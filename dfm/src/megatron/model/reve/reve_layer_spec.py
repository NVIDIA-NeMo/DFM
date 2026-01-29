# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pylint: disable=C0115,C0116,C0301

import copy
from dataclasses import dataclass
from typing import Literal, Optional, Union

import torch
import torch.nn as nn
from megatron.core import tensor_parallel
from megatron.core.jit import jit_fuser
from megatron.core.transformer.attention import (
    SelfAttentionSubmodules,
)
from megatron.core.transformer.custom_layers.transformer_engine import (
    TEColumnParallelLinear,
    TEDotProductAttention,
    TENorm,
    TERowParallelLinear,
)
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.mlp import MLP, MLPSubmodules
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.transformer_block import TransformerConfig
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.transformer_layer import TransformerLayer, TransformerLayerSubmodules
from megatron.core.utils import make_viewless_tensor

# to be imported from common
from dfm.src.megatron.model.common.dit_attention import (
    DiTCrossAttention,
    DiTCrossAttentionSubmodules,
    DiTSelfAttention,
)

from dfm.src.megatron.model.reve.reve_pytorch.layers import latency_tracker
import time

@dataclass
class ReveWithAdaLNSubmodules(TransformerLayerSubmodules):
    full_self_attention: Union[ModuleSpec, type] = IdentityOp


def l2_normalize(
    x: torch.Tensor, eps: float = 1e-6
) -> torch.Tensor:
    dtype = x.dtype
    compute_dtype = torch.float32
    norm = torch.norm(x, p=2, dim=-1, keepdim=True, dtype=compute_dtype)
    normalized = x / (norm + eps)
    return normalized.to(dtype)


class RMSNorm(nn.Module):
    def __init__(self, dims: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.dims = dims

    def forward(
        self, x: torch.Tensor
    ) -> torch.Tensor:
        input_dtype = x.dtype
        compute_dtype = torch.float32
        x = x.to(compute_dtype)
        x = torch.nn.functional.rms_norm(x, (self.dims,), eps=self.eps)
        return x.to(input_dtype)


class Modulation(nn.Module):
    def __init__(self, config: TransformerConfig, dims: int):
        super().__init__()

        # DEBUGGING
        modulation_config = copy.deepcopy(config)
        modulation_config.sequence_parallel = False

        self.lin = tensor_parallel.ColumnParallelLinear(
            dims,
            dims,
            config=modulation_config,
            init_method=torch.nn.init.xavier_normal_,
            bias=True,
            skip_bias_add=False,
            gather_output=True,
        )
        with torch.no_grad():
            self.lin.bias.zero_()
            self.lin.weight.zero_()
        self.act = nn.SiLU()

    def forward(
        self, vec: torch.Tensor
    ) -> torch.Tensor:
        scale, bias = self.lin(nn.functional.silu(vec))
        return scale + 1.0


class GateResiduals(nn.Module):
    def __init__(self, config: TransformerConfig, dims: int, do_modulation: bool, epsilon: float = 2e-2) -> None:
        super().__init__()
        if do_modulation:
            self.modulation = Modulation(config, dims)
        else:
            self.gate = nn.Parameter(torch.empty(dims))
            nn.init.constant_(self.gate, 0.0)
        self.do_modulation = do_modulation
        self.norm = RMSNorm(dims)
        self.epsilon = epsilon

    def forward(
        self,
        backbone: torch.Tensor,
        residual: torch.Tensor,
        vec: torch.Tensor | None,
    ) -> torch.Tensor:

        if self.do_modulation:
            gate = self.modulation(vec) + 2.9
        else:
            gate = self.gate + 3.9
            gate = gate.unsqueeze(0)
        gate = torch.sigmoid(gate)
        gate = gate * (1 - 2 * self.epsilon) + self.epsilon
        normalized_residual = self.norm(residual)
        if gate.ndim == 2:
            # NOTE: Need to unsqueeze(0) instead of unsqueeze(1) as in original Reve code to broadcast with the sequence
            gate = gate.unsqueeze(0)  # to broadcast with the sequence
        return backbone * gate + (1 - gate) * normalized_residual


class ReveLayerWithAdaLN(TransformerLayer):
    """A single transformer layer.

    Transformer layer takes input with size [s, b, h] and returns an
    output of the same size.

    Reve with Adapative Layer Normalization.
    """

    def __init__(
        self,
        config: TransformerConfig,
        submodules: TransformerLayerSubmodules,
        do_modulation: bool,
        do_cross_attn: bool,
        use_residual: bool = True,
        layer_number: int = 1,
        hidden_dropout: float = None,
        position_embedding_type: Literal["learned_absolute", "rope"] = "learned_absolute",
        pg_collection=None,
        vp_stage: Optional[int] = None,
    ):
        def _replace_no_cp_submodules(submodules):
            modified_submods = copy.deepcopy(submodules)
            modified_submods.cross_attention = IdentityOp
            return modified_submods

        # Replace any submodules that will have CP disabled and build them manually later after TransformerLayer init.
        # modified_submods = _replace_no_cp_submodules(submodules)
        super().__init__(
            config=config, submodules=submodules, layer_number=layer_number, hidden_dropout=hidden_dropout
        )

        self.do_modulation = do_modulation
        self.do_cross_attn = do_cross_attn
        self.use_residual = use_residual

        # Full self attention
        self.full_self_attention = build_module(
            submodules.full_self_attention,
            config=self.config,
            layer_number=layer_number,
        )

        # Cross attention
        # Override Cross Attention to disable CP.
        # Disable TP Comm overlap as well. Not disabling will attempt re-use of buffer size same as Q and lead to
        # incorrect tensor shapes.
        if self.do_cross_attn:
            self.cross_attention = build_module(
                submodules.cross_attention,
                config=self.config,
                layer_number=layer_number,
            )

        # Modulation
        if self.do_modulation:
            self.mod_self_attention = Modulation(config=self.config, dims=self.config.hidden_size)
            if self.do_cross_attn:
                self.mod_cross_attention = Modulation(config=self.config, dims=self.config.hidden_size)
            self.mod_mlp = Modulation(config=self.config, dims=self.config.hidden_size)

        # Gate residuals
        if self.use_residual:
            self.gate_residual_self_attention = GateResiduals(
                config=self.config, dims=self.config.hidden_size, do_modulation=self.do_modulation
            )
            if self.do_cross_attn:
                self.gate_residual_cross_attention = GateResiduals(
                    config=self.config, dims=self.config.hidden_size, do_modulation=self.do_modulation
                )
            self.gate_residual_mlp = GateResiduals(
                config=self.config, dims=self.config.hidden_size, do_modulation=self.do_modulation
            )

    def forward(
        self,
        hidden_states,
        attention_mask,
        context=None,
        context_mask=None,
        rotary_pos_emb=None,
        rotary_pos_cos=None,
        rotary_pos_sin=None,
        attention_bias=None,
        inference_params=None,
        packed_seq_params=None,
        sequence_len_offset=None,
        inference_context=None,
        rotary_pos_cos_sin=None,
        **kwargs,
    ):


        # DEBUGGING (benchmarking)
        full_layer_forward_start_time = time.time()

        vector_emb = attention_mask

        ########################## Self attention #################################

        # DEBUGGING (benchmarking)
        self_attention_start_time = time.time()

        # DEBUGGING (benchmarking)
        self_attention_modulation_start_time = time.time()

        # Modulation
        if self.do_modulation:
            scaled_hidden_states = self.mod_self_attention(vector_emb) * hidden_states
        else:
            scaled_hidden_states = hidden_states

        # DEBUGGING (benchmarking)
        self_attention_modulation_end_time = time.time()
        latency_tracker.update(f"[DEBUG]             (reve_layer_spec.py) - self attention modulation time: {self_attention_modulation_end_time - self_attention_modulation_start_time} seconds", now=self_attention_modulation_end_time)

        # DEBUGGING (benchmarking)
        self_attention_core_attention_start_time = time.time()

        # Attention
        attention_output, _ = self.full_self_attention(
            scaled_hidden_states,
            rotary_pos_emb=rotary_pos_emb,
            attention_mask=None,
            packed_seq_params=None if packed_seq_params is None else packed_seq_params["self_attention"],
        )

        # DEBUGGING (benchmarking)
        self_attention_core_attention_end_time = time.time()
        latency_tracker.update(f"[DEBUG]             (reve_layer_spec.py) - self attention core attention time: {self_attention_core_attention_end_time - self_attention_core_attention_start_time} seconds", now=self_attention_core_attention_end_time)

        # DEBUGGING (benchmarking)
        self_attention_gate_residuals_start_time = time.time()

        # Gate residuals
        if self.use_residual:
            hidden_states = self.gate_residual_self_attention(hidden_states, attention_output, vector_emb)
        else:
            hidden_states = attention_output

        # DEBUGGING (benchmarking)
        self_attention_gate_residuals_end_time = time.time()
        latency_tracker.update(f"[DEBUG]             (reve_layer_spec.py) - self attention gate residuals time: {self_attention_gate_residuals_end_time - self_attention_gate_residuals_start_time} seconds", now=self_attention_gate_residuals_end_time)

        # DEBUGGING (benchmarking)
        self_attention_end_time = time.time()
        latency_tracker.update(f"[DEBUG]         (reve_layer_spec.py) - self attention time: {self_attention_end_time - self_attention_start_time} seconds", now=self_attention_end_time)

        ########################## Cross attention #################################

        # DEBUGGING (benchmarking)
        cross_attention_start_time = time.time()

        if self.do_cross_attn:
            # DEBUGGING (benchmarking)
            cross_attention_modulation_start_time = time.time()

            # Modulation
            if self.do_modulation:
                scaled_hidden_states = self.mod_cross_attention(vector_emb) * hidden_states
            else:
                scaled_hidden_states = hidden_states

            # DEBUGGING (benchmarking)
            cross_attention_modulation_end_time = time.time()
            latency_tracker.update(f"[DEBUG]             (reve_layer_spec.py) - cross attention modulation time: {cross_attention_modulation_end_time - cross_attention_modulation_start_time} seconds", now=cross_attention_modulation_end_time)

            # DEBUGGING (benchmarking)
            cross_attention_core_attention_start_time = time.time()

            # Attention
            attention_output, _ = self.cross_attention(
                scaled_hidden_states,
                attention_mask=context_mask,
                key_value_states=context,
                packed_seq_params=None if packed_seq_params is None else packed_seq_params["cross_attention"],
            )

            # DEBUGGING (benchmarking)
            cross_attention_core_attention_end_time = time.time()
            latency_tracker.update(f"[DEBUG]             (reve_layer_spec.py) - cross attention core attention time: {cross_attention_core_attention_end_time - cross_attention_core_attention_start_time} seconds", now=cross_attention_core_attention_end_time)

            # DEBUGGING (benchmarking)
            cross_attention_gate_residuals_start_time = time.time()

            # Gate residuals
            if self.use_residual:
                hidden_states = self.gate_residual_cross_attention(hidden_states, attention_output, vector_emb)
            else:
                hidden_states = attention_output

            # DEBUGGING (benchmarking)
            cross_attention_gate_residuals_end_time = time.time()
            latency_tracker.update(f"[DEBUG]             (reve_layer_spec.py) - cross attention gate residuals time: {cross_attention_gate_residuals_end_time - cross_attention_gate_residuals_start_time} seconds", now=cross_attention_gate_residuals_end_time)

        # DEBUGGING (benchmarking)
        cross_attention_end_time = time.time()
        latency_tracker.update(f"[DEBUG]         (reve_layer_spec.py) - cross attention time: {cross_attention_end_time - cross_attention_start_time} seconds", now=cross_attention_end_time)

        ########################## MLP #################################

        # DEBUGGING (benchmarking)
        mlp_start_time = time.time()

        # DEBUGGING (benchmarking)
        mlp_modulation_start_time = time.time()

        # Modulation
        if self.do_modulation:
            scaled_hidden_states = self.mod_mlp(vector_emb) * hidden_states
        else:
            scaled_hidden_states = hidden_states

        # DEBUGGING (benchmarking)
        mlp_modulation_end_time = time.time()
        latency_tracker.update(f"[DEBUG]             (reve_layer_spec.py) - mlp modulation time: {mlp_modulation_end_time - mlp_modulation_start_time} seconds", now=mlp_modulation_end_time)

        # DEBUGGING (benchmarking)
        mlp_core_computation_start_time = time.time()

        # MLP
        mlp_output, _ = self.mlp(scaled_hidden_states)

        # DEBUGGING (benchmarking)
        mlp_core_computation_end_time = time.time()
        latency_tracker.update(f"[DEBUG]             (reve_layer_spec.py) - mlp core computation time: {mlp_core_computation_end_time - mlp_core_computation_start_time} seconds", now=mlp_core_computation_end_time)

        # DEBUGGING (benchmarking)
        mlp_gate_residuals_start_time = time.time()

        # Gate residuals
        if self.use_residual:
            hidden_states = self.gate_residual_mlp(hidden_states, mlp_output, vector_emb)
        else:
            hidden_states = mlp_output

        # DEBUGGING (benchmarking)
        mlp_gate_residuals_end_time = time.time()
        latency_tracker.update(f"[DEBUG]             (reve_layer_spec.py) - mlp gate residuals time: {mlp_gate_residuals_end_time - mlp_gate_residuals_start_time} seconds", now=mlp_gate_residuals_end_time)

        # Jit compiled function creates 'view' tensor. This tensor
        # potentially gets saved in the MPU checkpoint function context,
        # which rejects view tensors. While making a viewless tensor here
        # won't result in memory savings (like the data loader, or
        # p2p_communication), it serves to document the origin of this
        # 'view' tensor.
        output = make_viewless_tensor(inp=hidden_states, requires_grad=hidden_states.requires_grad, keep_graph=True)

        # DEBUGGING (benchmarking)
        mlp_end_time = time.time()
        latency_tracker.update(f"[DEBUG]         (reve_layer_spec.py) - mlp time: {mlp_end_time - mlp_start_time} seconds", now=mlp_end_time)

        # DEBUGGING (benchmarking)
        full_layer_forward_end_time = time.time()
        latency_tracker.update(f"[DEBUG]     (reve_layer_spec.py) - full layer forward time: {full_layer_forward_end_time - full_layer_forward_start_time} seconds", now=full_layer_forward_end_time)

        return output, context


def get_reve_adaln_block_with_transformer_engine_spec() -> ModuleSpec:
    params = {"attn_mask_type": AttnMaskType.padding}
    return ModuleSpec(
        module=ReveLayerWithAdaLN,
        params={"do_modulation": True, "do_cross_attn": True},
        submodules=ReveWithAdaLNSubmodules(
            full_self_attention=ModuleSpec(
                module=DiTSelfAttention,
                params=params,
                submodules=SelfAttentionSubmodules(
                    linear_qkv=TEColumnParallelLinear,
                    core_attention=TEDotProductAttention,
                    linear_proj=TERowParallelLinear,
                    q_layernorm=l2_normalize,
                    k_layernorm=l2_normalize,
                ),
            ),
            cross_attention=ModuleSpec(
                module=DiTCrossAttention,
                params=params,
                submodules=DiTCrossAttentionSubmodules(
                    linear_q=TEColumnParallelLinear,
                    linear_kv=TEColumnParallelLinear,
                    core_attention=TEDotProductAttention,
                    linear_proj=TERowParallelLinear,
                    q_layernorm=l2_normalize,
                    k_layernorm=l2_normalize,
                ),
            ),
            mlp=ModuleSpec(
                module=MLP,
                submodules=MLPSubmodules(
                    linear_fc1=TEColumnParallelLinear,
                    linear_fc2=TERowParallelLinear,
                ),
            ),
        ),
    )


def get_reve_adaln_text_block_with_transformer_engine_spec() -> ModuleSpec:
    params = {"attn_mask_type": AttnMaskType.padding}
    return ModuleSpec(
        module=ReveLayerWithAdaLN,
        params={"do_modulation": False, "do_cross_attn": False},
        submodules=ReveWithAdaLNSubmodules(
            full_self_attention=ModuleSpec(
                module=DiTSelfAttention,
                params=params,
                submodules=SelfAttentionSubmodules(
                    linear_qkv=TEColumnParallelLinear,
                    core_attention=TEDotProductAttention,
                    linear_proj=TERowParallelLinear,
                    q_layernorm=l2_normalize,
                    k_layernorm=l2_normalize,
                ),
            ),
            mlp=ModuleSpec(
                module=MLP,
                submodules=MLPSubmodules(
                    linear_fc1=TEColumnParallelLinear,
                    linear_fc2=TERowParallelLinear,
                ),
            ),
        ),
    )
