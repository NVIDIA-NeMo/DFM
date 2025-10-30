
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
from typing import Union, Optional

import torch
import torch.cuda.amp as amp
import torch.nn as nn
from megatron.core import parallel_state, tensor_parallel
from megatron.core.transformer.attention import (
    CrossAttention,
    CrossAttentionSubmodules,
    SelfAttention,
    SelfAttentionSubmodules,
)
from megatron.core.transformer.custom_layers.transformer_engine import (
    TEColumnParallelLinear,
    TEDotProductAttention,
    TERowParallelLinear,
)
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.mlp import MLP, MLPSubmodules
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.transformer_layer import TransformerLayer, TransformerLayerSubmodules
from megatron.core.utils import make_viewless_tensor
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.extensions.transformer_engine import TENorm

try:
    import transformer_engine  # pylint: disable=unused-import

    HAVE_TE = True
    from megatron.core.extensions.transformer_engine import SplitAlongDim
    
except ImportError:
    HAVE_TE = False
    SplitAlongDim = None


class WanLayerNorm(nn.LayerNorm):

    def __init__(self, dim, eps=1e-6, elementwise_affine=False):
        super().__init__(dim, elementwise_affine=elementwise_affine, eps=eps)

    def forward(self, x):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
        """
        return super().forward(x).type_as(x)


@dataclass
class WanSelfAttentionSubmodules:
    """
    Configuration class for specifying the submodules of a self-attention.
    """

    linear_qkv: Union[ModuleSpec, type] = None
    core_attention: Union[ModuleSpec, type] = None
    linear_proj: Union[ModuleSpec, type] = None
    layernorm_across_head: bool = False
    q_layernorm: Union[ModuleSpec, type] = None
    k_layernorm: Union[ModuleSpec, type] = None


@dataclass
class WanCrossAttentionSubmodules:
    """
    Configuration class for specifying the submodules of a cross-attention.
    """
    linear_q: Union[ModuleSpec, type] = None
    linear_kv: Union[ModuleSpec, type] = None
    core_attention: Union[ModuleSpec, type] = None
    linear_proj: Union[ModuleSpec, type] = None
    layernorm_across_head: bool = False
    q_layernorm: Union[ModuleSpec, type] = None
    k_layernorm: Union[ModuleSpec, type] = None


class WanSelfAttention(SelfAttention):
    def __init__(
        self,
        config: TransformerConfig,
        submodules: WanSelfAttentionSubmodules,
        layer_number: int,
        attn_mask_type: AttnMaskType,
        cp_comm_type: str = None,
        pg_collection: Optional[ProcessGroupCollection] = None,
    ):
        super().__init__(
            config,
            submodules,
            layer_number,
            attn_mask_type,
            cp_comm_type,
            pg_collection,
        )

        self.layernorm_across_head = submodules.layernorm_across_head

        # override q_layernorm
        if submodules.q_layernorm is not None:
            if self.layernorm_across_head:
                q_layernorm_size = self.query_projection_size
            else:
                q_layernorm_size = self.hidden_size_per_attention_head
            import transformer_engine as te
            norm_config = copy.deepcopy(self.config)
            norm_config.normalization = "RMSNorm"
            self.q_layernorm = build_module(
                submodules.q_layernorm,
                eps=norm_config.layernorm_epsilon,
                hidden_size=q_layernorm_size,
                config=norm_config,
            )
        else:
            self.q_layernorm = None

        # override k_layernorm
        if submodules.k_layernorm is not None:
            if self.layernorm_across_head:
                k_layernorm_size = self.kv_projection_size
            else:
                k_layernorm_size = self.hidden_size_per_attention_head
            import transformer_engine as te
            norm_config = copy.deepcopy(self.config)
            norm_config.normalization = "RMSNorm"
            self.k_layernorm = build_module(
                submodules.k_layernorm,
                eps=norm_config.layernorm_epsilon,
                hidden_size=k_layernorm_size,
                config=norm_config,
            )
        else:
            self.k_layernorm = None

    def get_query_key_value_tensors(self, hidden_states, key_value_states=None):
        """
        Derives `query`, `key` and `value` tensors from `hidden_states`.
        """
        # Attention heads [sq, b, h] --> [sq, b, ng * (np/ng + 2) * hn)]
        mixed_qkv, _ = self.linear_qkv(hidden_states)

        # [sq, b, hp] --> [sq, b, ng, (np/ng + 2) * hn]
        new_tensor_shape = mixed_qkv.size()[:-1] + (
            self.num_query_groups_per_partition,
            (
                (self.num_attention_heads_per_partition // self.num_query_groups_per_partition + 2)
                * self.hidden_size_per_attention_head
            ),
        )
        mixed_qkv = mixed_qkv.view(*new_tensor_shape)

        split_arg_list = [
            (
                self.num_attention_heads_per_partition
                // self.num_query_groups_per_partition
                * self.hidden_size_per_attention_head
            ),
            self.hidden_size_per_attention_head,
            self.hidden_size_per_attention_head,
        ]

        if SplitAlongDim is not None:

            # [sq, b, ng, (np/ng + 2) * hn]
            # --> [sq, b, ng, np/ng * hn], [sq, b, ng, hn], [sq, b, ng, hn]
            (query, key, value) = SplitAlongDim(mixed_qkv, 3, split_arg_list)
        else:

            # [sq, b, ng, (np/ng + 2) * hn]
            # --> [sq, b, ng, np/ng * hn], [sq, b, ng, hn], [sq, b, ng, hn]
            (query, key, value) = torch.split(mixed_qkv, split_arg_list, dim=3)

        # [sq, b, ng, np/ng * hn] -> [sq, b, np, hn]
        query = query.reshape(query.size(0), query.size(1), -1, self.hidden_size_per_attention_head)

        # gather query and key heads across TP ranks if self.layernorm_across_head is True
        if self.layernorm_across_head and parallel_state.get_tensor_model_parallel_world_size() > 1:
            query = query.transpose(-2, -1)
            key = key.transpose(-2, -1)
            query = tensor_parallel.gather_from_tensor_model_parallel_region(query)
            key = tensor_parallel.gather_from_tensor_model_parallel_region(key)
            query = query.transpose(-2, -1)
            key = key.transpose(-2, -1)

        if self.q_layernorm is not None:
            if self.layernorm_across_head:                
                q_flat = query.reshape(query.size(0), query.size(1), -1).contiguous()  # [sq, b, np*hn]
                q_flat = self.q_layernorm(q_flat)
                query = q_flat.view(query.size(0), query.size(1), -1, self.hidden_size_per_attention_head)  # [sq, b, np, hn]
            else:
                query = self.q_layernorm(query.contiguous())

        if self.k_layernorm is not None:
            if self.layernorm_across_head:
                k_flat = key.reshape(key.size(0), key.size(1), -1).contiguous()
                k_flat = self.k_layernorm(k_flat)
                key = k_flat.view(key.size(0), key.size(1), -1, self.hidden_size_per_attention_head)
            else:
                key = self.k_layernorm(key.contiguous())

        # scatter query and key heads across TP ranks if self.layernorm_across_head is True
        if self.layernorm_across_head and parallel_state.get_tensor_model_parallel_world_size() > 1:
            query = query.transpose(-2, -1)
            key = key.transpose(-2, -1)
            query = tensor_parallel.scatter_to_tensor_model_parallel_region(query)
            key = tensor_parallel.scatter_to_tensor_model_parallel_region(key)
            query = query.transpose(-2, -1)
            key = key.transpose(-2, -1)
            query = query.contiguous() # important becuase TE attention expects contiguous tensors
            key = key.contiguous() # important becuase TE attention expects contiguous tensors

        if self.config.test_mode:
            self.run_realtime_tests()

        return query, key, value


class WanCrossAttention(CrossAttention):
    def __init__(
        self,
        config: TransformerConfig,
        submodules: WanCrossAttentionSubmodules,
        layer_number: int,
        attn_mask_type: AttnMaskType,
        cp_comm_type: str = None,
        pg_collection: Optional[ProcessGroupCollection] = None,
    ):
        super().__init__(
            config,
            submodules,
            layer_number,
            attn_mask_type,
            cp_comm_type,
            pg_collection,
        )

        self.layernorm_across_head = submodules.layernorm_across_head

        # override q_layernorm
        if submodules.q_layernorm is not None:
            if self.layernorm_across_head:
                q_layernorm_size = self.query_projection_size
            else:
                q_layernorm_size = self.hidden_size_per_attention_head
            import transformer_engine as te
            norm_config = copy.deepcopy(self.config)
            norm_config.normalization = "RMSNorm"
            self.q_layernorm = build_module(
                submodules.q_layernorm,
                eps=norm_config.layernorm_epsilon,
                hidden_size=q_layernorm_size,
                config=norm_config,
            )
        else:
            self.q_layernorm = None

        # override k_layernorm
        if submodules.k_layernorm is not None:
            if self.layernorm_across_head:
                k_layernorm_size = self.kv_projection_size
            else:
                k_layernorm_size = self.hidden_size_per_attention_head
            import transformer_engine as te
            norm_config = copy.deepcopy(self.config)
            norm_config.normalization = "RMSNorm"
            self.k_layernorm = build_module(
                submodules.k_layernorm,
                eps=norm_config.layernorm_epsilon,
                hidden_size=k_layernorm_size,
                config=norm_config,
            )
        else:
            self.k_layernorm = None

    def get_query_key_value_tensors(self, hidden_states, key_value_states):
        """
        Derives `query` tensor from `hidden_states`, and `key`/`value` tensors
        from `key_value_states`.
        """
        # Attention heads [sk, b, h] --> [sk, b, (np * 2 * hn)]
        mixed_kv, _ = self.linear_kv(key_value_states)

        # [sk, b, (np * 2 * hn)] --> [sk, b, np, 2 * hn]
        new_tensor_shape = mixed_kv.size()[:-1] + (
            self.num_attention_heads_per_partition,
            2 * self.hidden_size_per_attention_head,
        )
        mixed_kv = mixed_kv.view(*new_tensor_shape)

        # [sk, b, np, 2 * hn] --> 2 [sk, b, np, hn]
        (key, value) = tensor_parallel.split_tensor_along_last_dim(mixed_kv, 2)

        # Attention head [sq, b, h] --> [sq, b, hp]
        query, _ = self.linear_q(hidden_states)

        # [sq, b, hp] --> [sq, b, np, hn]
        new_tensor_shape = query.size()[:-1] + (
            self.num_attention_heads_per_partition,
            self.hidden_size_per_attention_head,
        )
        query = query.view(*new_tensor_shape)

        # gather query and key heads across TP ranks if self.layernorm_across_head is True
        if self.layernorm_across_head and parallel_state.get_tensor_model_parallel_world_size() > 1:
            query = query.transpose(-2, -1)
            key = key.transpose(-2, -1)
            query = tensor_parallel.gather_from_tensor_model_parallel_region(query)
            key = tensor_parallel.gather_from_tensor_model_parallel_region(key)
            query = query.transpose(-2, -1)
            key = key.transpose(-2, -1)

        if self.q_layernorm is not None:
            if self.layernorm_across_head:
                q_flat = query.reshape(query.size(0), query.size(1), -1).contiguous()  # [sq, b, np*hn]
                q_flat = self.q_layernorm(q_flat)
                query = q_flat.view(query.size(0), query.size(1), -1, self.hidden_size_per_attention_head)  # [sq, b, np, hn]
            else:
                query = self.q_layernorm(query.contiguous())

        if self.k_layernorm is not None:
            if self.layernorm_across_head:
                k_flat = key.reshape(key.size(0), key.size(1), -1).contiguous()
                k_flat = self.k_layernorm(k_flat)
                key = k_flat.view(key.size(0), key.size(1), -1, self.hidden_size_per_attention_head)
            else:
                key = self.k_layernorm(key.contiguous())

        # scatter query and key heads across TP ranks if self.layernorm_across_head is True
        if self.layernorm_across_head and parallel_state.get_tensor_model_parallel_world_size() > 1:
            query = query.transpose(-2, -1)
            key = key.transpose(-2, -1)
            query = tensor_parallel.scatter_to_tensor_model_parallel_region(query)
            key = tensor_parallel.scatter_to_tensor_model_parallel_region(key)
            query = query.transpose(-2, -1)
            key = key.transpose(-2, -1)
            query = query.contiguous() # important becuase TE attention expects contiguous tensors
            key = key.contiguous() # important becuase TE attention expects contiguous tensors

        return query, key, value
        

@dataclass
class WanWithAdaLNSubmodules(TransformerLayerSubmodules):
    temporal_self_attention: Union[ModuleSpec, type] = IdentityOp
    full_self_attention: Union[ModuleSpec, type] = IdentityOp
    norm1: Union[ModuleSpec, type] = None
    norm3: Union[ModuleSpec, type] = None
    norm2: Union[ModuleSpec, type] = None


class WanAdaLN(MegatronModule):
    """
    Adaptive Layer Normalization Module for DiT.
    """

    def __init__(
        self, config: TransformerConfig
    ):
        super().__init__(config)
        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 6, config.hidden_size) / config.hidden_size**0.5)

        setattr(self.modulation, "sequence_parallel", config.sequence_parallel)

    def forward(self, timestep_emb):
        e = (self.modulation + timestep_emb).chunk(6, dim=1)
        return e

    # @jit_fuser
    def modulate(self, x, shift, scale):
        return x * (1 + scale) + shift

    # @jit_fuser
    def scale_add(self, residual, x, gate):
        return residual + gate * x


class WanLayerWithAdaLN(TransformerLayer):
    """A single transformer layer.

    Transformer layer takes input with size [s, b, h] and returns an
    output of the same size.

    DiT with Adapative Layer Normalization.
    """

    def __init__(
        self,
        config: TransformerConfig,
        submodules: TransformerLayerSubmodules,
        layer_number: int = 1,
        hidden_dropout: float = None,
        pg_collection: Optional[ProcessGroupCollection] = None,
        vp_stage: Optional[int] = None,
    ):
        super().__init__(
            config=config, submodules=submodules, layer_number=layer_number, hidden_dropout=hidden_dropout
        )

        # # TODO: Override Cross Attention to disable TP Comm overlap as well. ???
        # # Not disabling will attempt re-use of buffer size same as Q and lead to incorrect tensor shapes.
        # cp_override_config = copy.deepcopy(config)
        # cp_override_config.tp_comm_overlap = False
        # self.cross_attention = build_module(
        #     submodules.cross_attention,
        #     config=cp_override_config,
        #     layer_number=layer_number,
        # )

        self.full_self_attention = build_module(
            submodules.full_self_attention,
            config=self.config,
            layer_number=layer_number,
        )

        self.adaLN = WanAdaLN(config=self.config)
        self.norm1 = build_module(
            submodules.norm1,
            dim=config.hidden_size,
            eps=config.layernorm_epsilon,
            elementwise_affine=False
        )
        self.norm3 = build_module(
            submodules.norm3,
            dim=config.hidden_size,
            eps=config.layernorm_epsilon,
            elementwise_affine=True,
        )
        self.norm2 = build_module(
            submodules.norm2,
            dim=config.hidden_size,
            eps=config.layernorm_epsilon,
            elementwise_affine=False,
        )


    def forward(
        self,
        hidden_states,
        attention_mask=None,
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
    ):
        # the timestep embedding is stored in attention_mask argument
        timestep_emb = attention_mask
        rope_emb = rotary_pos_emb

        shift_full, scale_full, gate_full, shift_mlp, scale_mlp, gate_mlp = self.adaLN(timestep_emb)
        # transpose to bring it to [1, b, ...] format
        shift_full = shift_full.transpose(0, 1)
        scale_full = scale_full.transpose(0, 1)
        gate_full = gate_full.transpose(0, 1)
        shift_mlp = shift_mlp.transpose(0, 1)
        scale_mlp = scale_mlp.transpose(0, 1)
        gate_mlp = gate_mlp.transpose(0, 1)

        # ******************************************** full self attention *******************************************

        # adaLN with scale + shift + gate
        pre_full_attn_layernorm_output_ada = self.adaLN.modulate(
            self.norm1(hidden_states),
            shift=shift_full,
            scale=scale_full,
        )

        attention_output, bias = self.full_self_attention(
            pre_full_attn_layernorm_output_ada,
            attention_mask=None,
            rotary_pos_emb=rope_emb,
            rotary_pos_cos=rotary_pos_cos,
            rotary_pos_sin=rotary_pos_sin,
            packed_seq_params=packed_seq_params['self_attention'],
        )
        if bias is not None:
            attention_output = attention_output + bias

        hidden_states = self.adaLN.scale_add(residual=hidden_states, x=attention_output, gate=gate_full)

        # ******************************************** cross attention ******************************************************

        attention_output, bias = self.cross_attention(
            self.norm3(hidden_states),
            attention_mask=context_mask,
            key_value_states=context,
            packed_seq_params=packed_seq_params['cross_attention'],
        )
        if bias is not None:
            attention_output = attention_output + bias

        hidden_states = hidden_states + attention_output

        # ******************************************** mlp ******************************************************

        pre_mlp_layernorm_output_ada = self.adaLN.modulate(
            self.norm2(hidden_states),
            shift=shift_mlp,
            scale=scale_mlp,
        )

        mlp_output, bias = self.mlp(pre_mlp_layernorm_output_ada)
        if bias is not None:
           mlp_output = mlp_output + bias

        hidden_states = self.adaLN.scale_add(residual=hidden_states, x=mlp_output, gate=gate_mlp)

        # TODO: Jit compiled function creates 'view' tensor. This tensor
        # potentially gets saved in the MPU checkpoint function context,
        # which rejects view tensors. While making a viewless tensor here
        # won't result in memory savings (like the data loader, or
        # p2p_communication), it serves to document the origin of this
        # 'view' tensor. ???
        output = make_viewless_tensor(inp=hidden_states, requires_grad=hidden_states.requires_grad, keep_graph=True)
        # output = hidden_states

        return output, context


import transformer_engine as te
def get_wan_block_with_transformer_engine_spec() -> ModuleSpec:
    params = {"attn_mask_type": AttnMaskType.padding}
    return ModuleSpec(
        module=WanLayerWithAdaLN,
        submodules=WanWithAdaLNSubmodules(
            norm1=WanLayerNorm,
            norm3=WanLayerNorm,
            norm2=WanLayerNorm,
            full_self_attention=ModuleSpec(
                module=WanSelfAttention,
                params=params,
                submodules=WanSelfAttentionSubmodules(
                    linear_qkv=TEColumnParallelLinear,
                    core_attention=TEDotProductAttention,
                    linear_proj=TERowParallelLinear,
                    layernorm_across_head=True,     
                    q_layernorm=TENorm,
                    k_layernorm=TENorm,         
                ),
            ),
            cross_attention=ModuleSpec(
                module=WanCrossAttention,
                params=params,
                submodules=WanCrossAttentionSubmodules(
                    linear_q=TEColumnParallelLinear,
                    linear_kv=TEColumnParallelLinear,
                    core_attention=TEDotProductAttention,
                    linear_proj=TERowParallelLinear,
                    layernorm_across_head=True,
                    q_layernorm=TENorm,
                    k_layernorm=TENorm,
                ),
            ),
            mlp=ModuleSpec(
                module=MLP,
                submodules=MLPSubmodules(
                    linear_fc1=TEColumnParallelLinear,
                    # by default, activation_func is openai_gelu, which is equivalent to nn.GELU(approximate='tanh')
                    linear_fc2=TERowParallelLinear,
                ),
            ),
        ),
    )
