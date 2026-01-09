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

import math
from typing import Dict, List, Optional, Set, Tuple, Union

import torch
import torch.nn as nn
from diffusers.models.embeddings import Timesteps
from megatron.core import parallel_state, tensor_parallel
from megatron.core.dist_checkpointing.mapping import ShardedStateDict
from megatron.core.models.common.vision_module.vision_module import VisionModule
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.transformer.enums import ModelType
from megatron.core.transformer.transformer_block import TransformerBlock
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.utils import make_sharded_tensor_for_checkpoint
from torch import Tensor

from dfm.src.megatron.model.common.dit_embeddings import ParallelTimestepEmbedding
from dfm.src.megatron.model.wan.utils import patchify_compact, unpatchify_compact
from dfm.src.megatron.model.wan.wan_layer_spec import (
    get_wan_block_with_transformer_engine_spec as WanLayerWithAdaLNspec,
)

from .rope_utils import Wan3DRopeEmbeddings


def sinusoidal_embedding_1d(dim, position):
    # preprocess
    assert dim % 2 == 0
    half = dim // 2
    position = position

    # calculation
    sinusoid = torch.outer(position, torch.pow(10000, -torch.arange(half).to(position).div(half)))
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    return x


class Head(nn.Module):
    def __init__(self, dim, out_dim, patch_size, eps=1e-6):
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim
        self.patch_size = patch_size
        self.eps = eps

        # layers
        out_dim = math.prod(patch_size) * out_dim
        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
        self.head = nn.Linear(dim, out_dim)

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

    def forward(self, x, e):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            e(Tensor): Shape [B, C]
        """
        e = (self.modulation + e.unsqueeze(1)).chunk(2, dim=1)
        x = self.head(self.norm(x) * (1 + e[1]) + e[0])
        return x


class WanModel(VisionModule):
    """
    WanModel is a VisionModule that implements a Wan model.
    Attributes:
        config (TransformerConfig): Configuration for the transformer.
        pre_process (bool): Whether to apply pre-processing steps.
        post_process (bool): Whether to apply post-processing steps.
        fp16_lm_cross_entropy (bool): Whether to use fp16 for cross-entropy loss.
        parallel_output (bool): Whether to use parallel output.
        transformer_decoder_layer_spec: Specification for the transformer decoder layer. Auto-selects based on params_dtype if None.
        model_type (ModelType): Type of the model.
    """

    def __init__(
        self,
        config: TransformerConfig,
        pre_process: bool = True,
        post_process: bool = True,
        fp16_lm_cross_entropy: bool = False,
        parallel_output: bool = True,
        transformer_decoder_layer_spec=WanLayerWithAdaLNspec,
        **kwargs,
    ):
        super(WanModel, self).__init__(config=config)

        self.config: TransformerConfig = config

        self.transformer_decoder_layer_spec = transformer_decoder_layer_spec()
        self.pre_process = pre_process
        self.post_process = post_process
        self.fp16_lm_cross_entropy = fp16_lm_cross_entropy
        self.parallel_output = parallel_output

        # megatron core pipelining currently depends on model type
        # TODO: remove this dependency ?
        self.model_type = ModelType.encoder_or_decoder

        self.num_heads = self.config.num_attention_heads
        self.freq_dim = self.config.freq_dim
        self.in_channels = self.config.in_channels
        self.out_channels = self.config.out_channels
        self.patch_spatial = self.config.patch_spatial
        self.patch_temporal = self.config.patch_temporal
        self.patch_size = (self.patch_temporal, self.patch_spatial, self.patch_spatial)

        # these attributes are unused for images/videos, we just set because bridge training requires for LLMs
        self.share_embeddings_and_output_weights = False

        ######################################
        ########## Wan architecture ##########

        # embeddings
        if self.pre_process:
            self.patch_embedding = nn.Conv3d(
                self.in_channels, self.config.hidden_size, kernel_size=self.patch_size, stride=self.patch_size
            )

        self.text_embedding = nn.Sequential(
            nn.Linear(self.config.text_dim, self.config.crossattn_emb_size),
            nn.GELU(approximate="tanh"),
            nn.Linear(self.config.crossattn_emb_size, self.config.crossattn_emb_size),
        )

        # As in diffuser's Wan implementation
        self.timesteps_proj = Timesteps(num_channels=self.freq_dim, flip_sin_to_cos=True, downscale_freq_shift=0)
        self.time_embedder = ParallelTimestepEmbedding(
            in_channels=self.freq_dim, time_embed_dim=self.config.hidden_size
        )
        self.time_proj_act_fn = nn.SiLU()
        self.time_proj = nn.Linear(self.config.hidden_size, self.config.hidden_size * 6)

        self.rope_embeddings = Wan3DRopeEmbeddings(
            dim_head=self.config.hidden_size // self.num_heads, max_position_len=1024
        )

        # decoder blocks
        self.decoder = TransformerBlock(
            config=self.config,
            spec=self.transformer_decoder_layer_spec,
            pre_process=self.pre_process,
            post_process=self.post_process,
            post_layer_norm=False,
        )

        # output head
        if self.post_process:
            self.head = Head(self.config.hidden_size, self.out_channels, self.patch_size, eps=1e-6)

        # set attributes "average_gradients_across_tp_domain" for nn.Parameter objects
        # this is used for gradient averaging across TP domain with sequence parallelism
        self._mark_trainable_params_for_tp_grad_avg(
            [
                self.patch_embedding,
                self.text_embedding,
                self.time_embedder,
                self.time_proj,
                self.head,
            ]
        )

    def forward(
        self,
        x: Tensor,
        t: Tensor,
        context: Tensor,
        grid_sizes: list[Tuple[int, int, int]],
        fwd_pred_type: Optional[str] = None,
        packed_seq_params: PackedSeqParams = None,
        scale_t: bool = False,
        unpatchify_features: bool = False,
        feature_indices: Optional[Set[int]] = None,
        return_features_early: bool = False,
        **kwargs,
    ) -> Union[Tensor, List[Tensor]]:
        """Forward pass.

        Args:
            x List[Tensor]: list of vae encoded data (in_channel, f, h, w)
            grid_sizes List[Tuple[int, int, int]]: list of grid sizes (f, h, w)
            t Tensor: timesteps (in [0, 1] if scale_t=True, or [0, 1000] if scale_t=False)
            context List[Tensor]: list of context (text_len, hidden_size)
            packed_seq_params PackedSeqParams: packed sequence parameters
            scale_t: If True, rescale t from [0, 1] to [0, 1000] for timestep embeddings.
                     Use this when t comes from noise_scheduler (e.g., during distillation).
            feature_indices: A set of layer indices (0-based) from which to extract
                intermediate hidden states. Consistent with FastGen's Wan implementation.
            return_features_early: If True and feature_indices is non-empty, returns only
                the features list. If False (default), returns [output, features] when
                feature_indices is non-empty.

        Returns:
            Union[Tensor, List[Tensor]]:
                - If feature_indices is None or empty: output tensor
                - If feature_indices is non-empty and return_features_early is True: features list only
                - If feature_indices is non-empty and return_features_early is False: [output, features]
        """
        #################################
        ########## Wan forward ##########

        print("context passed to WanModel", context.flatten()[:20].tolist())
        x_input = x

        # ============= embedders =============
        if unpatchify_features:
            x = patchify_compact(x, self.patch_size)

        # run input embedding
        if self.pre_process:
            # x.shape [s, b, c * pF * pH * pW]
            seq_len, batch_size, _ = x.shape
            c = self.out_channels
            pF, pH, pW = self.patch_size
            x = x.reshape(seq_len * batch_size, pF, pH, pW, c)  # output: x.shape [s * b, pF, pH, pW, c]
            x = x.permute(0, 4, 1, 2, 3)  # output: x.shape [s * b, c, pF, pH, pW]
            x = self.patch_embedding(x)  # output: x.shape [s * b, hidden_size, 1, 1, 1]
            x = x.flatten(1)  # output: x.shape [s * b, hidden_size]
            x = x.reshape(seq_len, batch_size, -1)  # output: x.shape [s, b, hidden_size]

            # split sequence for sequence_parallel
            # TODO: for PP, do we move scatter_to_sequence_parallel_region here or after "x = self.decoder.input_tensor" ???
            if self.config.sequence_parallel:
                x = tensor_parallel.scatter_to_sequence_parallel_region(
                    x
                )  # output: x.shape [s * b // tp_size, hidden_size]

        else:
            # intermediate stage of pipeline
            x = self.decoder.input_tensor

        # ============= DEBUG: Patch embedding output =============
        print(
            f"[Megatron] x (after patch_embedding): shape={x.shape}, sum={x.sum().item():.4f}, first5={x[0, 0, :5].tolist()}"
        )

        # Rescale t from [0, 1] to [0, 1000] if scale_t is True
        # This is needed when t comes from noise_scheduler (e.g., during distillation with DMD2Model)
        input_t = t
        if scale_t:
            t = self.noise_scheduler.rescale_t(t)
        assert 1 <= t <= 1000, "t must be in [1, 1000]"

        # ============= DEBUG: Timestep value =============
        print(f"[Megatron] timestep t (rescaled): {t.item():.4f}")

        # time embeddings
        timestep_sinusoidal = self.timesteps_proj(t).to(x.dtype)
        print(
            f"[Megatron] timestep_sinusoidal: sum={timestep_sinusoidal.sum().item():.4f}, first5={timestep_sinusoidal[0, :5].tolist()}"
        )

        e = self.time_embedder(timestep_sinusoidal)
        print(f"[Megatron] temb (after time_embedder): sum={e.sum().item():.4f}, first5={e[0, :5].tolist()}")

        e0 = self.time_proj(self.time_proj_act_fn(e)).unflatten(1, (6, self.config.hidden_size))
        print(f"[Megatron] timestep_proj: shape={e0.shape}, sum={e0.sum().item():.4f}, first5={e0[0, 0, :5].tolist()}")

        # context embeddings
        context_input = context.clone()
        context = self.text_embedding(context)  # shape [text_len, b, hidden_size]
        print(
            f"[Megatron] context (after text_embedding): shape={context.shape}, sum={context.sum().item():.4f}, first5={context[0, 0, :5].tolist()}"
        )

        # ============= decoder =============
        # calculate rotary pos emb
        n_head, dim_head = self.num_heads, self.config.hidden_size // self.num_heads
        cu_seqlens_q_padded = packed_seq_params["self_attention"].cu_seqlens_q_padded
        rotary_pos_emb = self.rope_embeddings(
            n_head, dim_head, cu_seqlens_q_padded, grid_sizes, t.device
        )  # output: rotary_pos_emb.shape [s, b, 1, dim_head]
        # print(f'[Megatron] rotary_pos_emb: shape={rotary_pos_emb.shape}, values={rotary_pos_emb[0, 0, 0, :5].tolist()}')

        # run decoder
        decoder_output = self.decoder(
            hidden_states=x,
            attention_mask=e0,
            context=context,
            context_mask=None,
            rotary_pos_emb=rotary_pos_emb,
            rotary_pos_cos=None,
            rotary_pos_sin=None,
            packed_seq_params=packed_seq_params,
            feature_indices=feature_indices,
        )

        # Handle feature extraction (consistent with FastGen's Wan implementation)
        if feature_indices is not None and len(feature_indices) > 0:
            # decoder returns (hidden_states, features) tuple when feature_indices is non-empty
            x, features = decoder_output
            # Unpatchify features to spatial format using unpatchify_compact
            # For features, out_dim = hidden_size / (p_t * p_h * p_w)
            feature_out_dim = self.config.hidden_size // math.prod(self.patch_size)
            features = [unpatchify_compact(feat, grid_sizes, feature_out_dim, self.patch_size) for feat in features]

            if return_features_early:
                # Return only the features list (for discriminator feature extraction)
                return features
            # Otherwise, continue to compute output and return [output, features] below
        else:
            x = decoder_output
            features = None

        # If we have features but not returning early, x is already set from decoder_output unpacking
        # ============= DEBUG: After transformer blocks =============
        print(f"[Megatron] after decoder: shape={x.shape}, sum={x.sum().item():.4f}, first5={x[0, 0, :5].tolist()}")

        if fwd_pred_type is None:
            fwd_pred_type = self.net_pred_type

        # return if not post_process
        if not self.post_process:
            return x

        # head
        x = x.transpose(0, 1)  # head expects shape [b, s, hidden_size]
        x = self.head(x, e)  # output: x.shape [b, s, c * pF * pH * pW]
        x = x.transpose(0, 1)  # reshape back to shape [s, b, c * pF * pH * pW]

        # ============= DEBUG: After head =============
        print(f"[Megatron] after head: shape={x.shape}, sum={x.sum().item():.4f}, first5={x[0, 0, :5].tolist()}")

        # gather outputs for sequence_parallel
        # Note: in GPT models, because the vocab projection matrix is ColumnParallelLinear, the sequence is
        #   automatically gathered in ColumnParallelLinear forward pass.
        #   However, in Wan models, we need to gather the outputs manually.
        if self.config.sequence_parallel:
            x = tensor_parallel.gather_from_sequence_parallel_region(x)

        if unpatchify_features:
            x = unpatchify_compact(x, grid_sizes, self.config.z_dim, self.patch_size)
            print(
                f"[Megatron] after unpatchify: shape={x.shape}, sum={x.sum().item():.4f}, first5={x.flatten()[:5].tolist()}"
            )

        # ============= DEBUG: Before/After scheduler step =============
        print(f"[Megatron] before convert_model_output: sum={x.sum().item():.4f}")
        x = self.noise_scheduler.convert_model_output(
            x_input, x, input_t, src_pred_type=self.net_pred_type, target_pred_type=fwd_pred_type
        )
        print(f"[Megatron] after convert_model_output: sum={x.sum().item():.4f}, first5={x.flatten()[:5].tolist()}")

        # Return [output, features] when feature_indices was provided and not returning early
        # (consistent with FastGen's behavior when return_features_early=False)
        if features is not None:
            return [x, features]

        return x  # output: x.shape [s, b, c * pF * pH * pW]

    def set_input_tensor(self, input_tensor: Tensor) -> None:
        """Sets input tensor to the model.

        See megatron.model.transformer.set_input_tensor()

        Args:
            input_tensor (Tensor): Sets the input tensor for the model.
        """
        # This is usually handled in schedules.py but some inference code still
        # gives us non-lists or None
        if not isinstance(input_tensor, list):
            input_tensor = [input_tensor]

        assert len(input_tensor) == 1, "input_tensor should only be length 1 for gpt/bert"
        self.decoder.set_input_tensor(input_tensor[0])

    def sharded_state_dict(
        self, prefix: str = "module.", sharded_offsets: tuple = (), metadata: Optional[Dict] = None
    ) -> ShardedStateDict:
        """Sharded state dict implementation for GPTModel backward-compatibility (removing extra state).

        Args:
            prefix (str): Module name prefix.
            sharded_offsets (tuple): PP related offsets, expected to be empty at this module level.
            metadata (Optional[Dict]): metadata controlling sharded state dict creation.

        Returns:
            ShardedStateDict: sharded state dict for the GPTModel
        """
        sharded_state_dict = super().sharded_state_dict(prefix, sharded_offsets, metadata)

        # Ensure replica ids for non-transformer embedder weights include pipeline dimension
        for module in ["text_embedding", "time_embedding", "time_projection"]:
            if hasattr(self, module):
                for param_name, param in getattr(self, module).named_parameters():
                    weight_key = f"{prefix}{module}.{param_name}"
                    if weight_key in sharded_state_dict:
                        self._set_embedder_weights_replica_id(param, sharded_state_dict, weight_key)

        return sharded_state_dict

    def _mark_trainable_params_for_tp_grad_avg(self, modules: Optional[list] = None) -> None:
        """Mark selected modules' trainable parameters to average gradients across TP domain."""
        target_modules = modules if modules is not None else [self]
        for module in target_modules:
            for _name, param in module.named_parameters(recurse=True):
                if isinstance(param, nn.Parameter) and param.requires_grad:
                    setattr(param, "average_gradients_across_tp_domain", True)

    def _set_embedder_weights_replica_id(
        self, tensor: Tensor, sharded_state_dict: ShardedStateDict, embedder_weight_key: str
    ) -> None:
        """set replica ids of the weights in t_embedder for sharded state dict.

        Args:
            sharded_state_dict (ShardedStateDict): state dict with the weight to tie
            weight_key (str): key of the weight in the state dict.
                This entry will be replaced with a tied version

        Returns: None, acts in-place
        """
        tp_rank = parallel_state.get_tensor_model_parallel_rank()
        vpp_rank = parallel_state.get_virtual_pipeline_model_parallel_rank()
        vpp_rank = vpp_rank if vpp_rank else 0
        vpp_world = parallel_state.get_virtual_pipeline_model_parallel_world_size()
        vpp_world = vpp_world if vpp_world else 1
        pp_rank = parallel_state.get_pipeline_model_parallel_rank()
        del sharded_state_dict[embedder_weight_key]
        replica_id = (
            tp_rank,
            (vpp_rank + pp_rank * vpp_world),
            parallel_state.get_data_parallel_rank(with_context_parallel=True),
        )

        sharded_state_dict[embedder_weight_key] = make_sharded_tensor_for_checkpoint(
            tensor=tensor,
            key=embedder_weight_key,
            replica_id=replica_id,
            allow_shape_mismatch=False,
        )

    def load_state_dict(self, state_dict, strict=True):
        """Load state dict with automatic handling of 'module.' prefix mismatch.

        This method handles the case where checkpoints saved with DistributedDataParallel
        have a 'module.' prefix that needs to be removed when loading.

        Args:
            state_dict (dict): The state dictionary to load
            strict (bool): Whether to strictly enforce that the keys match

        Returns:
            NamedTuple: with 'missing_keys' and 'unexpected_keys' fields
        """
        # Check if state_dict has 'module.' prefix but model doesn't
        has_module_prefix = any(k.startswith("module.") for k in state_dict.keys())
        if has_module_prefix:
            new_state_dict = {}
            for key, value in state_dict.items():
                new_key = key.replace("module.", "", 1) if key.startswith("module.") else key
                new_state_dict[new_key] = value
            state_dict = new_state_dict

        return super().load_state_dict(state_dict, strict=strict)
