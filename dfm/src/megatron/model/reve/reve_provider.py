# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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


import logging
from dataclasses import dataclass, field
from typing import Callable

import torch
import torch.nn.functional as F
from megatron.bridge.models.model_provider import ModelProviderMixin
from megatron.bridge.models.transformer_config import TransformerConfig
from megatron.core import parallel_state
from megatron.core.models.common.vision_module.vision_module import VisionModule

from dfm.src.megatron.model.reve.reve_model import ReveModel


logger = logging.getLogger(__name__)


@dataclass
class ReveModelProvider(TransformerConfig, ModelProviderMixin[VisionModule]):

    num_layers: int = 26
    hidden_size: int = 6144
    ffn_hidden_size: int = 24576
    num_attention_heads: int = 24
    layernorm_epsilon: float = 1e-6
    normalization: str = "RMSNorm"
    layernorm_zero_centered_gamma: bool = False
    layernorm_across_heads: bool = False
    rotary_interleaved: bool = True
    activation_func: Callable = F.silu
    hidden_dropout: float = 0
    attention_dropout: float = 0
    fp16_lm_cross_entropy: bool = False
    parallel_output: bool = True
    bf16: bool = False
    params_dtype: torch.dtype = torch.float32
    qkv_format: str = "thd"  # "sbhd". NOTE: if we use context parallelism, we need to use "thd"
    apply_rope_fusion: bool = True
    # these attributes are unused for images/videos, we just set because bridge training requires for LLMs
    seq_length: int = 1024
    share_embeddings_and_output_weights: bool = False
    vocab_size: int = 25256 * 8
    make_vocab_size_divisible_by: int = 128

    # Reve model's specific architecture
    add_bias_linear: bool = False
    add_qkv_bias: bool = False
    gated_linear_unit: bool = False
    bias_activation_fusion: bool = False # M-LM: Only support fusion of gelu and swiglu
    softmax_scale: float = 16.0

    # Reve model text transformer block
    text_dims: int = 4096
    cross_dims_per_head: int = 256
    cross_num_heads: int = 24
    cross_num_layers: int = 8
    rope_max_wavelength: float = 512.0
    patch_size: int = 1
    patch_spatial: int = 1 # patch_spatial is the same as patch_size
    patch_temporal: int = 1 # patch_temporal is the same as patch_size
    latent_dims: int = 768
    in_channels: int = 768 # in_channels is the same as latent_dims
    rope_dims: list[int] = field(default_factory=lambda: [64, 64])
    cross_rope_dims: int = 128

    def provide(self, pre_process=None, post_process=None, vp_stage=None) -> ReveModel:
        vp_size = self.virtual_pipeline_model_parallel_size
        if vp_size:
            p_size = self.pipeline_model_parallel_size
            assert (self.num_layers // p_size) % vp_size == 0, (
                "Make sure the number of model chunks is the same across all pipeline stages."
            )

        model = ReveModel

        return model(
            self,
            pre_process=parallel_state.is_pipeline_first_stage(),
            post_process=parallel_state.is_pipeline_last_stage(),
            fp16_lm_cross_entropy=self.fp16_lm_cross_entropy,
            parallel_output=self.parallel_output,
        )


@dataclass
class ReveSmallModelProvider(ReveModelProvider):
    num_layers: int = 2
    hidden_size: int = 256
    ffn_hidden_size: int = 1024
    num_attention_heads: int = 4
    text_dims: int = 128
    cross_dims_per_head: int = 64
    cross_num_heads: int = 4
    cross_num_layers: int = 2
    rope_max_wavelength: float = 512.0
    patch_size: int = 1
    patch_spatial: int = 1 # patch_spatial is the same as patch_size
    patch_temporal: int = 1 # patch_temporal is the same as patch_size
    latent_dims: int = 16
    in_channels: int = 16 # in_channels is the same as latent_dims
    rope_dims: list[int] = field(default_factory=lambda: [16, 16])
    cross_rope_dims: int = 32


@dataclass
class ReveFullModelProvider(ReveModelProvider):
    num_layers: int = 26
    hidden_size: int = 6144
    ffn_hidden_size: int = 24576
    num_attention_heads: int = 24
    text_dims: int = 4096
    cross_dims_per_head: int = 256
    cross_num_heads: int = 24
    cross_num_layers: int = 8
    rope_max_wavelength: float = 512.0
    patch_size: int = 1
    patch_spatial: int = 1 # patch_spatial is the same as patch_size
    patch_temporal: int = 1 # patch_temporal is the same as patch_size
    latent_dims: int = 768
    in_channels: int = 768 # in_channels is the same as latent_dims
    rope_dims: list[int] = field(default_factory=lambda: [64, 64])
    cross_rope_dims: int = 128


@dataclass
class ReveHalfFullModelProvider(ReveFullModelProvider):
    num_layers: int = 13
    cross_num_layers: int = 4

@dataclass
class Reve1BModelProvider(ReveFullModelProvider):
    num_layers: int = 13
    hidden_size: int = 2048
    ffn_hidden_size: int = 8192
    num_attention_heads: int = 8
    cross_num_layers: int = 4
    cross_num_heads: int = 8
