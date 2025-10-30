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
from dataclasses import dataclass
from typing import Dict

from dfm.src.megatron.model.dit.dit_layer_spec import get_dit_adaln_block_with_transformer_engine_spec
from dfm.src.megatron.model.dit.dit_model import DiTCrossAttentionModel
import torch
from megatron.core import parallel_state
from megatron.core.transformer import ModuleSpec
from megatron.core.transformer.transformer_config import TransformerConfig
from dfm.src.megatron.model.dit.dit_utils import dynamic_import

from megatron.bridge.models.model_provider import ModelProviderMixin
from megatron.core.models.common.vision_module.vision_module import VisionModule

logger = logging.getLogger(__name__)


def dit_transformer_engine_layer_spec() -> ModuleSpec:
    """Create a Transformer Engine layer specification based on the provided config."""
    return get_dit_adaln_block_with_transformer_engine_spec()


def dit_forward_step(model, batch) -> torch.Tensor:
    return model(**batch)


def dit_data_step(module, dataloader_iter):
    batch = next(dataloader_iter)[0]
    batch = get_batch_on_this_cp_rank(batch)
    batch = {k: v.to(device="cuda", non_blocking=True) if torch.is_tensor(v) else v for k, v in batch.items()}
    batch["is_preprocessed"] = True  # assume data is preprocessed

    if ("seq_len_q" in batch) and ("seq_len_kv" in batch):
        cu_seqlens = batch["seq_len_q"].cumsum(dim=0).to(torch.int32)
        zero = torch.zeros(1, dtype=torch.int32, device="cuda")
        cu_seqlens = torch.cat((zero, cu_seqlens))

        cu_seqlens_kv = batch["seq_len_kv"].cumsum(dim=0).to(torch.int32)
        cu_seqlens_kv = torch.cat((zero, cu_seqlens_kv))

        batch["packed_seq_params"] = {
            "self_attention": PackedSeqParams(
                cu_seqlens_q=cu_seqlens,
                cu_seqlens_kv=cu_seqlens,
                qkv_format=module.qkv_format,
            ),
            "cross_attention": PackedSeqParams(
                cu_seqlens_q=cu_seqlens,
                cu_seqlens_kv=cu_seqlens_kv,
                qkv_format=module.qkv_format,
            ),
        }

    return batch


def get_batch_on_this_cp_rank(data: Dict):
    """Split the data for context parallelism."""
    from megatron.core import mpu

    cp_size = mpu.get_context_parallel_world_size()
    cp_rank = mpu.get_context_parallel_rank()

    t = 16
    if cp_size > 1:
        # cp split on seq_length, for video_latent, noise_latent and pos_ids
        assert t % cp_size == 0, "t must divisibly by cp_size"
        num_valid_tokens_in_ub = None
        if "loss_mask" in data and data["loss_mask"] is not None:
            num_valid_tokens_in_ub = data["loss_mask"].sum()

        for key, value in data.items():
            if (value is not None) and (key in ["video", "video_latent", "noise_latent", "pos_ids"]):
                if len(value.shape) > 5:
                    value = value.squeeze(0)
                B, C, T, H, W = value.shape
                if T % cp_size == 0:
                    # FIXME packed sequencing
                    data[key] = value.view(B, C, cp_size, T // cp_size, H, W)[:, :, cp_rank, ...].contiguous()
                else:
                    # FIXME packed sequencing
                    data[key] = value.view(B, C, T, cp_size, H // cp_size, W)[:, :, :, cp_rank, ...].contiguous()
        loss_mask = data["loss_mask"]
        data["loss_mask"] = loss_mask.view(loss_mask.shape[0], cp_size, loss_mask.shape[1] // cp_size)[
            :, cp_rank, ...
        ].contiguous()
        data["num_valid_tokens_in_ub"] = num_valid_tokens_in_ub

    return data


@dataclass
class DiTModelProvider(TransformerConfig, ModelProviderMixin[VisionModule]):
    """
    Config for DiT-XL model
    """

    crossattn_emb_size: int = 1024
    add_bias_linear: bool = False
    gated_linear_unit: bool = False

    num_layers: int = 28
    hidden_size: int = 1152
    max_img_h: int = 80
    max_img_w: int = 80
    max_frames: int = 34
    patch_spatial: int = 2
    num_attention_heads: int = 16
    layernorm_epsilon = 1e-6
    normalization = "RMSNorm"
    add_bias_linear = False
    qk_layernorm_per_head = True
    layernorm_zero_centered_gamma = False

    fp16_lm_cross_entropy: bool = False
    parallel_output: bool = True
    share_embeddings_and_output_weights: bool = True

    # max_position_embeddings: int = 5400
    hidden_dropout: float = 0
    attention_dropout: float = 0

    bf16: bool = True
    params_dtype: torch.dtype = torch.bfloat16
    vae_module: str = "megatron.bridge.models.DiTModel.diffusers_vae.AutoencoderKLVAE"
    vae_path: str = None
    sigma_data: float = 0.5

    in_channels: int = 16

    # remove these 2 parameters
    # data_step_fn = dit_data_step
    # forward_step_fn = dit_forward_step

    replicated_t_embedder = True
    qkv_format: str = 'sbhd'

    seq_length: int = 2048
    vocab_size: int = None
    make_vocab_size_divisible_by: int = 128


    def provide(self, pre_process=None, post_process=None, vp_stage=None) -> DiTCrossAttentionModel:
        vp_size = self.virtual_pipeline_model_parallel_size
        if vp_size:
            p_size = self.pipeline_model_parallel_size
            assert (self.num_layers // p_size) % vp_size == 0, (
                "Make sure the number of model chunks is the same across all pipeline stages."
            )

        model = DiTCrossAttentionModel

        return model(
            self,
            fp16_lm_cross_entropy=self.fp16_lm_cross_entropy,
            parallel_output=self.parallel_output,
            pre_process=parallel_state.is_pipeline_first_stage(),
            post_process=parallel_state.is_pipeline_last_stage(),
            max_img_h=self.max_img_h,
            max_img_w=self.max_img_w,
            max_frames=self.max_frames,
            patch_spatial=self.patch_spatial,
        )

    def configure_vae(self):
        return dynamic_import(self.vae_module)(self.vae_path)


# Add all the DIT configs here like DIT7B, 14B, cosmos, etc, etc,
# @dataclass
# class GPTProvider126M(GPTModelProvider):
#     """Configuration for a 126M parameter GPT model.

#     Predefined configuration for a small GPT model with 12 layers,
#     768 hidden size, and 12 attention heads.
#     """

#     seq_length: int = 2048
#     num_layers: int = 12
#     hidden_size: int = 768
#     ffn_hidden_size: int = 3072
#     num_attention_heads: int = 12
#     bias_activation_fusion: bool = True
#     bias_dropout_add_fusion: bool = True


# @dataclass
# class GPTProvider5B(GPTModelProvider):
#     """Configuration for a 5B parameter GPT model.

#     Predefined configuration for a medium-sized GPT model with 24 layers,
#     4096 hidden size, and 32 attention heads.
#     """

#     seq_length: int = 2048
#     num_layers: int = 24
#     hidden_size: int = 4096
#     ffn_hidden_size: int = 16384
#     num_attention_heads: int = 32
#     bias_activation_fusion: bool = True
#     bias_dropout_add_fusion: bool = True


# @dataclass
# class GPTProvider7B(GPTModelProvider):
#     """Configuration for a 7B parameter GPT model.

#     Predefined configuration for a medium-sized GPT model with 32 layers,
#     4096 hidden size, and 32 attention heads.
#     """

#     seq_length: int = 2048
#     num_layers: int = 32
#     hidden_size: int = 4096
#     ffn_hidden_size: int = 10880
#     num_attention_heads: int = 32
#     bias_activation_fusion: bool = True
#     bias_dropout_add_fusion: bool = True


# @dataclass
# class GPTProvider20B(GPTModelProvider):
#     """Configuration for a 20B parameter GPT model.

#     Predefined configuration for a large GPT model with 44 layers,
#     6144 hidden size, and 48 attention heads.
#     """

#     seq_length: int = 2048
#     num_layers: int = 44
#     hidden_size: int = 6144
#     ffn_hidden_size: int = 24576
#     num_attention_heads: int = 48
#     bias_activation_fusion: bool = True
#     bias_dropout_add_fusion: bool = True


# @dataclass
# class GPTProvider40B(GPTModelProvider):
#     """Configuration for a 40B parameter GPT model.

#     Predefined configuration for a large GPT model with 48 layers,
#     8192 hidden size, and 64 attention heads.
#     """

#     seq_length: int = 2048
#     num_layers: int = 48
#     hidden_size: int = 8192
#     ffn_hidden_size: int = 32768
#     num_attention_heads: int = 64
#     bias_activation_fusion: bool = True
#     bias_dropout_add_fusion: bool = True


# @dataclass
# class GPTProvider175B(GPTModelProvider):
#     """Configuration for a 175B parameter GPT model.

#     Predefined configuration for a massive GPT model with 96 layers,
#     12288 hidden size, and 96 attention heads.
#     """

#     seq_length: int = 2048
#     num_layers: int = 96
#     hidden_size: int = 12288
#     ffn_hidden_size: int = 49152
#     num_attention_heads: int = 96
#     hidden_dropout: float = 0.0
#     attention_dropout: float = 0.0
#     bias_activation_fusion: bool = True
#     bias_dropout_add_fusion: bool = True
#     layernorm_zero_centered_gamma: bool = True