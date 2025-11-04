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
from typing import Callable, Dict

from dfm.src.megatron.model.dit.dit_layer_spec import get_dit_adaln_block_with_transformer_engine_spec
from dfm.src.megatron.model.dit.dit_model import DiTCrossAttentionModel
import torch
from megatron.core import parallel_state
from megatron.core.transformer import ModuleSpec
from megatron.core.transformer.transformer_config import TransformerConfig
from dfm.src.common.utils.dynamic_import import dynamic_import

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
    add_bias_linear: bool = False
    qk_layernorm_per_head: bool = True
    layernorm_zero_centered_gamma: bool = False

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


@dataclass
class DiT7BModelProvider(DiTModelProvider):
    hidden_size: int = 4096
    max_img_h: int = 240
    max_img_w: int = 240
    max_frames: int = 128
    num_attention_heads: int = 32
    apply_rope_fusion: bool = True        # TODO: do we support this?
    additional_timestamp_channels = None  # TODO: do we support this?
    bf16: bool = True
    vae_module: str = "nemo.collections.diffusion.vae.video_vae.video_vae3_512"
    vae_path: str = None


@dataclass
class DiT14BModelProvider(DiTModelProvider):
    num_layers: int = 36
    hidden_size: int = 5120
    crossattn_emb_size: int = 1024
    max_img_h: int = 240
    max_img_w: int = 240
    max_frames: int = 128
    num_attention_heads: int = 40
    apply_rope_fusion: bool = True
    layernorm_zero_centered_gamma: bool = False
    additional_timestamp_channels = None
    bf16: bool = True
    vae_module: str = "nemo.collections.diffusion.vae.video_vae.video_vae3_512"
    vae_path: str = None
    loss_add_logvar: bool = True


@dataclass
class DiTLlama30BConfig(DiTModelProvider):
    num_layers: int = 48
    hidden_size: int = 6144
    ffn_hidden_size: int = 16384
    num_attention_heads: int = 48
    num_query_groups: int = 8
    gated_linear_unit: int = True
    bias_activation_fusion: int = True
    activation_func: Callable = torch.nn.functional.silu
    layernorm_epsilon: float = 1e-5
    max_frames: int = 128
    max_img_h: int = 240
    max_img_w: int = 240
    init_method_std: float = 0.01
    add_bias_linear: bool = False
    seq_length: int = 256
    masked_softmax_fusion: bool = True
    persist_layer_norm: bool = True
    bias_dropout_fusion: bool = True
