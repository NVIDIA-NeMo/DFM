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

from typing import Dict, Literal, Optional
import os

import torch
import torch.nn as nn
from diffusers.models.embeddings import Timesteps
from einops import rearrange
from megatron.core import parallel_state, tensor_parallel
from megatron.core.dist_checkpointing.mapping import ShardedStateDict
from megatron.core.models.common.vision_module.vision_module import VisionModule
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.transformer.enums import ModelType
from megatron.core.transformer.transformer_block import TransformerBlock
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.utils import make_sharded_tensor_for_checkpoint
from torch import Tensor

from dfm.src.megatron.model.reve.reve_layer_spec import (
    RMSNorm,
)
from dfm.src.megatron.model.reve.reve_layer_spec import (
    get_reve_adaln_block_with_transformer_engine_spec as ReveLayerWithAdaLN,
    get_reve_adaln_text_block_with_transformer_engine_spec as ReveTextLayerWithAdaLN,
)


def cis_embed(
    x: torch.Tensor, dims: int, max_wavelength: float, scale: float
) -> tuple[torch.Tensor, torch.Tensor]:
    compute_dtype = torch.float64
    output_dtype = torch.float32

    a = torch.div(
        torch.tensor(1.0, device=x.device, dtype=compute_dtype),
        torch.tensor(max_wavelength, device=x.device, dtype=compute_dtype),
    )
    exp = torch.div(
        torch.arange(dims, dtype=compute_dtype, device=x.device),
        torch.tensor(dims, device=x.device, dtype=compute_dtype),
    )
    freqs = torch.pow(a, exp)
    x = torch.mul(
        x.to(dtype=compute_dtype),
        torch.tensor(scale, dtype=compute_dtype, device=x.device),
    )
    out = torch.einsum("...,d->...d", x, freqs)
    return torch.cos(out).to(dtype=output_dtype), torch.sin(out).to(dtype=output_dtype)


class RoPEEmbed(nn.Module):
    def __init__(
        self, dims_per_axis: list[int], max_wavelength: float, scale: float = 1.0
    ):
        super().__init__()

        for dims in dims_per_axis:
            assert dims % 2 == 0, "dims must be even for RoPE cos/sin embeddings"
        self.dims_per_axis = dims_per_axis

        assert max_wavelength > 0, "max_wavelength must be positive"
        self.max_wavelength = max_wavelength

        self.cis_embed = cis_embed
        self.scale = scale

    def forward(
        self, pos_indices: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert pos_indices.shape[-1] == len(self.dims_per_axis), (
            f"number of dims ({pos_indices.shape[-1]}) must match `dims_per_axis` ({len(self.dims_per_axis)})"
        )
        all_cos = []
        all_sin = []
        for axis, dims in enumerate(self.dims_per_axis):
            assert dims % 2 == 0, "dims must be even"
            pos_idx = pos_indices[..., axis]
            cos, sin = self.cis_embed(
                pos_idx, dims // 2, self.max_wavelength, self.scale
            )
            all_cos.append(cos)
            all_sin.append(sin)

        return torch.concat(all_cos * 2, dim=-1), torch.concat(all_sin * 2, dim=-1)

    def forward_interleaved(
        self, pos_indices: torch.Tensor
    ) -> torch.Tensor:
        assert pos_indices.shape[-1] == len(self.dims_per_axis), (
            f"number of dims ({pos_indices.shape[-1]}) must match `dims_per_axis` ({len(self.dims_per_axis)})"
        )
        all_freqs = []
        compute_dtype = torch.float64
        output_dtype = torch.float32

        for axis, dims in enumerate(self.dims_per_axis):
            assert dims % 2 == 0, "dims must be even"
            pos_idx = pos_indices[..., axis]
            
            # Reimplementing frequency calculation from cis_embed to get angles directly
            half_dims = dims // 2
            a = torch.div(
                torch.tensor(1.0, device=pos_idx.device, dtype=compute_dtype),
                torch.tensor(self.max_wavelength, device=pos_idx.device, dtype=compute_dtype),
            )
            exp = torch.div(
                torch.arange(half_dims, dtype=compute_dtype, device=pos_idx.device),
                torch.tensor(half_dims, device=pos_idx.device, dtype=compute_dtype),
            )
            base_freqs = torch.pow(a, exp)
            
            # Apply scale to pos_idx before multiplying
            scaled_pos = torch.mul(
                pos_idx.to(dtype=compute_dtype),
                torch.tensor(self.scale, dtype=compute_dtype, device=pos_idx.device),
            )
            
            # Calculate angles: pos * base_freqs
            angles = torch.einsum("...,d->...d", scaled_pos, base_freqs)
            all_freqs.append(angles.to(dtype=output_dtype))

        full_freqs = torch.concat(all_freqs, dim=-1)
        # Interleave frequencies: [f1, f1, f2, f2, ...]
        return torch.repeat_interleave(full_freqs, 2, dim=-1)

    def forward_half_split(
        self, pos_indices: torch.Tensor
    ) -> torch.Tensor:
        assert pos_indices.shape[-1] == len(self.dims_per_axis), (
            f"number of dims ({pos_indices.shape[-1]}) must match `dims_per_axis` ({len(self.dims_per_axis)})"
        )
        all_freqs = []
        compute_dtype = torch.float64
        output_dtype = torch.float32

        for axis, dims in enumerate(self.dims_per_axis):
            pos_idx = pos_indices[..., axis]
            
            # 1. Calculate base frequencies for this axis
            half_dims = dims // 2
            a = torch.tensor(1.0 / self.max_wavelength, device=pos_idx.device, dtype=compute_dtype)
            exp = torch.arange(half_dims, dtype=compute_dtype, device=pos_idx.device) / half_dims
            base_freqs = torch.pow(a, exp)
            
            # 2. Scale positions and compute angles (theta)
            scaled_pos = pos_idx.to(dtype=compute_dtype) * self.scale
            angles = torch.einsum("...,d->...d", scaled_pos, base_freqs)
            
            all_freqs.append(angles.to(dtype=output_dtype))

        # 3. Concatenate all axes into one sequence of frequencies
        # If axes are X and Y, full_freqs is [X_freqs, Y_freqs]
        full_freqs = torch.concat(all_freqs, dim=-1)

        # 4. MATCH THE HALF-SPLIT PATTERN
        # Instead of repeat_interleave (which does [f1, f1, f2, f2]),
        # we concatenate the block with itself to get [f1, f2, f1, f2].
        return torch.concat([full_freqs, full_freqs], dim=-1)

class CosineEmbed(nn.Module):
    def __init__(self, dims: int, max_wavelength: float, scale: float):
        super().__init__()
        assert dims % 2 == 0
        self.dims = dims
        self.max_wavelength = max_wavelength
        self.scale = scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        cos, sin = cis_embed(x, self.dims // 2, self.max_wavelength, self.scale)
        return torch.concat([cos, sin], dim=-1)


class MLPEmbed(nn.Module):
    def __init__(self, in_dims: int, out_dims: int, apply_rms_norm: bool = False):
        super().__init__()
        self.in_layer = nn.Linear(in_dims, out_dims, bias=True)
        self.act = nn.SiLU()
        self.out_layer = nn.Linear(out_dims, out_dims, bias=True)
        self.apply_rms_norm = apply_rms_norm
        if self.apply_rms_norm:
            self.norm = RMSNorm(out_dims)

    def forward(
        self, x: torch.Tensor
    ) -> torch.Tensor:
        with torch.autocast(device_type="cuda", enabled=False):
            out = self.out_layer(self.act(self.in_layer(x)))
            if self.apply_rms_norm:
                out = self.norm(out)
        return out


class ReveModel(VisionModule):
    """
    ReveModel is a VisionModule that implements a Reve model.
    Attributes:
        config (TransformerConfig): Configuration for the transformer.
        text_config (TransformerConfig): Configuration for the text transformer.
        pre_process (bool): Whether to apply pre-processing steps.
        post_process (bool): Whether to apply post-processing steps.
        fp16_lm_cross_entropy (bool): Whether to use fp16 for cross-entropy loss.
        parallel_output (bool): Whether to use parallel output.
        position_embedding_type (Literal["learned_absolute", "rope"]): Type of position embedding.
        max_img_h (int): Maximum image height.
        max_img_w (int): Maximum image width.
        max_frames (int): Maximum number of frames.
        patch_spatial (int): Spatial patch size.
        patch_temporal (int): Temporal patch size.
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        transformer_decoder_layer_spec (ReveLayerWithAdaLN): Specification for the transformer decoder layer.
        transformer_text_decoder_layer_spec (ReveTextLayerWithAdaLN): Specification for the transformer text decoder layer.
        add_encoder (bool): Whether to add an encoder.
        add_decoder (bool): Whether to add a decoder.
        share_embeddings_and_output_weights (bool): Whether to share embeddings and output weights.
        concat_padding_mask (bool): Whether to concatenate padding mask.
        pos_emb_cls (str): Class of position embedding.
        model_type (ModelType): Type of the model.
        decoder (TransformerBlock): Transformer decoder block.
        t_embedder (torch.nn.Sequential): Time embedding layer.
        x_embedder (nn.Conv3d): Convolutional layer for input embedding.
        pos_embedder (SinCosPosEmb3D): Position embedding layer.
        final_layer_linear (torch.nn.Linear): Final linear layer.
        affline_norm (RMSNorm): Affine normalization layer.
    Methods:
        forward(x: Tensor, timesteps: Tensor, crossattn_emb: Tensor, packed_seq_params: PackedSeqParams = None, pos_ids: Tensor = None, **kwargs) -> Tensor:
            Forward pass of the model.
        set_input_tensor(input_tensor: Tensor) -> None:
            Sets input tensor to the model.
        sharded_state_dict(prefix: str = 'module.', sharded_offsets: tuple = (), metadata: Optional[Dict] = None) -> ShardedStateDict:
            Sharded state dict implementation for backward-compatibility.
        tie_embeddings_weights_state_dict(tensor, sharded_state_dict: ShardedStateDict, output_layer_weight_key: str, first_stage_word_emb_key: str) -> None:
            Ties the embedding and output weights in a given sharded state dict.
    """

    def __init__(
        self,
        config: TransformerConfig,
        pre_process: bool = True,
        post_process: bool = True,
        fp16_lm_cross_entropy: bool = False,
        parallel_output: bool = True,
        position_embedding_type: Literal["learned_absolute", "rope"] = "rope",
        max_img_h: int = 80,
        max_img_w: int = 80,
        max_frames: int = 34,
        patch_spatial: int = 1,
        patch_temporal: int = 1,
        in_channels: int = 16,
        out_channels: int = 16,
        transformer_decoder_layer_spec=ReveLayerWithAdaLN,
        transformer_text_decoder_layer_spec=ReveTextLayerWithAdaLN,
        pos_embedder=None,
        **kwargs,
    ):
        super(ReveModel, self).__init__(config=config)

        self.config: TransformerConfig = config
        self.transformer_decoder_layer_spec = transformer_decoder_layer_spec()
        self.transformer_text_decoder_layer_spec = transformer_text_decoder_layer_spec()
        self.pre_process = pre_process
        self.post_process = post_process
        self.add_encoder = True
        self.add_decoder = True
        self.fp16_lm_cross_entropy = fp16_lm_cross_entropy
        self.parallel_output = parallel_output
        self.position_embedding_type = position_embedding_type
        self.share_embeddings_and_output_weights = False
        self.concat_padding_mask = True
        self.pos_emb_cls = "sincos"
        self.patch_spatial = patch_spatial
        self.patch_temporal = patch_temporal

        # megatron core pipelining currently depends on model type
        # TODO: remove this dependency ?
        self.model_type = ModelType.encoder_or_decoder


        self.config.cross_model_dims = self.config.cross_dims_per_head * self.config.cross_num_heads
        self.config.dims_per_head = self.config.hidden_size // self.config.num_attention_heads


        # Text processing
        self.txt_in_norm: nn.Module = RMSNorm(self.config.cross_model_dims)
        self.txt_in = nn.Linear(self.config.text_dims, self.config.cross_model_dims, bias=True)
        self.txt_out = nn.Linear(self.config.cross_model_dims, self.config.hidden_size, bias=True)
        self.txt_rope_emb = RoPEEmbed(
            dims_per_axis=[self.config.cross_dims_per_head - self.config.cross_rope_dims, self.config.cross_rope_dims],
            max_wavelength=self.config.rope_max_wavelength,
        )

        # Image processing
        self.img_in_norm: nn.Module = RMSNorm(self.config.hidden_size)
        self.img_in = nn.Linear(self.config.latent_dims * self.config.patch_size**2, self.config.hidden_size, bias=True)
        self.img_rope_emb = RoPEEmbed(
            dims_per_axis=[self.config.dims_per_head - sum(self.config.rope_dims), *self.config.rope_dims],
            max_wavelength=self.config.rope_max_wavelength,
        )


        # Timestep and conditioning signal processing
        self.time_cos_embed = CosineEmbed(
            dims=self.config.dims_per_head, max_wavelength=10_000, scale=1000
        )
        self.time_mlp_embed = MLPEmbed(in_dims=self.config.dims_per_head, out_dims=self.config.hidden_size)
        self.conditioning_signal_cos_embed = CosineEmbed(
            dims=self.config.dims_per_head, max_wavelength=10_000, scale=1000
        )
        self.conditioning_signal_mlp_embed = MLPEmbed(
            in_dims=self.config.dims_per_head, out_dims=self.config.hidden_size
        )


        # Transformer textdecoder
        import copy
        self.text_config = copy.deepcopy(self.config)
        self.text_config.hidden_size = self.config.cross_model_dims
        self.text_config.num_attention_heads = self.config.cross_num_heads
        self.text_config.num_layers = self.config.cross_num_layers
        # # DEBUGGING (match forward pass in reve_pytorch/model.py)
        # if torch.distributed.get_rank() == 0:
        #     print(f"[DEBUG] (reve_model.py) text_config: {self.text_config}")
        self.text_decoder = TransformerBlock(
            config=self.text_config,
            spec=self.transformer_text_decoder_layer_spec,
            pre_process=self.pre_process,
            post_process=False,
            post_layer_norm=False,
        )


        # Transformer image decoder
        self.decoder = TransformerBlock(
            config=self.config,
            spec=self.transformer_decoder_layer_spec,
            pre_process=self.pre_process,
            post_process=False,
            post_layer_norm=False,
        )

        # Final layer
        self.final_layer = nn.Linear(self.config.hidden_size, self.config.latent_dims * self.config.patch_size**2, bias=True)


        # DEBUGGING (match forward pass in reve_pytorch/model.py)
        # Loading from "synced_checkpoints/reve_mcore_init.pt"
        if os.path.exists("synced_checkpoints/reve_mcore_init.pt"):
             if torch.distributed.get_rank() == 0:
                print(f"Loading weights from synced_checkpoints/reve_mcore_init.pt")
             state_dict = torch.load("synced_checkpoints/reve_mcore_init.pt", map_location="cpu")
             self.load_state_dict(state_dict, strict=True)
             # Convert to bfloat16
             self.to(torch.bfloat16)

             # Print model's parameters and their shapes, mean, std
             for name, param in self.named_parameters():
                    print(f"{name}: {param.shape}, dtype: {param.dtype}, mean: {param.mean()}, std: {param.std()}")
             # Also print total number of parameters
             total_params = sum(param.numel() for name, param in self.named_parameters())
             print(f"Total Parameters: {total_params}")
            #  print(stop_here)

        # DEBUGGING
        if torch.distributed.get_rank() == 0:
            total_params = 0
            print(f"\n{'='*20} Model Parameters {'='*20}")
            for name, param in self.named_parameters():
                print(f"{name}: {param.shape}")
                total_params += param.numel()
            print(f"Total Parameters: {total_params}")
            print(f"{'='*58}\n")



    def _compute_modulation_vector(
        self,
        dtype: torch.dtype,
        timestep: torch.Tensor,
        conditioning_signal: torch.Tensor,
    ) -> torch.Tensor:
        vector_emb = self.time_mlp_embed(
            self.time_cos_embed(timestep.float()).to(dtype=dtype)
        )
        conditioning_signal_emb = self.conditioning_signal_mlp_embed(
            self.conditioning_signal_cos_embed(conditioning_signal.float()).to(
                dtype=dtype
            )
        )
        vector_emb = vector_emb + conditioning_signal_emb

        return vector_emb


    def forward(
        self,
        number_packed_samples: int,
        x: torch.Tensor, # (number_packed_samples * img_seq_len, 1, hidden_size)
        x_position_ids: torch.Tensor, # (number_packed_samples, img_seq_len, 3)
        y: torch.Tensor, # (number_packed_samples * text_seq_len, 1, hidden_size)
        text_seq_len: int,
        timestep: torch.Tensor, # (1) - assumed to be the same for all the number_packed_samples
        conditioning_signal: torch.Tensor, # (1) - assumed to be the same for all the number_packed_samples
        packed_seq_params: PackedSeqParams = None,
        text_packed_seq_params: PackedSeqParams = None,
        **kwargs,
    ) -> Tensor:
        """Forward pass.
        """

        # # DEBUGGING (match forward pass in reve_pytorch/model.py)
        # # Mock values
        # device = self.img_in.weight.device
        # dtype = self.img_in.weight.dtype
        # number_packed_samples = 2
        # img_seq_len = 256
        # text_seq_len = 256
        # img_token_dim = self.img_in.in_features
        # txt_token_dim = self.txt_in.in_features
        # generator = torch.Generator(device=device)
        # generator.manual_seed(42)
        # x = torch.randn(
        #     number_packed_samples,
        #     img_seq_len,
        #     img_token_dim,
        #     dtype=dtype,
        #     device=device,
        #     generator=generator,
        # ).reshape(number_packed_samples * img_seq_len, 1, img_token_dim)
        # x_position_ids = torch.ones(
        #     number_packed_samples, img_seq_len, 3, dtype=dtype, device=device
        # )
        # timestep = torch.tensor([0.5], dtype=dtype, device=device)
        # y = torch.randn(
        #     number_packed_samples, text_seq_len, txt_token_dim, dtype=dtype, device=device, generator=generator,
        # ).reshape(number_packed_samples * text_seq_len, 1, txt_token_dim)
        # conditioning_signal = torch.tensor([0.7], dtype=dtype, device=device)
        # zero = torch.zeros(1, dtype=torch.int32, device=device)
        # seq_len_q = torch.tensor([img_seq_len] * number_packed_samples, device=device)
        # cu_seqlens_q = seq_len_q.cumsum(dim=0).to(torch.int32)
        # cu_seqlens_q = torch.cat((zero, cu_seqlens_q))
        # seq_len_kv = torch.tensor([text_seq_len] * number_packed_samples, device=device)
        # cu_seqlens_kv = seq_len_kv.cumsum(dim=0).to(torch.int32)
        # cu_seqlens_kv = torch.cat((zero, cu_seqlens_kv))
        # text_packed_seq_params = {
        #     "self_attention": PackedSeqParams(
        #         cu_seqlens_q=cu_seqlens_kv,
        #         cu_seqlens_q_padded=cu_seqlens_kv,
        #         cu_seqlens_kv=cu_seqlens_kv,
        #         cu_seqlens_kv_padded=cu_seqlens_kv,
        #         qkv_format='thd',
        #     ),
        # }
        # packed_seq_params = {
        #     "self_attention": PackedSeqParams(
        #         cu_seqlens_q=cu_seqlens_q,
        #         cu_seqlens_q_padded=cu_seqlens_q,
        #         cu_seqlens_kv=cu_seqlens_q,
        #         cu_seqlens_kv_padded=cu_seqlens_q,
        #         qkv_format='thd',
        #     ),
        #     "cross_attention": PackedSeqParams(
        #         cu_seqlens_q=cu_seqlens_q,
        #         cu_seqlens_q_padded=cu_seqlens_q,
        #         cu_seqlens_kv=cu_seqlens_kv,
        #         cu_seqlens_kv_padded=cu_seqlens_kv,
        #         qkv_format='thd',
        #     ),
        # }



        # # DEBUGGING
        # if torch.distributed.get_rank() == 0:
        #     print(f"[DEBUG] (reve_model.py) number_packed_samples: {number_packed_samples}")
        #     print(f"[DEBUG] (reve_model.py) x.shape: {x.shape}")
        #     # print(f"[DEBUG] (reve_model.py) x.mean() - x.std(): {x.mean()} - {x.std()}")
        #     # print(f"[DEBUG] (reve_model.py) x_position_ids: {x_position_ids.shape}")
        #     # print(f"[DEBUG] (reve_model.py) x_position_ids.mean() - x_position_ids.std(): {x_position_ids.mean()} - {x_position_ids.std()}")
        #     print(f"[DEBUG] (reve_model.py) y.shape: {y.shape}")
        #     # print(f"[DEBUG] (reve_model.py) y.mean() - y.std(): {y.mean()} - {y.std()}")
        #     # print(f"[DEBUG] (reve_model.py) text_seq_len: {text_seq_len}")
        #     # print(f"[DEBUG] (reve_model.py) timestep: {timestep}")
        #     # print(f"[DEBUG] (reve_model.py) conditioning_signal: {conditioning_signal}")
        #     # print(f"[DEBUG] (reve_model.py) text_packed_seq_params: {text_packed_seq_params}")
        #     # print(f"[DEBUG] (reve_model.py) packed_seq_params: {packed_seq_params}")
        #     print(f"[DEBUG] (reve_model.py) --------------------------------")

        ### Text processing
        txt = y
        # we ignore masking, those are taken care of by the packed_seq_params
        # txt_mask = y_mask.bool()
        # txt, txt_mask = self.pad_tokens(txt, txt_mask)
        txt = self.txt_in(txt)
        txt = self.txt_in_norm(txt)
        # rope embeddings
        txt_pos_ids = torch.arange(text_seq_len, dtype=torch.float32, device=txt.device)
        txt_nopos_ids = torch.zeros_like(txt_pos_ids)
        txt_pos_ids = torch.stack([txt_nopos_ids, txt_pos_ids], dim=-1)[None]
        txt_pos_ids = txt_pos_ids.repeat(number_packed_samples, 1, 1)
        # txt_rope_cis = self.txt_rope_emb.forward_interleaved(txt_pos_ids)
        txt_rope_cis = self.txt_rope_emb.forward_half_split(txt_pos_ids)
        txt_rope_cis = txt_rope_cis.reshape([-1, 1, txt_rope_cis.shape[-1]]) # (number_packed_samples, text_seq_len, hidden_size) -> ((number_packed_samples * text_seq_len), 1, hidden_size)
        txt_rope_cis = txt_rope_cis.unsqueeze(1) # ((number_packed_samples * text_seq_len), 1, hidden_size) -> (number_packed_samples, 1, 1, hidden_size)

        # # DEBUGGING (match forward pass in reve_pytorch/model.py)
        # if torch.distributed.get_rank() == 0:
        #     print(f"[DEBUG] (reve_model.py - before transformer blocks) txt.shape - txt.mean() - txt.std() - txt.norm(): {txt.shape} - {txt.mean()} - {txt.std()} - {txt.norm()}")
        #     print(f"[DEBUG] (reve_model.py - before transformer blocks) txt_rope_cis.shape - txt_rope_cis.mean() - txt_rope_cis.std() - txt_rope_cis.norm(): {txt_rope_cis.shape} - {txt_rope_cis.mean()} - {txt_rope_cis.std()} - {txt_rope_cis.norm()}")
        #     print(f"[DEBUG] (reve_model.py - before transformer blocks) text_packed_seq_params: {text_packed_seq_params}")

        ### Text decoder
        txt = self.text_decoder(
            hidden_states=txt,
            attention_mask=None,
            context=None,
            context_mask=None,
            rotary_pos_emb=txt_rope_cis,
            packed_seq_params=text_packed_seq_params,
        )

        # # DEBUGGING (match forward pass in reve_pytorch/model.py)
        # if torch.distributed.get_rank() == 0:
        #     print(f"[DEBUG] (reve_model.py - after transformer blocks) txt.shape - txt.mean() - txt.std() - txt.norm(): {txt.shape} - {txt.mean()} - {txt.std()} - {txt.norm()}")

        txt = self.txt_out(txt)

        # # DEBUGGING (match forward pass in reve_pytorch/model.py)
        # if torch.distributed.get_rank() == 0:
        #     print(f"[DEBUG] (reve_model.py) *** Finished text transformer blocks processing.")

        ### Image processing
        patched_img, img_ids = x, x_position_ids
        # we ignore masking, those are taken care of by the packed_seq_params
        # x_mask = img_ids[..., 0] >= 0
        # patched_img, x_mask = self.pad_tokens(patched_img, x_mask)
        patched_img = self.img_in(patched_img)
        patched_img = self.img_in_norm(patched_img)
        # rope embeddings
        img_ids = img_ids.clone().float()
        position_scale = torch.ones((number_packed_samples, 1, 3), device=img_ids.device)
        position_scale[:, :, 0] = 1.0
        img_ids = img_ids * position_scale
        # img_rope_cis = self.img_rope_emb.forward_interleaved(img_ids)
        img_rope_cis = self.img_rope_emb.forward_half_split(img_ids)
        img_rope_cis = img_rope_cis.reshape([-1, 1, img_rope_cis.shape[-1]]) # ((number_packed_samples * img_seq_len), 1, hidden_size) -> ((number_packed_samples * img_seq_len), 1, hidden_size)
        img_rope_cis = img_rope_cis.unsqueeze(1) # ((number_packed_samples * img_seq_len), 1, hidden_size) -> (number_packed_samples, 1, 1, hidden_size)

        ### Timestep and conditioning signal processing
        condition_vector_emb = self._compute_modulation_vector(
            patched_img.dtype,
            timestep,
            conditioning_signal,
        )


        # # DEBUGGING (match forward pass in reve_pytorch/model.py)
        # if torch.distributed.get_rank() == 0:
        #     print(f"[DEBUG] (reve_model.py) patched_img.shape - patched_img.mean() - patched_img.std() - patched_img.norm(): {patched_img.shape} - {patched_img.mean()} - {patched_img.std()} - {patched_img.norm()}")
        #     print(f"[DEBUG] (reve_model.py) txt.shape - txt.mean() - txt.std() - txt.norm(): {txt.shape} - {txt.mean()} - {txt.std()} - {txt.norm()}")
        #     print(f"[DEBUG] (reve_model.py) condition_vector_emb.shape - condition_vector_emb.mean() - condition_vector_emb.std() - condition_vector_emb.norm(): {condition_vector_emb.shape} - {condition_vector_emb.mean()} - {condition_vector_emb.std()} - {condition_vector_emb.norm()}")
        #     print(f"[DEBUG] (reve_model.py) img_rope_cis.shape - img_rope_cis.mean() - img_rope_cis.std() - img_rope_cis.norm(): {img_rope_cis.shape} - {img_rope_cis.mean()} - {img_rope_cis.std()} - {img_rope_cis.norm()}")
        #     print(f"[DEBUG] (reve_model.py) packed_seq_params: {packed_seq_params}")

        ### Image decoder
        patched_img = self.decoder(
            hidden_states=patched_img,
            attention_mask=condition_vector_emb,
            context=txt,
            context_mask=None,
            rotary_pos_emb=img_rope_cis,
            packed_seq_params=packed_seq_params,
        )

        # # DEBUGGING (match forward pass in reve_pytorch/model.py)
        # if torch.distributed.get_rank() == 0:
        #     print(f"[DEBUG] (reve_model.py) *** Finished image transformer blocks processing.")


        ### Final layer
        patched_img = self.final_layer(patched_img)


        # # DEBUGGING (match forward pass in reve_pytorch/model.py)
        # if torch.distributed.get_rank() == 0:
        #     patched_img_reshaped = patched_img.reshape(number_packed_samples, img_seq_len, img_token_dim)
        #     print(f"[DEBUG] (reve_model.py) patched_img_reshaped.shape - patched_img_reshaped.mean() - patched_img_reshaped.std() - patched_img_reshaped.norm(): {patched_img_reshaped.shape} - {patched_img_reshaped.mean()} - {patched_img_reshaped.std()} - {patched_img_reshaped.norm()}")
        #     print(f"[DEBUG] (reve_model.py) patched_img_reshaped: {patched_img_reshaped}")
        #     print(f"[DEBUG] (reve_model.py) --------------------------------")
        #     print(stop_here)

        return patched_img


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

        for module in ["t_embedder"]:
            for param_name, param in getattr(self, module).named_parameters():
                weight_key = f"{prefix}{module}.{param_name}"
                self._set_embedder_weights_replica_id(param, sharded_state_dict, weight_key)
        return sharded_state_dict

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

