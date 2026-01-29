"""Minimal ReveV2 diffusion model."""

import torch
from beartype import beartype
from jaxtyping import Bool, Float, Integer, jaxtyped
from layers import (
    CosineEmbed,
    MLPEmbed,
    PadToken,
    RMSNorm,
    RoPEEmbed,
    TransformerBlock,
    latency_tracker,
)
from torch import nn
import os
import time

class ReveV2(nn.Module):
    def __init__(
        self,
        latent_dims: int,
        text_dims: int,
        dims_per_head: int,
        num_heads: int,
        cross_dims_per_head: int,
        cross_num_heads: int,
        mlp_ratio: float,
        num_layers: int,
        cross_num_layers: int,
        rope_dims: list[int],
        cross_rope_dims: int,
        rope_max_wavelength: float,
        attn_scale: float,
        patch_size: int,
    ):
        super().__init__()

        self.patch_size = patch_size

        model_dims = dims_per_head * num_heads
        cross_model_dims = cross_dims_per_head * cross_num_heads

        self.img_in_norm: nn.Module = RMSNorm(model_dims)
        self.img_in = nn.Linear(latent_dims * patch_size**2, model_dims, bias=True)
        self.time_cos_embed = CosineEmbed(
            dims=dims_per_head, max_wavelength=10_000, scale=1000
        )

        self.time_mlp_embed = MLPEmbed(in_dims=dims_per_head, out_dims=model_dims)
        self.conditioning_signal_cos_embed = CosineEmbed(
            dims=dims_per_head, max_wavelength=10_000, scale=1000
        )
        self.conditioning_signal_mlp_embed = MLPEmbed(
            in_dims=dims_per_head, out_dims=model_dims
        )

        self.txt_in_norm: nn.Module = RMSNorm(cross_model_dims)
        self.txt_in = nn.Linear(text_dims, cross_model_dims, bias=True)
        self.txt_out = nn.Linear(cross_model_dims, model_dims, bias=True)
        self.txt_rope_emb = RoPEEmbed(
            dims_per_axis=[cross_dims_per_head - cross_rope_dims, cross_rope_dims],
            max_wavelength=rope_max_wavelength,
        )

        self.txt_blocks = nn.ModuleList(
            [
                TransformerBlock(
                    dims_per_head=cross_dims_per_head,
                    num_heads=cross_num_heads,
                    scale=attn_scale,
                    do_cross_attn=False,
                    do_modulation=False,
                    mlp_ratio=mlp_ratio,
                )
                for _ in range(cross_num_layers)
            ]
        )

        self.img_rope_emb = RoPEEmbed(
            dims_per_axis=[dims_per_head - sum(rope_dims), *rope_dims],
            max_wavelength=rope_max_wavelength,
        )

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    dims_per_head=dims_per_head,
                    num_heads=num_heads,
                    scale=attn_scale,
                    do_cross_attn=True,
                    do_modulation=True,
                    mlp_ratio=mlp_ratio,
                )
                for _ in range(num_layers)
            ]
        )

        self.final_layer = nn.Linear(model_dims, latent_dims * patch_size**2, bias=True)
        self.pad_tokens = PadToken()

        # # DEBUGGING (match forward pass in reve_pytorch/model.py)
        # # Loading from "synced_checkpoints/reve_simple_init.pt"
        # if os.path.exists("synced_checkpoints/reve_simple_init.pt"):
        #     if torch.distributed.get_rank() == 0:
        #        print(f"Loading weights from synced_checkpoints/reve_simple_init.pt")
        #     state_dict = torch.load("synced_checkpoints/reve_simple_init.pt", map_location="cpu")
        #     self.load_state_dict(state_dict, strict=True)
        #     # Convert to bfloat16
        #     self.to(torch.bfloat16)

        #     # Print model's parameters and their shapes, dtype, mean, std
        #     for name, param in self.named_parameters():
        #        print(f"{name}: {param.shape}, dtype: {param.dtype}, mean: {param.mean()}, std: {param.std()}")
        #     # Also print total number of parameters
        #     total_params = sum(param.numel() for name, param in self.named_parameters())
        #     print(f"Total Parameters: {total_params}")
        #     # print(stop_here)


        # DEBUGGING
        # Print model's parameters and their shapes
        total_params = 0
        print(f"\n{'='*20} ReveV2 Model Parameters {'='*20}")
        for name, param in self.named_parameters():
            print(f"{name}: {tuple(param.shape)}")
            total_params += param.numel()
        print(f"Total Parameters: {total_params}")
        print(f"{'='*58}\n")


    @jaxtyped(typechecker=beartype)
    def _encode_text(
        self,
        y: Float[torch.Tensor, "bs num_tokens token_dim"],
        y_mask: Bool[torch.Tensor, "bs num_tokens"]
        | Integer[torch.Tensor, "bs num_tokens"],
    ):
        txt = y
        txt_mask = y_mask.bool()
        txt, txt_mask = self.pad_tokens(txt, txt_mask)

        txt = self.txt_in(txt)
        txt = self.txt_in_norm(txt)

        txt_pos_ids = torch.arange(txt.shape[1], dtype=torch.float32, device=txt.device)
        txt_nopos_ids = torch.zeros_like(txt_pos_ids)
        txt_pos_ids = torch.stack([txt_nopos_ids, txt_pos_ids], dim=-1)[None]
        txt_pos_ids = txt_pos_ids.repeat(txt.shape[0], 1, 1)
        txt_rope_cis = self.txt_rope_emb(txt_pos_ids)

        # DEBUGGING (benchmarking)
        text_decoder_start_time = time.time()

        for block in self.txt_blocks:
            txt = block(txt, None, None, txt_rope_cis, txt_mask, None)

        # DEBUGGING (benchmarking)
        text_decoder_end_time = time.time()
        latency_tracker.update(f"[DEBUG] (reve_pytorch/model.py) - text decoder time: {text_decoder_end_time - text_decoder_start_time} seconds", now=text_decoder_end_time)

        txt = self.txt_out(txt)
        return txt, txt_mask

    def _embed_image(self, patched_img: torch.Tensor) -> torch.Tensor:
        patched_img = self.img_in(patched_img)
        patched_img = self.img_in_norm(patched_img)
        return patched_img

    def _compute_modulation_vector(
        self,
        dtype: torch.dtype,
        timestep: Float[torch.Tensor, " bs"],
        conditioning_signal: Float[torch.Tensor, " bs"],
    ) -> Float[torch.Tensor, "bs model_dims"]:
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

    @jaxtyped(typechecker=beartype)
    def forward(
        self,
        x: Float[torch.Tensor, "bs num_img_tokens img_token_dim"],
        x_position_ids: Float[torch.Tensor, "bs num_img_tokens 3"],
        timestep: Integer[torch.Tensor, " bs"] | Float[torch.Tensor, " bs"],
        y: Float[torch.Tensor, "bs num_txt_tokens txt_token_dim"],
        y_mask: Bool[torch.Tensor, "bs num_txt_tokens"],
        conditioning_signal: Float[torch.Tensor, " bs"],
    ) -> Float[torch.Tensor, "bs num_img_tokens img_token_dim"]:

        # # DEBUGGING (match forward pass in reve_pytorch/model.py)
        # # Mock values
        # device = self.img_in.weight.device
        # dtype = self.img_in.weight.dtype
        # bs = 2
        # num_image_tokens = 256
        # num_tokens = 256
        # img_token_dim = self.img_in.in_features
        # txt_token_dim = self.txt_in.in_features
        # generator = torch.Generator(device=device)
        # generator.manual_seed(42)
        # x = torch.randn(
        #     bs,
        #     num_image_tokens,
        #     img_token_dim,
        #     dtype=dtype,
        #     device=device,
        #     generator=generator,
        # )
        # x_position_ids = torch.ones(bs, num_image_tokens, 3, dtype=dtype, device=device)
        # timestep = torch.tensor([0.5], dtype=dtype, device=device).repeat(bs)
        # y = torch.randn(
        #     bs, num_tokens, txt_token_dim, dtype=dtype, device=device, generator=generator,
        # )
        # y_mask = torch.ones(bs, num_tokens, dtype=torch.bool, device=device)
        # conditioning_signal = torch.tensor([0.7], dtype=dtype, device=device).repeat(bs)


        # DEBUGGING
        if torch.distributed.get_rank() == 0:
            print(f"[DEBUG] (reve_pytorch/model.py) x.shape: {x.shape}")
            # print(f"[DEBUG] (reve_pytorch/model.py) x.mean() - x.std(): {x.mean()} - {x.std()}")
            # print(f"[DEBUG] (reve_pytorch/model.py) x_position_ids: {x_position_ids.shape}")
            # print(f"[DEBUG] (reve_pytorch/model.py) x_position_ids.mean() - x_position_ids.std(): {x_position_ids.mean()} - {x_position_ids.std()}")
            # print(f"[DEBUG] (reve_pytorch/model.py) timestep: {timestep}")
            print(f"[DEBUG] (reve_pytorch/model.py) y.shape: {y.shape}")
            # print(f"[DEBUG] (reve_pytorch/model.py) y.mean() - y.std(): {y.mean()} - {y.std()}")
            # print(f"[DEBUG] (reve_pytorch/model.py) y_mask: {y_mask.shape}")
            # print(f"[DEBUG] (reve_pytorch/model.py) y_mask: {y_mask}")
            # print(f"[DEBUG] (reve_pytorch/model.py) conditioning_signal: {conditioning_signal}")
            print(f"[DEBUG] (reve_pytorch/model.py) --------------------------------")


        # DEBUGGING (benchmarking)
        full_forward_start_time = time.time()
        latency_tracker.update(f"[DEBUG] (reve_pytorch/model.py) --------------------------------", now=full_forward_start_time)

        # DEBUGGING (benchmarking)
        text_processing_start_time = time.time()

        txt, txt_mask = self._encode_text(y, y_mask)

        # DEBUGGING (benchmarking)
        text_processing_end_time = time.time()
        latency_tracker.update(f"[DEBUG] (reve_pytorch/model.py) - text processing time: {text_processing_end_time - text_processing_start_time} seconds", now=text_processing_end_time)


        # DEBUGGING (benchmarking)
        image_processing_start_time = time.time()

        patched_img, img_ids = x, x_position_ids
        batch_size = patched_img.shape[0]

        img_ids = img_ids.clone().float()

        position_scale = torch.ones((batch_size, 1, 3), device=img_ids.device)
        position_scale[:, :, 0] = 1.0
        img_ids = img_ids * position_scale

        x_mask = img_ids[..., 0] >= 0
        patched_img, x_mask = self.pad_tokens(patched_img, x_mask)
        patched_img = self._embed_image(patched_img)
        img_rope_cis = self.img_rope_emb(img_ids)

        condition_vector_emb = self._compute_modulation_vector(
            patched_img.dtype,
            timestep,
            conditioning_signal,
        )

        # DEBUGGING (benchmarking)
        image_decoder_start_time = time.time()

        for block in self.blocks:
            patched_img = block(
                patched_img,
                txt,
                condition_vector_emb,
                img_rope_cis,
                x_mask,
                txt_mask,
            )


        # DEBUGGING (benchmarking)
        image_decoder_end_time = time.time()
        latency_tracker.update(f"[DEBUG] (reve_pytorch/model.py) - image decoder time: {image_decoder_end_time - image_decoder_start_time} seconds", now=image_decoder_end_time)

        # DEBUGGING (benchmarking)
        image_processing_end_time = time.time()
        latency_tracker.update(f"[DEBUG] (reve_pytorch/model.py) - image processing time: {image_processing_end_time - image_processing_start_time} seconds", now=image_processing_end_time)

        patched_img = self.final_layer(patched_img)

        # DEBUGGING (benchmarking)
        full_forward_end_time = time.time()
        latency_tracker.update(f"[DEBUG] (reve_pytorch/model.py) - full forward pass time: {full_forward_end_time - full_forward_start_time} seconds", now=full_forward_end_time)

        return patched_img

