"""Simple inference example."""

import argparse
import time

import torch

from model import ReveV2


def get_small_config():
    """Small config for fast testing."""
    return {
        "latent_dims": 16,
        "text_dims": 128,
        "dims_per_head": 64,
        "num_heads": 4,
        "cross_dims_per_head": 64,
        "cross_num_heads": 4,
        "mlp_ratio": 4.0,
        "num_layers": 2,
        "cross_num_layers": 2,
        "rope_dims": [16, 16],
        "cross_rope_dims": 32,
        "rope_max_wavelength": 512.0,
        "attn_scale": 16.0,
        "patch_size": 1,
    }


def get_full_config():
    """Full production config."""
    return {
        "latent_dims": 768,
        "text_dims": 4096,
        "dims_per_head": 256,
        "num_heads": 24,
        "cross_dims_per_head": 256,
        "cross_num_heads": 24,
        "mlp_ratio": 4.0,
        "num_layers": 26,
        "cross_num_layers": 8,
        "rope_dims": [64, 64],
        "cross_rope_dims": 128,
        "rope_max_wavelength": 512.0,
        "attn_scale": 16.0,
        "patch_size": 1,
    }


def run_inference(config, use_cuda=False):
    """Run inference with given config."""
    device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")
    dtype = (
        torch.bfloat16 if (use_cuda and torch.cuda.is_available()) else torch.float32
    )

    model = ReveV2(**config).to(device=device, dtype=dtype)
    model.eval()

    bs = 1
    num_image_tokens = 256
    num_tokens = 256
    text_dims = config["text_dims"]

    x_token_sequence = torch.randn(
        bs,
        num_image_tokens,
        config["latent_dims"] * config["patch_size"] ** 2,
        dtype=dtype,
        device=device,
    )
    x_position_ids = torch.ones(bs, num_image_tokens, 3, dtype=dtype, device=device)
    timestep = torch.tensor([0.5], dtype=dtype, device=device)
    y_token_sequence = torch.randn(
        bs, num_tokens, text_dims, dtype=dtype, device=device
    )
    y_mask = torch.ones(bs, num_tokens, dtype=torch.bool, device=device)
    conditioning_signal = torch.tensor([0.7], dtype=dtype, device=device)

    # Mask last 10 tokens (for demo)
    y_mask[0, num_tokens - 10 :] = 0

    start = time.time()
    with torch.no_grad():
        output = model(
            x=x_token_sequence,
            x_position_ids=x_position_ids,
            timestep=timestep,
            y=y_token_sequence,
            y_mask=y_mask,
            conditioning_signal=conditioning_signal,
        )
    end = time.time()

    print(f"Image input shape: {x_token_sequence.shape}")
    print(f"Text input shape: {y_token_sequence.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output dtype: {output.dtype}")
    print(f"Output mean: {output.mean().item():.6f}")
    print(f"Output std: {output.std().item():.6f}")
    print(f"Time taken: {end - start:.2f} s")

    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        choices=["small", "full"],
        default="small",
        help="Config size to use",
    )
    parser.add_argument("--cuda", action="store_true", help="Use CUDA if available")
    args = parser.parse_args()

    config = get_small_config() if args.config == "small" else get_full_config()

    print(f"Running inference with {args.config} config...")
    print(f"Device: {'CUDA' if args.cuda else 'CPU'}")
    print(f"Model size: {args.config}")
    print()

    run_inference(config, use_cuda=args.cuda)