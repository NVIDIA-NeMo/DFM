import argparse
import functools
import os
import time

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from layers import TransformerBlock
from model import ReveV2
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
)
from torch.distributed.fsdp import (
    MixedPrecision,
    ShardingStrategy,
)
from torch.distributed.fsdp.wrap import (
    transformer_auto_wrap_policy,
)


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

def get_1b_config():
    """Half full config."""
    return {
        "latent_dims": 768,
        "text_dims": 4096,
        "dims_per_head": 256,
        "num_heads": 8,
        "cross_dims_per_head": 256,
        "cross_num_heads": 8,
        "mlp_ratio": 4.0,
        "num_layers": 13,
        "cross_num_layers": 4,
        "rope_dims": [64, 64],
        "cross_rope_dims": 128,
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

def get_full_config_dimhead128():
    """Full production config with dims_per_head = 128."""
    return {
        "latent_dims": 768,
        "text_dims": 4096,
        "dims_per_head": 128,
        "num_heads": 48,
        "cross_dims_per_head": 128,
        "cross_num_heads": 48,
        "mlp_ratio": 4.0,
        "num_layers": 26,
        "cross_num_layers": 8,
        "rope_dims": [64, 64],
        "cross_rope_dims": 128,
        "rope_max_wavelength": 512.0,
        "attn_scale": 16.0,
        "patch_size": 1,
    }

def setup():
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")

def cleanup():
    dist.destroy_process_group()

def mock_train(config, micro_batch_size=8, num_img_tokens=256, num_txt_tokens=128, grad_accumulation=1):
    setup()
    
    rank = dist.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{local_rank}")
    global_batch_size = micro_batch_size * world_size * grad_accumulation

    if rank == 0:
        print(f"Using world size: {world_size}")
        print(f"micro_batch_size: {micro_batch_size}")
        print(f"grad_accumulation: {grad_accumulation}")
        print(f"num_img_tokens: {num_img_tokens}")
        print(f"num_txt_tokens: {num_txt_tokens}")
        print(f"global_batch_size: {global_batch_size}")

    # Set random seed for reproducibility
    torch.manual_seed(42)

    # Extract relevant params for data generation
    latent_dims = config["latent_dims"]
    text_dims = config["text_dims"]
    patch_size = config["patch_size"]
    
    ## set up mock data shape (micro_batch_size, num_img_tokens, num_txt_tokens from CLI)
    img_token_dim = latent_dims * (patch_size ** 2)
    txt_token_dim = text_dims

    # Initialize model
    if rank == 0:
        print("Initializing ReveV2 model...", flush=True)

    # Initialize on CPU first, then FSDP will move it
    model = ReveV2(**config)
    
    # Count parameters on CPU before FSDP wrapping/sharding to avoid OOM
    if rank == 0:
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Total parameters: {total_params:,}", flush=True)
    
    if rank == 0:
        print("Model initialized. wrapping with FSDP...", flush=True)

    # Mixed Precision Policy
    mp_policy = MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.bfloat16,
        buffer_dtype=torch.bfloat16,
    )

    # Auto wrap policy for Transformers
    auto_wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={TransformerBlock},
    )

    if world_size <= 8: 
            # If running on fewer devices than expected for a node (e.g. debugging with 1 GPU),
            # FULL_SHARD is equivalent (or NO_SHARD if world_size=1) but safer.
            print("Sharding strategy: FULL_SHARD")
            sharding_strategy = ShardingStrategy.FULL_SHARD
    else:
            print("Sharding strategy: HYBRID_SHARD")
            sharding_strategy = ShardingStrategy.HYBRID_SHARD

    model = FSDP(
        model,
        auto_wrap_policy=auto_wrap_policy,
        mixed_precision=mp_policy,
        device_id=device,
        sharding_strategy=sharding_strategy,
        limit_all_gathers=True,
        use_orig_params=False,
    )
    if rank == 0:
        print("FSDP wrapping done.", flush=True)

    # torch.compile for faster training (PyTorch 2.x).
    # Use mode="default" with FSDP: mode="reduce-overhead" uses CUDA graphs and can produce
    # "The CUDA Graph is empty" when capture runs on a different device/stream than FSDP's work.
    if rank == 0:
        print("Applying torch.compile with mode=default (FSDP-friendly)...", flush=True)
    # model = torch.compile(model, mode="default")
    if rank == 0:
        print("torch.compile applied.", flush=True)

    # Optimizer
    if rank == 0:
        print("Initializing optimizer...", flush=True)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    
    # Loss function (MSE for diffusion usually)
    criterion = nn.MSELoss()

    if rank == 0:
        print("Starting training loop for a number of steps...", flush=True)
    
    step_times = []
    forward_times = []
    backward_times = []
    optimizer_times = []
    
    # Wait for all ranks
    if rank == 0:
        print("Waiting for barrier...", flush=True)
    dist.barrier(device_ids=[local_rank])
    if rank == 0:
        print("Barrier passed. Entering loop.", flush=True)
    
    num_steps = 100
    for step in range(1, num_steps + 1):
        torch.cuda.synchronize()
        step_start = time.time()
        
        optimizer.zero_grad()
        step_loss = 0.0

        for acc_step in range(grad_accumulation):
            torch.manual_seed(42 + rank + step * grad_accumulation + acc_step)

            x = torch.randn(micro_batch_size, num_img_tokens, img_token_dim, device=device).to(torch.bfloat16)
            x_position_ids = torch.rand(micro_batch_size, num_img_tokens, 3, device=device).to(torch.bfloat16)
            timestep = torch.randint(0, 1000, (micro_batch_size,), device=device) # Integer
            y = torch.randn(micro_batch_size, num_txt_tokens, txt_token_dim, device=device).to(torch.bfloat16)

            y_mask = torch.ones(micro_batch_size, num_txt_tokens, dtype=torch.bool, device=device)
            y_mask[:, -2:] = False

            conditioning_signal = torch.randn(micro_batch_size, device=device).to(torch.bfloat16)

            target = torch.randn_like(x)

            # Forward pass
            torch.cuda.synchronize()
            forward_start = time.time()
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                output = model(
                    x=x,
                    x_position_ids=x_position_ids,
                    timestep=timestep,
                    y=y,
                    y_mask=y_mask,
                    conditioning_signal=conditioning_signal
                )

            # Compute loss (scale by 1/grad_accumulation for correct gradient accumulation)
            loss = criterion(output, target) / grad_accumulation
            step_loss += loss.detach() * grad_accumulation
            torch.cuda.synchronize()
            forward_end = time.time()
            forward_times.append(forward_end - forward_start)

            # Backward pass
            torch.cuda.synchronize()
            backward_start = time.time()
            loss.backward()
            torch.cuda.synchronize()
            backward_end = time.time()
            backward_times.append(backward_end - backward_start)

        # Optimization step (once per logical step)
        torch.cuda.synchronize()
        optim_start = time.time()
        optimizer.step()
        torch.cuda.synchronize()
        optim_end = time.time()
        optimizer_times.append(optim_end - optim_start)

        step_end = time.time()
        step_times.append(step_end - step_start)
        avg_loss = step_loss.item() / grad_accumulation

        if step % 10 == 0 and rank == 0:
            avg_step_time = sum(step_times[-10:]) / 10
            avg_fwd_time = sum(forward_times[-10 * grad_accumulation:]) / (10 * grad_accumulation)
            avg_bwd_time = sum(backward_times[-10 * grad_accumulation:]) / (10 * grad_accumulation)
            avg_opt_time = sum(optimizer_times[-10:]) / 10
            print(f"Step {step}/{num_steps} | Loss: {avg_loss:.6f} | Avg Step Time (last 10): {avg_step_time:.4f}s | Fwd: {avg_fwd_time:.4f}s | Bwd: {avg_bwd_time:.4f}s | Opt: {avg_opt_time:.4f}s")

    if rank == 0:
        print(f"Training finished. Overall Average Step Time: {sum(step_times) / len(step_times):.4f}s")
        print("\n"*10)
    cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        choices=["small", "1b", "full", "full_dimhead128"],
        default="small",
        help="Config size to use",
    )
    parser.add_argument(
        "--micro_batch_size",
        type=int,
        default=8,
        help="Batch size for mock training",
    )
    parser.add_argument(
        "--num_img_tokens",
        type=int,
        default=256,
        help="Number of image tokens per sample (e.g. 256 for 16x16)",
    )
    parser.add_argument(
        "--num_txt_tokens",
        type=int,
        default=128,
        help="Number of text tokens per sample",
    )
    parser.add_argument(
        "--grad_accumulation",
        type=int,
        default=1,
        help="Gradient accumulation steps (effective global batch = micro_batch_size * world_size * grad_accumulation)",
    )
    args = parser.parse_args()

    # get config
    if args.config == "small":
        config = get_small_config()
    elif args.config == "1b":
        config = get_1b_config()
    elif args.config == "full":
        config = get_full_config()
    elif args.config == "full_dimhead128":
        config = get_full_config_dimhead128()
    else:
        config = None    

    mock_train(
        config,
        micro_batch_size=args.micro_batch_size,
        num_img_tokens=args.num_img_tokens,
        num_txt_tokens=args.num_txt_tokens,
        grad_accumulation=args.grad_accumulation,
    )
