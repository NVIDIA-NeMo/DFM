#!/usr/bin/env python3
# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
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

"""
FLUX Inference Script

This script loads a trained FLUX transformer from a checkpoint and runs inference
using the diffusers library.

Usage:
    python flux_inference.py --checkpoint flux_pretrain_outputs/LATEST --prompt "A cat sitting on a beach"
    python flux_inference.py --checkpoint flux_pretrain_outputs/epoch_187_step_24002 --prompt "A beautiful sunset"
"""

import argparse
import os
from pathlib import Path

import torch
from diffusers import FluxPipeline, FluxTransformer2DModel


def parse_args():
    parser = argparse.ArgumentParser(description="FLUX Inference with trained checkpoint")
    
    parser.add_argument(
        "--model_id",
        type=str,
        default="black-forest-labs/FLUX.1-dev",
        help="Base FLUX model ID from HuggingFace (for text encoders, VAE, scheduler)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint directory containing model/ subfolder with .distcp files",
    )
    parser.add_argument(
        "--use-original",
        action="store_true",
        help="Use original FLUX model without loading custom checkpoint",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="A photorealistic cat sitting on a sunny beach, ocean waves in the background",
        help="Text prompt for image generation",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./inference_outputs",
        help="Directory to save generated images",
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=28,
        help="Number of inference steps",
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=3.5,
        help="Guidance scale for classifier-free guidance",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=512,
        help="Image height",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=512,
        help="Image width",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["float16", "bfloat16", "float32"],
        help="Data type for model",
    )
    
    return parser.parse_args()


def load_sharded_checkpoint(transformer, checkpoint_dir, device="cuda"):
    """
    Load sharded FSDP checkpoint (.distcp files) into transformer.
    
    This function initializes a single-process distributed group, wraps the
    transformer in FSDP, loads the sharded checkpoint, then unwraps it.
    
    Args:
        transformer: FluxTransformer2DModel instance
        checkpoint_dir: Path to the checkpoint directory (containing model/ subfolder)
        device: Device to load the model on (default: "cuda")
    
    Returns:
        transformer: Loaded transformer model
    """
    import torch.distributed as dist
    from torch.distributed.checkpoint import FileSystemReader
    from torch.distributed.checkpoint import load as dist_load
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp import StateDictType
    from torch.distributed.fsdp.api import ShardedStateDictConfig

    sharded_dir = os.path.join(checkpoint_dir, "model")
    
    if not os.path.isdir(sharded_dir):
        raise FileNotFoundError(f"Model directory not found: {sharded_dir}")
    
    distcp_files = [f for f in os.listdir(sharded_dir) if f.endswith(".distcp")]
    if not distcp_files:
        raise FileNotFoundError(f"No .distcp files found in {sharded_dir}")
    
    print(f"[INFO] Found {len(distcp_files)} .distcp files in {sharded_dir}")
    print("[INFO] Loading sharded checkpoint via PyTorch Distributed Checkpoint...")

    # Initialize a single-process group if not already initialized
    init_dist = False
    if not dist.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "localhost")
        os.environ.setdefault("MASTER_PORT", "29500")
        dist.init_process_group(backend="gloo", rank=0, world_size=1)
        init_dist = True

    try:
        # Ensure uniform dtype and move to CUDA before FSDP wrapping
        # FSDP requires the model to be on GPU
        transformer = transformer.to(device=device, dtype=torch.bfloat16)
        
        # Wrap transformer with FSDP to load sharded weights
        fsdp_transformer = FSDP(transformer, use_orig_params=True, device_id=torch.device(device))

        # Configure to expect sharded state dict
        FSDP.set_state_dict_type(
            fsdp_transformer,
            StateDictType.SHARDED_STATE_DICT,
            state_dict_config=ShardedStateDictConfig(offload_to_cpu=True),
        )

        # Load shards into the FSDP-wrapped model
        model_state = fsdp_transformer.state_dict()
        dist_load(state_dict=model_state, storage_reader=FileSystemReader(sharded_dir))
        fsdp_transformer.load_state_dict(model_state)

        # Unwrap back to the original module for inference
        transformer = fsdp_transformer.module
        
        print("[INFO] ✅ Successfully loaded sharded FSDP checkpoint")
        
    finally:
        if init_dist:
            dist.destroy_process_group()
    
    return transformer


def load_consolidated_checkpoint(transformer, checkpoint_path):
    """
    Load a consolidated (non-sharded) checkpoint.
    
    Args:
        transformer: FluxTransformer2DModel instance
        checkpoint_path: Path to the consolidated checkpoint file
    
    Returns:
        transformer: Loaded transformer model
    """
    print(f"[INFO] Loading consolidated checkpoint from {checkpoint_path}")
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    
    # Handle different checkpoint formats
    if "model_state_dict" in state_dict:
        state_dict = state_dict["model_state_dict"]
    
    transformer.load_state_dict(state_dict, strict=True)
    print("[INFO] ✅ Loaded consolidated checkpoint")
    
    return transformer


def main():
    args = parse_args()
    
    # Set dtype
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    torch_dtype = dtype_map[args.dtype]
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")
    print(f"[INFO] Using dtype: {torch_dtype}")
    
    # Determine if using original model or custom checkpoint
    use_original = args.use_original or args.checkpoint is None
    
    if use_original:
        print("=" * 80)
        print("FLUX Inference with Original Model")
        print("=" * 80)
        model_name = "original"
        checkpoint_dir = None
    else:
        print("=" * 80)
        print("FLUX Inference with Trained Checkpoint")
        print("=" * 80)
        
        # Resolve checkpoint path
        checkpoint_dir = args.checkpoint
        if not os.path.isabs(checkpoint_dir):
            # Make relative paths relative to script directory
            script_dir = Path(__file__).parent
            checkpoint_dir = script_dir / checkpoint_dir
        checkpoint_dir = Path(checkpoint_dir)
        model_name = checkpoint_dir.name
        
        print(f"\n[1] Checkpoint: {checkpoint_dir}")
        
        # Determine checkpoint type and paths
        sharded_dir = checkpoint_dir / "model"
        consolidated_path = checkpoint_dir / "consolidated_model.bin"
        ema_path = checkpoint_dir / "ema_shadow.pt"
        
        # Check what checkpoint type we have
        has_sharded = sharded_dir.exists() and any(f.endswith(".distcp") for f in os.listdir(sharded_dir))
        has_consolidated = consolidated_path.exists()
        has_ema = ema_path.exists()
        
        print(f"  - Sharded checkpoint (.distcp): {'✅' if has_sharded else '❌'}")
        print(f"  - Consolidated checkpoint: {'✅' if has_consolidated else '❌'}")
        print(f"  - EMA checkpoint: {'✅' if has_ema else '❌'}")
        
        if not any([has_sharded, has_consolidated, has_ema]):
            raise FileNotFoundError(
                f"No valid checkpoint found in {checkpoint_dir}. "
                "Expected model/ with .distcp files, consolidated_model.bin, or ema_shadow.pt"
            )
    
    # Load the full FLUX pipeline (includes text encoders, VAE, scheduler, and transformer)
    step_num = 1 if use_original else 2
    print(f"\n[{step_num}] Loading FLUX pipeline from: {args.model_id}")
    print("[INFO] Loading with low_cpu_mem_usage=True to reduce memory footprint...")
    pipe = FluxPipeline.from_pretrained(
        args.model_id,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
    )
    
    # Load checkpoint into transformer (only if not using original)
    # Must be done BEFORE enabling CPU offload for sharded checkpoints
    if not use_original:
        print(f"\n[3] Loading trained transformer weights...")
        
        if has_ema:
            # EMA checkpoint typically has best quality
            print("[INFO] Using EMA checkpoint (best quality)")
            ema_state = torch.load(ema_path, map_location="cpu")
            pipe.transformer.load_state_dict(ema_state, strict=True)
            print("[INFO] ✅ Loaded EMA checkpoint")
        elif has_consolidated:
            pipe.transformer = load_consolidated_checkpoint(pipe.transformer, str(consolidated_path))
        elif has_sharded:
            # Load sharded FSDP checkpoint (will be moved to CUDA inside the function)
            pipe.transformer = load_sharded_checkpoint(pipe.transformer, str(checkpoint_dir), device=device)
    else:
        print("[INFO] Using original FLUX model weights (no checkpoint loaded)")
    
    # Enable model CPU offloading to reduce GPU memory usage during inference
    # This moves components to GPU only when needed
    print("[INFO] Enabling model CPU offload for memory-efficient inference...")
    pipe.enable_model_cpu_offload()
    
    # Set up output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate image
    step_num = 2 if use_original else 4
    print(f"\n[{step_num}] Generating image...")
    print(f"  - Model: {model_name}")
    print(f"  - Prompt: {args.prompt}")
    print(f"  - Steps: {args.num_inference_steps}")
    print(f"  - Guidance scale: {args.guidance_scale}")
    print(f"  - Size: {args.width}x{args.height}")
    print(f"  - Seed: {args.seed}")
    
    # Set random seed for reproducibility
    # Use CPU generator when using model offloading for compatibility
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    
    with torch.no_grad():
        output = pipe(
            prompt=args.prompt,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            height=args.height,
            width=args.width,
            generator=generator,
        )
    
    # Save generated image
    image = output.images[0]
    
    # Create filename from prompt (truncated and sanitized) and include model name
    safe_prompt = "".join(c if c.isalnum() or c in " _-" else "" for c in args.prompt)[:50]
    safe_prompt = safe_prompt.strip().replace(" ", "_")
    output_path = output_dir / f"flux_{model_name}_{safe_prompt}_seed{args.seed}.png"
    
    image.save(output_path)
    
    step_num = 3 if use_original else 5
    print(f"\n[{step_num}] ✅ Image saved to: {output_path}")
    print("=" * 80)
    print("Inference complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
