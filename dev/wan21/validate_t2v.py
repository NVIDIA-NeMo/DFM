#!/usr/bin/env python3
# validate_t2v.py - Simple T2V validation (loads prompts from .meta files, infers frame count)

import argparse
import json
import os
import pickle
from pathlib import Path

import torch
from diffusers import WanPipeline
from diffusers.utils import export_to_video


def parse_args():
    p = argparse.ArgumentParser("WAN 2.1 T2V Validation")

    # Model configuration
    p.add_argument("--model_id", type=str, default="Wan-AI/Wan2.1-T2V-1.3B-Diffusers")
    p.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint (optional)")

    # Data - load from .meta files
    p.add_argument("--meta_folder", type=str, required=True,
                   help="Folder containing .meta files with prompts")

    # Generation settings
    p.add_argument("--num_samples", type=int, default=None, help="Number of samples (default: all)")
    p.add_argument("--num_inference_steps", type=int, default=50)
    p.add_argument("--guidance_scale", type=float, default=5.0)
    p.add_argument("--negative_prompt", type=str, default="")
    p.add_argument("--seed", type=int, default=42)
    
    # Video settings - now inferred from latents by default
    p.add_argument("--height", type=int, default=None, help="Height (default: infer from latents)")
    p.add_argument("--width", type=int, default=None, help="Width (default: infer from latents)")
    p.add_argument("--num_frames", type=int, default=None, help="Number of frames (default: infer from latents)")
    p.add_argument("--fps", type=int, default=16)

    # Output
    p.add_argument("--output_dir", type=str, default="./validation_outputs")

    return p.parse_args()


def infer_video_params_from_latents(latents):
    """
    Infer video generation parameters from latent shape.
    
    Args:
        latents: torch.Tensor or np.ndarray with shape (16, T_latent, H_latent, W_latent)
                 or (1, 16, T_latent, H_latent, W_latent)
    
    Returns:
        dict with num_frames, height, width
    """
    # Convert to tensor if numpy
    if not isinstance(latents, torch.Tensor):
        latents = torch.from_numpy(latents)
    
    # Handle batch dimension
    if latents.ndim == 5:
        latents = latents[0]  # Remove batch dim: (16, T_latent, H_latent, W_latent)
    
    C, T_latent, H_latent, W_latent = latents.shape
    
    # WAN 2.1 VAE compression ratios
    temporal_compression = 4
    spatial_compression = 8
    
    # Infer dimensions
    num_frames = (T_latent - 1) * temporal_compression + 1
    height = H_latent * spatial_compression
    width = W_latent * spatial_compression
    
    return {
        "num_frames": num_frames,
        "height": height,
        "width": width,
    }


def load_prompts_from_meta_files(meta_folder: str):
    """
    Load prompts from .meta files and infer video dimensions from latents.
    Each .meta file contains a 'metadata' dict with 'vila_caption'.
    
    Returns list of dicts: [{"prompt": "...", "name": "...", "num_frames": ..., ...}, ...]
    """
    meta_folder = Path(meta_folder)
    meta_files = sorted(list(meta_folder.glob("*.meta")))
    
    if not meta_files:
        raise FileNotFoundError(f"No .meta files found in {meta_folder}")
    
    print(f"[INFO] Found {len(meta_files)} .meta files")
    
    prompts = []
    
    for meta_file in meta_files:
        try:
            with open(meta_file, 'rb') as f:
                data = pickle.load(f)
            
            # Extract prompt from metadata
            metadata = data.get("metadata", {})
            prompt = metadata.get("vila_caption", "")
            
            if not prompt:
                print(f"[WARNING] No vila_caption in {meta_file.name}, skipping...")
                continue
            
            # Get filename without extension
            name = meta_file.stem
            
            # Try to infer video dimensions from latents if available
            video_params = None
            if "video_latents" in data:
                try:
                    video_params = infer_video_params_from_latents(data["video_latents"])
                except Exception as e:
                    print(f"[WARNING] Could not infer dimensions from {meta_file.name}: {e}")
            
            item = {
                "prompt": prompt,
                "name": name,
                "meta_file": str(meta_file)
            }
            
            # Add inferred dimensions if available
            if video_params:
                item.update(video_params)
            
            prompts.append(item)
            
        except Exception as e:
            print(f"[WARNING] Failed to load {meta_file.name}: {e}")
            continue
    
    if not prompts:
        raise ValueError(f"No valid prompts found in {meta_folder}")
    
    return prompts


def main():
    args = parse_args()
    
    print("=" * 80)
    print("WAN 2.1 Text-to-Video Validation")
    print("=" * 80)
    
    # Load prompts from .meta files
    print(f"\n[1] Loading prompts from .meta files in: {args.meta_folder}")
    prompts = load_prompts_from_meta_files(args.meta_folder)
    
    if args.num_samples:
        prompts = prompts[:args.num_samples]
    
    print(f"[INFO] Loaded {len(prompts)} prompts")
    
    # Show first few prompts with inferred dimensions
    print("\n[INFO] Sample prompts:")
    for i, item in enumerate(prompts[:3]):
        dims_str = ""
        if "num_frames" in item:
            dims_str = f" [{item['num_frames']} frames, {item['width']}x{item['height']}]"
        print(f"  {i+1}. {item['name']}{dims_str}: {item['prompt'][:60]}...")
    
    # Check dimension consistency
    items_with_dims = [p for p in prompts if "num_frames" in p]
    if items_with_dims:
        unique_dims = set((p['num_frames'], p['height'], p['width']) for p in items_with_dims)
        if len(unique_dims) == 1:
            num_frames, height, width = list(unique_dims)[0]
            print(f"\n[INFO] Inferred dimensions from latents: {num_frames} frames, {width}x{height}")
            # Use as defaults if not overridden
            if args.num_frames is None:
                args.num_frames = num_frames
            if args.height is None:
                args.height = height
            if args.width is None:
                args.width = width
        else:
            print(f"\n[WARNING] Different dimensions found across samples")
    
    # Set final defaults if still not set
    if args.num_frames is None:
        args.num_frames = 81
        print(f"[INFO] Using default num_frames: {args.num_frames}")
    if args.height is None:
        args.height = 480
        print(f"[INFO] Using default height: {args.height}")
    if args.width is None:
        args.width = 832
        print(f"[INFO] Using default width: {args.width}")
    
    # Load pipeline
    print(f"\n[2] Loading pipeline: {args.model_id}")
    pipe = WanPipeline.from_pretrained(args.model_id, torch_dtype=torch.bfloat16)
    pipe.to("cuda")
    
    # Enable VAE optimizations (critical for memory)
    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()
    print("[INFO] Enabled VAE slicing and tiling")
    
    # Load checkpoint if provided
    if args.checkpoint:
        print(f"\n[3] Loading checkpoint: {args.checkpoint}")
        
        # Try EMA checkpoint first (best quality)
        ema_path = os.path.join(args.checkpoint, "ema_shadow.pt")
        consolidated_path = os.path.join(args.checkpoint, "consolidated_model.bin")
        
        if os.path.exists(ema_path):
            print(f"[INFO] Loading EMA checkpoint (best quality)...")
            ema_state = torch.load(ema_path, map_location="cuda")
            pipe.transformer.load_state_dict(ema_state, strict=True)
            print(f"[INFO] Loaded from EMA checkpoint")
        elif os.path.exists(consolidated_path):
            print(f"[INFO] Loading consolidated checkpoint...")
            state_dict = torch.load(consolidated_path, map_location="cuda")
            pipe.transformer.load_state_dict(state_dict, strict=True)
            print(f"[INFO] Loaded from consolidated checkpoint")
        else:
            print(f"[WARNING] No consolidated or EMA checkpoint found")
            print(f"[INFO] Using base model")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Generate videos
    print(f"\n[4] Generating videos...")
    print(f"[INFO] Settings: {args.width}x{args.height}, {args.num_frames} frames, {args.num_inference_steps} steps")
    print(f"[INFO] Guidance scale: {args.guidance_scale}")
    
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    for i, item in enumerate(prompts):
        prompt = item["prompt"]
        name = item["name"]
        
        # Use per-item dimensions if available, else use global defaults
        num_frames = item.get("num_frames", args.num_frames)
        height = item.get("height", args.height)
        width = item.get("width", args.width)
        
        print(f"\n[{i+1}/{len(prompts)}] Generating: {name}")
        print(f"  Prompt: {prompt[:80]}...")
        print(f"  Dimensions: {num_frames} frames, {width}x{height}")
        
        try:
            # Generate from scratch (no latents needed!)
            generator = torch.Generator(device="cuda").manual_seed(args.seed + i)
            
            output = pipe(
                prompt=prompt,
                negative_prompt=args.negative_prompt,
                height=height,
                width=width,
                num_frames=num_frames,
                guidance_scale=args.guidance_scale,
                num_inference_steps=args.num_inference_steps,
                generator=generator,
            ).frames[0]
            
            # Save video
            output_path = os.path.join(args.output_dir, f"{name}.mp4")
            export_to_video(output, output_path, fps=args.fps)
            
            print(f"  Saved to {output_path}")
            
        except Exception as e:
            print(f"  Failed: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\n" + "=" * 80)
    print(f"Validation complete!")
    print(f"Videos saved to: {args.output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()


# ============================================================================
# USAGE EXAMPLES
# ============================================================================

# 1. Basic usage (dimensions auto-inferred from latents):
# python validate_t2v.py \
#     --meta_folder /linnanw/hdvilla_sample/pika/wan21_codes/1.3B_meta

# 2. Limited samples:
# python validate_t2v.py \
#     --meta_folder /linnanw/hdvilla_sample/pika/wan21_codes/1.3B_meta \
#     --num_samples 5

# 3. With checkpoint (uses EMA if available):
# python validate_t2v.py \
#     --meta_folder /linnanw/hdvilla_sample/pika/wan21_codes/1.3B_meta \
#     --checkpoint ./wan_t2v_all_fixes/checkpoint-5000 \
#     --num_samples 5

# 4. Override dimensions:
# python validate_t2v.py \
#     --meta_folder /linnanw/hdvilla_sample/pika/wan21_codes/1.3B_meta \
#     --checkpoint ./checkpoint-5000 \
#     --height 480 \
#     --width 832 \
#     --num_frames 65 \
#     --num_samples 10

# 5. Custom settings:
# python validate_t2v.py \
#     --meta_folder /linnanw/hdvilla_sample/pika/wan21_codes/1.3B_meta \
#     --checkpoint ./checkpoint-5000 \
#     --num_inference_steps 50 \
#     --guidance_scale 5.0 \
#     --num_samples 10