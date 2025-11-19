#!/usr/bin/env python3
# validate_t2v.py - T2V validation using precomputed text embeddings from .meta files

import argparse
import os
import pickle
import subprocess
from pathlib import Path

import numpy as np
import torch
from diffusers import WanPipeline
from diffusers.utils import export_to_video
from PIL import Image


try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("[WARNING] wandb not installed. Install with: pip install wandb")


def convert_to_gif(video_path):
    gif_path = Path(video_path).with_suffix(".gif")
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_path),
        "-vf",
        "fps=15,scale=512:-1:flags=lanczos",
        "-loop",
        "0",
        str(gif_path),
    ]
    subprocess.run(cmd, check=True)
    return str(gif_path)


def parse_args():
    p = argparse.ArgumentParser("WAN 2.1 T2V Validation with Precomputed Embeddings")

    # Model configuration
    p.add_argument("--model_id", type=str, default="Wan-AI/Wan2.1-T2V-1.3B-Diffusers")
    p.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint (optional)")

    # Data - load from .meta files
    p.add_argument("--meta_folder", type=str, required=True, help="Folder containing .meta files with embeddings")

    # Generation settings
    p.add_argument("--num_samples", type=int, default=10, help="Number of samples (default: 10)")
    p.add_argument("--num_inference_steps", type=int, default=50)
    p.add_argument("--guidance_scale", type=float, default=5.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--fps", type=int, default=16)

    # Output
    p.add_argument("--output_dir", type=str, default="./validation_outputs")

    # Wandb settings
    p.add_argument("--use_wandb", action="store_true", help="Upload results to Weights & Biases")
    p.add_argument("--wandb_project", type=str, default="wan_t2v_valid", help="Wandb project name")
    p.add_argument("--wandb_run_name", type=str, default=None, help="Wandb run name (default: auto-generated)")

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


def load_data_from_meta_files(meta_folder: str, num_samples: int = 10):
    """
    Load text embeddings and metadata from .meta files.

    Returns list of dicts: [{
        "prompt": "...",
        "name": "...",
        "text_embeddings": tensor,
        "num_frames": int,
        "height": int,
        "width": int
    }, ...]
    """
    meta_folder = Path(meta_folder)
    meta_files = sorted(list(meta_folder.glob("*.meta")))[:num_samples]

    if not meta_files:
        raise FileNotFoundError(f"No .meta files found in {meta_folder}")

    print(f"[INFO] Found {len(meta_files)} .meta files (limited to first {num_samples})")

    data_list = []

    for meta_file in meta_files:
        try:
            with open(meta_file, "rb") as f:
                data = pickle.load(f)

            # Extract prompt from metadata
            metadata = data.get("metadata", {})
            prompt = metadata.get("vila_caption", "")

            if not prompt:
                print(f"[WARNING] No vila_caption in {meta_file.name}, skipping...")
                continue

            # Get text embeddings
            text_embeddings = data.get("text_embeddings")
            if text_embeddings is None:
                print(f"[WARNING] No text_embeddings in {meta_file.name}, skipping...")
                continue

            # Convert to tensor and remove batch dimensions
            if not isinstance(text_embeddings, torch.Tensor):
                text_embeddings = torch.from_numpy(text_embeddings)

            # Squeeze out batch dimensions: (1, 1, seq_len, hidden_dim) -> (seq_len, hidden_dim)
            while text_embeddings.ndim > 2 and text_embeddings.shape[0] == 1:
                text_embeddings = text_embeddings.squeeze(0)

            # Get filename without extension
            name = meta_file.stem

            # Infer video dimensions from latents
            video_params = None
            if "video_latents" in data:
                try:
                    video_params = infer_video_params_from_latents(data["video_latents"])
                except Exception as e:
                    print(f"[WARNING] Could not infer dimensions from {meta_file.name}: {e}")

            item = {
                "prompt": prompt,
                "name": name,
                "text_embeddings": text_embeddings,
                "meta_file": str(meta_file),
            }

            # Add inferred dimensions if available
            if video_params:
                item.update(video_params)

            data_list.append(item)

        except Exception as e:
            print(f"[WARNING] Failed to load {meta_file.name}: {e}")
            continue

    if not data_list:
        raise ValueError(f"No valid data found in {meta_folder}")

    return data_list


def main():
    args = parse_args()

    print("=" * 80)
    print("WAN 2.1 Text-to-Video Validation (Using Precomputed Embeddings)")
    print("=" * 80)

    # Initialize wandb if requested
    wandb_run = None
    if args.use_wandb:
        if not WANDB_AVAILABLE:
            print("[ERROR] wandb requested but not installed. Install with: pip install wandb")
            print("[INFO] Continuing without wandb...")
        else:
            print("\n[WANDB] Initializing Weights & Biases...")
            print(f"[WANDB] Project: {args.wandb_project}")

            # Generate run name if not provided
            run_name = args.wandb_run_name
            if run_name is None:
                checkpoint_name = Path(args.checkpoint).name if args.checkpoint else "base_model"
                run_name = f"validation_{checkpoint_name}"

            wandb_run = wandb.init(
                project=args.wandb_project,
                name=run_name,
                config={
                    "model_id": args.model_id,
                    "checkpoint": args.checkpoint,
                    "num_samples": args.num_samples,
                    "num_inference_steps": args.num_inference_steps,
                    "guidance_scale": args.guidance_scale,
                    "seed": args.seed,
                    "fps": args.fps,
                },
            )
            print(f"[WANDB] Run name: {run_name}")
            print(f"[WANDB] Run URL: {wandb_run.get_url()}")

    # Load data from .meta files
    print(f"\n[1] Loading data from .meta files in: {args.meta_folder}")
    data_list = load_data_from_meta_files(args.meta_folder, args.num_samples)

    print(f"[INFO] Loaded {len(data_list)} samples")

    # Show first few samples with dimensions
    print("\n[INFO] Sample prompts:")
    for i, item in enumerate(data_list[:3]):
        dims_str = ""
        if "num_frames" in item:
            dims_str = f" [{item['num_frames']} frames, {item['width']}x{item['height']}]"
        emb_shape = item["text_embeddings"].shape
        print(f"  {i + 1}. {item['name']}{dims_str}")
        print(f"     Prompt: {item['prompt'][:60]}...")
        print(f"     Text embeddings: {emb_shape}")

    # Check dimension consistency
    items_with_dims = [p for p in data_list if "num_frames" in p]
    if items_with_dims:
        unique_dims = set((p["num_frames"], p["height"], p["width"]) for p in items_with_dims)
        if len(unique_dims) == 1:
            num_frames, height, width = list(unique_dims)[0]
            print(f"\n[INFO] All samples have consistent dimensions: {num_frames} frames, {width}x{height}")
        else:
            print(f"\n[INFO] Found {len(unique_dims)} different dimension sets across samples")
            for dims in unique_dims:
                count = sum(1 for p in items_with_dims if (p["num_frames"], p["height"], p["width"]) == dims)
                print(f"  - {dims[0]} frames, {dims[2]}x{dims[1]}: {count} samples")

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

        # Try consolidated checkpoint or EMA checkpoint
        consolidated_path = os.path.join(args.checkpoint, "consolidated_model.bin")
        ema_path = os.path.join(args.checkpoint, "ema_shadow.pt")

        if os.path.exists(consolidated_path):
            print("[INFO] Loading consolidated checkpoint...")
            state_dict = torch.load(consolidated_path, map_location="cuda")
            pipe.transformer.load_state_dict(state_dict, strict=True)
            print("[INFO] Loaded from consolidated checkpoint")
        elif os.path.exists(ema_path):
            print("[INFO] Loading EMA checkpoint (best quality)...")
            ema_state = torch.load(ema_path, map_location="cuda")
            pipe.transformer.load_state_dict(ema_state, strict=True)
            print("[INFO] Loaded from EMA checkpoint")
        else:
            print("[WARNING] No consolidated or EMA checkpoint found at specified path")
            print("[INFO] Using base WAN 2.1 model weights from pipeline")
    else:
        print("\n[3] No checkpoint specified, using base WAN 2.1 model weights")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Generate videos
    print("\n[4] Generating videos using precomputed text embeddings...")
    print(f"[INFO] Settings: {args.num_inference_steps} steps, guidance scale: {args.guidance_scale}")

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Track successful generations
    num_generated = 0

    for i, item in enumerate(data_list):
        prompt = item["prompt"]
        name = item["name"]
        text_embeddings = item["text_embeddings"]

        # Get dimensions for this sample
        num_frames = item.get("num_frames")
        height = item.get("height")
        width = item.get("width")

        if num_frames is None or height is None or width is None:
            print(f"\n[{i + 1}/{len(data_list)}] Skipping {name}: missing dimensions")
            continue

        print(f"\n[{i + 1}/{len(data_list)}] Generating: {name}")
        print(f"  Prompt: {prompt[:80]}...")
        print(f"  Dimensions: {num_frames} frames, {width}x{height}")
        print(f"  Text embeddings: {text_embeddings.shape}")

        try:
            # Move embeddings to GPU
            text_embeddings = text_embeddings.to(device="cuda", dtype=torch.bfloat16)

            # Add batch dimension if needed: (seq_len, hidden_dim) -> (1, seq_len, hidden_dim)
            if text_embeddings.ndim == 2:
                text_embeddings = text_embeddings.unsqueeze(0)

            # Generate using precomputed embeddings
            generator = torch.Generator(device="cuda").manual_seed(args.seed + i)

            # Call pipeline with prompt_embeds instead of prompt
            output = pipe(
                prompt_embeds=text_embeddings,
                negative_prompt="",  # Use empty string for negative prompt
                height=height,
                width=width,
                num_frames=num_frames,
                guidance_scale=args.guidance_scale,
                num_inference_steps=args.num_inference_steps,
                generator=generator,
            ).frames[0]

            # Save as image if single frame, otherwise as video
            if num_frames == 1:
                output_path = os.path.join(args.output_dir, f"{name}.png")

                # output is a numpy array, squeeze out extra dimensions
                frame = np.squeeze(output)  # Remove all dimensions of size 1

                # Ensure we have the right shape (H, W, C)
                if frame.ndim == 2:  # Grayscale
                    pass
                elif frame.ndim == 3 and frame.shape[-1] in [1, 3, 4]:  # RGB/RGBA
                    pass
                else:
                    raise ValueError(f"Unexpected frame shape: {frame.shape}")

                # Convert from float [0, 1] to uint8 [0, 255]
                if frame.dtype in [np.float32, np.float64]:
                    frame = (frame * 255).clip(0, 255).astype(np.uint8)

                image = Image.fromarray(frame)
                image.save(output_path)
                print(f"  ✅ Saved image to {output_path}")

                # Upload to wandb immediately
                if wandb_run is not None:
                    print("  📤 Uploading image to wandb...")
                    wandb_run.log(
                        {
                            f"image/{name}": wandb.Image(image, caption=prompt[:100]),
                            f"prompt/{name}": prompt,
                            f"dimensions/{name}": f"{width}x{height}",
                            "sample_index": i,
                        }
                    )
                    print("  ✅ Uploaded to wandb!")

            else:
                output_path = os.path.join(args.output_dir, f"{name}.mp4")
                export_to_video(output, output_path, fps=args.fps)
                print(f"  ✅ Saved video to {output_path}")
                gif_path = convert_to_gif(output_path)
                # Upload to wandb immediately
                if wandb_run is not None:
                    print("  📤 Uploading video to wandb...")
                    wandb_run.log(
                        {
                            f"video/{name}": wandb.Image(gif_path),
                            f"prompt/{name}": prompt,
                            f"dimensions/{name}": f"{num_frames} frames, {width}x{height}",
                            "sample_index": i,
                        }
                    )
                    print("  ✅ Uploaded to wandb!")

            num_generated += 1

        except Exception as e:
            print(f"  ❌ Failed: {e}")
            import traceback

            traceback.print_exc()
            continue

    print("\n" + "=" * 80)
    print("Validation complete!")
    print(f"Generated: {num_generated}/{len(data_list)} samples")
    print(f"Outputs saved to: {args.output_dir}")
    if wandb_run is not None:
        print(f"Wandb results: {wandb_run.get_url()}")
    print("=" * 80)

    # Finish wandb run
    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()


# ============================================================================
# USAGE EXAMPLES
# ============================================================================

# 1. Basic usage (uses precomputed text embeddings from .meta files):
# python validate_t2v.py \
#     --meta_folder /linnanw/hdvilla_sample/pika/wan21_codes/1.3B_meta

# 2. With wandb logging:
# python validate_t2v.py \
#     --meta_folder /linnanw/hdvilla_sample/pika/wan21_codes/1.3B_meta \
#     --use_wandb \
#     --wandb_project wan_t2v_valid \
#     --wandb_run_name "validation_checkpoint_5000"

# 3. With trained checkpoint and wandb:
# python validate_t2v.py \
#     --meta_folder /linnanw/hdvilla_sample/pika/wan21_codes/1.3B_meta \
#     --checkpoint ./wan_t2v_all_fixes/checkpoint-5000 \
#     --use_wandb

# 4. Limited samples with custom settings:
# python validate_t2v.py \
#     --meta_folder /linnanw/hdvilla_sample/pika/wan21_codes/1.3B_meta \
#     --checkpoint ./checkpoint-5000 \
#     --num_samples 5 \
#     --num_inference_steps 50 \
#     --guidance_scale 5.0 \
#     --use_wandb

# 5. If no checkpoint found, uses base WAN 2.1 weights:
# python validate_t2v.py \
#     --meta_folder /linnanw/hdvilla_sample/pika/wan21_codes/1.3B_meta \
#     --checkpoint ./nonexistent_checkpoint \
#     --use_wandb  # Will fall back to base model and log to wandb
