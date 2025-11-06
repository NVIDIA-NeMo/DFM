#!/usr/bin/env python3
# validate_t2v.py - T2V validation supporting both text prompts and precomputed embeddings

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
    """Convert video to GIF for wandb logging."""
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
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        return str(gif_path)
    except subprocess.CalledProcessError as e:
        print(f"[WARNING] Failed to convert to GIF: {e}")
        return None


def parse_args():
    p = argparse.ArgumentParser("WAN 2.1 T2V Validation (Dual Mode: Text or Embeddings)")

    # Model configuration
    p.add_argument("--model_id", type=str, default="Wan-AI/Wan2.1-T2V-1.3B-Diffusers")
    p.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint (optional)")

    # Data source - load from .meta files
    p.add_argument("--meta_folder", type=str, required=True, help="Folder containing .meta files")

    # MODE SELECTION - NEW!
    p.add_argument(
        "--use_embeddings",
        action="store_true",
        help="Use precomputed text embeddings from .meta files (new mode). Default: use text prompts (old mode)",
    )

    # Generation settings
    p.add_argument("--num_samples", type=int, default=10, help="Number of samples")
    p.add_argument("--num_inference_steps", type=int, default=50)
    p.add_argument("--guidance_scale", type=float, default=5.0)
    p.add_argument("--negative_prompt", type=str, default="", help="Negative prompt (only for text mode)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--fps", type=int, default=16)

    # Video settings (only used in text mode if not auto-inferred)
    p.add_argument("--height", type=int, default=480, help="Video height (text mode, or fallback)")
    p.add_argument("--width", type=int, default=832, help="Video width (text mode, or fallback)")
    p.add_argument("--num_frames", type=int, default=81, help="Number of frames (text mode, or fallback)")

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


def load_data_text_mode(meta_folder: str, num_samples: int = 10):
    """
    Load prompts from .meta files for text mode (old approach).

    Returns list of dicts: [{
        "prompt": "...",
        "name": "...",
        "num_frames": int (optional),
        "height": int (optional),
        "width": int (optional)
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

            name = meta_file.stem

            item = {
                "prompt": prompt,
                "name": name,
                "meta_file": str(meta_file),
            }

            # Try to infer dimensions from latents if available
            if "video_latents" in data:
                try:
                    video_params = infer_video_params_from_latents(data["video_latents"])
                    item.update(video_params)
                except Exception as e:
                    print(f"[WARNING] Could not infer dimensions from {meta_file.name}: {e}")

            data_list.append(item)

        except Exception as e:
            print(f"[WARNING] Failed to load {meta_file.name}: {e}")
            continue

    if not data_list:
        raise ValueError(f"No valid data found in {meta_folder}")

    return data_list


def load_data_embedding_mode(meta_folder: str, num_samples: int = 10):
    """
    Load text embeddings and metadata from .meta files for embedding mode (new approach).

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

            # Get text embeddings (required for embedding mode)
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

            name = meta_file.stem

            # Infer video dimensions from latents (required for embedding mode)
            video_params = None
            if "video_latents" in data:
                try:
                    video_params = infer_video_params_from_latents(data["video_latents"])
                except Exception as e:
                    print(f"[WARNING] Could not infer dimensions from {meta_file.name}: {e}")

            if video_params is None:
                print(f"[WARNING] Could not determine dimensions for {meta_file.name}, skipping...")
                continue

            item = {
                "prompt": prompt,
                "name": name,
                "text_embeddings": text_embeddings,
                "meta_file": str(meta_file),
                **video_params,
            }

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
    if args.use_embeddings:
        print("WAN 2.1 T2V Validation (EMBEDDING MODE - Using Precomputed Embeddings)")
    else:
        print("WAN 2.1 T2V Validation (TEXT MODE - Encoding Prompts On-the-Fly)")
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
                mode_suffix = "embeddings" if args.use_embeddings else "text"
                run_name = f"validation_{checkpoint_name}_{mode_suffix}"

            wandb_run = wandb.init(
                project=args.wandb_project,
                name=run_name,
                config={
                    "model_id": args.model_id,
                    "checkpoint": args.checkpoint,
                    "mode": "embeddings" if args.use_embeddings else "text",
                    "num_samples": args.num_samples,
                    "num_inference_steps": args.num_inference_steps,
                    "guidance_scale": args.guidance_scale,
                    "negative_prompt": args.negative_prompt if not args.use_embeddings else "N/A",
                    "seed": args.seed,
                    "fps": args.fps,
                },
            )
            print(f"[WANDB] Run name: {run_name}")
            print(f"[WANDB] Run URL: {wandb_run.get_url()}")

    # Load data from .meta files
    print(f"\n[1] Loading data from .meta files in: {args.meta_folder}")

    if args.use_embeddings:
        print("[INFO] Mode: EMBEDDING MODE (precomputed embeddings)")
        data_list = load_data_embedding_mode(args.meta_folder, args.num_samples)
    else:
        print("[INFO] Mode: TEXT MODE (encode prompts on-the-fly)")
        data_list = load_data_text_mode(args.meta_folder, args.num_samples)

    print(f"[INFO] Loaded {len(data_list)} samples")

    # Show first few samples
    print("\n[INFO] Sample data:")
    for i, item in enumerate(data_list[:3]):
        dims_str = ""
        if "num_frames" in item:
            dims_str = f" [{item['num_frames']} frames, {item['width']}x{item['height']}]"

        print(f"  {i + 1}. {item['name']}{dims_str}")
        print(f"     Prompt: {item['prompt'][:60]}...")

        if args.use_embeddings:
            emb_shape = item["text_embeddings"].shape
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
            print("[INFO] ✅ Loaded from consolidated checkpoint")
        elif os.path.exists(ema_path):
            print("[INFO] Loading EMA checkpoint (best quality)...")
            ema_state = torch.load(ema_path, map_location="cuda")
            pipe.transformer.load_state_dict(ema_state, strict=True)
            print("[INFO] ✅ Loaded from EMA checkpoint")
        else:
            print("[WARNING] No consolidated or EMA checkpoint found at specified path")
            print("[INFO] Using base WAN 2.1 model weights from pipeline")
    else:
        print("\n[3] No checkpoint specified, using base WAN 2.1 model weights")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Generate videos
    print("\n[4] Generating videos...")
    if args.use_embeddings:
        print("[INFO] Using precomputed text embeddings")
    else:
        print("[INFO] Encoding prompts on-the-fly")
        print(f"[INFO] Negative prompt: '{args.negative_prompt}'")

    print(f"[INFO] Settings: {args.num_inference_steps} steps, guidance scale: {args.guidance_scale}")

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Track successful generations
    num_generated = 0

    for i, item in enumerate(data_list):
        prompt = item["prompt"]
        name = item["name"]

        # Get dimensions - use inferred if available, otherwise use CLI args
        num_frames = item.get("num_frames", args.num_frames)
        height = item.get("height", args.height)
        width = item.get("width", args.width)

        print(f"\n[{i + 1}/{len(data_list)}] Generating: {name}")
        print(f"  Prompt: {prompt[:80]}...")
        print(f"  Dimensions: {num_frames} frames, {width}x{height}")

        if args.use_embeddings:
            text_embeddings = item["text_embeddings"]
            print(f"  Text embeddings: {text_embeddings.shape}")

        try:
            generator = torch.Generator(device="cuda").manual_seed(args.seed + i)

            # MODE SWITCH: Use embeddings or text
            if args.use_embeddings:
                # EMBEDDING MODE: Use precomputed embeddings
                text_embeddings = item["text_embeddings"]
                text_embeddings = text_embeddings.to(device="cuda", dtype=torch.bfloat16)

                # Add batch dimension if needed: (seq_len, hidden_dim) -> (1, seq_len, hidden_dim)
                if text_embeddings.ndim == 2:
                    text_embeddings = text_embeddings.unsqueeze(0)

                output = pipe(
                    prompt_embeds=text_embeddings,
                    negative_prompt="",  # Empty string in embedding mode
                    height=height,
                    width=width,
                    num_frames=num_frames,
                    guidance_scale=args.guidance_scale,
                    num_inference_steps=args.num_inference_steps,
                    generator=generator,
                ).frames[0]
            else:
                # TEXT MODE: Use prompt string (old approach)
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

            # Save as image if single frame, otherwise as video
            if num_frames == 1:
                output_path = os.path.join(args.output_dir, f"{name}.png")

                # output is a numpy array, squeeze out extra dimensions
                frame = np.squeeze(output)

                # Convert from float [0, 1] to uint8 [0, 255]
                if frame.dtype in [np.float32, np.float64]:
                    frame = (frame * 255).clip(0, 255).astype(np.uint8)

                image = Image.fromarray(frame)
                image.save(output_path)
                print(f"  ✅ Saved image to {output_path}")

                # Upload to wandb
                if wandb_run is not None:
                    wandb_run.log(
                        {
                            f"image/{name}": wandb.Image(image, caption=prompt[:100]),
                            f"prompt/{name}": prompt,
                            f"dimensions/{name}": f"{width}x{height}",
                            "sample_index": i,
                        }
                    )

            else:
                output_path = os.path.join(args.output_dir, f"{name}.mp4")
                export_to_video(output, output_path, fps=args.fps)
                print(f"  ✅ Saved video to {output_path}")

                # Upload to wandb
                if wandb_run is not None:
                    gif_path = convert_to_gif(output_path)
                    if gif_path:
                        wandb_run.log(
                            {
                                f"video/{name}": wandb.Image(gif_path),
                                f"prompt/{name}": prompt,
                                f"dimensions/{name}": f"{num_frames} frames, {width}x{height}",
                                "sample_index": i,
                            }
                        )

            num_generated += 1

        except Exception as e:
            print(f"  ❌ Failed: {e}")
            import traceback

            traceback.print_exc()
            continue

    print("\n" + "=" * 80)
    print("✅ Validation complete!")
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

# 1. TEXT MODE (old approach - encode prompts on-the-fly):
# python validate_t2v_dual_mode.py \
#     --meta_folder ./processed_meta \
#     --num_samples 10 \
#     --negative_prompt "blurry, low quality, distorted"

# 2. EMBEDDING MODE (new approach - use precomputed embeddings):
# python validate_t2v_dual_mode.py \
#     --meta_folder ./processed_meta \
#     --use_embeddings \
#     --num_samples 10

# 3. TEXT MODE with checkpoint and wandb:
# python validate_t2v_dual_mode.py \
#     --meta_folder ./processed_meta \
#     --checkpoint ./checkpoint-5000 \
#     --negative_prompt "blurry, low quality" \
#     --use_wandb \
#     --wandb_project wan_validation

# 4. EMBEDDING MODE with checkpoint and wandb:
# python validate_t2v_dual_mode.py \
#     --meta_folder ./processed_meta \
#     --checkpoint ./checkpoint-5000 \
#     --use_embeddings \
#     --use_wandb \
#     --wandb_project wan_validation

# 5. TEXT MODE with custom video dimensions:
# python validate_t2v_dual_mode.py \
#     --meta_folder ./processed_meta \
#     --height 480 \
#     --width 832 \
#     --num_frames 81 \
#     --num_inference_steps 50 \
#     --guidance_scale 5.0

# 6. EMBEDDING MODE (dimensions auto-inferred from latents):
# python validate_t2v_dual_mode.py \
#     --meta_folder ./processed_meta \
#     --use_embeddings \
#     --num_inference_steps 50 \
#     --guidance_scale 5.0
