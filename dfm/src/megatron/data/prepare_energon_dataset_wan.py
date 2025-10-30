import os
import json
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import webdataset as wds

from diffusers import AutoencoderKLWan
from transformers import AutoTokenizer, UMT5EncoderModel


def _map_interpolation(resize_mode: str) -> int:
    interpolation_map = {
        "bilinear": cv2.INTER_LINEAR,
        "bicubic": cv2.INTER_CUBIC,
        "nearest": cv2.INTER_NEAREST,
        "area": cv2.INTER_AREA,
        "lanczos": cv2.INTER_LANCZOS4,
    }
    if resize_mode not in interpolation_map:
        raise ValueError(f"Invalid resize_mode '{resize_mode}'. Choose from: {list(interpolation_map.keys())}")
    return interpolation_map[resize_mode]


def _calculate_resize_dimensions(
    original_height: int,
    original_width: int,
    target_size: Optional[Tuple[int, int]],
    maintain_aspect_ratio: bool,
) -> Tuple[int, int]:
    if target_size is None:
        return original_height, original_width

    target_height, target_width = target_size
    if not maintain_aspect_ratio:
        return target_height, target_width

    original_aspect = original_width / max(1, original_height)
    target_aspect = target_width / max(1, target_height)

    if original_aspect > target_aspect:
        new_width = target_width
        new_height = int(round(target_width / max(1e-6, original_aspect)))
    else:
        new_height = target_height
        new_width = int(round(target_height * original_aspect))

    return new_height, new_width


def _resize_frame(
    frame: np.ndarray,
    target_size: Optional[Tuple[int, int]],
    resize_mode: str,
    maintain_aspect_ratio: bool,
    center_crop: bool,
) -> np.ndarray:
    if target_size is None:
        return frame

    original_height, original_width = frame.shape[:2]
    resize_height, resize_width = _calculate_resize_dimensions(
        original_height, original_width, target_size, maintain_aspect_ratio
    )

    interpolation = _map_interpolation(resize_mode)
    resized_frame = cv2.resize(frame, (resize_width, resize_height), interpolation=interpolation)

    if maintain_aspect_ratio and center_crop:
        target_height, target_width = target_size
        if resize_height != target_height or resize_width != target_width:
            y_start = max(0, (resize_height - target_height) // 2)
            x_start = max(0, (resize_width - target_width) // 2)
            y_end = min(resize_height, y_start + target_height)
            x_end = min(resize_width, x_start + target_width)
            resized_frame = resized_frame[y_start:y_end, x_start:x_end]

            if resized_frame.shape[0] < target_height or resized_frame.shape[1] < target_width:
                pad_height = max(0, target_height - resized_frame.shape[0])
                pad_width = max(0, target_width - resized_frame.shape[1])
                resized_frame = np.pad(
                    resized_frame, ((0, pad_height), (0, pad_width), (0, 0)), mode="constant", constant_values=0
                )

    return resized_frame


def _read_sidecar_caption(jsonl_path: Path) -> str:
    if not jsonl_path.exists():
        return ""
    try:
        with open(jsonl_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                # Prefer keys used across datasets
                for key in ("vila_caption", "gemini_v2_caption", "caption", "text"):
                    if key in obj and isinstance(obj[key], str):
                        return obj[key]
                # If no known key, try first string value
                for v in obj.values():
                    if isinstance(v, str):
                        return v
                break
    except Exception:
        return ""
    return ""


def _get_total_frames(video_path: str) -> int:
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return max(0, total)


def _load_metadata(video_folder: Path) -> List[Dict]:
    meta_path = video_folder / "meta.json"
    if meta_path.exists():
        with open(meta_path, "r") as f:
            return json.load(f)

    # Fallback: scan for .mp4 files with sidecar .jsonl; use full frame range
    items: List[Dict] = []
    for entry in sorted(video_folder.iterdir()):
        if not entry.is_file():
            continue
        if entry.suffix.lower() != ".mp4":
            continue
        video_name = entry.name
        video_path = str(entry)
        total_frames = _get_total_frames(video_path)
        start_frame = 0
        end_frame = max(0, total_frames - 1)
        sidecar = entry.with_suffix("")
        # Handle names with additional dots gracefully
        sidecar_jsonl = Path(str(entry).rsplit(".", 1)[0] + ".jsonl")
        caption = _read_sidecar_caption(sidecar_jsonl)
        items.append(
            {
                "file_name": video_name,
                "start_frame": start_frame,
                "end_frame": end_frame,
                "vila_caption": caption,
            }
        )
    if not items:
        raise FileNotFoundError(f"No meta.json and no .mp4 files found in {video_folder}")
    return items


def _load_frames_cv2(
    video_path: str,
    start_frame: int,
    end_frame: int,
    target_size: Optional[Tuple[int, int]],
    resize_mode: str,
    maintain_aspect_ratio: bool,
    center_crop: bool,
    target_dtype: torch.dtype,
) -> torch.Tensor:
    cap = cv2.VideoCapture(video_path)
    frames: List[np.ndarray] = []

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    for frame_idx in range(start_frame, end_frame + 1):
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = _resize_frame(frame, target_size, resize_mode, maintain_aspect_ratio, center_crop)
        frame = frame.astype(np.float32) / 255.0
        frames.append(frame)
    cap.release()

    if not frames:
        raise ValueError(f"No frames loaded from {video_path}")

    video_array = np.array(frames)  # T, H, W, C in [0,1]
    video_tensor = torch.from_numpy(video_array)  # T, H, W, C
    video_tensor = video_tensor.permute(3, 0, 1, 2).unsqueeze(0)  # 1, C, T, H, W
    video_tensor = video_tensor.to(dtype=target_dtype)
    return video_tensor


@torch.no_grad()
def _init_hf_models(
    model_id: str,
    device: str,
    enable_memory_optimization: bool,
):
    dtype = torch.float16 if device.startswith("cuda") else torch.float32

    text_encoder = UMT5EncoderModel.from_pretrained(
        model_id,
        subfolder="text_encoder",
        torch_dtype=dtype,
    )
    text_encoder.to(device)
    text_encoder.eval()

    vae = AutoencoderKLWan.from_pretrained(
        model_id,
        subfolder="vae",
        torch_dtype=dtype,
    )
    vae.to(device)
    vae.eval()
    if enable_memory_optimization:
        vae.enable_slicing()
        vae.enable_tiling()

    tokenizer = AutoTokenizer.from_pretrained(model_id, subfolder="tokenizer")

    return vae, text_encoder, tokenizer, dtype


@torch.no_grad()
def _encode_text(
    tokenizer: AutoTokenizer,
    text_encoder: UMT5EncoderModel,
    device: str,
    caption: str,
) -> torch.Tensor:
    caption = caption.strip()
    inputs = tokenizer(
        caption,
        max_length=512,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
        return_attention_mask=True,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    outputs = text_encoder(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"]).last_hidden_state
    return outputs


@torch.no_grad()
def _encode_video_latents(
    vae: AutoencoderKLWan,
    device: str,
    video_tensor: torch.Tensor,
    deterministic_latents: bool,
) -> torch.Tensor:
    video_tensor = video_tensor.to(device=device, dtype=vae.dtype)
    video_tensor = video_tensor * 2.0 - 1.0  # [0,1] -> [-1,1]

    latent_dist = vae.encode(video_tensor)
    if deterministic_latents:
        video_latents = latent_dist.latent_dist.mean
    else:
        video_latents = latent_dist.latent_dist.sample()

    latent_mean = video_latents.mean().item()
    latent_std = video_latents.std().item()

    if abs(latent_mean) < 0.5 and 0.5 < latent_std < 2.0:
        final_latents = video_latents
    else:
        if not hasattr(vae.config, "latents_mean") or not hasattr(vae.config, "latents_std"):
            raise ValueError("Wan2.1 VAE requires latents_mean and latents_std in config")
        latents_mean = torch.tensor(vae.config.latents_mean, device=device, dtype=vae.dtype).view(1, -1, 1, 1, 1)
        latents_std = torch.tensor(vae.config.latents_std, device=device, dtype=vae.dtype).view(1, -1, 1, 1, 1)
        final_latents = (video_latents - latents_mean) / latents_std

    return final_latents


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Prepare WAN WebDataset shards using HF automodel encoders and resizing"
    )
    parser.add_argument("--video_folder", type=str, required=True, help="Folder containing videos and meta.json")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to write webdataset shards")
    parser.add_argument(
        "--model",
        default="Wan-AI/Wan2.1-T2V-14B-Diffusers",
        help="Wan2.1 model ID (e.g., Wan-AI/Wan2.1-T2V-14B-Diffusers or Wan-AI/Wan2.1-T2V-1.3B-Diffusers)",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use")
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Use stochastic encoding (sampling) instead of deterministic posterior mean",
    )
    parser.add_argument("--no-memory-optimization", action="store_true", help="Disable VAE slicing/tiling")
    parser.add_argument("--shard_maxcount", type=int, default=10000, help="Max samples per shard")

    # Resize arguments (match automodel)
    parser.add_argument("--height", type=int, default=None, help="Target height for video frames")
    parser.add_argument("--width", type=int, default=None, help="Target width for video frames")
    parser.add_argument(
        "--resize_mode",
        default="bilinear",
        choices=["bilinear", "bicubic", "nearest", "area", "lanczos"],
        help="Interpolation mode for resizing",
    )
    parser.add_argument("--no-aspect-ratio", action="store_true", help="Disable aspect ratio preservation")
    parser.add_argument("--center-crop", action="store_true", help="Center crop to exact target size after resize")

    args = parser.parse_args()

    video_folder = Path(args.video_folder)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    shard_pattern = str(output_dir / "shard-%06d.tar")

    # Target size
    target_size = None
    if args.height is not None and args.width is not None:
        target_size = (args.height, args.width)
    elif (args.height is None) ^ (args.width is None):
        parser.error("Both --height and --width must be specified together")

    # Init HF models
    vae, text_encoder, tokenizer, model_dtype = _init_hf_models(
        model_id=args.model,
        device=args.device,
        enable_memory_optimization=not args.no_memory_optimization,
    )

    # Load metadata list
    metadata_list = _load_metadata(video_folder)

    with wds.ShardWriter(shard_pattern, maxcount=args.shard_maxcount) as sink:
        written = 0
        for index, meta in enumerate(metadata_list):
            video_name = meta["file_name"]
            start_frame = int(meta["start_frame"])  # inclusive
            end_frame = int(meta["end_frame"])      # inclusive
            caption_text = meta.get("vila_caption", "")

            video_path = str(video_folder / video_name)
            # Load frames using the same OpenCV + resize path as automodel
            video_tensor = _load_frames_cv2(
                video_path=video_path,
                start_frame=start_frame,
                end_frame=end_frame,
                target_size=target_size,
                resize_mode=args.resize_mode,
                maintain_aspect_ratio=not args.no_aspect_ratio,
                center_crop=args.center_crop,
                target_dtype=model_dtype,
            )

            # Encode text and video with HF models exactly like automodel
            text_embed = _encode_text(tokenizer, text_encoder, args.device, caption_text)
            latents = _encode_video_latents(vae, args.device, video_tensor, deterministic_latents=not args.stochastic)

            # Move to CPU without changing dtype; keep exact values to match automodel outputs
            text_embed_cpu = text_embed.detach().to(device="cpu")
            latents_cpu = latents.detach().to(device="cpu")

            # Reshape to match Mcore's Wan input format
            text_embed_cpu = text_embed_cpu[0]
            latents_cpu = latents_cpu[0]

            # Build JSON side-info similar to prepare_energon script
            C, T, H, W = video_tensor.shape[1:]  # 1,C,T,H,W
            json_data = {
                "video_path": video_path,
                "processed_frames": int(T),
                "processed_height": int(H),
                "processed_width": int(W),
                "caption": caption_text,
                "deterministic_latents": bool(not args.stochastic),
                "memory_optimization": bool(not args.no_memory_optimization),
                "model_version": "wan2.1",
                "resize_settings": {
                    "target_size": target_size,
                    "resize_mode": args.resize_mode,
                    "maintain_aspect_ratio": bool(not args.no_aspect_ratio),
                    "center_crop": bool(args.center_crop),
                },
            }

            sample = {
                "__key__": f"{index:06}",
                "pth": latents_cpu,
                "pickle": pickle.dumps(text_embed_cpu),
                "json": json_data,
            }
            sink.write(sample)
            written += 1

    print("Done writing shards using HF automodel encoders.")


if __name__ == "__main__":
    main()


