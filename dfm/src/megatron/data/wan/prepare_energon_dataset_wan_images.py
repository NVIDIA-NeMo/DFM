import os
import io
import json
import tarfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Iterable

import cv2
import numpy as np
import torch

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
        raise ValueError(
            f"Invalid resize_mode '{resize_mode}'. Choose from: {list(interpolation_map.keys())}"
        )
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


def _decode_image_bytes_to_rgb(image_bytes: bytes) -> Optional[np.ndarray]:
    array = np.frombuffer(image_bytes, dtype=np.uint8)
    img_bgr = cv2.imdecode(array, cv2.IMREAD_COLOR)
    if img_bgr is None:
        return None
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return img_rgb


def _image_to_video_tensor(
    image_rgb: np.ndarray,
    target_size: Optional[Tuple[int, int]],
    resize_mode: str,
    maintain_aspect_ratio: bool,
    center_crop: bool,
    target_dtype: torch.dtype,
) -> torch.Tensor:
    frame = _resize_frame(image_rgb, target_size, resize_mode, maintain_aspect_ratio, center_crop)
    frame = frame.astype(np.float32) / 255.0  # H, W, C in [0,1]

    video_array = frame[None, ...]  # T=1, H, W, C
    video_tensor = torch.from_numpy(video_array)  # T, H, W, C
    video_tensor = video_tensor.permute(3, 0, 1, 2).unsqueeze(0)  # 1, C, T=1, H, W
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
    caption = (caption or "").strip()
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


def _iter_tar_images_and_captions(tf: tarfile.TarFile, image_exts: Tuple[str, ...]) -> Iterable[Tuple[tarfile.TarInfo, Optional[tarfile.TarInfo]]]:
    members = [m for m in tf.getmembers() if m.isfile()]
    # Map stem -> caption member
    txt_map: Dict[str, tarfile.TarInfo] = {}
    for m in members:
        name = os.path.basename(m.name)
        if name.lower().endswith(".txt"):
            stem = os.path.splitext(name)[0]
            txt_map[stem] = m

    for m in members:
        name = os.path.basename(m.name)
        lower = name.lower()
        if lower.endswith(image_exts):
            stem = os.path.splitext(name)[0]
            caption_member = txt_map.get(stem, None)
            yield m, caption_member


def _read_tar_member_bytes(tf: tarfile.TarFile, member: tarfile.TarInfo) -> bytes:
    f = tf.extractfile(member)
    if f is None:
        return b""
    with f:
        return f.read()


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Prepare WAN encodings for image tar shards. Each .tar is written to a same-named directory "
            "containing per-image VAE latents (.pth), T5 embeddings (.pkl), and metadata (.json)."
        )
    )
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing .tar shards of images")
    parser.add_argument("--output_root", type=str, required=True, help="Root directory to write per-tar output dirs")
    parser.add_argument(
        "--model",
        default="Wan-AI/Wan2.1-T2V-14B-Diffusers",
        help=(
            "Wan2.1 model ID (e.g., Wan-AI/Wan2.1-T2V-14B-Diffusers or Wan-AI/Wan2.1-T2V-1.3B-Diffusers)"
        ),
    )
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Use stochastic encoding (sampling) instead of deterministic posterior mean",
    )
    parser.add_argument("--no-memory-optimization", action="store_true", help="Disable VAE slicing/tiling")
    parser.add_argument(
        "--image_exts",
        type=str,
        default=".jpg,.jpeg,.png,.webp",
        help="Comma-separated list of image extensions to include",
    )
    parser.add_argument("--height", type=int, default=None, help="Target height for image frames")
    parser.add_argument("--width", type=int, default=None, help="Target width for image frames")
    parser.add_argument(
        "--resize_mode",
        default="bilinear",
        choices=["bilinear", "bicubic", "nearest", "area", "lanczos"],
        help="Interpolation mode for resizing",
    )
    parser.add_argument("--no-aspect-ratio", action="store_true", help="Disable aspect ratio preservation")
    parser.add_argument("--center-crop", action="store_true", help="Center crop to exact target size after resize")
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip images whose output files already exist (all three: .pth, .pkl, .json)",
    )

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    # Target size
    target_size = None
    if args.height is not None and args.width is not None:
        target_size = (args.height, args.width)
    elif (args.height is None) ^ (args.width is None):
        parser.error("Both --height and --width must be specified together")

    # Init HF models
    device = "cuda"
    vae, text_encoder, tokenizer, model_dtype = _init_hf_models(
        model_id=args.model,
        device=device,
        enable_memory_optimization=not args.no_memory_optimization,
    )

    image_exts = tuple(ext.strip().lower() for ext in args.image_exts.split(",") if ext.strip())

    # Find tar files
    tar_files = sorted([p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() == ".tar"])
    if not tar_files:
        raise FileNotFoundError(f"No .tar files found in {input_dir}")

    # DEBUGGING
    # for tar_path in tar_files:
    for tar_path in tar_files[:1]:
        tar_stem = tar_path.name[:-4]  # drop .tar
        out_dir = output_root / tar_stem
        out_dir.mkdir(parents=True, exist_ok=True)

        processed = 0
        failed = 0

        # Open tar for streaming read
        try:
            with tarfile.open(tar_path, mode="r:*") as tf:
                for img_member, cap_member in _iter_tar_images_and_captions(tf, image_exts):
                    img_name = os.path.basename(img_member.name)
                    stem = os.path.splitext(img_name)[0]

                    latents_path = out_dir / f"{stem}.pth"
                    text_path = out_dir / f"{stem}.pkl"
                    meta_path = out_dir / f"{stem}.json"

                    if args.skip_existing and latents_path.exists() and text_path.exists() and meta_path.exists():
                        continue

                    try:
                        img_bytes = _read_tar_member_bytes(tf, img_member)
                        if not img_bytes:
                            failed += 1
                            continue
                        rgb = _decode_image_bytes_to_rgb(img_bytes)
                        if rgb is None:
                            failed += 1
                            continue

                        caption_text = ""
                        if cap_member is not None:
                            try:
                                caption_bytes = _read_tar_member_bytes(tf, cap_member)
                                caption_text = caption_bytes.decode("utf-8", errors="ignore")
                            except Exception:
                                caption_text = ""

                        video_tensor = _image_to_video_tensor(
                            image_rgb=rgb,
                            target_size=target_size,
                            resize_mode=args.resize_mode,
                            maintain_aspect_ratio=not args.no_aspect_ratio,
                            center_crop=args.center_crop,
                            target_dtype=model_dtype,
                        )

                        # Encode
                        text_embed = _encode_text(tokenizer, text_encoder, device, caption_text)
                        latents = _encode_video_latents(
                            vae, device, video_tensor, deterministic_latents=not args.stochastic
                        )

                        # Move to CPU and drop batch dim
                        text_embed_cpu = text_embed.detach().to(device="cpu")[0]
                        latents_cpu = latents.detach().to(device="cpu")[0]

                        # Save outputs
                        torch.save(latents_cpu, latents_path)
                        # Use pickle for text embeddings to keep exact dtype/shape
                        with open(text_path, "wb") as f:
                            import pickle

                            pickle.dump(text_embed_cpu, f, protocol=pickle.HIGHEST_PROTOCOL)

                        # Metadata
                        C, T, H, W = video_tensor.shape[1:]
                        json_data = {
                            "source_tar": str(tar_path),
                            "tar_member": img_member.name,
                            "image_name": img_name,
                            "processed_frames": int(T),  # always 1
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
                        with open(meta_path, "w", encoding="utf-8") as f:
                            json.dump(json_data, f, ensure_ascii=False)

                        processed += 1
                    except Exception:
                        failed += 1
                        continue
        except Exception as e:
            print(f"Failed to process tar {tar_path}: {e}")
            continue

        print(f"Processed tar {tar_path.name}: {processed} ok, {failed} failed. Output -> {out_dir}")


if __name__ == "__main__":
    main()


