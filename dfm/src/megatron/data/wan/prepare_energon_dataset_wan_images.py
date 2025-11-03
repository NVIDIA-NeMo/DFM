import os
import io
import json
import pickle
import tarfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Iterable, Any
import multiprocessing as mp
import math

import cv2
import numpy as np
import torch
import webdataset as wds

from diffusers import AutoencoderKLWan
from transformers import AutoTokenizer, UMT5EncoderModel
from tqdm import tqdm


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
    target_height, target_width = target_size

    interpolation = _map_interpolation(resize_mode)

    if not maintain_aspect_ratio:
        resized_frame = cv2.resize(frame, (target_width, target_height), interpolation=interpolation)
        return resized_frame

    if center_crop:
        # Resize-to-cover: scale so both dims >= target, then center-crop to exact target
        scale = max(target_height / max(1, original_height), target_width / max(1, original_width))
        resize_height = int(math.ceil(original_height * scale))
        resize_width = int(math.ceil(original_width * scale))

        resized_frame = cv2.resize(frame, (resize_width, resize_height), interpolation=interpolation)

        y_start = max(0, (resize_height - target_height) // 2)
        x_start = max(0, (resize_width - target_width) // 2)
        y_end = y_start + target_height
        x_end = x_start + target_width

        # Bound checks (should be safe due to ceil, but guard anyway)
        y_start = min(y_start, max(0, resize_height - target_height))
        x_start = min(x_start, max(0, resize_width - target_width))
        y_end = min(y_end, resize_height)
        x_end = min(x_end, resize_width)

        cropped = resized_frame[y_start:y_end, x_start:x_end]

        # If due to rounding one dim is still short, pad minimally (rare)
        pad_h = max(0, target_height - cropped.shape[0])
        pad_w = max(0, target_width - cropped.shape[1])
        if pad_h > 0 or pad_w > 0:
            cropped = np.pad(
                cropped,
                ((pad_h // 2, pad_h - pad_h // 2), (pad_w // 2, pad_w - pad_w // 2), (0, 0)),
                mode="edge",
            )
            cropped = cropped[:target_height, :target_width]

        return cropped

    # Aspect-preserving resize-to-fit (no crop): may be smaller than target in one dim
    resize_height, resize_width = _calculate_resize_dimensions(
        original_height, original_width, target_size, maintain_aspect_ratio
    )
    resized_frame = cv2.resize(frame, (resize_width, resize_height), interpolation=interpolation)
    return resized_frame


def _decode_image_bytes_to_rgb(image_bytes: bytes) -> Optional[np.ndarray]:
    array = np.frombuffer(image_bytes, dtype=np.uint8)
    img_bgr = cv2.imdecode(array, cv2.IMREAD_COLOR)
    if img_bgr is None:
        return None
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return img_rgb


def _select_target_size_for_image(image_rgb: np.ndarray) -> Tuple[int, int]:
    h, w = image_rgb.shape[:2]
    if h <= w:
        return (480, 832)
    else:
        return (832, 480)


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
    # Pad to 512, then slice back to the non-padded length
    inputs = tokenizer(
        [caption],
        padding="max_length",
        truncation=True,
        max_length=512,
        return_tensors="pt",
        return_attention_mask=True,
        add_special_tokens=True,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    outputs = text_encoder(
        input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"]
    ).last_hidden_state  # [1, L, C]
    seq_len = int(inputs["attention_mask"][0].sum().item())
    return outputs[0, :seq_len, :]


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


def _process_tar_with_models(
    tar_path: Path,
    image_exts: Tuple[str, ...],
    opts: Dict[str, Any],
    device: str,
    vae: AutoencoderKLWan,
    text_encoder: UMT5EncoderModel,
    tokenizer: AutoTokenizer,
    model_dtype: torch.dtype,
    sink: Any,
    index: int,
    tqdm_position: int = 0,
) -> int:
    processed = 0
    failed = 0

    try:
        with tarfile.open(tar_path, mode="r:*") as tf:
            pairs = list(_iter_tar_images_and_captions(tf, image_exts))
            for img_member, cap_member in tqdm(
                pairs, total=len(pairs), desc=f"{tar_path.name}", unit="img", position=tqdm_position, leave=False
            ):
                img_name = os.path.basename(img_member.name)

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

                    target_size = _select_target_size_for_image(rgb)
                    video_tensor = _image_to_video_tensor(
                        image_rgb=rgb,
                        target_size=target_size,
                        resize_mode=opts["resize_mode"],
                        maintain_aspect_ratio=not opts.get("no_aspect_ratio", False),
                        center_crop=opts.get("center_crop", False),
                        target_dtype=model_dtype,
                    )

                    text_embed = _encode_text(tokenizer, text_encoder, device, caption_text)
                    latents = _encode_video_latents(
                        vae, device, video_tensor, deterministic_latents=not opts.get("stochastic", False)
                    )

                    # text_embed is already sliced to non-padded tokens: [L_actual, C]
                    text_embed_cpu = text_embed.detach().to(device="cpu")
                    latents_cpu = latents.detach().to(device="cpu")[0]

                    C, T, H, W = video_tensor.shape[1:]
                    json_data = {
                        "source_tar": str(tar_path),
                        "tar_member": img_member.name,
                        "image_name": img_name,
                        "processed_frames": int(T),
                        "processed_height": int(H),
                        "processed_width": int(W),
                        "caption": caption_text,
                        "deterministic_latents": bool(not opts.get("stochastic", False)),
                        "memory_optimization": bool(not opts.get("no_memory_optimization", False)),
                        "model_version": "wan2.1",
                        "resize_settings": {
                            "target_size": target_size,
                            "resize_mode": opts["resize_mode"],
                            "maintain_aspect_ratio": bool(not opts.get("no_aspect_ratio", False)),
                            "center_crop": bool(opts.get("center_crop", False)),
                        },
                    }

                    sample = {
                        "__key__": f"{index:09}",
                        "pth": latents_cpu,
                        "pickle": pickle.dumps(text_embed_cpu, protocol=pickle.HIGHEST_PROTOCOL),
                        "json": json_data,
                    }
                    sink.write(sample)

                    index += 1
                    processed += 1
                except Exception:
                    failed += 1
                    continue
    except Exception as e:
        print(f"Failed to process tar {tar_path}: {e}")
        return index

    print(f"Processed tar {tar_path.name}: {processed} ok, {failed} failed. WDS written")
    return index


def _worker_run(
    rank: int,
    device: str,
    tar_paths: List[str],
    in_root: str,
    out_root: str,
    image_exts: Tuple[str, ...],
    opts: Dict[str, Any],
):
    try:
        torch.cuda.set_device(int(device.split(":")[-1]))
    except Exception:
        pass

    vae, text_encoder, tokenizer, model_dtype = _init_hf_models(
        model_id=opts["model"],
        device=device,
        enable_memory_optimization=not opts.get("no_memory_optimization", False),
    )

    out_root_path = Path(out_root)
    in_root_path = Path(in_root)

    # DEBUGGING
    for tar_str in tar_paths:
    # for tar_str in tar_paths[:1]:
        tar_path = Path(tar_str)
        # Mirror the original directory structure from input_dir under output_root
        try:
            rel_parent = tar_path.parent.relative_to(in_root_path)
        except Exception:
            rel_parent = Path("")
        out_dir = out_root_path / rel_parent
        out_dir.mkdir(parents=True, exist_ok=True)

        out_tar = out_dir / f"{tar_path.stem}.tar"
        if opts.get("skip_existing") and out_tar.exists():
            continue

        index = 0
        with wds.TarWriter(str(out_tar)) as sink:
            index = _process_tar_with_models(
                tar_path=tar_path,
                image_exts=image_exts,
                opts=opts,
                device=device,
                vae=vae,
                text_encoder=text_encoder,
                tokenizer=tokenizer,
                model_dtype=model_dtype,
                sink=sink,
                index=index,
                tqdm_position=rank,
            )

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Prepare WAN encodings for image tar shards and write WebDataset shards (pth, pickle, json)."
        )
    )
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing .tar shards of images")
    parser.add_argument("--output_dir", type=str, required=False, help="Directory to write webdataset shards")
    parser.add_argument(
        "--output_root",
        type=str,
        required=False,
        help="Deprecated alias for --output_dir; if provided, will be used as output_dir",
    )
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
        help="No-op in WDS mode; retained for CLI compatibility",
    )
    parser.add_argument("--shard_maxcount", type=int, default=10000, help="Max samples per WDS shard")
    parser.add_argument(
        "--gpus",
        type=str,
        default="0",
        help="Comma-separated GPU indices to use (e.g., '0,1,2,3')",
    )

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    # Resolve output directory (support legacy --output_root)
    resolved_output_dir = args.output_dir or args.output_root
    if resolved_output_dir is None:
        parser.error("--output_dir must be specified (or legacy --output_root)")
    output_root = Path(resolved_output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    image_exts = tuple(ext.strip().lower() for ext in args.image_exts.split(",") if ext.strip())

    # Find tar files
    tar_files = sorted([p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() == ".tar"])
    if not tar_files:
        raise FileNotFoundError(f"No .tar files found in {input_dir}")

    # Parse GPU list and shard tars
    gpu_ids = [s.strip() for s in args.gpus.split(",") if s.strip()]
    devices = [f"cuda:{gid}" for gid in gpu_ids]
    num_workers = len(devices) if devices else 1

    shards: List[List[str]] = [[] for _ in range(num_workers)]
    for idx, tar_path in enumerate(tar_files):
        shards[idx % num_workers].append(str(tar_path))

    opts = {
        "model": args.model,
        "stochastic": bool(args.stochastic),
        "no_memory_optimization": bool(args.no_memory_optimization),
        "resize_mode": args.resize_mode,
        "no_aspect_ratio": bool(args.no_aspect_ratio),
        "center_crop": bool(args.center_crop),
        "skip_existing": bool(args.skip_existing),
        "shard_maxcount": int(args.shard_maxcount),
    }

    # Ensure CUDA-safe multiprocessing
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    if num_workers == 1:
        _worker_run(0, devices[0] if devices else "cuda:0", shards[0], str(input_dir), str(output_root), image_exts, opts)
    else:
        procs: List[mp.Process] = []
        for rank, device in enumerate(devices):
            if not shards[rank]:
                continue
            p = mp.Process(
                target=_worker_run,
                args=(rank, device, shards[rank], str(input_dir), str(output_root), image_exts, opts),
                daemon=False,
            )
            p.start()
            procs.append(p)
        for p in procs:
            p.join()


if __name__ == "__main__":
    main()


