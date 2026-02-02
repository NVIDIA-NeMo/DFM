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
Unified preprocessing tool for images and videos.

Supports:
- Images: FLUX (and other image models)
- Videos: Wan2.1, HunyuanVideo-1.5

Usage:
    # Image preprocessing
    python -m dfm.src.automodel.utils.preprocessing_multiprocess image \\
        --image_dir /path/to/images \\
        --output_dir /path/to/cache \\
        --processor flux

    # Video preprocessing
    python -m dfm.src.automodel.utils.preprocessing_multiprocess video \\
        --video_dir /path/to/videos \\
        --output_dir /path/to/cache \\
        --processor wan \\
        --resolution_preset 512p

    # List available processors
    python -m dfm.src.automodel.utils.preprocessing_multiprocess --list_processors
"""

import argparse
import hashlib
import json
import os
import pickle
import traceback
from multiprocessing import Pool
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from dfm.src.automodel.datasets.multiresolutionDataloader.multi_tier_bucketing import MultiTierBucketCalculator
from dfm.src.automodel.utils.processors import (
    BaseModelProcessor,
    BaseVideoProcessor,
    ProcessorRegistry,
    get_caption_loader,
)


# =============================================================================
# Global worker state (initialized once per process)
# =============================================================================
_worker_models: Optional[Dict[str, Any]] = None
_worker_processor: Optional[BaseModelProcessor] = None
_worker_calculator: Optional[MultiTierBucketCalculator] = None
_worker_device: Optional[str] = None
_worker_config: Optional[Dict[str, Any]] = None


# =============================================================================
# Image Preprocessing Functions
# =============================================================================


def _init_worker(processor_name: str, model_name: str, gpu_id: int, max_pixels: int):
    """Initialize worker process with models on assigned GPU."""
    global _worker_models, _worker_processor, _worker_calculator, _worker_device

    # Set CUDA_VISIBLE_DEVICES to isolate this GPU for the worker process.
    # After this, the selected GPU becomes cuda:0 (not cuda:{gpu_id}).
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    _worker_device = "cuda:0"

    _worker_processor = ProcessorRegistry.get(processor_name)
    _worker_models = _worker_processor.load_models(model_name, _worker_device)
    _worker_calculator = MultiTierBucketCalculator(quantization=64, max_pixels=max_pixels)

    print(f"Worker initialized on GPU {gpu_id}")


def _load_caption(image_path: Path, caption_field: str = "internvl") -> Optional[str]:
    """
    Load caption from JSON file for an image.

    DEPRECATED: Use _load_all_captions() instead for better performance.
    This function is kept for backward compatibility only.
    """
    image_name = image_path.name

    # Extract prefix: everything before '_sample'
    if "_sample" in image_name:
        prefix = image_name.rsplit("_sample", 1)[0]
    else:
        prefix = image_path.stem

    json_path = image_path.parent / f"{prefix}_internvl.json"

    if not json_path.exists():
        return None

    try:
        with open(json_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    if entry.get("file_name") == image_name:
                        return entry.get(caption_field, "")
                except json.JSONDecodeError:
                    continue
    except Exception:
        pass

    return None


def _load_all_captions(
    image_files: List[Path], caption_field: str = "internvl", verbose: bool = True
) -> Dict[str, str]:
    """
    Pre-load all captions from JSONL files into memory.

    This function eliminates the performance bottleneck of repeatedly opening
    and parsing the same JSONL files by loading all captions once upfront.

    Args:
        image_files: List of image file paths
        caption_field: Field name in JSONL to use ('internvl' or 'usr')
        verbose: Print progress information

    Returns:
        Dictionary mapping image filename to caption text
    """
    from collections import defaultdict

    if verbose:
        print("\nPre-loading captions from JSONL files...")

    # Group images by their JSONL file
    jsonl_to_images = defaultdict(list)

    for image_path in image_files:
        image_name = image_path.name

        # Extract prefix: everything before '_sample'
        if "_sample" in image_name:
            prefix = image_name.rsplit("_sample", 1)[0]
        else:
            prefix = image_path.stem

        json_path = image_path.parent / f"{prefix}_internvl.json"
        jsonl_to_images[json_path].append(image_name)

    # Load each JSONL file once and build caption dictionary
    caption_cache = {}
    loaded_files = 0
    missing_files = 0
    total_captions = 0

    for json_path, image_names in tqdm(jsonl_to_images.items(), desc="Loading JSONL files", disable=not verbose):
        if not json_path.exists():
            missing_files += 1
            # Images with missing JSONL will use filename fallback
            continue

        try:
            with open(json_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = json.loads(line)
                        file_name = entry.get("file_name")
                        if file_name and file_name in image_names:
                            caption = entry.get(caption_field, "")
                            if caption:
                                caption_cache[file_name] = caption
                                total_captions += 1
                    except json.JSONDecodeError:
                        continue
            loaded_files += 1
        except Exception as e:
            if verbose:
                print(f"Warning: Failed to load {json_path}: {e}")
            continue

    if verbose:
        print(f"Loaded {total_captions} captions from {loaded_files} JSONL files")
        if missing_files > 0:
            print(f"  {missing_files} JSONL files not found (will use filename fallback)")
        missing_captions = len(image_files) - total_captions
        if missing_captions > 0:
            print(f"  {missing_captions} images will use filename as caption")

    return caption_cache


def _validate_caption_files(image_files: List[Path], caption_field: str) -> Tuple[int, int, List[str]]:
    """
    Validate that caption files exist and are parseable.

    Args:
        image_files: List of image file paths
        caption_field: Field name to check in JSONL files

    Returns:
        (num_valid_files, num_missing_files, error_messages)
    """

    # Group images by their JSONL file
    jsonl_files = set()

    for image_path in image_files:
        image_name = image_path.name

        # Extract prefix: everything before '_sample'
        if "_sample" in image_name:
            prefix = image_name.rsplit("_sample", 1)[0]
        else:
            prefix = image_path.stem

        json_path = image_path.parent / f"{prefix}_internvl.json"
        jsonl_files.add(json_path)

    # Validate each JSONL file
    valid_files = 0
    missing_files = 0
    errors = []

    for json_path in jsonl_files:
        if not json_path.exists():
            missing_files += 1
            errors.append(f"Missing: {json_path}")
            continue

        try:
            with open(json_path, "r", encoding="utf-8") as f:
                line_count = 0
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    line_count += 1
                    try:
                        entry = json.loads(line)
                        # Basic validation: check structure
                        if "file_name" not in entry:
                            errors.append(f"Invalid format in {json_path}: missing 'file_name' field")
                            break
                    except json.JSONDecodeError as e:
                        errors.append(f"JSON error in {json_path} line {line_count}: {e}")
                        break
                else:
                    # File parsed successfully
                    valid_files += 1
        except Exception as e:
            errors.append(f"Failed to read {json_path}: {e}")
            continue

    return valid_files, missing_files, errors


def _process_image(args: Tuple) -> Optional[Dict]:
    """Process a single image using pre-initialized worker state."""
    image_path, output_dir, verify, caption = args

    try:
        image = Image.open(image_path).convert("RGB")
        orig_width, orig_height = image.size

        bucket = _worker_calculator.get_bucket_for_image(orig_width, orig_height)
        target_width, target_height = bucket["resolution"]

        resized_image, crop_offset = _worker_calculator.resize_and_crop(
            image, target_width, target_height, crop_mode="center"
        )

        image_tensor = _worker_processor.preprocess_image(resized_image)
        latent = _worker_processor.encode_image(image_tensor, _worker_models, _worker_device)

        if verify and not _worker_processor.verify_latent(latent, _worker_models, _worker_device):
            print(f"Verification failed: {image_path}")
            return None

        # Use pre-loaded caption with fallback to filename
        if not caption:
            caption = Path(image_path).stem.replace("_", " ")

        text_encodings = _worker_processor.encode_text(caption, _worker_models, _worker_device)

        # Save cache file
        resolution = f"{target_width}x{target_height}"
        cache_subdir = Path(output_dir) / resolution
        cache_subdir.mkdir(parents=True, exist_ok=True)

        cache_hash = hashlib.md5(f"{Path(image_path).absolute()}_{resolution}".encode()).hexdigest()
        cache_file = cache_subdir / f"{cache_hash}.pt"

        metadata = {
            "original_resolution": (orig_width, orig_height),
            "bucket_resolution": (target_width, target_height),
            "crop_offset": crop_offset,
            "prompt": caption,
            "image_path": str(Path(image_path).absolute()),
            "bucket_id": bucket["id"],
            "aspect_ratio": bucket["aspect_ratio"],
        }

        cache_data = _worker_processor.get_cache_data(latent, text_encodings, metadata)
        torch.save(cache_data, cache_file)

        return {
            "cache_file": str(cache_file),
            "image_path": str(Path(image_path).absolute()),
            "bucket_resolution": [target_width, target_height],
            "original_resolution": [orig_width, orig_height],
            "prompt": caption,
            "bucket_id": bucket["id"],
            "aspect_ratio": bucket["aspect_ratio"],
            "pixels": target_width * target_height,
            "model_type": _worker_processor.model_type,
        }

    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        traceback.print_exc()
        return None


def _get_image_files(image_dir: Path) -> List[Path]:
    """
    Recursively get all image files efficiently.

    Uses os.walk() for better performance on large directories compared to rglob().
    """
    image_files = []
    valid_extensions = {"jpg", "jpeg", "png", "webp", "bmp"}

    # Use os.walk for better performance on large directories
    for root, dirs, files in os.walk(image_dir):
        root_path = Path(root)
        for file in files:
            # Extract extension and check if it's a valid image file
            if "." in file:
                ext = file.lower().rsplit(".", 1)[-1]
                if ext in valid_extensions:
                    image_files.append(root_path / file)

    return sorted(image_files)


def _process_shard_on_gpu(
    gpu_id: int,
    image_files: List[Path],
    output_dir: str,
    processor_name: str,
    model_name: str,
    verify: bool,
    caption_cache: Dict[str, str],
    max_pixels: int,
) -> List[Dict]:
    """Process a shard of images on a specific GPU."""
    _init_worker(processor_name, model_name, gpu_id, max_pixels)

    results = []
    for image_path in tqdm(image_files, desc=f"GPU {gpu_id}", position=gpu_id):
        # Get caption from cache (or None if not found)
        caption = caption_cache.get(image_path.name)
        result = _process_image((str(image_path), output_dir, verify, caption))
        if result:
            results.append(result)

    return results


def preprocess_dataset(
    image_dir: str,
    output_dir: str,
    processor_name: str,
    model_name: Optional[str] = None,
    shard_size: int = 10000,
    verify: bool = False,
    caption_field: str = "internvl",
    max_images: Optional[int] = None,
    max_pixels: int = 256 * 256,
):
    """
    Preprocess image dataset with one process per GPU.

    Args:
        image_dir: Directory containing images
        output_dir: Output directory for cache
        processor_name: Name of processor to use (e.g., 'flux', 'sdxl')
        model_name: HuggingFace model name (uses processor default if None)
        shard_size: Number of images per metadata shard
        verify: Whether to verify latents can be decoded
        caption_field: Field to use from JSON captions ('internvl' or 'usr')
        max_images: Maximum number of images to process
        max_pixels: Maximum pixels per image
    """
    image_dir = Path(image_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get processor and resolve model name
    processor = ProcessorRegistry.get(processor_name)
    if model_name is None:
        model_name = processor.default_model_name

    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        raise RuntimeError("No GPUs available")

    print(f"Processor: {processor_name} ({processor.model_type})")
    print(f"Model: {model_name}")
    print(f"GPUs: {num_gpus}")
    print(f"Max pixels: {max_pixels}")

    # Get all image files
    print("\nScanning for images...")
    image_files = _get_image_files(image_dir)

    if max_images is not None:
        image_files = image_files[:max_images]

    print(f"Processing {len(image_files)} images")

    if not image_files:
        return

    # Validate caption files before processing
    print("\nValidating caption files...")
    num_valid, num_missing, errors = _validate_caption_files(image_files, caption_field)
    print(f"  Valid JSONL files: {num_valid}")
    print(f"  Missing JSONL files: {num_missing}")

    if errors and num_missing > len(set([img.parent / f"{img.stem}_internvl.json" for img in image_files])) * 0.5:
        print("\nWARNING: Many caption files missing or invalid. First 10 errors:")
        for err in errors[:10]:
            print(f"  {err}")
    elif errors and len(errors) <= 5:
        print("\nCaption file issues:")
        for err in errors:
            print(f"  {err}")

    # Pre-load all captions (PERFORMANCE OPTIMIZATION)
    caption_cache = _load_all_captions(image_files, caption_field, verbose=True)

    # Split images across GPUs
    chunks = [image_files[i::num_gpus] for i in range(num_gpus)]

    # Process with one worker per GPU
    all_metadata = []

    with Pool(processes=num_gpus) as pool:
        args = [
            (gpu_id, chunks[gpu_id], str(output_dir), processor_name, model_name, verify, caption_cache, max_pixels)
            for gpu_id in range(num_gpus)
        ]

        results = pool.starmap(_process_shard_on_gpu, args)

        for gpu_results in results:
            all_metadata.extend(gpu_results)

    # Save metadata in shards
    shard_files = []
    for shard_idx in range(0, len(all_metadata), shard_size):
        shard_data = all_metadata[shard_idx : shard_idx + shard_size]
        shard_file = output_dir / f"metadata_shard_{shard_idx // shard_size:04d}.json"
        with open(shard_file, "w") as f:
            json.dump(shard_data, f, indent=2)
        shard_files.append(shard_file.name)

    # Save config metadata (references shards instead of duplicating items)
    metadata_file = output_dir / "metadata.json"
    with open(metadata_file, "w") as f:
        json.dump(
            {
                "processor": processor_name,
                "model_name": model_name,
                "model_type": processor.model_type,
                "caption_field": caption_field,
                "max_pixels": max_pixels,
                "total_images": len(all_metadata),
                "num_shards": len(shard_files),
                "shard_size": shard_size,
                "shards": shard_files,
            },
            f,
            indent=2,
        )

    # Print summary
    print(f"\n{'=' * 50}")
    print(f"COMPLETE: {len(all_metadata)}/{len(image_files)} images")
    print(f"Output: {output_dir}")

    bucket_counts: Dict[str, int] = {}
    for item in all_metadata:
        res = f"{item['bucket_resolution'][0]}x{item['bucket_resolution'][1]}"
        bucket_counts[res] = bucket_counts.get(res, 0) + 1

    print("\nBucket distribution:")
    for res in sorted(bucket_counts.keys()):
        print(f"  {res}: {bucket_counts[res]}")


# =============================================================================
# Video Preprocessing Functions
# =============================================================================


def _init_video_worker(
    processor_name: str,
    model_name: str,
    gpu_id: int,
    max_pixels: int,
    video_config: Dict[str, Any],
):
    """Initialize video worker process with models on assigned GPU."""
    global _worker_models, _worker_processor, _worker_calculator, _worker_device, _worker_config

    # Set CUDA_VISIBLE_DEVICES to isolate this GPU for the worker process.
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    _worker_device = "cuda:0"
    _worker_config = video_config

    _worker_processor = ProcessorRegistry.get(processor_name)
    _worker_models = _worker_processor.load_models(model_name, _worker_device)

    # Create bucket calculator with processor's quantization (8 for video, 64 for image)
    quantization = getattr(_worker_processor, "quantization", 8)
    _worker_calculator = MultiTierBucketCalculator(quantization=quantization, max_pixels=max_pixels)

    print(f"Video worker initialized on GPU {gpu_id} (quantization={quantization})")


def _get_video_files(video_dir: Path) -> List[Path]:
    """
    Recursively get all video files.

    Uses os.walk() for better performance on large directories.
    """
    video_files = []
    valid_extensions = {"mp4", "avi", "mov", "mkv", "webm"}

    for root, dirs, files in os.walk(video_dir):
        root_path = Path(root)
        for file in files:
            if "." in file:
                ext = file.lower().rsplit(".", 1)[-1]
                if ext in valid_extensions:
                    video_files.append(root_path / file)

    return sorted(video_files)


def _get_video_dimensions(video_path: str) -> Tuple[int, int, int]:
    """Get video dimensions and frame count using OpenCV."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Failed to open video: {video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    return width, height, frame_count


def _extract_evenly_spaced_frames(
    video_path: str,
    num_frames: int,
    target_size: Tuple[int, int],
    resize_mode: str = "bilinear",
    center_crop: bool = True,
) -> Tuple[List[np.ndarray], List[int]]:
    """
    Extract evenly-spaced frames from a video.

    Args:
        video_path: Path to video file
        num_frames: Number of frames to extract
        target_size: Target (height, width) for resizing
        resize_mode: Interpolation mode for resizing
        center_crop: Whether to center crop to target aspect ratio

    Returns:
        Tuple of:
            - List of numpy arrays, each (H, W, C) in uint8
            - List of source frame indices (0-based)
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Failed to open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Calculate evenly-spaced frame indices
    if num_frames >= total_frames:
        frame_indices = list(range(total_frames))
    else:
        frame_indices = np.linspace(0, total_frames - 1, num_frames).astype(int).tolist()

    target_height, target_width = target_size

    # Map resize modes to OpenCV interpolation
    interp_map = {
        "bilinear": cv2.INTER_LINEAR,
        "bicubic": cv2.INTER_CUBIC,
        "nearest": cv2.INTER_NEAREST,
        "area": cv2.INTER_AREA,
        "lanczos": cv2.INTER_LANCZOS4,
    }
    interpolation = interp_map.get(resize_mode, cv2.INTER_LINEAR)

    frames = []
    actual_indices = []

    for target_idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, target_idx)
        ret, frame = cap.read()
        if not ret:
            continue

        # Convert BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Resize and optionally center crop
        if center_crop:
            # Calculate scale to cover target area
            scale = max(target_width / orig_width, target_height / orig_height)
            new_width = int(orig_width * scale)
            new_height = int(orig_height * scale)

            frame = cv2.resize(frame, (new_width, new_height), interpolation=interpolation)

            # Center crop
            start_x = (new_width - target_width) // 2
            start_y = (new_height - target_height) // 2
            frame = frame[start_y : start_y + target_height, start_x : start_x + target_width]
        else:
            # Direct resize (may change aspect ratio)
            frame = cv2.resize(frame, (target_width, target_height), interpolation=interpolation)

        frames.append(frame)
        actual_indices.append(target_idx)

    cap.release()
    return frames, actual_indices


def _frame_to_video_tensor(frame: np.ndarray, dtype: torch.dtype = torch.float16) -> torch.Tensor:
    """
    Convert a single frame to a 1-frame video tensor.

    Args:
        frame: (H, W, C) uint8 numpy array
        dtype: Target dtype

    Returns:
        (1, C, 1, H, W) tensor normalized to [-1, 1]
    """
    # (H, W, C) -> (C, H, W)
    tensor = torch.from_numpy(frame).float().permute(2, 0, 1)

    # Normalize to [-1, 1]
    tensor = tensor / 255.0
    tensor = (tensor - 0.5) / 0.5

    # Add batch and temporal dimensions: (C, H, W) -> (1, C, 1, H, W)
    tensor = tensor.unsqueeze(0).unsqueeze(2)

    return tensor.to(dtype)


def _process_video_frames_mode(args: Tuple) -> List[Dict]:
    """
    Process a video in frames mode - each frame becomes a separate sample.

    This matches Megatron behavior where each extracted frame is saved as a
    separate 1-frame sample for frame-level training.

    Args:
        args: Tuple of (video_path, output_dir, caption, config)

    Returns:
        List of result dictionaries, one per extracted frame
    """
    video_path, output_dir, caption, config = args

    try:
        # Get video dimensions
        orig_width, orig_height, total_frames = _get_video_dimensions(video_path)

        # Check if explicit target size is given (no bucketing)
        target_height = config.get("target_height")
        target_width = config.get("target_width")

        if target_height is not None and target_width is not None:
            # Explicit size: no bucketing
            bucket_id = None
            aspect_ratio = target_width / target_height
        else:
            # Use bucket calculator to find best resolution
            bucket = _worker_calculator.get_bucket_for_image(orig_width, orig_height)
            target_width, target_height = bucket["resolution"]
            bucket_id = bucket["id"]
            aspect_ratio = bucket["aspect_ratio"]

        # Extract evenly-spaced frames
        num_frames = config.get("num_frames", 10)
        frames, source_frame_indices = _extract_evenly_spaced_frames(
            video_path,
            num_frames=num_frames,
            target_size=(target_height, target_width),
            resize_mode=config.get("resize_mode", "bilinear"),
            center_crop=config.get("center_crop", True),
        )

        if not frames:
            print(f"No frames extracted from {video_path}")
            return []

        total_frames_extracted = len(frames)

        # Use caption with fallback to filename
        if not caption:
            caption = Path(video_path).stem.replace("_", " ")

        # Encode text ONCE (reuse for all frames)
        text_encodings = _worker_processor.encode_text(caption, _worker_models, _worker_device)

        # Process each frame individually
        results = []
        deterministic = config.get("deterministic", True)
        output_format = config.get("output_format", "meta")
        resolution = f"{target_width}x{target_height}"
        cache_subdir = Path(output_dir) / resolution
        cache_subdir.mkdir(parents=True, exist_ok=True)

        for frame_idx, (frame, source_idx) in enumerate(zip(frames, source_frame_indices)):
            # Convert single frame to 1-frame video tensor
            video_tensor = _frame_to_video_tensor(frame)

            # Encode with VAE
            latent = _worker_processor.encode_video(
                video_tensor,
                _worker_models,
                _worker_device,
                deterministic=deterministic,
            )

            # Prepare metadata for this frame
            # Note: first_frame and image_embeds are omitted in frames mode
            # (frames mode is intended for t2v training, not i2v conditioning)
            metadata = {
                "original_resolution": (orig_width, orig_height),
                "bucket_resolution": (target_width, target_height),
                "bucket_id": bucket_id,
                "aspect_ratio": aspect_ratio,
                "num_frames": 1,  # Always 1 for frame mode
                "total_original_frames": total_frames,
                "prompt": caption,
                "video_path": str(Path(video_path).absolute()),
                "deterministic": deterministic,
                "mode": "frames",
                # Frame-specific fields
                "frame_index": frame_idx + 1,  # 1-based index
                "total_frames_extracted": total_frames_extracted,
                "source_frame_index": source_idx,  # 0-based index in source video
            }

            # Get cache data from processor
            cache_data = _worker_processor.get_cache_data(latent, text_encodings, metadata)

            # Include frame index in hash to ensure unique filenames
            cache_hash = hashlib.md5(
                f"{Path(video_path).absolute()}_{resolution}_frame{frame_idx}".encode()
            ).hexdigest()

            if output_format == "meta":
                cache_file = cache_subdir / f"{cache_hash}.meta"
                with open(cache_file, "wb") as f:
                    pickle.dump(cache_data, f)
            else:  # pt format
                cache_file = cache_subdir / f"{cache_hash}.pt"
                torch.save(cache_data, cache_file)

            results.append(
                {
                    "cache_file": str(cache_file),
                    "video_path": str(Path(video_path).absolute()),
                    "bucket_resolution": [target_width, target_height],
                    "original_resolution": [orig_width, orig_height],
                    "num_frames": 1,
                    "prompt": caption,
                    "bucket_id": bucket_id,
                    "aspect_ratio": aspect_ratio,
                    "pixels": target_width * target_height,
                    "model_type": _worker_processor.model_type,
                    # Frame-specific fields
                    "frame_index": frame_idx + 1,
                    "total_frames_extracted": total_frames_extracted,
                    "source_frame_index": source_idx,
                }
            )

        return results

    except Exception as e:
        print(f"Error processing {video_path} in frames mode: {e}")
        traceback.print_exc()
        return []


def _process_video_video_mode(args: Tuple) -> Optional[Dict]:
    """
    Process a video in video mode - the original behavior.

    Extracts multiple frames and encodes them as a single multi-frame sample.

    Args:
        args: Tuple of (video_path, output_dir, caption, config)

    Returns:
        Result dictionary or None on error
    """
    video_path, output_dir, caption, config = args

    try:
        # Get video dimensions
        orig_width, orig_height, total_frames = _get_video_dimensions(video_path)

        # Check if explicit target size is given (no bucketing)
        target_height = config.get("target_height")
        target_width = config.get("target_width")

        if target_height is not None and target_width is not None:
            # Explicit size: no bucketing
            bucket_id = None
            aspect_ratio = target_width / target_height
        else:
            # Use bucket calculator to find best resolution
            bucket = _worker_calculator.get_bucket_for_image(orig_width, orig_height)
            target_width, target_height = bucket["resolution"]
            bucket_id = bucket["id"]
            aspect_ratio = bucket["aspect_ratio"]

        # Load video with target resolution
        num_frames = config.get("num_frames")
        target_frames = config.get("target_frames")

        video_tensor, first_frame = _worker_processor.load_video(
            video_path,
            target_size=(target_height, target_width),
            num_frames=target_frames or num_frames,
            resize_mode=config.get("resize_mode", "bilinear"),
            center_crop=config.get("center_crop", True),
        )

        actual_frames = video_tensor.shape[2]  # (1, C, T, H, W)

        # Use caption with fallback to filename
        if not caption:
            caption = Path(video_path).stem.replace("_", " ")

        # Encode video
        deterministic = config.get("deterministic", True)
        latent = _worker_processor.encode_video(
            video_tensor,
            _worker_models,
            _worker_device,
            deterministic=deterministic,
        )

        # Encode text
        text_encodings = _worker_processor.encode_text(caption, _worker_models, _worker_device)

        # Encode first frame for i2v (if processor supports it)
        image_embeds = None
        if hasattr(_worker_processor, "encode_first_frame"):
            image_embeds = _worker_processor.encode_first_frame(first_frame, _worker_models, _worker_device)

        # Prepare metadata
        metadata = {
            "original_resolution": (orig_width, orig_height),
            "bucket_resolution": (target_width, target_height),
            "bucket_id": bucket_id,
            "aspect_ratio": aspect_ratio,
            "num_frames": actual_frames,
            "total_original_frames": total_frames,
            "prompt": caption,
            "video_path": str(Path(video_path).absolute()),
            "first_frame": first_frame,
            "image_embeds": image_embeds,
            "deterministic": deterministic,
            "mode": config.get("mode", "video"),
        }

        # Get cache data from processor
        cache_data = _worker_processor.get_cache_data(latent, text_encodings, metadata)

        # Save cache file
        output_format = config.get("output_format", "meta")
        resolution = f"{target_width}x{target_height}"
        cache_subdir = Path(output_dir) / resolution
        cache_subdir.mkdir(parents=True, exist_ok=True)

        cache_hash = hashlib.md5(f"{Path(video_path).absolute()}_{resolution}_{actual_frames}".encode()).hexdigest()

        if output_format == "meta":
            cache_file = cache_subdir / f"{cache_hash}.meta"
            with open(cache_file, "wb") as f:
                pickle.dump(cache_data, f)
        else:  # pt format
            cache_file = cache_subdir / f"{cache_hash}.pt"
            torch.save(cache_data, cache_file)

        return {
            "cache_file": str(cache_file),
            "video_path": str(Path(video_path).absolute()),
            "bucket_resolution": [target_width, target_height],
            "original_resolution": [orig_width, orig_height],
            "num_frames": actual_frames,
            "prompt": caption,
            "bucket_id": bucket_id,
            "aspect_ratio": aspect_ratio,
            "pixels": target_width * target_height,
            "model_type": _worker_processor.model_type,
        }

    except Exception as e:
        print(f"Error processing {video_path}: {e}")
        traceback.print_exc()
        return None


def _process_video(args: Tuple) -> Optional[Union[Dict, List[Dict]]]:
    """
    Process a single video using pre-initialized worker state.

    Dispatches to the appropriate processing function based on mode:
    - 'video': Multi-frame encoding (original behavior)
    - 'frames': Frame-level encoding (each frame becomes a separate sample)

    Args:
        args: Tuple of (video_path, output_dir, caption, config)

    Returns:
        - In 'video' mode: Single result dict or None
        - In 'frames' mode: List of result dicts (one per frame)
    """
    video_path, output_dir, caption, config = args
    mode = config.get("mode", "video")

    if mode == "frames":
        return _process_video_frames_mode(args)
    else:
        return _process_video_video_mode(args)


def _process_video_shard_on_gpu(
    gpu_id: int,
    video_files: List[Path],
    output_dir: str,
    processor_name: str,
    model_name: str,
    caption_cache: Dict[str, str],
    max_pixels: int,
    video_config: Dict[str, Any],
) -> List[Dict]:
    """Process a shard of videos on a specific GPU."""
    _init_video_worker(processor_name, model_name, gpu_id, max_pixels, video_config)

    results = []
    for video_path in tqdm(video_files, desc=f"GPU {gpu_id}", position=gpu_id):
        caption = caption_cache.get(video_path.name)
        result = _process_video((str(video_path), output_dir, caption, video_config))

        if result is None:
            continue

        # Handle both single result (video mode) and list of results (frames mode)
        if isinstance(result, list):
            results.extend(result)
        else:
            results.append(result)

    return results


def preprocess_video_dataset(
    video_dir: str,
    output_dir: str,
    processor_name: str,
    model_name: Optional[str] = None,
    mode: str = "video",
    num_frames: int = 10,
    target_frames: Optional[int] = None,
    resolution_preset: Optional[str] = None,
    max_pixels: Optional[int] = None,
    target_height: Optional[int] = None,
    target_width: Optional[int] = None,
    resize_mode: str = "bilinear",
    center_crop: bool = True,
    deterministic: bool = True,
    output_format: str = "meta",
    caption_format: str = "sidecar",
    caption_field: str = "caption",
    shard_size: int = 10000,
    max_videos: Optional[int] = None,
):
    """
    Preprocess video dataset with one process per GPU.

    Args:
        video_dir: Directory containing videos
        output_dir: Output directory for cache
        processor_name: Name of processor ('wan', 'hunyuan')
        model_name: HuggingFace model name (uses processor default if None)
        mode: Processing mode ('video' or 'frames')
        num_frames: Number of frames for 'frames' mode
        target_frames: Target frame count (for HunyuanVideo 4n+1)
        resolution_preset: Resolution preset ('256p', '512p', '768p', '1024p', '1536p')
        max_pixels: Custom pixel budget (mutually exclusive with resolution_preset)
        target_height: Explicit target height (disables bucketing)
        target_width: Explicit target width (disables bucketing)
        resize_mode: Interpolation mode for resizing
        center_crop: Whether to center crop
        deterministic: Use deterministic latent encoding
        output_format: Output format ('meta' or 'pt')
        caption_format: Caption format ('sidecar', 'meta_json', 'jsonl')
        caption_field: Field name for captions
        shard_size: Number of videos per metadata shard
        max_videos: Maximum number of videos to process
    """
    video_dir = Path(video_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get processor and resolve model name
    processor = ProcessorRegistry.get(processor_name)
    if model_name is None:
        model_name = processor.default_model_name

    # Determine max_pixels
    if resolution_preset:
        if resolution_preset not in MultiTierBucketCalculator.RESOLUTION_PRESETS:
            raise ValueError(
                f"Unknown preset '{resolution_preset}'. "
                f"Available: {list(MultiTierBucketCalculator.RESOLUTION_PRESETS.keys())}"
            )
        max_pixels = MultiTierBucketCalculator.RESOLUTION_PRESETS[resolution_preset]
    elif max_pixels is None and target_height is None:
        # Default to 512p for videos
        max_pixels = 512 * 512

    # If explicit size given, disable bucketing
    use_bucketing = target_height is None or target_width is None
    if not use_bucketing and max_pixels is None:
        max_pixels = target_height * target_width  # Use explicit size as pixel budget

    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        raise RuntimeError("No GPUs available")

    print(f"Processor: {processor_name} ({processor.model_type})")
    print(f"Model: {model_name}")
    print(f"GPUs: {num_gpus}")
    print(f"Mode: {mode}")
    if use_bucketing:
        print(f"Max pixels: {max_pixels} (bucketing enabled)")
        print(f"Quantization: {getattr(processor, 'quantization', 8)}")
    else:
        print(f"Target size: {target_width}x{target_height} (bucketing disabled)")

    if hasattr(processor, "frame_constraint") and processor.frame_constraint:
        print(f"Frame constraint: {processor.frame_constraint}")

    # Get all video files
    print("\nScanning for videos...")
    video_files = _get_video_files(video_dir)

    if max_videos is not None:
        video_files = video_files[:max_videos]

    print(f"Found {len(video_files)} videos")

    if not video_files:
        return

    # Load captions using appropriate loader
    print(f"\nLoading captions (format: {caption_format}, field: {caption_field})...")
    caption_loader = get_caption_loader(caption_format)
    caption_cache = caption_loader.load_captions(video_files, caption_field)
    print(f"  Loaded {len(caption_cache)} captions")

    # Video config for workers
    video_config = {
        "mode": mode,
        "num_frames": num_frames,
        "target_frames": target_frames,
        "target_height": target_height if not use_bucketing else None,
        "target_width": target_width if not use_bucketing else None,
        "resize_mode": resize_mode,
        "center_crop": center_crop,
        "deterministic": deterministic,
        "output_format": output_format,
    }

    # Split videos across GPUs
    chunks = [video_files[i::num_gpus] for i in range(num_gpus)]

    # Process with one worker per GPU
    all_metadata = []

    with Pool(processes=num_gpus) as pool:
        args = [
            (
                gpu_id,
                chunks[gpu_id],
                str(output_dir),
                processor_name,
                model_name,
                caption_cache,
                max_pixels,
                video_config,
            )
            for gpu_id in range(num_gpus)
        ]

        results = pool.starmap(_process_video_shard_on_gpu, args)

        for gpu_results in results:
            all_metadata.extend(gpu_results)

    # Save metadata in shards
    shard_files = []
    for shard_idx in range(0, len(all_metadata), shard_size):
        shard_data = all_metadata[shard_idx : shard_idx + shard_size]
        shard_file = output_dir / f"metadata_shard_{shard_idx // shard_size:04d}.json"
        with open(shard_file, "w") as f:
            json.dump(shard_data, f, indent=2)
        shard_files.append(shard_file.name)

    # Save config metadata
    metadata_file = output_dir / "metadata.json"
    with open(metadata_file, "w") as f:
        json.dump(
            {
                "processor": processor_name,
                "model_name": model_name,
                "model_type": processor.model_type,
                "caption_format": caption_format,
                "caption_field": caption_field,
                "max_pixels": max_pixels,
                "mode": mode,
                "target_frames": target_frames,
                "total_videos": len(all_metadata),
                "num_shards": len(shard_files),
                "shard_size": shard_size,
                "shards": shard_files,
            },
            f,
            indent=2,
        )

    # Print summary
    print(f"\n{'=' * 50}")
    print(f"COMPLETE: {len(all_metadata)}/{len(video_files)} videos")
    print(f"Output: {output_dir}")

    bucket_counts: Dict[str, int] = {}
    for item in all_metadata:
        res = f"{item['bucket_resolution'][0]}x{item['bucket_resolution'][1]}"
        bucket_counts[res] = bucket_counts.get(res, 0) + 1

    print("\nBucket distribution:")
    for res in sorted(bucket_counts.keys()):
        print(f"  {res}: {bucket_counts[res]}")


# =============================================================================
# CLI Entry Point
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Unified preprocessing tool for images and videos",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Image preprocessing with FLUX
  python -m dfm.src.automodel.utils.preprocessing_multiprocess image \\
      --image_dir /data/images --output_dir /cache --processor flux

  # Video preprocessing with Wan2.1
  python -m dfm.src.automodel.utils.preprocessing_multiprocess video \\
      --video_dir /data/videos --output_dir /cache --processor wan \\
      --resolution_preset 512p --caption_format sidecar

  # Video preprocessing with HunyuanVideo
  python -m dfm.src.automodel.utils.preprocessing_multiprocess video \\
      --video_dir /data/videos --output_dir /cache --processor hunyuan \\
      --target_frames 121 --caption_format meta_json
        """,
    )

    parser.add_argument("--list_processors", action="store_true", help="List available processors and exit")

    subparsers = parser.add_subparsers(dest="command", help="Preprocessing type")

    # ===================
    # Image subcommand
    # ===================
    image_parser = subparsers.add_parser("image", help="Preprocess images")
    image_parser.add_argument("--image_dir", type=str, required=True, help="Input image directory")
    image_parser.add_argument("--output_dir", type=str, required=True, help="Output cache directory")
    image_parser.add_argument("--processor", type=str, default="flux", help="Processor name (default: flux)")
    image_parser.add_argument("--model_name", type=str, default=None, help="Model name (uses processor default)")
    image_parser.add_argument("--shard_size", type=int, default=10000, help="Metadata shard size")
    image_parser.add_argument("--verify", action="store_true", help="Verify latents can be decoded")
    image_parser.add_argument(
        "--caption_field", type=str, default="internvl", choices=["internvl", "usr"], help="Caption field in JSONL"
    )
    image_parser.add_argument("--max_images", type=int, default=None, help="Max images to process")

    # Resolution options (mutually exclusive)
    image_res_group = image_parser.add_mutually_exclusive_group()
    image_res_group.add_argument(
        "--resolution_preset",
        type=str,
        choices=["256p", "512p", "768p", "1024p", "1536p"],
        help="Resolution preset for bucketing",
    )
    image_res_group.add_argument("--max_pixels", type=int, help="Custom max pixel budget")

    # ===================
    # Video subcommand
    # ===================
    video_parser = subparsers.add_parser("video", help="Preprocess videos")
    video_parser.add_argument("--video_dir", type=str, required=True, help="Input video directory")
    video_parser.add_argument("--output_dir", type=str, required=True, help="Output cache directory")
    video_parser.add_argument(
        "--processor",
        type=str,
        required=True,
        choices=["wan", "wan2.1", "hunyuan", "hunyuanvideo", "hunyuanvideo-1.5"],
    )
    video_parser.add_argument("--model_name", type=str, default=None, help="Model name (uses processor default)")
    video_parser.add_argument("--mode", type=str, default="video", choices=["video", "frames"], help="Processing mode")
    video_parser.add_argument("--num_frames", type=int, default=10, help="Frames to extract in 'frames' mode")
    video_parser.add_argument(
        "--target_frames", type=int, default=None, help="Target frame count (e.g., 121 for HunyuanVideo)"
    )

    # Resolution options
    video_res_group = video_parser.add_mutually_exclusive_group()
    video_res_group.add_argument(
        "--resolution_preset",
        type=str,
        choices=["256p", "512p", "768p", "1024p", "1536p"],
        help="Resolution preset (videos bucketed by aspect ratio)",
    )
    video_res_group.add_argument("--max_pixels", type=int, help="Custom pixel budget for bucketing")

    # Explicit size options (disables bucketing)
    video_parser.add_argument("--height", type=int, default=None, help="Explicit height (disables bucketing)")
    video_parser.add_argument("--width", type=int, default=None, help="Explicit width (disables bucketing)")

    video_parser.add_argument(
        "--resize_mode",
        type=str,
        default="bilinear",
        choices=["bilinear", "bicubic", "nearest", "area", "lanczos"],
        help="Interpolation mode",
    )
    video_parser.add_argument("--center_crop", action="store_true", default=True, help="Center crop (default: True)")
    video_parser.add_argument("--no_center_crop", dest="center_crop", action="store_false", help="Disable center crop")
    video_parser.add_argument(
        "--deterministic", action="store_true", default=True, help="Use deterministic encoding (default: True)"
    )
    video_parser.add_argument(
        "--stochastic", dest="deterministic", action="store_false", help="Use stochastic (sampled) encoding"
    )
    video_parser.add_argument(
        "--caption_format",
        type=str,
        default="sidecar",
        choices=["sidecar", "meta_json", "jsonl"],
        help="Caption format",
    )
    video_parser.add_argument("--caption_field", type=str, default="caption", help="Caption field name")
    video_parser.add_argument(
        "--output_format", type=str, default="meta", choices=["meta", "pt"], help="Output file format"
    )
    video_parser.add_argument("--shard_size", type=int, default=10000, help="Metadata shard size")
    video_parser.add_argument("--max_videos", type=int, default=None, help="Max videos to process")

    args = parser.parse_args()

    # Handle --list_processors
    if args.list_processors:
        print("Available processors:")
        print()
        for name in ProcessorRegistry.list_available():
            proc = ProcessorRegistry.get(name)
            media_type = "video" if isinstance(proc, BaseVideoProcessor) else "image"
            quantization = getattr(proc, "quantization", 64)
            frame_constraint = getattr(proc, "frame_constraint", None) or "none"
            print(f"  {name}:")
            print(f"    type: {proc.model_type}")
            print(f"    media: {media_type}")
            print(f"    quantization: {quantization}")
            if media_type == "video":
                print(f"    frame_constraint: {frame_constraint}")
            print()
        return

    # Handle subcommands
    if args.command == "image":
        if args.resolution_preset:
            max_pixels = MultiTierBucketCalculator.RESOLUTION_PRESETS[args.resolution_preset]
        elif args.max_pixels:
            max_pixels = args.max_pixels
        else:
            max_pixels = 256 * 256

        preprocess_dataset(
            args.image_dir,
            args.output_dir,
            args.processor,
            args.model_name,
            args.shard_size,
            args.verify,
            args.caption_field,
            args.max_images,
            max_pixels,
        )

    elif args.command == "video":
        # Validate explicit size args
        if (args.height is None) != (args.width is None):
            parser.error("Both --height and --width must be specified together")

        preprocess_video_dataset(
            video_dir=args.video_dir,
            output_dir=args.output_dir,
            processor_name=args.processor,
            model_name=args.model_name,
            mode=args.mode,
            num_frames=args.num_frames,
            target_frames=args.target_frames,
            resolution_preset=args.resolution_preset,
            max_pixels=args.max_pixels,
            target_height=args.height,
            target_width=args.width,
            resize_mode=args.resize_mode,
            center_crop=args.center_crop,
            deterministic=args.deterministic,
            output_format=args.output_format,
            caption_format=args.caption_format,
            caption_field=args.caption_field,
            shard_size=args.shard_size,
            max_videos=args.max_videos,
        )

    else:
        # No subcommand - for backward compatibility, check for image_dir
        if hasattr(args, "image_dir") and args.image_dir:
            # Legacy mode
            if args.resolution_preset:
                max_pixels = MultiTierBucketCalculator.RESOLUTION_PRESETS[args.resolution_preset]
            elif hasattr(args, "max_pixels") and args.max_pixels:
                max_pixels = args.max_pixels
            else:
                max_pixels = 256 * 256

            preprocess_dataset(
                args.image_dir,
                args.output_dir,
                args.processor,
                args.model_name,
                args.shard_size,
                args.verify,
                args.caption_field,
                args.max_images,
                max_pixels,
            )
        else:
            parser.print_help()


if __name__ == "__main__":
    main()
