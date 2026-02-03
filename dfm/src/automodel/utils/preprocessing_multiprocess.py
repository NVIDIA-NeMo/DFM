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

import argparse
import hashlib
import json
import os
import traceback
from multiprocessing import Pool
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from PIL import Image
from tqdm import tqdm

from dfm.src.automodel.datasets.multiresolutionDataloader.multi_tier_bucketing import MultiTierBucketCalculator
from dfm.src.automodel.utils.processors import BaseModelProcessor, ProcessorRegistry


# Global worker state (initialized once per process)
_worker_models: Optional[Dict[str, Any]] = None
_worker_processor: Optional[BaseModelProcessor] = None
_worker_calculator: Optional[MultiTierBucketCalculator] = None
_worker_device: Optional[str] = None


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
            "crop_resolution": (target_width, target_height),
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
            "crop_resolution": [target_width, target_height],
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
    Preprocess dataset with one process per GPU.

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
        res = f"{item['crop_resolution'][0]}x{item['crop_resolution'][1]}"
        bucket_counts[res] = bucket_counts.get(res, 0) + 1

    print("\nBucket distribution:")
    for res in sorted(bucket_counts.keys()):
        print(f"  {res}: {bucket_counts[res]}")


def main():
    parser = argparse.ArgumentParser(description="Preprocess images (one process per GPU)")

    parser.add_argument("--list_processors", action="store_true", help="List available processors")
    parser.add_argument("--image_dir", type=str, help="Input image directory")
    parser.add_argument("--output_dir", type=str, help="Output cache directory")
    parser.add_argument("--processor", type=str, default="flux", help="Processor name")
    parser.add_argument("--model_name", type=str, default=None, help="Model name")
    parser.add_argument("--shard_size", type=int, default=10000, help="Metadata shard size")
    parser.add_argument("--verify", action="store_true", help="Verify latents")
    parser.add_argument("--caption_field", type=str, default="internvl", choices=["internvl", "usr"])
    parser.add_argument("--max_images", type=int, default=None, help="Max images to process")
    parser.add_argument("--max_pixels", type=int, default=None, help="Max pixels per image")
    parser.add_argument(
        "--resolution_preset", type=str, default=None, choices=["256p", "512p", "768p", "1024p", "1536p"]
    )

    args = parser.parse_args()

    if args.list_processors:
        print("Available processors:")
        for name in ProcessorRegistry.list_available():
            proc = ProcessorRegistry.get(name)
            print(f"  {name}: {proc.model_type}")
        return

    if not args.image_dir or not args.output_dir:
        parser.error("--image_dir and --output_dir are required")

    if args.resolution_preset and args.max_pixels:
        parser.error("Cannot specify both --resolution_preset and --max_pixels")

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


if __name__ == "__main__":
    main()
