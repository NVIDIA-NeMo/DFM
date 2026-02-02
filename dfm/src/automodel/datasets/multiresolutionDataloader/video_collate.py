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
Video-specific collate function and dataloader builder for multiresolution training.

Provides collation that transforms TextToVideoDataset output to match the
FlowMatchingPipeline expected batch format.
"""

import logging
from functools import partial
from typing import Dict, List, Tuple

import torch
from torch.utils.data import DataLoader

from dfm.src.automodel.datasets.multiresolutionDataloader.dataloader import SequentialBucketSampler
from dfm.src.automodel.datasets.multiresolutionDataloader.text_to_video_dataset import TextToVideoDataset


logger = logging.getLogger(__name__)


def collate_fn_video(batch: List[Dict], model_type: str = "wan") -> Dict:
    """
    Video-compatible collate function that transforms multiresolution batch output
    to match FlowMatchingPipeline expected format.

    Args:
        batch: List of samples from TextToVideoDataset
        model_type: Model type for handling model-specific fields ('wan', 'hunyuan')

    Returns:
        Dict compatible with FlowMatchingPipeline.step()
    """
    if len(batch) == 0:
        return {}

    # Verify all samples have same resolution (required for batching)
    resolutions = [tuple(item["bucket_resolution"].tolist()) for item in batch]
    assert len(set(resolutions)) == 1, f"Mixed resolutions in batch: {set(resolutions)}"

    # Stack video latents: each sample is (1, C, T, H, W) -> (B, C, T, H, W)
    # Use cat since samples have batch dim of 1
    video_latents = torch.cat([item["video_latents"] for item in batch], dim=0)

    # Stack text embeddings: each sample is (1, seq_len, embed_dim) -> (B, seq_len, embed_dim)
    text_embeddings = torch.cat([item["text_embeddings"] for item in batch], dim=0)

    # Build output dict compatible with FlowMatchingPipeline
    output = {
        "video_latents": video_latents,
        "text_embeddings": text_embeddings,
        "data_type": "video",
        "metadata": {
            "prompts": [item["prompt"] for item in batch],
            "video_paths": [item["video_path"] for item in batch],
            "bucket_ids": [item["bucket_id"] for item in batch],
            "aspect_ratios": [item["aspect_ratio"] for item in batch],
            "bucket_resolution": torch.stack([item["bucket_resolution"] for item in batch]),
            "original_resolution": torch.stack([item["original_resolution"] for item in batch]),
            "num_frames": [item["num_frames"] for item in batch],
        },
    }

    # Handle model-specific fields
    if model_type == "wan":
        # Wan models: optional text_mask
        if "text_mask" in batch[0]:
            output["text_mask"] = torch.cat([item["text_mask"] for item in batch], dim=0)

    elif model_type == "hunyuan":
        # HunyuanVideo: dual text encoders with masks
        if "text_mask" in batch[0]:
            output["text_mask"] = torch.cat([item["text_mask"] for item in batch], dim=0)
        if "text_embeddings_2" in batch[0]:
            output["text_embeddings_2"] = torch.cat([item["text_embeddings_2"] for item in batch], dim=0)
        if "text_mask_2" in batch[0]:
            output["text_mask_2"] = torch.cat([item["text_mask_2"] for item in batch], dim=0)
        if "image_embeds" in batch[0]:
            output["image_embeds"] = torch.cat([item["image_embeds"] for item in batch], dim=0)

    return output


def build_video_multiresolution_dataloader(
    *,
    # TextToVideoDataset parameters
    cache_dir: str,
    model_type: str = "wan",
    # Dataloader parameters
    batch_size: int = 1,
    dp_rank: int = 0,
    dp_world_size: int = 1,
    base_resolution: Tuple[int, int] = (512, 512),
    drop_last: bool = True,
    shuffle: bool = True,
    dynamic_batch_size: bool = False,
    num_workers: int = 4,
    pin_memory: bool = True,
    prefetch_factor: int = 2,
) -> Tuple[DataLoader, SequentialBucketSampler]:
    """
    Build a video-compatible multiresolution dataloader for TrainDiffusionRecipe.

    This wraps TextToVideoDataset and SequentialBucketSampler with a
    video-compatible collate function.

    Args:
        cache_dir: Directory containing preprocessed cache (metadata.json, shards, and resolution subdirs)
        model_type: Model type for handling model-specific fields ('wan', 'hunyuan')
        batch_size: Batch size per GPU
        dp_rank: Data parallel rank
        dp_world_size: Data parallel world size
        base_resolution: Base resolution for dynamic batch sizing
        drop_last: Drop incomplete batches
        shuffle: Shuffle data
        dynamic_batch_size: Scale batch size by resolution
        num_workers: DataLoader workers
        pin_memory: Pin memory for GPU transfer
        prefetch_factor: Prefetch batches per worker

    Returns:
        Tuple of (DataLoader, SequentialBucketSampler)
    """
    logger.info("Building video multiresolution dataloader:")
    logger.info(f"  cache_dir: {cache_dir}")
    logger.info(f"  model_type: {model_type}")
    logger.info(f"  batch_size: {batch_size}")
    logger.info(f"  dp_rank: {dp_rank}, dp_world_size: {dp_world_size}")

    # Create dataset
    dataset = TextToVideoDataset(
        cache_dir=cache_dir,
        model_type=model_type,
    )

    # Create sampler (reuses the same SequentialBucketSampler as image dataloader)
    sampler = SequentialBucketSampler(
        dataset,
        base_batch_size=batch_size,
        base_resolution=base_resolution,
        drop_last=drop_last,
        shuffle_buckets=shuffle,
        shuffle_within_bucket=shuffle,
        dynamic_batch_size=dynamic_batch_size,
        num_replicas=dp_world_size,
        rank=dp_rank,
    )

    # Create collate function with model_type bound
    collate_fn = partial(collate_fn_video, model_type=model_type)

    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_sampler=sampler,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        persistent_workers=num_workers > 0,
    )

    logger.info(f"  Dataset size: {len(dataset)}")
    logger.info(f"  Batches per epoch: {len(sampler)}")

    return dataloader, sampler
