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
Text-to-Image dataset for multiresolution training.

Loads preprocessed .pt files from preprocessing_multiprocess.py and groups
samples by bucket_resolution for efficient batch collation.
"""

import logging
from pathlib import Path
from typing import Dict

import torch

from dfm.src.automodel.datasets.multiresolutionDataloader.base_dataset import BaseMultiresolutionDataset


logger = logging.getLogger(__name__)


class TextToImageDataset(BaseMultiresolutionDataset):
    """Text-to-Image dataset with hierarchical bucket organization."""

    def __init__(
        self,
        cache_dir: str,
        train_text_encoder: bool = False,
    ):
        """
        Args:
            cache_dir: Directory containing preprocessed cache
            train_text_encoder: If True, returns tokens instead of embeddings
        """
        self.train_text_encoder = train_text_encoder

        # Initialize base class with image quantization (64)
        super().__init__(cache_dir, quantization=64)

        logger.info(f"Loaded image dataset with {len(self.metadata)} samples")

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Load a single sample from .pt file."""
        item = self.metadata[idx]
        cache_file = Path(item["cache_file"])

        # Load cached data (.pt files are torch format)
        data = torch.load(cache_file, map_location="cpu")

        # Support both old "crop_resolution" and new "bucket_resolution" keys for backward compatibility
        bucket_res = item.get("bucket_resolution", item.get("crop_resolution"))

        # Prepare output
        output = {
            "latent": data["latent"],
            "bucket_resolution": torch.tensor(bucket_res),
            "original_resolution": torch.tensor(item["original_resolution"]),
            "crop_offset": torch.tensor(data.get("crop_offset", bucket_res)),
            "prompt": data["prompt"],
            "image_path": data["image_path"],
            "bucket_id": item["bucket_id"],
            "aspect_ratio": item.get("aspect_ratio", 1.0),
        }

        if self.train_text_encoder:
            output["clip_tokens"] = data["clip_tokens"].squeeze(0)
            output["t5_tokens"] = data["t5_tokens"].squeeze(0)
        else:
            output["clip_hidden"] = data["clip_hidden"].squeeze(0)
            output["pooled_prompt_embeds"] = data["pooled_prompt_embeds"].squeeze(0)
            output["prompt_embeds"] = data["prompt_embeds"].squeeze(0)

        return output
