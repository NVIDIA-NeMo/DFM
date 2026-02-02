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
Text-to-Video dataset for multiresolution training.

Loads preprocessed .meta files from preprocessing_multiprocess.py and groups
samples by bucket_resolution for efficient batch collation.
"""

import logging
import pickle
from pathlib import Path
from typing import Dict

import torch

from dfm.src.automodel.datasets.multiresolutionDataloader.base_dataset import BaseMultiresolutionDataset


logger = logging.getLogger(__name__)


class TextToVideoDataset(BaseMultiresolutionDataset):
    """
    Text-to-Video dataset with hierarchical bucket organization.

    This dataset loads preprocessed .meta (pickle) files produced by
    preprocessing_multiprocess.py. Samples are grouped by bucket_resolution
    to enable efficient batching of videos with the same spatial dimensions.

    Supports multiple video models (Wan, Hunyuan, etc.) via model_type parameter.
    """

    def __init__(
        self,
        cache_dir: str,
        model_type: str = "wan",
        device: str = "cpu",
    ):
        """
        Args:
            cache_dir: Directory containing preprocessed cache (metadata.json and .meta files)
            model_type: Model type for handling model-specific fields ('wan', 'hunyuan')
            device: Device to load tensors to (default: 'cpu' for DataLoader workers)
        """
        self.model_type = model_type
        self.device = device

        # Initialize base class with video quantization (8)
        super().__init__(cache_dir, quantization=8)

        logger.info(f"Loaded video dataset with {len(self.metadata)} samples (model_type={model_type})")

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Load a single sample from .meta file."""
        item = self.metadata[idx]
        cache_file = Path(item["cache_file"])

        # Load cached data (.meta files are pickle format)
        with open(cache_file, "rb") as f:
            data = pickle.load(f)

        # Extract core fields
        video_latents = data["video_latents"].to(self.device)
        text_embeddings = data["text_embeddings"].to(self.device)

        # Prepare output with common fields
        output = {
            "video_latents": video_latents,
            "text_embeddings": text_embeddings,
            "bucket_resolution": torch.tensor(item["bucket_resolution"]),
            "original_resolution": torch.tensor(item["original_resolution"]),
            "num_frames": item.get("num_frames", video_latents.shape[2] if video_latents.ndim == 5 else 1),
            "prompt": item.get("prompt", data.get("metadata", {}).get("prompt", "")),
            "video_path": item.get("video_path", ""),
            "bucket_id": item.get("bucket_id"),
            "aspect_ratio": item.get("aspect_ratio", 1.0),
        }

        # Handle model-specific fields
        if self.model_type == "wan":
            # Wan models: text_embeddings is primary, optional text_mask
            if "text_mask" in data:
                output["text_mask"] = data["text_mask"].to(self.device)

        elif self.model_type == "hunyuan":
            # HunyuanVideo: dual text encoders with masks
            if "text_mask" in data:
                output["text_mask"] = data["text_mask"].to(self.device)
            if "text_embeddings_2" in data:
                output["text_embeddings_2"] = data["text_embeddings_2"].to(self.device)
            if "text_mask_2" in data:
                output["text_mask_2"] = data["text_mask_2"].to(self.device)
            if "image_embeds" in data:
                output["image_embeds"] = data["image_embeds"].to(self.device)

        return output
