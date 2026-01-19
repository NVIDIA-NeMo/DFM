# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

# pylint: disable=C0115,C0116,C0301

"""
Task encoder for WAN dataset with precomputed latents.

This module provides WanLatentTaskEncoder which handles precomputed VAE-encoded
video latents and text embeddings. It differs from WanTaskEncoder in that it
expects latents that are already VAE-encoded rather than raw video files.

Expected Energon dataset structure per sample:
  - latent.pth: RGB video latents (precomputed, already VAE-encoded) [C, T, H, W]
  - txt_emb.pth: Text embeddings (already padded to [512, dim])
  - depth_latent.pth: Depth latents (optional)
  - json: Metadata (resolution, fps, etc.)

The cook function maps these to the format expected by the parent WanTaskEncoder:
  - latent.pth -> pth (video latents)
  - txt_emb.pth -> pickle (text embeddings)
"""

from megatron.energon.task_encoder.cooking import Cooker, basic_sample_keys

from dfm.src.megatron.data.wan.wan_taskencoder import WanTaskEncoder


def cook_latent(sample: dict) -> dict:
    """
    Cook function for precomputed latent samples.

    Maps the precomputed latent file naming convention to the format
    expected by the parent WanTaskEncoder class.

    Args:
        sample (dict): Raw sample from Energon dataset containing:
            - "latent.pth": RGB video latents (precomputed)
            - "txt_emb.pth": Text embeddings
            - "depth_latent.pth" (optional): Depth latents
            - "json": Metadata

    Returns:
        dict: Processed sample with keys mapped to parent's expected format:
            - "pth": Video latent tensor (from latent.pth)
            - "pickle": Text embeddings (from txt_emb.pth)
            - "json": Metadata
    """
    return dict(
        **basic_sample_keys(sample),
        pth=sample["latent.pth"],  # Map latent.pth -> pth
        pickle=sample["txt_emb.pth"],  # Map txt_emb.pth -> pickle
        json=sample.get("json", {}),
    )


class WanLatentTaskEncoder(WanTaskEncoder):
    """
    Task encoder for WAN dataset with precomputed latents.

    This class inherits from WanTaskEncoder and only overrides the cook function
    to handle the different file naming convention used for precomputed latents:
      - latent.pth (precomputed VAE-encoded video) instead of raw video in pth
      - txt_emb.pth (pre-encoded text embeddings) instead of pickle

    All other processing is handled by the parent class:
      - Patchifying video latents
      - Grid size calculation
      - Text embedding padding to 512 tokens
      - Context parallelism padding
      - Sequence packing

    Attributes:
        use_depth_latent (bool): Whether to load and use depth latents.
            Note: Currently depth latents are loaded but not actively used
            in the encoding pipeline. They can be accessed via video_metadata.

    Example usage:
        task_encoder = WanLatentTaskEncoder(
            seq_length=500,
            packing_buffer_size=100,
            patch_spatial=2,
            patch_temporal=1,
            use_depth_latent=False,  # Set to True if needed
        )
    """

    cookers = [
        Cooker(cook_latent),
    ]

    def __init__(
        self,
        *args,
        use_depth_latent: bool = False,
        **kwargs,
    ):
        """
        Initialize the WanLatentTaskEncoder.

        Args:
            use_depth_latent (bool): Flag to enable depth latent loading.
                Defaults to False for memory optimization.
            *args: Additional positional arguments passed to parent WanTaskEncoder.
            **kwargs: Additional keyword arguments passed to parent WanTaskEncoder.
                Common kwargs include:
                - seq_length (int): Maximum sequence length
                - packing_buffer_size (int): Buffer size for sequence packing
                - patch_spatial (int): Spatial patch size (default: 2)
                - patch_temporal (int): Temporal patch size (default: 1)
        """
        super().__init__(*args, **kwargs)
        self.use_depth_latent = use_depth_latent
        # All other initialization (patchifying, grid calculation, etc.)
        # is handled by the parent WanTaskEncoder class
