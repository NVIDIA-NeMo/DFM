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

from typing import Tuple

import torch


def pack_latents(latents: torch.Tensor, batch_size: int, num_channels_latents: int, height: int, width: int) -> torch.Tensor:
    """Pack latents for FLUX processing.

    Rearranges [B, C, H, W] -> [B, (H/2)*(W/2), C*4].

    Args:
        latents: Input tensor [B, C, H, W].
        batch_size: Batch size.
        num_channels_latents: Number of latent channels.
        height: Latent height.
        width: Latent width.

    Returns:
        Packed tensor [B, num_patches, C*4].
    """
    latents = latents.view(batch_size, num_channels_latents, height // 2, 2, width // 2, 2)
    latents = latents.permute(0, 2, 4, 1, 3, 5)
    latents = latents.reshape(batch_size, (height // 2) * (width // 2), num_channels_latents * 4)
    return latents


def unpack_latents(latents: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """Unpack latents from FLUX format.

    Rearranges [B, num_patches, C*4] -> [B, C, H, W].

    Args:
        latents: Packed tensor [B, num_patches, C*4].
        height: Target height.
        width: Target width.

    Returns:
        Unpacked tensor [B, C, H, W].
    """
    batch_size, num_patches, channels = latents.shape

    # Adjust h and w for patching
    latents = latents.view(batch_size, height // 2, width // 2, channels // 4, 2, 2)
    latents = latents.permute(0, 3, 1, 4, 2, 5)

    latents = latents.reshape(batch_size, channels // 4, height, width)

    return latents


def prepare_latent_image_ids(batch_size: int, height: int, width: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Prepare latent image IDs for position encoding.

    Args:
        batch_size: Batch size.
        height: Latent height.
        width: Latent width.
        device: Device to place the tensor on.
        dtype: Data type for the tensor.

    Returns:
        Latent image IDs tensor [B, (H/2)*(W/2), 3].
    """
    latent_image_ids = torch.zeros(height // 2, width // 2, 3)
    latent_image_ids[..., 1] = latent_image_ids[..., 1] + torch.arange(height // 2)[:, None]
    latent_image_ids[..., 2] = latent_image_ids[..., 2] + torch.arange(width // 2)[None, :]

    latent_image_ids = latent_image_ids[None, :].repeat(batch_size, 1, 1, 1)
    latent_image_ids = latent_image_ids.reshape(batch_size, (height // 2) * (width // 2), 3)

    return latent_image_ids.to(device=device, dtype=dtype)


def prepare_text_ids(batch_size: int, text_seq_len: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Prepare text IDs for position encoding.

    Args:
        batch_size: Batch size.
        text_seq_len: Text sequence length.
        device: Device to place the tensor on.
        dtype: Data type for the tensor.

    Returns:
        Text IDs tensor [B, text_seq_len, 3].
    """
    text_ids = torch.zeros(batch_size, text_seq_len, 3, device=device, dtype=dtype)
    text_ids[..., 0] = torch.arange(text_seq_len, device=device, dtype=dtype)
    return text_ids



