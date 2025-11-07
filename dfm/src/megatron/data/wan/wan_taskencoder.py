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

from typing import List

import torch
import torch.nn.functional as F
from megatron.energon import SkipSample, stateless
from megatron.energon.task_encoder.cooking import Cooker, basic_sample_keys

from dfm.src.megatron.data.dit.diffusion_sample import DiffusionSample
from dfm.src.megatron.data.dit.diffusion_task_encoder_with_sp import DiffusionTaskEncoderWithSequencePacking
from dfm.src.megatron.model.wan.utils import grid_sizes_calculation, patchify_single_sample


def cook(sample: dict) -> dict:
    """
    Processes a raw sample dictionary from energon dataset and returns a new dictionary with specific keys.

    Args:
        sample (dict): The input dictionary containing the raw sample data.

    Returns:
        dict: A new dictionary containing the processed sample data with the following keys:
            - All keys from the result of `basic_sample_keys(sample)`
            - 'json': The contains meta data like resolution, aspect ratio, fps, etc.
            - 'pth': contains video latent tensor
            - 'pickle': contains text embeddings
    """
    return dict(
        **basic_sample_keys(sample),
        json=sample["json"],
        pth=sample["pth"],
        pickle=sample["pickle"],
    )


class WanTaskEncoder(DiffusionTaskEncoderWithSequencePacking):
    """
    Task encoder for Wan dataset.
    Attributes:
        cookers (list): A list of Cooker objects used for processing.
        patch_spatial (int): The spatial patch size. Defaults to 2.
        patch_temporal (int): The temporal patch size. Defaults to 1.
        seq_length (int): The sequence length. Defaults to 1024.
    """

    cookers = [
        Cooker(cook),
    ]

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

    ## actual encode_sample() for production
    def encode_sample(self, sample: dict) -> dict:
        video_latent = sample["pth"]
        context_embeddings = sample["pickle"]
        # sanity quality check
        if torch.isnan(video_latent).any() or torch.isinf(video_latent).any():
            raise SkipSample()
        if torch.max(torch.abs(video_latent)) > 1e3:
            raise SkipSample()

        # calculate grid size
        grid_size = grid_sizes_calculation(
            input_shape=video_latent.shape[1:],
            patch_size=(self.patch_temporal, self.patch_spatial, self.patch_spatial),
        )
        self.patch_size = (self.patch_temporal, self.patch_spatial, self.patch_spatial)

        ### Note: shape of sample's values
        video_tokens = patchify_single_sample(torch.tensor(video_latent), self.patch_size)

        seq_len_q = video_tokens.shape[0]
        seq_len_kv = context_embeddings.shape[0]

        if self.packing_buffer_size is None:
            pos_ids = F.pad(pos_ids, (0, 0, 0, self.seq_length - seq_len_q))
            loss_mask = torch.zeros(self.seq_length, dtype=torch.bfloat16)
            loss_mask[:seq_len_q] = 1
            video_latent = F.pad(video_latent, (0, 0, 0, self.seq_length - seq_len_q))
        else:
            loss_mask = torch.ones(seq_len_q, dtype=torch.bfloat16)

        return DiffusionSample(
            __key__=sample["__key__"],
            __restore_key__=sample["__restore_key__"],
            __subflavor__=None,
            __subflavors__=sample["__subflavors__"],
            video=video_tokens,
            context_embeddings=context_embeddings,
            context_mask=None,
            loss_mask=loss_mask,
            seq_len_q=torch.tensor([seq_len_q], dtype=torch.int32),
            seq_len_kv=torch.tensor([seq_len_kv], dtype=torch.int32),
            pos_ids=None,
            latent_shape=torch.tensor([grid_size], dtype=torch.int32),
        )

    @stateless
    def batch(self, samples: List[DiffusionSample]) -> dict:
        """Return dictionary with data for batch."""
        if self.packing_buffer_size is None:
            # no packing
            return super().batch(samples).to_dict()

        # packing
        sample = samples[0]
        return dict(
            video=sample.video.unsqueeze_(1),
            context_embeddings=sample.context_embeddings.unsqueeze_(1),
            context_mask=sample.context_mask.unsqueeze_(1) if sample.context_mask is not None else None,
            loss_mask=sample.loss_mask.unsqueeze_(1) if sample.loss_mask is not None else None,
            max_seq_len=torch.max(sample.seq_len_q),
            seq_len_q=sample.seq_len_q,
            seq_len_kv=sample.seq_len_kv,
            pos_ids=sample.pos_ids.unsqueeze_(1) if sample.pos_ids is not None else None,
            latent_shape=sample.latent_shape,
        )
