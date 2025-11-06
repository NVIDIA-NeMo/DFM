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

from dfm.src.megatron.model.wan.utils import grid_sizes_calculation, patchify
from megatron.core import parallel_state
from megatron.energon import DefaultTaskEncoder, SkipSample
from megatron.energon.task_encoder.cooking import basic_sample_keys, Cooker
import torch
import torch.nn.functional as F


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


class WanTaskEncoder(DefaultTaskEncoder):
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
        max_frames: int = None,
        patch_spatial: int = 2,
        patch_temporal: int = 1,
        seq_length: int = 1024,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.patch_spatial = patch_spatial
        self.patch_temporal = patch_temporal
        self.seq_length = seq_length

    ## actual encode_sample() for production
    def encode_sample(self, sample: dict) -> dict:

        video_latent = sample["pth"]
        context_embeddings = sample["pickle"]
        video_metadata = sample["json"]

        # sanity quality check
        if torch.isnan(video_latent).any() or torch.isinf(video_latent).any():
            raise SkipSample()
        if torch.max(torch.abs(video_latent)) > 1e3:
            raise SkipSample()

        # calculate grid size
        grid_size = grid_sizes_calculation(
            input_shape = video_latent.shape[1:], 
            patch_size = (self.patch_temporal, self.patch_spatial, self.patch_spatial),
        )

        ### Note: shape of sample's values
        # video_latent: [latents_channels, F_latents, W_latents, H_latents]
        # grid_size: [F_patches, W_patches, H_patches]
        # context_embeddings: [context_seq_len, text_embedding_dim]

        return dict(
            video_latent=video_latent,
            grid_size=grid_size,
            context_embeddings=context_embeddings,
            video_metadata=video_metadata,
        )

    ## mock encode_sample() for debugging
    # def encode_sample(self, sample: dict) -> dict:

    #     # mock encode sample
    #     F_latents = 24
    #     H_latents = 104
    #     W_latents = 60
    #     video_latent = torch.tensor(torch.randn(16, F_latents, H_latents, W_latents), dtype=torch.float32)
    #     grid_size = torch.tensor([video_latent.shape[1] // self.patch_temporal, video_latent.shape[2] // self.patch_spatial, video_latent.shape[3] // self.patch_spatial], dtype=torch.int32)
    #     context_embeddings = torch.tensor(torch.randn(512, 4096), dtype=torch.float32)
    #     video_metadata = {}

    #     return dict(
    #         video_latent=video_latent,
    #         grid_size=grid_size,
    #         context_embeddings=context_embeddings,
    #         video_metadata=video_metadata,
    #     )


    def batch(self, samples: list[dict]) -> dict:

        # process video latents
        # do padding here for video latents
        self.patch_size = (self.patch_temporal, self.patch_spatial, self.patch_spatial)

        # running patchify
        video_latents = patchify([sample["video_latent"] for sample in samples], self.patch_size)

        # build per-sample loss masks (1 for valid tokens pre-padding)
        loss_masks = [torch.ones(v.shape[0]) for v in video_latents]
        # calculate all sequence lengths of video latents for self-attention (for videos, we do this before padding to get original seq len)
        seq_len_q = [v.shape[0] for v in video_latents]
        seq_len_q = torch.tensor(seq_len_q, dtype=torch.int32)


        # padding and stack video latents
        max_video_seq_len = max([video_latent.shape[0] for video_latent in video_latents])
        # CAVEAT:
        #   when using pipeline parallelism, we need to set batch sequence length to DataModule's seq_length because
        #   because pipeline parallelism requires pre-specified sequence length to create buffer
        if parallel_state.get_pipeline_model_parallel_world_size() > 1:
            if max_video_seq_len > self.seq_length:
                raise ValueError(f"max_video_seq_len {max_video_seq_len} is greater than DataModule's seq_length {self.seq_length}")
            else:
                # set max_video_seq_len to DataModule's seq_length
                max_video_seq_len = self.seq_length
        # CAVEAT:
        #   when using context parallelism, we need to pad batch sequence length to be divisible by [cp_rank*2]
        #   (because TransformerEngine's context parallelism requires "AssertionError: Sequence length per GPU needs to be divisible by 2!")
        if parallel_state.get_context_parallel_world_size() > 1:
            batch_size = len(video_latents)
            assert batch_size == 1, "Error: Batch size must be 1 when using context parallelism"
            sharding_factor = parallel_state.get_context_parallel_world_size() * 2
            max_video_seq_len = ((max_video_seq_len + sharding_factor - 1) // sharding_factor) * sharding_factor
        video_latents = [F.pad(video_latent, (0, 0, 0, max_video_seq_len - video_latent.shape[0])) for video_latent in video_latents]
        video_latents = torch.stack(video_latents, dim=1)
        # pad and stack loss masks to shape [S_max, B]
        loss_masks = [F.pad(m, (0, max_video_seq_len - m.shape[0])) for m in loss_masks]
        loss_masks = torch.stack(loss_masks, dim=1)

        # process grid sizes
        grid_sizes = [torch.tensor(sample["grid_size"], dtype=torch.int32) for sample in samples]
        grid_sizes = torch.stack(grid_sizes, dim=0)

        # process text embeddings
        # pad here for text embeddings
        context_max_len = 512
        context_embeddings = [sample["context_embeddings"] for sample in samples]
        context_embeddings = [F.pad(context_embedding, (0, 0, 0, context_max_len - context_embedding.shape[0])) for context_embedding in context_embeddings]
        # calculate all sequence lengths of context embeddings for cross-attention (for videos, we do this after padding to get padded seq len)
        seq_len_kv = [c.shape[0] for c in context_embeddings]
        seq_len_kv = torch.tensor(seq_len_kv, dtype=torch.int32)
        # stack context embeddings
        context_embeddings = torch.stack(context_embeddings, dim=1)

        # process video metadata
        video_metadata = [sample["video_metadata"] for sample in samples]

        return dict(
            video_latents = video_latents,
            max_video_seq_len = max_video_seq_len,
            grid_sizes = grid_sizes,
            context_embeddings = context_embeddings,
            loss_mask = loss_masks,
            seq_len_q = seq_len_q,
            seq_len_kv = seq_len_kv,
            video_metadata = video_metadata,
        )