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

from dataclasses import dataclass
import torch

from megatron.bridge.data.utils import DatasetBuildContext, DatasetProvider

from dfm.src.megatron.data.dit.diffusion_energon_datamodule import DiffusionDataModule
from dfm.src.megatron.data.wan.wan_taskencoder import WanTaskEncoder


class WanMockTaskEncoder(WanTaskEncoder):
    """
    Mock task encoder for Wan dataset.
    Attributes:
        cookers (list): A list of Cooker objects used for processing.
        patch_spatial (int): The spatial patch size. Defaults to 2.
        patch_temporal (int): The temporal patch size. Defaults to 1.
        seq_length (int): The sequence length. Defaults to 1024.
    """

    F_latents: int
    H_latents: int
    W_latents: int
    context_seq_len: int
    context_embeddings_dim: int

    def __init__(
        self,
        *args,
        F_latents: int,
        H_latents: int,
        W_latents: int,
        context_seq_len: int,
        context_embeddings_dim: int,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.F_latents = F_latents
        self.H_latents = H_latents
        self.W_latents = W_latents
        self.context_seq_len = context_seq_len
        self.context_embeddings_dim = context_embeddings_dim

    # mock encode_sample() for debugging
    def encode_sample(self, sample: dict) -> dict:

        # mock encode sample
        video_latent = torch.tensor(torch.randn(16, self.F_latents, self.H_latents, self.W_latents), dtype=torch.float32)
        grid_size = torch.tensor([video_latent.shape[1] // self.patch_temporal, video_latent.shape[2] // self.patch_spatial, video_latent.shape[3] // self.patch_spatial], dtype=torch.int32)
        context_embeddings = torch.tensor(torch.randn(self.context_seq_len, self.context_embeddings_dim), dtype=torch.float32)
        video_metadata = {}

        # DEBUGGING
        output = ""
        output += "----------------------------------------\n"
        output += f"video_latent.shape: {video_latent.shape}\n"
        output += f"grid_size.shape: {grid_size.shape}\n"
        output += f"context_embeddings.shape: {context_embeddings.shape}\n"
        output += f"video_metadata: {video_metadata}\n"
        output += "----------------------------------------\n"
        print(output)

        return dict(
            video_latent=video_latent,
            grid_size=grid_size,
            context_embeddings=context_embeddings,
            video_metadata=video_metadata,
        )


@dataclass(kw_only=True)
class WanMockDataModuleConfig(DatasetProvider):
    path: str
    seq_length: int
    micro_batch_size: int
    global_batch_size: int
    num_workers: int
    dataloader_type: str = "external"
    F_latents: int = 3
    H_latents: int = 104
    W_latents: int = 60
    context_seq_len: int = 512
    context_embeddings_dim: int = 4096

    def __post_init__(self):
        self.dataset = DiffusionDataModule(
            path=self.path,
            seq_length=self.seq_length,
            task_encoder=WanMockTaskEncoder(
                seq_length=self.seq_length,
                F_latents=self.F_latents,
                H_latents=self.H_latents,
                W_latents=self.W_latents,
                context_seq_len=self.context_seq_len,
                context_embeddings_dim=self.context_embeddings_dim,
            ),
            micro_batch_size=self.micro_batch_size,
            global_batch_size=self.global_batch_size,
            num_workers=self.num_workers,
        )
        self.sequence_length = self.dataset.seq_length

    def build_datasets(self, context: DatasetBuildContext):
        return self.dataset.train_dataloader(), self.dataset.train_dataloader(), self.dataset.train_dataloader()
