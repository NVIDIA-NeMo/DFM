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
import logging
from typing import Any, Dict, Literal

from torch import int_repr

from dfm.src.megatron.data.Dit.data.diffusion_energon_datamodule import DiffusionDataModule
from dfm.src.megatron.data.wan.wan_taskencoder import WanTaskEncoder
from megatron.bridge.data.utils import DatasetBuildContext, DatasetProvider

@dataclass(kw_only=True)
class WanDataModuleConfig(DatasetProvider):
    path: str
    seq_length: int
    micro_batch_size: int
    global_batch_size: int
    num_workers: int_repr
    dataloader_type: str = "external"

    def __post_init__(self):
        self.dataset = DiffusionDataModule(
            path=self.path,
            seq_length=self.seq_length,
            task_encoder=WanTaskEncoder(seq_length=self.seq_length),
            micro_batch_size=self.micro_batch_size,
            global_batch_size=self.global_batch_size,
            num_workers=self.num_workers)
        self.sequence_length = self.dataset.seq_length
    
    def build_datasets(self, context: DatasetBuildContext):
        return self.dataset.train_dataloader(), self.dataset.train_dataloader(), self.dataset.train_dataloader()