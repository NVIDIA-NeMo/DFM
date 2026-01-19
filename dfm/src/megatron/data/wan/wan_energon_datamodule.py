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

from megatron.bridge.data.utils import DatasetBuildContext

from dfm.src.megatron.data.common.diffusion_energon_datamodule import DiffusionDataModule, DiffusionDataModuleConfig
from dfm.src.megatron.data.wan.wan_latent_taskencoder import WanLatentTaskEncoder
from dfm.src.megatron.data.wan.wan_taskencoder import WanTaskEncoder


@dataclass(kw_only=True)
class WanDataModuleConfig(DiffusionDataModuleConfig):
    # Only define new fields here; inherited fields come from DiffusionDataModuleConfig
    use_fastgen_dataset: bool = False  # Flag to determine which task encoder to use

    def __post_init__(self):
        # Instantiate the appropriate task encoder based on the flag
        if self.use_fastgen_dataset:
            task_encoder = WanLatentTaskEncoder(
                seq_length=self.task_encoder_seq_length,
                packing_buffer_size=self.packing_buffer_size,
            )
        else:
            task_encoder = WanTaskEncoder(
                seq_length=self.task_encoder_seq_length,
                packing_buffer_size=self.packing_buffer_size,
            )

        self.dataset = DiffusionDataModule(
            path=self.path,
            seq_length=self.seq_length,
            packing_buffer_size=self.packing_buffer_size,
            task_encoder=task_encoder,
            micro_batch_size=self.micro_batch_size,
            global_batch_size=self.global_batch_size,
            num_workers=self.num_workers,
            use_train_split_for_val=self.use_train_split_for_val,
        )
        self.sequence_length = self.dataset.seq_length

    def build_datasets(self, context: DatasetBuildContext):
        return self.dataset.train_dataloader(), self.dataset.train_dataloader(), self.dataset.train_dataloader()
