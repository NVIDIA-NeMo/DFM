# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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


import logging
from functools import partial
from typing import Iterable

import torch
from megatron.bridge.training.losses import masked_next_token_loss
from megatron.bridge.training.state import GlobalState
from megatron.core import parallel_state
from megatron.core.models.common.vision_module.vision_module import VisionModule
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.utils import get_model_config

from dfm.src.megatron.model.wan.flow_matching.flow_pipeline import FlowPipeline


logger = logging.getLogger(__name__)


def wan_data_step(qkv_format, dataloader_iter):
    batch = next(iter(dataloader_iter.iterable))

    batch = {k: v.to(device="cuda", non_blocking=True) if torch.is_tensor(v) else v for k, v in batch.items()}

    # Construct packed sequence parameters
    if ("seq_len_q" in batch) and ("seq_len_kv" in batch):
        cu_seqlens = batch["seq_len_q"].cumsum(dim=0).to(torch.int32)
        zero = torch.zeros(1, dtype=torch.int32, device="cuda")
        cu_seqlens = torch.cat((zero, cu_seqlens))

        cu_seqlens_kv = batch["seq_len_kv"].cumsum(dim=0).to(torch.int32)
        cu_seqlens_kv = torch.cat((zero, cu_seqlens_kv))

        batch["packed_seq_params"] = {
            "self_attention": PackedSeqParams(
                cu_seqlens_q=cu_seqlens,
                cu_seqlens_kv=cu_seqlens,
                qkv_format=qkv_format,
            ),
            "cross_attention": PackedSeqParams(
                cu_seqlens_q=cu_seqlens,
                cu_seqlens_kv=cu_seqlens_kv,
                qkv_format=qkv_format,
            ),
        }

    return batch


class WanForwardStep:
    def __init__(self):
        self.diffusion_pipeline = FlowPipeline()

    def __call__(
        self, state: GlobalState, data_iterator: Iterable, model: VisionModule
    ) -> tuple[torch.Tensor, partial]:
        """
        Forward training step.
        """
        timers = state.timers
        straggler_timer = state.straggler_timer

        config = get_model_config(model)

        timers("batch-generator", log_level=2).start()

        qkv_format = getattr(config, "qkv_format", "sbhd")
        with straggler_timer(bdata=True):
            batch = wan_data_step(qkv_format, data_iterator)
        timers("batch-generator").stop()

        check_for_nan_in_loss = state.cfg.rerun_state_machine.check_for_nan_in_loss
        check_for_spiky_loss = state.cfg.rerun_state_machine.check_for_spiky_loss

        # run diffusion training step
        with straggler_timer:
            if parallel_state.is_pipeline_last_stage():
                output_batch, loss, split_loss_mask = self.diffusion_pipeline.training_step(model, batch)
                output_tensor = torch.mean(loss, dim=-1)
                batch["loss_mask"] = split_loss_mask
            else:
                output_tensor = self.diffusion_pipeline.training_step(model, batch)

        # DEBUGGING
        # TODO: do we need to gather output with sequence or context parallelism here
        #       especially when we have pipeline parallelism

        loss = output_tensor
        if "loss_mask" not in batch or batch["loss_mask"] is None:
            loss_mask = torch.ones_like(loss)
        loss_mask = batch["loss_mask"]

        loss_function = self._create_loss_function(loss_mask, check_for_nan_in_loss, check_for_spiky_loss)

        return output_tensor, loss_function

    def _create_loss_function(
        self, loss_mask: torch.Tensor, check_for_nan_in_loss: bool, check_for_spiky_loss: bool
    ) -> partial:
        """Create a partial loss function with the specified configuration.

        Args:
            loss_mask: Used to mask out some portions of the loss
            check_for_nan_in_loss: Whether to check for NaN values in the loss
            check_for_spiky_loss: Whether to check for spiky loss values

        Returns:
            A partial function that can be called with output_tensor to compute the loss
        """
        return partial(
            masked_next_token_loss,
            loss_mask,
            check_for_nan_in_loss=check_for_nan_in_loss,
            check_for_spiky_loss=check_for_spiky_loss,
        )
