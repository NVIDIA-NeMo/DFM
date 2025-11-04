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
from megatron.core import parallel_state
from megatron.core.models.gpt import GPTModel
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.utils import get_batch_on_this_cp_rank, get_model_config
from megatron.bridge.training.losses import masked_next_token_loss
from megatron.bridge.training.state import GlobalState

from dfm.src.megatron.model.dit.edm.edm_pipeline import EDMPipeline
from nemo.collections.common.video_tokenizers.cosmos_tokenizer import CausalVideoTokenizer
from dfm.src.megatron.model.dit.dit_inference import run_diffusion_inference
import os


logger = logging.getLogger(__name__)


def dit_data_step(qkv_format, dataloader_iter):
    # import pdb;pdb.set_trace()
    batch = next(iter(dataloader_iter.iterable))
    batch = get_batch_on_this_cp_rank(batch)
    batch = {k: v.to(device="cuda", non_blocking=True) if torch.is_tensor(v) else v for k, v in batch.items()}
    batch["is_preprocessed"] = True  # assume data is preprocessed
    return encode_seq_length(batch, format=qkv_format)

def encode_seq_length(batch, format):
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
                qkv_format=format,
            ),
            "cross_attention": PackedSeqParams(
                cu_seqlens_q=cu_seqlens,
                cu_seqlens_kv=cu_seqlens_kv,
                qkv_format=format,
            ),
        }

    return batch

def get_batch_on_this_cp_rank(data):
    """Split the data for context parallelism."""
    from megatron.core import mpu

    cp_size = mpu.get_context_parallel_world_size()
    cp_rank = mpu.get_context_parallel_rank()

    t = 16
    if cp_size > 1:
        # cp split on seq_length, for video_latent, noise_latent and pos_ids
        assert t % cp_size == 0, "t must divisibly by cp_size"
        num_valid_tokens_in_ub = None
        if "loss_mask" in data and data["loss_mask"] is not None:
            num_valid_tokens_in_ub = data["loss_mask"].sum()

        for key, value in data.items():
            if (value is not None) and (key in ["video", "video_latent", "noise_latent", "pos_ids"]):
                if len(value.shape) > 5:
                    value = value.squeeze(0)
                B, C, T, H, W = value.shape
                if T % cp_size == 0:
                    # FIXME packed sequencing
                    data[key] = value.view(B, C, cp_size, T // cp_size, H, W)[:, :, cp_rank, ...].contiguous()
                else:
                    # FIXME packed sequencing
                    data[key] = value.view(B, C, T, cp_size, H // cp_size, W)[:, :, :, cp_rank, ...].contiguous()
        loss_mask = data["loss_mask"]
        data["loss_mask"] = loss_mask.view(loss_mask.shape[0], cp_size, loss_mask.shape[1] // cp_size)[
            :, cp_rank, ...
        ].contiguous()
        data["num_valid_tokens_in_ub"] = num_valid_tokens_in_ub

    return data

class DITForwardStep:
    def __init__(self):
        self.diffusion_pipeline = EDMPipeline(sigma_data=0.5)
        self.valid = False
        self.train = True
        self.step = 0

    def on_validation_start(self, batch, step):
        vae = CausalVideoTokenizer.from_pretrained("Cosmos-0.1-Tokenizer-CV4x8x8")
        vae.to("cuda")

        class Config:
            height = 512
            width = 768
            num_video_frames = 1
            fps = 1
            video_save_path = 'new_video'
            guidance = 7
            num_steps = 35

        os.makedirs('output_folder', exist_ok=True)
        Config.video_save_path = 'output_folder/' + Config.video_save_path + f'_{step}'
        state_shape = [len(batch['video']), 2048, 64]
        run_diffusion_inference(self.diffusion_pipeline, Config, batch, state_shape, vae)


    def __call__(
        self, state: GlobalState, data_iterator: Iterable, model: GPTModel, return_schedule_plan: bool = False
    ) -> tuple[torch.Tensor, partial]:
        """Forward training step.

        Args:
            state: Global state for the run
            data_iterator: Input data iterator
            model: The GPT Model
            return_schedule_plan (bool): Whether to return the schedule plan instead of the output tensor

        Returns:
            tuple containing the output tensor and the loss function
        """
        batch = self.__data_process__(state, data_iterator, model, return_schedule_plan)
        if model.training and self.valid:
            self.train = True
            self.valid = False
        elif (not model.training) and self.train:
            self.train = False
            self.valid = True
            self.step += 1
            self.on_validation_start(batch, step=self.step)
        return self.forward_step(state, batch, model, return_schedule_plan)

    def __data_process__(self, state: GlobalState, data_iterator: Iterable, model: GPTModel, return_schedule_plan: bool = False
    ) -> tuple[torch.Tensor, partial]:
        timers = state.timers
        straggler_timer = state.straggler_timer

        config = get_model_config(model)
 
        timers("batch-generator", log_level=2).start()
        # use_mtp = (getattr(config, "mtp_num_layers", None) or 0) > 0
        qkv_format =getattr(config, "qkv_format", "sbhd")
        with straggler_timer(bdata=True):
            batch = dit_data_step(
                qkv_format, data_iterator
            )
        return batch

    def forward_step(self, state, batch, model, return_schedule_plan: bool = False):
        timers = state.timers
        timers("batch-generator").stop()

        check_for_nan_in_loss = state.cfg.rerun_state_machine.check_for_nan_in_loss
        check_for_spiky_loss = state.cfg.rerun_state_machine.check_for_spiky_loss
        # import pdb;pdb.set_trace()
        straggler_timer = state.straggler_timer
        with straggler_timer:
            if parallel_state.is_pipeline_last_stage():
                # Self.diffusion_pipeline should not know anything about the model
                # TODO: we need to sepearte the noise ingection process from the pipeline itself
                output_batch, loss = self.diffusion_pipeline.training_step(model, batch, 0)
                output_tensor = torch.mean(loss, dim=-1)
            else:
                output_tensor = self.diffusion_pipeline.training_step(model, batch, 0)

        loss = output_tensor
        if "loss_mask" not in batch or batch["loss_mask"] is None:
            loss_mask = torch.ones_like(loss)
        loss_mask = batch["loss_mask"]
        

        loss_function = self._create_loss_function(loss_mask, check_for_nan_in_loss, check_for_spiky_loss)


        return output_tensor, loss_function


    def _create_loss_function(self, loss_mask: torch.Tensor, check_for_nan_in_loss: bool, check_for_spiky_loss: bool) -> partial:
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
