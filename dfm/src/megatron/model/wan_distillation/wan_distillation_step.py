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


# Add fastgen to the path so that internal fastgen imports work correctly
# _fastgen_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../fastgen"))
# if _fastgen_path not in sys.path:
#     sys.path.insert(0, _fastgen_path)

import logging
from functools import partial
from typing import Iterable

import torch
from diffusers import WanPipeline
from fastgen.methods.distribution_matching.dmd2_pipeline import DMD2Pipeline
from fastgen.networks.noise_schedule import get_noise_schedule
from megatron.bridge.training.losses import masked_next_token_loss
from megatron.bridge.training.state import GlobalState
from megatron.core import parallel_state
from megatron.core.models.common.vision_module.vision_module import VisionModule
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.utils import get_model_config, unwrap_model


logger = logging.getLogger(__name__)


def wan_data_step(qkv_format, dataloader_iter):
    batch = next(dataloader_iter)
    batch = {k: v.to(device="cuda", non_blocking=True) if torch.is_tensor(v) else v for k, v in batch.items()}
    # Construct packed sequence parameters
    if ("seq_len_q" in batch) and ("seq_len_kv" in batch):
        zero = torch.zeros(1, dtype=torch.int32, device="cuda")

        cu_seqlens = batch["seq_len_q"].cumsum(dim=0).to(torch.int32)
        cu_seqlens = torch.cat((zero, cu_seqlens))

        cu_seqlens_padded = batch["seq_len_q_padded"].cumsum(dim=0).to(torch.int32)
        cu_seqlens_padded = torch.cat((zero, cu_seqlens_padded))

        cu_seqlens_kv = batch["seq_len_kv"].cumsum(dim=0).to(torch.int32)
        cu_seqlens_kv = torch.cat((zero, cu_seqlens_kv))

        cu_seqlens_kv_padded = batch["seq_len_kv_padded"].cumsum(dim=0).to(torch.int32)
        cu_seqlens_kv_padded = torch.cat((zero, cu_seqlens_kv_padded))

        batch["packed_seq_params"] = {
            "self_attention": PackedSeqParams(
                cu_seqlens_q=cu_seqlens,
                cu_seqlens_q_padded=cu_seqlens_padded,
                cu_seqlens_kv=cu_seqlens,
                cu_seqlens_kv_padded=cu_seqlens_padded,
                qkv_format=qkv_format,
            ),
            "cross_attention": PackedSeqParams(
                cu_seqlens_q=cu_seqlens,
                cu_seqlens_q_padded=cu_seqlens_padded,
                cu_seqlens_kv=cu_seqlens_kv,
                cu_seqlens_kv_padded=cu_seqlens_kv_padded,
                qkv_format=qkv_format,
            ),
        }

    return batch


class WanDistillationStep:
    def __init__(
        self,
        config: dict = None,
        model_id="Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
    ):
        self.distillation_pipeline = DMD2Pipeline(config=config)
        self.wan_pipeline = WanPipeline.from_pretrained(
            model_id, vae=None, torch_dtype=torch.float32, text_encoder=None, cache_dir="/opt/artifacts"
        )
        self.valid = False
        self.train = False
        self.training_trigered = False

    def on_train_start(self, student, teacher, fake_score, state: GlobalState):
        student.noise_scheduler = get_noise_schedule("rf")
        self.distillation_pipeline.set_models(
            student=student,
            teacher=teacher,
            fake_score=fake_score,
        )

    def __call__(
        self, state: GlobalState, data_iterator: Iterable, model: VisionModule
    ) -> tuple[torch.Tensor, partial]:
        """
        Forward training step.
        """
        model_config = get_model_config(model)

        unwrapped_model = unwrap_model(model)
        if isinstance(unwrapped_model, list):
            unwrapped_model = unwrapped_model[0]
        student_proxy, teacher_proxy, fake_score_proxy = unwrapped_model.get_submodel_proxies(model)

        if model.training and not self.train:
            self.on_train_start(student_proxy, teacher_proxy, fake_score_proxy, state)
            self.train = True
        elif model.training and self.valid:
            self.train = True
            self.valid = False
        elif (not model.training) and self.train:
            self.train = False
            self.valid = True
        return self.forward_step(state, data_iterator, student_proxy, teacher_proxy, fake_score_proxy, model_config)

    def forward_step(
        self, state: GlobalState, data_iterator: Iterable, student_proxy, teacher_proxy, fake_score_proxy, model_config
    ) -> tuple[torch.Tensor, partial]:
        """
        Forward training step.
        """
        timers = state.timers
        straggler_timer = state.straggler_timer
        timers("batch-generator", log_level=2).start()

        qkv_format = getattr(model_config, "qkv_format", "sbhd")
        with straggler_timer(bdata=True):
            batch = wan_data_step(qkv_format, data_iterator)
        timers("batch-generator").stop()

        check_for_nan_in_loss = state.cfg.rerun_state_machine.check_for_nan_in_loss
        check_for_spiky_loss = state.cfg.rerun_state_machine.check_for_spiky_loss
        with straggler_timer:
            if parallel_state.is_pipeline_last_stage():
                loss_map, outputs = self.distillation_pipeline.training_step(
                    data=batch,
                    iteration=state.train_state.step,
                )
                output_tensor = torch.mean(loss_map["total_loss"], dim=-1)
                # batch["loss_mask"] = data_batch["loss_mask"]
            else:
                output_tensor = self.distillation_pipeline.single_train_step(
                    data=batch,
                    iteration=state.train_state.step,
                )

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
