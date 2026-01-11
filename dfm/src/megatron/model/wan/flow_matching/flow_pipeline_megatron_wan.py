# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
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

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Tuple

from megatron.core import parallel_state

from dfm.src.automodel.flow_matching.flow_matching_pipeline import FlowMatchingPipeline
from dfm.src.automodel.flow_matching.adapters.base import ModelAdapter, FlowMatchingContext
from dfm.src.megatron.model.wan.flow_matching.time_shift_utils import compute_density_for_timestep_sampling
from dfm.src.megatron.model.wan.utils import patchify, thd_split_inputs_cp
from dfm.src.automodel.flow_matching.adapters.base import FlowMatchingContext


class WanAdapter(ModelAdapter):
    """
    Model adapter for Wan model (Megatron version).
    
    Handles mapping of standard FlowMatchingContext to Wan specific inputs.
    """
    def prepare_inputs(self, context: FlowMatchingContext) -> Dict[str, Any]:

        grid_sizes = context.batch["grid_sizes"]
        noisy_latents = context.noisy_latents
        video_latents = context.video_latents
        loss_mask = context.batch["loss_mask"]
        context_embeddings = context.batch["context_embeddings"]
        timesteps = context.timesteps
        packed_seq_params = context.batch["packed_seq_params"]

        # # DEBUGGING
        # # tranpose back to have shape [seq_len, batch_size, ...]
        # # (before we reshaped to [batch_size, seq_len, ...] to be compatible with flow matching pipeline)
        # noisy_latents = noisy_latents.transpose(0, 1)
        # video_latents = video_latents.transpose(0, 1)
        # context_embeddings = context_embeddings.transpose(0, 1)
        # loss_mask = loss_mask.transpose(0, 1)


        # ========================================================================
        # Cast model inputs to bf16
        # ========================================================================

        noisy_latents = noisy_latents.to(torch.bfloat16)
        context_embeddings = context_embeddings.to(torch.bfloat16)

        # NOTE: investigate the affect of bf16 timesteps on embedding precision
        # CRITICAL: Keep timesteps in fp32 for embedding precision
        # timesteps = timesteps.float()  # NOT bf16!
        timesteps = timesteps.to(torch.bfloat16)

        # ========================================================================
        # Split accross context parallelism
        # ========================================================================

        if parallel_state.get_context_parallel_world_size() > 1:
            noisy_latents = thd_split_inputs_cp(
                noisy_latents,
                packed_seq_params["self_attention"].cu_seqlens_q_padded,
                parallel_state.get_context_parallel_group(),
            )
            # TODO (pmannan): Disable CP for CrossAttention as KV context is small.
            # We don't need to split context embeddings across context parallelism
            # if we disable context parallelism for cross-attention
            context_embeddings = thd_split_inputs_cp(
                context_embeddings,
                packed_seq_params["cross_attention"].cu_seqlens_kv_padded,
                parallel_state.get_context_parallel_group(),
            )
        else:
            noisy_latents = noisy_latents
            context_embeddings = context_embeddings

        return {
            "noisy_latents": noisy_latents,
            "grid_sizes": grid_sizes,
            "timesteps": timesteps,
            "context_embeddings": context_embeddings,
            "packed_seq_params": packed_seq_params,
        }

    def forward(self, model: nn.Module, inputs: Dict[str, Any]) -> torch.Tensor:
        """
        Execute forward pass for Wan model.

        Args:
            model: Wan model
            inputs: Dictionary from prepare_inputs()

        Returns:
            Model prediction tensor
        """

        model_pred = model(
            x=inputs["noisy_latents"],
            grid_sizes=inputs["grid_sizes"],
            t=inputs["timesteps"],
            context=inputs["context_embeddings"],
            packed_seq_params=inputs["packed_seq_params"],
        )
        return self.post_process_prediction(model_pred)


class WanFlowMatchingPipeline(FlowMatchingPipeline):
    """
    Wan-specific Flow Matching pipeline handling Context Parallelism and Custom Noise.
    
    This pipeline extends the standard FlowMatchingPipeline to support:
    1. Wan-specific noise generation (patching + padding)
    2. Context Parallelism (CP) splitting of inputs
    3. Masked loss computation
    """

    # def __init__(
    #     self,
    #     model_adapter: ModelAdapter,
    #     num_train_timesteps: int = 1000,
    #     timestep_sampling: str = "logit_normal",
    #     flow_shift: float = 3.0,
    #     i2v_prob: float = 0.3,
    #     # Logit-normal distribution parameters
    #     logit_mean: float = 0.0,
    #     logit_std: float = 1.0,
    #     # Mix sampling parameters
    #     mix_uniform_ratio: float = 0.1,
    #     # Sigma clamping for finetuning (pretrain uses [0.0, 1.0])
    #     sigma_min: float = 0.0,
    #     sigma_max: float = 1.0,
    #     # Loss weighting
    #     use_loss_weighting: bool = True,
    #     # Logging
    #     log_interval: int = 100,
    #     summary_log_interval: int = 10,
    #     device: Optional[torch.device] = None,
    # ):
    
    #     return None

    def determine_task_type(self, data_type: str) -> str:
        """Determine task type based on data type and randomization."""
        return "t2v"

    def compute_loss(
        self,
        model_pred: torch.Tensor,
        target: torch.Tensor,
        sigma: torch.Tensor,
        batch: Dict[str, Any],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        loss_mask = batch["loss_mask"]
        packed_seq_params = batch["packed_seq_params"]

        # ========================================================================
        # Split accross context parallelism
        # ========================================================================

        if parallel_state.get_context_parallel_world_size() > 1:
            target = thd_split_inputs_cp(
                target,
                packed_seq_params["self_attention"].cu_seqlens_q_padded,
                parallel_state.get_context_parallel_group(),
            )
            split_loss_mask = thd_split_inputs_cp(
                loss_mask,
                packed_seq_params["self_attention"].cu_seqlens_q_padded,
                parallel_state.get_context_parallel_group(),
            )
        else:
            target = target
            split_loss_mask = loss_mask
        
        batch["loss_mask"] = split_loss_mask
        weighted_loss, unweighted_loss, loss_weight, loss_mask = super().compute_loss(model_pred, target, sigma, batch)
        return weighted_loss, unweighted_loss, loss_weight, loss_mask

    # def step(
    #     self,
    #     model: nn.Module,
    #     batch: Dict[str, Any],
    #     device: torch.device,
    #     dtype: torch.dtype,
    #     global_step: int = 0,
    # ) -> Tuple[torch.Tensor, Dict[str, Any]]:
    #     return None