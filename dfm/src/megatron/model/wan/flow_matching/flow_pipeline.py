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

from typing import Any, Callable, Dict, Optional, Tuple, List

import numpy as np
import torch
from megatron.core import parallel_state
from torch import Tensor
from diffusers import WanPipeline
from megatron.bridge.models.wan.flow_matching.time_shift_utils import compute_density_for_timestep_sampling
from megatron.bridge.models.wan.utils.utils import patchify, split_inputs_cp

class FlowPipeline:

    def __init__(
        self,
        model_id="Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
        seed=1234,
    ):
        """
        Initializes the FlowPipeline with the given parameters.
        """
        self.pipe = WanPipeline.from_pretrained(model_id, vae=None, torch_dtype=torch.float32, text_encoder=None)


    def training_step(
        self, 
        model, 
        data_batch: dict[str, torch.Tensor],
        # Flow matching parameters
        use_sigma_noise: bool = True,
        timestep_sampling: str = "uniform",
        logit_mean: float = 0.0,
        logit_std: float = 1.0,
        flow_shift: float = 3.0,
        mix_uniform_ratio: float = 0.1,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        """
        Performs a single training step using flow matching algorithm.

        This method is responsible for executing one iteration of the model's training. It involves:
        1. Generate noise and add it to the input data.
        2. Pass the noisy data through the network to generate predictions.
        3. Compute the loss based on the difference between the predictions and target.
        """

        video_latents = data_batch['video_latents']
        max_video_seq_len = data_batch['max_video_seq_len']
        context_embeddings = data_batch['context_embeddings']
        loss_mask = data_batch['loss_mask']
        grid_sizes = data_batch['grid_sizes']
        packed_seq_params = data_batch['packed_seq_params']
        video_metadata = data_batch['video_metadata']

        self.model = model

        batch_size = video_latents.shape[1]
        device = video_latents.device

        # # # DEBUGGING precision
        # # import torch.cuda.amp as amp
        # # with amp.autocast(dtype=torch.bfloat16):
        # #     # Pass through model
        # #     ...

        # ========================================================================
        # Flow Matching Timestep Sampling
        # ========================================================================
        
        num_train_timesteps = self.pipe.scheduler.config.num_train_timesteps
        
        if use_sigma_noise:
            use_uniform = torch.rand(1).item() < mix_uniform_ratio
            
            if use_uniform or timestep_sampling == "uniform":
                # Pure uniform: u ~ U(0, 1)
                u = torch.rand(size=(batch_size,), device=device)
                sampling_method = "uniform"
            else:
                # Density-based sampling
                u = compute_density_for_timestep_sampling(
                    weighting_scheme=timestep_sampling,
                    batch_size=batch_size,
                    logit_mean=logit_mean,
                    logit_std=logit_std,
                ).to(device)
                sampling_method = timestep_sampling
            
            # Apply flow shift: σ = shift/(shift + (1/u - 1))
            u_clamped = torch.clamp(u, min=1e-5)  # Avoid division by zero
            sigma = flow_shift / (flow_shift + (1.0 / u_clamped - 1.0))
            sigma = torch.clamp(sigma, 0.0, 1.0)
            
        else:
            # Simple uniform without shift
            u = torch.rand(size=(batch_size,), device=device)
            sigma = u
            sampling_method = "uniform_no_shift"

        # ========================================================================
        # Manual Flow Matching Noise Addition
        # ========================================================================
        
        # Generate noise
        noise = torch.randn_like(torch.ones([1, 16, grid_sizes[0][0], grid_sizes[0][1]*2, grid_sizes[0][2]*2], device=video_latents.device), dtype=torch.float32)
        noise = patchify(noise, (1, 2, 2))[0].unsqueeze(1)
        # DEBUGGING
        # because video_latents might be padded, we need to make sure noise also be padded to have the same shape
        seq_noise = noise.shape[0]
        seq_video = video_latents.shape[0]
        if seq_noise < seq_video:
            pad_len = seq_video - seq_noise
            pad = torch.zeros((pad_len, noise.shape[1], noise.shape[2]), device=noise.device, dtype=noise.dtype)
            noise = torch.cat([noise, pad], dim=0)

        # CRITICAL: Manual flow matching (NOT scheduler.add_noise!)
        # x_t = (1 - σ) * x_0 + σ * ε
        sigma_reshaped = sigma.view(1, batch_size, 1)
        noisy_latents = (
            (1.0 - sigma_reshaped) * video_latents.float() 
            + sigma_reshaped * noise
        )
        
        # Timesteps for model [0, 1000]
        timesteps = sigma * num_train_timesteps

        # ========================================================================
        # Cast model inputs to bf16
        # ========================================================================

        video_latents = video_latents.to(torch.bfloat16)
        noisy_latents = noisy_latents.to(torch.bfloat16)
        context_embeddings = context_embeddings.to(torch.bfloat16)
        timesteps = timesteps.to(torch.bfloat16)

        # ========================================================================
        # Split accross context parallelism
        # ========================================================================
        
        if parallel_state.get_context_parallel_world_size() > 1:
            video_latents = split_inputs_cp(video_latents, 0)
            noisy_latents = split_inputs_cp(noisy_latents, 0)
            noise = split_inputs_cp(noise, 0)
            context_embeddings = split_inputs_cp(context_embeddings, 0)
            split_loss_mask = split_inputs_cp(loss_mask, 0)
        else:
            video_latents = video_latents
            noisy_latents = noisy_latents
            noise = noise
            context_embeddings = context_embeddings
            split_loss_mask = loss_mask


        # ========================================================================
        # Forward Pass
        # ========================================================================
        
        if parallel_state.is_pipeline_last_stage():
        
            model_pred = self.model(
                x = noisy_latents,
                grid_sizes = grid_sizes,
                t = timesteps,
                context = context_embeddings,
                max_seq_len = max_video_seq_len,
                packed_seq_params=packed_seq_params,
            )

            # ========================================================================
            # Target: Flow Matching Velocity
            # ========================================================================
            
            # Flow matching target: v = ε - x_0
            target = noise - video_latents.float()
            
            # ========================================================================
            # Loss with Flow Weighting
            # ========================================================================
            
            loss = torch.nn.functional.mse_loss(
                model_pred.float(),
                target.float(),
                reduction="none"
            )

            # Flow weight: w = 1 + shift * σ
            loss_weight = 1.0 + flow_shift * sigma # shape [batch_size]
            loss_weight = loss_weight.view(1, batch_size, 1).to(device) # shape [1, batch_size, 1]
            unweighted_loss = loss
            weighted_loss = (loss * loss_weight) # shape [seq_length / cp_size, batch_size, -1]

            # Safety check
            mean_weighted_loss = weighted_loss.mean()
            if torch.isnan(mean_weighted_loss) or mean_weighted_loss > 100:
                print(f"[ERROR] Loss explosion! Loss={mean_weighted_loss.item():.3f}")
                print(f"[DEBUG] Stopping training - check hyperparameters")
                raise ValueError(f"Loss exploded: {mean_weighted_loss.item()}")

            return model_pred, weighted_loss, split_loss_mask

        else:
            hidden_states = self.model(
                x = noisy_latents,
                grid_sizes = grid_sizes,
                t = timesteps,
                context = context_embeddings,
                max_seq_len = max_video_seq_len,
                packed_seq_params=packed_seq_params,
            )

            return hidden_states