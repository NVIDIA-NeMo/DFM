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


import torch
from diffusers import WanPipeline
from megatron.core import parallel_state

from dfm.src.megatron.model.wan.flow_matching.time_shift_utils import compute_density_for_timestep_sampling
from dfm.src.megatron.model.wan.utils import patchify, thd_split_inputs_cp


class FlowPipeline:
    def __init__(self, model_id="Wan-AI/Wan2.1-T2V-1.3B-Diffusers", seed=1234, cache_dir=None):
        """
        Initializes the FlowPipeline with the given parameters.
        """
        self.pipe = WanPipeline.from_pretrained(
            model_id, vae=None, torch_dtype=torch.float32, text_encoder=None, cache_dir=cache_dir
        )

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
        sigma_min: float = 0.0,  # Default: no clamping (pretrain)
        sigma_max: float = 1.0,  # Default: no clamping (pretrain)
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        """
        Performs a single training step using flow matching algorithm.

        This method is responsible for executing one iteration of the model's training. It involves:
        1. Generate noise and add it to the input data.
        2. Pass the noisy data through the network to generate predictions.
        3. Compute the loss based on the difference between the predictions and target.
        """

        video_latents = data_batch["video_latents"]
        context_embeddings = data_batch["context_embeddings"]
        loss_mask = data_batch["loss_mask"]
        grid_sizes = data_batch["grid_sizes"]
        packed_seq_params = data_batch["packed_seq_params"]
        video_metadata = data_batch["video_metadata"]

        self.model = model

        batch_size = video_latents.shape[1]
        device = video_latents.device

        # TODO: should we do as in Wan Github repo:
        # with amp.autocast(dtype=torch.bfloat16)
        #     # Pass through model

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

            # Clamp sigma (only if not full range [0,1])
            # Pretrain uses [0, 1], finetune uses [0.02, 0.55]
            if sigma_min > 0.0 or sigma_max < 1.0:
                sigma = torch.clamp(sigma, sigma_min, sigma_max)
            else:
                sigma = torch.clamp(sigma, 0.0, 1.0)

        else:
            # Simple uniform without shift
            u = torch.rand(size=(batch_size,), device=device)
            # Clamp sigma (only if not full range [0,1])
            if sigma_min > 0.0 or sigma_max < 1.0:
                sigma = torch.clamp(u, sigma_min, sigma_max)
            else:
                sigma = u
            sampling_method = "uniform_no_shift"

        # ========================================================================
        # Manual Flow Matching Noise Addition
        # ========================================================================

        # Generate noise for batch
        noise = []
        in_channels = model.config.in_channels
        patch_spatial = model.config.patch_spatial
        patch_temporal = model.config.patch_temporal
        for i, grid_size in enumerate(grid_sizes):
            sample_noise = torch.randn(
                1,
                in_channels,
                grid_size[0] * patch_temporal,
                grid_size[1] * patch_spatial,
                grid_size[2] * patch_spatial,
                dtype=torch.float32,
                device=video_latents.device,
            )
            sample_noise = patchify(sample_noise, (patch_temporal, patch_spatial, patch_spatial))[
                0
            ]  # shape [noise_seq, c * ( pF * pH * pW)]

            # because video_latents might be padded, we need to make sure noise also be padded to have the same shape
            sample_noise_seq_len = sample_noise.shape[0]
            cu_seqlens_q_padded = packed_seq_params["self_attention"].cu_seqlens_q_padded
            seq_len_q_padded = cu_seqlens_q_padded[i + 1] - cu_seqlens_q_padded[i]
            if sample_noise_seq_len < seq_len_q_padded:
                pad_len = seq_len_q_padded - sample_noise_seq_len
                pad = torch.zeros(
                    (pad_len, sample_noise.shape[1]), device=sample_noise.device, dtype=sample_noise.dtype
                )
                sample_noise = torch.cat([sample_noise, pad], dim=0)  # shape [padded_noise_seq, c * ( pF * pH * pW)]

            noise.append(sample_noise)
        noise = torch.cat(noise, dim=0)  # shape [concatenated_noise_seq, c * ( pF * pH * pW)]
        noise = noise.unsqueeze(1)  # shape [concatenated_noise_seq, 1, c * ( pF * pH * pW)]

        # CRITICAL: Manual flow matching (NOT scheduler.add_noise!)
        # x_t = (1 - σ) * x_0 + σ * ε
        # since we use sequence packing, the batch_size is 1)
        noisy_latents = (1.0 - sigma) * video_latents.float() + sigma * noise

        # Timesteps for model [0, 1000]
        timesteps = sigma * num_train_timesteps

        # ========================================================================
        # Cast model inputs to bf16
        # ========================================================================

        video_latents = video_latents.to(torch.bfloat16)
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
            video_latents = thd_split_inputs_cp(
                video_latents,
                packed_seq_params["self_attention"].cu_seqlens_q_padded,
                parallel_state.get_context_parallel_group(),
            )
            noisy_latents = thd_split_inputs_cp(
                noisy_latents,
                packed_seq_params["self_attention"].cu_seqlens_q_padded,
                parallel_state.get_context_parallel_group(),
            )
            noise = thd_split_inputs_cp(
                noise,
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
            split_loss_mask = thd_split_inputs_cp(
                loss_mask,
                packed_seq_params["self_attention"].cu_seqlens_q_padded,
                parallel_state.get_context_parallel_group(),
            )
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
                x=noisy_latents,
                grid_sizes=grid_sizes,
                t=timesteps,
                context=context_embeddings,
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

            loss = torch.nn.functional.mse_loss(model_pred.float(), target.float(), reduction="none")

            # Flow weight: w = 1 + shift * σ
            loss_weight = 1.0 + flow_shift * sigma
            unweighted_loss = loss
            # since we use sequence packing, the batch_size is 1
            weighted_loss = loss * loss_weight  # shape [seq_length / cp_size, 1, -1]

            # Safety check
            mean_weighted_loss = weighted_loss.mean()
            if torch.isnan(mean_weighted_loss) or mean_weighted_loss > 100:
                print(f"[ERROR] Loss explosion! Loss={mean_weighted_loss.item():.3f}")
                print("[DEBUG] Stopping training - check hyperparameters")
                raise ValueError(f"Loss exploded: {mean_weighted_loss.item()}")

            return model_pred, weighted_loss, split_loss_mask

        else:
            hidden_states = self.model(
                x=noisy_latents,
                grid_sizes=grid_sizes,
                t=timesteps,
                context=context_embeddings,
                packed_seq_params=packed_seq_params,
            )
            return hidden_states
