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

"""
FlowMatching Pipeline
"""

import logging
import math
import random
from typing import Dict, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class FlowMatchingPipeline:
    """
    Flow Matching Pipeline for video generation models.
    
    This class encapsulates all the logic for flow matching training including:
    - Noise scheduling
    - Timestep sampling with various strategies
    - Conditional latent generation
    - Training step execution
    """
    
    def __init__(
        self,
        num_train_timesteps: int = 1000,
        timestep_sampling: str = "lognorm",
        flow_shift: float = 3.0,
        i2v_prob: float = 0.3,
        device: torch.device = None,
    ):
        """
        Initialize the FlowMatching pipeline.
        
        Args:
            num_train_timesteps: Total number of timesteps for the flow
            timestep_sampling: Sampling strategy ("uniform", "lognorm", "mix", "mode")
            flow_shift: Shift parameter for timestep transformation
            i2v_prob: Probability of using image-to-video conditioning
            device: Device to use for computations
        """
        self.num_train_timesteps = num_train_timesteps
        self.timestep_sampling = timestep_sampling
        self.flow_shift = flow_shift
        self.i2v_prob = i2v_prob
        self.device = device if device is not None else torch.device("cuda")
        
        # Initialize components
        self.noise_schedule = LinearInterpolationSchedule(T=num_train_timesteps)
        self.timestep_sampler = TimestepSampler(
            T=num_train_timesteps,
            device=self.device,
            snr_type=timestep_sampling,
        )
    
    def get_condition_latents(self, latents: torch.Tensor, task_type: str) -> torch.Tensor:
        """
        Generate conditional latents based on task type.
        
        Args:
            latents: Input latents [B, C, F, H, W]
            task_type: Task type ("t2v" or "i2v")
            
        Returns:
            Conditional latents [B, C+1, F, H, W]
        """
        b, c, f, h, w = latents.shape
        cond = torch.zeros([b, c + 1, f, h, w], device=latents.device, dtype=latents.dtype)
            
        if task_type == "t2v":
            return cond
        elif task_type == "i2v":
            cond[:, :-1, :1] = latents[:, :, :1]
            cond[:, -1, 0] = 1
            return cond
        else:
            raise ValueError(f"Unsupported task type: {task_type}")
    
    def timestep_transform(self, timesteps: torch.Tensor) -> torch.Tensor:
        """
        Transform timesteps with shift parameter.
        
        Args:
            timesteps: Original timesteps
            
        Returns:
            Transformed timesteps
        """
        if self.flow_shift == 1.0:
            return timesteps
        timesteps_normalized = timesteps / self.num_train_timesteps
        timesteps_transformed = (
            self.flow_shift * timesteps_normalized / 
            (1 + (self.flow_shift - 1) * timesteps_normalized)
        )
        return timesteps_transformed * self.num_train_timesteps
    
    def determine_task_type(self, data_type: str) -> str:
        """
        Determine task type based on data type and randomization.
        
        Args:
            data_type: Type of data ("image" or "video")
            
        Returns:
            Task type ("t2v" or "i2v")
        """
        if data_type == "image":
            return "t2v"
        elif data_type == "video":
            return "i2v" if random.random() < self.i2v_prob else "t2v"
        else:
            return "t2v"
    
    def step(
        self,
        model: nn.Module,
        batch: Dict,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Execute a single training step with flow matching.
        
        Expected batch format:
        {
            "video_latents": torch.Tensor,  # [B, C, F, H, W]
            "text_embeddings": torch.Tensor,  # [B, seq_len, dim]
            "text_mask": torch.Tensor,  # [B, seq_len]
            "text_embeddings_2": torch.Tensor,  # [B, seq_len, dim]
            "text_mask_2": torch.Tensor,  # [B, seq_len]
            Optional: "data_type": str,  # "video" or "image"
        }
        
        Args:
            model: The model to train
            batch: Batch of training data
            device: Device to use
            dtype: Data type for operations
            
        Returns:
            loss: The computed loss
            metrics: Dictionary of training metrics
        """
        
        # Extract batch data
        video_latents = batch["video_latents"].to(device, dtype=dtype)  # [B, C, F, H, W]
        text_embeddings = batch["text_embeddings"].to(device, dtype=dtype)  # [B, seq_len, dim]
        text_mask = batch["text_mask"].to(device, dtype=dtype)  # [B, seq_len]
        text_embeddings_2 = batch["text_embeddings_2"].to(device, dtype=dtype)  # [B, seq_len, dim]
        text_mask_2 = batch["text_mask_2"].to(device, dtype=dtype)  # [B, seq_len]
        
        # Determine task type
        data_type = batch.get("data_type", "video")
        task_type = self.determine_task_type(data_type)
        if task_type == "i2v":
            image_embeds = batch["image_embeds"].to(device, dtype=dtype)
        else:
            image_embeds = torch.zeros(
                video_latents.shape[0],
                729,
                1152,
                dtype=dtype,
                device=device,
            )
        
        # Get condition latents
        cond_latents = self.get_condition_latents(video_latents, task_type)
        
        # Generate noise
        noise = torch.randn_like(video_latents, device=device)  # [B, C, F, H, W]
        
        # Sample and transform timesteps
        timesteps = self.timestep_sampler.sample(video_latents.shape[0], device=device)
        timesteps = self.timestep_transform(timesteps)
        
        # Flow matching: x_t = (1 - t/T) * x0 + (t/T) * noise
        latents_noised = self.noise_schedule.forward(video_latents, noise, timesteps)
        
        # Target is velocity: v = noise - x0
        target = noise - video_latents
        
        # Concatenate noised latents with conditional latents (for HunyuanVideo 1.5)
        # Model expects [B, 65, F, H, W] = [B, 32+32+1, F, H, W]
        latents_with_condition = torch.cat([latents_noised, cond_latents], dim=1)
        
        # Forward pass through model
        model_pred = model(
            latents_with_condition.to(dtype=dtype),
            timesteps,
            encoder_hidden_states=text_embeddings.to(dtype=dtype),
            encoder_attention_mask=text_mask.to(dtype=dtype),
            encoder_hidden_states_2=text_embeddings_2.to(dtype=dtype),
            encoder_attention_mask_2=text_mask_2.to(dtype=dtype),
            image_embeds=image_embeds.to(dtype=dtype),
            return_dict=False
        )[0]
        
        # Compute MSE loss
        target = target.to(dtype=model_pred.dtype)
        loss = nn.functional.mse_loss(model_pred, target)
        
        # Collect metrics
        metrics = {
            "task_type": task_type,
            "data_type": data_type,
        }
        
        return loss, metrics


class LinearInterpolationSchedule:
    """Simple linear interpolation schedule for flow matching"""
    
    def __init__(self, T: int = 1000):
        """
        Initialize the linear interpolation schedule.
        
        Args:
            T: Total number of timesteps
        """
        self.T = T
    
    def forward(self, x0: torch.Tensor, x1: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Linear interpolation: x_t = (1 - t/T) * x0 + (t/T) * x1
        
        Args:
            x0: Starting point (clean latents)
            x1: Ending point (noise)
            t: Timesteps
            
        Returns:
            Interpolated tensor at timestep t
        """
        t_normalized = t / self.T
        t_normalized = t_normalized.view(-1, *([1] * (x0.ndim - 1)))
        return (1 - t_normalized) * x0 + t_normalized * x1


class TimestepSampler:
    """Timestep sampler for flow matching with various strategies"""
    
    TRAIN_EPS = 1e-5
    SAMPLE_EPS = 1e-3
    
    def __init__(
        self, 
        T: int = 1000, 
        device: torch.device = None,
        snr_type: str = "lognorm",
    ):
        """
        Initialize the timestep sampler.
        
        Args:
            T: Total number of timesteps
            device: Device for tensor operations
            snr_type: Sampling strategy ("uniform", "lognorm", "mix", "mode")
        """
        self.T = T
        self.device = device
        self.snr_type = snr_type
    
    def _check_interval(self, eval: bool = False) -> Tuple[float, float]:
        """
        Get the sampling interval with epsilon margins.
        
        Args:
            eval: Whether in evaluation mode
            
        Returns:
            Tuple of (t0, t1) interval bounds
        """
        eps = self.SAMPLE_EPS if eval else self.TRAIN_EPS
        t0 = eps
        t1 = 1.0 - eps
        return t0, t1
    
    def sample(self, batch_size: int, device: torch.device = None) -> torch.Tensor:
        """
        Sample timesteps according to the configured strategy.
        
        Args:
            batch_size: Number of timesteps to sample
            device: Device for tensor operations
            
        Returns:
            Sampled timesteps [batch_size]
        """
        if device is None:
            device = self.device if self.device is not None else torch.device("cuda")
        
        t0, t1 = self._check_interval(eval=False)
        
        if self.snr_type == "uniform":
            t = torch.rand((batch_size,), device=device) * (t1 - t0) + t0
            
        elif self.snr_type == "lognorm":
            u = torch.normal(mean=0.0, std=1.0, size=(batch_size,), device=device)
            t = 1.0 / (1.0 + torch.exp(-u)) * (t1 - t0) + t0
            
        elif self.snr_type == "mix":
            u = torch.normal(mean=0.0, std=1.0, size=(batch_size,), device=device)
            t_lognorm = 1.0 / (1.0 + torch.exp(-u)) * (t1 - t0) + t0
            
            delta = 0.0
            t0_clip = t0 + delta
            t1_clip = t1 - delta
            t_clip_uniform = torch.rand((batch_size,), device=device) * (t1_clip - t0_clip) + t0_clip
            
            mask = (torch.rand((batch_size,), device=device) > 0.3).float()
            t = mask * t_lognorm + (1 - mask) * t_clip_uniform
            
        elif self.snr_type == "mode":
            mode_scale = 1.29
            u = torch.rand(size=(batch_size,), device=device)
            t = 1.0 - u - mode_scale * (torch.cos(math.pi * u / 2.0) ** 2 - 1.0 + u)
            t = t * (t1 - t0) + t0
        else:
            raise ValueError(f"Unknown SNR type: {self.snr_type}")
        
        timesteps = t * self.T
        return timesteps

