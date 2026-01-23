#!/usr/bin/env python3
"""
FLUX 训练步骤逻辑 - 对齐 HuggingFace 官方 FLUX 训练脚本

主要对齐点：
1. 使用 FlowMatchEulerDiscreteScheduler 的 timesteps/sigmas 表
2. 使用 compute_density_for_timestep_sampling 采样 u → timesteps
3. 使用 compute_loss_weighting_for_sd3 计算 loss 权重
4. Flow Matching 噪声添加：x_t = (1 - σ) x_0 + σ ε
5. 训练目标：velocity v = ε - x_0
"""

import copy
import logging
from typing import Dict, Tuple

import random
from diffusers import FluxPipeline

import torch
import torch.nn.functional as F
from diffusers import FlowMatchEulerDiscreteScheduler, FluxPipeline
from diffusers.training_utils import (
    compute_density_for_timestep_sampling,
    compute_loss_weighting_for_sd3,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FluxTrainingStep:
    """
    FLUX 训练步骤 - 对齐官方 FLUX 脚本（flow matching + noise_scheduler）
    """

    def __init__(
        self,
        # 采样 & weighting 参数（对应老师脚本里的 args）
        timestep_sampling: str = "logit_normal",  # 实际通过 weighting_scheme 控制
        logit_mean: float = 0.0,
        logit_std: float = 1.0,
        flow_shift: float = 3.0,           # 这里保留但不再用作 loss weight
        mix_uniform_ratio: float = 0.1,    # 若想混合 uniform，可扩展；当前遵循老师脚本主路径

        # Sigma 范围限制（可选）
        sigma_min: float = 0.0,  # pretrain: 0.0, finetune: 0.02
        sigma_max: float = 1.0,  # pretrain: 1.0, finetune: 0.55

        # scheduler / loss weighting
        num_train_timesteps: int = 1000,
        scheduler_pretrained_path: str = None,  # FLUX 模型目录，用于加载 scheduler
        weighting_scheme: str = "logit_normal",  # 和老师脚本中的 args.weighting_scheme 一致
        mode_scale: float = 1.29,                # compute_density_for_timestep_sampling 所需（mode 分布时）
    ):
        """
        初始化训练步骤

        Args:
            scheduler_pretrained_path: FLUX 基础模型路径（包含 scheduler 子目录），例如：
                "/high_perf_store4/evad-tech-vla/houzhiyi/FLUX/models/FLUX.1-dev"
        """
        self.timestep_sampling = timestep_sampling  # 兼容旧参数，真实使用看 weighting_scheme
        self.logit_mean = logit_mean
        self.logit_std = logit_std
        self.flow_shift = flow_shift
        self.mix_uniform_ratio = mix_uniform_ratio
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.num_train_timesteps = num_train_timesteps

        self.weighting_scheme = weighting_scheme
        self.mode_scale = mode_scale

        # ===== 加载官方 FlowMatchEulerDiscreteScheduler =====
        if scheduler_pretrained_path is not None:
            self.noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
                scheduler_pretrained_path,
                subfolder="scheduler",
            )
            self.noise_scheduler_copy = copy.deepcopy(self.noise_scheduler)
            logger.info(
                f"✓ Loaded FlowMatchEulerDiscreteScheduler from "
                f"{scheduler_pretrained_path}/scheduler"
            )
            logger.info(
                f"  num_train_timesteps (scheduler): "
                f"{self.noise_scheduler_copy.config.num_train_timesteps}"
            )
        else:
            self.noise_scheduler = None
            self.noise_scheduler_copy = None
            logger.warning(
                "⚠️ scheduler_pretrained_path 未提供，"
                "不能使用官方 noise_scheduler 的 sigma/time 表。"
                "当前实现需要 scheduler_pretrained_path 才能工作。"
            )

        logger.info("FluxTrainingStep initialized (对齐官方 FLUX 脚本):")
        logger.info(f"  weighting_scheme: {weighting_scheme}")
        logger.info(f"  logit_mean: {logit_mean}")
        logger.info(f"  logit_std: {logit_std}")
        logger.info(f"  sigma_range: [{sigma_min}, {sigma_max}]")
        logger.info(f"  num_train_timesteps (config): {num_train_timesteps}")

    # -------------------------------------------------------------------------
    # 从 timesteps 查 sigma（参考老师脚本中的 get_sigmas）
    # -------------------------------------------------------------------------
    def _get_sigmas_from_timesteps(
        self,
        timesteps: torch.Tensor,
        n_dim: int = 4,
        dtype: torch.dtype = torch.float32,
        device: torch.device = None,
    ) -> torch.Tensor:
        assert self.noise_scheduler_copy is not None, "noise_scheduler_copy 未初始化"

        device = device or timesteps.device
        sigmas_all = self.noise_scheduler_copy.sigmas.to(device=device, dtype=dtype)
        schedule_timesteps = self.noise_scheduler_copy.timesteps.to(device)

        timesteps = timesteps.to(device)
        # 对每个 t，在 scheduler 的 timesteps 表里找到对应的 index
        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

        sigma = sigmas_all[step_indices].flatten()  # [B]
        while sigma.dim() < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma  # shape: [B, 1, 1, 1] (对 4D latents 而言)

    # -------------------------------------------------------------------------
    # 采样 timesteps + sigmas：对齐老师脚本
    # -------------------------------------------------------------------------
    def _sample_timesteps_and_sigmas(
        self,
        batch_size: int,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        使用官方方式采样 timesteps + sigmas：

        1) u ~ compute_density_for_timestep_sampling(...)
        2) indices = floor(u * num_train_timesteps)
        3) timesteps = scheduler.timesteps[indices]
        4) sigmas = scheduler.sigmas[对应 index]
        """
        assert self.noise_scheduler_copy is not None, "noise_scheduler_copy 未初始化"

        # 1) 采样 u
        u = compute_density_for_timestep_sampling(
            weighting_scheme=self.weighting_scheme,
            batch_size=batch_size,
            logit_mean=self.logit_mean,
            logit_std=self.logit_std,
            mode_scale=self.mode_scale,
        ).to(device)

        # 2) u → index → timestep
        num_train_timesteps = self.noise_scheduler_copy.config.num_train_timesteps
        indices = (u * num_train_timesteps).long()
        indices = torch.clamp(indices, min=0, max=num_train_timesteps - 1)

        schedule_timesteps = self.noise_scheduler_copy.timesteps.to(device)
        timesteps = schedule_timesteps[indices]  # [B]

        # 3) timestep → sigma
        # 这里返回的是 shape [B,1,1,1]，方便和 (B,C,H,W) 广播
        sigmas = self._get_sigmas_from_timesteps(
            timesteps,
            n_dim=4,
            dtype=torch.float32,
            device=device,
        ).squeeze(-1).squeeze(-1).squeeze(-1)  # 再压回 [B]，后面 reshape

        # clamp sigma（可选）
        if self.sigma_min > 0.0 or self.sigma_max < 1.0:
            sigmas = torch.clamp(sigmas, self.sigma_min, self.sigma_max)
        else:
            sigmas = torch.clamp(sigmas, 0.0, 1.0)

        return timesteps, sigmas, u

    # -------------------------------------------------------------------------
    # 加噪：x_t = (1-σ) x_0 + σ ε
    # -------------------------------------------------------------------------
    def add_noise(
        self,
        latents: torch.Tensor,
        sigmas_1d: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        添加噪声 (Flow Matching)：x_t = (1 - σ) * x_0 + σ * ε

        Args:
            latents: (B, C, H, W)
            sigmas_1d: (B,) 每个样本的 sigma

        Returns:
            noisy_latents: (B, C, H, W)
            noise: (B, C, H, W)
        """
        noise = torch.randn_like(latents, dtype=torch.float32)
        sigma_reshaped = sigmas_1d.view(-1, 1, 1, 1)  # [B,1,1,1]
        noisy_latents = (1.0 - sigma_reshaped) * latents.float() + sigma_reshaped * noise
        return noisy_latents, noise

    # -------------------------------------------------------------------------
    # Loss 权重：使用 compute_loss_weighting_for_sd3
    # -------------------------------------------------------------------------
    def compute_loss_weight(
        self,
        sigmas_1d: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        """
        计算 loss 权重 - 对齐老师脚本中的 compute_loss_weighting_for_sd3
        """
        # sigmas_1d: [B]
        sigmas_for_weight = sigmas_1d.view(-1, 1, 1, 1).to(device)  # 让 compute_loss_weighting_for_sd3 能广播
        weighting = compute_loss_weighting_for_sd3(
            weighting_scheme=self.weighting_scheme,
            sigmas=sigmas_for_weight,
        )  # 返回 shape 可以广播到 (B,C,H,W)
        return weighting  # 后面直接乘在 MSE 上

        # -------------------------------------------------------------------------
        # 单步训练
        # -------------------------------------------------------------------------
    def training_step(
        self,
        batch: Dict[str, torch.Tensor],
        transformer: torch.nn.Module,
        device: torch.device,
        global_step: int = 0,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """单步训练 - 修复版"""
        
        # ===== 1. 提取数据 =====
        latents = batch["latents"].to(device)
        prompt_embeds = batch["prompt_embeds"].to(device)
        pooled_embeds = batch["pooled_prompt_embeds"].to(device)
        
        B, C, H, W = latents.shape
        
        # ✅ 新增：CFG dropout (10% 概率丢弃 text)
        cfg_dropout_prob = 0.1
        if random.random() < cfg_dropout_prob:
            prompt_embeds = torch.zeros_like(prompt_embeds)
            pooled_embeds = torch.zeros_like(pooled_embeds)
        
        # ===== 2. 采样 timesteps + sigmas =====
        timesteps, sigmas_1d, u = self._sample_timesteps_and_sigmas(B, device)
        
        # ===== 3. 添加噪声 =====
        noisy_latents, noise = self.add_noise(latents, sigmas_1d)
        
        # ===== 4. pack latents（使用官方方法）=====
        packed_noisy_latents = FluxPipeline._pack_latents(
            noisy_latents,
            batch_size=B,
            num_channels_latents=C,
            height=H,
            width=W,
        )
        
        # ===== 5. img_ids & txt_ids（修复维度）=====
        txt_seq_len = prompt_embeds.shape[1]
        txt_ids = torch.zeros(txt_seq_len, 3, device=device, dtype=packed_noisy_latents.dtype)
        
        # ✅ 修复：使用官方函数
        # img_ids = FluxPipeline._prepare_latent_image_ids(
        #     B, H, W, device, packed_noisy_latents.dtype
        # )
        
        img_ids = FluxPipeline._prepare_latent_image_ids(
            B, H // 2, W // 2, device, packed_noisy_latents.dtype
        )

        # ===== 6. Transformer 前向 =====
        timestep_float = (timesteps / 1000.0).float()
        
        # ✅ 修复：使用正确的 guidance scale
        guidance = torch.full((B,), 3.5, device=device, dtype=torch.float32)
        
        dtype = transformer.dtype
        
        model_output = transformer(
            hidden_states=packed_noisy_latents.to(dtype),
            encoder_hidden_states=prompt_embeds.to(dtype),
            pooled_projections=pooled_embeds.to(dtype),
            timestep=timestep_float,
            img_ids=img_ids,
            txt_ids=txt_ids,
            guidance=guidance,
            return_dict=False,
        )[0]
        
        # ===== 7. Unpack（使用官方方法）=====
        vae_scale_factor = 16  # 2^4
        
        model_pred = FluxPipeline._unpack_latents(
            model_output,
            height=H * vae_scale_factor,
            width=W * vae_scale_factor,
            vae_scale_factor=vae_scale_factor,
        )
        
        # ===== 8. velocity target =====
        target = noise - latents.float()
        
        # ===== 9. Loss =====
        mse_loss = F.mse_loss(model_pred.float(), target.float(), reduction="none")
        loss_weight = self.compute_loss_weight(sigmas_1d, device=device)
        
        while loss_weight.dim() < mse_loss.dim():
            loss_weight = loss_weight.unsqueeze(-1)
        
        unweighted_loss = mse_loss.mean()
        weighted_loss = (mse_loss * loss_weight).mean()
        
        # ===== 10. metrics =====
        metrics = {
            "loss": weighted_loss.item(),
            "unweighted_loss": unweighted_loss.item(),
            "sigma_min": sigmas_1d.min().item(),
            "sigma_max": sigmas_1d.max().item(),
            "sigma_mean": sigmas_1d.mean().item(),
            "loss_weight_min": loss_weight.min().item(),
            "loss_weight_max": loss_weight.max().item(),
            "loss_weight_mean": loss_weight.mean().item(),
        }
        
        return weighted_loss, metrics