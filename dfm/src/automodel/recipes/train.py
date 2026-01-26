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

from __future__ import annotations

import logging
import os
from math import ceil
from typing import Any, Dict, Optional

import torch
import torch.distributed as dist
import wandb
from nemo_automodel.components.checkpoint.checkpointing import Checkpointer, CheckpointingConfig
from nemo_automodel.components.loggers.log_utils import setup_logging
from nemo_automodel.components.loggers.wandb_utils import suppress_wandb_log_messages
from nemo_automodel.components.training.rng import StatefulRNG
from nemo_automodel.components.training.step_scheduler import StepScheduler
from nemo_automodel.recipes.base_recipe import BaseRecipe
from nemo_automodel.recipes.llm.train_ft import build_distributed, build_wandb
from torch.distributed.fsdp import MixedPrecisionPolicy
from transformers.utils.hub import TRANSFORMERS_CACHE

from dfm.src.automodel._diffusers.auto_diffusion_pipeline import NeMoAutoDiffusionPipeline
from dfm.src.automodel.flow_matching.flow_matching_pipeline import FlowMatchingPipeline, create_adapter


def build_model_and_optimizer(
    *,
    model_id: str,
    finetune_mode: bool,
    learning_rate: float,
    device: torch.device,
    dtype: torch.dtype,
    cpu_offload: bool = False,
    fsdp_cfg: Dict[str, Any] = {},
    attention_backend: Optional[str] = None,
    optimizer_cfg: Optional[Dict[str, Any]] = None,
    pipeline_spec: Optional[Dict[str, Any]] = None,
) -> tuple[Any, torch.optim.Optimizer, Any]:
    """
    Build the diffusion model, parallel scheme, and optimizer.

    Args:
        model_id: HuggingFace model ID or local path
        finetune_mode: If True, loads pretrained weights (finetuning).
                       If False, initializes with random weights (pretraining).
        learning_rate: Learning rate for optimizer
        device: Device to load model to
        dtype: Data type for model parameters
        cpu_offload: Whether to enable CPU offloading
        fsdp_cfg: FSDP configuration dict
        attention_backend: Optional attention backend to set
        optimizer_cfg: Optimizer configuration dict
        pipeline_spec: Required for pretraining (finetune_mode=False).
                       Dict with transformer_cls, subfolder, etc.
                       Not needed for finetuning.

    Returns:
        Tuple of (pipeline, optimizer, device_mesh)
    """
    logging.info("[INFO] Building NeMoAutoDiffusionPipeline with transformer parallel scheme...")

    if not dist.is_initialized():
        logging.info("[WARN] torch.distributed not initialized; proceeding in single-process mode")

    world_size = dist.get_world_size() if dist.is_initialized() else 1

    if fsdp_cfg.get("dp_size", None) is None:
        denom = max(1, fsdp_cfg.get("tp_size", 1) * fsdp_cfg.get("cp_size", 1) * fsdp_cfg.get("pp_size", 1))
        fsdp_cfg.dp_size = max(1, world_size // denom)

    manager_args: Dict[str, Any] = {
        "dp_size": fsdp_cfg.get("dp_size", None),
        "dp_replicate_size": fsdp_cfg.get("dp_replicate_size", None),
        "tp_size": fsdp_cfg.get("tp_size", 1),
        "cp_size": fsdp_cfg.get("cp_size", 1),
        "pp_size": fsdp_cfg.get("pp_size", 1),
        "backend": "nccl",
        "world_size": world_size,
        "use_hf_tp_plan": fsdp_cfg.get("use_hf_tp_plan", False),
        "activation_checkpointing": True,
        "mp_policy": MixedPrecisionPolicy(
            param_dtype=dtype,
            reduce_dtype=torch.float32,
            output_dtype=dtype,
        ),
    }

    parallel_scheme = {"transformer": manager_args}

    if finetune_mode:
        # Finetuning: use from_pretrained with auto-detection (no pipeline_spec needed)
        logging.info("[INFO] Finetuning mode: using NeMoAutoDiffusionPipeline.from_pretrained")
        pipe, created_managers = NeMoAutoDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=dtype,
            device=device,
            parallel_scheme=parallel_scheme,
            components_to_load=["transformer"],
            load_for_training=True,
        )
    else:
        # Pretraining: use from_config with YAML-specified transformer class
        if pipeline_spec is None:
            raise ValueError(
                "pipeline_spec is required for pretraining (finetune_mode=False). "
                "Add to your YAML config:\n"
                "  pipeline_spec:\n"
                "    transformer_cls: 'FluxTransformer2DModel'  # or WanTransformer3DModel, etc.\n"
                "    subfolder: 'transformer'"
            )
        logging.info("[INFO] Pretraining mode: using NeMoAutoDiffusionPipeline.from_config")
        pipe, created_managers = NeMoAutoDiffusionPipeline.from_config(
            model_id,
            pipeline_spec=pipeline_spec,
            torch_dtype=dtype,
            device=device,
            parallel_scheme=parallel_scheme,
            components_to_load=["transformer"],
        )
    fsdp2_manager = created_managers["transformer"]
    transformer_module = pipe.transformer
    if attention_backend is not None:
        logging.info(f"[INFO] Setting attention backend to {attention_backend}")
        transformer_module.set_attention_backend(attention_backend)

    trainable_params = [p for p in transformer_module.parameters() if p.requires_grad]
    if not trainable_params:
        raise RuntimeError("No trainable parameters found in transformer module!")

    optimizer_cfg = optimizer_cfg or {}
    weight_decay = optimizer_cfg.get("weight_decay", 0.01)
    betas = optimizer_cfg.get("betas", (0.9, 0.999))
    # TODO: Support other optimizers
    optimizer = torch.optim.AdamW(trainable_params, lr=learning_rate, weight_decay=weight_decay, betas=betas)

    logging.info("[INFO] Optimizer config: lr=%s, weight_decay=%s, betas=%s", learning_rate, weight_decay, betas)

    trainable_count = sum(1 for p in transformer_module.parameters() if p.requires_grad)
    frozen_count = sum(1 for p in transformer_module.parameters() if not p.requires_grad)
    logging.info(f"[INFO] Trainable parameters: {trainable_count}, Frozen parameters: {frozen_count}")

    if torch.cuda.is_available():
        memory_allocated = torch.cuda.memory_allocated(device) / 1024**3
        memory_reserved = torch.cuda.memory_reserved(device) / 1024**3
        logging.info(f"[INFO] GPU memory: {memory_allocated:.2f}GB allocated, {memory_reserved:.2f}GB reserved")

    logging.info("[INFO] NeMoAutoDiffusion setup complete (pipeline + optimizer)")

    return pipe, optimizer, getattr(fsdp2_manager, "device_mesh", None)


def build_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    num_epochs: int,
    steps_per_epoch: int,
    eta_min: float = 1e-6,
) -> torch.optim.lr_scheduler.CosineAnnealingLR:
    """Build the cosine annealing learning rate scheduler."""

    total_steps = max(1, num_epochs * max(1, steps_per_epoch))
    logging.info(f"[INFO] Scheduler configured for {total_steps} total steps")
    return torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=total_steps,
        eta_min=eta_min,
    )


def is_main_process():
    return (not dist.is_initialized()) or dist.get_rank() == 0


class TrainDiffusionRecipe(BaseRecipe):
    """Training recipe for diffusion models."""

    def __init__(self, cfg):
        self.cfg = cfg

    def setup(self):
        self.dist_env = build_distributed(self.cfg.get("dist_env", {}))
        setup_logging()

        if self.dist_env.is_main and hasattr(self.cfg, "wandb"):
            suppress_wandb_log_messages()
            run = build_wandb(self.cfg)
            if run is not None:
                logging.info("🚀 View run at {}".format(run.url))

        self.seed = self.cfg.get("seed", 42)
        self.rng = StatefulRNG(seed=self.seed, ranked=True)

        self.model_id = self.cfg.get("model.pretrained_model_name_or_path")
        self.attention_backend = self.cfg.get("model.attention_backend")
        self.learning_rate = self.cfg.get("optim.learning_rate", 5e-6)
        self.bf16 = torch.bfloat16

        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1
        if torch.cuda.is_available():
            self.device = torch.device("cuda", self.local_rank)
        else:
            self.device = torch.device("cpu")

        self.local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", self.world_size))
        self.local_world_size = max(self.local_world_size, 1)
        self.num_nodes = max(1, self.world_size // self.local_world_size)
        self.node_rank = dist.get_rank() // self.local_world_size if dist.is_initialized() else 0

        logging.info("[INFO] Diffusion Trainer with Flow Matching")
        logging.info(
            f"[INFO] Total GPUs: {self.world_size}, GPUs per node: {self.local_world_size}, Num nodes: {self.num_nodes}"
        )
        logging.info(f"[INFO] Node rank: {self.node_rank}, Local rank: {self.local_rank}")
        logging.info(f"[INFO] Learning rate: {self.learning_rate}")

        fsdp_cfg = self.cfg.get("fsdp", {})
        fm_cfg = self.cfg.get("flow_matching", {})

        self.cpu_offload = fsdp_cfg.get("cpu_offload", False)

        # Flow matching configuration
        self.adapter_type = fm_cfg.get("adapter_type", "simple")
        self.timestep_sampling = fm_cfg.get("timestep_sampling", "logit_normal")
        self.logit_mean = fm_cfg.get("logit_mean", 0.0)
        self.logit_std = fm_cfg.get("logit_std", 1.0)
        self.flow_shift = fm_cfg.get("flow_shift", 3.0)
        self.mix_uniform_ratio = fm_cfg.get("mix_uniform_ratio", 0.1)
        self.sigma_min = fm_cfg.get("sigma_min", 0.0)
        self.sigma_max = fm_cfg.get("sigma_max", 1.0)
        self.num_train_timesteps = fm_cfg.get("num_train_timesteps", 1000)
        self.i2v_prob = fm_cfg.get("i2v_prob", 0.3)
        self.use_loss_weighting = fm_cfg.get("use_loss_weighting", True)
        self.log_interval = fm_cfg.get("log_interval", 100)
        self.summary_log_interval = fm_cfg.get("summary_log_interval", 10)

        # Adapter-specific configuration
        adapter_kwargs = fm_cfg.get("adapter_kwargs", {})
        self.adapter_kwargs = adapter_kwargs.to_dict()

        logging.info("[INFO] Flow Matching V2 Pipeline")
        logging.info(f"[INFO]   - Adapter type: {self.adapter_type}")
        logging.info(f"[INFO]   - Timestep sampling: {self.timestep_sampling}")
        logging.info(f"[INFO]   - Flow shift: {self.flow_shift}")
        logging.info(f"[INFO]   - Mix uniform ratio: {self.mix_uniform_ratio}")
        logging.info(f"[INFO]   - Use loss weighting: {self.use_loss_weighting}")

        # Get pipeline_spec from config (required for pretraining, not needed for finetuning)
        finetune_mode = self.cfg.get("model.mode", "finetune").lower() == "finetune"
        pipeline_spec = None
        if not finetune_mode:
            # For pretraining, pipeline_spec is required
            pipeline_spec_cfg = self.cfg.get("model.pipeline_spec", None)
            if pipeline_spec_cfg is not None:
                # Convert config node to dict if needed
                pipeline_spec = pipeline_spec_cfg.to_dict() if hasattr(pipeline_spec_cfg, "to_dict") else dict(pipeline_spec_cfg)

        (self.pipe, self.optimizer, self.device_mesh) = build_model_and_optimizer(
            model_id=self.model_id,
            finetune_mode=finetune_mode,
            learning_rate=self.learning_rate,
            device=self.device,
            dtype=self.bf16,
            cpu_offload=self.cpu_offload,
            fsdp_cfg=fsdp_cfg,
            optimizer_cfg=self.cfg.get("optim.optimizer", {}),
            attention_backend=self.attention_backend,
            pipeline_spec=pipeline_spec,
        )

        self.model = self.pipe.transformer
        self.peft_config = None

        checkpoint_cfg = self.cfg.get("checkpoint", None)

        self.num_epochs = self.cfg.step_scheduler.num_epochs
        self.log_every = self.cfg.get("step_scheduler.log_every", 5)

        # Strictly require checkpoint config from YAML (no fallback)
        if checkpoint_cfg is None:
            raise ValueError(
                "checkpoint config is required in YAML (enabled, checkpoint_dir, model_save_format, save_consolidated)"
            )

        # Build BaseRecipe-style checkpointing configuration (DCP/TORCH_SAVE) from YAML
        model_state_dict_keys = list(self.model.state_dict().keys())
        model_cache_dir = self.cfg.get("model.cache_dir", None)
        self.checkpoint_config = CheckpointingConfig(
            enabled=checkpoint_cfg.get("enabled"),
            checkpoint_dir=checkpoint_cfg.get("checkpoint_dir"),
            model_save_format=checkpoint_cfg.get("model_save_format"),
            model_cache_dir=model_cache_dir if model_cache_dir is not None else TRANSFORMERS_CACHE,
            model_repo_id=self.model_id,
            save_consolidated=checkpoint_cfg.get("save_consolidated"),
            is_peft=False,
            model_state_dict_keys=model_state_dict_keys,
        )
        self.restore_from = checkpoint_cfg.get("restore_from", None)
        self.checkpointer = Checkpointer(
            config=self.checkpoint_config,
            dp_rank=self._get_dp_rank(include_cp=True),
            tp_rank=self._get_tp_rank(),
            pp_rank=self._get_pp_rank(),
            moe_mesh=None,
        )

        dataloader_cfg = self.cfg.get("data.dataloader")
        if not hasattr(dataloader_cfg, "instantiate"):
            raise RuntimeError("data.dataloader must be a config node with instantiate()")

        self.dataloader, self.sampler = dataloader_cfg.instantiate(
            dp_rank=self._get_dp_rank(),
            dp_world_size=self._get_dp_group_size(),
            batch_size=self.cfg.step_scheduler.local_batch_size,
        )

        self.raw_steps_per_epoch = len(self.dataloader)
        if self.raw_steps_per_epoch == 0:
            raise RuntimeError("Training dataloader is empty; cannot proceed with training")

        # Derive DP size consistent with model parallel config
        tp_size = fsdp_cfg.get("tp_size", 1)
        cp_size = fsdp_cfg.get("cp_size", 1)
        pp_size = fsdp_cfg.get("pp_size", 1)
        denom = max(1, tp_size * cp_size * pp_size)
        self.dp_size = fsdp_cfg.get("dp_size", None)
        if self.dp_size is None:
            self.dp_size = max(1, self.world_size // denom)

        # Infer local micro-batch size from dataloader if available
        self.local_batch_size = self.cfg.step_scheduler.local_batch_size
        # Desired global effective batch size across all DP ranks and nodes
        self.global_batch_size = self.cfg.step_scheduler.global_batch_size
        # Steps per epoch after gradient accumulation
        grad_acc_steps = max(1, self.global_batch_size // max(1, self.local_batch_size * self.dp_size))
        self.steps_per_epoch = ceil(self.raw_steps_per_epoch / grad_acc_steps)

        self.lr_scheduler = build_lr_scheduler(
            self.optimizer,
            num_epochs=self.num_epochs,
            steps_per_epoch=self.steps_per_epoch,
        )

        self.global_step = 0
        self.start_epoch = 0
        # Initialize StepScheduler for gradient accumulation and step/epoch bookkeeping
        self.step_scheduler = StepScheduler(
            global_batch_size=self.cfg.step_scheduler.global_batch_size,
            local_batch_size=self.cfg.step_scheduler.local_batch_size,
            dp_size=int(self.dp_size),
            ckpt_every_steps=self.cfg.step_scheduler.ckpt_every_steps,
            dataloader=self.dataloader,
            val_every_steps=None,
            start_step=int(self.global_step),
            start_epoch=int(self.start_epoch),
            num_epochs=int(self.num_epochs),
        )

        self.load_checkpoint(self.restore_from)

        # Init Flow Matching Pipeline V2 with model adapter
        model_adapter = create_adapter(self.adapter_type, **self.adapter_kwargs)
        self.flow_matching_pipeline = FlowMatchingPipeline(
            model_adapter=model_adapter,
            num_train_timesteps=self.num_train_timesteps,
            timestep_sampling=self.timestep_sampling,
            flow_shift=self.flow_shift,
            i2v_prob=self.i2v_prob,
            logit_mean=self.logit_mean,
            logit_std=self.logit_std,
            mix_uniform_ratio=self.mix_uniform_ratio,
            sigma_min=self.sigma_min,
            sigma_max=self.sigma_max,
            use_loss_weighting=self.use_loss_weighting,
            log_interval=self.log_interval,
            summary_log_interval=self.summary_log_interval,
            device=self.device,
        )
        logging.info(f"[INFO] Flow Matching Pipeline V2 initialized with {self.adapter_type} adapter")

        if is_main_process():
            os.makedirs(self.checkpoint_config.checkpoint_dir, exist_ok=True)

        if dist.is_initialized():
            dist.barrier()

    def run_train_validation_loop(self):
        logging.info("[INFO] Starting T2V training with Flow Matching")
        logging.info(f"[INFO] Global Batch size: {self.global_batch_size}; Local Batch size: {self.local_batch_size}")
        logging.info(f"[INFO] Num nodes: {self.num_nodes}; DP size: {self.dp_size}")

        # Keep global_step synchronized with scheduler
        global_step = int(self.step_scheduler.step)

        for epoch in self.step_scheduler.epochs:
            if self.sampler is not None and hasattr(self.sampler, "set_epoch"):
                self.sampler.set_epoch(epoch)

            # Optionally wrap dataloader with tqdm for rank-0
            if is_main_process():
                from tqdm import tqdm

                self.step_scheduler.dataloader = tqdm(self.dataloader, desc=f"Epoch {epoch + 1}/{self.num_epochs}")
            else:
                self.step_scheduler.dataloader = self.dataloader

            epoch_loss = 0.0
            num_steps = 0

            for batch_group in self.step_scheduler:
                self.optimizer.zero_grad(set_to_none=True)

                micro_losses = []
                for micro_batch in batch_group:
                    try:
                        loss, metrics = self.flow_matching_pipeline.step(
                            model=self.model,
                            batch=micro_batch,
                            device=self.device,
                            dtype=self.bf16,
                            global_step=global_step,
                        )
                    except Exception as exc:
                        logging.info(f"[ERROR] Training step failed at epoch {epoch}, step {num_steps}: {exc}")
                        video_shape = micro_batch.get("video_latents", torch.tensor([])).shape
                        text_shape = micro_batch.get("text_embeddings", torch.tensor([])).shape
                        logging.info(f"[DEBUG] Batch shapes - video: {video_shape}, text: {text_shape}")
                        raise

                    (loss / len(batch_group)).backward()
                    micro_losses.append(float(loss.item()))

                grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                grad_norm = float(grad_norm) if torch.is_tensor(grad_norm) else grad_norm

                self.optimizer.step()
                self.lr_scheduler.step()

                group_loss_mean = float(sum(micro_losses) / len(micro_losses))
                epoch_loss += group_loss_mean
                num_steps += 1
                global_step = int(self.step_scheduler.step)

                if self.log_every and self.log_every > 0 and is_main_process() and (global_step % self.log_every == 0):
                    avg_loss = epoch_loss / num_steps
                    log_dict = {
                        "train_loss": group_loss_mean,
                        "train_avg_loss": avg_loss,
                        "lr": self.optimizer.param_groups[0]["lr"],
                        "grad_norm": grad_norm,
                        "epoch": epoch,
                        "global_step": global_step,
                    }
                    if wandb.run is not None:
                        wandb.log(log_dict, step=global_step)

                    # Update tqdm if present
                    if hasattr(self.step_scheduler.dataloader, "set_postfix"):
                        self.step_scheduler.dataloader.set_postfix(
                            {
                                "loss": f"{group_loss_mean:.4f}",
                                "avg": f"{(avg_loss):.4f}",
                                "lr": f"{self.optimizer.param_groups[0]['lr']:.2e}",
                                "gn": f"{grad_norm:.2f}",
                            }
                        )

                if self.step_scheduler.is_ckpt_step:
                    self.save_checkpoint(epoch, global_step)

            avg_loss = epoch_loss / num_steps
            logging.info(f"[INFO] Epoch {epoch + 1} complete. avg_loss={avg_loss:.6f}")

            if is_main_process() and wandb.run is not None:
                wandb.log({"epoch/avg_loss": avg_loss, "epoch/num": epoch + 1}, step=global_step)

        if is_main_process():
            logging.info(f"[INFO] Saved final checkpoint at step {global_step}")
            if wandb.run is not None:
                wandb.finish()

        logging.info("[INFO] Training complete!")
