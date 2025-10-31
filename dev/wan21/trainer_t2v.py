# trainer_t2v.py - WAN 2.1 T2V Trainer with Gradient Accumulation (REVISED)

import os
from typing import Optional

import torch
import torch.distributed as dist
import wandb
from data_utils import create_dataloader
from diffusers import WanPipeline
from dist_utils import is_main_process, print0, setup_distributed
from fsdp2_utils_t2v import (
    get_fsdp_all_parameters,
    load_fsdp_checkpoint,
    save_fsdp_checkpoint,
    setup_fsdp_for_t2v_pipe,
    test_fsdp_forward_pass,
    verify_fsdp_setup,
)
from training_step_t2v import step_fsdp_transformer_t2v


class WanT2VTrainer:
    """
    WAN 2.1 T2V trainer with gradient accumulation.

    FSDP = Data-parallel training with sharded parameters/gradients/optimizer state.

    Features:
    - Mode-based configuration (pretrain/finetune)
    - FSDP for distributed training
    - Gradient accumulation support
    - Manual flow matching (no scheduler explosion)
    - Warmup + Cosine annealing LR schedule
    """

    def __init__(
        self,
        model_id: str = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
        mode: str = "finetune",
        learning_rate: float = 1e-5,
        cpu_offload: bool = True,
        # Optimizer config
        weight_decay: float = 0.01,
        beta1: float = 0.9,
        beta2: float = 0.999,
        grad_clip: float = 1.0,
        # LR schedule config
        warmup_steps: int = 0,
        lr_min: float = 1e-6,
        # Flow matching config
        use_sigma_noise: bool = True,
        timestep_sampling: str = "uniform",
        logit_mean: float = 0.0,
        logit_std: float = 1.0,
        flow_shift: float = 3.0,
        mix_uniform_ratio: float = 0.1,
    ):
        self.model_id = model_id
        self.mode = mode
        self.learning_rate = learning_rate
        self.cpu_offload = cpu_offload
        self.bf16 = torch.bfloat16

        # Optimizer config
        self.weight_decay = weight_decay
        self.beta1 = beta1
        self.beta2 = beta2
        self.grad_clip = grad_clip

        # LR schedule config
        self.warmup_steps = warmup_steps
        self.lr_min = lr_min

        # Flow matching config
        self.use_sigma_noise = use_sigma_noise
        self.timestep_sampling = timestep_sampling
        self.logit_mean = logit_mean
        self.logit_std = logit_std
        self.flow_shift = flow_shift
        self.mix_uniform_ratio = mix_uniform_ratio

        self.local_rank = setup_distributed()
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1
        self.device = torch.device("cuda", self.local_rank)

        # Calculate node information
        self.local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", self.world_size))
        self.num_nodes = self.world_size // self.local_world_size if self.local_world_size > 0 else 1
        self.node_rank = dist.get_rank() // self.local_world_size if dist.is_initialized() else 0

        print0(f"[INFO] WAN 2.1 T2V Trainer - Mode: {mode.upper()}")
        print0(
            f"[INFO] Total GPUs: {self.world_size}, GPUs per node: {self.local_world_size}, Num nodes: {self.num_nodes}"
        )
        print0(f"[INFO] Node rank: {self.node_rank}, Local rank: {self.local_rank}")

        # Important note about weight initialization
        if mode == "finetune":
            print0(f"[INFO] Weight initialization: PRETRAINED (from {model_id})")
        else:
            print0("[INFO] Weight initialization: RANDOM (training from scratch)")

    def setup_pipeline(self):
        """Load pipeline with appropriate initialization based on mode."""
        print0(f"[INFO] Loading pipeline: {self.model_id}")

        if self.mode == "finetune":
            # Load pretrained model
            print0("[INFO] Loading PRETRAINED model from HuggingFace...")
            self.pipe = WanPipeline.from_pretrained(
                self.model_id,
                torch_dtype=torch.bfloat16,
            )
            print0("[INFO] ✓ Pretrained model loaded")

        else:  # pretrain mode
            # Load config and reinitialize with random weights
            print0("[INFO] PRETRAIN mode: Initializing with RANDOM weights...")
            print0(f"[INFO] Loading config from {self.model_id}...")

            # Load just the config
            from diffusers import WanTransformer3DModel

            transformer = WanTransformer3DModel.from_pretrained(
                self.model_id,
                subfolder="transformer",
                torch_dtype=torch.bfloat16,
            )

            # Get config and reinitialize with random weights
            config = transformer.config
            print0("[INFO] Reinitializing transformer with RANDOM weights...")
            transformer = WanTransformer3DModel.from_config(config)

            # Load pipeline with random transformer
            self.pipe = WanPipeline.from_pretrained(
                self.model_id,
                transformer=transformer,
                torch_dtype=torch.bfloat16,
            )

            # Verify weights are random
            if hasattr(self.pipe.transformer, "blocks") and len(self.pipe.transformer.blocks) > 0:
                first_param = next(self.pipe.transformer.blocks[0].parameters())
                param_mean = first_param.abs().mean().item()
                param_std = first_param.std().item()

                print0("[INFO] ✓ Random initialization verified:")
                print0(f"[INFO]   - Mean: {param_mean:.4f} (should be near 0)")
                print0(f"[INFO]   - Std:  {param_std:.4f} (should be ~0.02)")

                # Random init should have very small mean
                if param_mean > 0.01:
                    print0(f"[WARNING] Mean seems high for random init: {param_mean:.4f}")

        # Remove VAE and text encoder (we only need transformer + scheduler)
        if hasattr(self.pipe, "vae") and self.pipe.vae is not None:
            print0("[INFO] Removing VAE from pipeline (not needed for training)...")
            del self.pipe.vae
            self.pipe.vae = None

        if hasattr(self.pipe, "text_encoder") and self.pipe.text_encoder is not None:
            print0("[INFO] Removing text encoder from pipeline (not needed for training)...")
            del self.pipe.text_encoder
            self.pipe.text_encoder = None

        # Clear any cached memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        mode_str = "pretrained" if self.mode == "finetune" else "random"
        print0(f"[INFO] ✓ Pipeline ready ({mode_str} transformer + scheduler)")

    def setup_fsdp(self):
        print0("[INFO] Setting up FSDP for full fine-tuning...")

        self.model_map = setup_fsdp_for_t2v_pipe(
            self.pipe,
            device=self.device,
            bf16=self.bf16,
            local_rank=self.local_rank,
            cpu_offload=self.cpu_offload,
        )

        verify_fsdp_setup(self.model_map)
        print0("[INFO] FSDP setup complete")

    def setup_optim(self, total_optimizer_steps: int):
        """
        Setup optimizer and scheduler.

        Args:
            total_optimizer_steps: Total number of optimizer steps (not micro-batch steps)
                                   = num_epochs × steps_per_epoch / grad_accum_steps
        """
        print0("[INFO] Setting up optimizer...")

        # Get ALL trainable parameters
        all_params = get_fsdp_all_parameters(self.model_map)

        if not all_params:
            raise RuntimeError("No trainable parameters found!")

        print0(f"[INFO] Optimizing {len(all_params)} parameters")

        self.optimizer = torch.optim.AdamW(
            all_params, lr=self.learning_rate, weight_decay=self.weight_decay, betas=(self.beta1, self.beta2)
        )

        # Create warmup + cosine annealing scheduler
        # Scheduler steps once per optimizer step (after gradient accumulation)
        if self.warmup_steps > 0:
            print0(f"[INFO] Using warmup ({self.warmup_steps} steps) + cosine annealing")

            # Lambda function for warmup + cosine
            def lr_lambda(current_step):
                if current_step < self.warmup_steps:
                    # Linear warmup
                    return float(current_step) / float(max(1, self.warmup_steps))
                else:
                    # Cosine annealing after warmup
                    progress = float(current_step - self.warmup_steps) / float(
                        max(1, total_optimizer_steps - self.warmup_steps)
                    )
                    cosine_decay = 0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.141592653589793)))
                    return max(self.lr_min / self.learning_rate, cosine_decay.item())

            self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lr_lambda)
        else:
            print0("[INFO] Using cosine annealing (no warmup)")

            def lr_lambda(current_step):
                progress = float(current_step) / float(max(1, total_optimizer_steps))
                cosine_decay = 0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.141592653589793)))
                return max(self.lr_min / self.learning_rate, cosine_decay.item())

            self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lr_lambda)

        print0(f"[INFO] Scheduler configured for {total_optimizer_steps} optimizer steps")

    def validate_setup(self):
        print0("[INFO] Validating FSDP setup...")
        test_fsdp_forward_pass(self.model_map, device=self.device, bf16=self.bf16)
        print0("[INFO] ✓ Validation complete")

        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated(self.device) / 1024**3
            memory_reserved = torch.cuda.memory_reserved(self.device) / 1024**3
            print0(f"[INFO] GPU memory: {memory_allocated:.2f}GB allocated, {memory_reserved:.2f}GB reserved")

    def train(
        self,
        meta_folder: str,
        num_epochs: int = 10,
        batch_size_per_gpu: int = 1,  # RENAMED: per GPU, not per node
        grad_accum_steps: int = 1,  # NEW: gradient accumulation
        save_every: int = 500,
        consolidate_every: int = 1000,
        log_every: int = 10,
        output_dir: str = "./wan_t2v_outputs",
        resume_checkpoint: Optional[str] = None,
    ):
        """
        Train the model with FSDP and gradient accumulation.

        Global batch size = batch_size_per_gpu × grad_accum_steps × world_size

        Args:
            meta_folder: Path to folder with .meta files
            num_epochs: Number of training epochs
            batch_size_per_gpu: Micro-batch size per GPU (per rank)
            grad_accum_steps: Number of gradient accumulation steps
            save_every: Save checkpoint every N optimizer steps
            consolidate_every: Consolidate checkpoint every N optimizer steps
            log_every: Log every N optimizer steps
            output_dir: Output directory for checkpoints
            resume_checkpoint: Path to checkpoint to resume from
        """
        # Calculate global batch size
        global_batch_size = batch_size_per_gpu * grad_accum_steps * self.world_size

        print0(f"\n[INFO] Starting T2V training - Mode: {self.mode.upper()}")
        print0("[INFO] Batch configuration:")
        print0(f"  - Micro-batch per GPU: {batch_size_per_gpu}")
        print0(f"  - Gradient accumulation steps: {grad_accum_steps}")
        print0(f"  - World size (total GPUs): {self.world_size}")
        print0(f"  - Global batch size: {global_batch_size}")
        print0(f"      = {batch_size_per_gpu} × {grad_accum_steps} × {self.world_size}")

        self.setup_pipeline()
        self.setup_fsdp()

        # Create dataloader
        dataloader, sampler = create_dataloader(
            meta_folder, batch_size_per_gpu=batch_size_per_gpu, num_nodes=self.num_nodes
        )

        # Calculate steps
        microbatch_steps_per_epoch = len(dataloader)
        optimizer_steps_per_epoch = microbatch_steps_per_epoch // grad_accum_steps
        total_optimizer_steps = num_epochs * optimizer_steps_per_epoch

        print0("[INFO] Training steps:")
        print0(f"  - Micro-batch steps per epoch: {microbatch_steps_per_epoch}")
        print0(f"  - Optimizer steps per epoch: {optimizer_steps_per_epoch}")
        print0(f"  - Total optimizer steps: {total_optimizer_steps}")

        # Setup optimizer with total OPTIMIZER steps for scheduler
        self.setup_optim(total_optimizer_steps)

        self.validate_setup()

        optimizer_step = 0
        start_epoch = 0

        if resume_checkpoint:
            loaded_step = load_fsdp_checkpoint(self.model_map, self.optimizer, self.lr_scheduler, resume_checkpoint)
            optimizer_step = loaded_step
            start_epoch = (loaded_step * grad_accum_steps) // microbatch_steps_per_epoch

        if is_main_process():
            os.makedirs(output_dir, exist_ok=True)

            config = {
                "model_id": self.model_id,
                "mode": self.mode,
                "learning_rate": self.learning_rate,
                "weight_decay": self.weight_decay,
                "beta1": self.beta1,
                "beta2": self.beta2,
                "grad_clip": self.grad_clip,
                "warmup_steps": self.warmup_steps,
                "lr_min": self.lr_min,
                "num_epochs": num_epochs,
                "batch_size_per_gpu": batch_size_per_gpu,
                "grad_accum_steps": grad_accum_steps,
                "global_batch_size": global_batch_size,
                "num_nodes": self.num_nodes,
                "gpus_per_node": self.local_world_size,
                "total_gpus": self.world_size,
                "microbatch_steps_per_epoch": microbatch_steps_per_epoch,
                "optimizer_steps_per_epoch": optimizer_steps_per_epoch,
                "total_optimizer_steps": total_optimizer_steps,
                "cpu_offload": self.cpu_offload,
                "use_sigma_noise": self.use_sigma_noise,
                "timestep_sampling": self.timestep_sampling,
                "flow_shift": self.flow_shift,
                "mix_uniform_ratio": self.mix_uniform_ratio,
                "training_method": "manual_flow_matching_fsdp_grad_accum",
            }

            wandb.init(
                project=f"wan-t2v-overfit-{self.mode}",
                config=config,
                resume=resume_checkpoint is not None,
            )

        if dist.is_initialized():
            dist.barrier()

        # Training loop
        for epoch in range(start_epoch, num_epochs):
            if sampler is not None:
                sampler.set_epoch(epoch)

            iterable = dataloader
            if is_main_process():
                from tqdm import tqdm

                iterable = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}")

            epoch_loss = 0.0
            accum_loss = 0.0
            microbatch_count = 0

            for microbatch_idx, batch in enumerate(iterable):
                # Determine if this is an accumulation step or optimizer step
                is_accumulating = (microbatch_idx + 1) % grad_accum_steps != 0

                # Zero gradients at start of accumulation cycle
                if microbatch_idx % grad_accum_steps == 0:
                    self.optimizer.zero_grad(set_to_none=True)

                try:
                    # Forward pass with manual flow matching
                    loss, metrics = step_fsdp_transformer_t2v(
                        pipe=self.pipe,
                        model_map=self.model_map,
                        batch=batch,
                        device=self.device,
                        bf16=self.bf16,
                        use_sigma_noise=self.use_sigma_noise,
                        timestep_sampling=self.timestep_sampling,
                        logit_mean=self.logit_mean,
                        logit_std=self.logit_std,
                        flow_shift=self.flow_shift,
                        mix_uniform_ratio=self.mix_uniform_ratio,
                        global_step=optimizer_step,
                    )

                except Exception as e:
                    print0(f"[ERROR] Training step failed at epoch {epoch}, microbatch {microbatch_idx}: {e}")
                    video_shape = batch.get("video_latents", torch.tensor([])).shape
                    text_shape = batch.get("text_embeddings", torch.tensor([])).shape
                    print0(f"[DEBUG] Batch shapes - video: {video_shape}, text: {text_shape}")
                    raise

                # Scale loss by accumulation steps (average over accumulated gradients)
                scaled_loss = loss / grad_accum_steps
                scaled_loss.backward()

                accum_loss += loss.item()
                epoch_loss += loss.item()
                microbatch_count += 1

                # Optimizer step after accumulation
                if not is_accumulating:
                    # Gradient clipping
                    trainable_params = [
                        p
                        for p in self.model_map["transformer"]["fsdp_transformer"].parameters()
                        if p.requires_grad and p.grad is not None
                    ]

                    grad_norm = 0.0
                    if trainable_params:
                        grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=self.grad_clip)
                        grad_norm = float(grad_norm) if torch.is_tensor(grad_norm) else grad_norm

                    # Take optimizer step
                    self.optimizer.step()
                    self.lr_scheduler.step()

                    optimizer_step += 1

                    # Logging (per optimizer step)
                    if is_main_process() and (optimizer_step % log_every == 0):
                        avg_accum_loss = accum_loss / grad_accum_steps

                        log_dict = {
                            "train_loss": avg_accum_loss,  # Average loss over accumulated steps
                            "train_avg_loss": epoch_loss / microbatch_count,
                            "lr": self.optimizer.param_groups[0]["lr"],
                            "grad_norm": grad_norm,
                            "epoch": epoch,
                            "optimizer_step": optimizer_step,
                            "microbatch_step": microbatch_idx + 1,
                            # Flow matching metrics (from last microbatch in accumulation)
                            "sigma_mean": metrics.get("sigma_mean", 0),
                            "sigma_min": metrics.get("sigma_min", 0),
                            "sigma_max": metrics.get("sigma_max", 0),
                            "timestep_min": metrics.get("timestep_min", 0),
                            "timestep_max": metrics.get("timestep_max", 0),
                        }

                        wandb.log(log_dict, step=optimizer_step)

                    # Reset accumulation loss
                    accum_loss = 0.0

                    # Save checkpoint (per optimizer step)
                    if optimizer_step % save_every == 0:
                        # Determine if we should consolidate
                        should_consolidate = optimizer_step % consolidate_every == 0

                        save_fsdp_checkpoint(
                            self.model_map,
                            self.optimizer,
                            self.lr_scheduler,
                            output_dir=output_dir,
                            step=optimizer_step,
                            consolidate=should_consolidate,
                        )

            # End of epoch logging
            if is_main_process():
                avg_epoch_loss = epoch_loss / microbatch_count
                print0(f"[INFO] Epoch {epoch + 1}/{num_epochs} complete - Avg loss: {avg_epoch_loss:.6f}")

            if dist.is_initialized():
                dist.barrier()

        # Save final checkpoint
        if is_main_process():
            save_fsdp_checkpoint(
                self.model_map,
                self.optimizer,
                self.lr_scheduler,
                output_dir=output_dir,
                step=optimizer_step,
                consolidate=True,  # Always consolidate final checkpoint
            )

        if dist.is_initialized():
            dist.barrier()

        print0(f"\n✅ {self.mode.upper()} complete!")
        print0(f"Total optimizer steps: {optimizer_step}")
        print0(f"Global batch size: {global_batch_size}")
