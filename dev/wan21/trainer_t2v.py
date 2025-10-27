# trainer_t2v_unified.py - Unified WAN 2.1 T2V Trainer (Pretrain + Finetune)

import os
from typing import Optional

import torch
import torch.distributed as dist
import wandb
from data_utils import create_dataloader
from diffusers import WanPipeline, WanTransformer3DModel
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
    Unified WAN 2.1 T2V trainer supporting both PRETRAIN and FINETUNE modes.
    
    Mode differences:
    - PRETRAIN: Higher LR, warmup, stronger regularization, better sampling
    - FINETUNE: Lower LR, no warmup, conservative updates, simpler sampling
    """

    def __init__(
        self,
        model_id: str = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
        mode: str = "finetune",  # "pretrain" or "finetune"
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
        if mode not in ["pretrain", "finetune"]:
            raise ValueError(f"Invalid mode: {mode}. Must be 'pretrain' or 'finetune'")
        
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
        print0(f"[INFO] Total GPUs: {self.world_size}, GPUs per node: {self.local_world_size}, Num nodes: {self.num_nodes}")
        print0(f"[INFO] Node rank: {self.node_rank}, Local rank: {self.local_rank}")
        print0(f"[INFO] Learning rate: {learning_rate} (warmup: {warmup_steps} steps, min: {lr_min})")
        print0(f"[INFO] Weight decay: {weight_decay}")
        print0(f"[INFO] Gradient clipping: {grad_clip}")
        print0(f"[INFO] Adam betas: ({beta1}, {beta2})")
        print0(f"[INFO] CPU offload: {'ENABLED' if cpu_offload else 'DISABLED'}")
        print0(f"[INFO] Flow matching: {'ENABLED' if use_sigma_noise else 'DISABLED'}")
        if use_sigma_noise:
            print0(f"[INFO]   - Timestep sampling: {timestep_sampling}")
            print0(f"[INFO]   - Flow shift: {flow_shift}")
            print0(f"[INFO]   - Logit std: {logit_std}")
            print0(f"[INFO]   - Mix uniform ratio: {mix_uniform_ratio}")

        self.pipe = None
        self.model_map = {}
        self.optimizer = None
        self.lr_scheduler = None

    def setup_pipeline(self):
        print0("[INFO] Loading WAN 2.1 T2V pipeline (transformer only)...")

        transformer = WanTransformer3DModel.from_config("./wan2.1_1.3B.config")

        # Load pipeline without VAE or text encoder
        self.pipe = WanPipeline.from_pretrained(
            self.model_id, 
            torch_dtype=torch.float32,
            vae=None,
            text_encoder=None,
            transformer=transformer,
        )

        # Explicitly delete VAE and text encoder if they were loaded
        if hasattr(self.pipe, "vae") and self.pipe.vae is not None:
            print0("[INFO] Removing VAE from pipeline...")
            del self.pipe.vae
            self.pipe.vae = None

        if hasattr(self.pipe, "text_encoder") and self.pipe.text_encoder is not None:
            print0("[INFO] Removing text encoder from pipeline...")
            del self.pipe.text_encoder
            self.pipe.text_encoder = None

        # Clear any cached memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print0("[INFO] Pipeline loaded (transformer + scheduler only)")

    def setup_fsdp(self):
        print0(f"[INFO] Setting up FSDP for {self.mode}...")

        self.model_map = setup_fsdp_for_t2v_pipe(
            self.pipe,
            device=self.device,
            bf16=self.bf16,
            local_rank=self.local_rank,
            cpu_offload=self.cpu_offload,
        )

        verify_fsdp_setup(self.model_map)
        print0("[INFO] FSDP setup complete")

    def setup_optim(self, total_steps: int):
        """Setup optimizer with mode-appropriate parameters."""
        print0(f"[INFO] Setting up optimizer for {self.mode}...")

        # Get ALL trainable parameters
        all_params = get_fsdp_all_parameters(self.model_map)

        if not all_params:
            raise RuntimeError("No trainable parameters found!")

        print0(f"[INFO] Optimizing {len(all_params)} parameters")
        print0(f"[INFO] Weight decay: {self.weight_decay}")
        print0(f"[INFO] Betas: ({self.beta1}, {self.beta2})")

        self.optimizer = torch.optim.AdamW(
            all_params, 
            lr=self.learning_rate, 
            weight_decay=self.weight_decay,
            betas=(self.beta1, self.beta2),
            eps=1e-8,
        )

        # Setup learning rate schedule
        if self.warmup_steps > 0:
            # With warmup (typical for pretraining)
            print0(f"[INFO] Setting up warmup + cosine LR schedule:")
            print0(f"[INFO]   - Warmup steps: {self.warmup_steps}")
            print0(f"[INFO]   - Total steps: {total_steps}")
            print0(f"[INFO]   - Min LR: {self.lr_min}")

            from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR

            # Warmup phase
            warmup_scheduler = LinearLR(
                self.optimizer,
                start_factor=0.1,  # Start at 10% of target LR
                end_factor=1.0,
                total_iters=self.warmup_steps,
            )

            # Cosine decay phase
            cosine_steps = max(1, total_steps - self.warmup_steps)
            cosine_scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=cosine_steps,
                eta_min=self.lr_min,
            )

            # Combine warmup + cosine
            self.lr_scheduler = SequentialLR(
                self.optimizer,
                schedulers=[warmup_scheduler, cosine_scheduler],
                milestones=[self.warmup_steps],
            )
        else:
            # No warmup (typical for finetuning)
            print0(f"[INFO] Setting up cosine LR schedule (no warmup):")
            print0(f"[INFO]   - Total steps: {total_steps}")
            print0(f"[INFO]   - Min LR: {self.lr_min}")

            from torch.optim.lr_scheduler import CosineAnnealingLR
            
            self.lr_scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=total_steps,
                eta_min=self.lr_min,
            )

        print0(f"[INFO] Optimizer and scheduler configured for {self.mode}")

    def validate_setup(self):
        """Validate FSDP setup with a test forward pass."""
        print0("[INFO] Validating FSDP setup...")

        test_fsdp_forward_pass(self.model_map, self.device, self.bf16)

        # Check memory
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated(self.device) / 1024**3
            memory_reserved = torch.cuda.memory_reserved(self.device) / 1024**3
            print0(f"[INFO] GPU memory: {memory_allocated:.2f}GB allocated, {memory_reserved:.2f}GB reserved")

    def train(
        self,
        meta_folder: str,
        num_epochs: int = 10,
        batch_size_per_node: int = 1,
        save_every: int = 500,
        consolidate_every: int = 1000,
        log_every: int = 5,
        output_dir: str = "./wan_t2v_outputs",
        resume_checkpoint: Optional[str] = None,
    ):
        """
        Train the model with flow matching.
        
        Args:
            batch_size_per_node: Batch size for each node (NOT per GPU)
            consolidate_every: Save consolidated model every N steps
        """
        print0("=" * 80)
        print0(f"[INFO] Starting T2V {self.mode.upper()} with Flow Matching")
        print0("=" * 80)
        print0(f"[INFO] Batch size per node: {batch_size_per_node}")
        print0(f"[INFO] Total effective batch size: {batch_size_per_node * self.num_nodes}")
        print0(f"[INFO] Number of epochs: {num_epochs}")

        self.setup_pipeline()
        self.setup_fsdp()

        # Create dataloader first to calculate total steps
        dataloader, sampler = create_dataloader(meta_folder, batch_size_per_node, self.num_nodes)

        steps_per_epoch = len(dataloader)
        total_steps = num_epochs * steps_per_epoch

        print0(f"[INFO] Steps per epoch: {steps_per_epoch}")
        print0(f"[INFO] Total training steps: {total_steps}")

        # Now setup optimizer with correct total_steps
        self.setup_optim(total_steps)
        self.validate_setup()

        global_step = 0
        start_epoch = 0

        if resume_checkpoint:
            global_step = load_fsdp_checkpoint(
                self.model_map, self.optimizer, self.lr_scheduler, resume_checkpoint
            )
            start_epoch = global_step // steps_per_epoch
            print0(f"[INFO] Resumed from step {global_step}, epoch {start_epoch}")

        if is_main_process():
            os.makedirs(output_dir, exist_ok=True)

            config = {
                "mode": self.mode,
                "model_id": self.model_id,
                "learning_rate": self.learning_rate,
                "weight_decay": self.weight_decay,
                "beta1": self.beta1,
                "beta2": self.beta2,
                "grad_clip": self.grad_clip,
                "warmup_steps": self.warmup_steps,
                "lr_min": self.lr_min,
                "num_epochs": num_epochs,
                "batch_size_per_node": batch_size_per_node,
                "total_batch_size": batch_size_per_node * self.num_nodes,
                "num_nodes": self.num_nodes,
                "gpus_per_node": self.local_world_size,
                "total_gpus": self.world_size,
                "total_steps": total_steps,
                "cpu_offload": self.cpu_offload,
                # Flow matching config
                "use_sigma_noise": self.use_sigma_noise,
                "timestep_sampling": self.timestep_sampling,
                "logit_mean": self.logit_mean,
                "logit_std": self.logit_std,
                "flow_shift": self.flow_shift,
                "mix_uniform_ratio": self.mix_uniform_ratio,
            }

            # Use different wandb project based on mode
            project_name = f"wan-t2v-{self.mode}"
            
            wandb.init(
                project=project_name,
                config=config,
                resume=resume_checkpoint is not None,
            )

        if dist.is_initialized():
            dist.barrier()

        # Training loop
        for epoch in range(start_epoch, num_epochs):
            sampler.set_epoch(epoch)

            iterable = dataloader
            if is_main_process():
                from tqdm import tqdm
                iterable = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}")

            epoch_loss = 0.0
            num_steps = 0

            for step, batch in enumerate(iterable):
                self.optimizer.zero_grad(set_to_none=True)

                try:
                    loss, metrics = step_fsdp_transformer_t2v(
                        pipe=self.pipe,
                        model_map=self.model_map,
                        batch=batch,
                        device=self.device,
                        bf16=self.bf16,
                        # Flow matching parameters
                        use_sigma_noise=self.use_sigma_noise,
                        timestep_sampling=self.timestep_sampling,
                        logit_mean=self.logit_mean,
                        logit_std=self.logit_std,
                        flow_shift=self.flow_shift,
                        mix_uniform_ratio=self.mix_uniform_ratio,
                        global_step=global_step,
                    )

                except Exception as e:
                    print0(f"[ERROR] Training step failed at epoch {epoch}, step {step}: {e}")
                    video_shape = batch.get("video_latents", torch.tensor([])).shape
                    text_shape = batch.get("text_embeddings", torch.tensor([])).shape
                    print0(f"[DEBUG] Batch shapes - video: {video_shape}, text: {text_shape}")
                    raise

                loss.backward()

                # Gradient clipping
                trainable_params = [
                    p for p in self.model_map["transformer"]["fsdp_transformer"].parameters() 
                    if p.requires_grad and p.grad is not None
                ]

                grad_norm = 0.0
                if trainable_params:
                    grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=self.grad_clip)
                    grad_norm = float(grad_norm) if torch.is_tensor(grad_norm) else grad_norm

                self.optimizer.step()
                self.lr_scheduler.step()

                epoch_loss += loss.item()
                num_steps += 1
                global_step += 1

                current_lr = self.optimizer.param_groups[0]["lr"]

                # Logging
                if is_main_process() and (global_step % log_every == 0):
                    log_dict = {
                        "train_loss": loss.item(),
                        "train_avg_loss": epoch_loss / num_steps,
                        "lr": current_lr,
                        "grad_norm": grad_norm,
                        "epoch": epoch,
                        "global_step": global_step,
                        # Flow matching metrics
                        "sigma_mean": metrics.get("sigma_mean", 0),
                        "timestep_mean": metrics.get("timestep_mean", 0),
                        "sampling_method": metrics.get("sampling_method", "unknown"),
                    }

                    wandb.log(log_dict, step=global_step)

                    if hasattr(iterable, "set_postfix"):
                        iterable.set_postfix(
                            {
                                "loss": f"{loss.item():.4f}",
                                "avg": f"{(epoch_loss / num_steps):.4f}",
                                "lr": f"{current_lr:.2e}",
                                "gn": f"{grad_norm:.2f}",
                            }
                        )

                # Checkpointing with smart consolidation
                if save_every and (global_step % save_every == 0):
                    # Only consolidate at specified intervals
                    
                    save_fsdp_checkpoint(
                        self.model_map,
                        self.optimizer,
                        self.lr_scheduler,
                        output_dir,
                        global_step,
                        consolidate=True,
                    )

            # Epoch summary
            avg_loss = epoch_loss / max(num_steps, 1)
            print0(f"[INFO] Epoch {epoch + 1} complete. avg_loss={avg_loss:.6f}")

            if is_main_process():
                wandb.log({"epoch/avg_loss": avg_loss, "epoch/num": epoch + 1}, step=global_step)

        # Final checkpoint (always consolidated)
        if is_main_process():
            print0(f"[INFO] {self.mode.upper()} complete, saving final consolidated checkpoint...")

            save_fsdp_checkpoint(
                self.model_map,
                self.optimizer,
                self.lr_scheduler,
                output_dir,
                global_step,
                consolidate=True,  # Always consolidate final checkpoint
            )

            print0(f"[INFO] Saved final checkpoint at step {global_step}")
            wandb.finish()

        print0("=" * 80)
        print0(f"[INFO] {self.mode.upper()} complete!")
        print0("=" * 80)