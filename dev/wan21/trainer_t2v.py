# trainer_t2v.py - WAN 2.1 T2V Trainer with Mode System (FIXED VERSION)

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
    WAN 2.1 T2V trainer with mode system (FIXED VERSION).
    
    FIXED: Uses manual flow matching instead of scheduler.add_noise()
    
    Features:
    - Mode-based configuration (pretrain/finetune)
    - FSDP for distributed training
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
        print0(f"[INFO] Total GPUs: {self.world_size}, GPUs per node: {self.local_world_size}, Num nodes: {self.num_nodes}")
        print0(f"[INFO] Node rank: {self.node_rank}, Local rank: {self.local_rank}")
        
        # Important note about weight initialization
        if mode == "finetune":
            print0(f"[INFO] Weight initialization: PRETRAINED (from {model_id})")
        else:
            print0(f"[INFO] Weight initialization: RANDOM (true from-scratch training)")
        
        print0(f"[INFO] Learning rate: {learning_rate}")
        print0(f"[INFO] Weight decay: {weight_decay}, Beta2: {beta2}, Grad clip: {grad_clip}")
        print0(f"[INFO] Warmup steps: {warmup_steps}")
        print0(f"[INFO] CPU offload: {'ENABLED' if cpu_offload else 'DISABLED'}")
        print0(f"[INFO] FIXED: Using manual flow matching (no scheduler explosion)")
        print0(f"[INFO] Flow matching config:")
        print0(f"[INFO]   - Enabled: {use_sigma_noise}")
        print0(f"[INFO]   - Timestep sampling: {timestep_sampling}")
        print0(f"[INFO]   - Flow shift: {flow_shift}")
        print0(f"[INFO]   - Mix uniform ratio: {mix_uniform_ratio}")

        self.pipe = None
        self.model_map = {}
        self.optimizer = None
        self.lr_scheduler = None

    def setup_pipeline(self):
        """Load pipeline with mode-aware weight initialization."""
        
        if self.mode == "finetune":
            # FINETUNE: Always load pretrained weights
            print0("[INFO] FINETUNE MODE: Loading pretrained weights...")
            
            self.pipe = WanPipeline.from_pretrained(
                self.model_id, 
                torch_dtype=torch.bfloat16,
            )
            
            print0(f"[INFO] ✓ Loaded pretrained transformer from {self.model_id}")
            
            # Verify weights are actually loaded
            if hasattr(self.pipe, 'transformer') and self.pipe.transformer is not None:
                if hasattr(self.pipe.transformer, 'blocks') and len(self.pipe.transformer.blocks) > 0:
                    first_param = next(self.pipe.transformer.blocks[0].parameters())
                    param_mean = first_param.abs().mean().item()
                    param_std = first_param.std().item()
                    
                    if param_mean < 1e-6:
                        raise RuntimeError(f"[ERROR] Weights are RANDOM! Mean: {param_mean:.2e}")
                    else:
                        print0(f"[INFO] ✓ Pretrained weights verified:")
                        print0(f"[INFO]   - Mean: {param_mean:.4f}")
                        print0(f"[INFO]   - Std:  {param_std:.4f}")
        
        else:  # pretrain mode
            # PRETRAIN: Random initialization (true from-scratch)
            print0("[INFO] PRETRAIN MODE: Random initialization (from scratch)...")
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
            transformer = WanTransformer3DModel(config)
            
            # Load pipeline with random transformer
            self.pipe = WanPipeline.from_pretrained(
                self.model_id,
                transformer=transformer,
                torch_dtype=torch.bfloat16,
            )
            
            # Verify weights are random
            if hasattr(self.pipe.transformer, 'blocks') and len(self.pipe.transformer.blocks) > 0:
                first_param = next(self.pipe.transformer.blocks[0].parameters())
                param_mean = first_param.abs().mean().item()
                param_std = first_param.std().item()
                
                print0(f"[INFO] ✓ Random initialization verified:")
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

    def setup_optim(self, total_steps: int):
        print0("[INFO] Setting up optimizer...")

        # Get ALL trainable parameters
        all_params = get_fsdp_all_parameters(self.model_map)

        if not all_params:
            raise RuntimeError("No trainable parameters found!")

        print0(f"[INFO] Optimizing {len(all_params)} parameters")

        self.optimizer = torch.optim.AdamW(
            all_params, 
            lr=self.learning_rate, 
            weight_decay=self.weight_decay,
            betas=(self.beta1, self.beta2)
        )

        # Create warmup + cosine annealing scheduler
        if self.warmup_steps > 0:
            print0(f"[INFO] Using warmup ({self.warmup_steps} steps) + cosine annealing")
            
            # Lambda function for warmup + cosine
            def lr_lambda(current_step):
                if current_step < self.warmup_steps:
                    # Linear warmup
                    return float(current_step) / float(max(1, self.warmup_steps))
                else:
                    # Cosine annealing after warmup
                    progress = float(current_step - self.warmup_steps) / float(max(1, total_steps - self.warmup_steps))
                    cosine_decay = 0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.141592653589793)))
                    return max(self.lr_min / self.learning_rate, cosine_decay.item())
            
            self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
                self.optimizer,
                lr_lambda=lr_lambda
            )
        else:
            print0(f"[INFO] Using cosine annealing (no warmup)")
            self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, 
                T_max=total_steps, 
                eta_min=self.lr_min
            )

    def validate_setup(self):
        """Validate FSDP setup with a test forward pass."""
        print0("[INFO] Validating FSDP setup...")
        test_fsdp_forward_pass(self.model_map, self.device, self.bf16)

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
        log_every: int = 10,
        output_dir: str = "./wan_t2v_outputs",
        resume_checkpoint: Optional[str] = None,
    ):
        """Train the model with mode-aware configuration."""
        print0(f"\n[INFO] Starting T2V training - Mode: {self.mode.upper()}")
        print0(f"[INFO] Batch size per node: {batch_size_per_node}")
        print0(f"[INFO] Total effective batch size: {batch_size_per_node * self.num_nodes}")

        self.setup_pipeline()
        self.setup_fsdp()

        # Create dataloader
        dataloader, sampler = create_dataloader(meta_folder, batch_size_per_node, self.num_nodes)

        steps_per_epoch = len(dataloader)
        total_steps = num_epochs * steps_per_epoch

        # Setup optimizer with total steps for scheduler
        self.setup_optim(total_steps)
        print0(f"[INFO] Scheduler configured for {total_steps} total steps")

        self.validate_setup()

        global_step = 0
        start_epoch = 0

        if resume_checkpoint:
            global_step = load_fsdp_checkpoint(
                self.model_map, self.optimizer, self.lr_scheduler, resume_checkpoint
            )
            start_epoch = global_step // steps_per_epoch

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
                "batch_size_per_node": batch_size_per_node,
                "total_batch_size": batch_size_per_node * self.num_nodes,
                "num_nodes": self.num_nodes,
                "gpus_per_node": self.local_world_size,
                "total_gpus": self.world_size,
                "cpu_offload": self.cpu_offload,
                "use_sigma_noise": self.use_sigma_noise,
                "timestep_sampling": self.timestep_sampling,
                "flow_shift": self.flow_shift,
                "mix_uniform_ratio": self.mix_uniform_ratio,
                "training_method": "manual_flow_matching_fixed",
            }

            wandb.init(
                project=f"wan-t2v-{self.mode}",
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
                    # FIXED: Uses manual flow matching (no scheduler.add_noise)
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

                # Logging
                if is_main_process() and (global_step % log_every == 0):
                    log_dict = {
                        "train_loss": loss.item(),
                        "train_avg_loss": epoch_loss / num_steps,
                        "lr": self.optimizer.param_groups[0]["lr"],
                        "grad_norm": grad_norm,
                        "epoch": epoch,
                        "global_step": global_step,
                        # Flow matching metrics
                        "sigma_mean": metrics.get("sigma_mean", 0),
                        "sigma_min": metrics.get("sigma_min", 0),
                        "sigma_max": metrics.get("sigma_max", 0),
                        "timestep_min": metrics.get("timestep_min", 0),
                        "timestep_max": metrics.get("timestep_max", 0),
                    }

                    wandb.log(log_dict, step=global_step)

                    if hasattr(iterable, "set_postfix"):
                        iterable.set_postfix(
                            {
                                "loss": f"{loss.item():.4f}",
                                "avg": f"{(epoch_loss / num_steps):.4f}",
                                "lr": f"{self.optimizer.param_groups[0]['lr']:.2e}",
                                "gn": f"{grad_norm:.2f}",
                            }
                        )

                # Checkpointing with consolidation control
                if save_every and (global_step % save_every == 0):
                    # Consolidate based on consolidate_every
                    should_consolidate = (global_step % consolidate_every == 0)
                    
                    save_fsdp_checkpoint(
                        self.model_map,
                        self.optimizer,
                        self.lr_scheduler,
                        output_dir,
                        global_step,
                        consolidate=should_consolidate,
                    )

            # Epoch summary
            avg_loss = epoch_loss / max(num_steps, 1)
            print0(f"[INFO] Epoch {epoch + 1} complete. avg_loss={avg_loss:.6f}")

            if is_main_process():
                wandb.log({"epoch/avg_loss": avg_loss, "epoch/num": epoch + 1}, step=global_step)

        # Final checkpoint (always consolidate)
        if is_main_process():
            print0("[INFO] Training complete, saving final checkpoint...")

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

        print0("[INFO] Training complete!")