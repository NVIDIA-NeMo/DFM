#!/usr/bin/env python3
# main_t2v_unified.py - Unified entry point for WAN 2.1 T2V Training
# Supports both PRETRAINING and FINETUNING modes

import argparse

from dist_utils import print0
from trainer_t2v import WanT2VTrainer


def parse_args():
    p = argparse.ArgumentParser("WAN 2.1 T2V Training (Pretrain or Finetune)")

    # ========================================================================
    # MODE SELECTION - THIS IS THE KEY FLAG
    # ========================================================================
    p.add_argument("--mode", type=str, default="finetune", 
                   choices=["pretrain", "finetune"],
                   help="Training mode: 'pretrain' (from scratch) or 'finetune' (adapt pretrained)")

    # Model configuration
    p.add_argument("--model_id", type=str, default="Wan-AI/Wan2.1-T2V-1.3B-Diffusers", 
                   help="HuggingFace model ID")

    # Training configuration
    p.add_argument("--meta_folder", type=str, required=True, 
                   help="Path to folder containing .meta files")
    p.add_argument("--num_epochs", type=int, default=None,
                   help="Number of training epochs (default: mode-dependent)")
    p.add_argument("--batch_size_per_node", type=int, default=None,
                   help="Batch size per NODE (default: mode-dependent)")
    p.add_argument("--learning_rate", type=float, default=None,
                   help="Learning rate (default: mode-dependent)")

    # Optimizer settings
    p.add_argument("--weight_decay", type=float, default=None,
                   help="Weight decay (default: mode-dependent)")
    p.add_argument("--beta1", type=float, default=0.9,
                   help="Adam beta1")
    p.add_argument("--beta2", type=float, default=None,
                   help="Adam beta2 (default: mode-dependent)")
    p.add_argument("--grad_clip", type=float, default=None,
                   help="Gradient clipping (default: mode-dependent)")

    # Learning rate schedule
    p.add_argument("--warmup_steps", type=int, default=None,
                   help="Warmup steps (default: mode-dependent)")
    p.add_argument("--lr_min", type=float, default=None,
                   help="Minimum LR (default: mode-dependent)")

    # Memory optimization
    p.add_argument("--cpu_offload", action="store_true", default=True,
                   help="Enable CPU offloading")
    p.add_argument("--no_cpu_offload", action="store_false", dest="cpu_offload",
                   help="Disable CPU offloading")

    # Flow matching arguments
    p.add_argument("--use_sigma_noise", action="store_true", default=True,
                   help="Use flow matching noise scheduling")
    p.add_argument("--no_sigma_noise", action="store_false", dest="use_sigma_noise",
                   help="Disable flow matching")
    p.add_argument("--timestep_sampling", type=str, default=None,
                   choices=["uniform", "logit_normal", "mode"],
                   help="Timestep sampling strategy (default: mode-dependent)")
    p.add_argument("--logit_mean", type=float, default=0.0,
                   help="Mean for logit-normal distribution")
    p.add_argument("--logit_std", type=float, default=None,
                   help="Std for logit-normal (default: mode-dependent)")
    p.add_argument("--flow_shift", type=float, default=None,
                   help="Flow matching shift (default: mode-dependent)")
    p.add_argument("--mix_uniform_ratio", type=float, default=None,
                   help="Uniform sampling ratio (default: mode-dependent)")

    # Checkpointing
    p.add_argument("--save_every", type=int, default=None,
                   help="Save checkpoint every N steps (default: mode-dependent)")
    p.add_argument("--consolidate_every", type=int, default=None,
                   help="Consolidate every N steps (default: mode-dependent)")
    p.add_argument("--log_every", type=int, default=None,
                   help="Log every N steps (default: mode-dependent)")
    p.add_argument("--output_dir", type=str, default=None,
                   help="Output directory (default: mode-dependent)")
    p.add_argument("--resume_checkpoint", type=str, default=None, 
                   help="Path to checkpoint to resume from")

    return p.parse_args()


def apply_mode_defaults(args):
    """Apply mode-specific defaults for parameters not explicitly set."""
    
    mode = args.mode
    
    # Define defaults for each mode
    if mode == "pretrain":
        defaults = {
            "num_epochs": 1,
            "batch_size_per_node": 4,
            "learning_rate": 1e-4,
            "weight_decay": 0.1,
            "beta2": 0.95,
            "grad_clip": 2.0,
            "warmup_steps": 1000,
            "lr_min": 1e-5,
            "timestep_sampling": "logit_normal",
            "logit_std": 1.5,
            "flow_shift": 5.0,
            "mix_uniform_ratio": 0.2,
            "save_every": 2000,
            "consolidate_every": 10000,
            "log_every": 10,
            "output_dir": "./wan_t2v_pretrain_outputs",
        }
    else:  # finetune
        defaults = {
            "num_epochs": 10,
            "batch_size_per_node": 1,
            "learning_rate": 1e-5,
            "weight_decay": 0.01,
            "beta2": 0.999,
            "grad_clip": 1.0,
            "warmup_steps": 0,  # No warmup for finetuning
            "lr_min": 1e-6,
            "timestep_sampling": "uniform",
            "logit_std": 1.0,
            "flow_shift": 3.0,
            "mix_uniform_ratio": 0.1,
            "save_every": 500,
            "consolidate_every": 1000,  # More frequent for finetuning
            "log_every": 5,
            "output_dir": "./wan_t2v_finetune_outputs",
        }
    
    # Apply defaults for None values
    for key, default_value in defaults.items():
        if getattr(args, key) is None:
            setattr(args, key, default_value)
    
    return args


def print_config_summary(args):
    """Print a summary of the configuration."""
    print0("\n" + "=" * 80)
    print0(f"WAN 2.1 T2V TRAINING - MODE: {args.mode.upper()}")
    print0("=" * 80)
    
    print0("\n📋 MODEL & DATA:")
    print0(f"  Model ID: {args.model_id}")
    print0(f"  Meta folder: {args.meta_folder}")
    print0(f"  Output dir: {args.output_dir}")
    if args.resume_checkpoint:
        print0(f"  Resume from: {args.resume_checkpoint}")
    
    print0("\n⚙️  TRAINING PARAMETERS:")
    print0(f"  Epochs: {args.num_epochs}")
    print0(f"  Batch size per node: {args.batch_size_per_node}")
    print0(f"  Learning rate: {args.learning_rate:.2e}")
    print0(f"  Weight decay: {args.weight_decay}")
    print0(f"  Gradient clip: {args.grad_clip}")
    print0(f"  Adam betas: ({args.beta1}, {args.beta2})")
    
    print0("\n📈 LEARNING RATE SCHEDULE:")
    if args.warmup_steps > 0:
        print0(f"  Warmup steps: {args.warmup_steps}")
        print0(f"  Schedule: Linear warmup → Cosine decay")
    else:
        print0(f"  Schedule: Cosine decay (no warmup)")
    print0(f"  Min LR: {args.lr_min:.2e}")
    
    print0("\n🌊 FLOW MATCHING:")
    print0(f"  Enabled: {args.use_sigma_noise}")
    if args.use_sigma_noise:
        print0(f"  Timestep sampling: {args.timestep_sampling}")
        print0(f"  Flow shift: {args.flow_shift}")
        print0(f"  Logit std: {args.logit_std}")
        print0(f"  Mix uniform ratio: {args.mix_uniform_ratio}")
    
    print0("\n💾 CHECKPOINTING:")
    print0(f"  Save every: {args.save_every} steps")
    print0(f"  Consolidate every: {args.consolidate_every} steps")
    print0(f"  Log every: {args.log_every} steps")
    print0(f"  CPU offload: {args.cpu_offload}")
    
    print0("\n" + "=" * 80 + "\n")


def main():
    args = parse_args()
    
    # Apply mode-specific defaults
    args = apply_mode_defaults(args)
    
    # Print configuration summary
    print_config_summary(args)
    
    # Confirm mode
    if args.mode == "pretrain":
        print0("🚀 Starting PRETRAINING (learning from scratch)")
        print0("   • Higher learning rate for faster learning")
        print0("   • Stronger regularization to prevent overfitting")
        print0("   • Better timestep coverage for comprehensive training")
    else:
        print0("🎯 Starting FINETUNING (adapting pretrained model)")
        print0("   • Lower learning rate to preserve pretrained features")
        print0("   • Conservative updates to avoid catastrophic forgetting")
        print0("   • Multiple epochs on smaller dataset")
    
    print0("")
    
    # Create unified trainer
    trainer = WanT2VTrainer(
        model_id=args.model_id,
        mode=args.mode,
        learning_rate=args.learning_rate,
        cpu_offload=args.cpu_offload,
        # Optimizer config
        weight_decay=args.weight_decay,
        beta1=args.beta1,
        beta2=args.beta2,
        grad_clip=args.grad_clip,
        # LR schedule config
        warmup_steps=args.warmup_steps,
        lr_min=args.lr_min,
        # Flow matching config
        use_sigma_noise=args.use_sigma_noise,
        timestep_sampling=args.timestep_sampling,
        logit_mean=args.logit_mean,
        logit_std=args.logit_std,
        flow_shift=args.flow_shift,
        mix_uniform_ratio=args.mix_uniform_ratio,
    )

    # Start training
    trainer.train(
        meta_folder=args.meta_folder,
        num_epochs=args.num_epochs,
        batch_size_per_node=args.batch_size_per_node,
        save_every=args.save_every,
        consolidate_every=args.consolidate_every,
        log_every=args.log_every,
        output_dir=args.output_dir,
        resume_checkpoint=args.resume_checkpoint,
    )

    print0(f"\n✅ {args.mode.upper()} complete!")


if __name__ == "__main__":
    main()


# ============================================================================
# USAGE EXAMPLES
# ============================================================================

# 1. FINETUNING with all defaults:
# torchrun --nproc-per-node=8 main_t2v_unified.py \
#     --mode finetune \
#     --meta_folder /path/to/data

# 2. PRETRAINING with all defaults:
# torchrun --nproc-per-node=8 main_t2v_unified.py \
#     --mode pretrain \
#     --meta_folder /path/to/large/dataset

# 3. FINETUNING with custom learning rate:
# torchrun --nproc-per-node=8 main_t2v_unified.py \
#     --mode finetune \
#     --meta_folder /path/to/data \
#     --learning_rate 5e-6

# 4. PRETRAINING with custom batch size and LR:
# torchrun --nproc-per-node=8 main_t2v_unified.py \
#     --mode pretrain \
#     --meta_folder /path/to/large/dataset \
#     --batch_size_per_node 8 \
#     --learning_rate 5e-4

# 5. Multi-node PRETRAINING:
# torchrun \
#     --nnodes=10 \
#     --nproc-per-node=8 \
#     --rdzv-backend=c10d \
#     --rdzv-endpoint=<master_addr>:<master_port> \
#     main_t2v_unified.py \
#     --mode pretrain \
#     --meta_folder /path/to/large/dataset

# 6. FINETUNING with resume:
# torchrun --nproc-per-node=8 main_t2v_unified.py \
#     --mode finetune \
#     --meta_folder /path/to/data \
#     --resume_checkpoint ./wan_t2v_finetune_outputs/checkpoint-5000

# 7. Custom PRETRAINING (override all parameters):
# torchrun --nproc-per-node=8 main_t2v_unified.py \
#     --mode pretrain \
#     --meta_folder /path/to/data \
#     --batch_size_per_node 16 \
#     --learning_rate 1e-3 \
#     --weight_decay 0.15 \
#     --flow_shift 7.0 \
#     --warmup_steps 2000

# 8. FINETUNING without CPU offload (more memory, faster):
# torchrun --nproc-per-node=8 main_t2v_unified.py \
#     --mode finetune \
#     --meta_folder /path/to/data \
#     --no_cpu_offload

# 9. Debug mode with frequent logging:
# DEBUG_TRAINING=1 torchrun --nproc-per-node=8 main_t2v_unified.py \
#     --mode pretrain \
#     --meta_folder /path/to/data \
#     --log_every 1

# 10. Compare modes side-by-side:
# First run: python compare_configs.py
# Then choose your mode and run the appropriate command above