# fsdp2_utils_t2v.py - FSDP setup for WAN 2.1 T2V FULL fine-tuning
import os
from typing import Dict

import torch
import torch.distributed as dist
from dist_utils import cast_model_to_dtype, is_main_process, print0
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    CheckpointImpl,
    apply_activation_checkpointing,
    checkpoint_wrapper,
)
from torch.distributed.checkpoint import (
    FileSystemWriter,
    FileSystemReader,
    save as dist_save,
    load as dist_load,
)
from torch.distributed.fsdp import (
    CPUOffload,
    MixedPrecision,
    ShardingStrategy,
    StateDictType,
    FullyShardedDataParallel as FSDP,
)
from torch.distributed.fsdp.api import (
    ShardedOptimStateDictConfig,
    ShardedStateDictConfig,
    FullStateDictConfig,
    FullOptimStateDictConfig,
)
from torch.distributed.fsdp.wrap import ModuleWrapPolicy


def is_main_process():
    return (not dist.is_initialized()) or dist.get_rank() == 0

def print0(*args, **kwargs):
    if is_main_process():
        print(*args, **kwargs)

def _summarize_optimizer_state(tag: str, state_dict: dict):
    if state_dict is None or not isinstance(state_dict, dict):
        print0(f"[FSDP][OPT-DEBUG] {tag}: state_dict is None/invalid")
        return

    param_state = state_dict.get("state", {})
    param_groups = state_dict.get("param_groups", [])

    num_params_total = len(param_state)
    num_params_non_empty = sum(
        1 for v in param_state.values()
        if isinstance(v, dict) and len(v) > 0
    )
    has_momentum_like = any(
        isinstance(v, dict) and (
            "momentum" in v
            or "exp_avg" in v
            or "exp_avg_sq" in v
            or "exp_avg_sq_row" in v
            or "exp_avg_sq_col" in v
        )
        for v in param_state.values()
    )

    print0(
        f"[FSDP][OPT-DEBUG] {tag}: "
        f"param_groups={len(param_groups)}, "
        f"state_entries={num_params_total}, "
        f"non_empty_state={num_params_non_empty}, "
        f"has_momentum_like={has_momentum_like}"
    )


def create_fsdp_mixed_precision_policy(bf16):
    """
    Create mixed precision policy for FSDP.

    CRITICAL: Keep param_dtype as fp32 for optimizer stability.
    - param_dtype=fp32: Master weights in fp32 (prevents gradient update loss)
    - reduce_dtype=bf16: Gradient reductions in bf16 (communication efficiency)
    - buffer_dtype=bf16: Buffers in bf16 (memory efficiency)
    """
    return MixedPrecision(
        param_dtype=bf16,  # FIXED: fp32 for optimizer master weights
        reduce_dtype=bf16,
        buffer_dtype=bf16,
    )


def detect_transformer_block_types(transformer):
    """Detect transformer block types for activation checkpointing."""
    block_types = set()

    if hasattr(transformer, "blocks") and transformer.blocks is not None:
        if len(transformer.blocks) > 0:
            block_type = type(transformer.blocks[0])
            block_types.add(block_type)
            print0(f"[FSDP] Detected transformer block type: {block_type.__name__}")

    return block_types


def create_fsdp_wrap_policy(transformer):
    """Create FSDP wrapping policy for WAN 2.1 transformer."""
    wrap_module_types = set()

    if hasattr(transformer, "blocks") and transformer.blocks is not None:
        if len(transformer.blocks) > 0:
            block_type = type(transformer.blocks[0])
            wrap_module_types.add(block_type)
            print0(f"[FSDP] Will wrap transformer blocks of type: {block_type.__name__}")

    # Store references to modules we should NOT wrap
    ignore_modules = set()

    if hasattr(transformer, "condition_embedder"):
        ignore_modules.add(transformer.condition_embedder)
        print0("[FSDP] Will NOT wrap condition_embedder (prevents dimension mismatch)")

    if hasattr(transformer, "patch_embedding"):
        ignore_modules.add(transformer.patch_embedding)
        print0("[FSDP] Will NOT wrap patch_embedding (prevents dimension mismatch)")

    if not wrap_module_types:
        print0("[FSDP] WARNING: No transformer blocks found")
        return None

    return ModuleWrapPolicy(wrap_module_types)


def apply_fsdp_activation_checkpointing(fsdp_model, transformer_block_types):
    """Apply activation checkpointing to transformer blocks."""
    if not transformer_block_types:
        print0("[FSDP] No block types for activation checkpointing")
        return

    def check_fn(submodule):
        return type(submodule) in transformer_block_types

    apply_activation_checkpointing(
        fsdp_model,
        checkpoint_wrapper_fn=lambda module: checkpoint_wrapper(
            module,
            checkpoint_impl=CheckpointImpl.NO_REENTRANT,
        ),
        check_fn=check_fn,
    )

    print0("[FSDP] Applied activation checkpointing")


def setup_fsdp_for_t2v_pipe(
    pipe,
    device,
    bf16,
    local_rank: int,
    cpu_offload: bool = False,
):
    """Setup FSDP for WAN 2.1 T2V transformer - FULL fine-tuning."""

    world_size = dist.get_world_size()
    rank = dist.get_rank()

    print0(f"[FSDP] Setting up T2V FULL fine-tuning with world_size={world_size}, rank={rank}")
    print0(f"[FSDP] CPU offloading: {'ENABLED' if cpu_offload else 'DISABLED'}")

    cpu_offload_config = None
    if cpu_offload:
        cpu_offload_config = CPUOffload(offload_params=True)
        print0("[FSDP] CPU offload will move parameters to CPU when not in use")

    # WAN 2.1 T2V has single transformer
    if not hasattr(pipe, "transformer") or pipe.transformer is None:
        raise RuntimeError("transformer not found in pipeline")

    print0("[INFO] Setting up single transformer for T2V")

    base_transformer = pipe.transformer

    cast_model_to_dtype(base_transformer, bf16)
    base_transformer.to(device)

    # FULL FINE-TUNING: All parameters trainable
    for param in base_transformer.parameters():
        param.requires_grad = True

    trainable_params = sum(p.numel() for p in base_transformer.parameters() if p.requires_grad)
    print0(f"[FSDP] Full fine-tuning: {trainable_params:,} trainable parameters")

    # Disable built-in gradient checkpointing
    if hasattr(base_transformer, "gradient_checkpointing"):
        base_transformer.gradient_checkpointing = False
        print0("[FSDP] Disabled built-in gradient_checkpointing")

    auto_wrap_policy = create_fsdp_wrap_policy(base_transformer)
    mixed_precision_policy = create_fsdp_mixed_precision_policy(bf16)

    print0("[FSDP] Wrapping transformer with FSDP...")

    fsdp_transformer = FSDP(
        base_transformer,
        auto_wrap_policy=auto_wrap_policy,
        mixed_precision=mixed_precision_policy,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        cpu_offload=cpu_offload_config,
        device_id=torch.cuda.current_device(),
        use_orig_params=True,
    )

    print0("[FSDP] Successfully wrapped transformer")
    if cpu_offload:
        print0("[FSDP] CPU offload enabled - parameters will be offloaded to CPU")

    # Configure sharded state dict
    FSDP.set_state_dict_type(
        fsdp_transformer,
        StateDictType.SHARDED_STATE_DICT,
        state_dict_config=ShardedStateDictConfig(offload_to_cpu=True),
        optim_state_dict_config=ShardedOptimStateDictConfig(offload_to_cpu=True),
    )

    print0("[FSDP] Configured optimizer state sharding")

    # Apply activation checkpointing
    block_types = detect_transformer_block_types(base_transformer)
    if block_types:
        print0("[FSDP] Applying activation checkpointing...")
        apply_fsdp_activation_checkpointing(fsdp_transformer, block_types)

    # Verify all parameters are trainable
    trainable_count = sum(1 for p in fsdp_transformer.parameters() if p.requires_grad)
    frozen_count = sum(1 for p in fsdp_transformer.parameters() if not p.requires_grad)
    print0(f"[FSDP] Trainable parameters: {trainable_count}, Frozen: {frozen_count}")

    model_map = {
        "transformer": {
            "fsdp_transformer": fsdp_transformer,
            "base_transformer": base_transformer,
        }
    }

    pipe.transformer = fsdp_transformer

    print0("[FSDP] Transformer setup complete")

    return model_map


def verify_fsdp_setup(model_map: Dict):
    """Verify FSDP setup."""
    print0("[FSDP] Verifying setup...")

    fsdp_model = model_map["transformer"]["fsdp_transformer"]

    print0("[FSDP] transformer:")
    print0(f"  - FSDP wrapped: {isinstance(fsdp_model, FSDP)}")

    trainable_count = sum(1 for p in fsdp_model.parameters() if p.requires_grad)
    frozen_count = sum(1 for p in fsdp_model.parameters() if not p.requires_grad)

    print0(f"  - Trainable parameters: {trainable_count}")
    print0(f"  - Frozen parameters: {frozen_count}")

    if trainable_count == 0:
        raise RuntimeError("No trainable parameters found!")


def test_fsdp_forward_pass(model_map: Dict, device, bf16):
    """Test forward pass."""
    print0("[FSDP] Testing forward pass...")

    model = model_map["transformer"]["fsdp_transformer"]
    model.train()

    # T2V uses 16-channel input (no conditioning concatenation)
    dummy_input = torch.randn(1, 16, 8, 32, 32, device=device, dtype=bf16)
    dummy_timestep = torch.randint(0, 1000, (1,), device=device)
    dummy_encoder_hidden = torch.randn(1, 77, 4096, device=device, dtype=bf16)

    print0("[FSDP] Testing transformer")

    try:
        with torch.no_grad():
            output = model(
                hidden_states=dummy_input,
                timestep=dummy_timestep,
                encoder_hidden_states=dummy_encoder_hidden,
                return_dict=False,
            )
            if isinstance(output, tuple):
                output = output[0]

            print0(f"[FSDP] Transformer forward pass successful! Output shape: {output.shape}")
    except Exception as e:
        print0(f"[WARNING] Transformer test failed: {e}")
        import traceback

        traceback.print_exc()


def get_fsdp_all_parameters(model_map: Dict):
    """Get ALL trainable parameters for full fine-tuning."""
    fsdp_model = model_map["transformer"]["fsdp_transformer"]
    all_params = [p for p in fsdp_model.parameters() if p.requires_grad]
    print0(f"[FSDP] Collected {len(all_params)} trainable parameters")

    if len(all_params) == 0:
        raise RuntimeError("No trainable parameters found!")

    return all_params


def save_fsdp_checkpoint(model_map, optimizer, scheduler, output_dir: str, step: int, consolidate = True):
    if not dist.is_initialized():
        raise RuntimeError("FSDP checkpointing requires torch.distributed to be initialized")

    fsdp_model = model_map["transformer"]["fsdp_transformer"]

    ckpt_dir = os.path.join(output_dir, f"checkpoint-{step}")
    if is_main_process():
        os.makedirs(ckpt_dir, exist_ok=True)
    dist.barrier()

    print0(f"[FSDP] Saving checkpoint to {ckpt_dir}")

    # ==========================
    # 1) Save sharded model via DCP
    # ==========================
    model_path = os.path.join(ckpt_dir, "transformer_model")
    if is_main_process():
        os.makedirs(model_path, exist_ok=True)
    dist.barrier()

    model_state = {"model": fsdp_model.state_dict()}
    dist_save(model_state, FileSystemWriter(model_path))
    dist.barrier()
    print0("[FSDP] ✓ Saved sharded transformer model")

    # ==========================
    # 2) Save FULL optimizer state (rank0-only write, all-rank compute)
    # ==========================
    optim_file = os.path.join(ckpt_dir, "optimizer.pt")

    # All ranks must participate in FSDP.optim_state_dict, even with rank0_only=True.
    with FSDP.state_dict_type(
        fsdp_model,
        StateDictType.FULL_STATE_DICT,
        FullStateDictConfig(rank0_only=True, offload_to_cpu=True),
        FullOptimStateDictConfig(rank0_only=True, offload_to_cpu=True),
    ):
        global_optim_state = FSDP.optim_state_dict(fsdp_model, optimizer)

    # At this point:
    # - rank 0 has the full CPU optimizer dict
    # - other ranks have an empty dict (by contract)
    if is_main_process():
        _summarize_optimizer_state("Global FSDP.optim_state_dict (save)", global_optim_state)
        torch.save(global_optim_state, optim_file)
        print0(f"[FSDP] ✓ Saved global optimizer state to {optim_file}")
        if len(global_optim_state.get("state", {})) == 0:
            print0(
                "[FSDP][OPT-DEBUG][WARNING] Saved optimizer state has 0 entries. "
                "Resume will use a fresh optimizer (expect loss spike)."
            )

    dist.barrier()

    # ==========================
    # 3) Save training state
    # ==========================
    if is_main_process():
        training_state = {
            "step": int(step),
            "scheduler": scheduler.state_dict(),
        }
        torch.save(training_state, os.path.join(ckpt_dir, "training_state.pt"))
        print0("[FSDP] ✓ Saved training_state.pt")

    dist.barrier()



def load_fsdp_checkpoint(model_map, optimizer, scheduler, ckpt_path: str) -> int:
    if not os.path.exists(ckpt_path):
        print0(f"[FSDP] Checkpoint {ckpt_path} not found")
        return 0

    print0(f"[FSDP] Loading checkpoint from {ckpt_path}")
    fsdp_model = model_map["transformer"]["fsdp_transformer"]

    # ============================================================
    # 1) Load sharded transformer model via DCP
    # ============================================================
    model_path = os.path.join(ckpt_path, "transformer_model")

    # Make sure all ranks agree whether model_path exists
    if dist.is_initialized():
        if is_main_process():
            model_dir_exists = os.path.exists(model_path)
        else:
            model_dir_exists = False
        flag = [model_dir_exists]
        dist.broadcast_object_list(flag, src=0)
        model_dir_exists = flag[0]
    else:
        model_dir_exists = os.path.exists(model_path)

    if not model_dir_exists:
        print0("[FSDP][WARNING] Model checkpoint directory not found")
        return 0

    model_state = {"model": fsdp_model.state_dict()}
    dist_load(model_state, FileSystemReader(model_path))
    missing, unexpected = fsdp_model.load_state_dict(model_state["model"], strict=False)
    print0(
        f"[FSDP] Loaded transformer model state "
        f"(missing={len(missing)}, unexpected={len(unexpected)})"
    )

    # ============================================================
    # 2) Load optimizer from FULL optimizer.pt (global, synced)
    # ============================================================
    optim_file = os.path.join(ckpt_path, "optimizer.pt")

    # Global check for optimizer.pt existence (avoid branchy collectives)
    if dist.is_initialized():
        if is_main_process():
            has_optim = os.path.exists(optim_file)
        else:
            has_optim = False
        flag = [has_optim]
        dist.broadcast_object_list(flag, src=0)
        has_optim = flag[0]
    else:
        has_optim = os.path.exists(optim_file)

    base_optim_state = None
    if has_optim:
        if is_main_process():
            try:
                # PyTorch 2.6+: weights_only=True by default is too strict for optimizer dicts.
                base_optim_state = torch.load(
                    optim_file,
                    map_location="cpu",
                    weights_only=False,
                )
            except TypeError:
                # Older PyTorch: no weights_only arg
                base_optim_state = torch.load(optim_file, map_location="cpu")
            except Exception as e:
                print0(
                    f"[FSDP][OPT-DEBUG][WARNING] Failed to load optimizer.pt: {e}. "
                    "Treating as missing optimizer state."
                )
                base_optim_state = None

        # All ranks participate in the same broadcast
        if dist.is_initialized():
            obj_list = [base_optim_state]
            dist.broadcast_object_list(obj_list, src=0)
            base_optim_state = obj_list[0]

        _summarize_optimizer_state("Loaded optimizer.pt (global)", base_optim_state)

        if (
            base_optim_state is None
            or not isinstance(base_optim_state, dict)
            or len(base_optim_state.get("state", {})) == 0
        ):
            print0(
                "[FSDP][OPT-DEBUG][WARNING] optimizer.pt is empty/invalid; "
                "using fresh optimizer (expect possible loss bump)."
            )
        else:
            mapped_osd = FSDP.optim_state_dict_to_load(
                optim_state_dict=base_optim_state,
                model=fsdp_model,
                optim=optimizer,
            )
            _summarize_optimizer_state("Mapped optimizer state", mapped_osd)

            optimizer.load_state_dict(mapped_osd)
            _summarize_optimizer_state(
                "After optimizer.load_state_dict",
                optimizer.state_dict(),
            )
            print0("[FSDP] ✓ Optimizer state loaded from optimizer.pt")
    else:
        print0(
            "[FSDP][WARNING] optimizer.pt not found in checkpoint; "
            "using fresh optimizer."
        )

    # ============================================================
    # 3) Load scheduler + global step (no collectives inside branches)
    # ============================================================
    step = 0
    training_state_path = os.path.join(ckpt_path, "training_state.pt")

    # For scheduler/step it's okay if only rank0 reads the file; no collectives needed.
    if os.path.exists(training_state_path):
        training_state = torch.load(training_state_path, map_location="cpu")
        if "scheduler" in training_state:
            try:
                scheduler.load_state_dict(training_state["scheduler"])
                print0("[FSDP] Scheduler state loaded.")
            except Exception as e:
                print0(f"[FSDP][WARNING] Failed to load scheduler state: {e}")
        step = int(training_state.get("step", 0))
        print0(f"[FSDP] Resumed from global step={step}")
    else:
        print0(
            "[FSDP][WARNING] training_state.pt not found; "
            "starting scheduler and step from 0."
        )

    # ============================================================
    # 4) Final sync
    # ============================================================
    if dist.is_initialized():
        dist.barrier()

    return step
