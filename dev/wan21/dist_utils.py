import logging
import os
import warnings

import torch
import torch.distributed as dist


def is_main_process() -> bool:
    return not dist.is_initialized() or dist.get_rank() == 0


def init_dist():
    """Initialize distributed training - FIXED to use LOCAL_WORLD_SIZE."""
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl", init_method="env://")

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)

    world_size = dist.get_world_size()
    rank = dist.get_rank()

    # CRITICAL FIX: Use LOCAL_WORLD_SIZE (not FSDP_LOCAL_WORLD_SIZE!)
    # This must match what you export in your shell script
    local_world = int(os.environ.get("LOCAL_WORLD_SIZE", "8"))

    node_id = rank // local_world
    node_base = node_id * local_world
    local_ranks = list(range(node_base, node_base + local_world))
    pg_local = dist.new_group(ranks=local_ranks)

    if rank == 0:
        print(f"[DIST] Initialized: world_size={world_size}, local_world_size={local_world}")
        print("[DIST] Node groups created successfully")

    return rank, world_size, local_rank, pg_local


def print0(*a, **k):
    if is_main_process():
        print(*a, **k)


def configure_logging():
    level = logging.INFO if is_main_process() else logging.ERROR
    logging.basicConfig(level=level)
    if dist.is_initialized() and dist.get_rank() != 0:
        warnings.filterwarnings("ignore")


def setup_distributed():
    """Setup distributed training with proper device handling."""
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    # Initialize process group first
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
        print0("[DIST] Initialized process group")

    # THEN set device
    torch.cuda.set_device(local_rank)

    configure_logging()

    # BF16 only; WAN models expect this
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    # seed
    seed = 42 + (dist.get_rank() if dist.is_initialized() else 0)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    print0(
        f"[DIST] Setup complete - Rank: {dist.get_rank() if dist.is_initialized() else 0}, Device: cuda:{local_rank}"
    )
    return local_rank


def cast_model_to_dtype(module, dtype):
    for p in module.parameters(recurse=True):
        if p.dtype.is_floating_point:
            p.data = p.data.to(dtype)
    for b in module.buffers(recurse=True):
        if b.dtype.is_floating_point:
            b.data = b.data.to(dtype)
