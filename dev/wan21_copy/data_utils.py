import torch.distributed as dist
from dataloader import MetaFilesDataset, collate_fn
from torch.utils.data import DataLoader, DistributedSampler
import torch

def _seed_worker(worker_id):
    """Seed each dataloader worker for reproducibility."""
    import random
    import numpy as np
    import torch
    
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def create_dataloader(meta_folder: str, batch_size_per_gpu: int, num_nodes: int):
    """
    Create distributed dataloader for FSDP training.

    FSDP = Data-parallel training with sharded parameters/gradients/optimizer state.

    Communication pattern:
    - Forward: all-gather parameters per layer (reconstruct temporarily)
    - Backward: reduce-scatter gradients per layer (average and shard)

    Args:
        meta_folder: Path to folder containing .meta files
        batch_size_per_gpu: Micro-batch size per GPU (per rank)
        num_nodes: Number of nodes (for logging only)

    Returns:
        Tuple of (DataLoader, DistributedSampler)
    """
    dataset = MetaFilesDataset(meta_folder=meta_folder)

    if not dist.is_initialized():
        # Single process mode
        generator = torch.Generator()
        generator.manual_seed(42)
        
        loader = DataLoader(
            dataset,
            batch_size=batch_size_per_gpu,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=2,
            pin_memory=True,
            drop_last=True,
            worker_init_fn=_seed_worker,
            generator=generator,
        )
        return loader, None

    # Get distributed info
    world_size = dist.get_world_size()  # Total GPUs across all nodes
    rank = dist.get_rank()  # This GPU's rank (0 to world_size-1)

    # Standard data-parallel: each rank gets different data
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,  # Each GPU is a replica
        rank=rank,  # Each GPU's unique rank
        shuffle=True,
    )

    # Create generator for reproducibility (each rank gets different seed)
    generator = torch.Generator()
    generator.manual_seed(42 + rank)

    loader = DataLoader(
        dataset,
        batch_size=batch_size_per_gpu,
        sampler=sampler,
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=True,
        drop_last=True,  # FIXED: Prevent partial batches that misalign with grad accumulation
        worker_init_fn=_seed_worker,
        generator=generator,
    )

    # Debug logging
    from dist_utils import print0

    if rank == 0:
        gpus_per_node = world_size // num_nodes

        print0("\n[DATA] FSDP Data-Parallel Configuration:")
        print0(f"  - World size (total GPUs): {world_size}")
        print0(f"  - Nodes: {num_nodes}")
        print0(f"  - GPUs per node: {gpus_per_node}")
        print0(f"  - Micro-batch per GPU: {batch_size_per_gpu}")
        print0("  - Each GPU processes different data")
        print0("  - Data-parallel with sharded states (not model-parallel)")
        print0("  - Forward: all-gather params per layer")
        print0("  - Backward: reduce-scatter gradients per layer")
        print0(f"  - Sampler: rank={rank}, num_replicas={world_size}\n")

    return loader, sampler