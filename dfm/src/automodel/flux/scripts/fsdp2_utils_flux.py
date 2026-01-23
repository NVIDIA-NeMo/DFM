#!/usr/bin/env python3
"""
FSDP2 工具函数

原始功能:
1. setup_distributed / cleanup_distributed
2. setup_fsdp2_model
3. print_model_size / get_gradient_norm
4. save_checkpoint / load_checkpoint (full_tensor 版，暂不建议使用)

新增功能:
1. save_sharded_checkpoint: 保存 FSDP2 分片 checkpoint（每个 rank 一份），用于多卡推理/恢复训练
"""

import os
import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, ShardingStrategy
from torch.distributed._tensor import DTensor
import logging

logger = logging.getLogger(__name__)


def setup_distributed():
    """
    设置分布式训练环境
    
    Returns:
        rank: 当前进程的 rank
        world_size: 总进程数
        device: 当前设备
    """
    # 从环境变量获取
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    
    # 初始化进程组
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    
    # 设置设备
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    
    if rank == 0:
        logger.info(f"✓ 分布式环境已设置: rank={rank}, world_size={world_size}")
    
    return rank, world_size, device


def cleanup_distributed():
    """清理分布式环境"""
    if dist.is_initialized():
        dist.destroy_process_group()
        logger.info("✓ 分布式环境已清理")


def setup_fsdp2_model(model, world_size):
    """
    使用 FSDP2 包装模型
    
    Args:
        model: 原始模型
        world_size: 总进程数
        
    Returns:
        包装后的 FSDP 模型
    """
    from torch.distributed._composable.fsdp import fully_shard
    
    # 应用 FSDP2
    # 注意：fully_shard 会就地修改模型
    fully_shard(model)
    
    logger.info(f"✓ FSDP2 已应用 (world_size={world_size})")
    
    return model


def print_model_size(model):
    """
    打印模型大小
    
    Args:
        model: 模型
    """
    param_size = 0
    param_count = 0
    
    for param in model.parameters():
        param_count += param.numel()
        param_size += param.numel() * param.element_size()
    
    buffer_size = 0
    buffer_count = 0
    
    for buffer in model.buffers():
        buffer_count += buffer.numel()
        buffer_size += buffer.numel() * buffer.element_size()
    
    size_all_mb = (param_size + buffer_size) / 1024**2
    
    logger.info(f"Model size:")
    logger.info(f"  Parameters: {param_count:,} ({param_size/1024**2:.2f} MB)")
    logger.info(f"  Buffers: {buffer_count:,} ({buffer_size/1024**2:.2f} MB)")
    logger.info(f"  Total: {size_all_mb:.2f} MB")


def get_gradient_norm(model):
    """
    计算模型的梯度范数
    
    Args:
        model: 模型
        
    Returns:
        total_norm: 梯度范数
    """
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    return total_norm


# ================== 原 full_tensor 版本（暂不建议在训练中使用） ==================

def save_checkpoint(
    model,
    optimizer,
    epoch,
    step,
    loss,
    save_path,
    rank,
):
    """
    保存 checkpoint - full_tensor 版（会尝试收集完整模型）

    在 FSDP2 训练中，model.state_dict() 返回的是分片后的参数(DTensor)。
    本函数通过 DTensor.full_tensor() 收集所有分片，然后只在 rank 0 保存。

    ⚠️ 注意:
      - full_tensor() 会触发全局 all_gather，如果其它 rank 没有参与，会导致 NCCL 超时。
      - FLUX-dev 模型约 12B 参数，完整 bfloat16 权重接近 24GB，收集和保存的开销都很大。
      - 当前推荐在训练中改用 save_sharded_checkpoint，只在 eval 或导出时再考虑 full ckpt。
    """
    if rank != 0:
        # 非 rank 0 进程等待
        dist.barrier()
        return
    
    logger.info(f"\n{'='*80}")
    logger.info("开始保存 checkpoint (full_tensor 版)")
    logger.info(f"{'='*80}")
    logger.info(f"保存路径: {save_path}")
    
    # ===== 收集完整的模型参数 =====
    logger.info("正在收集完整模型参数...")
    full_state_dict = {}
    
    # 获取原始 state_dict (包含 DTensor)
    sharded_state_dict = model.state_dict()
    
    total_params = len(sharded_state_dict)
    dtensor_count = 0
    regular_count = 0
    
    with torch.no_grad():
        for name, param in sharded_state_dict.items():
            if isinstance(param, DTensor):
                # DTensor: 需要收集所有分片
                try:
                    full_param = param.full_tensor()
                    full_state_dict[name] = full_param.detach().cpu()
                    dtensor_count += 1
                    
                    if dtensor_count % 100 == 0:
                        logger.info(f"  已处理 {dtensor_count}/{total_params} 个参数...")
                        
                except Exception as e:
                    logger.warning(f"⚠️ 收集参数 {name} 失败: {e}")
                    logger.warning(f"   尝试使用 _local_tensor...")
                    # Fallback: 使用本地分片（不完整，但至少能保存）
                    full_state_dict[name] = param._local_tensor.detach().cpu()
                    dtensor_count += 1
            else:
                # 普通 Tensor: 直接保存
                full_state_dict[name] = param.detach().cpu()
                regular_count += 1
    
    logger.info(f"✓ 参数收集完成:")
    logger.info(f"  DTensor 参数: {dtensor_count}")
    logger.info(f"  普通 Tensor: {regular_count}")
    logger.info(f"  总计: {total_params}")
    
    # ===== 保存 checkpoint =====
    checkpoint = {
        'epoch': epoch,
        'step': step,
        'model_state_dict': full_state_dict,
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    
    logger.info(f"正在保存到 {save_path}...")
    torch.save(checkpoint, save_path)
    
    file_size_mb = os.path.getsize(save_path) / (1024 ** 2)
    logger.info(f"✓ Checkpoint 保存成功!")
    logger.info(f"  文件大小: {file_size_mb:.2f} MB")
    
    if file_size_mb < 10000:  # 小于 10GB
        logger.warning(f"\n⚠️ 警告: checkpoint 大小看起来不像完整 FLUX 模型 (~24GB)")
    
    logger.info(f"{'='*80}\n")
    
    dist.barrier()


def load_checkpoint(
    model,
    optimizer,
    checkpoint_path,
):
    """
    加载 full_tensor 版 checkpoint
    """
    logger.info(f"Loading checkpoint from {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    epoch = checkpoint['epoch']
    step = checkpoint['step']
    loss = checkpoint['loss']
    
    logger.info(f"✓ Checkpoint loaded (epoch {epoch}, step {step}, loss {loss:.4f})")
    
    return epoch, step, loss


# ================== 新增: FSDP2 分片保存 ckpt（推荐用于训练+多卡推理） ==================

def save_sharded_checkpoint(
    model,
    optimizer,
    epoch,
    step,
    loss,
    save_dir,
    rank,
):
    """
    保存 FSDP2 分片 checkpoint（每个 rank 一份），不做 full_tensor 收集。

    Args:
        model: FSDP2 包装的模型
        optimizer: 优化器
        epoch: 当前 epoch
        step: 当前 step
        loss: 当前 avg loss
        save_dir: 存放所有 rank 分片的目录，例如:
            outputs/flux_training/ckpt_epoch_10_sharded
        rank: 当前 rank
    """
    if not dist.is_initialized():
        raise RuntimeError("save_sharded_checkpoint: distributed is not initialized")

    world_size = dist.get_world_size()

    # 只让 rank0 创建目录
    if rank == 0:
        os.makedirs(save_dir, exist_ok=True)

    # 所有进程等目录创建好
    dist.barrier()

    shard_path = os.path.join(save_dir, f"model_rank{rank}.pt")

    # 直接保存 FSDP2 的 sharded state_dict（不做 full_tensor）
    shard_state = {
        "epoch": epoch,
        "step": step,
        "loss": loss,
        "rank": rank,
        "world_size": world_size,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }

    torch.save(shard_state, shard_path)

    # 只在 rank0 打一个汇总日志
    if rank == 0:
        logger.info(f"✓ Sharded checkpoint saved to {save_dir}")
        logger.info(f"  每个 rank 保存一个分片: model_rank0.pt ... model_rank{world_size-1}.pt")