#!/usr/bin/env python3
import os
from pathlib import Path
import argparse

import torch
import torch.distributed as dist

from torch.distributed._composable.fsdp import fully_shard
from torch.distributed._tensor import DTensor  # 识别 + full_tensor

from diffusers import FluxTransformer2DModel
from fsdp2_utils_flux import setup_distributed, cleanup_distributed


def parse_args():
    parser = argparse.ArgumentParser(
        description="Merge FSDP2 sharded ckpts into a full ckpt (rank0 only, using DTensor.full_tensor)."
    )

    parser.add_argument(
        "--model_id",
        type=str,
        required=True,
        help="FLUX base model path, e.g. /high_perf_store4/.../FLUX.1-dev",
    )
    parser.add_argument(
        "--ckpt_dir",
        type=str,
        required=True,
        help="Directory containing sharded ckpts, e.g. outputs/.../ckpt_epoch_50_sharded",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to save merged full ckpt, e.g. outputs/.../flux_epoch50_full.pt",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    rank, world_size, device = setup_distributed()

    ckpt_dir = Path(args.ckpt_dir)
    shard_path = ckpt_dir / f"model_rank{rank}.pt"
    if not shard_path.exists():
        raise FileNotFoundError(f"Shard checkpoint not found: {shard_path}")

    if rank == 0:
        print(f"[Rank 0] Building FluxTransformer2DModel structure from "
              f"{args.model_id}/transformer (from config, no pretrained weights) ...")

    # 1. 构建 transformer 结构（必须和训练时一致），只用 config，不加载预训练权重
    config = FluxTransformer2DModel.load_config(
        args.model_id,
        subfolder="transformer",
    )
    transformer = FluxTransformer2DModel.from_config(config)
    transformer = transformer.to(torch.bfloat16)

    # 应用 fully_shard（结构必须和训练时一致）
    fully_shard(transformer)

    # 2. 各 rank 加载自己的 sharded state_dict（里面是 DTensor）
    ckpt = torch.load(shard_path, map_location="cpu")
    sharded_state = ckpt["model_state_dict"]
    transformer.load_state_dict(sharded_state)

    epoch = ckpt.get("epoch", -1)
    step = ckpt.get("step", -1)
    loss = ckpt.get("loss", -1.0)

    dist.barrier()

    # 3. 在所有 rank 上，从 FSDP2 模型的 state_dict 中 “full_tensor” 出完整参数
    if rank == 0:
        print("[Rank 0] Gathering FULL tensors from DTensors via full_tensor() ...")

    sharded_state_dict = transformer.state_dict()
    full_state_dict = {}

    # 注意：这里所有 rank 都要参与 full_tensor() 的通信
    with torch.no_grad():
        for name, param in sharded_state_dict.items():
            if isinstance(param, DTensor):
                # 这里会触发 all_gather，把所有 shard 拼成一个完整 Tensor
                try:
                    full_param = param.full_tensor()
                except Exception as e:
                    if rank == 0:
                        print(f"[Rank 0] full_tensor failed on {name}: {e}")
                    raise
                # 只在 rank0 上保留最终的 full tensor
                if rank == 0:
                    full_state_dict[name] = full_param.detach().cpu()
            else:
                # 非 DTensor 的 buffer / 参数，直接保存（只在 rank0）
                if rank == 0:
                    full_state_dict[name] = param.detach().cpu()

    dist.barrier()

    # 4. 只有 rank0 把 full_state_dict 写盘
    if rank == 0:
        any_key = next(iter(full_state_dict))
        print("[Rank 0] Example param:", any_key,
              "shape:", full_state_dict[any_key].shape,
              "type:", type(full_state_dict[any_key]))

        print("[Rank 0] Saving merged FULL ckpt to disk ...")
        out_ckpt = {
            "epoch": epoch,
            "step": step,
            "loss": loss,
            "model_state_dict": full_state_dict,
        }
        out_path = Path(args.output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        torch.save(out_ckpt, out_path)
        file_size_mb = os.path.getsize(out_path) / (1024 ** 2)
        print(f"[Rank 0] Saved merged full ckpt to {out_path} "
              f"(size: {file_size_mb:.2f} MB)")

    dist.barrier()
    cleanup_distributed()


if __name__ == "__main__":
    main()