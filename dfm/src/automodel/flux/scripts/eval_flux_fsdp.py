#!/usr/bin/env python3
import argparse
from pathlib import Path

import torch
from diffusers import FluxTransformer2DModel, FluxPipeline
from torch.distributed._tensor import DTensor  # 用于简单检查，防止误用 sharded ckpt


def parse_args():
    parser = argparse.ArgumentParser(
        description="Single-GPU inference with merged full Flux ckpt (from-config model)"
    )

    parser.add_argument(
        "--model_id",
        type=str,
        required=True,
        help="FLUX base model path, e.g. /high_perf_store4/.../FLUX.1-dev",
    )
    parser.add_argument(
        "--full_ckpt_path",
        type=str,
        required=True,
        help="Merged full ckpt path, e.g. outputs/.../flux_epoch40_full.pt",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./eval_outputs",
        help="Directory to save generated images",
    )

    parser.add_argument("--num_inference_steps", type=int, default=28)
    parser.add_argument("--guidance_scale", type=float, default=3.5)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)

    return parser.parse_args()


def main():
    args = parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. 加载 full ckpt
    print(f"Loading full ckpt from {args.full_ckpt_path} ...")
    ckpt = torch.load(args.full_ckpt_path, map_location="cpu")
    full_state_dict = ckpt["model_state_dict"]

    # 简单检查一下类型，防止误用分片 ckpt
    k0 = next(iter(full_state_dict))
    v0 = full_state_dict[k0]
    print("[Eval] example key:", k0, "shape:", v0.shape, "type:", type(v0))
    if isinstance(v0, DTensor):
        raise RuntimeError(
            f"Checkpoint {args.full_ckpt_path} still contains DTensor (key: {k0}). "
            "请确保这是合并后的 FULL ckpt (flux_epochXX_full.pt)，而不是 FSDP 分片 ckpt。"
        )

    epoch = ckpt.get("epoch", -1)
    step = ckpt.get("step", -1)
    loss = ckpt.get("loss", -1.0)
    print(f"ckpt epoch: {epoch}, step: {step}, loss: {loss:.4f}")

    # 2. 用和训练时相同的 config 构建 transformer，并加载训练好的 full 权重
    print(f"Building transformer from config at {args.model_id}/transformer ...")
    config = FluxTransformer2DModel.load_config(
        args.model_id,
        subfolder="transformer",
    )
    transformer = FluxTransformer2DModel.from_config(config).to(torch.bfloat16).to(device)

    print("Loading full_state_dict into transformer (strict=True) ...")
    missing, unexpected = transformer.load_state_dict(full_state_dict, strict=True)
    print("missing keys:", missing)
    print("unexpected keys:", unexpected)

    # 3. 加载 pipeline，并替换 transformer
    print(f"Loading full FluxPipeline from {args.model_id} ...")
    pipe = FluxPipeline.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16,
    ).to(device)

    print("Replacing pipeline.transformer with trained transformer ...")
    pipe.transformer = transformer

    # 4. 准备 prompts
    prompts = [
        "People outside a building on a street with a gay crossing, girl with bicycle, and a motorcycle rider with helmut.",
        "A young woman stands by the post with a pink umbrella.",
        "A row of motorcyclists lined up while passer byers watch.",
        "Two cat lying on a floor playing with each other",
        "A picture of some people about to board a plane.",
        "A bathroom with two sinks and two mirrors.",
    ]

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 5. 执行推理
    for i, prompt in enumerate(prompts):
        print(f"[Eval] epoch {epoch}, prompt[{i}]: {prompt}")
        with torch.no_grad():
            out = pipe(
                prompt=prompt,
                num_inference_steps=args.num_inference_steps,
                guidance_scale=args.guidance_scale,
                height=args.height,
                width=args.width,
            )
        image = out.images[0]
        image_path = out_dir / f"epoch{epoch}_step{step}_prompt{i}.png"
        image.save(image_path)
        print(f"[Eval] Saved image to {image_path}")


if __name__ == "__main__":
    main()