#!/usr/bin/env python3
"""
FLUX 图像生成模型训练主程序 (TensorBoard + 文生图验证, 分片 ckpt 保存)
"""

import os
import sys
import argparse
from pathlib import Path
import logging

import torch
import torch.distributed as dist
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dataloader_flux import create_dataloader
from training_step_flux import FluxTrainingStep
from fsdp2_utils_flux import (
    setup_distributed,
    setup_fsdp2_model,
    cleanup_distributed,
    print_model_size,
    get_gradient_norm,
    save_sharded_checkpoint,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class FluxTrainer:
    """FLUX 训练器 (TensorBoard + 文生图验证, 分片 ckpt 保存)"""

    def __init__(self, args):
        self.args = args

        # 分布式
        self.rank, self.world_size, self.device = setup_distributed()

        # 输出目录 & 日志文件
        self.output_dir = Path(args.output_dir)
        if self.rank == 0:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            log_file = self.output_dir / "train.log"
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(
                logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
            )
            logger.addHandler(file_handler)

            # TensorBoard 日志目录：优先用平台注入的 TENSORBOARD_LOG_PATH
            tb_log_dir = os.getenv("TENSORBOARD_LOG_PATH", str(self.output_dir / "tb_logs"))
            logger.info(f"TensorBoard log dir: {tb_log_dir}")
            self.writer = SummaryWriter(log_dir=tb_log_dir)
        else:
            self.writer = None

        if self.rank == 0:
            logger.info("=" * 70)
            logger.info("FLUX 图像生成模型训练 (TensorBoard + 文生图验证, 分片 ckpt 保存)")
            logger.info("=" * 70)
            logger.info(f"输出目录: {self.output_dir}")
            logger.info(f"World size: {self.world_size}")
            logger.info(f"Batch size per GPU: {args.batch_size_per_gpu}")
            logger.info(f"Global batch size: {args.batch_size_per_gpu * self.world_size}")
            logger.info(f"Learning rate: {args.learning_rate}")
            logger.info(f"Num epochs: {args.num_epochs}")
            logger.info("")
            logger.info("Flow Matching 参数:")
            logger.info(f"  flow_shift: {args.flow_shift}")
            logger.info(f"  mix_uniform_ratio: {args.mix_uniform_ratio}")
            logger.info(f"  sigma_range: [{args.sigma_min}, {args.sigma_max}]")
            logger.info("")
            logger.info("验证推理参数:")
            logger.info(f"  validate_every: {args.validate_every} iters")
            logger.info(f"  val_num_inference_steps: {args.val_num_inference_steps}")
            logger.info(f"  val_guidance_scale: {args.val_guidance_scale}")
            logger.info(f"  val_resolution: {args.val_height}x{args.val_width}")

        # 1. 构建 Transformer 模型（从 config 随机初始化）
        self.model = self.load_model()

        # 2. 应用 FSDP2 包裹训练模型
        if self.rank == 0:
            logger.info("\n应用 FSDP2...")
        self.model = setup_fsdp2_model(self.model, self.world_size)

        # 3. 训练优化器
        self.optimizer = self.create_optimizer()

        # 4. DataLoader（先建 dataloader 再建 scheduler）
        self.train_loader = self.create_dataloader()

        # 5. 学习率调度器
        self.scheduler = self.create_scheduler()

        # 6. 训练步骤逻辑（Flow Matching）
        self.training_step = FluxTrainingStep(
            logit_mean=args.logit_mean,
            logit_std=args.logit_std,
            flow_shift=args.flow_shift,
            mix_uniform_ratio=args.mix_uniform_ratio,
            sigma_min=args.sigma_min,
            sigma_max=args.sigma_max,
            num_train_timesteps=args.num_train_timesteps,
            scheduler_pretrained_path=args.model_id,
            weighting_scheme="logit_normal",
            mode_scale=1.29,
        )

        # 7. 验证用 Pipeline（当前关闭，如果后续需要可以再打开）
        self.val_pipeline = None

        self.current_epoch = 0
        self.current_step = 0
        self.best_loss = float("inf")

    def load_model(self):
        """构建待训练的 FLUX Transformer 模型（从 config 随机初始化）"""
        if self.rank == 0:
            logger.info("\n构建 FLUX Transformer 模型（从 config 随机初始化，不加载预训练权重）...")

        from diffusers import FluxTransformer2DModel

        config = FluxTransformer2DModel.load_config(
            self.args.model_id,
            subfolder="transformer",
        )

        model = FluxTransformer2DModel.from_config(config)
        model = model.to(torch.bfloat16)

        # ✅ 启用 gradient checkpointing
        if hasattr(model, 'enable_gradient_checkpointing'):
            model.enable_gradient_checkpointing()
            if self.rank == 0:
                logger.info("✅ Gradient checkpointing 已启用 (可省 40-60% 显存)")
        else:
            if self.rank == 0:
                logger.warning("⚠️ 模型不支持 gradient_checkpointing")

        if self.rank == 0:
            logger.info("✓ Transformer 模型从 config 构建完成（未加载预训练权重）")
            print_model_size(model)

        return model

    def create_optimizer(self):
        optimizer = AdamW(
            self.model.parameters(),
            lr=self.args.learning_rate,
            betas=(0.9, 0.999),
            weight_decay=self.args.weight_decay,
            eps=1e-8,
        )
        if self.rank == 0:
            logger.info("\n✓ 优化器创建成功 (AdamW)")
        return optimizer

    def create_scheduler(self):
        """创建学习率调度器（使用已创建好的 self.train_loader）"""
        total_steps = len(self.train_loader) * self.args.num_epochs

        scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=total_steps,
            eta_min=self.args.learning_rate * 0.1,
        )

        if self.rank == 0:
            logger.info(f"✓ 学习率调度器创建成功 (Cosine Annealing)")
            logger.info(f"  Total steps: {total_steps}")

        return scheduler

    def create_dataloader(self):
        dataloader = create_dataloader(
            meta_folder=self.args.meta_folder,
            batch_size=self.args.batch_size_per_gpu,
            num_workers=self.args.num_workers,
            shuffle=True,
            pin_memory=True,
        )
        if self.rank == 0:
            logger.info(f"\n✓ DataLoader 创建成功")
            logger.info(f"  Total samples: {len(dataloader.dataset)}")
            logger.info(f"  Batches per epoch: {len(dataloader)}")
        return dataloader

    @torch.no_grad()
    def run_validation(self, epoch: int, step: int):
        """使用当前模型做一次真实的文生图验证（当前 val_pipeline 关闭，如需可恢复）"""
        if self.rank != 0:
            return
        if self.val_pipeline is None:
            logger.warning("⚠️ 没有可用的验证 pipeline，跳过 validation")
            return

        self.model.eval()

        prompts = [
            "a cute cat in watercolor style",
            "a futuristic city at night, highly detailed, 4k",
            "一只在雪地里奔跑的哈士奇，摄影作品，清晰，高质量",
        ]

        images = []
        for i, prompt in enumerate(prompts):
            logger.info(f"[Validation] epoch {epoch}, step {step}, prompt[{i}]: {prompt}")
            out = self.val_pipeline(
                prompt=prompt,
                num_inference_steps=self.args.val_num_inference_steps,
                guidance_scale=self.args.val_guidance_scale,
                height=self.args.val_height,
                width=self.args.val_width,
            )
            img = out.images[0]
            images.append((prompt, img))

        from torchvision.transforms.functional import to_tensor

        val_dir = self.output_dir / "validation_images"
        val_dir.mkdir(parents=True, exist_ok=True)

        for i, (prompt, img) in enumerate(images):
            img_path = val_dir / f"epoch{epoch}_step{step}_idx{i}.png"
            img.save(img_path)
            logger.info(f"✓ Validation image saved to {img_path}")

            if self.writer is not None:
                img_tensor = to_tensor(img)
                self.writer.add_image(
                    f"validation/image_{i}",
                    img_tensor,
                    global_step=step,
                )
                self.writer.add_text(
                    f"validation/prompt_{i}",
                    f"epoch {epoch}, step {step}: {prompt}",
                    global_step=step,
                )

        self.model.train()

    def train_one_epoch(self, epoch: int):
        self.model.train()

        total_loss = 0.0
        total_steps = 0

        if self.rank == 0:
            pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}/{self.args.num_epochs}")
        else:
            pbar = self.train_loader

        for batch_idx, batch in enumerate(pbar):
            # 单步训练
            loss, metrics = self.training_step.training_step(
                batch=batch,
                transformer=self.model,
                device=self.device,
                global_step=self.current_step,
            )

            loss.backward()

            grad_norm = get_gradient_norm(self.model)
            if self.args.max_grad_norm > 0 and grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.args.max_grad_norm,
                )

            self.optimizer.step()
            self.optimizer.zero_grad()
            self.scheduler.step()

            total_loss += loss.item()
            total_steps += 1
            self.current_step += 1

            if self.rank == 0:
                pbar.set_postfix(
                    {
                        "loss": f"{loss.item():.4f}",
                        "σ": f'{metrics["sigma_mean"]:.3f}',
                        "w": f'{metrics["loss_weight_mean"]:.2f}',
                        "lr": f"{self.scheduler.get_last_lr()[0]:.2e}",
                    }
                )

            if self.rank == 0 and (batch_idx + 1) % self.args.log_every == 0:
                avg_loss = total_loss / total_steps
                logger.info(
                    f"Epoch [{epoch}/{self.args.num_epochs}] "
                    f"Step [{batch_idx + 1}/{len(self.train_loader)}] "
                    f"GlobalStep [{self.current_step}] "
                    f"Loss: {loss.item():.4f} "
                    f"Avg Loss: {avg_loss:.4f} "
                    f"σ: {metrics['sigma_mean']:.3f} "
                    f"Weight: {metrics['loss_weight_mean']:.2f} "
                    f"LR: {self.scheduler.get_last_lr()[0]:.2e} "
                    f"Grad Norm: {grad_norm:.4f}"
                )

                if self.writer is not None:
                    self.writer.add_scalar("train/loss", loss.item(), self.current_step)
                    self.writer.add_scalar("train/avg_loss", avg_loss, self.current_step)
                    self.writer.add_scalar(
                        "train/sigma_mean", metrics["sigma_mean"], self.current_step
                    )
                    self.writer.add_scalar(
                        "train/loss_weight_mean",
                        metrics["loss_weight_mean"],
                        self.current_step,
                    )
                    self.writer.add_scalar(
                        "train/lr", self.scheduler.get_last_lr()[0], self.current_step
                    )
                    self.writer.add_scalar(
                        "train/grad_norm", grad_norm, self.current_step
                    )

        return total_loss / total_steps if total_steps > 0 else 0.0

    def train(self):
        if self.rank == 0:
            logger.info("\n" + "=" * 70)
            logger.info("开始训练")
            logger.info("=" * 70)

        for epoch in range(1, self.args.num_epochs + 1):
            self.current_epoch = epoch
            avg_loss = self.train_one_epoch(epoch)

            # 日志 & best loss
            if self.rank == 0:
                logger.info(f"[Epoch {epoch}] Avg Loss: {avg_loss:.4f}")
                if self.writer is not None:
                    self.writer.add_scalar("epoch/avg_loss", avg_loss, epoch)
                if avg_loss < self.best_loss:
                    self.best_loss = avg_loss

            # === 每 save_every 个 epoch 保存一次分片 ckpt ===
            if epoch % self.args.save_every == 0:
                save_dir = self.output_dir / f"ckpt_epoch_{epoch}_sharded"
                save_sharded_checkpoint(
                    model=self.model,
                    optimizer=self.optimizer,
                    epoch=epoch,
                    step=self.current_step,
                    loss=avg_loss,
                    save_dir=str(save_dir),
                    rank=self.rank,
                )
            # ====================================================

        if self.rank == 0:
            logger.info("\n" + "=" * 70)
            logger.info("训练完成!")
            logger.info("=" * 70)
            logger.info(f"最佳 Loss: {self.best_loss:.4f}")
            logger.info(f"输出目录: {self.output_dir}")
            if self.writer is not None:
                self.writer.close()


def main():
    parser = argparse.ArgumentParser(
        description="FLUX 训练 (TensorBoard + 文生图验证, 分片 ckpt 保存)"
    )

    parser.add_argument(
        "--local-rank", "--local_rank", type=int, default=os.getenv("LOCAL_RANK", 0)
    )

    # 数据参数
    parser.add_argument(
        "--meta_folder",
        type=str,
        required=True,
        help="预处理后的 .meta 文件夹",
    )

    # 模型参数
    parser.add_argument(
        "--model_id",
        type=str,
        default="black-forest-labs/FLUX.1-dev",
        help="FLUX 模型路径",
    )

    # 训练参数
    parser.add_argument("--batch_size_per_gpu", type=int, default=4)
    parser.add_argument("--num_epochs", type=int, default=50)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)

    # Flow Matching 参数
    parser.add_argument("--logit_mean", type=float, default=0.0)
    parser.add_argument("--logit_std", type=float, default=1.0)
    parser.add_argument("--flow_shift", type=float, default=3.0)
    parser.add_argument("--mix_uniform_ratio", type=float, default=0.1)
    parser.add_argument("--sigma_min", type=float, default=0.0)
    parser.add_argument("--sigma_max", type=float, default=1.0)
    parser.add_argument("--num_train_timesteps", type=int, default=1000)

    # DataLoader 参数
    parser.add_argument("--num_workers", type=int, default=4)

    # 输出参数
    parser.add_argument(
        "--output_dir", type=str, default="./outputs/flux_training"
    )
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--save_every", type=int, default=10)

    # 验证推理参数（当前未启用）
    parser.add_argument(
        "--validate_every",
        type=int,
        default=0,
        help="多少个 iteration 做一次文生图验证（0 表示关闭）",
    )
    parser.add_argument(
        "--val_num_inference_steps",
        type=int,
        default=28,
        help="验证推理采样步数",
    )
    parser.add_argument(
        "--val_guidance_scale",
        type=float,
        default=3.5,
        help="验证推理 guidance scale",
    )
    parser.add_argument("--val_height", type=int, default=512)
    parser.add_argument("--val_width", type=int, default=512)

    args = parser.parse_args()

    trainer = FluxTrainer(args)
    try:
        trainer.train()
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()