#!/usr/bin/env python3
"""
FLUX 图像预处理脚本
将图像处理成训练所需的 .meta 文件格式

用法:
python preprocess_flux_images.py \
    --input_folder ./data/raw_data \
    --output_folder ./data/processed_meta \
    --caption_file ./data/captions.json \
    --height 512 \
    --width 512 \
    --max_samples 3  # ✅ 新增：只处理 3 张图像

Caption 文件格式支持：
1. JSON 格式（推荐）:
   [
     {"file_name": "image1.jpg", "caption": "a dog"},
     {"file_name": "image2.jpg", "caption": "a cat"}
   ]

2. TXT 格式（兼容）:
   - filename|caption
   - filename\tcaption
   - 每行一个 caption（按文件名排序）

参考: WAN2.1 的 preprocess_resize.py
修改: 视频处理 → 图像处理, 5D → 4D
"""

import os
import argparse
import pickle
import torch
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import logging
import random  # ✅ 新增：用于随机采样

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FluxImagePreprocessor:
    """FLUX 图像预处理器"""
    
    def __init__(
        self,
        model_id: str = "black-forest-labs/FLUX.1-dev",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        height: int = 512,
        width: int = 512,
    ):
        """
        初始化预处理器
        
        Args:
            model_id: FLUX 模型 ID
            device: 运行设备
            height: 目标高度
            width: 目标宽度
        """
        self.model_id = model_id
        self.device = device
        self.height = height
        self.width = width
        
        logger.info(f"初始化 FLUX 预处理器...")
        logger.info(f"  模型: {model_id}")
        logger.info(f"  设备: {device}")
        logger.info(f"  分辨率: {height}x{width}")
        
        # 加载模型组件
        self._load_models()
        
    def _load_models(self):
        """加载 FLUX 的所有模型组件"""
        from diffusers import FluxPipeline
        
        logger.info("加载 FLUX 模型组件（首次运行会下载，约 24GB）...")
        
        try:
            # 加载完整的 pipeline（包含所有组件）
            pipeline = FluxPipeline.from_pretrained(
                self.model_id,
                torch_dtype=torch.bfloat16
            )
            
            # 提取各个组件
            self.vae = pipeline.vae.to(self.device)
            self.text_encoder = pipeline.text_encoder.to(self.device)  # CLIP
            self.text_encoder_2 = pipeline.text_encoder_2.to(self.device)  # T5
            self.tokenizer = pipeline.tokenizer
            self.tokenizer_2 = pipeline.tokenizer_2
            
            # 设置为评估模式
            self.vae.eval()
            self.text_encoder.eval()
            self.text_encoder_2.eval()
            
            logger.info("✓ VAE 加载成功")
            logger.info("✓ CLIP Text Encoder 加载成功")
            logger.info("✓ T5 Text Encoder 加载成功")
            
            # 记录 VAE 配置
            logger.info(f"VAE 配置:")
            logger.info(f"  scaling_factor: {self.vae.config.scaling_factor}")
            logger.info(f"  shift_factor: {self.vae.config.shift_factor}")
            
        except Exception as e:
            logger.error(f"加载模型失败: {e}")
            raise
    
    def load_image(self, image_path: str) -> torch.Tensor:
        """
        加载并预处理图像
        
        Args:
            image_path: 图像路径
            
        Returns:
            tensor: (1, C, H, W) 格式的图像 tensor
        """
        # 加载图像
        image = Image.open(image_path).convert("RGB")
        
        # Resize 到目标尺寸
        image = image.resize((self.width, self.height), Image.LANCZOS)
        
        # 转为 numpy array
        image = np.array(image).astype(np.float32)
        
        # 归一化到 [-1, 1]
        image = image / 255.0
        image = (image - 0.5) / 0.5
        
        # 转为 tensor: (H, W, C) -> (C, H, W)
        image = torch.from_numpy(image).permute(2, 0, 1)
        
        # 添加 batch 维度: (C, H, W) -> (1, C, H, W)
        image = image.unsqueeze(0)
        
        return image
    
    def encode_image(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """
        使用 VAE 编码图像
        
        完全对齐官方 FluxPipeline 的解码逻辑（Line 868-870）
        
        官方解码: latents = latents / scale + shift → vae.decode
        所以编码: vae.encode → (latents - shift) * scale
        
        FLUX VAE 默认配置:
        - shift_factor: 0.1159
        - scaling_factor: 0.3611
        """
        with torch.no_grad():
            image_tensor = image_tensor.to(device=self.device, dtype=torch.bfloat16)
            
            # Step 1: VAE 编码
            posterior = self.vae.encode(image_tensor).latent_dist
            latents = posterior.sample()
            
            # Step 2: 归一化（完全对齐官方 pipeline_flux.py Line 868）
            # 官方解码用: latents / scale + shift
            # 所以编码用: (latents - shift) * scale
            latents = (latents - self.vae.config.shift_factor) * self.vae.config.scaling_factor
            
            # 日志验证（仅打印一次）
            if not hasattr(self, '_logged_vae_config'):
                logger.info("=" * 70)
                logger.info("VAE 编码配置（对齐官方 FluxPipeline）:")
                logger.info(f"  shift_factor: {self.vae.config.shift_factor}")
                logger.info(f"  scaling_factor: {self.vae.config.scaling_factor}")
                logger.info(f"  官方解码公式: latents / {self.vae.config.scaling_factor} + {self.vae.config.shift_factor}")
                logger.info(f"  当前编码公式: (latents - {self.vae.config.shift_factor}) * {self.vae.config.scaling_factor}")
                logger.info(f"  编码后 latents 统计:")
                logger.info(f"    范围: [{latents.min().item():.4f}, {latents.max().item():.4f}]")
                logger.info(f"    均值: {latents.mean().item():.4f}")
                logger.info(f"    标准差: {latents.std().item():.4f}")
                logger.info("=" * 70)
                self._logged_vae_config = True
            
            latents = latents.cpu()
        return latents
    
    def encode_text(self, caption: str) -> tuple:
        """
        使用 CLIP 和 T5 编码文本
        
        Args:
            caption: 文本描述
            
        Returns:
            (prompt_embeds, pooled_prompt_embeds): T5 和 CLIP 的 embeddings
        """
        with torch.no_grad():
            # 1. CLIP 编码（用于 pooled embeddings）
            clip_inputs = self.tokenizer(
                caption,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt"
            )
            
            clip_outputs = self.text_encoder(
                clip_inputs.input_ids.to(self.device),
                output_hidden_states=True
            )
            
            # CLIP 的 pooled output（使用最后一层的 [CLS] token）
            pooled_prompt_embeds = clip_outputs.pooler_output.cpu()
            
            # 2. T5 编码（用于 sequence embeddings）
            t5_inputs = self.tokenizer_2(
                caption,
                padding="max_length",
                max_length=self.tokenizer_2.model_max_length,
                truncation=True,
                return_tensors="pt"
            )
            
            t5_outputs = self.text_encoder_2(
                t5_inputs.input_ids.to(self.device),
                output_hidden_states=False
            )
            
            # T5 的 sequence output
            prompt_embeds = t5_outputs.last_hidden_state.cpu()
            
        return prompt_embeds, pooled_prompt_embeds
    
    def process_single_image(
        self,
        image_path: str,
        caption: str,
        output_path: str
    ):
        """
        处理单张图像
        
        Args:
            image_path: 图像路径
            caption: 文本描述
            output_path: 输出 .meta 文件路径
        """
        logger.info(f"处理: {os.path.basename(image_path)}")
        
        try:
            # 1. 加载图像
            logger.info("  [1/3] 加载图像...")
            image_tensor = self.load_image(image_path)
            logger.info(f"        图像 shape: {image_tensor.shape}")
            
            # 2. 编码图像
            logger.info("  [2/3] VAE 编码...")
            latents = self.encode_image(image_tensor)
            logger.info(f"        Latents shape: {latents.shape}")
            
            # 3. 编码文本
            logger.info("  [3/3] Text 编码...")
            prompt_embeds, pooled_prompt_embeds = self.encode_text(caption)
            logger.info(f"        Prompt embeds shape: {prompt_embeds.shape}")
            logger.info(f"        Pooled embeds shape: {pooled_prompt_embeds.shape}")
            
            # 4. 保存 .meta 文件
            data = {
                "prompt_embeds": prompt_embeds,
                "pooled_prompt_embeds": pooled_prompt_embeds,
                "model_input_512": latents,  # 固定名称，方便后续使用
                "metadata": {
                    "image_path": image_path,
                    "caption": caption,
                    "height": self.height,
                    "width": self.width,
                    "original_shape": image_tensor.shape,
                    "latent_shape": latents.shape,
                }
            }
            
            with open(output_path, 'wb') as f:
                pickle.dump(data, f)
            
            logger.info(f"  ✓ 保存到: {output_path}")
            
        except Exception as e:
            logger.error(f"  ✗ 处理失败: {e}")
            raise
    
    def process_folder(
        self,
        input_folder: str,
        output_folder: str,
        caption_file: str = None,
        default_caption: str = "a photo",
        max_samples: int = None,  # ✅ 新增参数
    ):
        """
        处理整个文件夹的图像
        
        Args:
            input_folder: 输入图像文件夹
            output_folder: 输出 .meta 文件夹
            caption_file: caption 文件路径
                - JSON 格式: 只处理 JSON 中指定的图像
                - TXT 格式: 处理文件夹中所有图像
            default_caption: 默认 caption
            max_samples: 最大处理样本数（None 表示全部处理）
        """
        input_folder = Path(input_folder)
        output_folder = Path(output_folder)
        output_folder.mkdir(parents=True, exist_ok=True)
        
        # 加载 captions
        captions = {}
        image_files_to_process = []
        
        if caption_file and os.path.exists(caption_file):
            logger.info(f"加载 captions from {caption_file}")
            
            # 检测文件格式
            if caption_file.endswith('.json'):
                # JSON 格式：只处理 JSON 中指定的图像
                import json
                logger.info("检测到格式: JSON")
                with open(caption_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # ✅ 新增：如果指定了 max_samples，随机采样
                if max_samples is not None and len(data) > max_samples:
                    logger.info(f"从 {len(data)} 个样本中随机采样 {max_samples} 个")
                    data = random.sample(data, max_samples)
                
                # 从 JSON 构建要处理的文件列表
                for entry in data:
                    filename = entry.get('file_name', entry.get('filename'))
                    caption = entry.get('caption', '')
                    if filename and caption:
                        captions[filename] = caption
                        # 构建完整路径
                        image_path = input_folder / filename
                        if image_path.exists():
                            image_files_to_process.append(image_path)
                        else:
                            logger.warning(f"图像不存在: {image_path}")
                
                logger.info(f"JSON 中指定了 {len(data)} 个图像")
                logger.info(f"实际找到 {len(image_files_to_process)} 个图像")
                
            else:
                # TXT 格式：处理文件夹中所有图像（兼容旧版本）
                # 支持的图像格式
                image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
                
                # 找到所有图像文件
                image_files_to_process = [
                    f for f in input_folder.iterdir()
                    if f.suffix.lower() in image_extensions
                ]
                
                # 排序
                image_files_to_process = sorted(image_files_to_process)
                
                # ✅ 新增：如果指定了 max_samples，采样
                if max_samples is not None and len(image_files_to_process) > max_samples:
                    logger.info(f"从 {len(image_files_to_process)} 个图像中随机采样 {max_samples} 个")
                    image_files_to_process = random.sample(image_files_to_process, max_samples)
                
                with open(caption_file, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                
                # 检测格式
                if '|' in lines[0]:
                    # 格式: filename|caption
                    logger.info("检测到格式: filename|caption")
                    for line in lines:
                        parts = line.strip().split('|', 1)
                        if len(parts) == 2:
                            filename, caption = parts
                            captions[filename] = caption
                elif '\t' in lines[0]:
                    # 格式: filename\tcaption
                    logger.info("检测到格式: filename\\tcaption")
                    for line in lines:
                        parts = line.strip().split('\t')
                        if len(parts) == 2:
                            filename, caption = parts
                            captions[filename] = caption
                else:
                    # 格式: 每行一个 caption
                    logger.info("检测到格式: 每行一个 caption")
                    for i, line in enumerate(lines):
                        if i < len(image_files_to_process):
                            captions[image_files_to_process[i].name] = line.strip()
        else:
            # 没有 caption 文件：处理所有图像
            image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
            image_files_to_process = [
                f for f in input_folder.iterdir()
                if f.suffix.lower() in image_extensions
            ]
            image_files_to_process = sorted(image_files_to_process)
            
            # ✅ 新增：如果指定了 max_samples，采样
            if max_samples is not None and len(image_files_to_process) > max_samples:
                logger.info(f"从 {len(image_files_to_process)} 个图像中随机采样 {max_samples} 个")
                image_files_to_process = random.sample(image_files_to_process, max_samples)
        
        if not image_files_to_process:
            logger.warning(f"没有要处理的图像文件!")
            return
        
        logger.info(f"准备处理 {len(image_files_to_process)} 张图像")
        
        # 处理每张图像
        processed = 0
        for image_file in tqdm(image_files_to_process, desc="处理图像"):
            # 生成输出文件名
            output_file = output_folder / f"{image_file.stem}.meta"
            
            # 跳过已处理的文件
            if output_file.exists():
                logger.debug(f"跳过已存在: {output_file.name}")
                continue
            
            # 获取 caption
            caption = captions.get(image_file.name, default_caption)
            
            # 处理图像
            try:
                self.process_single_image(
                    str(image_file),
                    caption,
                    str(output_file)
                )
                processed += 1
            except Exception as e:
                logger.error(f"处理 {image_file} 失败: {e}")
                continue
        
        logger.info(f"✓ 完成! 成功处理了 {processed}/{len(image_files_to_process)} 张图像")
        logger.info(f"✓ 输出目录: {output_folder}")


def main():
    parser = argparse.ArgumentParser(description="FLUX 图像预处理")
    
    parser.add_argument(
        "--input_folder",
        type=str,
        required=True,
        help="输入图像文件夹路径"
    )
    
    parser.add_argument(
        "--output_folder",
        type=str,
        required=True,
        help="输出 .meta 文件夹路径"
    )
    
    parser.add_argument(
        "--caption_file",
        type=str,
        default=None,
        help="Caption 文件路径（每行一个 caption，对应一张图像）"
    )
    
    parser.add_argument(
        "--default_caption",
        type=str,
        default="a photo",
        help="默认 caption（如果没有提供 caption_file）"
    )
    
    parser.add_argument(
        "--height",
        type=int,
        default=512,
        help="目标高度"
    )
    
    parser.add_argument(
        "--width",
        type=int,
        default=512,
        help="目标宽度"
    )
    
    parser.add_argument(
        "--model_id",
        type=str,
        default="black-forest-labs/FLUX.1-dev",
        help="FLUX 模型 ID"
    )
    
    # ✅ 新增参数
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="最大处理样本数（默认处理全部）"
    )
    
    args = parser.parse_args()
    
    # 创建预处理器
    preprocessor = FluxImagePreprocessor(
        model_id=args.model_id,
        height=args.height,
        width=args.width
    )
    
    # 处理文件夹
    preprocessor.process_folder(
        input_folder=args.input_folder,
        output_folder=args.output_folder,
        caption_file=args.caption_file,
        default_caption=args.default_caption,
        max_samples=args.max_samples,  # ✅ 传入新参数
    )


if __name__ == "__main__":
    main()