#!/usr/bin/env python3
"""
FLUX DataLoader - 加载预处理后的 .meta 文件

参考: WAN2.1 的 dataloader.py
修改: 5D 视频数据 → 4D 图像数据
"""

import os
import pickle
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FluxMetaDataset(Dataset):
    """
    FLUX 训练数据集
    
    加载预处理后的 .meta 文件，每个文件包含：
    - prompt_embeds: T5 编码的文本 embeddings
    - pooled_prompt_embeds: CLIP pooled embeddings  
    - model_input_512: VAE 编码的 latents (4D)
    - metadata: 元数据信息
    """
    
    def __init__(
        self,
        meta_folder: str,
        max_samples: int = None,
    ):
        """
        初始化数据集
        
        Args:
            meta_folder: .meta 文件所在文件夹
            max_samples: 最大样本数（用于快速测试）
        """
        self.meta_folder = Path(meta_folder)
        
        if not self.meta_folder.exists():
            raise ValueError(f"Meta folder not found: {meta_folder}")
        
        # 找到所有 .meta 文件
        self.meta_files = sorted(list(self.meta_folder.glob("*.meta")))
        
        if len(self.meta_files) == 0:
            raise ValueError(f"No .meta files found in {meta_folder}")
        
        # 限制样本数（用于测试）
        if max_samples is not None:
            self.meta_files = self.meta_files[:max_samples]
        
        logger.info(f"Found {len(self.meta_files)} .meta files in {meta_folder}")
    
    def __len__(self):
        return len(self.meta_files)
    
    def __getitem__(self, idx):
        """
        加载单个样本
        
        Returns:
            dict: 包含所有训练需要的数据
                - prompt_embeds: (1, seq_len, hidden_dim) 
                - pooled_prompt_embeds: (1, hidden_dim)
                - latents: (1, C, H, W) - 4D 图像 latents
                - metadata: dict
        """
        meta_file = self.meta_files[idx]
        
        try:
            with open(meta_file, 'rb') as f:
                data = pickle.load(f)
            
            # 验证数据
            required_keys = ['prompt_embeds', 'pooled_prompt_embeds', 'model_input_512']
            for key in required_keys:
                if key not in data:
                    raise ValueError(f"Missing key '{key}' in {meta_file}")
            
            # 提取数据
            sample = {
                'prompt_embeds': data['prompt_embeds'],           # (1, seq_len, dim)
                'pooled_prompt_embeds': data['pooled_prompt_embeds'],  # (1, dim)
                'latents': data['model_input_512'],              # (1, C, H, W)
                'metadata': data.get('metadata', {}),
            }
            
            # 验证形状
            assert sample['latents'].dim() == 4, f"Expected 4D latents, got {sample['latents'].dim()}D"
            
            return sample
            
        except Exception as e:
            logger.error(f"Error loading {meta_file}: {e}")
            raise
    
    def get_sample_info(self, idx):
        """获取样本信息（用于调试）"""
        sample = self[idx]
        info = {
            'file': self.meta_files[idx].name,
            'prompt_embeds_shape': tuple(sample['prompt_embeds'].shape),
            'pooled_embeds_shape': tuple(sample['pooled_prompt_embeds'].shape),
            'latents_shape': tuple(sample['latents'].shape),
            'caption': sample['metadata'].get('caption', 'N/A'),
        }
        return info


def collate_fn(batch):
    """
    自定义 collate 函数
    
    由于 FLUX 是固定分辨率 (512x512)，所有样本形状相同，
    可以直接 stack
    
    Args:
        batch: list of samples from dataset
        
    Returns:
        dict: batched data
            - prompt_embeds: (B, seq_len, hidden_dim)
            - pooled_prompt_embeds: (B, hidden_dim)  
            - latents: (B, C, H, W)
            - metadata: list of metadata dicts
    """
    # 检查 batch 是否为空
    if len(batch) == 0:
        raise ValueError("Empty batch")
    
    # Stack 所有样本
    # 注意：需要去掉第一个维度 (batch 中每个样本的 shape 是 (1, ...))
    batched = {
        'prompt_embeds': torch.cat([s['prompt_embeds'] for s in batch], dim=0),
        'pooled_prompt_embeds': torch.cat([s['pooled_prompt_embeds'] for s in batch], dim=0),
        'latents': torch.cat([s['latents'] for s in batch], dim=0),
        'metadata': [s['metadata'] for s in batch],
    }
    
    return batched


def create_dataloader(
    meta_folder: str,
    batch_size: int = 4,
    num_workers: int = 4,
    shuffle: bool = True,
    max_samples: int = None,
    pin_memory: bool = True,
):
    """
    创建 DataLoader
    
    Args:
        meta_folder: .meta 文件文件夹
        batch_size: batch 大小
        num_workers: 数据加载线程数
        shuffle: 是否打乱
        max_samples: 最大样本数（测试用）
        pin_memory: 是否使用 pin memory（GPU 训练推荐）
        
    Returns:
        DataLoader
    """
    dataset = FluxMetaDataset(
        meta_folder=meta_folder,
        max_samples=max_samples,
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
        drop_last=True,  # 丢弃最后不完整的 batch
    )
    
    logger.info(f"Created DataLoader:")
    logger.info(f"  Total samples: {len(dataset)}")
    logger.info(f"  Batch size: {batch_size}")
    logger.info(f"  Num batches: {len(dataloader)}")
    logger.info(f"  Num workers: {num_workers}")
    
    return dataloader


# ===================== 测试代码 =====================

def test_dataloader(meta_folder: str, batch_size: int = 2):
    """
    测试 DataLoader
    
    Args:
        meta_folder: .meta 文件文件夹
        batch_size: batch 大小
    """
    print("=" * 70)
    print("测试 FLUX DataLoader")
    print("=" * 70)
    
    # 创建 DataLoader
    dataloader = create_dataloader(
        meta_folder=meta_folder,
        batch_size=batch_size,
        num_workers=0,  # 测试时用 0，避免多进程问题
        shuffle=False,
        max_samples=10,  # 只测试 10 个样本
    )
    
    # 测试第一个 batch
    print("\n测试第一个 batch:")
    print("-" * 70)
    
    for batch_idx, batch in enumerate(dataloader):
        print(f"\nBatch {batch_idx + 1}:")
        print(f"  prompt_embeds shape: {batch['prompt_embeds'].shape}")
        print(f"  pooled_embeds shape: {batch['pooled_prompt_embeds'].shape}")
        print(f"  latents shape: {batch['latents'].shape}")
        print(f"  num samples: {len(batch['metadata'])}")
        
        # 显示每个样本的 caption
        for i, meta in enumerate(batch['metadata']):
            caption = meta.get('caption', 'N/A')
            print(f"  Sample {i + 1} caption: {caption}")
        
        # 只测试第一个 batch
        if batch_idx == 0:
            break
    
    print("\n" + "=" * 70)
    print("✅ DataLoader 测试通过!")
    print("=" * 70)
    
    # 验证形状
    expected_shapes = {
        'prompt_embeds': (batch_size, 512, 4096),  # T5 embeddings
        'pooled_prompt_embeds': (batch_size, 768),  # CLIP pooled
        'latents': (batch_size, 16, 64, 64),        # VAE latents (512x512 → 64x64)
    }
    
    print("\n验证数据形状:")
    print("-" * 70)
    for key, expected_shape in expected_shapes.items():
        actual_shape = tuple(batch[key].shape)
        match = "✓" if actual_shape == expected_shape else "✗"
        print(f"{match} {key}: {actual_shape} (期望: {expected_shape})")
    
    return dataloader


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="测试 FLUX DataLoader")
    parser.add_argument(
        "--meta_folder",
        type=str,
        required=True,
        help=".meta 文件夹路径"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=2,
        help="batch 大小"
    )
    
    args = parser.parse_args()
    
    test_dataloader(args.meta_folder, args.batch_size)