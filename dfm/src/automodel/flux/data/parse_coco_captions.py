#!/usr/bin/env python3
"""
解析 MSCOCO captions_train2017.json 生成 caption 文件

输入：captions_train2017.json
输出：captions.txt (每行: filename|caption)
"""

import json
import argparse
from pathlib import Path
from tqdm import tqdm


def parse_coco_captions(
    annotation_file,
    image_folder,
    output_file,
    max_samples=None,
    output_format='json',  # 'json' or 'txt'
):
    """
    解析 COCO captions
    
    Args:
        annotation_file: captions_train2017.json 路径
        image_folder: 图像文件夹路径
        output_file: 输出文件路径（.json 或 .txt）
        max_samples: 限制样本数（测试用）
        output_format: 输出格式 'json' 或 'txt'
    """
    print("=" * 70)
    print("解析 MSCOCO Captions")
    print("=" * 70)
    
    # 读取 annotations
    print(f"读取: {annotation_file}")
    with open(annotation_file, 'r') as f:
        data = json.load(f)
    
    # 提取图像信息
    images = {img['id']: img for img in data['images']}
    print(f"✓ 找到 {len(images)} 张图像")
    
    # 提取 captions
    print(f"解析 captions...")
    image_captions = {}
    
    for anno in tqdm(data['annotations']):
        img_id = anno['image_id']
        caption = anno['caption']
        
        if img_id not in image_captions:
            image_captions[img_id] = []
        image_captions[img_id].append(caption)
    
    print(f"✓ {len(image_captions)} 张图像有 caption")
    
    # 生成数据
    print(f"\n生成数据...")
    dataset = []
    
    image_folder = Path(image_folder)
    count = 0
    
    for img_id, captions in tqdm(image_captions.items()):
        if max_samples and count >= max_samples:
            break
        
        img_info = images[img_id]
        img_filename = img_info['file_name']
        img_path = image_folder / img_filename
        
        # 检查图像是否存在
        if not img_path.exists():
            continue
        
        # 构建数据条目
        entry = {
            "image_id": img_id,
            "file_name": img_filename,
            "image_path": str(img_path),
            "caption": captions[0].strip(),  # 使用第一个 caption
            "all_captions": [c.strip() for c in captions],  # 保存所有 5 个 captions
            "width": img_info.get('width', 0),
            "height": img_info.get('height', 0),
        }
        
        dataset.append(entry)
        count += 1
    
    # 保存
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    if output_format == 'json':
        # JSON 格式（推荐）
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, indent=2, ensure_ascii=False)
        print(f"\n✅ 保存为 JSON 格式")
    else:
        # TXT 格式（兼容）
        lines = [f"{entry['file_name']}|{entry['caption']}" for entry in dataset]
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))
        print(f"\n✅ 保存为 TXT 格式")
    
    print(f"输出文件: {output_file}")
    print(f"样本数: {len(dataset)}")
    
    return output_file, len(dataset)


def main():
    parser = argparse.ArgumentParser(description="解析 MSCOCO captions")
    
    parser.add_argument(
        "--annotation_file",
        type=str,
        required=True,
        help="captions_train2017.json 路径"
    )
    
    parser.add_argument(
        "--image_folder",
        type=str,
        required=True,
        help="图像文件夹路径"
    )
    
    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="输出文件路径（.json 或 .txt）"
    )
    
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="最大样本数（测试用）"
    )
    
    parser.add_argument(
        "--format",
        type=str,
        default="json",
        choices=["json", "txt"],
        help="输出格式（默认: json）"
    )
    
    args = parser.parse_args()
    
    # 解析
    caption_file, num_samples = parse_coco_captions(
        annotation_file=args.annotation_file,
        image_folder=args.image_folder,
        output_file=args.output_file,
        max_samples=args.max_samples,
        output_format=args.format,
    )
    
    print("\n" + "=" * 70)
    print("下一步：运行预处理")
    print("=" * 70)
    print(f"python scripts/preprocess_flux_images.py \\")
    print(f"    --input_folder {args.image_folder} \\")
    print(f"    --output_folder ./data/processed_meta \\")
    print(f"    --caption_file {caption_file} \\")
    print(f"    --height 512 --width 512 \\")
    print(f"    --model_id ./models/FLUX.1-dev")
    print("=" * 70)


if __name__ == "__main__":
    main()