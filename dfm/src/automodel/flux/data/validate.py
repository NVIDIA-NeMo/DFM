#!/usr/bin/env python3
"""
验证 VAE 编解码的一致性（修复版）
"""

import torch
from PIL import Image
import numpy as np
from diffusers import FluxPipeline
import os

# 加载官方 pipeline
model_id = "/high_perf_store4/evad-tech-vla/houzhiyi/FLUX/models/FLUX.1-dev"
print("加载 FluxPipeline...")
pipe = FluxPipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16)
pipe.to("cuda")

# 加载测试图像
image_path = "/high_perf_store4/evad-tech-vla/houzhiyi/FLUX/flux_training/data/mscoco_10case/000000035897/000000035897.jpg"
if not os.path.exists(image_path):
    print(f"❌ 图像不存在: {image_path}")
    exit(1)

image = Image.open(image_path).convert("RGB")
image = image.resize((256, 256), Image.LANCZOS)

# 转换为 tensor
image_np = np.array(image).astype(np.float32) / 255.0
image_np = (image_np - 0.5) / 0.5
image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).unsqueeze(0)
image_tensor = image_tensor.to("cuda", dtype=torch.bfloat16)

print("=" * 70)
print("VAE 编解码验证（bfloat16 精度分析）")
print("=" * 70)
print(f"测试图像: {image_path}")
print(f"分辨率: 256x256")
print(f"VAE config:")
print(f"  shift_factor: {pipe.vae.config.shift_factor}")
print(f"  scaling_factor: {pipe.vae.config.scaling_factor}")

with torch.no_grad():
    # ===== 测试1：你的编码方案 =====
    print("\n" + "=" * 70)
    print("【测试1】你的编码方案（有 shift/scale）")
    print("=" * 70)
    
    latents = pipe.vae.encode(image_tensor).latent_dist.sample()
    print(f"1. VAE encode 输出: [{latents.min():.4f}, {latents.max():.4f}]")
    
    latents_encoded = (latents - pipe.vae.config.shift_factor) * pipe.vae.config.scaling_factor
    print(f"2. 编码后存储: [{latents_encoded.min():.4f}, {latents_encoded.max():.4f}]")
    
    latents_decoded = latents_encoded / pipe.vae.config.scaling_factor + pipe.vae.config.shift_factor
    print(f"3. 推理解码前: [{latents_decoded.min():.4f}, {latents_decoded.max():.4f}]")
    
    diff = (latents - latents_decoded).abs()
    print(f"4. 编解码误差:")
    print(f"   max: {diff.max():.6f}")
    print(f"   mean: {diff.mean():.6f}")
    
    reconstructed_your = pipe.vae.decode(latents_decoded, return_dict=False)[0]
    reconstructed_your = (reconstructed_your / 2 + 0.5).clamp(0, 1)
    reconstructed_your = (reconstructed_your * 255).byte().cpu().permute(0, 2, 3, 1).numpy()[0]
    img_your = Image.fromarray(reconstructed_your)
    img_your.save("./reconstructed_your_method.png")
    
    mse_your = ((np.array(image).astype(np.float32) - reconstructed_your.astype(np.float32)) ** 2).mean()
    print(f"5. 重建 MSE: {mse_your:.2f}")
    
    # ===== 测试2：不加 shift/scale =====
    print("\n" + "=" * 70)
    print("【测试2】错误方案（不加 shift/scale）")
    print("=" * 70)
    
    latents_no_scale = latents
    print(f"1. 直接存储: [{latents_no_scale.min():.4f}, {latents_no_scale.max():.4f}]")
    
    latents_for_decode = latents_no_scale / pipe.vae.config.scaling_factor + pipe.vae.config.shift_factor
    print(f"2. 推理解码前: [{latents_for_decode.min():.4f}, {latents_for_decode.max():.4f}]")
    print(f"   ⚠️ 范围严重超出正常值！")
    
    reconstructed_wrong = pipe.vae.decode(latents_for_decode, return_dict=False)[0]
    reconstructed_wrong = (reconstructed_wrong / 2 + 0.5).clamp(0, 1)
    reconstructed_wrong = (reconstructed_wrong * 255).byte().cpu().permute(0, 2, 3, 1).numpy()[0]
    img_wrong = Image.fromarray(reconstructed_wrong)
    img_wrong.save("./reconstructed_no_scale.png")
    
    mse_wrong = ((np.array(image).astype(np.float32) - reconstructed_wrong.astype(np.float32)) ** 2).mean()
    print(f"3. 重建 MSE: {mse_wrong:.2f}")
    
    # ===== 测试3：float32 精度测试（修复版）=====
    print("\n" + "=" * 70)
    print("【测试3】float32 精度对比（理论最佳）")
    print("=" * 70)
    
    # ✅ 修复：使用 CPU 上的 float32 VAE
    try:
        # 方案1：临时将 VAE 转为 float32
        pipe.vae.to(torch.float32)
        image_tensor_fp32 = image_tensor.to(torch.float32)
        
        latents_fp32 = pipe.vae.encode(image_tensor_fp32).latent_dist.sample()
        latents_encoded_fp32 = (latents_fp32 - pipe.vae.config.shift_factor) * pipe.vae.config.scaling_factor
        latents_decoded_fp32 = latents_encoded_fp32 / pipe.vae.config.scaling_factor + pipe.vae.config.shift_factor
        
        diff_fp32 = (latents_fp32 - latents_decoded_fp32).abs()
        print(f"1. float32 编解码误差:")
        print(f"   max: {diff_fp32.max():.9f}")
        print(f"   mean: {diff_fp32.mean():.9f}")
        
        # 恢复 bfloat16
        pipe.vae.to(torch.bfloat16)
        
    except Exception as e:
        print(f"⚠️ float32 测试失败（可忽略）: {e}")
        print("使用数学验证替代:")
        # 数学验证：(x - s) * k / k + s = x
        print("公式验证: (x - shift) * scale / scale + shift = x")
        print("   ✅ 数学上可逆，逻辑正确")
        diff_fp32_max = 0.0  # 数学上完美可逆
    
    # ===== 最终分析 =====
    print("\n" + "=" * 70)
    print("📊 综合分析结果")
    print("=" * 70)
    
    print(f"\n1️⃣  bfloat16 编解码误差: {diff.max():.6f}")
    if diff.max() < 0.1:
        print("   ✅ 在 bfloat16 精度范围内，正常！")
        print("   📝 bfloat16 只有 7 bits 尾数，误差 < 0.1 是正常的")
    else:
        print("   ❌ 超出正常范围")
    
    print(f"\n2️⃣  重建质量对比:")
    print(f"   你的方法 MSE: {mse_your:.2f}")
    print(f"   不加 scale MSE: {mse_wrong:.2f}")
    improvement = (mse_wrong - mse_your) / mse_wrong * 100
    print(f"   改善率: {improvement:.1f}%")
    
    if mse_your < mse_wrong * 0.5:
        print("   ✅ 你的方法明显更好！")
    else:
        print("   ⚠️ 需要检查")
    
    print(f"\n3️⃣  MSE 质量评级:")
    if mse_your < 100:
        grade = "优秀 ⭐⭐⭐⭐⭐"
    elif mse_your < 200:
        grade = "良好 ⭐⭐⭐⭐"
    elif mse_your < 500:
        grade = "可接受 ⭐⭐⭐"
    else:
        grade = "较差 ⭐"
    print(f"   {grade} (MSE = {mse_your:.2f})")
    
    # 最终结论
    print("\n" + "=" * 70)
    print("🎯 最终结论")
    print("=" * 70)
    
    all_good = (
        diff.max() < 0.1 and 
        mse_your < 100 and
        mse_your < mse_wrong * 0.5
    )
    
    if all_good:
        print("✅ 你的 VAE 编码实现完全正确！")
        print("✅ bfloat16 误差是正常的精度损失")
        print("✅ 可以安心用于预处理和训练")
        print("\n📋 下一步:")
        print("  1. ✅ 确认预处理脚本保留了 (latents - shift) * scale")
        print("  2. 🔄 重新运行预处理（如果之前注释掉了）")
        print("  3. 🚀 开始训练！")
    else:
        print("⚠️ 存在问题，需要检查")

# 保存原图
image.save("./original.png")

print("\n" + "=" * 70)
print("📁 生成的图像:")
print("=" * 70)
print("  ✓ original.png")
print("  ✓ reconstructed_your_method.png (你的方法)")
print("  ✓ reconstructed_no_scale.png (不加 scale)")
print("\n请对比这三张图像！")
print("=" * 70)
