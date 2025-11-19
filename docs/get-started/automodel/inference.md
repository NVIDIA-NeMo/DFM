---
description: "Generate videos from fine-tuned Auto model checkpoints"
categories: ["getting-started", "automodel"]
tags: ["inference", "generation", "how-to"]
personas: ["data-scientist-focused", "mle-focused"]
difficulty: "beginner"
content_type: "how-to"
---

(gs-automodel-inference)=

# Generate Videos with Automodel

Generate videos from your fine-tuned WAN2.1 checkpoint or use pretrained models from Hugging Face.

## Goal

Generate high-quality videos from text prompts using your trained model.

**Time**: 5-10 minutes per video

## Prerequisites

- ✅ Complete [Installation](../installation.md)
- ✅ Either:
  - Fine-tuned checkpoint from [training](training.md), OR
  - Pretrained Hugging Face model (`Wan-AI/Wan2.2-T2V-A14B-Diffusers`)
- ✅ GPU with sufficient memory (16GB+ recommended)

## Overview

**What happens during inference**:
1. Load model (from checkpoint or Hugging Face)
2. Configure distributed parallelism (optional)
3. Generate video from text prompt
4. Save video file

**Generation time**: 2-5 minutes per video (single GPU), faster with parallelism

## Step 1: Generate from Pretrained Model

Start with a pretrained model to verify your setup.

### Single GPU Generation

Generate a video using default settings:

```bash
python dfm/examples/automodel/generate/wan_generate.py \
    --prompt "A butterfly flying over colorful flowers in a garden" \
    --height 480 \
    --width 848 \
    --num-frames 111 \
    --output butterfly_garden.mp4
```

**What this does**:
1. Downloads `Wan-AI/Wan2.2-T2V-A14B-Diffusers` from Hugging Face (if not cached)
2. Generates 111 frames at 480×848 resolution
3. Saves video to `butterfly_garden.mp4`

**Expected output**:

```text
[Loading] Loading VAE and pipeline...
[Setup] Pipeline loaded and parallelized via NeMoAutoDiffusionPipeline
[Inference] Starting distributed inference...
[Inference] Saved butterfly_garden.mp4
[Complete] Automodel FSDP2 inference completed!
```

**Output file**:
- Filename: `butterfly_garden.mp4`
- Size: 5-15 MB
- Duration: ~4.6 seconds (111 frames at 24 FPS)

### View the Video

```bash
# Play with ffplay
ffplay butterfly_garden.mp4

# Or open with default player
open butterfly_garden.mp4  # macOS
xdg-open butterfly_garden.mp4  # Linux
```

## Step 2: Generate from Your Checkpoint

Use your fine-tuned checkpoint from training.

### Load Custom Checkpoint

The generation script can load from:
1. **Consolidated checkpoint** (single `.pt` file)
2. **Sharded checkpoint** (distributed `.distcp` files)

**For consolidated checkpoints**:

```bash
python dfm/examples/automodel/generate/wan_generate.py \
    --prompt "A robot cooking in a kitchen" \
    --checkpoint-path /path/to/checkpoints/wan2_1_finetuning/iter_10000/consolidated_checkpoint.pt \
    --output robot_cooking.mp4
```

**For sharded checkpoints**:

The script automatically detects and loads sharded checkpoints from the directory.

```bash
python dfm/examples/automodel/generate/wan_generate.py \
    --prompt "A robot cooking in a kitchen" \
    --checkpoint-path /path/to/checkpoints/wan2_1_finetuning/iter_10000/ \
    --output robot_cooking.mp4
```

## Step 3: Multi-GPU Generation (Optional)

Speed up generation using tensor parallelism across multiple GPUs.

```bash
torchrun --nproc-per-node 2 \
    dfm/examples/automodel/generate/wan_generate.py \
    --prompt "A robot cooking in a kitchen" \
    --height 720 \
    --width 1280 \
    --num-frames 149 \
    --tp-size 2 \
    --output robot_cooking_hd.mp4
```

**Parallelism options**:
- `--tp-size 2`: Split model across 2 GPUs (tensor parallelism)
- `--cp-size`: Context parallelism (rarely needed for inference)
- `--pp-size`: Pipeline parallelism (for very large models)

**When to use multi-GPU**:
- High-resolution videos (720p, 1080p)
- Long videos (200+ frames)
- Faster generation (reduces time by ~40-60%)

## Generation Parameters

### Common Parameters

| Parameter | Description | Default | Range/Options |
|-----------|-------------|---------|---------------|
| `--prompt` | Text description of video | Required | Any text |
| `--height` | Video height (pixels) | `480` | 360, 480, 720, 1080 |
| `--width` | Video width (pixels) | `848` | 640, 848, 1280, 1920 |
| `--num-frames` | Number of frames | `111` | 51, 111, 149 (4n+1 format) |
| `--output` | Output filename | `t2v_fsdp2_rank0.mp4` | Any `.mp4` path |
| `--seed` | Random seed | `42` | Any integer |

### Quality vs. Speed Parameters

| Parameter | Description | Default | Range |
|-----------|-------------|---------|-------|
| `--num-inference-steps` | Diffusion steps (more = better quality) | `20` | 10-50 |
| `--guidance-scale` | Prompt adherence strength | `4.0` | 1.0-10.0 |
| `--guidance-scale-2` | Secondary guidance | `3.0` | 1.0-10.0 |
| `--fps` | Frames per second | `24` | 12, 24, 30 |

### Frame Count Format

**Important**: `--num-frames` must follow the `4n+1` format:
- Valid: 51, 111, 149, 189, 229
- Invalid: 50, 100, 150

This ensures compatibility with the model's temporal patching.

## Advanced Usage

### High-Quality Generation

Maximum quality settings (slower generation):

```bash
python dfm/examples/automodel/generate/wan_generate.py \
    --prompt "A serene lake at sunset with mountains in the background" \
    --height 720 \
    --width 1280 \
    --num-frames 149 \
    --num-inference-steps 50 \
    --guidance-scale 7.0 \
    --output sunset_lake_hq.mp4
```

**Changes**:
- More inference steps (50 vs. 20): Smoother, more detailed
- Higher guidance scale (7.0 vs. 4.0): Stronger prompt adherence
- Higher resolution (720p vs. 480p): Sharper video

**Trade-off**: ~3-5x longer generation time

### Fast Generation

Quick generation for prototyping:

```bash
python dfm/examples/automodel/generate/wan_generate.py \
    --prompt "A cat playing with yarn" \
    --height 360 \
    --width 640 \
    --num-frames 51 \
    --num-inference-steps 10 \
    --output cat_yarn_fast.mp4
```

**Changes**:
- Fewer inference steps (10 vs. 20): Faster but less refined
- Lower resolution (360p vs. 480p): Faster processing
- Fewer frames (51 vs. 111): Shorter video

**Trade-off**: Lower quality, but ~4-5x faster

### Reproducible Generation

Generate the same video multiple times:

```bash
python dfm/examples/automodel/generate/wan_generate.py \
    --prompt "A dog running on a beach" \
    --seed 12345 \
    --output dog_beach_v1.mp4

# Run again with same seed → identical output
python dfm/examples/automodel/generate/wan_generate.py \
    --prompt "A dog running on a beach" \
    --seed 12345 \
    --output dog_beach_v2.mp4

# dog_beach_v1.mp4 and dog_beach_v2.mp4 are identical
```

## Prompt Engineering Tips

### Effective Prompts

**Good prompts are**:
- **Specific**: Include details (colors, actions, setting)
- **Descriptive**: Paint a visual picture
- **Concise**: 1-3 sentences

**Examples**:

✅ **Good**:
```
"A teal robot cooking food in a cozy kitchen. Steam rises from a simmering pot 
as the robot chops vegetables on a wooden cutting board. Sunlight streams through 
a window, illuminating copper pans hanging from an overhead rack."
```

❌ **Too vague**:
```
"A robot"
```

❌ **Too long**:
```
"In a futuristic kitchen with advanced technology and sophisticated equipment where 
a mechanical being of teal coloration undertakes various culinary tasks including 
but not limited to the preparation and cooking of food items..."
```

### Prompt Structure

**Recommended structure**:
1. **Subject**: What/who is the focus?
2. **Action**: What are they doing?
3. **Setting**: Where is this happening?
4. **Details**: Colors, lighting, mood

**Example**:
```
Subject: "The teal robot"
Action: "is cooking food in a kitchen"
Setting: "on a wooden cutting board with copper pans hanging above"
Details: "Steam rises from a pot, afternoon light through the window"
```

## Troubleshooting

### Out of Memory Errors

```
RuntimeError: CUDA out of memory
```

**Solution 1**: Reduce resolution and frames:

```bash
python dfm/examples/automodel/generate/wan_generate.py \
    --prompt "Your prompt" \
    --height 360 \
    --width 640 \
    --num-frames 51 \
    --output output.mp4
```

**Solution 2**: Use tensor parallelism:

```bash
torchrun --nproc-per-node 2 \
    dfm/examples/automodel/generate/wan_generate.py \
    --prompt "Your prompt" \
    --tp-size 2 \
    --output output.mp4
```

### Slow Generation

**Expected times** (single GPU, 480p, 111 frames):
- 20 steps: 2-3 minutes
- 50 steps: 5-7 minutes

**Speed up**:
1. Reduce `--num-inference-steps` to 10-15
2. Use multi-GPU with `--tp-size 2`
3. Lower resolution/frame count

### Poor Quality Results

**Symptoms**: Blurry, artifacts, doesn't match prompt

**Solutions**:
1. Increase `--num-inference-steps` to 30-50
2. Increase `--guidance-scale` to 6.0-7.5
3. Refine your prompt (more specific, descriptive)
4. Try different `--seed` values

### Model Loading Errors

```
FileNotFoundError: Model not found
```

**For pretrained models**:

```bash
# Login to Hugging Face
huggingface-cli login

# Check internet connection
ping huggingface.co
```

**For custom checkpoints**:

```bash
# Verify checkpoint exists
ls -lh /path/to/checkpoint/

# Check for consolidated or sharded format
ls /path/to/checkpoint/*.pt
ls /path/to/checkpoint/*.distcp
```

## Next Steps

After generating videos:

1. **Evaluate quality**: Compare outputs to training data
2. **Iterate on prompts**: Refine prompts for better results
3. **Experiment with parameters**: Find optimal quality/speed balance
4. **Scale up**: Use multi-GPU for high-resolution production

## Related Pages

- **[Automodel Training](training.md)**: Fine-tune your own model
- **[Diffusion Models](../../about/concepts/diffusion-models.md)**: Understand how generation works
- **[Distributed Training](../../reference/distributed-training.md)**: Multi-GPU inference optimization

