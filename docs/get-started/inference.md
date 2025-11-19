---
description: "Inference quickstart guide for NeMo DFM"
categories: ["getting-started"]
tags: ["inference", "quickstart", "tutorial"]
personas: ["mle-focused", "data-scientist-focused"]
difficulty: "beginner"
content_type: "tutorial"
---

(gs-inference)=

# Inference Quickstart

Learn how to generate videos from text prompts using NeMo DFM. This tutorial walks you through two inference approaches: Automodel for Hugging Face models and Megatron for custom checkpoints.

**What you'll learn**:
- Generate videos using pre-trained models
- Configure distributed parallelism for faster inference
- Adjust generation parameters for quality vs. speed
- Troubleshoot common inference issues

**Time to complete**: 10-15 minutes

## Prerequisites

Before starting:

- Complete the [Installation Quickstart](gs-installation)
- Have a CUDA-capable GPU available
- Choose your model source:
  - Automodel: Hugging Face model identifier
  - Megatron: Local checkpoint directory

## Step 1: Choose Your Inference Path

NeMo DFM supports two inference approaches:

| Approach | Model Source | Best For |
|----------|--------------|----------|
| **Automodel** | Hugging Face models | Quick start, pre-trained models |
| **Megatron** | Custom checkpoints | Custom models, fine-tuned weights |

Choose Automodel if you want to start quickly with pre-trained models. Choose Megatron if you have custom checkpoints from training.

## Step 2: Run Automodel Inference

Automodel inference generates videos from Hugging Face models with optional distributed parallelism.

### Single GPU Generation

Generate a video from a text prompt:

```bash
python dfm/examples/automodel/generate/wan_generate.py \
    --prompt "A cat playing piano" \
    --height 480 \
    --width 848 \
    --num-frames 111 \
    --output output.mp4
```

This command:
1. Loads the model from Hugging Face (`Wan-AI/Wan2.2-T2V-A14B-Diffusers`)
2. Generates 111 frames at 480×848 resolution
3. Saves the video to `output.mp4`

**Expected output**:
- Generation time: 2-5 minutes (depending on GPU)
- Output file: `output.mp4` (approximately 5-10 MB)

### Multi-GPU Generation (Optional)

Speed up generation using distributed parallelism:

```bash
torchrun --nproc-per-node 2 \
    dfm/examples/automodel/generate/wan_generate.py \
    --prompt "A cat playing piano" \
    --height 480 \
    --width 848 \
    --num-frames 111 \
    --tp-size 2 \
    --output output.mp4
```

**Common parameters**:

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--prompt` | Text description of video | Required |
| `--height` | Video height in pixels | `480` |
| `--width` | Video width in pixels | `848` |
| `--num-frames` | Number of frames (4n+1 format) | `111` |
| `--output` | Output filename | `t2v_fsdp2_rank0.mp4` |

<details>
<summary>View all parameters</summary>

**Generation control**:

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--guidance-scale` | Classifier-free guidance scale | `4.0` |
| `--num-inference-steps` | Number of diffusion steps | `20` |
| `--fps` | Frames per second | `24` |
| `--seed` | Random seed for reproducibility | Random |

**Parallelism options**:

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--tp-size` | Tensor parallel group size | `1` |
| `--cp-size` | Context parallel group size | `1` |
| `--pp-size` | Pipeline parallel group size | `1` |
| `--dp-size` | Data parallel group size | `1` |

</details>

## Step 3: Run Megatron Inference (Alternative)

Use Megatron inference to generate videos from custom checkpoints.

### Single Prompt Generation

Generate from your custom checkpoint:

```bash
torchrun --nproc-per-node 2 \
    examples/megatron/recipes/wan/inference_wan.py \
    --task t2v-14B \
    --checkpoint_dir /path/to/checkpoint \
    --prompts "A cat playing piano" \
    --sizes 1280*720 \
    --frame_nums 111
```

This command:
1. Loads your checkpoint from the specified directory
2. Generates a 1280×720 video with 111 frames
3. Saves the video to the current directory with a timestamped filename

**Expected output**:
- Filename format: `t2v-14B_DefaultExp_videoindex0_size1280*720_prompt_timestamp.mp4`

### Batch Generation (Optional)

Generate multiple videos at once:

```bash
torchrun --nproc-per-node 2 \
    examples/megatron/recipes/wan/inference_wan.py \
    --task t2v-14B \
    --checkpoint_dir /path/to/checkpoint \
    --prompts "A cat playing piano" "A dog running in a park" \
    --sizes 1280*720 832*480 \
    --frame_nums 111 149
```

**Common parameters**:

| Parameter | Description | Required |
|-----------|-------------|----------|
| `--task` | Model architecture (`t2v-14B` or `t2v-1.3B`) | Yes |
| `--checkpoint_dir` | Path to checkpoint directory | Yes |
| `--prompts` | Text prompts (space-separated) | Yes |
| `--sizes` | Video sizes in WIDTH*HEIGHT format | Optional |
| `--frame_nums` | Frame counts (must be 4n+1) | Optional |

<details>
<summary>View all parameters and supported configurations</summary>

**Advanced parameters**:

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--checkpoint_step` | Specific checkpoint step to load | Latest |
| `--sample_steps` | Number of sampling steps | `50` |
| `--sample_guide_scale` | Guidance scale strength | `5.0` |
| `--sample_shift` | Noise schedule shift | `5.0` |

**Supported configurations**:

**t2v-14B** (14 billion parameter model):
- Supported sizes: `720*1280`, `1280*720`, `480*832`, `832*480`
- Default frames: 111

**t2v-1.3B** (1.3 billion parameter model):
- Supported sizes: `416*240`, `480*832`, `832*480`
- Default frames: 111

</details>

## Step 4: View Your Generated Video

Check that your video was created:

```bash
ls -lh output.mp4
```

Play the video:

```bash
# Using ffplay
ffplay output.mp4

# Or open with your default video player
open output.mp4  # macOS
xdg-open output.mp4  # Linux
```

**Expected results**:
- Video file size: 5-50 MB (depending on resolution and frame count)
- Video duration: 4-6 seconds at 24 FPS for 111 frames
- Quality: HD video matching your prompt description

**Megatron output location**:
For Megatron inference, videos save to the current directory with timestamped filenames:
- `t2v-14B_DefaultExp_videoindex0_size1280*720_prompt_timestamp.mp4`

## Experiment with Generation Settings

Now that you have a working inference setup, try adjusting parameters to see their effects.

### Improve Video Quality

Increase quality at the cost of generation time:

```bash
python dfm/examples/automodel/generate/wan_generate.py \
    --prompt "A cat playing piano" \
    --num-inference-steps 50 \
    --guidance-scale 7.0 \
    --height 720 \
    --width 1280 \
    --output high_quality.mp4
```

**Changes**:
- More inference steps (50 vs. 20): Smoother, more detailed results
- Higher guidance scale (7.0 vs. 4.0): Stronger prompt adherence
- Higher resolution (720p vs. 480p): Sharper video

### Speed Up Generation

Reduce generation time while maintaining acceptable quality:

```bash
python dfm/examples/automodel/generate/wan_generate.py \
    --prompt "A cat playing piano" \
    --num-inference-steps 10 \
    --height 360 \
    --width 640 \
    --num-frames 51 \
    --output fast.mp4
```

**Changes**:
- Fewer inference steps (10 vs. 20): Faster but less refined
- Lower resolution (360p vs. 480p): Faster processing
- Fewer frames (51 vs. 111): Shorter video, faster generation

### Reproduce Results

Generate the same video multiple times:

```bash
python dfm/examples/automodel/generate/wan_generate.py \
    --prompt "A cat playing piano" \
    --seed 42 \
    --output reproducible.mp4
```

Using `--seed` ensures identical results across runs.

## Troubleshooting

### Out of Memory Errors

```
RuntimeError: CUDA out of memory
```

**Solution**: Reduce memory usage:

```bash
python dfm/examples/automodel/generate/wan_generate.py \
    --prompt "A cat playing piano" \
    --height 360 \
    --width 640 \
    --num-frames 51 \
    --output lower_memory.mp4
```

Or use distributed parallelism to split memory across GPUs:

```bash
torchrun --nproc-per-node 2 \
    dfm/examples/automodel/generate/wan_generate.py \
    --prompt "A cat playing piano" \
    --tp-size 2 \
    --output distributed.mp4
```

### Slow Generation

If generation takes more than 10 minutes for a single video:

1. Reduce inference steps: `--num-inference-steps 10`
2. Lower resolution: `--height 360 --width 640`
3. Enable parallelism: `--tp-size 2` with `torchrun --nproc-per-node 2`

### Poor Quality Results

If videos are blurry, artifacts are visible, or prompt is not followed:

1. Increase inference steps: `--num-inference-steps 50`
2. Increase guidance scale: `--guidance-scale 7.0`
3. Refine your prompt (be more specific and descriptive)
4. Use higher resolution: `--height 720 --width 1280`

### Model Loading Errors

```
FileNotFoundError: Checkpoint not found
```

**For Automodel**: Check internet connection and Hugging Face access:

```bash
huggingface-cli login
```

**For Megatron**: Verify checkpoint path:

```bash
ls -lh /path/to/checkpoint/
# Should contain model files and configuration
```

## Summary and Next Steps

You learned how to:
- ✅ Generate videos from text prompts using Automodel or Megatron
- ✅ Adjust generation parameters for quality vs. speed trade-offs
- ✅ Use distributed parallelism for faster inference
- ✅ Troubleshoot common inference issues

**Continue learning**:

- **[Training Quickstart](gs-training)**: Train and fine-tune your own video generation models
- **[Concepts: Diffusion Models](about-concepts-diffusion-models)**: Understand how video generation works
- **[Reference: Distributed Training](ref-distributed-training)**: Deep dive into parallelism strategies
