---
description: "Experimental comparison between AutoModel and Megatron training paths for WAN 2.1"
categories: ["concepts-architecture"]
tags: ["comparison", "automodel", "megatron", "wan", "experimental"]
personas: ["mle-focused", "data-scientist-focused"]
difficulty: "intermediate"
content_type: "explanation"
---

(about-comparison)=

# AutoModel vs Megatron Comparison

Experimental comparison of two training paths for WAN 2.1: the AutoModel (Diffusers) path versus the Megatron-Core (Megatron-Bridge) path.

## Experiment Overview

**Goal**: Compare two training paths for WAN 2.1:

1. **[Diffusers](https://huggingface.co/docs/diffusers/en/index) implementation + [AutoModel](https://github.com/NVIDIA-NeMo/Automodel/tree/diffusion) training path**
2. **[Megatron-Core](https://github.com/NVIDIA/Megatron-LM) implementation + [Megatron-Bridge](https://github.com/NVIDIA-NeMo/Megatron-Bridge) training path**

**Training Approach**: Two-stage training

- **Stage 1**: Text → Image - Learn to connect textual embeddings with visual concepts
- **Stage 2**: Text → Video - Learn visual movements aligning with prompts

**Dataset**: 3,000 videos; frames extracted from videos are used for text-to-image training stage.

:::{note}
This experiment is a partial convergence test and only demonstrates the model's ability to reconstruct images and videos from input prompts. With only 3,000 videos, the model cannot generalize to generate novel content. Such generalization can be achieved with larger training datasets and increased training resources.
:::

---

## Dataset Configuration

:::: {tab-set}

::: {tab-item} Stage 1: Text-to-Image

**Dataset**:
- Extract 40 frames per video → **120k images**
- Resolution: **240 × 416**
- Each frame uses same caption as parent video

**Training Setup**:
- Global batch size: 2560 images
- Learning rate: warmup 10k → 5e-5 constant
- Hardware: 10 nodes (80 GPUs)

| Path | Parallelism | Notes |
|------|-------------|-------|
| Megatron-Core | TP=1, PP=1, CP=1 | Sequence packing (32 samples/pack) |
| AutoModel | FSDP | micro_batch_size = 32 |

:::

::: {tab-item} Stage 2: Text-to-Video

**Dataset**:
- Full videos → **3,000 videos**
- Resolution: **240 × 416**, duration 4–8 seconds

**Training Setup**:
- Global batch size: 80 videos
- Learning rate: 5e-5 constant
- Hardware: 10 nodes (80 GPUs)

| Path | Parallelism | Notes |
|------|-------------|-------|
| Megatron-Core | TP=1, PP=1, CP=1 | micro_batch_size = 1 |
| AutoModel | FSDP | micro_batch_size = 1 |

:::

::::

---

## Results

### Stage 1 — Loss vs. Steps

```{image} ../medias/training_curves/lm_loss_text2image_3kvids.png
:alt: Training loss curve for Stage 1 (Text-to-Image)
:width: 700px
```

### Stage 2 — Loss vs. Steps

```{image} ../medias/training_curves/lm_loss_text2video_3kvids.png
:alt: Training loss curve for Stage 2 (Text-to-Video)
:width: 700px
```

:::{note}
Training loss is smoothed with 50 steps averaging.
:::

### Analysis

The training curves for both stages have similar value ranges, although they do not match exactly. This is expected due to differences in implementation and training loop setups.

:::{dropdown} Important Caveat: Megatron-Core Timestep Handling
:icon: alert

In the current Megatron-Core implementation, the same diffusion time steps are applied to all samples within a pack for each step, rather than different time steps for each sample. As a result, the training loss for Megatron-Core fluctuates more significantly than for AutoModel, especially at the beginning of training.
:::

---

## Key Takeaways

- Both paths achieve similar training loss ranges
- Implementation differences lead to curve variations (expected)
- Megatron-Core shows more loss fluctuation due to timestep handling in sequence packing
- Both paths successfully learn reconstruction from prompts

---

## Related Documentation

- [Training Paradigms](concepts/training-paradigms.md) - Detailed comparison of paradigms
- [Performance Benchmarks](../reference/performance.md) - Training throughput metrics
- [Get Started](../get-started/index.md) - Start training with either path

