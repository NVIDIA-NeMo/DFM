---
description: "End-to-end Automodel quickstart: fine-tune and generate videos"
categories: ["getting-started", "automodel"]
tags: ["quickstart", "tutorial", "automodel"]
personas: ["data-scientist-focused"]
difficulty: "beginner"
content_type: "tutorial"
---

(gs-automodel)=

# Automodel Quickstart

Complete end-to-end tutorial for fine-tuning and generating videos using NeMo DFM's Automodel approach.

**What you'll accomplish**:
1. Fine-tune the WAN2.1 model on your dataset
2. Generate videos from your trained model
3. Experiment with generation parameters

**Time**: 30-45 minutes (depending on training duration)

**Prerequisites**:
- Complete [Installation](../installation.md)
- Multi-GPU setup (recommended: 8 GPUs)
- Dataset in Energon format or custom dataloader

## Automodel Approach

**Best for**: Quick prototyping, fine-tuning pretrained models

**Key features**:
- Recipe-based training with YAML configuration
- Automatic FSDP2 parallelism (no manual setup)
- Uses Hugging Face models
- Simpler configuration vs. Megatron

**When to use this**:
- Fine-tuning pretrained models
- Rapid experimentation
- Production inference with standard models
- Teams comfortable with PyTorch and Hugging Face

## Quickstart Steps

```{toctree}
---
maxdepth: 1
---
training
inference
```

### Step 1: Training
[Fine-tune WAN2.1 model](training.md) with automatic parallelism

### Step 2: Inference
[Generate videos](inference.md) from your trained checkpoint

## Next Steps

After completing this quickstart:

- **Scale up**: [Distributed Training Reference](../reference/distributed-training.md)
- **Understand the architecture**: [Diffusion Models](../about/concepts/diffusion-models.md)
- **Explore alternatives**: [Megatron Quickstart](../megatron/index.md) for large-scale pretraining

## Need Help?

**Not sure if Automodel is right for you?**

Consider [Megatron Quickstart](../megatron/index.md) if you need:
- Full control over distributed training
- Large-scale pretraining from scratch
- Custom parallelism strategies
- Advanced optimization techniques

