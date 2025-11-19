---
description: "End-to-end Megatron quickstart: prepare data, train, and generate videos"
categories: ["getting-started", "megatron"]
tags: ["quickstart", "tutorial", "megatron"]
personas: ["mle-focused"]
difficulty: "intermediate"
content_type: "tutorial"
---

(gs-megatron)=

# Megatron Quickstart

Complete end-to-end tutorial for pretraining a DiT model and generating videos using NeMo DFM's Megatron approach.

**What you'll accomplish**:
1. Prepare the Smithsonian Butterflies dataset
2. Train a DiT model from scratch
3. Generate videos from your trained model

**Time**: 1-2 hours (depending on training duration)

**Prerequisites**:
- Complete [Installation](../installation.md)
- Multi-GPU setup (minimum: 2 GPUs, recommended: 8+ GPUs)
- ~50GB storage for dataset and checkpoints

## Megatron Approach

**Best for**: Large-scale pretraining, full distributed control

**Key features**:
- Manual control over parallelism (TP+CP+PP+DP)
- Three-layer configuration (recipe + YAML + CLI)
- Webdataset format for scalability
- Advanced optimization techniques

**When to use this**:
- Pretraining models from scratch
- Large-scale distributed training (100+ GPUs)
- Custom parallelism strategies
- Performance-critical training
- Teams familiar with Megatron-LM

## Quickstart Steps

```{toctree}
---
maxdepth: 1
---
prepare-data
training
inference
```

### Step 1: Prepare Dataset
[Prepare Smithsonian Butterflies dataset](prepare-data.md) in webdataset format

### Step 2: Training
[Train DiT model](training.md) with distributed parallelism

### Step 3: Inference
[Generate videos](inference.md) from your trained checkpoint

## Example: Train on Butterfly Images

This quickstart uses the **Smithsonian Butterflies dataset** from Hugging Face:
- **Source**: `huggan/smithsonian_butterflies_subset`
- **Size**: ~800 images with captions
- **Task**: Image-to-video generation (DiT model)
- **Why butterflies?**: Small, fast dataset perfect for learning the workflow

**Real-world application**: Replace with your production dataset after completing this tutorial.

## Next Steps

After completing this quickstart:

- **Scale up**: [Distributed Training Reference](../../reference/distributed-training.md)
- **Optimize**: [Training Paradigms](../../about/concepts/training-paradigms.md)
- **Compare approaches**: [Automodel Quickstart](../automodel/index.md) for simpler workflows

## Need Help?

**Not sure if Megatron is right for you?**

Consider [Automodel Quickstart](../automodel/index.md) if you need:
- Simpler configuration
- Faster prototyping
- Fine-tuning pretrained models
- Automatic parallelism

