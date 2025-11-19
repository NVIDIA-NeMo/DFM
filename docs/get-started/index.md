---
description: "Get started with NeMo DFM for video generation"
categories: ["getting-started"]
tags: ["quickstart", "overview"]
personas: ["data-scientist-focused", "mle-focused"]
difficulty: "beginner"
content_type: "tutorial"
---

(gs-index)=

# Get Started with NeMo DFM

Start generating and training video diffusion models with NeMo DFM.

## Installation

**First step for all users**: Install NeMo DFM

→ **[Installation Quickstart](installation.md)**

Install via Docker, pip, or from source. Takes 10-15 minutes.

---

## Choose Your Path

After installation, choose the approach that matches your goals:

### Automodel: Quick Prototyping

**Best for**: Fine-tuning pretrained models, rapid experimentation

```{card}
**Automodel Quickstart**
^^^
**What you'll do**:
- Fine-tune WAN2.1 model from Hugging Face
- Generate videos from your checkpoint
- Experiment with generation parameters

**Time**: 30-45 minutes

**Complexity**: ⭐⭐☆☆☆ Beginner-friendly

**Key features**:
- Recipe-based training (YAML configuration)
- Automatic FSDP2 parallelism
- Use Hugging Face models directly
- Simpler setup vs. Megatron

+++
{bdg-primary}`Recommended for data scientists` {bdg-success}`Fast start`

[Start Automodel Track →](automodel/index.md)
```

### Megatron: Large-Scale Training

**Best for**: Pretraining from scratch, full distributed control

```{card}
**Megatron Quickstart**
^^^
**What you'll do**:
- Prepare Smithsonian Butterflies dataset
- Train DiT model from scratch
- Generate videos from your checkpoint

**Time**: 1-2 hours

**Complexity**: ⭐⭐⭐☆☆ Intermediate

**Key features**:
- Manual parallelism configuration (TP+CP+PP+DP)
- Three-layer config system (recipe + YAML + CLI)
- Webdataset format for scalability
- Advanced optimization

+++
{bdg-primary}`Recommended for MLEs` {bdg-info}`Full control`

[Start Megatron Track →](megatron/index.md)
```

---

## Quick Comparison

Not sure which path to choose? Compare the approaches:

| Feature | Automodel | Megatron |
|---------|-----------|----------|
| **Best for** | Fine-tuning pretrained models | Pretraining from scratch |
| **Configuration** | Single YAML file | Recipe + YAML + CLI overrides |
| **Parallelism** | Automatic (FSDP2) | Manual (TP+CP+PP+DP) |
| **Model source** | Hugging Face models | Custom checkpoints |
| **Data format** | Energon or custom dataloader | Webdataset |
| **Setup time** | Fast (~10 mins) | Moderate (~30 mins) |
| **Complexity** | ⭐⭐☆☆☆ | ⭐⭐⭐☆☆ |
| **Control** | Less (automatic) | More (manual) |

### Decision Guide

Choose **Automodel** if you:
- Want to fine-tune existing models
- Prefer simpler configuration
- Need faster experimentation
- Work with standard Hugging Face models

Choose **Megatron** if you:
- Need to pretrain from scratch
- Require full control over parallelism
- Train on large clusters (100+ GPUs)
- Need maximum performance optimization

**Still unsure?** Start with [Automodel](automodel/index.md)—it's faster to learn and you can always switch to Megatron later.

---

## What's Next?

After completing a quickstart track:

```{toctree}
---
maxdepth: 2
---
automodel/index
megatron/index
installation
inference
training
```

### Learn the Concepts

- **[Diffusion Models](../about/concepts/diffusion-models.md)**: How video generation works
- **[Training Paradigms](../about/concepts/training-paradigms.md)**: Automodel vs. Megatron deep dive
- **[Distributed Training](../about/concepts/distributed-training.md)**: Parallelism strategies
- **[Configuration](../about/concepts/configuration.md)**: YAML configuration system

### Explore Examples

- **Automodel examples**: `dfm/examples/automodel/`
- **Megatron examples**: `examples/megatron/recipes/`

### Reference Documentation

- **[Distributed Training Reference](../reference/distributed-training.md)**: Advanced parallelism
- **[Data Loading Reference](../reference/data-loading.md)**: Dataset preparation
- **[API Reference](../apidocs/index.rst)**: Full API documentation

---

## Need Help?

**Common questions**:
- **"Which approach should I use?"** → See [Decision Guide](#decision-guide) above
- **"How do I install NeMo DFM?"** → [Installation Quickstart](installation.md)
- **"Where are the code examples?"** → `dfm/examples/` (Automodel) and `examples/megatron/` (Megatron)

**Get support**:
- GitHub Issues: [Report bugs or request features](https://github.com/NVIDIA-NeMo/DFM/issues)
- GitHub Discussions: [Ask questions](https://github.com/NVIDIA-NeMo/DFM/discussions)
