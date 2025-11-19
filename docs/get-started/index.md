(gs-index)=

# Get Started with NeMo DFM

**Estimated Time**: 1-2 hours (depending on chosen path)

This guide helps you get started with training video diffusion models using NeMo DFM. Each tutorial is a comprehensive, end-to-end guide that takes you from installation through training and inference.

**By completing a tutorial, you will have:**

✅ A working NeMo DFM installation
✅ Hands-on experience with video model training and inference
✅ Understanding of Automodel vs. Megatron approaches
✅ Ability to generate videos from trained checkpoints

## Before You Start

Make sure you have these prerequisites ready before beginning the tutorials:

- **Python 3.10+**
- **Multi-GPU system** (recommended: 8 GPUs for optimal performance)
- **Git** for cloning the repository
- **~50GB storage** for datasets and checkpoints
- Basic command-line familiarity

---

## Getting Started Path

Follow these steps to build your first video generation model:

::::::{grid} 1 1 1 1

:::::{grid-item-card} {octicon}`package;1.5em;sd-mr-1` 1. Installation
:link: installation
:link-type: doc

Get NeMo DFM installed and verify your setup with a quick test.
+++
{bdg-secondary}`environment` {bdg-secondary}`first-run`
:::::

:::::{grid-item}
:margin: 0
:padding: 0

::::{grid} 1 2 2 2
:margin: 3 1 0 0
:gutter: 3
:padding: 3

:::{grid-item-card} {octicon}`zap;1.5em;sd-mr-1` 2a. Automodel Tutorial
:link: automodel
:link-type: doc

Fine-tune pretrained models with automatic parallelism. Best for quick prototyping.
+++
{bdg-secondary}`automodel` {bdg-success}`Fast start` {bdg-primary}`Data scientists`
:::

:::{grid-item-card} {octicon}`server;1.5em;sd-mr-1` 2b. Megatron Tutorial
:link: megatron
:link-type: doc

Train from scratch with full distributed control. Best for large-scale training.
+++
{bdg-secondary}`megatron` {bdg-info}`Full control` {bdg-primary}`MLEs`
:::

::::
:::::

::::::

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

**Still unsure?** Start with [Automodel](automodel.md)—it's faster to learn and you can always switch to Megatron later.
