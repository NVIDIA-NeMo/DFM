---
description: "Advanced training recipes and comprehensive tutorials for NeMo DFM"
categories: ["tutorials"]
tags: ["recipes", "training", "advanced"]
personas: ["mle-focused", "data-scientist-focused"]
difficulty: "intermediate"
content_type: "tutorial"
---

(tutorials-index)=

# Tutorials Overview

Comprehensive training recipes and advanced tutorials for NeMo DFM. These tutorials provide detailed configurations, advanced topics, and in-depth guidance for production training workflows.

**When to use these tutorials**:
- You've completed the [Get Started](../get-started/index.md) guides
- You need detailed configuration options and advanced settings
- You're setting up production training workflows
- You want to understand all available options and parameters

## Available Tutorials

::::{grid} 1 2 2 2
:gutter: 1 1 1 2

:::{grid-item-card} {octicon}`zap;1.5em;sd-mr-1` Fine-Tuning Pretrained Models
:link: fine-tuning-pretrained-models
:link-type: doc

Fine-tune pretrained models with automatic parallelism. Detailed configuration options, preprocessing modes, and advanced topics for quick prototyping workflows.
+++
{bdg-secondary}`automodel` {bdg-secondary}`wan` {bdg-success}`Quick start` {bdg-info}`Advanced`
:::

:::{grid-item-card} {octicon}`server;1.5em;sd-mr-1` Training from Scratch
:link: training-from-scratch
:link-type: doc

Train DiT models from scratch with full distributed control. Complete recipe with sequence packing, Energon format details, and parallelism configuration for large-scale training.
+++
{bdg-secondary}`megatron` {bdg-secondary}`dit` {bdg-info}`Full control` {bdg-info}`Advanced`
:::

:::{grid-item-card} {octicon}`video;1.5em;sd-mr-1` Text-to-Video Training
:link: text-to-video-training
:link-type: doc

Train production-scale text-to-video models with WAN 2.1. Detailed recipe with WebDataset preparation, pretraining workflows, and inference for video generation.
+++
{bdg-secondary}`megatron` {bdg-secondary}`wan` {bdg-info}`Video models` {bdg-info}`Advanced`
:::

::::

---

## Relationship to Get Started Guides

These tutorials complement the [Get Started](../get-started/index.md) guides:

- **Get Started**: Quick start, basic workflows, essential steps
- **Tutorials**: Comprehensive recipes, detailed configurations, advanced topics

If you're new to NeMo DFM, start with the [Get Started](../get-started/index.md) guides, then refer to these tutorials for advanced configuration and detailed options.

---
