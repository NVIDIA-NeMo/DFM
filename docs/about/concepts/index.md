---
description: "Core concepts and terminology for NeMo DFM including training paradigms, diffusion models, video data representation, and distributed training"
categories: ["concepts-architecture"]
tags: ["concepts", "fundamentals", "diffusion", "training", "distributed"]
personas: ["data-scientist-focused", "mle-focused"]
difficulty: "beginner"
content_type: "concept"
modality: "universal"
---

(about-concepts)=

# Concepts

Learn about the core concepts you need to understand before using NeMo DFM.

## Core Concepts

These concepts are essential for understanding how NeMo DFM works and making informed decisions about your training and inference workflows.

::::{grid} 1 1 1 2
:gutter: 1 1 1 2

:::{grid-item-card} {octicon}`git-branch;1.5em;sd-mr-1` Training Paradigms
:link: about-concepts-training-paradigms
:link-type: ref

Understand the two main training approaches: Automodel (recipe-based) and Megatron (large-scale distributed), and when to use each.
:::

:::{grid-item-card} {octicon}`graph;1.5em;sd-mr-1` Diffusion Models for Video
:link: about-concepts-diffusion-models
:link-type: ref

Learn how diffusion models work for video generation, including EDM and Flow Matching paradigms.
:::

:::{grid-item-card} {octicon}`database;1.5em;sd-mr-1` Video Data Representation
:link: about-concepts-video-data
:link-type: ref

Understand how video data is represented in DFM: latents, VAE encoding, tokenization, and data formats.
:::

:::{grid-item-card} {octicon}`server;1.5em;sd-mr-1` Distributed Training
:link: about-concepts-distributed-training
:link-type: ref

Learn about parallelism strategies: tensor parallelism (TP), context parallelism (CP), pipeline parallelism (PP), and data parallelism (DP).
:::

:::{grid-item-card} {octicon}`gear;1.5em;sd-mr-1` Configuration System
:link: about-concepts-configuration
:link-type: ref

Understand how DFM's configuration system works: YAML files, CLI overrides, and configuration precedence.
:::

::::

```{toctree}
:hidden:
:maxdepth: 2

about/concepts/index.md
Training Paradigms <training-paradigms.md>
Diffusion Models for Video <diffusion-models.md>
Video Data Representation <video-data.md>
Distributed Training <distributed-training.md>
Configuration System <configuration.md>
```
