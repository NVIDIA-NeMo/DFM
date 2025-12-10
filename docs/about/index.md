---
description: "Overview of NeMo DFM, a framework for large-scale training and inference of video diffusion models with Automodel and Megatron support"
categories: ["getting-started"]
tags: ["overview", "platform", "diffusion", "video-models", "getting-started"]
personas: ["data-scientist-focused", "mle-focused", "admin-focused", "devops-focused"]
difficulty: "beginner"
content_type: "concept"
modality: "universal"
---

(about-overview)=

# Overview of NeMo DFM

NeMo DFM (Diffusion Foundation Models) trains and runs inference on video diffusion models at scale. It combines two training approaches—Automodel for recipe-based workflows and Megatron for multi-node distributed training—with support for multiple architectures including DiT, WAN, and EDM.

**Use NeMo DFM to:**

- Train video diffusion models using Flow Matching or EDM paradigms
- Scale training across GPUs and nodes with tensor, context, and pipeline parallelism
- Run efficient video generation inference on trained models
- Experiment with different architectures (DiT, WAN, EDM) using the same framework

## Who Should Use DFM

- **Machine Learning Engineers**: Train video foundation models using diffusion and autoregressive architectures with configuration-driven workflows.
- **Data Scientists**: Process video datasets with VAE encoding and tokenization pipelines for diffusion model training.
- **Cluster Administrators**: Deploy and monitor large-scale distributed training jobs across multi-node GPU clusters.
- **Researchers**: Experiment with diffusion architectures (DiT, EDM, WAN), training paradigms (Flow Matching, EDM), and parallelism strategies.

## What DFM Provides

**Two Training Paradigms**:

- **Automodel**: Recipe-based training with DTensor for 3D parallelism, optimized for experimentation and prototyping
- **Megatron**: Large-scale distributed training with comprehensive parallelism support (TP, CP, PP, DP) for production workloads

**Architectures**:

- **DiT** (Diffusion Transformer): Transformer-based diffusion models for video generation
- **WAN**: Flow Matching architecture for alternative training dynamics
- **EDM** (Elucidating Diffusion Models): Improved diffusion training with better convergence

**Video Processing**:

- VAE encoding for latent space representation
- Tokenization pipelines for efficient video data handling
- Support for variable-length videos and diverse resolutions

**Distributed Training**:

- Tensor parallelism (TP) for splitting model layers across GPUs
- Context parallelism (CP) for long-sequence training
- Pipeline parallelism (PP) for splitting models across stages
- Data parallelism (DP) for scaling batch sizes

## Learn Core Concepts

Understand the foundational concepts before training or deploying video diffusion models.

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

Understand how DFM represents video data: latents, VAE encoding, tokenization, and data formats.
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
