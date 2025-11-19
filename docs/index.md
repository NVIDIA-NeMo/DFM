---
description: "NeMo DFM is a state-of-the-art framework for fast, large-scale training and inference of video world models using diffusion-based and autoregressive techniques"
categories:
  - documentation
  - home
tags:
  - diffusion-models
  - video-generation
  - large-scale-training
  - distributed
personas:
  - Data Scientists
  - Machine Learning Engineers
  - Cluster Administrators
  - DevOps Professionals
difficulty: beginner
content_type: index
modality: universal
---

(dfm-home)=

# NeMo DFM Documentation

Welcome to the NeMo DFM documentation.

## Introduction to NeMo DFM

Learn about NeMo DFM, how it works at a high-level, and the key features.

::::{grid} 1 2 2 2
:gutter: 1 1 1 2

:::{grid-item-card} {octicon}`book;1.5em;sd-mr-1` About NeMo DFM
:link: about-overview
:link-type: ref
Overview of NeMo DFM and its capabilities.
+++
{bdg-secondary}`target-users` {bdg-secondary}`how-it-works`
:::

:::{grid-item-card} {octicon}`book;1.5em;sd-mr-1` Concepts
:link: about-concepts
:link-type: ref
Explore the core concepts for diffusion models, architectures, and training in NeMo DFM.
+++
{bdg-secondary}`architectures` {bdg-secondary}`training` {bdg-secondary}`data-handling`
:::

::::

## Get Started

Install NeMo DFM and choose your training path: Automodel for quick prototyping or Megatron for large-scale training.

::::::{grid} 1 1 1 1

:::::{grid-item-card} {octicon}`package;1.5em;sd-mr-1` 1. Installation
:link: gs-installation
:link-type: ref

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
:link: gs-automodel
:link-type: ref

Fine-tune pretrained models with automatic parallelism. Best for quick prototyping.
+++
{bdg-secondary}`automodel` {bdg-success}`Fast start` {bdg-primary}`Data scientists`
:::

:::{grid-item-card} {octicon}`server;1.5em;sd-mr-1` 2b. Megatron Tutorial
:link: gs-megatron
:link-type: ref

Train from scratch with full distributed control. Best for large-scale training.
+++
{bdg-secondary}`megatron` {bdg-info}`Full control` {bdg-primary}`MLEs`
:::

::::
:::::

::::::

---

::::{toctree}
:hidden:
Home <self>
::::

::::{toctree}
:hidden:
:caption: About NeMo DFM
:maxdepth: 1
about/index.md
about/concepts/index.md
::::

::::{toctree}
:hidden:
:caption: Get Started
:maxdepth: 2

get-started/index.md
Installation <get-started/installation>
Automodel <get-started/automodel>
Megatron <get-started/megatron>
::::

::::{toctree}
:hidden:
:caption: Reference
:maxdepth: 2

About References <reference/index.md>
apidocs/index.rst
::::
