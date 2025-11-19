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

## Quickstarts

Install and run NeMo DFM for training and inference.

::::{grid} 1 2 2 2
:gutter: 1 1 1 2

:::{grid-item-card} {octicon}`rocket;1.5em;sd-mr-1` Installation Quickstart
:link: gs-installation
:link-type: ref
Set up your environment and install NeMo DFM.
:::

:::{grid-item-card} {octicon}`play;1.5em;sd-mr-1` Training Quickstart
:link: gs-index
:link-type: ref
Run your first video diffusion model training job.
:::

:::{grid-item-card} {octicon}`image;1.5em;sd-mr-1` Inference Quickstart
:link: gs-index
:link-type: ref
Generate videos using trained models.
:::

::::

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
