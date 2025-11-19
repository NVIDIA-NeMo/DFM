---
description: "Understanding the two training paradigms in NeMo DFM: Automodel and Megatron, and when to use each"
categories: ["concepts-architecture"]
tags: ["training", "automodel", "megatron", "paradigms"]
personas: ["mle-focused", "data-scientist-focused"]
difficulty: "beginner"
content_type: "explanation"
---

(about-concepts-training-paradigms)=

# Training Paradigms

NeMo DFM offers two training paradigms: **Automodel** for quick prototyping and fine-tuning, and **Megatron** for large-scale production training. Each paradigm uses different configuration systems, parallelism strategies, and data loading approaches.

## Overview

Choose between two approaches based on your training goal:

| Paradigm | Best For | Complexity | Configuration | Example |
|----------|----------|------------|---------------|---------|
| **Automodel** | Quick prototyping, fine-tuning, research | Lower | YAML-based recipes | `finetune.py` |
| **Megatron** | Large-scale pretraining, production training | Higher | Python recipes + YAML + CLI | `pretrain_dit_model.py` |

## Understanding the Paradigms

### Key Features

Each paradigm takes a different approach to configuration, parallelism, and data loading. Understanding these differences helps you choose the right paradigm for your training workflow.

::::{tab-set}
:sync-group: paradigm

:::{tab-item} Automodel
:sync: automodel

Automodel provides recipe-based training that abstracts distributed training complexity behind a single YAML configuration file. Pre-built recipes handle model initialization, data loading, and training loops automatically.

**Configuration**: Single YAML file controls all training parameters. The recipe provides sensible defaults, and you override only what you need to change.

**Parallelism**: FSDP2 automatically distributes training across GPUs using tensor parallelism (TP), context parallelism (CP), pipeline parallelism (PP), and data parallelism (DP). You configure parallelism strategy in the `fsdp` section without managing low-level details.

**Data Loading**: Uses PyTorch DataLoader with standard dataset interfaces. Works with common formats like images, text, and Hugging Face datasets.

**Model Integration**: Works directly with Hugging Face Diffusers models, making fine-tuning pre-trained models straightforward.
:::

:::{tab-item} Megatron
:sync: megatron

Megatron provides explicit control over every aspect of distributed training, from parallelism dimensions to data loading pipelines. Built for large-scale pretraining, it supports multi-node clusters with thousands of GPUs and custom model architectures.

**Configuration**: Three-level configuration system provides maximum flexibility:

1. Base recipe (Python) defines training logic and default parameters
2. YAML override files modify specific parameters for experiments
3. CLI overrides (highest precedence) enable quick parameter sweeps

This layered approach supports Hydra-style syntax for complex configuration changes.

**Parallelism**: Explicit control over all parallelism dimensions. You specify tensor parallel size, context parallel size, pipeline parallel stages, and data parallel degree independently. This fine-grained control enables optimal scaling for different model architectures and cluster configurations.

**Data Loading**: Uses Energon data loader with webdataset format, optimized for distributed training at scale. Supports efficient data streaming across nodes and advanced features like sample reweighting and mixing multiple datasets.

**Model Customization**: Full access to model architecture, forward pass logic, and training step. You define custom `ForwardStep` functions and modify model components directly.
:::
::::

### Use Cases

Your training goal determines which paradigm fits best. Here are the scenarios where each paradigm excels.

::::{tab-set}
:sync-group: paradigm

:::{tab-item} Automodel
:sync: automodel

- **Fine-tuning**: Adapt pre-trained models to your dataset
- **Research prototyping**: Test ideas quickly without infrastructure overhead
- **Small-scale training**: Single-node or small multi-node setups
- **Standard architectures**: Using existing model recipes without customization
:::

:::{tab-item} Megatron
:sync: megatron

- **Large-scale pretraining**: Training foundation models from scratch on multi-node clusters
- **Production workflows**: Reproducible training with version-controlled configurations
- **Custom architectures**: Implementing novel model designs not available in standard recipes
- **Performance optimization**: Tuning parallelism and memory usage for specific hardware
- **Multi-stage training**: Complex workflows with different training phases
:::
::::

### Architecture

Both paradigms organize code into layers that separate configuration from execution. The layer structure reflects each paradigm's design philosophy.

::::{tab-set}
:sync-group: paradigm

:::{tab-item} Automodel
:sync: automodel

Automodel uses a three-layer architecture:

1. **Recipe layer**: Pre-built training recipes (such as `TrainWan21DiffusionRecipe`) encapsulate training logic
2. **Config layer**: YAML files specify hyperparameters, data paths, and parallelism
3. **Execution layer**: `recipe.run_train_validation_loop()` handles training iteration
:::

:::{tab-item} Megatron
:sync: megatron

Megatron uses a modular architecture with clear separation of concerns:

1. **Recipe layer**: Base Python configuration (`pretrain_config()`) defines model, optimizer, and training parameters
2. **Override layer**: YAML files and CLI arguments modify base configuration
3. **Execution layer**: `pretrain()` function orchestrates distributed training with custom forward steps
4. **Bridge layer**: Megatron-Bridge handles low-level distributed training mechanics
:::
::::

## Comparing the Paradigms

The paradigms differ fundamentally in how they balance ease of use against control and scalability.

::::{tab-set}
:sync-group: paradigm

:::{tab-item} Automodel
:sync: automodel

**Configuration**: Single YAML file with recipe defaults

**Parallelism**: Automatic FSDP2 (less control)

**Data Loading**: PyTorch DataLoader, standard formats

**Scalability**: Small multi-node

**Setup Complexity**: Low

**Customization**: Recipe-level only

**Best For**: Quick experiments, fine-tuning
:::

:::{tab-item} Megatron
:sync: megatron

**Configuration**: Python base + YAML overrides + CLI

**Parallelism**: Explicit TP/CP/PP/DP (full control)

**Data Loading**: Energon data loader with distributed streaming

**Scalability**: Large multi-node clusters

**Setup Complexity**: High

**Customization**: Full code-level access

**Best For**: Large-scale pretraining, production
:::
::::

### Configuration Systems

::::{tab-set}
:sync-group: paradigm

:::{tab-item} Automodel
:sync: automodel

Uses a single YAML file where you specify training parameters. The recipe provides defaults for most settings, so you only override what matters for your experiment. Configuration is simple and flat.
:::

:::{tab-item} Megatron
:sync: megatron

Uses a three-level system: start with a Python recipe that defines base configuration, override specific parameters with YAML files for experiments, and apply final tweaks via CLI for parameter sweeps. This complexity enables reproducible experiments with version control.
:::
::::

### Parallelism Strategies

::::{tab-set}
:sync-group: paradigm

:::{tab-item} Automodel
:sync: automodel

Automatically configures FSDP2 to distribute your model across GPUs. You specify high-level parallelism settings in the `fsdp` section, and the framework determines optimal shard placement. This works well for standard model architectures.
:::

:::{tab-item} Megatron
:sync: megatron

Requires you to explicitly set tensor parallel size, context parallel size, pipeline stages, and data parallel degree. This granular control enables optimal memory usage and communication patterns for very large models or custom architectures.
:::
::::

### Data Loading Pipelines

::::{tab-set}
:sync-group: paradigm

:::{tab-item} Automodel
:sync: automodel

Uses PyTorch DataLoader with standard Python datasets. This familiar interface works with images, text files, and Hugging Face datasets without preprocessing.
:::

:::{tab-item} Megatron
:sync: megatron

Uses the Energon data loader optimized for distributed training at scale. This loader enables efficient streaming of massive datasets across nodes and supports advanced features like deterministic sampling and dataset mixing.
:::
::::

## Selecting Your Paradigm

Your training goal determines which paradigm to use.

::::{tab-set}
:sync-group: paradigm

:::{tab-item} Automodel
:sync: automodel

**Fine-tuning existing models**: Automodel integrates directly with Hugging Face models and provides pre-built fine-tuning recipes.

**Research experiments**: Quick iteration with YAML-only configuration changes. Test hypotheses in minutes instead of hours.

**Small-scale training**: Training on single-node or small multi-node setups where automatic parallelism configuration works well.

**Standard architectures**: Using proven model architectures without custom modifications.
:::

:::{tab-item} Megatron
:sync: megatron

**Pretraining foundation models**: Large-scale training from scratch where Energon's data loading efficiency and explicit parallelism control are essential.

**Production deployments**: Reproducible training with version-controlled Python recipes and configuration overrides.

**Custom model architectures**: Implementing novel designs that require code-level modifications to model structure and training steps.

**Performance-critical training**: Optimizing memory usage and communication patterns for specific hardware configurations.

**Large clusters**: Training on large multi-node clusters where explicit parallelism management becomes necessary.
:::
::::

## Paradigm Interoperability

Model checkpoints from one paradigm can often be loaded in the other, but training workflows are not interchangeable. The paradigms use different:

- **Configuration formats**: YAML-only versus Python + YAML + CLI
- **Data formats**: PyTorch datasets versus webdataset
- **Parallelism APIs**: FSDP2 versus explicit Megatron parallelism

Plan to use one paradigm consistently throughout your project. Converting training infrastructure between paradigms requires rewriting configuration and data loading code.

**Inference**: Both paradigms can export models to standard formats for inference deployment.
