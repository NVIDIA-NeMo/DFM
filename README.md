<div align="center">

# NeMo DFM: Diffusion Foundation Models


<!-- We are still using Mbridge CICD NeMo. @pablo can we get our own? and the same for star gazer-->

<!-- Not includeing codecov for now since we have not worked on it extensively-->

[![CICD NeMo](https://github.com/NVIDIA-NeMo/Megatron-Bridge/actions/workflows/cicd-main.yml/badge.svg)](https://github.com/NVIDIA-NeMo/Megatron-Bridge/actions/workflows/cicd-main.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![GitHub Stars](https://img.shields.io/github/stars/NVIDIA-NeMo/DFM.svg?style=social&label=Star&cacheSeconds=14400)](https://github.com/NVIDIA-NeMo/DFM/stargazers/)

**State-of-the-art framework for fast, large-scale training and inference of diffusion models**

[Documentation](https://github.com/NVIDIA-NeMo/DFM/tree/main/docs) | [Supported Models](#supported-models) | [Examples](https://github.com/NVIDIA-NeMo/DFM/tree/main/examples) | [Contributing](https://github.com/NVIDIA-NeMo/Megatron-Bridge/tree/main/CONTRIBUTING.md)

</div>

## Overview

NeMo DFM (Diffusion Foundation Models) is a comprehensive collection of diffusion models for **Video**, **Image**, and **Text** generation. It unifies cutting-edge diffusion-based architectures and training techniques, prioritizing efficiency and performance from research prototyping to production deployment.

**Dual-Path Architecture**: DFM provides two complementary training paths to maximize flexibility:

- **🌉 Megatron Bridge Path**: Built on [Megatron Bridge](https://github.com/NVIDIA-NeMo/Megatron-Bridge) and [Megatron Core](https://github.com/NVIDIA/Megatron-LM/tree/main/megatron/core) for maximum scalability with tensor, pipeline, and context parallelism
- **🚀 AutoModel Path**: Built on [NeMo AutoModel](https://github.com/NVIDIA-NeMo/Automodel) for PyTorch DTensor-native SPMD training with seamless 🤗 Hugging Face integration

Choose the path that best fits your workflow—or use both for different stages of development!

<!-- Once we have updated images of how DFM fits into NeMo journey. Put them here. @Eliiot can help.-->
## 🔧 Installation

### 🐳 Built your own Container

#### 1. Build the container
```bash
# Initialize all submodules (Megatron-Bridge, Automodel, and nested Megatron-LM)
git submodule update --init --recursive

# Build the container
docker build -f docker/Dockerfile.ci -t dfm:dev .
```

#### 2. Start the container

```bash
docker run --rm -it --gpus all \
  --entrypoint bash \
  -v $(pwd):/opt/DFM -it dfm:dev
```



### 📦 Using DFM Docker (Coming Soon)

## ⚡ Quickstart

### Megatron Bridge Path

#### Run a Receipe
You can find all predefined recipes under [recipes](https://github.com/NVIDIA-NeMo/DFM/tree/main/examples/megatron/recipes) directory.

> **Note:** You will have to use [uv](https://docs.astral.sh/uv/) to run the recipes. Please use `--group` as `megatron-bridge`.


<!-- @Huy please update the below command after you change defaults-->

```bash
uv run --group megatron-bridge python -m torch.distributed.run --nproc_per_node=2 examples/megatron/recipes/wan/pretrain_wan.py model.qkv_format=thd --mock
```

### AutoModel Path

Train with PyTorch-native DTensor parallelism and direct 🤗 HF integration:

<!-- @Linnan, @Alex please add this thanks a ton-->
```bash
# TODO
# Fine-tune a video diffusion model with FSDP2
uv run torchrun --nproc-per-node=8 \
  dfm/src/automodel/recipes/finetune.py \
  --config examples/automodel/wan21_finetune.yaml

# Override parameters via CLI
# TODO
uv run torchrun --nproc-per-node=8 \
  dfm/src/automodel/recipes/finetune.py \
  --config examples/automodel/wan21_finetune.yaml \
  --step_scheduler.local_batch_size 4 \
  --model.pretrained_model_name_or_path "your-model-id"
```

## 🚀 Key Features

### Dual Training Paths

- **Megatron Bridge Path**
  - 🔄 Bidirectional HuggingFace ↔ Megatron checkpoint conversion
  - 🎯 Advanced parallelism: Tensor (TP), Pipeline (PP), Context (CP), Expert (EP)
  - 📈 Near-linear scalability to thousands of nodes
  - 🔧 Production-ready recipes with optimized hyperparameters

- **AutoModel Path**
  - 🌐 PyTorch DTensor-native SPMD training
  - 🔀 FSDP2-based Hybrid Sharding Data Parallelism (HSDP)
  - 📦 Sequence packing for efficient training
  - 🎨 Minimal ceremony with YAML-driven configs

### Shared Capabilities

- **🎥 Multi-Modal Diffusion**: Support for video, image, and text generation
- **🔬 Advanced Samplers**: EDM, Flow Matching, and custom diffusion schedules
- **🎭 Flexible Architectures**: DiT (Diffusion Transformers), WAN (World Action Networks)
- **📊 Efficient Data Loading**: Data pipelines with sequence packing
- **💾 Distributed Checkpointing**: SafeTensors-based sharded checkpoints
- **🌟 Memory Optimization**: Gradient checkpointing, mixed precision, efficient attention

## Supported Models

DFM provides out-of-the-box support for state-of-the-art diffusion architectures:

| Model | Type | Megatron Bridge | AutoModel | Description |
|-------|------|-----------------|-----------|-------------|
| **DiT** | Image/Video | [pretrain](https://github.com/NVIDIA-NeMo/DFM/blob/main/examples/megatron/recipes/dit/pretrain_dit_model.py), [inference](https://github.com/NVIDIA-NeMo/DFM/blob/main/examples/megatron/recipes/dit/inference_dit_model.py)  | 🔜 | Diffusion Transformers with scalable architecture |
| **WAN 2.1** | Video | [inference](https://github.com/NVIDIA-NeMo/DFM/blob/main/examples/megatron/recipes/wan/inference_wan.py), [pretrain, finetune](https://github.com/NVIDIA-NeMo/DFM/blob/main/examples/megatron/recipes/wan/pretrain_wan.py), conversion(@Huy) | @Linnan, @Alex | World Action Networks for video generation |

## Performance Benchmarking

For detailed performance benchmarks including throughput metrics across different GPU systems and model configurations, see the (Performance Summary)[https://github.com/NVIDIA-NeMo/DFM/blob/main/docs/performance-summary.md] in our documentation.

## Project Structure

```
DFM/
├── dfm/
│   └── src/
│       ├── megatron/              # Megatron Bridge path
│       │   ├── base/              # Base utilities for Megatron
│       │   ├── data/              # Data loaders and task encoders
│       │   │   ├── common/        # Shared data utilities
│       │   │   ├── <model_name>/  # model-specific data handling
│       │   ├── model/             # Model implementations
│       │   │   ├── common/        # Shared model components
│       │   │   ├── <model_name>/  # model-specific implementations
│       │   └── recipes/           # Training recipes
│       │       ├── <model_name>/  # model-specific training configs
│       ├── automodel (@linnan, @alex)/             # AutoModel path (DTensor-native)
│       │   ├── _diffusers/        # Diffusion pipeline integrations
│       │   ├── datasets/          # Dataset implementations
│       │   ├── distributed/       # Parallelization strategies
│       │   ├── flow_matching/     # Flow matching implementations
│       │   ├── recipes/           # Training scripts
│       │   └── utils/             # Utilities and validation
│       └── common/                # Shared across both paths
│           ├── data/              # Common data utilities
│           └── utils/             # Batch ops, video utils, etc.
├── examples/                      # Example scripts and configs
```

## 🎯 Choosing Your Path

| Feature | Megatron Bridge | AutoModel |
|---------|-----------------|-----------|
| **Best For** | Maximum scale (1000+ GPUs) | Flexibility & fast iteration |
| **Parallelism** | TP, PP, CP, EP, VPP | FSDP2, TP, SP, CP |
| **HF Integration** | Via bridge/conversion | PyTorch-native DTensor |
| **Checkpoint Format** | Megatron + HF export | SafeTensors DCP |
| **Learning Curve** | Steeper (more knobs) | Gentler (YAML-driven) |
| **Performance** | Highest at scale | Excellent, pytorch-native |

**Recommendation**:
- Start with **AutoModel** for quick prototyping and HF model compatibility
- Move to **Megatron Bridge** when scaling to 100+ GPUs or need advanced parallelism
- Use **both**: prototype with AutoModel, scale with Megatron Bridge!


## 🤝 Contributing

We welcome contributions! Please see our Contributing Guide for details on:

- Setting up your development environment
- Code style and testing guidelines
- Submitting pull requests
- Reporting issues

For questions or discussions, please open an issue on GitHub.

## Acknowledgements

NeMo DFM builds upon the excellent work of:

- [Megatron-LM](https://github.com/NVIDIA/Megatron-LM) - Advanced model parallelism
- [Megatron Bridge](https://github.com/NVIDIA-NeMo/Megatron-Bridge) - HuggingFace ↔ Megatron bridge
- [NeMo AutoModel](https://github.com/NVIDIA-NeMo/Automodel) - PyTorch-native SPMD training
- [PyTorch Distributed](https://pytorch.org/docs/stable/distributed.html) - Foundation for distributed training
- [Diffusers](https://github.com/huggingface/diffusers) - Diffusion model implementations
