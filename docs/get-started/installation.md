---
description: "Installation guide for NeMo DFM"
categories: ["getting-started"]
tags: ["installation", "setup", "prerequisites"]
personas: ["mle-focused", "admin-focused"]
difficulty: "beginner"
content_type: "how-to"
---

(gs-installation)=

# Installation Quickstart

Set up your environment for training and inference with NeMo DFM. This guide covers three installation methods: Docker (recommended), pip, and source.

## Prerequisites

Verify you have the following before installation:

- **Python 3.10 or later**: Check with `python --version`
- **NVIDIA GPU with CUDA support**: Required for training and inference
- **Docker** (recommended): Provides pre-configured environment with all dependencies
- **Git**: Required for cloning the repository

## Installation Methods

Choose the installation method that best fits your use case:

::::{tab-set}

:::{tab-item} Docker (Recommended)

Docker installation provides a pre-configured environment with all dependencies. This method is recommended for development and testing.

1. Clone the repository:

   ```bash
   git clone https://github.com/NVIDIA-NeMo/DFM.git
   cd DFM
   ```

2. Initialize submodules:

   ```bash
   git submodule update --init --recursive
   ```

3. Build the container:

   ```bash
   docker build -f docker/Dockerfile.ci -t dfm:latest .
   ```

4. Run the container:

   ```bash
   docker run --gpus all -v $(pwd):/opt/DFM -it dfm:latest bash
   ```

5. Install DFM inside the container:

   The Docker image includes all dependencies installed during build. For development, install DFM in editable mode:

   ```bash
   source /opt/venv/bin/activate
   uv pip install --no-deps -e .
   ```

   ```{note}
   The `--no-deps` flag prevents reinstalling dependencies that are already installed during the Docker build process. This step is only needed for editable development installs.
   ```

The Docker image includes:

- PyTorch 25.09 with CUDA support
- All required dependencies (accelerate, diffusers, megatron-energon)
- Pre-configured virtual environment

:::

:::{tab-item} Pip

Install NeMo DFM directly from the repository using pip.

1. Clone the repository:

   ```bash
   git clone https://github.com/NVIDIA-NeMo/DFM.git
   cd DFM
   ```

2. Initialize submodules:

   ```bash
   git submodule update --init --recursive
   ```

3. Install dependencies:

   ```bash
   pip install -e .
   ```

**Optional** Install with extra features:

```bash
# Install with Automodel support
pip install -e ".[automodel]"

# Install with Megatron-Bridge support
pip install -e ".[megatron-bridge]"
```

:::

:::{tab-item} Source

For development or custom configurations, install from source.

1. Clone the repository:

   ```bash
   git clone https://github.com/NVIDIA-NeMo/DFM.git
   cd DFM
   ```

2. Initialize submodules:

   ```bash
   git submodule update --init --recursive
   ```

3. Create a virtual environment:

   ```bash
   python3.10 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

4. Install build dependencies:

   ```bash
   pip install -e ".[build]"
   ```

5. Install DFM:

   ```bash
   pip install -e .
   ```

:::

::::

## Verify Installation

Confirm your installation succeeded by running these verification checks.

::::{tab-set}

:::{tab-item} Python Import

```python
import dfm
print("DFM installed successfully!")
```

:::

:::{tab-item} GPU Availability

```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU count: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"GPU name: {torch.cuda.get_device_name(0)}")
```

:::

:::{tab-item} Package Version

```python
import dfm
print(f"DFM version: {dfm.__version__}")
```

:::

::::

## Core Dependencies

All core dependencies install automatically with NeMo DFM:

- `accelerate`: Distributed training acceleration
- `diffusers==0.35.1`: Hugging Face Diffusers library for diffusion models
- `easydict`: Dictionary access utilities
- `ftfy`: Text encoding fixes
- `megatron-energon`: Megatron-based efficient data loading
- `imageio`, `imageio-ffmpeg`: Video I/O operations
- `opencv-python-headless==4.10.0.84`: Image processing without GUI dependencies

### Optional Dependencies

Install these with extras flags:

- `nemo-automodel`: Automodel support via `pip install -e ".[automodel]"`
- `megatron-bridge`: Megatron-Bridge support via `pip install -e ".[megatron-bridge]"`

## Next Steps

Now that installation is complete:

1. **[Run training](gs-training)**: Start your first training job with sample data
2. **[Generate videos](gs-inference)**: Use pre-trained models for inference
3. **[Learn core concepts](about-concepts)**: Understand DiT, WAN, and EDM architectures
