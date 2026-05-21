---
description: "Understanding NeMo DFM's configuration system: YAML files, CLI overrides, and configuration precedence"
categories: ["concepts-architecture"]
tags: ["configuration", "yaml", "cli", "overrides"]
personas: ["mle-focused", "data-scientist-focused"]
difficulty: "beginner"
content_type: "explanation"
---

(about-concepts-configuration)=

# Configuration System

NeMo DFM uses a layered configuration system: base recipes provide defaults, YAML files define reusable settings, and CLI overrides enable quick experimentation. Each layer overrides the previous, with CLI arguments taking highest precedence.

## Configuration Layers

Configuration precedence: Base Recipe < YAML File < CLI Overrides

1. **Base recipes**: Python functions with framework defaults
2. **YAML files**: Reusable configuration templates
3. **CLI overrides**: Runtime argument overrides (highest precedence)

## Automodel Configuration

Automodel is a separate training framework in DFM that uses a simplified, YAML-first configuration approach. It requires the Automodel submodule from `3rdparty/Automodel`.

### YAML-Based Configuration

Automodel uses a single YAML file for all configuration:

```yaml
seed: 42

model:
  pretrained_model_name_or_path: Wan-AI/Wan2.1-T2V-1.3B-Diffusers

data:
  dataloader:
    _target_: Automodel.datasets.build_wan21_dataloader
    meta_folder: /path/to/dataset/meta/
    batch_size: 1
    num_workers: 2

batch:
  batch_size_per_node: 8

training:
  num_epochs: 100

optim:
  learning_rate: 5e-6
  optimizer:
    weight_decay: 0.01
    betas: [0.9, 0.999]

fsdp:
  tp_size: 1
  cp_size: 1
  pp_size: 1
  dp_size: 8
```

### Loading Configuration

Load configuration using Automodel's argument parser:

```python
# From Automodel package (3rdparty/Automodel)
from nemo_automodel.components.config._arg_parser import parse_args_and_load_config

cfg = parse_args_and_load_config("config.yaml")
```

The `nemo_automodel` package is provided by the Automodel submodule in `3rdparty/Automodel`.

## Megatron Configuration

### Multi-Level Configuration

Megatron supports three configuration levels:

#### 1. Base Recipe Configuration

Python functions define base configurations:

```python
from dfm.src.megatron.recipes.dit.dit import pretrain_config

cfg = pretrain_config(dataset_path="/path/to/dataset", mock=False)
```

#### 2. YAML Override Files

YAML files override base configuration:

```yaml
model:
  tensor_model_parallel_size: 4
train:
  global_batch_size: 512
```

#### 3. CLI Overrides

Command-line arguments override everything:

```bash
python pretrain_dit_model.py \
  --config-file config.yaml \
  model.tensor_model_parallel_size=8 \
  train.global_batch_size=1024
```

## CLI Override Syntax

### Basic Syntax

```bash
key=value
```

### Nested Keys

Use dot notation for nested configuration:

```bash
model.tensor_model_parallel_size=4
train.global_batch_size=512
optimizer.learning_rate=1e-4
```

### Adding New Keys

Use `+` prefix to add new configuration keys:

```bash
+new_key=value
+model.custom_setting=42
```

### Removing Keys

Use `~` prefix to remove configuration keys:

```bash
~key_to_remove
~model.unused_setting
```

### Type Conversion

CLI overrides automatically convert types:

```bash
model.tensor_model_parallel_size=4        # int
train.learning_rate=1e-4                  # float
model.use_mixed_precision=true            # bool
model.model_name="my_model"               # string
```

### Complex Types

PyTorch types use string representations that are parsed by OmegaConf:

```bash
model.pipeline_dtype=torch.bfloat16       # torch dtype (common: torch.float16, torch.bfloat16, torch.float32)
```

For function references and complex objects, define them in YAML files rather than CLI overrides.

## Configuration Structure

Configuration files organize settings into logical sections:

**Model**: Architecture and parallelism

```yaml
model:
  tensor_model_parallel_size: 4
  pipeline_model_parallel_size: 2
  pipeline_dtype: torch.bfloat16
```

**Training**: Batch sizes and iteration control

```yaml
train:
  global_batch_size: 512
  max_steps: 10000
  save_interval: 1000
```

**Data**: Dataset paths and loading

```yaml
data:
  dataset_path: /path/to/data
  num_workers: 8
```

**Optimizer**: Learning rates and schedules

```yaml
optim:
  learning_rate: 1e-4
  weight_decay: 0.01
```

## Configuration Patterns

### Experiment Workflows

Base configuration with CLI variations:

```bash
# Base run
python train.py --config-file base_config.yaml

# Learning rate sweep
python train.py --config-file base_config.yaml train.learning_rate=2e-4
python train.py --config-file base_config.yaml train.learning_rate=5e-4

# Scale model parallelism
python train.py --config-file base_config.yaml \
  model.tensor_model_parallel_size=8 \
  model.pipeline_model_parallel_size=2
```

### Verify Final Configuration

Print merged configuration in Megatron to verify all overrides:

```python
from megatron.bridge.utils.common_utils import get_rank_safe

if get_rank_safe() == 0:
    cfg.print_yaml()
```

This displays the final configuration after all merging, showing effective values for model, training, data, and optimizer settings.

## Environment Variables

Set runtime behavior with environment variables:

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3  # Select GPUs
export NCCL_DEBUG=INFO               # Debug distributed communication
```

