---
description: "Train video diffusion models with Automodel or Megatron approaches"
categories: ["getting-started"]
tags: ["training", "quickstart", "how-to"]
personas: ["mle-focused", "data-scientist-focused"]
difficulty: "beginner"
content_type: "how-to"
---

(gs-training)=

# Training Quickstart

Train video diffusion models using NeMo DFM with recipe-based (Automodel) or large-scale distributed (Megatron) approaches.

## Prerequisites

Complete these steps before training:

- [Installation Quickstart](gs-installation) - Install NeMo DFM
- Dataset in Energon or webdataset format
- Multi-GPU setup for distributed training

## Choose Your Approach

| Approach | Best For | Complexity |
|----------|----------|------------|
| **Automodel** | Quick prototyping, fine-tuning pretrained models | Lower |
| **Megatron** | Large-scale pretraining, full distributed control | Higher |

## Automodel Training

Automodel uses recipe-based training with YAML configuration and automatic FSDP2 parallelism.

### Fine-Tune WAN2.1 Model

**Step 1: Create configuration file**

Create a YAML configuration file with your training parameters:

```yaml
seed: 42

model:
  pretrained_model_name_or_path: Wan-AI/Wan2.1-T2V-1.3B-Diffusers

data:
  dataloader:
    _target_: Automodel.datasets.build_wan21_dataloader
    meta_folder: /path/to/your/dataset/meta/
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

checkpoint:
  enabled: true
  checkpoint_dir: /path/to/checkpoints/
  save_consolidated: false
```

**Step 2: Run training**

```bash
python dfm/examples/automodel/finetune/finetune.py /path/to/config.yaml
```

Omit the path to use the default configuration at `dfm/examples/automodel/finetune/wan2_1_t2v_flow.yaml`.

**Training process**:

1. `TrainWan21DiffusionRecipe` loads configuration and initializes model
2. FSDP2 parallelism applies automatically based on `fsdp` settings
3. Training loop executes with automatic checkpointing
4. Checkpoints save to `checkpoint.checkpoint_dir` at intervals defined by `logging.save_every`

## Megatron Training

Megatron training provides fine-grained control over distributed training for large-scale pretraining.

### Pretrain DiT Model

**Step 1: Prepare webdataset**

Organize your dataset in webdataset format with tar shards:

```text
/path/to/dataset/
  ├── shard_000000.tar
  ├── shard_000001.tar
  └── ...
```

**Step 2: Run distributed training**

```bash
torchrun --nproc-per-node 8 \
    examples/megatron/recipes/dit/pretrain_dit_model.py \
    --dataset-path "/path/to/your/dataset"
```

**With custom configuration and overrides**:

```bash
torchrun --nproc-per-node 8 \
    examples/megatron/recipes/dit/pretrain_dit_model.py \
    --config-file /path/to/config.yaml \
    --dataset-path "/path/to/your/dataset" \
    model.tensor_model_parallel_size=4 \
    train.global_batch_size=512
```

**Training process**:

1. `torchrun` initializes distributed environment across GPUs
2. Base configuration merges with YAML file overrides and CLI parameters
3. Energon data module loads webdataset shards
4. DiT model initializes with specified parallelism (TP+CP+PP+DP)
5. Training loop executes using `DITForwardStep`

### Configuration Overrides

Configure Megatron training using three layers with increasing precedence:

**Layer 1: Base configuration** (recipe defaults)

Built-in defaults from the training recipe.

**Layer 2: YAML file overrides** (`--config-file`)

```yaml
model:
  tensor_model_parallel_size: 4
train:
  global_batch_size: 512
```

**Layer 3: CLI overrides** (highest precedence)

```bash
model.tensor_model_parallel_size=4 train.global_batch_size=512
```

CLI parameters override YAML settings, which override recipe defaults.

## Monitor Training Progress

### Training Logs

Monitor console output for:

- **Loss values**: Per-iteration training loss
- **Learning rate**: Current LR from scheduler
- **Checkpoint saves**: Confirmation of saved checkpoints
- **Validation metrics**: Accuracy or loss metrics (if validation enabled)

### Checkpoints

Checkpoints save to the configured directory with this structure:

```text
checkpoints/
  ├── iter_1000/
  │   ├── model_weights.pt
  │   └── optimizer_states.pt
  ├── iter_2000/
  └── latest/
```

Each checkpoint contains:

- Model weights
- Optimizer states
- Training metadata (step count, epoch, RNG states)

## Key Configuration Parameters

### Automodel

| Parameter | Description | Default | Range |
|-----------|-------------|---------|-------|
| `batch.batch_size_per_node` | Batch size per node | `8` | 1-64 |
| `training.num_epochs` | Training epochs | `100` | 1+ |
| `optim.learning_rate` | Learning rate | `5e-6` | 1e-7 to 1e-3 |
| `fsdp.tp_size` | Tensor parallel size | `1` | 1, 2, 4, 8 |
| `fsdp.dp_size` | Data parallel size | `8` | 1+ |
| `checkpoint.save_every` | Checkpoint interval (iterations) | `1000` | 1+ |
| `logging.log_every` | Logging interval (iterations) | `2` | 1+ |

### Megatron

| Parameter | Description | Default | Range |
|-----------|-------------|---------|-------|
| `--nproc-per-node` | GPUs per node | `8` | 1-8 |
| `--dataset-path` | Webdataset directory path | Required | Valid path |
| `model.tensor_model_parallel_size` | Tensor parallel size | Varies | 1, 2, 4, 8 |
| `train.global_batch_size` | Global batch size across all GPUs | Varies | 1+ |

## Troubleshooting

### Out of Memory Errors

**Symptom**: `RuntimeError: CUDA out of memory`

**Solutions**:

1. Reduce `batch_size_per_node` or `global_batch_size`
2. Increase `gradient_accumulation_steps` to maintain effective batch size
3. Enable tensor parallelism: Set `fsdp.tp_size=2` or `fsdp.tp_size=4`
4. Enable pipeline parallelism: Set `fsdp.pp_size=2` for large models

### Data Loading Issues

**Symptom**: `FileNotFoundError` or slow data loading

**Solutions**:

1. Verify dataset format matches requirements (webdataset tar shards for Megatron, Energon format for both)
2. Check file permissions: `ls -l /path/to/dataset`
3. Increase `data.dataloader.num_workers` to 4-8 for faster loading
4. Verify dataset path in configuration matches actual location

### Distributed Training Errors

**Symptom**: `NCCL error` or training hangs at initialization

**Solutions**:

1. Verify NCCL installation: `python -c "import torch; print(torch.cuda.nccl.version())"`
2. For multi-node: Test network connectivity between nodes
3. Match `--nproc-per-node` to available GPUs: `nvidia-smi --list-gpus | wc -l`
4. Set environment variable: `export NCCL_DEBUG=INFO` for detailed NCCL logs

## Next Steps

After training:

- **[Inference Quickstart](gs-inference)**: Generate videos from your trained model
- **[Reference: Distributed Training](ref-distributed-training)**: Advanced distributed training configuration
- **[Reference: Data Loading](ref-data-loading)**: Dataset preparation and loading
