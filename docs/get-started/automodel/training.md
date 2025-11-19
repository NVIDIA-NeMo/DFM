---
description: "Fine-tune WAN2.1 video generation model with Automodel"
categories: ["getting-started", "automodel"]
tags: ["training", "fine-tuning", "how-to"]
personas: ["data-scientist-focused", "mle-focused"]
difficulty: "beginner"
content_type: "how-to"
---

(gs-automodel-training)=

# Fine-Tune WAN2.1 Model

Fine-tune the WAN2.1 text-to-video model using Automodel's recipe-based training approach.

## Goal

By the end of this guide, you'll have a fine-tuned WAN2.1 model checkpoint ready for video generation.

**Time**: 20-30 minutes setup + training time

## Prerequisites

Before starting:

- ✅ Complete [Installation](../installation.md)
- ✅ Multi-GPU system (recommended: 8 GPUs for optimal performance)
- ✅ Dataset prepared (see [Data Requirements](#data-requirements))
- ✅ Checkpoint storage location (`~50GB per checkpoint`)

## Overview

**What happens during training**:
1. Load pretrained WAN2.1 model from Hugging Face
2. Configure FSDP2 parallelism automatically
3. Train on your dataset with flow matching
4. Save checkpoints periodically

**Key concept**: Automodel handles parallelism automatically using FSDP2—no manual tensor or pipeline parallelism configuration needed.

## Step 1: Prepare Your Dataset

### Data Requirements

Automodel expects a dataset with:
- **Video files**: MP4, WebM, or similar
- **Text captions**: Descriptions for each video
- **Metadata**: Frame count, resolution, FPS

### Dataset Format

Create a custom dataloader or use the WAN2.1 format. Example structure:

```text
/path/to/dataset/
  meta/
    ├── 00000.json    # {"caption": "...", "video_path": "..."}
    ├── 00001.json
    └── ...
  videos/
    ├── 00000.mp4
    ├── 00001.mp4
    └── ...
```

### Example Dataloader

The training script uses a custom dataloader specified in the config:

```yaml
data:
  dataloader:
    _target_: Automodel.datasets.build_wan21_dataloader
    meta_folder: /path/to/your/dataset/meta/
    batch_size: 1
    num_workers: 2
```

## Step 2: Create Training Configuration

Create a YAML configuration file with your training parameters.

**Create** `wan2_1_finetune.yaml`:

```yaml
seed: 42

wandb:
  project: wan-t2v-finetuning
  mode: online
  name: wan2_1_finetuning_run_1

dist_env:
  backend: nccl
  timeout_minutes: 30

model:
  pretrained_model_name_or_path: Wan-AI/Wan2.1-T2V-1.3B-Diffusers

data:
  dataloader:
    _target_: Automodel.datasets.build_wan21_dataloader
    meta_folder: /path/to/your/dataset/meta/
    batch_size: 1
    num_workers: 2
    device: cpu

batch:
  batch_size_per_node: 8

training:
  num_epochs: 100

optim:
  learning_rate: 5e-6
  optimizer:
    weight_decay: 0.01
    betas: [0.9, 0.999]

flow_matching:
  use_sigma_noise: true
  timestep_sampling: uniform
  logit_mean: 0.0
  logit_std: 1.0
  flow_shift: 3.0
  mix_uniform_ratio: 0.1

fsdp:
  tp_size: 1
  cp_size: 1
  pp_size: 1
  dp_replicate_size: 1
  dp_size: 8

logging:
  save_every: 1000
  log_every: 2

checkpoint:
  enabled: true
  checkpoint_dir: /path/to/checkpoints/wan2_1_finetuning/
  model_save_format: torch_save
  save_consolidated: false
  restore_from: null
```

### Key Configuration Parameters

| Parameter | Description | Default | Recommended |
|-----------|-------------|---------|-------------|
| `model.pretrained_model_name_or_path` | Hugging Face model ID | Required | `Wan-AI/Wan2.1-T2V-1.3B-Diffusers` |
| `data.dataloader.meta_folder` | Dataset metadata location | Required | Your dataset path |
| `batch.batch_size_per_node` | Batch size per node | `8` | 4-8 (depends on GPU memory) |
| `training.num_epochs` | Training epochs | `100` | Adjust based on dataset size |
| `optim.learning_rate` | Learning rate | `5e-6` | 1e-6 to 1e-5 |
| `fsdp.dp_size` | Data parallel size | `8` | Match GPU count |
| `checkpoint.checkpoint_dir` | Where to save checkpoints | Required | Path with enough storage |
| `logging.save_every` | Checkpoint interval (iterations) | `1000` | 500-2000 |

**Parallelism settings** (`fsdp`):
- `tp_size=1`: Tensor parallelism disabled (automatic for this model size)
- `cp_size=1`: Context parallelism disabled
- `pp_size=1`: Pipeline parallelism disabled
- `dp_size=8`: Data parallelism across 8 GPUs

## Step 3: Run Training

Execute the training script with your configuration:

```bash
python dfm/examples/automodel/finetune/finetune.py /path/to/wan2_1_finetune.yaml
```

**Alternative**: Use the default configuration:

```bash
python dfm/examples/automodel/finetune/finetune.py
```

This uses the default config at `dfm/examples/automodel/finetune/wan2_1_t2v_flow.yaml`.

### What Happens During Training

1. **Initialization** (2-5 minutes):
   - Downloads WAN2.1 model from Hugging Face (if not cached)
   - Initializes FSDP2 parallelism across GPUs
   - Loads your dataset

2. **Training loop**:
   - Processes batches across distributed GPUs
   - Logs loss every `log_every` iterations
   - Saves checkpoints every `save_every` iterations

3. **Checkpoint saves**:
   - Checkpoints save to `checkpoint.checkpoint_dir`
   - Each checkpoint is ~50GB (model weights + optimizer states)

### Expected Output

```text
[INFO] Loading pretrained model: Wan-AI/Wan2.1-T2V-1.3B-Diffusers
[INFO] Initializing FSDP2 with dp_size=8
[INFO] Starting training loop...
[INFO] Epoch 1/100, Iter 1/5000, Loss: 0.234
[INFO] Epoch 1/100, Iter 2/5000, Loss: 0.221
...
[INFO] Checkpoint saved: /path/to/checkpoints/wan2_1_finetuning/iter_1000/
```

## Step 4: Monitor Training

### Check Training Logs

Monitor console output for:
- **Loss values**: Should decrease over time
- **Learning rate**: Follows scheduler (if configured)
- **Checkpoint saves**: Confirms periodic saving

### WandB Monitoring (Optional)

If `wandb.mode: online`, view metrics in WandB dashboard:
- Training loss over time
- Learning rate schedule
- GPU utilization

### Verify Checkpoints

Check that checkpoints are being saved:

```bash
ls -lh /path/to/checkpoints/wan2_1_finetuning/
```

Expected structure:

```text
/path/to/checkpoints/wan2_1_finetuning/
  ├── iter_1000/
  │   ├── model_weights.pt
  │   └── optimizer_states.pt
  ├── iter_2000/
  └── latest/
```

## Configuration Tips

### Reduce Memory Usage

If you encounter OOM errors:

```yaml
batch:
  batch_size_per_node: 4  # Reduce from 8

data:
  dataloader:
    batch_size: 1  # Keep at 1
```

### Speed Up Training

Enable tensor parallelism for large models:

```yaml
fsdp:
  tp_size: 2
  dp_size: 4  # Adjust to maintain tp_size * dp_size = GPU count
```

### Multi-Node Training

For multi-node setups, use the multi-node config:

```bash
python dfm/examples/automodel/finetune/finetune.py \
    dfm/examples/automodel/finetune/wan2_1_t2v_flow_multinode.yaml
```

Ensure nodes can communicate via NCCL.

## Troubleshooting

### Out of Memory Errors

```
RuntimeError: CUDA out of memory
```

**Solution**: Reduce `batch.batch_size_per_node`:

```yaml
batch:
  batch_size_per_node: 4  # or 2
```

### Data Loading Slow

**Solution**: Increase `data.dataloader.num_workers`:

```yaml
data:
  dataloader:
    num_workers: 4  # or 8
```

### Model Download Fails

**Solution**: Set Hugging Face cache directory:

```bash
export HF_HOME=/path/to/cache
python dfm/examples/automodel/finetune/finetune.py ...
```

### NCCL Errors

```
NCCL error: unhandled system error
```

**Solution**: Verify GPU communication:

```bash
nvidia-smi topo -m
```

Set NCCL debug mode:

```bash
export NCCL_DEBUG=INFO
python dfm/examples/automodel/finetune/finetune.py ...
```

## Next Steps

After training completes:

1. **[Generate videos](inference.md)** from your fine-tuned checkpoint
2. **Evaluate quality**: Compare generated videos to training data
3. **Iterate**: Adjust hyperparameters and retrain if needed

## Related Pages

- **[Automodel Inference](inference.md)**: Generate videos from your checkpoint
- **[Configuration Reference](../../about/concepts/configuration.md)**: Understand YAML configuration
- **[Distributed Training](../../reference/distributed-training.md)**: Deep dive into parallelism

