---
description: "Train DiT model on butterfly dataset with Megatron"
categories: ["getting-started", "megatron"]
tags: ["training", "pretraining", "how-to"]
personas: ["mle-focused"]
difficulty: "intermediate"
content_type: "how-to"
---

(gs-megatron-training)=

# Train DiT Model

Pretrain a Diffusion Transformer (DiT) model on the butterfly dataset using Megatron's distributed training.

## Goal

Train a DiT model from scratch with full control over distributed parallelism.

**Time**: 30 minutes setup + training time

## Prerequisites

- ✅ Complete [Installation](../installation.md)
- ✅ Prepared dataset from [data preparation](prepare-data.md)
- ✅ Multi-GPU system (minimum: 2 GPUs)
- ✅ ~50GB storage for checkpoints

## Overview

**What happens during training**:
1. Initialize distributed environment with `torchrun`
2. Load webdataset shards via Energon data module
3. Initialize DiT model with specified parallelism
4. Train using EDM (Elucidating Diffusion Models) pipeline
5. Save checkpoints periodically

**Key concept**: Megatron requires manual parallelism configuration (TP, CP, PP, DP) for maximum control and optimization.

## Step 1: Understand Configuration Layers

Megatron uses a **three-layer configuration system** with increasing precedence:

```yaml
Layer 1: Recipe defaults (pretrain_config() function)
  ↓
Layer 2: YAML file overrides (--config-file)
  ↓
Layer 3: CLI overrides (highest precedence)
```

**Example**:
```bash
torchrun pretrain_dit_model.py \
    --config-file my_config.yaml \  # Layer 2
    model.tensor_model_parallel_size=4  # Layer 3 overrides Layer 2
```

CLI parameters override YAML settings, which override recipe defaults.

## Step 2: Run Training with Defaults

Start training using default configuration:

```bash
cd /opt/DFM  # Or your DFM installation path

torchrun --nproc-per-node 2 \
    examples/megatron/recipes/dit/pretrain_dit_model.py \
    --dataset-path "/path/to/butterfly_webdataset"
```

**Command breakdown**:
- `torchrun --nproc-per-node 2`: Use 2 GPUs on this node
- `--dataset-path`: Path to your webdataset shards

### What Happens During Training

1. **Initialization** (1-2 minutes):
   - Initializes NCCL distributed backend
   - Loads DiT model configuration
   - Creates Energon data module for webdataset
   - Initializes model with parallelism settings

2. **Training loop**:
   - Loads batches from webdataset shards
   - Runs forward pass with EDM diffusion
   - Computes loss and backpropagates
   - Saves checkpoints at intervals

3. **Checkpoint saves**:
   - Saves model weights and optimizer states
   - Default interval: every 1000 iterations

### Expected Output

```text
[INFO] Megatron-Bridge DiT Pretraining Script with YAML & CLI Overrides
[INFO] Loaded base configuration
[INFO] Starting pretraining...
[INFO] Iteration    1/10000, Loss: 0.456
[INFO] Iteration    2/10000, Loss: 0.442
[INFO] Iteration  100/10000, Loss: 0.312
[INFO] Checkpoint saved: checkpoints/dit_butterfly/iter_1000/
```

## Step 3: Custom Configuration

### Create YAML Override File

Create `dit_butterfly_config.yaml`:

```yaml
# Model parallelism
model:
  tensor_model_parallel_size: 2
  pipeline_model_parallel_size: 1
  context_parallel_size: 1
  
# Training parameters
train:
  global_batch_size: 64
  micro_batch_size: 2
  train_iters: 10000
  
# Optimizer
optimizer:
  lr: 0.0001
  weight_decay: 0.01
  
# Checkpointing
checkpoint:
  save_interval: 500
  checkpoint_dir: /path/to/checkpoints/dit_butterfly/
```

### Run with Custom Configuration

```bash
torchrun --nproc-per-node 2 \
    examples/megatron/recipes/dit/pretrain_dit_model.py \
    --config-file dit_butterfly_config.yaml \
    --dataset-path "/path/to/butterfly_webdataset"
```

### Add CLI Overrides

Override specific parameters on command line:

```bash
torchrun --nproc-per-node 4 \
    examples/megatron/recipes/dit/pretrain_dit_model.py \
    --config-file dit_butterfly_config.yaml \
    --dataset-path "/path/to/butterfly_webdataset" \
    model.tensor_model_parallel_size=4 \
    train.global_batch_size=128
```

**Result**: `tensor_model_parallel_size=4` overrides the YAML value of `2`.

## Configuration Parameters

### Key Training Parameters

| Parameter | Description | Default | Recommended |
|-----------|-------------|---------|-------------|
| `--dataset-path` | Webdataset directory | Required | Path to butterfly_webdataset |
| `--nproc-per-node` | GPUs per node | Required | 2, 4, or 8 |
| `train.train_iters` | Training iterations | Varies | 5000-10000 |
| `train.global_batch_size` | Total batch across GPUs | Varies | 32-128 |
| `train.micro_batch_size` | Batch per GPU | Varies | 1-4 |
| `optimizer.lr` | Learning rate | Varies | 1e-4 to 5e-4 |

### Parallelism Parameters

| Parameter | Description | Constraint |
|-----------|-------------|------------|
| `model.tensor_model_parallel_size` (TP) | Model tensor split across GPUs | Power of 2 |
| `model.pipeline_model_parallel_size` (PP) | Model layer split across GPUs | 1+ |
| `model.context_parallel_size` (CP) | Sequence split across GPUs | 1+ |
| DP (Data Parallel) | Computed automatically | `DP = num_gpus / (TP * PP * CP)` |

**Example** (8 GPUs):
```yaml
TP: 2, PP: 1, CP: 1 → DP: 4
TP: 4, PP: 2, CP: 1 → DP: 1
```

### Checkpoint Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `checkpoint.save_interval` | Save every N iterations | `1000` |
| `checkpoint.checkpoint_dir` | Checkpoint save location | `checkpoints/` |
| `checkpoint.load_checkpoint` | Resume from checkpoint | `null` |

## Multi-Node Training

### Setup Multi-Node Environment

**Node 0** (master):

```bash
export MASTER_ADDR=node0.cluster.com
export MASTER_PORT=6000

torchrun --nproc-per-node 8 \
    --nnodes 2 \
    --node-rank 0 \
    --master-addr $MASTER_ADDR \
    --master-port $MASTER_PORT \
    examples/megatron/recipes/dit/pretrain_dit_model.py \
    --dataset-path "/shared/butterfly_webdataset"
```

**Node 1** (worker):

```bash
export MASTER_ADDR=node0.cluster.com
export MASTER_PORT=6000

torchrun --nproc-per-node 8 \
    --nnodes 2 \
    --node-rank 1 \
    --master-addr $MASTER_ADDR \
    --master-port $MASTER_PORT \
    examples/megatron/recipes/dit/pretrain_dit_model.py \
    --dataset-path "/shared/butterfly_webdataset"
```

**Requirements**:
- Nodes can communicate via network
- Shared filesystem for dataset and checkpoints
- NCCL configured correctly

## Monitor Training

### Training Logs

Monitor console output for:

```text
[INFO] Iteration 100/10000, Loss: 0.312, LR: 0.0001
[INFO] Iteration 200/10000, Loss: 0.289, LR: 0.00009
[INFO] Iteration 500/10000, Loss: 0.245, LR: 0.00007
[INFO] Checkpoint saved: checkpoints/dit_butterfly/iter_500/
```

**Key metrics**:
- **Loss**: Should decrease over time (expect 0.5 → 0.1 range)
- **LR**: Learning rate (may change if using scheduler)
- **Iteration speed**: ~1-3 seconds per iteration (depends on hardware)

### Verify Checkpoints

Check checkpoint structure:

```bash
ls -lh checkpoints/dit_butterfly/
```

Expected structure:

```text
checkpoints/dit_butterfly/
  ├── iter_0001000/
  │   ├── model_weights.pt
  │   └── optimizer_states.pt
  ├── iter_0002000/
  └── latest_checkpointed_iteration.txt
```

## Resume from Checkpoint

Resume training from a saved checkpoint:

```bash
torchrun --nproc-per-node 2 \
    examples/megatron/recipes/dit/pretrain_dit_model.py \
    --dataset-path "/path/to/butterfly_webdataset" \
    checkpoint.load_checkpoint=/path/to/checkpoints/dit_butterfly/iter_5000/
```

Training continues from iteration 5000.

## Troubleshooting

### Out of Memory Errors

```
RuntimeError: CUDA out of memory
```

**Solution 1**: Reduce batch size:

```bash
torchrun --nproc-per-node 2 \
    examples/megatron/recipes/dit/pretrain_dit_model.py \
    --dataset-path "/path/to/butterfly_webdataset" \
    train.micro_batch_size=1 \
    train.global_batch_size=32
```

**Solution 2**: Enable tensor parallelism:

```bash
torchrun --nproc-per-node 4 \
    examples/megatron/recipes/dit/pretrain_dit_model.py \
    --dataset-path "/path/to/butterfly_webdataset" \
    model.tensor_model_parallel_size=2
```

### NCCL Errors

```
NCCL error: unhandled system error
```

**Solution**: Check NCCL installation and GPU communication:

```bash
# Verify NCCL
python -c "import torch; print(torch.cuda.nccl.version())"

# Check GPU topology
nvidia-smi topo -m

# Enable NCCL debug logging
export NCCL_DEBUG=INFO
```

### Data Loading Slow

**Symptom**: Long delays between iterations

**Solution 1**: Check dataset location (prefer SSD over NFS)

**Solution 2**: Increase data loader workers (edit `pretrain_dit_model.py`):

```python
# Line ~143
data_module = EnergonDataModule(
    dataset_path=args.dataset_path,
    num_workers=8  # Increase from 4
)
```

### Loss Not Decreasing

**Symptom**: Loss stays constant or increases

**Solutions**:
1. **Check learning rate**: May be too high or too low
   ```bash
   optimizer.lr=0.0001  # Try 1e-4
   ```

2. **Verify data**: Ensure dataset loaded correctly
   ```bash
   # Check webdataset samples
   python -c "import webdataset as wds; print(next(iter(wds.WebDataset('butterfly_webdataset/rank0-000000.tar'))))"
   ```

3. **Check parallelism**: Ensure TP/PP/CP values are valid

## Next Steps

After training completes:

1. **[Generate videos](inference.md)** from your trained checkpoint
2. **Evaluate quality**: Compare generated samples to training data
3. **Scale up**: Train on larger datasets with more GPUs

## Related Pages

- **[Megatron Inference](inference.md)**: Generate from your checkpoint
- **[Distributed Training](../../about/concepts/distributed-training.md)**: Deep dive into parallelism
- **[Training Paradigms](../../about/concepts/training-paradigms.md)**: Compare Automodel vs. Megatron

