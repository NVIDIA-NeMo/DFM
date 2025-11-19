---
description: "End-to-end Megatron quickstart: prepare data, train, and generate videos"
categories: ["getting-started", "megatron"]
tags: ["quickstart", "tutorial", "megatron"]
personas: ["mle-focused"]
difficulty: "intermediate"
content_type: "tutorial"
---

(gs-megatron)=

# Megatron Workflow

Complete end-to-end tutorial for pretraining a DiT model and generating videos using NeMo DFM's Megatron approach.

:::{card}

**Goal**: Pretrain a DiT model from scratch with manual control over distributed training parallelism (TP/PP/CP/DP).

^^^

**In this tutorial, you will**:

1. Prepare dataset (Smithsonian Butterflies: `huggan/smithsonian_butterflies_subset`)
2. Train DiT model with custom parallelism configuration
3. Generate videos from trained checkpoint

**Time**: 1-2 hours (depending on training duration)

:::

:::{button-ref} gs-index
:color: secondary
:outline:
:ref-type: ref

← Back to Get Started
:::

## Before You Start

Make sure you have completed:

- ✅ [Installation](installation.md)
- ✅ Multi-GPU setup (minimum: 2 GPUs, recommended: 8+ GPUs)
- ✅ ~50GB storage for dataset and checkpoints

---

(gs-megatron-prepare-data-section)=

## Prepare Dataset

Convert the Smithsonian Butterflies dataset from Hugging Face into webdataset format for Megatron training.

**Dataset**: `huggan/smithsonian_butterflies_subset` (~800 images with captions)

**Requirements**: ~10GB free storage, internet connection for download

### 1. Verify Dependencies

1. Ensure required packages are installed:

   ```bash
   pip install pandas webdataset transformers mediapy
   ```

2. Check the preparation script exists:

   ```bash
   ls -l examples/megatron/recipes/dit/prepare_energon_dataset_butterfly.py
   ```

### 2. Run Data Preparation

:::: {tab-set}

::: {tab-item} Single GPU

```bash
cd /opt/DFM  # Or your DFM installation path

torchrun --nproc-per-node 1 \
    examples/megatron/recipes/dit/prepare_energon_dataset_butterfly.py \
    --output-dir butterfly_webdataset
```

**Processing time**: ~30 minutes

:::

::: {tab-item} Multi-GPU

```bash
torchrun --nproc-per-node 4 \
    examples/megatron/recipes/dit/prepare_energon_dataset_butterfly.py \
    --output-dir butterfly_webdataset
```

**Processing time**: ~8 minutes (each GPU processes a subset in parallel)

:::

::::

### 3. Verify Dataset

Check that webdataset shards were created:

```bash
ls -lh butterfly_webdataset/
```

Expected: `rank0-000000.tar`, `rank1-000000.tar`, etc. Each tar contains ~200 samples with `.pth` (latents), `.pickle` (text embeddings), and `.json` (metadata) files.

:::{dropdown} Inspect Sample Format
:icon: info

```python
import webdataset as wds

dataset = wds.WebDataset("butterfly_webdataset/rank0-000000.tar")
sample = next(iter(dataset))

print(sample.keys())  # ['__key__', '.pth', '.pickle', '.json']
```

Each sample: `.pth` (image latents), `.pickle` (T5 embeddings), `.json` (metadata)
:::

### Troubleshooting

:::{dropdown} Out of Memory During Preparation
:icon: alert

```text
RuntimeError: CUDA out of memory
```

**Solution**: Use more GPUs to split work:

```bash
torchrun --nproc-per-node 8 \
    examples/megatron/recipes/dit/prepare_energon_dataset_butterfly.py \
    --output-dir butterfly_webdataset
```
:::

---

(gs-megatron-training-section)=

## Train DiT Model

Pretrain a Diffusion Transformer (DiT) model on the butterfly dataset using Megatron's distributed training.

**Requirements**: Prepared dataset from [data preparation](#gs-megatron-prepare-data-section), ~50GB storage for checkpoints

### 1. Understand Configuration Layers

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

### 2. Run Training with Defaults

Start training using default configuration:

```bash
cd /opt/DFM  # Or your DFM installation path

torchrun --nproc-per-node 2 \
    examples/megatron/recipes/dit/pretrain_dit_model.py \
    --dataset-path "/path/to/butterfly_webdataset"
```

#### Expected Output

```text
[INFO] Megatron-Bridge DiT Pretraining Script with YAML & CLI Overrides
[INFO] Loaded base configuration
[INFO] Starting pretraining...
[INFO] Iteration    1/10000, Loss: 0.456
[INFO] Iteration    2/10000, Loss: 0.442
[INFO] Iteration  100/10000, Loss: 0.312
[INFO] Checkpoint saved: checkpoints/dit_butterfly/iter_2000/
```

### 3. Custom Configuration

1. Create YAML override file `dit_butterfly_config.yaml`:

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

2. Run with custom configuration:

   ```bash
   torchrun --nproc-per-node 2 \
       examples/megatron/recipes/dit/pretrain_dit_model.py \
       --config-file dit_butterfly_config.yaml \
       --dataset-path "/path/to/butterfly_webdataset"
   ```

3. Add CLI overrides (optional):

   ```bash
   torchrun --nproc-per-node 4 \
       examples/megatron/recipes/dit/pretrain_dit_model.py \
       --config-file dit_butterfly_config.yaml \
       --dataset-path "/path/to/butterfly_webdataset" \
       model.tensor_model_parallel_size=4 \
       train.global_batch_size=128
   ```

   **Result**: `tensor_model_parallel_size=4` overrides the YAML value of `2`.

### Configuration Parameters

:::{dropdown} Key Parameters
:icon: info

**Training**: `train.global_batch_size` (32-128), `train.micro_batch_size` (1-4), `train.train_iters` (5000-10000), `optimizer.lr` (1e-4 to 5e-4)

**Parallelism**: `model.tensor_model_parallel_size` (TP, power of 2), `model.pipeline_model_parallel_size` (PP), `model.context_parallel_size` (CP). DP computed as `num_gpus / (TP * PP * CP)`

**Checkpointing**: `checkpoint.save_interval` (default: 2000), `checkpoint.checkpoint_dir`, `checkpoint.load_checkpoint`
:::

### Monitor Training

Monitor console output for decreasing loss values and checkpoint saves.

Verify checkpoints are being saved:

```bash
ls -lh checkpoints/dit_butterfly/
```

Expected: `iter_0001000/`, `iter_0002000/` directories with `model_weights.pt` and `optimizer_states.pt` files.

:::{dropdown} Resume from Checkpoint
:icon: repeat

Resume training from a saved checkpoint:

```bash
torchrun --nproc-per-node 2 \
    examples/megatron/recipes/dit/pretrain_dit_model.py \
    --dataset-path "/path/to/butterfly_webdataset" \
    checkpoint.load_checkpoint=/path/to/checkpoints/dit_butterfly/iter_5000/
```

Training continues from iteration 5000.
:::

### Troubleshooting

:::{dropdown} Out of Memory Errors
:icon: alert

```text
RuntimeError: CUDA out of memory
```

**Solution**: Reduce batch size:

```bash
torchrun --nproc-per-node 2 \
    examples/megatron/recipes/dit/pretrain_dit_model.py \
    --dataset-path "/path/to/butterfly_webdataset" \
    train.micro_batch_size=1 \
    train.global_batch_size=32
```
:::

---

(gs-megatron-inference-section)=

## Generate Videos

Generate videos from your trained DiT model checkpoint using Megatron inference.

**Generation time**: 3-8 minutes per video (depends on resolution and steps)

**Requirements**: Trained checkpoint from [training](#gs-megatron-training-section), Cosmos tokenizer for video decoding

### 1. Prepare Model Checkpoint

The inference script expects a consolidated `model.pth` file. Training saves checkpoints in `checkpoints/dit_butterfly/iter_5000/` with `model.pth` and `extra_state.pt` files.

:::{dropdown} Consolidate Sharded Checkpoint (If Needed)
:icon: alert

If your checkpoint is distributed across multiple files, consolidate:

```python
import torch

checkpoint = {}
for i in range(num_gpus):
    shard = torch.load(f"checkpoints/iter_5000/model_rank_{i}.pt")
    checkpoint.update(shard)

torch.save(checkpoint, "model.pth")
```
:::

### 2. Run Inference

:::: {tab-set}

::: {tab-item} Basic Generation

```bash
cd /opt/DFM  # Or your DFM installation path

torchrun --nproc-per-node 2 \
    examples/megatron/recipes/dit/inference_dit_model.py \
    --prompt "A beautiful monarch butterfly with orange and black wings" \
    --height 704 \
    --width 1280 \
    --num-video-frames 121 \
    --video-save-path butterfly_monarch.mp4
```

**Note**: The script requires `model.pth` in the current directory (line 247). Update path if needed.

:::

::: {tab-item} Custom Settings

```bash
torchrun --nproc-per-node 2 \
    examples/megatron/recipes/dit/inference_dit_model.py \
    --prompt "A blue morpho butterfly in a rainforest" \
    --height 704 \
    --width 1280 \
    --num-video-frames 121 \
    --num-steps 50 \
    --guidance 9.0 \
    --seed 42 \
    --cp-size 2 \
    --video-save-path morpho_rainforest.mp4
```

**Additional parameters**: `--num-steps` (default: 35), `--guidance` (default: 7), `--seed`, `--cp-size`

:::

::::

### Generation Parameters

**Required**: `--prompt`, `--height` (divisible by 16), `--width` (divisible by 16), `--num-video-frames` (common: 61, 121, 241), `--video-save-path`

**Optional**: `--num-steps` (default: 35), `--guidance` (default: 7.0), `--seed` (default: 1), `--cp-size`

### 3. View Generated Video

Check that video was created:

```bash
ls -lh idx=0_rank=0_butterfly_monarch.mp4
```

**Note**: Megatron inference adds prefix `idx={i}_rank={rank}_` to filename.

### Troubleshooting

:::{dropdown} Model Loading Error
:icon: alert

```text
FileNotFoundError: model.pth not found
```

**Solution**: Verify checkpoint path in script (line 247) or copy `model.pth` to working directory:

```bash
cp checkpoints/dit_butterfly/iter_5000/model.pth .
```
:::

:::{dropdown} Out of Memory Errors
:icon: alert

```text
RuntimeError: CUDA out of memory
```

**Solution**: Reduce resolution and frames:

```bash
--height 480 --width 848 --num-video-frames 61
```
:::
