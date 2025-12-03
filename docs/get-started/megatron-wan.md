---
description: "End-to-end Megatron WAN quickstart: prepare data, train, and generate videos"
categories: ["getting-started", "megatron"]
tags: ["quickstart", "tutorial", "megatron", "wan"]
personas: ["mle-focused"]
difficulty: "intermediate"
content_type: "tutorial"
---

(gs-megatron-wan)=

# Megatron WAN Workflow

Complete end-to-end tutorial for pretraining a WAN 2.1 model and generating videos using NeMo DFM's Megatron approach.

:::{card}

**Goal**: Pretrain a WAN 2.1 model from scratch with manual control over distributed training parallelism (TP/PP/CP/DP).

^^^

**In this tutorial, you will**:

1. Prepare WAN dataset (raw videos → WebDataset shards)
2. Train WAN model with custom parallelism configuration
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

## Overview

WAN 2.1 is an open-source implementation of large-scale text-to-video/image generative models built on top of Megatron-Core and Megatron-Bridge. It supports advanced parallelism strategies (data, tensor, sequence, and context parallelism) and optimized kernels (e.g., Transformer Engine fused attention).

**Currently Supported**: WAN 2.1 Text-to-Video (1.3B and 14B models)

---

(gs-megatron-wan-prepare-data-section)=

## Prepare WAN Dataset

Convert raw videos into WAN-ready WebDataset format for Megatron training.

**Requirements**: Raw `.mp4` videos with corresponding `.json` metadata files containing captions

### 1. Set Up Dataset Paths

```bash
# Define your input (raw videos) and output (WebDataset shards) folders
DATASET_SRC=/opt/raw_videos            # contains .mp4 and per-video .jsonl captions
DATASET_PATH=/opt/wan_webdataset      # output WebDataset shards
```

### 2. Optional: Hugging Face Token

If your WAN models require authentication on first download:

```bash
export HF_TOKEN=<your_huggingface_token>
```

### 3. Create WAN Shards

WAN's VAE encoder and T5 encoder extract videos' latents and caption embeddings offline before training:

```bash
uv run --group megatron-bridge python -m torch.distributed.run --nproc-per-node 1 \
  examples/megatron/recipes/wan/prepare_energon_dataset_wan.py \
  --video_folder "${DATASET_SRC}" \
  --output_dir "${DATASET_PATH}" \
  --model "Wan-AI/Wan2.1-T2V-1.3B-Diffusers" \
  --height 480 --width 832 \
  --center-crop
```

**Key arguments**:
- `--height/--width`: Control resize target (832×480 is supported for both 1.3B and 14B models)
- `--center-crop`: Run center crop to exact target size after resize

### 4. Process with Energon

Use Energon to process shards and create metadata:

```bash
energon prepare "${DATASET_PATH}"
```

**Interactive prompts**:
- Enter a train/val/test split, e.g., "8,1,1"
- When asked for the sample type, choose: **"Crude sample (plain dict for cooking)"**

### 5. What Gets Produced

Each shard contains:
- `pth`: WAN video latents
- `pickle`: Text embeddings
- `json`: Useful side-info (text caption, sizes, processing choices, etc.)

Energon writes a `.nv-meta` directory with dataset info and a `dataset.yaml` you can version/control.

You're ready to launch training. In the training config, point the WAN config (or CLI overrides) to the processed data output directory as `dataset.path=${DATASET_PATH}`.

:::{dropdown} Quick Start with Mock Dataset
:icon: beaker

If you want to run without a real dataset (for debugging or performance measurement), pass `--mock`:

```bash
uv run --group megatron-bridge python -m torch.distributed.run --nproc-per-node $num_gpus \
  examples/megatron/recipes/wan/pretrain_wan.py \
  --config-file examples/megatron/recipes/wan/conf/wan_1_3B.yaml \
  --training-mode pretrain \
  --mock
```

You may adjust mock shapes (`F_latents`, `H_latents`, `W_latents`) and packing behavior (`number_packed_samples`) in `WanMockDataModuleConfig` (see `dfm/src/megatron/recipes/wan/wan.py`) to simulate different data scenarios.
:::

---

(gs-megatron-wan-training-section)=

## Train WAN Model

Pretrain a WAN 2.1 model using Megatron's distributed training.

**Requirements**: Prepared dataset from [data preparation](#gs-megatron-wan-prepare-data-section), ~50GB storage for checkpoints

### Sequence Packing for WAN

This recipe leverages sequence packing to maximize throughput. When batches contain videos with different shapes or resolutions, naive batching and padding require significant padded tokens. Sequence packing stacks multiple samples (with different resolutions) into a single sequence instead of padding; hence no computation is wasted on padded tokens.

**Requirements**:
- Set `train.micro_batch_size=1` and `dataset.micro_batch_size=1`
- Ensure `model.qkv_format=thd` (required with context parallelism and recommended with sequence packing)

### Training Mode Presets

The script exposes a `--training-mode` with `pretrain` and `finetune` presets for flow-matching hyperparameters as a starting point for experiments.

:::{dropdown} Understanding Training Mode Presets
:icon: info

**Pretraining preset** (`--training-mode pretrain`):
- Uses noisier, biased sampling (e.g., logit-normal, higher logit_std, lower flow_shift)
- Purpose: Stability and broad learning

**Finetuning preset** (`--training-mode finetune`):
- Uses uniform, lower-noise settings (e.g., uniform sampling, lower logit_std, higher flow_shift)
- Purpose: Refine details and improve quality
:::

### 1. Prepare Configuration

We provide example configs for running 1.3B and 14B model sizes on mock dataset (see `wan_1_3B.yaml` and `wan_14B.yaml` under `examples/megatron/recipes/wan/conf`).

Copy and edit one of the example configs:

```bash
cp examples/megatron/recipes/wan/conf/wan_1_3B.yaml examples/megatron/recipes/wan/conf/my_wan.yaml
```

Edit `my_wan.yaml` to set:
- `dataset.path`: Path to your WebDataset directory
- `train.global_batch_size/micro_batch_size`: Keep micro_batch_size=1
- `model.tensor_model_parallel_size` / `model.context_parallel_size`: Based on GPUs
- `checkpoint.save` and `checkpoint.load`: Checkpoint directory

:::{note}
Users can learn more about argument details at [Megatron-Bridge docs](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/main/docs/megatron-lm-to-megatron-bridge.md).
:::

### 2. Run Training

```bash
uv run --group megatron-bridge python -m torch.distributed.run --nproc-per-node $num_gpus \
  examples/megatron/recipes/wan/pretrain_wan.py \
  --training-mode pretrain \
  --config-file examples/megatron/recipes/wan/conf/my_wan.yaml
```

### 3. CLI Overrides

You can also override any config values from the command line:

```bash
uv run --group megatron-bridge python -m torch.distributed.run --nproc-per-node $num_gpus \
  examples/megatron/recipes/wan/pretrain_wan.py \
  --config-file examples/megatron/recipes/wan/conf/my_wan.yaml \
  --training-mode pretrain \
  dataset.path=/opt/wan_webdataset \
  train.global_batch_size=8 \
  train.micro_batch_size=1 \
  model.tensor_model_parallel_size=2 \
  model.context_parallel_size=4 \
  checkpoint.save=/opt/pretrained_checkpoints \
  checkpoint.load=/opt/pretrained_checkpoints
```

:::{note}
If you use `logger.wandb_project` and `logger.wandb_exp_name`, export `WANDB_API_KEY`.
:::

### Monitor Training

Monitor console output for decreasing loss values and checkpoint saves.

Verify checkpoints are being saved:

```bash
ls -lh /opt/pretrained_checkpoints/
```

Expected: `iter_0001000/`, `iter_0002000/` directories with checkpoint files.

---

(gs-megatron-wan-inference-section)=

## Generate Videos

Generate videos from your trained WAN model checkpoint.

**Requirements**: Trained checkpoint from [training](#gs-megatron-wan-training-section)

### Run Inference

After training, run inference with `examples/megatron/recipes/wan/inference_wan.py`:

```bash
uv run --group megatron-bridge python -m torch.distributed.run --nproc-per-node 1 \
  examples/megatron/recipes/wan/inference_wan.py \
  --task t2v-1.3B \
  --frame_nums 81 \
  --sizes 480*832 \
  --checkpoint_dir /opt/pretrained_checkpoints \
  --checkpoint_step 10000 \
  --prompts "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage." \
  --sample_steps 50
```

**Parameters**:
- `--checkpoint_step`: Use specific checkpoint for inference
- `--sizes`: Specify video shape (height*width, e.g., 480*832)
- `--frame_nums`: Specify number of frames (e.g., 81)
- `--sample_steps`: Number of noise diffusion steps (default: 50)

:::{note}
Current inference path is single-GPU. Parallel inference is not yet supported.
:::

---

## Parallelism Support

The table below shows current parallelism support for different WAN model sizes:

| Model | Data Parallel | Tensor Parallel | Sequence Parallel | Context Parallel | FSDP |
|-------|---------------|-----------------|-------------------|------------------|------|
| 1.3B | ✅ | ✅ | ✅ | ✅ | Coming Soon |
| 14B  | ✅ | ✅ | ✅ | ✅ | Coming Soon |

## Related Tutorials

- [Megatron DiT Tutorial](megatron.md) - DiT model training workflow
- [Automodel Tutorial](automodel.md) - Fine-tune models with automatic parallelism

## Related Documentation

- [Training Paradigms](../about/concepts/training-paradigms.md) - Understand Megatron approach
- [Distributed Training](../about/concepts/distributed-training.md) - Parallelism strategies
- [Performance Benchmarks](../reference/performance.md) - Training throughput metrics

---

## References

WAN Team. (2025). Wan: Open and advanced large-scale video generative models (WAN 2.1). GitHub. https://github.com/Wan-Video/Wan2.1/

