---
description: "Comprehensive guide for training production-scale text-to-video models with WAN 2.1, WebDataset preparation, and inference workflows"
categories: ["tutorials", "megatron"]
tags: ["training", "recipe", "megatron", "wan", "advanced"]
personas: ["mle-focused"]
difficulty: "intermediate"
content_type: "tutorial"
---

(tutorial-text-to-video-training)=

# Text-to-Video Training

Comprehensive guide for training large-scale text-to-video generation models using WAN 2.1 architecture. This approach uses Megatron-Core and Megatron-Bridge for scalable training with advanced parallelism strategies (data, tensor, sequence, and context parallelism) and optimized kernels (e.g., Transformer Engine fused attention).

**Use case**: Train production-scale text-to-video models with full control over distributed training parallelism.

:::{note}
For a quick start guide, see [Megatron WAN Workflow](../get-started/megatron-wan.md). This tutorial provides detailed configuration options and advanced topics.
:::

---

## Dataset Preparation

This recipe uses NVIDIA's [Megatron-Energon](https://github.com/NVIDIA/Megatron-Energon) as an efficient multi-modal data loader. Datasets should be in the WebDataset-compatible format (typically sharded `.tar` archives). Energon supports large-scale distributed loading, sharding, and sampling for video-text and image-text pairs. Set `dataset.path` to your WebDataset directory or shard pattern. See Megatron-Energon docs for format details, subflavors, and advanced options.

If you do not have a dataset yet or only need to validate performance/plumbing, see the "Quick Start with Mock Dataset" section below.

### Dataset Preparation Example

Starting with a directory containing raw .mp4 videos and their corresponding .json metadata files containing captions, you can turn the data into WAN-ready WebDataset shards using our helper script. We then use Energon to process those shards and create its metadata. After this, you can set training script's `dataset.path` argument to the output processed data folder and start training.

```bash
# 1) Define your input (raw videos) and output (WebDataset shards) folders. For example:
DATASET_SRC=/opt/raw_videos            # contains .mp4 and  per-video .jsonl captions
DATASET_PATH=/opt/wan_webdataset      # output WebDataset shards

# 2) (Optional) If your WAN models require auth on first download
export HF_TOKEN=<your_huggingface_token>

# 3) Create WAN shards with latents + text embeddings
# Wan's VAE encoder and T5 encoder is used to extract videos' latents and caption embeddings offline before training, using the following core arguments:
#    --height/--width: control resize target (832x480 is supported for both 1.3B and 14B model)
#    --center-crop: run center crop to exact target size after resize
uv run --group megatron-bridge python -m torch.distributed.run --nproc-per-node 1 \
  examples/megatron/recipes/wan/prepare_energon_dataset_wan.py \
  --video_folder "${DATASET_SRC}" \
  --output_dir "${DATASET_PATH}" \
  --model "Wan-AI/Wan2.1-T2V-1.3B-Diffusers" \
  --height 480 --width 832 \
  --center-crop

# 4) Use Energon to process shards and create its metadata/spec
energon prepare "${DATASET_PATH}"
# In the interactive prompts:
# - Enter a train/val/test split, e.g., "8,1,1"
# - When asked for the sample type, choose: "Crude sample (plain dict for cooking)"
```

What gets produced:
- Each shard contains:
  - pth: contain WAN video latents
  - pickle: contain text embeddings
  - json: contain useful side-info (text caption, sizes, processing choices, etc.)
- Energon writes a `.nv-meta` directory with dataset info and a `dataset.yaml` you can version/control.

You're ready to launch training. In the training config, we will point the WAN config (or CLI overrides) to the processed data output directory as `dataset.path=${DATASET_PATH}`.

---

## Build Container

Please follow the instructions in the container section of the main README:

- DFM container guide: https://github.com/NVIDIA-NeMo/DFM#-built-your-own-container

---

## Pretraining

This recipe leverages sequence packing to maximize throughput. When a batch containing videos with different shapes or resolution, naive batching and padding method require significant number of padded tokens, due to the inherent size of videos. Sequence packing stacks multiple samples (with different resolutions) into a single sequence instead of padding; hence no computation is wasted on padded tokens. When using sequence packing:
- Set `train.micro_batch_size=1` and `dataset.micro_batch_size=1`
- Ensure `model.qkv_format=thd` (required with context parallelism and recommended with sequence packing)

Multiple parallelism techniques including tensor, sequence, and context parallelism are supported and configurable per your hardware.

Wan training is driven by `examples/megatron/recipes/wan/pretrain_wan.py`, which supports both a YAML config file and CLI overrides.

The script exposes a `--training-mode` with `pretrain` and `finetune` presets for flow-matching hyperparameters as a starting point for experiments. This presets specify that pretraining uses noisier, biased sampling (e.g., logit-normal, higher logit_std, lower flow_shift) for stability and broad learning, while finetuning uses uniform, lower-noise settings (e.g., uniform sampling, lower logit_std, higher flow_shift) to refine details and improve quality.

**Notes**: If you use `logger.wandb_project` and `logger.wandb_exp_name`, export `WANDB_API_KEY`.

### Pretraining Script Example

We provide example scripts for running 1.3B and 14B model sizes on mock dataset (see `wan_1_3B.yaml` and `wan_14B.yaml` under `examples/megatron/recipes/wan/conf`). From these starting points, users can set their own configuration by copy one of the example override configs and update it with your settings (e.g., with actual processed data path, and specific configurations based on available hardware, etc.). Users can learn more about arguments detail at [Megatron-Bridge docs](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/main/docs/megatron-lm-to-megatron-bridge.md).

```bash
cp examples/megatron/recipes/wan/conf/wan_1_3B.yaml examples/megatron/recipes/wan/conf/my_wan.yaml
# Edit my_wan.yaml to set:
# - dataset.path: Path to your WebDataset directory
# - train.global_batch_size/micro_batch_size: Keep micro_batch_size=1
# - model.tensor_model_parallel_size / model.context_parallel_size: Based on GPUs
# - checkpoint.save and checkpoint.load: Checkpoint directory
```

Then run:

```bash
uv run --group megatron-bridge python -m torch.distributed.run --nproc-per-node $num_gpus \
  examples/megatron/recipes/wan/pretrain_wan.py \
  --training-mode pretrain \
  --config-file examples/megatron/recipes/wan/conf/my_wan.yaml
```

You can also override any config values from the command line. For example:

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

### Quick Start with Mock Dataset

If you want to run without a real dataset (for debugging or performance measurement), pass `--mock`:

```bash
uv run --group megatron-bridge python -m torch.distributed.run --nproc-per-node $num_gpus \
  examples/megatron/recipes/wan/pretrain_wan.py \
  --config-file examples/megatron/recipes/wan/conf/wan_1_3B.yaml \
  --training-mode pretrain \
  --mock
```

You may adjust mock shapes (`F_latents`, `H_latents`, `W_latents`) and packing behavior (`number_packed_samples`) in `WanMockDataModuleConfig` (see `dfm/src/megatron/recipes/wan/wan.py`) to simulate different data scenarios.

---

## Inference

After training, users can run inferencing with `examples/megatron/recipes/wan/inference_wan.py`. Set `--checkpoint_step` to use specific checkpoint for inference. Set `--sizes` and `--frame_nums` to specify video shape (frames, height, width). Set `--sample_steps` (default to 50) for number of noise diffusion steps.

```bash
uv run --group megatron-bridge python -m torch.distributed.run --nproc-per-node 1 \
  examples/megatron/recipes/wan/inference_wan.py  \
  --task t2v-1.3B \
  --frame_nums 81 \
  --sizes 480*832 \
  --checkpoint_dir /opt/pretrained_checkpoints \
  --checkpoint_step 10000 \
  --prompts "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage." \
  --sample_steps 50
```

**Note**: Current inference path is single-GPU. Parallel inference is not yet supported.

---

## Parallelism Support

The table below shows current parallelism support for different model sizes:

| Model | Data Parallel | Tensor Parallel | Sequence Parallel | Context Parallel | FSDP |
|---|---|---|---|---|---|
| 1.3B | ✅ | ✅ | ✅ | ✅ |Coming Soon|
| 14B  | ✅ | ✅ | ✅ | ✅ |Coming Soon|

---

## References

Wan Team. (2025). Wan: Open and advanced large-scale video generative models (WAN 2.1). GitHub. https://github.com/Wan-Video/Wan2.1/

---

## Related Documentation

- [Megatron WAN Quick Start](../get-started/megatron-wan.md) - Quick start guide
- [Training Paradigms](../about/concepts/training-paradigms.md) - Understanding Megatron approach
- [Performance Benchmarks](../reference/performance.md) - Training throughput metrics

