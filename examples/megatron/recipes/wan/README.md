## 🚀 Megatron WAN

### 📋 Overview
An open-source implementation of [WAN 2.1](https://github.com/Wan-Video/Wan2.1) (large-scale text-to-video/image generative models) built on top of [Megatron-Core](https://github.com/NVIDIA/Megatron-LM) and [Megatron-Bridge](https://github.com/NVIDIA-NeMo/Megatron-Bridge)for scalable and efficient training. It supports advanced parallelism strategies (data, tensor, sequence, and context parallelism) and optimized kernels (e.g., Transformer Engine fused attention).

---

### 📦 Dataset Preparation
This recipe uses NVIDIA's Megatron-Energon as an efficient multi-modal data loader. Datasets should be in the WebDataset-compatible format (typically sharded `.tar` archives). Energon supports large-scale distributed loading, sharding, and sampling for video-text and image-text pairs.

- Set `dataset.path` to your WebDataset directory or shard pattern.
- See Megatron-Energon docs for format details, subflavors, and advanced options.

If you do not have a dataset yet or only need to validate performance/plumbing, see the "Quick Start with Mock Dataset" section below.

---

#### 🗂️ Dataset Preparation Example
Starting with a directory containing raw .mp4 videos and their corresponding metadata .json files containing captions, we’ll turn the data into WAN-ready WebDataset shards using our helper script, and then ask Energon to process those shards and create its metadata. After this, you can point `dataset.path` at the output folder and start training.

```bash
# 1) Define your input (raw videos) and output (WebDataset shards) folders. For example:
DATASET_SRC=/opt/raw_videos            # contains .mp4 and  per-video .jsonl captions
DATASET_PATH=/opt/wan_webdataset      # output WebDataset shards

# 2) (Optional) If your WAN models require auth on first download
export HF_TOKEN=<your_huggingface_token>

# 3) Create WAN shards with latents + text embeddings
# Wan's VAE encoder and T5 encoder is used to extract videos' latents and caption embeddings
#    --height/--width: arguments control resize target (832x480 is one supported option for 1.3B and 14B model)
#    --center-crop: arguments for center crop to exact target size after resize
uv run --group megatron-bridge python -m torch.distributed.run --nproc-per-node $num_gpus \
  examples/megatron/recipes/wan/prepare_energon_dataset_wan.py \
  --video_folder "${DATASET_SRC}" \
  --output_dir "${DATASET_PATH}" \
  --model "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"" \
  --height 480 --width 832 \
  --center-crop

# 4) Ask Energon to process shards and create its metadata/spec
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

Next steps:
- Point your WAN config (or CLI overrides) at `dataset.path=${DATASET_PATH}`
- You’re ready to launch pretraining

---

### 🐳 Build Container

Please follow the instructions in the container section of the main README:

- DFM container guide: https://github.com/NVIDIA-NeMo/DFM#-built-your-own-container

---

### 🏋️ Pretraining

This recipe leverages sequence packing to maximize throughput. When a batch containing videos with different shapes or resolution, naive batching and padding method require significant numner of padded tokens, due to the inherit size of videos. Sequence packing stacks multiple samples (with dirrent resolutions) into a single sequence instead of padding; hence no computation is wasted on padded tokens. When using sequence packing:
- Set `train.micro_batch_size=1` and `dataset.micro_batch_size=1`
- Ensure `model.qkv_format=thd` (required with context parallelism and recommended with sequence packing)

Multiple parallelism techniques including tensor, sequence, and context parallelism are supported and configurable per your hardware.

WAN training is driven by `examples/megatron/recipes/wan/pretrain_wan.py`, which supports both a YAML config file and CLI overrides. 

The script exposes a `--training-mode` with `pretrain` and `finetune` presets for flow-matching hyperparameters. As a starting point for experiments, this presets specify that pretraining uses noisier, biased sampling (e.g., logit-normal, higher logit_std, lower flow_shift) for stability and broad learning, while finetuning uses uniform, lower-noise settings (e.g., uniform sampling, lower logit_std, higher flow_shift) to refine details and improve quality.

Notes:
- If you use `logger.wandb_project` and `logger.wandb_exp_name`, export `WANDB_API_KEY`.
- Checkpointing is controlled via the `checkpoint.*` section. Use the same path for `save` and `load` to resume training.

#### Pretraining script example

We provide example scripts for running 1.3B and 14B model sizes on mock dataset (see `wan_1_3B.yaml` and `wan_14B.yaml` under `examples/megatron/recipes/wan/conf`). From these starting points, users can set their own configuration by copy one of the example override configs and update it with your settings (e.g., with actual processed data, and specific configurations based on available hardware):

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
NVTE_FUSED_ATTN=1 uv run --group megatron-bridge python -m torch.distributed.run --nproc-per-node $num_gpus \
  examples/megatron/recipes/wan/pretrain_wan.py \
  --training-mode pretrain \
  --config-file examples/megatron/recipes/wan/conf/my_wan.yaml
```

You can also override any config values from the command line. For example:

```bash
NVTE_FUSED_ATTN=1 uv run --group megatron-bridge python -m torch.distributed.run --nproc-per-node $num_gpus \
  examples/megatron/recipes/wan/pretrain_wan.py \
  --config-file examples/megatron/recipes/wan/conf/my_wan.yaml \
  --training-mode pretrain \
  dataset.path=/opt/wan_webdataset \
  train.global_batch_size=8 \
  train.micro_batch_size=1 \
  model.tensor_model_parallel_size=2 \
  model.context_parallel_size=4 \
  checkpoint.save=/opt/pretrained_checkpoint \
  checkpoint.load=/opt/pretrained_checkpoint
```

#### 🧪 Quick Start with Mock Dataset
If you want to run without a real dataset (for debugging or performance measurement), pass `--mock`:

```bash
NVTE_FUSED_ATTN=1 uv run --group megatron-bridge python -m torch.distributed.run --nproc-per-node $num_gpus \
  examples/megatron/recipes/wan/pretrain_wan.py \
  --config-file examples/megatron/recipes/wan/conf/wan_1_3B.yaml \
  --training-mode pretrain \
  --mock
```

You may adjust mock shapes (`F_latents`, `H_latents`, `W_latents`) and packing behavior (`number_packed_samples`) in `WanMockDataModuleConfig` (see `dfm/src/megatron/recipes/wan/wan.py`) to simulate different data scenarios.

---

### 🎬 Inference

After training, users can run inferencing with `examples/megatron/recipes/wan/inference_wan.py`. Set `--checkpoint_step` to use specific checkpoint for inference. Set `--sizes` and `--frame_nums` to specify video shape (frames, height, width). Set `--sample_steps` (default to 50) for number of noise diffusion steps.

```bash
NVTE_FUSED_ATTN=1 uv run --group megatron-bridge python -m torch.distributed.run --nproc-per-node 1 \
  examples/megatron/recipes/wan/inference_wan.py  \
  --task t2v-1.3B \
  --sizes 480*832 \
  --checkpoint_dir /path/to/checkpoint \
  --checkpoint_step 0 \
  --frame_nums 81 \
  --prompts "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage." \
  --sample_steps 50
```

Note: Current inference path is single-GPU. Parallel inference is not yet supported.

---

### ⚡ Parallelism Support

The table below shows current parallelism support for different model sizes:

| Model | Data Parallel | Tensor Parallel | Sequence Parallel | Context Parallel |
|---|---|---|---|---|
| 1.3B | ✅ | ✅ | ✅ | ✅ |
| 14B  | ✅ | ✅ | ✅ | ✅ |


### Citation
```bibtex
@article{wan2.1,
  title   = {Wan: Open and Advanced Large‐Scale Video Generative Models},
  author  = {Wan Team},
  year    = {2025},
  note    = {Open­source video foundation model series (Wan 2.1), https://github.com/Wan-Video/Wan2.1/}
}
```

