## 🚀 Megatron FLUX

### 📋 Overview
An open-source implementation of [FLUX.1](https://github.com/black-forest-labs/flux) (state-of-the-art text-to-image diffusion models) built on top of [Megatron-Core](https://github.com/NVIDIA/Megatron-LM) and [Megatron-Bridge](https://github.com/NVIDIA-NeMo/Megatron-Bridge) for scalable and efficient training. FLUX combines double (MMDiT-style) and single transformer blocks with flow matching for high-quality image generation. It supports advanced parallelism strategies (data, tensor) and optimized kernels (e.g., Transformer Engine fused attention).

---

### 📦 Dataset Preparation
This recipe uses NVIDIA's [Megatron-Energon](https://github.com/NVIDIA/Megatron-Energon) as an efficient multi-modal data loader. Datasets should be in the WebDataset-compatible format (typically sharded `.tar` archives). Energon supports large-scale distributed loading, sharding, and sampling for image-text pairs. Set `dataset.path` to your WebDataset directory or shard pattern. See Megatron-Energon docs for format details, subflavors, and advanced options.

If you do not have a dataset yet or only need to validate performance/plumbing, see the "Quick Start with Mock Dataset" section below.

---

#### 🗂️ Dataset Preparation Example
Starting with a directory containing raw images and their corresponding metadata files containing captions, you can turn the data into FLUX-ready WebDataset shards using our helper script. We then use Energon to process those shards and create its metadata. After this, you can set training script's `dataset.path` argument to the output processed data folder and start training.

```bash
# 1) Define your input (raw images) and output (WebDataset shards) folders. For example:
DATASET_SRC=/opt/raw_images            # contains images and per-image .json/.jsonl captions
DATASET_PATH=/opt/flux_webdataset      # output WebDataset shards

# 2) (Optional) If FLUX models require auth on first download
export HF_TOKEN=<your_huggingface_token>

# 3) Create FLUX shards with latents + text embeddings
# FLUX's VAE encoder, CLIP encoder, and T5 encoder are used to extract images' latents and caption embeddings offline before training, using the following core arguments:
#    --output_format: select output format of "automodel" or "energon"
#    --height/--width: control resize target (1024x1024 is standard for FLUX)
#    --center-crop: run center crop to exact target size after resize
#    --model: HuggingFace model path (e.g., "black-forest-labs/FLUX.1-dev")
uv run --group megatron-bridge python -m torch.distributed.run --nproc-per-node 8 \
  examples/megatron/recipes/flux/prepare_energon_dataset_flux.py \
  --data_folder "${DATASET_SRC}" \
  --output_folder "${DATASET_PATH}" \
  --output_format energon \
  --model "black-forest-labs/FLUX.1-dev" \
  --height 1024 \
  --width 1024 \
  --resize_mode bilinear \
  --center-crop

# 4) Use Energon to process shards and create its metadata/spec
energon prepare "${DATASET_PATH}"
# In the interactive prompts:
# - Enter a train/val/test split, e.g., "8,1,1"
# - When asked for the sample type, choose: "Crude sample (plain dict for cooking)"
```

What gets produced:
- Each shard contains:
  - pth: contain FLUX image latents
  - pickle: contain text embeddings (both T5 and CLIP pooled)
  - json: contain useful side-info (text caption, sizes, processing choices, etc.)
- Energon writes a `.nv-meta` directory with dataset info and a `dataset.yaml` you can version/control.

You're ready to launch training. In the training config, we will point the FLUX config (or CLI overrides) to the processed data output directory as `dataset.path=${DATASET_PATH}`.

---

### 🐳 Build Container

Please follow the instructions in the container section of the main README:

- DFM container guide: https://github.com/NVIDIA-NeMo/DFM#-built-your-own-container

---

### 🏋️ Pretraining

FLUX training is driven by `examples/megatron/recipes/flux/pretrain_flux.py`, which supports both a YAML config file and CLI overrides.

The script exposes flow matching hyperparameters through CLI flags including:
- `--timestep-sampling`: Timestep sampling strategy (logit_normal, uniform, mode)
- `--logit-mean` and `--logit-std`: Parameters for logit-normal sampling
- `--guidance-scale`: Classifier-free guidance scale for training
- `--scheduler-steps`: Number of diffusion timesteps

Multiple parallelism techniques including tensor are supported and configurable per your hardware.

**Notes**: If you use `logger.wandb_project` and `logger.wandb_exp_name`, export `WANDB_API_KEY`.

#### Pretraining script example

We provide example scripts and configurations for running FLUX model training. From these starting points, users can set their own configuration by copying one of the example override configs and update it with your settings (e.g., with actual processed data path, and specific configurations based on available hardware, etc.). Users can learn more about arguments detail at [Megatron-Bridge docs](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/main/docs/megatron-lm-to-megatron-bridge.md).

```bash
cp examples/megatron/recipes/flux/conf/flux_pretrain_override_example.yaml examples/megatron/recipes/flux/conf/my_flux.yaml
# Edit my_flux.yaml to set:
# - dataset.path: Path to your WebDataset directory
# - train.global_batch_size/micro_batch_size
# - model.tensor_model_parallel_size / model.context_parallel_size: Based on GPUs
# - model.num_joint_layers / model.num_single_layers: FLUX architecture
# - checkpoint.save and checkpoint.load: Checkpoint directory
```

Then run:

```bash
uv run --group megatron-bridge python -m torch.distributed.run --nproc-per-node $num_gpus \
  examples/megatron/recipes/flux/pretrain_flux.py \
  --config-file examples/megatron/recipes/flux/conf/my_flux.yaml \
  --timestep-sampling logit_normal \
  --logit-mean 0.0 \
  --logit-std 1.0 \
  --guidance-scale 3.5
```

You can also override any config values from the command line. For example:

```bash
uv run --group megatron-bridge python -m torch.distributed.run --nproc-per-node $num_gpus \
  examples/megatron/recipes/flux/pretrain_flux.py \
  --config-file examples/megatron/recipes/flux/conf/my_flux.yaml \
  dataset.path=/opt/flux_webdataset \
  train.global_batch_size=16 \
  train.micro_batch_size=2 \
  model.tensor_model_parallel_size=2 \
  model.context_parallel_size=1 \
  model.num_joint_layers=19 \
  model.num_single_layers=38 \
  checkpoint.save=/opt/pretrained_checkpoints \
  checkpoint.load=/opt/pretrained_checkpoints \
  --timestep-sampling logit_normal \
  --logit-mean 0.0 \
  --logit-std 1.0 \
  --guidance-scale 3.5
```

#### 🧪 Quick Start with Mock Dataset
If you want to run without a real dataset (for debugging or performance measurement), pass `--mock`:

```bash
uv run --group megatron-bridge python -m torch.distributed.run --nproc-per-node $num_gpus \
  examples/megatron/recipes/flux/pretrain_flux.py \
  --config-file examples/megatron/recipes/flux/conf/flux_pretrain_override_example.yaml \
  --mock
```

You may adjust mock shapes (`image_H`, `image_W`) in the config to simulate different data scenarios.

---

### 🔧 Fine-tuning

FLUX fine-tuning allows you to continue training from a pretrained checkpoint with your custom dataset. Fine-tuning typically uses lower learning rates and fewer training iterations compared to pretraining. The training iteration counter is reset to 0, regardless of the checkpoint's saved iteration.

FLUX fine-tuning is driven by `examples/megatron/recipes/flux/finetune_flux.py`, which supports both a YAML config file and CLI overrides, similar to the pretraining script.

#### Fine-tuning script example

The fine-tuning script requires a `--load-checkpoint` argument pointing to your pretrained checkpoint directory. You can point to either:
- A base checkpoint directory (loads the latest checkpoint)
- A specific iteration directory (e.g., `iter_0000000`)

```bash
uv run --group megatron-bridge python -m torch.distributed.run --nproc-per-node $num_gpus \
  examples/megatron/recipes/flux/finetune_flux.py \
  --load-checkpoint /opt/pretrained_checkpoints \
  --config-file examples/megatron/recipes/flux/conf/my_flux.yaml \
  dataset.path=/opt/finetune_webdataset \
  train.train_iters=5000 \
  optimizer.lr=5e-6 \
  --timestep-sampling logit_normal
```

You can also override any config values from the command line:

```bash
uv run --group megatron-bridge python -m torch.distributed.run --nproc-per-node $num_gpus \
  examples/megatron/recipes/flux/finetune_flux.py \
  --load-checkpoint /opt/pretrained_checkpoints/iter_0000000 \
  dataset.path=/opt/finetune_webdataset \
  train.global_batch_size=8 \
  train.micro_batch_size=1 \
  train.train_iters=10000 \
  optimizer.lr=1e-5 \
  model.tensor_model_parallel_size=2 \
  checkpoint.save=/opt/finetuned_checkpoints \
  --timestep-sampling logit_normal \
  --guidance-scale 3.5
```

**Notes**:
- Fine-tuning always starts from iteration 0, regardless of the checkpoint's saved iteration
- Optimizer and RNG states are not loaded (fresh optimizer and RNG)
- Use lower learning rates (typically 1e-5 to 5e-6) compared to pretraining
- Fewer training iterations are typically needed (e.g., 5,000-10,000 vs 50,000+ for pretraining)
- If you use `logger.wandb_project` and `logger.wandb_exp_name`, export `WANDB_API_KEY`

---

### 🎨 Inference

After training, users can run inference with `examples/megatron/recipes/flux/inference_flux.py`. Set `--flux_ckpt` to your trained checkpoint directory. Set `--height` and `--width` to specify image dimensions. Set `--num_inference_steps` (default 10) for number of denoising diffusion steps.

```bash
uv run --group megatron-bridge python -m torch.distributed.run --nproc-per-node 1 \
  examples/megatron/recipes/flux/inference_flux.py \
  --flux_ckpt /opt/pretrained_checkpoints/iter_0010000 \
  --vae_ckpt black-forest-labs/FLUX.1-dev \
  --t5_version google/t5-v1_1-xxl \
  --clip_version openai/clip-vit-large-patch14 \
  --prompts "A cat holding a sign that says hello world" \
  --prompts "A futuristic cityscape at sunset" \
  --height 1024 \
  --width 1024 \
  --num_inference_steps 20 \
  --guidance_scale 3.5 \
  --output_path ./flux_output
```

**Note**: Current inference path is single-GPU. Multi-GPU parallel inference is not yet supported.

---

### 🔄 Checkpoint Converting (optional)

If you plan to fine-tune FLUX using a pre-trained model, you can convert between HuggingFace and Megatron checkpoint formats. The provided script supports bidirectional conversion, allowing you to move between HuggingFace and Megatron formats as needed.

Follow these steps to convert your checkpoints:
```bash
# Download the HF checkpoint locally
huggingface-cli download black-forest-labs/FLUX.1-dev \
  --local-dir /root/.cache/huggingface/flux.1-dev \
  --local-dir-use-symlinks False

# Import a HuggingFace model to Megatron format
python examples/megatron/recipes/flux/conversion/convert_checkpoints.py import \
  --hf-model /root/.cache/huggingface/flux.1-dev \
  --megatron-path /workspace/checkpoints/megatron_checkpoints/flux_dev

# Export a Megatron checkpoint to HuggingFace format
python examples/megatron/recipes/flux/conversion/convert_checkpoints.py export \
  --hf-model /root/.cache/huggingface/flux.1-dev \
  --megatron-path /workspace/checkpoints/megatron_checkpoints/flux_dev/iter_0000000 \
  --hf-path /workspace/checkpoints/hf_checkpoints/flux_dev_hf
```

**Note**: The exported checkpoint from Megatron to HuggingFace (`/workspace/checkpoints/hf_checkpoints/flux_dev_hf`) contains only the transformer weights. To run inference, you still require the other pipeline components (VAE, text encoders, etc.).
To assemble a functional inference directory:
- Duplicate the original HF checkpoint directory.
- Replace the `./transformer` folder in that directory with your newly exported `/transformer` folder.

---

### ⚡ Parallelism Support

The table below shows current parallelism support for FLUX models:

| Model | Data Parallel | Tensor Parallel | Sequence Parallel | Context Parallel | FSDP |
|---|---|---|---|---|---|
| FLUX.1-dev (12B) | ✅ | ✅ | Coming Soon | Coming Soon | Coming Soon |

---

### 📊 Model Configurations

FLUX supports:

**FLUX.1-dev:**
- `num_joint_layers=19` (double/MMDiT blocks)
- `num_single_layers=38` (single transformer blocks)
- `hidden_size=3072`
- `num_attention_heads=24`
- `guidance_embed=True` (classifier-free guidance)
- `in_channels=64` (VAE latent channels after packing)
- `context_dim=4096` (T5 text encoder dimension)
- `vae_channels=16` (before packing)
- `vae_scale_factor=8` (8x8 downsampling)
- Total parameters: ~12B

---

### References
Black Forest Labs. (2024). FLUX.1: State-of-the-art text-to-image generation. GitHub. https://github.com/black-forest-labs/flux

