---
description: "End-to-end Automodel quickstart: fine-tune and generate videos"
categories: ["getting-started", "automodel"]
tags: ["quickstart", "tutorial", "automodel"]
personas: ["data-scientist-focused"]
difficulty: "beginner"
content_type: "tutorial"
---

(gs-automodel)=

# Automodel Workflow

Complete end-to-end tutorial for fine-tuning and generating videos using NeMo DFM's Automodel approach.

:::{card}

**Goal**: Fine-tune a pretrained video model and generate videos from your checkpoint.

^^^

**In this tutorial, you will**:

1. Fine-tune the WAN2.1 model on your dataset
2. Generate videos from your trained model
3. Experiment with generation parameters

**Time**: 30-45 minutes (depending on training duration)

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
- ✅ Multi-GPU setup (recommended: 8 GPUs)
- ✅ Dataset in Energon format or custom dataloader

---

(gs-automodel-training-section)=
## Fine-Tune WAN2.1 Model

Fine-tune the WAN2.1 text-to-video model using Automodel's recipe-based training approach.

**Key concept**: Automodel handles parallelism automatically using FSDP2—no manual tensor or pipeline parallelism configuration needed.

:::{dropdown} What happens during training
:icon: info

1. Load pretrained WAN2.1 model from Hugging Face
2. Configure FSDP2 parallelism automatically
3. Train on your dataset with flow matching
4. Save checkpoints periodically
:::

### 1. Prepare Your Dataset

(gs-automodel-data-requirements)=

You can prepare your dataset in two ways:

- **Start with raw videos**: Place your `.mp4` files in a folder and use data-preparation scripts to scan videos and generate a `meta.json` entry for each sample
- **Bring your own `meta.json`**: If you already have annotations, create `meta.json` yourself following the schema below

#### Dataset Structure

```text
<your_video_folder>/
├── video1.mp4
├── video2.mp4
└── meta.json
```

:::{note}
If you have captions, you can also include per-video named `<video>.jsonl` files; the scripts will pick up the text automatically.
:::

#### meta.json Format

:::{dropdown} Complete meta.json Schema
:icon: info

Each entry in `meta.json` should include:

```json
[
  {
    "file_name": "video1.mp4",
    "width": 1280,
    "height": 720,
    "start_frame": 0,
    "end_frame": 121,
    "vila_caption": "A detailed description of the video1.mp4 contents..."
  },
  {
    "file_name": "video2.mp4",
    "width": 1280,
    "height": 720,
    "start_frame": 0,
    "end_frame": 12,
    "vila_caption": "A detailed description of the video2.mp4 contents..."
  }
]
```

**Fields**:
- `file_name`: Name of the video file
- `width`: Video width in pixels
- `height`: Video height in pixels
- `start_frame`: Starting frame index (usually 0)
- `end_frame`: Ending frame index
- `vila_caption`: Text description/caption for the video
:::

#### Preprocess Videos to .meta Files

There are two preprocessing modes. Choose the right mode for your use case:

:::: {tab-set}

::: {tab-item} Full Video Mode (Recommended)

**What it is**: Converts each source video into a single `.meta` file that preserves the full temporal sequence as latents. Training can sample temporal windows/clips from the sequence on the fly.

**When to use**: Fine-tuning text-to-video models where motion and temporal consistency matter. This is the recommended default for most training runs.

**Output**: Creates one `.meta` file per video

```bash
python dfm/src/automodel/utils/data/preprocess_resize.py \
  --mode video \
  --video_folder <your_video_folder> \
  --output_folder ./processed_meta \
  --model Wan-AI/Wan2.1-T2V-1.3B-Diffusers \
  --height 480 \
  --width 720 \
  --center-crop
```

**Key arguments**:
- `--mode video`: Process full videos
- `--height/--width`: Target resolution
- `--center-crop`: Crop to exact size after aspect-preserving resize

:::

::: {tab-item} Extract Frames Mode

**What it is**: Uniformly samples `N` frames per video and writes each as its own one-frame `.meta` sample (no temporal continuity).

**When to use**: Image/frame-level training objectives, quick smoke tests, or ablations where learning motion is not required.

**Output**: Creates one `.meta` file per frame (treated as 1-frame videos)

```bash
python dfm/src/automodel/utils/data/preprocess_resize.py \
  --mode frames \
  --num-frames 40 \
  --video_folder <your_video_folder> \
  --output_folder ./processed_frames \
  --model Wan-AI/Wan2.1-T2V-1.3B-Diffusers \
  --height 240 \
  --width 416 \
  --center-crop
```

**Key arguments**:
- `--mode frames`: Extract evenly-spaced frames
- `--num-frames`: Number of frames to extract
- `--height/--width`: Target resolution
- `--center-crop`: Crop to exact size after aspect-preserving resize

:::

::::

**Output**: Both modes create `.meta` files containing:
- Encoded video latents (normalized)
- Text embeddings (from UMT5)
- First frame as JPEG (video mode only)
- Metadata

### 2. Create Training Configuration

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

#### Key Configuration Parameters

:::{list-table} Configuration Parameters
:header-rows: 1
:name: config-params

* - Parameter
  - Description
  - Default
  - Recommended
* - `model.pretrained_model_name_or_path`
  - Hugging Face model ID
  - Required
  - `Wan-AI/Wan2.1-T2V-1.3B-Diffusers`
* - `data.dataloader.meta_folder`
  - Dataset metadata location
  - Required
  - Your dataset path
* - `batch.batch_size_per_node`
  - Batch size per node
  - `8`
  - 4-8 (depends on GPU memory)
* - `training.num_epochs`
  - Training epochs
  - `100`
  - Adjust based on dataset size
* - `optim.learning_rate`
  - Learning rate
  - `5e-6`
  - 1e-6 to 1e-5
* - `fsdp.dp_size`
  - Data parallel size
  - `8`
  - Match GPU count
* - `checkpoint.checkpoint_dir`
  - Where to save checkpoints
  - Required
  - Path with enough storage
* - `logging.save_every`
  - Checkpoint interval (iterations)
  - `1000`
  - 500-2000
:::

:::{dropdown} Parallelism settings (`fsdp`)
:icon: gear

- `tp_size=1`: Tensor parallelism disabled (automatic for this model size)
- `cp_size=1`: Context parallelism disabled
- `pp_size=1`: Pipeline parallelism disabled
- `dp_size=8`: Data parallelism across 8 GPUs
:::

### 3. Run Training

Execute the training script:

::::: {tab-set}

:::: {tab-item} Custom Configuration

```bash
python dfm/examples/automodel/finetune/finetune.py /path/to/wan2_1_finetune.yaml
```

:::{dropdown} Multi-Node with SLURM
:icon: server

For multi-node training with SLURM, use this script:

```bash
#!/bin/bash
#SBATCH -N 2
#SBATCH --ntasks-per-node 1
#SBATCH --gpus-per-node=8
#SBATCH --exclusive

export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=29500
export NUM_GPUS=8

# Per-rank UV cache to avoid conflicts
unset UV_PROJECT_ENVIRONMENT
mkdir -p /opt/uv_cache/${SLURM_JOB_ID}_${SLURM_PROCID}
export UV_CACHE_DIR=/opt/uv_cache/${SLURM_JOB_ID}_${SLURM_PROCID}

uv run --group automodel --with . \
  torchrun \
  --nnodes=$SLURM_NNODES \
  --nproc-per-node=$NUM_GPUS \
  --rdzv_backend=c10d \
  --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
  examples/automodel/finetune/finetune.py \
  -c examples/automodel/finetune/wan2_1_t2v_flow_multinode.yaml
```

**Key differences for multi-node**:
- Uses `wan2_1_t2v_flow_multinode.yaml` config
- Sets `MASTER_ADDR` and `MASTER_PORT` for distributed coordination
- Configures per-rank UV cache to avoid conflicts
- Uses `--nnodes` and `--rdzv_backend=c10d` for multi-node setup
:::

::::

::::{tab-item} Default Configuration

```bash
python dfm/examples/automodel/finetune/finetune.py
```

This uses the default config at `dfm/examples/automodel/finetune/wan2_1_t2v_flow.yaml` (relative to the DFM installation directory).

::::

:::::

:::{dropdown} What happens during training
:icon: info

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
:::

#### Expected Output

```text
[INFO] Loading pretrained model: Wan-AI/Wan2.1-T2V-1.3B-Diffusers
[INFO] Initializing FSDP2 with dp_size=8
[INFO] Starting training loop...
[INFO] Epoch 1/100, Iter 1/5000, Loss: 0.234
[INFO] Epoch 1/100, Iter 2/5000, Loss: 0.221
...
[INFO] Checkpoint saved: /path/to/checkpoints/wan2_1_finetuning/iter_1000/
```

### 4. Validate Training

Use the validation script to perform a quick qualitative check of a trained checkpoint.

:::{dropdown} Validation Script Details
:icon: info

The validation script (`wan_validate.py`):
- Reads prompts from `.meta` files in `--meta_folder` (uses `metadata.vila_caption`; latents are ignored)
- Loads the `WanPipeline` and, if provided, restores weights from `--checkpoint`
- Checkpoint loading priority: `ema_shadow.pt` → `consolidated_model.bin` → sharded FSDP `model/*.distcp`
- Generates short videos for each prompt with specified settings (`--guidance_scale`, `--num_inference_steps`, `--height/--width`, `--num_frames`, `--fps`, `--seed`)
- Writes videos to `--output_dir`
- Intended for qualitative comparison across checkpoints; does not compute quantitative metrics
:::

```bash
uv run --group automodel --with . \
  python examples/automodel/generate/wan_validate.py \
  --meta_folder <your_meta_folder> \
  --guidance_scale 5 \
  --checkpoint ./checkpoints/step_1000 \
  --num_samples 10
```

:::{note}
You can use `--checkpoint ./checkpoints/LATEST` to automatically use the most recent checkpoint.
:::

### 5. Monitor Training

Monitor console output for decreasing loss values and checkpoint saves. If `wandb.mode: online`, view metrics in the WandB dashboard.

Verify checkpoints are being saved:

```bash
ls -lh /path/to/checkpoints/wan2_1_finetuning/
```

Expected: `iter_1000/`, `iter_2000/`, `latest/` directories with `model_weights.pt` and `optimizer_states.pt` files.

### Hardware Requirements

:::{dropdown} System Requirements
:icon: server

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU | A100 40GB | A100 80GB / H100 |
| GPUs | 4 | 8+ |
| RAM | 128 GB | 256 GB+ |
| Storage | 500 GB SSD | 2 TB NVMe |
:::

### Supported Models

| Model | Parameters | Parallelization | Status |
|-------|------------|-----------------|--------|
| WAN 2.1 T2V 1.3B | 1.3B | FSDP2 via Automodel + DDP | ✅ |
| WAN 2.1 T2V 14B | 14B | FSDP2 via Automodel + DDP | ✅ |
| FLUX | TBD | TBD | 🔄 In Progress |

### Advanced Topics

:::{dropdown} Pretraining vs Fine-tuning
:icon: gear

| Setting | Fine-tuning | Pretraining |
|---------|-------------|-------------|
| `learning_rate` | 5e-6 | 5e-5 |
| `weight_decay` | 0.01 | 0.1 |
| `flow_shift` | 3.0 | 2.5 |
| `logit_std` | 1.0 | 1.5 |
| Dataset size | 100s-1000s | 10K+ |
:::

:::{dropdown} Custom Parallelization
:icon: gear

You can customize parallelism settings in your config:

```yaml
fsdp:
  tp_size: 2  # Tensor parallel
  dp_size: 4  # Data parallel
```
:::

:::{dropdown} Checkpoint Management
:icon: gear

Clean up old checkpoints to save storage:

```python
from pathlib import Path
import shutil

def cleanup_old_checkpoints(checkpoint_dir, keep_last_n=3):
    checkpoints = sorted(Path(checkpoint_dir).glob("step_*"))
    for old_ckpt in checkpoints[:-keep_last_n]:
        shutil.rmtree(old_ckpt)
```
:::

### Troubleshooting

:::{dropdown} Out of Memory Errors
:icon: alert

```
RuntimeError: CUDA out of memory
```

**Solution**: Reduce `batch.batch_size_per_node`:

```yaml
batch:
  batch_size_per_node: 4  # or 2
```
:::

---

(gs-automodel-inference-section)=
## Generate Videos

Generate videos using pretrained models from Hugging Face.

:::{note} The examples in this section use `Wan-AI/Wan2.2-T2V-A14B-Diffusers` (a newer, larger model) for inference, while the training section uses `Wan-AI/Wan2.1-T2V-1.3B-Diffusers` (smaller model suitable for fine-tuning). Both models follow the same workflow.
:::

**Generation time**: 2-5 minutes per video (single GPU), faster with parallelism

**Requirements**: Pretrained Hugging Face model (`Wan-AI/Wan2.2-T2V-A14B-Diffusers`), GPU with 16GB+ memory recommended

:::{dropdown} What happens during inference
:icon: info

1. Load pretrained model from Hugging Face
2. Configure distributed parallelism (optional)
3. Generate video from text prompt
4. Save video file
:::

### Generate from Pretrained Model

#### Generate a Video

```bash
python dfm/examples/automodel/generate/wan_generate.py \
    --prompt "A butterfly flying over colorful flowers in a garden" \
    --height 480 \
    --width 848 \
    --num-frames 111 \
    --output butterfly_garden.mp4
```

:::: {tab-set}

::: {tab-item} Expected Output

```text
[Loading] Loading VAE and pipeline...
[Setup] Pipeline loaded and parallelized via NeMoAutoDiffusionPipeline
[Inference] Starting distributed inference...
[Inference] Saved butterfly_garden.mp4
[Complete] Automodel FSDP2 inference completed!
```

:::

::: {tab-item} Output File

- Filename: `butterfly_garden.mp4`
- Size: 5-15 MB
- Duration: ~4.6 seconds (111 frames at 24 FPS)

:::

::::

#### View the Video

```bash
# Play with ffplay
ffplay butterfly_garden.mp4

# Or open with default player
open butterfly_garden.mp4  # macOS
xdg-open butterfly_garden.mp4  # Linux
```

### Generation Parameters

:::{list-table} Generation Parameters
:header-rows: 1
:name: generation-params

* - Parameter
  - Description
  - Default
  - Notes
* - `--prompt`
  - Text description of video
  - Required
  - Be specific and descriptive
* - `--height`
  - Video height (pixels)
  - `480`
  - Common: 360, 480, 720
* - `--width`
  - Video width (pixels)
  - `848`
  - Common: 640, 848, 1280
* - `--num-frames`
  - Number of frames
  - `111`
  - Must be 4n+1 format (51, 111, 149, 189, 229)
* - `--output`
  - Output filename
  - `t2v_fsdp2_rank0.mp4`
  - Any `.mp4` path
* - `--num-inference-steps`
  - Diffusion steps
  - `20`
  - More steps = better quality, slower
* - `--seed`
  - Random seed
  - `42`
  - Use same seed for reproducible results
:::

### Troubleshooting

:::{dropdown} Out of Memory Errors
:icon: alert

```
RuntimeError: CUDA out of memory
```

**Solution**: Reduce resolution and frames:

```bash
python dfm/examples/automodel/generate/wan_generate.py \
    --prompt "Your prompt" \
    --height 360 \
    --width 640 \
    --num-frames 51 \
    --output output.mp4
```

:::

---

## Related Tutorials

- [Megatron DiT Tutorial](megatron.md) - Train DiT models from scratch
- [Megatron WAN Tutorial](megatron-wan.md) - Train WAN models with Megatron

## Related Documentation

- [Training Paradigms](../about/concepts/training-paradigms.md) - Understand AutoModel vs Megatron differences
- [Performance Benchmarks](../reference/performance.md) - Training throughput metrics
- [AutoModel vs Megatron Comparison](../about/comparison.md) - Experimental comparison

