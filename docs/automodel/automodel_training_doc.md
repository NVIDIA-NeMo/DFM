# Diffusion Model Fine-tuning with Automodel Backend

Train diffusion models with distributed training support using NeMo Automodel and flow matching.

**Currently Supported:** Wan 2.1 Text-to-Video (1.3B and 14B models)

---

## Quick Start

### 1. Docker Setup

```bash
# Build image
docker build -f docker/Dockerfile.ci -t dfm-training .

# Run container
docker run --gpus all -it \
  -v $(pwd):/workspace \
  -v /path/to/data:/data \
  --ipc=host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  dfm-training bash

# Inside container: Initialize submodules
export UV_PROJECT_ENVIRONMENT=
git submodule update --init --recursive 3rdparty/
```

### 2. Prepare Data

We provide two ways to prepare your dataset:

- Start with raw videos: Place your `.mp4` files in a folder and use our data-preparation scripts to scan the videos and generate a `meta.json` entry for each sample (which includes `width`, `height`, `start_frame`, `end_frame`, and a caption). If you have captions, you can also include per-video named `<video>.jsonl`; the scripts will pick up the text automatically. The final dataset layout is shown below.
- Bring your own `meta.json`: If you already have annotations, create `meta.json` yourself following the schema shown below.

**Create video dataset:**
In the following exaample we use two video files, solely for demonstration purposes. Actual training datasets will have a large number of files.
```
<your_video_folder>/
├── video1.mp4
├── video2.mp4
└── meta.json
```

**meta.json format:**
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

**Preprocess videos to .meta files:**

There are two preprocessing modes. Use this guide to choose the right mode:

- **Full Video (`--mode video`)**
  - **What it is**: Converts each source video into a single `.meta` that preserves the full temporal sequence as latents. Training can sample temporal windows/clips from the sequence on the fly.
  - **When to use**: Fine-tuning text-to-video models where motion and temporal consistency matter. This is the recommended default for most training runs.

- **Extract frames (`--mode frames`)**
  - **What it is**: Uniformly samples `N` frames per video and writes each as its own one-frame `.meta` sample (no temporal continuity).
  - **When to use**: Image/frame-level training objectives, quick smoke tests, or ablations where learning motion is not required.

**Mode 1: Full video (recommended for training)**
```bash
python examples/megatron/recipes/wan/prepare_dataset_wan.py \
  --video_folder <your_video_folder> \
  --output_folder ./processed_meta \
  --output_format automodel \
  --model "Wan-AI/Wan2.1-T2V-1.3B-Diffusers" \
  --mode video \
  --height 480 \
  --width 832 \
  --resize_mode bilinear \
  --center-crop
```

**Mode 2: Extract frames (for frame-based training)**
```bash
python examples/megatron/recipes/wan/prepare_dataset_wan.py \
  --video_folder <your_video_folder> \
  --output_folder ./processed_meta \
  --output_format automodel \
  --model "Wan-AI/Wan2.1-T2V-1.3B-Diffusers" \
  --mode frames \
  --num-frames 40 \
  --height 480 \
  --width 832 \
  --resize_mode bilinear \
  --center-crop
```

**Key arguments:**
- `--mode`: `video` (full video) or `frames` (extract evenly-spaced frames)
- `--num-frames`: Number of frames to extract (only for `frames` mode)
- `--height/--width`: Target resolution
- `--center-crop`: Crop to exact size after aspect-preserving resize

**Preprocessing modes:**
- **`video` mode**: Processes entire video sequence, creates one `.meta` file per video
- **`frames` mode**: Extracts N evenly-spaced frames, creates one `.meta` file per frame (treated as 1-frame videos)

**Output:** Creates `.meta` files containing:
- Encoded video latents (normalized)
- Text embeddings (from UMT5)
- First frame as JPEG (video mode only)
- Metadata

### 3. Train

**Single-node (8 GPUs):**
```bash
export UV_PROJECT_ENVIRONMENT=

uv run --group automodel --with . \
  torchrun --nproc-per-node=8 \
  examples/automodel/finetune/finetune.py \
  -c examples/automodel/finetune/wan2_1_t2v_flow.yaml
```

**Multi-node with SLURM:**
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

### 4. Validate

Use this step to perform a quick qualitative check of a trained checkpoint. The validation script:
- Reads prompts from `.meta` files in `--meta_folder` (uses `metadata.vila_caption`; latents are ignored).
- Loads the `WanPipeline` and, if provided, restores weights from `--checkpoint` (prefers `ema_shadow.pt`, then `consolidated_model.bin`, then sharded FSDP `model/*.distcp`).
- Generates short videos for each prompt with the specified settings (`--guidance_scale`, `--num_inference_steps`, `--height/--width`, `--num_frames`, `--fps`, `--seed`) and writes them to `--output_dir`.
- Intended for qualitative comparison across checkpoints; it does not compute quantitative metrics.

```bash
uv run --group automodel --with . \
  python examples/automodel/generate/wan_validate.py \
  --meta_folder <your_meta_folder> \
  --guidance_scale 5 \
  --checkpoint ./checkpoints/step_1000 \
  --num_samples 10
```

**Note:** You can use `--checkpoint ./checkpoints/LATEST` to automatically use the most recent checkpoint.

---

## Configuration

### Fine-tuning Config (`wan2_1_t2v_flow.yaml`)

Note: The inline configuration below is provided for quick reference. The canonical, up-to-date files are maintained in the repository: [examples/automodel/](../../examples/automodel/), [examples/automodel/finetune/wan2_1_t2v_flow.yaml](../../examples/automodel/finetune/wan2_1_t2v_flow.yaml), and [examples/automodel/finetune/wan2_1_t2v_flow_multinode.yaml](../../examples/automodel/finetune/wan2_1_t2v_flow_multinode.yaml).

```yaml
model:  # Base pretrained model to fine-tune
  pretrained_model_name_or_path: Wan-AI/Wan2.1-T2V-1.3B-Diffusers  # HF repo or local path

step_scheduler:  # Global training schedule
  global_batch_size: 8  # Effective batch size across all GPUs
  local_batch_size: 1  # Per-GPU batch size
  num_epochs: 10  # Number of passes over the dataset
  ckpt_every_steps: 100  # Save a checkpoint every N steps

data:  # Data input configuration
  dataloader:  # DataLoader parameters
    meta_folder: "<your_processed_meta_folder>"  # Folder containing .meta files
    num_workers: 2  # Worker processes per rank

optim:  # Optimizer/training hyperparameters
  learning_rate: 5e-6  # Base learning rate

flow_matching:  # Flow-matching training settings
  timestep_sampling: "uniform"  # Strategy for sampling timesteps
  flow_shift: 3.0  # Scalar shift applied to the target flow

fsdp:  # Distributed training (e.g., FSDP) configuration
  dp_size: 8  # Total data-parallel replicas (single node: 8 GPUs)

checkpoint:  # Checkpointing behavior
  enabled: true  # Enable periodic checkpoint saving
  checkpoint_dir: "./checkpoints"  # Output directory for checkpoints
```

### Multi-node Config Differences

```yaml
fsdp:  # Overrides for multi-node runs
  dp_size: 16           # Total data-parallel replicas (2 nodes × 8 GPUs)
  dp_replicate_size: 2  # Number of replicated groups across nodes
```

### Pretraining vs Fine-tuning

| Setting | Fine-tuning | Pretraining |
|---------|-------------|-------------|
| `learning_rate` | 5e-6 | 5e-5 |
| `weight_decay` | 0.01 | 0.1 |
| `flow_shift` | 3.0 | 2.5 |
| `logit_std` | 1.0 | 1.5 |
| Dataset size | 100s-1000s | 10K+ |

---

## Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU | A100 40GB | A100 80GB / H100 |
| GPUs | 4 | 8+ |
| RAM | 128 GB | 256 GB+ |
| Storage | 500 GB SSD | 2 TB NVMe |

---

## Features

- ✅ **Flow Matching**: Pure flow matching training
- ✅ **Distributed**: FSDP2 + Tensor Parallelism
- ✅ **Mixed Precision**: BF16 by default
- ✅ **WandB**: Automatic logging
- ✅ **Checkpointing**: consolidated, and sharded formats
- ✅ **Multi-node**: SLURM and torchrun support

---

## Supported Models

| Model | Parameters | Parallelization | Status |
|-------|------------|-----------------|--------|
| Wan 2.1 T2V 1.3B | 1.3B | FSDP2 via Automodel + DDP | ✅ |
| Wan 2.1 T2V 14B | 14B | FSDP2 via Automodel + DDP | ✅ |
| FLUX | TBD | TBD | 🔄 In Progress |

---

## Advanced

**Custom parallelization:**
```yaml
fsdp:
  tp_size: 2  # Tensor parallel
  dp_size: 4  # Data parallel
```

**Checkpoint cleanup:**
```python
from pathlib import Path
import shutil

def cleanup_old_checkpoints(checkpoint_dir, keep_last_n=3):
    checkpoints = sorted(Path(checkpoint_dir).glob("step_*"))
    for old_ckpt in checkpoints[:-keep_last_n]:
        shutil.rmtree(old_ckpt)
```
