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

**Create video dataset:**
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
    "vila_caption": "A detailed description of the video content..."
  }
]
```

**Preprocess videos to .meta files:**

There are two preprocessing modes:

**Mode 1: Full video (recommended for training)**
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

**Mode 2: Extract frames (for frame-based training)**
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

```bash
uv run --group automodel --with . \
  python examples/automodel/generate/wan_validate.py \
  --meta_folder <your_meta_folder> \
  --guidance_scale 5 \
  --checkpoint ./checkpoints/step_1000 \
  --num_samples 10
```

**With WandB logging:**
```bash
uv run --group automodel --with . \
  python examples/automodel/generate/wan_validate.py \
  --meta_folder <your_meta_folder> \
  --guidance_scale 5 \
  --checkpoint ./checkpoints/step_1000 \
  --num_samples 10 \
  --use_wandb \
  --wandb_project wan_validation
```

**Note:** You can use `--checkpoint ./checkpoints/LATEST` to automatically use the most recent checkpoint.

---

## Configuration

### Fine-tuning Config (`wan2_1_t2v_flow.yaml`)

```yaml
model:
  pretrained_model_name_or_path: Wan-AI/Wan2.1-T2V-1.3B-Diffusers

step_scheduler:
  global_batch_size: 8
  local_batch_size: 1
  num_epochs: 10
  ckpt_every_steps: 100

data:
  dataloader:
    meta_folder: "<your_processed_meta_folder>"
    num_workers: 2

optim:
  learning_rate: 5e-6

flow_matching:
  timestep_sampling: "uniform"
  flow_shift: 3.0

fsdp:
  dp_size: 8  # Single node: 8 GPUs

checkpoint:
  enabled: true
  checkpoint_dir: "./checkpoints"
```

### Multi-node Config Differences

```yaml
fsdp:
  dp_size: 16           # 2 nodes × 8 GPUs
  dp_replicate_size: 2  # Replicate across 2 nodes
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
- ✅ **Checkpointing**: EMA, consolidated, and sharded formats
- ✅ **Multi-node**: SLURM and torchrun support

---

## Supported Models

| Model | Parameters | Resolution | Frames | Status |
|-------|------------|------------|--------|--------|
| Wan 2.1 T2V 1.3B | 1.3B | 480×832 | 81 | ✅ |
| Wan 2.1 T2V 14B | 14B | 480×832 | 81 | ✅ |
| FLUX | TBD | TBD | TBD | 🔄 In Progress |

**Request a model:** Submit an issue on [GitHub](https://github.com/your-org/dfm/issues) with `model-request` label.

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
