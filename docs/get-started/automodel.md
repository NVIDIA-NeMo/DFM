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
:ref-type: doc

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

:::: {tab-set}

::: {tab-item} Dataset Format

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

:::

::: {tab-item} Data Requirements

Automodel expects a dataset with:
- **Video files**: MP4, WebM, or similar
- **Text captions**: Descriptions for each video
- **Metadata**: Frame count, resolution, FPS

:::

::: {tab-item} Dataloader Config

The training script uses a custom dataloader specified in the config:

```yaml
data:
  dataloader:
    _target_: Automodel.datasets.build_wan21_dataloader
    meta_folder: /path/to/your/dataset/meta/
    batch_size: 1
    num_workers: 2
```

:::

::::

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
:icon: settings

- `tp_size=1`: Tensor parallelism disabled (automatic for this model size)
- `cp_size=1`: Context parallelism disabled
- `pp_size=1`: Pipeline parallelism disabled
- `dp_size=8`: Data parallelism across 8 GPUs
:::

### 3. Run Training

Execute the training script:

:::: {tab-set}

::: {tab-item} Custom Configuration

```bash
python dfm/examples/automodel/finetune/finetune.py /path/to/wan2_1_finetune.yaml
```

:::

::: {tab-item} Default Configuration

```bash
python dfm/examples/automodel/finetune/finetune.py
```

This uses the default config at `dfm/examples/automodel/finetune/wan2_1_t2v_flow.yaml` (relative to the DFM installation directory).

:::

::::

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

### 4. Monitor Training

Monitor console output for decreasing loss values and checkpoint saves. If `wandb.mode: online`, view metrics in the WandB dashboard.

Verify checkpoints are being saved:

```bash
ls -lh /path/to/checkpoints/wan2_1_finetuning/
```

Expected: `iter_1000/`, `iter_2000/`, `latest/` directories with `model_weights.pt` and `optimizer_states.pt` files.

### Troubleshooting

:::{dropdown} Out of Memory Errors
:icon: warning

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
:icon: warning

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

