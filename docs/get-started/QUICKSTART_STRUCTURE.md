# Get Started Quickstart Structure

## Why Three Quickstarts?

The `get-started/` section supports three distinct quickstarts because NeMo DFM has three primary user journeys:

### 1. **Installation Quickstart** (`installation.md`)
**Purpose**: Get the environment set up and ready to use DFM

**Why separate**: Installation is a prerequisite for both training and inference, but users may:
- Want to install without immediately training/inferring
- Need different installation methods (Docker vs. pip vs. source)
- Have different system requirements (development vs. production)

**User journey**: "I want to use DFM → How do I install it?"

### 2. **Training Quickstart** (`training.md`)
**Purpose**: Run your first training job with minimal setup

**Why separate**: Training is a distinct workflow that requires:
- Understanding distributed training setup (torchrun, multi-GPU)
- Data preparation (Energon datasets, webdatasets)
- Configuration files (YAML configs, override patterns)
- Different from inference (no model loading, different parallelism)

**User journey**: "I have data → How do I train a model?"

### 3. **Inference Quickstart** (`inference.md`)
**Purpose**: Generate videos using pre-trained models

**Why separate**: Inference is a distinct workflow that requires:
- Model loading (checkpoints, Hugging Face models)
- Different parallelism (inference-optimized)
- No training loop, just generation
- Different from training (simpler setup, faster to run)

**User journey**: "I have a model → How do I generate videos?"

---

## Example Content in Source

### Installation Examples

**Location**: `CONTRIBUTING.md`, `docker/Dockerfile.ci`

**Key patterns found**:
```bash
# Docker-based installation (recommended for development)
docker build -f docker/Dockerfile.ci -t dfm:latest .
docker run --gpus all -v $(pwd):/opt/DFM -it dfm:latest bash

# Inside container
source /opt/venv/bin/activate
uv pip install --no-deps -e .
```

**Dependencies** (from `pyproject.toml`):
- Core: `accelerate`, `diffusers==0.35.1`, `megatron-energon`
- Video: `imageio`, `imageio-ffmpeg`, `opencv-python-headless`
- Optional: `nemo-automodel` (for Automodel support)

### Training Examples

**Location**: 
- `examples/megatron/recipes/dit/pretrain_dit_model.py` - DiT training
- `examples/megatron/recipes/wan/pretrain_wan.py` - WAN training
- `dfm/examples/automodel/finetune/finetune.py` - Automodel fine-tuning

**Key patterns found**:

#### Megatron Training (DiT/WAN)
```python
# Distributed training with torchrun
torchrun --nproc-per-node 2 \
    examples/megatron/recipes/dit/pretrain_dit_model.py \
    --dataset_path "/opt/VFM/butterfly_webdataset"
```

**Structure**:
1. Parse arguments (config file, dataset path, overrides)
2. Load configuration (YAML + CLI overrides)
3. Initialize distributed environment
4. Setup data module (Energon-based)
5. Initialize model (DiT/WAN)
6. Run training loop

**Example from `pretrain_dit_model.py`**:
- Uses `pretrain_config()` recipe function
- Supports YAML config files + CLI overrides
- Uses `DITForwardStep` for training step
- Integrates with Megatron-Bridge training infrastructure

#### Automodel Training
```python
# Simple recipe-based training
from Automodel.recipes.finetune import TrainWan21DiffusionRecipe

cfg = parse_args_and_load_config(default_config_path)
recipe = TrainWan21DiffusionRecipe(cfg)
recipe.setup()
recipe.run_train_validation_loop()
```

**Structure**:
1. Load config (YAML-based)
2. Create recipe instance
3. Setup (model, data, optimizers)
4. Run training loop

### Inference Examples

**Location**:
- `dfm/examples/automodel/generate/wan_generate.py` - Automodel inference
- `examples/megatron/recipes/dit/inference_dit_model.py` - DiT inference
- `examples/megatron/recipes/wan/inference_wan.py` - WAN inference
- `dfm/src/automodel/utils/validate_t2v.py` - Validation/inference utility

**Key patterns found**:

#### Automodel Inference
```python
# Load pipeline with distributed parallelism
pipe, _ = NeMoAutoDiffusionPipeline.from_pretrained(
    "Wan-AI/Wan2.2-T2V-A14B-Diffusers",
    vae=vae,
    torch_dtype=torch.bfloat16,
    parallel_scheme=parallel_scheme  # TP+CP+PP+DP
)

# Generate video
out = pipe(
    prompt=args.prompt,
    height=args.height,
    width=args.width,
    num_frames=args.num_frames,
    guidance_scale=args.guidance_scale,
    num_inference_steps=args.num_inference_steps,
).frames[0]

# Export video
export_to_video(out, args.output, fps=args.fps)
```

**Structure**:
1. Initialize distributed environment
2. Load VAE and pipeline (with parallelism)
3. Generate video from prompt
4. Export video to file

**Key parameters**:
- `--prompt`: Text prompt for generation
- `--height`, `--width`: Video resolution
- `--num-frames`: Number of frames (e.g., 111)
- `--guidance-scale`: CFG scale (e.g., 4.0)
- `--num-inference-steps`: Diffusion steps (e.g., 20)
- `--tp-size`, `--cp-size`, `--pp-size`, `--dp-size`: Parallelism config

#### Megatron Inference (WAN)
```python
# Load inference pipeline
pipeline = FlowInferencePipeline(
    inference_cfg,
    model_id="Wan-AI/Wan2.1-T2V-14B-Diffusers",
    checkpoint_dir=args.checkpoint_dir,
    tensor_parallel_size=args.tensor_parallel_size,
    context_parallel_size=args.context_parallel_size,
    pipeline_parallel_size=args.pipeline_parallel_size,
)

# Generate videos
videos = pipeline.generate(
    prompts=prompts,
    sizes=[SIZE_CONFIGS[size] for size in size_keys],
    frame_nums=frame_nums,
    shift=args.sample_shift,
    sampling_steps=args.sample_steps,
    guide_scale=args.sample_guide_scale,
    seed=args.base_seed,
    offload_model=args.offload_model,
)
```

**Structure**:
1. Parse arguments (checkpoint, parallelism, prompts)
2. Load inference pipeline with parallelism
3. Generate videos (batch support)
4. Save videos to files

---

## Recommended Quickstart Structure

### Installation Quickstart (`installation.md`)

**Sections**:
1. **Prerequisites**
   - Python 3.10+
   - CUDA-capable GPU
   - Docker (optional, recommended)

2. **Installation Methods**
   - Docker (recommended for development)
   - pip install (for users)
   - Source install (for developers)

3. **Verify Installation**
   - Simple import test
   - Check GPU availability

4. **Next Steps**
   - Link to training quickstart
   - Link to inference quickstart

**Example content**:
```markdown
## Docker Installation (Recommended)

```bash
# Build container
docker build -f docker/Dockerfile.ci -t dfm:latest .

# Run container
docker run --gpus all -v $(pwd):/opt/DFM -it dfm:latest bash

# Install DFM
source /opt/venv/bin/activate
uv pip install --no-deps -e .
```

## Verify Installation

```python
import dfm
print("DFM installed successfully!")
```
```

### Training Quickstart (`training.md`)

**Sections**:
1. **Prerequisites**
   - Installation complete
   - Dataset prepared (Energon format or webdataset)
   - Multi-GPU setup (for distributed training)

2. **Choose Your Path**
   - **Automodel Training**: Simpler, recipe-based
   - **Megatron Training**: More control, large-scale

3. **Automodel Training Example**
   - Show `finetune.py` example
   - Explain config file structure
   - Run command

4. **Megatron Training Example**
   - Show `pretrain_dit_model.py` example
   - Explain distributed setup (torchrun)
   - Run command

5. **Monitor Training**
   - Check logs
   - Monitor checkpoints

**Example content**:
```markdown
## Automodel Training (Simpler)

```bash
python dfm/examples/automodel/finetune/finetune.py \
    --config-path /path/to/config.yaml
```

## Megatron Training (Large-Scale)

```bash
torchrun --nproc-per-node 8 \
    examples/megatron/recipes/dit/pretrain_dit_model.py \
    --dataset_path "/path/to/dataset"
```
```

### Inference Quickstart (`inference.md`)

**Sections**:
1. **Prerequisites**
   - Installation complete
   - Pre-trained model (checkpoint or Hugging Face model)

2. **Choose Your Path**
   - **Automodel Inference**: Simpler, Hugging Face models
   - **Megatron Inference**: More control, custom checkpoints

3. **Automodel Inference Example**
   - Show `wan_generate.py` example
   - Explain parallelism options
   - Run command

4. **Megatron Inference Example**
   - Show `inference_wan.py` example
   - Explain checkpoint loading
   - Run command

5. **View Results**
   - Check output video files
   - Adjust generation parameters

**Example content**:
```markdown
## Automodel Inference

```bash
python dfm/examples/automodel/generate/wan_generate.py \
    --prompt "A cat playing piano" \
    --height 480 --width 848 \
    --num-frames 111 \
    --output output.mp4
```

## Megatron Inference

```bash
torchrun --nproc-per-node 2 \
    examples/megatron/recipes/wan/inference_wan.py \
    --checkpoint-dir /path/to/checkpoint \
    --prompts "A cat playing piano"
```
```

---

## Key Differences: Training vs. Inference

| Aspect | Training | Inference |
|--------|----------|-----------|
| **Setup** | Data preparation, config files | Model loading, checkpoint paths |
| **Parallelism** | Full distributed (TP+CP+PP+DP) | Inference-optimized (often TP only) |
| **Time** | Hours/days | Minutes |
| **Output** | Model checkpoints | Video files |
| **Complexity** | High (training loop, validation) | Lower (single forward pass) |
| **Examples** | `pretrain_*.py`, `finetune.py` | `inference_*.py`, `wan_generate.py` |

---

## Next Steps

After completing quickstarts, users should:
1. **Read Concepts**: Understand architectures (DiT, WAN, EDM)
2. **Explore Examples**: Review full examples in `examples/` directory
3. **Reference Docs**: Check API reference for detailed parameters
4. **Advanced Topics**: Distributed training, custom architectures, optimization

