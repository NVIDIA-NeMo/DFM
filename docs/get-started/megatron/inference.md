---
description: "Generate videos from DiT checkpoint with Megatron"
categories: ["getting-started", "megatron"]
tags: ["inference", "generation", "how-to"]
personas: ["mle-focused"]
difficulty: "intermediate"
content_type: "how-to"
---

(gs-megatron-inference)=

# Generate Videos from DiT Checkpoint

Generate videos from your trained DiT model checkpoint using Megatron inference.

## Goal

Generate videos from the DiT model you trained on the butterfly dataset.

**Time**: 5-10 minutes per video

## Prerequisites

- ✅ Complete [Installation](../installation.md)
- ✅ Trained checkpoint from [training](training.md)
- ✅ Multi-GPU system (recommended: 2+ GPUs)
- ✅ Cosmos tokenizer for video decoding

## Overview

**What happens during inference**:
1. Initialize distributed environment with context parallelism
2. Load DiT model from checkpoint
3. Encode text prompt to T5 embeddings
4. Generate video latents using EDM sampling
5. Decode latents to video with Cosmos tokenizer
6. Save video file

**Generation time**: 3-8 minutes per video (depends on resolution and steps)

## Step 1: Prepare Model Checkpoint

### Checkpoint Format

The training saves checkpoints in this structure:

```text
checkpoints/dit_butterfly/
  ├── iter_5000/
  │   ├── model.pth         # Model weights
  │   └── extra_state.pt    # Training metadata
  └── latest_checkpointed_iteration.txt
```

**Note**: The inference script currently expects a consolidated `model.pth` file. If your checkpoint is sharded, consolidate it first.

### Consolidate Checkpoint (If Needed)

If your checkpoint is distributed across multiple files, consolidate:

```python
# consolidate_checkpoint.py
import torch

# Load sharded checkpoints
checkpoint = {}
for i in range(num_gpus):
    shard = torch.load(f"checkpoints/iter_5000/model_rank_{i}.pt")
    checkpoint.update(shard)

# Save consolidated
torch.save(checkpoint, "model.pth")
```

## Step 2: Run Inference

### Basic Generation

Generate a video using your checkpoint:

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

**Command breakdown**:
- `torchrun --nproc-per-node 2`: Use 2 GPUs with context parallelism
- `--prompt`: Text description of video to generate
- `--height` / `--width`: Video resolution
- `--num-video-frames`: Frame count
- `--video-save-path`: Output filename

**Note**: The script requires `model.pth` in the current directory (line 247). Update path if needed:

```python
# Edit inference_dit_model.py line 247:
state = torch.load("path/to/your/model.pth")
```

### With Custom Settings

Adjust generation quality and speed:

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

**Additional parameters**:
- `--num-steps`: Diffusion sampling steps (default: 35)
- `--guidance`: Classifier-free guidance scale (default: 7)
- `--seed`: Random seed for reproducibility
- `--cp-size`: Context parallel size (should match `nproc-per-node`)

## Generation Parameters

### Common Parameters

| Parameter | Description | Default | Range/Options |
|-----------|-------------|---------|---------------|
| `--prompt` | Text description | Required | Any text |
| `--negative-prompt` | What to avoid | `None` | Any text |
| `--height` | Video height (pixels) | `704` | 480, 704, 1024 |
| `--width` | Video width (pixels) | `1280` | 848, 1280, 1920 |
| `--num-video-frames` | Number of frames | `121` | 61, 121, 241 |
| `--fps` | Frames per second | `24` | 12, 24, 30 |
| `--video-save-path` | Output filename | `outputs` | Any path |

### Quality vs. Speed Parameters

| Parameter | Description | Default | Range | Effect |
|-----------|-------------|---------|-------|--------|
| `--num-steps` | Sampling steps | `35` | 10-100 | More = better quality, slower |
| `--guidance` | Guidance scale | `7.0` | 1.0-15.0 | Higher = stronger prompt adherence |
| `--cp-size` | Context parallel size | `1` | 1, 2, 4 | Higher = faster (multi-GPU) |
| `--seed` | Random seed | `1` | Any int | Same seed = reproducible output |

### Resolution Guidelines

**Supported resolutions** (DiT model):

| Resolution | Aspect Ratio | Use Case | Memory |
|------------|--------------|----------|--------|
| 480×848 | 16:9 (portrait) | Mobile, quick tests | ~8GB |
| 704×1280 | 16:9 (landscape) | Desktop, default | ~12GB |
| 1024×1920 | 16:9 (landscape) | High quality | ~20GB |

**Important**: Ensure height and width are divisible by 16 (tokenizer requirement).

## Step 3: View Generated Video

Check that video was created:

```bash
ls -lh idx=0_rank=0_butterfly_monarch.mp4
```

**Note**: Megatron inference adds prefix `idx={i}_rank={rank}_` to filename.

### Play Video

```bash
# Using ffplay
ffplay idx=0_rank=0_butterfly_monarch.mp4

# Or default player
open idx=0_rank=0_butterfly_monarch.mp4  # macOS
xdg-open idx=0_rank=0_butterfly_monarch.mp4  # Linux
```

## Advanced Usage

### High-Quality Generation

Maximum quality settings:

```bash
torchrun --nproc-per-node 2 \
    examples/megatron/recipes/dit/inference_dit_model.py \
    --prompt "A swallowtail butterfly landing on a purple flower" \
    --height 1024 \
    --width 1920 \
    --num-video-frames 241 \
    --num-steps 100 \
    --guidance 12.0 \
    --video-save-path swallowtail_hq.mp4
```

**Changes**:
- Higher resolution (1080p vs. 704p)
- More frames (241 vs. 121)
- More sampling steps (100 vs. 35)
- Stronger guidance (12.0 vs. 7.0)

**Trade-off**: ~5-10x longer generation time, ~3x more memory

### Fast Prototyping

Quick generation for testing:

```bash
torchrun --nproc-per-node 2 \
    examples/megatron/recipes/dit/inference_dit_model.py \
    --prompt "A small white butterfly" \
    --height 480 \
    --width 848 \
    --num-video-frames 61 \
    --num-steps 15 \
    --video-save-path butterfly_fast.mp4
```

**Changes**:
- Lower resolution (480p)
- Fewer frames (61 vs. 121)
- Fewer steps (15 vs. 35)

**Trade-off**: ~5x faster, lower quality

### Negative Prompts

Guide what NOT to generate:

```bash
torchrun --nproc-per-node 2 \
    examples/megatron/recipes/dit/inference_dit_model.py \
    --prompt "A butterfly in a garden" \
    --negative-prompt "blurry, low quality, distorted, watermark" \
    --video-save-path butterfly_clean.mp4
```

## Prompt Engineering Tips

### Effective Prompts for DiT

**Good prompts are**:
- **Specific**: Mention species, colors, actions
- **Visual**: Describe what you want to see
- **Concise**: 1-2 sentences optimal

**Examples**:

✅ **Good**:
```
"A monarch butterfly with vibrant orange and black wings fluttering over yellow wildflowers in bright sunlight"
```

❌ **Too vague**:
```
"A butterfly"
```

✅ **Good**:
```
"A blue morpho butterfly resting on a green leaf with sunlight filtering through rainforest canopy"
```

❌ **Too complex**:
```
"In a tropical environment characterized by high humidity and dense vegetation, a lepidopteran specimen of the morpho genus exhibits iridescent blue coloration..."
```

### Prompt Structure

**Recommended format**:
1. **Subject**: What butterfly species?
2. **Colors/Details**: Wing patterns, colors
3. **Action**: Flying, resting, feeding
4. **Environment**: Background, lighting

**Example**:
```
Subject: "A swallowtail butterfly"
Colors: "with yellow and black striped wings"
Action: "feeding on"
Environment: "purple lavender flowers in a sunny garden"
```

## Troubleshooting

### Model Loading Error

```
FileNotFoundError: model.pth not found
```

**Solution**: Verify checkpoint path in script (line 247):

```python
# inference_dit_model.py line 247
state = torch.load("/path/to/your/checkpoints/iter_5000/model.pth")
```

Or copy `model.pth` to working directory:

```bash
cp checkpoints/dit_butterfly/iter_5000/model.pth .
```

### Out of Memory Errors

```
RuntimeError: CUDA out of memory
```

**Solution 1**: Reduce resolution/frames:

```bash
--height 480 --width 848 --num-video-frames 61
```

**Solution 2**: Increase context parallelism:

```bash
torchrun --nproc-per-node 4 \
    examples/megatron/recipes/dit/inference_dit_model.py \
    --cp-size 4 \
    ...
```

### T5 Encoder Download Fails

```
ConnectionError: Failed to download T5 model
```

**Solution**: Set cache and download manually:

```bash
export HF_HOME=/path/to/cache
export TRANSFORMERS_CACHE=/path/to/cache

python -c "from transformers import T5EncoderModel, T5TokenizerFast; \
    T5TokenizerFast.from_pretrained('google-t5/t5-11b', cache_dir='/path/to/cache'); \
    T5EncoderModel.from_pretrained('google-t5/t5-11b', cache_dir='/path/to/cache')"
```

Then specify cache in script:

```python
# inference_dit_model.py line 150 (prepare_data_batch)
tokenizer = T5TokenizerFast.from_pretrained("google-t5/t5-11b", cache_dir="/path/to/cache")
text_encoder = T5EncoderModel.from_pretrained("google-t5/t5-11b", cache_dir="/path/to/cache")
```

### Cosmos Tokenizer Error

```
FileNotFoundError: Cosmos-0.1-Tokenizer-CV4x8x8 not found
```

**Solution**: Download tokenizer explicitly:

```python
from dfm.src.common.tokenizers.cosmos.cosmos1.causal_video_tokenizer import CausalVideoTokenizer

# Pre-download
vae = CausalVideoTokenizer.from_pretrained("Cosmos-0.1-Tokenizer-CV4x8x8")
```

### Poor Quality Results

**Symptoms**: Blurry, artifacts, doesn't match prompt

**Solutions**:
1. **Increase sampling steps**: `--num-steps 50` or `100`
2. **Increase guidance**: `--guidance 10.0` or `12.0`
3. **Check checkpoint**: Ensure model trained sufficiently (>5000 iters)
4. **Refine prompt**: More specific, descriptive
5. **Try different seeds**: `--seed` values

## Customize Inference Script

### Load Different Checkpoint

Edit `inference_dit_model.py` line 244-252:

```python
# Load from specific checkpoint
checkpoint_path = "/path/to/checkpoints/iter_10000/model.pth"
state = torch.load(checkpoint_path)

new_state = {}
for key, value in state.items():
    if "extra_state" in key:
        continue
    new_state[key.replace("0.module.", "")] = value

model.load_state_dict(new_state, strict=False)
```

### Batch Generation

Generate multiple videos from list of prompts:

```python
# Add to inference_dit_model.py after line 78
prompts = [
    "A monarch butterfly on a sunflower",
    "A blue butterfly in rain",
    "A white butterfly near water"
]

for idx, prompt in enumerate(prompts):
    args.prompt = prompt
    args.video_save_path = f"butterfly_{idx}.mp4"
    main(args)
```

## Next Steps

After generating videos:

1. **Evaluate quality**: Compare to training data and expectations
2. **Iterate on training**: Adjust training if quality is poor
3. **Scale up**: Train on larger datasets for better results
4. **Production deployment**: Optimize inference for serving

## Related Pages

- **[Megatron Training](training.md)**: Train better models for improved generation
- **[Distributed Training](../../about/concepts/distributed-training.md)**: Optimize multi-GPU inference
- **[Diffusion Models](../../about/concepts/diffusion-models.md)**: Understand EDM sampling

