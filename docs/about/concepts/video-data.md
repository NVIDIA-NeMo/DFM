---
description: "How video data is represented in NeMo DFM: latents, VAE encoding, tokenization, and data formats"
categories: ["concepts-architecture"]
tags: ["data", "video", "latents", "vae", "tokenization"]
personas: ["data-scientist-focused", "mle-focused"]
difficulty: "intermediate"
content_type: "explanation"
---

(about-concepts-video-data)=

# Video Data Representation

NeMo DFM processes videos in latent space rather than pixel space, reducing memory requirements and accelerating training by up to 64×.

## Overview

Videos in DFM follow a four-stage pipeline:

1. **Encode to latents**: VAE (Variational Autoencoder) compresses raw pixels into latent space
2. **Store as tensors**: Compressed latents are saved with text embeddings
3. **Process with diffusion**: Models operate on compact latent representations
4. **Decode to pixels**: VAE reconstructs final video frames

**Key benefit**: A 1080p video (1920×1080×3 channels×120 frames = 746 million values) compresses to latents of 16×15×135×240 = 8.6 million values—a 64× reduction.

## Video Latents

### Tensor Format

Video latents are 4D tensors with shape `(C, T, H, W)`:

| Dimension | Description | Example Values |
|-----------|-------------|----------------|
| **C** | Channels | 16 (standard for most VAEs) |
| **T** | Temporal frames | 15, 30, 60, 120 (varies by video length) |
| **H** | Latent height | 135 for 1080p (1080÷8) |
| **W** | Latent width | 240 for 1920p (1920÷8) |

**Spatial compression**: VAEs downsample by 8× in both height and width. A 1920×1080 frame becomes 240×135 in latent space.

**Temporal compression**: Some VAEs also compress temporally. A 120-frame video might compress to 15 latent frames (8× temporal compression).

### Why Latents?

**Memory efficiency**: Latent representation is 64× smaller than raw pixels.

- Raw 1080p video (120 frames): 746 MB
- Latent representation: 12 MB
- Enables training on longer videos with limited GPU memory

**Training speed**: Diffusion models process 8.6 million values instead of 746 million values—approximately 8× faster per iteration.

**Quality preservation**: VAE reconstruction maintains perceptual quality. Peak Signal-to-Noise Ratio (PSNR) remains above 30 dB for most VAE models.

## VAE Encoding and Decoding

### Encoding Process

The VAE encoder transforms raw video frames into compact latent tensors:

```python
import torch
from diffusers import AutoencoderKLWan

# Load video: (batch, channels, time, height, width)
video_frames = torch.randn(1, 3, 120, 1080, 1920)  # 1080p, 120 frames

# Normalize to [-1, 1] range
video_frames = video_frames * 2.0 - 1.0

# Initialize VAE (WAN 2.1)
vae = AutoencoderKLWan.from_pretrained(
    "Wan-AI/Wan2.1-T2V-14B-Diffusers",
    subfolder="vae"
)

# Encode to latents
latent_dist = vae.encode(video_frames)
latents = latent_dist.latent_dist.mean  # Use mean for deterministic encoding
# Output shape: (1, 16, 120, 135, 240)
# Compression: 1× in time (no temporal compression), 8× in height, 8× in width
```

**Encoding steps**:

1. Normalize input frames to VAE's expected range (usually [-1, 1])
2. Pass through encoder network
3. Quantize or sample latent distribution
4. Output compressed latent tensor

### Decoding Process

The VAE decoder reconstructs video frames from latents:

```python
# Generate or load latents
latents = torch.randn(1, 16, 120, 135, 240)

# Decode to video frames
reconstructed_video = vae.decode(latents).sample
# Output shape: (1, 3, 120, 1080, 1920)

# Denormalize from [-1, 1] to [0, 255] for video output
video_uint8 = ((reconstructed_video + 1.0) * 127.5).clamp(0, 255).to(torch.uint8)
```

**Decoding steps**:

1. Pass latents through decoder network
2. Upsample to original spatial and temporal resolution
3. Denormalize to pixel value range
4. Output reconstructed video frames

### VAE Models

DFM supports multiple VAE architectures:

**Cosmos Tokenizer** (Continuous Video: `Cosmos-Tokenizer-CV8x8x8`):

- Compression: 8×8×8 (time × height × width)
- Channels: 16 latent channels
- Use case: DiT models, continuous latent diffusion
- Normalization: Input frames in [-1, 1]

**Cosmos Tokenizer** (Discrete Video: `Cosmos-Tokenizer-DV4x8x8`):

- Compression: 4×8×8 (time × height × width)
- Channels: 6 discrete code channels (codebook size 64K)
- Use case: Autoregressive models, discrete token generation
- Normalization: Input frames in [-1, 1]

**WAN VAE**:

- Compression: 1×8×8 (no temporal compression)
- Channels: 16 latent channels
- Use case: WAN models, Flow Matching models
- Normalization: Input frames converted to [-1, 1] internally

Each VAE requires specific normalization. Check model documentation before preprocessing.

## Data Formats

### Training Data Formats

DFM supports two paradigms with different data formats:

#### Automodel Format

Automodel uses pickled `.meta` files containing preprocessed latents:

```python
# Example .meta file structure
{
    "video_latents": torch.Tensor,         # Shape: (C, T, H, W)
    "text_embeddings": torch.Tensor,       # Shape: (S, D)
    "first_frame": np.ndarray,             # First frame (H, W, 3) in [0, 255]
    "metadata": dict,                      # Original video metadata
    "num_frames": int,                     # Frame count
    "original_filename": str,              # Source video filename
    "original_video_path": str,            # Source video path
    "deterministic_latents": bool,         # Encoding mode used
    "memory_optimization": bool,           # Memory optimization enabled
    "model_version": str,                  # VAE model version (e.g., "wan2.1")
    "resize_settings": dict,               # Resize configuration
}
```

**File organization**:

```text
dataset/
├── sample_0000.meta
├── sample_0001.meta
├── sample_0002.meta
└── ...
```

#### Megatron Format

Megatron supports two distributed data formats:

**Webdataset format**:

- Tar archives containing video samples
- Each sample is a set of files with shared basename
- Example: `sample001.latent.pth`, `sample001.text.pth`, `sample001.json`

**Energon format**:

- Optimized for distributed data loading across nodes
- Supports efficient sharding and data parallelism
- Recommended for multi-node training at scale

Both formats include latents, text embeddings, and metadata per sample.

### DiffusionSample Structure

The `DiffusionSample` class represents a training sample:

```python
@dataclass
class DiffusionSample:
    video: torch.Tensor              # Video latents (C, T, H, W)
    context_embeddings: torch.Tensor  # Text embeddings (S, D)
    context_mask: torch.Tensor       # Text mask
    image_size: torch.Tensor         # [height, width]
    fps: torch.Tensor                # Frame rate
    num_frames: torch.Tensor         # Frame count
    # ... additional metadata
```

## Text Conditioning

### Text Embeddings

Text prompts guide video generation through learned embeddings. DFM uses T5 or similar transformer-based text encoders.

**Embedding dimensions**:

| Encoder | Sequence Length (S) | Embedding Dim (D) | Model Size |
|---------|---------------------|-------------------|------------|
| T5-Base | Up to 512 tokens | 768 | 220M params |
| T5-Large | Up to 512 tokens | 1024 | 770M params |
| T5-XXL | Up to 512 tokens | 4096 | 11B params |

**Process**: Text → Tokenizer → Token IDs → Encoder → Embeddings `(S, D)`

### Text Encoding Example

```python
from transformers import AutoTokenizer, UMT5EncoderModel
import torch

# Initialize UMT5 encoder (used by WAN models)
tokenizer = AutoTokenizer.from_pretrained(
    "Wan-AI/Wan2.1-T2V-14B-Diffusers",
    subfolder="text_encoder"
)
text_encoder = UMT5EncoderModel.from_pretrained(
    "Wan-AI/Wan2.1-T2V-14B-Diffusers",
    subfolder="text_encoder"
)

# Encode prompt
prompt = "A robot cooking pasta in a modern kitchen"
inputs = tokenizer(
    prompt,
    max_length=512,
    padding="max_length",
    truncation=True,
    return_tensors="pt",
    return_attention_mask=True,
)

with torch.no_grad():
    text_embeddings = text_encoder(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"]
    ).last_hidden_state
# Output shape: (1, 512, D) where D is embedding dimension

# Embeddings condition the diffusion model
# via cross-attention layers during generation
```

**Attention masking**: Padding tokens are masked so the model only attends to real tokens, not padding.

## Video Tokenization

Some models discretize continuous latents into tokens for autoregressive generation.

### Cosmos Video Tokenizer

The Cosmos tokenizer converts continuous latents into discrete token sequences:

**Process**:

1. Encode video to continuous latents: `(C, T, H, W)`
2. Quantize latents using learned codebook
3. Output discrete token indices: `(T×H×W,)` flattened sequence

**Use cases**:

- Autoregressive video models (predict next token)
- Enables language model-style training on videos
- Supports efficient caching during generation

### Causal Video Tokenizer

Causal tokenizers maintain temporal causality for autoregressive models:

- **Temporal masking**: Each frame can only see previous frames
- **Autoregressive generation**: Generate frame-by-frame sequentially
- **Architecture compatibility**: Required for GPT-style video models

**Example**: Generating a 120-frame video autoregressively produces frames 1→2→3→...→120, where each frame conditions on all previous frames.

## Sequence Packing

Sequence packing improves GPU utilization during distributed training:

**Without packing**:

```text
Batch 1: [sequence_A (50 tokens), padding (14 tokens)]  # 22% wasted
Batch 2: [sequence_B (40 tokens), padding (24 tokens)]  # 37% wasted
```

**With packing**:

```text
Batch 1: [sequence_A (50 tokens), sequence_B (14 tokens)]  # 0% wasted
```

**Implementation**:

- Combine multiple sequences into fixed-length batches
- Use attention masks to separate sequences
- Track sequence boundaries for gradient computation

**Benefits**: Up to 2× throughput improvement on datasets with variable-length videos.

## Data Preprocessing

### Preparation Pipeline

Preprocessing transforms raw videos into training-ready samples:

1. **Load raw video**: Read MP4, AVI, or other video formats
2. **Resize and crop**: Standardize to target resolution (for example, 1080p)
3. **Normalize frames**: Convert to expected range ([-1, 1] or [0, 1])
4. **Encode to latents**: Apply VAE encoder
5. **Encode text prompts**: Apply text encoder
6. **Package sample**: Create `DiffusionSample` with metadata
7. **Save to disk**: Write as `.meta` file or webdataset entry

**Batch processing**: Process videos in parallel to maximize throughput. Use multi-GPU encoding for large datasets.

### Preprocessing Example

```python
from dfm.src.automodel.utils.data.preprocess_resize import VideoPreprocessor
from pathlib import Path

# Initialize preprocessor
preprocessor = VideoPreprocessor(
    video_folder="raw_videos",
    wan21_model_id="Wan-AI/Wan2.1-T2V-14B-Diffusers",
    output_folder="processed_meta",
    device="cuda",
    deterministic_latents=True,  # Use deterministic encoding (no flares)
    target_size=(1080, 1920),    # Target resolution (height, width)
    resize_mode="bilinear",
    maintain_aspect_ratio=True,
)

# Process all videos in folder
# Requires meta.json with video metadata in video_folder
preprocessor.process_all_videos()

# Or load existing processed data
data = preprocessor.load_processed_data("sample_0000.meta")

# Data contains:
# - video_latents: (16, T, 135, 240)
# - text_embeddings: (1, 512, D)
# - first_frame: (1080, 1920, 3)
# - metadata: Original video metadata
```

### Preprocessing Tools

DFM provides command-line tools and Python APIs:

**Command-line preprocessing**:

```bash
python dfm/src/automodel/utils/data/preprocess_resize.py \
    --video_folder raw_videos/ \
    --output_folder processed_meta/ \
    --model Wan-AI/Wan2.1-T2V-14B-Diffusers \
    --height 1080 \
    --width 1920 \
    --resize_mode bilinear \
    --device cuda
```

**Python API**:

- `VideoPreprocessor`: End-to-end video preprocessing (`dfm.src.automodel.utils.data.preprocess_resize`)
- `AutoencoderKLWan.encode()` / `.decode()`: Manual latent encoding (Diffusers library)
- `UMT5EncoderModel`: Text prompt encoding (Transformers library)
- `DiffusionSample`: Training sample dataclass (`dfm.src.megatron.data.common.diffusion_sample`)

## Metadata

Each training sample includes metadata for proper model conditioning:

| Metadata Field | Type | Purpose | Example |
|----------------|------|---------|---------|
| **image_size** | `(int, int)` | Original video resolution | `(1080, 1920)` |
| **fps** | `int` | Frame rate | `24`, `30`, `60` |
| **num_frames** | `int` | Total frame count | `120` |
| **padding_mask** | `torch.Tensor` | Valid vs padded regions | Binary mask |
| **position_ids** | `torch.Tensor` | Spatial/temporal positions | 3D position indices |

**Why metadata matters**:

- **Resolution conditioning**: Models can generate videos at different resolutions
- **FPS conditioning**: Control playback speed and motion dynamics
- **Frame count conditioning**: Generate videos of varying lengths
- **Padding masks**: Prevent model from learning on invalid padded regions

**Example usage**:

```python
# Model conditions on metadata during training
loss = model(
    latents=sample.video,
    text_embeddings=sample.context_embeddings,
    image_size=sample.image_size,  # Conditions generation
    fps=sample.fps,                # Conditions motion dynamics
    num_frames=sample.num_frames,  # Conditions temporal length
)
```
