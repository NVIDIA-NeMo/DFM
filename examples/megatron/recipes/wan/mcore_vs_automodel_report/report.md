# Wan 2.1 – Partial Convergence Comparison  
### Diffusers (Automodel path) vs. Megatron-Core (Megatron-Bridge path)  

---

## 1. Experiment Overview
- Goal: Compare two training paths for Wan 2.1:  
  **(1) [Diffusers](https://huggingface.co/docs/diffusers/en/index) implementaion + [Automodel](https://github.com/NVIDIA-NeMo/Automodel/tree/diffusion) training path** vs. **(2) [Megatron-Core](https://github.com/NVIDIA/Megatron-LM) implementaion + [Megatron-Bridge](https://github.com/NVIDIA-NeMo/Megatron-Bridge) training path**
- Two-stage training:
  - **Stage 1:** Text → Image - Learn to connect textual embeddings with visual concepts.
  - **Stage 2:** Text → Video - Learn visual movements aligning with prompts.
- Dataset: 3,000 videos; frames extracted from videos are used for text-to-image training stage.


## 2. Dataset

### Stage 1 (Text-to-Image)
- Extract 40 frames per video → **120k images**
- Resolution: **240 × 416**
- Each frame uses same caption as parent video.

### Stage 2 (Text-to-Video)
- Full videos → **3,000 videos**
- Resolution: **240 × 416**, duration 4–8 seconds.

**Note**: This experiment is a partial convergence test and only demonstrates the model's ability to reconstruct images and videos from input prompts. With only 3,000 videos, the model cannot generalize to generate novel content. Such generalization can be achieved with larger training dataset and increased training resources.

## 3. Training Setup

### Stage 1
- Global batch size: 2560 images
- Learning rate: warmup 10k → 5e-5 constant
- Hardware: 10 nodes (80 GPUs)

| Path | Parallelism | Notes |
|------|-------------|-------|
| Megatron-Core | TP=1, PP=1, CP=1 | Sequence packing (32 samples/pack) |
| Automodel | FSDP | micro_batch_size = 32 |

### Stage 2
- Global batch size: 80 videos
- Learning rate: 5e-5 constant
- Hardware: 10 nodes (80 GPUs)

| Path | Parallelism | Notes |
|------|-------------|-------|
| Megatron-Core | TP=1, PP=1, CP=1 | micro_batch_size = 1 |
| Automodel | FSDP | micro_batch_size = 1 |


## 4. Results
#### Stage 1 — Loss vs. Steps
<img src="./medias/training_curves/lm_loss_text2image_3kvids.png" width="700">

#### Stage 2 — Loss vs. Steps
<img src="./medias/training_curves/lm_loss_text2video_3kvids.png" width="700">
**Note**: Training loss is smoothened with 50 steps averaging.


The training curves for both stages have similar value ranges, although they do not match exactly. This is expected due to differences in implementation and training loop setups.

One important caveat: In the current Megatron-Core implementation, the same diffusion time steps are applied to all samples within a pack for each step, rather than different time steps for each sample. As a result, the training loss for Megatron-Core fluctuates more significantly than for AutoModel, especially at the beginning of training.