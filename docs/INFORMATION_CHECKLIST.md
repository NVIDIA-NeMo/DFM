# Information Preservation Checklist

**Purpose**: Verify all unique information from old docs is captured in new structure.

**How to Use**: Check off each item as it's integrated into the new docs. Items can be integrated anywhere logical in the new IA.

---

## 1. Performance Benchmarks (`performance-summary.md`)

**Target Location**: `docs/reference/performance.md` (REFERENCE)

### Nomenclature Definitions
- [ ] **GBS**: Global Batch Size
- [ ] **MBS**: Micro Batch Size
- [ ] **FSDP**: Fully Sharded Data Parallel
  - [ ] FSDP = 1: use FSDP
  - [ ] FSDP = 0: use DDP (Distributed Data Parallel)
- [ ] **TP**: Tensor Parallel Size
- [ ] **SP**: Sequence Parallel
- [ ] **PP**: Pipeline Parallel Size
- [ ] **CP**: Context Parallel Size
- [ ] **VP**: Virtual Pipeline Parallel Size
- [ ] **EP**: Expert Parallel Size

### Performance Metrics
- [ ] **Tokens/sec/GPU**: Throughput per GPU (explanation)
- [ ] **Model TFLOP/sec/GPU**: Model floating-point operations per second per GPU (explanation)

### Benchmark Tables

#### Megatron-Core Pre-Training Performance

**DGX-GB200**:
- [ ] WAN 2.1 14B benchmark row (32 GPUs, GBS=64, MBS=1, SeqLen=37440, FSDP=0, TP=1, SP=0, PP=1, CP=4, VP=0, EP=0, TFLOP=787.59)

**DGX-GB300**:
- [ ] WAN 2.1 14B benchmark row (32 GPUs, GBS=64, MBS=1, SeqLen=37440, FSDP=0, TP=1, SP=0, PP=1, CP=2, VP=0, EP=0, TFLOP=1,022.26)

**DGX-H100**:
- [ ] WAN 2.1 14B benchmark row (128 GPUs, GBS=128, MBS=1, SeqLen=37440, FSDP=0, TP=2, SP=1, PP=1, CP=4, VP=0, EP=0, TFLOP=325.77)

#### NeMo Automodel Pre-Training Performance

**DGX-H100**:
- [ ] WAN 2.1 14B benchmark row (8 GPUs, GBS=8, MBS=1, SeqLen=37440, FSDP=1, DP=8, TP=1, SP=1, PP=1, CP=1, VP=0, EP=0, TFLOP=175.88)
- [ ] WAN 2.1 14B benchmark row (64 GPUs, GBS=64, MBS=1, SeqLen=37440, FSDP=1, DP=64, TP=1, SP=1, PP=1, CP=1, VP=0, EP=0, TFLOP=228.85)

### Context Information
- [ ] Note about referring to `examples/megatron/recipes/wan/conf` for updated YAML configs
- [ ] Statement about ongoing optimization

---

## 2. Paradigm Comparison (`mcore_automodel_comparision_wan21.md`)

**Target Location**: `docs/about/comparison.md` OR integrate into `docs/about/concepts/training-paradigms.md` (EXPLANATION)

### Experiment Overview
- [ ] Goal: Compare two training paths for WAN 2.1
- [ ] Path 1: Diffusers + Automodel training path (with links)
- [ ] Path 2: Megatron-Core + Megatron-Bridge training path (with links)
- [ ] Two-stage training approach explanation
- [ ] Dataset: 3,000 videos (frames extracted for Stage 1)

### Stage 1: Text-to-Image
- [ ] Extract 40 frames per video → 120k images
- [ ] Resolution: 240 × 416
- [ ] Each frame uses same caption as parent video
- [ ] Global batch size: 2560 images
- [ ] Learning rate: warmup 10k → 5e-5 constant
- [ ] Hardware: 10 nodes (80 GPUs)
- [ ] Megatron-Core parallelism: TP=1, PP=1, CP=1, Sequence packing (32 samples/pack)
- [ ] Automodel parallelism: FSDP, micro_batch_size = 32
- [ ] Training curve image: `lm_loss_text2image_3kvids.png`

### Stage 2: Text-to-Video
- [ ] Full videos → 3,000 videos
- [ ] Resolution: 240 × 416, duration 4–8 seconds
- [ ] Global batch size: 80 videos
- [ ] Learning rate: 5e-5 constant
- [ ] Hardware: 10 nodes (80 GPUs)
- [ ] Megatron-Core parallelism: TP=1, PP=1, CP=1, micro_batch_size = 1
- [ ] Automodel parallelism: FSDP, micro_batch_size = 1
- [ ] Training curve image: `lm_loss_text2video_3kvids.png`

### Results Analysis
- [ ] Note: Training loss smoothed with 50 steps averaging
- [ ] Observation: Training curves have similar value ranges but don't match exactly
- [ ] Explanation: Expected due to differences in implementation and training loop setups
- [ ] **Critical Caveat**: Megatron-Core applies same diffusion timesteps to all samples in pack (not different timesteps per sample)
- [ ] **Critical Caveat**: Training loss for Megatron-Core fluctuates more than AutoModel, especially at beginning

### Context Notes
- [ ] Note: Partial convergence test (3K videos insufficient for generalization)
- [ ] Note: Only demonstrates reconstruction ability, not novel generation

---

## 3. Automodel Training Information (`automodel_training_doc.md`)

**Target Location**: Integrate into `docs/get-started/automodel.md` (TUTORIAL with progressive disclosure)

### Overview
- [ ] Currently Supported: WAN 2.1 Text-to-Video (1.3B and 14B models)

### Docker Setup
- [ ] Build command: `docker build -f docker/Dockerfile.ci -t dfm-training .`
- [ ] Run command with all flags (--gpus, -v mounts, --ipc=host, ulimit settings)
- [ ] Inside container: Initialize submodules command

### Data Preparation

#### Dataset Options
- [ ] Option 1: Start with raw videos (use data-preparation scripts)
- [ ] Option 2: Bring your own `meta.json`

#### Dataset Structure
- [ ] Folder structure example (`<your_video_folder>/` with videos and `meta.json`)
- [ ] Note about per-video `.jsonl` captions being picked up automatically

#### meta.json Schema
- [ ] Complete JSON schema with all fields:
  - [ ] `file_name`
  - [ ] `width`
  - [ ] `height`
  - [ ] `start_frame`
  - [ ] `end_frame`
  - [ ] `vila_caption`
- [ ] Example with two video entries

#### Preprocessing Modes

**Full Video Mode (`--mode video`)**:
- [ ] What it is: Converts each source video into single `.meta` preserving full temporal sequence
- [ ] When to use: Fine-tuning text-to-video models where motion/temporal consistency matter
- [ ] Status: Recommended default for most training runs
- [ ] Command example with all flags
- [ ] Output: Creates one `.meta` file per video

**Extract Frames Mode (`--mode frames`)**:
- [ ] What it is: Uniformly samples N frames, writes each as one-frame `.meta` sample
- [ ] When to use: Image/frame-level training, quick smoke tests, ablations
- [ ] Command example with `--num-frames` flag
- [ ] Output: Creates one `.meta` file per frame

#### Preprocessing Key Arguments
- [ ] `--mode`: `video` or `frames` explanation
- [ ] `--num-frames`: Number of frames to extract (frames mode only)
- [ ] `--height/--width`: Target resolution
- [ ] `--center-crop`: Crop to exact size after aspect-preserving resize

#### Preprocessing Output
- [ ] Encoded video latents (normalized)
- [ ] Text embeddings (from UMT5)
- [ ] First frame as JPEG (video mode only)
- [ ] Metadata

### Training

#### Single-Node Training
- [ ] Command: `uv run --group automodel --with . torchrun --nproc-per-node=8 ...`
- [ ] Config file: `examples/automodel/finetune/wan2_1_t2v_flow.yaml`
- [ ] Note about `UV_PROJECT_ENVIRONMENT` export

#### Multi-Node SLURM Training
- [ ] Complete SLURM script with all SBATCH directives
- [ ] MASTER_ADDR setup from SLURM_JOB_NODELIST
- [ ] MASTER_PORT setup
- [ ] Per-rank UV cache setup to avoid conflicts
- [ ] UV_CACHE_DIR per job/rank
- [ ] torchrun command with multi-node flags
- [ ] Config file: `wan2_1_t2v_flow_multinode.yaml`

### Validation

#### Validation Script Details
- [ ] Purpose: Quick qualitative check of trained checkpoint
- [ ] Reads prompts from `.meta` files in `--meta_folder`
- [ ] Uses `metadata.vila_caption` (latents ignored)
- [ ] Loads `WanPipeline`
- [ ] Checkpoint loading priority: `ema_shadow.pt` → `consolidated_model.bin` → sharded FSDP `model/*.distcp`
- [ ] Generation settings: `--guidance_scale`, `--num_inference_steps`, `--height/--width`, `--num_frames`, `--fps`, `--seed`
- [ ] Output: Writes videos to `--output_dir`
- [ ] Note: Qualitative comparison only, no quantitative metrics
- [ ] Command example
- [ ] Note: `--checkpoint ./checkpoints/LATEST` automatically uses most recent checkpoint

### Configuration

#### Fine-tuning Config (`wan2_1_t2v_flow.yaml`)
- [ ] Complete YAML config with all sections:
  - [ ] `model.pretrained_model_name_or_path`
  - [ ] `step_scheduler` (global_batch_size, local_batch_size, num_epochs, ckpt_every_steps)
  - [ ] `data.dataloader` (meta_folder, num_workers)
  - [ ] `optim.learning_rate`
  - [ ] `flow_matching` (timestep_sampling, flow_shift)
  - [ ] `fsdp.dp_size`
  - [ ] `checkpoint` (enabled, checkpoint_dir)
- [ ] Note about canonical files in repository

#### Multi-Node Config Differences
- [ ] `fsdp.dp_size`: Total data-parallel replicas (2 nodes × 8 GPUs = 16)
- [ ] `fsdp.dp_replicate_size`: Number of replicated groups across nodes (2)

#### Pretraining vs Fine-tuning Comparison Table
- [ ] `learning_rate`: Fine-tuning (5e-6) vs Pretraining (5e-5)
- [ ] `weight_decay`: Fine-tuning (0.01) vs Pretraining (0.1)
- [ ] `flow_shift`: Fine-tuning (3.0) vs Pretraining (2.5)
- [ ] `logit_std`: Fine-tuning (1.0) vs Pretraining (1.5)
- [ ] Dataset size: Fine-tuning (100s-1000s) vs Pretraining (10K+)

### Hardware Requirements Table
- [ ] GPU: Minimum (A100 40GB) vs Recommended (A100 80GB / H100)
- [ ] GPUs: Minimum (4) vs Recommended (8+)
- [ ] RAM: Minimum (128 GB) vs Recommended (256 GB+)
- [ ] Storage: Minimum (500 GB SSD) vs Recommended (2 TB NVMe)

### Features List
- [ ] Flow Matching: Pure flow matching training
- [ ] Distributed: FSDP2 + Tensor Parallelism
- [ ] Mixed Precision: BF16 by default
- [ ] WandB: Automatic logging
- [ ] Checkpointing: consolidated and sharded formats
- [ ] Multi-node: SLURM and torchrun support

### Supported Models Table
- [ ] WAN 2.1 T2V 1.3B: 1.3B params, FSDP2 via Automodel + DDP, Status ✅
- [ ] WAN 2.1 T2V 14B: 14B params, FSDP2 via Automodel + DDP, Status ✅
- [ ] FLUX: TBD params, TBD parallelization, Status 🔄 In Progress

### Advanced Topics

#### Custom Parallelization
- [ ] Example YAML: `fsdp.tp_size: 2`, `fsdp.dp_size: 4`

#### Checkpoint Cleanup
- [ ] Python function: `cleanup_old_checkpoints(checkpoint_dir, keep_last_n=3)`
- [ ] Complete code example with Path and shutil usage

---

## 4. DiT Model Information (`megatron/models/dit/README.md`)

**Target Location**: Integrate into `docs/get-started/megatron.md` (TUTORIAL with progressive disclosure)

### Overview
- [ ] DiT description: Open-source implementation of Diffusion Transformers
- [ ] Purpose: Training text-to-image/video models with EDM Pipeline
- [ ] Based on: Megatron-Core and Megatron-Bridge
- [ ] Parallelism support: Tensor, sequence, and context parallelism

### Dataset Preparation

#### Energon Data Loader
- [ ] Uses NVIDIA's Megatron-Energon
- [ ] WebDataset-compatible format (sharded `.tar` archives)
- [ ] Supports: Large-scale distributed loading, sharding, sampling for multi-modal pairs
- [ ] Set `dataset.path` to WebDataset location or shard pattern

#### Butterfly Dataset Example
- [ ] Dataset: `huggan/smithsonian_butterflies_subset` on Hugging Face
- [ ] Script: `prepare_energon_dataset_butterfly.py`
- [ ] Command with `--nproc-per-node`
- [ ] Optional arguments: `--t5_cache_dir`, `--tokenizer_cache_dir`

#### Energon Prepare Workflow
- [ ] Command: `energon prepare $dataset_path`
- [ ] Interactive prompts explanation:
  - [ ] Train/val/test split entry (e.g., "1,0,0")
  - [ ] Sample type selection: "Crude sample (plain dict for cooking)" (option 11)
- [ ] Sample structure: keys include `json`, `pickle`, `pth`
- [ ] Sample JSON content example (`image_height`, `image_width`)
- [ ] Note: CrudeWebdataset doesn't need field map
- [ ] Note: Need to provide `Cooker` in `TaskEncoder`
- [ ] Note: Can add `subflavors` in meta dataset specification

### Container Build
- [ ] Reference to container section in main README

### Pretraining

#### Sequence Packing
- [ ] Purpose: Maximize training efficiency
- [ ] How it works: Stacks multiple samples into single sequence instead of padding
- [ ] Requirement: `micro_batch_size` must be set to 1
- [ ] Requirement: `qkv_format` should be set to `thd` (signals Transformer Engine)
- [ ] Link to NeMo sequence packing documentation

#### Sequence Packing Parameters
- [ ] `task_encoder_seq_length`: Controls maximum sequence length passed to model
- [ ] `packing_buffer_size`: Determines number of samples processed to create buckets
- [ ] Reference to `select_samples_to_pack` and `pack_selected_samples` methods
- [ ] Link to DiffusionTaskEncoderWithSequencePacking code
- [ ] Link to Energon packing documentation

#### Parallelism
- [ ] Multiple parallelism techniques supported (tensor, sequence, context)
- [ ] Configurable based on computational requirements

#### Model Architecture Customization
- [ ] Parameters: `num_layers`, `num_attention_heads`
- [ ] Link to Megatron-Bridge documentation for comprehensive options

#### WandB Notes
- [ ] If using `wandb_project` and `wandb_exp_name`, export `WANDB_API_KEY`

#### Validation Details
- [ ] Model generates one sample per GPU at start of each validation round
- [ ] Samples saved to `validation_generation` folder within `checkpoint_dir`
- [ ] Logged to WandB if `WANDB_API_KEY` configured
- [ ] Requires access to video tokenizer used during dataset preparation
- [ ] Specify VAE artifacts location using `vae_cache_folder` argument
- [ ] Otherwise downloaded in first validation round

#### Pretraining Script Example
- [ ] Copy config file: `cp examples/megatron/recipes/dit/conf/dit_pretrain_example.yaml ...`
- [ ] Edit instructions for `my_config.yaml`:
  - [ ] `model.vae_cache_folder`: Path to VAE cache folder
  - [ ] `dataset.path`: Path to dataset folder
  - [ ] `checkpoint.save` and `checkpoint.load`: Path to checkpoint folder
  - [ ] `train.global_batch_size`: Set to be divisible by NUM_GPUs
  - [ ] `logger.wandb_exp_name`: Your experiment name
- [ ] Run command with `--config-file`
- [ ] CLI override example: `train.train_iters=20000`, `model.num_layers=32`

#### Training Split Note
- [ ] If 100% data to training, pass `dataset.use_train_split_for_val=true`
- [ ] Uses subset of training data for validation
- [ ] Command example with this flag

#### Mock Dataset
- [ ] Use `--mock` flag for performance measurement without dataset
- [ ] Command example with `--mock` flag

### Inference

#### Inference Script
- [ ] Script: `inference_dit_model.py`
- [ ] Requires: Trained checkpoint (`--checkpoint_path`), save path (`--video_save_path`)
- [ ] Optional: `--t5_cache_dir`, `--tokenizer_cache_dir` (avoid re-downloading)
- [ ] Command example with all parameters:
  - [ ] `--t5_cache_dir`
  - [ ] `--tokenizer_cache_dir`
  - [ ] `--tokenizer_model Cosmos-0.1-Tokenizer-CV4x8x8`
  - [ ] `--checkpoint_path`
  - [ ] `--num_video_frames 10`
  - [ ] `--height 240`
  - [ ] `--width 416`
  - [ ] `--video_save_path`
  - [ ] `--prompt`

### Parallelism Support Table
- [ ] DiT-S (330M): Data Parallel (TBD), Tensor Parallel (TBD), Sequence Parallel (TBD), Context Parallel (TBD)
- [ ] DiT-L (450M): Data Parallel (TBD), Tensor Parallel (TBD), Sequence Parallel (TBD), Context Parallel (TBD)
- [ ] DiT-XL (700M): Data Parallel (✅), Tensor Parallel (✅), Sequence Parallel (✅), Context Parallel (✅)

---

## 5. WAN Recipe Information (`megatron/recipes/wan/wan2.1.md`)

**Target Location**: `docs/get-started/megatron-wan.md` OR integrate into `docs/get-started/megatron.md` with tabs (TUTORIAL/HOW-TO)

### Overview
- [ ] WAN 2.1 description: Open-source implementation of large-scale text-to-video/image generative models
- [ ] Built on: Megatron-Core and Megatron-Bridge
- [ ] Supports: Advanced parallelism strategies (data, tensor, sequence, context parallelism)
- [ ] Optimized kernels: Transformer Engine fused attention

### Dataset Preparation

#### Energon Data Loader
- [ ] Uses NVIDIA's Megatron-Energon
- [ ] WebDataset-compatible format (sharded `.tar` archives)
- [ ] Supports: Large-scale distributed loading, sharding, sampling for video-text and image-text pairs
- [ ] Set `dataset.path` to WebDataset directory or shard pattern
- [ ] Link to Megatron-Energon docs for format details, subflavors, advanced options

#### Mock Dataset Note
- [ ] If no dataset: See "Quick Start with Mock Dataset" section

#### WAN Dataset Preparation Example
- [ ] Input: Directory with raw `.mp4` videos and `.json` metadata files with captions
- [ ] Output: WAN-ready WebDataset shards
- [ ] Step 1: Define input/output folders (`DATASET_SRC`, `DATASET_PATH`)
- [ ] Step 2: Optional HF_TOKEN export if auth required
- [ ] Step 3: Create WAN shards with latents + text embeddings
  - [ ] Script: `prepare_energon_dataset_wan.py`
  - [ ] Uses WAN's VAE encoder and T5 encoder
  - [ ] Extracts videos' latents and caption embeddings offline
  - [ ] Arguments: `--height/--width` control resize target (832x480 supported for 1.3B and 14B)
  - [ ] `--center-crop`: Run center crop to exact target size after resize
  - [ ] Command example with all flags
- [ ] Step 4: Use Energon to process shards
  - [ ] Command: `energon prepare "${DATASET_PATH}"`
  - [ ] Interactive prompts: Enter train/val/test split (e.g., "8,1,1")
  - [ ] Sample type: Choose "Crude sample (plain dict for cooking)"

#### What Gets Produced
- [ ] Each shard contains:
  - [ ] `pth`: WAN video latents
  - [ ] `pickle`: Text embeddings
  - [ ] `json`: Useful side-info (text caption, sizes, processing choices)
- [ ] Energon writes `.nv-meta` directory with dataset info
- [ ] Energon writes `dataset.yaml` (can version/control)

#### Training Config Setup
- [ ] Point WAN config to processed data: `dataset.path=${DATASET_PATH}`

### Container Build
- [ ] Reference to DFM container guide in main README

### Pretraining

#### Sequence Packing for WAN
- [ ] Purpose: Maximize throughput
- [ ] Problem: Naive batching/padding requires significant padded tokens for videos
- [ ] Solution: Sequence packing stacks multiple samples (different resolutions) into single sequence
- [ ] Benefit: No computation wasted on padded tokens
- [ ] Requirements:
  - [ ] Set `train.micro_batch_size=1` and `dataset.micro_batch_size=1`
  - [ ] Ensure `model.qkv_format=thd` (required with context parallelism, recommended with sequence packing)

#### Parallelism
- [ ] Multiple parallelism techniques supported (tensor, sequence, context parallelism)
- [ ] Configurable per hardware

#### Training Script
- [ ] Script: `examples/megatron/recipes/wan/pretrain_wan.py`
- [ ] Supports: YAML config file and CLI overrides

#### Training Mode Presets
- [ ] `--training-mode` with `pretrain` and `finetune` presets
- [ ] Purpose: Flow-matching hyperparameters as starting point
- [ ] **Pretraining preset**:
  - [ ] Uses noisier, biased sampling
  - [ ] Examples: logit-normal, higher logit_std, lower flow_shift
  - [ ] Purpose: Stability and broad learning
- [ ] **Finetuning preset**:
  - [ ] Uses uniform, lower-noise settings
  - [ ] Examples: uniform sampling, lower logit_std, higher flow_shift
  - [ ] Purpose: Refine details and improve quality

#### WandB Notes
- [ ] If using `logger.wandb_project` and `logger.wandb_exp_name`, export `WANDB_API_KEY`

#### Pretraining Script Example
- [ ] Example configs: `wan_1_3B.yaml` and `wan_14B.yaml` under `examples/megatron/recipes/wan/conf`
- [ ] Copy and edit instructions:
  - [ ] `dataset.path`: Path to WebDataset directory
  - [ ] `train.global_batch_size/micro_batch_size`: Keep micro_batch_size=1
  - [ ] `model.tensor_model_parallel_size` / `model.context_parallel_size`: Based on GPUs
  - [ ] `checkpoint.save` and `checkpoint.load`: Checkpoint directory
- [ ] Run command with `--training-mode pretrain` and `--config-file`
- [ ] CLI override example with all parameters:
  - [ ] `dataset.path`
  - [ ] `train.global_batch_size`
  - [ ] `train.micro_batch_size`
  - [ ] `model.tensor_model_parallel_size`
  - [ ] `model.context_parallel_size`
  - [ ] `checkpoint.save`
  - [ ] `checkpoint.load`
- [ ] Link to Megatron-Bridge docs for argument details

#### Mock Dataset
- [ ] Use `--mock` flag for debugging or performance measurement
- [ ] Command example with `--mock` flag
- [ ] Note: Can adjust mock shapes (`F_latents`, `H_latents`, `W_latents`) and packing behavior (`number_packed_samples`) in `WanMockDataModuleConfig`
- [ ] Reference: See `dfm/src/megatron/recipes/wan/wan.py`

### Inference

#### Inference Script
- [ ] Script: `examples/megatron/recipes/wan/inference_wan.py`
- [ ] `--checkpoint_step`: Use specific checkpoint for inference
- [ ] `--sizes`: Specify video shape (height, width)
- [ ] `--frame_nums`: Specify number of frames
- [ ] `--sample_steps`: Number of noise diffusion steps (default: 50)
- [ ] Command example with all parameters:
  - [ ] `--task t2v-1.3B`
  - [ ] `--frame_nums 81`
  - [ ] `--sizes 480*832`
  - [ ] `--checkpoint_dir`
  - [ ] `--checkpoint_step 10000`
  - [ ] `--prompts` (example prompt)
  - [ ] `--sample_steps 50`
- [ ] **Note**: Current inference path is single-GPU. Parallel inference not yet supported.

### Parallelism Support Table
- [ ] 1.3B model: Data Parallel (✅), Tensor Parallel (✅), Sequence Parallel (✅), Context Parallel (✅), FSDP (Coming Soon)
- [ ] 14B model: Data Parallel (✅), Tensor Parallel (✅), Sequence Parallel (✅), Context Parallel (✅), FSDP (Coming Soon)

### References
- [ ] WAN Team citation: (2025). Wan: Open and advanced large-scale video generative models (Wan 2.1). GitHub. https://github.com/Wan-Video/Wan2.1/

---

## Verification Summary

**Total Information Items**: ~200+ discrete pieces

**Checklist Status**:
- [ ] All items from `performance-summary.md` captured
- [ ] All items from `mcore_automodel_comparision_wan21.md` captured
- [ ] All items from `automodel_training_doc.md` captured
- [ ] All items from `megatron/models/dit/README.md` captured
- [ ] All items from `megatron/recipes/wan/wan2.1.md` captured

**Integration Verification**:
- [ ] Each item checked off as integrated
- [ ] Location documented (which file/section)
- [ ] Progressive disclosure applied (Layer 1/2/3/4)
- [ ] Links and references verified
- [ ] Images copied and paths updated

---

## Notes

- **Information can be integrated anywhere logical** - doesn't need to match old file structure
- **Progressive disclosure**: Layer 3/4 items can be in dropdowns/tabs/separate pages
- **Cross-references**: Related information can be linked rather than duplicated
- **Verification**: Check off items as you integrate them, note location

