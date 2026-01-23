# FLUX.1 Fine-tuning with FSDP2

Train FLUX.1 image generation models from scratch or fine-tune with PyTorch FSDP2.

---

## 📋 Pipeline Overview

```
Model Download → Data Preprocessing → Training → Checkpoint Merging → Inference
```

---

## ⚡ Quick Start

### 0️⃣ Environment Setup

```bash
# Install dependencies
bash setup.sh

# Or manually:
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

**Requirements:**
- Python 3.10+
- CUDA 11.8+
- 8× A100 GPUs (80GB VRAM)

---

### 1️⃣ Download FLUX Model

```bash
# Option 1: Hugging Face CLI (recommended)
huggingface-cli download black-forest-labs/FLUX.1-dev --local-dir ./models/FLUX.1-dev

# Option 2: Python
python -c "
from huggingface_hub import snapshot_download
snapshot_download('black-forest-labs/FLUX.1-dev', local_dir='./models/FLUX.1-dev')
"
```

**Model size:** ~24GB

---

### 2️⃣ Data Preprocessing

**Step 1: Parse COCO captions**

Edit `data/run_parse_coco.sh` to configure:
```bash
--annotation_file /path/to/mscoco/annotations/captions_train2017.json
--image_folder /path/to/mscoco/train2017
--output_file mscoco_captions.json
--max_samples 1000
```

Run:
```bash
bash data/run_parse_coco.sh
```

Output format (`mscoco_captions.json`):
```json
[
  {"file_name": "000000108838.jpg", "caption": "A dog playing in the park"},
  {"file_name": "000000142261.jpg", "caption": "A cat sitting on a chair"}
]
```

---

**Step 2: Preprocess images**

Edit `data/run_process_coco.sh` to configure:
```bash
--input_folder /path/to/mscoco/train2017
--output_folder ./data/processed_meta
--caption_file mscoco_captions.json
--height 256 --width 256
--model_id ./models/FLUX.1-dev
--max_samples 3
```

Run:
```bash
bash data/run_process_coco.sh
```

---

**Step 3: Validate preprocessing**

```bash
python data/validate.py
```

---

### 3️⃣ Training

**Edit `train.sh` to configure:**
```bash
# Model & Data
MODEL_ID="./models/FLUX.1-dev"           # FLUX model path
META_FOLDER="./data/processed_meta"      # Preprocessed data

# Training
NUM_GPUS=8                               # Number of GPUs
BATCH_SIZE_PER_GPU=1                     # Batch size per GPU
NUM_EPOCHS=10000                         # Training epochs
LEARNING_RATE=1e-5                       # Learning rate
SAVE_EVERY=100                           # Save checkpoint every N epochs
```

**Start training:**
```bash
bash train.sh
```

**Training outputs:**
- Checkpoints: `outputs/flux_pretraining_ckpt/ckpt_epoch_X_sharded/`
- Logs: `outputs/flux_pretraining_ckpt/train.log`

---

### 4️⃣ Checkpoint Merging

Training saves **sharded checkpoints** (8 files, one per GPU). Merge them:

**Edit `convert_ckpt.sh`:**
```bash
MODEL_ID="./models/FLUX.1-dev"
CKPT_DIR="outputs/flux_pretraining_ckpt/ckpt_epoch_100_sharded"
OUTPUT_PATH="outputs/flux_pretraining_ckpt/flux_epoch100_full.pt"
```

**Run merging (requires 8 GPUs):**
```bash
bash convert_ckpt.sh
```

**Output:** `flux_epoch100_full.pt` (~24GB)

---

### 5️⃣ Inference

**Edit `run_inference.sh`:**
```bash
MODEL_ID="./models/FLUX.1-dev"
FULL_CKPT="outputs/flux_pretraining_ckpt/flux_epoch100_full.pt"
OUTPUT_DIR="inference_outputs"
```

**Generate images:**
```bash
bash run_inference.sh
```

**Generated images:** `inference_outputs/epoch100_step1000_prompt0.png`

**Test prompts (in `scripts/eval_flux_fsdp.py`):**
- "People outside a building on a street with a gay crossing..."
- "A young woman stands by the post with a pink umbrella."
- "A row of motorcyclists lined up while passer byers watch."
- ...

---

## 📁 Project Structure

```
flux-finetuning/
├── data/                          # Data preparation
│   ├── preprocess_flux_images.py  # Main preprocessing script
│   ├── parse_coco_captions.py     # Parse COCO-style captions
│   ├── validate.py                # Validate preprocessed data
│   └── mscoco_3case/              # Example dataset
├── scripts/                       # Core training code
│   ├── main_flux.py               # Main trainer (WandB + resume + time limit)
│   ├── training_step_flux.py      # Flow Matching training logic
│   ├── fsdp2_utils_flux.py        # FSDP2 utilities (save/load checkpoints)
│   ├── dataloader_flux.py         # Data loading
│   ├── merge.py                   # Checkpoint merging
│   └── eval_flux_fsdp.py          # Inference script
├── train.sh                       # Training launcher
├── convert_ckpt.sh                # Checkpoint merging launcher
├── run_inference.sh               # Inference launcher
├── setup.sh                       # Environment setup
├── requirements.txt               # Dependencies
└── README.md                      # This file
```

---

## 🔧 Configuration

### Training Hyperparameters

Edit `train.sh`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `BATCH_SIZE_PER_GPU` | 1 | Batch size per GPU |
| `NUM_EPOCHS` | 10000 | Training epochs |
| `LEARNING_RATE` | 1e-5 | AdamW learning rate |
| `SAVE_EVERY` | 100 | Save checkpoint every N epochs |
| `FLOW_SHIFT` | 3.0 | Flow Matching shift (**DO NOT CHANGE**) |
| `SIGMA_MIN` | 0.0 | Min sigma (pretrain: 0.0, finetune: 0.02) |
| `SIGMA_MAX` | 1.0 | Max sigma (pretrain: 1.0, finetune: 0.55) |

### Image Resolution

Edit `data/preprocess_flux_images.py`:

```bash
--height 256 --width 256   # 256×256 (fast, low quality)
--height 512 --width 512   # 512×512 (balanced)
--height 1024 --width 1024 # 1024×1024 (slow, high quality)
```

---

## 🙏 Acknowledgments

- [FLUX.1](https://github.com/black-forest-labs/flux) by Black Forest Labs
- [Diffusers](https://github.com/huggingface/diffusers) by Hugging Face
- Flow Matching inspired by [Pika Labs](https://pika.art/)

---

## 📄 License

Apache License 2.0

---

## 💬 Support

For issues or questions:
- **GitHub Issues:** [Your repo issues page]
- **Discussions:** [Your repo discussions page]