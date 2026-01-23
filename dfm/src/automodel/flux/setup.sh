#!/bin/bash
# FLUX Fine-tuning Environment Setup

set -e

echo "🚀 FLUX Fine-tuning Setup"
echo ""

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python version: $python_version"

# Check CUDA
if command -v nvidia-smi &> /dev/null; then
    echo "✓ CUDA available"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1
else
    echo "⚠️  CUDA not detected"
fi
echo ""

# Install PyTorch (CUDA 11.8)
echo "📦 Installing PyTorch (CUDA 11.8)..."
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu118
echo ""

# Install other dependencies
echo "📦 Installing other dependencies..."
pip install diffusers==0.35.2 transformers==4.57.1 accelerate==1.11.0 peft==0.18.0
pip install pillow opencv-python huggingface-hub safetensors
pip install numpy tqdm pyyaml omegaconf
pip install tensorboard  # Optional
echo ""

# Verify
echo "🔍 Verifying installation..."
python3 -c "
import torch
import diffusers
import transformers
print('✅ PyTorch:', torch.__version__)
print('✅ CUDA:', torch.version.cuda if torch.cuda.is_available() else 'Not available')
print('✅ GPUs:', torch.cuda.device_count() if torch.cuda.is_available() else 0)
print('✅ Diffusers:', diffusers.__version__)
print('✅ Transformers:', transformers.__version__)
"

echo ""
echo "✅ Setup complete!"
echo ""
echo "Next steps:"
echo "  1. Download FLUX model: huggingface-cli download black-forest-labs/FLUX.1-dev"
echo "  2. Prepare data: python data/preprocess_flux_images.py --help"
echo "  3. Start training: bash train.sh"
