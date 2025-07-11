# Sparse Attention

## 1. Initialize Repositories

### 1.1. Clone Repositories

```bash
git clone https://gitlab-master.nvidia.com/sgovande/sparse-attention && cd sparse-attention
git clone https://github.com/nvidia-cosmos/cosmos-predict1
```

### 1.2. Initialize Cosmos Dependencies

Follow the instructions in [cosmos-predict1/INSTALL.md](https://github.com/nvidia-cosmos/cosmos-predict1/blob/main/INSTALL.md) to set your environment up. You will need to install dependencies for both the **Inference** and the **Post-Training** sections.


### 1.3. Install Sparse Attention

```bash
pip install git+https://github.com/sandyresearch/chipmunk@master --no-build-isolation
pip install flash-attn==2.4.2 --no-build-isolation
```

### 1.4. Update Inference Script
Patch the inference script to activate sparse attention.
```bash
mv accel-config.yml cosmos-predict1/accel-config.yml
mv sparse_attn.py cosmos-predict1/sparse_attn.py
sed -i '1i from sparse_attn import setup_sparse_attn' cosmos-predict1/cosmos_predict1/diffusion/inference/text2world.py
sed -i '2i setup_sparse_attn("accel-config.yml")' cosmos-predict1/cosmos_predict1/diffusion/inference/text2world.py
```

## 2. Modify the Sparsity Configuration

The sparsity configuration file is located in [`accel-config.yml`](../../../accel-config.yml). There is a certain set of parameters controlling the sparsity level to modulate the speed-quality tradeoff, and there are other parameters for the overall structure of the model.

### 2.1. Configure the Speed-Quality Tradeoff

On 75% sparsity (the recommended default), you can expect to see a generation time of 201 seconds on a H100 GPU. This is over 2.0x down from the original generation time of 410 seconds. There are two key hyperparameters to control the level of sparsity:

- **Main Sparsity Parameter**: `attn.top_keys`: This affects how many keys/values are selected for every query,
- **Other Sparsity Parameters**: `attn.full_step_every`: This affects how often we interleave a dense step with the sparse steps.

### 2.2. Configure the Model Shapes

Ensure that the `model_config` key in the configuration file aligns with the characteristics of the model, including the model's latent vector shape **(W, H, D)**, the number of heads, and the number of layers.

## 3. Generate a video.

Run the `text2world.py` script to generate a video.

```bash
export NVTE_FUSED_ATTN=0
export NVTE_UNFUSED_ATTN=0
export NVTE_FLASH_ATTN=1
```


```bash
export PROMPT="Hello World"
cd cosmos-predict1
CUDA_HOME=$CONDA_PREFIX PYTHONPATH=$(pwd) python cosmos_predict1/diffusion/inference/text2world.py \
    --checkpoint_dir checkpoints \
    --diffusion_transformer_dir Cosmos-Predict1-7B-Text2World \
    --prompt "${PROMPT}" \
    --video_save_name diffusion-text2world-7b \
    --disable_prompt_upsampler \
    --disable_guardrail
```