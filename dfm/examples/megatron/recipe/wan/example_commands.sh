#Let's make a md file instead

### set path to Megatron-Bridge
DFM_PATH=/path/to/dfm
MBRIDGE_PATH=/path/to/megatron-bridge
export PYTHONPATH="${DFM_PATH}/.:${MBRIDGE_PATH}/src/.:/opt/NeMo-Framework-Launcher/launcher_scripts"


### install dependencies
pip install --upgrade git+https://github.com/NVIDIA/Megatron-LM.git@ce8185cbbe04f38beb74360e878450f2e8525885
python3 -m pip install --upgrade diffusers
pip install easydict
pip install imageio
pip install imageio-ffmpeg


### Convert checkpoint
# See ${MBRIDGE_PATH}/examples/conversion/convert_wan_checkpoints.py for details.


### Finetuning
export HF_TOKEN=...
export WANDB_API_KEY=...
EXP_NAME=...
PRETRAINED_CHECKPOINT=/path/to/pretrained_checkpoint
CHECKPOINT_DIR=/path/to/checkpoint_dir
DATASET_PATH=/path/to/dataset
cd ${MBRIDGE_PATH}
NVTE_FUSED_ATTN=1 torchrun --nproc_per_node=8 examples/recipes/wan/pretrain_wan.py \
  model.tensor_model_parallel_size=1 \
  model.pipeline_model_parallel_size=1 \
  model.context_parallel_size=4 \
  model.sequence_parallel=false \
  model.qkv_format=thd \
  dataset.path=${DATASET_PATH} \
  checkpoint.save=${CHECKPOINT_DIR} \
  checkpoint.load=${PRETRAINED_CHECKPOINT} \
  checkpoint.load_optim=false \
  checkpoint.save_interval=200 \
  optimizer.lr=5e-6 \
  optimizer.min_lr=5e-6 \
  train.eval_iters=0 \
  scheduler.lr_decay_style=constant \
  scheduler.lr_warmup_iters=0 \
  model.seq_length=2048 \
  dataset.seq_length=2048 \
  train.global_batch_size=1 \
  train.micro_batch_size=1 \
  dataset.global_batch_size=1 \
  dataset.micro_batch_size=1 \
  logger.log_interval=1 \
  logger.wandb_project="wan" \
  logger.wandb_exp_name=${EXP_NAME} \
  logger.wandb_save_dir=${CHECKPOINT_DIR}


### Inferencing
# Download T5 weights and VAE weights from "https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B/tree/main"
#   T5: models_t5_umt5-xxl-enc-bf16.pth, google
#   VAE: Wan2.1_VAE.pth
export HF_TOKEN=...
CHECKPOINT_DIR=/path/to/checkpoint_dir
T5_DIR=/path/to/t5_weights
VAE_DIR=/path/to/vae_weights
cd ${MBRIDGE_PATH}
NVTE_FUSED_ATTN=1 torchrun --nproc_per_node=1 examples/recipes/wan/inference_wan.py \
  --task t2v-1.3B \
  --sizes 832*480 \
  --checkpoint_dir ${CHECKPOINT_DIR} \
  --checkpoint_step 1000 \
  --t5_checkpoint_dir ${T5_DIR} \
  --vae_checkpoint_dir ${VAE_DIR} \
  --prompts "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage." \
  --frame_nums 81 \
  --tensor_parallel_size 1 \
  --context_parallel_size 1 \
  --pipeline_parallel_size 1 \
  --sequence_parallel False \
  --base_seed 42 \
  --sample_steps 50