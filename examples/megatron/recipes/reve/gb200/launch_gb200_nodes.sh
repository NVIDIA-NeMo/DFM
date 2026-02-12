#!/bin/bash

# --- 1. Fix Container Environment ---
export PYTHONIOENCODING=utf-8
export LANG=C.UTF-8
export LC_ALL=C.UTF-8

# --- 2. Paths and Python Setup ---
DFM_PATH=/lustre/fsw/portfolios/coreai/users/huvu/codes/dfm_customers/DFM_reve_optim
MBRIDGE_PATH=/lustre/fsw/portfolios/coreai/users/huvu/codes/dfm_customers/Megatron-Bridge_latest
MLM_PATH=/lustre/fsw/portfolios/coreai/users/huvu/codes/dfm_customers/megatron-lm_latest
export PYTHONPATH="${DFM_PATH}:${MBRIDGE_PATH}/src:${MLM_PATH}:/opt/NeMo-Framework-Launcher/launcher_scripts"

# GB200 High-Speed Fabric Optimizations
export NVTE_FUSED_ATTN=1
export NVTE_FLASH_ATTN=0
export NCCL_MNNVL_ENABLE=1 

# Install dependencies (Quietly to avoid encoding issues)
pip install beartype jaxtyping --quiet --root-user-action=ignore

# --- 3. Dataset params (override with env vars; defaults shown) ---
NUMBER_PACKED_SAMPLES=${NUMBER_PACKED_SAMPLES:-4}
GBS=${GBS:-8}
CONTEXT_SEQ_LEN=${CONTEXT_SEQ_LEN:-256}
H_LATENTS=${H_LATENTS:-64}
W_LATENTS=${W_LATENTS:-64}
RECOMPUTE_NUM_LAYERS=${RECOMPUTE_NUM_LAYERS:-0}

# --- 4. Distributed Setup (Injected from Host) ---
# We use MY_MASTER_ADDR which we will pass in the srun command
export NODE_RANK=$SLURM_PROCID
echo "INIT: Node $NODE_RANK reporting. Master is $MY_MASTER_ADDR"

# --- 5. Execution ---
cd $DFM_PATH
export HF_TOKEN=<hf_token>
export WANDB_API_KEY=<wandb_api_key>
unset CUDA_DEVICE_MAX_CONNECTIONS

EXP_NAME=reve_debug_testmockdatamodule
CHECKPOINT_DIR=/lustre/fsw/portfolios/coreai/users/huvu/data/dfm_customers/results/reve_finetune/${EXP_NAME}


torchrun \
  --nproc_per_node=4 \
  --nnodes=2 \
  --node_rank=$NODE_RANK \
  --master_addr=$MY_MASTER_ADDR \
  --master_port=6000 \
  examples/megatron/recipes/reve/pretrain_reve.py \
  --mock \
  --model-size full_dimhead128 \
  model.tensor_model_parallel_size=1 \
  model.pipeline_model_parallel_size=1 \
  model.context_parallel_size=1 \
  model.sequence_parallel=false \
  model.recompute_granularity=full \
  model.recompute_method=block \
  model.recompute_num_layers=${RECOMPUTE_NUM_LAYERS} \
  model.qkv_format=thd \
  dataset.path="" \
  dataset.number_packed_samples=${NUMBER_PACKED_SAMPLES} \
  dataset.context_seq_len=${CONTEXT_SEQ_LEN} \
  dataset.H_latents=${H_LATENTS} \
  dataset.W_latents=${W_LATENTS} \
  checkpoint.save="" \
  checkpoint.load="" \
  checkpoint.load_optim=false \
  checkpoint.save_interval=20000 \
  optimizer.lr=5e-6 \
  optimizer.min_lr=5e-6 \
  train.eval_iters=0 \
  scheduler.lr_decay_style=constant \
  scheduler.lr_warmup_iters=0 \
  model.seq_length=2048 \
  dataset.seq_length=2048 \
  train.global_batch_size=${GBS} \
  train.micro_batch_size=1 \
  dataset.global_batch_size=${GBS} \
  dataset.micro_batch_size=1 \
  logger.log_interval=10 \
  logger.wandb_project="reve" \
  logger.wandb_exp_name=${EXP_NAME} \
  logger.wandb_save_dir=${CHECKPOINT_DIR} \
  logger.log_timers_to_tensorboard=true \
  logger.timing_log_level=2
