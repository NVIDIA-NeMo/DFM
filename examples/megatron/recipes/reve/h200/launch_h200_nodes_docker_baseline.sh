#!/bin/bash

# --- 1. Fix Container Environment ---
export PYTHONIOENCODING=utf-8
export LANG=C.UTF-8
export LC_ALL=C.UTF-8

# --- 2. Paths and Python Setup ---
DFM_PATH=/home/scratch.svc_compute_arch/huvu/codes/dfm_customers/DFM_reve_optim
MBRIDGE_PATH=/home/scratch.svc_compute_arch/huvu/codes/dfm_customers/Megatron-Bridge_latest # the same commit as the one in EOS
MLM_PATH=/home/scratch.svc_compute_arch/huvu/codes/dfm_customers/megatron-lm_latest # the same commit as the one in EOS
export PYTHONPATH="${DFM_PATH}/.:${MBRIDGE_PATH}/src/.:${MLM_PATH}/.:/opt/NeMo-Framework-Launcher/launcher_scripts"

# GB200 High-Speed Fabric Optimizations
export NVTE_FUSED_ATTN=1
export NVTE_FLASH_ATTN=0
export NCCL_MNNVL_ENABLE=1 

# Install dependencies (Quietly to avoid encoding issues)
pip install beartype jaxtyping --quiet --root-user-action=ignore

# --- 3. Mock data params (override with env vars; defaults shown) ---
MBS=${MBS:-8}
NUM_IMG_TOKENS=${NUM_IMG_TOKENS:-256}
NUM_TXT_TOKENS=${NUM_TXT_TOKENS:-128}
GRAD_ACCUMULATION=${GRAD_ACCUMULATION:-1}
# => total_batch_size = MBS * DATA_PARALLEL_SIZE * GRAD_ACCUMULATION

# --- 4. Execution ---
cd $DFM_PATH
torchrun --nproc_per_node=8 dfm/src/megatron/model/reve/baseline_reve/mock_train_reve.py \
  --config full \
  --micro_batch_size ${MBS} \
  --num_img_tokens ${NUM_IMG_TOKENS} \
  --num_txt_tokens ${NUM_TXT_TOKENS} \
  --grad_accumulation ${GRAD_ACCUMULATION}