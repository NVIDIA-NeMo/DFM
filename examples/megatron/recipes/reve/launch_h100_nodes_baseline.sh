#!/bin/bash

# --- 1. Fix Container Environment ---
export PYTHONIOENCODING=utf-8
export LANG=C.UTF-8
export LC_ALL=C.UTF-8

# --- 2. Paths and Python Setup ---
DFM_PATH=/lustre/fsw/coreai_dlalgo_genai/huvu/codes/dfm_customers/DFM_reve_optim
MBRIDGE_PATH=/lustre/fsw/coreai_dlalgo_genai/huvu/codes/nemo_vfm/Megatron-Bridge_latest
MLM_PATH=/lustre/fsw/coreai_dlalgo_genai/huvu/codes/nemo_vfm/megatron-lm_latest
export PYTHONPATH="${DFM_PATH}/.:${MBRIDGE_PATH}/src/.:${MLM_PATH}/.:/opt/NeMo-Framework-Launcher/launcher_scripts"

# GB200 High-Speed Fabric Optimizations
export NVTE_FUSED_ATTN=1
export NVTE_FLASH_ATTN=0
export NCCL_MNNVL_ENABLE=1 

# Install dependencies (Quietly to avoid encoding issues)
pip install beartype jaxtyping --quiet --root-user-action=ignore

# --- 3. Execution ---
cd $DFM_PATH
torchrun --nproc_per_node=8 dfm/src/megatron/model/reve/baseline_reve/mock_train_reve.py \
  --config full