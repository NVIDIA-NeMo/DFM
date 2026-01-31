#!/bin/bash

# Slurm parameters (parsed by run_example.sh)
NUM_NODES=1
TIME=00:30:00

EXP_NAME=automodel_wan_1.3B_pretrain_text2video_nightly
CHECKPOINT_DIR=${CHECKPOINT_BASE_DIR}/${EXP_NAME}

# Multi-node env
export NUM_GPUS=8
export WORLD_SIZE=$(($NUM_GPUS * $SLURM_NNODES))
export CUDA_DEVICE_MAX_CONNECTIONS=1
export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export NCCL_NVLS_ENABLE=0

torchrun --nnodes=${SLURM_NNODES} \
    --nproc-per-node=${NUM_GPUS} \
    --rdzv_backend=c10d \
    --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} \
    examples/automodel/pretrain/pretrain.py \
    -c examples/automodel/pretrain/cicd/wan21_cicd_nightly_video.yaml \
    data.dataloader.meta_folder=${DATASET_BASE_DIR}/Wan21/nightly/video \
    checkpoint.checkpoint_dir=${CHECKPOINT_DIR} \
    checkpoint.restore_from=${CHECKPOINT_DIR}/automodel_wan_1.3B_pretrain_text2image_nightly/LATEST
