#!/bin/bash

# Slurm parameters (parsed by run_example.sh)
NUM_NODES=1
TIME=00:30:00


EXP_NAME=sbatch_wan_1.3B_pretrain_text2video_cicd_3000vids_example
CHECKPOINT_DIR=/lustre/fsw/coreai_dlalgo_genai/huvu/data/nemo_vfm/results/wan_finetune/${EXP_NAME}
PROJECT=wan
MBS=1
GBS=2
LR=5e-5
WARMUP_ITERS=1000
CHECKPOINT_DIR=${CHECKPOINT_BASE_DIR}/${EXP_NAME}
# set this PRETRAIN_CHECKPOINT_DIR to CHECKPOINT_DIR to train from scratch
PRETRAIN_CHECKPOINT_DIR=${CHECKPOINT_DIR}

NVTE_FUSED_ATTN=1 MASTER_ADDR=${MASTER_ADDR} MASTER_PORT=${MASTER_PORT} torchrun \
  --nnodes=${SLURM_JOB_NUM_NODES} \
  --nproc_per_node=8 \
  --rdzv-backend=c10d \
  --rdzv-endpoint=${MASTER_ADDR}:${RDZV_PORT} \
  --rdzv-id=${SLURM_JOB_ID} \
  --rdzv-conf=timeout=6000 \
  examples/megatron/recipes/wan/pretrain_wan.py \
  --training-mode pretrain \
  model.tensor_model_parallel_size=1 \
  model.pipeline_model_parallel_size=1 \
  model.context_parallel_size=4 \
  model.sequence_parallel=false \
  model.qkv_format=thd \
  dataset.path="${DATASET_BASE_DIR}/OpenVid-1M/OpenVidHD/OpenVidHD_part_1_3000vids_text2image_wds" \
  dataset.packing_buffer_size=50 \
  dataset.num_workers=10 \
  checkpoint.save=${CHECKPOINT_DIR} \
  checkpoint.load=${PRETRAIN_CHECKPOINT_DIR} \
  checkpoint.load_optim=true \
  checkpoint.save_interval=100 \
  optimizer.lr=${LR} \
  optimizer.min_lr=${LR} \
  optimizer.weight_decay=0.1 \
  optimizer.adam_beta2=0.95 \
  optimizer.clip_grad=2.0 \
  train.eval_iters=0 \
  train.train_iters=200 \
  scheduler.lr_decay_style=constant \
  scheduler.lr_warmup_iters=${WARMUP_ITERS} \
  model.seq_length=80000 \
  dataset.seq_length=80000 \
  train.global_batch_size=${GBS} \
  train.micro_batch_size=${MBS} \
  dataset.global_batch_size=${GBS} \
  dataset.micro_batch_size=${MBS} \
  logger.log_interval=1
