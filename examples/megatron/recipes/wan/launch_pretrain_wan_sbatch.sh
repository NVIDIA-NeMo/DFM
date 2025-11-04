#!/bin/bash

# Parameters
#SBATCH --account=coreai_dlalgo_llm
#SBATCH --job-name=coreai_dlalgo_llm-run:dfm
#SBATCH --nodes=1
#SBATCH --partition=batch
#SBATCH --time=04:00:00


EXP_NAME=sbatch_wan_1.3B_square_images_pretrain_mbs64gbs512_1tar
CHECKPOINT_DIR=/lustre/fsw/coreai_dlalgo_genai/huvu/data/nemo_vfm/results/wan_finetune/${EXP_NAME}
PROJECT=wan
MBS=64
GBS=512
LR=1e-4
WARMUP_ITERS=10000
# set this PRETRAIN_CHECKPOINT_DIR to CHECKPOINT_DIR to train from scratch
PRETRAIN_CHECKPOINT_DIR=${CHECKPOINT_DIR}
# PRETRAIN_CHECKPOINT_DIR=/lustre/fsw/coreai_dlalgo_genai/huvu/data/nemo_vfm/wan_checkpoints/megatron_checkpoint_1.3B

# create checkpoint directory
mkdir -p ${CHECKPOINT_DIR}

cmd="

# install
DFM_PATH=/lustre/fsw/coreai_dlalgo_genai/huvu/codes/nemo_vfm/DFM_mcore_wan
MBRIDGE_PATH=/lustre/fsw/coreai_dlalgo_genai/huvu/codes/nemo_vfm/Megatron-Bridge_mcore_wan_official
MLM_PATH=/lustre/fsw/coreai_dlalgo_genai/huvu/codes/nemo_vfm/megatron-lm_latest
export PYTHONPATH="\$DFM_PATH/.:\$MBRIDGE_PATH/src/.:\$MLM_PATH/.:/opt/NeMo-Framework-Launcher/launcher_scripts"


# install dependencies
python3 -m pip install --upgrade diffusers
pip install easydict
pip install imageio
pip install imageio-ffmpeg
[apt update; apt install ffmpeg -y] -> for data preparation


cd \$DFM_PATH
export HF_TOKEN=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
export WANDB_API_KEY=497a93e5ac7cf1e0ec821741ef7bac27b36f2db8
NVTE_FUSED_ATTN=1 torchrun --standalone --nproc_per_node=8 dfm/examples/megatron/recipe/wan/pretrain_wan.py \
  model.tensor_model_parallel_size=1 \
  model.pipeline_model_parallel_size=1 \
  model.context_parallel_size=1 \
  model.sequence_parallel=false \
  model.qkv_format=sbhd \
  dataset.path="/lustre/fsw/coreai_dlalgo_genai/huvu/data/nemo_vfm/datasets/coyo_dataset_wan/processed_coyo_dataset_wan/part_00000_square_wds_1tar" \
  checkpoint.save=${CHECKPOINT_DIR} \
  checkpoint.load=${PRETRAIN_CHECKPOINT_DIR} \
  checkpoint.load_optim=false \
  checkpoint.save_interval=2500 \
  optimizer.lr=${LR} \
  optimizer.min_lr=${LR} \
  train.eval_iters=0 \
  train.train_iters=1000000 \
  scheduler.lr_decay_style=constant \
  scheduler.lr_decay_style=constant \
  scheduler.lr_warmup_iters=${WARMUP_ITERS} \
  model.seq_length=2048 \
  dataset.seq_length=2048 \
  train.global_batch_size=${GBS} \
  train.micro_batch_size=${MBS} \
  dataset.global_batch_size=${GBS} \
  dataset.micro_batch_size=${MBS} \
  logger.log_interval=1 \
  logger.wandb_project=${PROJECT} \
  logger.wandb_exp_name=${EXP_NAME} \
  logger.wandb_save_dir=${CHECKPOINT_DIR}

"  

CONT="nvcr.io/nvidia/nemo:25.09.00"
MOUNT="/lustre/fsw/:/lustre/fsw/"
OUTFILE=$CHECKPOINT_DIR/slurm-%j.out
ERRFILE=$CHECKPOINT_DIR/error-%j.out
export CUDA_DEVICE_MAX_CONNECTIONS=1

echo "Running training script."
srun -o ${OUTFILE} -e ${ERRFILE} --mpi=pmix \
    --container-image="${CONT}" --container-mounts="${MOUNT}" \
    --no-container-mount-home \
    --ntasks-per-node=1 \
    -N ${SLURM_JOB_NUM_NODES}  \
    bash -c "${cmd}"