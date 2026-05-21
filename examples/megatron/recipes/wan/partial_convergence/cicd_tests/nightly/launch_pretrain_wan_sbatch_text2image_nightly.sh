#!/bin/bash

# Parameters
#SBATCH --account=coreai_dlalgo_llm
#SBATCH --job-name=coreai_dlalgo_llm-run:dfm
#SBATCH --nodes=1
#SBATCH --partition=batch
#SBATCH --time=04:00:00


EXP_NAME=sbatch_wan_1.3B_pretrain_text2image_cicd_3000vids_nightly_example
CHECKPOINT_DIR=/lustre/fsw/coreai_dlalgo_genai/huvu/data/nemo_vfm/results/wan_finetune/${EXP_NAME}
PROJECT=wan
MBS=1
GBS=8
LR=5e-5
WARMUP_ITERS=1000
# set this PRETRAIN_CHECKPOINT_DIR to CHECKPOINT_DIR to train from scratch
PRETRAIN_CHECKPOINT_DIR=${CHECKPOINT_DIR}

# compute rendezvous/master addresses and ports (avoid port collision)
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
RDZV_PORT=${RDZV_PORT:-29500}
MASTER_PORT=${MASTER_PORT:-29600}

# create checkpoint directory
mkdir -p ${CHECKPOINT_DIR}
# create barrier directory for synchronization
BARRIER_DIR=${CHECKPOINT_DIR}/setup_barrier
rm -rf ${BARRIER_DIR}
mkdir -p ${BARRIER_DIR}

cmd="

# install
DFM_PATH=/lustre/fsw/coreai_dlalgo_genai/huvu/codes/nemo_vfm/DFM_wan_cicd
MBRIDGE_PATH=/lustre/fsw/coreai_dlalgo_genai/huvu/codes/nemo_vfm/Megatron-Bridge_latest
MLM_PATH=/lustre/fsw/coreai_dlalgo_genai/huvu/codes/nemo_vfm/megatron-lm_latest
export PYTHONPATH=\$DFM_PATH/.:\$MBRIDGE_PATH/src/.:\$MLM_PATH/.:/opt/NeMo-Framework-Launcher/launcher_scripts

cd \$DFM_PATH
export HF_TOKEN=...
export WANDB_API_KEY=...

# Synchronization barrier to ensure all nodes have finished installation before starting torchrun
echo \"Node \${SLURM_NODEID} finished setup. Waiting for others...\"
touch \"${BARRIER_DIR}/node_\${SLURM_NODEID}.ready\"
while [ \$(ls -1 \"${BARRIER_DIR}\"/node_*.ready | wc -l) -lt ${SLURM_JOB_NUM_NODES} ]; do
   sleep 5
done
echo \"All nodes ready. Starting training...\"

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
  model.context_parallel_size=1 \
  model.sequence_parallel=false \
  model.qkv_format=thd \
  dataset.path="/lustre/fsw/coreai_dlalgo_genai/huvu/data/nemo_vfm/datasets/cicd_tests/OpenVid-1M/OpenVidHD/OpenVidHD_part_1_3000vids_text2image_wds" \
  dataset.packing_buffer_size=150 \
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
  train.train_iters=100 \
  scheduler.lr_decay_style=constant \
  scheduler.lr_warmup_iters=${WARMUP_ITERS} \
  model.seq_length=12480 \
  dataset.seq_length=12480 \
  train.global_batch_size=${GBS} \
  train.micro_batch_size=${MBS} \
  dataset.global_batch_size=${GBS} \
  dataset.micro_batch_size=${MBS} \
  logger.log_interval=1

"  

CONT="nvcr.io/nvidian/nemo:25.11.rc4"
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