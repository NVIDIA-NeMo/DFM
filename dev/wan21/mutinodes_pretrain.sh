#!/bin/bash
# wan21_run.sh - FINAL FIX with all environment variables

set -euo pipefail

# ============================================================================
# MASTER CONFIGURATION
# ============================================================================
MASTER_PORT=29500
MASTER_ADDR=$(scontrol show hostname $SLURM_NODELIST | head -n 1 | cut -d"." -f1)
echo "$MASTER_ADDR, $MASTER_PORT"

# ============================================================================
# TRAINING CONFIGURATION
# ============================================================================
MODE=pretrain
META_FOLDER=/linnanw/hdvilla_sample/pika/wan21_codes/dmd_1.3B_meta
BATCH_PER_NODE=1
LR=1e-4
EPOCHS=1000000
SAVE_EVERY=50
OUTDIR=./wan_pretrain_1.3B
LOG_EVERY=1

TRAIN_CMD="main_t2v.py \
  --mode ${MODE} \
  --meta_folder ${META_FOLDER} \
  --batch_size_per_node ${BATCH_PER_NODE} \
  --learning_rate ${LR} \
  --num_epochs ${EPOCHS} \
  --save_every ${SAVE_EVERY} \
  --output_dir ${OUTDIR} \
  --log_every ${LOG_EVERY}"

# ============================================================================
# CONTAINER CONFIGURATION
# ============================================================================
WORK_DIR="/linnanw/wan2.1"
MOUNTS="/lustre/fsw/portfolios/coreai/users/linnanw:/linnanw"
CONTAINER="/lustre/fsw/portfolios/coreai/users/linnanw/dockers/pika.sqsh"

# ============================================================================
# NODE CONFIGURATION
# ============================================================================
NNODES=${NNODES:-2}

echo "=========================================="
echo "Multi-Node Training Launch"
echo "=========================================="
echo "Number of nodes: $NNODES"
echo "GPUs per node: 8"
echo "Total GPUs: $((NNODES * 8))"
echo "Master: $MASTER_ADDR:$MASTER_PORT"
echo "=========================================="
echo ""

# ============================================================================
# LAUNCH NODES
# ============================================================================
for NODE_RANK in $(seq 0 $((NNODES - 1))); do
  NODE_INDEX=$((NODE_RANK + 1))
  
  if [ $NODE_RANK -eq $((NNODES - 1)) ]; then
    # Last node in foreground
    echo "Launching node $NODE_RANK (foreground)..."
    srun --signal=SIGUSR1@240 --nodes=1 --ntasks=1 \
      --nodelist=$(scontrol show hostnames $SLURM_JOB_NODELIST | sed -n "${NODE_INDEX}p") \
      --container-mounts=$MOUNTS \
      --container-image=$CONTAINER \
      --container-workdir=$WORK_DIR \
      --export=ALL \
      --container-env=SLURM_NODEID,SLURM_PROCID,SLURM_NODELIST,MASTER_ADDR,MASTER_PORT,LOCAL_WORLD_SIZE \
      bash -c "
        set -x
        echo \"[NODE $NODE_RANK] Container started on \$(hostname)\"
        echo \"[NODE $NODE_RANK] MASTER_ADDR=\$MASTER_ADDR\"
        echo \"[NODE $NODE_RANK] MASTER_PORT=\$MASTER_PORT\"
        echo \"[NODE $NODE_RANK] LOCAL_WORLD_SIZE=\$LOCAL_WORLD_SIZE\"
        
        export HF_HOME=/linnanw/hdvilla_sample/cache
        export LOCAL_WORLD_SIZE=8
        export MASTER_ADDR=$MASTER_ADDR
        export MASTER_PORT=$MASTER_PORT
        
        exec torchrun \
          --nproc_per_node=8 \
          --nnodes=$NNODES \
          --node_rank=$NODE_RANK \
          --master_addr=\$MASTER_ADDR \
          --master_port=\$MASTER_PORT \
          --rdzv_backend=c10d \
          --rdzv_endpoint=\$MASTER_ADDR:\$MASTER_PORT \
          $TRAIN_CMD
      "
  else
    # Other nodes in background
    echo "Launching node $NODE_RANK (background)..."
    srun --signal=SIGUSR1@240 --nodes=1 --ntasks=1 \
      --nodelist=$(scontrol show hostnames $SLURM_JOB_NODELIST | sed -n "${NODE_INDEX}p") \
      --container-mounts=$MOUNTS \
      --container-image=$CONTAINER \
      --container-workdir=$WORK_DIR \
      --export=ALL \
      --container-env=SLURM_NODEID,SLURM_PROCID,SLURM_NODELIST,MASTER_ADDR,MASTER_PORT,LOCAL_WORLD_SIZE \
      bash -c "
        set -x
        echo \"[NODE $NODE_RANK] Container started on \$(hostname)\"
        echo \"[NODE $NODE_RANK] MASTER_ADDR=\$MASTER_ADDR\"
        echo \"[NODE $NODE_RANK] MASTER_PORT=\$MASTER_PORT\"
        echo \"[NODE $NODE_RANK] LOCAL_WORLD_SIZE=\$LOCAL_WORLD_SIZE\"
        
        export HF_HOME=/linnanw/hdvilla_sample/cache
        export LOCAL_WORLD_SIZE=8
        export MASTER_ADDR=$MASTER_ADDR
        export MASTER_PORT=$MASTER_PORT
        
        exec torchrun \
          --nproc_per_node=8 \
          --nnodes=$NNODES \
          --node_rank=$NODE_RANK \
          --master_addr=\$MASTER_ADDR \
          --master_port=\$MASTER_PORT \
          --rdzv_backend=c10d \
          --rdzv_endpoint=\$MASTER_ADDR:\$MASTER_PORT \
          $TRAIN_CMD
      " &
  fi
done

echo ""
echo "All nodes launched!"