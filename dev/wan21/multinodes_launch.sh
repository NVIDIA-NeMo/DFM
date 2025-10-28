#!/bin/bash
# wan21_launch.sh - Automatic multi-node launcher
# Set NNODES before running, or it defaults to 2

# ============================================================================
# CONFIGURATION
# ============================================================================
# 🎯 CHANGE THIS VALUE TO SCALE TO ANY NUMBER OF NODES!
NNODES=${NNODES:-2}  # Default to 2 nodes if not set

# You can also set this via environment variable:
# export NNODES=3
# ./wan21_launch.sh
# NNODES=5 ./multinodes_launch.sh

echo "=========================================="
echo "Multi-Node Training Configuration"
echo "=========================================="
echo "Number of nodes: $NNODES"
echo "GPUs per node: 8"
echo "Total GPUs: $((NNODES * 8))"
echo "=========================================="
echo ""

# ============================================================================
# LAUNCH LOOP
# ============================================================================
for i in {1..1}; do
  echo "Iteration $i: Allocating resources..."
  
  salloc -t 04:00:00 \
    -N $NNODES \
    --gres=gpu:8 \
    --job-name=wan21-t2v-${NNODES}n-pretrain \
    --account=coreai_dlalgo_llm \
    --partition=batch \
    bash -c "export NNODES=$NNODES; /lustre/fsw/portfolios/coreai/users/linnanw/wan2.1/mutinodes_pretrain.sh"
  
  EXIT_CODE=$?
  
  if [ $EXIT_CODE -eq 0 ]; then
    echo "Iteration $i completed successfully!"
  else
    echo "Iteration $i failed with exit code $EXIT_CODE"
  fi
  
  echo ""
done