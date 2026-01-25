#!/bin/bash

# Generic sbatch script for running example scripts
# Usage: sbatch --account=<account> --nodes=<nodes> --time=<time> examples/run_example.sh --example-script <path> --container <container> --checkpoint-base-dir <dir> --dataset-base-dir <dir> [--partition <partition>] [--mount <mount>]
#
# NUM_NODES and TIME should be parsed from the example script and passed to sbatch via --nodes and --time

#SBATCH --partition=batch
#SBATCH --job-name=dfm-example

# Default values
MOUNT="/lustre:/lustre"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --example-script)
            EXAMPLE_SCRIPT="$2"
            shift 2
            ;;
        --container)
            CONTAINER="$2"
            shift 2
            ;;
        --checkpoint-base-dir)
            CHECKPOINT_BASE_DIR="$2"
            shift 2
            ;;
        --dataset-base-dir)
            DATASET_BASE_DIR="$2"
            shift 2
            ;;
        --mount)
            MOUNT="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: sbatch --account=<account> --nodes=<nodes> --time=<time> $0 --example-script <path> --container <container> --checkpoint-base-dir <dir> --dataset-base-dir <dir> [--mount <mount>]"
            echo ""
            echo "Required sbatch arguments:"
            echo "  --account             Slurm account to use"
            echo "  --nodes               Number of nodes (parse NUM_NODES from example script)"
            echo "  --time                Job timeout (parse TIME from example script)"
            echo ""
            echo "Required script arguments:"
            echo "  --example-script      Path to the torchrun script to execute"
            echo "  --container           Container image to use"
            echo "  --checkpoint-base-dir Base directory for checkpoints"
            echo "  --dataset-base-dir    Base directory for datasets"
            echo ""
            echo "Optional arguments:"
            echo "  --mount               Slurm mount to use (default: /lustre:/lustre)"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Validate required arguments
if [[ -z "$EXAMPLE_SCRIPT" ]]; then
    echo "Error: --example-script is required"
    exit 1
fi

if [[ -z "$CONTAINER" ]]; then
    echo "Error: --container is required"
    exit 1
fi

if [[ -z "$CHECKPOINT_BASE_DIR" ]]; then
    echo "Error: --checkpoint-base-dir is required"
    exit 1
fi

if [[ -z "$DATASET_BASE_DIR" ]]; then
    echo "Error: --dataset-base-dir is required"
    exit 1
fi

# Validate example script exists
if [[ ! -f "$EXAMPLE_SCRIPT" ]]; then
    echo "Error: Example script not found: $EXAMPLE_SCRIPT"
    exit 1
fi

# Get the example script name for checkpoint directory
SCRIPT_NAME=$(basename "$EXAMPLE_SCRIPT" .sh)
CHECKPOINT_DIR=${CHECKPOINT_BASE_DIR}/${SCRIPT_NAME}

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

# Read the torchrun command from the example script
TORCHRUN_CMD=$(cat "${EXAMPLE_SCRIPT}")

cmd="
# Synchronization barrier to ensure all nodes have finished installation before starting torchrun
echo \"Node \${SLURM_NODEID} finished setup. Waiting for others...\"
touch \"${BARRIER_DIR}/node_\${SLURM_NODEID}.ready\"
while [ \$(ls -1 \"${BARRIER_DIR}\"/node_*.ready | wc -l) -lt ${SLURM_JOB_NUM_NODES} ]; do
   sleep 5
done
echo \"All nodes ready. Starting training...\"

# Set environment variables for distributed training
export MASTER_ADDR=${MASTER_ADDR}
export MASTER_PORT=${MASTER_PORT}
export RDZV_PORT=${RDZV_PORT}
export CHECKPOINT_DIR=${CHECKPOINT_DIR}
export CHECKPOINT_BASE_DIR=${CHECKPOINT_BASE_DIR}
export DATASET_BASE_DIR=${DATASET_BASE_DIR}

${TORCHRUN_CMD}
"

export CUDA_DEVICE_MAX_CONNECTIONS=1

echo "Running training script: ${EXAMPLE_SCRIPT}"
echo "CHECKPOINT_DIR: ${CHECKPOINT_DIR}"

srun --mpi=pmix \
    --container-image="${CONTAINER}" --container-mounts="${MOUNT}" \
    --no-container-mount-home \
    --ntasks-per-node=1 \
    -N ${SLURM_JOB_NUM_NODES} \
    bash -c "${cmd}"
