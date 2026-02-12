# allocate node
salloc -p h200@cr+mp/viking@cr+mp/8gpu-224cpu-2048gb --time 4:00:00 -N 1 -A nemo-mcore --gres=gpu:8

# srun setup
CONT="nvcr.io/nvidian/nemo:25.11.rc4"
export M_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)


# baseline Reve
srun --jobid <job_id> -N 1 --ntasks-per-node=1 \
  --container-image="${CONT}" \
  --container-mounts=/lustre:/lustre \
  --export=ALL,MY_MASTER_ADDR="$M_ADDR" \
  bash -c '
    cd home/huvu/codes/dfm_customers/DFM_reve_optim

    MBS=16 NUM_IMG_TOKENS=256 NUM_TXT_TOKENS=128 GRAD_ACCUMULATION=1 \
      bash ./examples/megatron/recipes/reve/h200/launch_h200_nodes_baseline.sh
    MBS=8 NUM_IMG_TOKENS=256 NUM_TXT_TOKENS=128 GRAD_ACCUMULATION=2 \
      bash ./examples/megatron/recipes/reve/h200/launch_h200_nodes_baseline.sh
  '


# DFM Reve
srun --jobid <job_id> -N 1 --ntasks-per-node=1 \
  --container-image="${CONT}" \
  --container-mounts=/lustre:/lustre \
  --export=ALL,MY_MASTER_ADDR="$M_ADDR" \
  bash  -c '
    cd home/huvu/codes/dfm_customers/DFM_reve_optim

    NUMBER_PACKED_SAMPLES=16 GBS=8 CONTEXT_SEQ_LEN=128 H_LATENTS=16 W_LATENTS=16 \
      bash ./examples/megatron/recipes/reve/h200/launch_h200_nodes.sh
  '
