# allocate node
salloc -p batch --time 4:00:00 -N 1 -A coreai_dlalgo_llm

# srun setup
CONT="nvcr.io/nvidian/nemo:25.11.rc4"
export M_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)

# baseline Reve
srun --jobid <job_id> -N 1 --ntasks-per-node=1 \
  --container-image="${CONT}" \
  --container-mounts=/lustre:/lustre \
  --export=ALL,MY_MASTER_ADDR="$M_ADDR" \
  bash -c '
    MBS=16 NUM_IMG_TOKENS=256 NUM_TXT_TOKENS=128 GRAD_ACCUMULATION=1 \
      bash /lustre/fsw/coreai_dlalgo_genai/huvu/codes/dfm_customers/DFM_reve_optim/examples/megatron/recipes/reve/h100/launch_h100_nodes_baseline.sh
    MBS=8 NUM_IMG_TOKENS=256 NUM_TXT_TOKENS=128 GRAD_ACCUMULATION=2 \
      bash /lustre/fsw/coreai_dlalgo_genai/huvu/codes/dfm_customers/DFM_reve_optim/examples/megatron/recipes/reve/h100/launch_h100_nodes_baseline.sh
  '

# DFM Reve
srun --jobid <job_id> -N 1 --ntasks-per-node=1 \
  --container-image="${CONT}" \
  --container-mounts=/lustre:/lustre \
  --export=ALL,MY_MASTER_ADDR="$M_ADDR" \
  bash -c '
    NUMBER_PACKED_SAMPLES=16 GBS=8 CONTEXT_SEQ_LEN=128 H_LATENTS=16 W_LATENTS=16 \
      bash /lustre/fsw/coreai_dlalgo_genai/huvu/codes/dfm_customers/DFM_reve_optim/examples/megatron/recipes/reve/h100/launch_h100_nodes.sh
  '
