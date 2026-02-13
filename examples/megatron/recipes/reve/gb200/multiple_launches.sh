# allocate node
salloc -p batch --time 4:00:00 -N 2 -A coreai_dlalgo_llm --gres=gpu:4

# srun setup
CONT="nvcr.io/nvidian/nemo:25.11.rc4"
export M_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)


# DFM Reve
srun --jobid 1722911 -N 2 --ntasks-per-node=1 \
  --container-image="${CONT}" \
  --container-mounts=/lustre:/lustre \
  --export=ALL,MY_MASTER_ADDR="$M_ADDR" \
  bash -c '
    cd /lustre/fsw/portfolios/coreai/users/huvu/codes/dfm_customers/DFM_reve_optim

    NUMBER_PACKED_SAMPLES=4 GBS=8 CONTEXT_SEQ_LEN=128 H_LATENTS=16 W_LATENTS=16 \
      bash ./examples/megatron/recipes/reve/gb200/launch_gb200_nodes.sh
    NUMBER_PACKED_SAMPLES=8 GBS=8 CONTEXT_SEQ_LEN=128 H_LATENTS=16 W_LATENTS=16 \
      bash ./examples/megatron/recipes/reve/gb200/launch_gb200_nodes.sh
    NUMBER_PACKED_SAMPLES=16 GBS=8 CONTEXT_SEQ_LEN=128 H_LATENTS=16 W_LATENTS=16 \
      bash ./examples/megatron/recipes/reve/gb200/launch_gb200_nodes.sh
    NUMBER_PACKED_SAMPLES=32 GBS=8 CONTEXT_SEQ_LEN=128 H_LATENTS=16 W_LATENTS=16 \
      bash ./examples/megatron/recipes/reve/gb200/launch_gb200_nodes.sh

    NUMBER_PACKED_SAMPLES=2 GBS=8 CONTEXT_SEQ_LEN=256 H_LATENTS=32 W_LATENTS=32 \
      bash ./examples/megatron/recipes/reve/gb200/launch_gb200_nodes.sh
    NUMBER_PACKED_SAMPLES=4 GBS=8 CONTEXT_SEQ_LEN=256 H_LATENTS=32 W_LATENTS=32 \
      bash ./examples/megatron/recipes/reve/gb200/launch_gb200_nodes.sh
    NUMBER_PACKED_SAMPLES=8 GBS=8 CONTEXT_SEQ_LEN=256 H_LATENTS=32 W_LATENTS=32 \
      bash ./examples/megatron/recipes/reve/gb200/launch_gb200_nodes.sh
    NUMBER_PACKED_SAMPLES=16 GBS=8 CONTEXT_SEQ_LEN=256 H_LATENTS=32 W_LATENTS=32 RECOMPUTE_NUM_LAYERS=13 \
      bash ./examples/megatron/recipes/reve/gb200/launch_gb200_nodes.sh

    NUMBER_PACKED_SAMPLES=1 GBS=8 CONTEXT_SEQ_LEN=256 H_LATENTS=64 W_LATENTS=64 \
      bash ./examples/megatron/recipes/reve/gb200/launch_gb200_nodes.sh
    NUMBER_PACKED_SAMPLES=2 GBS=8 CONTEXT_SEQ_LEN=256 H_LATENTS=64 W_LATENTS=64 \
      bash ./examples/megatron/recipes/reve/gb200/launch_gb200_nodes.sh
    NUMBER_PACKED_SAMPLES=4 GBS=8 CONTEXT_SEQ_LEN=256 H_LATENTS=64 W_LATENTS=64 RECOMPUTE_NUM_LAYERS=13 \
      bash ./examples/megatron/recipes/reve/gb200/launch_gb200_nodes.sh
    NUMBER_PACKED_SAMPLES=8 GBS=8 CONTEXT_SEQ_LEN=256 H_LATENTS=64 W_LATENTS=64 RECOMPUTE_NUM_LAYERS=26 \
      bash ./examples/megatron/recipes/reve/gb200/launch_gb200_nodes.sh
  '
