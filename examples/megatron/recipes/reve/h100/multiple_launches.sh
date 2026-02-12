# allocate node
salloc -p batch --time 4:00:00 -N 1 -A coreai_dlalgo_llm
salloc -p batch --time 4:00:00 -N 1 -A coreai_dlalgo_genai
salloc -p interactive --time 2:00:00 -N 1 -A coreai_dlalgo_llm
salloc -p interactive --time 2:00:00 -N 1 -A coreai_dlalgo_genai

# srun setup
CONT="nvcr.io/nvidian/nemo:25.11.rc4"
export M_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)

# baseline Reve
srun --jobid 4679313 -N 1 --ntasks-per-node=1 \
  --container-image="${CONT}" \
  --container-mounts=/lustre:/lustre \
  --export=ALL,MY_MASTER_ADDR="$M_ADDR" \
  bash -c '
    cd /lustre/fsw/coreai_dlalgo_genai/huvu/codes/dfm_customers/DFM_reve_optim
  
    MBS=4 NUM_IMG_TOKENS=256 NUM_TXT_TOKENS=128 GRAD_ACCUMULATION=1 \
      bash ./examples/megatron/recipes/reve/h100/launch_h100_nodes_baseline.sh
    MBS=8 NUM_IMG_TOKENS=256 NUM_TXT_TOKENS=128 GRAD_ACCUMULATION=1 \
      bash ./examples/megatron/recipes/reve/h100/launch_h100_nodes_baseline.sh
    MBS=16 NUM_IMG_TOKENS=256 NUM_TXT_TOKENS=128 GRAD_ACCUMULATION=1 \
      bash ./examples/megatron/recipes/reve/h100/launch_h100_nodes_baseline.sh
    MBS=32 NUM_IMG_TOKENS=256 NUM_TXT_TOKENS=128 GRAD_ACCUMULATION=1 \
      bash ./examples/megatron/recipes/reve/h100/launch_h100_nodes_baseline.sh

    MBS=2 NUM_IMG_TOKENS=1024 NUM_TXT_TOKENS=256 GRAD_ACCUMULATION=1 \
      bash ./examples/megatron/recipes/reve/h100/launch_h100_nodes_baseline.sh
    MBS=4 NUM_IMG_TOKENS=1024 NUM_TXT_TOKENS=256 GRAD_ACCUMULATION=1 \
      bash ./examples/megatron/recipes/reve/h100/launch_h100_nodes_baseline.sh
    MBS=8 NUM_IMG_TOKENS=1024 NUM_TXT_TOKENS=256 GRAD_ACCUMULATION=1 \
      bash ./examples/megatron/recipes/reve/h100/launch_h100_nodes_baseline.sh
    MBS=16 NUM_IMG_TOKENS=1024 NUM_TXT_TOKENS=256 GRAD_ACCUMULATION=1 \
      bash ./examples/megatron/recipes/reve/h100/launch_h100_nodes_baseline.sh

    MBS=1 NUM_IMG_TOKENS=4096 NUM_TXT_TOKENS=256 GRAD_ACCUMULATION=1 \
      bash ./examples/megatron/recipes/reve/h100/launch_h100_nodes_baseline.sh
    MBS=2 NUM_IMG_TOKENS=4096 NUM_TXT_TOKENS=256 GRAD_ACCUMULATION=1 \
      bash ./examples/megatron/recipes/reve/h100/launch_h100_nodes_baseline.sh
    MBS=4 NUM_IMG_TOKENS=4096 NUM_TXT_TOKENS=256 GRAD_ACCUMULATION=1 \
      bash ./examples/megatron/recipes/reve/h100/launch_h100_nodes_baseline.sh
    MBS=8 NUM_IMG_TOKENS=4096 NUM_TXT_TOKENS=256 GRAD_ACCUMULATION=1 \
      bash ./examples/megatron/recipes/reve/h100/launch_h100_nodes_baseline.sh
  '

# DFM Reve
srun --jobid 4679314 -N 1 --ntasks-per-node=1 \
  --container-image="${CONT}" \
  --container-mounts=/lustre:/lustre \
  --export=ALL,MY_MASTER_ADDR="$M_ADDR" \
  bash -c '
    cd /lustre/fsw/coreai_dlalgo_genai/huvu/codes/dfm_customers/DFM_reve_optim

    NUMBER_PACKED_SAMPLES=4 GBS=8 CONTEXT_SEQ_LEN=128 H_LATENTS=16 W_LATENTS=16 \
      bash ./examples/megatron/recipes/reve/h100/launch_h100_nodes.sh
    NUMBER_PACKED_SAMPLES=8 GBS=8 CONTEXT_SEQ_LEN=128 H_LATENTS=16 W_LATENTS=16 \
      bash ./examples/megatron/recipes/reve/h100/launch_h100_nodes.sh
    NUMBER_PACKED_SAMPLES=16 GBS=8 CONTEXT_SEQ_LEN=128 H_LATENTS=16 W_LATENTS=16 \
      bash ./examples/megatron/recipes/reve/h100/launch_h100_nodes.sh
    NUMBER_PACKED_SAMPLES=32 GBS=8 CONTEXT_SEQ_LEN=128 H_LATENTS=16 W_LATENTS=16 \
      bash ./examples/megatron/recipes/reve/h100/launch_h100_nodes.sh

    NUMBER_PACKED_SAMPLES=2 GBS=8 CONTEXT_SEQ_LEN=256 H_LATENTS=32 W_LATENTS=32 \
      bash ./examples/megatron/recipes/reve/h100/launch_h100_nodes.sh
    NUMBER_PACKED_SAMPLES=4 GBS=8 CONTEXT_SEQ_LEN=256 H_LATENTS=32 W_LATENTS=32 \
      bash ./examples/megatron/recipes/reve/h100/launch_h100_nodes.sh
    NUMBER_PACKED_SAMPLES=8 GBS=8 CONTEXT_SEQ_LEN=256 H_LATENTS=32 W_LATENTS=32 \
      bash ./examples/megatron/recipes/reve/h100/launch_h100_nodes.sh
    NUMBER_PACKED_SAMPLES=16 GBS=8 CONTEXT_SEQ_LEN=256 H_LATENTS=32 W_LATENTS=32 \
      bash ./examples/megatron/recipes/reve/h100/launch_h100_nodes.sh

    NUMBER_PACKED_SAMPLES=1 GBS=8 CONTEXT_SEQ_LEN=256 H_LATENTS=64 W_LATENTS=64 \
      bash ./examples/megatron/recipes/reve/h100/launch_h100_nodes.sh
    NUMBER_PACKED_SAMPLES=2 GBS=8 CONTEXT_SEQ_LEN=256 H_LATENTS=64 W_LATENTS=64 \
      bash ./examples/megatron/recipes/reve/h100/launch_h100_nodes.sh
    NUMBER_PACKED_SAMPLES=4 GBS=8 CONTEXT_SEQ_LEN=256 H_LATENTS=64 W_LATENTS=64 \
      bash ./examples/megatron/recipes/reve/h100/launch_h100_nodes.sh
    NUMBER_PACKED_SAMPLES=8 GBS=8 CONTEXT_SEQ_LEN=256 H_LATENTS=64 W_LATENTS=64 \
      bash ./examples/megatron/recipes/reve/h100/launch_h100_nodes.sh
  '
