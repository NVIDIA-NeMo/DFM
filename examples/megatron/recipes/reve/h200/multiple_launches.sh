# allocate node
salloc -p h200@cr+mp/viking@cr+mp/8gpu-224cpu-2048gb --time 4:00:00 -N 1 -A nemo-mcore --gres=gpu:8

# srun setup
CONT="nvcr.io/nvidian/nemo:25.11.rc4"
export M_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)


# baseline Reve
srun --jobid 1261072 -N 1 --ntasks-per-node=1 \
  --container-image="${CONT}" \
  --container-mounts=/home/scratch.svc_compute_arch:/home/scratch.svc_compute_arch \
  --export=ALL,MY_MASTER_ADDR="$M_ADDR" \
  bash -c '
    cd /home/scratch.svc_compute_arch/huvu/codes/dfm_customers/DFM_reve_optim

    MBS=4 NUM_IMG_TOKENS=256 NUM_TXT_TOKENS=128 GRAD_ACCUMULATION=1 \
      bash ./examples/megatron/recipes/reve/h200/launch_h200_nodes_docker_baseline.sh
    MBS=8 NUM_IMG_TOKENS=256 NUM_TXT_TOKENS=128 GRAD_ACCUMULATION=1 \
      bash ./examples/megatron/recipes/reve/h200/launch_h200_nodes_docker_baseline.sh
    MBS=16 NUM_IMG_TOKENS=256 NUM_TXT_TOKENS=128 GRAD_ACCUMULATION=1 \
      bash ./examples/megatron/recipes/reve/h200/launch_h200_nodes_docker_baseline.sh
    MBS=32 NUM_IMG_TOKENS=256 NUM_TXT_TOKENS=128 GRAD_ACCUMULATION=1 \
      bash ./examples/megatron/recipes/reve/h200/launch_h200_nodes_docker_baseline.sh
    MBS=32 NUM_IMG_TOKENS=256 NUM_TXT_TOKENS=128 GRAD_ACCUMULATION=2 \
      bash ./examples/megatron/recipes/reve/h200/launch_h200_nodes_docker_baseline.sh
    MBS=64 NUM_IMG_TOKENS=256 NUM_TXT_TOKENS=128 GRAD_ACCUMULATION=1 \
      bash ./examples/megatron/recipes/reve/h200/launch_h200_nodes_docker_baseline.sh


    MBS=2 NUM_IMG_TOKENS=1024 NUM_TXT_TOKENS=256 GRAD_ACCUMULATION=1 \
      bash ./examples/megatron/recipes/reve/h200/launch_h200_nodes_docker_baseline.sh
    MBS=4 NUM_IMG_TOKENS=1024 NUM_TXT_TOKENS=256 GRAD_ACCUMULATION=1 \
      bash ./examples/megatron/recipes/reve/h200/launch_h200_nodes_docker_baseline.sh
    MBS=8 NUM_IMG_TOKENS=1024 NUM_TXT_TOKENS=256 GRAD_ACCUMULATION=1 \
      bash ./examples/megatron/recipes/reve/h200/launch_h200_nodes_docker_baseline.sh
    MBS=8 NUM_IMG_TOKENS=1024 NUM_TXT_TOKENS=256 GRAD_ACCUMULATION=2 \
      bash ./examples/megatron/recipes/reve/h200/launch_h200_nodes_docker_baseline.sh
    MBS=16 NUM_IMG_TOKENS=1024 NUM_TXT_TOKENS=256 GRAD_ACCUMULATION=1 \
      bash ./examples/megatron/recipes/reve/h200/launch_h200_nodes_docker_baseline.sh

    MBS=1 NUM_IMG_TOKENS=4096 NUM_TXT_TOKENS=256 GRAD_ACCUMULATION=1 \
      bash ./examples/megatron/recipes/reve/h200/launch_h200_nodes_docker_baseline.sh
    MBS=2 NUM_IMG_TOKENS=4096 NUM_TXT_TOKENS=256 GRAD_ACCUMULATION=1 \
      bash ./examples/megatron/recipes/reve/h200/launch_h200_nodes_docker_baseline.sh
    MBS=2 NUM_IMG_TOKENS=4096 NUM_TXT_TOKENS=256 GRAD_ACCUMULATION=2 \
      bash ./examples/megatron/recipes/reve/h200/launch_h200_nodes_docker_baseline.sh
    MBS=2 NUM_IMG_TOKENS=4096 NUM_TXT_TOKENS=256 GRAD_ACCUMULATION=4 \
      bash ./examples/megatron/recipes/reve/h200/launch_h200_nodes_docker_baseline.sh
    MBS=4 NUM_IMG_TOKENS=4096 NUM_TXT_TOKENS=256 GRAD_ACCUMULATION=1 \
      bash ./examples/megatron/recipes/reve/h200/launch_h200_nodes_docker_baseline.sh
    MBS=8 NUM_IMG_TOKENS=4096 NUM_TXT_TOKENS=256 GRAD_ACCUMULATION=1 \
      bash ./examples/megatron/recipes/reve/h200/launch_h200_nodes_docker_baseline.sh
  '


# DFM Reve 
srun --jobid 1261458 -N 1 --ntasks-per-node=1 \
  --container-image="${CONT}" \
  --container-mounts=/home/scratch.svc_compute_arch:/home/scratch.svc_compute_arch \
  --export=ALL,MY_MASTER_ADDR="$M_ADDR" \
  bash  -c '
    cd /home/scratch.svc_compute_arch/huvu/codes/dfm_customers/DFM_reve_optim

    NUMBER_PACKED_SAMPLES=4 GBS=8 CONTEXT_SEQ_LEN=128 H_LATENTS=16 W_LATENTS=16 \
      bash ./examples/megatron/recipes/reve/h200/launch_h200_nodes_docker.sh
    NUMBER_PACKED_SAMPLES=8 GBS=8 CONTEXT_SEQ_LEN=128 H_LATENTS=16 W_LATENTS=16 \
      bash ./examples/megatron/recipes/reve/h200/launch_h200_nodes_docker.sh
    NUMBER_PACKED_SAMPLES=16 GBS=8 CONTEXT_SEQ_LEN=128 H_LATENTS=16 W_LATENTS=16 \
      bash ./examples/megatron/recipes/reve/h200/launch_h200_nodes_docker.sh
    NUMBER_PACKED_SAMPLES=32 GBS=8 CONTEXT_SEQ_LEN=128 H_LATENTS=16 W_LATENTS=16 RECOMPUTE_NUM_LAYERS=13 \
      bash ./examples/megatron/recipes/reve/h200/launch_h200_nodes_docker.sh

    NUMBER_PACKED_SAMPLES=2 GBS=8 CONTEXT_SEQ_LEN=256 H_LATENTS=32 W_LATENTS=32 \
      bash ./examples/megatron/recipes/reve/h200/launch_h200_nodes_docker.sh
    NUMBER_PACKED_SAMPLES=4 GBS=8 CONTEXT_SEQ_LEN=256 H_LATENTS=32 W_LATENTS=32 \
      bash ./examples/megatron/recipes/reve/h200/launch_h200_nodes_docker.sh
    NUMBER_PACKED_SAMPLES=8 GBS=8 CONTEXT_SEQ_LEN=256 H_LATENTS=32 W_LATENTS=32 RECOMPUTE_NUM_LAYERS=13 \
      bash ./examples/megatron/recipes/reve/h200/launch_h200_nodes_docker.sh
    NUMBER_PACKED_SAMPLES=16 GBS=8 CONTEXT_SEQ_LEN=256 H_LATENTS=32 W_LATENTS=32 RECOMPUTE_NUM_LAYERS=26 \
      bash ./examples/megatron/recipes/reve/h200/launch_h200_nodes_docker.sh

    NUMBER_PACKED_SAMPLES=1 GBS=8 CONTEXT_SEQ_LEN=256 H_LATENTS=64 W_LATENTS=64 \
      bash ./examples/megatron/recipes/reve/h200/launch_h200_nodes_docker.sh
    NUMBER_PACKED_SAMPLES=2 GBS=8 CONTEXT_SEQ_LEN=256 H_LATENTS=64 W_LATENTS=64 RECOMPUTE_NUM_LAYERS=13 \
      bash ./examples/megatron/recipes/reve/h200/launch_h200_nodes_docker.sh
    NUMBER_PACKED_SAMPLES=4 GBS=8 CONTEXT_SEQ_LEN=256 H_LATENTS=64 W_LATENTS=64 RECOMPUTE_NUM_LAYERS=26 \
      bash ./examples/megatron/recipes/reve/h200/launch_h200_nodes_docker.sh
    NUMBER_PACKED_SAMPLES=8 GBS=8 CONTEXT_SEQ_LEN=256 H_LATENTS=64 W_LATENTS=64 RECOMPUTE_NUM_LAYERS=26 \
      bash ./examples/megatron/recipes/reve/h200/launch_h200_nodes_docker.sh
  '

# DFM Reve (full_dimhead128)
srun --jobid 1261458 -N 1 --ntasks-per-node=1 \
  --container-image="${CONT}" \
  --container-mounts=/home/scratch.svc_compute_arch:/home/scratch.svc_compute_arch \
  --export=ALL,MY_MASTER_ADDR="$M_ADDR" \
  bash  -c '
    cd /home/scratch.svc_compute_arch/huvu/codes/dfm_customers/DFM_reve_optim

    NUMBER_PACKED_SAMPLES=4 GBS=8 CONTEXT_SEQ_LEN=128 H_LATENTS=16 W_LATENTS=16 \
      bash ./examples/megatron/recipes/reve/h200/launch_h200_nodes_dimhead128_docker.sh
    NUMBER_PACKED_SAMPLES=8 GBS=8 CONTEXT_SEQ_LEN=128 H_LATENTS=16 W_LATENTS=16 \
      bash ./examples/megatron/recipes/reve/h200/launch_h200_nodes_dimhead128_docker.sh
    NUMBER_PACKED_SAMPLES=16 GBS=8 CONTEXT_SEQ_LEN=128 H_LATENTS=16 W_LATENTS=16 \
      bash ./examples/megatron/recipes/reve/h200/launch_h200_nodes_dimhead128_docker.sh
    NUMBER_PACKED_SAMPLES=32 GBS=8 CONTEXT_SEQ_LEN=128 H_LATENTS=16 W_LATENTS=16 RECOMPUTE_NUM_LAYERS=13 \
      bash ./examples/megatron/recipes/reve/h200/launch_h200_nodes_dimhead128_docker.sh

    NUMBER_PACKED_SAMPLES=2 GBS=8 CONTEXT_SEQ_LEN=256 H_LATENTS=32 W_LATENTS=32 \
      bash ./examples/megatron/recipes/reve/h200/launch_h200_nodes_dimhead128_docker.sh
    NUMBER_PACKED_SAMPLES=4 GBS=8 CONTEXT_SEQ_LEN=256 H_LATENTS=32 W_LATENTS=32 \
      bash ./examples/megatron/recipes/reve/h200/launch_h200_nodes_dimhead128_docker.sh
    NUMBER_PACKED_SAMPLES=8 GBS=8 CONTEXT_SEQ_LEN=256 H_LATENTS=32 W_LATENTS=32 RECOMPUTE_NUM_LAYERS=13 \
      bash ./examples/megatron/recipes/reve/h200/launch_h200_nodes_dimhead128_docker.sh
    NUMBER_PACKED_SAMPLES=16 GBS=8 CONTEXT_SEQ_LEN=256 H_LATENTS=32 W_LATENTS=32 RECOMPUTE_NUM_LAYERS=26 \
      bash ./examples/megatron/recipes/reve/h200/launch_h200_nodes_dimhead128_docker.sh

    NUMBER_PACKED_SAMPLES=1 GBS=8 CONTEXT_SEQ_LEN=256 H_LATENTS=64 W_LATENTS=64 \
      bash ./examples/megatron/recipes/reve/h200/launch_h200_nodes_dimhead128_docker.sh
    NUMBER_PACKED_SAMPLES=2 GBS=8 CONTEXT_SEQ_LEN=256 H_LATENTS=64 W_LATENTS=64 RECOMPUTE_NUM_LAYERS=13 \
      bash ./examples/megatron/recipes/reve/h200/launch_h200_nodes_dimhead128_docker.sh
    NUMBER_PACKED_SAMPLES=4 GBS=8 CONTEXT_SEQ_LEN=256 H_LATENTS=64 W_LATENTS=64 RECOMPUTE_NUM_LAYERS=26 \
      bash ./examples/megatron/recipes/reve/h200/launch_h200_nodes_dimhead128_docker.sh
    NUMBER_PACKED_SAMPLES=8 GBS=8 CONTEXT_SEQ_LEN=256 H_LATENTS=64 W_LATENTS=64 RECOMPUTE_NUM_LAYERS=26 \
      bash ./examples/megatron/recipes/reve/h200/launch_h200_nodes_dimhead128_docker.sh
  '
