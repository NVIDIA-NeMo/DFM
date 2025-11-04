#!/bin/bash

# Parameters
#SBATCH --account=coreai_dlalgo_llm
#SBATCH --job-name=coreai_dlalgo_llm-run:dfm
#SBATCH --nodes=1
#SBATCH --partition=batch
#SBATCH --time=04:00:00


OUTPUT_DIR=/lustre/fsw/coreai_dlalgo_genai/huvu/data/nemo_vfm/datasets/coyo_dataset_wan/processed_coyo_dataset_wan

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
python dfm/src/megatron/data/wan/prepare_energon_dataset_wan_square_images.py \
  --input_dir /lustre/fsw/coreai_dlalgo_genai/datasets/coyo-700m/part_00000 \
  --output_dir /lustre/fsw/coreai_dlalgo_genai/huvu/data/nemo_vfm/datasets/coyo_dataset_wan/processed_coyo_dataset_wan/part_00000_square_wds \
  --model Wan-AI/Wan2.1-T2V-14B-Diffusers \
  --gpus 0,1,2,3,4,5,6,7 \
  --size 256 \
  --resize_mode bilinear \
  --save-image \
  --skip-existing \
  --stochastic

"  

CONT="nvcr.io/nvidia/nemo:25.09.00"
MOUNT="/lustre/fsw/:/lustre/fsw/"
OUTFILE=$OUTPUT_DIR/slurm-%j.out
ERRFILE=$OUTPUT_DIR/error-%j.out
export CUDA_DEVICE_MAX_CONNECTIONS=1

echo "Running training script."
srun -o ${OUTFILE} -e ${ERRFILE} --mpi=pmix \
    --container-image="${CONT}" --container-mounts="${MOUNT}" \
    --no-container-mount-home \
    --ntasks-per-node=1 \
    -N ${SLURM_JOB_NUM_NODES}  \
    bash -c "${cmd}"