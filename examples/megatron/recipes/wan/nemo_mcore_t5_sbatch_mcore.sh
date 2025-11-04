#!/bin/bash

# Parameters
#SBATCH --account=coreai_dlalgo_llm
#SBATCH --job-name=coreai_dlalgo_llm-run:nemo_mcore_t5
#SBATCH --nodes=1
#SBATCH --partition=batch
#SBATCH --time=04:00:00

# sbatch script
mcore_t5=False

NUM_DEVICES=8
NEMO_DIR=/lustre/fsw/coreai_dlalgo_genai/huvu/codes/T5_nemo/NeMo
EXPNAME=nemo_nonmcore_t5_fp32_sbatch_mcore
PROJECT=t5_pretrain_sbatch

CONFIG_NAME='megatron_t5_config'

PRECISION=32
AMP_O2=False
MICRO_BATCH_SIZE=64
GLOBAL_BATCH_SIZE=512
ACCUMULATE_GRAD_BATCHES=1
TENSOR_MODEL_PARALLEL_SIZE=1
VAL_CHECK_INTERVAL=2000
MAX_STEPS=1000000
# BLEND="[.0333,/lustre/fsw/coreai_dlalgo_genai/huvu/data/t5/training_data/symlinks/my-t5_00_bert_tokenizer_text_document,.0333,/lustre/fsw/coreai_dlalgo_genai/huvu/data/t5/training_data/symlinks/my-t5_01_bert_tokenizer_text_document,.0333,/lustre/fsw/coreai_dlalgo_genai/huvu/data/t5/training_data/symlinks/my-t5_02_bert_tokenizer_text_document,.0333,/lustre/fsw/coreai_dlalgo_genai/huvu/data/t5/training_data/symlinks/my-t5_03_bert_tokenizer_text_document,.0333,/lustre/fsw/coreai_dlalgo_genai/huvu/data/t5/training_data/symlinks/my-t5_04_bert_tokenizer_text_document,.0333,/lustre/fsw/coreai_dlalgo_genai/huvu/data/t5/training_data/symlinks/my-t5_05_bert_tokenizer_text_document,.0333,/lustre/fsw/coreai_dlalgo_genai/huvu/data/t5/training_data/symlinks/my-t5_06_bert_tokenizer_text_document,.0333,/lustre/fsw/coreai_dlalgo_genai/huvu/data/t5/training_data/symlinks/my-t5_07_bert_tokenizer_text_document,.0333,/lustre/fsw/coreai_dlalgo_genai/huvu/data/t5/training_data/symlinks/my-t5_08_bert_tokenizer_text_document,.0333,/lustre/fsw/coreai_dlalgo_genai/huvu/data/t5/training_data/symlinks/my-t5_09_bert_tokenizer_text_document,.0333,/lustre/fsw/coreai_dlalgo_genai/huvu/data/t5/training_data/symlinks/my-t5_10_bert_tokenizer_text_document,.0333,/lustre/fsw/coreai_dlalgo_genai/huvu/data/t5/training_data/symlinks/my-t5_11_bert_tokenizer_text_document,.0333,/lustre/fsw/coreai_dlalgo_genai/huvu/data/t5/training_data/symlinks/my-t5_12_bert_tokenizer_text_document,.0333,/lustre/fsw/coreai_dlalgo_genai/huvu/data/t5/training_data/symlinks/my-t5_13_bert_tokenizer_text_document,.0333,/lustre/fsw/coreai_dlalgo_genai/huvu/data/t5/training_data/symlinks/my-t5_14_bert_tokenizer_text_document,.0333,/lustre/fsw/coreai_dlalgo_genai/huvu/data/t5/training_data/symlinks/my-t5_15_bert_tokenizer_text_document,.0333,/lustre/fsw/coreai_dlalgo_genai/huvu/data/t5/training_data/symlinks/my-t5_16_bert_tokenizer_text_document,.0333,/lustre/fsw/coreai_dlalgo_genai/huvu/data/t5/training_data/symlinks/my-t5_17_bert_tokenizer_text_document,.0333,/lustre/fsw/coreai_dlalgo_genai/huvu/data/t5/training_data/symlinks/my-t5_18_bert_tokenizer_text_document,.0333,/lustre/fsw/coreai_dlalgo_genai/huvu/data/t5/training_data/symlinks/my-t5_19_bert_tokenizer_text_document,.0333,/lustre/fsw/coreai_dlalgo_genai/huvu/data/t5/training_data/symlinks/my-t5_20_bert_tokenizer_text_document,.0333,/lustre/fsw/coreai_dlalgo_genai/huvu/data/t5/training_data/symlinks/my-t5_21_bert_tokenizer_text_document,.0333,/lustre/fsw/coreai_dlalgo_genai/huvu/data/t5/training_data/symlinks/my-t5_22_bert_tokenizer_text_document,.0333,/lustre/fsw/coreai_dlalgo_genai/huvu/data/t5/training_data/symlinks/my-t5_23_bert_tokenizer_text_document,.0333,/lustre/fsw/coreai_dlalgo_genai/huvu/data/t5/training_data/symlinks/my-t5_24_bert_tokenizer_text_document,.0333,/lustre/fsw/coreai_dlalgo_genai/huvu/data/t5/training_data/symlinks/my-t5_25_bert_tokenizer_text_document,.0333,/lustre/fsw/coreai_dlalgo_genai/huvu/data/t5/training_data/symlinks/my-t5_26_bert_tokenizer_text_document,.0333,/lustre/fsw/coreai_dlalgo_genai/huvu/data/t5/training_data/symlinks/my-t5_27_bert_tokenizer_text_document,.0333,/lustre/fsw/coreai_dlalgo_genai/huvu/data/t5/training_data/symlinks/my-t5_28_bert_tokenizer_text_document,.0334,/lustre/fsw/coreai_dlalgo_genai/huvu/data/t5/training_data/symlinks/my-t5_29_bert_tokenizer_text_document]"
BLEND="[/lustre/fsw/coreai_dlalgo_genai/huvu/data/t5/training_data/symlinks/my-t5_00_bert_tokenizer_text_document]"

# Model architecture
SEQ_LENGTH=512
SEQ_LENGTH_DEC=128
NUM_LAYERS=12
HIDDEN_SIZE=768
NUM_ATTENTION_HEADS=12

home_dir=/lustre/fsw/coreai_dlalgo_genai/huvu/data/t5/results/t5_nemo
exp_dir=$home_dir/${EXPNAME}
mkdir ${exp_dir}

cmd="

cd /lustre/fsw/coreai_dlalgo_genai/huvu/codes/T5_nemo/NeMo
pip install -e .
NEMO=/lustre/fsw/coreai_dlalgo_genai/huvu/codes/T5_nemo/NeMo
export PYTHONPATH="${NEMO}/.:${PYTHONPATH}" 
## Megatron-LM
cd /lustre/fsw/coreai_dlalgo_genai/huvu/codes/T5_nemo/megatron-lm
pip install -e .
MEGATRONLM=/lustre/fsw/coreai_dlalgo_genai/huvu/codes/T5_nemo/megatron-lm
export PYTHONPATH="${MEGATRONLM}/.:${PYTHONPATH}"
export PYTHONPATH="/lustre/fsw/coreai_dlalgo_genai/huvu/codes/T5_nemo/megatron-lm/.:/lustre/fsw/coreai_dlalgo_genai/huvu/codes/T5_nemo/NeMo/.:/opt/NeMo-Megatron-Launcher/launcher_scripts"

export WANDB_API_KEY=497a93e5ac7cf1e0ec821741ef7bac27b36f2db8

if [[ ${PRECISION} != "32" ]]; then
  export NVTE_FUSED_ATTN=0
  export NVTE_FLASH_ATTN=0
fi

python ${NEMO_DIR}/examples/nlp/language_modeling/megatron_t5_pretraining.py \
  --config-name=${CONFIG_NAME} \
  trainer.num_nodes=1 \
  trainer.devices=${NUM_DEVICES} \
  trainer.max_epochs=null \
  trainer.max_steps=${MAX_STEPS} \
  trainer.val_check_interval=${VAL_CHECK_INTERVAL} \
  trainer.accumulate_grad_batches=${ACCUMULATE_GRAD_BATCHES} \
  trainer.precision=${PRECISION} \
  trainer.log_every_n_steps=1 \
  model.megatron_amp_O2=${AMP_O2} \
  model.micro_batch_size=${MICRO_BATCH_SIZE} \
  model.global_batch_size=${GLOBAL_BATCH_SIZE} \
  model.tensor_model_parallel_size=${TENSOR_MODEL_PARALLEL_SIZE} \
  model.max_position_embeddings=${SEQ_LENGTH} \
  model.seq_length=${SEQ_LENGTH} \
  model.encoder.hidden_size=${HIDDEN_SIZE} \
  model.decoder.hidden_size=${HIDDEN_SIZE} \
  model.encoder.num_layers=${NUM_LAYERS} \
  model.decoder.num_layers=${NUM_LAYERS} \
  model.encoder.num_attention_heads=${NUM_ATTENTION_HEADS} \
  model.decoder.num_attention_heads=${NUM_ATTENTION_HEADS} \
  model.encoder.init_method_std=0.015 \
  model.decoder.init_method_std=0.015 \
  model.encoder.transformer_block_type='pre_ln' \
  model.decoder.transformer_block_type='pre_ln' \
  model.data.data_prefix=${BLEND} \
  model.data.seq_length=${SEQ_LENGTH} \
  model.tokenizer.vocab_file=/lustre/fsw/coreai_dlalgo_genai/huvu/data/t5/training_data/symlinks/bert-large-cased-vocab.txt \
  model.data.seq_length_dec=${SEQ_LENGTH_DEC} \
  model.data.splits_string=\'99982,9,9\' \
  model.data.num_workers=4 \
  model.optim.name=distributed_fused_adam \
  model.mcore_t5=${mcore_t5} \
  model.transformer_engine=True \
  +model.kv_channels=64 \
  exp_manager.create_wandb_logger=True \
  exp_manager.wandb_logger_kwargs.name=${EXPNAME} \
  exp_manager.wandb_logger_kwargs.project=${PROJECT} \
  +exp_manager.wandb_logger_kwargs.resume=True \
  exp_manager.explicit_log_dir=${exp_dir} \
  exp_manager.resume_if_exists=True \
  exp_manager.resume_ignore_no_checkpoint=True \
  exp_manager.create_checkpoint_callback=True \
  exp_manager.checkpoint_callback_params.monitor=val_loss \
  exp_manager.checkpoint_callback_params.save_top_k=3 \
  exp_manager.checkpoint_callback_params.mode=min \
  exp_manager.checkpoint_callback_params.always_save_nemo=False \
  ++exp_manager.checkpoint_callback_params.save_nemo_on_train_end=False \
  ++exp_manager.log_step_timing=True \
"  

# ++model.async_grad_allreduce=False for training with bf16, O2, FusedAdam

CONT="gitlab-master.nvidia.com/dl/joc/nemo-ci/train:pipe.14465850"
MOUNT="/lustre/fsw/:/lustre/fsw/"
OUTFILE=$exp_dir/slurm-%j.out
ERRFILE=$exp_dir/error-%j.out
export CUDA_DEVICE_MAX_CONNECTIONS=1

echo "Running training script."
srun -o ${OUTFILE} -e ${ERRFILE} --mpi=pmix \
    --container-image="${CONT}" --container-mounts="${MOUNT}" \
    --no-container-mount-home \
    --ntasks-per-node=8 \
    -N ${SLURM_JOB_NUM_NODES}  \
    bash -c "${cmd}"