# torchrun \
#   --nproc-per-node=8 \
#   --master_port=29501 \
#   scripts/merge_sharded_ckpt_to_full.py \
#   --model_id "/high_perf_store4/evad-tech-vla/houzhiyi/FLUX/models/FLUX.1-dev" \
#   --ckpt_dir "outputs/flux_training_ckpt/ckpt_epoch_400_sharded" \
#   --output_path "outputs/flux_training_ckpt/flux_epoch400_full.pt"

  torchrun \
  --nproc-per-node=8 \
  --master_port=29501 \
  scripts/merge.py \
  --model_id "./models/FLUX.1-dev" \
  --ckpt_dir "outputs/flux_pretraining_ckpt/ckpt_epoch_2000_sharded" \
  --output_path "outputs/flux_pretraining_ckpt/flux_epoch2000_full.pt"