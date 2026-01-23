#!/usr/bin/env bash

set -e

MODEL_ID="/high_perf_store4/evad-tech-vla/houzhiyi/FLUX/models/FLUX.1-dev"
FULL_CKPT="outputs/flux_pretraining_ckpt/flux_epoch2000_full.pt"
OUTPUT_DIR="infer_case/eval_epoch2000_full_single"

NUM_STEPS=28
GUIDANCE=3.5
HEIGHT=256
WIDTH=256

  python3 scripts/eval_flux_fsdp.py \
  --model_id "${MODEL_ID}" \
  --full_ckpt_path "${FULL_CKPT}" \
  --output_dir "${OUTPUT_DIR}" \
  --num_inference_steps "${NUM_STEPS}" \
  --guidance_scale "${GUIDANCE}" \
  --height "${HEIGHT}" \
  --width "${WIDTH}"