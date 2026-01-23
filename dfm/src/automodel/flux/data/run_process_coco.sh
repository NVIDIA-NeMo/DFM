python preprocess_flux_images.py \
    --input_folder /high_perf_store2/evad-osc-datasets/datasets/mscoco/train2017 \
    --output_folder ./processed_256_256_3meta_debug_1 \
    --caption_file mscoco_captions_1k.json \
    --height 256 --width 256 \
    --model_id /high_perf_store4/evad-tech-vla/houzhiyi/FLUX/models/FLUX.1-dev \
    --max_samples 3