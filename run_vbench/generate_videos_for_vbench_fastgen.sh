num_gpus=8

DATA_ROOT_DIR='/opt/datasets' FASTGEN_OUTPUT_ROOT='/opt/artifacts/FASTGEN_OUTPUT' HF_HOME=/opt/artifacts/ LOCAL_FILES_ONLY=True uv run --group megatron-bridge python \
    -m torch.distributed.run \
    --nproc-per-node $num_gpus \
    examples/megatron/recipes/wan/vbench_eval/generate_vbench_videos.py \
    --fastgen_config dfm/src/fastgen/fastgen/configs/experiments/Wan/config_dmd2_vidprom_latent.py \
    --checkpoint_dir /opt/artifacts/FASTGEN_OUTPUT/fastgen/wan_dmd2/convergence_test_take_3_dmd2s5_m2p10_wan_8N_KD_vidpL_msl3_lr1e-5_g0.03/checkpoints/0000400.pth \
    --output_dir /opt/artifacts/vbench_outputs/fastgen_student_standard_take_3/ \
    --load_fastgen_checkpoint \
    --student_sample_steps 4 \
    --num_videos_per_prompt 5 \
    --student_sample_type sde \
    --t_list 0.999 0.967 0.908 0.769 0.0 \
    --checkpoint_step 400 \
    --batch_size 1 \
    --seed 42
