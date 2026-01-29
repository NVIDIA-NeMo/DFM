# Adjust based on your training parallelism
num_gpus=8

uv run --group megatron-bridge python \
    -m torch.distributed.run \
    --nproc-per-node $num_gpus \
    examples/megatron/recipes/wan/vbench_eval/generate_vbench_videos.py \
    --checkpoint_dir /opt/artifacts/convergence_test_take_3_wan_dmd_latent_distill_gan_0.03_experiment_vidprom_wan_vidprom_wan_480p_latents_cp_1_pp_1_tp_1 \
    --output_dir /opt/artifacts/vbench_outputs/megatron_student_standard_take_3/ \
    --use_student_model \
    --student_sample_steps 4 \
    --num_videos_per_prompt 5 \
    --student_sample_type sde \
    --t_list 0.999 0.967 0.908 0.769 0.0 \
    --checkpoint_step 400 \
    --batch_size 1 \
    --seed 42
