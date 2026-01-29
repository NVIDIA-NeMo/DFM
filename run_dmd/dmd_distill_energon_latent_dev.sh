num_gpus=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
# num_gpus=1
# Using precomputed latent dataset
dataset="vidprom_wan/vidprom_wan_480p_latents"
num_train_step=100000
gan_loss_weight=0.03
custom_name="dev_wan_dmd_latent_distill_gan_${gan_loss_weight}_"
cp=1
pp=1
tp=1
exp_name="${custom_name}experiment_${dataset//\//_}_cp_${cp}_pp_${pp}_tp_${tp}"

# Force PyTorch optimizer instead of FusedAdam from TE/Apex
# Note: env var must be passed BEFORE uv run command on the same line
HF_HOME="/opt/artifacts/" MEGATRON_USE_PYTORCH_OPTIMIZER=1 uv run --group megatron-bridge python -m torch.distributed.run \
        --nproc-per-node $num_gpus examples/megatron/recipes/wan/wan_dmd.py\
        --use-fastgen-dataset \
        --config-file examples/megatron/recipes/wan/conf/wan_dmd.yaml \
	train.global_batch_size=8\
	model.tensor_model_parallel_size=$tp\
        model.pipeline_model_parallel_size=$pp\
	model.context_parallel_size=$cp\
	train.eval_interval=25\
	model.fast_gen_config.gan_loss_weight_gen=${gan_loss_weight}\
	model.fast_gen_config.student_update_freq=5\
	model.fast_gen_config.student_sample_steps=4\
	model.fast_gen_config.sample_t_cfg.t_list="[0.999, 0.967, 0.908, 0.769, 0]"\
	model.fast_gen_config.sample_t_cfg.time_dist_type="shifted" \
    model.fast_gen_config.sample_t_cfg.shift=5.0 \
	model.fast_gen_config.discriminator.disc_type=multiscale_down_mlp_large\
	model.fast_gen_config.discriminator.feature_indices="[15, 22, 29]"\
	model.fast_gen_config.gan_use_same_t_noise=True\
	model.fast_gen_config.fake_score_pred_type=x0\
	optimizer.lr=1e-5\
	optimizer.min_lr=1e-5\
	optimizer.weight_decay=0.01\
	optimizer.adam_beta2=0.999\
	optimizer.clip_grad=10.0\
	optimizer.adam_eps=1e-8\
	scheduler.start_weight_decay=0.01\
	scheduler.end_weight_decay=0.01\
	optimizer.skip_fp32_master_weights=true\
	optimizer.use_distributed_optimizer=false\
	ddp.use_distributed_optimizer=false\
	checkpoint.save="/opt/artifacts/$exp_name" \
	checkpoint.load="/opt/artifacts/$exp_name" \
	dataset.path="/opt/datasets/${dataset}/" \
	train.train_iters=100000 \
	logger.wandb_exp_name=${exp_name}
