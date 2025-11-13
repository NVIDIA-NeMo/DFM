## Megatron WAN 2.1

### Overview
WAN 2.1 is an open, large-scale video generative model series focused on high-quality text-to-video and text-to-image generation. This recipe re-implements WAN using [Megatron-Core](https://github.com/NVIDIA/Megatron-LM) to improve training efficiency and scalability via advanced parallelism schemes and throughput optimizations, including data/tensor/sequence/context parallelism and fused kernels (e.g., NVTE fused attention).


### Dataset Preparation
- This recipe uses NVIDIA's [Megatron-Energon](https://github.com/NVIDIA/Megatron-Energon) as an efficient multi-modal data loader.
- Datasets should be in the WebDataset-compatible format (typically sharded `.tar` archives). Energon efficiently supports large-scale distributed loading, sharding, and sampling for multi-modal pairs (e.g., text-image, text-video).
- Point `dataset.path` to your WebDataset location or shard pattern (e.g., a directory containing shards). See the Megatron-Energon documentation for format details and advanced options.


### Training and Finetuning
- Use `--training-mode` to select the correct flow-matching hyper-parameters:
  - `pretrain`: default pretraining configuration
  - `finetune`: finetuning configuration (uses different flow-matching hyper-parameters)

Set environment variables like `EXP_NAME` and `CHECKPOINT_DIR` as desired before running.

#### Example: Pretrain WAN 1.3B
```bash
NVTE_FUSED_ATTN=1 torchrun --nproc_per_node=8 examples/megatron/recipes/wan/pretrain_wan.py \
  --training-mode pretrain \
  model.tensor_model_parallel_size=1 \
  model.pipeline_model_parallel_size=1 \
  model.context_parallel_size=4 \
  model.crossattn_emb_size=1536 \
  model.hidden_size=1536 \
  model.ffn_hidden_size=8960 \
  model.num_attention_heads=12 \
  model.num_layers=30 \
  model.qkv_format=thd \
  dataset.path=/path/to/dataset \
  checkpoint.save=/path/to/checkpoint_dir \
  checkpoint.load=/path/to/checkpoint_dir \
  checkpoint.load_optim=true \
  checkpoint.save_interval=200 \
  optimizer.lr=5e-6 \
  optimizer.min_lr=5e-6 \
  train.eval_iters=0 \
  scheduler.lr_decay_style=constant \
  scheduler.lr_warmup_iters=0 \
  model.seq_length=2048 \
  dataset.seq_length=2048 \
  train.global_batch_size=2 \
  train.micro_batch_size=1 \
  dataset.global_batch_size=2 \
  dataset.micro_batch_size=1 \
  logger.log_interval=1 \
  logger.wandb_project="wan" \
  logger.wandb_exp_name="${EXP_NAME}" \
  logger.wandb_save_dir="${CHECKPOINT_DIR}"
```

#### Finetuning
- Switch `--training-mode finetune` to enable the finetuning flow-matching setup. Adjust dataset and optimization parameters (learning rate, warmup steps, etc.) as needed for your task and hardware.

### Inference
```bash
NVTE_FUSED_ATTN=1 torchrun --nproc_per_node=1 examples/megatron/recipes/wan/inference_wan.py  \
  --task t2v-1.3B \
  --sizes 480*832 \
  --checkpoint_dir /path/to/checkpoint \
  --checkpoint_step 0 \
  --frame_nums 81 \
  --prompts "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage." \
  --tensor_parallel_size 1 \
  --context_parallel_size 1 \
  --pipeline_parallel_size 1 \
  --sequence_parallel False \
  --base_seed 42 \
  --sample_steps 50
```

### Parallelism Support
The table below shows current parallelisms support for corresponding Wan model size.

  | Model | Data Parallel | Tensor Parallel | Sequence Parallel | Pipeline Parallel | Context Parallel | FSDP |
  |---|---|---|---|---|---|---|
  | **1.3B** | ✅ | ✅ | ✅ |  | ✅ |  |
  | **14B**  | ✅ | ✅ | ✅ |  | ✅ |  |


### Performance
The table below shows performances of corresponding Wan model size on a variety of Nvidia hardware (measured by TFLOPs/GPU).

  | Model | H100 | GB200 | GB300 |
  |---|---|---|---|
  | **1.3B** |  |    |  |
  | **14B** | 308 |  790  | 1000 |


### Citation
```bibtex
@article{wan2.1,
  title   = {Wan: Open and Advanced Large‐Scale Video Generative Models},
  author  = {Wan Team},
  year    = {2025},
  note    = {Open­source video foundation model series (Wan 2.1), https://github.com/Wan-Video/Wan2.1/}
}
```

