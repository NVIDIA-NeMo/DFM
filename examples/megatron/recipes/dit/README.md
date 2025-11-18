## Megatron DiT

### Overview
An open source implementation of Diffusion Transformers (DiTs) that can be used to train text-to-image/video models. The implementation is based on [Megatron-Core](https://github.com/NVIDIA/Megatron-LM) and [Megatron-Bridge](https://github.com/NVIDIA-NeMo/Megatron-Bridge) to bring both scalability and efficiency. Various parallelization techniques such as tensor, sequence, and context parallelism are currently supported.


### Dataset Preparation
This recipe uses NVIDIA's [Megatron-Energon](https://github.com/NVIDIA/Megatron-Energon) as an efficient multi-modal data loader. Datasets should be in the WebDataset-compatible format (typically sharded `.tar` archives). Energon efficiently supports large-scale distributed loading, sharding, and sampling for multi-modal pairs (e.g., text-image, text-video). Set `dataset.path` to your WebDataset location or shard pattern (e.g., a directory containing shards). See the Megatron-Energon documentation for format details and advanced options.

#### Dataset Preparation Example

As an example you can use [butterfly-dataset](https://huggingface.co/datasets/huggan/smithsonian_butterflies_subset) available on Hugging Face.

The script below prepares the dataset to be compatible with Energon. t5_folder and tokenizer_cache_dir are optional parameters pointing to a T5 model and Video Tokenizer of your choice, otherwise the code downloads such artifacts.
``` bash
uv run --group megatron-bridge python -m torch.distributed.run --nproc-per-node $num_gpus\
       examples/megatron/recipes/dit/prepare_energon_dataset_butterfly.py\
       --t5_cache_dir $t5_folder\
       --tokenizer_cache_dir $tokenizer_cache_dir
```
Then you need to run `energon prepare $dataset_path` and choose `CrudeWebdataset` as the sample type:

```bash
energon prepare ./
/usr/local/lib/python3.12/dist-packages/torch/cuda/__init__.py:63: FutureWarning: The pynvml package is deprecated. Please install nvidia-ml-py instead. If you did not install pynvml directly, please report this to the maintainers of the package that installed pynvml for you.
  import pynvml  # type: ignore[import]
Found 8 tar files in total. The first and last ones are:
- rank0-000000.tar
- rank7-000000.tar
If you want to exclude some of them, cancel with ctrl+c and specify an exclude filter in the command line.
Please enter a desired train/val/test split like "0.5, 0.2, 0.3" or "8,1,1": 1,0,0
Saving info to /opt/datasets/butterfly_webdataset_new/.nv-meta/.info.yaml
Sample 0, keys:
 - json
 - pickle
 - pth
Json content of sample 0 of rank0-000000.tar:
{
  "image_height": 1,
  "image_width": 512
}
Sample 1, keys:
 - json
 - pickle
 - pth
Json content of sample 1 of rank0-000000.tar:
{
  "image_height": 1,
  "image_width": 512
}
Found the following part types in the dataset: pth, json, pickle
Do you want to create a dataset.yaml interactively? [Y/n]: y
The following sample types are available:
0. CaptioningSample
1. ImageClassificationSample
2. ImageSample
3. InterleavedSample
4. MultiChoiceVQASample
5. OCRSample
6. Sample
7. SimilarityInterleavedSample
8. TextSample
9. VQASample
10. VidQASample
11. Crude sample (plain dict for cooking)
Please enter a number to choose a class: 11
CrudeWebdataset does not need a field map. You will need to provide a `Cooker` for your dataset samples in your `TaskEncoder`.
Furthermore, you might want to add `subflavors` in your meta dataset specification.
Done
```

### Pretraining
These scripts assume you're using the Docker container provided by the repo. Use them to pre-train a DiT model on your own dataset.

**Note:** Set the `WANDB_API_KEY` environment variable if you're using the `wandb_project` and `wandb_exp_name` arguments.

```bash
uv run --group megatron-bridge python -m torch.distributed.run \
       --nproc-per-node $NUM_GPUS examples/megatron/recipes/dit/pretrain_dit_model.py\
        model.tensor_model_parallel_size=1 \
        model.pipeline_model_parallel_size=1 \
        model.context_parallel_size=1 \
        model.qkv_format=thd \
        model.num_attention_heads=16\
        model.vae_cache_folder=$CACHE_FOLDER\
        dataset.path=$DATA_FOLDER \
        dataset.task_encoder_seq_length=15360\
        dataset.packing_buffer_size=100\
        dataset.num_workers=20\
        checkpoint.save=$CHECKPOINT_FOLDER \
        checkpoint.load=$CHECKPOINT_FOLDER \
        checkpoint.load_optim=true \
        checkpoint.save_interval=1000 \
        train.eval_interval=1000\
        train.train_iters=10000\
        train.eval_iters=32 \
        train.global_batch_size=$NUM_GPUS\
        train.micro_batch_size=1\
        logger.log_interval=10\
        logger.wandb_project="DiT"\
        logger.wandb_exp_name=$WANDB_NAME
```

### Inference
``` bash
uv run --group megatron-bridge python -m torch.distributed.run --nproc-per-node $num_gpus examples/megatron/recipes/dit/inference_dit_model.py \
	--t5_cache_dir $artifact_dir \
    --tokenizer_cache_dir $tokenizer_cache_dir \
    --tokenizer_model Cosmos-0.1-Tokenizer-CV4x8x8\
	--checkpoint_path $checkpoint_dir \
	--num_video_frames 10 \
	--height 240 \
	--width 416 \
	--video_save_path $save_path \
	--prompt $prompt
```

### Parallelism Support
The table below shows current parallelism support.

  | Model | Data Parallel | Tensor Parallel | Sequence Parallel | Pipeline Parallel | Context Parallel | FSDP |
  |---|---|---|---|---|---|---|
  | **DiT-XL (700M)** | ✅ | ✅ | ✅ |  | ✅ |  |
  | **DiT 7B**  | | | | | |  |


### Mock Dataset

For performance measurement purposes you can use the mock dataset by passing the `--mock` argument.

``` bash
uv run --group megatron-bridge python -m torch.distributed.run \
       --nproc-per-node $NUM_GPUS examples/megatron/recipes/dit/pretrain_dit_model.py\
        model.tensor_model_parallel_size=1 \
        model.pipeline_model_parallel_size=1 \
        model.context_parallel_size=1 \
        model.qkv_format=thd \
        model.num_attention_heads=16\
        model.vae_cache_folder=$CACHE_FOLDER\
        dataset.path=$DATA_FOLDER \
        dataset.task_encoder_seq_length=15360\
        dataset.packing_buffer_size=100\
        dataset.num_workers=20\
        checkpoint.save=$CHECKPOINT_FOLDER \
        checkpoint.load=$CHECKPOINT_FOLDER \
        checkpoint.load_optim=true \
        checkpoint.save_interval=1000 \
        train.eval_interval=1000\
        train.train_iters=10000\
        train.eval_iters=32 \
        train.global_batch_size=$NUM_GPUS\
        train.micro_batch_size=1\
        logger.log_interval=10\
        --mock
```
