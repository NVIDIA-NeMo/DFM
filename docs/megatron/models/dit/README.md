## 🚀 Megatron DiT

### 📋 Overview
An open-source implementation of [Diffusion Transformers (DiTs)](https://github.com/facebookresearch/DiT) for training text-to-image/video models with [EDMPipeline](https://arxiv.org/abs/2206.00364). The implementation is based on [Megatron-Core](https://github.com/NVIDIA/Megatron-LM) and [Megatron-Bridge](https://github.com/NVIDIA-NeMo/Megatron-Bridge) to bring both scalability and efficiency. Various parallelization techniques such as tensor, sequence, and context parallelism are currently supported.

---

### 📦 Dataset Preparation
This recipe uses NVIDIA's [Megatron-Energon](https://github.com/NVIDIA/Megatron-Energon) as an efficient multi-modal data loader. Datasets should be in the WebDataset-compatible format (typically sharded `.tar` archives). Energon efficiently supports large-scale distributed loading, sharding, and sampling for multi-modal pairs (e.g., text-image, text-video). Set `dataset.path` to your WebDataset location or shard pattern. See the Megatron-Energon documentation for format details and advanced options.

#### 🦋 Dataset Preparation Example

As an example, you can use the [butterfly-dataset](https://huggingface.co/datasets/huggan/smithsonian_butterflies_subset) available on Hugging Face.

The script below packs the Hugging Face dataset into WebDataset format, which Energon requires.
```bash
uv run --group megatron-bridge python -m torch.distributed.run --nproc-per-node $num_gpus \
       examples/megatron/recipes/dit/prepare_energon_dataset_butterfly.py
```

In case you already have the T5 model or video tokenizer downloaded, you can point to them with optional arguments `--t5_cache_dir` and `--tokenizer_cache_dir`.


```bash
uv run --group megatron-bridge python -m torch.distributed.run --nproc-per-node $num_gpus \
       examples/megatron/recipes/dit/prepare_energon_dataset_butterfly.py \
       --t5_cache_dir $t5_cache_dir \
       --tokenizer_cache_dir $tokenizer_cache_dir
```

Then you need to run `energon prepare $dataset_path` and choose `CrudeWebdataset` as the sample type:

```bash
energon prepare ./
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

---

### 🐳 Build Container

Please follow the instructions in the [container](https://github.com/NVIDIA-NeMo/DFM#build-your-own-container) section of the main README.

---

### 🏋️ Pretraining

Once you have the dataset and container ready, you can start training the DiT model on your own dataset. This repository leverages [sequence packing](https://docs.nvidia.com/nemo-framework/user-guide/24.09/nemotoolkit/features/optimizations/sequence_packing.html) to maximize training efficiency. Sequence packing stacks multiple samples into a single sequence instead of padding individual samples to a fixed length; therefore, `micro_batch_size` must be set to 1. Additionally, `qkv_format` should be set to `thd` to signal to Transformer Engine that sequence packing is enabled.

For data loading, Energon provides two key hyperparameters related to sequence packing: `task_encoder_seq_length` and `packing_buffer_size`. The `task_encoder_seq_length` parameter controls the maximum sequence length passed to the model, while `packing_buffer_size` determines the number of samples processed to create different buckets. You can look at `select_samples_to_pack` and `pack_selected_samples` methods of [DiffusionTaskEncoderWithSequencePacking](https://github.com/NVIDIA-NeMo/DFM/blob/main/dfm/src/megatron/data/common/diffusion_task_encoder_with_sp.py#L50) to get a better sense of these parameters. For further details you can look at [Energon packing](https://nvidia.github.io/Megatron-Energon/advanced/packing.html) documenation.

Multiple parallelism techniques including tensor, sequence, and context parallelism are supported and can be configured based on your computational requirements.

The model architecture can be customized through parameters such as `num_layers` and `num_attention_heads`. A comprehensive list of configuration options is available in the [Megatron-Bridge documentation](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/main/docs/megatron-lm-to-megatron-bridge.md).


**Note:** If using the `wandb_project` and `wandb_exp_name` arguments, ensure the `WANDB_API_KEY` environment variable is set.


**Note:** During validation, the model generates one sample per GPU at the start of each validation round. These samples are saved to a `validation_generation` folder within `checkpoint_dir` and are also logged to Wandb if the `WANDB_API_KEY` environment variable is configured. To decode the generated latent samples, the model requires access to the video tokenizer used during dataset preparation. Specify the VAE artifacts location using the `vae_cache_folder` argument, otherwise they will be downloaded in the first validation round.

#### Pretraining script example
First, copy the example config file and update it with your own settings:

```bash
cp examples/megatron/recipes/dit/conf/dit_pretrain.yaml examples/megatron/recipes/dit/conf/my_config.yaml
# Edit my_config.yaml to set:
# - model.vae_cache_folder: Path to VAE cache folder
# - dataset.path: Path to your dataset folder
# - checkpoint.save and checkpoint.load: Path to checkpoint folder
# - train.global_batch_size: Set to be divisible by NUM_GPUs
# - logger.wandb_exp_name: Your experiment name
```

Then run:

```bash
uv run --group megatron-bridge python -m torch.distributed.run \
       --nproc-per-node $NUM_GPUS examples/megatron/recipes/dit/pretrain_dit_model.py \
       --config-file examples/megatron/recipes/dit/conf/my_config.yaml
```

You can still override any config values from the command line:

```bash
uv run --group megatron-bridge python -m torch.distributed.run \
       --nproc-per-node $num_gpus examples/megatron/recipes/dit/pretrain_dit_model.py \
       --config-file examples/megatron/recipes/dit/conf/my_config.yaml \
       train.train_iters=20000 \
       model.num_layers=32
```

**Note:** If you dedicate 100% of the data to training, you need to pass `dataset.use_train_split_for_val=true` to use a subset of training data for validation purposes.

```bash
uv run --group megatron-bridge python -m torch.distributed.run \
       --nproc-per-node $num_gpus examples/megatron/recipes/dit/pretrain_dit_model.py \
       --config-file examples/megatron/recipes/dit/conf/my_config.yaml \
       dataset.use_train_split_for_val=true
```

#### 🧪 Quick Start with Mock Dataset

If you want to run the code without having the dataset ready (for performance measurement purposes, for example), you can pass the `--mock` flag to activate a mock dataset.

```bash
uv run --group megatron-bridge python -m torch.distributed.run \
       --nproc-per-node $num_gpus examples/megatron/recipes/dit/pretrain_dit_model.py \
       --config-file examples/megatron/recipes/dit/conf/dit_pretrain.yaml \
       --mock
```

### 🎬 Inference

Once training completes, you can run inference using [inference_dit_model.py](https://github.com/NVIDIA-NeMo/DFM/blob/main/examples/megatron/recipes/dit/inference_dit_model.py). The script requires your trained model checkpoint (`--checkpoint_path`) and a path to save generated videos (`--video_save_path`). You can pass two optional arguments, `--t5_cache_dir` and `--tokenizer_cache_dir`, to avoid re-downloading artifacts if they are already downloaded.

```bash
uv run --group megatron-bridge python -m torch.distributed.run --nproc-per-node $num_gpus \
    examples/megatron/recipes/dit/inference_dit_model.py \
    --t5_cache_dir $artifact_dir \
    --tokenizer_cache_dir $tokenizer_cache_dir \
    --tokenizer_model Cosmos-0.1-Tokenizer-CV4x8x8 \
    --checkpoint_path $checkpoint_dir \
    --num_video_frames 10 \
    --height 240 \
    --width 416 \
    --video_save_path $save_path \
    --prompt "$prompt"
```

---

### ⚡ Parallelism Support

The table below shows current parallelism support for different model sizes:

| Model | Data Parallel | Tensor Parallel | Sequence Parallel | Context Parallel |
|---|---|---|---|---|
| **DiT-S (330M)** | TBD | TBD | TBD | TBD |
| **DiT-L (450M)** | TBD | TBD | TBD| TBD |
| **DiT-XL (700M)** | ✅ | ✅ | ✅ | ✅ |
