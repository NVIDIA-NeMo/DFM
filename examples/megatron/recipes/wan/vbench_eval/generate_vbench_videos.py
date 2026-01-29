#!/usr/bin/env python3
# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Generate videos for VBench evaluation using WAN model.

Usage:
    # Single GPU with batching
    python generate_vbench_videos.py \
        --checkpoint_dir /path/to/checkpoint \
        --output_dir ./vbench_outputs \
        --batch_size 4

    # Multi-GPU data parallelism (automatically detected - each GPU processes different prompts)
    # 4 GPUs = ~4x faster generation
    torchrun --nproc_per_node=4 generate_vbench_videos.py \
        --checkpoint_dir /path/to/checkpoint \
        --output_dir ./vbench_outputs \
        --batch_size 8

    # Fastgen checkpoint with 8 GPUs
    torchrun --nproc_per_node=8 generate_vbench_videos.py \
        --checkpoint_dir /path/to/fastgen_checkpoint.pth \
        --load_fastgen_checkpoint \
        --fastgen_config dfm/src/fastgen/fastgen/configs/experiments/Wan/config_dmd2_vidprom_latent.py \
        --student_sample_steps 4 \
        --student_sample_type sde \
        --t_list 0.999 0.967 0.908 0.769 0.0 \
        --output_dir ./vbench_outputs \
        --batch_size 16

    # Multi-GPU with tensor parallelism (model split across GPUs for large models)
    torchrun --nproc_per_node=8 generate_vbench_videos.py \
        --checkpoint_dir /path/to/checkpoint \
        --output_dir ./vbench_outputs \
        --tensor_parallel_size 8 \
        --batch_size 4
"""

import argparse
import json
import logging
import math
import os
import subprocess
import sys
import warnings
from functools import partial
from pathlib import Path
from types import SimpleNamespace


# Add VBench to Python path
sys.path.insert(0, "/opt/dfm/VBench")

warnings.filterwarnings("ignore")

import torch
import torch.distributed as dist
import torch.nn.functional as F
from megatron.core import parallel_state as ps
from tqdm import tqdm

from dfm.src.megatron.model.wan.flow_matching.flow_inference_pipeline import (
    FlowInferencePipeline,
)


def _encode_text_batch(
    tokenizer,
    text_encoder,
    device,
    captions,
    max_length=512,
):
    """
    Encode multiple text captions in batch for efficiency.

    Args:
        tokenizer: Text tokenizer
        text_encoder: Text encoder model
        device: Device to run encoding on
        captions: List of text captions
        max_length: Maximum sequence length

    Returns:
        List of encoded text tensors (one per caption, each trimmed to true length)
    """
    if len(captions) == 0:
        return []

    # Strip all captions
    captions = [caption.strip() for caption in captions]

    # Tokenize all captions in batch
    inputs = tokenizer(
        captions,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
        return_attention_mask=True,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Encode all captions in batch
    outputs = text_encoder(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"]).last_hidden_state

    # Split outputs and trim each to true length
    result = []
    for i in range(len(captions)):
        true_len = int(inputs["attention_mask"][i].sum().item())
        result.append(outputs[i, :true_len, :])

    return result


# Fastgen utilities for loading checkpoints
from fastgen.methods.model import FastGenModel
from fastgen.utils import instantiate
from fastgen.utils.checkpointer import Checkpointer
from megatron.bridge.training.model_load_save import build_and_load_model

from dfm.src.megatron.model.wan.flow_matching.flow_inference_pipeline import _select_checkpoint_dir
from dfm.src.megatron.model.wan.inference import SIZE_CONFIGS
from dfm.src.megatron.model.wan.inference.utils import cache_video, str2bool
from dfm.src.megatron.model.wan_dmd.wan_dmd_model_provider import WanDMDCombinedModelProvider, WanDMDModelProvider


# VBench dimensions - each dimension has specific prompts
VBENCH_DIMENSIONS = [
    "subject_consistency",
    "background_consistency",
    "temporal_flickering",
    "motion_smoothness",
    "dynamic_degree",
    "aesthetic_quality",
    "imaging_quality",
    "object_class",
    "multiple_objects",
    "human_action",
    "color",
    "spatial_relationship",
    "scene",
    "appearance_style",
    "temporal_style",
    "overall_consistency",
]


def load_vbench_prompts(dimension=None):
    """
    Load prompts from VBench package.

    Args:
        dimension: If specified, load only prompts for this dimension.
                   If None, load prompts for all dimensions.

    Returns:
        dict: Dictionary mapping dimension -> list of (prompt, video_id) tuples
    """
    try:
        import vbench

        vbench_path = Path(vbench.__file__).parent
        info_file = vbench_path / "VBench_full_info.json"

        if not info_file.exists():
            # Try alternative location
            info_file = vbench_path / "data" / "VBench_full_info.json"

        if not info_file.exists():
            raise FileNotFoundError(
                f"VBench_full_info.json not found in {vbench_path}. Please ensure VBench is properly installed."
            )

        with open(info_file, "r") as f:
            vbench_data = json.load(f)

        # VBench_full_info.json is a list of items with 'dimension' and 'prompt_en' keys
        prompts_dict = {}

        dimensions_to_load = [dimension] if dimension else VBENCH_DIMENSIONS

        # Initialize prompts lists for each dimension
        for dim in dimensions_to_load:
            prompts_dict[dim] = []

        # Parse the VBench data list
        for idx, item in enumerate(vbench_data):
            if not isinstance(item, dict):
                continue

            # Get dimension(s) - can be a list
            item_dimensions = item.get("dimension", [])
            if not isinstance(item_dimensions, list):
                item_dimensions = [item_dimensions]

            # Get prompt
            prompt = item.get("prompt_en", "").strip()
            if not prompt:
                continue

            # Add prompt to all matching dimensions
            for item_dim in item_dimensions:
                if item_dim in dimensions_to_load:
                    # Use the item index as video_id
                    prompts_dict[item_dim].append((prompt, idx))

        # Log results and remove empty dimensions
        prompts_dict_filtered = {}
        for dim in dimensions_to_load:
            if dim in prompts_dict and len(prompts_dict[dim]) > 0:
                prompts_dict_filtered[dim] = prompts_dict[dim]
                logging.info(f"Loaded {len(prompts_dict[dim])} prompts for dimension: {dim}")
            else:
                logging.warning(f"No prompts found for dimension '{dim}'. Skipping.")

        return prompts_dict_filtered

    except ImportError as e:
        logging.error(
            f"VBench could not be imported: {e}\n"
            "Make sure VBench is either:\n"
            "  1. Cloned at /opt/dfm/VBench, or\n"
            "  2. Installed with: pip install vbench"
        )
        sys.exit(1)
    except Exception as e:
        logging.error(f"Error loading VBench prompts: {e}")
        sys.exit(1)


def _validate_args(args):
    """Validate command line arguments."""
    assert args.checkpoint_dir is not None, "checkpoint_dir must be specified"

    if args.sample_steps is None:
        args.sample_steps = 50

    if args.sample_shift is None:
        args.sample_shift = 5.0

    # Validate size
    if args.size not in SIZE_CONFIGS:
        raise ValueError(f"Unsupported size {args.size}. Supported sizes: {', '.join(SIZE_CONFIGS.keys())}")

    # Validate frame count
    if (args.frames - 1) % 4 != 0:
        raise ValueError(f"Frame count must be 4n+1. Got {args.frames}")

    # Validate dimensions
    if args.dimensions:
        for dim in args.dimensions:
            if dim not in VBENCH_DIMENSIONS:
                raise ValueError(f"Unknown dimension: {dim}. Valid dimensions: {', '.join(VBENCH_DIMENSIONS)}")


def _parse_args():
    parser = argparse.ArgumentParser(description="Generate videos for VBench evaluation using WAN model")

    # Model arguments
    parser.add_argument(
        "--task",
        type=str,
        default="t2v-14B",
        help="Task type (t2v-1.3B or t2v-14B)",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        required=True,
        help="Path to the main WAN checkpoint directory",
    )
    parser.add_argument(
        "--checkpoint_step",
        type=int,
        default=None,
        help="Optional training step to load. If not provided, uses latest checkpoint.",
    )
    parser.add_argument(
        "--t5_checkpoint_dir",
        type=str,
        default=None,
        help="Optional directory containing T5 checkpoint/tokenizer",
    )
    parser.add_argument(
        "--vae_checkpoint_dir",
        type=str,
        default=None,
        help="Optional directory containing VAE checkpoint",
    )

    # add seq_length parameter
    parser.add_argument(
        "--seq_length",
        type=int,
        default=33000,
        help="Sequence length for the model",
    )

    # VBench arguments
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./vbench_outputs",
        help="Directory to save generated videos",
    )
    parser.add_argument(
        "--dimensions",
        type=str,
        nargs="+",
        default=None,
        help="Specific VBench dimensions to evaluate. If not specified, evaluates all dimensions.",
    )

    # Generation parameters
    parser.add_argument(
        "--size",
        type=str,
        default="832*480",
        help="Video resolution (WIDTH*HEIGHT)",
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=81,
        help="Number of frames to generate (must be 4n+1)",
    )
    parser.add_argument(
        "--sample_steps",
        type=int,
        default=None,
        help="Number of sampling steps (default: 50)",
    )
    parser.add_argument(
        "--sample_shift",
        type=float,
        default=None,
        help="Sampling shift factor (default: 5.0)",
    )
    parser.add_argument(
        "--sample_guide_scale",
        type=float,
        default=5.0,
        help="Classifier-free guidance scale",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--num_videos_per_prompt",
        type=int,
        default=1,
        help="Number of videos to generate per prompt (for consistency metrics)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Number of prompts to process in parallel (batching for faster generation)",
    )

    # Fastgen student model arguments
    parser.add_argument(
        "--use_student_model",
        action="store_true",
        default=False,
        help="Use fastgen-style student generation with FastGenModel.generator_fn (for distilled models)",
    )
    parser.add_argument(
        "--student_sample_steps",
        type=int,
        default=1,
        help="Number of sampling steps for student model (e.g., 1, 4, 5)",
    )
    parser.add_argument(
        "--student_sample_type",
        type=str,
        default="sde",
        choices=["sde", "ode"],
        help="Type of sampling for student model",
    )
    parser.add_argument(
        "--t_list",
        type=float,
        nargs="+",
        default=None,
        help="Custom timestep list for student sampling (e.g., 1.0 0.5 0.0)",
    )
    parser.add_argument(
        "--load_fastgen_checkpoint",
        action="store_true",
        default=False,
        help="Use fastgen approach: load model from fastgen config and fastgen checkpoint (instead of Megatron-Bridge)",
    )
    parser.add_argument(
        "--fastgen_config",
        type=str,
        default=None,
        help="Path to fastgen config file (required when --load_fastgen_checkpoint is used)",
    )

    # Model configuration
    parser.add_argument(
        "--offload_model",
        default=False,
        action="store_true",
        help="Whether to offload model to CPU between forward passes",
    )
    parser.add_argument(
        "--t5_cpu",
        action="store_true",
        default=False,
        help="Whether to place T5 model on CPU",
    )
    parser.add_argument(
        "--tensor_parallel_size",
        type=int,
        default=1,
        help="Tensor parallel size",
    )
    parser.add_argument(
        "--context_parallel_size",
        type=int,
        default=1,
        help="Context parallel size",
    )
    parser.add_argument(
        "--pipeline_parallel_size",
        type=int,
        default=1,
        help="Pipeline parallel size",
    )
    parser.add_argument(
        "--sequence_parallel",
        type=str2bool,
        default=False,
        help="Sequence parallel",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="Cache directory for downloading models",
    )

    # Evaluation arguments
    parser.add_argument(
        "--run_eval",
        type=str2bool,
        default=True,
        help="Automatically run VBench evaluation after video generation completes (default: True)",
    )
    parser.add_argument(
        "--eval_output_dir",
        type=str,
        default="/opt/artifacts/vbench_results/",
        help="Base directory to save evaluation results. The most nested folder from output_dir will be appended (default: /opt/artifacts/vbench_results/)",
    )

    args = parser.parse_args()
    _validate_args(args)

    return args


def _init_logging(rank):
    """Initialize logging."""
    if rank == 0:
        logging.basicConfig(
            level=logging.INFO,
            format="[%(asctime)s] %(levelname)s: %(message)s",
            handlers=[logging.StreamHandler(stream=sys.stdout)],
        )
    else:
        logging.basicConfig(level=logging.ERROR)


def initialize_distributed(tensor_model_parallel_size=1, pipeline_model_parallel_size=1, context_parallel_size=1):
    """Initialize distributed training."""
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    torch.distributed.init_process_group(backend="nccl")
    ps.initialize_model_parallel(
        tensor_model_parallel_size,
        pipeline_model_parallel_size,
        context_parallel_size=context_parallel_size,
    )


def load_fastgen_model_and_checkpoint(config_path, checkpoint_path, device):
    """
    Load a fastgen model using fastgen config and checkpoint.

    Args:
        config_path: Path to fastgen config file
        checkpoint_path: Path to fastgen checkpoint (.pth file or directory for FSDP)
        device: Device to load model to (int, str, or torch.device)

    Returns:
        Loaded fastgen model
    """
    logging.info("=" * 80)
    logging.info("LOADING FASTGEN MODEL")
    logging.info(f"Config: {config_path}")
    logging.info(f"Checkpoint: {checkpoint_path}")
    logging.info("=" * 80)

    # Convert device to proper format early
    if isinstance(device, int):
        device_str = f"cuda:{device}"
        device = torch.device(device_str)
    elif isinstance(device, str):
        device = torch.device(device)
    elif not isinstance(device, torch.device):
        device = torch.device("cuda")

    # Import config loading utilities
    import importlib.util

    spec = importlib.util.spec_from_file_location("config_module", config_path)
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)
    config = config_module.create_config()

    # Override device in config
    config.model.device = str(device)

    logging.info(f"Loaded config: {config.log_config.name}")
    logging.info(f"  - Model architecture: {config.model.net}")
    logging.info(f"  - Device: {config.model.device}")

    # Initialize model using fastgen approach
    config.model_class.config = config.model
    model = instantiate(config.model_class)
    config.model_class.config = None

    # Initialize checkpointer
    checkpointer = Checkpointer(config.trainer.checkpointer)

    # Load checkpoint
    if checkpoint_path and (os.path.isdir(checkpoint_path + ".net_model") or os.path.isfile(checkpoint_path)):
        logging.info(f"Loading checkpoint from: {checkpoint_path}")

        # Build model dict for loading
        model_dict_infer = torch.nn.ModuleDict({"net": model.net, **model.ema_dict})

        ckpt_iter = checkpointer.load(model_dict_infer, path=checkpoint_path, device=device)
        logging.info(f"Successfully loaded checkpoint iteration {ckpt_iter}")
    else:
        logging.warning(f"No checkpoint found at {checkpoint_path}, using initialized weights")
        ckpt_iter = None

    # Set to eval mode
    model.net.eval()
    model.net.to(device=device)

    logging.info("Fastgen model loaded successfully")
    logging.info("=" * 80)

    return model, config


def generate_student_latents(
    pipeline,
    prompts,
    size_config,
    frame_num,
    seeds,
    student_sample_steps,
    student_sample_type,
    t_list,
    guide_scale,
    offload_model,
):
    """
    Generate video latents using FastGen student model (supports batching).

    This prepares all necessary inputs (noise, text embeddings, packed_seq_params)
    and calls FastGenModel.generator_fn via create_generator_fn.

    Args:
        pipeline: FlowInferencePipeline instance (provides model, VAE, text encoder)
        prompts: List of text prompts
        size_config: Video size configuration (width, height)
        frame_num: Number of frames
        seeds: List of random seeds (one per prompt)
        student_sample_steps: Number of sampling steps
        student_sample_type: Type of sampling
        t_list: Optional custom timestep list
        guide_scale: CFG scale
        offload_model: Whether to offload model

    Returns:
        Generated latents tensor (batch_size, ...)
    """
    device = pipeline.device
    rank = pipeline.rank
    batch_size = len(prompts)

    # Calculate target latent shape
    target_shape = (
        pipeline.vae.config.z_dim,
        (frame_num - 1) // pipeline.vae_stride[0] + 1,
        size_config[1] // pipeline.vae_stride[1],
        size_config[0] // pipeline.vae_stride[2],
    )

    # Generate batched noise with different seeds
    noise_list = []
    for seed in seeds:
        seed_g = torch.Generator(device=device)
        seed_g.manual_seed(seed)
        noise = torch.randn(
            target_shape[0],
            target_shape[1],
            target_shape[2],
            target_shape[3],
            dtype=torch.float32,
            device=device,
            generator=seed_g,
        )
        noise_list.append(noise)

    # Stack noise into batch: [batch_size, C, T, H, W]
    noise_batch = torch.stack(noise_list, dim=0)

    # Encode text in batch (much faster than per-prompt encoding)
    context_max_len = pipeline.text_len

    if not pipeline.t5_cpu:
        pipeline.text_encoder.to(device)
        contexts = _encode_text_batch(
            pipeline.tokenizer, pipeline.text_encoder, device, prompts, max_length=context_max_len
        )
        if offload_model:
            pipeline.text_encoder.cpu()
    else:
        contexts = _encode_text_batch(
            pipeline.tokenizer, pipeline.text_encoder, torch.device("cpu"), prompts, max_length=context_max_len
        )
        contexts = [ctx.to(device) for ctx in contexts]

    # Pad contexts to context_max_len and stack into batch
    contexts_padded = []
    for context in contexts:
        context_padded = F.pad(context, (0, 0, 0, context_max_len - context.shape[0]))
        contexts_padded.append(context_padded)

    # Stack contexts: [seq_len, batch_size, hidden]
    context_batch = torch.stack(contexts_padded, dim=1)

    # Calculate grid_sizes using pipeline helper method
    # Pass list of noise tensors (unbatched) to calculate_grid_sizes
    grid_sizes = pipeline.calculate_grid_sizes(noise_list)

    # Calculate sequence length for packed_seq_params
    seq_len = (
        math.ceil(
            (target_shape[2] * target_shape[3])
            / (pipeline.patch_size[1] * pipeline.patch_size[2])
            * target_shape[1]
            / pipeline.sp_size
        )
        * pipeline.sp_size
    )

    # Create packed_seq_params using pipeline helper method
    if not args.load_fastgen_checkpoint:
        packed_seq_params = pipeline.create_packed_seq_params([seq_len] * batch_size, [context_max_len] * batch_size)
    else:
        packed_seq_params = None

    gen_rand_func = partial(
        FastGenModel.generator_fn,
        net=pipeline.model,
        noise=noise_batch,
        context=context_batch,
        condition=context_batch.transpose(0, 1)
        if args.load_fastgen_checkpoint
        else context_batch,  # [batch_size, seq_len, hidden]
        student_sample_steps=student_sample_steps,
        student_sample_type=student_sample_type,
        t_list=t_list,
        precision_amp=torch.bfloat16,
        grid_sizes=grid_sizes,
        packed_seq_params=packed_seq_params,
        unpatchify_features=True,
        scale_t=True,
    )

    # Call the generator function to generate latents
    with torch.no_grad():
        latents = gen_rand_func()

    return latents


def generate_vbench_videos(args):
    """Main function to generate VBench videos."""

    # Check if running with torchrun/distributed
    use_distributed = (
        "RANK" in os.environ
        or "LOCAL_RANK" in os.environ
        or args.tensor_parallel_size > 1
        or args.context_parallel_size > 1
        or args.pipeline_parallel_size > 1
    )

    if use_distributed:
        # Get the actual world size before initializing model parallel
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)

        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(backend="nccl")

        global_rank = torch.distributed.get_rank()
        global_world_size = torch.distributed.get_world_size()
        device = torch.cuda.current_device()

        # Now initialize model parallelism
        ps.initialize_model_parallel(
            args.tensor_parallel_size,
            args.pipeline_parallel_size,
            context_parallel_size=args.context_parallel_size,
        )

        # After model parallel init, check if we're in data parallel mode
        # If no model parallelism is configured, all GPUs do data parallelism
        total_model_parallel = args.tensor_parallel_size * args.pipeline_parallel_size * args.context_parallel_size

        using_data_parallel = (global_world_size > 1) and (total_model_parallel == 1)

        if using_data_parallel:
            # Pure data parallelism: use global rank and world size
            rank = global_rank
            world_size = global_world_size
            data_parallel_rank = global_rank
            data_parallel_size = global_world_size
        else:
            # Model parallelism: use Megatron's data parallel group
            rank = global_rank
            world_size = global_world_size
            data_parallel_rank = ps.get_data_parallel_rank() if ps.is_initialized() else 0
            data_parallel_size = ps.get_data_parallel_world_size() if ps.is_initialized() else 1
    else:
        rank = 0
        world_size = 1
        device = 0
        using_data_parallel = False
        data_parallel_rank = 0
        data_parallel_size = 1

    _init_logging(rank)

    # Log from all ranks in data parallel, only rank 0 in model parallel
    should_log_init = using_data_parallel or (rank == 0)
    if should_log_init:
        gpu_prefix = f"[GPU {data_parallel_rank}] " if using_data_parallel else ""
        logging.info(f"{gpu_prefix}Distributed setup:")
        logging.info(f"{gpu_prefix}  Global world size: {world_size}")
        logging.info(f"{gpu_prefix}  Global rank: {rank}")
        logging.info(f"{gpu_prefix}  Device: {device}")
        logging.info(f"{gpu_prefix}  Using data parallelism: {using_data_parallel}")
        if using_data_parallel:
            logging.info(f"{gpu_prefix}  Data parallel rank: {data_parallel_rank}/{data_parallel_size}")
            logging.info(f"{gpu_prefix}  This GPU will process every {data_parallel_size}th prompt")
        else:
            logging.info(f"{gpu_prefix}  Tensor parallel size: {args.tensor_parallel_size}")
            logging.info(f"{gpu_prefix}  Pipeline parallel size: {args.pipeline_parallel_size}")
            logging.info(f"{gpu_prefix}  Context parallel size: {args.context_parallel_size}")
            logging.info(f"{gpu_prefix}  Data parallel size: {data_parallel_size}")

    # Load VBench prompts
    if rank == 0:
        logging.info("Loading VBench prompts...")
        if args.dimensions:
            logging.info(f"Loading prompts for dimensions: {', '.join(args.dimensions)}")
        else:
            logging.info("Loading prompts for all VBench dimensions")

    prompts_dict = load_vbench_prompts()

    # Filter dimensions if specified
    if args.dimensions:
        prompts_dict = {dim: prompts for dim, prompts in prompts_dict.items() if dim in args.dimensions}

    if rank == 0:
        total_prompts = sum(len(prompts) for prompts in prompts_dict.values())
        total_videos = total_prompts * args.num_videos_per_prompt
        logging.info(f"Total prompts: {total_prompts}")
        logging.info(f"Total videos to generate: {total_videos}")
        logging.info(f"Batch size: {args.batch_size}")
        if using_data_parallel:
            logging.info(f"Videos per GPU: ~{total_videos // data_parallel_size}")
        logging.info(f"Total batches: {(total_videos + args.batch_size - 1) // args.batch_size}")

    # Setup inference configuration
    inference_cfg = SimpleNamespace(
        # T5
        t5_dtype=torch.bfloat16,
        text_len=512,
        # VAE
        vae_stride=(4, 8, 8),
        # Transformer
        param_dtype=torch.bfloat16,
        patch_size=(1, 2, 2),
        # Others
        num_train_timesteps=1000,
        sample_fps=16,
        chinese_sample_neg_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
        english_sample_neg_prompt="Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards",
    )

    if rank == 0:
        logging.info(f"Generation args: {args}")
        logging.info(f"Inference config: {inference_cfg}")

    # Create inference pipeline
    if rank == 0:
        logging.info("Creating flow inference pipeline...")

    student_model = None
    fastgen_model = None

    assert not (args.load_fastgen_checkpoint and args.use_student_model), "Cannot use both fastgen and student model"

    if args.load_fastgen_checkpoint:
        # === LOAD FASTGEN MODEL AND CHECKPOINT ===
        # Use fastgen's config system to instantiate and load the model
        fastgen_model, fastgen_config = load_fastgen_model_and_checkpoint(
            config_path=args.fastgen_config,
            checkpoint_path=args.checkpoint_dir,
            device=device,
        )
        student_model = fastgen_model.net

    elif args.use_student_model:
        # === LOAD MEGATRON-BRIDGE CHECKPOINT ===
        wan_checkpoint_dir = _select_checkpoint_dir(args.checkpoint_dir, args.checkpoint_step)
        print("checkpoint_dir", wan_checkpoint_dir)
        # Create model provider with proper configuration
        params_dtype = torch.float32
        provider = WanDMDCombinedModelProvider(
            params_dtype=params_dtype,
            # Separate providers for fake_score and teacher
            fake_score_model_provider=WanDMDModelProvider(
                params_dtype=params_dtype,
                seq_length=args.seq_length,
            ),
            teacher_model_provider=WanDMDModelProvider(
                params_dtype=params_dtype,
                seq_length=args.seq_length,
            ),
            # Parallelism config
            tensor_model_parallel_size=args.tensor_parallel_size,
            pipeline_model_parallel_size=args.pipeline_parallel_size,
            pipeline_dtype=params_dtype,
            virtual_pipeline_model_parallel_size=None,
            context_parallel_size=args.context_parallel_size,
            sequence_parallel=args.sequence_parallel,
        )
        provider.fast_gen_config.gan_loss_weight_gen = 0.03
        provider.fast_gen_config.discriminator.disc_type = "multiscale_down_mlp_large"
        provider.fast_gen_config.discriminator.feature_indices = {15, 22, 29}

        # Use build_and_load_model to create model from provider and load weights
        # skip_temp_dist_context=True because distributed is already initialized
        dmd_model = build_and_load_model(
            checkpoint_path=wan_checkpoint_dir,
            model_cfg=provider,
            skip_temp_dist_context=True,
        )

        # Unwrap if needed
        if isinstance(dmd_model, list):
            dmd_model = dmd_model[0]
        if hasattr(dmd_model, "module"):
            dmd_model = dmd_model.module

        student_model = dmd_model.net
        student_model.eval().cuda()

    pipeline = FlowInferencePipeline(
        inference_cfg=inference_cfg,
        checkpoint_dir=args.checkpoint_dir,
        model_id="Wan-AI/Wan2.1-T2V-14B-Diffusers",
        model=student_model,
        checkpoint_step=args.checkpoint_step,
        t5_checkpoint_dir=args.t5_checkpoint_dir,
        vae_checkpoint_dir=args.vae_checkpoint_dir,
        device_id=device,
        rank=rank,
        t5_cpu=args.t5_cpu,
        tensor_parallel_size=args.tensor_parallel_size,
        context_parallel_size=args.context_parallel_size,
        pipeline_parallel_size=args.pipeline_parallel_size,
        sequence_parallel=args.sequence_parallel,
        pipeline_dtype=torch.float32,
        cache_dir=args.cache_dir,
    )

    if rank == 0:
        print(f"Running inference with tensor_parallel_size: {args.tensor_parallel_size}")
        print(f"Running inference with context_parallel_size: {args.context_parallel_size}")
        print(f"Running inference with pipeline_parallel_size: {args.pipeline_parallel_size}")
        print(f"Running inference with sequence_parallel: {args.sequence_parallel}")
        print("")
        if args.load_fastgen_checkpoint or args.use_student_model:
            logging.info("=" * 80)
            logging.info("USING FASTGEN STUDENT GENERATION")
            if args.load_fastgen_checkpoint:
                logging.info("Mode: Fastgen checkpoint (using fastgen config)")
            else:
                logging.info("Mode: Megatron-Bridge checkpoint")
            logging.info(f"Student sample steps: {args.student_sample_steps}")
            logging.info(f"Student sample type: {args.student_sample_type}")
            if args.t_list:
                logging.info(f"Custom t_list: {args.t_list}")
            logging.info("=" * 80)
        else:
            logging.info("=" * 80)
            logging.info("USING TEACHER GENERATION (FlowInferencePipeline)")
            logging.info(f"Sampling steps: {args.sample_steps}")
            logging.info(f"Sampling shift: {args.sample_shift}")
            logging.info("=" * 80)
        print("")

    # Get size configuration
    size_config = SIZE_CONFIGS[args.size]

    # Create output directory
    output_dir = Path(args.output_dir)
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)

    # Synchronize base seed across all ranks
    if dist.is_initialized():
        base_seed = [args.seed] if rank == 0 else [None]
        dist.broadcast_object_list(base_seed, src=0)
        args.seed = base_seed[0]

    # Generate videos for each dimension
    for dimension, prompts_list in prompts_dict.items():
        if rank == 0:
            logging.info(f"\n{'=' * 80}")
            logging.info(f"Generating videos for dimension: {dimension}")
            logging.info(f"Number of prompts: {len(prompts_list)}")
            logging.info(f"{'=' * 80}\n")

        # Create dimension output directory
        dim_output_dir = output_dir / dimension
        if rank == 0:
            dim_output_dir.mkdir(parents=True, exist_ok=True)

        # ========================================================================
        # BATCHED GENERATION WITH MULTI-GPU SUPPORT
        # Process prompts in batches for faster generation
        # Each GPU processes a different subset when using data parallelism
        # ========================================================================

        # Prepare batches of prompts with metadata
        # Each batch item: (prompt_idx, prompt, video_id, video_num, video_seed)
        all_tasks = []
        for prompt_idx, (prompt, video_id) in enumerate(prompts_list):
            for video_num in range(args.num_videos_per_prompt):
                video_seed = args.seed + prompt_idx * args.num_videos_per_prompt + video_num
                all_tasks.append((prompt_idx, prompt, video_id, video_num, video_seed))

        # Distribute tasks across GPUs for data parallelism
        if using_data_parallel:
            # Each GPU gets every Nth task starting at its rank
            gpu_tasks = [task for i, task in enumerate(all_tasks) if i % data_parallel_size == data_parallel_rank]
            logging.info(
                f"[GPU {data_parallel_rank}] Processing {len(gpu_tasks)}/{len(all_tasks)} tasks for dimension '{dimension}'"
            )
        else:
            gpu_tasks = all_tasks
            if rank == 0:
                logging.info(f"Processing {len(gpu_tasks)} tasks for dimension '{dimension}'")

        # Process tasks in batches
        num_batches = (len(gpu_tasks) + args.batch_size - 1) // args.batch_size

        for batch_idx in tqdm(
            range(num_batches),
            desc=f"{dimension} [GPU {data_parallel_rank}]" if using_data_parallel else f"{dimension}",
            disable=(rank != 0 and not using_data_parallel),
        ):
            # Get batch slice from this GPU's tasks
            batch_start = batch_idx * args.batch_size
            batch_end = min(batch_start + args.batch_size, len(gpu_tasks))
            batch_tasks = gpu_tasks[batch_start:batch_end]

            # Unpack batch data
            batch_prompt_idxs = [task[0] for task in batch_tasks]
            batch_prompts = [task[1] for task in batch_tasks]
            batch_video_ids = [task[2] for task in batch_tasks]
            batch_video_nums = [task[3] for task in batch_tasks]
            batch_seeds = [task[4] for task in batch_tasks]

            # Log progress (rank 0 for model parallelism, all ranks for data parallelism)
            should_log = (rank == 0) or using_data_parallel
            print("using data parallelism: ", using_data_parallel)
            print("rank: ", rank)
            print("should_log: ", should_log)
            if should_log:
                gpu_str = f" [GPU {data_parallel_rank}]" if using_data_parallel else ""
                logging.info(f"\nBatch {batch_idx + 1}/{num_batches}{gpu_str} (size: {len(batch_tasks)})")
                for i, (prompt_idx, prompt, video_num, seed) in enumerate(
                    zip(batch_prompt_idxs, batch_prompts, batch_video_nums, batch_seeds)
                ):
                    logging.info(
                        f"  [{i}] Prompt {prompt_idx}/{len(prompts_list)}: '{prompt[:60]}...' (video {video_num}, seed {seed})"
                    )

            # Choose generation path: student (fastgen) or teacher (standard pipeline)
            if args.load_fastgen_checkpoint or args.use_student_model:
                # === FASTGEN STYLE GENERATION (BATCHED) ===
                if should_log:
                    logging.info(
                        f"Using FastGen student generation with {args.student_sample_steps} steps (batch size: {len(batch_tasks)})"
                    )

                # Generate latents using fastgen generator_fn
                with torch.no_grad():
                    latents = generate_student_latents(
                        pipeline=pipeline,
                        prompts=batch_prompts,
                        size_config=size_config,
                        frame_num=args.frames,
                        seeds=batch_seeds,
                        student_sample_steps=args.student_sample_steps,
                        student_sample_type=args.student_sample_type,
                        t_list=args.t_list,
                        guide_scale=args.sample_guide_scale,
                        offload_model=args.offload_model,
                    )

                # Decode latents to video
                # In data parallel mode: each GPU decodes its own latents
                # In model parallel mode: only rank 0 decodes
                should_decode = using_data_parallel or (rank == 0)
                if should_decode and latents is not None:
                    with torch.no_grad():
                        videos = pipeline._decode_latents(latents, sample=False)
                else:
                    videos = None
            else:
                # === TEACHER GENERATION (BATCHED) ===
                if should_log:
                    logging.info(
                        f"Using teacher generation with {args.sample_steps} steps (batch size: {len(batch_tasks)})"
                    )

                # Pre-generate noise with per-sample seeds for proper seed handling
                initial_noises = []
                for seed in batch_seeds:
                    target_shape = (
                        pipeline.vae.config.z_dim,
                        (args.frames - 1) // pipeline.vae_stride[0] + 1,
                        size_config[1] // pipeline.vae_stride[1],
                        size_config[0] // pipeline.vae_stride[2],
                    )
                    seed_g = torch.Generator(device=pipeline.device)
                    seed_g.manual_seed(seed)
                    noise = torch.randn(
                        target_shape[0],
                        target_shape[1],
                        target_shape[2],
                        target_shape[3],
                        dtype=torch.float32,
                        device=pipeline.device,
                        generator=seed_g,
                    )
                    initial_noises.append(noise)

                videos = pipeline.generate(
                    prompts=batch_prompts,
                    sizes=[size_config] * len(batch_tasks),
                    frame_nums=[args.frames] * len(batch_tasks),
                    shift=args.sample_shift,
                    sampling_steps=args.sample_steps,
                    guide_scale=args.sample_guide_scale,
                    seed=-1,  # Disable internal noise generation
                    initial_latents=initial_noises,  # Use pre-generated noise with per-sample seeds
                    offload_model=args.offload_model,
                )

            # Save videos
            # In data parallel mode: each GPU saves its own videos
            # In model parallel mode: only rank 0 saves
            should_save = using_data_parallel or (rank == 0)
            if should_save and videos is not None:
                # videos is a tensor with shape [batch_size, C, T, H, W]
                # Convert to list if it's a tensor
                if isinstance(videos, torch.Tensor):
                    if videos.dim() == 4:
                        # Single video without batch dimension
                        videos = [videos]
                    else:
                        # Batched videos - split into list
                        videos = [videos[i] for i in range(videos.shape[0])]

                for i, (prompt, video_num) in enumerate(zip(batch_prompts, batch_video_nums)):
                    # VBench expects videos named as: {prompt}-{video_num}.mp4
                    # This matches the format expected by mode='vbench_standard'
                    video_filename = f"{prompt}-{video_num}.mp4"
                    video_path = dim_output_dir / video_filename

                    gpu_str = f" [GPU {data_parallel_rank}]" if using_data_parallel else ""
                    logging.info(f"Saving video {i + 1}/{len(batch_tasks)}{gpu_str} to: {video_path}")
                    cache_video(
                        tensor=videos[i][None],  # Add batch dimension back
                        save_file=str(video_path),
                        fps=inference_cfg.sample_fps,
                        nrow=1,
                        normalize=True,
                        value_range=(-1, 1),
                    )

        # Synchronize GPUs after each dimension
        if dist.is_initialized():
            dist.barrier()

        if should_log:
            gpu_str = f" [GPU {data_parallel_rank}]" if using_data_parallel else ""
            logging.info(f"Completed dimension: {dimension}{gpu_str}")

    # Final synchronization
    if dist.is_initialized():
        dist.barrier()

    if rank == 0:
        logging.info("\n" + "=" * 80)
        logging.info("Video generation complete!")
        if using_data_parallel:
            logging.info(f"All {data_parallel_size} GPUs have finished processing")
        logging.info(f"Videos saved to: {output_dir}")
        logging.info("=" * 80)

    # Return output_dir for evaluation
    return output_dir, rank, world_size, args


def run_evaluation(output_dir, eval_output_path, dimensions, save_json, rank, world_size):
    """
    Run VBench evaluation after video generation.

    Args:
        output_dir: Path to generated videos
        eval_output_path: Path to save evaluation results
        dimensions: List of dimensions to evaluate (or None for all)
        save_json: Whether to save detailed JSON results
        rank: Current process rank
        world_size: Total number of processes
    """
    if rank != 0:
        # Only rank 0 coordinates the evaluation subprocess
        return

    logging.info("\n" + "=" * 80)
    logging.info("Starting VBench evaluation...")
    logging.info("=" * 80 + "\n")

    # Build the evaluation command
    eval_script = Path(__file__).parent / "evaluate_vbench.py"

    cmd = [sys.executable, str(eval_script)]
    cmd.extend(["--videos_path", str(output_dir)])
    cmd.extend(["--output_path", str(eval_output_path)])

    if dimensions:
        cmd.extend(["--dimensions"] + dimensions)

    if save_json:
        cmd.append("--save_json")

    # If we were running with multiple GPUs, use torchrun for evaluation too
    if world_size > 1:
        # Use torchrun to launch evaluation with same number of GPUs
        torchrun_cmd = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            f"--nproc_per_node={world_size}",
            str(eval_script),
            "--videos_path",
            str(output_dir),
            "--output_path",
            str(eval_output_path),
        ]
        if dimensions:
            torchrun_cmd.extend(["--dimensions"] + dimensions)
        if save_json:
            torchrun_cmd.append("--save_json")
        cmd = torchrun_cmd

    # Print the command before executing
    cmd_str = " ".join(cmd)
    logging.info("\n" + "-" * 80)
    logging.info("Evaluation command:")
    logging.info(f"  {cmd_str}")
    logging.info("-" * 80 + "\n")
    print(f"\n[Eval Command] {cmd_str}\n", flush=True)

    try:
        result = subprocess.run(cmd, check=True)
        logging.info("\n" + "=" * 80)
        logging.info("VBench evaluation completed successfully!")
        logging.info(f"Results saved to: {eval_output_path}")
        logging.info("=" * 80)
    except subprocess.CalledProcessError as e:
        logging.error(f"Evaluation failed with return code {e.returncode}")
        logging.error("You can run evaluation manually with:")
        logging.error("   python evaluate_vbench.py \\")
        logging.error(f"       --videos_path {output_dir} \\")
        logging.error(f"       --output_path {eval_output_path}")
    except Exception as e:
        logging.error(f"Failed to run evaluation: {e}")
        logging.error("You can run evaluation manually with:")
        logging.error("   python evaluate_vbench.py \\")
        logging.error(f"       --videos_path {output_dir} \\")
        logging.error(f"       --output_path {eval_output_path}")


if __name__ == "__main__":
    args = _parse_args()
    output_dir, rank, world_size, args = generate_vbench_videos(args)

    if args.run_eval:
        # Append the most nested folder from output_dir to eval_output_dir
        output_folder_name = Path(output_dir).name
        eval_output_path = str(Path(args.eval_output_dir) / output_folder_name)

        run_evaluation(
            output_dir=output_dir,
            eval_output_path=eval_output_path,
            dimensions=args.dimensions,
            save_json=True,
            rank=rank,
            world_size=world_size,
        )
    elif rank == 0:
        logging.info("\nNext steps:")
        logging.info("1. Run evaluation:")
        logging.info("   python evaluate_vbench.py \\")
        logging.info(f"       --videos_path {output_dir} \\")
        logging.info("       --output_path ./vbench_results")
        logging.info("\n2. Visualize results:")
        logging.info("   python visualize_results.py \\")
        logging.info("       --results_path ./vbench_results \\")
        logging.info("       --output_dir ./vbench_plots")
        logging.info("\nTip: Use --run_eval flag to automatically run evaluation after generation.")
