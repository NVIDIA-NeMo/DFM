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

import os
from typing import List, Optional, Union

import torch
from megatron.bridge.recipes.utils.optimizer_utils import distributed_fused_adam_with_cosine_annealing
from megatron.bridge.recipes.utils.tokenizer_utils import DEFAULT_NULL_TOKENIZER_VOCAB_SIZE
from megatron.bridge.training.comm_overlap import CommOverlapConfig
from megatron.bridge.training.config import (
    CheckpointConfig,
    ConfigContainer,
    LoggerConfig,
    RNGConfig,
    TokenizerConfig,
    TrainingConfig,
    DistributedInitConfig,
)
from megatron.bridge.training.mixed_precision import MixedPrecisionConfig, get_mixed_precision_config
from megatron.core.distributed import DistributedDataParallelConfig

from dfm.src.megatron.data.reve.reve_mock_datamodule import ReveMockDataModuleConfig
from dfm.src.megatron.model.reve.reve_provider import ReveModelProvider, ReveSmallModelProvider, ReveFullModelProvider, ReveHalfFullModelProvider, Reve1BModelProvider


def model_config(
    model_size: str = "small",
    tensor_parallelism: int = 1,
    pipeline_parallelism: int = 1,
    pipeline_parallelism_dtype: Optional[torch.dtype] = torch.bfloat16,
    virtual_pipeline_parallelism: Optional[int] = 1,
    context_parallelism: int = 1,
    sequence_parallelism: bool = False,
    seq_length: int = 1024,
) -> ReveModelProvider:
    """
    Configure the Reve model.

    Args:
        tensor_parallelism (int): Degree of tensor model parallelism.
        pipeline_parallelism (int): Degree of pipeline model parallelism.
        pipeline_parallelism_dtype (Optional[torch.dtype]): Data type for pipeline parallelism.
        virtual_pipeline_parallelism (Optional[int]): Size of virtual pipeline parallelism.
        context_parallelism (int): Degree of context parallelism.
        sequence_parallelism (bool): Whether to use sequence parallelism.
        seq_length (int): Sequence length for the model.
    Returns:
        ReveModelProvider: Configuration for the Reve model.
    """
    if model_size == "small":
        return ReveSmallModelProvider(
            tensor_model_parallel_size=tensor_parallelism,
            pipeline_model_parallel_size=pipeline_parallelism,
            pipeline_dtype=pipeline_parallelism_dtype,
            virtual_pipeline_model_parallel_size=None,
            context_parallel_size=context_parallelism,
            sequence_parallel=sequence_parallelism,
            seq_length=seq_length,
        )
    
    elif model_size == "full":
        return ReveFullModelProvider(
            tensor_model_parallel_size=tensor_parallelism,
            pipeline_model_parallel_size=pipeline_parallelism,
            pipeline_dtype=pipeline_parallelism_dtype,
            virtual_pipeline_model_parallel_size=None,
            context_parallel_size=context_parallelism,
            sequence_parallel=sequence_parallelism,
            seq_length=seq_length,
        )
    elif model_size == "half_full":
        return ReveHalfFullModelProvider(
            tensor_model_parallel_size=tensor_parallelism,
            pipeline_model_parallel_size=pipeline_parallelism,
            pipeline_dtype=pipeline_parallelism_dtype,
            virtual_pipeline_model_parallel_size=None,
            context_parallel_size=context_parallelism,
            sequence_parallel=sequence_parallelism,
            seq_length=seq_length,
        )
    elif model_size == "1b":
        return Reve1BModelProvider(
            tensor_model_parallel_size=tensor_parallelism,
            pipeline_model_parallel_size=pipeline_parallelism,
            pipeline_dtype=pipeline_parallelism_dtype,
            virtual_pipeline_model_parallel_size=None,
            context_parallel_size=context_parallelism,
            sequence_parallel=sequence_parallelism,
            seq_length=seq_length,
        )
    else:
        assert False, "Invalid model size"

def pretrain_config(
    dir: Optional[str] = None,
    name: str = "default",
    # Dataset configuration
    data_paths: Optional[List[str]] = None,
    data_args_path: Optional[str] = None,
    train_data_path: Optional[List[str]] = None,
    valid_data_path: Optional[List[str]] = None,
    test_data_path: Optional[List[str]] = None,
    per_split_data_args_path: Optional[str] = None,
    mock: bool = False,
    model_size: str = "small",
    # Model configuration
    tensor_parallelism: int = 1,
    pipeline_parallelism: int = 1,
    pipeline_parallelism_dtype: Optional[torch.dtype] = torch.bfloat16,
    virtual_pipeline_parallelism: Optional[int] = 1,
    context_parallelism: int = 1,
    sequence_parallelism: bool = False,
    # DEBUGGING (use FSDP)
    use_megatron_fsdp: bool = False,
    use_torch_fsdp2: bool = False,
    # Training hyperparameters
    train_iters: int = 10000,
    global_batch_size: int = 4,
    micro_batch_size: int = 1,
    lr: float = 0.9e-4,
    lr_warmup_iters: int = 2000,
    # Precision recipe
    precision_config: Optional[Union[MixedPrecisionConfig, str]] = "bf16_mixed",
    comm_overlap_config: Optional[CommOverlapConfig] = None,
) -> ConfigContainer:
    """
    Create a pre-training configuration for GPT3 175B model.

    The default configuration is expected to run on 64 nodes with 8 GPUs each.

    Args:
        dir (Optional[str]): Base directory for saving logs and checkpoints.
        name (str): Name of the pre-training run.
        data_paths (Optional[List[str]]): List of paths to dataset files. If None, mock data will be used.
        data_args_path (Optional[str]): Path to file containing data arguments.
        train_data_path (Optional[List[str]]): List of training data paths.
        valid_data_path (Optional[List[str]]): List of validation data paths.
        test_data_path (Optional[List[str]]): List of test data paths.
        per_split_data_args_path (Optional[str]): Path to JSON file with per-split data configuration.
        mock (bool): Whether to use mock data. If True, ignores data_paths.
        tensor_parallelism (int): Degree of tensor model parallelism.
        pipeline_parallelism (int): Degree of pipeline model parallelism.
        pipeline_parallelism_dtype (Optional[torch.dtype]): Data type for pipeline parallelism.
        virtual_pipeline_parallelism (Optional[int]): Size of virtual pipeline parallelism.
        context_parallelism (int): Degree of context parallelism to be passed to model_config.
        sequence_parallelism (bool): Whether to use sequence parallelism.
        train_iters (int): Total number of training iterations.
        global_batch_size (int): Global batch size for training.
        micro_batch_size (int): Micro batch size for training.
        seq_length (int): Sequence length for training data.
        lr (float): Learning rate.
        min_lr (float): Minimum learning rate for cosine decay.
        lr_warmup_iters (int): Number of warmup iterations for the learning rate.
        precision_config (Optional[Union[MixedPrecisionConfig, str]]): Precision configuration for the model.
        comm_overlap_config (Optional[CommOverlapConfig]): Communication overlap configuration for the model.

    Returns:
        ConfigContainer: Configuration for pre-training.
    """
    base_output_dir = dir if dir is not None else os.path.join(os.getcwd(), "nemo_experiments")
    run_output_dir = os.path.join(base_output_dir, name)
    checkpoint_dir = os.path.join(run_output_dir, "checkpoints")
    tensorboard_dir = os.path.join(run_output_dir, "tb_logs")

    model_cfg = model_config(
        model_size=model_size,
        tensor_parallelism=tensor_parallelism,
        pipeline_parallelism=pipeline_parallelism,
        pipeline_parallelism_dtype=pipeline_parallelism_dtype,
        virtual_pipeline_parallelism=virtual_pipeline_parallelism,
        context_parallelism=context_parallelism,
        sequence_parallelism=sequence_parallelism,
        seq_length=1024,
    )

    opt_config, scheduler = distributed_fused_adam_with_cosine_annealing(
        lr_warmup_iters=lr_warmup_iters,
        lr_decay_iters=train_iters,
        max_lr=lr,
    )
    opt_config.use_precision_aware_optimizer = False

    if isinstance(precision_config, str):
        precision_config = get_mixed_precision_config(precision_config)

    precision_config.grad_reduce_in_fp32 = False

    if mock:
        if model_size == "small":
            in_channels = 16
            context_embeddings_dim = 128
            F_latents=1
            H_latents=16
            W_latents=16
            context_seq_len=256
            number_packed_samples=1
        elif model_size == "full" or model_size == "half_full" or model_size == "1b":
            in_channels = 768
            context_embeddings_dim = 4096

            # ## config 1
            # F_latents=1
            # H_latents=8
            # W_latents=8
            # context_seq_len=16
            # number_packed_samples=2

            # ## config 2
            # F_latents=1
            # H_latents=32
            # W_latents=32
            # context_seq_len=256
            # number_packed_samples=8

            ## config 3 (FA3 test)
            F_latents=1
            H_latents=16
            W_latents=16
            context_seq_len=128
            number_packed_samples=72

            # ## config 4 (FSDP test)
            # F_latents=1
            # H_latents=16
            # W_latents=16
            # context_seq_len=128
            # number_packed_samples=8

            # ## testing config
            # F_latents=1
            # H_latents=160
            # W_latents=160
            # context_seq_len=128
            # number_packed_samples=1

        else:
            assert False, "Invalid model size"

        dataset = ReveMockDataModuleConfig(
            path=None,
            seq_length=1024,  # we don't need to use this value, just add because Bridge training requires for LLMs
            in_channels=in_channels,
            F_latents=F_latents,
            H_latents=H_latents,
            W_latents=W_latents,
            patch_spatial=1,
            patch_temporal=1,
            number_packed_samples=number_packed_samples,
            context_seq_len=context_seq_len,
            context_embeddings_dim=context_embeddings_dim,
            micro_batch_size=micro_batch_size,
            global_batch_size=global_batch_size,
            num_workers=0,
            packing_buffer_size=None,
        )
    else:
        assert False, "Reve data module is not implemented yet"

    # Config Container
    cfg = ConfigContainer(
        model=model_cfg,
        train=TrainingConfig(
            train_iters=train_iters,
            eval_interval=2000,
            eval_iters=32,
            global_batch_size=global_batch_size,
            micro_batch_size=micro_batch_size,
            manual_gc=True,
            manual_gc_interval=100,
            manual_gc_eval=100,
        ),
        optimizer=opt_config,
        scheduler=scheduler,
        ddp=DistributedDataParallelConfig(
            check_for_nan_in_grad=True,
            grad_reduce_in_fp32=True,
            # DEBUGGING (use FSDP)
            # overlap_grad_reduce=False,
            # overlap_param_gather=False,
            overlap_grad_reduce=True if use_megatron_fsdp else False,
            overlap_param_gather=True if use_megatron_fsdp else False,
            average_in_collective=True,
            use_distributed_optimizer=True,
            # DEBUGGING (use FSDP)
            use_megatron_fsdp=use_megatron_fsdp,  # need use_distributed_optimizer=True
            data_parallel_sharding_strategy="optim_grads_params",
        ),
        # DEBUGGING (use FSDP)
        dist=DistributedInitConfig(use_megatron_fsdp=use_megatron_fsdp, use_torch_fsdp2=use_torch_fsdp2),
        dataset=dataset,
        logger=LoggerConfig(
            log_interval=10,
            tensorboard_dir=tensorboard_dir,
            log_timers_to_tensorboard=True,
        ),
        tokenizer=TokenizerConfig(tokenizer_type="NullTokenizer", vocab_size=DEFAULT_NULL_TOKENIZER_VOCAB_SIZE),
        checkpoint=CheckpointConfig(
            save_interval=2000,
            save=checkpoint_dir,
            load=checkpoint_dir,
            # DEBUGGING (use FSDP)
            ckpt_format="fsdp_dtensor" if use_megatron_fsdp else "torch_dist",
            fully_parallel_save=True,
        ),
        rng=RNGConfig(seed=1234),
        comm_overlap=comm_overlap_config,
        mixed_precision=precision_config,
    )

    return cfg
