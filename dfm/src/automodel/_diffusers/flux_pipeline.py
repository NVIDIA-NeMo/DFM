# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
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
NeMo Flux Pipeline - Wrapper for FLUX model loading with FSDP2 support.

This module provides a Flux-specific pipeline wrapper that handles:
- Loading from pretrained weights or config (random init)
- FSDP2 parallelization for distributed training
- Gradient checkpointing for memory efficiency
"""

import copy
import logging
from typing import Any, Dict, Iterable, Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def _choose_device(device: Optional[torch.device] = None) -> torch.device:
    """Choose device, defaulting to CUDA if available."""
    if device is not None:
        return device
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _ensure_params_trainable(module: nn.Module, module_name: Optional[str] = None) -> int:
    """Ensure all parameters in module are trainable."""
    num_trainable = 0
    for param in module.parameters():
        param.requires_grad = True
        num_trainable += param.numel()
    logger.info(f"[INFO] Made {num_trainable:,} parameters trainable in {module_name}")
    return num_trainable


class NeMoFluxPipeline:
    """
    Flux pipeline wrapper that handles:
    - Loading from pretrained or config (random init)
    - FSDP2 parallelization
    - Gradient checkpointing

    This class provides a consistent interface for loading FLUX models
    that is compatible with TrainDiffusionRecipe.
    """

    def __init__(self, transformer, **kwargs):
        """
        Initialize NeMoFluxPipeline.

        Args:
            transformer: The FluxTransformer2DModel instance
            **kwargs: Additional attributes to set on the pipeline
        """
        self.transformer = transformer
        for k, v in kwargs.items():
            setattr(self, k, v)

    @classmethod
    def from_pretrained(
        cls,
        model_id: str,
        torch_dtype: torch.dtype = torch.bfloat16,
        device: Optional[torch.device] = None,
        parallel_scheme: Optional[Dict[str, Dict[str, Any]]] = None,
        components_to_load: Optional[Iterable[str]] = None,
        load_for_training: bool = False,
        low_cpu_mem_usage: bool = True,
        **kwargs,
    ):
        """
        Load Flux model from pretrained weights.

        Args:
            model_id: HuggingFace model ID or local path
            torch_dtype: Data type for model parameters
            device: Device to load model to
            parallel_scheme: Dict mapping component names to FSDP2Manager kwargs
            components_to_load: Which components to load (default: ["transformer"])
            load_for_training: Whether to make parameters trainable
            low_cpu_mem_usage: Use low CPU memory mode during loading
            **kwargs: Additional arguments passed to from_pretrained

        Returns:
            Tuple of (NeMoFluxPipeline, Dict[str, FSDP2Manager])
        """
        from diffusers import FluxTransformer2DModel
        from nemo_automodel.components.distributed.fsdp2 import FSDP2Manager
        from torch.distributed.fsdp import MixedPrecisionPolicy

        logger.info(f"[INFO] Loading Flux transformer from {model_id}")

        # Load transformer from pretrained
        transformer = FluxTransformer2DModel.from_pretrained(
            model_id,
            subfolder="transformer",
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=low_cpu_mem_usage,
        )

        dev = _choose_device(device)
        transformer = transformer.to(dev)

        # Enable gradient checkpointing for memory efficiency
        if hasattr(transformer, "enable_gradient_checkpointing"):
            transformer.enable_gradient_checkpointing()
            logger.info("[INFO] Enabled gradient checkpointing for Flux transformer")

        # Make parameters trainable if requested
        if load_for_training:
            _ensure_params_trainable(transformer, "transformer")

        # Apply FSDP2 if requested
        created_managers: Dict[str, FSDP2Manager] = {}
        if parallel_scheme is not None and "transformer" in parallel_scheme:
            manager_args = parallel_scheme["transformer"]
            logger.info(f"[INFO] Applying FSDP2 to transformer with args: {manager_args}")
            manager = FSDP2Manager(**manager_args)
            transformer = manager.parallelize(transformer)
            created_managers["transformer"] = manager

        pipe = cls(transformer=transformer)
        return pipe, created_managers

    @classmethod
    def from_config(
        cls,
        model_id: str,
        torch_dtype: torch.dtype = torch.bfloat16,
        device: Optional[torch.device] = None,
        parallel_scheme: Optional[Dict[str, Dict[str, Any]]] = None,
        components_to_load: Optional[Iterable[str]] = None,
        **kwargs,
    ):
        """
        Initialize Flux with random weights from config (for pretraining).

        Args:
            model_id: HuggingFace model ID or local path (for config only)
            torch_dtype: Data type for model parameters
            device: Device to load model to
            parallel_scheme: Dict mapping component names to FSDP2Manager kwargs
            components_to_load: Which components to load (default: ["transformer"])
            **kwargs: Additional arguments

        Returns:
            Tuple of (NeMoFluxPipeline, Dict[str, FSDP2Manager])
        """
        from diffusers import FluxTransformer2DModel
        from nemo_automodel.components.distributed.fsdp2 import FSDP2Manager
        from torch.distributed.fsdp import MixedPrecisionPolicy

        logger.info(f"[INFO] Initializing Flux transformer from config (random weights): {model_id}")

        # Load config only
        config = FluxTransformer2DModel.load_config(model_id, subfolder="transformer")

        # Create model with random weights
        transformer = FluxTransformer2DModel.from_config(config)
        transformer = transformer.to(torch_dtype)

        dev = _choose_device(device)
        transformer = transformer.to(dev)

        # Enable gradient checkpointing for memory efficiency
        # if hasattr(transformer, "enable_gradient_checkpointing"):
        #     transformer.enable_gradient_checkpointing()
        #     logger.info("[INFO] Enabled gradient checkpointing for Flux transformer")

        # Make parameters trainable (always true for pretraining)
        _ensure_params_trainable(transformer, "transformer")

        # Apply FSDP2 if requested
        created_managers: Dict[str, FSDP2Manager] = {}
        if parallel_scheme is not None and "transformer" in parallel_scheme:
            manager_args = parallel_scheme["transformer"]
            logger.info(f"[INFO] Applying FSDP2 to transformer with args: {manager_args}")
            manager = FSDP2Manager(**manager_args)
            transformer = manager.parallelize(transformer)
            created_managers["transformer"] = manager

        pipe = cls(transformer=transformer)
        return pipe, created_managers
