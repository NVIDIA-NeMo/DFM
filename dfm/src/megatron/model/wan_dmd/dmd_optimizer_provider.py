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

"""DMD Optimizer Provider with ConditionalChainedOptimizer.

This module provides an optimizer for DMD (Distribution Matching Distillation) training
that uses a ConditionalChainedOptimizer to separately optimize three different components:
1. fake_score - the fake score model used for distillation
2. student (net) - the main student model being distilled
3. discriminator - the discriminator for GAN-based training (if enabled)

Each component has its own optimizer that can be conditionally executed based on training
iteration or other training state.
"""

from dataclasses import dataclass, field
from typing import Callable, List, Optional

from megatron.bridge.training.config import OptimizerConfig
from megatron.core.optimizer import OptimizerConfig, get_megatron_optimizer
from megatron.core.optimizer.optimizer import MegatronOptimizer
from megatron.core.transformer.module import MegatronModule

from .conditional_chained_optimizer import ConditionalChainedOptimizer


@dataclass
class DMDOptimizerProvider(OptimizerConfig):
    """Optimizer provider for DMD training with separate optimizers for each component.

    This provider creates a ConditionalChainedOptimizer that contains three separate
    optimizers:
    - fake_score_optimizer: Optimizes the fake_score model parameters
    - student_optimizer: Optimizes the student (net) model parameters
    - discriminator_optimizer: Optimizes the discriminator parameters (if present)

    Each optimizer is conditionally executed based on the provided condition functions,
    allowing for flexible training strategies like alternating updates.

    Args:
        fake_score_condition: Callable that takes iteration (int) and returns True when fake_score should be optimized
        student_condition: Callable that takes iteration (int) and returns True when student should be optimized
        discriminator_condition: Callable that takes iteration (int) and returns True when discriminator should be optimized
        fake_score_lr_mult: Learning rate multiplier for fake_score optimizer (default: 1.0)
        student_lr_mult: Learning rate multiplier for student optimizer (default: 1.0)
        discriminator_lr_mult: Learning rate multiplier for discriminator optimizer (default: 1.0)
    """

    # DMD-specific fields
    fake_score_condition: Optional[Callable] = field(default=None)
    student_condition: Optional[Callable] = field(default=None)
    discriminator_condition: Optional[Callable] = field(default=None)
    fake_score_lr_mult: float = field(default=1.0)
    student_lr_mult: float = field(default=1.0)
    discriminator_lr_mult: float = field(default=1.0)

    def __post_init__(self):
        """Set default conditions if not provided."""
        # Call parent's __post_init__ to ensure OptimizerConfig validation runs
        super().__post_init__()

        # Default: always optimize all components (iteration parameter is ignored)
        if self.fake_score_condition is None:
            self.fake_score_condition = lambda iteration: True
        if self.student_condition is None:
            self.student_condition = lambda iteration: True
        if self.discriminator_condition is None:
            self.discriminator_condition = lambda iteration: True

    def provide(
        self,
        model: List[MegatronModule],
        no_weight_decay_cond: Optional[Callable] = None,
        scale_lr_cond: Optional[Callable] = None,
        lr_mult: float = 1.0,
        use_gloo_process_groups: bool = False,
    ) -> MegatronOptimizer:
        """Create a ConditionalChainedOptimizer with separate optimizers for each DMD component.

        This method creates three separate optimizers:
        1. Fake Score Optimizer - for parameters in fake_score model
        2. Student Optimizer - for parameters in net (student) model
        3. Discriminator Optimizer - for parameters in discriminator model (if exists)

        Args:
            config: Optimizer configuration containing learning rate, weight decay, etc.
            model: List of model chunks (should be DMDDistillModel instances)
            no_wd_decay_cond: Function to determine if a parameter should skip weight decay
            scale_lr_cond: Function to determine if a parameter should have scaled learning rate
            lr_mult: Base learning rate multiplier
            use_gloo_process_groups: Whether to use Gloo process groups for distributed optimization
            default_skip_embedding_weight_decay: Skip weight decay for embeddings by default
            dump_param_to_param_group_map: Path to dump parameter-to-group mapping for debugging

        Returns:
            ConditionalChainedOptimizer containing three conditional optimizers
        """
        print("[INFO] Creating DMD ConditionalChainedOptimizer with separate optimizers")

        # Wrapper class that filters parameters by component name
        class FilteredModelChunk:
            """Wrapper that filters a model chunk's parameters by component name.

            This wrapper delegates all attribute access to the original chunk while
            overriding named_parameters() to filter by component name.
            """

            def __init__(self, original_chunk, component_name):
                self._original_chunk = original_chunk
                self._component_name = component_name

            def named_parameters(self, *args, **kwargs):
                """Yield only parameters belonging to the target component."""
                for name, param in self._original_chunk.named_parameters(*args, **kwargs):
                    if self._component_name in name:
                        yield name, param

            def __getattr__(self, name):
                """Delegate all other attribute access to the original chunk."""
                if name in ["_original_chunk", "_component_name"]:
                    return object.__getattribute__(self, name)
                return getattr(self._original_chunk, name)

        # Collect filtered parameters for each component
        fake_score_chunks = [FilteredModelChunk(chunk, "fake_score") for chunk in model]
        student_chunks = [FilteredModelChunk(chunk, "net") for chunk in model]
        discriminator_chunks = [FilteredModelChunk(chunk, "discriminator") for chunk in model]

        # Check if discriminator exists in the model
        has_discriminator = False
        for chunk in model:
            for name, param in chunk.named_parameters():
                if "discriminator" in name and param.requires_grad:
                    has_discriminator = True
                    break
            if has_discriminator:
                break

        # Define optimizer components configuration
        optimizer_components = [
            {
                "name": "fake_score",
                "chunks": fake_score_chunks,
                "lr_mult": self.fake_score_lr_mult,
                "condition": self.fake_score_condition,
                "enabled": True,
            },
            {
                "name": "student (net)",
                "chunks": student_chunks,
                "lr_mult": self.student_lr_mult,
                "condition": self.student_condition,
                "enabled": True,
            },
            {
                "name": "discriminator",
                "chunks": discriminator_chunks,
                "lr_mult": self.discriminator_lr_mult,
                "condition": self.discriminator_condition,
                "enabled": has_discriminator,
            },
        ]

        # Print configuration
        for component in optimizer_components:
            if component["enabled"]:
                print(f"[INFO] {component['name']} LR multiplier: {component['lr_mult']}")

        # Create optimizers for each enabled component
        conditional_optimizers = []
        for component in optimizer_components:
            if not component["enabled"]:
                print(f"[INFO] Skipping {component['name']} optimizer (not enabled)")
                continue

            print(f"[INFO] Creating optimizer for {component['name']}")
            optimizer = get_megatron_optimizer(
                config=self,
                model_chunks=component["chunks"],
                no_weight_decay_cond=no_weight_decay_cond,
                scale_lr_cond=scale_lr_cond,
                lr_mult=component["lr_mult"] * lr_mult,
                use_gloo_process_groups=use_gloo_process_groups,
            )
            conditional_optimizers.append((optimizer, component["condition"]))

        return ConditionalChainedOptimizer(conditional_optimizers)
