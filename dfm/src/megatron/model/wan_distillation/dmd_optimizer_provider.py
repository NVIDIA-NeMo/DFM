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

import math
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

import torch
from megatron.bridge.training.config import OptimizerConfig
from megatron.core.optimizer import OptimizerConfig, get_megatron_optimizer
from megatron.core.optimizer.clip_grads import clip_grad_by_total_norm_fp32, count_zeros_fp32, get_grad_norm_fp32
from megatron.core.optimizer.optimizer import ChainedOptimizer, MegatronOptimizer
from megatron.core.transformer.module import MegatronModule


class ConditionalChainedOptimizer(ChainedOptimizer):
    """ConditionalChainedOptimizer extends ChainedOptimizer with conditional execution.

    Each optimizer is executed only if its associated condition is met during training.
    This is useful for dynamic optimization strategies like alternating updates or
    conditional gradient updates based on training state.

    Args:
        conditional_optimizers: a list of tuples (optimizer, condition), where:
            - optimizer: a MegatronOptimizer instance
            - condition: a callable that takes iteration (int) and returns True if the optimizer should be executed
    """

    def __init__(self, conditional_optimizers: List[Tuple[MegatronOptimizer, Callable[[int], bool]]]):
        """Initialize with list of (optimizer, condition) tuples."""
        # Store conditions separately
        self.conditions = [condition for _, condition in conditional_optimizers]

        # Extract optimizers and call parent constructor
        optimizers = [opt for opt, _ in conditional_optimizers]
        super().__init__(optimizers)

        # Track current iteration for conditional execution
        self.iteration = 0

    def set_iteration(self, iteration: int):
        """Set the current training iteration.

        This should be called by the training loop before optimizer step to enable
        iteration-dependent conditional execution.

        Args:
            iteration: Current training iteration number
        """
        self.iteration = iteration

    def _should_execute(self, idx: int) -> bool:
        """Check if the optimizer at index idx should be executed.

        Args:
            idx: Index of the optimizer to check

        Returns:
            True if the optimizer should be executed, False otherwise
        """
        return self.conditions[idx](self.iteration)

    def zero_grad(self, set_to_none=True):
        """Zero gradients only for optimizers whose conditions are met."""
        for idx, optimizer in enumerate(self.chained_optimizers):
            if self._should_execute(idx):
                optimizer.zero_grad(set_to_none)

    @torch.no_grad()
    def prepare_grads(self) -> bool:
        """Pre-process gradients only for optimizers whose conditions are met."""
        found_inf_flag = False
        for idx, optimizer in enumerate(self.chained_optimizers):
            if self._should_execute(idx):
                found_inf_flag |= optimizer.prepare_grads()
        return found_inf_flag

    @torch.no_grad()
    def step_with_ready_grads(self) -> bool:
        """Step only optimizers whose conditions are met."""
        success = True
        for optimizer_idx, optimizer in enumerate(self.chained_optimizers):
            if self._should_execute(optimizer_idx):
                print(f"stepping optimizer {optimizer_idx}")
                success &= optimizer.step_with_ready_grads()
                if self.config.overlap_param_gather_with_optimizer_step and optimizer_idx == 0:
                    assert success
                    assert len(optimizer.model_chunks) == 1
                    optimizer.model_chunks[0].start_param_sync(force_dispatch=True)
        return success

    @torch.no_grad()
    def get_grad_norm(self):
        """Get gradient norm, only considering optimizers whose conditions are met."""
        if len(self.chained_optimizers) == 1:
            if self._should_execute(0):
                return self.chained_optimizers[0].get_grad_norm()
            else:
                return 0.0

        if self.grads_states_parallel_group_is_shared():
            grads_for_norm = []
            for idx, optimizer in enumerate(self.chained_optimizers):
                if self._should_execute(idx):
                    grads_for_norm += optimizer.get_main_grads_for_grad_norm()
            if grads_for_norm:
                grad_norm = get_grad_norm_fp32(
                    grads_for_norm, grad_stats_parallel_group=self.get_grad_stats_parallel_group()
                )
            else:
                grad_norm = 0.0
        else:
            grad_norms = []
            for idx, optimizer in enumerate(self.chained_optimizers):
                if self._should_execute(idx):
                    _grad_norm = optimizer.get_grad_norm()
                    grad_norms += [_grad_norm if _grad_norm else 0.0]
            grad_norm = math.sqrt(sum([x**2 for x in grad_norms]))
        return grad_norm

    @torch.no_grad()
    def count_zeros(self):
        """Count zeros only in gradients of optimizers whose conditions are met."""
        if self.grads_states_parallel_group_is_shared():
            params = []
            for idx, optimizer in enumerate(self.chained_optimizers):
                if self._should_execute(idx):
                    params += optimizer.get_parameters()
            if params:
                return count_zeros_fp32(
                    params,
                    grad_stats_parallel_group=self.get_grad_stats_parallel_group(),
                    use_decoupled_grad=self.config.use_precision_aware_optimizer_no_fp8_or_ds_fp8,
                )
            else:
                return 0
        else:
            num_zeros_in_grad = 0
            for idx, optimizer in enumerate(self.chained_optimizers):
                if self._should_execute(idx):
                    num_zeros_in_grad += optimizer.count_zeros() if optimizer.config.log_num_zeros_in_grad else 0
            return num_zeros_in_grad

    @torch.no_grad()
    def step(self):
        """Step only optimizers whose conditions are met."""
        found_inf_flag = self.prepare_grads()
        if found_inf_flag:
            return False, None, None

        grad_norm = self.get_grad_norm()

        # Clip gradients only for optimizers whose conditions are met
        for idx, optimizer in enumerate(self.chained_optimizers):
            if not self._should_execute(idx):
                continue
            if hasattr(optimizer, "is_stub_optimizer") and optimizer.is_stub_optimizer:
                continue
            parameters = optimizer.get_parameters()
            if len(parameters) == 0:
                continue
            if optimizer.config.clip_grad > 0.0:
                clip_grad_by_total_norm_fp32(
                    parameters,
                    max_norm=optimizer.config.clip_grad,
                    total_norm=grad_norm,
                    use_decoupled_grad=(optimizer.config.use_precision_aware_optimizer_no_fp8_or_ds_fp8),
                )

        # Count the zeros in the grads
        num_zeros_in_grad = self.count_zeros() if self.config.log_num_zeros_in_grad else None

        update_successful = self.step_with_ready_grads()

        return update_successful, grad_norm, num_zeros_in_grad

    def _synchronize_steps(self):
        """
        Override to skip step synchronization.
        In ConditionalChainedOptimizer, different optimizers may be executed
        at different frequencies, so their steps are expected to diverge.
        No synchronization is needed or desired.
        """
        # No synchronization - each optimizer maintains its own independent step count
        return None


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
