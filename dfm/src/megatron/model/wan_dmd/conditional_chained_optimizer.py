import math
from typing import Callable, List, Tuple

import torch
from megatron.core.optimizer.clip_grads import clip_grad_by_total_norm_fp32, count_zeros_fp32, get_grad_norm_fp32
from megatron.core.optimizer.optimizer import ChainedOptimizer, MegatronOptimizer


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
        self.iteration = iteration + 1

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
