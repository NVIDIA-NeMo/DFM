# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.


import math
from typing import Optional, Tuple

import torch
from torch import Tensor
from torch.nn.attention.flex_attention import flex_attention

from megatron.core import parallel_state, tensor_parallel
from megatron.core.dist_checkpointing.mapping import ShardedStateDict
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.utils import (
    attention_mask_func,
    is_layer_window_attention,
    make_sharded_tensors_for_checkpoint,
)
from megatron.core.utils import divide


class FlexDotProductAttention(MegatronModule):
    """
    FlexAttention-based attention implementation that provides flexible attention patterns
    through PyTorch's FlexAttention API. This allows for easy implementation of various
    attention variants like causal attention, sliding window, etc.
    
    We use the following notation:
     h: hidden size
     n: number of attention heads
     p: number of tensor model parallel partitions
     b: batch size
     s: sequence length
    """

    def __init__(
        self,
        config: TransformerConfig,
        layer_number: int,
        attn_mask_type: AttnMaskType,
        attention_type: str,
        attention_dropout: float = None,
        softmax_scale: float = None,
        cp_comm_type: str = None,
        pg_collection: ProcessGroupCollection = None,
    ):
        super().__init__(config=config)

        self.config: TransformerConfig = config

        assert (
            self.config.context_parallel_size == 1
        ), "Context parallelism is only supported by TEDotProductAttention!"

        print(f"config.num_query_groups: {self.config.num_query_groups}")
        # We do not support GQA for now
        assert self.config.num_query_groups == self.config.num_attention_heads, "GQA is not supported by FlexDotProductAttention!"

        self.layer_number = max(1, layer_number)
        self.attn_mask_type = attn_mask_type
        self.attention_type = attention_type

        projection_size = self.config.kv_channels * self.config.num_attention_heads

        # Per attention head and per partition values.
        if pg_collection is None:
            pg_collection = ProcessGroupCollection.use_mpu_process_groups(required_pgs=['tp'])
        else:
            assert hasattr(
                pg_collection, 'tp'
            ), "FlexDotProductAttention pg_collection must have tp process group"

        world_size = pg_collection.tp.size()
        self.hidden_size_per_partition = divide(projection_size, world_size)
        self.hidden_size_per_attention_head = divide(projection_size, config.num_attention_heads)
        self.num_attention_heads_per_partition = divide(self.config.num_attention_heads, world_size)

        # Softmax scaling
        if softmax_scale is None:
            self.softmax_scale = 1.0 / math.sqrt(self.hidden_size_per_attention_head)
        else:
            self.softmax_scale = softmax_scale

        if self.config.apply_query_key_layer_scaling:
            coeff = self.layer_number
            self.softmax_scale /= coeff

        # Dropout. Note that for a single iteration, this layer will generate
        # different outputs on different number of parallel partitions but
        # on average it should not be partition dependent.
        self.attention_dropout = torch.nn.Dropout(
            self.config.attention_dropout if attention_dropout is None else attention_dropout
        )

        # Window attention support
        self.window_size = None
        if is_layer_window_attention(
            self.config.window_size, self.config.window_attn_skip_freq, layer_number
        ):
            self.window_size = self.config.window_size

        # Score modification functions for different attention types
        self._setup_score_mod_functions()

    def _setup_score_mod_functions(self):
        """Setup score modification functions for different attention patterns."""
        
        def noop_score_mod(score, b, h, q_idx, kv_idx):
            """No-op score modification for full attention."""
            return score * self.softmax_scale

        def causal_score_mod(score, b, h, q_idx, kv_idx):
            """Causal (autoregressive) attention score modification."""
            scaled_score = score * self.softmax_scale
            # Apply causal mask: -inf for future positions
            return torch.where(q_idx >= kv_idx, scaled_score, torch.tensor(float('-inf'), device=score.device, dtype=score.dtype))

        def sliding_window_score_mod(score, b, h, q_idx, kv_idx):
            """Sliding window attention score modification."""
            scaled_score = score * self.softmax_scale
            # Apply sliding window mask
            if self.window_size is not None:
                mask = (kv_idx >= q_idx - self.window_size + 1) & (kv_idx <= q_idx)
                return torch.where(mask, scaled_score, torch.tensor(float('-inf'), device=score.device, dtype=score.dtype))
            return scaled_score

        def causal_sliding_window_score_mod(score, b, h, q_idx, kv_idx):
            """Combined causal and sliding window attention."""
            scaled_score = score * self.softmax_scale
            causal_mask = q_idx >= kv_idx
            if self.window_size is not None:
                window_mask = kv_idx >= q_idx - self.window_size + 1
                combined_mask = causal_mask & window_mask
            else:
                combined_mask = causal_mask
            return torch.where(combined_mask, scaled_score, torch.tensor(float('-inf'), device=score.device, dtype=score.dtype))

        # Map attention mask types to score mod functions
        self.score_mod_functions = {
            AttnMaskType.no_mask: noop_score_mod,
            AttnMaskType.causal: causal_score_mod,
            AttnMaskType.no_mask: sliding_window_score_mod, # The no_mask is just a random Enum used here. We need a better way to map.
            AttnMaskType.padding: causal_sliding_window_score_mod, # The padding is just a random Enum used here. We need a better way to map.
        }

    def _get_score_mod_function(self):
        """Get the appropriate score modification function based on attention mask type."""
        return self.score_mod_functions.get(self.attn_mask_type, self.score_mod_functions[AttnMaskType.no_mask])

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attention_mask: Tensor = None,
        attn_mask_type: AttnMaskType = None,
        attention_bias: Tensor = None,
        packed_seq_params: Optional[PackedSeqParams] = None,
    ):
        """Forward pass using FlexAttention."""
        assert packed_seq_params is None, (
            "Packed sequence is not supported by FlexDotProductAttention."
        )
        
        # Handle input shapes: [sq, b, np, hn]
        sq, batch_size, num_heads, head_dim = query.shape
        sk = key.size(0)
        
        # Reshape for FlexAttention: [b, num_heads, seq_len, head_dim]
        query = query.transpose(0, 1)  # [b, sq, np, hn]
        key = key.transpose(0, 1)      # [b, sk, np, hn]  
        value = value.transpose(0, 1)  # [b, sk, np, hn]
        
        query = query.transpose(1, 2)  # [b, np, sq, hn]
        key = key.transpose(1, 2)      # [b, np, sk, hn]
        value = value.transpose(1, 2)  # [b, np, sk, hn]

        # Get the appropriate score modification function
        score_mod = self._get_score_mod_function()

        # Apply FlexAttention
        context = flex_attention(query, key, value, score_mod=score_mod)

        # Apply dropout
        if not self.config.sequence_parallel:
            with tensor_parallel.get_cuda_rng_tracker().fork():
                context = self.attention_dropout(context)
        else:
            context = self.attention_dropout(context)

        # Reshape back to original format: [sq, b, hp]
        context = context.transpose(1, 2)  # [b, sq, np, hn]
        context = context.transpose(0, 1)  # [sq, b, np, hn]
        
        # Reshape to [sq, b, hp] where hp = hidden_size_per_partition
        new_context_shape = context.size()[:-2] + (self.hidden_size_per_partition,)
        context = context.contiguous().view(*new_context_shape)

        return context

    def sharded_state_dict(
        self,
        prefix: str = '',
        sharded_offsets: Tuple[Tuple[int, int, int]] = (),
        metadata: Optional[dict] = None,
    ) -> ShardedStateDict:
        """Sharded state dict - FlexAttention doesn't have learnable parameters by default."""
        return {}
