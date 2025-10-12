# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.


import math
from typing import Optional, Tuple

import torch
from torch import Tensor
from torch.nn.attention.flex_attention import create_block_mask, flex_attention

from megatron.core import tensor_parallel
from megatron.core.dist_checkpointing.mapping import ShardedStateDict
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.utils import is_layer_window_attention
from megatron.core.utils import divide


# Fused flex attention implementation (similar to mask.py)
@torch.compile(fullgraph=True, mode="max-autotune-no-cudagraphs", dynamic=False)
def fused_flex_attention(q, k, v, score_mod=None, block_mask=None):
    """Compiled flex attention for better performance."""
    return flex_attention(q, k, v, score_mod=score_mod, block_mask=block_mask)


class FlexDotProductAttention(MegatronModule):
    """
    FlexAttention-based attention implementation that provides flexible attention patterns
    through PyTorch's FlexAttention API. This allows for easy implementation of various
    attention variants like causal attention, sliding window, block diffusion, etc.
    
    Supported attention patterns:
    - No mask (full attention)
    - Causal (autoregressive)
    - Sliding window
    - Causal + sliding window
    - Block diffusion (for diffusion language models)
    
    Block Diffusion Attention:
    The block diffusion pattern is designed for diffusion language models and splits
    the sequence into xt (noisy) and x0 (clean) tokens. It implements three masks:
    - Block Diagonal (M_BD): Self-attention within blocks
    - Offset Block-Causal (M_OBC): Cross-attention from xt to x0
    - Block-Causal (M_BC): Causal attention within x0
    
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

        # Block diffusion parameters
        self.block_size = getattr(config, 'block_size', 2)
        self.seq_length = getattr(config, 'seq_length', 4096)
        self.use_fused_attention = getattr(config, 'use_fused_flex_attention', False)
        
        # Precompute block masks if using block diffusion
        self.block_diff_mask = None
        if self.block_size is not None and self.seq_length is not None:
            self.block_diff_mask = self._compute_block_mask(
                mode='block_diff', 
                block_size=self.block_size
            )

        # Set torch dynamo cache size for better compilation performance
        import torch._dynamo.config as dcfg
        dcfg.cache_size_limit = 512

        # Score modification functions for different attention types
        self._setup_score_mod_functions()

        self.use_block_diffusion_attention()

    def _compute_block_mask(self, mode: str, block_size: int = None, q_len: int = None):
        """
        Compute block mask for different attention patterns.
        Following the pattern from mask.py.
        
        Args:
            mode: Attention mode ('block_diff', 'causal', 'bidirectional', etc.)
            block_size: Block size for block-based attention
            q_len: Query length (defaults to seq_length or 2*seq_length for block_diff)
        """
        def block_diff_mask(block_size, b, h, q_idx, kv_idx, n):
            """
            Constructs the specialized block diffusion attention mask for training
            composed of three masks:
            - **Block Diagonal Mask (M_BD)**: Self-attention within noised blocks
            - **Offset Block Causal Mask (M_OBC)**: Cross-attention for conditional context
            - **Block Causal Mask (M_BC)**: Attention to update x0

            Args:
                block_size: Defines the block structure.
                b, h: Batch and head indices (ignored for mask logic).
                q_idx, kv_idx: Query and Key indices.
                n: Half sequence length (total sequence is 2*n for xt + x0)

            Returns:
                A boolean attention mask.
            """
            # Indicate whether token belongs to xt or x0
            x0_flag_q = (q_idx >= n)
            x0_flag_kv = (kv_idx >= n)

            # Compute block indices
            block_q = torch.where(x0_flag_q == 1,
                                    (q_idx - n) // block_size,
                                    q_idx // block_size)
            block_kv = torch.where(x0_flag_kv == 1,
                                    (kv_idx - n) // block_size,
                                    kv_idx // block_size)

            # **1. Block Diagonal Mask (M_BD) **
            block_diagonal = (block_q == block_kv) & (x0_flag_q == x0_flag_kv)

            # **2. Offset Block-Causal Mask (M_OBC) **
            offset_block_causal = (
                (block_q > block_kv)
                & (x0_flag_kv == 1)
                & (x0_flag_q == 0)
            )

            # **3. Block-Causal Mask (M_BC) **
            block_causal = (block_q >= block_kv) & (x0_flag_kv == 1) & (x0_flag_q == 1)

            # **4. Combine Masks **
            return block_diagonal | offset_block_causal | block_causal

        if mode == 'block_diff':
            assert block_size is not None, "block_size must be provided for block_diff mode"
            attn_mask = lambda b, h, q, kv: block_diff_mask(block_size, b, h, q, kv, self.seq_length)
        else:
            raise ValueError(f"Unknown attention mode: {mode}")

        # Determine query/key length
        if q_len is not None:
            Q_LEN = q_len
        else:
            if mode == 'block_diff':
                Q_LEN = self.seq_length * 2  # doubled for xt + x0
            else:
                Q_LEN = self.seq_length

        # Create block mask
        block_mask = create_block_mask(
            attn_mask, B=None, H=None, Q_LEN=Q_LEN, KV_LEN=Q_LEN
        )

        return block_mask

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
        # Note: For block_diff, we use block_mask instead of score_mod
        self.score_mod_functions = {
            AttnMaskType.no_mask: noop_score_mod,
            AttnMaskType.causal: causal_score_mod,
            AttnMaskType.no_mask: sliding_window_score_mod, # The no_mask is just a random Enum used here. We need a better way to map.
            AttnMaskType.padding: causal_sliding_window_score_mod, # The padding is just a random Enum used here. We need a better way to map.
        }

    def _get_score_mod_function(self):
        """Get the appropriate score modification function based on attention mask type."""
        return self.score_mod_functions.get(self.attn_mask_type, self.score_mod_functions[AttnMaskType.no_mask])
    
    def set_block_diffusion_params(self, block_size: int = None, seq_length: int = None):
        """
        Update block diffusion parameters dynamically and recompute block mask.
        
        Args:
            block_size: Size of each block for block diffusion attention
            seq_length: Length of the sequence (will be doubled for block diffusion: xt + x0)
        """
        recompute = False
        if block_size is not None and block_size != self.block_size:
            self.block_size = block_size
            recompute = True
        if seq_length is not None and seq_length != self.seq_length:
            self.seq_length = seq_length
            recompute = True
        
        # Recompute block mask if parameters changed
        if recompute and self.block_size is not None and self.seq_length is not None:
            self.block_diff_mask = self._compute_block_mask(
                mode='block_diff',
                block_size=self.block_size
            )
    
    def use_block_diffusion_attention(self):
        """
        Switch to block diffusion attention pattern.
        Ensures block mask is computed if not already done.
        """
        self.attn_mask_type = 'block_diff'
        if self.block_diff_mask is None and self.block_size is not None and self.seq_length is not None:
            self.block_diff_mask = self._compute_block_mask(
                mode='block_diff',
                block_size=self.block_size
            )

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

        # Determine whether to use block_mask or score_mod
        block_mask = None
        score_mod = None
        
        if self.attn_mask_type == 'block_diff':
            # Use block_mask for block diffusion
            if self.block_diff_mask is None or sq != self.block_diff_mask.shape[-2]:
                # Compute or recompute block mask for different sequence length
                assert self.block_size is not None and self.seq_length is not None, \
                    "block_size and seq_length must be set for block_diff attention"
                block_mask = self._compute_block_mask(
                    mode='block_diff',
                    block_size=self.block_size,
                    q_len=sq
                )
            else:
                block_mask = self.block_diff_mask
        else:
            # Use score_mod for other attention patterns
            score_mod = self._get_score_mod_function()

        # Apply FlexAttention (fused or standard)
        if self.use_fused_attention:
            context = fused_flex_attention(query, key, value, score_mod=score_mod, block_mask=block_mask)
        else:
            context = flex_attention(query, key, value, score_mod=score_mod, block_mask=block_mask)

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
