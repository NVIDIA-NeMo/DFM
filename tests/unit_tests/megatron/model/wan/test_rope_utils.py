import pytest
import torch

from dfm.src.megatron.model.wan.rope_utils import Wan3DRopeEmbeddings


def test_wan3d_rope_embeddings_shapes_and_padding():
    # Small, CPU-friendly config
    n_head = 2
    dim_head = 8  # must be divisible with the internal splits
    max_position_len = 16
    rope = Wan3DRopeEmbeddings(dim_head=dim_head, max_position_len=max_position_len)

    # Two samples with different (f, h, w)
    grid_sizes = torch.tensor([[2, 3, 2], [4, 1, 1]], dtype=torch.int32)
    seq_lens = [(2 * 3 * 2), (4 * 1 * 1)]
    padded_lens = [seq_lens[0] + 2, seq_lens[1]]  # pad first sample

    cu_seqlens_q_padded = torch.tensor([0, padded_lens[0], padded_lens[0] + padded_lens[1]], dtype=torch.int32)

    out = rope(
        n_head=n_head,
        dim_head=dim_head,
        cu_seqlens_q_padded=cu_seqlens_q_padded,
        grid_sizes=grid_sizes,
        device=torch.device("cpu"),
    )

    # Total concatenated length equals sum of padded lens
    assert out.shape == (sum(padded_lens), 1, 1, dim_head)

    # Check that padding region for the first sample is zero
    first_seq_len = seq_lens[0]
    first_padded_len = padded_lens[0]
    tail = out[first_seq_len:first_padded_len]
    assert torch.all(tail == 0), "Padded region should be zeros"


