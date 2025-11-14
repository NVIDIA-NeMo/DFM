import torch

from dfm.src.megatron.model.wan.wan_model import sinusoidal_embedding_1d


def test_sinusoidal_embedding_1d_shape_and_dtype():
    dim = 16
    pos = torch.arange(10, dtype=torch.float32)
    emb = sinusoidal_embedding_1d(dim, pos)
    assert emb.shape == (pos.shape[0], dim)
    assert emb.dtype == torch.float32


