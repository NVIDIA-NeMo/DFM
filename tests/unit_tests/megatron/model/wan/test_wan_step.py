import pytest
import torch

from dfm.src.megatron.model.wan.wan_step import wan_data_step, WanForwardStep


class _DummyIter:
    def __init__(self, batch):
        # mimic attribute used inside wan_data_step
        self.iterable = [batch]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="wan_data_step moves tensors to CUDA")
def test_wan_data_step_builds_packed_seq_params_cuda_guarded():
    # Construct minimal batch with required seq_len fields
    batch = {
        "seq_len_q": torch.tensor([3, 5], dtype=torch.int32),
        "seq_len_q_padded": torch.tensor([4, 6], dtype=torch.int32),
        "seq_len_kv": torch.tensor([2, 7], dtype=torch.int32),
        "seq_len_kv_padded": torch.tensor([2, 8], dtype=torch.int32),
        # include a tensor field to exercise device transfer
        "video_latents": torch.randn(8, 1, 4, dtype=torch.float32),
    }
    it = _DummyIter(batch)
    qkv_format = "sbhd"
    out = wan_data_step(qkv_format, it)

    assert "packed_seq_params" in out
    for k in ["self_attention", "cross_attention"]:
        assert k in out["packed_seq_params"]
        p = out["packed_seq_params"][k]
        assert hasattr(p, "cu_seqlens_q")
        assert hasattr(p, "cu_seqlens_q_padded")
        assert hasattr(p, "cu_seqlens_kv")
        assert hasattr(p, "cu_seqlens_kv_padded")
    # spot-check CUDA device after move
    assert out["video_latents"].is_cuda


def test_wan_forward_step_loss_partial_creation():
    step = WanForwardStep()
    mask = torch.ones(4, dtype=torch.float32)
    loss_fn = step._create_loss_function(mask, check_for_nan_in_loss=False, check_for_spiky_loss=False)
    # Just validate it's callable and is a functools.partial
    import functools

    assert isinstance(loss_fn, functools.partial)
    assert callable(loss_fn)


