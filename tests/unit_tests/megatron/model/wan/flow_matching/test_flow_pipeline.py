import types
import torch

from dfm.src.megatron.model.wan.flow_matching import flow_pipeline as flow_pipeline_mod
from dfm.src.megatron.model.wan.flow_matching.flow_pipeline import FlowPipeline


class _DummyModel:
    class _Cfg:
        in_channels = 2
        patch_spatial = 1
        patch_temporal = 1

    def __init__(self):
        self.config = self._Cfg()

    def __call__(self, x, grid_sizes, t, context, packed_seq_params):
        # Return zeros matching input shape (seq_len, 1, latent_dim)
        return torch.zeros_like(x)


def test_flow_pipeline_training_step_cpu_stub(monkeypatch):
    # Bypass heavy diffusers init
    def _stub_init(self, model_id="x", seed=0):
        self.pipe = types.SimpleNamespace(scheduler=types.SimpleNamespace(config=types.SimpleNamespace(num_train_timesteps=1000)))
    monkeypatch.setattr(FlowPipeline, "__init__", _stub_init)

    # Make patchify accept both tensor and list for this test
    def _safe_patchify(x, patch_size):
        # Always delegate to the real implementation in utils to avoid recursion
        from dfm.src.megatron.model.wan import utils as wan_utils
        impl = wan_utils.patchify
        # Normalize inputs to expected 4D [C, F, H, W] without batch dim
        if isinstance(x, list):
            x_norm = []
            for t in x:
                if isinstance(t, torch.Tensor) and t.dim() == 5 and t.size(0) == 1:
                    x_norm.append(t.squeeze(0))
                else:
                    x_norm.append(t)
        else:
            t = x
            if isinstance(t, torch.Tensor) and t.dim() == 5 and t.size(0) == 1:
                t = t.squeeze(0)
            x_norm = [t]
        return impl(x_norm, patch_size)
    monkeypatch.setattr(flow_pipeline_mod, "patchify", _safe_patchify)

    # Disable context parallelism and force last pipeline stage
    from megatron.core import parallel_state
    monkeypatch.setattr(parallel_state, "get_context_parallel_world_size", lambda: 1, raising=False)
    monkeypatch.setattr(parallel_state, "is_pipeline_last_stage", lambda: True, raising=False)

    pipe = FlowPipeline()
    model = _DummyModel()

    # Build a minimal, consistent batch: seq_len = F*H*W = 2*2*2 = 8, latent_dim = in_channels * pF * pH * pW = 2
    F, H, W = 2, 2, 2
    seq_len = F * H * W
    latent_dim = model.config.in_channels

    video_latents = torch.randn(seq_len, 1, latent_dim, dtype=torch.float32)
    context_embeddings = torch.randn(4, 1, 8, dtype=torch.float32)
    loss_mask = torch.ones(seq_len, dtype=torch.bfloat16)
    grid_sizes = torch.tensor([[F, H, W]], dtype=torch.int32)

    # Packed seq params with simple cumulative lengths
    from megatron.core.packed_seq_params import PackedSeqParams
    cu = torch.tensor([0, seq_len], dtype=torch.int32)
    packed_seq_params = {
        "self_attention": PackedSeqParams(
            cu_seqlens_q=cu, cu_seqlens_q_padded=cu, cu_seqlens_kv=cu, cu_seqlens_kv_padded=cu, qkv_format="sbhd"
        ),
        "cross_attention": PackedSeqParams(
            cu_seqlens_q=cu, cu_seqlens_q_padded=cu, cu_seqlens_kv=cu, cu_seqlens_kv_padded=cu, qkv_format="sbhd"
        ),
    }

    batch = {
        "video_latents": video_latents,
        "context_embeddings": context_embeddings,
        "loss_mask": loss_mask,
        "grid_sizes": grid_sizes,
        "packed_seq_params": packed_seq_params,
        "video_metadata": {},
    }

    model_pred, weighted_loss, split_loss_mask = pipe.training_step(
        model,
        batch,
        use_sigma_noise=True,
        timestep_sampling="uniform",
        flow_shift=3.0,
        mix_uniform_ratio=1.0,  # force uniform branch
        sigma_min=0.0,
        sigma_max=1.0,
    )

    # Basic shape checks
    assert model_pred.shape == video_latents.shape
    assert weighted_loss.shape[:2] == video_latents.shape[:2]
    assert split_loss_mask.shape == loss_mask.shape


