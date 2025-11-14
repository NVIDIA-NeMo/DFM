import torch

from torch.utils.data import DataLoader

from dfm.src.megatron.data.wan.wan_mock_datamodule import WanMockDataModuleConfig


def test_wan_mock_datamodule_build_and_batch_shapes():
    cfg = WanMockDataModuleConfig(
        path="",
        seq_length=128,
        packing_buffer_size=2,
        micro_batch_size=2,
        global_batch_size=8,
        num_workers=0,
        # Use small shapes for a light-weight test run
        F_latents=4,
        H_latents=8,
        W_latents=6,
        patch_spatial=2,
        patch_temporal=1,
        number_packed_samples=2,
        context_seq_len=16,
        context_embeddings_dim=64,
    )
    train_dl, val_dl, test_dl = cfg.build_datasets(_context=None)
    assert isinstance(train_dl, DataLoader)
    assert train_dl is val_dl and val_dl is test_dl

    batch = next(iter(train_dl))
    expected_keys = {
        "video_latents",
        "context_embeddings",
        "loss_mask",
        "seq_len_q",
        "seq_len_q_padded",
        "seq_len_kv",
        "seq_len_kv_padded",
        "grid_sizes",
        "video_metadata",
    }
    assert expected_keys.issubset(set(batch.keys()))

    # Basic sanity checks on shapes/dtypes
    assert batch["video_latents"].dim() == 3 and batch["video_latents"].shape[1] == 1
    assert batch["context_embeddings"].dim() == 3 and batch["context_embeddings"].shape[1] == 1
    assert batch["loss_mask"].dim() == 2 and batch["loss_mask"].shape[1] == 1
    assert batch["seq_len_q"].dtype == torch.int32
    assert batch["seq_len_q_padded"].dtype == torch.int32
    assert batch["seq_len_kv"].dtype == torch.int32
    assert batch["seq_len_kv_padded"].dtype == torch.int32
    assert batch["grid_sizes"].dtype == torch.int32


