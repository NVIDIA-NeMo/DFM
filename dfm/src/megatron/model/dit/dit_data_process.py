# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

import torch
from megatron.core.packed_seq_params import PackedSeqParams


def dit_data_step(qkv_format, dataloader_iter):
    # import pdb;pdb.set_trace()
    batch = next(iter(dataloader_iter.iterable))
    batch = get_batch_on_this_cp_rank(batch)
    batch = {k: v.to(device="cuda", non_blocking=True) if torch.is_tensor(v) else v for k, v in batch.items()}
    batch["is_preprocessed"] = True  # assume data is preprocessed
    return encode_seq_length(batch, format=qkv_format)


def encode_seq_length(batch, format):
    if ("seq_len_q" in batch) and ("seq_len_kv" in batch):
        zero = torch.zeros([1], dtype=torch.int32, device="cuda")

        cu_seqlens_q = batch["seq_len_q"].cumsum(dim=0).to(torch.int32)
        cu_seqlens_q = torch.cat((zero, cu_seqlens_q))

        cu_seqlens_kv = batch["seq_len_kv"].cumsum(dim=0).to(torch.int32)
        cu_seqlens_kv = torch.cat((zero, cu_seqlens_kv))

        batch["packed_seq_params"] = {
            "self_attention": PackedSeqParams(
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_kv=cu_seqlens_q,
                cu_seqlens_q_padded=None,
                cu_seqlens_kv_padded=None,
                qkv_format=format,
            ),
            "cross_attention": PackedSeqParams(
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_kv=cu_seqlens_kv,
                cu_seqlens_q_padded=None,
                cu_seqlens_kv_padded=None,
                qkv_format=format,
            ),
        }

    return batch


def get_batch_on_this_cp_rank(data):
    """Split the data for context parallelism."""
    from megatron.core import mpu

    cp_size = mpu.get_context_parallel_world_size()
    cp_rank = mpu.get_context_parallel_rank()

    t = 16
    if cp_size > 1:
        # cp split on seq_length, for video_latent, noise_latent and pos_ids
        assert t % cp_size == 0, "t must divisibly by cp_size"
        num_valid_tokens_in_ub = None
        if "loss_mask" in data and data["loss_mask"] is not None:
            num_valid_tokens_in_ub = data["loss_mask"].sum()

        for key, value in data.items():
            if (value is not None) and (key in ["video", "video_latent", "noise_latent", "pos_ids"]):
                if len(value.shape) > 5:
                    value = value.squeeze(0)
                B, C, T, H, W = value.shape
                if T % cp_size == 0:
                    # FIXME packed sequencing
                    data[key] = value.view(B, C, cp_size, T // cp_size, H, W)[:, :, cp_rank, ...].contiguous()
                else:
                    # FIXME packed sequencing
                    data[key] = value.view(B, C, T, cp_size, H // cp_size, W)[:, :, :, cp_rank, ...].contiguous()
        loss_mask = data["loss_mask"]
        data["loss_mask"] = loss_mask.view(loss_mask.shape[0], cp_size, loss_mask.shape[1] // cp_size)[
            :, cp_rank, ...
        ].contiguous()
        data["num_valid_tokens_in_ub"] = num_valid_tokens_in_ub

    return data
