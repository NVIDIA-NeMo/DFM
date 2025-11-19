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


import pytest
import torch
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.model.dit.dit_data_process import (
    encode_seq_length,
)


class TestEncodeSeqLength:
    """Test suite for encode_seq_length function."""

    @pytest.fixture
    def basic_batch(self):
        """Create a basic batch with sequence length information."""
        return {
            "seq_len_q": torch.tensor([10, 20, 15], dtype=torch.int32, device="cuda"),
            "seq_len_kv": torch.tensor([5, 10, 8], dtype=torch.int32, device="cuda"),
            "seq_len_q_padded": torch.tensor([12, 24, 16], dtype=torch.int32, device="cuda"),
            "seq_len_kv_padded": torch.tensor([8, 12, 10], dtype=torch.int32, device="cuda"),
            "video": torch.randn(3, 100, 512, device="cuda"),
        }

    def test_encode_seq_length_with_seq_lens(self, basic_batch):
        """Test encode_seq_length creates packed_seq_params when seq_len_q and seq_len_kv are present."""
        qkv_format = "thd"
        result = encode_seq_length(basic_batch, format=qkv_format)

        # Check that packed_seq_params is created
        assert "packed_seq_params" in result
        assert "self_attention" in result["packed_seq_params"]
        assert "cross_attention" in result["packed_seq_params"]

        # Check self_attention params
        self_attn = result["packed_seq_params"]["self_attention"]
        assert isinstance(self_attn, PackedSeqParams)
        assert self_attn.qkv_format == qkv_format

        # Verify cumulative sum for q (self_attention uses cu_seqlens_q for both q and kv)
        expected_cu_seqlens_q = torch.tensor([0, 10, 30, 45], dtype=torch.int32, device="cuda")
        assert torch.equal(self_attn.cu_seqlens_q, expected_cu_seqlens_q)
        assert torch.equal(self_attn.cu_seqlens_kv, expected_cu_seqlens_q)

        # Verify cumulative sum for q_padded
        expected_cu_seqlens_q_padded = torch.tensor([0, 12, 36, 52], dtype=torch.int32, device="cuda")
        assert torch.equal(self_attn.cu_seqlens_q_padded, expected_cu_seqlens_q_padded)
        assert torch.equal(self_attn.cu_seqlens_kv_padded, expected_cu_seqlens_q_padded)

        # Check cross_attention params
        cross_attn = result["packed_seq_params"]["cross_attention"]
        assert isinstance(cross_attn, PackedSeqParams)
        assert cross_attn.qkv_format == qkv_format

        # Verify cumulative sum for kv (cross_attention uses different kv lengths)
        expected_cu_seqlens_kv = torch.tensor([0, 5, 15, 23], dtype=torch.int32, device="cuda")
        assert torch.equal(cross_attn.cu_seqlens_q, expected_cu_seqlens_q)
        assert torch.equal(cross_attn.cu_seqlens_kv, expected_cu_seqlens_kv)

        # Verify cumulative sum for kv_padded
        expected_cu_seqlens_kv_padded = torch.tensor([0, 8, 20, 30], dtype=torch.int32, device="cuda")
        assert torch.equal(cross_attn.cu_seqlens_q_padded, expected_cu_seqlens_q_padded)
        assert torch.equal(cross_attn.cu_seqlens_kv_padded, expected_cu_seqlens_kv_padded)
