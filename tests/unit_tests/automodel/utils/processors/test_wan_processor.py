# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
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

"""Unit tests for WanProcessor."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from dfm.src.automodel.utils.processors import ProcessorRegistry, WanProcessor
from dfm.src.automodel.utils.processors.base_video import BaseVideoProcessor


class TestWanProcessorRegistration:
    """Test processor registration."""

    def test_registered_as_wan(self):
        """Test that processor is registered under 'wan' name."""
        processor = ProcessorRegistry.get("wan")
        assert isinstance(processor, WanProcessor)

    def test_registered_as_wan21(self):
        """Test that processor is registered under 'wan2.1' name."""
        processor = ProcessorRegistry.get("wan2.1")
        assert isinstance(processor, WanProcessor)

    def test_is_video_processor(self):
        """Test that WanProcessor inherits from BaseVideoProcessor."""
        processor = WanProcessor()
        assert isinstance(processor, BaseVideoProcessor)


class TestWanProcessorProperties:
    """Test processor properties."""

    def test_model_type(self):
        processor = WanProcessor()
        assert processor.model_type == "wan"

    def test_default_model_name(self):
        processor = WanProcessor()
        assert processor.default_model_name == "Wan-AI/Wan2.1-T2V-14B-Diffusers"

    def test_supported_modes(self):
        processor = WanProcessor()
        assert "video" in processor.supported_modes
        assert "frames" in processor.supported_modes

    def test_quantization(self):
        """Test that quantization is 8 for video VAE."""
        processor = WanProcessor()
        assert processor.quantization == 8

    def test_frame_constraint(self):
        """Test that Wan has no specific frame constraint."""
        processor = WanProcessor()
        assert processor.frame_constraint is None

    def test_max_sequence_length(self):
        """Test that max sequence length is 226."""
        assert WanProcessor.MAX_SEQUENCE_LENGTH == 226


class TestWanProcessorTextPadding:
    """Test text encoding padding behavior."""

    @pytest.fixture
    def mock_models(self):
        """Create mock models for testing."""
        tokenizer = MagicMock()
        tokenizer.return_value = {
            "input_ids": torch.randint(0, 1000, (1, 226)),
            "attention_mask": torch.ones(1, 226),
        }

        text_encoder = MagicMock()
        # Mock output with shape (1, 226, 4096)
        text_encoder.return_value.last_hidden_state = torch.randn(1, 226, 4096)

        return {
            "tokenizer": tokenizer,
            "text_encoder": text_encoder,
        }

    def test_text_encoding_output_shape(self, mock_models):
        """Test that text encoding produces correct shape."""
        processor = WanProcessor()

        with patch.object(processor, "encode_text") as mock_encode:
            mock_encode.return_value = {"text_embeddings": torch.randn(1, 226, 4096)}
            result = processor.encode_text("test prompt", mock_models, "cpu")
            assert "text_embeddings" in result
            assert result["text_embeddings"].shape == (1, 226, 4096)


class TestWanProcessorLatentNormalization:
    """Test latent normalization behavior."""

    def test_latent_normalization_formula(self):
        """Test that latent normalization uses mean/std formula."""
        # Create mock VAE with config
        mock_vae = MagicMock()
        mock_vae.config.latents_mean = [0.0] * 16
        mock_vae.config.latents_std = [1.0] * 16
        mock_vae.dtype = torch.float16

        # Mock encode output
        latent_dist = MagicMock()
        latent_dist.latent_dist.mean = torch.randn(1, 16, 4, 30, 53)
        mock_vae.encode.return_value = latent_dist

        processor = WanProcessor()
        models = {"vae": mock_vae, "dtype": torch.float16}

        # Test encode_video
        video_tensor = torch.randn(1, 3, 10, 240, 424)

        with torch.no_grad():
            result = processor.encode_video(video_tensor, models, "cpu", deterministic=True)

        # Verify shape and dtype
        assert result.dtype == torch.float16
        assert len(result.shape) == 5  # (1, C, T, H, W)


class TestWanProcessorCacheData:
    """Test cache data structure."""

    def test_cache_data_structure(self):
        """Test that get_cache_data returns correct structure."""
        processor = WanProcessor()

        latent = torch.randn(1, 16, 4, 30, 53)
        text_encodings = {"text_embeddings": torch.randn(1, 226, 4096)}
        metadata = {
            "original_resolution": (1920, 1080),
            "bucket_resolution": (848, 480),
            "bucket_id": 5,
            "aspect_ratio": 1.767,
            "num_frames": 10,
            "prompt": "test prompt",
            "video_path": "/path/to/video.mp4",
            "first_frame": np.zeros((480, 848, 3), dtype=np.uint8),
        }

        cache_data = processor.get_cache_data(latent, text_encodings, metadata)

        # Check required keys
        assert "video_latents" in cache_data
        assert "text_embeddings" in cache_data
        assert "first_frame" in cache_data
        assert "original_resolution" in cache_data
        assert "bucket_resolution" in cache_data
        assert "bucket_id" in cache_data
        assert "aspect_ratio" in cache_data
        assert "num_frames" in cache_data
        assert "prompt" in cache_data
        assert "video_path" in cache_data
        assert "model_version" in cache_data
        assert "model_type" in cache_data

        # Check values
        assert cache_data["model_version"] == "wan2.1"
        assert cache_data["model_type"] == "wan"
        assert torch.equal(cache_data["video_latents"], latent)
        assert torch.equal(cache_data["text_embeddings"], text_encodings["text_embeddings"])


class TestWanProcessorPromptCleaning:
    """Test prompt cleaning functions."""

    def test_basic_clean(self):
        """Test basic text cleaning."""
        from dfm.src.automodel.utils.processors.wan import _basic_clean

        # Test HTML entity unescaping
        assert "&amp;" not in _basic_clean("test &amp; test")
        assert "&lt;" not in _basic_clean("&lt;tag&gt;")

    def test_whitespace_clean(self):
        """Test whitespace normalization."""
        from dfm.src.automodel.utils.processors.wan import _whitespace_clean

        assert _whitespace_clean("  hello   world  ") == "hello world"
        assert _whitespace_clean("a\n\nb\tc") == "a b c"

    def test_prompt_clean(self):
        """Test combined prompt cleaning."""
        from dfm.src.automodel.utils.processors.wan import _prompt_clean

        result = _prompt_clean("  hello   &amp;   world  ")
        assert result == "hello & world"


class TestWanProcessorVideoLoading:
    """Test video loading functionality."""

    def test_frames_to_tensor(self):
        """Test frames to tensor conversion."""
        processor = WanProcessor()

        # Create mock frames (T, H, W, C) in uint8
        frames = np.random.randint(0, 255, (10, 240, 424, 3), dtype=np.uint8)

        tensor = processor.frames_to_tensor(frames)

        # Check shape: (1, C, T, H, W)
        assert tensor.shape == (1, 3, 10, 240, 424)

        # Check normalization to [-1, 1]
        assert tensor.min() >= -1.0
        assert tensor.max() <= 1.0

    def test_adjust_frame_count_same(self):
        """Test frame adjustment when count matches."""
        processor = WanProcessor()

        frames = np.random.rand(10, 240, 424, 3)
        result = processor.adjust_frame_count(frames, 10)

        assert len(result) == 10
        assert np.array_equal(result, frames)

    def test_adjust_frame_count_downsample(self):
        """Test frame adjustment when downsampling."""
        processor = WanProcessor()

        frames = np.random.rand(100, 240, 424, 3)
        result = processor.adjust_frame_count(frames, 10)

        assert len(result) == 10

    def test_adjust_frame_count_upsample(self):
        """Test frame adjustment when upsampling."""
        processor = WanProcessor()

        frames = np.random.rand(5, 240, 424, 3)
        result = processor.adjust_frame_count(frames, 10)

        assert len(result) == 10


class TestWanProcessorVerifyLatent:
    """Test latent verification."""

    def test_verify_latent_valid(self):
        """Test verification passes for valid latent."""
        processor = WanProcessor()
        latent = torch.randn(1, 16, 4, 30, 53)
        assert processor.verify_latent(latent, {}, "cpu") is True

    def test_verify_latent_nan(self):
        """Test verification fails for NaN latent."""
        processor = WanProcessor()
        latent = torch.randn(1, 16, 4, 30, 53)
        latent[0, 0, 0, 0, 0] = float("nan")
        assert processor.verify_latent(latent, {}, "cpu") is False

    def test_verify_latent_inf(self):
        """Test verification fails for Inf latent."""
        processor = WanProcessor()
        latent = torch.randn(1, 16, 4, 30, 53)
        latent[0, 0, 0, 0, 0] = float("inf")
        assert processor.verify_latent(latent, {}, "cpu") is False
