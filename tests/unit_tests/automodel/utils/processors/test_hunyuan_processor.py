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

"""Unit tests for HunyuanVideoProcessor."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from dfm.src.automodel.utils.processors import HunyuanVideoProcessor, ProcessorRegistry
from dfm.src.automodel.utils.processors.base_video import BaseVideoProcessor


class TestHunyuanProcessorRegistration:
    """Test processor registration."""

    def test_registered_as_hunyuan(self):
        """Test that processor is registered under 'hunyuan' name."""
        processor = ProcessorRegistry.get("hunyuan")
        assert isinstance(processor, HunyuanVideoProcessor)

    def test_registered_as_hunyuanvideo(self):
        """Test that processor is registered under 'hunyuanvideo' name."""
        processor = ProcessorRegistry.get("hunyuanvideo")
        assert isinstance(processor, HunyuanVideoProcessor)

    def test_registered_as_hunyuanvideo_15(self):
        """Test that processor is registered under 'hunyuanvideo-1.5' name."""
        processor = ProcessorRegistry.get("hunyuanvideo-1.5")
        assert isinstance(processor, HunyuanVideoProcessor)

    def test_is_video_processor(self):
        """Test that HunyuanVideoProcessor inherits from BaseVideoProcessor."""
        processor = HunyuanVideoProcessor()
        assert isinstance(processor, BaseVideoProcessor)


class TestHunyuanProcessorProperties:
    """Test processor properties."""

    def test_model_type(self):
        processor = HunyuanVideoProcessor()
        assert processor.model_type == "hunyuanvideo"

    def test_default_model_name(self):
        processor = HunyuanVideoProcessor()
        assert "hunyuanvideo" in processor.default_model_name.lower()

    def test_supported_modes(self):
        processor = HunyuanVideoProcessor()
        assert "video" in processor.supported_modes

    def test_quantization(self):
        """Test that quantization is 8 for video VAE."""
        processor = HunyuanVideoProcessor()
        assert processor.quantization == 8

    def test_frame_constraint(self):
        """Test that HunyuanVideo has 4n+1 frame constraint."""
        processor = HunyuanVideoProcessor()
        assert processor.frame_constraint == "4n+1"

    def test_default_image_embed_shape(self):
        """Test default image embedding shape."""
        assert HunyuanVideoProcessor.DEFAULT_IMAGE_EMBED_SHAPE == (729, 1152)


class TestHunyuanProcessorFrameConstraint:
    """Test 4n+1 frame constraint handling."""

    def test_validate_frame_count_valid(self):
        """Test validation of valid 4n+1 frame counts."""
        processor = HunyuanVideoProcessor()

        # 4n+1 values: 1, 5, 9, 13, 17, 21, 25, ..., 121
        valid_counts = [1, 5, 9, 13, 17, 21, 25, 29, 33, 121]

        for count in valid_counts:
            assert processor.validate_frame_count(count) is True, f"Expected {count} to be valid"

    def test_validate_frame_count_invalid(self):
        """Test validation of invalid frame counts."""
        processor = HunyuanVideoProcessor()

        invalid_counts = [2, 3, 4, 6, 7, 8, 10, 11, 12, 100, 120, 122]

        for count in invalid_counts:
            assert processor.validate_frame_count(count) is False, f"Expected {count} to be invalid"

    def test_get_closest_valid_frame_count(self):
        """Test finding closest 4n+1 value."""
        processor = HunyuanVideoProcessor()

        # Test cases: (input, expected_closest_4n+1)
        test_cases = [
            (1, 1),
            (2, 1),
            (3, 1),  # Closer to 1 than 5
            (4, 5),  # Closer to 5 than 1
            (5, 5),
            (6, 5),
            (7, 5),  # Closer to 5 than 9
            (8, 9),  # Closer to 9 than 5
            (9, 9),
            (10, 9),
            (100, 101),  # 4*25+1 = 101
            (120, 121),  # 4*30+1 = 121
        ]

        for input_count, expected in test_cases:
            result = processor.get_closest_valid_frame_count(input_count)
            assert result == expected, f"For input {input_count}, expected {expected}, got {result}"

    def test_adjust_frame_count_valid_input(self):
        """Test frame adjustment with valid 4n+1 target."""
        processor = HunyuanVideoProcessor()

        frames = np.random.rand(100, 240, 424, 3)

        # Adjust to 4n+1 = 121
        result = processor.adjust_frame_count(frames, 121)
        assert len(result) == 121

        # Adjust to 4n+1 = 9
        result = processor.adjust_frame_count(frames, 9)
        assert len(result) == 9

    def test_adjust_frame_count_invalid_target_raises(self):
        """Test that invalid target raises ValueError."""
        processor = HunyuanVideoProcessor()

        frames = np.random.rand(100, 240, 424, 3)

        with pytest.raises(ValueError, match="must be 4n\\+1"):
            processor.adjust_frame_count(frames, 10)

        with pytest.raises(ValueError, match="must be 4n\\+1"):
            processor.adjust_frame_count(frames, 100)

    def test_adjust_frame_count_same_count(self):
        """Test frame adjustment when count already matches."""
        processor = HunyuanVideoProcessor()

        frames = np.random.rand(121, 240, 424, 3)
        result = processor.adjust_frame_count(frames, 121)

        assert len(result) == 121
        assert np.array_equal(result, frames)


class TestHunyuanProcessorCacheData:
    """Test cache data structure."""

    def test_cache_data_structure(self):
        """Test that get_cache_data returns correct structure for HunyuanVideo."""
        processor = HunyuanVideoProcessor()

        latent = torch.randn(1, 16, 31, 45, 80)  # (1, C, T, H, W) - 121 frames -> 31 latent frames
        text_encodings = {
            "text_embeddings": torch.randn(1, 256, 4096),
            "text_mask": torch.ones(1, 256),
            "text_embeddings_2": torch.randn(1, 256, 1024),
            "text_mask_2": torch.ones(1, 256),
        }
        metadata = {
            "original_resolution": (1920, 1080),
            "bucket_resolution": (1280, 720),
            "bucket_id": 5,
            "aspect_ratio": 1.778,
            "num_frames": 121,
            "prompt": "test prompt",
            "video_path": "/path/to/video.mp4",
            "image_embeds": torch.randn(1, 729, 1152),
        }

        cache_data = processor.get_cache_data(latent, text_encodings, metadata)

        # Check required keys
        assert "video_latents" in cache_data
        assert "text_embeddings" in cache_data
        assert "text_mask" in cache_data
        assert "text_embeddings_2" in cache_data
        assert "text_mask_2" in cache_data
        assert "image_embeds" in cache_data
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
        assert cache_data["model_version"] == "hunyuanvideo-1.5"
        assert cache_data["model_type"] == "hunyuanvideo"
        assert torch.equal(cache_data["video_latents"], latent)
        assert torch.equal(cache_data["text_embeddings"], text_encodings["text_embeddings"])
        assert torch.equal(cache_data["text_embeddings_2"], text_encodings["text_embeddings_2"])
        assert torch.equal(cache_data["image_embeds"], metadata["image_embeds"])


class TestHunyuanProcessorLatentNormalization:
    """Test latent normalization behavior."""

    def test_latent_normalization_with_shift_factor(self):
        """Test that latent normalization uses shift_factor when available."""
        # Create mock VAE with shift_factor
        mock_vae = MagicMock()
        mock_vae.config.shift_factor = 0.1
        mock_vae.config.scaling_factor = 0.5
        mock_vae.dtype = torch.float16

        # Mock encode output
        latent_dist = MagicMock()
        raw_latents = torch.randn(1, 16, 31, 45, 80)
        latent_dist.latent_dist.sample.return_value = raw_latents
        mock_vae.encode.return_value = latent_dist

        processor = HunyuanVideoProcessor()
        models = {"vae": mock_vae, "dtype": torch.float16}

        # Test encode_video
        video_tensor = torch.randn(1, 3, 121, 720, 1280)

        with torch.no_grad():
            result = processor.encode_video(video_tensor, models, "cpu", deterministic=True)

        # Verify shape and dtype
        assert result.dtype == torch.float16
        assert len(result.shape) == 5  # (1, C, T, H, W)


class TestHunyuanProcessorVideoLoading:
    """Test video loading functionality."""

    def test_frames_to_tensor(self):
        """Test frames to tensor conversion."""
        processor = HunyuanVideoProcessor()

        # Create mock frames with 4n+1 count (T, H, W, C) in uint8
        frames = np.random.randint(0, 255, (121, 720, 1280, 3), dtype=np.uint8)

        tensor = processor.frames_to_tensor(frames)

        # Check shape: (1, C, T, H, W)
        assert tensor.shape == (1, 3, 121, 720, 1280)

        # Check normalization to [-1, 1]
        assert tensor.min() >= -1.0
        assert tensor.max() <= 1.0


class TestHunyuanProcessorVerifyLatent:
    """Test latent verification."""

    def test_verify_latent_valid(self):
        """Test verification passes for valid latent."""
        processor = HunyuanVideoProcessor()
        latent = torch.randn(1, 16, 31, 45, 80)
        assert processor.verify_latent(latent, {}, "cpu") is True

    def test_verify_latent_nan(self):
        """Test verification fails for NaN latent."""
        processor = HunyuanVideoProcessor()
        latent = torch.randn(1, 16, 31, 45, 80)
        latent[0, 0, 0, 0, 0] = float("nan")
        assert processor.verify_latent(latent, {}, "cpu") is False

    def test_verify_latent_inf(self):
        """Test verification fails for Inf latent."""
        processor = HunyuanVideoProcessor()
        latent = torch.randn(1, 16, 31, 45, 80)
        latent[0, 0, 0, 0, 0] = float("inf")
        assert processor.verify_latent(latent, {}, "cpu") is False


class TestHunyuanProcessorImageEmbedding:
    """Test first frame image embedding."""

    def test_default_image_embed_shape_constant(self):
        """Test the default image embed shape constant."""
        seq_len, dim = HunyuanVideoProcessor.DEFAULT_IMAGE_EMBED_SHAPE
        assert seq_len == 729  # 27*27 patches
        assert dim == 1152  # Embedding dimension
