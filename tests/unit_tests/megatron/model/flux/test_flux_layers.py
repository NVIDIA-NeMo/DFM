# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

from dfm.src.megatron.model.flux.layers import (
    EmbedND,
    MLPEmbedder,
    TimeStepEmbedder,
    Timesteps,
    get_timestep_embedding,
    rope,
)


pytestmark = [pytest.mark.unit]


class TestRopeFunction:
    """Test the rope function for rotary position embeddings."""

    def test_rope_basic_shape(self):
        """Test rope function returns correct shape."""
        pos = torch.tensor([[1.0, 2.0, 3.0]])
        dim = 64
        theta = 10000

        result = rope(pos, dim, theta)

        # Output shape should be [..., dim//2]
        assert result.shape == (1, 3, dim // 2)

    def test_rope_requires_even_dimension(self):
        """Test that rope function requires even dimension."""
        pos = torch.tensor([[1.0, 2.0]])
        dim = 63  # Odd dimension
        theta = 10000

        with pytest.raises(AssertionError, match="The dimension must be even"):
            rope(pos, dim, theta)

    def test_rope_output_is_float(self):
        """Test that rope output is float32."""
        pos = torch.tensor([[1.0, 2.0]], dtype=torch.float64)
        dim = 64
        theta = 10000

        result = rope(pos, dim, theta)

        assert result.dtype == torch.float32

    def test_rope_with_different_thetas(self):
        """Test rope function with different theta values."""
        pos = torch.tensor([[1.0]])
        dim = 32

        result_10k = rope(pos, dim, 10000)
        result_5k = rope(pos, dim, 5000)

        # Different theta values should produce different results
        assert not torch.allclose(result_10k, result_5k)

    def test_rope_with_zero_positions(self):
        """Test rope function with zero positions."""
        pos = torch.zeros(2, 3)
        dim = 64
        theta = 10000

        result = rope(pos, dim, theta)

        # Zero positions should produce zero embeddings
        assert torch.allclose(result, torch.zeros_like(result), atol=1e-6)

    def test_rope_batched_input(self):
        """Test rope function with batched input."""
        batch_size = 4
        seq_len = 16
        pos = torch.randn(batch_size, seq_len)
        dim = 64
        theta = 10000

        result = rope(pos, dim, theta)

        assert result.shape == (batch_size, seq_len, dim // 2)
        assert not torch.isnan(result).any()


class TestEmbedND:
    """Test the EmbedND class for N-dimensional rotary embeddings."""

    def test_embednd_initialization(self):
        """Test EmbedND initialization."""
        dim = 128
        theta = 10000
        axes_dim = [16, 56, 56]

        embed = EmbedND(dim, theta, axes_dim)

        assert embed.dim == dim
        assert embed.theta == theta
        assert embed.axes_dim == axes_dim

    def test_embednd_forward_shape(self):
        """Test EmbedND forward pass output shape."""
        dim = 128
        theta = 10000
        axes_dim = [16, 56, 56]
        batch_size = 2
        seq_len = 100
        n_axes = 3

        embed = EmbedND(dim, theta, axes_dim)
        ids = torch.randn(batch_size, seq_len, n_axes)

        output = embed(ids)

        # Output should be reshaped and stacked
        assert output.ndim == 4
        assert not torch.isnan(output).any()

    def test_embednd_with_different_axes_dims(self):
        """Test EmbedND with different axes dimensions."""
        dim = 256
        theta = 10000
        axes_dim = [8, 32, 32]

        embed = EmbedND(dim, theta, axes_dim)
        ids = torch.randn(1, 50, 3)

        output = embed(ids)

        assert output.ndim == 4
        assert not torch.isnan(output).any()

    def test_embednd_output_is_finite(self):
        """Test that EmbedND output contains no inf values."""
        embed = EmbedND(128, 10000, [16, 56, 56])
        ids = torch.randn(2, 100, 3)

        output = embed(ids)

        assert torch.isfinite(output).all()


class TestMLPEmbedder:
    """Test the MLPEmbedder class."""

    def test_mlpembedder_initialization(self):
        """Test MLPEmbedder initialization."""
        in_dim = 256
        hidden_dim = 512

        embedder = MLPEmbedder(in_dim, hidden_dim)

        assert embedder.in_layer.in_features == in_dim
        assert embedder.in_layer.out_features == hidden_dim
        assert embedder.out_layer.in_features == hidden_dim
        assert embedder.out_layer.out_features == hidden_dim
        assert isinstance(embedder.silu, torch.nn.SiLU)

    def test_mlpembedder_forward_shape(self):
        """Test MLPEmbedder forward pass output shape."""
        in_dim = 256
        hidden_dim = 512
        batch_size = 4

        embedder = MLPEmbedder(in_dim, hidden_dim)
        x = torch.randn(batch_size, in_dim)

        output = embedder(x)

        assert output.shape == (batch_size, hidden_dim)

    def test_mlpembedder_forward_with_different_input_shapes(self):
        """Test MLPEmbedder with different input shapes."""
        embedder = MLPEmbedder(128, 256)

        # 2D input
        x_2d = torch.randn(4, 128)
        out_2d = embedder(x_2d)
        assert out_2d.shape == (4, 256)

        # 3D input
        x_3d = torch.randn(2, 8, 128)
        out_3d = embedder(x_3d)
        assert out_3d.shape == (2, 8, 256)

    def test_mlpembedder_output_is_finite(self):
        """Test that MLPEmbedder output is finite."""
        embedder = MLPEmbedder(256, 512)
        x = torch.randn(10, 256)

        output = embedder(x)

        assert torch.isfinite(output).all()

    def test_mlpembedder_has_bias(self):
        """Test that MLPEmbedder layers have bias."""
        embedder = MLPEmbedder(256, 512)

        assert embedder.in_layer.bias is not None
        assert embedder.out_layer.bias is not None


class TestGetTimestepEmbedding:
    """Test the get_timestep_embedding function."""

    def test_get_timestep_embedding_basic_shape(self):
        """Test get_timestep_embedding returns correct shape."""
        timesteps = torch.tensor([0, 100, 500, 999])
        embedding_dim = 128

        emb = get_timestep_embedding(timesteps, embedding_dim)

        assert emb.shape == (4, embedding_dim)

    def test_get_timestep_embedding_requires_1d_timesteps(self):
        """Test that get_timestep_embedding requires 1D timesteps."""
        timesteps = torch.tensor([[0, 1], [2, 3]])  # 2D
        embedding_dim = 128

        with pytest.raises(AssertionError, match="Timesteps should be a 1d-array"):
            get_timestep_embedding(timesteps, embedding_dim)

    def test_get_timestep_embedding_with_flip_sin_cos(self):
        """Test get_timestep_embedding with flip_sin_to_cos option."""
        timesteps = torch.tensor([100.0])
        embedding_dim = 64

        emb_flipped = get_timestep_embedding(timesteps, embedding_dim, flip_sin_to_cos=True)
        emb_not_flipped = get_timestep_embedding(timesteps, embedding_dim, flip_sin_to_cos=False)

        # Results should be different with different flip settings
        assert not torch.allclose(emb_flipped, emb_not_flipped)

    def test_get_timestep_embedding_with_scale(self):
        """Test get_timestep_embedding with different scales."""
        timesteps = torch.tensor([100.0])
        embedding_dim = 64

        emb_scale1 = get_timestep_embedding(timesteps, embedding_dim, scale=1.0)
        emb_scale2 = get_timestep_embedding(timesteps, embedding_dim, scale=2.0)

        # Scale affects the input to sin/cos, so outputs should differ but not be simple multiples
        # Verify that they are different and have the same shape
        assert emb_scale1.shape == emb_scale2.shape
        assert not torch.allclose(emb_scale1, emb_scale2, atol=1e-5)

    def test_get_timestep_embedding_with_odd_dimension(self):
        """Test get_timestep_embedding with odd embedding dimension."""
        timesteps = torch.tensor([100.0])
        embedding_dim = 63  # Odd

        emb = get_timestep_embedding(timesteps, embedding_dim)

        # Should be zero-padded to the correct dimension
        assert emb.shape == (1, embedding_dim)

    def test_get_timestep_embedding_fractional_timesteps(self):
        """Test get_timestep_embedding with fractional timesteps."""
        timesteps = torch.tensor([0.5, 1.5, 2.5])
        embedding_dim = 64

        emb = get_timestep_embedding(timesteps, embedding_dim)

        assert emb.shape == (3, embedding_dim)
        assert torch.isfinite(emb).all()

    def test_get_timestep_embedding_zero_timestep(self):
        """Test get_timestep_embedding with zero timestep."""
        timesteps = torch.tensor([0.0])
        embedding_dim = 64

        emb = get_timestep_embedding(timesteps, embedding_dim)

        assert emb.shape == (1, embedding_dim)
        assert torch.isfinite(emb).all()

    def test_get_timestep_embedding_max_period(self):
        """Test get_timestep_embedding with different max_period values."""
        timesteps = torch.tensor([100.0])
        embedding_dim = 64

        emb_10k = get_timestep_embedding(timesteps, embedding_dim, max_period=10000)
        emb_5k = get_timestep_embedding(timesteps, embedding_dim, max_period=5000)

        # Different max_period should produce different embeddings
        assert not torch.allclose(emb_10k, emb_5k)


class TestTimesteps:
    """Test the Timesteps module."""

    def test_timesteps_initialization(self):
        """Test Timesteps module initialization."""
        embedding_dim = 128

        timesteps = Timesteps(embedding_dim)

        assert timesteps.embedding_dim == embedding_dim
        assert timesteps.flip_sin_to_cos is True
        assert timesteps.downscale_freq_shift == 0
        assert timesteps.scale == 1
        assert timesteps.max_period == 10000

    def test_timesteps_initialization_custom(self):
        """Test Timesteps module with custom parameters."""
        timesteps = Timesteps(
            embedding_dim=256, flip_sin_to_cos=False, downscale_freq_shift=1.0, scale=2.0, max_period=5000
        )

        assert timesteps.embedding_dim == 256
        assert timesteps.flip_sin_to_cos is False
        assert timesteps.downscale_freq_shift == 1.0
        assert timesteps.scale == 2.0
        assert timesteps.max_period == 5000

    def test_timesteps_forward(self):
        """Test Timesteps forward pass."""
        timesteps_module = Timesteps(embedding_dim=128)
        timesteps_input = torch.tensor([0.0, 100.0, 500.0])

        output = timesteps_module(timesteps_input)

        assert output.shape == (3, 128)
        assert torch.isfinite(output).all()

    def test_timesteps_is_nn_module(self):
        """Test that Timesteps is an nn.Module."""
        timesteps = Timesteps(128)

        assert isinstance(timesteps, torch.nn.Module)


class TestTimeStepEmbedder:
    """Test the TimeStepEmbedder class."""

    def test_timestepembedder_initialization(self):
        """Test TimeStepEmbedder initialization."""
        embedding_dim = 256
        hidden_dim = 512

        embedder = TimeStepEmbedder(embedding_dim, hidden_dim)

        assert isinstance(embedder.time_proj, Timesteps)
        assert isinstance(embedder.time_embedder, MLPEmbedder)
        assert embedder.time_proj.embedding_dim == embedding_dim

    def test_timestepembedder_forward_shape(self):
        """Test TimeStepEmbedder forward pass output shape."""
        embedding_dim = 256
        hidden_dim = 512
        batch_size = 4

        embedder = TimeStepEmbedder(embedding_dim, hidden_dim)
        timesteps = torch.tensor([0.0, 100.0, 500.0, 999.0])

        output = embedder(timesteps)

        assert output.shape == (batch_size, hidden_dim)

    def test_timestepembedder_with_custom_params(self):
        """Test TimeStepEmbedder with custom parameters."""
        embedder = TimeStepEmbedder(
            embedding_dim=128,
            hidden_dim=256,
            flip_sin_to_cos=False,
            downscale_freq_shift=1.0,
            scale=2.0,
            max_period=5000,
        )

        timesteps = torch.tensor([100.0, 200.0])
        output = embedder(timesteps)

        assert output.shape == (2, 256)
        assert torch.isfinite(output).all()

    def test_timestepembedder_output_is_finite(self):
        """Test that TimeStepEmbedder output is finite."""
        embedder = TimeStepEmbedder(256, 512)
        timesteps = torch.randn(10).abs() * 1000

        output = embedder(timesteps)

        assert torch.isfinite(output).all()

    def test_timestepembedder_is_nn_module(self):
        """Test that TimeStepEmbedder is an nn.Module."""
        embedder = TimeStepEmbedder(256, 512)

        assert isinstance(embedder, torch.nn.Module)

    def test_timestepembedder_components_connected(self):
        """Test that TimeStepEmbedder components are properly connected."""
        embedder = TimeStepEmbedder(256, 512)
        timesteps = torch.tensor([100.0])

        # Forward through time_proj
        proj_output = embedder.time_proj(timesteps)
        assert proj_output.shape == (1, 256)

        # Forward through time_embedder
        emb_output = embedder.time_embedder(proj_output)
        assert emb_output.shape == (1, 512)

        # Full forward pass
        full_output = embedder(timesteps)
        assert torch.allclose(full_output, emb_output, atol=1e-6)


class TestLayersIntegration:
    """Integration tests for layers module."""

    def test_rope_to_embednd_pipeline(self):
        """Test using rope in EmbedND."""
        embed = EmbedND(dim=128, theta=10000, axes_dim=[16, 56, 56])
        ids = torch.randn(2, 50, 3)

        # This internally uses rope function
        output = embed(ids)

        assert output.ndim == 4
        assert torch.isfinite(output).all()

    def test_timestep_embedding_pipeline(self):
        """Test complete timestep embedding pipeline."""
        # Create embedder
        embedder = TimeStepEmbedder(embedding_dim=256, hidden_dim=3072)

        # Generate timesteps
        timesteps = torch.linspace(0, 1000, 10)

        # Get embeddings
        embeddings = embedder(timesteps)

        assert embeddings.shape == (10, 3072)
        assert torch.isfinite(embeddings).all()
        # Different timesteps should produce different embeddings
        assert not torch.allclose(embeddings[0], embeddings[-1])

    def test_mlp_embedder_in_timestep_embedder(self):
        """Test that MLPEmbedder works correctly within TimeStepEmbedder."""
        embedder = TimeStepEmbedder(128, 256)

        timesteps = torch.tensor([0.0, 500.0, 1000.0])
        output = embedder(timesteps)

        # Should pass through both time_proj and time_embedder
        assert output.shape == (3, 256)


class TestLayersEdgeCases:
    """Test edge cases for layers module."""

    def test_rope_with_large_positions(self):
        """Test rope function with large position values."""
        pos = torch.tensor([[1000.0, 2000.0, 3000.0]])
        dim = 64
        theta = 10000

        result = rope(pos, dim, theta)

        assert torch.isfinite(result).all()

    def test_embednd_with_small_batch(self):
        """Test EmbedND with batch size 1."""
        embed = EmbedND(64, 10000, [8, 16, 16])
        ids = torch.randn(1, 10, 3)

        output = embed(ids)

        assert output.shape[1] == 1  # Batch dimension

    def test_mlpembedder_with_zero_input(self):
        """Test MLPEmbedder with zero input."""
        embedder = MLPEmbedder(256, 512)
        x = torch.zeros(4, 256)

        output = embedder(x)

        assert output.shape == (4, 512)
        assert torch.isfinite(output).all()

    def test_timestep_embedding_with_negative_timesteps(self):
        """Test get_timestep_embedding with negative timesteps."""
        timesteps = torch.tensor([-100.0, -50.0, 0.0, 50.0])
        embedding_dim = 64

        emb = get_timestep_embedding(timesteps, embedding_dim)

        assert emb.shape == (4, embedding_dim)
        assert torch.isfinite(emb).all()

    def test_timestepembedder_with_very_small_timesteps(self):
        """Test TimeStepEmbedder with very small fractional timesteps."""
        embedder = TimeStepEmbedder(256, 512)
        timesteps = torch.tensor([0.001, 0.01, 0.1])

        output = embedder(timesteps)

        assert output.shape == (3, 512)
        assert torch.isfinite(output).all()

    def test_rope_different_dimensions(self):
        """Test rope with various even dimensions."""
        pos = torch.tensor([[1.0, 2.0]])

        for dim in [32, 64, 128, 256]:
            result = rope(pos, dim, 10000)
            assert result.shape[-1] == dim // 2
            assert torch.isfinite(result).all()
