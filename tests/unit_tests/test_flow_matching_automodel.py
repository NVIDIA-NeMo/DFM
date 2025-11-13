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

import math

import numpy as np
import pytest
import torch

from dfm.src.Automodel.flow_matching.time_shift_utils import (
    compute_density_for_timestep_sampling,
    get_flow_match_loss_weight,
    time_shift,
)


class TestTimeShift:
    """Test time shifting functions for flow matching"""

    def test_time_shift_constant_mode_default(self):
        """Test constant shift mode with default constant=3.0 (Pika default)"""
        # Test with t in range [0.1, 0.9] to avoid division by zero
        t = torch.tensor([0.1, 0.3, 0.5, 0.7, 0.9])
        image_seq_len = 1024  # dummy value, not used in constant mode
        constant = 3.0

        sigma = time_shift(
            t=t,
            image_seq_len=image_seq_len,
            shift_type="constant",
            constant=constant,
        )

        # Verify formula: σ = constant / (constant + (1/t - 1))
        expected = constant / (constant + (1.0 / t - 1.0))

        assert torch.allclose(sigma, expected, atol=1e-5), f"Expected {expected}, got {sigma}"
        assert sigma.shape == t.shape, "Output shape should match input shape"
        print("✓ Constant mode with default constant=3.0 passed")

    def test_time_shift_constant_mode_different_constants(self):
        """Test constant shift mode with different constant values"""
        t = torch.tensor([0.2, 0.5, 0.8])
        image_seq_len = 1024

        for constant in [1.0, 3.0, 5.0, 10.0]:
            sigma = time_shift(
                t=t,
                image_seq_len=image_seq_len,
                shift_type="constant",
                constant=constant,
            )

            # Verify all values are in [0, 1]
            assert torch.all(sigma >= 0) and torch.all(sigma <= 1), f"Sigma not in [0,1] for constant={constant}"

        print("✓ Constant mode with different constants passed")

    def test_time_shift_linear_mode(self):
        """Test linear interpolation mode"""
        t = torch.tensor([0.2, 0.5, 0.8])
        image_seq_len = 4096
        base_shift = 0.5
        max_shift = 1.15

        sigma = time_shift(
            t=t,
            image_seq_len=image_seq_len,
            shift_type="linear",
            base_shift=base_shift,
            max_shift=max_shift,
        )

        # Verify linear interpolation: mu = base_shift + (max_shift - base_shift) * (image_seq_len / 4096)
        mu = base_shift + (max_shift - base_shift) * (image_seq_len / 4096)
        expected = math.exp(mu) / (math.exp(mu) + (1.0 / t - 1.0))

        assert torch.allclose(sigma, expected, atol=1e-5), f"Expected {expected}, got {sigma}"
        assert torch.all(sigma >= 0) and torch.all(sigma <= 1), "Sigma not in [0,1]"
        print("✓ Linear mode passed")

    def test_time_shift_sqrt_mode(self):
        """Test sqrt scaling (Flux-style) mode"""
        t = torch.tensor([0.2, 0.5, 0.8])
        image_seq_len = 128 * 128  # 128x128 latent space

        sigma = time_shift(
            t=t,
            image_seq_len=image_seq_len,
            shift_type="sqrt",
        )

        # Verify sqrt formula: mu = max(1.0, sqrt(image_seq_len / (128*128)) * 3.0)
        mu = np.maximum(1.0, np.sqrt(image_seq_len / (128.0 * 128.0)) * 3.0)
        expected = mu / (mu + (1.0 / t.numpy() - 1.0))

        assert torch.allclose(sigma, torch.tensor(expected, dtype=sigma.dtype), atol=1e-5)
        assert torch.all(sigma >= 0) and torch.all(sigma <= 1), "Sigma not in [0,1]"
        print("✓ Sqrt mode passed")

    def test_time_shift_output_range(self):
        """Verify all outputs are in [0, 1] range for all modes"""
        t_values = torch.linspace(0.01, 0.99, 100)  # Avoid exact 0 and 1
        image_seq_lens = [512, 1024, 2048, 4096, 8192]
        modes = ["constant", "linear", "sqrt"]

        for mode in modes:
            for seq_len in image_seq_lens:
                sigma = time_shift(t=t_values, image_seq_len=seq_len, shift_type=mode)

                assert torch.all(sigma >= 0) and torch.all(sigma <= 1), (
                    f"Sigma out of [0,1] for mode={mode}, seq_len={seq_len}"
                )

        print("✓ Output range test passed for all modes")

    def test_time_shift_edge_cases(self):
        """Test edge cases: very small t, very large t, extreme seq_len"""
        # Very small t (close to 0)
        t_small = torch.tensor([0.001, 0.01])
        sigma_small = time_shift(t=t_small, image_seq_len=1024, shift_type="constant", constant=3.0)
        assert torch.all(torch.isfinite(sigma_small)), "Non-finite values for small t"

        # Very large t (close to 1)
        t_large = torch.tensor([0.99, 0.999])
        sigma_large = time_shift(t=t_large, image_seq_len=1024, shift_type="constant", constant=3.0)
        assert torch.all(torch.isfinite(sigma_large)), "Non-finite values for large t"

        # Extreme sequence lengths
        t_normal = torch.tensor([0.5])
        for seq_len in [64, 16384]:
            sigma = time_shift(t=t_normal, image_seq_len=seq_len, shift_type="sqrt")
            assert torch.all(torch.isfinite(sigma)), f"Non-finite values for seq_len={seq_len}"

        print("✓ Edge cases test passed")

    def test_time_shift_no_shift_mode(self):
        """Test that invalid shift_type returns original t"""
        t = torch.tensor([0.1, 0.3, 0.5, 0.7, 0.9])
        sigma = time_shift(t=t, image_seq_len=1024, shift_type="none")

        # Should return original t when shift_type is not recognized
        assert torch.allclose(sigma, t), "Should return original t for invalid shift_type"
        print("✓ No-shift mode passed")


class TestTimestepSampling:
    """Test timestep sampling distributions"""

    def test_uniform_sampling(self):
        """Test uniform distribution [0, 1]"""
        batch_size = 1000

        u = compute_density_for_timestep_sampling(
            weighting_scheme="uniform",
            batch_size=batch_size,
        )

        # Check shape
        assert u.shape == (batch_size,), f"Expected shape ({batch_size},), got {u.shape}"

        # Check range [0, 1]
        assert torch.all(u >= 0) and torch.all(u <= 1), "Values not in [0, 1]"

        # Check uniform distribution properties (approximate)
        mean = u.mean().item()
        assert 0.45 <= mean <= 0.55, f"Mean {mean} not close to 0.5 for uniform distribution"

        std = u.std().item()
        # Uniform [0,1] has std = 1/sqrt(12) ≈ 0.289
        expected_std = 1.0 / math.sqrt(12)
        assert abs(std - expected_std) < 0.05, f"Std {std} not close to {expected_std}"

        print("✓ Uniform sampling passed")

    def test_logit_normal_sampling(self):
        """Test logit-normal distribution (SD3-style)"""
        batch_size = 10000
        logit_mean = 0.0
        logit_std = 1.0

        u = compute_density_for_timestep_sampling(
            weighting_scheme="logit_normal",
            batch_size=batch_size,
            logit_mean=logit_mean,
            logit_std=logit_std,
        )

        # Check shape
        assert u.shape == (batch_size,), f"Expected shape ({batch_size},), got {u.shape}"

        # Check range [0, 1] (sigmoid output)
        assert torch.all(u >= 0) and torch.all(u <= 1), "Values not in [0, 1]"

        # For logit_mean=0, logit_std=1, the sigmoid(N(0,1)) should be roughly centered at 0.5
        mean = u.mean().item()
        assert 0.45 <= mean <= 0.55, f"Mean {mean} not close to 0.5 for logit_normal"

        print("✓ Logit-normal sampling passed")

    def test_logit_normal_different_parameters(self):
        """Test logit-normal with different mean/std parameters"""
        batch_size = 5000

        # Test with positive mean (shifts distribution toward 1)
        u_positive = compute_density_for_timestep_sampling(
            weighting_scheme="logit_normal",
            batch_size=batch_size,
            logit_mean=2.0,
            logit_std=1.0,
        )
        mean_positive = u_positive.mean().item()
        assert mean_positive > 0.5, "Positive logit_mean should shift mean toward 1"

        # Test with negative mean (shifts distribution toward 0)
        u_negative = compute_density_for_timestep_sampling(
            weighting_scheme="logit_normal",
            batch_size=batch_size,
            logit_mean=-2.0,
            logit_std=1.0,
        )
        mean_negative = u_negative.mean().item()
        assert mean_negative < 0.5, "Negative logit_mean should shift mean toward 0"

        # Test with larger std (more spread)
        u_large_std = compute_density_for_timestep_sampling(
            weighting_scheme="logit_normal",
            batch_size=batch_size,
            logit_mean=0.0,
            logit_std=2.0,
        )
        std_large = u_large_std.std().item()

        u_small_std = compute_density_for_timestep_sampling(
            weighting_scheme="logit_normal",
            batch_size=batch_size,
            logit_mean=0.0,
            logit_std=0.5,
        )
        std_small = u_small_std.std().item()

        assert std_large > std_small, "Larger logit_std should have larger std"

        print("✓ Logit-normal with different parameters passed")

    def test_mode_sampling(self):
        """Test mode-based sampling"""
        batch_size = 1000
        mode_scale = 1.29

        u = compute_density_for_timestep_sampling(
            weighting_scheme="mode",
            batch_size=batch_size,
            mode_scale=mode_scale,
        )

        # Check shape
        assert u.shape == (batch_size,), f"Expected shape ({batch_size},), got {u.shape}"

        # Check range [0, 1]
        assert torch.all(u >= 0) and torch.all(u <= 1), "Values not in [0, 1]"

        # Mode sampling formula: u_out = 1 - u - mode_scale * (cos(π*u/2)^2 - 1 + u)
        # This creates a specific distribution that concentrates samples around certain timesteps

        print("✓ Mode sampling passed")

    def test_sampling_batch_size(self):
        """Test different batch sizes"""
        for batch_size in [1, 10, 100, 1000]:
            u = compute_density_for_timestep_sampling(
                weighting_scheme="uniform",
                batch_size=batch_size,
            )

            assert u.shape == (batch_size,), f"Expected shape ({batch_size},), got {u.shape}"

        print("✓ Batch size test passed")

    def test_sampling_determinism_with_seed(self):
        """Test that sampling is deterministic with same seed"""
        batch_size = 100

        # Set seed and sample
        torch.manual_seed(42)
        u1 = compute_density_for_timestep_sampling(
            weighting_scheme="uniform",
            batch_size=batch_size,
        )

        # Reset seed and sample again
        torch.manual_seed(42)
        u2 = compute_density_for_timestep_sampling(
            weighting_scheme="uniform",
            batch_size=batch_size,
        )

        assert torch.allclose(u1, u2), "Results should be identical with same seed"

        print("✓ Determinism test passed")


class TestLossWeighting:
    """Test flow matching loss weights"""

    def test_loss_weight_formula(self):
        """Test weight = 1 + shift * sigma"""
        sigma = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0])
        shift = 3.0

        weight = get_flow_match_loss_weight(sigma, shift)

        # Expected: weight = 1 + shift * sigma
        expected = 1.0 + shift * sigma

        assert torch.allclose(weight, expected, atol=1e-6), f"Expected {expected}, got {weight}"
        print("✓ Loss weight formula test passed")

    def test_loss_weight_shape_broadcasting(self):
        """Test broadcasting for 5D tensors (B, C, T, H, W)"""
        batch_size = 2
        sigma = torch.tensor([0.3, 0.7])  # Shape: (2,)
        shift = 3.0

        weight = get_flow_match_loss_weight(sigma, shift)

        # Weight should have same shape as sigma
        assert weight.shape == sigma.shape, f"Expected shape {sigma.shape}, got {weight.shape}"

        # Now test with reshaped sigma for 5D broadcasting
        sigma_5d = sigma.view(-1, 1, 1, 1, 1)  # Shape: (2, 1, 1, 1, 1)
        weight_5d = get_flow_match_loss_weight(sigma_5d, shift)

        assert weight_5d.shape == sigma_5d.shape, f"Expected shape {sigma_5d.shape}, got {weight_5d.shape}"

        print("✓ Shape broadcasting test passed")

    def test_loss_weight_different_shifts(self):
        """Test with different shift values (1.0, 3.0, 5.0)"""
        sigma = torch.tensor([0.5])

        for shift in [1.0, 3.0, 5.0, 10.0]:
            weight = get_flow_match_loss_weight(sigma, shift)
            expected = 1.0 + shift * sigma

            assert torch.allclose(weight, expected, atol=1e-6), f"Failed for shift={shift}"

            # Verify weight increases with shift
            assert weight > 1.0, f"Weight should be > 1.0 for positive sigma, got {weight}"

        print("✓ Different shifts test passed")

    def test_loss_weight_edge_cases(self):
        """Test edge cases: sigma=0, sigma=1"""
        shift = 3.0

        # sigma = 0 → weight = 1
        weight_zero = get_flow_match_loss_weight(torch.tensor([0.0]), shift)
        assert torch.allclose(weight_zero, torch.tensor([1.0])), "Weight should be 1.0 for sigma=0"

        # sigma = 1 → weight = 1 + shift
        weight_one = get_flow_match_loss_weight(torch.tensor([1.0]), shift)
        expected = torch.tensor([1.0 + shift])
        assert torch.allclose(weight_one, expected), f"Weight should be {expected} for sigma=1"

        print("✓ Edge cases test passed")


class TestFlowMatchingMath:
    """Test core flow matching equations"""

    def test_noisy_latent_generation(self):
        """Test x_t = (1-σ)x_0 + σ*ε (manual flow matching)"""
        batch_size = 2
        channels = 16
        frames = 1
        height = 8
        width = 8

        # Create clean latents and noise
        x_0 = torch.randn(batch_size, channels, frames, height, width)
        noise = torch.randn_like(x_0)

        # Test with different sigma values
        for sigma_val in [0.0, 0.25, 0.5, 0.75, 1.0]:
            sigma = torch.tensor([sigma_val, sigma_val])
            sigma_reshaped = sigma.view(-1, 1, 1, 1, 1)

            # Apply flow matching: x_t = (1-σ)x_0 + σ*ε
            x_t = (1.0 - sigma_reshaped) * x_0 + sigma_reshaped * noise

            # Verify shape
            assert x_t.shape == x_0.shape, "Output shape should match input shape"

            # Verify edge cases
            if sigma_val == 0.0:
                assert torch.allclose(x_t, x_0, atol=1e-6), "x_t should equal x_0 when sigma=0"
            elif sigma_val == 1.0:
                assert torch.allclose(x_t, noise, atol=1e-6), "x_t should equal noise when sigma=1"

        print("✓ Noisy latent generation test passed")

    def test_flow_matching_target(self):
        """Test target = ε - x_0 (velocity formulation)"""
        batch_size = 2
        channels = 16
        frames = 1
        height = 8
        width = 8

        x_0 = torch.randn(batch_size, channels, frames, height, width)
        noise = torch.randn_like(x_0)

        # Flow matching target (velocity)
        target = noise - x_0

        # Verify shape
        assert target.shape == x_0.shape, "Target shape should match input shape"

        # Verify formula
        expected = noise - x_0
        assert torch.allclose(target, expected), "Target should be ε - x_0"

        print("✓ Flow matching target test passed")

    def test_sigma_normalization(self):
        """Test sigma clamping and normalization"""
        # Test clamping to [0, 1]
        sigma_raw = torch.tensor([-0.5, 0.0, 0.5, 1.0, 1.5])
        sigma_clamped = torch.clamp(sigma_raw, 0.0, 1.0)

        expected = torch.tensor([0.0, 0.0, 0.5, 1.0, 1.0])
        assert torch.allclose(sigma_clamped, expected), "Sigma should be clamped to [0, 1]"

        print("✓ Sigma normalization test passed")

    def test_flow_matching_interpolation_property(self):
        """Test that x_t is a proper interpolation between x_0 and noise"""
        batch_size = 2
        channels = 16
        frames = 1
        height = 8
        width = 8

        x_0 = torch.randn(batch_size, channels, frames, height, width)
        noise = torch.randn_like(x_0)

        # Test multiple sigma values
        sigma_values = torch.linspace(0, 1, 11)

        for sigma_val in sigma_values:
            sigma = torch.full((batch_size,), sigma_val)
            sigma_reshaped = sigma.view(-1, 1, 1, 1, 1)

            x_t = (1.0 - sigma_reshaped) * x_0 + sigma_reshaped * noise

            # x_t should be finite
            assert torch.all(torch.isfinite(x_t)), f"x_t contains non-finite values for sigma={sigma_val}"

            # For sigma=0.5, x_t should be roughly between x_0 and noise
            if abs(sigma_val - 0.5) < 0.01:
                # Check that x_t is reasonably between x_0 and noise
                x_t_mean = x_t.mean().item()
                x_0_mean = x_0.mean().item()
                noise_mean = noise.mean().item()

                # The mean should be between the two (approximately)
                min_mean = min(x_0_mean, noise_mean)
                max_mean = max(x_0_mean, noise_mean)

                # Allow some tolerance
                tolerance = abs(max_mean - min_mean) * 0.3
                assert min_mean - tolerance <= x_t_mean <= max_mean + tolerance, (
                    f"x_t mean {x_t_mean} not between x_0 mean {x_0_mean} and noise mean {noise_mean}"
                )

        print("✓ Flow matching interpolation property test passed")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
