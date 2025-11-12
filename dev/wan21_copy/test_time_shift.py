#!/usr/bin/env python3
"""
Real unit tests for time_shift_utils.py

Tests the actual functions in the module.
"""

import pytest
import torch
import sys
import math

# Add uploads directory to path
sys.path.insert(0, '/mnt/user-data/uploads')

from time_shift_utils import (
    time_shift,
    compute_density_for_timestep_sampling,
    get_flow_match_loss_weight
)


class TestTimeShift:
    """Test the time_shift function"""
    
    def setup_method(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def test_constant_shift(self):
        """Test constant shift mode (default)"""
        t = torch.tensor([0.1, 0.5, 0.9], device=self.device)
        image_seq_len = 1024
        constant = 3.0
        
        sigma = time_shift(
            t=t,
            image_seq_len=image_seq_len,
            shift_type='constant',
            constant=constant
        )
        
        # Should match formula: constant / (constant + (1/t - 1))
        expected = constant / (constant + (1.0 / t - 1.0))
        
        assert torch.allclose(sigma, expected, atol=1e-6)
    
    def test_constant_shift_default(self):
        """Test that default constant is 3.0"""
        t = torch.tensor([0.5], device=self.device)
        image_seq_len = 1024
        
        sigma = time_shift(t, image_seq_len, shift_type='constant')
        
        # With t=0.5, constant=3: sigma = 3/(3 + 1) = 0.75
        expected = 3.0 / 4.0
        assert torch.isclose(sigma, torch.tensor(expected, device=self.device), atol=1e-6)
    
    def test_linear_shift(self):
        """Test linear shift mode"""
        t = torch.tensor([0.5], device=self.device)
        image_seq_len = 4096  # Standard size
        base_shift = 0.5
        max_shift = 1.15
        
        sigma = time_shift(
            t=t,
            image_seq_len=image_seq_len,
            shift_type='linear',
            base_shift=base_shift,
            max_shift=max_shift
        )
        
        # mu should be max_shift since image_seq_len == 4096
        mu = base_shift + (max_shift - base_shift) * (image_seq_len / 4096)
        assert abs(mu - max_shift) < 1e-6
        
        expected = math.exp(mu) / (math.exp(mu) + (1 / 0.5 - 1))
        assert torch.isclose(sigma, torch.tensor(expected, device=self.device), atol=1e-5)
    
    def test_sqrt_shift(self):
        """Test sqrt shift mode (Flux-style)"""
        t = torch.tensor([0.5], device=self.device)
        image_seq_len = 128 * 128  # Standard latent space
        
        sigma = time_shift(
            t=t,
            image_seq_len=image_seq_len,
            shift_type='sqrt'
        )
        
        # mu should be 3.0 for 128x128
        import numpy as np
        mu = np.maximum(1.0, np.sqrt(image_seq_len / (128.0 * 128.0)) * 3.0)
        assert abs(mu - 3.0) < 1e-6
        
        expected = mu / (mu + (1 / 0.5 - 1))
        assert torch.isclose(sigma, torch.tensor(expected, device=self.device), atol=1e-5)
    
    def test_no_shift(self):
        """Test with no shift (returns original t)"""
        t = torch.tensor([0.1, 0.5, 0.9], device=self.device)
        image_seq_len = 1024
        
        sigma = time_shift(
            t=t,
            image_seq_len=image_seq_len,
            shift_type='none'
        )
        
        # Should return original t
        assert torch.allclose(sigma, t, atol=1e-6)
    
    def test_sigma_range(self):
        """Test that sigma stays in valid range [0, 1]"""
        t = torch.linspace(0.01, 0.99, 100, device=self.device)
        image_seq_len = 1024
        
        for shift_type in ['constant', 'linear', 'sqrt']:
            sigma = time_shift(t, image_seq_len, shift_type=shift_type)
            
            assert (sigma >= 0).all(), f"{shift_type}: sigma below 0"
            assert (sigma <= 1).all(), f"{shift_type}: sigma above 1"
    
    def test_monotonicity(self):
        """Test that sigma increases monotonically with t"""
        t = torch.linspace(0.01, 0.99, 100, device=self.device)
        image_seq_len = 1024
        
        for shift_type in ['constant', 'linear', 'sqrt']:
            sigma = time_shift(t, image_seq_len, shift_type=shift_type)
            
            # Should be monotonically increasing
            diffs = sigma[1:] - sigma[:-1]
            assert (diffs >= -1e-6).all(), f"{shift_type}: not monotonic"
    
    def test_different_sequence_lengths(self):
        """Test with different sequence lengths"""
        t = torch.tensor([0.5], device=self.device)
        
        for seq_len in [256, 1024, 4096, 16384]:
            sigma = time_shift(t, seq_len, shift_type='constant', constant=3.0)
            
            # With constant shift, seq_len shouldn't matter
            expected = 3.0 / 4.0
            assert torch.isclose(sigma, torch.tensor(expected, device=self.device), atol=1e-6)


class TestComputeDensityForTimestepSampling:
    """Test the compute_density_for_timestep_sampling function"""
    
    def setup_method(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        torch.manual_seed(42)  # For reproducibility
    
    def test_uniform_sampling(self):
        """Test uniform sampling"""
        batch_size = 1000
        
        u = compute_density_for_timestep_sampling(
            weighting_scheme='uniform',
            batch_size=batch_size
        )
        
        # Should be on CPU initially
        assert u.device.type == 'cpu'
        
        # Should be in [0, 1]
        assert (u >= 0).all()
        assert (u <= 1).all()
        
        # Should be approximately uniform
        mean = u.mean()
        assert 0.45 < mean < 0.55, f"Mean {mean} not close to 0.5"
    
    def test_logit_normal_sampling(self):
        """Test logit-normal sampling"""
        batch_size = 1000
        logit_mean = 0.0
        logit_std = 1.5
        
        u = compute_density_for_timestep_sampling(
            weighting_scheme='logit_normal',
            batch_size=batch_size,
            logit_mean=logit_mean,
            logit_std=logit_std
        )
        
        # Should be in [0, 1]
        assert (u >= 0).all()
        assert (u <= 1).all()
        
        # Shape should be correct
        assert u.shape == (batch_size,)
    
    def test_mode_sampling(self):
        """Test mode-based sampling"""
        batch_size = 1000
        mode_scale = 1.29
        
        u = compute_density_for_timestep_sampling(
            weighting_scheme='mode',
            batch_size=batch_size,
            mode_scale=mode_scale
        )
        
        # Should be in [0, 1]
        assert (u >= 0).all()
        assert (u <= 1).all()
        
        # Should have different distribution than uniform
        # (mode sampling concentrates around certain values)
        assert u.shape == (batch_size,)
    
    def test_different_batch_sizes(self):
        """Test with various batch sizes"""
        for batch_size in [1, 10, 100, 1000]:
            u = compute_density_for_timestep_sampling(
                weighting_scheme='uniform',
                batch_size=batch_size
            )
            
            assert u.shape == (batch_size,)
            assert (u >= 0).all()
            assert (u <= 1).all()
    
    def test_logit_normal_parameters(self):
        """Test different logit-normal parameters"""
        batch_size = 1000
        
        # Test different means
        for mean in [-1.0, 0.0, 1.0]:
            u = compute_density_for_timestep_sampling(
                weighting_scheme='logit_normal',
                batch_size=batch_size,
                logit_mean=mean,
                logit_std=1.0
            )
            
            assert (u >= 0).all()
            assert (u <= 1).all()
        
        # Test different stds
        for std in [0.5, 1.0, 2.0]:
            u = compute_density_for_timestep_sampling(
                weighting_scheme='logit_normal',
                batch_size=batch_size,
                logit_mean=0.0,
                logit_std=std
            )
            
            assert (u >= 0).all()
            assert (u <= 1).all()
    
    def test_mode_scale_parameter(self):
        """Test different mode scale values"""
        batch_size = 1000
        
        for scale in [0.5, 1.0, 1.29, 2.0]:
            u = compute_density_for_timestep_sampling(
                weighting_scheme='mode',
                batch_size=batch_size,
                mode_scale=scale
            )
            
            assert (u >= 0).all()
            assert (u <= 1).all()
    
    def test_invalid_scheme(self):
        """Test that invalid scheme falls back to uniform"""
        batch_size = 100
        
        u = compute_density_for_timestep_sampling(
            weighting_scheme='invalid_scheme',
            batch_size=batch_size
        )
        
        # Should fall back to uniform
        assert (u >= 0).all()
        assert (u <= 1).all()


class TestGetFlowMatchLossWeight:
    """Test the get_flow_match_loss_weight function"""
    
    def setup_method(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def test_weight_formula(self):
        """Test weight = 1 + shift * sigma"""
        sigma = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0], device=self.device)
        shift = 3.0
        
        weights = get_flow_match_loss_weight(sigma, shift)
        
        expected = 1.0 + shift * sigma
        assert torch.allclose(weights, expected, atol=1e-6)
    
    def test_weight_at_boundaries(self):
        """Test weight values at sigma boundaries"""
        shift = 3.0
        
        # At sigma=0, weight should be 1.0
        sigma_zero = torch.tensor([0.0], device=self.device)
        weight_zero = get_flow_match_loss_weight(sigma_zero, shift)
        assert torch.isclose(weight_zero, torch.tensor(1.0, device=self.device))
        
        # At sigma=1, weight should be 1+shift
        sigma_one = torch.tensor([1.0], device=self.device)
        weight_one = get_flow_match_loss_weight(sigma_one, shift)
        assert torch.isclose(weight_one, torch.tensor(1.0 + shift, device=self.device))
    
    def test_weight_monotonicity(self):
        """Test that weight increases monotonically with sigma"""
        sigma = torch.linspace(0, 1, 100, device=self.device)
        shift = 3.0
        
        weights = get_flow_match_loss_weight(sigma, shift)
        
        # Should be monotonically increasing
        diffs = weights[1:] - weights[:-1]
        assert (diffs >= -1e-6).all()
    
    def test_different_shifts(self):
        """Test with different shift values"""
        sigma = torch.tensor([0.5], device=self.device)
        
        for shift in [0.0, 1.0, 3.0, 5.0, 10.0]:
            weights = get_flow_match_loss_weight(sigma, shift)
            
            expected = 1.0 + shift * 0.5
            assert torch.isclose(weights, torch.tensor(expected, device=self.device), atol=1e-6)
    
    def test_weight_range(self):
        """Test that weights are always >= 1.0"""
        sigma = torch.linspace(0, 1, 100, device=self.device)
        shift = 3.0
        
        weights = get_flow_match_loss_weight(sigma, shift)
        
        # All weights should be >= 1.0
        assert (weights >= 1.0).all()
    
    def test_batch_weights(self):
        """Test with batch of sigma values"""
        sigma = torch.rand(32, device=self.device)
        shift = 3.0
        
        weights = get_flow_match_loss_weight(sigma, shift)
        
        # Shape should match
        assert weights.shape == sigma.shape
        
        # All should be valid
        assert (weights >= 1.0).all()
        assert (weights <= 1.0 + shift).all()
    
    def test_weight_effect_on_loss(self):
        """Test that weights properly scale loss"""
        loss = torch.ones(4, device=self.device)
        sigma = torch.tensor([0.0, 0.33, 0.67, 1.0], device=self.device)
        shift = 3.0
        
        weights = get_flow_match_loss_weight(sigma, shift)
        weighted_loss = loss * weights
        
        # Weighted loss should match expected values
        expected = torch.tensor([1.0, 2.0, 3.0, 4.0], device=self.device)
        assert torch.allclose(weighted_loss, expected, atol=0.01)
    
    def test_zero_shift(self):
        """Test with zero shift (no weighting)"""
        sigma = torch.linspace(0, 1, 100, device=self.device)
        shift = 0.0
        
        weights = get_flow_match_loss_weight(sigma, shift)
        
        # All weights should be 1.0
        assert torch.allclose(weights, torch.ones_like(weights))


class TestTimeshiftUtilsIntegration:
    """Integration tests combining multiple functions"""
    
    def setup_method(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def test_full_pipeline_uniform(self):
        """Test full pipeline with uniform sampling"""
        batch_size = 10
        image_seq_len = 1024
        shift = 3.0
        
        # Sample timesteps
        u = compute_density_for_timestep_sampling(
            weighting_scheme='uniform',
            batch_size=batch_size
        )
        u = u.to(self.device)
        
        # Apply time shift
        sigma = time_shift(u, image_seq_len, shift_type='constant', constant=shift)
        
        # Get loss weights
        weights = get_flow_match_loss_weight(sigma, shift)
        
        # Validate pipeline
        assert sigma.shape == (batch_size,)
        assert weights.shape == (batch_size,)
        assert (sigma >= 0).all() and (sigma <= 1).all()
        assert (weights >= 1.0).all()
    
    def test_full_pipeline_logit_normal(self):
        """Test full pipeline with logit-normal sampling"""
        batch_size = 10
        image_seq_len = 1024
        shift = 3.0
        
        # Sample timesteps
        u = compute_density_for_timestep_sampling(
            weighting_scheme='logit_normal',
            batch_size=batch_size,
            logit_mean=0.0,
            logit_std=1.5
        )
        u = u.to(self.device)
        
        # Apply time shift
        sigma = time_shift(u, image_seq_len, shift_type='constant', constant=shift)
        
        # Get loss weights
        weights = get_flow_match_loss_weight(sigma, shift)
        
        # Validate
        assert (sigma >= 0).all() and (sigma <= 1).all()
        assert (weights >= 1.0).all()
    
    def test_pretrain_vs_finetune_settings(self):
        """Test different settings for pretrain vs finetune"""
        batch_size = 100
        image_seq_len = 1024
        
        # Pretrain settings
        u_pretrain = compute_density_for_timestep_sampling(
            weighting_scheme='logit_normal',
            batch_size=batch_size,
            logit_std=1.5
        ).to(self.device)
        sigma_pretrain = time_shift(u_pretrain, image_seq_len, shift_type='constant', constant=2.5)
        weights_pretrain = get_flow_match_loss_weight(sigma_pretrain, shift=2.5)
        
        # Finetune settings
        u_finetune = compute_density_for_timestep_sampling(
            weighting_scheme='uniform',
            batch_size=batch_size
        ).to(self.device)
        sigma_finetune = time_shift(u_finetune, image_seq_len, shift_type='constant', constant=3.0)
        weights_finetune = get_flow_match_loss_weight(sigma_finetune, shift=3.0)
        
        # Both should be valid
        assert (sigma_pretrain >= 0).all() and (sigma_pretrain <= 1).all()
        assert (sigma_finetune >= 0).all() and (sigma_finetune <= 1).all()
        assert (weights_pretrain >= 1.0).all()
        assert (weights_finetune >= 1.0).all()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])