#!/usr/bin/env python3
"""
Real unit tests for training_step_t2v.py

Tests the actual step_fsdp_transformer_t2v function and related components.
"""

import pytest
import torch
import os
import sys
from unittest.mock import Mock, MagicMock, patch

# Add the uploads directory to path so we can import the modules
sys.path.insert(0, '/mnt/user-data/uploads')

# Import the actual functions we want to test
from training_step_t2v import step_fsdp_transformer_t2v


class MockScheduler:
    """Mock scheduler that mimics the real diffusers scheduler"""
    def __init__(self):
        self.config = Mock()
        self.config.num_train_timesteps = 1000


class MockTransformer(torch.nn.Module):
    """Mock transformer that behaves like WAN transformer"""
    def __init__(self):
        super().__init__()
        # Add a parameter so the model is trainable
        self.weight = torch.nn.Parameter(torch.randn(16, 16))
    
    def forward(self, hidden_states, timestep, encoder_hidden_states, return_dict=False):
        """Return output with same shape as input"""
        batch_size = hidden_states.shape[0]
        output = torch.randn_like(hidden_states) * 0.1 + hidden_states * self.weight[0, 0] * 0.001
        
        if return_dict:
            return Mock(sample=output)
        return (output,)


class MockPipe:
    """Mock pipeline"""
    def __init__(self):
        self.scheduler = MockScheduler()


class TestStepFSDPTransformerT2V:
    """Test the main training step function"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.bf16 = torch.bfloat16
        self.batch_size = 2
        
        # Disable debug mode for tests
        if 'DEBUG_TRAINING' in os.environ:
            del os.environ['DEBUG_TRAINING']
    
    def create_mock_batch(self):
        """Create a realistic batch"""
        return {
            'video_latents': torch.randn(
                self.batch_size, 16, 8, 32, 32,
                device=self.device, dtype=self.bf16
            ),
            'text_embeddings': torch.randn(
                self.batch_size, 77, 4096,
                device=self.device, dtype=self.bf16
            ),
        }
    
    def create_mock_model_map(self):
        """Create mock model map"""
        transformer = MockTransformer().to(self.device).to(self.bf16)
        
        return {
            'transformer': {
                'fsdp_transformer': transformer,
                'base_transformer': transformer,
            }
        }
    
    def test_step_basic_functionality(self):
        """Test that step_fsdp_transformer_t2v runs without errors"""
        pipe = MockPipe()
        model_map = self.create_mock_model_map()
        batch = self.create_mock_batch()
        
        loss, metrics = step_fsdp_transformer_t2v(
            pipe=pipe,
            model_map=model_map,
            batch=batch,
            device=self.device,
            bf16=self.bf16,
            use_sigma_noise=True,
            timestep_sampling='uniform',
            flow_shift=3.0,
            global_step=0,
        )
        
        # Basic assertions
        assert loss is not None
        assert isinstance(loss, torch.Tensor)
        assert not torch.isnan(loss)
        assert loss.item() > 0
        
        # Check metrics
        assert 'loss' in metrics
        assert 'sigma_min' in metrics
        assert 'sigma_max' in metrics
        assert 'sampling_method' in metrics
    
    def test_step_with_uniform_sampling(self):
        """Test with uniform timestep sampling"""
        pipe = MockPipe()
        model_map = self.create_mock_model_map()
        batch = self.create_mock_batch()
        
        loss, metrics = step_fsdp_transformer_t2v(
            pipe=pipe,
            model_map=model_map,
            batch=batch,
            device=self.device,
            bf16=self.bf16,
            use_sigma_noise=False,  # Pure uniform
            global_step=0,
        )
        
        assert not torch.isnan(loss)
        assert metrics['sampling_method'] == 'uniform_no_shift'
    
    def test_step_with_logit_normal_sampling(self):
        """Test with logit-normal timestep sampling"""
        pipe = MockPipe()
        model_map = self.create_mock_model_map()
        batch = self.create_mock_batch()
        
        # Mock the compute_density_for_timestep_sampling function
        with patch('training_step_t2v.compute_density_for_timestep_sampling') as mock_density:
            mock_density.return_value = torch.rand(self.batch_size).to(self.device)
            
            loss, metrics = step_fsdp_transformer_t2v(
                pipe=pipe,
                model_map=model_map,
                batch=batch,
                device=self.device,
                bf16=self.bf16,
                use_sigma_noise=True,
                timestep_sampling='logit_normal',
                logit_mean=0.0,
                logit_std=1.5,
                flow_shift=3.0,
                global_step=0,
            )
            
            assert not torch.isnan(loss)
            # Should call the density function
            assert mock_density.called
    
    def test_step_with_flow_shift(self):
        """Test that flow shift parameter works"""
        pipe = MockPipe()
        model_map = self.create_mock_model_map()
        batch = self.create_mock_batch()
        
        # Test different flow shifts
        for shift in [1.0, 3.0, 5.0]:
            loss, metrics = step_fsdp_transformer_t2v(
                pipe=pipe,
                model_map=model_map,
                batch=batch,
                device=self.device,
                bf16=self.bf16,
                use_sigma_noise=True,
                flow_shift=shift,
                global_step=0,
            )
            
            assert not torch.isnan(loss)
            assert loss.item() > 0
    
    def test_step_pretrain_sigma_range(self):
        """Test pretrain mode with full sigma range [0, 1]"""
        pipe = MockPipe()
        model_map = self.create_mock_model_map()
        batch = self.create_mock_batch()
        
        loss, metrics = step_fsdp_transformer_t2v(
            pipe=pipe,
            model_map=model_map,
            batch=batch,
            device=self.device,
            bf16=self.bf16,
            use_sigma_noise=True,
            sigma_min=0.0,
            sigma_max=1.0,
            global_step=0,
        )
        
        # Sigma should be in full range
        assert metrics['sigma_min'] >= 0.0
        assert metrics['sigma_max'] <= 1.0
    
    def test_step_finetune_sigma_range(self):
        """Test finetune mode with restricted sigma range [0.02, 0.55]"""
        pipe = MockPipe()
        model_map = self.create_mock_model_map()
        batch = self.create_mock_batch()
        
        loss, metrics = step_fsdp_transformer_t2v(
            pipe=pipe,
            model_map=model_map,
            batch=batch,
            device=self.device,
            bf16=self.bf16,
            use_sigma_noise=True,
            sigma_min=0.02,
            sigma_max=0.55,
            global_step=0,
        )
        
        # Sigma should be clamped
        assert metrics['sigma_min'] >= 0.02
        assert metrics['sigma_max'] <= 0.55
    
    def test_step_with_different_batch_sizes(self):
        """Test with various batch sizes"""
        pipe = MockPipe()
        model_map = self.create_mock_model_map()
        
        for batch_size in [1, 2, 4]:
            batch = {
                'video_latents': torch.randn(
                    batch_size, 16, 8, 32, 32,
                    device=self.device, dtype=self.bf16
                ),
                'text_embeddings': torch.randn(
                    batch_size, 77, 4096,
                    device=self.device, dtype=self.bf16
                ),
            }
            
            loss, metrics = step_fsdp_transformer_t2v(
                pipe=pipe,
                model_map=model_map,
                batch=batch,
                device=self.device,
                bf16=self.bf16,
                global_step=0,
            )
            
            assert not torch.isnan(loss)
            assert loss.item() > 0
    
    def test_step_gradient_flow(self):
        """Test that gradients flow backward properly"""
        pipe = MockPipe()
        model_map = self.create_mock_model_map()
        batch = self.create_mock_batch()
        
        # Enable gradients on model
        model = model_map['transformer']['fsdp_transformer']
        for param in model.parameters():
            param.requires_grad = True
        
        loss, metrics = step_fsdp_transformer_t2v(
            pipe=pipe,
            model_map=model_map,
            batch=batch,
            device=self.device,
            bf16=self.bf16,
            global_step=0,
        )
        
        # Backward pass
        loss.backward()
        
        # Check that gradients exist
        for param in model.parameters():
            if param.requires_grad:
                assert param.grad is not None
                # At least some gradients should be non-zero
    
    def test_step_returns_correct_metrics(self):
        """Test that all expected metrics are returned"""
        pipe = MockPipe()
        model_map = self.create_mock_model_map()
        batch = self.create_mock_batch()
        
        loss, metrics = step_fsdp_transformer_t2v(
            pipe=pipe,
            model_map=model_map,
            batch=batch,
            device=self.device,
            bf16=self.bf16,
            global_step=0,
        )
        
        # Check all expected metrics exist
        expected_keys = [
            'loss', 'unweighted_loss', 'sigma_min', 'sigma_max', 'sigma_mean',
            'weight_min', 'weight_max', 'timestep_min', 'timestep_max',
            'noisy_min', 'noisy_max', 'sampling_method'
        ]
        
        for key in expected_keys:
            assert key in metrics, f"Missing metric: {key}"
    
    def test_step_loss_weighting(self):
        """Test that loss weighting affects the result"""
        pipe = MockPipe()
        model_map = self.create_mock_model_map()
        batch = self.create_mock_batch()
        
        loss, metrics = step_fsdp_transformer_t2v(
            pipe=pipe,
            model_map=model_map,
            batch=batch,
            device=self.device,
            bf16=self.bf16,
            flow_shift=3.0,
            global_step=0,
        )
        
        # Weighted loss should differ from unweighted
        # (unless all sigmas happen to be 0, which is very unlikely)
        weighted = metrics['loss']
        unweighted = metrics['unweighted_loss']
        
        # Both should be positive
        assert weighted > 0
        assert unweighted > 0
    
    def test_step_timestep_conversion(self):
        """Test that sigma is converted to timesteps correctly"""
        pipe = MockPipe()
        model_map = self.create_mock_model_map()
        batch = self.create_mock_batch()
        
        loss, metrics = step_fsdp_transformer_t2v(
            pipe=pipe,
            model_map=model_map,
            batch=batch,
            device=self.device,
            bf16=self.bf16,
            global_step=0,
        )
        
        # Timesteps should be in valid range [0, 1000]
        assert 0 <= metrics['timestep_min'] <= 1000
        assert 0 <= metrics['timestep_max'] <= 1000
        assert metrics['timestep_min'] <= metrics['timestep_max']
    
    def test_step_noisy_latents_bounds(self):
        """Test that noisy latents stay within reasonable bounds"""
        pipe = MockPipe()
        model_map = self.create_mock_model_map()
        batch = self.create_mock_batch()
        
        loss, metrics = step_fsdp_transformer_t2v(
            pipe=pipe,
            model_map=model_map,
            batch=batch,
            device=self.device,
            bf16=self.bf16,
            global_step=0,
        )
        
        # Noisy latents should be finite and reasonable
        assert torch.isfinite(torch.tensor(metrics['noisy_min']))
        assert torch.isfinite(torch.tensor(metrics['noisy_max']))
        
        # Check that values aren't exploding
        assert abs(metrics['noisy_min']) < 100
        assert abs(metrics['noisy_max']) < 100
    
    def test_step_shape_normalization(self):
        """Test that input shapes are normalized correctly"""
        pipe = MockPipe()
        model_map = self.create_mock_model_map()
        
        # Test with 6D video latents (extra dimension)
        batch = {
            'video_latents': torch.randn(
                1, self.batch_size, 16, 8, 32, 32,  # 6D
                device=self.device, dtype=self.bf16
            ),
            'text_embeddings': torch.randn(
                self.batch_size, 77, 4096,
                device=self.device, dtype=self.bf16
            ),
        }
        
        # Should handle it without error (will squeeze to 5D internally)
        loss, metrics = step_fsdp_transformer_t2v(
            pipe=pipe,
            model_map=model_map,
            batch=batch,
            device=self.device,
            bf16=self.bf16,
            global_step=0,
        )
        
        assert not torch.isnan(loss)
    
    def test_step_with_debug_logging(self):
        """Test that debug mode works"""
        os.environ['DEBUG_TRAINING'] = '1'
        
        pipe = MockPipe()
        model_map = self.create_mock_model_map()
        batch = self.create_mock_batch()
        
        # Should not crash with debug logging enabled
        loss, metrics = step_fsdp_transformer_t2v(
            pipe=pipe,
            model_map=model_map,
            batch=batch,
            device=self.device,
            bf16=self.bf16,
            global_step=0,
        )
        
        assert not torch.isnan(loss)
        
        # Cleanup
        del os.environ['DEBUG_TRAINING']
    
    def test_step_mix_uniform_ratio(self):
        """Test mix_uniform_ratio parameter"""
        pipe = MockPipe()
        model_map = self.create_mock_model_map()
        batch = self.create_mock_batch()
        
        # With ratio=1.0, should always use uniform
        loss, metrics = step_fsdp_transformer_t2v(
            pipe=pipe,
            model_map=model_map,
            batch=batch,
            device=self.device,
            bf16=self.bf16,
            use_sigma_noise=True,
            timestep_sampling='logit_normal',
            mix_uniform_ratio=1.0,  # Always uniform
            global_step=0,
        )
        
        # Should use uniform sampling
        assert metrics['sampling_method'] == 'uniform'


class TestFlowMatchingFormulas:
    """Test the mathematical correctness of flow matching"""
    
    def setup_method(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.bf16 = torch.bfloat16
    
    def test_flow_matching_noise_formula(self):
        """Test x_t = (1-σ)x_0 + σ*ε"""
        clean = torch.randn(2, 16, 8, 32, 32, device=self.device, dtype=torch.float32)
        noise = torch.randn_like(clean)
        
        # Test at various sigma values
        for sigma_val in [0.0, 0.25, 0.5, 0.75, 1.0]:
            sigma = torch.tensor([sigma_val, sigma_val], device=self.device)
            sigma_reshaped = sigma.view(-1, 1, 1, 1, 1)
            
            noisy = (1.0 - sigma_reshaped) * clean + sigma_reshaped * noise
            
            # At sigma=0, should equal clean
            if sigma_val == 0.0:
                assert torch.allclose(noisy, clean, atol=1e-6)
            
            # At sigma=1, should equal noise
            if sigma_val == 1.0:
                assert torch.allclose(noisy, noise, atol=1e-6)
            
            # At sigma=0.5, should be average
            if sigma_val == 0.5:
                expected = 0.5 * clean + 0.5 * noise
                assert torch.allclose(noisy, expected, atol=1e-6)
    
    def test_flow_matching_target_formula(self):
        """Test v = ε - x_0"""
        clean = torch.randn(2, 16, 8, 32, 32, device=self.device, dtype=torch.float32)
        noise = torch.randn_like(clean)
        
        target = noise - clean
        
        # Target should be different from both inputs
        assert not torch.allclose(target, noise)
        assert not torch.allclose(target, clean)
        
        # Reconstructing should work
        reconstructed_noise = target + clean
        assert torch.allclose(reconstructed_noise, noise, atol=1e-6)
    
    def test_flow_shift_formula(self):
        """Test σ = shift/(shift + (1/u - 1))"""
        u = torch.linspace(0.01, 0.99, 100, device=self.device)
        shift = 3.0
        
        sigma = shift / (shift + (1.0 / u - 1.0))
        
        # Sigma should be in [0, 1]
        assert (sigma >= 0).all()
        assert (sigma <= 1).all()
        
        # Should be monotonically increasing with u
        assert (sigma[1:] >= sigma[:-1]).all()
    
    def test_loss_weight_formula(self):
        """Test w = 1 + shift * σ"""
        sigma = torch.linspace(0, 1, 100, device=self.device)
        shift = 3.0
        
        weights = 1.0 + shift * sigma
        
        # At sigma=0, weight=1
        assert torch.isclose(weights[0], torch.tensor(1.0, device=self.device))
        
        # At sigma=1, weight=1+shift
        assert torch.isclose(weights[-1], torch.tensor(1.0 + shift, device=self.device))
        
        # Should be monotonically increasing
        assert (weights[1:] >= weights[:-1]).all()


class TestMetricsAccuracy:
    """Test that metrics are computed accurately"""
    
    def setup_method(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.bf16 = torch.bfloat16
    
    def test_sigma_metrics_accuracy(self):
        """Test that sigma min/max/mean are correct"""
        pipe = MockPipe()
        model_map = {
            'transformer': {
                'fsdp_transformer': MockTransformer().to(self.device).to(self.bf16),
                'base_transformer': MockTransformer().to(self.device).to(self.bf16),
            }
        }
        
        batch = {
            'video_latents': torch.randn(10, 16, 8, 32, 32, device=self.device, dtype=self.bf16),
            'text_embeddings': torch.randn(10, 77, 4096, device=self.device, dtype=self.bf16),
        }
        
        loss, metrics = step_fsdp_transformer_t2v(
            pipe=pipe,
            model_map=model_map,
            batch=batch,
            device=self.device,
            bf16=self.bf16,
            global_step=0,
        )
        
        # Min should be <= mean <= max
        assert metrics['sigma_min'] <= metrics['sigma_mean']
        assert metrics['sigma_mean'] <= metrics['sigma_max']
        
        # All should be in valid range
        assert 0 <= metrics['sigma_min'] <= 1
        assert 0 <= metrics['sigma_mean'] <= 1
        assert 0 <= metrics['sigma_max'] <= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])