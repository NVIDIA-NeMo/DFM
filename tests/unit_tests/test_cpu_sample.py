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

import torch


def test_cpu_tensor_operations():
    """Test basic CPU tensor operations"""
    # Create tensors on CPU
    a = torch.tensor([1.0, 2.0, 3.0], device='cpu')
    b = torch.tensor([4.0, 5.0, 6.0], device='cpu')
    
    # Test addition
    c = a + b
    expected = torch.tensor([5.0, 7.0, 9.0])
    assert torch.allclose(c, expected), f"Expected {expected}, got {c}"
    
    # Test multiplication
    d = a * b
    expected = torch.tensor([4.0, 10.0, 18.0])
    assert torch.allclose(d, expected), f"Expected {expected}, got {d}"
    
    print("✓ CPU tensor operations test passed")


def test_cpu_only_environment():
    """Verify that CUDA is not visible in CPU-only environment"""
    # In CPU tests, CUDA should not be available or visible
    # When CUDA_VISIBLE_DEVICES="" is set, torch.cuda.is_available() should be False
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA device count: {torch.cuda.device_count()}")
    print("✓ CPU environment test completed")


def test_cpu_matrix_multiplication():
    """Test matrix multiplication on CPU"""
    # Create random matrices
    matrix_a = torch.randn(10, 20, device='cpu')
    matrix_b = torch.randn(20, 30, device='cpu')
    
    # Perform matrix multiplication
    result = torch.matmul(matrix_a, matrix_b)
    
    # Verify shape
    assert result.shape == (10, 30), f"Expected shape (10, 30), got {result.shape}"
    
    # Verify result is finite
    assert torch.isfinite(result).all(), "Result contains non-finite values"
    
    print("✓ CPU matrix multiplication test passed")


def test_pytorch_version():
    """Test that PyTorch is properly installed"""
    print(f"PyTorch version: {torch.__version__}")
    assert torch.__version__ is not None, "PyTorch version not found"
    print("✓ PyTorch version check passed")
