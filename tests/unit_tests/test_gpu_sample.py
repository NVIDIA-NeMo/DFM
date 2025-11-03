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


def test_gpu_availability():
    """Test that GPU is available and accessible"""
    assert torch.cuda.is_available(), "CUDA is not available"
    
    gpu_count = torch.cuda.device_count()
    print(f"Number of GPUs available: {gpu_count}")
    assert gpu_count >= 1, f"Expected at least 1 GPU, found {gpu_count}"
    
    # Print GPU information
    for i in range(gpu_count):
        gpu_name = torch.cuda.get_device_name(i)
        print(f"GPU {i}: {gpu_name}")
        print(f"  Memory allocated: {torch.cuda.memory_allocated(i) / 1024**2:.2f} MB")
        print(f"  Memory reserved: {torch.cuda.memory_reserved(i) / 1024**2:.2f} MB")
    
    print("✓ GPU availability test passed")


def test_gpu_tensor_operations():
    """Test basic GPU tensor operations"""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    
    # Create tensors on GPU
    device = torch.device('cuda:0')
    a = torch.tensor([1.0, 2.0, 3.0], device=device)
    b = torch.tensor([4.0, 5.0, 6.0], device=device)
    
    # Verify tensors are on GPU
    assert a.is_cuda, "Tensor a is not on GPU"
    assert b.is_cuda, "Tensor b is not on GPU"
    
    # Test addition
    c = a + b
    expected = torch.tensor([5.0, 7.0, 9.0], device=device)
    assert torch.allclose(c, expected), f"Expected {expected}, got {c}"
    assert c.is_cuda, "Result tensor is not on GPU"
    
    # Test multiplication
    d = a * b
    expected = torch.tensor([4.0, 10.0, 18.0], device=device)
    assert torch.allclose(d, expected), f"Expected {expected}, got {d}"
    
    print("✓ GPU tensor operations test passed")


def test_gpu_matrix_multiplication():
    """Test matrix multiplication on GPU"""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    
    device = torch.device('cuda:0')
    
    # Create random matrices on GPU
    matrix_a = torch.randn(100, 200, device=device)
    matrix_b = torch.randn(200, 300, device=device)
    
    # Perform matrix multiplication
    result = torch.matmul(matrix_a, matrix_b)
    
    # Verify shape
    assert result.shape == (100, 300), f"Expected shape (100, 300), got {result.shape}"
    
    # Verify result is on GPU
    assert result.is_cuda, "Result is not on GPU"
    
    # Verify result is finite
    assert torch.isfinite(result).all(), "Result contains non-finite values"
    
    print("✓ GPU matrix multiplication test passed")


def test_multi_gpu_tensor_transfer():
    """Test tensor transfer between GPUs if multiple GPUs are available"""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    
    gpu_count = torch.cuda.device_count()
    print(f"Testing with {gpu_count} GPU(s)")
    
    if gpu_count < 2:
        print("Only 1 GPU available, testing single GPU operations")
        device = torch.device('cuda:0')
        tensor = torch.randn(10, 10, device=device)
        assert tensor.is_cuda, "Tensor is not on GPU"
    else:
        print("Multiple GPUs available, testing cross-GPU transfer")
        # Create tensor on GPU 0
        tensor_gpu0 = torch.randn(10, 10, device='cuda:0')
        assert tensor_gpu0.device.index == 0, "Tensor not on GPU 0"
        
        # Transfer to GPU 1
        tensor_gpu1 = tensor_gpu0.to('cuda:1')
        assert tensor_gpu1.device.index == 1, "Tensor not on GPU 1"
        
        # Verify data is preserved
        assert torch.allclose(tensor_gpu0.cpu(), tensor_gpu1.cpu()), "Data changed during transfer"
    
    print("✓ Multi-GPU tensor transfer test passed")


def test_gpu_memory_allocation():
    """Test GPU memory allocation and deallocation"""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    
    device = torch.device('cuda:0')
    
    # Record initial memory
    torch.cuda.empty_cache()
    initial_memory = torch.cuda.memory_allocated(0)
    print(f"Initial GPU memory allocated: {initial_memory / 1024**2:.2f} MB")
    
    # Allocate large tensor
    large_tensor = torch.randn(1000, 1000, device=device)
    memory_after_alloc = torch.cuda.memory_allocated(0)
    print(f"Memory after allocation: {memory_after_alloc / 1024**2:.2f} MB")
    
    # Verify memory increased
    assert memory_after_alloc > initial_memory, "GPU memory did not increase after allocation"
    
    # Delete tensor and clear cache
    del large_tensor
    torch.cuda.empty_cache()
    memory_after_dealloc = torch.cuda.memory_allocated(0)
    print(f"Memory after deallocation: {memory_after_dealloc / 1024**2:.2f} MB")
    
    print("✓ GPU memory allocation test passed")


def test_cuda_compute_capability():
    """Test CUDA compute capability"""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    
    for i in range(torch.cuda.device_count()):
        capability = torch.cuda.get_device_capability(i)
        print(f"GPU {i} compute capability: {capability[0]}.{capability[1]}")
        
        # Verify compute capability is reasonable (at least 3.5)
        assert capability[0] >= 3, f"GPU {i} compute capability too old: {capability}"
    
    print("✓ CUDA compute capability test passed")

