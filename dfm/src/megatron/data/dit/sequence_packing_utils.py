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


from typing import List
import torch


def find_first_bin_that_fits(bins: List[List[int]], s: int, bin_size: int) -> int:
    """
    Finds the first bin in a list of bins that has enough space to fit a sequence of size 's'.

    Args:
      bins: A list of lists, where each inner list represents a bin and contains the current elements in that bin.
      s: The size of the sequence to be placed in a bin.
      bin_size: The maximum capacity of each bin.

    Returns:
      The index of the first bin that can fit the sequence 's', or -1 if no such bin exists.
    """
    for i, abin in enumerate(bins):
        if sum(abin) + s <= bin_size:
            return i
    return -1


def first_fit(seqlens: List[int], pack_size: int) -> List[List[int]]:
    """
    Packs sequences of varying lengths into bins using the First-Fit algorithm.

    Args:
      seqlens: A list of integers, representing the lengths of the sequences to be packed.
      pack_size: The maximum capacity of each bin.

    Returns:
      A list of lists, where each inner list represents a bin and contains the indices
        of the sequences assigned to that bin.
    """
    res = []
    for s in seqlens:
        first_bin = find_first_bin_that_fits(res, s, pack_size)
        if first_bin == -1:  # open a new bin
            res.append([s])
        else:
            res[first_bin].append(s)
    return res

def first_fit_decreasing(seqlens: List[int], pack_size: int) -> List[List[int]]:
    """
    Packs sequences of varying lengths into bins using the First-Fit Decreasing algorithm.

    This is a variation of the First-Fit algorithm where the sequences are sorted by decreasing length before packing.

    Args:
      seqlens: A list of integers, representing the lengths of the sequences to be packed.
      pack_size: The maximum capacity of each bin.

    Returns:
      A list of lists, similar to the output of the 'first_fit' function.
    """
    sorted_seqlens = sorted(seqlens, reverse=True)
    return first_fit(sorted_seqlens, pack_size)

def concat_pad(tensor_list, max_seq_length):
    """
    Efficiently concatenates a list of tensors along the first dimension and pads with zeros
    to reach max_seq_length.

    Args:
        tensor_list (list of torch.Tensor): List of tensors to concatenate and pad.
        max_seq_length (int): The desired size of the first dimension of the output tensor.

    Returns:
        torch.Tensor: A tensor of shape [max_seq_length, ...], where ... represents the remaining dimensions.
    """
    import torch

    # Get common properties from the first tensor
    other_shape = tensor_list[0].shape[1:]
    dtype = tensor_list[0].dtype
    device = tensor_list[0].device

    # Initialize the result tensor with zeros
    result = torch.zeros((max_seq_length, *other_shape), dtype=dtype, device=device)

    current_index = 0
    for tensor in tensor_list:
        length = tensor.shape[0]
        # Directly assign the tensor to the result tensor without checks
        result[current_index : current_index + length] = tensor
        current_index += length

    return result

