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

import imageio
import numpy as np
import torch


def save_video(
    grid: np.ndarray,
    fps: int,
    H: int,
    W: int,
    video_save_quality: int,
    video_save_path: str,
):
    kwargs = {
        "fps": fps,
        "quality": video_save_quality,
        "macro_block_size": 1,
        "ffmpeg_params": ["-s", f"{W}x{H}"],
        "output_params": ["-f", "mp4"],
    }

    print("video_save_path", video_save_path)
    imageio.mimsave(video_save_path, grid, "mp4", **kwargs)


def print_dict(dict):
    for key, value in dict.items():
        if isinstance(value, torch.Tensor):
            print(key, value.shape)
        else:
            print(key, value)
    print("-" * 40)
