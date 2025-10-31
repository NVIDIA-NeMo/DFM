import imageio
import numpy as np


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

    print('video_save_path', video_save_path)
    imageio.mimsave(video_save_path, grid, "mp4", **kwargs)