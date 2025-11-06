from dataclasses import dataclass
from typing import Optional, Any
from megatron.energon import Sample
import torch

@dataclass
class DiffusionSample(Sample):
    """
    Data class representing a sample for diffusion tasks.

    Attributes:
        video (torch.Tensor): Video latents (C T H W).
        t5_text_embeddings (torch.Tensor): Text embeddings (S D).
        t5_text_mask (torch.Tensor): Mask for text embeddings.
        loss_mask (torch.Tensor): Mask indicating valid positions for loss computation.
        image_size (Optional[torch.Tensor]): Tensor containing image dimensions.
        fps (Optional[torch.Tensor]): Frame rate of the video.
        num_frames (Optional[torch.Tensor]): Number of frames in the video.
        padding_mask (Optional[torch.Tensor]): Mask indicating padding positions.
        seq_len_q (Optional[torch.Tensor]): Sequence length for query embeddings.
        seq_len_kv (Optional[torch.Tensor]): Sequence length for key/value embeddings.
        pos_ids (Optional[torch.Tensor]): Positional IDs.
        latent_shape (Optional[torch.Tensor]): Shape of the latent tensor.
    """

    video: torch.Tensor  # video latents (C T H W)
    t5_text_embeddings: torch.Tensor  # (S D)
    t5_text_mask: torch.Tensor  # 1
    loss_mask: torch.Tensor
    image_size: Optional[torch.Tensor] = None
    fps: Optional[torch.Tensor] = None
    num_frames: Optional[torch.Tensor] = None
    padding_mask: Optional[torch.Tensor] = None
    seq_len_q: Optional[torch.Tensor] = None
    padding_mask: Optional[torch.Tensor] = None
    seq_len_kv: Optional[torch.Tensor] = None
    pos_ids: Optional[torch.Tensor] = None
    latent_shape: Optional[torch.Tensor] = None

    def to_dict(self) -> dict:
        """Converts the sample to a dictionary."""
        return dict(
            video=self.video,
            t5_text_embeddings=self.t5_text_embeddings,
            t5_text_mask=self.t5_text_mask,
            loss_mask=self.loss_mask,
            image_size=self.image_size,
            fps=self.fps,
            num_frames=self.num_frames,
            padding_mask=self.padding_mask,
            seq_len_q=self.seq_len_q,
            seq_len_kv=self.seq_len_kv,
            pos_ids=self.pos_ids,
            latent_shape=self.latent_shape,
        )

    def __add__(self, other: Any) -> int:
        """Adds the sequence length of this sample with another sample or integer."""
        if isinstance(other, DiffusionSample):
            # Combine the values of the two instances
            return self.seq_len_q.item() + other.seq_len_q.item()
        elif isinstance(other, int):
            # Add an integer to the value
            return self.seq_len_q.item() + other
        raise NotImplementedError

    def __radd__(self, other: Any) -> int:
        """Handles reverse addition for summing with integers."""
        # This is called if sum or other operations start with a non-DiffusionSample object.
        # e.g., sum([DiffusionSample(1), DiffusionSample(2)]) -> the 0 + DiffusionSample(1) calls __radd__.
        if isinstance(other, int):
            return self.seq_len_q.item() + other
        raise NotImplementedError

    def __lt__(self, other: Any) -> bool:
        """Compares this sample's sequence length with another sample or integer."""
        if isinstance(other, DiffusionSample):
            return self.seq_len_q.item() < other.seq_len_q.item()
        elif isinstance(other, int):
            return self.seq_len_q.item() < other
        raise NotImplementedError

