import random
from abc import ABC, abstractmethod
from typing import List

import torch
from megatron.energon import DefaultTaskEncoder
from megatron.energon.task_encoder.base import stateless
from megatron.energon.task_encoder.cooking import Cooker, basic_sample_keys

from dfm.src.megatron.data.dit.diffusion_sample import DiffusionSample
from dfm.src.megatron.data.dit.sequence_packing_utils import first_fit_decreasing


def cook(sample: dict) -> dict:
    """
    Processes a raw sample dictionary from energon dataset and returns a new dictionary with specific keys.

    Args:
        sample (dict): The input dictionary containing the raw sample data.

    Returns:
        dict: A new dictionary containing the processed sample data with the following keys:
            - All keys from the result of `basic_sample_keys(sample)`
            - 'json': The contains meta data like resolution, aspect ratio, fps, etc.
            - 'pth': contains video latent tensor
            - 'pickle': contains text embeddings
    """
    return dict(
        **basic_sample_keys(sample),
        json=sample[".json"],
        pth=sample[".pth"],
        pickle=sample[".pickle"],
    )


class DiffusionTaskEncoderWithSequencePacking(DefaultTaskEncoder, ABC):
    cookers = [
        Cooker(cook),
    ]

    def __init__(
        self,
        *args,
        max_frames: int = None,
        text_embedding_padding_size: int = 512,
        seq_length: int = None,
        patch_spatial: int = 2,
        patch_temporal: int = 1,
        packing_buffer_size: int = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.max_frames = max_frames
        self.text_embedding_padding_size = text_embedding_padding_size
        self.seq_length = seq_length
        self.patch_spatial = patch_spatial
        self.patch_temporal = patch_temporal
        self.packing_buffer_size = packing_buffer_size

    @abstractmethod
    def encode_sample(self, sample: dict) -> dict:
        raise NotImplementedError

    def select_samples_to_pack(self, samples: List[DiffusionSample]) -> List[List[DiffusionSample]]:
        """
        Selects sequences to pack for mixed image-video training.
        """
        results = first_fit_decreasing(samples, self.seq_length)
        random.shuffle(results)
        return results

    @stateless
    def pack_selected_samples(self, samples: List[DiffusionSample]) -> DiffusionSample:
        """Construct a new Diffusion sample by concatenating the sequences."""

        def stack(attr):
            return torch.stack([getattr(sample, attr) for sample in samples], dim=0)

        def cat(attr):
            return torch.cat([getattr(sample, attr) for sample in samples], dim=0)

        return DiffusionSample(
            __key__=",".join([s.__key__ for s in samples]),
            __restore_key__=(),  # Will be set by energon based on `samples`
            __subflavor__=None,
            __subflavors__=samples[0].__subflavors__,
            video=cat("video"),
            context_embeddings=cat("context_embeddings"),
            loss_mask=cat("loss_mask"),
            seq_len_q=cat("seq_len_q"),
            seq_len_kv=cat("seq_len_kv"),
            pos_ids=cat("pos_ids"),
            latent_shape=stack("latent_shape"),
        )

    @stateless
    def batch(self, samples: List[DiffusionSample]) -> dict:
        raise NotImplementedError