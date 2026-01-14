import math
from typing import List, Optional, Set, Tuple, Union

from megatron.core import tensor_parallel
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.transformer.transformer_config import TransformerConfig
from torch import Tensor

from dfm.src.megatron.model.wan.utils import patchify_compact, unpatchify_compact
from dfm.src.megatron.model.wan.wan_model import WanModel


class WanDMDModel(WanModel):
    def __init__(self, config: TransformerConfig, **kwargs):
        super().__init__(config, **kwargs)

    def forward(
        self,
        x: Tensor,
        t: Tensor,
        context: Tensor,
        grid_sizes: list[Tuple[int, int, int]],
        fwd_pred_type: Optional[str] = None,
        packed_seq_params: PackedSeqParams = None,
        scale_t: bool = False,
        unpatchify_features: bool = False,
        feature_indices: Optional[Set[int]] = None,
        return_features_early: bool = False,
        **kwargs,
    ) -> Union[Tensor, List[Tensor]]:
        """Forward pass.

        Args:
            x List[Tensor]: list of vae encoded data (in_channel, f, h, w)
            grid_sizes List[Tuple[int, int, int]]: list of grid sizes (f, h, w)
            t Tensor: timesteps (in [0, 1] if scale_t=True, or [0, 1000] if scale_t=False)
            context List[Tensor]: list of context (text_len, hidden_size)
            packed_seq_params PackedSeqParams: packed sequence parameters
            scale_t: If True, rescale t from [0, 1] to [0, 1000] for timestep embeddings.
                     Use this when t comes from noise_scheduler (e.g., during distillation).
            feature_indices: A set of layer indices (0-based) from which to extract
                intermediate hidden states. Consistent with FastGen's Wan implementation.
            return_features_early: If True and feature_indices is non-empty, returns only
                the features list. If False (default), returns [output, features] when
                feature_indices is non-empty.

        Returns:
            Union[Tensor, List[Tensor]]:
                - If feature_indices is None or empty: output tensor
                - If feature_indices is non-empty and return_features_early is True: features list only
                - If feature_indices is non-empty and return_features_early is False: [output, features]
        """
        x_input = x
        if unpatchify_features:
            x = patchify_compact(x, self.patch_size)

        # run input embedding
        if self.pre_process:
            # x.shape [s, b, c * pF * pH * pW]
            seq_len, batch_size, _ = x.shape
            c = self.out_channels
            pF, pH, pW = self.patch_size
            x = x.reshape(seq_len * batch_size, pF, pH, pW, c)  # output: x.shape [s * b, pF, pH, pW, c]
            x = x.permute(0, 4, 1, 2, 3)  # output: x.shape [s * b, c, pF, pH, pW]
            x = self.patch_embedding(x)  # output: x.shape [s * b, hidden_size, 1, 1, 1]
            x = x.flatten(1)  # output: x.shape [s * b, hidden_size]
            x = x.reshape(seq_len, batch_size, -1)  # output: x.shape [s, b, hidden_size]

            # split sequence for sequence_parallel
            # TODO: for PP, do we move scatter_to_sequence_parallel_region here or after "x = self.decoder.input_tensor" ???
            if self.config.sequence_parallel:
                x = tensor_parallel.scatter_to_sequence_parallel_region(
                    x
                )  # output: x.shape [s * b // tp_size, hidden_size]

        else:
            # intermediate stage of pipeline
            x = self.decoder.input_tensor

        # Rescale t from [0, 1] to [0, 1000] if scale_t is True
        # This is needed when t comes from noise_scheduler (e.g., during distillation with DMD2Model)
        input_t = t
        if scale_t:
            t = self.noise_scheduler.rescale_t(t)
        assert 1 <= t <= 1000, "t must be in [1, 1000]"

        timestep_sinusoidal = self.timesteps_proj(t).to(x.dtype)
        e = self.time_embedder(timestep_sinusoidal)
        e0 = self.time_proj(self.time_proj_act_fn(e)).unflatten(1, (6, self.config.hidden_size))
        context = self.text_embedding(context)  # shape [text_len, b, hidden_size]

        # calculate rotary pos emb
        n_head, dim_head = self.num_heads, self.config.hidden_size // self.num_heads
        cu_seqlens_q_padded = packed_seq_params["self_attention"].cu_seqlens_q_padded
        rotary_pos_emb = self.rope_embeddings(
            n_head, dim_head, cu_seqlens_q_padded, grid_sizes, t.device
        )  # output: rotary_pos_emb.shape [s, b, 1, dim_head]

        # run decoder
        decoder_output = self.decoder(
            hidden_states=x,
            attention_mask=e0,
            context=context,
            context_mask=None,
            rotary_pos_emb=rotary_pos_emb,
            rotary_pos_cos=None,
            rotary_pos_sin=None,
            packed_seq_params=packed_seq_params,
            feature_indices=feature_indices,
        )

        # Handle feature extraction (consistent with FastGen's Wan implementation)
        if feature_indices is not None and len(feature_indices) > 0:
            # decoder returns (hidden_states, features) tuple when feature_indices is non-empty
            x, features = decoder_output
            # Unpatchify features to spatial format using unpatchify_compact
            # For features, out_dim = hidden_size / (p_t * p_h * p_w)
            feature_out_dim = self.config.hidden_size // math.prod(self.patch_size)
            features = [unpatchify_compact(feat, grid_sizes, feature_out_dim, self.patch_size) for feat in features]

            if return_features_early:
                # Return only the features list (for discriminator feature extraction)
                return features
            # Otherwise, continue to compute output and return [output, features] below
        else:
            x = decoder_output
            features = None

        if fwd_pred_type is None:
            fwd_pred_type = self.net_pred_type

        # return if not post_process
        if not self.post_process:
            return x

        # head
        x = x.transpose(0, 1)  # head expects shape [b, s, hidden_size]
        x = self.head(x, e)  # output: x.shape [b, s, c * pF * pH * pW]
        x = x.transpose(0, 1)  # reshape back to shape [s, b, c * pF * pH * pW]

        if self.config.sequence_parallel:
            x = tensor_parallel.gather_from_sequence_parallel_region(x)

        if unpatchify_features:
            # Infer z_dim from input shape: z_dim = feature_dim / (pF * pH * pW)
            z_dim = x.shape[-1] // math.prod(self.patch_size)
            x = unpatchify_compact(x, grid_sizes, z_dim, self.patch_size)

        x = self.noise_scheduler.convert_model_output(
            x_input, x, input_t, src_pred_type=self.net_pred_type, target_pred_type=fwd_pred_type
        )
        if features is not None:
            return [x, features]

        return x  # output: x.shape [s, b, c * pF * pH * pW]

    def load_state_dict(self, state_dict, strict=True):
        """Load state dict with automatic handling of 'module.' prefix mismatch.

        This method handles the case where checkpoints saved with DistributedDataParallel
        have a 'module.' prefix that needs to be removed when loading.

        Args:
            state_dict (dict): The state dictionary to load
            strict (bool): Whether to strictly enforce that the keys match

        Returns:
            NamedTuple: with 'missing_keys' and 'unexpected_keys' fields
        """
        # Check if state_dict has 'module.' prefix but model doesn't
        has_module_prefix = any(k.startswith("module.") for k in state_dict.keys())
        if has_module_prefix:
            new_state_dict = {}
            for key, value in state_dict.items():
                new_key = key.replace("module.", "", 1) if key.startswith("module.") else key
                new_state_dict[new_key] = value
            state_dict = new_state_dict

        return super().load_state_dict(state_dict, strict=strict)
