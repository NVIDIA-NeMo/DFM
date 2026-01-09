from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import torch
from fastgen.configs.config_dmd2 import ModelConfig
from fastgen.configs.experiments.Wan.config_dmd2 import create_config as create_wan_dmd2_config
from fastgen.methods.distribution_matching.dmd2 import DMD2Model
from fastgen.networks.noise_schedule import get_noise_schedule
from fastgen.networks.Wan.network import WanTextEncoder, WanVideoEncoder
from fastgen.utils import instantiate
from fastgen.utils.basic_utils import PRECISION_MAP
from megatron.core.models.common.vision_module.vision_module import VisionModule
from torch import Tensor

from dfm.src.megatron.model.wan.wan_provider import WanModelProvider


def _default_wan_fast_gen_config() -> ModelConfig:
    """Create ModelConfig with WAN-specific defaults from fastgen."""
    return create_wan_dmd2_config().model


@dataclass
class DMDModelProvider(WanModelProvider):
    teacher_model_provider: Optional[WanModelProvider] = None
    fake_score_model_provider: Optional[WanModelProvider] = None
    fast_gen_config: ModelConfig = field(default_factory=_default_wan_fast_gen_config)
    z_dim: int = 16  # Latent dimension for Wan VAE

    def __getattr__(self, name: str):
        try:
            fast_gen_config = object.__getattribute__(self, "fast_gen_config")
        except AttributeError:
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
        _sentinel = object()
        result = getattr(fast_gen_config, name, _sentinel)
        if result is not _sentinel:
            return result
        try:
            return fast_gen_config[name]
        except (KeyError, TypeError, AttributeError):
            pass
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def provide(self, pre_process=None, post_process=None, vp_stage=None):
        # Set recompute/gradient checkpointing settings FIRST (before copying to children)
        self.recompute_granularity = "full"
        self.recompute_method = "uniform"
        self.recompute_num_layers = 1

        # Copy parallelism config and recompute config to child providers
        for provider in [self.teacher_model_provider, self.fake_score_model_provider]:
            if provider is not None:
                provider.tensor_model_parallel_size = self.tensor_model_parallel_size
                provider.pipeline_model_parallel_size = self.pipeline_model_parallel_size
                provider.virtual_pipeline_model_parallel_size = self.virtual_pipeline_model_parallel_size
                provider.context_parallel_size = self.context_parallel_size
                provider.sequence_parallel = self.sequence_parallel
                # Copy recompute/gradient checkpointing settings
                provider.recompute_granularity = self.recompute_granularity
                provider.recompute_method = self.recompute_method
                provider.recompute_num_layers = self.recompute_num_layers
                if hasattr(provider, "finalize"):
                    provider.finalize()

        student_provider = super().provide
        self.training_mode = "finetune"
        self.fake_score_model_provider.training_mode = "pretrain"
        self.fake_score_model_provider.z_dim = self.z_dim
        self.teacher_model_provider.training_mode = "pretrain"
        self.teacher_model_provider.z_dim = self.z_dim
        # Capture model_id from the provider for text encoder initialization
        model_id = getattr(self, "model_id", "Wan-AI/Wan2.1-T2V-1.3B-Diffusers")

        class DMDDistillModel(VisionModule, DMD2Model):
            def __init__(second_self, config):
                DMD2Model.__init__(second_self, config)
                second_self._text_encoder = None
                second_self._vae = None
                second_self._model_id = model_id

            def get_text_encoder(second_self) -> WanTextEncoder:
                """Lazy-load the WAN text encoder for encoding prompts."""
                print("dtype", PRECISION_MAP[self.t_precision])
                if second_self._text_encoder is None:
                    second_self._text_encoder = WanTextEncoder(
                        model_id=second_self._model_id,
                    )
                    # Move to CUDA and set precision
                    print("moving text encoder to cuda and setting precision", PRECISION_MAP[self.t_precision])
                    second_self._text_encoder.to(device="cuda", dtype=PRECISION_MAP[self.t_precision])
                return second_self._text_encoder

            def get_vae(second_self) -> WanVideoEncoder:
                """Lazy-load the WAN video encoder for decoding latents."""
                if second_self._vae is None:
                    second_self._vae = WanVideoEncoder(
                        model_id=second_self._model_id,
                    )
                    second_self._vae.to(device="cuda", dtype=torch.bfloat16)
                return second_self._vae

            def encode_prompt(second_self, prompt: str | list[str], precision: torch.dtype = None) -> torch.Tensor:
                """Encode a text prompt using the WAN text encoder.

                Args:
                    prompt: A single prompt string or list of prompt strings
                    precision: Optional dtype for the output embeddings

                Returns:
                    Encoded prompt embeddings tensor
                """
                if precision is None:
                    precision = PRECISION_MAP[self.t_precision]
                return second_self.get_text_encoder().encode(prompt, precision=precision)

            def build_model(second_self):
                second_self.net = student_provider(pre_process, post_process, vp_stage)
                second_self.net.noise_scheduler = get_noise_schedule("rf")
                second_self.net.net_pred_type = "flow"

                second_self.fake_score = self.fake_score_model_provider.provide(pre_process, post_process, vp_stage)
                second_self.fake_score.noise_scheduler = get_noise_schedule("rf")
                second_self.fake_score.net_pred_type = "flow"
                # Load weights from net to fake_score
                second_self.fake_score.load_state_dict(second_self.net.state_dict())

                second_self.teacher = self.teacher_model_provider.provide(pre_process, post_process, vp_stage)
                second_self.teacher.noise_scheduler = get_noise_schedule("rf")
                second_self.teacher.net_pred_type = "flow"
                # Load weights from net to teacher
                second_self.teacher.load_state_dict(second_self.net.state_dict())
                second_self.teacher.eval().requires_grad_(False)

                # CRITICAL: Ensure fake_score starts with requires_grad=True so it's included in optimizer
                # The _setup_grad_requirements method will toggle this based on iteration
                second_self.fake_score.train().requires_grad_(True)

                # Also ensure student (net) starts trainable
                second_self.net.train().requires_grad_(True)

                # Instantiate discriminator if GAN loss is enabled
                if second_self.config.gan_loss_weight_gen > 0:
                    print(f"[INFO] gan_loss_weight_gen: {second_self.config.gan_loss_weight_gen}")
                    print("[INFO] Instantiating the discriminator")
                    if getattr(second_self.config.discriminator, "disc_type", None) is not None:
                        print(f"[INFO] Discriminator type: {second_self.config.discriminator.disc_type}")
                    second_self.discriminator = instantiate(second_self.config.discriminator)
                    second_self.discriminator.train().requires_grad_(True)
                    # Move discriminator to the same device as the student

                # Debug: verify recompute settings are applied to each model
                print(f"[DEBUG] student recompute_granularity: {second_self.net.config.recompute_granularity}")
                print(
                    f"[DEBUG] fake_score recompute_granularity: {second_self.fake_score.config.recompute_granularity}"
                )
                print(f"[DEBUG] teacher recompute_granularity: {second_self.teacher.config.recompute_granularity}")

            def set_input_tensor(second_self, input_tensor: Tensor) -> None:
                """Sets input tensor to the model.

                See megatron.model.transformer.set_input_tensor()

                Args:
                    input_tensor (Tensor): Sets the input tensor for the model.
                """
                # Delegate to the main network (net) for pipeline parallelism
                if not isinstance(input_tensor, list):
                    input_tensor = [input_tensor]

                assert len(input_tensor) == 1, "input_tensor should only be length 1 for gpt/bert"
                second_self.net.decoder.set_input_tensor(input_tensor[0])

            def _extract_fwd_kwargs(second_self, data: Dict[str, Any]) -> Dict[str, Any]:
                """Extract model-specific forward kwargs from data dict.

                Override this method in subclasses to extract custom forward arguments
                like grid_sizes, packed_seq_params, etc. These kwargs will be passed
                to all network forward calls.

                Args:
                    data: Data dict from the dataloader

                Returns:
                    Dict of additional kwargs to pass to network forward calls
                """
                fwd_kwargs = {}
                # Extract common model-specific kwargs if present in data
                if "grid_sizes" in data:
                    fwd_kwargs["grid_sizes"] = data["grid_sizes"]
                if "packed_seq_params" in data:
                    fwd_kwargs["packed_seq_params"] = data["packed_seq_params"]
                if "context_embeddings" in data:
                    fwd_kwargs["context"] = data["context_embeddings"].to(PRECISION_MAP[self.t_precision])
                # Enable timestep scaling from [0, 1] to [0, 1000] for distillation
                # DMD2Model generates timesteps in [0, 1] range, but WanModel expects [0, 1000]
                fwd_kwargs["scale_t"] = True
                fwd_kwargs["unpatchify_features"] = True
                return fwd_kwargs

        # Convert to ModelConfig if it was serialized to dict by omegaconf/megatron
        return DMDDistillModel(config=self)
