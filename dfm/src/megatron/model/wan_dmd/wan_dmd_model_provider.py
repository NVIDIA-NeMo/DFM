from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from fastgen.configs.config_dmd2 import ModelConfig
from fastgen.configs.experiments.Wan.config_dmd2 import create_config as create_wan_dmd2_config
from fastgen.methods.distribution_matching.dmd2 import DMD2Model
from fastgen.networks.noise_schedule import get_noise_schedule
from fastgen.networks.Wan.network import WanTextEncoder
from fastgen.utils import instantiate
from fastgen.utils.basic_utils import PRECISION_MAP
from megatron.bridge.training.checkpointing import _load_model_weights_from_checkpoint
from megatron.core import parallel_state
from megatron.core.models.common.vision_module.vision_module import VisionModule
from torch import Tensor

from dfm.src.megatron.model.wan.wan_provider import WanModelProvider
from dfm.src.megatron.model.wan_dmd.wan_dmd_model import WanDMDModel


@dataclass
class WanDMDModelProvider(WanModelProvider):
    def provide(self, pre_process=None, post_process=None, vp_stage=None) -> WanDMDModel:
        vp_size = self.virtual_pipeline_model_parallel_size
        if vp_size:
            p_size = self.pipeline_model_parallel_size
            assert (self.num_layers // p_size) % vp_size == 0, (
                "Make sure the number of model chunks is the same across all pipeline stages."
            )

        model = WanDMDModel

        return model(
            self,
            pre_process=parallel_state.is_pipeline_first_stage(),
            post_process=parallel_state.is_pipeline_last_stage(),
            fp16_lm_cross_entropy=self.fp16_lm_cross_entropy,
            parallel_output=self.parallel_output,
        )


def _default_wan_fast_gen_config() -> ModelConfig:
    """Create ModelConfig with WAN-specific defaults from fastgen."""
    return create_wan_dmd2_config().model


@dataclass
class WanDMDCombinedModelProvider(WanDMDModelProvider):
    teacher_model_provider: Optional[WanDMDModelProvider] = None
    fake_score_model_provider: Optional[WanDMDModelProvider] = None
    fast_gen_config: ModelConfig = field(default_factory=_default_wan_fast_gen_config)
    # z_dim: int = 16  # Latent dimension for Wan VAE

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
        # Capture model_id from the provider for text encoder initialization
        model_id = getattr(self, "model_id", "Wan-AI/Wan2.1-T2V-1.3B-Diffusers")

        class WanDMDCombinedModel(VisionModule, DMD2Model):
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

            def build_model(second_self):
                second_self.net = student_provider(pre_process, post_process, vp_stage)
                megatron_checkpoint_path = "/opt/artifacts/megatron_checkpoint_1.3B/iter_0000000"
                _load_model_weights_from_checkpoint(
                    checkpoint_path=megatron_checkpoint_path,
                    model=[second_self.net],
                    strict=True,
                )
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
                second_self.fake_score.train().requires_grad_(True)
                second_self.net.train().requires_grad_(True)

                # Instantiate discriminator if GAN loss is enabled
                if second_self.config.gan_loss_weight_gen > 0:
                    if getattr(second_self.config.discriminator, "disc_type", None) is not None:
                        print(f"[INFO] Discriminator type: {second_self.config.discriminator.disc_type}")
                    second_self.discriminator = instantiate(second_self.config.discriminator)
                    second_self.discriminator.train().requires_grad_(True)
                    # Move discriminator to the same device as the student

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
        return WanDMDCombinedModel(config=self)
