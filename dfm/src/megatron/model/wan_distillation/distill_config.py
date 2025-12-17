from dataclasses import dataclass, field
from typing import List

from fastgen.configs.config import SampleTConfig


@dataclass(kw_only=True)
class BaseModelConfig:
    guidance_scale: float = 1.0
    skip_layers: List[int] | None = None
    # optimizer and scheduler for the main net (i.e., one-step generator in DMD)
    # net_optimizer
    # net_scheduler

    # sampling t from a given distribution
    sample_t_cfg: SampleTConfig = field(default_factory=SampleTConfig)
    use_ema: bool = False

    precision: str = "float32"  # "float32", "float16" or "bfloat16"
    device: str = "cuda"
    # Enable Gradient Scaler
    grad_scaler_enabled: bool = False
    grad_scaler_init_scale: float = 65536.0
    grad_scaler_growth_interval: int = 2000
    student_sample_steps: int = 1
    student_sample_type: str = "sde"
    precision_infer: str | None = None


@dataclass(kw_only=True)
class DMD2DistillConfig(BaseModelConfig):
    input_shape: List[int] = field(default_factory=lambda: [3, 32, 32])  # CIFAR-10, by default
    student_update_freq: int = 5
    gan_loss_weight_gen: float = 0.0
    guidance_scale: float = 1.0
    gan_use_same_t_noise: bool = False
    student_sample_steps: int = 1
    sample_t_cfg: SampleTConfig = field(default_factory=SampleTConfig)
    skip_layers: List[int] | None = None
    gan_r1_reg_weight: float = 0.0
    gan_r1_reg_alpha: float = 0.0
    fake_score_pred_type: str = "eps"
    t_precision: str = "bfloat16"

    # flow_head_steps
    # flow_head_fuse_type
    # flow_head_num_blocks
    # flow_head_t_list
    # main_t_to_flow_r_map
    # flow_head_target_type

    # fake_score_optimizer
    # fake_score_scheduler
    # fake_score_pred_type
    # discriminator
    # discriminator_optimizer
    # discriminator_scheduler
    # add_teacher_to_model_dict
