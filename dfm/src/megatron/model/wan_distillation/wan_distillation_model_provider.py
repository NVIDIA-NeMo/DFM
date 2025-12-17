from dataclasses import dataclass
from typing import Optional

from dfm.src.megatron.model.wan.wan_provider import WanModelProvider
from dfm.src.megatron.model.wan_distillation.wan_distillation_model import DMDDistillModel


@dataclass
class DMDModelProvider(WanModelProvider):
    teacher_model_provider: Optional[WanModelProvider] = None
    fake_score_model_provider: Optional[WanModelProvider] = None

    def provide(self, pre_process=None, post_process=None, vp_stage=None):
        # Copy parallelism config to child providers
        for provider in [self.teacher_model_provider, self.fake_score_model_provider]:
            if provider is not None:
                provider.tensor_model_parallel_size = self.tensor_model_parallel_size
                provider.pipeline_model_parallel_size = self.pipeline_model_parallel_size
                provider.virtual_pipeline_model_parallel_size = self.virtual_pipeline_model_parallel_size
                provider.context_parallel_size = self.context_parallel_size
                provider.sequence_parallel = self.sequence_parallel
                # Finalize to compute derived fields (e.g., kv_channels)
                if hasattr(provider, "finalize"):
                    provider.finalize()

        student_model = super().provide(pre_process, post_process, vp_stage)
        fake_score_model = self.fake_score_model_provider.provide(pre_process, post_process, vp_stage)
        teacher_model = self.teacher_model_provider.provide(pre_process, post_process, vp_stage)
        return DMDDistillModel(
            config=self,
            student=student_model,
            fake_score=fake_score_model,
            teacher=teacher_model,
        )
