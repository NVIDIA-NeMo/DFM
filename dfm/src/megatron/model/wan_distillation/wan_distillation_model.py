from megatron.bridge.models.transformer_config import TransformerConfig
from megatron.core.models.common.vision_module.vision_module import VisionModule
from torch import Tensor

from dfm.src.megatron.model.wan.wan_model import WanModel


class SubModelProxy:
    """Proxy that routes forward calls through the wrapped composite model.

    This preserves DDP gradient sync and Float16Module precision handling.
    """

    def __init__(self, wrapped_model, submodel_attr: str):
        """
        Args:
            wrapped_model: The DDP/Float16-wrapped WanDMD2Model
            submodel_attr: 'student', 'fake_score', or 'teacher'
        """
        self._wrapped_model = wrapped_model
        self._submodel_attr = submodel_attr

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, *args, **kwargs):
        """Forward through the full wrapper stack with submodel routing."""
        kwargs["_route_to_submodel"] = self._submodel_attr
        # Keep outputs in model precision (bf16) for consistent backward pass
        # Float16Module defaults to fp32_output=True which causes dtype mismatch
        kwargs["fp32_output"] = False
        return self._wrapped_model(*args, **kwargs)

    def _get_submodel(self):
        """Get the unwrapped submodel."""
        from megatron.core.utils import unwrap_model

        dmd2 = unwrap_model(self._wrapped_model)
        if isinstance(dmd2, list):
            dmd2 = dmd2[0]
        return getattr(dmd2, self._submodel_attr)

    def train(self, mode=True):
        """Set training mode for the submodel."""
        return self._get_submodel().train(mode)

    def eval(self):
        """Set evaluation mode for the submodel."""
        return self._get_submodel().eval()

    def requires_grad_(self, requires_grad=True):
        """Set requires_grad for all parameters in the submodel."""
        for param in self._get_submodel().parameters():
            param.requires_grad_(requires_grad)
        return self


class DMDDistillModel(VisionModule):
    def __init__(self, config: TransformerConfig, student: WanModel, fake_score: WanModel, teacher: WanModel):
        super().__init__(config)
        self.teacher = teacher
        self.student = student
        self.fake_score = fake_score

    def forward(
        self,
        x,
        t,
        condition=None,
        grid_sizes=None,
        _route_to_submodel=None,  # <- routing flag
        packed_seq_params=None,
        **kwargs,
    ):
        """
        If _route_to_submodel is set, only run that submodel.
        Otherwise, run the full DMD2 pipeline.
        """
        if _route_to_submodel == "student":
            return self.student(
                x, t=t, context=condition, grid_sizes=grid_sizes, packed_seq_params=packed_seq_params, **kwargs
            )
        elif _route_to_submodel == "teacher":
            return self.teacher(
                x, t=t, context=condition, grid_sizes=grid_sizes, packed_seq_params=packed_seq_params, **kwargs
            )
        elif _route_to_submodel == "fake_score":
            return self.fake_score(
                x, t=t, context=condition, grid_sizes=grid_sizes, packed_seq_params=packed_seq_params, **kwargs
            )

    def set_input_tensor(self, input_tensor: Tensor) -> None:
        """Sets input tensor to the model.

        See megatron.model.transformer.set_input_tensor()

        Args:
            input_tensor (Tensor): Sets the input tensor for the model.
        """
        # Delegate to the main network (net) for pipeline parallelism
        if not isinstance(input_tensor, list):
            input_tensor = [input_tensor]

        assert len(input_tensor) == 1, "input_tensor should only be length 1 for gpt/bert"
        self.student.decoder.set_input_tensor(input_tensor[0])

    def get_submodel_proxies(self, wrapped_model=None):
        """Create proxies for FastGen that route through wrappers.

        Args:
            wrapped_model: The DDP/Float16-wrapped version of this model.
                          If None, uses self (for non-DDP cases).

        Returns:
            tuple: (student_proxy, teacher_proxy, fake_score_proxy)
        """
        return (
            SubModelProxy(wrapped_model, "student"),
            SubModelProxy(wrapped_model, "teacher"),
            SubModelProxy(wrapped_model, "fake_score"),
        )
