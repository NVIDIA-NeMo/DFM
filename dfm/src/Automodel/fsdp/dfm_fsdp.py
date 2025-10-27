from nemo_automodel.components.distributed.fsdp2 import FSDP2Manager
from Automodel.fsdp.parallelize import get_parallelization_strategy

class DFMFSDPManager(FSDP2Manager):
    def parallelize_diffusion_model(self, model):
        strategy = get_parallelization_strategy(model)
        return strategy.parallelize(model)