from megatron.bridge.models.hf_pretrained.wan import PreTrainedWAN
from megatron.bridge.models.wan.wan_bridge import WanBridge
from megatron.bridge.training.model_load_save import save_megatron_model
import os, random
os.environ["MASTER_ADDR"] = "127.0.0.1"
os.environ["MASTER_PORT"] = str(29500 + random.randint(0, 1000))
os.environ["RANK"] = "0"
os.environ["WORLD_SIZE"] = "1"
os.environ["LOCAL_RANK"] = "0"
#
hf = PreTrainedWAN("Wan-AI/Wan2.1-T2V-1.3B-Diffusers")
# hf = PreTrainedWAN("Wan-AI/Wan2.1-T2V-14B-Diffusers")
bridge = WanBridge()
#
provider = bridge.provider_bridge(hf)
provider.perform_initialization = False
megatron_models = provider.provide_distributed_model(wrap_with_ddp=False, use_cpu_initialization=True)
#
bridge.load_weights_hf_to_megatron(hf, megatron_models)
save_megatron_model(megatron_models, "/opt/megatron_checkpoint", hf_tokenizer_path=None)