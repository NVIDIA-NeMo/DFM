import sys
import torch
import torch.nn as nn
from unittest.mock import MagicMock
import os

# Mock missing dependencies
sys.modules["beartype"] = MagicMock()
sys.modules["jaxtyping"] = MagicMock()
sys.modules["einops"] = MagicMock()
sys.modules["einops.layers"] = MagicMock()
sys.modules["einops.layers.torch"] = MagicMock()

# Mock specific imports from jaxtyping that are used as types
class MockType:
    def __getitem__(self, args):
        return self
    
    def __or__(self, other):
        return self
        
    def __ror__(self, other):
        return self

mock_type_instance = MockType()

sys.modules["jaxtyping"].Bool = mock_type_instance
sys.modules["jaxtyping"].Float = mock_type_instance
sys.modules["jaxtyping"].Integer = mock_type_instance
sys.modules["jaxtyping"].jaxtyped = lambda *args, **kwargs: (lambda x: x)
sys.modules["beartype"].beartype = lambda x: x
sys.modules["einops.layers.torch"].Rearrange = lambda *args, **kwargs: nn.Identity()

# Add current directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# Now import the model
try:
    from model import ReveV2
except ImportError:
    # If running from a different directory, we might need to adjust
    print("Could not import model.ReveV2 directly. Trying to adjust path...")
    sys.path.append(current_dir)
    from model import ReveV2

def get_full_config():
    """Full production config."""
    return {
        "latent_dims": 768,
        "text_dims": 4096,
        "dims_per_head": 256,
        "num_heads": 24,
        "cross_dims_per_head": 256,
        "cross_num_heads": 24,
        "mlp_ratio": 4.0,
        "num_layers": 26,
        "cross_num_layers": 8,
        "rope_dims": [64, 64],
        "cross_rope_dims": 128,
        "rope_max_wavelength": 512.0,
        "attn_scale": 16.0,
        "patch_size": 1,
    }

def main():
    config = get_full_config()
    print("Instantiating ReveV2 with config:")
    for k, v in config.items():
        print(f"  {k}: {v}")
    print("-" * 50)
    
    try:
        model = ReveV2(**config)
    except Exception as e:
        print(f"Error instantiating model: {e}")
        import traceback
        traceback.print_exc()
        return

    total_params = 0
    print(f"{'Parameter Name':<60} {'Shape'}")
    print("-" * 80)
    
    for name, param in model.named_parameters():
        shape_str = str(tuple(param.shape))
        print(f"{name:<60} {shape_str}")
        total_params += param.numel()
        
    print("-" * 80)
    print(f"Total Parameters: {total_params:,}")

if __name__ == "__main__":
    main()
