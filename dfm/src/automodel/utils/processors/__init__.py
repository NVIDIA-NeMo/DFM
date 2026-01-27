"""
Model processors for preprocessing pipeline.

This module provides a model-agnostic interface for preprocessing different
model architectures. Each processor handles model loading, image encoding,
text encoding, and cache formatting specific to its architecture.

Supported Processors:
    - flux: FLUX.1 architecture (VAE + CLIP + T5)
    
Usage:
    from processors import ProcessorRegistry, BaseModelProcessor
    
    # Get available processors
    print(ProcessorRegistry.list_available())  # ['flux']
    
    # Get a processor instance
    processor = ProcessorRegistry.get("flux")
    
    # Load models
    models = processor.load_models("black-forest-labs/FLUX.1-dev", "cuda")
    
    # Encode image and text
    latent = processor.encode_image(image_tensor, models, "cuda")
    text_enc = processor.encode_text("a photo of a cat", models, "cuda")

Adding New Processors:
    1. Create a new file in this directory (e.g., sdxl.py)
    2. Subclass BaseModelProcessor
    3. Use @ProcessorRegistry.register("name") decorator
    4. Import the file in this __init__.py
"""

from .base import BaseModelProcessor
from .registry import ProcessorRegistry

# Import processors to register them
from . import flux  # noqa: F401 - imported for side effect (registration)

# Future processors can be added here:
# from . import sdxl  # noqa: F401
# from . import sd15  # noqa: F401
# from . import sd3   # noqa: F401

__all__ = [
    "BaseModelProcessor",
    "ProcessorRegistry",
]

