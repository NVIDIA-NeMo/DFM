# Megatron Examples

Advanced recipes and configuration overrides for training models using the Megatron-Core backend.

## Available Model Recipes

| Recipe | Key Scripts | Description |
|--------|-------------|-------------|
| **[DiT](recipes/dit/README.md)** | • [Pretrain](recipes/dit/pretrain_dit_model.py) <br> • [Inference](recipes/dit/inference_dit_model.py) | Diffusion Transformer (DiT) training on butterfly dataset |
| **[Wan](recipes/wan/README.md)** | • [Pretrain](recipes/wan/pretrain_wan.py) <br> • [Inference](recipes/wan/inference_wan.py) | Wan 2.1 model pre-training and inference |

## Directory Structure

| Directory | Description |
|-----------|-------------|
| **[recipes](recipes/)** | Source code and scripts for the models above |
| **[override_configs](override_configs/)** | Configuration overrides for customizing parallelism (TP/CP/SP) |
