# Configuration Overrides

Collection of YAML configuration files used to override default settings in Megatron training recipes. These are typically used for specifying parallelization strategies (Tensor Parallelism, Context Parallelism, Sequence Parallelism) or data configurations.

## Files

| File | Description |
|------|-------------|
| `wan_pretrain_sample_data.yaml` | Sample data configuration for Wan pre-training. |

## Usage

These configs can be passed to the training script arguments to override defaults.
