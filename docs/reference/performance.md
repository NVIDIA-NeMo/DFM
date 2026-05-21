---
description: "Performance benchmarks and training throughput metrics for NeMo DFM across different GPU systems"
categories: ["reference"]
tags: ["performance", "benchmarks", "throughput", "gpu"]
personas: ["mle-focused", "admin-focused"]
difficulty: "reference"
content_type: "reference"
---

(ref-performance)=

# Performance Benchmarks

NeMo DFM provides current performance benchmarks for models across different GPU systems and configurations. These benchmarks help you understand expected training throughput and optimize your training setup.

:::{note}
For updated YAML configurations, refer to `examples/megatron/recipes/wan/conf` in the repository.
:::

## Nomenclature

Understanding the terminology used in performance benchmarks:

:::{dropdown} Parallelism Abbreviations
:icon: info

- **GBS**: Global Batch Size
- **MBS**: Micro Batch Size
- **FSDP**: Fully Sharded Data Parallel
  - FSDP = 1: use FSDP
  - FSDP = 0: use DDP (Distributed Data Parallel)
- **TP**: Tensor Parallel Size
- **SP**: Sequence Parallel
- **PP**: Pipeline Parallel Size
- **CP**: Context Parallel Size
- **VP**: Virtual Pipeline Parallel Size
- **EP**: Expert Parallel Size
:::

## Performance Metrics

We measure performance using:

- **Tokens/sec/GPU**: Throughput per GPU
- **Model TFLOP/sec/GPU**: Model floating-point operations per second per GPU

## Performance Summary

The performance data includes:

- **Pre-training Performance**: Throughput metrics for various model sizes and architectures
- **System Configurations**: Results across different GPU systems (DGX-GB200, DGX-GB300, DGX-H100)

---

## Megatron-Core Pre-Training Performance

Performance benchmarks using the Megatron-Core backend with Megatron-Bridge.

:::: {tab-set}

::: {tab-item} DGX-GB200

| Model | #-GPUs | GBS | MBS | Sequence Length | FSDP | TP | SP | PP | CP | VP | EP | Model TFLOP / sec / GPU |
|-------|--------|-----|-----|-----------------|------|----|----|----|----|----|----|-------------------------|
| WAN 2.1 14B | 32 | 64 | 1 | 37440 | 0 | 1 | 0 | 1 | 4 | 0 | 0 | 787.59 |

:::

::: {tab-item} DGX-GB300

| Model | #-GPUs | GBS | MBS | Sequence Length | FSDP | TP | SP | PP | CP | VP | EP | Model TFLOP / sec / GPU |
|-------|--------|-----|-----|-----------------|------|----|----|----|----|----|----|-------------------------|
| WAN 2.1 14B | 32 | 64 | 1 | 37440 | 0 | 1 | 0 | 1 | 2 | 0 | 0 | 1,022.26 |

:::

::: {tab-item} DGX-H100

| Model | #-GPUs | GBS | MBS | Sequence Length | FSDP | TP | SP | PP | CP | VP | EP | Model TFLOP / sec / GPU |
|-------|--------|-----|-----|-----------------|------|----|----|----|----|----|----|-------------------------|
| WAN 2.1 14B | 128 | 128 | 1 | 37440 | 0 | 2 | 1 | 1 | 4 | 0 | 0 | 325.77 |

:::

::::

---

## NeMo AutoModel Pre-Training Performance

Performance benchmarks using the NeMo AutoModel backend with FSDP2.

### System: DGX-H100

| Model | #-GPUs | GBS | MBS | Sequence Length | FSDP | DP | TP | SP | PP | CP | VP | EP | Model TFLOP / sec / GPU |
|-------|--------|-----|-----|-----------------|------|----|----|----|----|----|----|----|-------------------------|
| WAN 2.1 14B | 8 | 8 | 1 | 37440 | 1 | 8 | 1 | 1 | 1 | 1 | 0 | 0 | 175.88 |
| WAN 2.1 14B | 64 | 64 | 1 | 37440 | 1 | 64 | 1 | 1 | 1 | 1 | 0 | 0 | 228.85 |

## Related Documentation

- [Distributed Training](../about/concepts/distributed-training.md) - Learn about parallelism strategies
- [Training Paradigms](../about/concepts/training-paradigms.md) - Understand AutoModel vs Megatron differences
- [Get Started](../get-started/index.md) - Start training with NeMo DFM
