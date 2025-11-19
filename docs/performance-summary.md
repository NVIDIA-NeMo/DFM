# Performance

As part of the NVIDIA NeMo Framework, DFM, provides the most recent training techniques for training advanced generative AI models, such as model parallelization, optimized attention mechanisms, and more, to achieve high training throughput.

This page provides the current performance benchmarks for large language models using DFM across different GPU systems and configurations as we continue to optimize the model for optimal performance. Please refer to `examples/megatron/recipes/wan/conf` for updated YAML configurations.

## Nomenclature

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

## Performance Metrics

Performance is measured using:
- **Tokens/sec/GPU**: Throughput per GPU
- **Model TFLOP/sec/GPU**: Model floating-point operations per second per GPU

```{contents}
:local:
:depth: 2
```

## Performance Summary for Large Language Models

Below are performance benchmarks for various large language models organized by release version.

The performance data includes:

- **Pre-training Performance**: Throughput metrics for various model sizes and architectures
- **System Configurations**: Results across different GPU systems (DGX-GB200, DGX-GB300, DGX-H100)

---

## Megatron-Core Pre-Training Performance

#### System: DGX-GB200

| Model | #-GPUs | GBS | MBS | Sequence Length | FSDP | TP | SP | PP | CP | VP | EP | Tokens / sec / GPU | Model TFLOP / sec / GPU |
|-------|--------|-----|-----|-----------------|------|----|----|----|----|----|----|-----------------------|-------------------------|
|Wan 2.1 14B|32|64|1|37440|0|1|0|1|4|0|0|4747.17|787.59|


#### System: DGX-GB300

| Model | #-GPUs | GBS | MBS | Sequence Length | FSDP | TP | SP | PP | CP | VP | EP | Tokens / sec / GPU | Model TFLOP / sec / GPU |
|-------|--------|-----|-----|-----------------|------|----|----|----|----|----|----|-----------------------|-------------------------|
|Wan 2.1 14B|32|64|1|37440|0|1|0|1|2|0|0|6161.63|1,022.26|

#### System: DGX-H100

| Model | #-GPUs | GBS | MBS | Sequence Length | FSDP | TP | SP | PP | CP | VP | EP | Tokens / sec / GPU | Model TFLOP / sec / GPU |
|-------|--------|-----|-----|-----------------|------|----|----|----|----|----|----|-----------------------|-------------------------|
|Wan 2.1 14B|64|64|1|37440|0|2|1|1|4|0|0|1866.47|309.66|

## Automodel Pre-Training Performance

