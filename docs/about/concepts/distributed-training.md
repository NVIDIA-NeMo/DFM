---
description: "Understanding distributed training parallelism in NeMo DFM: tensor parallelism, context parallelism, pipeline parallelism, and data parallelism"
categories: ["concepts-architecture"]
tags: ["distributed", "parallelism", "training", "tensor-parallelism"]
personas: ["mle-focused", "admin-focused"]
difficulty: "intermediate"
content_type: "explanation"
---

(about-concepts-distributed-training)=

# Distributed Training

NeMo DFM scales training across multiple GPUs and nodes using four parallelism strategies. These strategies address different bottlenecks: model size (TP, PP), sequence length (CP), and throughput (DP).

## Overview

| Type | What It Splits | When to Use | Communication |
|------|----------------|-------------|---------------|
| **Tensor Parallelism (TP)** | Model weights across GPUs | Model >40 GB per GPU | High-bandwidth (NVLink) |
| **Context Parallelism (CP)** | Sequence tokens across GPUs | Sequences >32K tokens | High-bandwidth (NVLink) |
| **Pipeline Parallelism (PP)** | Model layers across GPUs | Very deep models, multi-node | Low-bandwidth (point-to-point) |
| **Data Parallelism (DP)** | Training batches across GPUs | Standard scaling | Standard (all-reduce) |

**Example**: A 70B parameter model with 16K sequence length on 128 GPUs might use TP=4, CP=2, PP=2, DP=8.

## Tensor Parallelism (TP)

Splits model weights across GPUs within each layer. A 40 GB layer with TP=4 uses 10 GB per GPU.

### How It Works

For a matrix multiplication `Y = XW`:
1. Weight matrix `W` is split column-wise across GPUs
2. Each GPU computes partial result using its weight shard
3. Results are combined via all-reduce operation

**Example**: For a 12,288 × 12,288 weight matrix with TP=4, each GPU holds 12,288 × 3,072.

### When to Use

- **Model size**: Model parameters >40 GB per GPU
- **Layer size**: Individual layers >10 GB
- **Hardware**: GPUs connected via NVLink or high-speed interconnect

**Typical configurations**:
- TP=2: 70B-175B models on A100 80GB
- TP=4: 175B-400B models on H100 80GB
- TP=8: >400B models or limited GPU memory

### Configuration

**Automodel**:
```yaml
fsdp:
  tp_size: 4  # Split across 4 GPUs
  cp_size: 1
  pp_size: 1
  dp_size: 2  # Calculated automatically if not specified
```

**Megatron**:
```python
model.tensor_model_parallel_size = 4
```

### Performance Impact

- **Memory**: Reduces per-GPU memory by `1/tp_size`
- **Communication**: All-reduce after each layer forward/backward pass
- **Bandwidth requirement**: High-bandwidth interconnect (NVLink, NVSwitch) required for efficient scaling

## Context Parallelism (CP)

Splits sequence tokens across GPUs. A 64K token sequence with CP=2 processes 32K tokens per GPU.

### How It Works

For attention computation:
1. Sequence split into chunks across GPUs
2. Each GPU computes attention for its chunk
3. Key-value pairs shared via all-gather
4. Results combined for full attention

**Example**: A 64K token sequence with CP=4 splits into 4 chunks of 16K tokens, reducing attention memory by 75%.

### When to Use

- **Sequence length**: >32K tokens or frames
- **Memory bottleneck**: Attention memory exceeds 40% of total
- **Use case**: Video generation (100+ frames), long-context language models

**Typical configurations**:
- CP=2: 32K-64K token sequences
- CP=4: 64K-128K token sequences
- CP=8: >128K token sequences

### Configuration

**Automodel**:
```yaml
fsdp:
  tp_size: 1
  cp_size: 2  # Split sequence across 2 GPUs
  pp_size: 1
  dp_size: 4
```

**Megatron**:
```python
model.context_parallel_size = 2
```

### Performance Impact

- **Memory**: Reduces attention memory by `1/cp_size`
- **Communication**: All-gather for key-value pairs per attention layer
- **Scaling**: Most effective when attention is memory bottleneck

## Pipeline Parallelism (PP)

Splits model layers across GPUs or nodes. A 48-layer model with PP=4 assigns 12 layers per stage.

### How It Works

Model divided into sequential stages:
1. Stage 1 (GPU 0): Layers 1-12
2. Stage 2 (GPU 1): Layers 13-24
3. Stage 3 (GPU 2): Layers 25-36
4. Stage 4 (GPU 3): Layers 37-48

Activations flow forward through stages; gradients flow backward. Microbatching overlaps computation to reduce idle time.

### When to Use

- **Multi-node training**: Minimizes inter-node bandwidth requirements
- **Very deep models**: >80 layers that don't fit with TP alone
- **Heterogeneous networks**: Lower bandwidth between nodes than within

**Typical configurations**:
- PP=2: 2-node training with fast inter-node links
- PP=4: 4+ node training
- PP=8: Large-scale multi-node deployments

### Configuration

**Automodel**:
```yaml
fsdp:
  tp_size: 2
  cp_size: 1
  pp_size: 4  # 4 pipeline stages
  dp_size: 1
```

**Megatron**:
```python
model.pipeline_model_parallel_size = 4
```

### Performance Impact

- **Memory**: Reduces per-GPU memory by ~`1/pp_size`
- **Communication**: Point-to-point activation/gradient transfers between stages
- **Efficiency**: Pipeline bubbles cause idle time during stage transitions; mitigated by microbatching and virtual pipeline parallelism

## Data Parallelism (DP)

Replicates the model and splits batches across GPUs. Each GPU processes different data with the same model.

### How It Works

For batch size 64 with DP=8:
1. Each GPU gets 8 samples
2. Each GPU computes gradients independently
3. Gradients averaged across all GPUs via all-reduce
4. All GPUs update with averaged gradients

This increases effective batch size and training throughput.

### When to Use

- **Scaling throughput**: Increase samples per second
- **Batch size**: Increase effective batch size
- **Standard case**: After applying TP/CP/PP, use remaining GPUs for DP

**Typical configurations**:
- DP=8: Single 8-GPU node
- DP=16-32: Multi-node without model parallelism
- DP=4-16: Remaining GPUs after TP/CP/PP

### Configuration

**Automodel**:
```yaml
fsdp:
  tp_size: 1
  cp_size: 1
  pp_size: 1
  dp_size: 8  # 8 data parallel replicas
```

**Megatron**:
```python
# Automatically calculated: DP = total_gpus / (TP × CP × PP)
# Example: 32 GPUs with TP=4, CP=2, PP=2 → DP = 32/(4×2×2) = 2
```

### Performance Impact

- **Memory**: No memory savings (full model copy per GPU)
- **Communication**: All-reduce for gradients after each backward pass
- **Scaling**: Near-linear speedup; efficiency depends on batch size

## Combining Parallelism Strategies

All four parallelism types can be combined. Total GPUs = TP × CP × PP × DP.

### Real-World Examples

**Small model, long sequences (8 GPUs)**:
```yaml
# Video generation: 13B model, 128K frames
fsdp:
  tp_size: 1   # Model fits on single GPU
  cp_size: 4   # Split long sequence
  pp_size: 1   # No pipeline needed
  dp_size: 2   # Use remaining GPUs for throughput
```

**Large model, standard sequences (64 GPUs)**:
```yaml
# Language model: 175B model, 8K tokens
fsdp:
  tp_size: 4   # Split large model
  cp_size: 1   # Sequence fits in memory
  pp_size: 2   # 2-node deployment
  dp_size: 8   # Scale throughput
```

**Massive model, multi-node (256 GPUs)**:
```yaml
# 500B+ model across 32 nodes
fsdp:
  tp_size: 8   # Within-node parallelism
  cp_size: 2   # Moderate sequences
  pp_size: 4   # Across-node parallelism
  dp_size: 4   # Remaining GPUs
```

### Design Principles

1. **Start with TP**: If model doesn't fit, add TP first (requires high bandwidth)
2. **Add CP if needed**: For sequences >32K tokens
3. **Use PP for multi-node**: Pipeline across nodes to reduce inter-node traffic
4. **Fill with DP**: Use remaining GPUs for data parallelism

## Choosing Parallelism Strategy

### Decision Flowchart

**Step 1**: Model fits on single GPU?
- **Yes**: Use DP only (simplest, most efficient)
- **No**: Go to Step 2

**Step 2**: Single node or multi-node?
- **Single node (8 GPUs)**: Use TP=2 or TP=4, then DP
- **Multi-node (16+ GPUs)**: Go to Step 3

**Step 3**: Configure multi-node strategy
1. Use **PP** across nodes (minimize inter-node bandwidth)
2. Use **TP** within nodes (leverage NVLink)
3. Add **CP** if sequences >32K tokens
4. Use **DP** for remaining GPUs

### Hardware-Specific Guidance

**8x A100 80GB (single node)**:
```yaml
# 70B model, 8K tokens
fsdp:
  tp_size: 2
  cp_size: 1
  pp_size: 1
  dp_size: 4
```

**4 nodes × 8 H100 80GB (32 GPUs)**:
```yaml
# 175B model, 16K tokens
fsdp:
  tp_size: 4   # Within node
  cp_size: 2   # Long sequences
  pp_size: 2   # Across nodes (4 → 2 nodes per stage)
  dp_size: 2   # Remaining GPUs
```

**32 nodes × 8 H100 80GB (256 GPUs)**:
```yaml
# 500B model, 8K tokens
fsdp:
  tp_size: 8   # Full node
  cp_size: 1   # Standard sequences
  pp_size: 4   # Across nodes
  dp_size: 8   # Remaining GPUs
```

### Performance vs Memory Trade-offs

| Priority | Strategy | Rationale |
|----------|----------|-----------|
| **Maximum speed** | DP only | No communication overhead, if model fits |
| **Fit large model** | TP first | Most memory reduction per communication cost |
| **Long sequences** | CP | Only option for >32K tokens |
| **Multi-node scaling** | PP | Minimizes expensive inter-node bandwidth |

## Implementation Details

### Automodel (FSDP2)

Automodel uses FSDP2 (Fully Sharded Data Parallel) with automatic optimizations:

- **Weight sharding**: Distributes model weights across DP ranks
- **Gradient synchronization**: Overlaps communication with computation
- **Optimizer state sharding**: Distributes optimizer states across DP ranks to reduce per-GPU memory
- **Checkpointing**: Saves only one copy regardless of DP size

Best for: Standard training workflows with minimal tuning.

**Note**: Configure all parallelism dimensions in the `fsdp:` section of your YAML config. The framework handles DP calculation automatically if `dp_size` is not specified.

### Megatron

Megatron provides explicit control over parallelism configuration:

- **Fine-grained tuning**: Set communication schedules and buffer sizes
- **Custom patterns**: Optimize for specific network topologies
- **Large-scale focus**: Optimized for 100+ GPU deployments

Best for: Large-scale training requiring custom optimization.

### Verifying Parallelism Configuration

To check your current parallelism settings at runtime:

**Megatron**:
```python
from megatron.core import parallel_state as ps

tp_size = ps.get_tensor_model_parallel_world_size()
cp_size = ps.get_context_parallel_world_size()
pp_size = ps.get_pipeline_model_parallel_world_size()
# DP is calculated: dp_size = world_size / (tp_size * cp_size * pp_size)
```

**Automodel**:
Check your configuration YAML or training logs for the applied parallelism settings. The framework logs parallelism configuration at initialization.
