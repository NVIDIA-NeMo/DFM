# Reve Training Commands

Quick reference for running and configuring DFM Reve vs baseline Reve.

<br>

## Running scripts

| Variant       | Script |
|---------------|--------|
| **DFM Reve**  | `./examples/megatron/recipes/reve/launch_h100_nodes.sh`, `./examples/megatron/recipes/reve/launch_gb200_nodes.sh` |
| **Baseline Reve** | `./examples/megatron/recipes/reve/launch_h100_nodes_baseline.sh` |

<br>

## Modifying model size

| Variant       | Where to change |
|---------------|-----------------|
| **DFM Reve**  | `./examples/megatron/recipes/reve/launch_h100_nodes.sh` — adjust `--model-size` argument |
| **Baseline Reve** | `./examples/megatron/recipes/reve/launch_h100_nodes_baseline.sh` — adjust `--config` argument |

**Memory tuning:** For DFM Reve, we have the options to reduce memory via layers recomputation, which is needed for running full-size (25B) model with large input data,  by setting:
- `model.recompute_granularity=full`
- `model.recompute_method=block`
- `model.recompute_num_layers=N` (e.g. N=13, 20, or 26 for a balance of recomputation vs memory)

<br>

## Modifying data batch

| Variant          | CLI settings |
|------------------|--------------|
| **DFM Reve**     | `dataset.number_packed_samples`, `dataset.context_seq_len`, `dataset.H_latents`, `dataset.W_latents` |
| **Baseline Reve** | `bs`, `num_img_tokens`, `num_txt_tokens` |


**Note:** For DFM Reve, because we traing with sequence packing, the literal `micro_batch_size` per data-parallel rank is always 1; the actual effective `micro_batch_size` is determined by `number_packed_samples` (number of samples packed in one input sequence). For example, DFM Reve's setup of `micro_batch_size=1, number_packed_samples=8` corresponds to baseline Reve's setup of `bs=8`.

<br>

## Running on H100 cluster (1 node × 8 GPUs per node)

Request an interactive node:

```bash
salloc -p batch --time 4:00:00 -N 1 -A coreai_dlalgo_llm --gres=gpu:8
```

Setting container:
```
CONT="nvcr.io/nvidian/nemo:25.11.rc4"
```

**DFM Reve:**

```bash
export M_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
srun --jobid <job_id> -N 1 --ntasks-per-node=1 \
  --container-image="${CONT}" \
  --container-mounts=/lustre:/lustre \
  --export=ALL,MY_MASTER_ADDR="$M_ADDR" \
  bash ./examples/megatron/recipes/reve/launch_h100_nodes.sh
```

**Baseline Reve:**

```bash
export M_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
srun --jobid <job_id> -N 1 --ntasks-per-node=1 \
  --container-image="${CONT}" \
  --container-mounts=/lustre:/lustre \
  --export=ALL,MY_MASTER_ADDR="$M_ADDR" \
  bash ./examples/megatron/recipes/reve/launch_h100_nodes_baseline.sh
```

<br>

## Running on GB200 cluster (2 nodes × 4 GPUs per node)

Request nodes:

```bash
salloc -p batch --time 4:00:00 -N 2 -A coreai_dlalgo_llm --gres=gpu:4
```

Setting container:
```
CONT="nvcr.io/nvidian/nemo:25.11.rc4"
```

**DFM Reve:**

```bash
export M_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
srun --jobid <job_id> -N 2 --ntasks-per-node=1 \
  --container-image="${CONT}" \
  --container-mounts=/lustre:/lustre \
  --export=ALL,MY_MASTER_ADDR="$M_ADDR" \
  bash ./examples/megatron/recipes/reve/launch_gb200_nodes.sh
```
