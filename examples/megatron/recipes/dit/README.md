# DiT (Diffusion Transformer) Model Setup

This guide provides instructions for setting up and running the DiT model on the butterfly dataset.

## Overview

Megatron-LM and Megatron-Bridge coming with the docker image are not compatible with DiT model. This setup guide will walk you through configuring the environment properly.

## Setup Instructions

### 1. Clone Required Repositories

The following repositories need to be cloned with specific commit hashes:

#### Megatron-LM
```bash
git clone https://github.com/NVIDIA/Megatron-LM.git
cd Megatron-LM
git checkout aecce9e95624ddfedbd2bd3ce599e36cd96da065
cd ..
```

#### Megatron-Bridge
```bash
git clone https://github.com/NVIDIA-NeMo/Megatron-Bridge.git
cd Megatron-Bridge
git checkout 83f90524f9bc467e8f864e2f9bd1da1246594ab9
cd ..
```

#### DFM Repository
```bash
git clone https://github.com/NVIDIA-NeMo/DFM.git
cd DFM
git checkout dit_debug
cd ..
```

### 2. Dataset Location

The butterfly webdataset is accesible on eos clusters in the path below:
```
/home/snorouzi/code/butterfly_webdataset
```

## Docker Setup

Run the following Docker command to start the container with all necessary volume mounts:

```bash
sudo docker run --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 --gpus all -w /opt/dfm --rm \
    -v ${DATA_PATH}/butterfly_webdataset:/opt/VFM/butterfly_webdataset \
    -v ${CODE_PATH}/Megatron-LM:/opt/megatron-lm \
    -v ${CODE_PATH}/Megatron-Bridge/:/opt/Megatron-Bridge/ \
    -v ${CODE_PATH}/DFM:/opt/dfm \
    -it nvcr.io/nvidian/nemo:25.09.rc6
```

**Note:** Set the `DATA_PATH` and `CODE_PATH` environment variables to point to your local directories before running this command.

## Installation

Once inside the container, install the required Python packages:

```bash
pip install --upgrade transformers
pip install imageio==2.24
pip install imageio[ffmpeg]
```

## Running the Model

Execute the DiT model training with the following command:

```bash
torchrun --nproc-per-node 2 examples/megatron/recipes/dit/pretrain_dit_model.py --dataset_path "/opt/VFM/butterfly_webdataset"
```