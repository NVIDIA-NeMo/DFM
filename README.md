# NeMo VFM: video foundation model collection

NeMo VFM is a state-of-the-art framework for fast, large-scale training and inference of video world models. It unifies the latest diffusion-based and autoregressive techniques, prioritizing efficiency and performance from research prototyping to production deployment.

## Projects

This collection consists of 4 projects:
1. [Scalable diffusion training framework](nemo_vfm/diffusion/readme.rst)
2. [Accelerated diffusion world models](nemo_vfm/physicalai/Cosmos/cosmos1/models/diffusion/README.md)
3. [Accelerated autoregressive world models](nemo_vfm/physicalai/Cosmos/cosmos1/models/autoregressive/README.md)
4. [Sparse attention for efficient diffusion inference](nemo_vfm/sparse_attention/README.md)

## Citations

If you find our code useful, please consider citing the following papers:
```bibtex
@article{patel2025training,
  title={Training Video Foundation Models with NVIDIA NeMo},
  author={Patel, Zeeshan and He, Ethan and Mannan, Parth and Ren, Xiaowei and Wolf, Ryan and Agarwal, Niket and Huffman, Jacob and Wang, Zhuoyao and Wang, Carl and Chang, Jack and others},
  journal={arXiv preprint arXiv:2503.12964},
  year={2025}
}

@article{agarwal2025cosmos,
  title={Cosmos world foundation model platform for physical ai},
  author={Agarwal, Niket and Ali, Arslan and Bala, Maciej and Balaji, Yogesh and Barker, Erik and Cai, Tiffany and Chattopadhyay, Prithvijit and Chen, Yongxin and Cui, Yin and Ding, Yifan and others},
  journal={arXiv preprint arXiv:2501.03575},
  year={2025}
}
```


# How to run

1. Setup the environment and mount the repo:
```bash
docker run --gpus all  --ipc=host --ulimit memlock=-1 --ulimit stack=6710886 -it -v ~/.cache:/root/.cache  -v  ~/code/cursor/VFM:/workspace/VFM gitlab-master.nvidia.com:5005/dl/nemo/nemo-vfm:25.02.rc3
```

2. Inside the docker container. Make the data. Use single GPU for now:

```bash
export CUDA_VISIBLE_DEVICES=0
torchrun --nproc-per-node 1 nemo_vfm/diffusion/data/prepare_energon_dataset_butterfly.py --factory prepare_dummy_image_dataset
```

3. Step 2 will create 1000 samples of images in webdataset format. Use energon
```bash
energon prepare .
```
In this step choose to prepare the dataset.yaml interactively and use CrudeWebDataset format.


4. The energon does not product the right dataset.yaml file, so copy the one we have

```bash
cp dataset.yaml <PATH_TO_WEBPATH>/.nv-meta/
```

5. Run training 
```bash
   torchrun --nproc-per-node 8 nemo/collections/diffusion/train.py --yes --factory pretrain_xl
```
