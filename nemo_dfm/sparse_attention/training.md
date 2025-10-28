Clone the repository

```bash
git clone https://gitlab-master.nvidia.com/sgovande/Cosmos/ Cosmos-gitlab
git clone https://gitlab-master.nvidia.com/dl/nemo/nemo-vfm nemo-dfm
cd nemo-dfm && git checkout training-sparse-attn && cd ..
```

Modify `Cosmos-gitlab/cosmos1/models/diffusion/nemo/post_training/multinode.sh` with the following:
1. Update paths to `nemo-dfm`
2. Update the environment variable `MOUNT` to point to your new location

```bash
cd Cosmos-gitlab
cd cosmos1/models/diffusion/nemo/post_training
sbatch multinode.sh
```