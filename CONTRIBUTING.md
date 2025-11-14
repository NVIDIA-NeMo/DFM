# Contributing To NeMo DFM
## 🛠️ Setting Up Your Environment

Use the instructions below to setup a dev environment and a dev container

### Building a container
```bash
# We recommend you to get the latest commits for Megatron-Bridge and Autmodel
# The easiest way to do that might be to remove the 3rdparty directly completely before running the following commands
git submodule update --init --recursive --remote # Get all the 3rd party submodules
cd 3rdparty/Megatron-Bridge/3rdparty/Megatron-LM # Megatron LM commit might be wrong
# Get the right megatron commit from here: https://github.com/NVIDIA-NeMo/Megatron-Bridge/tree/main/3rdparty
git checkout <commit_hash>
cd ../../../../
docker build -f docker/Dockerfile.ci -t dfm:latest .
```

### Run the container
```bash
docker run --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 --gpus all $(pwd):/opt/DFM -it dfm:latest bash
```

### inside the container
```bash
# Add DFM to PYTHONPATH
export PYTHONPATH=$PYTHONPATH:/opt/DFM

# Run a Mock Run:
```

## Signing Your Work

* We require that all contributors "sign-off" on their commits. This certifies that the contribution is your original work, or you have rights to submit it under the same license, or a compatible license.

  * Any contribution which contains commits that are not Signed-Off will not be accepted.

* To sign off on a commit you simply use the `--signoff` (or `-s`) option when committing your changes:
  ```bash
  $ git commit -s -m "Add cool feature."
  ```
  This will append the following to your commit message:
  ```
  Signed-off-by: Your Name <your@email.com>
  ```

* Full text of the DCO:

  ```
  Developer Certificate of Origin
  Version 1.1

  Copyright (C) 2004, 2006 The Linux Foundation and its contributors.

  Everyone is permitted to copy and distribute verbatim copies of this
  license document, but changing it is not allowed.


  Developer's Certificate of Origin 1.1

  By making a contribution to this project, I certify that:

  (a) The contribution was created in whole or in part by me and I
      have the right to submit it under the open source license
      indicated in the file; or

  (b) The contribution is based upon previous work that, to the best
      of my knowledge, is covered under an appropriate open source
      license and I have the right under that license to submit that
      work with modifications, whether created in whole or in part
      by me, under the same open source license (unless I am
      permitted to submit under a different license), as indicated
      in the file; or

  (c) The contribution was provided directly to me by some other
      person who certified (a), (b) or (c) and I have not modified
      it.

  (d) I understand and agree that this project and the contribution
      are public and that a record of the contribution (including all
      personal information I submit with it, including my sign-off) is
      maintained indefinitely and may be redistributed consistent with
      this project or the open source license(s) involved.
  ```
