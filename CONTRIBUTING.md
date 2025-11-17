# Contributing To NeMo DFM
## 🛠️ Setting Up Your Environment

Use the instructions below to setup a dev environment and a dev container

### Building a container
```bash
# Initialize all submodules (Megatron-Bridge, Automodel, and nested Megatron-LM)
git submodule update --init --recursive

# Build the container
docker build -f docker/Dockerfile.ci -t dfm:latest .
```

### Run the container
```bash
docker run --gpus all -v $(pwd):/opt/DFM -it dfm:latest bash
```

### Inside the container
```bash
# Install DFM in editable mode (automatically handles Python path)
source /opt/venv/bin/activate
uv pip install --no-deps -e .

# Run a Mock Run:
```

## Signing Your Work

### Sign-Off (Required)

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

### Commit Verification (Recommended)

* We recommend signing your commits with SSH or GPG to get the "Verified" badge on GitHub.

* **SSH Signing (Easiest):**
  ```bash
  # Configure SSH signing
  git config --global gpg.format ssh
  git config --global user.signingkey ~/.ssh/id_rsa.pub  # or id_ed25519.pub
  git config --global commit.gpgsign true

  # Add your SSH key as a "Signing Key" on GitHub: https://github.com/settings/keys
  ```

* **GPG Signing (Alternative):**
  ```bash
  # Generate a GPG key
  gpg --full-generate-key

  # Configure GPG signing
  git config --global user.signingkey YOUR_GPG_KEY_ID
  git config --global commit.gpgsign true

  # Add your GPG key to GitHub: https://github.com/settings/keys
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
