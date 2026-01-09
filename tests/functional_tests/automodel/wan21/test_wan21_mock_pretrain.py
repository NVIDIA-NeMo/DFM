# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Functional smoke tests for Automodel WAN pretrain mock runs.

This test runs the WAN 2.1 pretrain recipe with mock data using 2 GPUs
to verify the training pipeline works end-to-end.
"""

import os
import subprocess

import pytest


class TestAutomodelWanPretrain:
    """Test class for Automodel WAN pretrain functional tests."""

    @pytest.mark.run_only_on("GPU")
    def test_wan_pretrain_mock_short(self, tmp_path):
        """
        Short functional test for WAN pretrain with mock data.
        """
        # Set up temporary directory for checkpoints
        checkpoint_dir = os.path.join(tmp_path, "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Path to the mock config file
        mock_config_path = "tests/functional_tests/automodel/wan21/mock_configs/wan2_1_t2v_flow_mock.yaml"

        # Build the command for the mock run with minimal settings
        cmd = [
            "python",
            "-m",
            "torch.distributed.run",
            "--nproc_per_node=2",
            "examples/automodel/pretrain/pretrain.py",
            "-c",
            mock_config_path,
        ]

        # Run the command with a timeout
        result = None
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=900,  # 15 minute timeout for short test
                check=True,
            )

            # Basic verification that the run completed
            assert result.returncode == 0, f"Command failed with return code {result.returncode}"

        except subprocess.TimeoutExpired:
            pytest.fail("WAN pretrain mock short run exceeded timeout of 900 seconds (15 minutes)")
        except subprocess.CalledProcessError as e:
            result = e
            pytest.fail(f"WAN pretrain mock short run failed with return code {e.returncode}")
        finally:
            # Always print output for debugging
            if result is not None:
                print("STDOUT:", result.stdout)
                print("STDERR:", result.stderr)
