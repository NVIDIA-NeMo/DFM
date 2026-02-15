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


import logging
import os

from dfm.src.automodel.datasets.multiresolutionDataloader import build_flux_multiresolution_dataloader


def test_real_dataloader(cache_path: str):
    # Configure logging to see the initialization details
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if not os.path.exists(cache_path):
        print(f"ERROR: Cache directory not found at {cache_path}")
        return

    try:
        # 1. Initialize the real dataloader
        dataloader, sampler = build_flux_multiresolution_dataloader(
            cache_dir=cache_path,
            batch_size=2,  # Small batch for printing
            num_workers=2,  # Use a couple of workers to test multi-processing
            dynamic_batch_size=True,  # Test the bucket logic
            shuffle=True,
        )

        print("\n" + "=" * 50)
        print("DATALOADER LOADED SUCCESSFULLY")
        print(f"Total Batches: {len(dataloader)}")
        print("=" * 50 + "\n")

        # 2. Iterate through the first 2 batches
        pathes = []
        for batch_idx, batch in enumerate(dataloader):
            # if batch_idx >= 2:  # Stop after 2 batches to avoid flooding the console
            #    break

            print(f"--- Batch {batch_idx} ---")
            print(f"Keys in batch: {list(batch.keys())}")

            # Print Tensor Shapes
            print(f"Image Latents Shape: {batch['image_latents'].shape} (B, C, H, W)")

            if "text_embeddings" in batch:
                print(f"Text Embeds Shape:  {batch['text_embeddings']}")
                print(f"Pooled Embeds Shape: {batch['pooled_prompt_embeds']}")

            # Print Metadata for the first sample in the batch
            metadata = batch.get("metadata", {})
            print("\nSample Metadata (First item in batch):")
            print(f"  - Prompt: {metadata['prompts'][0][:100]}...")  # Truncated
            print(f"  - Path:   {metadata['image_paths'][0]}")
            print(f"  - Res:    {metadata['original_resolution'][0]} -> {metadata['crop_resolution'][0]}")
            print(f"  - Aspect: {metadata['aspect_ratios'][0]}")
            print("-" * 30 + "\n")
            pathes.append(metadata["image_paths"][0])
        unique_paths = list(set(pathes))
        print(f"Total paths: {len(pathes)}")
        print(f"Unique paths: {len(unique_paths)}")

    except Exception as e:
        logging.error(f"Failed to run dataloader: {e}", exc_info=True)


if __name__ == "__main__":
    # SET YOUR ACTUAL PATH HERE
    MY_CACHE_DIR = "/linnanw/Diffuser/FLUX/DATA"

    test_real_dataloader(MY_CACHE_DIR)
