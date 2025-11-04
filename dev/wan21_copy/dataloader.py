import logging
import os
import pickle
from pathlib import Path
from typing import Dict, List, Optional

import torch
from torch.utils.data import DataLoader, Dataset


# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MetaFilesDataset(Dataset):
    """
    Optimized PyTorch Dataset for loading .meta files with prefetching support.

    Key optimizations:
    - Always loads to CPU in workers (pin_memory handles GPU transfer)
    - Supports high num_workers for parallel I/O
    - Optional in-memory caching for small datasets
    - Shape validation for batch compatibility
    """

    def __init__(
        self,
        meta_folder: str,
        transform_text: Optional[callable] = None,
        transform_video: Optional[callable] = None,
        filter_fn: Optional[callable] = None,
        max_files: Optional[int] = None,
        cache_in_memory: bool = False,
        validate_shapes: bool = True,  # NEW: Validate shapes for batching
    ):
        """
        Initialize the dataset.

        Args:
            meta_folder: Path to folder containing .meta files
            transform_text: Optional transform to apply to text embeddings
            transform_video: Optional transform to apply to video latents
            filter_fn: Optional function to filter files (takes metadata dict, returns bool)
            max_files: Optional limit on number of files to load (for testing)
            cache_in_memory: If True, cache all data in RAM (only for small datasets!)
            validate_shapes: If True, validate that all samples have consistent shapes
        """
        self.meta_folder = Path(meta_folder)
        self.transform_text = transform_text
        self.transform_video = transform_video
        self.filter_fn = filter_fn
        self.cache_in_memory = cache_in_memory
        self.validate_shapes = validate_shapes
        self.cache = {} if cache_in_memory else None

        # Find all .meta files
        self.meta_files = sorted(list(self.meta_folder.glob("*.meta")))

        # Apply file limit if specified
        if max_files is None:
            max_files_env = os.environ.get("MAX_META_FILES", None)
            if max_files_env is not None:
                max_files = int(max_files_env)

        if max_files is not None and max_files > 0:
            self.meta_files = self.meta_files[:max_files]
            logger.info(f"Limited to {max_files} files for testing")

        if not self.meta_files:
            raise ValueError(f"No .meta files found in {meta_folder}")

        # Filter files if filter function is provided
        if self.filter_fn:
            logger.info("Applying filter function to .meta files...")
            filtered_files = []
            for meta_file in self.meta_files:
                try:
                    with open(meta_file, "rb") as f:
                        data = pickle.load(f)
                    if self.filter_fn(data["metadata"]):
                        filtered_files.append(meta_file)
                except Exception as e:
                    logger.warning(f"Error loading {meta_file} for filtering: {e}")

            self.meta_files = filtered_files
            logger.info(f"Filtered to {len(self.meta_files)} files")

        logger.info(f"Found {len(self.meta_files)} .meta files in {meta_folder}")

        # Pre-cache if requested
        if self.cache_in_memory:
            logger.info("⚠️  Caching all data in memory - only use for small datasets!")
            self._preload_cache()

        # Validate shapes for batching compatibility
        if self.validate_shapes:
            self._validate_batch_compatibility()

        # Log some statistics about the dataset
        self._log_dataset_stats()

    def _validate_batch_compatibility(self):
        """
        Validate that all samples have compatible shapes for batching.
        This is CRITICAL for batch_size > 1.
        """
        logger.info("🔍 Validating shape compatibility for batching...")

        sample_size = min(10, len(self.meta_files))
        text_shapes = []
        video_shapes = []

        for i in range(sample_size):
            try:
                if self.cache_in_memory and i in self.cache:
                    data = self.cache[i]
                else:
                    with open(self.meta_files[i], "rb") as f:
                        data = pickle.load(f)

                # Get shapes after squeezing batch dimensions
                text_emb = data["text_embeddings"]
                while text_emb.ndim > 3 and text_emb.shape[0] == 1:
                    text_emb = text_emb.squeeze(0)
                text_shapes.append(tuple(text_emb.shape))

                vid_lat = data["video_latents"]
                while vid_lat.ndim > 5 and vid_lat.shape[0] == 1:
                    vid_lat = vid_lat.squeeze(0)
                video_shapes.append(tuple(vid_lat.shape))

            except Exception as e:
                logger.warning(f"Error sampling {self.meta_files[i]}: {e}")

        # Check for shape consistency
        if text_shapes:
            unique_text_shapes = set(text_shapes)
            unique_video_shapes = set(video_shapes)

            if len(unique_text_shapes) > 1:
                logger.warning(f"⚠️  INCONSISTENT TEXT SHAPES DETECTED: {unique_text_shapes}")
                logger.warning("This will cause batching issues with batch_size > 1!")
            else:
                logger.info(f"✅ Text embeddings have consistent shape: {list(unique_text_shapes)[0]}")

            if len(unique_video_shapes) > 1:
                logger.warning(f"⚠️  INCONSISTENT VIDEO SHAPES DETECTED: {unique_video_shapes}")
                logger.warning("This will cause batching issues with batch_size > 1!")
                logger.warning("Consider filtering dataset or using batch_size=1")
            else:
                logger.info(f"✅ Video latents have consistent shape: {list(unique_video_shapes)[0]}")

    def _preload_cache(self):
        """Preload all data into memory - only for small datasets!"""
        logger.info(f"Preloading {len(self.meta_files)} files into memory...")

        for idx, meta_file in enumerate(self.meta_files):
            try:
                with open(meta_file, "rb") as f:
                    data = pickle.load(f)
                self.cache[idx] = data

                if (idx + 1) % 100 == 0:
                    logger.info(f"Cached {idx + 1}/{len(self.meta_files)} files")
            except Exception as e:
                logger.error(f"Error caching {meta_file}: {e}")
                self.cache[idx] = None

        logger.info(f"✅ Cached {len(self.cache)} files in memory")

    def _log_dataset_stats(self):
        """Log statistics about the dataset."""
        if len(self.meta_files) == 0:
            return

        # Sample a few files to get statistics
        sample_size = min(5, len(self.meta_files))
        text_shapes = []
        video_shapes = []
        encoding_modes = []

        for i in range(sample_size):
            try:
                # Use cache if available
                if self.cache_in_memory and i in self.cache:
                    data = self.cache[i]
                else:
                    with open(self.meta_files[i], "rb") as f:
                        data = pickle.load(f)

                text_shapes.append(tuple(data["text_embeddings"].shape))
                video_shapes.append(tuple(data["video_latents"].shape))
                encoding_modes.append(data.get("deterministic_latents", "unknown"))
            except Exception as e:
                logger.warning(f"Error sampling {self.meta_files[i]}: {e}")

        if text_shapes:
            logger.info(f"Sample text embedding shapes: {text_shapes[:3]}")
            logger.info(f"Sample video latent shapes: {video_shapes[:3]}")
            logger.info(f"Sample encoding modes: {set(encoding_modes)}")

    def __len__(self) -> int:
        """Return the number of .meta files."""
        return len(self.meta_files)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Load and return data from a .meta file.

        CRITICAL: Always returns tensors on CPU for proper DataLoader behavior.
        pin_memory=True in DataLoader will handle GPU transfer efficiently.

        Args:
            idx: Index of the file to load

        Returns:
            Dictionary containing tensors on CPU
        """
        meta_file = self.meta_files[idx]

        try:
            # Load from cache or disk
            if self.cache_in_memory and idx in self.cache:
                data = self.cache[idx]
                if data is None:
                    raise ValueError(f"Cached data is None for index {idx}")
            else:
                with open(meta_file, "rb") as f:
                    data = pickle.load(f)
        except Exception as e:
            logger.error(f"Error loading {meta_file}: {e}")
            raise

        # CRITICAL: Always keep tensors on CPU in dataset
        # DataLoader with pin_memory=True will handle GPU transfer
        text_embeddings = data["text_embeddings"].cpu()
        video_latents = data["video_latents"].cpu()

        # Apply transforms if provided
        if self.transform_text:
            text_embeddings = self.transform_text(text_embeddings)

        if self.transform_video:
            video_latents = self.transform_video(video_latents)

        # Prepare file info
        file_info = {
            "meta_filename": meta_file.name,
            "original_filename": data.get("original_filename", "unknown"),
            "original_video_path": data.get("original_video_path", "unknown"),
            "deterministic_latents": data.get("deterministic_latents", "unknown"),
            "memory_optimization": data.get("memory_optimization", "unknown"),
            "frame_index": data.get("frame_index", "unknown"),
            "total_frames_in_video": data.get("total_frames_in_video", "unknown"),
        }

        return {
            "text_embeddings": text_embeddings,
            "video_latents": video_latents,
            "metadata": data["metadata"],
            "file_info": file_info,
        }

    def get_file_info(self, idx: int) -> Dict:
        """Get file information without loading the full data."""
        meta_file = self.meta_files[idx]

        try:
            # Use cache if available
            if self.cache_in_memory and idx in self.cache:
                data = self.cache[idx]
            else:
                with open(meta_file, "rb") as f:
                    data = pickle.load(f)

            return {
                "meta_filename": meta_file.name,
                "original_filename": data.get("original_filename", "unknown"),
                "vila_caption": data["metadata"].get("vila_caption", "No caption"),
                "start_frame": data["metadata"].get("start_frame", "unknown"),
                "end_frame": data["metadata"].get("end_frame", "unknown"),
                "text_shape": tuple(data["text_embeddings"].shape),
                "video_shape": tuple(data["video_latents"].shape),
                "deterministic_latents": data.get("deterministic_latents", "unknown"),
                "memory_optimization": data.get("memory_optimization", "unknown"),
                "frame_index": data.get("frame_index", "unknown"),
            }
        except Exception as e:
            logger.error(f"Error getting info for {meta_file}: {e}")
            return {"error": str(e)}


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Custom collate function for batching .meta file data.

    FIXED: Enhanced debugging and error handling for shape mismatches.
    CRITICAL: Validates shapes before stacking to prevent hangs.

    Args:
        batch: List of data dictionaries from __getitem__

    Returns:
        Batched data dictionary
    """
    debug_mode = os.environ.get("DEBUG_DATALOADER", "0") == "1"

    # ========================================================================
    # Stack Text Embeddings
    # ========================================================================
    try:
        # Squeeze out batch dimension from each item before stacking
        text_embeddings_list = []
        text_shapes_before = []
        text_shapes_after = []

        for i, item in enumerate(batch):
            text_emb = item["text_embeddings"]
            text_shapes_before.append(text_emb.shape)

            # Remove batch dimension if present
            # FIXED: >= instead of > to properly squeeze (1, 512, 4096) -> (512, 4096)
            while text_emb.ndim >= 3 and text_emb.shape[0] == 1:
                text_emb = text_emb.squeeze(0)

            text_shapes_after.append(text_emb.shape)
            text_embeddings_list.append(text_emb)

        # Validate shapes before stacking
        first_shape = text_shapes_after[0]
        if len(batch) > 1:
            for i, shape in enumerate(text_shapes_after[1:], 1):
                if shape != first_shape:
                    logger.error("❌ TEXT EMBEDDING SHAPE MISMATCH!")
                    logger.error(f"   Sample 0: {first_shape}")
                    logger.error(f"   Sample {i}: {shape}")
                    logger.error(f"   Shapes before squeeze: {text_shapes_before}")
                    raise RuntimeError(f"Cannot batch text embeddings with different shapes: {first_shape} vs {shape}")

        text_embeddings = torch.stack(text_embeddings_list)

        if debug_mode:
            logger.info(f"✅ Text embeddings stacked: {text_embeddings.shape}")

    except RuntimeError as e:
        # Enhanced error logging
        logger.error(f"❌ Error stacking text_embeddings: {e}")
        logger.error(f"   Batch size: {len(batch)}")
        logger.error(f"   Original shapes: {text_shapes_before}")
        logger.error(f"   After squeeze: {text_shapes_after}")
        raise

    # ========================================================================
    # Stack Video Latents
    # ========================================================================
    try:
        # Squeeze out batch dimension from each item before stacking
        video_latents_list = []
        video_shapes_before = []
        video_shapes_after = []

        for i, item in enumerate(batch):
            vid_lat = item["video_latents"]
            video_shapes_before.append(vid_lat.shape)

            # Remove batch dimension if present
            # FIXED: >= instead of > to properly squeeze (1, 16, 1, 30, 52) -> (16, 1, 30, 52)
            while vid_lat.ndim >= 5 and vid_lat.shape[0] == 1:
                vid_lat = vid_lat.squeeze(0)

            video_shapes_after.append(vid_lat.shape)
            video_latents_list.append(vid_lat)

        # Validate shapes before stacking
        first_shape = video_shapes_after[0]
        if len(batch) > 1:
            for i, shape in enumerate(video_shapes_after[1:], 1):
                if shape != first_shape:
                    logger.error("❌ VIDEO LATENT SHAPE MISMATCH!")
                    logger.error(f"   Sample 0: {first_shape}")
                    logger.error(f"   Sample {i}: {shape}")
                    logger.error(f"   Shapes before squeeze: {video_shapes_before}")
                    logger.error(f"   File info sample 0: {batch[0]['file_info']['meta_filename']}")
                    logger.error(f"   File info sample {i}: {batch[i]['file_info']['meta_filename']}")
                    raise RuntimeError(f"Cannot batch video latents with different shapes: {first_shape} vs {shape}")

        video_latents = torch.stack(video_latents_list)

        # if debug_mode or len(batch) > 1:
        #     logger.info(f"✅ Video latents stacked: {video_latents.shape}")

    except RuntimeError as e:
        # Enhanced error logging
        logger.error(f"❌ Error stacking video_latents: {e}")
        logger.error(f"   Batch size: {len(batch)}")
        logger.error(f"   Original shapes: {video_shapes_before}")
        logger.error(f"   After squeeze: {video_shapes_after}")
        logger.error("   This usually indicates videos have different dimensions!")
        logger.error("   Solution: Use batch_size=1 or filter dataset to uniform dimensions")
        raise

    # Collect metadata and file info as lists
    metadata_list = [item["metadata"] for item in batch]
    file_info_list = [item["file_info"] for item in batch]

    return {
        "text_embeddings": text_embeddings,
        "video_latents": video_latents,
        "metadata": metadata_list,
        "file_info": file_info_list,
    }


def create_dataloader(
    meta_folder: str,
    batch_size: int = 4,
    shuffle: bool = True,
    num_workers: int = 4,
    prefetch_factor: int = 2,
    persistent_workers: bool = True,
    transform_text: Optional[callable] = None,
    transform_video: Optional[callable] = None,
    filter_fn: Optional[callable] = None,
    max_files: Optional[int] = None,
    cache_in_memory: bool = False,
    validate_shapes: bool = True,  # NEW: Validate shapes
) -> DataLoader:
    """
    Create an optimized DataLoader for .meta files with prefetching.

    Args:
        meta_folder: Path to folder containing .meta files
        batch_size: Batch size for the DataLoader
        shuffle: Whether to shuffle the data
        num_workers: Number of worker processes (recommended: 4-8 per GPU)
        prefetch_factor: Number of batches to prefetch per worker
        persistent_workers: Keep workers alive between epochs
        transform_text: Optional transform for text embeddings
        transform_video: Optional transform for video latents
        filter_fn: Optional function to filter files
        max_files: Optional limit on number of files to load
        cache_in_memory: Cache all data in RAM (only for small datasets!)
        validate_shapes: Validate shapes for batch compatibility

    Returns:
        DataLoader instance
    """
    dataset = MetaFilesDataset(
        meta_folder=meta_folder,
        transform_text=transform_text,
        transform_video=transform_video,
        filter_fn=filter_fn,
        max_files=max_files,
        cache_in_memory=cache_in_memory,
        validate_shapes=validate_shapes,
    )

    # Warn if batch_size > 1 and shape validation found issues
    if batch_size > 1 and validate_shapes:
        logger.info(f"⚠️  Using batch_size={batch_size} - ensure all samples have identical shapes!")

    # Optimize num_workers based on dataset size
    if len(dataset) < 100:
        # Small dataset - fewer workers, but keep at least 2 for prefetching
        num_workers = min(num_workers, 2)
        logger.info(f"Small dataset detected, using {num_workers} workers")

    # If caching in memory, fewer workers needed
    if cache_in_memory:
        num_workers = min(num_workers, 2)
        logger.info(f"In-memory cache enabled, using {num_workers} workers")

    # Ensure at least 2 workers for effective prefetching (if requested)
    if num_workers == 0:
        logger.warning("num_workers=0 disables prefetching - consider using 2-4 workers for better performance")

    # Create optimized dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,  # CRITICAL: Enables async GPU transfer
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        persistent_workers=persistent_workers if num_workers > 0 else False,
        drop_last=True,  # Drop incomplete batches for training stability
        timeout=60 if num_workers > 0 else 0,  # Timeout only works with workers
    )

    logger.info("✅ Created DataLoader:")
    logger.info(f"   - Batch size: {batch_size}")
    logger.info(f"   - Num workers: {num_workers}")
    if num_workers > 0:
        logger.info(f"   - Prefetch factor: {prefetch_factor} (prefetches {prefetch_factor * num_workers} batches)")
        logger.info(f"   - Persistent workers: {persistent_workers}")
        logger.info("   - Timeout: 60s")
    else:
        logger.info("   - Prefetch: DISABLED (num_workers=0)")
    logger.info("   - Pin memory: True")
    logger.info(f"   - Cache in memory: {cache_in_memory}")

    return dataloader


# Example filter functions
def filter_by_duration(min_frames: int = 10, max_frames: int = 100):
    """Create a filter function based on video duration."""

    def filter_fn(metadata):
        duration = metadata["end_frame"] - metadata["start_frame"] + 1
        return min_frames <= duration <= max_frames

    return filter_fn


def filter_by_caption_length(min_length: int = 10, max_length: int = 200):
    """Create a filter function based on caption length."""

    def filter_fn(metadata):
        caption = metadata.get("vila_caption", "")
        return min_length <= len(caption) <= max_length

    return filter_fn


def filter_by_shape(expected_video_shape: tuple):
    """
    Create a filter function that only allows videos with specific shape.
    CRITICAL for batch_size > 1!

    Args:
        expected_video_shape: e.g., (16, 21, 60, 104) for channels, frames, height, width

    Returns:
        Filter function
    """

    def filter_fn(metadata):
        # This would need access to the actual data, so this is示意性的
        # In practice, you'd need to load the file to check shape
        # Better to validate shapes during dataset initialization
        return True

    return filter_fn


# Example transforms
def normalize_text_embeddings(embeddings):
    """Normalize text embeddings to unit length."""
    return torch.nn.functional.normalize(embeddings, dim=-1)


def add_noise_to_video_latents(noise_scale: float = 0.01):
    """Add small amount of noise to video latents."""

    def transform_fn(latents):
        noise = torch.randn_like(latents) * noise_scale
        return latents + noise

    return transform_fn
