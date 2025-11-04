import html
import json
import logging
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import regex as re
import torch
from diffusers import AutoencoderKLWan
from diffusers.utils import is_ftfy_available
from transformers import AutoTokenizer, UMT5EncoderModel

if is_ftfy_available():
    import ftfy

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


# Text cleaning functions (matching WanPipeline implementation)
def basic_clean(text):
    """Fix text encoding issues and unescape HTML entities."""
    if is_ftfy_available():
        text = ftfy.fix_text(text)
    text = html.unescape(html.unescape(text))
    return text.strip()


def whitespace_clean(text):
    """Normalize whitespace by replacing multiple spaces with single space."""
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    return text


def prompt_clean(text):
    """Clean prompt text exactly as done in WanPipeline."""
    text = whitespace_clean(basic_clean(text))
    return text


class VideoPreprocessor:
    def __init__(
        self,
        video_folder: str = "clipped_video",
        wan21_model_id: str = "Wan-AI/Wan2.1-T2V-14B-Diffusers",
        output_folder: str = "processed_meta",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        deterministic_latents: bool = True,
        enable_memory_optimization: bool = True,
        target_size: Optional[Tuple[int, int]] = None,
        resize_mode: str = "bilinear",
        maintain_aspect_ratio: bool = True,
        center_crop: bool = False,
        mode: str = "video",  # "video" or "frames"
        num_frames_per_video: int = 10,  # Only used in "frames" mode
    ):
        """
        Initialize the video preprocessor for Wan2.1 fine-tuning.

        Args:
            video_folder: Path to folder containing videos and meta.json
            wan21_model_id: Hugging Face model ID for Wan2.1
            output_folder: Path to folder where .meta files will be saved
            device: Device to run inference on
            deterministic_latents: If True, use posterior mean instead of sampling
            enable_memory_optimization: Enable Wan's built-in slicing and tiling
            target_size: Target (height, width) for resizing. If None, no resizing is performed
            resize_mode: Interpolation mode for resizing ('bilinear', 'bicubic', 'nearest', 'area')
            maintain_aspect_ratio: If True, maintain aspect ratio when resizing
            center_crop: If True, center crop to target size after resizing (when maintaining aspect ratio)
            mode: Processing mode - "video" for full videos, "frames" for extracting frames
            num_frames_per_video: Number of evenly-spaced frames to extract (only for "frames" mode)
        """
        self.video_folder = Path(video_folder)
        self.output_folder = Path(output_folder)
        self.device = device
        self.wan21_model_id = wan21_model_id
        self.deterministic_latents = deterministic_latents
        self.enable_memory_optimization = enable_memory_optimization
        self.mode = mode
        self.num_frames_per_video = num_frames_per_video

        # Validate mode
        if self.mode not in ["video", "frames"]:
            raise ValueError(f"Invalid mode '{self.mode}'. Choose from: 'video', 'frames'")

        logger.info(f"Processing mode: {self.mode.upper()}")
        if self.mode == "frames":
            logger.info(f"Will extract {self.num_frames_per_video} evenly-spaced frames per video")
        else:
            logger.info("Will process full videos")

        # Resize parameters
        self.target_size = target_size
        self.resize_mode = resize_mode
        self.maintain_aspect_ratio = maintain_aspect_ratio
        self.center_crop = center_crop

        # Validate resize parameters
        if self.target_size is not None:
            if len(self.target_size) != 2 or any(s <= 0 for s in self.target_size):
                raise ValueError("target_size must be a tuple of (height, width) with positive values")
            logger.info(f"Video resizing enabled: target size = {self.target_size} (H x W)")
            logger.info(f"Resize mode: {self.resize_mode}")
            logger.info(f"Maintain aspect ratio: {self.maintain_aspect_ratio}")
            if self.maintain_aspect_ratio and self.center_crop:
                logger.info("Center crop enabled (will crop to exact target size after aspect-preserving resize)")
        else:
            logger.info("Video resizing disabled - using original video dimensions")

        # Map resize modes to OpenCV interpolation flags
        self.interpolation_map = {
            "bilinear": cv2.INTER_LINEAR,
            "bicubic": cv2.INTER_CUBIC,
            "nearest": cv2.INTER_NEAREST,
            "area": cv2.INTER_AREA,
            "lanczos": cv2.INTER_LANCZOS4,
        }

        if self.resize_mode not in self.interpolation_map:
            raise ValueError(
                f"Invalid resize_mode '{self.resize_mode}'. Choose from: {list(self.interpolation_map.keys())}"
            )

        # Log the encoding mode
        if self.deterministic_latents:
            logger.info("Using DETERMINISTIC latents (posterior mean) - no flares expected")
        else:
            logger.info("Using STOCHASTIC latents (sampling) - may cause temporal flares")

        # Log memory optimization setting
        if self.enable_memory_optimization:
            logger.info("Using Wan's built-in memory optimization (slicing + tiling)")
        else:
            logger.info("Memory optimization disabled - using full tensors")

        # Create output directory if it doesn't exist
        self.output_folder.mkdir(parents=True, exist_ok=True)
        logger.info(f"Output folder created/verified: {self.output_folder}")

        # Load Wan2.1 components
        logger.info(f"Loading Wan2.1 components from {wan21_model_id}...")
        self.text_encoder = self._load_text_encoder()
        self.vae = self._load_vae()
        self.tokenizer = self._load_tokenizer()

        # Load metadata
        self.metadata = self._load_metadata()

    def _load_text_encoder(self):
        """Load Wan2.1 UMT5 text encoder from Hugging Face."""
        logger.info("Loading UMT5 text encoder...")
        text_encoder = UMT5EncoderModel.from_pretrained(
            self.wan21_model_id,
            subfolder="text_encoder",
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
        )
        text_encoder.to(self.device)
        text_encoder.eval()
        logger.info("UMT5 text encoder loaded successfully")
        return text_encoder

    def _load_vae(self):
        """Load Wan2.1 VAE from Hugging Face with memory optimization."""
        logger.info("Loading Wan2.1 VAE...")

        # Load Wan2.1 VAE with correct subfolder
        try:
            vae = AutoencoderKLWan.from_pretrained(
                self.wan21_model_id,
                subfolder="vae",
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            )
        except Exception as e:
            logger.error(f"Failed to load VAE from {self.wan21_model_id}/vae")
            logger.error(f"Error: {e}")
            logger.info("Make sure you're using a valid Wan2.1 model:")
            logger.info("- Wan-AI/Wan2.1-T2V-14B-Diffusers")
            logger.info("- Wan-AI/Wan2.1-T2V-1.3B-Diffusers")
            raise

        vae.to(self.device)
        vae.eval()

        # Enable Wan's built-in memory optimization
        if self.enable_memory_optimization:
            logger.info("Enabling Wan VAE memory optimization...")
            vae.enable_slicing()
            vae.enable_tiling()
            logger.info("✅ Enabled slicing and tiling for memory efficiency")
        else:
            logger.info("Memory optimization disabled - using full tensors")

        # Log VAE configuration
        logger.info("Wan2.1 VAE loaded successfully")
        if hasattr(vae.config, "z_dim"):
            logger.info(f"VAE z_dim (latent channels): {vae.config.z_dim}")
        if hasattr(vae.config, "latents_mean"):
            logger.info(f"VAE latents_mean: {vae.config.latents_mean}")
        if hasattr(vae.config, "latents_std"):
            logger.info(f"VAE latents_std: {vae.config.latents_std}")

        scale_factor_temporal = getattr(vae.config, "scale_factor_temporal", 4)
        scale_factor_spatial = getattr(vae.config, "scale_factor_spatial", 8)
        logger.info(f"VAE scale_factor_temporal: {scale_factor_temporal}")
        logger.info(f"VAE scale_factor_spatial: {scale_factor_spatial}")

        return vae

    def _load_tokenizer(self):
        """Load UMT5 tokenizer from Hugging Face."""
        logger.info("Loading UMT5 tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(self.wan21_model_id, subfolder="tokenizer")
        logger.info("Tokenizer loaded successfully")
        return tokenizer

    def _load_metadata(self) -> List[Dict]:
        """Load video metadata from meta.json."""
        meta_json_path = self.video_folder / "meta.json"

        if not meta_json_path.exists():
            raise FileNotFoundError(f"meta.json not found in {self.video_folder}")

        logger.info(f"Loading metadata from {meta_json_path}...")
        with open(meta_json_path, "r") as f:
            metadata = json.load(f)

        logger.info(f"Loaded metadata for {len(metadata)} videos")
        return metadata

    def _calculate_resize_dimensions(self, original_height: int, original_width: int) -> Tuple[int, int]:
        """
        Calculate resize dimensions based on target size and aspect ratio settings.

        Returns:
            Tuple of (resize_height, resize_width)
        """
        if self.target_size is None:
            return original_height, original_width

        target_height, target_width = self.target_size

        if not self.maintain_aspect_ratio:
            # Stretch to exact target size
            return target_height, target_width

        # Maintain aspect ratio
        original_aspect = original_width / original_height
        target_aspect = target_width / target_height

        if self.center_crop:
            # Resize so the smaller dimension matches target, then crop
            if original_aspect > target_aspect:
                # Original is wider, match height
                resize_height = target_height
                resize_width = int(target_height * original_aspect)
            else:
                # Original is taller, match width
                resize_width = target_width
                resize_height = int(target_width / original_aspect)
        else:
            # Resize so the larger dimension matches target (fit inside)
            if original_aspect > target_aspect:
                # Original is wider, match width
                resize_width = target_width
                resize_height = int(target_width / original_aspect)
            else:
                # Original is taller, match height
                resize_height = target_height
                resize_width = int(target_height * original_aspect)

        return resize_height, resize_width

    def _resize_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Resize a single frame according to the configured settings.

        Args:
            frame: RGB numpy array (H, W, C) with values in [0, 255]

        Returns:
            Resized frame with same format
        """
        if self.target_size is None:
            return frame

        original_height, original_width = frame.shape[:2]
        resize_height, resize_width = self._calculate_resize_dimensions(original_height, original_width)

        # Resize using OpenCV
        interpolation = self.interpolation_map[self.resize_mode]
        resized_frame = cv2.resize(frame, (resize_width, resize_height), interpolation=interpolation)

        # Apply center crop if enabled
        if self.maintain_aspect_ratio and self.center_crop:
            target_height, target_width = self.target_size

            # Calculate crop coordinates
            if resized_frame.shape[0] > target_height or resized_frame.shape[1] > target_width:
                y_start = max(0, (resized_frame.shape[0] - target_height) // 2)
                x_start = max(0, (resized_frame.shape[1] - target_width) // 2)
                y_end = y_start + target_height
                x_end = x_start + target_width

                # Crop the frame
                resized_frame = resized_frame[y_start:y_end, x_start:x_end]

                # Pad if necessary
                if resized_frame.shape[0] < target_height or resized_frame.shape[1] < target_width:
                    pad_height = max(0, target_height - resized_frame.shape[0])
                    pad_width = max(0, target_width - resized_frame.shape[1])
                    resized_frame = np.pad(
                        resized_frame, ((0, pad_height), (0, pad_width), (0, 0)), mode="constant", constant_values=0
                    )

        return resized_frame

    def extract_first_frame(self, video_path: str, start_frame: int) -> np.ndarray:
        """
        Extract the first frame from the video as RGB image.

        Args:
            video_path: Path to video file
            start_frame: Starting frame index

        Returns:
            First frame as RGB numpy array (H, W, C) with values in [0, 255]
        """
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        ret, frame = cap.read()
        cap.release()

        if not ret:
            raise ValueError(f"Could not read frame {start_frame} from {video_path}")

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_rgb = self._resize_frame(frame_rgb)

        logger.info(f"Extracted first frame with shape: {frame_rgb.shape}")
        return frame_rgb

    def extract_evenly_spaced_frames(
        self, video_path: str, start_frame: int, end_frame: int, num_frames: int = 10
    ) -> List[np.ndarray]:
        """
        Extract evenly spaced frames from the video as RGB images.

        Args:
            video_path: Path to video file
            start_frame: Starting frame index
            end_frame: Ending frame index
            num_frames: Number of frames to extract

        Returns:
            List of frames as RGB numpy arrays (H, W, C) with values in [0, 255]
        """
        cap = cv2.VideoCapture(video_path)

        # Calculate frame indices
        total_frames = end_frame - start_frame + 1
        if num_frames > total_frames:
            logger.warning(
                f"Requested {num_frames} frames but video only has {total_frames} frames. Extracting all available."
            )
            num_frames = total_frames

        # Calculate evenly spaced indices
        if num_frames == 1:
            frame_indices = [start_frame]
        else:
            frame_indices = np.linspace(start_frame, end_frame, num_frames, dtype=int).tolist()

        logger.info(f"Extracting {num_frames} frames at indices: {frame_indices}")

        extracted_frames = []
        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()

            if not ret:
                logger.warning(f"Could not read frame {frame_idx} from {video_path}, skipping...")
                continue

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_rgb = self._resize_frame(frame_rgb)
            extracted_frames.append(frame_rgb)

        cap.release()

        if not extracted_frames:
            raise ValueError(f"Could not extract any frames from {video_path}")

        logger.info(f"Extracted {len(extracted_frames)} frames with shape: {extracted_frames[0].shape}")
        return extracted_frames

    def load_video_frames(self, video_path: str, start_frame: int, end_frame: int) -> torch.Tensor:
        """
        Load video frames and convert to tensor for Wan VAE.

        Args:
            video_path: Path to video file
            start_frame: Starting frame index
            end_frame: Ending frame index

        Returns:
            Video tensor of shape (batch, channels, num_frames, height, width)
        """
        cap = cv2.VideoCapture(video_path)
        frames = []

        # Get original video dimensions for logging
        original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if self.target_size is not None:
            resize_height, resize_width = self._calculate_resize_dimensions(original_height, original_width)
            logger.info(f"Original dimensions: {original_height}x{original_width}")
            logger.info(f"Resize dimensions: {resize_height}x{resize_width}")
            if self.maintain_aspect_ratio and self.center_crop:
                logger.info(f"Final dimensions (after crop): {self.target_size[0]}x{self.target_size[1]}")

        # Set to start frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        for frame_idx in range(start_frame, end_frame + 1):
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = self._resize_frame(frame)
            frames.append(frame)

        cap.release()

        if not frames:
            raise ValueError(f"Could not load any frames from {video_path}")

        logger.info(f"Loaded {len(frames)} frames with shape: {frames[0].shape}")

        # Convert to tensor: (batch, channels, num_frames, height, width)
        frames_array = np.stack(frames, axis=0)
        frames_tensor = torch.from_numpy(frames_array).float() / 255.0
        frames_tensor = frames_tensor.permute(0, 3, 1, 2)
        frames_tensor = frames_tensor.permute(1, 0, 2, 3).unsqueeze(0)

        logger.info(f"Video tensor shape: {frames_tensor.shape}")
        return frames_tensor

    def encode_text(self, caption: str, max_sequence_length: int = 226) -> torch.Tensor:
        """
        Encode text caption using Wan2.1 UMT5 text encoder.
        Matches the exact preprocessing used in WanPipeline._get_t5_prompt_embeds.

        Args:
            caption: Text description of the video
            max_sequence_length: Maximum sequence length (default: 226, matching WanPipeline)

        Returns:
            Text embedding tensor of shape (1, max_sequence_length, hidden_dim)
        """
        # Clean the prompt EXACTLY as done in Wan2.1 pipeline
        caption = prompt_clean(caption)

        # Tokenize text with UMT5 settings (matching WanPipeline)
        text_inputs = self.tokenizer(
            caption,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_attention_mask=True,
            return_tensors="pt",
        )

        # Move to device
        text_input_ids = text_inputs.input_ids.to(self.device)
        mask = text_inputs.attention_mask.to(self.device)

        # Calculate actual sequence length (excluding padding)
        seq_lens = mask.gt(0).sum(dim=1).long()

        # Encode text using UMT5 encoder
        with torch.no_grad():
            prompt_embeds = self.text_encoder(text_input_ids, mask).last_hidden_state
            prompt_embeds = prompt_embeds.to(dtype=self.text_encoder.dtype, device=self.device)

            # CRITICAL: Trim to actual length and re-pad with zeros
            # This removes embeddings for padding tokens and replaces with proper zeros
            prompt_embeds = [u[:v] for u, v in zip(prompt_embeds, seq_lens)]
            prompt_embeds = torch.stack(
                [
                    torch.cat([u, u.new_zeros(max_sequence_length - u.size(0), u.size(1))])
                    for u in prompt_embeds
                ],
                dim=0,
            )

        logger.info(f"Text embeddings shape: {prompt_embeds.shape}")
        return prompt_embeds

    def encode_video(self, video_tensor: torch.Tensor) -> torch.Tensor:
        """
        Encode video using Wan2.1 VAE with per-channel normalization.

        Args:
            video_tensor: Video tensor of shape (batch, channels, num_frames, height, width)

        Returns:
            Normalized video latent tensor
        """
        logger.info(f"Input video tensor shape: {video_tensor.shape}, dtype: {video_tensor.dtype}")

        # Ensure tensor is on correct device and dtype
        video_tensor = video_tensor.to(device=self.device, dtype=self.vae.dtype)

        # Convert to [-1, 1] range for VAE
        video_tensor = video_tensor * 2.0 - 1.0

        logger.info("Encoding with Wan2.1 VAE...")

        with torch.no_grad():
            latent_dist = self.vae.encode(video_tensor)

            if self.deterministic_latents:
                video_latents = latent_dist.latent_dist.mean
                logger.info("Using deterministic posterior mean (no flares)")
            else:
                video_latents = latent_dist.latent_dist.sample()
                logger.info("Using stochastic sampling (may cause flares)")

        # Apply Wan2.1 VAE per-channel normalization
        if not hasattr(self.vae.config, "latents_mean") or not hasattr(self.vae.config, "latents_std"):
            raise ValueError("Wan2.1 VAE requires latents_mean and latents_std in config")

        latents_mean = torch.tensor(self.vae.config.latents_mean, device=self.device, dtype=self.vae.dtype)
        latents_std = torch.tensor(self.vae.config.latents_std, device=self.device, dtype=self.vae.dtype)

        # Reshape for broadcasting: (1, C, 1, 1, 1) for 5D tensors
        latents_mean = latents_mean.view(1, -1, 1, 1, 1)
        latents_std = latents_std.view(1, -1, 1, 1, 1)

        # Apply normalization
        normalized_latents = (video_latents - latents_mean) / latents_std

        logger.info(f"Output video latents shape: {normalized_latents.shape}")
        return normalized_latents

    def encode_single_frame_as_video(self, frame: np.ndarray) -> torch.Tensor:
        """
        Encode a single frame by treating it as a 1-frame video.

        Args:
            frame: RGB numpy array (H, W, C) with values in [0, 255]

        Returns:
            Video latent tensor of shape (batch, channels, 1, latent_h, latent_w)
        """
        # Normalize to [0, 1]
        frame_normalized = frame.astype(np.float32) / 255.0

        # Convert to tensor: (batch, channels, 1, height, width)
        frame_tensor = torch.from_numpy(frame_normalized)
        frame_tensor = frame_tensor.permute(2, 0, 1)  # (channels, height, width)
        frame_tensor = frame_tensor.unsqueeze(0).unsqueeze(2)  # (batch, channels, 1, height, width)

        target_dtype = torch.float16 if self.device == "cuda" else torch.float32
        frame_tensor = frame_tensor.to(dtype=target_dtype, device=self.device)

        logger.info(f"Frame tensor shape before encoding: {frame_tensor.shape}")

        # Encode using the video encoding method
        frame_latents = self.encode_video(frame_tensor)

        logger.info(f"Frame latents shape after encoding: {frame_latents.shape}")
        return frame_latents

    def save_processed_data(
        self,
        video_name: str,
        text_embeddings: torch.Tensor,
        video_latents: torch.Tensor,
        first_frame: Optional[np.ndarray],
        metadata: Dict,
        frame_index: Optional[int] = None,
    ):
        """
        Save processed text embeddings and video latents to binary file.

        Args:
            video_name: Original video filename
            text_embeddings: Encoded text embeddings
            video_latents: Encoded video latents
            first_frame: First frame as RGB numpy array (optional for frame mode)
            metadata: Original metadata for the video
            frame_index: Index of this frame (only for frames mode)
        """
        # Create output filename
        video_stem = Path(video_name).stem
        if self.mode == "frames" and frame_index is not None:
            output_path = self.output_folder / f"{video_stem}_{frame_index}.meta"
        else:
            output_path = self.output_folder / f"{video_stem}.meta"

        # Prepare data for saving
        processed_data = {
            "text_embeddings": text_embeddings.cpu(),
            "video_latents": video_latents.cpu(),
            "metadata": metadata,
            "original_filename": video_name,
            "original_video_path": str(self.video_folder / video_name),
            "deterministic_latents": self.deterministic_latents,
            "memory_optimization": self.enable_memory_optimization,
            "model_version": "wan2.1",
            "processing_mode": self.mode,
            "resize_settings": {
                "target_size": self.target_size,
                "resize_mode": self.resize_mode,
                "maintain_aspect_ratio": self.maintain_aspect_ratio,
                "center_crop": self.center_crop,
            },
        }

        # Add mode-specific metadata
        if self.mode == "frames" and frame_index is not None:
            processed_data["frame_index"] = frame_index
            processed_data["total_frames_in_video"] = self.num_frames_per_video
        elif first_frame is not None:
            processed_data["first_frame"] = first_frame

        # Save as pickle file
        with open(output_path, "wb") as f:
            pickle.dump(processed_data, f)

        file_size_mb = output_path.stat().st_size / (1024 * 1024)

        logger.info(f"✅ Saved processed data to {output_path}")
        logger.info(f"   Video latents shape: {video_latents.shape}")
        logger.info(f"   Text embeddings shape: {text_embeddings.shape}")
        logger.info(f"   File size: {file_size_mb:.2f} MB")

    def process_single_video_full(self, video_metadata: Dict):
        """Process a single video as a complete sequence."""
        video_name = video_metadata["file_name"]
        video_path = self.video_folder / video_name

        if not video_path.exists():
            logger.warning(f"Video file {video_path} not found, skipping...")
            return

        logger.info(f"Processing {video_name} (FULL VIDEO MODE)...")
        
        # Print caption for VIDEO mode
        print(f"\n[VIDEO] {video_name}")
        print(f"Caption: {video_metadata['vila_caption']}\n")

        try:
            # Extract first frame
            logger.info("Step 1: Extracting first frame...")
            first_frame = self.extract_first_frame(str(video_path), video_metadata["start_frame"])

            # Load video frames
            logger.info("Step 2: Loading video frames...")
            video_tensor = self.load_video_frames(
                str(video_path), video_metadata["start_frame"], video_metadata["end_frame"]
            )

            # Encode text caption
            logger.info("Step 3: Encoding text caption...")
            text_embeddings = self.encode_text(video_metadata["vila_caption"])

            # Encode video
            logger.info("Step 4: Encoding video with VAE...")
            video_latents = self.encode_video(video_tensor)

            # Save processed data
            logger.info("Step 5: Saving processed data...")
            self.save_processed_data(video_name, text_embeddings, video_latents, first_frame, video_metadata)

            logger.info(f"✅ Successfully processed {video_name}")

        except Exception as e:
            import traceback

            logger.error(f"Error processing {video_name}: {e}")
            logger.error(f"Full traceback:\n{traceback.format_exc()}")

    def process_single_video_frames(self, video_metadata: Dict):
        """Process a single video by extracting and encoding evenly-spaced frames."""
        video_name = video_metadata["file_name"]
        video_path = self.video_folder / video_name

        if not video_path.exists():
            logger.warning(f"Video file {video_path} not found, skipping...")
            return

        logger.info(f"Processing {video_name} (FRAMES MODE - extracting {self.num_frames_per_video} frames)...")
        
        # Print caption for FRAMES mode (only once per video, not per frame)
        print(f"\n[FRAMES] {video_name} (extracting {self.num_frames_per_video} frames)")
        print(f"Caption: {video_metadata['vila_caption']}\n")

        try:
            # Step 1: Extract evenly spaced frames
            logger.info(f"Step 1: Extracting {self.num_frames_per_video} evenly-spaced frames...")
            frames = self.extract_evenly_spaced_frames(
                str(video_path), video_metadata["start_frame"], video_metadata["end_frame"], self.num_frames_per_video
            )

            # Step 2: Encode text (once for all frames)
            logger.info("Step 2: Encoding text caption...")
            text_embeddings = self.encode_text(video_metadata["vila_caption"])

            # Step 3: Process each frame
            logger.info(f"Step 3: Encoding {len(frames)} frames...")
            saved_files = []

            for frame_idx, frame in enumerate(frames, start=1):
                logger.info(f"Processing frame {frame_idx}/{len(frames)}...")

                # Encode single frame as 1-frame video
                frame_latents = self.encode_single_frame_as_video(frame)

                # Save this frame's data
                self.save_processed_data(video_name, text_embeddings, frame_latents, None, video_metadata, frame_idx)
                saved_files.append(f"{Path(video_name).stem}_{frame_idx}.meta")

            logger.info(f"✅ Successfully processed all {len(frames)} frames from {video_name}")

        except Exception as e:
            import traceback

            logger.error(f"Error processing {video_name}: {e}")
            logger.error(f"Full traceback:\n{traceback.format_exc()}")

    def process_all_videos(self):
        """Process all videos in the folder."""
        logger.info(f"Starting to process {len(self.metadata)} videos...")
        logger.info(f"Model: Wan2.1 ({self.wan21_model_id})")
        logger.info(f"Mode: {self.mode.upper()}")
        if self.mode == "frames":
            logger.info(f"Extracting {self.num_frames_per_video} frames per video")

        for i, video_metadata in enumerate(self.metadata):
            logger.info(f"Progress: Video {i + 1}/{len(self.metadata)}")

            if self.mode == "video":
                self.process_single_video_full(video_metadata)
            else:  # mode == "frames"
                self.process_single_video_frames(video_metadata)

        logger.info("=" * 80)
        logger.info("✅ Finished processing all videos!")
        logger.info(f"Output folder: {self.output_folder}")
        logger.info("=" * 80)


def main():
    """Main function to run the preprocessing."""
    import argparse

    parser = argparse.ArgumentParser(description="Preprocess videos for Wan2.1 fine-tuning")
    parser.add_argument(
        "--video_folder", default="clipped_video", help="Path to folder containing videos and meta.json"
    )
    parser.add_argument(
        "--output_folder", default="processed_meta", help="Path to folder where .meta files will be saved"
    )
    parser.add_argument(
        "--model",
        default="Wan-AI/Wan2.1-T2V-14B-Diffusers",
        help="Wan2.1 model ID",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use")
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Use stochastic encoding (sampling) instead of deterministic",
    )
    parser.add_argument(
        "--no-memory-optimization", action="store_true", help="Disable Wan's built-in memory optimization"
    )

    # Processing mode
    parser.add_argument(
        "--mode",
        default="video",
        choices=["video", "frames"],
        help="Processing mode: 'video' for full videos, 'frames' to extract frames",
    )
    parser.add_argument(
        "--num-frames", type=int, default=10, help="Number of evenly-spaced frames to extract (frames mode only)"
    )

    # Resize arguments
    parser.add_argument("--height", type=int, default=None, help="Target height for video frames")
    parser.add_argument("--width", type=int, default=None, help="Target width for video frames")
    parser.add_argument(
        "--resize_mode",
        default="bilinear",
        choices=["bilinear", "bicubic", "nearest", "area", "lanczos"],
        help="Interpolation mode for resizing",
    )
    parser.add_argument(
        "--no-aspect-ratio",
        action="store_true",
        help="Disable aspect ratio preservation",
    )
    parser.add_argument(
        "--center-crop", action="store_true", help="Center crop to exact target size after aspect-preserving resize"
    )

    args = parser.parse_args()

    # Set target size
    target_size = None
    if args.height is not None and args.width is not None:
        target_size = (args.height, args.width)
    elif args.height is not None or args.width is not None:
        parser.error("Both --height and --width must be specified together")

    # Initialize preprocessor
    preprocessor = VideoPreprocessor(
        video_folder=args.video_folder,
        wan21_model_id=args.model,
        output_folder=args.output_folder,
        device=args.device,
        deterministic_latents=not args.stochastic,
        enable_memory_optimization=not args.no_memory_optimization,
        target_size=target_size,
        resize_mode=args.resize_mode,
        maintain_aspect_ratio=not args.no_aspect_ratio,
        center_crop=args.center_crop,
        mode=args.mode,
        num_frames_per_video=args.num_frames,
    )

    # Process all videos
    preprocessor.process_all_videos()


if __name__ == "__main__":
    main()