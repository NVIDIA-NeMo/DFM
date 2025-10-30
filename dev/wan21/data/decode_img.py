import logging
import os
import pickle
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from diffusers import AutoencoderKLWan
from PIL import Image


# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ImageDecoder:
    def __init__(
        self,
        wan21_model_id: str = "Wan-AI/Wan2.1-T2V-14B-Diffusers",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        enable_memory_optimization: bool = True,
    ):
        """
        Initialize the image decoder for converting .meta files back to images.

        Args:
            wan21_model_id: Hugging Face model ID for Wan2.1 VAE
            device: Device to run inference on
            enable_memory_optimization: Enable Wan's built-in slicing and tiling
        """
        self.device = device
        self.wan21_model_id = wan21_model_id
        self.enable_memory_optimization = enable_memory_optimization

        # Load Wan2.1 VAE for decoding
        logger.info(f"Loading Wan2.1 VAE from {wan21_model_id}...")
        self.vae = self._load_vae()

    def _load_vae(self):
        """Load Wan2.1 VAE from Hugging Face with memory optimization."""
        logger.info("Loading Wan2.1 VAE for decoding...")
        vae = AutoencoderKLWan.from_pretrained(
            self.wan21_model_id, subfolder="vae", torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
        )
        vae.to(self.device)
        vae.eval()

        # Enable Wan's built-in memory optimization
        if self.enable_memory_optimization:
            logger.info("Enabling Wan VAE memory optimization...")
            vae.enable_slicing()  # Reduce peak memory by slicing batch
            vae.enable_tiling()  # Tile H/W during encode+decode
            logger.info("✅ Enabled slicing and tiling for memory efficiency")
        else:
            logger.info("Memory optimization disabled - using full tensors")

        # Log VAE config for debugging
        logger.info("Wan2.1 VAE loaded successfully")

        # Verify this is a Wan2.1 VAE
        if hasattr(vae.config, "latents_mean") and hasattr(vae.config, "latents_std"):
            logger.info("✅ Found latents_mean and latents_std in VAE config (Wan2.1 format)")
            logger.info(f"   z_dim: {vae.config.z_dim if hasattr(vae.config, 'z_dim') else 'unknown'}")
        else:
            logger.error("❌ No latents_mean/latents_std found in VAE config")
            logger.error("This doesn't appear to be a Wan2.1 VAE!")
            raise ValueError("VAE config missing latents_mean and latents_std (required for Wan2.1)")

        return vae

    def load_meta_file(self, meta_path: str) -> Dict:
        """
        Load processed data from .meta file.

        Args:
            meta_path: Path to .meta file

        Returns:
            Dictionary containing text_embeddings, video_latents, and metadata
        """
        logger.info(f"Loading .meta file: {meta_path}")

        with open(meta_path, "rb") as f:
            data = pickle.load(f)

        # Log information about the loaded data
        logger.info(f"Loaded data keys: {list(data.keys())}")
        logger.info(f"Video latents shape: {data['video_latents'].shape}")
        logger.info(f"Text embeddings shape: {data['text_embeddings'].shape}")
        logger.info(f"Original filename: {data.get('original_filename', 'N/A')}")

        # Check if this is a single-frame latent
        latent_shape = data["video_latents"].shape
        if len(latent_shape) == 5:
            num_frames = latent_shape[2]  # (B, C, T, H, W)
            logger.info(f"Number of frames in latent: {num_frames}")
            if num_frames != 1:
                logger.warning(f"⚠️  Expected 1 frame but found {num_frames} frames!")

        # Check frame index info
        frame_index = data.get("frame_index", "N/A")
        total_frames = data.get("total_frames_in_video", "N/A")
        logger.info(f"Frame index: {frame_index}/{total_frames}")

        # Check model version
        model_version = data.get("model_version", "unknown")
        logger.info(f"Model version: {model_version}")
        if model_version != "wan2.1":
            logger.warning(f"⚠️  This .meta file was created with {model_version}, but you're using a Wan2.1 decoder!")
            logger.warning("   Decoding may not work correctly if versions don't match.")

        # Check encoding mode
        encoding_mode = data.get("deterministic_latents", "unknown")
        logger.info(f"Encoding mode: {encoding_mode}")

        return data

    def decode_image_latents(self, image_latents: torch.Tensor) -> torch.Tensor:
        """
        Decode single-frame image latents back to an image using Wan2.1 VAE.

        Args:
            image_latents: Normalized image latents from .meta file
                          Shape: (batch, channels, 1, latent_h, latent_w)

        Returns:
            Decoded image tensor in [0, 1] range
            Shape: (batch, channels, 1, height, width)
        """
        logger.info(f"Decoding image latents: {image_latents.shape}")
        logger.info(f"Input latents range: [{image_latents.min():.3f}, {image_latents.max():.3f}]")

        # Verify this is a single frame
        if len(image_latents.shape) == 5:
            if image_latents.shape[2] != 1:
                logger.warning(f"⚠️  Expected 1 frame but got {image_latents.shape[2]} frames. Using first frame only.")
                image_latents = image_latents[:, :, 0:1, :, :]

        # Move to device and ensure correct dtype
        image_latents = image_latents.to(device=self.device, dtype=self.vae.dtype)

        # De-normalize latents (reverse the Wan2.1 encoding normalization)
        # Wan2.1 uses per-channel normalization: (z - mean) / std
        # So we reverse it: z * std + mean
        if not (hasattr(self.vae.config, "latents_mean") and hasattr(self.vae.config, "latents_std")):
            raise ValueError("Wan2.1 VAE requires latents_mean and latents_std in config")

        latents_mean = torch.tensor(self.vae.config.latents_mean, device=self.device, dtype=self.vae.dtype)
        latents_std = torch.tensor(self.vae.config.latents_std, device=self.device, dtype=self.vae.dtype)

        logger.info("Using Wan2.1 per-channel de-normalization")
        logger.info(f"latents_mean: {latents_mean.tolist()}")
        logger.info(f"latents_std: {latents_std.tolist()}")

        # Reshape for broadcasting: (1, C, 1, 1, 1) for 5D tensors
        latents_mean = latents_mean.view(1, -1, 1, 1, 1)
        latents_std = latents_std.view(1, -1, 1, 1, 1)

        # De-normalize: z * std + mean (reverse of (z - mean) / std)
        image_latents = image_latents * latents_std + latents_mean
        logger.info("Applied Wan2.1 VAE de-normalization")
        logger.info(f"De-normalized latents range: [{image_latents.min():.3f}, {image_latents.max():.3f}]")

        logger.info(f"De-normalized latents shape: {image_latents.shape}")

        # Decode using VAE
        with torch.no_grad():
            logger.info("Decoding with Wan2.1 VAE (using built-in memory optimization)")
            decoded_image = self.vae.decode(image_latents).sample
            logger.info(f"Decoded image range: [{decoded_image.min():.3f}, {decoded_image.max():.3f}]")

        # Convert from [-1, 1] to [0, 1] range
        decoded_image = (decoded_image + 1.0) / 2.0
        logger.info(f"After [-1,1] to [0,1] conversion: [{decoded_image.min():.3f}, {decoded_image.max():.3f}]")

        # Clamp to ensure values are in [0, 1]
        decoded_image = torch.clamp(decoded_image, 0.0, 1.0)
        logger.info(f"After clamping: [{decoded_image.min():.3f}, {decoded_image.max():.3f}]")

        logger.info(f"Final decoded image shape: {decoded_image.shape}")
        return decoded_image

    def save_image(self, image_tensor: torch.Tensor, output_path: str, quality: int = 95, format: str = "png"):
        """
        Save decoded image tensor as image file (PNG or JPEG).

        Args:
            image_tensor: Decoded image tensor of shape (B, C, 1, H, W) in [0, 1] range
            output_path: Path to save the image file
            quality: Quality for JPEG (1-100, ignored for PNG)
            format: Output format ('png' or 'jpeg'/'jpg')
        """
        logger.info(f"Saving image to: {output_path}")

        # Convert tensor to numpy
        # From (B, C, 1, H, W) to (H, W, C)
        image_tensor = image_tensor.squeeze(0)  # Remove batch dimension: (C, 1, H, W)
        image_tensor = image_tensor.squeeze(1)  # Remove temporal dimension: (C, H, W)
        image_tensor = image_tensor.permute(1, 2, 0)  # (H, W, C)
        image_np = image_tensor.cpu().numpy()

        # Convert from [0, 1] to [0, 255] and to uint8
        image_np = (image_np * 255).astype(np.uint8)

        logger.info(f"Image array shape: {image_np.shape}")
        logger.info(f"Image array dtype: {image_np.dtype}")
        logger.info(f"Image array range: [{image_np.min()}, {image_np.max()}]")

        # Create output directory if it doesn't exist
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        # Determine output format
        format = format.lower()
        if format not in ["png", "jpg", "jpeg"]:
            logger.warning(f"Unknown format '{format}', defaulting to PNG")
            format = "png"

        # Use PIL for better quality control
        if image_np.shape[2] == 3:  # RGB
            pil_image = Image.fromarray(image_np, mode="RGB")
        elif image_np.shape[2] == 1:  # Grayscale
            pil_image = Image.fromarray(image_np.squeeze(2), mode="L")
        else:
            raise ValueError(f"Unexpected number of channels: {image_np.shape[2]}")

        # Save with appropriate settings
        if format == "png":
            pil_image.save(output_path, format="PNG", optimize=True)
        else:  # JPEG
            pil_image.save(output_path, format="JPEG", quality=quality, optimize=True)

        logger.info(f"✅ Successfully saved image: {output_path}")
        logger.info(f"   Image dimensions: {image_np.shape[0]}x{image_np.shape[1]}")
        logger.info(f"   Format: {format.upper()}")

    def decode_meta_to_image(self, meta_path: str, output_path: str, quality: int = 95, format: str = "png"):
        """
        Complete pipeline: load .meta file and save decoded image.

        Args:
            meta_path: Path to input .meta file
            output_path: Path to output image file
            quality: Quality for JPEG (1-100, ignored for PNG)
            format: Output format ('png' or 'jpeg'/'jpg')
        """
        logger.info(f"Converting {meta_path} -> {output_path}")

        try:
            # Load meta file
            data = self.load_meta_file(meta_path)

            # Extract image latents (stored as "video_latents" but it's a single frame)
            image_latents = data["video_latents"]

            # Decode latents to image
            decoded_image = self.decode_image_latents(image_latents)

            # Save as image
            self.save_image(decoded_image, output_path, quality, format)

            logger.info(f"✅ Successfully converted {meta_path} to {output_path}")

        except Exception as e:
            logger.error(f"❌ Error converting {meta_path}: {e}")
            import traceback

            logger.error(f"Full traceback:\n{traceback.format_exc()}")
            raise

    def decode_folder(self, meta_folder: str, output_folder: str, quality: int = 95, format: str = "png"):
        """
        Decode all .meta files in a folder to images.

        Args:
            meta_folder: Path to folder containing .meta files
            output_folder: Path to folder for output images
            quality: Quality for JPEG (1-100, ignored for PNG)
            format: Output format ('png' or 'jpeg'/'jpg')
        """
        meta_folder = Path(meta_folder)
        output_folder = Path(output_folder)

        # Create output directory
        output_folder.mkdir(parents=True, exist_ok=True)

        # Find all .meta files
        meta_files = sorted(list(meta_folder.glob("*.meta")))

        if not meta_files:
            logger.warning(f"No .meta files found in {meta_folder}")
            return

        logger.info(f"Found {len(meta_files)} .meta files to decode")

        # Determine file extension
        ext = ".png" if format.lower() == "png" else ".jpg"

        for i, meta_file in enumerate(meta_files):
            logger.info(f"Progress: {i + 1}/{len(meta_files)}")

            # Generate output filename (keep the same stem, change extension)
            output_file = output_folder / f"{meta_file.stem}{ext}"

            try:
                self.decode_meta_to_image(str(meta_file), str(output_file), quality, format)
            except Exception as e:
                logger.error(f"Failed to decode {meta_file}: {e}")
                continue

        logger.info(f"✅ Finished decoding {len(meta_files)} images to {output_folder}")


def main():
    """Main function to run the image decoding."""
    import argparse

    parser = argparse.ArgumentParser(description="Decode Wan2.1 single-frame .meta files to images")
    parser.add_argument("--input", "-i", required=True, help="Input .meta file or folder containing .meta files")
    parser.add_argument("--output", "-o", required=True, help="Output image file or folder for images")
    parser.add_argument(
        "--format", "-f", default="png", choices=["png", "jpg", "jpeg"], help="Output image format (default: png)"
    )
    parser.add_argument(
        "--quality", "-q", type=int, default=95, help="JPEG quality (1-100, default: 95, ignored for PNG)"
    )
    parser.add_argument(
        "--model",
        default="Wan-AI/Wan2.1-T2V-14B-Diffusers",
        help="Wan2.1 model ID (default: Wan-AI/Wan2.1-T2V-14B-Diffusers)",
    )
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use (default: auto-detect)"
    )
    parser.add_argument(
        "--no-memory-optimization", action="store_true", help="Disable Wan's built-in memory optimization"
    )

    args = parser.parse_args()

    # Initialize decoder
    decoder = ImageDecoder(
        wan21_model_id=args.model, device=args.device, enable_memory_optimization=not args.no_memory_optimization
    )

    input_path = Path(args.input)
    output_path = Path(args.output)

    if input_path.is_file() and input_path.suffix == ".meta":
        # Single file mode
        if not output_path.suffix:
            # If output path has no extension, add one based on format
            ext = ".png" if args.format == "png" else ".jpg"
            output_path = output_path / f"{input_path.stem}{ext}"

        decoder.decode_meta_to_image(str(input_path), str(output_path), args.quality, args.format)

    elif input_path.is_dir():
        # Folder mode
        decoder.decode_folder(str(input_path), str(output_path), args.quality, args.format)

    else:
        logger.error(f"Invalid input: {input_path} (must be .meta file or directory)")


if __name__ == "__main__":
    main()


# Example usage:
"""
# Decode single .meta file to PNG
python decode_image.py --input processed_meta/video1_1.meta --output decoded_images/frame1.png

# Decode single .meta file to JPEG with custom quality
python decode_image.py --input processed_meta/video1_1.meta --output decoded_images/frame1.jpg --format jpg --quality 90

# Decode all .meta files in a folder to PNG
python decode_image.py --input processed_meta/ --output decoded_images/

# Decode all .meta files in a folder to JPEG
python decode_image.py --input processed_meta/ --output decoded_images/ --format jpg --quality 95

# Use Wan2.1 1.3B model
python decode_image.py --input processed_meta/ --output decoded_images/ --model Wan-AI/Wan2.1-T2V-1.3B-Diffusers

# Disable memory optimization if you have enough VRAM
python decode_image.py --input processed_meta/ --output decoded_images/ --no-memory-optimization

# Programmatic usage
from decode_image import ImageDecoder

decoder = ImageDecoder("Wan-AI/Wan2.1-T2V-14B-Diffusers")

# Single file
decoder.decode_meta_to_image("processed_meta/video1_1.meta", "output/frame1.png")

# Entire folder
decoder.decode_folder("processed_meta", "decoded_images", quality=95, format="png")
"""
