import cv2
import numpy as np
from typing import Optional, Dict, Any, List
from pathlib import Path

from .grayscale import to_grayscale
from .denoise import denoise
from .deskew import deskew
from .normalize import normalize
from .resize import resize


class PreprocessingPipeline:
    """
    Configurable image preprocessing pipeline for document OCR training.

    Chains multiple preprocessing steps with configurable parameters.
    Each step can be enabled/disabled independently.
    """

    def __init__(
        self,
        # Step enable/disable flags
        enable_grayscale: bool = True,
        enable_denoise: bool = True,
        enable_deskew: bool = True,
        enable_resize: bool = True,
        enable_normalize: bool = True,
        # Grayscale params
        grayscale_copy: bool = False,
        # Denoise params
        denoise_method: str = "bilateral",  # Faster than nlm for training
        denoise_strength: float = 10.0,
        # Deskew params
        deskew_threshold: int = 0,
        deskew_return_angle: bool = False,
        # Resize params
        resize_width: Optional[int] = None,
        resize_height: Optional[int] = None,
        resize_mode: str = "fit",
        resize_interpolation: str = "auto",
        # Normalize params
        normalize_method: str = "zero_one",
        normalize_mean: Optional[tuple] = None,
        normalize_std: Optional[tuple] = None,
    ):
        """
        Initialize preprocessing pipeline with step configurations.

        Args:
            enable_grayscale: Convert to grayscale
            enable_denoise: Apply denoising
            enable_deskew: Correct skew
            enable_resize: Resize image
            enable_normalize: Normalize pixel values
            grayscale_copy: Copy in grayscale step
            denoise_method: Denoising algorithm
            denoise_strength: Denoising strength
            deskew_threshold: Threshold for deskew detection
            deskew_return_angle: Return detected angle
            resize_width: Target width
            resize_height: Target height
            resize_mode: Resize mode (fit/fill/stretch)
            resize_interpolation: Interpolation method
            normalize_method: Normalization method
            normalize_mean: Mean for mean_std normalization
            normalize_std: Std for mean_std normalization
        """
        # Store configuration
        self.config = {
            "enable_grayscale": enable_grayscale,
            "enable_denoise": enable_denoise,
            "enable_deskew": enable_deskew,
            "enable_resize": enable_resize,
            "enable_normalize": enable_normalize,
            "grayscale_copy": grayscale_copy,
            "denoise_method": denoise_method,
            "denoise_strength": denoise_strength,
            "deskew_threshold": deskew_threshold,
            "deskew_return_angle": deskew_return_angle,
            "resize_width": resize_width,
            "resize_height": resize_height,
            "resize_mode": resize_mode,
            "resize_interpolation": resize_interpolation,
            "normalize_method": normalize_method,
            "normalize_mean": normalize_mean,
            "normalize_std": normalize_std,
        }

    def __call__(self, img: np.ndarray) -> Dict[str, Any]:
        """
        Process a single image through the pipeline.

        Args:
            img: Input image as numpy array

        Returns:
            Dictionary containing:
                - 'image': Preprocessed image
                - 'metadata': Processing metadata (original_shape, steps_applied, etc.)
        """
        if img is None:
            raise ValueError("Input image is None")

        if not isinstance(img, np.ndarray):
            raise TypeError(f"Expected numpy array, got {type(img)}")

        # Initialize metadata
        metadata = {
            "original_shape": img.shape,
            "steps_applied": [],
        }

        processed = img

        # Step 1: Grayscale
        if self.config["enable_grayscale"]:
            processed = to_grayscale(
                processed,
                copy=self.config["grayscale_copy"]
            )
            metadata["steps_applied"].append("grayscale")

        # Step 2: Denoise
        if self.config["enable_denoise"]:
            processed = denoise(
                processed,
                method=self.config["denoise_method"],
                strength=self.config["denoise_strength"],
                copy=False,  # Already copied in grayscale if needed
            )
            metadata["steps_applied"].append("denoise")

        # Step 3: Deskew
        if self.config["enable_deskew"]:
            if self.config["deskew_return_angle"]:
                processed, angle = deskew(
                    processed,
                    threshold=self.config["deskew_threshold"],
                    copy=False,
                    return_angle=True,
                )
                metadata["deskew_angle"] = angle
            else:
                processed = deskew(
                    processed,
                    threshold=self.config["deskew_threshold"],
                    copy=False,
                    return_angle=False,
                )
            metadata["steps_applied"].append("deskew")

        # Step 4: Resize
        if self.config["enable_resize"]:
            if self.config["resize_width"] or self.config["resize_height"]:
                processed = resize(
                    processed,
                    width=self.config["resize_width"],
                    height=self.config["resize_height"],
                    mode=self.config["resize_mode"],
                    interpolation=self.config["resize_interpolation"],
                    copy=False,
                )
                metadata["steps_applied"].append("resize")

        # Step 5: Normalize
        if self.config["enable_normalize"]:
            processed = normalize(
                processed,
                method=self.config["normalize_method"],
                mean=self.config["normalize_mean"],
                std=self.config["normalize_std"],
                copy=False,
            )
            metadata["steps_applied"].append("normalize")

        metadata["final_shape"] = processed.shape

        return {
            "image": processed,
            "metadata": metadata,
        }

    def process_image_path(self, image_path: str) -> Dict[str, Any]:
        """
        Load and process an image from file path.

        Args:
            image_path: Path to image file

        Returns:
            Dictionary with preprocessed image and metadata

        Raises:
            FileNotFoundError: If image file doesn't exist
            ValueError: If image cannot be loaded
        """
        path = Path(image_path)

        if not path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        # Load image
        img = cv2.imread(str(path))

        if img is None:
            raise ValueError(f"Failed to load image: {image_path}")

        # Process through pipeline
        result = self(img)
        result["metadata"]["source_path"] = str(path)

        return result

    def process_batch(
        self,
        images: List[np.ndarray]
    ) -> List[Dict[str, Any]]:
        """
        Process a batch of images through the pipeline.

        Args:
            images: List of images as numpy arrays

        Returns:
            List of dictionaries with preprocessed images and metadata
        """
        results = []

        for idx, img in enumerate(images):
            try:
                result = self(img)
                result["metadata"]["batch_index"] = idx
                results.append(result)
            except Exception as e:
                # Log error but continue processing
                print(f"Warning: Failed to process image {idx}: {e}")
                results.append({
                    "image": None,
                    "metadata": {
                        "batch_index": idx,
                        "error": str(e),
                    }
                })

        return results

    def get_config(self) -> Dict[str, Any]:
        """Get current pipeline configuration."""
        return self.config.copy()


# Convenience function for simple preprocessing
def preprocess_image(
    img: np.ndarray,
    target_size: Optional[tuple] = None,
    normalize_method: str = "zero_one",
) -> np.ndarray:
    """
    Simple preprocessing function for quick use.

    Args:
        img: Input image
        target_size: (width, height) tuple for resizing. None to skip resize.
        normalize_method: Normalization method

    Returns:
        Preprocessed image
    """
    pipeline = PreprocessingPipeline(
        enable_grayscale=True,
        enable_denoise=True,
        enable_deskew=True,
        enable_resize=target_size is not None,
        enable_normalize=True,
        resize_width=target_size[0] if target_size else None,
        resize_height=target_size[1] if target_size else None,
        normalize_method=normalize_method,
    )

    result = pipeline(img)
    return result["image"]
