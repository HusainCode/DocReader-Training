import cv2
import numpy as np
import time
import hashlib
import pickle
from typing import Optional, Dict, Any, List
from pathlib import Path

from .grayscale import to_grayscale
from .denoise import denoise
from .deskew import deskew
from .normalize import normalize
from .resize import resize
from ..utils.logger import get_logger

# Setup logger for preprocessing
logger = get_logger(__name__)


class PreprocessingPipeline:
    """
    Configurable image preprocessing pipeline for document OCR training.

    Chains multiple preprocessing steps with configurable parameters.
    Each step can be enabled/disabled independently.

    Features for training:
    - Caching: Cache preprocessed images to avoid reprocessing
    - Profiling: Track timing for each preprocessing step
    - Config tracking: Save/load pipeline configuration
    - Logging: Debug data quality issues
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
        # Training-specific params
        cache_dir: Optional[str] = None,
        enable_profiling: bool = False,
        enable_logging: bool = True,
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
            cache_dir: Directory to cache preprocessed images. None to disable caching.
            enable_profiling: Track timing for each step
            enable_logging: Enable debug logging
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

        # Training-specific settings
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.enable_profiling = enable_profiling
        self.enable_logging = enable_logging

        # Create cache directory if needed
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            if self.enable_logging:
                logger.info(f"Caching enabled at: {self.cache_dir}")

        # Profiling stats
        self.profiling_stats = {
            "grayscale": [],
            "denoise": [],
            "deskew": [],
            "resize": [],
            "normalize": [],
            "total": [],
        }

    def _get_cache_key(self, img: np.ndarray) -> str:
        """Generate cache key from image and config."""
        # Hash image content + config
        img_hash = hashlib.md5(img.tobytes()).hexdigest()
        config_str = str(sorted(self.config.items()))
        config_hash = hashlib.md5(config_str.encode()).hexdigest()
        return f"{img_hash}_{config_hash}"

    def _load_from_cache(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Load preprocessed image from cache."""
        if not self.cache_dir:
            return None

        cache_file = self.cache_dir / f"{cache_key}.pkl"
        if cache_file.exists():
            try:
                with open(cache_file, "rb") as f:
                    return pickle.load(f)
            except Exception as e:
                if self.enable_logging:
                    logger.warning(f"Failed to load cache: {e}")
        return None

    def _save_to_cache(self, cache_key: str, result: Dict[str, Any]):
        """Save preprocessed image to cache."""
        if not self.cache_dir:
            return

        cache_file = self.cache_dir / f"{cache_key}.pkl"
        try:
            with open(cache_file, "wb") as f:
                pickle.dump(result, f)
        except Exception as e:
            if self.enable_logging:
                logger.warning(f"Failed to save cache: {e}")

    def __call__(
        self, img: np.ndarray, cache_key: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Process a single image through the pipeline.

        Args:
            img: Input image as numpy array
            cache_key: Optional cache key. If None, auto-generated from image.

        Returns:
            Dictionary containing:
                - 'image': Preprocessed image
                - 'metadata': Processing metadata (original_shape, steps_applied, etc.)
        """
        start_time = time.time()

        if img is None:
            raise ValueError("Input image is None")

        if not isinstance(img, np.ndarray):
            raise TypeError(f"Expected numpy array, got {type(img)}")

        # Check cache
        if cache_key is None and self.cache_dir:
            cache_key = self._get_cache_key(img)

        if cache_key:
            cached_result = self._load_from_cache(cache_key)
            if cached_result is not None:
                if self.enable_logging:
                    logger.debug(f"Cache hit for key: {cache_key}")
                return cached_result

        # Initialize metadata
        metadata = {
            "original_shape": img.shape,
            "steps_applied": [],
            "step_timings": {} if self.enable_profiling else None,
        }

        processed = img

        # Step 1: Grayscale
        if self.config["enable_grayscale"]:
            step_start = time.time()
            processed = to_grayscale(processed, copy=self.config["grayscale_copy"])
            if self.enable_profiling:
                elapsed = time.time() - step_start
                metadata["step_timings"]["grayscale"] = elapsed
                self.profiling_stats["grayscale"].append(elapsed)
            metadata["steps_applied"].append("grayscale")

        # Step 2: Denoise
        if self.config["enable_denoise"]:
            step_start = time.time()
            processed = denoise(
                processed,
                method=self.config["denoise_method"],
                strength=self.config["denoise_strength"],
                copy=False,
            )
            if self.enable_profiling:
                elapsed = time.time() - step_start
                metadata["step_timings"]["denoise"] = elapsed
                self.profiling_stats["denoise"].append(elapsed)
            metadata["steps_applied"].append("denoise")

        # Step 3: Deskew
        if self.config["enable_deskew"]:
            step_start = time.time()
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
            if self.enable_profiling:
                elapsed = time.time() - step_start
                metadata["step_timings"]["deskew"] = elapsed
                self.profiling_stats["deskew"].append(elapsed)
            metadata["steps_applied"].append("deskew")

        # Step 4: Resize
        if self.config["enable_resize"]:
            if self.config["resize_width"] or self.config["resize_height"]:
                step_start = time.time()
                processed = resize(
                    processed,
                    width=self.config["resize_width"],
                    height=self.config["resize_height"],
                    mode=self.config["resize_mode"],
                    interpolation=self.config["resize_interpolation"],
                    copy=False,
                )
                if self.enable_profiling:
                    elapsed = time.time() - step_start
                    metadata["step_timings"]["resize"] = elapsed
                    self.profiling_stats["resize"].append(elapsed)
                metadata["steps_applied"].append("resize")

        # Step 5: Normalize
        if self.config["enable_normalize"]:
            step_start = time.time()
            processed = normalize(
                processed,
                method=self.config["normalize_method"],
                mean=self.config["normalize_mean"],
                std=self.config["normalize_std"],
                copy=False,
            )
            if self.enable_profiling:
                elapsed = time.time() - step_start
                metadata["step_timings"]["normalize"] = elapsed
                self.profiling_stats["normalize"].append(elapsed)
            metadata["steps_applied"].append("normalize")

        metadata["final_shape"] = processed.shape

        # Total time
        total_time = time.time() - start_time
        if self.enable_profiling:
            metadata["total_time"] = total_time
            self.profiling_stats["total"].append(total_time)

        # Data quality checks (logging)
        if self.enable_logging:
            # Check for anomalies
            if processed.size == 0:
                logger.warning("Preprocessed image is empty!")
            if np.all(processed == 0):
                logger.warning("Preprocessed image is all black!")
            if np.all(processed == processed.flat[0]):
                logger.warning("Preprocessed image is constant!")

        result = {
            "image": processed,
            "metadata": metadata,
        }

        # Save to cache
        if cache_key:
            self._save_to_cache(cache_key, result)

        return result

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

    def process_batch(self, images: List[np.ndarray]) -> List[Dict[str, Any]]:
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
                results.append(
                    {
                        "image": None,
                        "metadata": {
                            "batch_index": idx,
                            "error": str(e),
                        },
                    }
                )

        return results

    def get_config(self) -> Dict[str, Any]:
        """Get current pipeline configuration."""
        return self.config.copy()

    def save_config(self, path: str):
        """
        Save pipeline configuration to file for reproducibility.

        Args:
            path: Path to save config (JSON or YAML format based on extension)
        """
        import json

        path = Path(path)

        with open(path, "w") as f:
            json.dump(self.config, f, indent=2)

        if self.enable_logging:
            logger.info(f"Pipeline config saved to: {path}")

    @classmethod
    def load_config(cls, path: str) -> "PreprocessingPipeline":
        """
        Load pipeline from saved configuration.

        Args:
            path: Path to config file

        Returns:
            PreprocessingPipeline instance with loaded config
        """
        import json

        path = Path(path)

        with open(path, "r") as f:
            config = json.load(f)

        return cls(**config)

    def get_profiling_summary(self) -> Dict[str, Dict[str, float]]:
        """
        Get profiling statistics summary.

        Returns:
            Dictionary with mean, min, max times for each step
        """
        summary = {}

        for step, times in self.profiling_stats.items():
            if times:
                summary[step] = {
                    "mean": np.mean(times),
                    "min": np.min(times),
                    "max": np.max(times),
                    "std": np.std(times),
                    "total": np.sum(times),
                    "count": len(times),
                }

        return summary

    def print_profiling_summary(self):
        """Print profiling summary to console."""
        summary = self.get_profiling_summary()

        print("\n" + "=" * 60)
        print("PREPROCESSING PROFILING SUMMARY")
        print("=" * 60)

        for step, stats in summary.items():
            if stats["count"] > 0:
                print(f"\n{step.upper()}:")
                print(f"  Count:    {stats['count']}")
                print(f"  Mean:     {stats['mean'] * 1000:.2f} ms")
                print(f"  Std:      {stats['std'] * 1000:.2f} ms")
                print(f"  Min:      {stats['min'] * 1000:.2f} ms")
                print(f"  Max:      {stats['max'] * 1000:.2f} ms")
                print(f"  Total:    {stats['total']:.2f} s")

        print("=" * 60 + "\n")

    def clear_cache(self):
        """Clear all cached preprocessed images."""
        if self.cache_dir and self.cache_dir.exists():
            import shutil

            shutil.rmtree(self.cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            if self.enable_logging:
                logger.info("Cache cleared")


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
