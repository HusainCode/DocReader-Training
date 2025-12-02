"""
Production-ready OCR dataset loader for PyTorch training.

Designed for large-scale document recognition with defensive programming,
error handling, and performance optimization patterns used at Meta/Facebook.
"""

import warnings
from pathlib import Path
from typing import Optional, Callable, Dict, Any, List, Tuple, Literal

import cv2
import numpy as np
from torch.utils.data import Dataset

from ..utils.logger import get_logger

logger = get_logger(__name__)


class OCRDataset(Dataset):
    """
    Production-grade PyTorch dataset for OCR/document image recognition.

    Features:
    - Automatic project root detection (works in WSL, Docker, notebooks)
    - Robust error handling with fallback mechanisms
    - Image-annotation pair validation
    - Corrupted image detection and graceful degradation
    - Memory-efficient lazy loading
    - Preprocessing pipeline integration
    - Comprehensive logging for debugging
    - Train/val/test split support

    Architecture:
    - Uses pathlib for cross-platform compatibility
    - Validates all files on initialization (fail-fast principle)
    - Lazy loads images to minimize memory footprint
    - Supports custom transforms (TorchVision or custom preprocessing)

    Example:
        >>> from torchvision import transforms
        >>> transform = transforms.Compose([
        ...     transforms.ToTensor(),
        ...     transforms.Normalize(mean=[0.5], std=[0.5])
        ... ])
        >>> train_dataset = OCRDataset(
        ...     split="train",
        ...     transform=transform,
        ...     validate_on_init=True
        ... )
        >>> img, label = train_dataset[0]
    """

    # Supported image formats
    VALID_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}

    # Supported annotation formats
    VALID_ANNOTATION_EXTENSIONS = {".txt", ".json"}

    def __init__(
        self,
        split: Literal["train", "val", "test", "training", "validation", "testing"] = "train",
        data_root: Optional[Path] = None,
        transform: Optional[Callable] = None,
        annotation_format: Literal["txt", "json"] = "txt",
        validate_on_init: bool = True,
        skip_corrupted: bool = False,
        max_samples: Optional[int] = None,
    ):
        """
        Initialize OCR dataset with robust validation and error handling.

        Args:
            split: Dataset split to load ("train", "val", "test" or legacy names)
            data_root: Root directory containing data/dataset (default: auto-detect)
            transform: Optional transform/preprocessing pipeline
            annotation_format: Format of annotation files ("txt" or "json")
            validate_on_init: Whether to validate all images on initialization
                             (slower startup, catches errors early)
            skip_corrupted: If True, skip corrupted images instead of raising error
                           (useful for dirty datasets)
            max_samples: Maximum number of samples to load (for debugging/testing)

        Raises:
            ValueError: If split is invalid or data_root doesn't exist
            RuntimeError: If no images found or critical validation fails
            FileNotFoundError: If required directories are missing
        """
        # Input validation
        if not isinstance(split, str):
            raise TypeError(f"split must be a string, got {type(split)}")

        # Normalize split names (support both "train" and "training")
        split_mapping = {
            "train": "training",
            "training": "training",
            "val": "validation",
            "validation": "validation",
            "test": "testing",
            "testing": "testing",
        }

        normalized_split = split_mapping.get(split.lower())
        if normalized_split is None:
            raise ValueError(
                f"Invalid split: {split}. "
                f"Must be one of: {list(split_mapping.keys())}"
            )

        self.split = normalized_split
        self.transform = transform
        self.annotation_format = annotation_format
        self.skip_corrupted = skip_corrupted

        # Auto-detect project root if not provided
        if data_root is None:
            # Navigate up from src/train/dataset.py to project root
            data_root = Path(__file__).resolve().parents[3]

        self.data_root = Path(data_root)

        if not self.data_root.exists():
            raise FileNotFoundError(
                f"Data root directory does not exist: {self.data_root}"
            )

        # Construct dataset paths
        dataset_root = self.data_root / "data" / "dataset"

        # Support both formats: training_data and train
        split_dir_candidates = [
            dataset_root / f"{self.split}_data",  # training_data, validation_data
            dataset_root / self.split,  # training, validation
        ]

        split_root = None
        for candidate in split_dir_candidates:
            if candidate.exists():
                split_root = candidate
                break

        if split_root is None:
            raise FileNotFoundError(
                f"Split directory not found. Tried: {split_dir_candidates}"
            )

        self.split_root = split_root
        self.img_dir = split_root / "images"
        self.ann_dir = split_root / "annotations"

        # Validate required directories
        if not self.img_dir.exists():
            raise FileNotFoundError(
                f"Image directory does not exist: {self.img_dir}"
            )

        if not self.ann_dir.exists():
            logger.warning(
                f"Annotation directory not found: {self.ann_dir}. "
                f"Dataset will return dummy labels."
            )
            self.has_annotations = False
        else:
            self.has_annotations = True

        # Discover all image files
        self.image_paths = self._discover_images()

        if len(self.image_paths) == 0:
            raise RuntimeError(
                f"No valid images found in {self.img_dir}. "
                f"Supported formats: {self.VALID_IMAGE_EXTENSIONS}"
            )

        # Apply max_samples limit if specified (for debugging)
        if max_samples is not None and max_samples > 0:
            self.image_paths = self.image_paths[:max_samples]
            logger.info(f"Limited dataset to {max_samples} samples for debugging")

        # Build annotation paths
        self.annotation_paths = self._build_annotation_paths()

        # Validate image-annotation pairs
        self._validate_pairs()

        # Optional: Validate all images can be loaded (expensive but catches errors early)
        if validate_on_init:
            self._validate_all_images()

        logger.info(
            f"OCRDataset initialized: split={self.split}, "
            f"samples={len(self.image_paths)}, "
            f"root={self.split_root}"
        )

    def _discover_images(self) -> List[Path]:
        """
        Discover all valid image files in the image directory.

        Returns:
            Sorted list of image file paths

        Note:
            Sorts paths to ensure deterministic ordering across runs
        """
        image_paths = []

        for ext in self.VALID_IMAGE_EXTENSIONS:
            # Use glob for efficient file discovery
            image_paths.extend(self.img_dir.glob(f"*{ext}"))
            image_paths.extend(self.img_dir.glob(f"*{ext.upper()}"))

        # Sort for deterministic ordering
        image_paths = sorted(image_paths)

        logger.debug(f"Discovered {len(image_paths)} images in {self.img_dir}")

        return image_paths

    def _build_annotation_paths(self) -> List[Optional[Path]]:
        """
        Build corresponding annotation paths for each image.

        Returns:
            List of annotation paths (None if annotation doesn't exist)
        """
        annotation_paths = []

        for img_path in self.image_paths:
            if not self.has_annotations:
                annotation_paths.append(None)
                continue

            # Annotation filename matches image stem (e.g., image001.txt)
            ann_filename = f"{img_path.stem}.{self.annotation_format}"
            ann_path = self.ann_dir / ann_filename

            annotation_paths.append(ann_path)

        return annotation_paths

    def _validate_pairs(self) -> None:
        """
        Validate that each image has a corresponding annotation.

        Raises:
            FileNotFoundError: If any annotation is missing (when has_annotations=True)
        """
        if not self.has_annotations:
            logger.info("Skipping annotation validation (no annotation directory)")
            return

        missing_annotations = []

        for img_path, ann_path in zip(self.image_paths, self.annotation_paths):
            if ann_path is None or not ann_path.exists():
                missing_annotations.append((img_path.name, ann_path))

        if missing_annotations:
            error_msg = (
                f"Found {len(missing_annotations)} images without annotations:\n"
            )
            for img_name, ann_path in missing_annotations[:5]:  # Show first 5
                error_msg += f"  - {img_name} -> {ann_path}\n"

            if len(missing_annotations) > 5:
                error_msg += f"  ... and {len(missing_annotations) - 5} more\n"

            raise FileNotFoundError(error_msg)

        logger.debug(f"Validated {len(self.image_paths)} image-annotation pairs")

    def _validate_all_images(self) -> None:
        """
        Validate that all images can be loaded without corruption.

        This is an expensive operation but catches errors early during initialization
        rather than during training (fail-fast principle).

        Raises:
            RuntimeError: If any image is corrupted (when skip_corrupted=False)
        """
        logger.info("Validating all images (this may take a while)...")

        corrupted_images = []

        for idx, img_path in enumerate(self.image_paths):
            try:
                img = cv2.imread(str(img_path))

                if img is None:
                    corrupted_images.append(img_path)
                    continue

                # Additional validation: check if image has valid shape
                if img.ndim != 3 or img.shape[2] != 3:
                    logger.warning(
                        f"Image has unexpected shape: {img_path} -> {img.shape}"
                    )

            except Exception as e:
                logger.error(f"Failed to load image {img_path}: {e}")
                corrupted_images.append(img_path)

            # Progress logging for large datasets
            if (idx + 1) % 1000 == 0:
                logger.debug(f"Validated {idx + 1}/{len(self.image_paths)} images")

        if corrupted_images:
            error_msg = (
                f"Found {len(corrupted_images)} corrupted images:\n"
                + "\n".join(f"  - {p}" for p in corrupted_images[:10])
            )

            if len(corrupted_images) > 10:
                error_msg += f"\n  ... and {len(corrupted_images) - 10} more"

            if self.skip_corrupted:
                logger.warning(error_msg)
                # Remove corrupted images from dataset
                self.image_paths = [
                    p for p in self.image_paths if p not in corrupted_images
                ]
                self.annotation_paths = [
                    self.annotation_paths[i]
                    for i, p in enumerate(self.image_paths)
                    if p not in corrupted_images
                ]
                logger.info(
                    f"Removed {len(corrupted_images)} corrupted images. "
                    f"Remaining: {len(self.image_paths)}"
                )
            else:
                raise RuntimeError(error_msg)

        logger.info(f"✓ All {len(self.image_paths)} images validated successfully")

    def __len__(self) -> int:
        """Return the total number of samples in the dataset."""
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, str]:
        """
        Load and return a single sample from the dataset.

        Args:
            idx: Index of the sample to load

        Returns:
            Tuple of (image, label):
                - image: RGB image as numpy array or transformed tensor
                - label: Text label (string)

        Raises:
            IndexError: If idx is out of range
            RuntimeError: If image loading fails

        Note:
            Images are loaded in BGR by OpenCV and converted to RGB for PyTorch.
        """
        if idx < 0 or idx >= len(self.image_paths):
            raise IndexError(
                f"Index {idx} out of range for dataset of size {len(self)}"
            )

        img_path = self.image_paths[idx]
        ann_path = self.annotation_paths[idx]

        # Load image (OpenCV loads as BGR)
        img = cv2.imread(str(img_path))

        if img is None:
            error_msg = (
                f"Failed to load image: {img_path}\n"
                f"This might indicate:\n"
                f"  - Corrupted file\n"
                f"  - Unsupported format\n"
                f"  - Insufficient permissions\n"
                f"  - File moved/deleted during training"
            )

            if self.skip_corrupted:
                logger.warning(error_msg)
                # Return a dummy sample (all zeros)
                img = np.zeros((224, 224, 3), dtype=np.uint8)
            else:
                raise RuntimeError(error_msg)

        # Convert BGR -> RGB (PyTorch/TorchVision convention)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Load annotation
        if ann_path is None or not ann_path.exists():
            # Dummy label if no annotation
            label = ""
        else:
            try:
                with open(ann_path, "r", encoding="utf-8") as f:
                    if self.annotation_format == "txt":
                        label = f.read().strip()
                    elif self.annotation_format == "json":
                        import json
                        label = json.load(f).get("text", "")
            except Exception as e:
                logger.error(f"Failed to load annotation {ann_path}: {e}")
                label = ""

        # Apply transform/preprocessing if provided
        if self.transform:
            try:
                img = self.transform(img)
            except Exception as e:
                logger.error(
                    f"Transform failed for {img_path}: {e}. "
                    f"Returning untransformed image."
                )
                warnings.warn(
                    f"Transform failed for index {idx}. Check your transform pipeline."
                )

        return img, label

    def get_sample_info(self, idx: int) -> Dict[str, Any]:
        """
        Get metadata about a sample without loading the image.

        Args:
            idx: Sample index

        Returns:
            Dictionary with sample metadata:
                - image_path: Path to image file
                - annotation_path: Path to annotation file
                - image_name: Image filename
                - image_size_bytes: File size in bytes
                - exists: Whether files exist

        Example:
            >>> info = dataset.get_sample_info(0)
            >>> print(info['image_name'])
        """
        img_path = self.image_paths[idx]
        ann_path = self.annotation_paths[idx]

        return {
            "image_path": str(img_path),
            "annotation_path": str(ann_path) if ann_path else None,
            "image_name": img_path.name,
            "image_size_bytes": img_path.stat().st_size if img_path.exists() else 0,
            "exists": img_path.exists(),
        }

    def __repr__(self) -> str:
        """Return string representation of the dataset."""
        return (
            f"OCRDataset(\n"
            f"  split={self.split},\n"
            f"  samples={len(self)},\n"
            f"  root={self.split_root},\n"
            f"  has_annotations={self.has_annotations},\n"
            f"  transform={self.transform is not None}\n"
            f")"
        )
