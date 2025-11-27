import cv2
import numpy as np
from typing import Literal, Optional, Tuple


def normalize(
    img: np.ndarray,
    method: Literal["zero_one", "neg_one_one", "mean_std", "min_max"] = "zero_one",
    mean: Optional[Tuple[float, float, float]] = None,
    std: Optional[Tuple[float, float, float]] = None,
    min_val: float = 0.0,
    max_val: float = 1.0,
    copy: bool = True,
) -> np.ndarray:
    """
    Normalize image pixel values using various normalization methods.

    Args:
        img: Input image as numpy array (grayscale or BGR color)
        method: Normalization method:
            - 'zero_one': Scale to [0, 1] range (default)
            - 'neg_one_one': Scale to [-1, 1] range (common for GANs)
            - 'mean_std': Standardize using mean and std (z-score normalization)
            - 'min_max': Scale to custom [min_val, max_val] range
        mean: Mean values for 'mean_std' method.
            - For grayscale: (mean,) or scalar
            - For color: (mean_b, mean_g, mean_r) for BGR
            - Default: None (computed from image)
            - ImageNet: (0.485, 0.456, 0.406) for RGB
        std: Std values for 'mean_std' method.
            - For grayscale: (std,) or scalar
            - For color: (std_b, std_g, std_r) for BGR
            - Default: None (computed from image)
            - ImageNet: (0.229, 0.224, 0.225) for RGB
        min_val: Minimum value for 'min_max' method. Default 0.0.
        max_val: Maximum value for 'min_max' method. Default 1.0.
        copy: If True, avoid mutating the original array. Default True.

    Returns:
        Normalized image as float32 numpy array

    Raises:
        ValueError: If image is None, empty, or invalid parameters
        TypeError: If input is not a numpy array

    Examples:
        >>> # Standard [0, 1] normalization
        >>> normalized = normalize(img)

        >>> # [-1, 1] normalization for GANs
        >>> normalized = normalize(img, method='neg_one_one')

        >>> # ImageNet normalization (after converting BGR to RGB)
        >>> normalized = normalize(img, method='mean_std',
        ...                       mean=(0.485, 0.456, 0.406),
        ...                       std=(0.229, 0.224, 0.225))
    """

    if img is None:
        raise ValueError("Input image is None. Cannot normalize.")

    if not isinstance(img, np.ndarray):
        raise TypeError(f"Expected numpy array, got {type(img)}")

    if img.size == 0:
        raise ValueError("Input image is empty.")

    # Copy and convert to float32
    if copy:
        img = img.astype(np.float32)
    else:
        # In-place conversion if not copying
        if img.dtype != np.float32:
            img = img.astype(np.float32)

    # Apply normalization based on method
    if method == "zero_one":
        # Scale to [0, 1]
        if img.max() > 1.0:  # Only scale if not already normalized
            img /= 255.0

    elif method == "neg_one_one":
        # Scale to [-1, 1]
        if img.max() > 1.0:  # Assume [0, 255] range
            img = (img / 127.5) - 1.0
        else:  # Assume [0, 1] range
            img = (img * 2.0) - 1.0

    elif method == "mean_std":
        # Z-score normalization: (x - mean) / std
        # First scale to [0, 1] if needed
        if img.max() > 1.0:
            img /= 255.0

        if mean is None or std is None:
            # Compute mean and std from image
            if mean is None:
                mean = np.mean(img, axis=(0, 1)) if len(img.shape) == 3 else np.mean(img)
            if std is None:
                std = np.std(img, axis=(0, 1)) if len(img.shape) == 3 else np.std(img)
                # Avoid division by zero
                std = np.maximum(std, 1e-8)

        # Convert to arrays for broadcasting
        if isinstance(mean, (int, float)):
            mean = np.array([mean])
        else:
            mean = np.array(mean)

        if isinstance(std, (int, float)):
            std = np.array([std])
        else:
            std = np.array(std)

        # Validate shapes for color images
        if len(img.shape) == 3:
            if len(mean) != img.shape[2]:
                raise ValueError(
                    f"Mean length {len(mean)} doesn't match channels {img.shape[2]}"
                )
            if len(std) != img.shape[2]:
                raise ValueError(
                    f"Std length {len(std)} doesn't match channels {img.shape[2]}"
                )

        # Apply standardization
        img = (img - mean) / std

    elif method == "min_max":
        # Scale to custom [min_val, max_val] range
        if min_val >= max_val:
            raise ValueError(f"min_val ({min_val}) must be less than max_val ({max_val})")

        img_min = img.min()
        img_max = img.max()

        # Avoid division by zero
        if img_max - img_min < 1e-8:
            # Image is constant, set to min_val
            img = np.full_like(img, min_val)
        else:
            # Scale to [0, 1] then to [min_val, max_val]
            img = (img - img_min) / (img_max - img_min)
            img = img * (max_val - min_val) + min_val

    else:
        raise ValueError(
            f"Unknown normalization method '{method}'. "
            f"Choose from: 'zero_one', 'neg_one_one', 'mean_std', 'min_max'"
        )

    return img
