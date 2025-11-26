import cv2
import numpy as np
from typing import Literal


def denoise(
    img: np.ndarray,
    method: Literal["nlm", "bilateral", "gaussian", "median"] = "nlm",
    strength: float = 10.0,
    copy: bool = True,
) -> np.ndarray:
    """
    Remove noise from image using various denoising algorithms.

    Args:
        img: Input image as numpy array (grayscale or BGR color)
        method: Denoising method to use:
            - 'nlm': Non-Local Means (best quality, slowest)
            - 'bilateral': Bilateral filter (good edge preservation, moderate speed)
            - 'gaussian': Gaussian blur (fast, simple smoothing)
            - 'median': Median filter (good for salt-and-pepper noise, fast)
        strength: Denoising strength parameter. Higher = more smoothing.
            - nlm: filter strength (default 10, typical range 3-15)
            - bilateral: sigma color (default 10, typical range 5-50)
            - gaussian: kernel size will be computed from this (default 10)
            - median: kernel size (default 10, must be odd, will be rounded)
        copy: If True, avoid mutating the original array. Default True.

    Returns:
        Denoised image as numpy array

    Raises:
        ValueError: If image is None, empty, invalid, or method unknown
        TypeError: If input is not a numpy array
        RuntimeError: If denoising operation fails

    Notes:
        - Non-Local Means is recommended for document images but is slowest
        - Bilateral filter offers good balance of quality and speed
        - Gaussian and Median are fastest but less sophisticated
    """

    if img is None:
        raise ValueError("Input image is None. Cannot denoise.")

    if not isinstance(img, np.ndarray):
        raise TypeError(f"Expected numpy array, got {type(img)}")

    if img.size == 0:
        raise ValueError("Input image is empty.")

    if strength <= 0:
        raise ValueError(f"Strength must be positive, got {strength}")

    # Copy input if requested to avoid mutation
    if copy:
        img = img.copy()

    # Detect if color or grayscale
    is_color = len(img.shape) == 3 and img.shape[2] == 3

    try:
        if method == "nlm":
            # Non-Local Means Denoising
            if is_color:
                # For color images: (src, dst, h, hColor, templateWindowSize, searchWindowSize)
                denoised = cv2.fastNlMeansDenoisingColored(
                    img, None, h=strength, hColor=strength, templateWindowSize=7, searchWindowSize=21
                )
            else:
                # For grayscale: (src, dst, h, templateWindowSize, searchWindowSize)
                denoised = cv2.fastNlMeansDenoising(
                    img, None, h=strength, templateWindowSize=7, searchWindowSize=21
                )

        elif method == "bilateral":
            # Bilateral filter: preserves edges while smoothing
            # Parameters: (src, d, sigmaColor, sigmaSpace)
            d = 9  # Diameter of pixel neighborhood
            sigma_space = 75  # Spatial sigma
            denoised = cv2.bilateralFilter(img, d=d, sigmaColor=strength, sigmaSpace=sigma_space)

        elif method == "gaussian":
            # Gaussian blur: simple smoothing
            # Compute kernel size from strength (must be odd)
            ksize = int(strength) if int(strength) % 2 == 1 else int(strength) + 1
            ksize = max(3, ksize)  # Minimum kernel size of 3
            denoised = cv2.GaussianBlur(img, (ksize, ksize), sigmaX=0)

        elif method == "median":
            # Median filter: good for salt-and-pepper noise
            # Kernel size must be odd
            ksize = int(strength) if int(strength) % 2 == 1 else int(strength) + 1
            ksize = max(3, ksize)  # Minimum kernel size of 3
            denoised = cv2.medianBlur(img, ksize)

        else:
            raise ValueError(
                f"Unknown denoising method '{method}'. "
                f"Choose from: 'nlm', 'bilateral', 'gaussian', 'median'"
            )

    except cv2.error as e:
        raise RuntimeError(f"Denoising with method '{method}' failed: {e}")

    return denoised
