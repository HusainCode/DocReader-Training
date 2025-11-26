import cv2
import numpy as np

def to_grayscale(img: np.ndarray, copy: bool = True) -> np.ndarray:
    """
    Convert BGR image to grayscale with safe error handling.

    Args:
        img: Input image as numpy array in BGR format (from cv2.imread)
        copy: If True, avoid mutating the original array. Default True.

    Returns:
        Grayscale image as numpy array (single channel)

    Raises:
        ValueError: If image is None, empty, or invalid shape
        TypeError: If input is not a numpy array
        RuntimeError: If OpenCV conversion fails
    """

    if img is None:
        raise ValueError("Input image is None. Cannot convert to grayscale.")

    # Error: image is not a valid array
    if not isinstance(img, np.ndarray):
        raise TypeError(f"Expected numpy array, got {type(img)}")

    if img.size == 0:
        raise ValueError("Input image is empty.")

    # Check if already grayscale
    if len(img.shape) == 2:
        return img.copy() if copy else img

    # Check for valid BGR image (3 channels)
    if len(img.shape) != 3 or img.shape[2] != 3:
        raise ValueError(f"Expected BGR image with 3 channels, got shape {img.shape}")

    # Copy input if requested to avoid mutation
    if copy:
        img = img.copy()

    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    except cv2.error as e:
        raise RuntimeError(f"OpenCV grayscale conversion failed: {e}")

    return gray