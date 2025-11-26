import cv2
import numpy as np
from typing import Tuple


def deskew(
    img: np.ndarray, threshold: int = 0, copy: bool = True, return_angle: bool = False
) -> np.ndarray | Tuple[np.ndarray, float]:
    """
    Correct image skew by detecting and rotating to align text horizontally.

    Uses minimum area rectangle fitting on foreground pixels to detect skew angle.
    Assumes dark text on light background for binary/grayscale images.

    Args:
        img: Input image as numpy array (grayscale or color)
        threshold: Pixel intensity threshold for foreground detection. Default 0.
        copy: If True, avoid mutating the original array. Default True.
        return_angle: If True, return tuple of (deskewed_image, angle). Default False.

    Returns:
        Deskewed image, or tuple of (deskewed_image, detected_angle_degrees)

    Raises:
        ValueError: If image is None, empty, or invalid
        TypeError: If input is not a numpy array
        RuntimeError: If deskew operation fails
    """

    if img is None:
        raise ValueError("Input image is None. Cannot deskew.")

    if not isinstance(img, np.ndarray):
        raise TypeError(f"Expected numpy array, got {type(img)}")

    if img.size == 0:
        raise ValueError("Input image is empty.")

    # Copy input if requested to avoid mutation
    if copy:
        img = img.copy()

    # Convert to grayscale if color image
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img

    # Find coordinates of foreground pixels
    coords = np.column_stack(np.where(gray > threshold))

    # Handle case where no foreground pixels found
    if coords.size == 0:
        if return_angle:
            return img, 0.0
        return img

    try:
        # Detect skew angle using minimum area rectangle
        angle = cv2.minAreaRect(coords)[-1]

        # Correct angle to [-45, 45] range
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle

    except cv2.error as e:
        raise RuntimeError(f"Angle detection failed: {e}")

    # Skip rotation if angle is negligible (< 0.5 degrees)
    if abs(angle) < 0.5:
        if return_angle:
            return img, angle
        return img

    # Get image dimensions
    (h, w) = img.shape[:2]

    # Compute rotation matrix
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)

    try:
        # Apply affine transformation
        deskewed = cv2.warpAffine(
            img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
        )
    except cv2.error as e:
        raise RuntimeError(f"Deskew rotation failed: {e}")

    if return_angle:
        return deskewed, angle

    return deskewed
