import cv2
import numpy as np
from typing import Literal, Optional, Tuple


def resize(
    img: np.ndarray,
    width: Optional[int] = None,
    height: Optional[int] = None,
    mode: Literal["fit", "fill", "stretch"] = "fit",
    interpolation: Literal["auto", "nearest", "linear", "cubic", "area", "lanczos"] = "auto",
    pad_color: Tuple[int, int, int] = (0, 0, 0),
    copy: bool = True,
) -> np.ndarray:
    """
    Resize image with multiple modes and smart interpolation selection.

    Args:
        img: Input image as numpy array (grayscale or BGR color)
        width: Target width in pixels. If None, computed from height.
        height: Target height in pixels. If None, computed from width.
        mode: Resize mode:
            - 'fit': Maintain aspect ratio, scale to fit within target (default)
            - 'fill': Maintain aspect ratio, scale to fill target, then center crop
            - 'stretch': Ignore aspect ratio, stretch to exact dimensions
        interpolation: Interpolation method:
            - 'auto': Smart selection (area for downscale, cubic for upscale)
            - 'nearest': Fastest, lowest quality (cv2.INTER_NEAREST)
            - 'linear': Fast, good quality (cv2.INTER_LINEAR)
            - 'cubic': Slower, better quality (cv2.INTER_CUBIC)
            - 'area': Best for downscaling (cv2.INTER_AREA)
            - 'lanczos': Highest quality, slowest (cv2.INTER_LANCZOS4)
        pad_color: RGB/BGR color for padding in 'fit' mode. Default (0,0,0) black.
        copy: If True, avoid mutating the original array. Default True.

    Returns:
        Resized image as numpy array

    Raises:
        ValueError: If image is None, empty, invalid dimensions, or both width/height None
        TypeError: If input is not a numpy array
        RuntimeError: If resize operation fails

    Examples:
        >>> # Resize to 224x224, maintaining aspect ratio with padding
        >>> resized = resize(img, width=224, height=224, mode='fit')

        >>> # Resize to max width 800, keeping aspect ratio
        >>> resized = resize(img, width=800, mode='fit')

        >>> # Stretch to exact dimensions
        >>> resized = resize(img, width=640, height=480, mode='stretch')
    """

    if img is None:
        raise ValueError("Input image is None. Cannot resize.")

    if not isinstance(img, np.ndarray):
        raise TypeError(f"Expected numpy array, got {type(img)}")

    if img.size == 0:
        raise ValueError("Input image is empty.")

    if width is None and height is None:
        raise ValueError("Must provide at least one of width or height for resizing.")

    if width is not None and width <= 0:
        raise ValueError(f"Width must be positive, got {width}")

    if height is not None and height <= 0:
        raise ValueError(f"Height must be positive, got {height}")

    # Copy input if requested to avoid mutation
    if copy:
        img = img.copy()

    # Get original dimensions
    h, w = img.shape[:2]

    # Determine target dimensions based on mode
    if mode == "stretch":
        # Stretch: use exact dimensions (both must be provided)
        if width is None or height is None:
            raise ValueError("Both width and height required for 'stretch' mode")
        target_w, target_h = width, height

    elif mode in ["fit", "fill"]:
        # Fit/Fill: maintain aspect ratio
        if width is not None and height is not None:
            # Both dimensions provided: compute scale to fit/fill
            scale_w = width / float(w)
            scale_h = height / float(h)

            if mode == "fit":
                # Fit: scale to fit inside (use smaller scale)
                scale = min(scale_w, scale_h)
            else:  # fill
                # Fill: scale to cover (use larger scale)
                scale = max(scale_w, scale_h)

            target_w = int(w * scale)
            target_h = int(h * scale)

        elif width is not None:
            # Only width provided: compute height from aspect ratio
            scale = width / float(w)
            target_w = width
            target_h = int(h * scale)

        else:  # height is not None
            # Only height provided: compute width from aspect ratio
            scale = height / float(h)
            target_w = int(w * scale)
            target_h = height

    else:
        raise ValueError(f"Unknown resize mode '{mode}'. Choose from: 'fit', 'fill', 'stretch'")

    # Select interpolation method
    interp_map = {
        "nearest": cv2.INTER_NEAREST,
        "linear": cv2.INTER_LINEAR,
        "cubic": cv2.INTER_CUBIC,
        "area": cv2.INTER_AREA,
        "lanczos": cv2.INTER_LANCZOS4,
    }

    if interpolation == "auto":
        # Smart selection: area for downscale, cubic for upscale
        if target_w < w or target_h < h:
            interp_method = cv2.INTER_AREA
        else:
            interp_method = cv2.INTER_CUBIC
    else:
        interp_method = interp_map.get(interpolation)
        if interp_method is None:
            raise ValueError(
                f"Unknown interpolation '{interpolation}'. "
                f"Choose from: 'auto', 'nearest', 'linear', 'cubic', 'area', 'lanczos'"
            )

    try:
        # Perform resize
        resized = cv2.resize(img, (target_w, target_h), interpolation=interp_method)
    except cv2.error as e:
        raise RuntimeError(f"Resize operation failed: {e}")

    # Handle padding/cropping for fit/fill modes
    if mode == "fit" and width is not None and height is not None:
        # Pad to exact dimensions if needed
        if target_w < width or target_h < height:
            # Create canvas with pad_color
            if len(img.shape) == 2:
                # Grayscale: use first value of pad_color
                canvas = np.full((height, width), pad_color[0], dtype=img.dtype)
            else:
                # Color: use full pad_color
                canvas = np.full((height, width, img.shape[2]), pad_color, dtype=img.dtype)

            # Center the resized image on canvas
            y_offset = (height - target_h) // 2
            x_offset = (width - target_w) // 2
            canvas[y_offset:y_offset + target_h, x_offset:x_offset + target_w] = resized
            resized = canvas

    elif mode == "fill" and width is not None and height is not None:
        # Crop to exact dimensions if needed
        if target_w > width or target_h > height:
            # Center crop
            y_offset = (target_h - height) // 2
            x_offset = (target_w - width) // 2
            resized = resized[y_offset:y_offset + height, x_offset:x_offset + width]

    return resized
