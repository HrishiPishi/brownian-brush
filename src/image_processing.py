"""Image loading and preprocessing for Brownian Brush."""

from __future__ import annotations

from pathlib import Path
from typing import BinaryIO, Tuple

import numpy as np
from PIL import Image, ImageOps


Array = np.ndarray


def load_image(uploaded_file: str | Path | BinaryIO) -> Image.Image:
    """Load an uploaded image or path as an RGB PIL image.

    Supports PNG, JPG, and JPEG inputs. EXIF orientation is respected so phone
    photos do not appear rotated incorrectly.
    """
    try:
        image = Image.open(uploaded_file)
        image = ImageOps.exif_transpose(image)
        return image.convert("RGB")
    except Exception as exc:  # pragma: no cover - exact PIL failures vary
        raise ValueError("Could not load image. Please use a valid PNG, JPG, or JPEG file.") from exc


def resize_image(image: Image.Image, max_size: int = 512) -> Image.Image:
    """Resize an image while preserving aspect ratio.

    The longest side becomes at most ``max_size``. Images already smaller than
    ``max_size`` are copied rather than enlarged.
    """
    if max_size <= 0:
        raise ValueError("max_size must be positive")
    width, height = image.size
    longest = max(width, height)
    if longest <= max_size:
        return image.copy()
    scale = max_size / float(longest)
    new_size = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
    return image.resize(new_size, Image.Resampling.LANCZOS)


def normalize_image(gray: Array) -> Array:
    """Normalize an image-like array into float values in [0, 1]."""
    arr = np.asarray(gray, dtype=np.float64)
    if arr.size == 0:
        raise ValueError("image array cannot be empty")
    finite = np.isfinite(arr)
    if not finite.all():
        raise ValueError("image array contains nan or infinite values")
    if arr.min() >= 0 and arr.max() <= 1:
        return np.clip(arr, 0.0, 1.0)
    if arr.max() == arr.min():
        return np.zeros_like(arr, dtype=np.float64)
    if arr.min() >= 0 and arr.max() <= 255:
        return np.clip(arr / 255.0, 0.0, 1.0)
    return np.clip((arr - arr.min()) / (arr.max() - arr.min()), 0.0, 1.0)


def to_grayscale(image: Image.Image | Array) -> Array:
    """Convert an RGB image to a grayscale float array in [0, 1].

    Uses luminance weights rather than a simple channel average.
    """
    if isinstance(image, Image.Image):
        arr = np.asarray(image.convert("RGB"), dtype=np.float64) / 255.0
    else:
        arr = normalize_image(np.asarray(image, dtype=np.float64))
        if arr.ndim == 2:
            return arr
    if arr.ndim == 2:
        return normalize_image(arr)
    if arr.ndim != 3 or arr.shape[2] < 3:
        raise ValueError("expected a grayscale array or RGB image")
    gray = 0.2126 * arr[..., 0] + 0.7152 * arr[..., 1] + 0.0722 * arr[..., 2]
    return np.clip(gray, 0.0, 1.0)


def compute_darkness(gray: Array, invert: bool = False) -> Array:
    """Return a darkness map in [0, 1].

    With the default setting, black pixels have value 1 and white pixels have
    value 0. If ``invert`` is true, white pixels are treated as important.
    """
    normalized = normalize_image(gray)
    darkness = normalized if invert else 1.0 - normalized
    return np.clip(darkness, 0.0, 1.0)


def compute_edges(gray: Array) -> Array:
    """Compute a simple normalized finite-difference edge-strength map."""
    normalized = normalize_image(gray)
    if normalized.ndim != 2:
        raise ValueError("edge detection expects a 2d grayscale array")
    gy, gx = np.gradient(normalized)
    magnitude = np.sqrt(gx * gx + gy * gy)
    max_value = float(magnitude.max())
    if max_value <= 0:
        return np.zeros_like(normalized, dtype=np.float64)
    return np.clip(magnitude / max_value, 0.0, 1.0)


def validate_image_array(arr: Array) -> Tuple[int, int]:
    """Validate a 2D image array and return ``(height, width)``."""
    data = np.asarray(arr)
    if data.ndim != 2:
        raise ValueError("expected a 2d image array")
    height, width = data.shape
    if height <= 0 or width <= 0:
        raise ValueError("image dimensions must be positive")
    return height, width
