"""Rendering helpers for Brownian Brush."""

from __future__ import annotations

from io import BytesIO
from typing import Iterable, Sequence

import numpy as np
from PIL import Image, ImageColor, ImageDraw

Array = np.ndarray


def parse_color(color: str | tuple[int, int, int]) -> tuple[int, int, int]:
    """Parse a user color into an RGB tuple."""
    if isinstance(color, tuple):
        if len(color) != 3:
            raise ValueError("RGB color tuples must have length 3")
        return tuple(int(np.clip(c, 0, 255)) for c in color)
    try:
        return ImageColor.getrgb(color)
    except Exception as exc:  # pragma: no cover - PIL error text varies
        raise ValueError(f"invalid color: {color}") from exc


def _base(image_shape: tuple[int, int], background_color: str | tuple[int, int, int]) -> Image.Image:
    height, width = image_shape
    if height <= 0 or width <= 0:
        raise ValueError("image_shape must be positive")
    return Image.new("RGBA", (width, height), parse_color(background_color) + (255,))


def render_dots(
    points: Array,
    image_shape: tuple[int, int],
    dot_size: float = 1.5,
    opacity: float = 0.85,
    dot_color: str | tuple[int, int, int] = "black",
    background_color: str | tuple[int, int, int] = "white",
) -> Image.Image:
    """Render a list of sampled points as semi-transparent dots."""
    if dot_size <= 0:
        raise ValueError("dot_size must be positive")
    alpha = int(np.clip(opacity, 0.0, 1.0) * 255)
    image = _base(image_shape, background_color)
    draw = ImageDraw.Draw(image, "RGBA")
    color = parse_color(dot_color) + (alpha,)
    pts = np.asarray(points, dtype=np.float64)
    if pts.size == 0:
        return image.convert("RGB")
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError("points must have shape (n, 2)")
    radius = dot_size / 2.0
    height, width = image_shape
    for x, y in pts:
        if x < 0 or y < 0 or x >= width or y >= height:
            continue
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color)
    return image.convert("RGB")


def render_strokes(
    strokes: Sequence[Array],
    image_shape: tuple[int, int],
    stroke_width: int = 1,
    opacity: float = 0.55,
    stroke_color: str | tuple[int, int, int] = "black",
    background_color: str | tuple[int, int, int] = "white",
) -> Image.Image:
    """Render Brownian paths as semi-transparent connected line segments."""
    if stroke_width <= 0:
        raise ValueError("stroke_width must be positive")
    alpha = int(np.clip(opacity, 0.0, 1.0) * 255)
    image = _base(image_shape, background_color)
    draw = ImageDraw.Draw(image, "RGBA")
    color = parse_color(stroke_color) + (alpha,)
    height, width = image_shape
    for stroke in strokes:
        path = np.asarray(stroke, dtype=np.float64)
        if path.size == 0:
            continue
        if path.ndim != 2 or path.shape[1] != 2:
            raise ValueError("each stroke must have shape (n, 2)")
        clipped = np.column_stack((np.clip(path[:, 0], 0, width - 1), np.clip(path[:, 1], 0, height - 1)))
        xy = [tuple(map(float, pair)) for pair in clipped]
        if len(xy) == 1:
            x, y = xy[0]
            r = max(0.5, stroke_width / 2)
            draw.ellipse((x - r, y - r, x + r, y + r), fill=color)
        else:
            draw.line(xy, fill=color, width=stroke_width, joint="curve")
    return image.convert("RGB")


def side_by_side(left: Image.Image, right: Image.Image, gutter: int = 16, background_color: str = "white") -> Image.Image:
    """Place two images next to each other on one canvas."""
    if gutter < 0:
        raise ValueError("gutter must be nonnegative")
    h = max(left.height, right.height)
    w = left.width + right.width + gutter
    canvas = Image.new("RGB", (w, h), parse_color(background_color))
    canvas.paste(left.convert("RGB"), (0, (h - left.height) // 2))
    canvas.paste(right.convert("RGB"), (left.width + gutter, (h - right.height) // 2))
    return canvas


def to_png_bytes(image: Image.Image) -> bytes:
    """Encode a PIL image as PNG bytes."""
    buf = BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()
