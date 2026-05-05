"""Text exports for the exact drawing primitives."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

Array = np.ndarray


def _num(value: float) -> str:
    """Format a float so the export stays readable but precise enough."""
    text = f"{float(value):.4f}"
    text = text.rstrip("0").rstrip(".")
    return text if text not in {"-0", ""} else "0"


def line_equation(p0: Array, p1: Array) -> str:
    """Return parametric and cartesian equations for one drawn segment."""
    x0, y0 = map(float, p0)
    x1, y1 = map(float, p1)
    dx = x1 - x0
    dy = y1 - y0

    parametric = (
        f"x(t) = {_num(x0)} + {_num(dx)} t, "
        f"y(t) = {_num(y0)} + {_num(dy)} t, 0 <= t <= 1"
    )

    if abs(dx) < 1e-12:
        cartesian = f"x = {_num(x0)}"
    else:
        slope = dy / dx
        intercept = y0 - slope * x0
        cartesian = f"y = {_num(slope)}x + {_num(intercept)}"

    endpoints = f"({_num(x0)}, {_num(y0)}) -> ({_num(x1)}, {_num(y1)})"
    return f"endpoints: {endpoints}\nparametric: {parametric}\ncartesian: {cartesian}"


def stroke_equations_text(
    strokes: Sequence[Array],
    image_shape: tuple[int, int],
    *,
    max_segments: int | None = None,
) -> str:
    """Build a text file containing equations for all Brownian stroke segments."""
    height, width = image_shape
    lines: list[str] = [
        "brownian brush line equations",
        "",
        f"image width: {width}",
        f"image height: {height}",
        f"number of strokes: {len(strokes)}",
        "coordinate system: origin at top left, x goes right, y goes down",
        "note: each stroke is drawn as straight segments between consecutive sampled points",
        "",
    ]

    written = 0
    for stroke_idx, stroke in enumerate(strokes, start=1):
        path = np.asarray(stroke, dtype=np.float64)
        if path.ndim != 2 or path.shape[1] != 2:
            raise ValueError("each stroke must have shape (n, 2)")
        if len(path) < 2:
            continue

        lines.append(f"stroke {stroke_idx}")
        for segment_idx in range(len(path) - 1):
            if max_segments is not None and written >= max_segments:
                lines.append("")
                lines.append(f"stopped after {max_segments} segments")
                lines.append("download the full file for every line equation")
                return "\n".join(lines)

            lines.append(f"  segment {segment_idx + 1}")
            equation = line_equation(path[segment_idx], path[segment_idx + 1])
            for equation_line in equation.splitlines():
                lines.append(f"    {equation_line}")
            written += 1
        lines.append("")

    lines.append(f"total line segments: {written}")
    return "\n".join(lines)


def dot_coordinates_text(points: Array, image_shape: tuple[int, int], *, max_points: int | None = None) -> str:
    """Build a text file containing dot coordinates for non-line modes."""
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError("points must have shape (n, 2)")
    height, width = image_shape
    limit = len(pts) if max_points is None else min(max_points, len(pts))
    lines = [
        "brownian brush dot coordinates",
        "",
        f"image width: {width}",
        f"image height: {height}",
        f"number of dots: {len(pts)}",
        "coordinate system: origin at top left, x goes right, y goes down",
        "",
    ]
    for idx, (x, y) in enumerate(pts[:limit], start=1):
        lines.append(f"dot {idx}: x = {_num(x)}, y = {_num(y)}")
    if max_points is not None and len(pts) > max_points:
        lines.append("")
        lines.append(f"stopped after {max_points} dots")
        lines.append("download the full file for every dot coordinate")
    return "\n".join(lines)
