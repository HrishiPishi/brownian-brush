"""Progressive frame helpers."""

from __future__ import annotations

from typing import Iterator, Sequence

import numpy as np

Array = np.ndarray


def _validate_batch_size(batch_size: int) -> None:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")


def progressive_dot_frames(points: Array, batch_size: int) -> Iterator[Array]:
    """Yield cumulative dot batches for animation."""
    _validate_batch_size(batch_size)
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError("points must have shape (n, 2)")
    for end in range(batch_size, len(pts) + batch_size, batch_size):
        yield pts[: min(end, len(pts))]


def progressive_stroke_frames(strokes: Sequence[Array], batch_size: int) -> Iterator[list[Array]]:
    """Yield cumulative stroke batches for animation."""
    _validate_batch_size(batch_size)
    for end in range(batch_size, len(strokes) + batch_size, batch_size):
        yield list(strokes[: min(end, len(strokes))])
