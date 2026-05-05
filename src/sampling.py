"""Sampling methods for stochastic image reconstruction."""

from __future__ import annotations

from typing import Iterable, List

import numpy as np

Array = np.ndarray


def _rng(seed: int | None = None) -> np.random.Generator:
    return np.random.default_rng(seed)


def make_probability_map(density: Array, gamma: float = 1.0, epsilon: float = 1e-12) -> Array:
    """Convert a nonnegative density map into a valid probability map.

    ``gamma`` sharpens or softens the distribution. Values above 1 concentrate
    samples in darker or more important regions. Values below 1 spread samples
    more evenly.
    """
    if gamma <= 0:
        raise ValueError("gamma must be positive")
    arr = np.asarray(density, dtype=np.float64)
    if arr.ndim != 2 or arr.size == 0:
        raise ValueError("density must be a nonempty 2d array")
    if not np.isfinite(arr).all():
        raise ValueError("density contains nan or infinite values")
    weighted = np.power(np.clip(arr, 0.0, None), gamma)
    total = float(weighted.sum())
    if total <= epsilon:
        return np.full(arr.shape, 1.0 / arr.size, dtype=np.float64)
    return weighted / total


def sample_points(probability_map: Array, num_points: int, seed: int | None = None) -> Array:
    """Sample ``(x, y)`` points from a 2D probability map."""
    if num_points < 0:
        raise ValueError("num_points must be nonnegative")
    prob = np.asarray(probability_map, dtype=np.float64)
    if prob.ndim != 2 or prob.size == 0:
        raise ValueError("probability_map must be a nonempty 2d array")
    total = float(prob.sum())
    if not np.isfinite(prob).all() or total <= 0:
        raise ValueError("probability_map must have positive finite mass")
    prob = (prob / total).ravel()
    height, width = probability_map.shape
    indices = _rng(seed).choice(height * width, size=num_points, replace=True, p=prob)
    ys, xs = np.divmod(indices, width)
    return np.column_stack((xs, ys)).astype(np.float64)


def sample_edge_weighted_points(
    darkness: Array,
    edges: Array,
    num_points: int,
    edge_weight: float = 1.0,
    gamma: float = 1.0,
    seed: int | None = None,
    darkness_weight: float = 1.0,
) -> Array:
    """Sample from a combined darkness/edge density."""
    if edge_weight < 0 or darkness_weight < 0:
        raise ValueError("weights must be nonnegative")
    dark = np.asarray(darkness, dtype=np.float64)
    edge = np.asarray(edges, dtype=np.float64)
    if dark.shape != edge.shape:
        raise ValueError("darkness and edges must have the same shape")
    density = darkness_weight * np.clip(dark, 0.0, 1.0) + edge_weight * np.clip(edge, 0.0, 1.0)
    probability_map = make_probability_map(density, gamma=gamma)
    return sample_points(probability_map, num_points=num_points, seed=seed)


def generate_brownian_strokes(
    targets: Array,
    image_shape: tuple[int, int],
    stroke_length: int = 20,
    jitter: float = 2.0,
    seed: int | None = None,
    drift_strength: float = 0.12,
) -> List[Array]:
    """Generate short Brownian-style random walks around target points.

    The walk has a small attraction back to its sampled target point. That keeps
    strokes local enough to reveal the source image while still looking random.
    """
    if stroke_length <= 0:
        raise ValueError("stroke_length must be positive")
    if jitter < 0:
        raise ValueError("jitter must be nonnegative")
    if drift_strength < 0:
        raise ValueError("drift_strength must be nonnegative")
    height, width = image_shape
    if height <= 0 or width <= 0:
        raise ValueError("image_shape must be positive")
    pts = np.asarray(targets, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError("targets must have shape (n, 2)")
    rng = _rng(seed)
    strokes: List[Array] = []
    for target in pts:
        current = target + rng.normal(0.0, jitter * 0.5, size=2)
        current[0] = np.clip(current[0], 0, width - 1)
        current[1] = np.clip(current[1], 0, height - 1)
        path = [current.copy()]
        for _ in range(stroke_length - 1):
            pull = drift_strength * (target - current)
            noise = rng.normal(0.0, jitter, size=2)
            current = current + pull + noise
            current[0] = np.clip(current[0], 0, width - 1)
            current[1] = np.clip(current[1], 0, height - 1)
            path.append(current.copy())
        strokes.append(np.asarray(path, dtype=np.float64))
    return strokes
