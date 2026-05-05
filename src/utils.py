"""Small utilities for Brownian Brush."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class RenderMetadata:
    mode: str
    samples: int
    gamma: float
    seed: int
    width: int
    height: int

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)


def set_seed(seed: int | None = None) -> np.random.Generator:
    """Return a NumPy random generator for reproducible sampling."""
    return np.random.default_rng(seed)


def validate_positive_int(value: int, name: str) -> int:
    if int(value) <= 0:
        raise ValueError(f"{name} must be positive")
    return int(value)


def save_output_image(image: Image.Image, path: str | Path = "brownian_brush_output.png") -> Path:
    """Save a final render to disk and return its path."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path, format="PNG")
    return output_path
