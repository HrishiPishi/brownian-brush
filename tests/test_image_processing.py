import numpy as np
from PIL import Image

from src.image_processing import compute_darkness, compute_edges, normalize_image, resize_image, to_grayscale


def test_grayscale_output_shape_and_range():
    image = Image.new("RGB", (12, 7), (30, 120, 210))
    gray = to_grayscale(image)
    assert gray.shape == (7, 12)
    assert np.all(gray >= 0)
    assert np.all(gray <= 1)


def test_normalized_image_values_are_between_zero_and_one():
    arr = np.array([[0, 128, 255]], dtype=np.uint8)
    normalized = normalize_image(arr)
    assert normalized.min() >= 0
    assert normalized.max() <= 1
    assert np.isclose(normalized[0, 0], 0)
    assert np.isclose(normalized[0, 2], 1)


def test_darkness_map_values_are_between_zero_and_one():
    gray = np.array([[0.0, 0.25, 1.0]])
    darkness = compute_darkness(gray)
    assert darkness.shape == gray.shape
    assert np.all(darkness >= 0)
    assert np.all(darkness <= 1)
    assert np.isclose(darkness[0, 0], 1)
    assert np.isclose(darkness[0, 2], 0)


def test_resize_preserves_aspect_ratio():
    image = Image.new("RGB", (1000, 500))
    resized = resize_image(image, max_size=250)
    assert resized.size == (250, 125)


def test_edges_have_correct_shape_and_range():
    gray = np.zeros((10, 10), dtype=float)
    gray[:, 5:] = 1.0
    edges = compute_edges(gray)
    assert edges.shape == gray.shape
    assert edges.min() >= 0
    assert edges.max() <= 1
    assert edges.max() > 0
