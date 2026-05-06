import numpy as np
from PIL import Image

from src.rendering import render_dots, render_strokes, side_by_side, to_png_bytes


def test_render_dots_has_expected_dimensions():
    points = np.array([[1, 1], [5, 4], [9, 2]], dtype=float)
    image = render_dots(points, (8, 10), dot_size=2)
    assert isinstance(image, Image.Image)
    assert image.size == (10, 8)


def test_empty_point_list_does_not_crash():
    image = render_dots(np.empty((0, 2)), (4, 6))
    assert image.size == (6, 4)


def test_large_number_of_points_does_not_crash():
    rng = np.random.default_rng(1)
    points = np.column_stack((rng.integers(0, 50, 5000), rng.integers(0, 40, 5000)))
    image = render_dots(points, (40, 50), dot_size=1)
    assert image.size == (50, 40)


def test_render_strokes_has_expected_dimensions():
    strokes = [np.array([[0, 0], [1, 1], [2, 1]], dtype=float)]
    image = render_strokes(strokes, (8, 10), stroke_width=1)
    assert image.size == (10, 8)


def test_side_by_side_and_png_export():
    left = Image.new("RGB", (10, 8), "white")
    right = Image.new("RGB", (12, 8), "white")
    combined = side_by_side(left, right, gutter=3)
    assert combined.size == (25, 8)
    data = to_png_bytes(combined)
    assert data.startswith(b"\x89PNG")


def test_render_dots_can_use_per_point_colors():
    points = np.array([[1, 1], [3, 1]], dtype=float)
    colors = np.array([[255, 0, 0], [0, 0, 255]], dtype=np.uint8)
    image = render_dots(points, (5, 5), dot_size=2, opacity=1.0, point_colors=colors)
    assert image.getpixel((1, 1))[0] > 200
    assert image.getpixel((3, 1))[2] > 200


def test_render_strokes_can_use_per_stroke_colors():
    strokes = [
        np.array([[0, 0], [0, 3]], dtype=float),
        np.array([[4, 0], [4, 3]], dtype=float),
    ]
    colors = [(255, 0, 0), (0, 128, 255)]
    image = render_strokes(strokes, (5, 5), stroke_width=1, opacity=1.0, stroke_colors=colors)
    assert image.getpixel((0, 1))[0] > 200
    assert image.getpixel((4, 1))[2] > 200
