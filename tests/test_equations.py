import numpy as np

from src.equations import dot_coordinates_text, line_equation, stroke_equations_text


def test_line_equation_includes_parametric_and_cartesian():
    text = line_equation(np.array([0, 1]), np.array([2, 5]))
    assert "x(t) = 0 + 2 t" in text
    assert "y(t) = 1 + 4 t" in text
    assert "y = 2x + 1" in text


def test_vertical_line_equation():
    text = line_equation(np.array([3, 1]), np.array([3, 5]))
    assert "cartesian: x = 3" in text


def test_stroke_equations_text_counts_segments():
    strokes = [np.array([[0, 0], [1, 1], [2, 1]], dtype=float)]
    text = stroke_equations_text(strokes, (10, 20))
    assert "number of strokes: 1" in text
    assert "segment 1" in text
    assert "segment 2" in text
    assert "total line segments: 2" in text


def test_dot_coordinates_text():
    points = np.array([[1, 2], [3.5, 4]], dtype=float)
    text = dot_coordinates_text(points, (10, 20))
    assert "number of dots: 2" in text
    assert "dot 1: x = 1, y = 2" in text
