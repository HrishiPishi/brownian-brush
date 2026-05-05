import numpy as np

from src.sampling import generate_brownian_strokes, make_probability_map, sample_edge_weighted_points, sample_points


def test_probability_map_sums_to_one():
    density = np.array([[0.0, 1.0], [2.0, 3.0]])
    prob = make_probability_map(density, gamma=1.4)
    assert prob.shape == density.shape
    assert np.isclose(prob.sum(), 1.0)
    assert np.all(prob >= 0)


def test_zero_density_becomes_uniform():
    prob = make_probability_map(np.zeros((4, 5)))
    assert np.isclose(prob.sum(), 1.0)
    assert np.allclose(prob, 1 / 20)


def test_sampled_points_have_valid_coordinates():
    prob = make_probability_map(np.ones((6, 8)))
    points = sample_points(prob, 1000, seed=1)
    assert points.shape == (1000, 2)
    assert np.all(points[:, 0] >= 0)
    assert np.all(points[:, 0] < 8)
    assert np.all(points[:, 1] >= 0)
    assert np.all(points[:, 1] < 6)


def test_same_seed_produces_same_points():
    prob = make_probability_map(np.arange(1, 10).reshape(3, 3))
    a = sample_points(prob, 50, seed=44)
    b = sample_points(prob, 50, seed=44)
    assert np.array_equal(a, b)


def test_darker_region_gets_sampled_more_often():
    density = np.ones((10, 10)) * 0.05
    density[:, :5] = 10.0
    prob = make_probability_map(density)
    points = sample_points(prob, 5000, seed=10)
    left_count = np.sum(points[:, 0] < 5)
    right_count = np.sum(points[:, 0] >= 5)
    assert left_count > right_count * 10


def test_edge_weighted_points_are_valid():
    darkness = np.ones((8, 9)) * 0.2
    edges = np.zeros((8, 9))
    edges[3, :] = 1
    points = sample_edge_weighted_points(darkness, edges, 200, edge_weight=3, seed=2)
    assert points.shape == (200, 2)
    assert np.all(points[:, 0] >= 0)
    assert np.all(points[:, 0] < 9)
    assert np.all(points[:, 1] >= 0)
    assert np.all(points[:, 1] < 8)


def test_brownian_strokes_stay_inside_image():
    targets = np.array([[5, 5], [9, 0], [0, 9]], dtype=float)
    strokes = generate_brownian_strokes(targets, (10, 10), stroke_length=12, jitter=3, seed=4)
    assert len(strokes) == 3
    for stroke in strokes:
        assert stroke.shape == (12, 2)
        assert np.all(stroke[:, 0] >= 0)
        assert np.all(stroke[:, 0] < 10)
        assert np.all(stroke[:, 1] >= 0)
        assert np.all(stroke[:, 1] < 10)
