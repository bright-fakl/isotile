"""EdgeShape constraints, enforce_symmetry (index-based projection), is_symmetric and interpolate."""

import numpy as np
import pytest

from isotile import EdgeShape, enforce_symmetry, interpolate, is_symmetric

SHAPES = ("I", "J", "U", "S")


def random_points(rng: np.random.Generator, m: int, turn_back: bool = True) -> np.ndarray:
    """m control points; x is not monotonic when ``turn_back``."""
    x = rng.uniform(-0.3, 1.3, m) if turn_back else np.sort(rng.uniform(0, 1, m))
    return np.column_stack([x, rng.uniform(-0.5, 0.5, m)])


def mirror(p: np.ndarray, shape: str) -> np.ndarray:
    """The image of the points under the symmetry, in reversed order."""
    q = p[::-1].copy()
    q[:, 0] = 1.0 - q[:, 0]
    if shape == "S":
        q[:, 1] = -q[:, 1]
    return q


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("m", [2, 3, 4, 5, 8, 11, 40])
def test_idempotent_and_symmetric(shape, m):
    rng = np.random.default_rng(m)
    for _ in range(50):
        p = random_points(rng, m)
        once = enforce_symmetry(p, shape)
        twice = enforce_symmetry(once, shape)
        np.testing.assert_array_equal(once, twice)  # exactly, not approximately
        assert is_symmetric(once, shape, tol=0.0)
        if shape in ("U", "S"):
            np.testing.assert_allclose(once, mirror(once, shape), atol=1e-15)


@pytest.mark.parametrize("shape", ("U", "S"))
def test_first_half_is_kept_by_index(shape):
    rng = np.random.default_rng(3)
    for m in (4, 5, 9):
        p = random_points(rng, m)  # x is not sorted: pairing is by index, not by position
        out = enforce_symmetry(p, shape)
        np.testing.assert_array_equal(out[: m // 2], p[: m // 2])


def test_middle_point_is_moved_onto_the_axis():
    p = np.array([[0, 0], [0.3, 0.2], [0.6, 0.4], [0.7, 0.2], [1, 0]], dtype=float)
    np.testing.assert_array_equal(enforce_symmetry(p, "U")[2], [0.5, 0.4])
    np.testing.assert_array_equal(enforce_symmetry(p, "S")[2], [0.5, 0.0])


def test_known_values():
    p = [[0, 0], [0.2, 0.1], [0.4, 0.3], [0.9, 0.9], [1, 0]]
    np.testing.assert_allclose(enforce_symmetry(p, "U"), [[0, 0], [0.2, 0.1], [0.5, 0.3], [0.8, 0.1], [1, 0]])
    np.testing.assert_allclose(enforce_symmetry(p, "S"), [[0, 0], [0.2, 0.1], [0.5, 0.0], [0.8, -0.1], [1, 0]])
    np.testing.assert_array_equal(enforce_symmetry(p, "J"), p)
    np.testing.assert_array_equal(enforce_symmetry(p, "I"), [[0, 0], [1, 0]])


@pytest.mark.parametrize("shape", ("U", "S"))
def test_symmetric_input_is_unchanged(shape):
    """A curve that already has the symmetry of its shape is returned as it is, whatever its x order."""
    rng = np.random.default_rng(7)
    for m in (3, 6, 7):
        half = random_points(rng, m // 2)
        half[0] = [0.0, 0.0]
        mid = np.array([[0.5, 0.0]]) if m % 2 else np.empty((0, 2))
        p = np.vstack([half, mid, mirror(half, shape)])
        np.testing.assert_array_equal(enforce_symmetry(p, shape), p)
        assert is_symmetric(p, shape)


def test_is_symmetric_detects_violations():
    p = np.array([[0, 0], [0.25, 0.1], [0.5, 0], [0.75, 0.1], [1, 0]])
    assert is_symmetric(p, "U")
    assert not is_symmetric(p, "S")
    assert is_symmetric(p, "J")
    assert not is_symmetric(p, "I")
    assert is_symmetric([[0, 0], [1, 0]], "I")
    assert is_symmetric(p + 1e-12, "U")
    assert not is_symmetric(p + 1e-12, "U", tol=0.0)


@pytest.mark.parametrize(
    "bad", [[[0, 0]], [[0, 0, 0], [1, 0, 0]], [0, 1], [[0, 0], [np.nan, 0]], [[0, 0], [np.inf, 0]]]
)
def test_invalid_points_are_rejected(bad):
    with pytest.raises(ValueError, match="edge curve"):
        enforce_symmetry(bad, "J")
    with pytest.raises(ValueError, match="edge curve"):
        interpolate(bad)


def test_unknown_shape_and_mode():
    with pytest.raises(ValueError, match="shape"):
        enforce_symmetry([[0, 0], [1, 0]], "X")
    with pytest.raises(ValueError, match="shape"):
        is_symmetric([[0, 0], [1, 0]], "X")
    with pytest.raises(ValueError, match="mode"):
        interpolate([[0, 0], [1, 0]], "cubic")
    with pytest.raises(ValueError, match="n_points"):
        interpolate([[0, 0], [1, 0]], n_points=1)


def test_input_is_not_modified():
    p = np.array([[0, 0], [0.2, 0.1], [0.4, 0.3], [0.9, 0.9], [1, 0]])
    before = p.copy()
    for shape in SHAPES:
        enforce_symmetry(p, shape)
    np.testing.assert_array_equal(p, before)


# -- interpolate -----------------------------------------------------------------------------------------------


def test_interpolate_default_counts():
    assert len(interpolate([[0, 0], [0.5, 0.1], [1, 0]])) == 100
    p = np.column_stack([np.linspace(0, 1, 12), np.sin(np.arange(12))])
    p[0], p[-1] = [0, 0], [1, 0]
    assert len(interpolate(p)) == 120
    assert len(interpolate(p, n_points=7)) == 7


def test_interpolate_passes_through_control_points_and_endpoints():
    p = np.array([[0, 0], [0.2, 0.1], [0.5, -0.2], [0.8, 0.1], [1, 0]])
    s = interpolate(p, n_points=401)  # control points sit at the chord-length parameters
    np.testing.assert_array_equal(s[[0, -1]], p[[0, -1]])
    t = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))])
    idx = np.round(t / t[-1] * 400).astype(int)
    # the control points are on the curve: the nearest sample is within the sampling step
    dist = np.linalg.norm(s[None] - p[:, None], axis=2).min(axis=1)
    assert dist.max() < 3e-3
    assert np.all(idx >= 0)


def test_interpolate_modes():
    p = np.array([[0, 0], [0.3, 0.2], [0.7, -0.1], [1, 0]])
    np.testing.assert_array_equal(interpolate(p, "linear"), p)
    np.testing.assert_array_equal(interpolate(p, "linear", n_points=50), p)
    np.testing.assert_array_equal(interpolate([[0, 0], [1, 0]]), [[0, 0], [1, 0]])


def test_interpolate_turns_back_in_x():
    p = np.array([[0, 0], [0.4, 0.3], [0.2, 0.5], [0.6, 0.6], [1, 0]])  # x is not monotonic
    s = interpolate(p)
    assert np.all(np.isfinite(s))
    assert np.any(np.diff(s[:, 0]) < 0)


def test_interpolate_duplicate_points_count_once():
    p = np.array([[0, 0], [0.5, 0.1], [0.5, 0.1], [1, 0]])
    np.testing.assert_allclose(interpolate(p), interpolate(p[[0, 1, 3]]))
    np.testing.assert_array_equal(interpolate([[0, 0], [0, 0], [1, 0]]), [[0, 0], [1, 0]])


@pytest.mark.parametrize("shape", ("U", "S"))
@pytest.mark.parametrize("m", [3, 4, 7, 12])
def test_interpolation_keeps_the_symmetry(shape, m):
    rng = np.random.default_rng(m)
    p = enforce_symmetry(np.vstack([[0, 0], random_points(rng, m - 2), [1, 0]]), shape)
    s = interpolate(p, n_points=101)
    np.testing.assert_allclose(s, mirror(s, shape), atol=1e-12)
    s_even = interpolate(p, n_points=100)
    np.testing.assert_allclose(s_even, mirror(s_even, shape), atol=1e-12)


# -- EdgeShape -------------------------------------------------------------------------------------------------


def test_edge_shape_from_points():
    e = EdgeShape.from_points("S", [[0, 0], [0.3, 0.2], [0.5, 0.1], [0.4, 0.9], [1, 0]])
    assert e.shape == "S" and e.mode == "spline"
    assert is_symmetric(e.points, "S")
    assert not e.points.flags.writeable
    np.testing.assert_array_equal(e.points[[0, -1]], [[0, 0], [1, 0]])
    assert len(e.samples()) == 100
    assert len(e.samples(30)) == 30
    straight = EdgeShape.from_points("I")
    np.testing.assert_array_equal(straight.points, [[0, 0], [1, 0]])
    assert EdgeShape.from_points("J", [[0, 0], [0.5, 0.2], [1, 0]]) == EdgeShape.from_points(
        "J", [[0, 0], [0.5, 0.2], [1, 0]]
    )
    assert EdgeShape.from_points("J", [[0, 0], [0.5, 0.2], [1, 0]], "linear") != EdgeShape.from_points(
        "J", [[0, 0], [0.5, 0.2], [1, 0]]
    )
    assert EdgeShape.from_points("J") != "J"


def test_edge_shape_rejects_wrong_endpoints_and_modes():
    with pytest.raises(ValueError, match=r"from \(0, 0\) to \(1, 0\)"):
        EdgeShape.from_points("J", [[0, 0], [0.5, 0.2], [1, 0.1]])
    with pytest.raises(ValueError, match=r"from \(0, 0\) to \(1, 0\)"):
        EdgeShape.from_points("U", [[0.1, 0], [0.5, 0.2], [1, 0]])
    with pytest.raises(ValueError, match="mode"):
        EdgeShape.from_points("J", None, "cubic")
    with pytest.raises(ValueError, match="shape"):
        EdgeShape.from_points("Q")
