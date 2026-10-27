"""Named constructors, frame operations and the frame as public API."""

import numpy as np
import pytest

from isotile import IsohedralTiling, geometric_adjacency


def same_polygon(a: np.ndarray, b: np.ndarray, tol: float = 1e-9) -> bool:
    """True if the closed polygons have the same vertices in the same cyclic order, either direction."""
    for candidate in (b, b[::-1]):
        for shift in range(len(b)):
            if np.allclose(a, np.roll(candidate, shift, axis=0), atol=tol):
                return True
    return False


def random_triangle(rng: np.random.Generator) -> np.ndarray:
    while True:
        pts = rng.uniform(-1, 1, (3, 2))
        if (
            abs((pts[1, 0] - pts[0, 0]) * (pts[2, 1] - pts[0, 1]) - (pts[1, 1] - pts[0, 1]) * (pts[2, 0] - pts[0, 0]))
            > 0.05
        ):
            return pts


def random_simple_quadrilateral(rng: np.random.Generator) -> np.ndarray:
    """Random simple quadrilateral, convex or not: three points in angular order around the fourth plus a vertex."""
    while True:
        pts = rng.uniform(-1, 1, (4, 2))
        center = pts.mean(axis=0)
        pts = pts[np.argsort(np.arctan2(*(pts - center).T[::-1]))]  # star-shaped about the mean: simple
        area = 0.5 * np.sum(pts[:, 0] * np.roll(pts[:, 1], -1) - np.roll(pts[:, 0], -1) * pts[:, 1])
        if abs(area) > 0.05:
            return pts


def check_lattice_equals_geometry(tiling: IsohedralTiling) -> None:
    extent = 6.0 * np.sqrt(tiling.area)
    tiles = tiling.cover((-extent, -extent, extent, extent))
    edge, vertex_only = geometric_adjacency(tiles.vertices(), tol=1e-7 * np.sqrt(tiling.area))
    assert (tiles.adjacency("edge") != edge).nnz == 0
    assert (tiles.adjacency("vertex-only") != vertex_only).nnz == 0


def test_square():
    t = IsohedralTiling.square()
    assert t.type.number == 76
    assert same_polygon(t.vertices, np.array([[1.0, 0], [1, 1], [0, 1], [0, 0]]))
    tiles = t.cover((-5, -5, 5, 5))
    assert len(tiles) == 144  # tiles that only touch the closed region count
    inner = np.linalg.norm(tiles.centroids, axis=1) < 3
    assert np.all(np.asarray(tiles.adjacency("edge").sum(axis=1)).ravel()[inner] == 4)
    assert np.all(np.asarray(tiles.adjacency("vertex-only").sum(axis=1)).ravel()[inner] == 4)


def extent(points: np.ndarray) -> np.ndarray:
    return points.max(axis=0) - points.min(axis=0)


def test_hexagon_orientation():
    flat = IsohedralTiling.hexagon(pointy_top=False).vertices
    pointy = IsohedralTiling.hexagon(pointy_top=True).vertices
    np.testing.assert_allclose(extent(flat), [2.0, np.sqrt(3)], atol=1e-9)
    np.testing.assert_allclose(extent(pointy), [np.sqrt(3), 2.0], atol=1e-9)
    assert same_polygon(IsohedralTiling.hexagon(False).rotated(30).vertices, pointy)
    assert np.any(np.isclose(flat[:, 1], 0) & np.isclose(flat[:, 0], 1))  # a vertex at angle 0
    assert np.any(np.isclose(pointy[:, 0], 0) & np.isclose(pointy[:, 1], 1))  # a vertex at the top


def test_hexagon_fill_region_keeps_orientation():
    for pointy_top in (True, False):
        tiles = IsohedralTiling.hexagon(pointy_top).fill_region((0, 0, 10, 10), tile_size=0.5)
        width, height = extent(tiles.vertices()[0])
        assert (height > width) == pointy_top


def test_hexagon_neighbors():
    for pointy_top in (True, False):
        tiles = IsohedralTiling.hexagon(pointy_top).cover((-10, -10, 10, 10))
        edge = np.asarray(tiles.adjacency("edge").sum(axis=1)).ravel()
        assert edge.max() == 6
        assert tiles.adjacency("vertex-only").nnz == 0


def test_default_triangle_and_quadrilateral():
    t = IsohedralTiling.triangle()
    assert t.type.number == 84
    assert t.area == pytest.approx(np.sqrt(3) / 4)
    assert np.allclose(np.linalg.norm(np.roll(t.vertices, -1, axis=0) - t.vertices, axis=1), 1.0)
    q = IsohedralTiling.quadrilateral()
    assert q.type.number == 46
    assert same_polygon(q.vertices, np.array([[0.0, 0], [1, 0], [1, 1], [0, 1]]))


@pytest.mark.parametrize("seed", range(25))
def test_any_triangle(seed):
    rng = np.random.default_rng(seed)
    pts = random_triangle(rng)
    t = IsohedralTiling.triangle(pts)
    assert same_polygon(t.vertices, pts) or same_polygon(t.vertices, pts[::-1])
    check_lattice_equals_geometry(t)


@pytest.mark.parametrize("seed", range(25))
def test_any_simple_quadrilateral(seed):
    rng = np.random.default_rng(seed)
    pts = random_simple_quadrilateral(rng)
    t = IsohedralTiling.quadrilateral(pts)
    assert same_polygon(t.vertices, pts) or same_polygon(t.vertices, pts[::-1])
    check_lattice_equals_geometry(t)


def test_clockwise_input_is_accepted():
    pts = np.array([[0.0, 0], [0, 1], [2, 1], [3, 0]])
    t = IsohedralTiling.quadrilateral(pts)
    assert same_polygon(t.vertices, pts)


def test_self_intersecting_quadrilateral_is_rejected():
    with pytest.raises(ValueError, match="invalid prototile"):
        IsohedralTiling.quadrilateral([(0, 0), (2, 1), (1, 0), (0, 3)])
    with pytest.raises(ValueError, match="invalid prototile"):
        IsohedralTiling.quadrilateral([(0, 0), (1, 1), (1, 0), (0, 1)])


def test_degenerate_and_malformed_polygons_are_rejected():
    with pytest.raises(ValueError):
        IsohedralTiling.triangle([(0, 0), (1, 0), (2, 0)])
    with pytest.raises(ValueError, match="needs 3"):
        IsohedralTiling.triangle([(0, 0), (1, 0)])
    with pytest.raises(ValueError, match="needs 4"):
        IsohedralTiling.quadrilateral([(0, 0), (1, 0), (1, 1)])
    with pytest.raises(ValueError, match="no extent"):
        IsohedralTiling.triangle([(0, 0), (0, 0), (0, 0)])


def test_frame_is_public_and_validated():
    t = IsohedralTiling(7, frame=[[2, 0, 1], [0, 2, -1]])
    np.testing.assert_allclose(t.frame, [[2, 0, 1], [0, 2, -1]])
    np.testing.assert_allclose(t.vertices, IsohedralTiling(7).vertices * 2 + [1, -1], atol=1e-12)
    np.testing.assert_allclose(t.translations, IsohedralTiling(7).translations * 2, atol=1e-12)
    assert t.area == pytest.approx(4 * IsohedralTiling(7).area)
    t.frame[0, 0] = 99  # a copy
    assert t.frame[0, 0] == 2
    with pytest.raises(ValueError, match="frame"):
        IsohedralTiling(7, frame=np.eye(3))
    with pytest.raises(ValueError, match="singular"):
        IsohedralTiling(7, frame=[[1, 2, 0], [2, 4, 0]])
    with pytest.raises(ValueError, match="frame"):
        IsohedralTiling(7, frame=[[np.nan, 0, 0], [0, 1, 0]])


def test_frame_operations():
    t = IsohedralTiling(7)
    r = t.rotated(90)
    np.testing.assert_allclose(r.vertices, t.vertices @ np.array([[0, 1], [-1, 0]]), atol=1e-12)
    s = t.scaled(3.0)
    assert s.area == pytest.approx(9 * t.area)
    np.testing.assert_array_equal(s.parameters, t.parameters)
    w = t.with_frame([[1, 0, 5], [0, 1, 6]])
    np.testing.assert_allclose(w.vertices, t.vertices + np.array([5, 6]), atol=1e-12)
    # the original is unchanged
    np.testing.assert_allclose(t.frame, np.eye(3)[:2])
    assert t.rotated(33).scaled(0.7).type is t.type


def test_reflecting_frame_keeps_adjacency():
    t = IsohedralTiling(7, frame=[[1, 0, 0], [0, -1, 0]])
    tiles = t.cover((-6, -6, 6, 6))
    edge, vertex_only = geometric_adjacency(tiles.vertices(), tol=1e-7)
    assert (tiles.adjacency("edge") != edge).nnz == 0
    assert (tiles.adjacency("vertex-only") != vertex_only).nnz == 0
