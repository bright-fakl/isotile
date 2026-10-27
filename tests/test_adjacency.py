"""Lattice adjacency equals geometric adjacency (shared-vertex counting) for all 81 types."""

import numpy as np
import pytest

from isotile import IsohedralTiling, geometric_adjacency, get_type

from .conftest import random_tilings


def patch(tiling: IsohedralTiling):
    """About 150-400 tiles around the origin, whatever the size of the type's tile."""
    extent = 6.0 * np.sqrt(tiling.area)
    return tiling.cover((-extent, -extent, extent, extent))


def same(a, b) -> bool:
    return (a != b).nnz == 0


def check(tiling: IsohedralTiling) -> int:
    tiles = patch(tiling)
    edge, vertex_only = geometric_adjacency(tiles.vertices(), tol=1e-6 * max(1.0, np.sqrt(tiling.area)))
    assert same(tiles.adjacency("edge"), edge)
    assert same(tiles.adjacency("vertex-only"), vertex_only)
    assert same(tiles.adjacency("vertex"), edge + vertex_only)
    assert same(tiles.adjacency("edge"), tiles.adjacency("edge").T.tocsr())
    return len(tiles)


def test_default_parameters(number):
    assert check(IsohedralTiling(number)) > 50


@pytest.mark.parametrize("spread", [0.05, 0.15, 0.3])
def test_random_parameters(number, spread):
    tilings = random_tilings(number, 3, spread, seed=7)
    if get_type(number).n_params == 0:
        pytest.skip("no parameters")
    assert tilings, "no valid random parameters"
    for t in tilings:
        check(t)


def test_edge_neighbor_count_equals_vertex_count(number):
    tiling = IsohedralTiling(number)
    tiles = patch(tiling)
    degree = np.asarray(tiles.adjacency("edge").sum(axis=1)).ravel()
    interior = np.linalg.norm(tiles.centroids, axis=1) < 0.5 * np.abs(tiles.centroids).max()
    assert np.all(degree[interior] == tiling.type.n_vertices)


def test_table_offsets_are_small(number):
    assert np.abs(get_type(number).adjacency[:, 1:3]).max() <= 2


def test_square_and_hexagon_neighbors():
    sq = IsohedralTiling.square().scaled(2.0).cover((-20, -20, 20, 20))
    deg = np.asarray(sq.adjacency("edge").sum(axis=1)).ravel()
    dvert = np.asarray(sq.adjacency("vertex-only").sum(axis=1)).ravel()
    assert deg.max() == 4
    assert dvert.max() == 4
    hx = IsohedralTiling.hexagon(True).cover((-20, -20, 20, 20))
    assert np.asarray(hx.adjacency("edge").sum(axis=1)).max() == 6
    assert hx.adjacency("vertex-only").nnz == 0


def test_neighbors_of_one_tile():
    tiles = IsohedralTiling(7).cover((-6, -6, 6, 6))
    i = int(np.argmin(np.linalg.norm(tiles.centroids, axis=1)))
    assert len(tiles.neighbors(i, "edge")) == 6


def test_adjacency_kind_is_validated():
    with pytest.raises(ValueError, match="kind"):
        IsohedralTiling(7).cover((-3, -3, 3, 3)).adjacency("corner")


@pytest.mark.parametrize("number", [1, 7, 28])
@pytest.mark.parametrize("kind", ["edge", "vertex", "vertex-only"])
def test_neighbor_lists_equal_neighbors(number, kind):
    tiles = patch(IsohedralTiling(number))
    lists = tiles.neighbor_lists(kind)
    assert len(lists) == len(tiles)
    for i in range(0, len(tiles), 7):
        assert lists[i].dtype == np.int64
        np.testing.assert_array_equal(lists[i], tiles.neighbors(i, kind))
    assert all(np.all(np.diff(a) > 0) for a in lists)


def test_neighbor_lists_default_kind_is_edge():
    tiles = patch(IsohedralTiling(7))
    for a, b in zip(tiles.neighbor_lists(), tiles.neighbor_lists("edge"), strict=True):
        np.testing.assert_array_equal(a, b)


def test_neighbor_lists_of_empty_and_single_tile_sets():
    tiles = IsohedralTiling.square().cover((0, 0, 3, 3))
    assert tiles.select(np.zeros(len(tiles), dtype=bool)).neighbor_lists() == []
    single = tiles.select(np.array([0]))
    assert [len(a) for a in single.neighbor_lists()] == [0]


def test_neighbor_lists_of_10000_tiles_are_fast():
    import time

    tiles = IsohedralTiling.hexagon().fill_region((-1, -1, 1, 1), n_tiles=10000)
    assert len(tiles) >= 10000
    start = time.perf_counter()
    lists = tiles.neighbor_lists()
    assert time.perf_counter() - start < 0.1
    assert len(lists) == len(tiles)
