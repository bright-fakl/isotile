"""Valid curved prototiles tile the plane: no overlaps, no gaps, unchanged adjacency, for all 81 types."""

import numpy as np
import pytest
from scipy.spatial import cKDTree

from isotile import IsohedralTiling, geometric_adjacency, tiling_types

from .conftest import curvable_ids, random_frame, random_valid_curved_tiling

CURVABLE = [n for n in tiling_types() if curvable_ids(n)]
slow = pytest.mark.slow


@pytest.fixture(params=CURVABLE, ids=[f"IH{n}" for n in CURVABLE])
def curved_number(request) -> int:
    return request.param


def patch(tiling: IsohedralTiling, half: float = 1.5, margin: float = 2.5):
    """Tiles that cover a square of side 2 * half tile sizes around the centroid, and the square.

    The tiles are those whose straight polygon meets the square grown by ``margin`` tile sizes; curves reach
    less far than that.
    """
    s, c = np.sqrt(tiling.area), tiling.centroid
    box = (c[0] - half * s, c[1] - half * s, c[0] + half * s, c[1] + half * s)
    margin *= s
    return tiling.cover((box[0] - margin, box[1] - margin, box[2] + margin, box[3] + margin)), box


def test_neighbors_share_their_edge_curve_exactly(curved_number):
    """Along every shared edge the two tiles have the same polyline."""
    tiling = random_valid_curved_tiling(curved_number, 1)
    assert tiling is not None
    tiles, _ = patch(tiling, 1.0, 1.0)
    outlines = tiles.outlines()
    adjacency = tiles.adjacency("edge").tocoo()
    tol = 1e-9 * np.sqrt(tiling.area)
    lengths = {len(line) for line in tiling.edge_outlines()}  # points of one shared edge, end points included
    checked = 0
    pairs = [(i, j) for i, j in zip(adjacency.row.tolist(), adjacency.col.tolist(), strict=True) if i < j]
    for i, j in pairs[:: max(1, len(pairs) // 40)]:
        near = cKDTree(outlines[j]).query(outlines[i], distance_upper_bound=tol)[0] < tol
        shared = int(near.sum())
        assert shared in lengths, (i, j, shared, lengths)
        checked += 1
    assert checked >= 10


def test_curved_tiles_leave_the_adjacency_unchanged(curved_number):
    """Shared-point counting on the curved outlines gives the lattice adjacency of the straight tiling."""
    tiling = random_valid_curved_tiling(curved_number, 2)
    assert tiling is not None
    tiles, _ = patch(tiling, 1.0, 1.0)
    edge, vertex_only = geometric_adjacency(tiles.outlines(), tol=1e-9 * np.sqrt(tiling.area))
    for kind, found in (("edge", edge), ("vertex-only", vertex_only)):
        assert (tiles.adjacency(kind) != found).nnz == 0, kind


@pytest.mark.parametrize("reflect", [False, True])
def test_tiles_cover_a_region_once(curved_number, reflect):
    """Random points of a region lie in exactly one curved tile (no gap, no overlap); frames included."""
    shapely = pytest.importorskip("shapely")
    rng = np.random.default_rng(curved_number)
    frame = random_frame(rng, reflect)
    tiling = random_valid_curved_tiling(curved_number, 3, frame=frame)
    assert tiling is not None
    tiles, box = patch(tiling)
    polygons = shapely.polygons(tiles.outlines())
    assert shapely.is_valid(polygons).all()
    points = shapely.points(rng.uniform([box[0], box[1]], [box[2], box[3]], (800, 2)))
    point_index, _ = shapely.STRtree(polygons).query(points, predicate="within")
    counts = np.bincount(point_index, minlength=len(points))
    assert counts.min() == 1 and counts.max() == 1, (int((counts == 0).sum()), int((counts > 1).sum()))
    # every tile has the area of the straight tile
    np.testing.assert_allclose(shapely.area(polygons), tiling.area, rtol=1e-9)


@slow
@pytest.mark.parametrize("amplitude", [0.15, 1.5])
def test_every_valid_draw_is_a_tiling(curved_number, amplitude):
    """The validity rule is sufficient: random curves of growing amplitude, every draw the check accepts tiles the plane."""
    shapely = pytest.importorskip("shapely")
    from isotile import check_prototile

    from .conftest import random_curves, random_tilings

    rng = np.random.default_rng(10 * curved_number + int(amplitude * 10))
    accepted = 0
    for seed in range(4):
        base = random_tilings(curved_number, 1, 0.1, seed=seed)[0]
        curves = random_curves(curved_number, rng, amplitude)
        if not check_prototile(curved_number, base.parameters, edge_curves=curves).valid:
            continue
        tiling = IsohedralTiling(curved_number, base.parameters, edge_curves=curves)
        tiles, box = patch(tiling, 1.5, 4.0)
        polygons = shapely.polygons(tiles.outlines())
        points = shapely.points(rng.uniform([box[0], box[1]], [box[2], box[3]], (400, 2)))
        point_index, _ = shapely.STRtree(polygons).query(points, predicate="within")
        counts = np.bincount(point_index, minlength=len(points))
        assert counts.min() == 1 and counts.max() == 1
        accepted += 1
    if amplitude <= 0.15:
        assert accepted > 0
