"""cover: a tile is included iff its straight-edge polygon intersects the closed rectangle."""

import numpy as np
import pytest

from isotile import IsohedralTiling, tiling_types

from .conftest import random_tilings

shapely = pytest.importorskip("shapely")


def keys(tiles):
    return {tuple(r) for r in np.column_stack([tiles.cell, tiles.aspect]).tolist()}


def reference(tiling, bounds):
    """Tiles of a much larger region filtered by shapely's intersects (closed geometries)."""
    x0, y0, x1, y1 = bounds
    pad = 4 * np.ptp(tiling.vertices, axis=0).max() + 2 * np.abs(tiling.translations).sum()
    big = tiling.cover((x0 - pad, y0 - pad, x1 + pad, y1 + pad))
    box = shapely.box(x0, y0, x1, y1)
    hit = shapely.intersects(np.asarray(big.polygons()), box)
    return keys(big.select(hit))


@pytest.mark.parametrize("seed", range(3))
def test_cover_equals_shapely_intersects(number, seed):
    rng = np.random.default_rng(1000 * seed + number)
    tiling = random_tilings(number, 1, 0.1, seed=seed)[0] if seed and number in (1, 7) else IsohedralTiling(number)
    scale = np.sqrt(tiling.area)
    for _ in range(3):
        x0, y0 = rng.uniform(-5, 5, 2) * scale
        width, height = rng.uniform(0.05, 6, 2) * scale
        bounds = (x0, y0, x0 + width, y0 + height)
        assert keys(tiling.cover(bounds)) == reference(tiling, bounds)


def test_random_parameters_equal_shapely():
    rng = np.random.default_rng(3)
    for number in tiling_types():
        for tiling in random_tilings(number, 1, 0.15, seed=5):
            scale = np.sqrt(tiling.area)
            x0, y0 = rng.uniform(-3, 3, 2) * scale
            width, height = rng.uniform(0.1, 4, 2) * scale
            bounds = (x0, y0, x0 + width, y0 + height)
            assert keys(tiling.cover(bounds)) == reference(tiling, bounds)


def test_touching_counts_as_intersecting():
    squares = IsohedralTiling.square()
    assert len(squares.cover((0.0, 0.0, 1.0, 1.0))) == 9  # the tile and its 8 neighbors touch the closed square
    assert len(squares.cover((0.25, 0.25, 0.75, 0.75))) == 1
    assert (
        len(squares.cover((1.0, 0.25, 2.0, 0.75))) == 3
    )  # both tiles touching along the edge x = 1 and x = 2, and the middle one


def test_rectangle_inside_one_tile_and_region_inside_tile_with_no_vertex_inside():
    hexagon = IsohedralTiling.hexagon(False)
    assert len(hexagon.cover((-0.1, -0.1, 0.1, 0.1))) == 1
    # a thin rectangle crossing a tile with no tile vertex inside it
    assert len(hexagon.cover((-0.2, -5.0, 0.2, 5.0))) >= 5


def test_cover_never_includes_a_tile_that_misses_the_rectangle():
    tiling = IsohedralTiling(1)
    tiles = tiling.cover((0.0, 0.0, 3.0, 2.0))
    box = shapely.box(0, 0, 3, 2)
    assert all(p.intersects(box) for p in tiles.polygons())


def test_degenerate_point_region():
    assert len(IsohedralTiling.square().cover((0.2, 0.2, 0.2, 0.2))) == 1
