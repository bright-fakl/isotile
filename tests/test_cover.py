"""cover: a tile is included iff its straight-edge polygon overlaps the rectangle with positive area."""

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
    polygons = np.asarray(big.polygons())
    hit = shapely.area(shapely.intersection(polygons, box)) > 1e-9 * tiling.area
    return keys(big.select(hit))


@pytest.mark.parametrize("seed", range(3))
def test_cover_equals_shapely_overlap_area(number, seed):
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


def test_touching_only_is_excluded():
    squares = IsohedralTiling.square()
    assert len(squares.cover((0.0, 0.0, 1.0, 1.0))) == 1  # the neighbors only touch the square
    assert len(squares.cover((0.25, 0.25, 0.75, 0.75))) == 1
    assert len(squares.cover((1.0, 0.25, 2.0, 0.75))) == 1  # touching along the edges x = 1 and x = 2 does not count
    assert len(squares.cover((0.0, 0.0, 2.0, 2.0))) == 4
    assert len(squares.cover((0.5, 0.5, 1.0000001, 1.0))) == 2  # a real overlap of 5e-8 of the neighbor's width
    hexagons = IsohedralTiling.hexagon(False)  # the tile (0, 0) has its vertex (1, 0) at the left edge of this region
    tiles = hexagons.cover((1.0, -0.2, 2.0, 0.2))
    assert not np.any(np.linalg.norm(tiles.centroids, axis=1) < 1e-9)
    assert len(tiles) > 0


def test_degenerate_regions_raise():
    tiling = IsohedralTiling.square()
    for bounds in [(0, 0, 0, 1), (0, 0, 1, 0), (1, 0, 0, 1), (0, 1, 1, 0), (0, 0, np.inf, 1), (0, 0, np.nan, 1)]:
        with pytest.raises(ValueError, match="bounds"):
            tiling.cover(bounds)


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


def test_square_tiling_on_tile_edges_far_from_the_origin():
    # the boundary of the region lies on tile edges: 12 x 12 tiles at any distance
    for c in (0.0, 1e3, 1e5, 1e6):
        region = (c - 1.2, c - 1.2, c + 1.2, c + 1.2)
        assert len(IsohedralTiling.square().fill_region(region, tile_size=0.2)) == 144


def test_no_holes_far_from_the_origin():
    import shapely
    from shapely.geometry import Polygon

    rng = np.random.default_rng(0)
    shapes = []
    while len(shapes) < 100:
        p = rng.uniform(-1, 1, (3, 2))
        if Polygon(p).area > 0.02:
            shapes.append(p)
    c = 1e6
    region = (c - 1, c - 1, c + 1, c + 1)
    points = shapely.points(np.random.default_rng(5).uniform(region[:2], region[2:], (20000, 2)))
    for pts in shapes:
        unit = pts - Polygon(pts).centroid.coords[0]
        unit /= np.sqrt(Polygon(unit).area)
        polygons = IsohedralTiling.triangle(unit).fill_region(region, tile_size=0.25).polygons()
        covered = np.unique(shapely.STRtree(polygons).query(points, predicate="within")[0])
        assert len(covered) == len(points)


CURVED_PRESETS = ("wavy_square", "puzzle_piece", "scalloped_hexagon", "wavy_triangle", "arrow_quad", "flowers")


def uncovered_fraction(tiles, bounds):
    import shapely

    box = shapely.box(*bounds)
    return box.difference(shapely.union_all(tiles.polygons())).area / box.area


@pytest.mark.parametrize("name", CURVED_PRESETS)
def test_curved_outline_covers_the_region_at_margin_zero(name):
    tiling = IsohedralTiling.from_preset(name)
    worst = 0.0
    for seed in range(6):
        rng = np.random.default_rng(seed)
        x0, y0 = rng.uniform(-5, 5, 2)
        bounds = (x0, y0, x0 + rng.uniform(3, 12), y0 + rng.uniform(3, 12))
        tiles = tiling.fill_region(bounds, n_tiles=int(rng.integers(20, 300)))  # the default for a curved tiling
        worst = max(worst, uncovered_fraction(tiles, bounds))
    assert worst < 1e-9


def test_straight_outline_leaves_slivers_that_curved_covers():
    tiling = IsohedralTiling.from_preset("wavy_triangle")
    bounds = (0.0, 0.0, 10.0, 6.0)
    straight = tiling.fill_region(bounds, n_tiles=100, outline="straight")
    curved = tiling.fill_region(bounds, n_tiles=100, outline="curved")
    assert uncovered_fraction(straight, bounds) > 1e-3
    assert uncovered_fraction(curved, bounds) < 1e-9
    assert keys(straight) < keys(curved)


@pytest.mark.parametrize("seed", range(2))
def test_curved_outline_equals_shapely_overlap_of_the_curved_polygons(number, seed):
    from .conftest import random_valid_curved_tiling

    tiling = random_valid_curved_tiling(number, seed)
    if tiling is None:
        pytest.skip("no curvable edge")
    rng = np.random.default_rng(seed)
    scale = np.sqrt(tiling.area)
    x0, y0 = rng.uniform(-3, 3, 2) * scale
    bounds = (x0, y0, x0 + rng.uniform(0.3, 4) * scale, y0 + rng.uniform(0.3, 4) * scale)
    pad = 4 * np.ptp(tiling.vertices, axis=0).max() + 2 * np.abs(tiling.translations).sum()
    big = tiling.cover((bounds[0] - pad, bounds[1] - pad, bounds[2] + pad, bounds[3] + pad), outline="curved")
    hit = shapely.area(shapely.intersection(np.asarray(big.polygons()), shapely.box(*bounds))) > 1e-9 * tiling.area
    assert keys(tiling.cover(bounds, outline="curved")) == keys(big.select(hit))


def test_outline_modes_agree_without_curves(number):
    tiling = IsohedralTiling(number)
    for bounds in ((0.0, 0.0, 5.0, 3.0), (-1.3, 0.2, 2.9, 7.1)):
        assert keys(tiling.cover(bounds)) == keys(tiling.cover(bounds, outline="curved"))
        assert keys(tiling.fill_region(bounds, n_tiles=40)) == keys(
            tiling.fill_region(bounds, n_tiles=40, outline="curved")
        )


def test_default_outline_is_curved_with_curves_and_straight_without(number):
    from .conftest import random_valid_curved_tiling

    bounds = (0.0, 0.0, 6.0, 4.0)
    plain = IsohedralTiling(number)
    assert keys(plain.fill_region(bounds, n_tiles=30)) == keys(
        plain.fill_region(bounds, n_tiles=30, outline="straight")
    )
    curved = random_valid_curved_tiling(number, 0)
    if curved is not None:
        assert keys(curved.fill_region(bounds, n_tiles=30)) == keys(
            curved.fill_region(bounds, n_tiles=30, outline="curved")
        )
        assert keys(curved.cover(bounds)) == keys(curved.cover(bounds, outline="curved"))


def test_default_on_a_curved_preset_covers_at_margin_zero_and_straight_is_the_former_result():
    tiling = IsohedralTiling.from_preset("puzzle_piece")
    bounds = (0.0, 0.0, 10.0, 6.0)
    assert len(tiling.fill_region(bounds, n_tiles=100, margin=0.1)) == 126
    straight = tiling.fill_region(bounds, n_tiles=100, margin=0.1, outline="straight")
    assert len(straight) == 112
    assert keys(straight) == keys(tiling.with_edge_curves(None).fill_region(bounds, n_tiles=100, margin=0.1))
    assert uncovered_fraction(tiling.fill_region(bounds, n_tiles=100), bounds) < 1e-9
    assert uncovered_fraction(tiling.fill_region(bounds, n_tiles=100, outline="straight"), bounds) > 1e-4


def test_invalid_outline():
    for bad in ("rounded", "Straight", 1):
        with pytest.raises(ValueError, match="outline"):
            IsohedralTiling.square().cover((0, 0, 1, 1), outline=bad)  # type: ignore[arg-type]
