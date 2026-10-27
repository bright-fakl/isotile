"""fill_region: tile counts, tile size, margin, frame handling and coverage of the region."""

import numpy as np
import pytest

from isotile import IsohedralTiling, tiling_types

from .conftest import random_tilings

shapely = pytest.importorskip("shapely")


def test_exactly_one_of_n_tiles_and_tile_size():
    t = IsohedralTiling(7)
    with pytest.raises(ValueError, match="exactly one"):
        t.fill_region((0, 0, 1, 1))
    with pytest.raises(ValueError, match="exactly one"):
        t.fill_region((0, 0, 1, 1), n_tiles=10, tile_size=0.1)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"n_tiles": 0},
        {"n_tiles": -3},
        {"tile_size": 0.0},
        {"tile_size": -1.0},
        {"tile_size": np.inf},
        {"n_tiles": 10, "margin": -0.5},
    ],
)
def test_invalid_arguments(kwargs):
    with pytest.raises(ValueError):
        IsohedralTiling(7).fill_region((0, 0, 1, 1), **kwargs)


def test_invalid_bounds():
    for bounds in [(0, 0, 0, 1), (1, 0, 0, 1), (0, 1, 1, 0), (0, 0, np.inf, 1)]:
        with pytest.raises(ValueError, match="bounds"):
            IsohedralTiling(7).fill_region(bounds, n_tiles=10)


def test_tile_area_is_tile_size_squared(number):
    tiles = IsohedralTiling(number).fill_region((-2, 1, 5, 4), tile_size=0.37)
    assert tiles.tiling.area == pytest.approx(0.37**2)
    assert len(tiles) > 0


def test_n_tiles_sets_the_tile_size_from_the_region_area():
    bounds = (0.0, 0.0, 4.0, 2.0)
    for number in (7, 46, 76, 84):
        tiles = IsohedralTiling(number).fill_region(bounds, n_tiles=200)
        assert tiles.tiling.area == pytest.approx(8.0 / 200)


@pytest.mark.parametrize("n_tiles", [1, 7, 50, 400, 3000])
def test_tile_count_semantics(number, n_tiles):
    bounds = (-3.0, 2.0, 5.0, 6.0)
    tiling = IsohedralTiling(number)
    tiles = tiling.fill_region(bounds, n_tiles=n_tiles)
    # the tiles cover the region, so their total area is at least the region's area
    assert len(tiles) >= n_tiles
    # boundary tiles come on top: all tiles meet the region, so lie in the region grown by the tile diameter
    v = tiles.vertices()
    diameter = np.ptp(v[0], axis=0).max()
    grown = (8.0 + 2 * diameter) * (4.0 + 2 * diameter)
    assert len(tiles) * tiles.tiling.area <= grown * (1 + 1e-9)


def test_margin_expands_the_region():
    t = IsohedralTiling(7)
    with_margin = t.fill_region((0, 0, 10, 4), tile_size=0.3, margin=0.25)
    explicit = t.fill_region((-2.5, -1.0, 12.5, 5.0), tile_size=0.3)
    assert len(with_margin) == len(explicit)
    np.testing.assert_allclose(with_margin.transforms, explicit.transforms)
    by_count = t.fill_region((0, 0, 10, 4), n_tiles=100, margin=0.25)
    assert by_count.tiling.area == pytest.approx(15.0 * 6.0 / 100)


def test_region_is_centered_on_the_native_origin():
    tiles = IsohedralTiling.hexagon().fill_region((10, 20, 14, 26), tile_size=0.5)
    nearest = np.argmin(np.linalg.norm(tiles.centroids - [12, 23], axis=1))
    np.testing.assert_allclose(tiles.centroids[nearest], [12, 23], atol=1e-9)
    assert tuple(tiles.cell[nearest]) == (0, 0)


def test_the_frame_scale_and_position_are_replaced_but_not_the_orientation():
    base = IsohedralTiling.quadrilateral([(0, 0), (3, 0), (4, 2), (1, 2)]).rotated(17)
    tiles = base.fill_region((0, 0, 5, 5), tile_size=0.4)
    a = base.frame[:, :2]
    b = tiles.tiling.frame[:, :2]
    ratio = b / a
    np.testing.assert_allclose(ratio, np.full((2, 2), ratio[0, 0]), atol=1e-9)
    np.testing.assert_allclose(tiles.tiling.frame[:, 2], [2.5, 2.5])


def test_the_tiling_itself_is_unchanged():
    t = IsohedralTiling(7)
    frame = t.frame
    t.fill_region((0, 0, 3, 3), n_tiles=30)
    np.testing.assert_array_equal(t.frame, frame)


def coverage_cases():
    rng = np.random.default_rng(2024)
    numbers = rng.choice(tiling_types(), 20, replace=False)
    for number in numbers.tolist():
        tilings = random_tilings(number, 1, 0.1, seed=3)
        tiling = tilings[0] if tilings else IsohedralTiling(number)
        x0, y0 = rng.uniform(-5, 5, 2)
        width, height = rng.uniform(0.5, 6, 2)
        yield tiling, (x0, y0, x0 + width, y0 + height), int(rng.integers(5, 300)), float(rng.uniform(0, 0.3))


@pytest.mark.parametrize("case", list(coverage_cases()), ids=lambda c: f"IH{c[0].type.number}")
def test_tiles_cover_the_region(case):
    tiling, bounds, n_tiles, margin = case
    tiles = tiling.fill_region(bounds, n_tiles=n_tiles, margin=margin)
    x0, y0, x1, y1 = bounds
    w, h = x1 - x0, y1 - y0
    region = shapely.box(x0 - w * margin, y0 - h * margin, x1 + w * margin, y1 + h * margin)
    polygons = tiles.polygons()
    assert all(p.is_valid for p in polygons)
    assert tiles.tiling.area == pytest.approx(region.area / n_tiles)
    # union of the tiles contains the region
    missing = region.difference(shapely.union_all(polygons))
    assert missing.area < 1e-9 * region.area
    # random points of the region lie in a tile; every tile meets the region
    rng = np.random.default_rng(1)
    pts = shapely.points(rng.uniform(region.bounds[:2], region.bounds[2:], (3000, 2)))
    hits = shapely.STRtree(polygons).query(pts, predicate="intersects")
    assert len(np.unique(hits[0])) == len(pts)
    assert all(p.bounds[0] < region.bounds[2] and p.bounds[2] > region.bounds[0] for p in polygons)
    # no tile appears twice
    assert len(np.unique(np.column_stack([tiles.cell, tiles.aspect]), axis=0)) == len(tiles)


def test_named_constructors_cover_too():
    rng = np.random.default_rng(5)
    constructors = [
        IsohedralTiling.square(),
        IsohedralTiling.hexagon(True),
        IsohedralTiling.hexagon(False),
        IsohedralTiling.triangle(rng.uniform(-1, 1, (3, 2)) + np.array([[0, 0], [3, 0], [0, 3]])),
        IsohedralTiling.quadrilateral([(0, 0), (4, 0), (1, 0.2), (0, 3)]),  # elongated, non-convex
    ]
    for tiling in constructors:
        tiles = tiling.fill_region((0, 0, 3, 2), n_tiles=60)
        region = shapely.box(0, 0, 3, 2)
        assert region.difference(shapely.union_all(tiles.polygons())).area < 1e-9 * region.area


def test_polygons_need_shapely_message(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "shapely":
            raise ImportError("no shapely")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(ImportError, match=r"isotile\[shapely\]"):
        IsohedralTiling.square().cover((0, 0, 1, 1)).polygons()
