"""isotile reproduces tactile for all 81 types."""

import numpy as np
import pytest

from isotile import IsohedralTiling, find_types, get_type, tiling_types

from .conftest import random_tilings

tactile = pytest.importorskip("tactile")
Tactile = tactile.IsohedralTiling

pytestmark = pytest.mark.tactile


def as_array(points) -> np.ndarray:
    return np.array([[p.x, p.y] for p in points], dtype=float)


def assert_same_geometry(ours: IsohedralTiling, ref: Tactile) -> None:
    np.testing.assert_allclose(ours.vertices, as_array(ref.vertices), atol=1e-12)
    np.testing.assert_allclose(ours.aspects, np.array(ref.aspects, dtype=float).reshape(-1, 2, 3), atol=1e-12)
    np.testing.assert_allclose(ours.translation_u, [ref.t1.x, ref.t1.y], atol=1e-12)
    np.testing.assert_allclose(ours.translation_v, [ref.t2.x, ref.t2.y], atol=1e-12)
    np.testing.assert_allclose(ours.edge_transforms, np.array(ref.edges, dtype=float).reshape(-1, 2, 3), atol=1e-12)
    np.testing.assert_array_equal(ours.edge_reversed, ref.reversals)


def test_type_numbers():
    assert tiling_types() == tuple(tactile.tiling_types)
    assert len(tiling_types()) == 81


def test_defaults(number):
    ours, ref = IsohedralTiling(number), Tactile(number)
    np.testing.assert_allclose(ours.parameters, ref.parameters)
    assert (ours.type.n_vertices, ours.type.n_aspects, ours.type.n_params) == (
        ref.num_vertices,
        ref.num_aspects,
        ref.num_parameters,
    )
    assert_same_geometry(ours, ref)


def test_random_parameters(number):
    tilings = random_tilings(number, 5, 0.15, seed=1)
    if get_type(number).n_params == 0:
        assert tilings == [] or len(tilings) == 5
        return
    assert len(tilings) >= 1, "no valid random parameters"
    for ours in tilings:
        ref = Tactile(number)
        ref.parameters = [float(x) for x in ours.parameters]
        assert_same_geometry(ours, ref)


def test_edge_shapes(number):
    t, ref = get_type(number), Tactile(number)
    assert t.shape_types == tuple(s.name for s in ref.ttd.edge_shapes)
    assert t.edge_shapes == tuple(s.shape.name for s in ref.shapes)
    np.testing.assert_array_equal(t.edge_shape_ids, ref.ttd.edge_shape_ids)


def reference_tiles(ref: Tactile, bounds) -> dict[tuple[int, int, int], np.ndarray]:
    """Tactile's tiles by lattice triple; a dict keeps the first occurrence of repeated tiles, in tactile's order."""
    try:
        shapes = list(ref.fill_region_bounds(*bounds))
    except ZeroDivisionError:  # tactile fails on lattices aligned with the region; ours returns a result
        return {}
    return {(s.t1, s.t2, s.aspect): np.array(s.T, dtype=float).reshape(2, 3) for s in shapes}  # tactile's t1, t2


@pytest.mark.parametrize("seed", range(3))
def test_fill_equals_tactile(number, seed):
    rng = np.random.default_rng(100 * seed + number)
    tilings = [IsohedralTiling(number), *random_tilings(number, 1, 0.1, seed=seed)]
    for ours in tilings:
        ref = Tactile(number)
        ref.parameters = [float(x) for x in ours.parameters]
        x0, y0 = rng.uniform(-5, 5, 2)
        w, h = rng.uniform(3, 12, 2)
        bounds = (x0, y0, x0 + w, y0 + h)
        expected = reference_tiles(ref, bounds)
        tiles = ours.fill(bounds)
        keys = np.column_stack([tiles.cell, tiles.aspect]).tolist()
        got = {tuple(k): m for k, m in zip(keys, tiles.transforms, strict=True)}
        assert len(got) == len(tiles), "fill returned duplicates"
        if expected:
            assert list(got) == list(expected), "tile set or order differs from tactile"
            for k, m in expected.items():
                np.testing.assert_allclose(got[k], m, atol=1e-10)


def test_fill_is_deduplicated_and_larger_than_zero(number):
    tiles = IsohedralTiling(number).fill((-0.31, -0.27, 7.4, 6.1))
    assert len(tiles) > 0
    assert len(np.unique(np.column_stack([tiles.cell, tiles.aspect]), axis=0)) == len(tiles)


def test_cover_contains_every_tile_meeting_the_rectangle(number):
    ours = IsohedralTiling(number)
    bounds = (-1.3, -0.9, 4.2, 3.7)
    covered = ours.cover(bounds)
    big = ours.cover((-30, -30, 30, 30))
    v = big.vertices()
    meets = (
        (v[:, :, 0].max(axis=1) > bounds[0])
        & (v[:, :, 0].min(axis=1) < bounds[2])
        & (v[:, :, 1].max(axis=1) > bounds[1])
        & (v[:, :, 1].min(axis=1) < bounds[3])
    )

    def keys(tiles):
        return {tuple(r) for r in np.column_stack([tiles.cell, tiles.aspect]).tolist()}

    assert keys(big.select(meets)) == keys(covered)


def test_colors_equal_tactile(number):
    ref = Tactile(number)
    cells = [(a, b) for a in range(-4, 5) for b in range(-4, 5) for _ in range(ref.num_aspects)]
    aspects = [k for _ in range(81) for k in range(ref.num_aspects)]
    tiles = IsohedralTiling(number).tiles(cells, aspects)
    expected = [ref.get_color(a, b, k) for (a, b), k in zip(cells, aspects, strict=True)]
    np.testing.assert_array_equal(tiles.colors(), expected)


def test_parameter_validation():
    t = IsohedralTiling(7)
    before = t.parameters
    with pytest.raises(ValueError, match="parameters"):
        t.parameters = [0.1]
    with pytest.raises(ValueError, match="finite"):
        t.parameters = [np.nan, 0.5]
    with pytest.raises(ValueError, match="invalid prototile"):
        t.parameters = [0.25, -1.0]
    np.testing.assert_array_equal(t.parameters, before)
    with pytest.raises(ValueError, match="Invalid isohedral tiling type"):
        IsohedralTiling(19)


def test_zero_param_type_rejects_parameters():
    with pytest.raises(ValueError, match="parameters"):
        IsohedralTiling(76, [0.5])


def test_number_of_edge_shapes_equals_tactile(number):
    assert get_type(number).n_edge_shapes == Tactile(number).num_edge_shapes


def test_find_types_equals_tactile_derived_filter():
    assert find_types() == list(tactile.tiling_types)
