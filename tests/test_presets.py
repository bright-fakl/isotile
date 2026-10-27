"""Tiling presets and edge curve presets."""

import numpy as np
import pytest

from isotile import (
    IsohedralTiling,
    check_prototile,
    edge_curve_preset,
    edge_curves_from_preset,
    is_symmetric,
    list_edge_curve_presets,
    list_presets,
    tiling_types,
)

PRESETS = list(list_presets())
CURVES = list(list_edge_curve_presets())


def test_the_fifteen_presets():
    assert len(PRESETS) == 15
    assert PRESETS[:3] == ["regular_hexagon", "square", "equilateral_triangle"]
    assert all(isinstance(d, str) and d for d in list_presets().values())


@pytest.mark.parametrize("name", PRESETS)
def test_preset_builds_and_is_valid(name):
    t = IsohedralTiling.from_preset(name)
    assert check_prototile(t.type, t.parameters, edge_curves=t.edge_curves, edge_curve_modes=t.edge_curve_modes).valid
    x, y = t.prototile().T
    assert abs(0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y) - t.area) < 1e-9 * t.area
    assert list_presets()[name].startswith(f"IH{t.type.number}.")
    tiles = t.fill_region((0, 0, 4, 3), n_tiles=30)
    assert len(tiles) >= 30


@pytest.mark.parametrize("name", PRESETS)
def test_preset_curves_have_the_symmetry_of_their_shape(name):
    t = IsohedralTiling.from_preset(name)
    from isotile.presets import preset_arguments

    for k, raw in preset_arguments(name).get("edge_curves", {}).items():
        np.testing.assert_allclose(t.edge_curves[k], np.asarray(raw, dtype=float), atol=1e-12)


def test_preset_overrides_and_errors():
    t = IsohedralTiling.from_preset("wide_parallelogram", parameters=[0.3, 0.3])
    np.testing.assert_array_equal(t.parameters, [0.3, 0.3])
    straight = IsohedralTiling.from_preset("scalloped_hexagon", edge_curves={})
    assert straight.edge_curves == {}
    framed = IsohedralTiling.from_preset("cairo", frame=[[2, 0, 1], [0, 2, 1]])
    np.testing.assert_allclose(framed.area, 4 * IsohedralTiling.from_preset("cairo").area)
    with pytest.raises(ValueError, match="unknown preset"):
        IsohedralTiling.from_preset("nonexistent")


def test_presets_are_not_shared_between_tilings():
    a = IsohedralTiling.from_preset("scalloped_hexagon")
    a.edge_curves[0][:] = 0
    np.testing.assert_array_equal(IsohedralTiling.from_preset("scalloped_hexagon").edge_curves[0][1], [0.2, 0.08])


# -- edge curve presets ------------------------------------------------------------------------------------------


@pytest.mark.parametrize("name", CURVES)
@pytest.mark.parametrize("shape", ["I", "J", "U", "S"])
def test_curve_preset_respects_the_shape(name, shape):
    p = edge_curve_preset(name, shape, 0.1)
    assert is_symmetric(p, shape, tol=0.0)
    np.testing.assert_array_equal(p[[0, -1]], [[0, 0], [1, 0]])
    assert np.all(np.abs(p[:, 1]) <= 0.1 + 1e-12)
    if shape == "I" or name == "straight":
        np.testing.assert_array_equal(p, [[0, 0], [1, 0]])
    else:
        assert len(p) == 21


def test_curve_preset_values():
    np.testing.assert_allclose(edge_curve_preset("bump", "J", 0.2)[10], [0.5, 0.2])
    np.testing.assert_allclose(edge_curve_preset("dent", "J", 0.2)[10], [0.5, -0.2])
    wave = edge_curve_preset("wave", "J", 0.1)
    np.testing.assert_allclose(wave[5], [0.25, 0.1])
    np.testing.assert_allclose(wave[15], [0.75, -0.1])
    assert np.abs(edge_curve_preset("s_curve", "J", 0.1)[:, 1]).max() == pytest.approx(0.1, rel=1e-2)
    with pytest.raises(ValueError, match="unknown edge curve preset"):
        edge_curve_preset("zigzag")
    with pytest.raises(ValueError, match="shape"):
        edge_curve_preset("wave", "Q")


@pytest.mark.parametrize("name", CURVES)
def test_curve_presets_fit_every_type(name):
    """A small amplitude gives a valid prototile for each of the 81 types, whatever the edge shapes."""
    for number in tiling_types():
        curves = edge_curves_from_preset(number, name, 0.04)
        report = check_prototile(number, edge_curves=curves)
        assert report.valid, (number, name, report.message)
        t = IsohedralTiling(number, edge_curves=curves)
        assert set(t.edge_curves) == set(curves)
        for e in t.edges:
            assert e.shape in ("I", "J", "U", "S")


def test_curves_from_preset_skip_straight_edges():
    assert edge_curves_from_preset(76, "wave") == {}
    assert set(edge_curves_from_preset(4, "wave")) == {0, 1, 2, 3, 4}
    assert set(edge_curves_from_preset(28, "wave")) == {0, 1, 2}
