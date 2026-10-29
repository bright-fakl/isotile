"""with_parameters, no re-validation in with_frame, repr, InvalidPrototileError and TileSet.to_geojson."""

import json

import numpy as np
import pytest

import isotile
from isotile import InvalidPrototileError, IsohedralTiling, check_prototile, list_presets, tiling_types
from isotile import tiling as tiling_module

BOW = [[0, 0], [0.8, 0.5], [0.2, 0.5], [1, 0]]  # the first and the last segment cross
FRAME = [[1.3, -0.4, 2.0], [0.4, 1.3, -1.0]]


def same_state(a: IsohedralTiling, b: IsohedralTiling) -> None:
    assert a.type is b.type
    for name in ("parameters", "frame", "vertices", "aspects", "translations", "edge_transforms", "prototile"):
        x, y = getattr(a, name), getattr(b, name)
        x, y = (x(), y()) if callable(x) else (x, y)
        assert np.array_equal(x, y), name
    assert a.edge_curve_modes == b.edge_curve_modes
    assert a.edge_curves.keys() == b.edge_curves.keys()
    for k in a.edge_curves:
        assert np.array_equal(a.edge_curves[k], b.edge_curves[k])


def all_tilings() -> list[IsohedralTiling]:
    return [IsohedralTiling(n) for n in tiling_types()] + [IsohedralTiling.from_preset(p) for p in list_presets()]


# -- with_frame without re-validation ------------------------------------------------------------


def test_with_frame_does_not_check_the_prototile(monkeypatch):
    t = IsohedralTiling.from_preset("puzzle_piece")
    calls = []
    monkeypatch.setattr(tiling_module, "check_curves", lambda *a, **k: calls.append("curves"))
    monkeypatch.setattr(tiling_module, "check_prototile", lambda *a, **k: calls.append("prototile"))
    t.with_frame(FRAME)
    t.rotated(30).scaled(2.0)
    t.fill_region((0, 0, 4, 3), n_tiles=20, offset=(0.2, 0.1))
    assert calls == []


@pytest.mark.parametrize("tiling", all_tilings(), ids=lambda t: f"IH{t.type.number}-{len(t.edge_curves)}")
def test_with_frame_equals_the_constructor(tiling):
    built = IsohedralTiling(
        tiling.type,
        tiling.parameters,
        edge_curves=tiling.edge_curves,
        edge_curve_modes=tiling.edge_curve_modes,
        frame=FRAME,
    )
    same_state(tiling.with_frame(FRAME), built)


@pytest.mark.parametrize("name", ["puzzle_piece", "wavy_triangle", "fish_hexagon"])
@pytest.mark.parametrize("outline", ["straight", "curved"])
def test_fill_region_equals_the_constructor_path(name, outline):
    if name not in list_presets():
        pytest.skip("preset not available")
    t = IsohedralTiling.from_preset(name).rotated(17)
    tiles = t.fill_region((-2, -2, 2, 2), n_tiles=60, margin=0.1, outline=outline, offset=(0.3, 0.2))
    anchor = tiles.tiling
    built = IsohedralTiling(
        t.type, t.parameters, edge_curves=t.edge_curves, edge_curve_modes=t.edge_curve_modes, frame=anchor.frame
    )
    same_state(anchor, built)
    again = built.cover(tiles_bounds(tiles, anchor), outline=outline)
    assert len(again) >= len(tiles) > 0


def tiles_bounds(tiles, anchor):
    pts = tiles.outlines(curved=False).reshape(-1, 2)
    return (*pts.min(axis=0), *pts.max(axis=0))


def test_with_frame_rejects_a_bad_frame():
    t = IsohedralTiling(7)
    with pytest.raises(ValueError, match="frame"):
        t.with_frame(np.zeros((2, 3)))
    with pytest.raises(ValueError, match="frame"):
        t.with_frame(np.eye(3))


# -- with_parameters -------------------------------------------------------------------------------


def test_with_parameters_returns_a_new_tiling():
    t = IsohedralTiling.from_preset("puzzle_piece").rotated(10)
    before = t.parameters
    p = before + 0.01
    new = t.with_parameters(p)
    assert np.array_equal(t.parameters, before)
    assert np.array_equal(new.parameters, p)
    same_state(
        new,
        IsohedralTiling(t.type, p, edge_curves=t.edge_curves, edge_curve_modes=t.edge_curve_modes, frame=t.frame),
    )
    t.parameters = p
    same_state(new, t)


def test_with_parameters_rejects_what_the_setter_rejects():
    t = IsohedralTiling(7)
    with pytest.raises(InvalidPrototileError) as caught:
        t.with_parameters([0.25, -1.0])
    assert caught.value.report.bad_edges == (0, 2, 5)
    with pytest.raises(ValueError, match="parameters"):
        t.with_parameters([0.25])
    assert np.array_equal(t.parameters, IsohedralTiling(7).parameters)


def test_with_parameters_checks_the_curves():
    t = IsohedralTiling.from_preset("puzzle_piece")
    p = [-0.13645195008204428, 0.11322210842282326]  # a valid straight prototile whose curved outline is not
    assert check_prototile(t.type, p).valid
    expected = check_prototile(t.type, p, edge_curves=t.edge_curves, edge_curve_modes=t.edge_curve_modes)
    assert not expected.valid
    with pytest.raises(InvalidPrototileError) as caught:
        t.with_parameters(p)
    assert caught.value.report == expected
    assert IsohedralTiling(t.type, p).type is t.type  # without the curves it is a tiling


# -- InvalidPrototileError ---------------------------------------------------------------------------


def test_invalid_prototile_error_carries_the_report():
    expected = check_prototile(41, edge_curves={0: BOW}, edge_curve_modes={0: "linear"})
    with pytest.raises(ValueError, match="invalid prototile") as caught:
        IsohedralTiling(41, edge_curves={0: BOW}, edge_curve_modes={0: "linear"})
    assert isinstance(caught.value, InvalidPrototileError)
    assert caught.value.report == expected
    assert isotile.InvalidPrototileError is InvalidPrototileError


def test_invalid_prototile_error_from_every_entry_point():
    t = IsohedralTiling(41)
    with pytest.raises(InvalidPrototileError) as a:
        t.with_edge_curves({0: BOW}, {0: "linear"})
    assert a.value.report.bad_edges == (0, 2)
    with pytest.raises(InvalidPrototileError):
        t.parameters = [np.nan] * len(t.parameters)
    with pytest.raises(InvalidPrototileError):
        IsohedralTiling(7, [0.25, -1.0])


def test_malformed_arguments_stay_plain_value_errors():
    with pytest.raises(ValueError) as caught:
        IsohedralTiling(41, edge_curves={9: BOW})
    assert not isinstance(caught.value, InvalidPrototileError)
    with pytest.raises(ValueError) as caught:
        IsohedralTiling(7, [0.25])
    assert not isinstance(caught.value, InvalidPrototileError)


# -- repr -----------------------------------------------------------------------------------------------


@pytest.mark.parametrize("tiling", all_tilings(), ids=lambda t: f"IH{t.type.number}-{len(t.edge_curves)}")
def test_repr_round_trips(tiling):
    for t in (tiling, tiling.rotated(33).scaled(1.7)):
        text = repr(t)
        assert text.startswith("IsohedralTiling(")
        same_state(eval(text, {"IsohedralTiling": IsohedralTiling}), t)  # noqa: S307


def test_repr_omits_what_is_not_set():
    assert repr(IsohedralTiling(76)) == "IsohedralTiling(76, parameters=[])"
    text = repr(IsohedralTiling(41, edge_curves={0: [[0, 0], [0.5, 0.2], [1, 0]]}))
    assert "edge_curves={0: [[0.0, 0.0], [0.5, 0.2], [1.0, 0.0]]}" in text
    assert "edge_curve_modes={0: 'spline'}" in text
    assert "frame" not in text
    assert "frame=" in repr(IsohedralTiling.hexagon())


# -- to_geojson ----------------------------------------------------------------------------------------------


def test_to_geojson_matches_the_tiles():
    t = IsohedralTiling.from_preset("puzzle_piece")
    tiles = t.fill_region((0, 0, 4, 3), n_tiles=20, outline="straight")
    data = tiles.to_geojson()
    assert json.loads(json.dumps(data)) == data
    assert data["type"] == "FeatureCollection" and len(data["features"]) == len(tiles)
    outlines = tiles.outlines()
    colors = tiles.colors()
    for i, feature in enumerate(data["features"]):
        assert feature["type"] == "Feature"
        assert feature["properties"] == {
            "cell": tiles.cell[i].tolist(),
            "aspect": int(tiles.aspect[i]),
            "color": int(colors[i]),
        }
        (ring,) = feature["geometry"]["coordinates"]
        assert feature["geometry"]["type"] == "Polygon"
        ring = np.array(ring)
        assert np.array_equal(ring[0], ring[-1]) and len(ring) == outlines.shape[1] + 1
        area = 0.5 * np.sum(ring[:-1, 0] * ring[1:, 1] - ring[1:, 0] * ring[:-1, 1])
        assert area > 0.0
        assert {tuple(p) for p in ring[:-1]} == {tuple(p) for p in outlines[i]}


def test_to_geojson_orients_reflected_tiles_counterclockwise():
    t = IsohedralTiling.square().with_frame([[1, 0, 0], [0, -1, 0]])  # a reflection: the tile outlines are clockwise
    tiles = t.fill_region((0, 0, 3, 3), n_tiles=9)
    for f in tiles.to_geojson()["features"]:
        ring = np.array(f["geometry"]["coordinates"][0])
        assert 0.5 * np.sum(ring[:-1, 0] * ring[1:, 1] - ring[1:, 0] * ring[:-1, 1]) > 0.0


def test_to_geojson_properties_and_options():
    tiles = IsohedralTiling.from_preset("puzzle_piece").fill_region((0, 0, 4, 3), n_tiles=12, outline="straight")
    n = len(tiles)
    data = tiles.to_geojson({"index": np.arange(n), "name": [f"t{i}" for i in range(n)], "color": np.zeros(n, int)})
    assert data["features"][3]["properties"]["index"] == 3
    assert data["features"][3]["properties"]["name"] == "t3"
    assert all(f["properties"]["color"] == 0 for f in data["features"])
    json.dumps(data)
    straight = tiles.to_geojson(curved=False)
    assert len(straight["features"][0]["geometry"]["coordinates"][0]) == tiles.tiling.type.n_vertices + 1
    coarse = tiles.to_geojson(n_points=4)
    assert len(coarse["features"][0]["geometry"]["coordinates"][0]) < len(
        data["features"][0]["geometry"]["coordinates"][0]
    )
    with pytest.raises(ValueError, match="one element per tile"):
        tiles.to_geojson({"x": [1, 2]})


def test_to_geojson_of_no_tiles():
    tiles = IsohedralTiling.square().fill_region((0, 0, 2, 2), n_tiles=4)
    assert tiles.select(np.zeros(len(tiles), bool)).to_geojson() == {"type": "FeatureCollection", "features": []}
