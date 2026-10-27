"""Serialization: a JSON-compatible dict, deterministic, round-trippable; version 1.0 and foreign keys are read."""

import json

import numpy as np
import pytest

from isotile import IsohedralTiling

from .conftest import curvable_ids, random_frame, random_valid_curved_tiling


def same(a: IsohedralTiling, b: IsohedralTiling) -> None:
    assert a.type.number == b.type.number
    np.testing.assert_array_equal(a.parameters, b.parameters)
    np.testing.assert_array_equal(a.frame, b.frame)
    assert a.edge_curve_modes == b.edge_curve_modes
    assert a.edge_curves.keys() == b.edge_curves.keys()
    for k in a.edge_curves:
        np.testing.assert_array_equal(a.edge_curves[k], b.edge_curves[k])
    np.testing.assert_array_equal(a.prototile(), b.prototile())


def test_round_trip_straight(number):
    t = IsohedralTiling(number)
    data = t.to_dict()
    assert data == {"version": "1.1", "tiling_type": number, "parameters": t.parameters.tolist()}
    same(IsohedralTiling.from_dict(data), t)


@pytest.mark.parametrize("seed", range(2))
def test_round_trip_with_curves_modes_and_frame(number, seed):
    frame = random_frame(np.random.default_rng(seed), reflect=bool(seed))
    t = random_valid_curved_tiling(number, seed, frame=frame)
    if t is None:
        pytest.skip("no curvable edge")
    data = t.to_dict()
    text = json.dumps(data)  # JSON-compatible
    assert json.loads(text) == data
    same(IsohedralTiling.from_dict(json.loads(text)), t)
    assert json.dumps(IsohedralTiling.from_dict(data).to_dict()) == text  # a fixed point


def test_modes_are_kept():
    curve = [[0, 0], [0.3, 0.1], [0.7, -0.1], [1, 0]]
    t = IsohedralTiling(41, edge_curves={0: curve, 1: curve}, edge_curve_modes={0: "linear"})
    assert t.edge_curve_modes == {0: "linear", 1: "spline"}
    back = IsohedralTiling.from_dict(t.to_dict())
    assert back.edge_curve_modes == {0: "linear", 1: "spline"}
    assert not np.array_equal(back.prototile(), back.with_edge_curves(back.edge_curves).prototile())


def test_format():
    t = IsohedralTiling(84, edge_curves={0: [[0, 0], [0.25, 0.1], [0.5, 0], [0.75, -0.1], [1, 0]]})
    data = t.to_dict()
    assert list(data) == ["version", "tiling_type", "parameters", "edge_curves", "edge_curve_modes"]
    assert data["edge_curves"]["0"][1] == [0.25, 0.1]
    assert "frame" not in data
    assert "frame" in t.scaled(2.0).to_dict()


def test_keys_are_ordered_by_shape_id():
    curve = [[0, 0], [0.5, 0.1], [1, 0]]
    t = IsohedralTiling(4, edge_curves={1: curve, 0: [[0, 0], [0.5, 0], [1, 0]]})
    assert list(t.to_dict()["edge_curves"]) == ["0", "1"]


def test_save_and_load(tmp_path):
    t = random_valid_curved_tiling(28, 0, frame=random_frame(np.random.default_rng(0), True))
    assert t is not None
    path = tmp_path / "tiling.json"
    t.save(path)
    text = path.read_text(encoding="utf-8")
    assert text.endswith("}\n")
    assert json.loads(text) == t.to_dict()
    same(IsohedralTiling.load(path), t)
    t.save(tmp_path / "again.json")
    assert (tmp_path / "again.json").read_text(encoding="utf-8") == text  # deterministic
    from isotile import io

    same(io.load(path), t)
    io.save(t, tmp_path / "third.json")
    assert (tmp_path / "third.json").read_text(encoding="utf-8") == text


def test_reads_version_1_0_and_foreign_keys():
    data = {
        "version": "1.0",
        "tiling_type": 41,
        "parameters": [0.3, 0.6],
        "edge_curves": {"0": [[0, 0], [0.5, 0.1], [1, 0]]},
        "description": "from an editor",
        "metadata": {"author": "x"},
    }
    t = IsohedralTiling.from_dict(data)
    assert t.edge_curve_modes == {0: "spline"}
    np.testing.assert_array_equal(t.parameters, [0.3, 0.6])
    assert IsohedralTiling.from_dict({"tiling_type": 7}).type.number == 7  # no version, no parameters
    assert IsohedralTiling.from_dict({"tiling_type": 41, "parameters": None, "edge_curves": {}}).edge_curves == {}


@pytest.mark.parametrize(
    ("data", "match"),
    [
        ({"version": "2.0", "tiling_type": 7}, "version"),
        ({"version": "1.1"}, "tiling_type"),
        ({"tiling_type": 7, "parameters": [0.1]}, "parameters"),
        ({"tiling_type": 999}, "type"),
        ({"tiling_type": 41, "edge_curves": {"x": [[0, 0], [1, 0]]}}, "shape ids"),
        ({"tiling_type": 41, "edge_curve_modes": {"0": "spline"}}, "without a curve"),
        ({"tiling_type": 41, "edge_curves": {"0": [[0, 0], [1, 0.5]]}}, "from"),
        ({"tiling_type": 41, "frame": [[1, 0, 0], [0, 0, 0]]}, "singular"),
    ],
)
def test_invalid_data_raises_value_error(data, match):
    with pytest.raises(ValueError, match=match):
        IsohedralTiling.from_dict(data)


def test_no_pickle_and_plain_types():
    t = random_valid_curved_tiling(28, 0)
    assert t is not None

    def walk(x):
        if isinstance(x, dict):
            assert all(isinstance(k, str) for k in x)
            for v in x.values():
                walk(v)
        elif isinstance(x, list):
            for v in x:
                walk(v)
        else:
            assert type(x) in (int, float, str), type(x)

    walk(t.to_dict())
    assert curvable_ids(28)
