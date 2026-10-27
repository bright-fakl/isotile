"""find_types, info and describe against values generated once from carto-flow's tactile-based implementation."""

import json
from pathlib import Path

import numpy as np
import pytest

from isotile import IsohedralTiling, find_types, get_type, tiling_types

EXPECTED = json.loads((Path(__file__).parent / "data" / "carto_flow_expected.json").read_text())
INFO_KEYS = {
    "num_parameters",
    "num_vertices",
    "num_aspects",
    "num_edge_shapes",
    "edge_shapes",
    "default_parameters",
    "edges",
    "has_reflections",
    "customizable_shape_ids",
    "parameters",
}


@pytest.mark.parametrize("case", EXPECTED["find_types"], ids=lambda c: str(c["filters"]))
def test_find_types_equals_carto_flow(case):
    assert find_types(**case["filters"]) == case["result"]


def test_find_types_accepts_any_collection_of_edge_types():
    expected = find_types(edge_types={"J", "S"})
    assert expected
    assert find_types(edge_types=["S", "J"]) == expected
    assert find_types(edge_types=frozenset("JS")) == expected
    assert find_types(edge_types=("J", "S", "J")) == expected


def test_find_types_without_filters_returns_all_types_sorted():
    assert find_types() == sorted(tiling_types())


@pytest.mark.parametrize("number", [int(n) for n in EXPECTED["info"]])
def test_info_equals_carto_flow(number):
    expected = EXPECTED["info"][str(number)]
    info = get_type(number).info()
    assert set(info) == INFO_KEYS == set(expected)
    for key in INFO_KEYS - {"parameters", "default_parameters"}:
        assert info[key] == expected[key], key
    np.testing.assert_allclose(info["default_parameters"], expected["default_parameters"], atol=1e-12)
    assert len(info["parameters"]) == len(expected["parameters"])
    for got, want in zip(info["parameters"], expected["parameters"], strict=True):
        assert set(got) == set(want)
        assert got["index"] == want["index"]
        assert got["default"] == pytest.approx(want["default"], abs=1e-12)
        assert got["description"] == want["description"]
        assert [a["vertex"] for a in got["affected_vertices"]] == [a["vertex"] for a in want["affected_vertices"]]
        for a, b in zip(got["affected_vertices"], want["affected_vertices"], strict=True):
            assert a["dx_dp"] == pytest.approx(b["dx_dp"], abs=1.1e-4)
            assert a["dy_dp"] == pytest.approx(b["dy_dp"], abs=1.1e-4)


@pytest.mark.parametrize("number", [int(n) for n in EXPECTED["describe"]])
def test_describe_equals_carto_flow(number):
    assert get_type(number).describe() == EXPECTED["describe"][str(number)]


def test_tiling_describe_shows_the_current_values():
    tiling = IsohedralTiling(7, [0.62, 0.21])
    text = tiling.describe()
    assert "[0] value=0.62 (default 0.6)" in text
    assert "[1] value=0.21 (default 0.1964)" in text
    assert text.splitlines()[0] == get_type(7).describe().splitlines()[0]
    assert IsohedralTiling(84).describe().count("value=") == 2
    assert "value=" not in get_type(7).describe()
    assert "Parameters: none" in IsohedralTiling(76).describe()


def test_tiling_info_describes_the_instance():
    tiling = IsohedralTiling(7, [0.62, 0.21], frame=[[2, 0, 1], [0, 2, 1]])
    info, type_info = tiling.info(), get_type(7).info()
    assert set(info) == set(type_info) | {"parameter_values"}
    assert info["parameter_values"] == [0.62, 0.21]
    assert info["default_parameters"] == type_info["default_parameters"] == [0.6, pytest.approx(0.196416770201)]
    assert [p["value"] for p in info["parameters"]] == [0.62, 0.21]
    assert [p["default"] for p in info["parameters"]] == [p["default"] for p in type_info["parameters"]]
    for key in set(type_info) - {"parameters"}:
        assert info[key] == type_info[key]
    assert "parameter_values" not in type_info
    assert all("value" not in p for p in type_info["parameters"])
    assert IsohedralTiling(7).info()["parameter_values"] == IsohedralTiling(7).info()["default_parameters"]


def test_affected_vertices_follow_the_coefficients_exactly(number):
    t = get_type(number)
    for p in t.info()["parameters"]:
        column = t.vertex_coeffs[:, :, p["index"]]
        moved = [v for v in range(t.n_vertices) if np.any(np.abs(column[v]) > 1e-6)]
        assert [a["vertex"] for a in p["affected_vertices"]] == moved


def test_type_properties(number):
    t = get_type(number)
    assert t.n_vertices in (3, 4, 5, 6)
    assert t.safe_ranges.shape == (t.n_params, 2)
    assert t.n_edge_shapes == len(set(t.edge_shape_ids.tolist()))
