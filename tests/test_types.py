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
    assert IsohedralTiling(number).describe() == EXPECTED["describe"][str(number)]


def test_tiling_info_is_the_type_info():
    assert IsohedralTiling(7).info() == get_type(7).info()


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
