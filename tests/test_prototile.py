"""check_prototile, the constructor's validation, safe ranges and the inscribed radius."""

import numpy as np
import pytest

from isotile import IsohedralTiling, PrototileReport, check_prototile, get_type, tiling_types
from isotile._build_data import critical_values
from isotile.prototile import inscribed_circle

PARAMETRIC = [n for n in tiling_types() if get_type(n).n_params]


def find_invalid(number: int, seed: int = 0) -> np.ndarray:
    """Parameters at which the prototile of the type is invalid.

    Random search around the defaults; failing that (a triangle is degenerate only at isolated values), the
    exact roots of the orientation determinants along each parameter axis.
    """
    t = get_type(number)
    rng = np.random.default_rng(seed + number)
    for spread in (0.5, 1.0, 2.0, 4.0):
        for _ in range(500):
            p = t.default_params + rng.uniform(-spread, spread, t.n_params)
            if not check_prototile(number, p).valid:
                return p
    for index in range(t.n_params):
        for root in critical_values(t, index):
            p = t.default_params.copy()
            p[index] = root
            if not check_prototile(number, p).valid:
                return p
    raise AssertionError(f"no invalid parameters found for IH{number}")


def test_defaults_are_valid(number):
    report = check_prototile(number)
    assert report == PrototileReport(True, (), "valid")
    IsohedralTiling(number)


@pytest.mark.parametrize("number", PARAMETRIC)
def test_invalid_parameters_are_reported_and_rejected(number):
    p = find_invalid(number)
    report = check_prototile(number, p)
    assert not report.valid
    assert report.bad_edges
    assert report.message != "valid"
    assert all(0 <= e < get_type(number).n_vertices for e in report.bad_edges)
    assert list(report.bad_edges) == sorted(set(report.bad_edges))
    with pytest.raises(ValueError, match="invalid prototile"):
        IsohedralTiling(number, p)
    tiling = IsohedralTiling(number)
    before = tiling.parameters
    with pytest.raises(ValueError, match="invalid prototile"):
        tiling.parameters = p
    np.testing.assert_array_equal(tiling.parameters, before)


@pytest.mark.parametrize("number", PARAMETRIC)
def test_check_agrees_with_constructor_on_random_parameters(number):
    t = get_type(number)
    rng = np.random.default_rng(number)
    for _ in range(30):
        p = t.default_params + rng.uniform(-1.5, 1.5, t.n_params)
        valid = check_prototile(number, p).valid
        try:
            IsohedralTiling(number, p)
        except ValueError:
            assert not valid
        else:
            assert valid


def test_crossing_edges_are_named():
    # IH7 at (0.25, -1.0) was rejected as self-intersecting by the prototype
    report = check_prototile(7, [0.25, -1.0])
    assert not report.valid
    assert "cross" in report.message or "touch" in report.message or "fold" in report.message
    assert len(report.bad_edges) >= 2


def test_non_finite_and_wrongly_shaped_parameters():
    assert not check_prototile(7, [np.nan, 0.5]).valid
    with pytest.raises(ValueError, match="parameters"):
        check_prototile(7, [0.1])
    with pytest.raises(ValueError, match="Invalid isohedral tiling type"):
        check_prototile(19)


def test_check_does_not_construct_a_tiling(monkeypatch):
    def fail(*args, **kwargs):
        raise AssertionError("constructed a tiling")

    monkeypatch.setattr(IsohedralTiling, "__init__", fail)
    assert check_prototile(7).valid


def test_degenerate_and_folded_polygons():
    from isotile.prototile import check_polygon

    zero_edge = np.array([[0.0, 0], [1, 0], [1, 0], [1, 1], [0, 1]])
    assert 1 in check_polygon(zero_edge).bad_edges
    folded = np.array([[0.0, 0], [2, 0], [1, 0], [1, 1]])
    report = check_polygon(folded)
    assert not report.valid
    assert {0, 1} <= set(report.bad_edges)
    flat = np.array([[0.0, 0], [1, 0], [2, 0]])
    assert not check_polygon(flat).valid
    bow_tie = np.array([[0.0, 0], [1, 1], [1, 0], [0, 1]])
    assert check_polygon(bow_tie).bad_edges == (0, 2)


@pytest.mark.parametrize("number", PARAMETRIC)
def test_safe_ranges_contain_the_defaults_and_have_valid_ends(number):
    t = get_type(number)
    assert t.safe_ranges.shape == (t.n_params, 2)
    for i in range(t.n_params):
        low, high = t.safe_ranges[i]
        assert low <= t.default_params[i] <= high
        for value in (low, high, (low + t.default_params[i]) / 2, (high + t.default_params[i]) / 2):
            p = t.default_params.copy()
            p[i] = value
            assert check_prototile(number, p).valid, (i, value)
        # a finite end is a validity limit: the offset is rounded down to a multiple of 1e-6, so a critical value
        # at which the prototile is invalid (or invalid just beyond it) lies within 1e-6 beyond the end
        for end, sign in ((low, -1.0), (high, 1.0)):
            if abs(end - t.default_params[i]) < 4.99:
                beyond = [r for r in critical_values(t, i) if 0 < sign * (r - end) <= 1.0001e-6]
                assert beyond, (i, end)
                invalid = []
                for r in beyond:
                    for value in (r, r + sign * 1e-7):
                        p = t.default_params.copy()
                        p[i] = value
                        invalid.append(not check_prototile(number, p).valid)
                assert any(invalid), (i, end)


def test_many_types_have_finite_ends():
    finite = [
        n for n in PARAMETRIC if np.any(np.abs(get_type(n).safe_ranges - get_type(n).default_params[:, None]) < 4.99)
    ]
    assert len(finite) >= 20


def test_zero_parameter_types_have_empty_ranges():
    for number in tiling_types():
        if get_type(number).n_params == 0:
            assert get_type(number).safe_ranges.shape == (0, 2)


def test_inscribed_circle_of_known_polygons():
    square = np.array([[0.0, 0], [2, 0], [2, 2], [0, 2]])
    center, radius = inscribed_circle(square)
    assert radius == pytest.approx(1.0, abs=1e-9)
    np.testing.assert_allclose(center, [1, 1], atol=1e-8)
    triangle = np.array([[0.0, 0], [4, 0], [0, 3]])  # 3-4-5: inradius = area / semiperimeter = 6 / 6
    assert inscribed_circle(triangle)[1] == pytest.approx(1.0, abs=1e-9)
    l_shape = np.array([[0.0, 0], [3, 0], [3, 1], [1, 1], [1, 3], [0, 3]])  # arms of width 1
    assert inscribed_circle(l_shape)[1] == pytest.approx(2 - np.sqrt(2), abs=1e-6)
    rectangle = np.array([[0.0, 0], [10, 0], [10, 1], [0, 1]])
    assert inscribed_circle(rectangle)[1] == pytest.approx(0.5, abs=1e-9)


def test_inscribed_radius_of_tilings(number):
    tiling = IsohedralTiling(number)
    radius = tiling.inscribed_radius
    _center, same = inscribed_circle(tiling.vertices)
    assert radius == same > 0
    assert radius <= np.sqrt(tiling.area / np.pi) * (1 + 1e-9)  # a circle inside a shape has at most its area
    assert IsohedralTiling.square().inscribed_radius == pytest.approx(0.5)
    assert IsohedralTiling.hexagon().inscribed_radius == pytest.approx(np.sqrt(3) / 2)


def test_inscribed_radius_is_not_beaten_by_random_circles(number):
    tiling = IsohedralTiling(number)
    v = tiling.vertices
    rng = np.random.default_rng(number)
    pts = rng.uniform(v.min(axis=0), v.max(axis=0), (2000, 2))
    from isotile.prototile import _signed_distance

    assert _signed_distance(v, pts).max() <= tiling.inscribed_radius + 1e-9


@pytest.mark.parametrize("number", PARAMETRIC)
def test_no_invalid_value_inside_a_safe_range(number):
    t = get_type(number)
    rng = np.random.default_rng(number)
    for i in range(t.n_params):
        low, high = t.safe_ranges[i]
        values = [r for r in critical_values(t, i) if low <= r <= high]
        values += rng.uniform(low, high, 100).tolist()
        for value in values:
            p = t.default_params.copy()
            p[i] = value
            assert check_prototile(number, p).valid, (i, value)
