"""The certified validity test: clearance, folds, loops, independence of the sampling, agreement with other tests."""

import inspect

import numpy as np
import pytest

from isotile import (
    IsohedralTiling,
    check_prototile,
    edge_curves_from_preset,
    get_type,
    list_edge_curve_presets,
    tiling_types,
)
from isotile.clearance import bounding_box, check_pieces, signed_area
from isotile.curves import CLEARANCE, check_curves, native_edge_transforms, normalize_curves, outline

from .conftest import curvable_ids, random_curves, random_tilings
from .sampled_check import check_outline


def segments(points: list[tuple[float, float]]) -> np.ndarray:
    """Closed chain of straight pieces through the points."""
    p = np.asarray(points, dtype=float)
    a, b = p, np.roll(p, -1, axis=0)
    return np.stack([a, (2 * a + b) / 3, (a + 2 * b) / 3, b], axis=1)


def verdict(pieces: np.ndarray) -> tuple[bool, str]:
    low, high = bounding_box(pieces)
    diameter = float((high - low).max())
    _bad, problems = check_pieces(pieces, np.arange(len(pieces)), len(pieces), CLEARANCE * diameter)
    return not problems and signed_area(pieces) > 0, "; ".join(problems)


def slot(width: float) -> np.ndarray:
    """A square with a slot of the given width cut from the top, walls 0.8 long."""
    return segments([
        (0, 0),
        (1, 0),
        (1, 1),
        (0.5 + width / 2, 1),
        (0.5 + width / 2, 0.2),
        (0.5 - width / 2, 0.2),
        (0.5 - width / 2, 1),
        (0, 1),
    ])


def band(thickness: float) -> np.ndarray:
    """A thin curved band: two cubic arcs thickness apart, closed by two short straight pieces."""
    arc = np.array([[0, 0], [1 / 3, 0.3], [2 / 3, 0.3], [1, 0]], dtype=float)
    lower, upper = arc, arc + np.array([0.0, thickness])
    right = segments([(1, 0), (1, thickness)])[:1]
    left = segments([(0, thickness), (0, 0)])[:1]
    return np.concatenate([lower[None], right, upper[::-1][None], left])


@pytest.mark.parametrize("make", [slot, band])
def test_clearance_is_one_millionth_of_the_diameter(make):
    diameter = 1.0  # the larger side of both shapes
    assert verdict(make(0.5 * CLEARANCE * diameter))[0] is False
    assert verdict(make(0.9 * CLEARANCE * diameter))[0] is False
    assert verdict(make(1.5 * CLEARANCE * diameter))[0] is True
    assert verdict(make(2.0 * CLEARANCE * diameter))[0] is True
    assert verdict(make(1e-3))[0] is True


def test_clearance_scales_with_the_diameter():
    for scale in (1e-3, 1.0, 1e4):
        assert verdict(slot(0.5e-6) * scale)[0] is False
        assert verdict(slot(2e-6) * scale)[0] is True


def test_touching_and_crossing_pieces_are_reported():
    touching = segments([(0, 0), (1, 0), (1, 1), (0.5, 1), (0.5, 0.2), (0.5, 0.2 + 1e-12), (0.5, 1), (0, 1)])
    assert not verdict(touching)[0]
    bow = segments([(0, 0), (1, 1), (1, 0), (0, 1)])
    ok, message = verdict(bow)
    assert not ok and "cross" in message


@pytest.mark.parametrize(("number", "shape_id", "sign"), [(5, 2, 1), (28, 1, -1), (28, 2, 1)])
def test_threshold_is_the_clearance_between_edges(number, shape_id, sign):
    """A spike on one shape id grows until it nears another edge; the threshold is 1e-6 times the diameter.

    The distance is measured independently with shapely on the exact polylines (linear mode).
    """
    shapely = pytest.importorskip("shapely")
    t = get_type(number)
    modes = {shape_id: "linear"}

    def curve(length: float) -> dict[int, list[tuple[float, float]]]:
        w = 0.02
        return {shape_id: [(0, 0), (0.5 - w, 0), (0.5, sign * length), (0.5 + w, 0), (1, 0)]}

    def valid(length: float) -> bool:
        return check_prototile(number, edge_curves=curve(length), edge_curve_modes=modes).valid

    low, high = 0.0, 3.0
    assert valid(low) and not valid(high)
    for _ in range(60):
        mid = (low + high) / 2
        low, high = (mid, high) if valid(mid) else (low, mid)

    ratios = []
    for length in (low, high):
        points, starts = outline(
            t, native_edge_transforms(t, t.default_params), normalize_curves(t, curve(length), modes)
        )
        closed = np.vstack([points, points[:1]])
        n = t.n_vertices
        lines = [shapely.LineString(closed[starts[i] : starts[i + 1] + 1]) for i in range(n)]
        diameter = float(np.ptp(points, axis=0).max())
        gaps = [lines[i].distance(lines[j]) for i in range(n) for j in range(i + 2, n) if not (i == 0 and j == n - 1)]
        ratios.append(min(gaps) / (CLEARANCE * diameter))
    assert ratios == pytest.approx([1.0, 1.0], abs=1e-3)


def test_fold_back_at_a_join():
    curve = {0: [(0, 0), (0.5, 0.1), (1, 0.5), (1, 0)]}  # arrives at (1, 0) from above; the next edge goes up
    report = check_prototile(41, edge_curves=curve, edge_curve_modes={0: "linear"})
    assert not report.valid and "fold" in report.message
    assert 0 in report.bad_edges and 1 in report.bad_edges
    spline = check_prototile(41, edge_curves={0: [(0, 0), (0.5, 0.1), (1.0, 0.5), (1.0, 0.0)]})
    assert not spline.valid


def test_loop_and_cusp_within_one_piece():
    tail = [(0.5, -1.0)]
    loop = np.array([[0, 0], [1, 1], [0, 1], [1, 0]], dtype=float)  # a cubic with a loop
    cusp = np.array([[0, 0], [1, 0], [0, 0], [1, 0]], dtype=float)  # goes forward, back and forward on a line
    for piece in (loop, cusp):
        chain = np.concatenate([piece[None], segments([(1, 0), *tail, (0, 0)])[:2]])
        ok, message = verdict(chain)
        assert not ok and "fold" in message


def test_straight_pieces_and_zero_length():
    assert verdict(segments([(0, 0), (1, 0), (1, 1), (0, 1)]))[0]
    ok, message = verdict(segments([(0, 0), (1, 0), (1, 0), (1, 1), (0, 1)]))
    assert not ok and "zero-length" in message
    assert not verdict(segments([(0, 0), (1, 0), (1, 1), (0, 1)])[::-1, ::-1])[0]  # clockwise: negative area


def test_thin_loop_that_the_sample_spacing_misses():
    """A loop narrower than the sample spacing (0.01) is invisible to the polyline test and found by the new one."""
    t = get_type(41)
    s = 1e-3
    curves = {0: [(0, 0), (0.5 - s, 0), (0.5 + s, s), (0.5 - s, s), (0.5 + s, 0), (1, 0)]}
    points, starts = outline(t, native_edge_transforms(t, t.default_params), normalize_curves(t, curves, None))
    assert check_outline(points, starts)[0] == ()  # the sampled test accepts it
    report = check_prototile(41, edge_curves=curves)
    assert not report.valid and report.bad_edges == (0, 2)
    with pytest.raises(ValueError, match="invalid prototile"):
        IsohedralTiling(41, edge_curves=curves)


def test_validity_does_not_depend_on_the_sampling():
    assert "n_points" not in inspect.signature(check_prototile).parameters
    assert "n_points" not in inspect.signature(check_curves).parameters
    curve = {0: [(0, 0), (0.3, 0.2), (0.7, -0.2), (1, 0)]}
    t = IsohedralTiling(41, edge_curves=curve)
    before = check_prototile(41, edge_curves=curve)
    sizes = {n: len(t.prototile(n_points=n)) for n in (20, 100, 1000)}
    assert len(set(sizes.values())) == 3  # the output depends on n_points
    assert check_prototile(41, edge_curves=curve) == before and before.valid
    thin = {0: [(0, 0), (0.499, 0), (0.501, 1e-3), (0.499, 1e-3), (0.501, 0), (1, 0)]}
    for n in (20, 100, 1000):
        IsohedralTiling(41, edge_curves=curve).prototile(n_points=n)  # drawing never validates
    assert not check_prototile(41, edge_curves=thin).valid


# -- agreement with the sampled test and GEOS ----------------------------------------------------------------------


def draws(seed: int, amplitudes=(0.1, 0.4, 1.0, 2.0)):
    rng = np.random.default_rng(seed)
    for number in tiling_types():
        if not curvable_ids(number):
            continue
        for amplitude in amplitudes:
            base = random_tilings(number, 1, 0.1, seed=seed)[0]
            yield number, base.parameters, random_curves(number, rng, amplitude)


def test_agreement_with_the_sampled_test_and_geos():
    """The certified test rejects what the polyline test rejects, and accepts what GEOS and the polyline accept.

    Allowed differences: a polyline that is simple although the curve folds back (a single piece that turns by
    more than 180 degrees), and crossings or near-touches finer than the sample spacing.
    """
    shapely = pytest.importorskip("shapely")
    accepted_by_sampling_only = []
    rejected_by_sampling_only = []
    n = 0
    for number, parameters, curves in draws(11):
        t = get_type(number)
        certified = check_prototile(number, parameters, edge_curves=curves).valid
        points, starts = outline(t, native_edge_transforms(t, parameters), normalize_curves(t, curves, None))
        sampled = check_outline(points, starts)[0] == ()
        geos = shapely.Polygon(points).is_valid and shapely.Polygon(points).area > 0
        assert sampled == geos
        n += 1
        if sampled and not certified:
            accepted_by_sampling_only.append(number)
        if certified and not sampled:
            rejected_by_sampling_only.append(number)
    assert n > 200
    assert rejected_by_sampling_only == []
    assert len(accepted_by_sampling_only) <= 0.02 * n


def test_certified_acceptance_tiles_the_plane_once():
    """Every draw that the certified test accepts covers a region exactly once (the sufficiency of the rule)."""
    shapely = pytest.importorskip("shapely")
    accepted = 0
    for number, parameters, curves in draws(12, (0.2, 1.5)):
        if not check_prototile(number, parameters, edge_curves=curves).valid:
            continue
        tiling = IsohedralTiling(number, parameters, edge_curves=curves)
        s, c = np.sqrt(tiling.area), tiling.centroid
        box = (c[0] - 1.5 * s, c[1] - 1.5 * s, c[0] + 1.5 * s, c[1] + 1.5 * s)
        tiles = tiling.cover((box[0] - 4 * s, box[1] - 4 * s, box[2] + 4 * s, box[3] + 4 * s))
        polygons = shapely.polygons(tiles.outlines())
        rng = np.random.default_rng(number)
        points = shapely.points(rng.uniform([box[0], box[1]], [box[2], box[3]], (300, 2)))
        index, _ = shapely.STRtree(polygons).query(points, predicate="within")
        counts = np.bincount(index, minlength=300)
        assert counts.min() == 1 and counts.max() == 1, number
        accepted += 1
    assert accepted > 50


# -- presets ------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("name", list(list_edge_curve_presets()))
def test_curve_presets_at_two_amplitudes(name):
    """Amplitude 0.04 is valid for all 81 types; the larger amplitude 0.1 is invalid only for the listed types."""
    invalid_at_01 = {"bump": {39}, "dent": {39}, "wave": {36, 38, 86}, "s_curve": {36, 38, 86}, "straight": set()}[name]
    for number in tiling_types():
        assert check_prototile(number, edge_curves=edge_curves_from_preset(number, name, 0.04)).valid
        valid = check_prototile(number, edge_curves=edge_curves_from_preset(number, name, 0.1)).valid
        assert valid == (number not in invalid_at_01), (number, name)
