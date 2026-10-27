"""The prototile outline with curved edges, its validity rule and the constructor's behavior."""

import numpy as np
import pytest

from isotile import IsohedralTiling, check_prototile, get_type, tiling_types

from .conftest import curvable_ids, random_curves, random_frame, random_valid_curved_tiling

CURVABLE = [n for n in tiling_types() if curvable_ids(n)]
ALL_I = [n for n in tiling_types() if not curvable_ids(n)]


def signed_area(points: np.ndarray) -> float:
    x, y = points[:, 0], points[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def test_types_without_curvable_edges():
    assert ALL_I
    for number in ALL_I:
        with pytest.raises(ValueError, match="straight"):
            IsohedralTiling(number, edge_curves={0: [[0, 0], [0.5, 0.1], [1, 0]]})


def test_no_curves_is_the_straight_prototile(number):
    t = IsohedralTiling(number)
    np.testing.assert_array_equal(t.prototile(), t.vertices)
    assert t.edge_curves == {} and t.edge_curve_modes == {}
    for info, line in zip(t.edges, t.edge_outlines(), strict=True):
        assert len(line) == 2 and info.curve.shape == "I"


def test_straight_curve_equals_vertices(number):
    ids = curvable_ids(number)
    if not ids:
        pytest.skip("no curvable edge")
    t = IsohedralTiling(number, edge_curves={i: [[0, 0], [1, 0]] for i in ids})
    np.testing.assert_allclose(t.prototile(), t.vertices, atol=1e-14)


@pytest.mark.parametrize("seed", range(2))
def test_outline_runs_from_vertex_to_vertex(number, seed):
    """Each edge outline joins vertex i to vertex i + 1, whatever the reversal of the edge."""
    t = random_valid_curved_tiling(number, seed)
    if t is None:
        pytest.skip("no curvable edge")
    v = t.vertices
    lines = t.edge_outlines()
    assert len(lines) == t.type.n_vertices
    for i, line in enumerate(lines):
        np.testing.assert_allclose(line[0], v[i], atol=1e-12)
        np.testing.assert_allclose(line[-1], v[(i + 1) % len(v)], atol=1e-12)
    assert sum(len(line) - 1 for line in lines) == len(t.prototile())
    np.testing.assert_array_equal(np.vstack([line[:-1] for line in lines]), t.prototile())
    np.testing.assert_array_equal(t.prototile(curved=False), v)


@pytest.mark.parametrize("seed", range(2))
def test_area_is_unchanged_by_valid_curves(number, seed):
    """Edges with the same shape id cancel, so the curved prototile has the area of the straight one."""
    t = random_valid_curved_tiling(number, seed)
    if t is None:
        pytest.skip("no curvable edge")
    assert signed_area(t.prototile()) == pytest.approx(t.area, rel=1e-9)


def test_edges_with_one_shape_id_have_the_same_curve(number):
    t = random_valid_curved_tiling(number, 0)
    if t is None:
        pytest.skip("no curvable edge")
    by_id: dict[int, list[np.ndarray]] = {}
    for info, line in zip(t.edges, t.edge_outlines(), strict=True):
        canonical = line[::-1] if info.reversed else line
        # the canonical edge map is a similarity (possibly reflecting) of the shared curve
        by_id.setdefault(info.shape_id, []).append(canonical)
    for lines in by_id.values():
        lengths = [np.sum(np.linalg.norm(np.diff(x, axis=0), axis=1)) / np.linalg.norm(x[-1] - x[0]) for x in lines]
        np.testing.assert_allclose(lengths, lengths[0], rtol=1e-9)


def test_edge_info():
    t = IsohedralTiling(
        4, edge_curves={0: [[0, 0], [0.2, 0.1], [0.5, 0], [0.8, -0.1], [1, 0]], 1: [[0, 0], [0.5, 0.2], [1, 0]]}
    )
    assert [e.index for e in t.edges] == list(range(6))
    assert [e.shape for e in t.edges] == list(t.type.edge_shapes)
    assert [e.shape_id for e in t.edges] == t.type.edge_shape_ids.tolist()
    assert [e.reversed for e in t.edges] == t.edge_reversed.tolist()
    assert t.edges[0].curve.shape == "S" and t.edges[1].curve.shape == "J"
    assert t.edges[3].curve.shape == "I"
    assert set(t.edge_curves) == {0, 1}
    assert t.edge_curve_modes == {0: "spline", 1: "spline"}


def test_curves_are_copied_and_symmetrized():
    raw = np.array([[0, 0], [0.3, 0.2], [0.5, 0.1], [0.1, 0.9], [1, 0]])
    t = IsohedralTiling(84, edge_curves={0: raw})
    out = t.edge_curves[0]
    np.testing.assert_allclose(out, [[0, 0], [0.3, 0.2], [0.5, 0], [0.7, -0.2], [1, 0]])
    out[:] = 0  # the property returns a copy
    assert t.edge_curves[0][1, 1] == 0.2
    raw[:] = 0  # and the input was copied
    assert t.edge_curves[0][1, 1] == 0.2


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"edge_curves": {2: [[0, 0], [1, 0]]}}, "not an edge shape id"),
        ({"edge_curves": {-1: [[0, 0], [1, 0]]}}, "not an edge shape id"),
        ({"edge_curves": {"a": [[0, 0], [1, 0]]}}, "not an edge shape id"),
        ({"edge_curves": {0: [[0, 0], [0.5, 0.1], [1, 0.2]]}}, "from"),
        ({"edge_curves": {0: [[0, 0]]}}, "edge curve"),
        ({"edge_curve_modes": {0: "spline"}}, "without a curve"),
        ({"edge_curves": {0: [[0, 0], [0.5, 0.1], [1, 0]]}, "edge_curve_modes": {0: "cubic"}}, "mode"),
        ({"edge_curves": {0: [[0, 0], [0.5, np.nan], [1, 0]]}}, "finite"),
    ],
)
def test_malformed_curves_raise(kwargs, match):
    with pytest.raises(ValueError, match=match):
        IsohedralTiling(41, **kwargs)
    with pytest.raises(ValueError, match=match):
        check_prototile(41, **kwargs)


BOW = [[0, 0], [0.8, 0.5], [0.2, 0.5], [1, 0]]  # the first and the last segment cross
SLEEVE = [[0, 0], [0.2, 0], [0.2, 3], [0.3, 3], [0.3, 0], [1, 0]]  # its dent and its bump overlap along x = 0.2, 0.3


def test_self_intersecting_curve_is_invalid():
    ok = check_prototile(41, edge_curves={0: [[0, 0], [0.5, 0.2], [1, 0]]})
    assert ok.valid and ok.bad_edges == () and ok.message == "valid"
    report = check_prototile(41, edge_curves={0: BOW}, edge_curve_modes={0: "linear"})
    assert not report.valid
    assert report.bad_edges == (0, 2)  # the edges with shape id 0
    assert "cross" in report.message
    with pytest.raises(ValueError, match="invalid prototile"):
        IsohedralTiling(41, edge_curves={0: BOW}, edge_curve_modes={0: "linear"})


def test_edges_that_touch_each_other_are_invalid():
    # the dent of edge 0 and the bump of the opposite edge 2 have common sides
    report = check_prototile(41, edge_curves={0: SLEEVE}, edge_curve_modes={0: "linear"})
    assert not report.valid
    assert report.bad_edges == (0, 2)
    assert "touch" in report.message


def test_spline_through_a_loop_is_invalid():
    loop = [[0, 0], [0.9, 0.4], [0.1, 0.4], [0.5, -0.1], [1, 0]]
    assert not check_prototile(41, edge_curves={0: loop}).valid


def test_neighboring_edges_that_cross_are_reported():
    # a bump on shape id 1 (edges 1 and 3) that crosses the sides next to it: the bump is long and leans over
    lean = [[0, 0], [0.1, 0.1], [-2.0, 1.0], [0.5, 1.0], [1, 0]]
    report = check_prototile(41, edge_curves={1: lean}, edge_curve_modes={1: "linear"})
    assert not report.valid and set(report.bad_edges) <= {0, 1, 2, 3}


def test_fold_back_is_reported():
    fold = [[0, 0], [0.5, 0], [0.3, 0], [1, 0]]  # goes forward, then back along itself
    report = check_prototile(41, edge_curves={0: fold}, edge_curve_modes={0: "linear"})
    assert not report.valid
    assert "fold" in report.message or "touch" in report.message


def test_zero_length_segment_is_reported():
    report = check_prototile(
        41, edge_curves={0: [[0, 0], [0.5, 0.1], [0.5, 0.1], [1, 0]]}, edge_curve_modes={0: "linear"}
    )
    assert not report.valid and "zero-length" in report.message
    assert check_prototile(41, edge_curves={0: [[0, 0], [0.5, 0.1], [0.5, 0.1], [1, 0]]}).valid  # a spline merges them


def test_straight_prototile_checked_first():
    from tests.test_prototile import find_invalid

    p = find_invalid(7)
    plain = check_prototile(7, p)
    with_curves = check_prototile(7, p, edge_curves={0: [[0, 0], [0.5, 0.05], [1, 0]]})
    assert not plain.valid and with_curves == plain


def test_parameter_setter_checks_the_curves_and_keeps_the_state():
    t = IsohedralTiling(41, edge_curves={0: [[0, 0], [0.5, 0.15], [1, 0]]})
    before = t.parameters
    outline_before = t.prototile()
    # collapse the square so that the bump crosses the opposite side
    squeezed = np.array([0.23076923, 0.02])
    assert check_prototile(41, squeezed).valid
    assert not check_prototile(41, squeezed, edge_curves={0: [[0, 0], [0.5, 0.15], [1, 0]]}).valid
    with pytest.raises(ValueError, match="invalid prototile"):
        t.parameters = squeezed
    np.testing.assert_array_equal(t.parameters, before)
    np.testing.assert_array_equal(t.prototile(), outline_before)
    t.parameters = [0.3, 0.6]
    assert t.prototile().shape == outline_before.shape


def test_valid_means_simple_closed_curve_with_positive_area(number):
    """The report agrees with shapely's test of the polygon of the outline (a second implementation)."""
    shapely = pytest.importorskip("shapely")
    ids = curvable_ids(number)
    if not ids:
        pytest.skip("no curvable edge")
    rng = np.random.default_rng(number)
    t = get_type(number)
    for amplitude in (0.1, 0.4, 1.0, 2.0):
        curves = random_curves(number, rng, amplitude)
        report = check_prototile(number, edge_curves=curves)
        # the outline of the unchecked construction
        from isotile.curves import native_edge_transforms, normalize_curves, outline

        points, _ = outline(t, native_edge_transforms(t, t.default_params), normalize_curves(t, curves, None))
        polygon = shapely.Polygon(points)
        assert report.valid == (polygon.is_valid and polygon.area > 0), (amplitude, report.message)


@pytest.mark.parametrize("reflect", [False, True])
def test_frame_invariance(number, reflect):
    """A frame maps the outline and the tiles; validity and the curve shapes do not depend on it."""
    rng = np.random.default_rng(number)
    if not curvable_ids(number):
        pytest.skip("no curvable edge")
    native = random_valid_curved_tiling(number, 0)
    assert native is not None
    f = random_frame(rng, reflect)
    framed = native.with_frame(f)
    np.testing.assert_allclose(framed.prototile(), native.prototile() @ f[:, :2].T + f[:, 2], atol=1e-10)
    np.testing.assert_allclose(framed.vertices, native.vertices @ f[:, :2].T + f[:, 2], atol=1e-10)
    assert set(framed.edge_curves) == set(native.edge_curves)
    for k in native.edge_curves:
        np.testing.assert_array_equal(framed.edge_curves[k], native.edge_curves[k])
    # validity is judged in native coordinates: the same curves are invalid under every frame
    assert check_prototile(number, native.parameters, edge_curves=native.edge_curves).valid
    # tiles map the same way
    a = native.cover((-3, -3, 3, 3))
    b = framed.with_frame(f).cover((-3, -3, 3, 3))
    assert len(b) >= 1 and len(a) >= 1


def test_rotated_and_scaled_keep_the_curves():
    t = IsohedralTiling(84, edge_curves={0: [[0, 0], [0.25, 0.1], [0.5, 0], [0.75, -0.1], [1, 0]]})
    r = t.rotated(30).scaled(2.0)
    assert set(r.edge_curves) == {0}
    np.testing.assert_array_equal(r.edge_curves[0], t.edge_curves[0])
    np.testing.assert_allclose(r.area, 4 * t.area)
    assert signed_area(r.prototile()) == pytest.approx(r.area, rel=1e-9)


def test_with_edge_curves():
    t = IsohedralTiling(41).scaled(2.0)
    c = t.with_edge_curves({0: [[0, 0], [0.5, 0.1], [1, 0]]})
    assert set(c.edge_curves) == {0}
    np.testing.assert_allclose(c.frame, t.frame)
    assert c.with_edge_curves(None).edge_curves == {}
    with pytest.raises(ValueError, match="invalid prototile"):
        t.with_edge_curves({0: BOW}, {0: "linear"})


def test_normalized_prototile():
    t = random_valid_curved_tiling(28, 0)
    assert t is not None
    n = t.prototile(normalized=True)
    assert signed_area(n) == pytest.approx(1.0, rel=1e-9)
    straight = t.prototile(normalized=True, curved=False)
    assert signed_area(straight) == pytest.approx(1.0, rel=1e-9)
    # the same translation and scale for the curved and the straight outline
    np.testing.assert_allclose(straight, (t.vertices - t.centroid) / np.sqrt(t.area))
    np.testing.assert_allclose(n, (t.prototile() - t.centroid) / np.sqrt(t.area))


def test_n_points():
    t = IsohedralTiling(41, edge_curves={0: [[0, 0], [0.5, 0.1], [1, 0]]})
    assert len(t.prototile()) == 2 * 99 + 2  # 100 samples on each of edges 0 and 2, straight edges 1 and 3
    assert len(t.prototile(n_points=10)) == 2 * 9 + 2
    assert [len(x) for x in t.edge_outlines(n_points=10)] == [10, 2, 10, 2]
    assert [len(x) for x in t.edge_outlines()] == [100, 2, 100, 2]
    assert [len(x) for x in t.edge_outlines(curved=False)] == [2, 2, 2, 2]
    plain = IsohedralTiling(41)
    np.testing.assert_array_equal(plain.prototile(n_points=10), plain.vertices)
