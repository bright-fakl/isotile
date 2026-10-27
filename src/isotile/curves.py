"""The prototile with curved edges: its outline in the plane and its validity.

A curved prototile is valid when its outline, the closed curve that runs along the edge curves in order, is a simple
closed curve with positive area (and the straight prototile of the same parameters is valid). The edge shapes make the
copies of the prototile fit along every shared edge, so no further condition is needed.

The outline is a polyline: each curved edge is sampled with [`interpolate`][isotile.interpolate] and the test works on
these samples.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .edges import EdgeShape
from .prototile import PrototileReport
from .types import TilingType

Array = NDArray[np.float64]
IntArray = NDArray[np.int64]

_ORIENT_TOL = 1e-12  # relative to the squared extent of the outline
_LENGTH_TOL = 1e-12  # relative to the extent of the outline
_AREA_TOL = 1e-9  # relative to the squared extent of the outline


@dataclass(frozen=True, eq=False)
class EdgeInfo:
    """One edge of the prototile.

    Attributes
    ----------
    index : int
        Position of the edge: it joins vertex ``index`` to vertex ``index + 1``.
    shape_id : int
        Id of the edge shape; edges with the same id have the same curve.
    shape : {"I", "J", "U", "S"}
        Constraint on the curve of the edge.
    reversed : bool
        True if the edge is traversed against the direction of its shape's canonical edge.
    curve : EdgeShape
        The curve of the shape id on the canonical edge (the segment for an edge without a curve).
    """

    index: int
    shape_id: int
    shape: str
    reversed: bool
    curve: EdgeShape


def native_edge_transforms(tiling_type: TilingType, parameters: Array) -> Array:
    """(n_vertices, 2, 3) maps of the canonical edge onto each edge of the prototile, in native coordinates.

    The map of an edge takes ``(0, 0)`` to the vertex where the shape starts, which is the end vertex when the edge is
    reversed (the orientation flags of the type are included).
    """
    t = tiling_type
    v = t.vertex_coeffs @ np.append(parameters, 1.0)
    w = np.roll(v, -1, axis=0)
    seg = np.zeros((t.n_vertices, 2, 3))  # unit edge (0,0)-(1,0) -> v[i]-v[i+1]
    seg[:, 0] = np.stack([w[:, 0] - v[:, 0], v[:, 1] - w[:, 1], v[:, 0]], axis=1)
    seg[:, 1] = np.stack([w[:, 1] - v[:, 1], w[:, 0] - v[:, 0], v[:, 1]], axis=1)
    orient = _ORIENTS[2 * t.edge_orientations[:, 0] + t.edge_orientations[:, 1]]
    out: Array = np.empty_like(seg)
    out[:, :, :2] = seg[:, :, :2] @ orient[:, :, :2]
    out[:, :, 2] = np.einsum("nij,nj->ni", seg[:, :, :2], orient[:, :, 2]) + seg[:, :, 2]
    return out


# The four edge orientation matrices (identity, rotate, flip, rotate + flip), index 2 * flip + rotate.
_ORIENTS: Array = np.array(
    [
        [[1, 0, 0], [0, 1, 0]],
        [[-1, 0, 1], [0, -1, 0]],
        [[-1, 0, 1], [0, 1, 0]],
        [[1, 0, 0], [0, -1, 0]],
    ],
    dtype=float,
)

STRAIGHT = EdgeShape.from_points("I")


def normalize_curves(
    tiling_type: TilingType,
    edge_curves: Mapping[int, ArrayLike] | None,
    edge_curve_modes: Mapping[int, str] | None,
) -> dict[int, EdgeShape]:
    """The curve of each shape id that has one, with the symmetry of its shape enforced.

    Raises
    ------
    ValueError
        If an id is not a shape id of the type or belongs to an ``I`` edge, if a mode is given for an id without a
        curve, or if a curve or mode is invalid.
    """
    curves: dict[int, EdgeShape] = {}
    modes = {} if edge_curve_modes is None else dict(edge_curve_modes)
    n_ids = len(tiling_type.shape_types)
    for key in [*(edge_curves or {}), *modes]:
        if not isinstance(key, (int, np.integer)) or isinstance(key, bool) or not 0 <= key < n_ids:
            raise ValueError(f"IH{tiling_type.number}: {key!r} is not an edge shape id (0 to {n_ids - 1})")
    for key, points in (edge_curves or {}).items():
        shape = tiling_type.shape_types[int(key)]
        if shape == "I":
            raise ValueError(f"IH{tiling_type.number}: edge shape id {key} is a straight (I) edge and cannot be curved")
        curves[int(key)] = EdgeShape.from_points(shape, points, modes.get(key, "spline"))
    extra = sorted(set(modes) - set(curves))
    if extra:
        raise ValueError(f"IH{tiling_type.number}: edge_curve_modes for shape ids without a curve: {extra}")
    return curves


def outline(
    tiling_type: TilingType,
    edge_maps: Array,
    curves: Mapping[int, EdgeShape],
    n_points: int | None = None,
) -> tuple[Array, IntArray]:
    """Closed outline of the prototile and the index in it where each edge starts.

    Parameters
    ----------
    tiling_type : TilingType
    edge_maps : (n_vertices, 2, 3)
        Maps of the canonical edge onto the edges (``edge_transforms``), in any coordinates.
    curves : mapping of shape id to EdgeShape
        Curves of the shape ids; ids without a curve are straight.
    n_points : int, optional
        Samples per spline, see [`interpolate`][isotile.interpolate].

    Returns
    -------
    points : (M, 2)
        Outline without the repeated first point; edge ``i`` is ``points[starts[i]:starts[i + 1]]`` followed by
        the first point of edge ``i + 1``.
    starts : (n_vertices + 1,) int
        Index in ``points`` where each edge starts; the last entry is ``M``.
    """
    samples = {k: c.samples(n_points) for k, c in curves.items()}
    reversed_ = tiling_type.edge_orientations[:, 0] != tiling_type.edge_orientations[:, 1]
    pieces: list[Array] = []
    for i, (sid, m) in enumerate(zip(tiling_type.edge_shape_ids.tolist(), edge_maps, strict=True)):
        s = samples.get(sid, STRAIGHT.points)
        placed = s @ m[:, :2].T + m[:, 2]
        pieces.append(placed[::-1][:-1] if reversed_[i] else placed[:-1])
    starts = np.concatenate([[0], np.cumsum([len(p) for p in pieces])]).astype(np.int64)
    return np.concatenate(pieces), starts


def edge_infos(tiling_type: TilingType, curves: Mapping[int, EdgeShape]) -> tuple[EdgeInfo, ...]:
    """The [`EdgeInfo`][isotile.EdgeInfo] of each edge."""
    reversed_ = tiling_type.edge_orientations[:, 0] != tiling_type.edge_orientations[:, 1]
    return tuple(
        EdgeInfo(i, sid, tiling_type.shape_types[sid], bool(reversed_[i]), curves.get(sid, STRAIGHT))
        for i, sid in enumerate(tiling_type.edge_shape_ids.tolist())
    )


def check_outline(points: Array, starts: IntArray) -> tuple[tuple[int, ...], str]:
    """Test that the closed polyline is simple and has positive area.

    Returns the sorted edges at fault (edge ``i`` owns the segments ``starts[i]`` to ``starts[i + 1] - 1``) and the
    message; the message is ``"valid"`` and the edges empty if there is no fault. Faults: segments of zero length;
    two segments that are not neighbors and cross, touch or overlap; neighboring segments that fold back onto each
    other; an area that is not positive (up to a relative ``1e-9`` of the squared extent).
    """
    n = len(points)
    nxt = np.roll(points, -1, axis=0)
    seg = nxt - points
    length = np.linalg.norm(seg, axis=1)
    extent = float(np.ptp(points, axis=0).max())
    owner = np.repeat(np.arange(len(starts) - 1), np.diff(starts))
    problems: list[str] = []
    bad: set[int] = set()

    short = np.flatnonzero(length <= _LENGTH_TOL * extent)
    if len(short):
        bad.update(owner[short].tolist())
        problems.append(f"zero-length segments in edges {sorted(set(owner[short].tolist()))}")

    tol = _ORIENT_TOL * extent**2

    def side(p: Array, q: Array, r: Array) -> Array:
        c = (q[:, 0] - p[:, 0]) * (r[:, 1] - p[:, 1]) - (q[:, 1] - p[:, 1]) * (r[:, 0] - p[:, 0])
        signs: Array = np.where(np.abs(c) <= tol, 0.0, np.sign(c))
        return signs

    def on_segment(p: Array, q: Array, r: Array) -> NDArray[np.bool_]:
        """r lies in the bounding box of pq (r is known to be collinear with p and q)."""
        inside: NDArray[np.bool_] = (
            (np.minimum(p[:, 0], q[:, 0]) <= r[:, 0])
            & (r[:, 0] <= np.maximum(p[:, 0], q[:, 0]))
            & (np.minimum(p[:, 1], q[:, 1]) <= r[:, 1])
            & (r[:, 1] <= np.maximum(p[:, 1], q[:, 1]))
        )
        return inside

    lo, hi = np.minimum(points, nxt), np.maximum(points, nxt)
    eps = 1e-9 * extent
    close = np.ones((n, n), dtype=bool)
    for axis in (0, 1):
        close &= (lo[:, None, axis] <= hi[None, :, axis] + eps) & (lo[None, :, axis] <= hi[:, None, axis] + eps)
    candidate = np.triu(close, 2)
    candidate[0, n - 1] = False
    i, j = np.nonzero(candidate)
    p, q, r, s = points[i], nxt[i], points[j], nxt[j]
    d1, d2, d3, d4 = side(p, q, r), side(p, q, s), side(r, s, p), side(r, s, q)
    hit = (
        ((d1 * d2 < 0) & (d3 * d4 < 0))
        | ((d1 == 0) & on_segment(p, q, r))
        | ((d2 == 0) & on_segment(p, q, s))
        | ((d3 == 0) & on_segment(r, s, p))
        | ((d4 == 0) & on_segment(r, s, q))
    )
    if hit.any():
        pairs = sorted({(int(owner[a]), int(owner[b])) for a, b in zip(i[hit], j[hit], strict=True)})
        bad.update(x for pair in pairs for x in pair)
        problems.append(
            "edges cross or touch: " + ", ".join(f"{a}-{b}" if a != b else f"{a} (itself)" for a, b in pairs)
        )

    following = np.roll(seg, -1, axis=0)
    cross = seg[:, 0] * following[:, 1] - seg[:, 1] * following[:, 0]
    folded = np.flatnonzero((np.abs(cross) <= tol) & (np.sum(seg * following, axis=1) < 0))
    if len(folded):
        edges = sorted(set(owner[folded].tolist()) | set(owner[(folded + 1) % n].tolist()))
        bad.update(edges)
        problems.append(f"the curve folds back onto itself in edges {edges}")

    area = 0.5 * float(np.sum(points[:, 0] * nxt[:, 1] - nxt[:, 0] * points[:, 1]))
    if not area > _AREA_TOL * extent**2:
        if not bad:
            bad.update(range(len(starts) - 1))
        problems.append("the outline does not have positive area")

    if not problems:
        return (), "valid"
    return tuple(sorted(bad)), "; ".join(problems)


def check_curves(
    tiling_type: TilingType, parameters: Array, curves: Mapping[int, EdgeShape], n_points: int | None = None
) -> PrototileReport:
    """Validity of the outline of a prototile with curves, in native coordinates (the straight prototile is valid)."""
    points, starts = outline(tiling_type, native_edge_transforms(tiling_type, parameters), curves, n_points)
    bad, message = check_outline(points, starts)
    return PrototileReport(not bad and message == "valid", bad, message)
