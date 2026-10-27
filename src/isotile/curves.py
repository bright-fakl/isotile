"""The prototile with curved edges: its outline in the plane and its validity.

A curved prototile is valid when the straight prototile of the same parameters is valid and the outline, the closed
curve that runs along the edge curves in order, is a simple closed curve with positive area whose parts that are not
directly consecutive are at least ``1e-6`` times the diameter of the outline apart. The edge shapes make the copies
of the prototile fit along every shared edge, so no further condition is needed. The test is exact for the curve
itself (see `isotile.clearance`); it does not depend on how the curve is sampled for drawing.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .clearance import bounding_box, check_pieces, signed_area
from .edges import EdgeShape
from .prototile import PrototileReport
from .types import TilingType

Array = NDArray[np.float64]
IntArray = NDArray[np.int64]

CLEARANCE = 1e-6  # smallest distance between parts of the outline, relative to its diameter
_AREA_TOL = 1e-9  # smallest area of the outline, relative to the squared diameter


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


def placed_pieces(tiling_type: TilingType, edge_maps: Array, curves: Mapping[int, EdgeShape]) -> tuple[Array, IntArray]:
    """Cubic Bezier pieces ``(N, 4, 2)`` of the closed outline in order, and the edge of each piece.

    Straight edges are one straight piece. Reversed edges are traversed from their end.
    """
    reversed_ = tiling_type.edge_orientations[:, 0] != tiling_type.edge_orientations[:, 1]
    pieces: list[Array] = []
    owner: list[int] = []
    for i, (sid, m) in enumerate(zip(tiling_type.edge_shape_ids.tolist(), edge_maps, strict=True)):
        local = curves.get(sid, STRAIGHT).pieces()
        placed = local @ m[:, :2].T + m[:, 2]
        if reversed_[i]:
            placed = placed[::-1, ::-1]
        pieces.append(placed)
        owner.extend([i] * len(placed))
    return np.concatenate(pieces), np.array(owner, dtype=np.int64)


def check_curves(tiling_type: TilingType, parameters: Array, curves: Mapping[int, EdgeShape]) -> PrototileReport:
    """Validity of the outline of a prototile with curves, in native coordinates (the straight prototile is valid).

    The outline must be a simple closed curve with positive area, and parts of it that are not directly consecutive
    must be at least ``1e-6`` times its diameter apart (the larger side of its bounding box). The result does not
    depend on how the curves are sampled.
    """
    pieces, owner = placed_pieces(tiling_type, native_edge_transforms(tiling_type, parameters), curves)
    low, high = bounding_box(pieces)
    diameter = float((high - low).max())
    n_edges = tiling_type.n_vertices
    bad, problems = check_pieces(pieces, owner, n_edges, CLEARANCE * diameter)
    if not problems and not signed_area(pieces) > _AREA_TOL * diameter**2:
        bad = tuple(range(n_edges))
        problems = ["the outline does not have positive area"]
    return PrototileReport(not problems, bad, "; ".join(problems) if problems else "valid")
