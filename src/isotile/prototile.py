"""Validity of the straight prototile and polygon measures."""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import linprog

from .types import TilingType, get_type

_LENGTH_TOL = 1e-9  # relative to the longest edge
_AREA_TOL = 1e-9  # relative to the squared longest edge
_ORIENTATION_TOL = 1e-12  # relative to the squared longest edge
_MAX_SQUARES = 200_000  # branch and bound of inscribed_circle


@dataclass(frozen=True)
class PrototileReport:
    """Result of :func:`check_prototile`.

    Attributes
    ----------
    valid : bool
        True if the straight prototile is a simple polygon with nonzero edge lengths and area.
    bad_edges : tuple of int
        Sorted indices of the edges at fault (edge ``i`` joins vertex ``i`` to vertex ``i + 1``): zero-length edges,
        edges of every pair that crosses or touches without being neighbors, and neighboring edges that fold
        back onto each other. All edges when the polygon has zero area without another fault. Empty if valid.
    message : str
        Description of the problem; ``"valid"`` if there is none.
    """

    valid: bool
    bad_edges: tuple[int, ...]
    message: str


def polygon_area(vertices: NDArray[np.float64]) -> float:
    """Signed area of the polygon (positive if counterclockwise)."""
    x, y = vertices[:, 0], vertices[:, 1]
    return float(0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def polygon_centroid(vertices: NDArray[np.float64]) -> NDArray[np.float64]:
    """Area centroid of the polygon."""
    w = np.roll(vertices, -1, axis=0)
    cross = vertices[:, 0] * w[:, 1] - w[:, 0] * vertices[:, 1]
    centroid: NDArray[np.float64] = ((vertices + w) * cross[:, None]).sum(axis=0) / (3.0 * cross.sum())
    return centroid


def check_polygon(vertices: NDArray[np.float64]) -> PrototileReport:
    """Check that the closed polygon with the given (n, 2) vertices is simple, with nonzero edges and area."""
    n = len(vertices)
    nxt = np.roll(vertices, -1, axis=0)
    edge = nxt - vertices
    length = np.linalg.norm(edge, axis=1)
    scale = float(length.max())
    problems: list[str] = []
    bad: set[int] = set()

    short = np.flatnonzero(length < _LENGTH_TOL * max(scale, 1.0))
    if len(short):
        bad.update(short.tolist())
        problems.append(f"zero-length edges {short.tolist()}")

    tol = _ORIENTATION_TOL * scale**2

    def side(p: NDArray[np.float64], q: NDArray[np.float64], r: NDArray[np.float64]) -> NDArray[np.float64]:
        c = (q[:, 0] - p[:, 0]) * (r[:, 1] - p[:, 1]) - (q[:, 1] - p[:, 1]) * (r[:, 0] - p[:, 0])
        signs: NDArray[np.float64] = np.where(np.abs(c) < tol, 0.0, np.sign(c))
        return signs

    i, j = np.triu_indices(n, 2)
    keep = ~((i == 0) & (j == n - 1))
    i, j = i[keep], j[keep]
    d1, d2 = side(vertices[i], nxt[i], vertices[j]), side(vertices[i], nxt[i], nxt[j])
    d3, d4 = side(vertices[j], nxt[j], vertices[i]), side(vertices[j], nxt[j], nxt[i])
    hit = (d1 * d2 <= 0) & (d3 * d4 <= 0)
    if hit.any():
        pairs = sorted(zip(i[hit].tolist(), j[hit].tolist(), strict=True))
        bad.update(itertools.chain.from_iterable(pairs))
        problems.append(f"edges cross or touch: {pairs}")

    following = np.roll(edge, -1, axis=0)
    cross = edge[:, 0] * following[:, 1] - edge[:, 1] * following[:, 0]
    dot = np.sum(edge * following, axis=1)
    folded = np.flatnonzero((np.abs(cross) < tol) & (dot < 0))
    if len(folded):
        bad.update(folded.tolist())
        bad.update(((folded + 1) % n).tolist())
        problems.append(f"neighboring edges fold back: {[(k, (k + 1) % n) for k in folded.tolist()]}")

    if abs(polygon_area(vertices)) < _AREA_TOL * scale**2:
        if not bad:
            bad.update(range(n))
        problems.append("zero area")

    if not problems:
        return PrototileReport(True, (), "valid")
    return PrototileReport(False, tuple(sorted(bad)), "; ".join(problems))


def check_prototile(tiling_type: int | TilingType, parameters: ArrayLike | None = None) -> PrototileReport:
    """Report whether the straight prototile of a type at the given parameters is a valid simple polygon.

    This is the test the :class:`isotile.IsohedralTiling` constructor applies; it builds no tiling, so an editor can
    call it for every intermediate state. The prototile is checked in the type's native coordinates (a frame does
    not change validity).

    Parameters
    ----------
    tiling_type : int or TilingType
    parameters : array-like, optional
        ``n_params`` values; the defaults when omitted.

    Raises
    ------
    ValueError
        If the type number is invalid or the number of parameters is wrong.
    """
    t = tiling_type if isinstance(tiling_type, TilingType) else get_type(tiling_type)
    p = t.default_params if parameters is None else np.asarray(parameters, dtype=float)
    if p.shape != (t.n_params,):
        raise ValueError(f"IH{t.number} takes {t.n_params} parameters, got shape {p.shape}")
    if not np.all(np.isfinite(p)):
        return PrototileReport(False, tuple(range(t.n_vertices)), "parameters must be finite")
    return check_polygon(t.vertex_coeffs @ np.append(p, 1.0))


def _signed_distance(polygon: NDArray[np.float64], points: NDArray[np.float64]) -> NDArray[np.float64]:
    """Distance of (m, 2) points to the polygon boundary, positive inside, negative outside."""
    a, b = polygon, np.roll(polygon, -1, axis=0)
    ab = b - a
    ap = points[:, None, :] - a[None]
    t = np.clip(np.sum(ap * ab, axis=2) / np.sum(ab * ab, axis=1), 0.0, 1.0)
    distance = np.linalg.norm(ap - t[..., None] * ab, axis=2).min(axis=1)
    y = points[:, 1:2]
    straddles = (a[None, :, 1] > y) != (b[None, :, 1] > y)
    x_cross = a[None, :, 0] + (y - a[None, :, 1]) * ab[None, :, 0] / np.where(ab[:, 1] == 0, 1.0, ab[:, 1])
    inside = (np.sum(straddles & (points[:, 0:1] < x_cross), axis=1) % 2) == 1
    signed: NDArray[np.float64] = np.where(inside, distance, -distance)
    return signed


def inscribed_circle(polygon: NDArray[np.float64]) -> tuple[NDArray[np.float64], float]:
    """Center and radius of the largest circle inside a simple polygon with at least 3 vertices.

    A convex polygon is solved exactly as a linear program (maximize ``r`` subject to ``r`` not exceeding the
    distance to each edge line). Any other polygon is solved by branch and bound on squares: a square is dropped
    when the signed distance at its center plus its half-diagonal cannot exceed the best circle found by more
    than ``1e-7`` times the longer side of the bounding box, so the radius is exact to that tolerance. The search
    stops when more than 200000 squares would remain, which happens only when circles of equal radius fill a long
    stretch (a long strip of constant width); the radius is then that of a circle that exists, not an upper bound.
    """
    oriented = polygon if polygon_area(polygon) > 0 else polygon[::-1]
    edge = np.roll(oriented, -1, axis=0) - oriented
    turn = edge[:, 0] * np.roll(edge[:, 1], -1) - edge[:, 1] * np.roll(edge[:, 0], -1)
    if np.all(turn >= -1e-12 * float(np.abs(edge).max()) ** 2):
        normal = np.column_stack([-edge[:, 1], edge[:, 0]]) / np.linalg.norm(edge, axis=1)[:, None]
        constraints = np.column_stack([-normal, np.ones(len(normal))])
        result = linprog(
            [0.0, 0.0, -1.0],
            A_ub=constraints,
            b_ub=-np.sum(normal * oriented, axis=1),
            bounds=(None, None),
            method="highs",
        )
        solution = np.asarray(result.x, dtype=float)
        return solution[:2], float(solution[2])

    low, high = polygon.min(axis=0), polygon.max(axis=0)
    size = float((high - low).max())
    tol = 1e-7 * size
    half = size / 2.0
    counts = np.maximum(1, np.ceil((high - low) / size - 1e-12)).astype(int)
    grid = np.stack(np.meshgrid(np.arange(counts[0]), np.arange(counts[1]), indexing="ij"), axis=-1).reshape(-1, 2)
    centers: NDArray[np.float64] = low + half + grid * size
    best_distance = -np.inf
    best = centers[0]
    while len(centers):
        distance = _signed_distance(polygon, centers)
        top = int(np.argmax(distance))
        if distance[top] > best_distance:
            best_distance, best = float(distance[top]), centers[top]
        centers = centers[distance + half * np.sqrt(2.0) > best_distance + tol]
        if half * np.sqrt(2.0) <= tol or 4 * len(centers) > _MAX_SQUARES:
            break
        half /= 2.0
        offsets = half * np.array([[-1.0, -1.0], [1.0, -1.0], [-1.0, 1.0], [1.0, 1.0]])
        centers = (centers[:, None, :] + offsets).reshape(-1, 2)
    return best, best_distance
