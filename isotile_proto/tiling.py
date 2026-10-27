"""Isohedral tiling: geometry of one type at given parameters, tile generation, adjacency."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .adjacency import lattice_adjacency
from .types import TilingType, get_type

if TYPE_CHECKING:
    from scipy.sparse import csr_array

Bounds = tuple[float, float, float, float]
AdjacencyKind = Literal["edge", "vertex", "vertex-only"]

# Tactile's four edge orientation matrices (identity, rotate, flip, rotate+flip), index 2 * flip + rotate.
_ORIENTS = np.array(
    [
        [[1, 0, 0], [0, 1, 0]],
        [[-1, 0, 1], [0, -1, 0]],
        [[-1, 0, 1], [0, 1, 0]],
        [[1, 0, 0], [0, -1, 0]],
    ],
    dtype=float,
)


def _homogeneous(m: NDArray[np.float64]) -> NDArray[np.float64]:
    """(..., 2, 3) -> (..., 3, 3)."""
    out = np.zeros((*m.shape[:-2], 3, 3))
    out[..., :2, :] = m
    out[..., 2, 2] = 1.0
    return out


def _is_simple(v: NDArray[np.float64]) -> bool:
    """True if the closed polygon with vertices ``v`` has no crossing or touching non-adjacent edges."""
    n = len(v)
    a, b = v, np.roll(v, -1, axis=0)
    i, j = np.triu_indices(n, 2)
    keep = ~((i == 0) & (j == n - 1))
    i, j = i[keep], j[keep]

    def side(p, q, r):
        c = (q[:, 0] - p[:, 0]) * (r[:, 1] - p[:, 1]) - (q[:, 1] - p[:, 1]) * (r[:, 0] - p[:, 0])
        return np.where(np.abs(c) < 1e-12, 0.0, np.sign(c))

    d1, d2 = side(a[i], b[i], a[j]), side(a[i], b[i], b[j])
    d3, d4 = side(a[j], b[j], a[i]), side(a[j], b[j], b[i])
    return not np.any((d1 * d2 <= 0) & (d3 * d4 <= 0))


class IsohedralTiling:
    """One of the 81 isohedral types at given parameters.

    Parameters
    ----------
    tiling_type : int or TilingType
    parameters : array-like, optional
        ``n_params`` values; the type's defaults when omitted. Any real values are accepted as long as the
        prototile is a simple polygon (the data carries no valid ranges).
    frame : (2, 3) array-like, optional
        Affine map applied to the native geometry (vertices, translation vectors, aspects, edges). Used to
        scale, rotate or reflect the tiling, e.g. a pointy-top hexagon.
    """

    def __init__(
        self, tiling_type: int | TilingType, parameters: ArrayLike | None = None, frame: ArrayLike | None = None
    ) -> None:
        self.type = tiling_type if isinstance(tiling_type, TilingType) else get_type(tiling_type)
        self._frame = _homogeneous(np.asarray(np.eye(3)[:2] if frame is None else frame, dtype=float))
        self._parameters = np.empty(0)
        self.parameters = self.type.default_params if parameters is None else parameters

    @property
    def frame(self) -> NDArray[np.float64]:
        """(2, 3) affine map from native to world coordinates."""
        return self._frame[:2].copy()

    @property
    def parameters(self) -> NDArray[np.float64]:
        return self._parameters.copy()

    @parameters.setter
    def parameters(self, value: ArrayLike) -> None:
        p = np.asarray(value, dtype=float)
        if p.shape != (self.type.n_params,):
            raise ValueError(f"IH{self.type.number} takes {self.type.n_params} parameters, got shape {p.shape}")
        if not np.all(np.isfinite(p)):
            raise ValueError("parameters must be finite")
        self._set(p)

    def _set(self, p: NDArray[np.float64]) -> None:
        """Derive vertices, aspects, translations and edge transforms; keep the old state if invalid."""
        t, f = self.type, self._frame
        p1 = np.append(p, 1.0)
        v = t.vertex_coeffs @ p1
        edge_len = np.linalg.norm(np.roll(v, -1, axis=0) - v, axis=1)
        x, y = v[:, 0], v[:, 1]
        area = 0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)
        scale = edge_len.max()
        if edge_len.min() < 1e-9 * max(scale, 1.0) or abs(area) < 1e-9 * scale**2 or not _is_simple(v):
            raise ValueError(f"IH{t.number}: parameters {p.tolist()} give a degenerate or self-intersecting tile")
        w = np.roll(v, -1, axis=0)
        seg = np.zeros((t.n_vertices, 3, 3))  # tactile's matchSeg: unit edge (0,0)-(1,0) -> v[i]-v[i+1]
        seg[:, 0] = np.stack([w[:, 0] - v[:, 0], v[:, 1] - w[:, 1], v[:, 0]], axis=1)
        seg[:, 1] = np.stack([w[:, 1] - v[:, 1], w[:, 0] - v[:, 0], v[:, 1]], axis=1)
        seg[:, 2, 2] = 1.0
        orient = _homogeneous(_ORIENTS[2 * t.edge_orientations[:, 0] + t.edge_orientations[:, 1]])
        finv = np.linalg.inv(f)
        self._parameters = p
        self._vertices = v @ f[:2, :2].T + f[:2, 2]
        self._aspects = (f @ _homogeneous(t.aspect_coeffs @ p1) @ finv)[:, :2]
        self._translations = (t.translation_coeffs @ p1) @ f[:2, :2].T
        self._edges = (f @ seg @ orient)[:, :2]

    @property
    def vertices(self) -> NDArray[np.float64]:
        """(n_vertices, 2) prototile vertices."""
        return self._vertices.copy()

    @property
    def aspects(self) -> NDArray[np.float64]:
        """(n_aspects, 2, 3) affine transforms of the aspects."""
        return self._aspects.copy()

    @property
    def t1(self) -> NDArray[np.float64]:
        return self._translations[0].copy()

    @property
    def t2(self) -> NDArray[np.float64]:
        return self._translations[1].copy()

    @property
    def edge_transforms(self) -> NDArray[np.float64]:
        """(n_vertices, 2, 3) maps of the canonical edge ``(0, 0)-(1, 0)`` onto each prototile edge."""
        return self._edges.copy()

    @property
    def edge_reversed(self) -> NDArray[np.bool_]:
        """(n_vertices,) True where the edge is traversed against its shape direction (flip != rotate)."""
        return self.type.edge_orientations[:, 0] != self.type.edge_orientations[:, 1]

    @property
    def centroid(self) -> NDArray[np.float64]:
        """Area centroid of the straight-edge prototile."""
        v, w = self._vertices, np.roll(self._vertices, -1, axis=0)
        cross = v[:, 0] * w[:, 1] - w[:, 0] * v[:, 1]
        return ((v + w) * cross[:, None]).sum(axis=0) / (3.0 * cross.sum())

    @property
    def area(self) -> float:
        v, w = self._vertices, np.roll(self._vertices, -1, axis=0)
        return float(abs(0.5 * np.sum(v[:, 0] * w[:, 1] - w[:, 0] * v[:, 1])))

    # -- tile generation -------------------------------------------------------------------------

    def tiles(self, lattice: ArrayLike) -> TileSet:
        """Tiles for explicit lattice coordinates ``(t1, t2, aspect)``, shape (n, 3)."""
        lat = np.asarray(lattice, dtype=np.int64).reshape(-1, 3)
        m = self._aspects[lat[:, 2]].copy()
        m[:, :, 2] += lat[:, :1] * self._translations[0] + lat[:, 1:2] * self._translations[1]
        return TileSet(self, lat, m)

    def _lattice_coords(self, points: NDArray[np.float64]) -> NDArray[np.float64]:
        """Coordinates of world points in the (t1, t2) basis."""
        return points @ np.linalg.inv(self._translations)

    def fill(self, bounds: Bounds) -> TileSet:
        """Tiles generated by tactile's scanline fill of the rectangle, in tactile's order, without repeats.

        Every lattice point ``(t1, t2)`` whose image lies in the rectangle (floor on the lower side, 1e-7 slack
        on the upper side, one extra row at the top) contributes all its aspects. Tiles near the boundary may
        reach outside the rectangle and tiles outside it may reach in; use :meth:`cover` for exact coverage.
        """
        x0, y0, x1, y1 = bounds
        pts = self._lattice_coords(np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]]))
        t1, t2 = self._translations
        if t1[0] * t2[1] - t2[0] * t1[1] < 0.0:
            pts[[1, 3]] = pts[[3, 1]]
        lat = _scanline_points(pts)
        return self.tiles(_with_aspects(lat, self.type.n_aspects))

    def cover(self, bounds: Bounds) -> TileSet:
        """All tiles whose bounding box overlaps the rectangle."""
        x0, y0, x1, y1 = bounds
        corners = self._lattice_coords(np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]]))
        proto = self._lattice_coords(np.einsum("aij,vj->avi", self._aspects[:, :, :2], self._vertices))
        proto += self._lattice_coords(self._aspects[:, None, :, 2])
        lo = np.floor(corners.min(axis=0) - proto.max(axis=(0, 1))).astype(np.int64)
        hi = np.ceil(corners.max(axis=0) - proto.min(axis=(0, 1))).astype(np.int64)
        i, j = np.meshgrid(np.arange(lo[0], hi[0] + 1), np.arange(lo[1], hi[1] + 1), indexing="ij")
        tiles = self.tiles(_with_aspects(np.stack([i.ravel(), j.ravel()], axis=1), self.type.n_aspects))
        v = tiles.vertices()
        keep = (
            (v[:, :, 0].max(axis=1) > x0)
            & (v[:, :, 0].min(axis=1) < x1)
            & (v[:, :, 1].max(axis=1) > y0)
            & (v[:, :, 1].min(axis=1) < y1)
        )
        return tiles.select(keep)


def _with_aspects(points: NDArray[np.int64], n_aspects: int) -> NDArray[np.int64]:
    """(m, 2) lattice points -> (m * n_aspects, 3) rows ``(t1, t2, aspect)``."""
    rep = np.repeat(points, n_aspects, axis=0)
    return np.column_stack([rep, np.tile(np.arange(n_aspects), len(points))])


def _scanline_points(pts: NDArray[np.float64]) -> NDArray[np.int64]:
    """Port of tactile's ``_fill_region_quad`` on lattice-space corners ``pts`` (A, B, C, D), without repeats."""
    chunks: list[NDArray[np.int64]] = []
    last_y: float | None = None

    def slope(p, q):
        dy = q[1] - p[1]
        return (q[0] - p[0]) / dy if dy != 0.0 else 0.0

    def do_fill(a, b, c, d, do_top):
        nonlocal last_y
        dx1, dx2 = slope(a, d), slope(b, c)
        ymax = c[1] + (1.0 if do_top else 0.0)
        y0 = math.floor(a[1])
        if last_y:
            y0 = max(last_y, y0)
        n_rows = max(math.ceil(ymax - y0), 0)
        k = np.arange(n_rows)
        x_lo = np.floor(a[0] + k * dx1).astype(np.int64)
        x_hi = b[0] + k * dx2 + 1e-7
        counts = np.maximum(np.ceil(x_hi - x_lo).astype(np.int64), 0)
        rows = np.repeat(np.arange(n_rows), counts)
        starts = np.cumsum(counts) - counts
        xs = x_lo[rows] + np.arange(counts.sum()) - starts[rows]
        chunks.append(np.column_stack([xs, y0 + rows]))
        last_y = y0 + n_rows

    def fix_x(a, b, c, d, do_top):
        do_fill(*((b, a, d, c) if a[0] > b[0] else (a, b, c, d)), do_top)

    def fix_y(a, b, c, d, do_top):
        do_fill(*((c, d, a, b) if a[1] > c[1] else (a, b, c, d)), do_top)

    a, b, c, d = pts
    if abs(a[1] - b[1]) < 1e-7:
        fix_y(a, b, c, d, True)
    elif abs(b[1] - c[1]) < 1e-7:
        fix_y(b, c, d, a, True)
    else:
        lowest = int(np.argmin(pts[:, 1]))
        bottom, left, top, right = (pts[(lowest + k) % 4] for k in range(4))
        if left[0] > right[0]:
            left, right = right, left
        if left[1] < right[1]:
            r1 = _at_height(bottom, right, left[1])
            l2 = _at_height(left, top, right[1])
            fix_x(bottom, bottom, r1, left, False)
            fix_x(left, r1, right, l2, False)
            fix_x(l2, right, top, top, True)
        else:
            l1 = _at_height(bottom, left, right[1])
            r2 = _at_height(right, top, left[1])
            fix_x(bottom, bottom, right, l1, False)
            fix_x(l1, right, r2, left, False)
            fix_x(left, r2, top, top, True)
    points = np.concatenate(chunks)
    _, first = np.unique(points, axis=0, return_index=True)
    return points[np.sort(first)]  # drop repeated points, keep tactile's order of first occurrence


def _at_height(p, q, y):
    t = (y - p[1]) / (q[1] - p[1])
    return np.array([(1.0 - t) * p[0] + t * q[0], y])


@dataclass(eq=False)
class TileSet:
    """Tiles of a tiling: lattice coordinates ``(t1, t2, aspect)`` and the affine transform of each tile."""

    tiling: IsohedralTiling
    lattice: NDArray[np.int64]  # (n, 3)
    transforms: NDArray[np.float64]  # (n, 2, 3), maps the prototile onto the tile

    def __len__(self) -> int:
        return len(self.lattice)

    def select(self, mask: ArrayLike) -> TileSet:
        return TileSet(self.tiling, self.lattice[mask], self.transforms[mask])

    def vertices(self) -> NDArray[np.float64]:
        """(n, n_vertices, 2) straight-edge tile polygons."""
        return (
            np.einsum("nij,vj->nvi", self.transforms[:, :, :2], self.tiling.vertices) + self.transforms[:, None, :, 2]
        )

    @property
    def centroids(self) -> NDArray[np.float64]:
        c = self.tiling.centroid
        return self.transforms[:, :, :2] @ c + self.transforms[:, :, 2]

    def colors(self) -> NDArray[np.int64]:
        """Tactile's tile coloring (``get_color``) for all tiles."""
        clr, period = self.tiling.type.coloring, self.tiling.type.coloring[18]
        mt1, mt2 = self.lattice[:, 0] % period, self.lattice[:, 1] % period
        col = clr[self.lattice[:, 2]]
        for k in range(period - 1):
            col = np.where(mt1 > k, clr[12 + col], col)
        for k in range(period - 1):
            col = np.where(mt2 > k, clr[15 + col], col)
        return col

    def adjacency(self, kind: AdjacencyKind = "edge") -> csr_array:
        """(n, n) boolean sparse adjacency from the lattice table of the type.

        ``edge``: share an edge; ``vertex``: share an edge or a vertex; ``vertex-only``: share exactly a vertex.
        Pairs where one tile is not in the set are dropped.
        """
        return lattice_adjacency(self.lattice, self.tiling.type, kind)

    def neighbors(self, i: int, kind: AdjacencyKind = "edge") -> NDArray[np.int64]:
        adj = self.adjacency(kind)
        return adj.indices[adj.indptr[i] : adj.indptr[i + 1]]
