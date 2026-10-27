"""Isohedral tiling: geometry of one type at given parameters, tile generation, adjacency."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .adjacency import AdjacencyKind, lattice_adjacency
from .curves import EdgeInfo, check_curves, edge_infos, native_edge_transforms, normalize_curves, outline
from .prototile import check_prototile, inscribed_circle, polygon_area, polygon_centroid
from .types import TilingType, get_type

if TYPE_CHECKING:
    from scipy.sparse import csr_array

Bounds = tuple[float, float, float, float]
_MIN_OVERLAP = 1e-9  # smallest overlap area counted by cover, as a fraction of the tile area
Array = NDArray[np.float64]
IntArray = NDArray[np.int64]

SQUARE, HEXAGON, TRIANGLE, QUADRILATERAL = 76, 20, 84, 46  # types of the named constructors


def _homogeneous(m: Array) -> Array:
    """(..., 2, 3) -> (..., 3, 3)."""
    out = np.zeros((*m.shape[:-2], 3, 3))
    out[..., :2, :] = m
    out[..., 2, 2] = 1.0
    return out


def _similarity(z: complex, t: complex) -> Array:
    """(2, 3) matrix of x -> z * x + t on points written as complex numbers."""
    return np.array([[z.real, -z.imag, t.real], [z.imag, z.real, t.imag]])


class IsohedralTiling:
    """One of the 81 isohedral types at given parameters, placed by an affine frame.

    A tile is identified by its cell ``(u, v)``, the parallelogram of the lattice spanned by
    [`translation_u`][isotile.IsohedralTiling.translation_u] and
    [`translation_v`][isotile.IsohedralTiling.translation_v], and its aspect, the index of the tile within the cell.

    Parameters
    ----------
    tiling_type : int or TilingType
        Type number (see [`tiling_types`][isotile.tiling_types]) or the type data.
    parameters : array-like, optional
        ``n_params`` values; the type's defaults when omitted. Any finite values for which the straight prototile
        is a valid simple polygon are accepted (see [`check_prototile`][isotile.check_prototile]).
    edge_curves : mapping of int to (m, 2) array-like, optional
        Curves for the edge shape ids (`TilingType.edge_shape_ids`): control points on the canonical edge, from
        ``(0, 0)`` to ``(1, 0)``. Every edge with the shape id gets the curve. The symmetry of the shape is
        enforced (see [`enforce_symmetry`][isotile.enforce_symmetry]). Ids that are not given stay straight; an id
        of an ``I`` edge is an error.
    edge_curve_modes : mapping of int to str, optional
        ``"spline"`` (default) or ``"linear"`` for the shape ids in ``edge_curves``, see
        [`interpolate`][isotile.interpolate].
    frame : (2, 3) array-like, optional
        Affine map from the type's native coordinates to world coordinates, applied to the vertices, translation
        vectors, aspects and edges. Identity when omitted.

    Attributes
    ----------
    type : TilingType
        The data of the type.

    Raises
    ------
    ValueError
        If the type, the number of parameters or the frame is invalid, if an edge curve is malformed (an id that
        is not a shape id or belongs to an ``I`` edge, points that do not run from ``(0, 0)`` to ``(1, 0)``), or
        if the prototile or its outline with the curves is not valid (see
        [`check_prototile`][isotile.check_prototile]). The parameter setter keeps the previous state in that case.
    """

    _vertices: Array
    _aspects: Array
    _translations: Array
    _edges: Array
    _outline: Array
    _outline_starts: IntArray

    def __init__(
        self,
        tiling_type: int | TilingType,
        parameters: ArrayLike | None = None,
        *,
        edge_curves: Mapping[int, ArrayLike] | None = None,
        edge_curve_modes: Mapping[int, str] | None = None,
        frame: ArrayLike | None = None,
    ) -> None:
        self.type = tiling_type if isinstance(tiling_type, TilingType) else get_type(tiling_type)
        self._curves = normalize_curves(self.type, edge_curves, edge_curve_modes)
        f = np.asarray(np.eye(3)[:2] if frame is None else frame, dtype=float)
        if f.shape != (2, 3) or not np.all(np.isfinite(f)):
            raise ValueError(f"frame must be a finite (2, 3) array, got shape {f.shape}")
        if abs(np.linalg.det(f[:, :2])) <= 1e-12 * float(np.abs(f[:, :2]).max()) ** 2:
            raise ValueError("the linear part of the frame is singular")
        self._frame = _homogeneous(f)
        self._parameters: Array = np.empty(0)
        self.parameters = self.type.default_params if parameters is None else parameters

    # -- named constructors ----------------------------------------------------------------------

    @classmethod
    def square(cls) -> IsohedralTiling:
        """Unit squares (type 76); the tile ``(0, 0)`` is ``[0, 1] x [0, 1]``."""
        return cls(SQUARE)

    @classmethod
    def hexagon(cls, pointy_top: bool = True) -> IsohedralTiling:
        """Regular hexagons with circumradius 1 (type 20), centered at the origin for the tile ``(0, 0)``.

        With ``pointy_top`` the tiles have a vertex at the top; otherwise they are flat-top (a vertex at angle 0).
        """
        frame = _similarity(complex(math.cos(math.pi / 6), math.sin(math.pi / 6)) if pointy_top else 1.0, 0.0)
        return cls(HEXAGON, frame=frame)

    @classmethod
    def triangle(cls, vertices: ArrayLike | None = None) -> IsohedralTiling:
        """Tiling by copies of a triangle that are related by point reflections about the edge midpoints (type 84).

        ``vertices`` is any triangle as 3 points; the prototile equals it (up to the starting vertex and the
        orientation, which is counterclockwise). The default is the equilateral triangle with side 1.
        """
        default = [[0.0, 0.0], [1.0, 0.0], [0.5, math.sqrt(3.0) / 2.0]]
        return _fit(cls, TRIANGLE, default if vertices is None else vertices)

    @classmethod
    def quadrilateral(cls, vertices: ArrayLike | None = None) -> IsohedralTiling:
        """Tiling by copies of a quadrilateral related by point reflections about the edge midpoints (type 46).

        ``vertices`` is any simple quadrilateral as 4 points, convex or not; the prototile equals it (up to the
        starting vertex and the orientation, which is counterclockwise). The default is the unit square.
        A self-intersecting quadrilateral raises ``ValueError``.
        """
        default = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]
        return _fit(cls, QUADRILATERAL, default if vertices is None else vertices)

    @classmethod
    def from_preset(cls, name: str, **overrides: Any) -> IsohedralTiling:
        """A tiling from a named preset, see [`list_presets`][isotile.list_presets].

        Parameters
        ----------
        name : str
            Preset name.
        **overrides
            Replace arguments of the preset (``parameters``, ``edge_curves``, ``edge_curve_modes``, ``frame``).

        Raises
        ------
        ValueError
            If the name is unknown or the result is not a valid tiling.
        """
        from .presets import preset_arguments

        return cls(**{**preset_arguments(name), **overrides})

    # -- frame -----------------------------------------------------------------------------------

    @property
    def frame(self) -> Array:
        """(2, 3) affine map from native to world coordinates."""
        return self._frame[:2].copy()

    def with_frame(self, frame: ArrayLike) -> IsohedralTiling:
        """The same type, parameters and edge curves with the frame replaced."""
        return IsohedralTiling(
            self.type,
            self._parameters,
            edge_curves=self.edge_curves,
            edge_curve_modes=self.edge_curve_modes,
            frame=frame,
        )

    def _after(self, m: Array) -> IsohedralTiling:
        return self.with_frame((_homogeneous(m) @ self._frame)[:2])

    def rotated(self, angle_degrees: float) -> IsohedralTiling:
        """A copy rotated counterclockwise about the world origin."""
        a = math.radians(angle_degrees)
        return self._after(_similarity(complex(math.cos(a), math.sin(a)), 0.0))

    def scaled(self, factor: float) -> IsohedralTiling:
        """A copy scaled about the world origin."""
        return self._after(_similarity(complex(factor), 0.0))

    # -- parameters and geometry -----------------------------------------------------------------

    @property
    def parameters(self) -> Array:
        """(n_params,) current parameters. Assigning checks the prototile and raises ``ValueError`` if it is invalid."""
        return self._parameters.copy()

    @parameters.setter
    def parameters(self, value: ArrayLike) -> None:
        p = np.asarray(value, dtype=float)
        report = check_prototile(self.type, p)
        if report.valid and self._curves:
            report = check_curves(self.type, p, self._curves)
        if not report.valid:
            raise ValueError(f"IH{self.type.number}: invalid prototile for parameters {p.tolist()}: {report.message}")
        self._set(p)

    def _set(self, p: Array) -> None:
        """Derive vertices, aspects, translations and edge transforms."""
        t, f = self.type, self._frame
        p1 = np.append(p, 1.0)
        v = t.vertex_coeffs @ p1
        self._parameters = p
        self._vertices = v @ f[:2, :2].T + f[:2, 2]
        self._aspects = (f @ _homogeneous(t.aspect_coeffs @ p1) @ np.linalg.inv(f))[:, :2]
        self._translations = (t.translation_coeffs @ p1) @ f[:2, :2].T
        self._edges = (f @ _homogeneous(native_edge_transforms(t, p)))[:, :2]
        self._outline, self._outline_starts = outline(t, self._edges, self._curves)

    @property
    def vertices(self) -> Array:
        """(n_vertices, 2) prototile vertices."""
        return self._vertices.copy()

    @property
    def aspects(self) -> Array:
        """(n_aspects, 2, 3) affine transforms of the aspects."""
        return self._aspects.copy()

    @property
    def translations(self) -> Array:
        """(2, 2) lattice translation vectors; row k is vector k."""
        return self._translations.copy()

    @property
    def translation_u(self) -> Array:
        """First lattice translation vector."""
        return np.array(self._translations[0], dtype=float)

    @property
    def translation_v(self) -> Array:
        """Second lattice translation vector."""
        return np.array(self._translations[1], dtype=float)

    @property
    def edge_transforms(self) -> Array:
        """(n_vertices, 2, 3) maps of the canonical edge ``(0, 0)-(1, 0)`` onto each prototile edge."""
        return self._edges.copy()

    @property
    def edge_reversed(self) -> NDArray[np.bool_]:
        """(n_vertices,) True where the edge is traversed against its shape direction (flip != rotate)."""
        reversed_: NDArray[np.bool_] = self.type.edge_orientations[:, 0] != self.type.edge_orientations[:, 1]
        return reversed_

    @property
    def edge_shape_ids(self) -> IntArray:
        """(n_vertices,) id of the shape of each edge; edges with the same id have the same shape."""
        return self.type.edge_shape_ids.copy()

    @property
    def edge_shapes(self) -> tuple[str, ...]:
        """Shape of each edge: ``"I"`` (straight), ``"J"``, ``"U"`` or ``"S"``."""
        return self.type.edge_shapes

    @property
    def edge_curves(self) -> dict[int, Array]:
        """Control points of the curve of each shape id that has one, with the symmetry of its shape enforced."""
        return {k: np.array(c.points) for k, c in self._curves.items()}

    @property
    def edge_curve_modes(self) -> dict[int, str]:
        """Interpolation mode (``"spline"`` or ``"linear"``) of each shape id in `edge_curves`."""
        return {k: c.mode for k, c in self._curves.items()}

    @property
    def edges(self) -> tuple[EdgeInfo, ...]:
        """The edges of the prototile in order, with shape id, shape, reversal and curve."""
        return edge_infos(self.type, self._curves)

    def with_edge_curves(
        self,
        edge_curves: Mapping[int, ArrayLike] | None,
        edge_curve_modes: Mapping[int, str] | None = None,
    ) -> IsohedralTiling:
        """The same type, parameters and frame with the edge curves replaced (``None`` removes all curves).

        Raises
        ------
        ValueError
            As the constructor, if the curves are malformed or the outline is not valid.
        """
        return IsohedralTiling(
            self.type, self._parameters, edge_curves=edge_curves, edge_curve_modes=edge_curve_modes, frame=self.frame
        )

    @property
    def area(self) -> float:
        """Area of the straight prototile."""
        return abs(polygon_area(self._vertices))

    @property
    def centroid(self) -> Array:
        """Area centroid of the straight prototile."""
        return polygon_centroid(self._vertices)

    def prototile(self, normalized: bool = False, *, curved: bool = True, n_points: int | None = None) -> Array:
        """(M, 2) closed outline of the prototile in world coordinates, without the repeated first point.

        The outline starts at vertex 0 and runs along the edges in order; each edge with a curve contributes its
        samples, the others only their start vertex. Without curves (or with ``curved=False``) the outline is
        `vertices`.

        With ``normalized`` the outline is translated so that the centroid of the straight prototile is at the
        origin and scaled to unit area of the straight prototile (the curved prototile has the same area).

        Parameters
        ----------
        normalized : bool, default False
        curved : bool, default True
            Replace the edges by their curves.
        n_points : int, optional
            Samples per spline edge, see [`interpolate`][isotile.interpolate]. It only sets how finely the outline is
            drawn; validity does not depend on it.
        """
        points = self._curved_outline(n_points)[0] if curved else self._vertices
        if not normalized:
            return points.copy()
        return (points - self.centroid) / math.sqrt(self.area)

    def _curved_outline(self, n_points: int | None) -> tuple[Array, IntArray]:
        if n_points is None or not self._curves:
            return self._outline, self._outline_starts
        return outline(self.type, self._edges, self._curves, n_points)

    def edge_outlines(self, curved: bool = True, *, n_points: int | None = None) -> list[Array]:
        """One polyline per edge in world coordinates, from vertex ``i`` to vertex ``i + 1``.

        Parameters
        ----------
        curved : bool, default True
            Use the curves; otherwise every edge is the segment between its vertices.
        n_points : int, optional
            Samples per spline edge, see [`interpolate`][isotile.interpolate].

        Returns
        -------
        list of (k_i, 2) ndarray
            Edge ``i`` has at least 2 points (its end vertices); consecutive edges share their end points.
        """
        if not curved:
            return [np.stack([a, b]) for a, b in zip(self._vertices, np.roll(self._vertices, -1, axis=0), strict=True)]
        points, starts = self._curved_outline(n_points)
        closed = np.vstack([points, points[:1]])
        return [closed[starts[i] : starts[i + 1] + 1].copy() for i in range(self.type.n_vertices)]

    @property
    def inscribed_radius(self) -> float:
        """Radius of the largest circle inside the straight prototile (to a relative accuracy of 1e-10)."""
        return inscribed_circle(self._vertices)[1]

    def info(self) -> dict[str, Any]:
        """Metadata of the type as in [`TilingType.info`][isotile.TilingType.info], for the current parameters.

        Adds the key ``parameter_values`` (the current parameters) and, in each entry of ``parameters``, the key
        ``value``. ``default_parameters`` and ``default`` keep the type's defaults.
        """
        info = self.type.info()
        info["parameter_values"] = self._parameters.tolist()
        for entry, value in zip(info["parameters"], self._parameters.tolist(), strict=True):
            entry["value"] = value
        return info

    def describe(self) -> str:
        """Human-readable summary of the type with the current parameter values, see [`TilingType.describe`][isotile.TilingType.describe]."""
        return self.type.describe(self._parameters)

    # -- serialization ---------------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """The tiling as a JSON-compatible dict: type, parameters, edge curves and modes, frame.

        The format is described in [`isotile.io`][isotile.io].
        """
        from .io import to_dict

        return to_dict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> IsohedralTiling:
        """The tiling that a dict from [`to_dict`][isotile.IsohedralTiling.to_dict] describes.

        Raises
        ------
        ValueError
            If the data are not a valid version 1.x tiling.
        """
        from .io import from_dict

        return from_dict(data)

    def save(self, path: str | Path) -> None:
        """Write the tiling to a JSON file, see [`isotile.io.save`][isotile.io.save]."""
        from .io import save

        save(self, path)

    @classmethod
    def load(cls, path: str | Path) -> IsohedralTiling:
        """Read a tiling from a JSON file written by [`save`][isotile.IsohedralTiling.save]."""
        from .io import load

        return load(path)

    # -- tile generation -------------------------------------------------------------------------

    def tiles(self, cell: ArrayLike, aspect: ArrayLike) -> TileSet:
        """The tiles with the given cells and aspects.

        Parameters
        ----------
        cell : (n, 2) array-like of int
            Cells ``(u, v)``.
        aspect : int or (n,) array-like of int
            Aspect of each tile; a single value applies to all cells.

        Returns
        -------
        TileSet

        Raises
        ------
        ValueError
            If an aspect is outside ``[0, n_aspects)``.
        """
        cells = np.asarray(cell, dtype=np.int64).reshape(-1, 2)
        aspects = np.broadcast_to(np.asarray(aspect, dtype=np.int64), (len(cells),)).copy()
        if len(aspects) and (aspects.min() < 0 or aspects.max() >= self.type.n_aspects):
            raise ValueError(f"aspect must be in [0, {self.type.n_aspects})")
        m = self._aspects[aspects].copy()
        m[:, :, 2] += cells[:, :1] * self._translations[0] + cells[:, 1:2] * self._translations[1]
        return TileSet(self, cells, aspects, m)

    def _cell_coordinates(self, points: Array) -> Array:
        """Coordinates of world points in the basis of the translation vectors."""
        coordinates: Array = points @ np.linalg.inv(self._translations)
        return coordinates

    def fill(self, bounds: Bounds) -> TileSet:
        """Tiles generated by the scanline fill of the rectangle, in scanline order, without repeats.

        Every cell whose origin lies in the rectangle (floor on the lower side, 1e-7 slack on the upper side, one
        extra row at the top) contributes all its aspects. Tiles near the boundary may reach outside the
        rectangle and tiles outside it may reach in; use [`cover`][isotile.IsohedralTiling.cover] for exact
        coverage.

        Parameters
        ----------
        bounds : (x0, y0, x1, y1)
            The rectangle.

        Returns
        -------
        TileSet
        """
        x0, y0, x1, y1 = bounds
        pts = self._cell_coordinates(np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]]))
        u, v = self._translations
        if u[0] * v[1] - v[0] * u[1] < 0.0:
            pts[[1, 3]] = pts[[3, 1]]
        origins = _scanline_cells(pts)
        return self.tiles(*_with_aspects(origins, self.type.n_aspects))

    def cover(self, bounds: Bounds) -> TileSet:
        """All tiles whose straight-edge polygon overlaps the rectangle ``bounds`` with positive area.

        Parameters
        ----------
        bounds : (x0, y0, x1, y1)
            The rectangle.

        Returns
        -------
        TileSet
            The tiles sorted by cell ``u``, then ``v``, then aspect.

        Raises
        ------
        ValueError
            If the rectangle does not have ``x1 > x0`` and ``y1 > y0`` (a region of zero width or height has no
            positive-area overlap) or has non-finite bounds.

        Notes
        -----
        A tile that only touches the rectangle at a point or along an edge is not included. The overlap area must
        exceed ``1e-9`` times the tile area, which separates real overlaps from rounding errors at contacts. The
        test is exact for any simple polygon, convex or not: after a bounding-box prefilter the polygon is clipped
        against the four sides of the rectangle (Sutherland-Hodgman, vectorized over the tiles) and the area of
        the result is measured.
        """
        x0, y0, x1, y1 = (float(b) for b in bounds)
        if not (x1 > x0 and y1 > y0 and math.isfinite(x1 - x0) and math.isfinite(y1 - y0)):
            raise ValueError("bounds must be finite with x1 > x0 and y1 > y0")
        corners = self._cell_coordinates(np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]]))
        proto = self._cell_coordinates(np.einsum("aij,vj->avi", self._aspects[:, :, :2], self._vertices))
        proto += self._cell_coordinates(self._aspects[:, None, :, 2])
        lo = np.floor(corners.min(axis=0) - proto.max(axis=(0, 1))).astype(np.int64)
        hi = np.ceil(corners.max(axis=0) - proto.min(axis=(0, 1))).astype(np.int64)
        i, j = np.meshgrid(np.arange(lo[0], hi[0] + 1), np.arange(lo[1], hi[1] + 1), indexing="ij")
        tiles = self.tiles(*_with_aspects(np.stack([i.ravel(), j.ravel()], axis=1), self.type.n_aspects))
        v = tiles.vertices()
        candidate = (
            (v[:, :, 0].max(axis=1) > x0)
            & (v[:, :, 0].min(axis=1) < x1)
            & (v[:, :, 1].max(axis=1) > y0)
            & (v[:, :, 1].min(axis=1) < y1)
        )
        keep = np.zeros(len(tiles), dtype=bool)
        # areas are measured relative to the center of the region: the rounding error of the clipped areas grows
        # with the square of the distance from the origin and would otherwise drop tiles far from it
        center = np.array([(x0 + x1) / 2.0, (y0 + y1) / 2.0])
        local = (x0 - center[0], y0 - center[1], x1 - center[0], y1 - center[1])
        keep[candidate] = _overlap_area(v[candidate] - center, local) > _MIN_OVERLAP * self.area
        return tiles.select(keep)

    def fill_region(
        self,
        bounds: Bounds,
        *,
        n_tiles: int | None = None,
        tile_size: float | None = None,
        margin: float = 0.0,
    ) -> TileSet:
        """Tiles that cover a region, with the tile size set by a tile count or given directly.

        The tile area is the area of the straight prototile. The frame of this tiling gives the orientation (its
        linear part is scaled by ``tile_size / sqrt(area)``); its own scale and position are replaced: the type's
        native origin is placed at the center of the region.

        The lattice is anchored at the center of the region (after the margin is applied). A region that is expanded
        by the same amount on all four sides therefore keeps the lattice at the same tile size: every tile of the
        region is a tile of the expanded region.

        Parameters
        ----------
        bounds : (x0, y0, x1, y1)
            The region, a rectangle.
        n_tiles : int, optional
            Target number of tiles. The tile size is ``sqrt(region_area / n_tiles)``, so the tiles that fit into
            the area have that count; the tiles along the boundary that reach into the region come on top.
        tile_size : float, optional
            Square root of the area of one tile. Give exactly one of ``n_tiles`` and ``tile_size``.
        margin : float, default 0.0
            The region is expanded on every side by ``margin`` times its width (left and right) or height (top and
            bottom).

        Returns
        -------
        TileSet
            [`cover`][isotile.IsohedralTiling.cover] of the expanded region by the resulting tiling (the tiles that
            overlap the region with positive area), available as ``tiles.tiling``.

        Raises
        ------
        ValueError
            If both or neither of ``n_tiles`` and ``tile_size`` are given, if a value is not positive, if the
            bounds are empty or if ``margin <= -0.5``.
        """
        if (n_tiles is None) == (tile_size is None):
            raise ValueError("give exactly one of n_tiles and tile_size")
        if not margin > -0.5:
            raise ValueError("margin must be greater than -0.5")
        x0, y0, x1, y1 = (float(b) for b in bounds)
        width, height = x1 - x0, y1 - y0
        if not (width > 0.0 and height > 0.0 and math.isfinite(width * height)):
            raise ValueError("bounds must be finite with x1 > x0 and y1 > y0")
        x0, y0, x1, y1 = x0 - width * margin, y0 - height * margin, x1 + width * margin, y1 + height * margin
        if tile_size is None:
            if n_tiles is None or not n_tiles > 0:
                raise ValueError("n_tiles must be positive")
            size = math.sqrt((x1 - x0) * (y1 - y0) / n_tiles)
        else:
            if not (tile_size > 0.0 and math.isfinite(tile_size)):
                raise ValueError("tile_size must be positive and finite")
            size = float(tile_size)
        factor = size / math.sqrt(self.area)
        frame = np.column_stack([factor * self._frame[:2, :2], [(x0 + x1) / 2.0, (y0 + y1) / 2.0]])
        return self.with_frame(frame).cover((x0, y0, x1, y1))


def _overlap_area(polygons: Array, bounds: Bounds) -> Array:
    """(n,) area of the intersection of each simple polygon (n, k, 2) with the rectangle.

    Sutherland-Hodgman clipping against the four sides. An edge yields up to two points (the crossing with the
    side, then its end point); points that are not emitted are replaced by the previous emitted point, which only
    adds zero-length edges, so all polygons keep a rectangular array. Clipping a non-convex polygon can add edges
    along the rectangle's sides that enclose no area, so the shoelace area of the result is the area of the
    intersection.
    """
    x0, y0, x1, y1 = bounds
    points = polygons
    alive = np.ones(len(polygons), dtype=bool)
    for axis, value, sign in ((0, x0, 1.0), (0, x1, -1.0), (1, y0, 1.0), (1, y1, -1.0)):
        side = sign * (points[..., axis] - value)
        previous, previous_side = np.roll(points, 1, axis=1), np.roll(side, 1, axis=1)
        inside, was_inside = side >= 0.0, previous_side >= 0.0
        crossing = inside != was_inside
        t = np.where(crossing, previous_side / np.where(crossing, previous_side - side, 1.0), 0.0)
        crossing_point = previous + t[..., None] * (points - previous)
        n, m = side.shape
        emitted = np.stack([crossing_point, points], axis=2).reshape(n, 2 * m, 2)
        valid = np.stack([crossing, inside], axis=2).reshape(n, 2 * m)
        alive &= valid.any(axis=1)
        index = np.maximum.accumulate(np.where(valid, np.arange(2 * m), -1), axis=1)
        last = index[:, -1:]  # leading gaps take the last emitted point: the polygon is cyclic
        points = np.take_along_axis(emitted, np.where(index < 0, last, index)[..., None], axis=1)
    x, y = points[..., 0], points[..., 1]
    area: Array = 0.5 * np.abs(np.sum(x * np.roll(y, -1, axis=1) - np.roll(x, -1, axis=1) * y, axis=1))
    return np.where(alive, area, 0.0)


def _fit(cls: type[IsohedralTiling], number: int, polygon: ArrayLike) -> IsohedralTiling:
    """Type ``number`` whose prototile is ``polygon`` (a similar copy by the parameters, then a similarity frame)."""
    tt = get_type(number)
    pts = np.asarray(polygon, dtype=float)
    if pts.shape != (tt.n_vertices, 2) or not np.all(np.isfinite(pts)):
        raise ValueError(f"IH{number} needs {tt.n_vertices} finite vertices, got shape {pts.shape}")
    if polygon_area(pts) < 0:
        pts = pts[::-1]  # the prototile is counterclockwise
    lengths = np.linalg.norm(np.roll(pts, -1, axis=0) - pts, axis=1)
    if lengths.max() == 0.0:
        raise ValueError("the polygon has no extent")
    pts = np.roll(pts, -int(np.argmax(lengths)), axis=0)  # longest edge first keeps the parameters moderate
    const = tt.vertex_coeffs[:, :, -1]
    if np.any(tt.vertex_coeffs[:2, :, :-1]):
        raise ValueError(f"IH{number}: the first edge depends on the parameters")
    p = pts[:, 0] + 1j * pts[:, 1]
    n = const[:, 0] + 1j * const[:, 1]
    z = (n[1] - n[0]) / (p[1] - p[0])  # similarity from the polygon to native coordinates
    w = n[0] - z * p[0]
    q = z * p[2:] + w
    a = tt.vertex_coeffs[2:, :, :-1].reshape(-1, tt.n_params)
    b = (np.column_stack([q.real, q.imag]) - const[2:]).reshape(-1)
    params, *_ = np.linalg.lstsq(a, b, rcond=None)
    if not np.allclose(a @ params, b, atol=1e-9):
        raise ValueError(f"IH{number} cannot express this polygon")
    return cls(number, params, frame=_similarity(1 / z, -w / z))


def _with_aspects(cells: IntArray, n_aspects: int) -> tuple[IntArray, IntArray]:
    """(m, 2) cells -> cells (m * n_aspects, 2) and aspects (m * n_aspects,), all aspects of each cell."""
    return np.repeat(cells, n_aspects, axis=0), np.tile(np.arange(n_aspects, dtype=np.int64), len(cells))


def _scanline_cells(pts: Array) -> IntArray:
    """Scanline fill of the quadrilateral ``pts`` (A, B, C, D) given in cell coordinates; cells without repeats."""
    chunks: list[IntArray] = []
    last_y: float | None = None

    def slope(p: Array, q: Array) -> float:
        dy = q[1] - p[1]
        return float((q[0] - p[0]) / dy) if dy != 0.0 else 0.0

    def do_fill(a: Array, b: Array, c: Array, d: Array, do_top: bool) -> None:
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

    def fix_x(a: Array, b: Array, c: Array, d: Array, do_top: bool) -> None:
        do_fill(*((b, a, d, c) if a[0] > b[0] else (a, b, c, d)), do_top)

    def fix_y(a: Array, b: Array, c: Array, d: Array, do_top: bool) -> None:
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
    cells = np.concatenate(chunks)
    _, first = np.unique(cells, axis=0, return_index=True)
    unique: IntArray = cells[np.sort(first)]  # drop repeated cells, keep the order of first occurrence
    return unique


def _at_height(p: Array, q: Array, y: float) -> Array:
    t = (y - p[1]) / (q[1] - p[1])
    return np.array([(1.0 - t) * p[0] + t * q[0], y])


@dataclass(eq=False)
class TileSet:
    """Tiles of a tiling: cell ``(u, v)``, aspect and affine transform of each tile.

    Attributes
    ----------
    tiling : IsohedralTiling
        The tiling that the tiles belong to.
    cell : (n, 2) int
        Columns ``u`` and ``v``: the tile's cell in the basis of the translation vectors.
    aspect : (n,) int
        Index of the tile within its cell.
    transforms : (n, 2, 3)
        Affine map of the prototile onto each tile.
    """

    tiling: IsohedralTiling
    cell: IntArray
    aspect: IntArray
    transforms: Array

    def __len__(self) -> int:
        return len(self.cell)

    def select(self, mask: NDArray[np.bool_] | IntArray) -> TileSet:
        """The tiles that a boolean mask or an integer index array selects.

        Parameters
        ----------
        mask : (n,) array of bool or of int
            Boolean mask over the tiles, or indices of the tiles to keep.

        Returns
        -------
        TileSet
        """
        return TileSet(self.tiling, self.cell[mask], self.aspect[mask], self.transforms[mask])

    def vertices(self) -> Array:
        """(n, n_vertices, 2) straight-edge tile polygons, in world coordinates."""
        vertices: Array = (
            np.einsum("nij,vj->nvi", self.transforms[:, :, :2], self.tiling.vertices) + self.transforms[:, None, :, 2]
        )
        return vertices

    @property
    def centroids(self) -> Array:
        """(n, 2) centroids of the tiles."""
        centroids: Array = self.transforms[:, :, :2] @ self.tiling.centroid + self.transforms[:, :, 2]
        return centroids

    def outlines(self, curved: bool = True, *, n_points: int | None = None) -> Array:
        """(n, M, 2) tile outlines in world coordinates: the prototile outline mapped onto each tile.

        With ``curved`` (the default) the edges are replaced by their curves, see
        [`IsohedralTiling.prototile`][isotile.IsohedralTiling.prototile]; otherwise the outlines are
        [`vertices`][isotile.TileSet.vertices]. Tiles that share an edge share the points of its curve.
        """
        base = self.tiling.prototile(curved=curved, n_points=n_points)
        outlines: Array = np.einsum("nij,vj->nvi", self.transforms[:, :, :2], base) + self.transforms[:, None, :, 2]
        return outlines

    def polygons(self, curved: bool = True, *, n_points: int | None = None) -> list[Any]:
        """The tiles as a list of shapely polygons (needs the ``shapely`` extra), with curved edges by default.

        The polygons are the [`outlines`][isotile.TileSet.outlines] of the tiles.

        Raises
        ------
        ImportError
            If shapely is not installed.
        """
        try:
            import shapely
        except ImportError:
            raise ImportError("TileSet.polygons needs shapely: pip install isotile[shapely]") from None
        return list(shapely.polygons(self.outlines(curved, n_points=n_points)))

    def colors(self) -> IntArray:
        """(n,) color index of each tile from the type's coloring table."""
        table, period = self.tiling.type.coloring, self.tiling.type.coloring[18]
        mu, mv = self.cell[:, 0] % period, self.cell[:, 1] % period
        color: IntArray = table[self.aspect]
        for k in range(period - 1):
            color = np.where(mu > k, table[12 + color], color)
        for k in range(period - 1):
            color = np.where(mv > k, table[15 + color], color)
        return color

    def adjacency(self, kind: AdjacencyKind = "edge") -> csr_array:
        """(n, n) boolean sparse adjacency from the lattice table of the type.

        Pairs where one tile is not in the set are dropped.

        Parameters
        ----------
        kind : {"edge", "vertex", "vertex-only"}, default "edge"
            ``"edge"``: the tiles share an edge; ``"vertex"``: they share an edge or a vertex; ``"vertex-only"``:
            they share exactly a vertex.

        Returns
        -------
        scipy.sparse.csr_array
            Symmetric boolean matrix; entry ``(i, j)`` is true if tile ``j`` is adjacent to tile ``i``.
        """
        return lattice_adjacency(self.cell, self.aspect, self.tiling.type, kind)

    def neighbors(self, i: int, kind: AdjacencyKind = "edge") -> IntArray:
        """Indices of the tiles adjacent to tile ``i``, sorted.

        Parameters
        ----------
        i : int
            Index of the tile in this tile set.
        kind : {"edge", "vertex", "vertex-only"}, default "edge"
            Kind of adjacency, see [`adjacency`][isotile.TileSet.adjacency].

        Notes
        -----
        Each call builds the adjacency matrix of the whole tile set. For the neighbors of many tiles use
        [`neighbor_lists`][isotile.TileSet.neighbor_lists], which builds it once.
        """
        adj = self.adjacency(kind)
        neighbors: IntArray = np.sort(adj.indices[adj.indptr[i] : adj.indptr[i + 1]]).astype(np.int64)
        return neighbors

    def neighbor_lists(self, kind: AdjacencyKind = "edge") -> list[IntArray]:
        """Sorted indices of the adjacent tiles of every tile, from one adjacency matrix.

        Parameters
        ----------
        kind : {"edge", "vertex", "vertex-only"}, default "edge"
            Kind of adjacency, see [`adjacency`][isotile.TileSet.adjacency].

        Returns
        -------
        list of (k,) int arrays
            Entry ``i`` equals ``neighbors(i, kind)``; the list is empty for an empty tile set.
        """
        adj = self.adjacency(kind)
        adj.sort_indices()
        indices = adj.indices.astype(np.int64)
        return list(np.split(indices, adj.indptr[1:-1])) if len(self) else []
