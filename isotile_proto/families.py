"""Named constructors: square, hexagon, triangle and quadrilateral tilings as isohedral types.

Square: IH76. Regular hexagon: IH20. Any triangle: IH84 (three S edges, point reflection about every edge midpoint).
Any simple quadrilateral: IH46 (four S edges). The triangle and quadrilateral types have free vertex parameters, so a
polygon is expressed as parameters plus a similarity frame.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .tiling import IsohedralTiling
from .types import get_type

SQUARE, HEXAGON, TRIANGLE, QUADRILATERAL = 76, 20, 84, 46


def _similarity(z: complex, t: complex) -> NDArray[np.float64]:
    """(2, 3) matrix of x -> z * x + t on points written as complex numbers."""
    return np.array([[z.real, -z.imag, t.real], [z.imag, z.real, t.imag]])


def square(size: float = 1.0, center: tuple[float, float] = (0.0, 0.0)) -> IsohedralTiling:
    """Square grid with side ``size``; the tile (0, 0, 0) is centered at ``center``."""
    return IsohedralTiling(
        SQUARE, frame=_similarity(complex(size), complex(center[0] - size / 2, center[1] - size / 2))
    )


def hexagon(pointy_top: bool = True, size: float = 1.0, center: tuple[float, float] = (0.0, 0.0)) -> IsohedralTiling:
    """Regular hexagons with circumradius ``size``; the tile (0, 0, 0) is centered at ``center``."""
    z = size * (complex(math.cos(math.pi / 6), math.sin(math.pi / 6)) if pointy_top else 1.0)
    return IsohedralTiling(HEXAGON, frame=_similarity(z, complex(*center)))


def _fit(number: int, polygon: ArrayLike) -> IsohedralTiling:
    """Type ``number`` whose prototile is exactly ``polygon`` (a similar copy with parameters, then a frame)."""
    tt = get_type(number)
    pts = np.asarray(polygon, dtype=float)
    if pts.shape != (tt.n_vertices, 2):
        raise ValueError(f"IH{number} needs {tt.n_vertices} vertices, got shape {pts.shape}")
    x, y = pts[:, 0], pts[:, 1]
    if np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y) < 0:
        pts = pts[::-1]  # the type's prototile is counterclockwise
    lengths = np.linalg.norm(np.roll(pts, -1, axis=0) - pts, axis=1)
    pts = np.roll(pts, -int(np.argmax(lengths)), axis=0)  # longest edge first: keeps parameters moderate
    const = tt.vertex_coeffs[:, :, -1]
    if np.any(tt.vertex_coeffs[:2, :, :-1]):
        raise ValueError(f"IH{number}: the first edge depends on the parameters")
    p = pts[:, 0] + 1j * pts[:, 1]
    n = const[:, 0] + 1j * const[:, 1]
    z = (n[1] - n[0]) / (p[1] - p[0])  # similarity p -> native
    w = n[0] - z * p[0]
    q = z * p[2:] + w
    a = tt.vertex_coeffs[2:, :, :-1].reshape(-1, tt.n_params)
    b = (np.column_stack([q.real, q.imag]) - const[2:]).reshape(-1)
    params, *_ = np.linalg.lstsq(a, b, rcond=None)
    if not np.allclose(a @ params, b, atol=1e-9):
        raise ValueError(f"IH{number} cannot express this polygon")
    return IsohedralTiling(number, params, frame=_similarity(1 / z, -w / z))


def triangle(vertices: ArrayLike) -> IsohedralTiling:
    """Tiling by congruent copies of the triangle ``vertices`` (3, 2), related by point reflections (IH84)."""
    return _fit(TRIANGLE, vertices)


def quadrilateral(vertices: ArrayLike) -> IsohedralTiling:
    """Tiling by congruent copies of the simple quadrilateral ``vertices`` (4, 2), point reflections (IH46)."""
    return _fit(QUADRILATERAL, vertices)
