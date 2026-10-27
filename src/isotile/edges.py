"""Edge curves: the shape constraints, the symmetry projection and the interpolation.

An edge curve is a polyline through control points ``(x, y)`` on the canonical edge, which runs from ``(0, 0)`` to
``(1, 0)``. The shape of an edge restricts the curve:

======  ==============================================================================================
Shape   Constraint
======  ==============================================================================================
``I``   straight: the points ``(0, 0)`` and ``(1, 0)``
``J``   none
``U``   mirror symmetric: point ``k`` is ``(x, y)``, point ``m - 1 - k`` is ``(1 - x, y)``
``S``   point symmetric: point ``k`` is ``(x, y)``, point ``m - 1 - k`` is ``(1 - x, -y)``
======  ==============================================================================================

The symmetries pair points by position along the curve (index ``k`` with ``m - 1 - k``), never by coordinates, so a
curve may turn back in ``x``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, cast

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.interpolate import CubicSpline

Array = NDArray[np.float64]
Shape = Literal["I", "J", "U", "S"]
Mode = Literal["spline", "linear"]

SHAPES: tuple[str, ...] = ("I", "J", "U", "S")
MODES: tuple[str, ...] = ("spline", "linear")
_DUPLICATE_TOL = 1e-12  # consecutive points closer than this are one point for the spline
_ENDPOINT_TOL = 1e-9


def _points(points: ArrayLike) -> Array:
    p = np.array(points, dtype=float)
    if p.ndim != 2 or p.shape[1] != 2 or len(p) < 2:
        raise ValueError(f"an edge curve needs at least 2 points of shape (m, 2), got shape {p.shape}")
    if not np.all(np.isfinite(p)):
        raise ValueError("edge curve points must be finite")
    return p


def _check_shape(shape: str) -> None:
    if shape not in SHAPES:
        raise ValueError(f"unknown edge shape {shape!r}; expected one of {', '.join(SHAPES)}")


def enforce_symmetry(points: ArrayLike, shape: str) -> Array:
    """Project edge curve control points onto the curves that the edge shape allows.

    The projection is by index. For ``U`` and ``S``, point ``k`` of the first half is kept and point ``m - 1 - k``
    is replaced by its mirror image (``U``: ``(1 - x, y)``) or its image under a rotation by 180 degrees about
    ``(0.5, 0)`` (``S``: ``(1 - x, -y)``). The middle point of an odd number of points is moved onto the axis of the
    mirror (``U``: ``x = 0.5``, ``y`` kept) or onto the center of the rotation (``S``: ``(0.5, 0)``). ``J``
    returns the points unchanged and ``I`` the segment ``(0, 0)``, ``(1, 0)``.

    Applying the function to its own result returns the same array, and a curve that already has the symmetry of its
    shape is returned unchanged.

    Parameters
    ----------
    points : (m, 2) array-like
        Control points, ``m >= 2``.
    shape : {"I", "J", "U", "S"}
        Edge shape.

    Returns
    -------
    (m, 2) ndarray
        The projected points (``(2, 2)`` for ``I``).

    Raises
    ------
    ValueError
        If the shape is unknown or ``points`` is not a finite array of at least 2 points of shape ``(m, 2)``.
    """
    _check_shape(shape)
    p = _points(points)
    if shape == "I":
        return np.array([[0.0, 0.0], [1.0, 0.0]])
    if shape == "J":
        return p
    m = len(p)
    half = m // 2
    out = p.copy()
    mirrored = p[:half].copy()
    mirrored[:, 0] = 1.0 - mirrored[:, 0]
    if shape == "S":
        mirrored[:, 1] = -mirrored[:, 1]
    out[m - half :] = mirrored[::-1]
    if m % 2:
        out[half, 0] = 0.5
        if shape == "S":
            out[half, 1] = 0.0
    return out


def is_symmetric(points: ArrayLike, shape: str, tol: float = 1e-9) -> bool:
    """True if the control points have the symmetry of the edge shape, to within ``tol`` in each coordinate.

    ``J`` is always true; ``I`` requires the points to be the segment ``(0, 0)``, ``(1, 0)``. Equivalent to
    ``np.allclose(enforce_symmetry(points, shape), points, rtol=0, atol=tol)`` for the other shapes.

    Parameters
    ----------
    points : (m, 2) array-like
    shape : {"I", "J", "U", "S"}
    tol : float, default 1e-9
        Largest allowed absolute difference.

    Raises
    ------
    ValueError
        As for [`enforce_symmetry`][isotile.enforce_symmetry].
    """
    p = _points(points)
    projected = enforce_symmetry(p, shape)
    if shape == "J":
        return True
    return projected.shape == p.shape and bool(np.all(np.abs(projected - p) <= tol))


def default_n_points(n_control: int) -> int:
    """Number of samples of a spline through ``n_control`` control points: ``max(100, 10 * n_control)``."""
    return max(100, 10 * n_control)


def interpolate(points: ArrayLike, mode: str = "spline", n_points: int | None = None) -> Array:
    """Sample an edge curve.

    ``"spline"`` is a parametric natural cubic spline ``(x(t), y(t))`` through the control points, with ``t`` the
    cumulative chord length scaled to ``[0, 1]``, evaluated at ``n_points`` equally spaced values of ``t``. The
    curve passes through every control point, may turn back in ``x`` and keeps the symmetry of its control points
    (the samples at index ``k`` and ``n_points - 1 - k`` are related by it). Consecutive control points that
    coincide count once. With one or two distinct points the result is the control points. ``"linear"`` returns the
    control points unchanged (``n_points`` is ignored).

    Parameters
    ----------
    points : (m, 2) array-like
        Control points.
    mode : {"spline", "linear"}, default "spline"
    n_points : int, optional
        Number of samples of the spline, at least 2. Default ``max(100, 10 * m)``.

    Returns
    -------
    (k, 2) ndarray

    Raises
    ------
    ValueError
        If the mode is unknown, ``n_points < 2`` or ``points`` is not a finite array of at least 2 points of shape
        ``(m, 2)``.
    """
    if mode not in MODES:
        raise ValueError(f"unknown interpolation mode {mode!r}; expected 'spline' or 'linear'")
    p = _points(points)
    if n_points is not None and n_points < 2:
        raise ValueError("n_points must be at least 2")
    if mode == "linear":
        return p
    keep = np.ones(len(p), dtype=bool)
    keep[1:] = np.linalg.norm(np.diff(p, axis=0), axis=1) > _DUPLICATE_TOL
    q: Array = p[keep]
    if len(q) <= 2:
        return q
    t: Array = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(q, axis=0), axis=1))])
    t /= t[-1]
    spline = CubicSpline(t, q, bc_type="natural")
    n = default_n_points(len(p)) if n_points is None else n_points
    samples: Array = spline(np.linspace(0.0, 1.0, n))
    samples[0], samples[-1] = q[0], q[-1]
    return samples


@dataclass(frozen=True, eq=False)
class EdgeShape:
    """A curve on the canonical edge together with the shape that constrains it.

    Create it with [`EdgeShape.from_points`][isotile.EdgeShape.from_points], which projects the points onto the
    constraint of the shape; the constructor stores the arrays as given.

    Attributes
    ----------
    shape : {"I", "J", "U", "S"}
        Edge shape.
    points : (m, 2) ndarray
        Control points from ``(0, 0)`` to ``(1, 0)`` (read-only), with the symmetry of the shape.
    mode : {"spline", "linear"}
        How the curve between the control points is drawn, see [`interpolate`][isotile.interpolate].
    """

    shape: Shape
    points: Array
    mode: Mode = "spline"

    @classmethod
    def from_points(cls, shape: str, points: ArrayLike | None = None, mode: str = "spline") -> EdgeShape:
        """The curve with the symmetry of ``shape`` that [`enforce_symmetry`][isotile.enforce_symmetry] makes of the points.

        Parameters
        ----------
        shape : {"I", "J", "U", "S"}
        points : (m, 2) array-like, optional
            Control points; the segment when omitted. Ignored for ``"I"``.
        mode : {"spline", "linear"}, default "spline"

        Raises
        ------
        ValueError
            If the shape or mode is unknown, the points are not a finite ``(m, 2)`` array with ``m >= 2`` or the
            projected curve does not run from ``(0, 0)`` to ``(1, 0)``.
        """
        _check_shape(shape)
        if mode not in MODES:
            raise ValueError(f"unknown interpolation mode {mode!r}; expected 'spline' or 'linear'")
        projected = enforce_symmetry([[0.0, 0.0], [1.0, 0.0]] if points is None else points, shape)
        if not (
            np.all(np.abs(projected[0]) <= _ENDPOINT_TOL)
            and np.all(np.abs(projected[-1] - [1.0, 0.0]) <= _ENDPOINT_TOL)
        ):
            raise ValueError(
                f"an edge curve must run from (0, 0) to (1, 0), got {projected[0].tolist()} to {projected[-1].tolist()}"
            )
        projected = projected.copy()
        projected[0], projected[-1] = (0.0, 0.0), (1.0, 0.0)
        projected.setflags(write=False)
        return cls(cast("Shape", shape), projected, cast("Mode", mode))

    def samples(self, n_points: int | None = None) -> Array:
        """The curve as a polyline, see [`interpolate`][isotile.interpolate]."""
        return interpolate(self.points, self.mode, n_points)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, EdgeShape):
            return NotImplemented
        return self.shape == other.shape and self.mode == other.mode and np.array_equal(self.points, other.points)

    __hash__ = None  # type: ignore[assignment]
