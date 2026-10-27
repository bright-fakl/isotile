"""Ready-made tilings and edge curves.

The tiling presets are named configurations (type, parameters, edge curves) built by
[`IsohedralTiling.from_preset`][isotile.IsohedralTiling.from_preset]. The edge curve presets are curves with a name
that [`edge_curve_preset`][isotile.edge_curve_preset] returns for any edge shape, with the constraint of the shape
enforced.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from .edges import enforce_symmetry
from .types import TilingType, get_type

Array = NDArray[np.float64]

_SQUARE_KNOB = [
    (0, 0),
    (0.35, 0),
    (0.35, 0.06),
    (0.38, 0.12),
    (0.44, 0.16),
    (0.5, 0.18),
    (0.56, 0.16),
    (0.62, 0.12),
    (0.65, 0.06),
    (0.65, 0),
    (1, 0),
]
_WAVE_5 = [(0, 0), (0.25, 0.12), (0.5, 0), (0.75, -0.12), (1, 0)]
_WAVE_5_LARGE = [(0, 0), (0.25, 0.15), (0.5, 0), (0.75, -0.15), (1, 0)]
_BULGE_5 = [(0, 0), (0.25, 0.08), (0.5, 0.12), (0.75, 0.08), (1, 0)]

# name -> configuration; the keys of a configuration are the arguments of IsohedralTiling.
_PRESETS: dict[str, dict[str, Any]] = {
    "regular_hexagon": {
        "type": 10,
        "description": "IH10. Regular hexagon with straight edges; no free parameters.",
    },
    "square": {
        "type": 41,
        "description": "IH41. Square with straight edges at the default parameters; 2 free parameters.",
    },
    "equilateral_triangle": {
        "type": 93,
        "description": "IH93. Equilateral triangle with straight edges; no free parameters.",
    },
    "rhombus": {
        "type": 68,
        "description": "IH68. Rhombus with straight edges; 1 free parameter sets the acute angle.",
    },
    "cairo": {
        "type": 28,
        "description": "IH28. Pentagon of the Cairo tiling with straight edges; 2 free parameters.",
    },
    "wide_parallelogram": {
        "type": 43,
        "parameters": [0.4, 0.3],
        "description": "IH43. Wide parallelogram with 2 aspects related by a rotation of 180 degrees.",
    },
    "scalloped_hexagon": {
        "type": 10,
        "edge_curves": {0: [(0, 0), (0.2, 0.08), (0.5, 0.12), (0.8, 0.08), (1, 0)]},
        "description": "IH10. Regular hexagon whose six edges bulge outward in the same curve.",
    },
    "wavy_square": {
        "type": 41,
        "edge_curves": {0: _WAVE_5, 1: [(0, 0), (0.25, -0.12), (0.5, 0), (0.75, 0.12), (1, 0)]},
        "description": "IH41. Square with zigzag edges; opposite edges use mirrored curves that interlock.",
    },
    "puzzle_piece": {
        "type": 41,
        "edge_curves": {0: _SQUARE_KNOB, 1: _SQUARE_KNOB},
        "description": "IH41. Square with a knob on every edge that fits the socket of the neighbor.",
    },
    "arrow_quad": {
        "type": 43,
        "edge_curves": {
            0: [(0, 0), (0.3, 0), (0.5, 0.15), (0.7, 0), (1, 0)],
            1: [(0, 0), (0.3, 0), (0.5, -0.15), (0.7, 0), (1, 0)],
        },
        "edge_curve_modes": {0: "linear", 1: "linear"},
        "description": "IH43. Quadrilateral with a triangular point or notch in the middle of each edge.",
    },
    "wavy_triangle": {
        "type": 84,
        "edge_curves": {0: _WAVE_5_LARGE, 1: _WAVE_5_LARGE, 2: _WAVE_5_LARGE},
        "description": "IH84. Triangle with S-shaped wavy edges; 2 aspects related by a rotation of 180 degrees.",
    },
    "interlocking_hexagon": {
        "type": 10,
        "edge_curves": {0: [(0, 0), (-0.12, 0.2), (0.5, 0.6), (1.12, 0.2), (1, 0)]},
        "description": "IH10. Regular hexagon whose six edges bulge strongly into a lens shape.",
    },
    "flowers": {
        "type": 21,
        "parameters": [0.104512294489, 0.65],
        "edge_curves": {
            0: [(0, 0), (0.25, 0.08), (0.5, 0.0), (0.75, -0.08), (1.0, 0)],
            1: _BULGE_5,
            2: _BULGE_5,
        },
        "description": "IH21. Pentagon with one S-shaped edge and two pairs of gently scalloped edges; 6 aspects.",
    },
    "scalloped_quad": {
        "type": 33,
        "edge_curves": {0: _BULGE_5, 1: _BULGE_5},
        "description": "IH33. Quadrilateral whose four edges bulge outward; 3 aspects, no free parameters.",
    },
    "tree": {
        "type": 4,
        "parameters": [0.027230055765024366, 0.5, 0.315470053838, 0.5, 0.315470053838, 0.5],
        "edge_curves": {
            0: [(0, 0), (0.17, -0.16), (0.5, 0.0), (0.83, 0.16), (1.0, 0)],
            1: [(0, 0), (0.5, 0.3), (1, 0)],
            2: [(0, 0), (0.17, 0.16), (0.5, 0.0), (0.83, -0.16), (1.0, 0)],
            3: [(0, 0), (0.2, 0.1), (0.4, -0.1), (0.5, 0.0), (0.6, 0.1), (0.8, -0.1), (1.0, 0)],
            4: [(0, 0), (0.2, -0.1), (0.4, 0.1), (0.5, 0.0), (0.6, -0.1), (0.8, 0.1), (1.0, 0)],
        },
        "description": "IH4. Hexagon with 5 edge shapes whose S-curves and wavy edges give a tree-like outline; 2 aspects.",
    },
}


def list_presets() -> dict[str, str]:
    """The names of the tiling presets and their descriptions."""
    return {name: p["description"] for name, p in _PRESETS.items()}


def preset_arguments(name: str) -> dict[str, Any]:
    """The constructor arguments (``tiling_type``, optionally ``parameters``, ``edge_curves``, ``edge_curve_modes``) of a preset."""
    if name not in _PRESETS:
        raise ValueError(f"unknown preset {name!r}; available presets: {', '.join(sorted(_PRESETS))}")
    p = _PRESETS[name]
    args: dict[str, Any] = {"tiling_type": p["type"]}
    for key in ("parameters", "edge_curves", "edge_curve_modes"):
        if key in p:
            args[key] = p[key]
    return args


# ---- Edge curve presets -------------------------------------------------------------------------------------

_N_CONTROL = 21  # control points of a preset curve; odd, so the middle point is on the axis of symmetry

_CURVES: dict[str, str] = {
    "straight": "The straight edge.",
    "bump": "One half sine wave, a bump of height `amplitude` to the left of the direction of travel.",
    "dent": "A bump of depth `amplitude` to the right of the direction of travel.",
    "wave": "One period of a sine wave: a bump followed by a dent.",
    "s_curve": "A cubic S: a dip followed by a rise of the same height, with a gentle slope at both ends.",
}


def list_edge_curve_presets() -> dict[str, str]:
    """The names of the edge curve presets and their descriptions."""
    return dict(_CURVES)


def edge_curve_preset(name: str, shape: str = "J", amplitude: float = 0.1) -> Array:
    """Control points of a named curve on the canonical edge, with the constraint of the edge shape enforced.

    The curve is built for a free (``J``) edge and then projected by
    [`enforce_symmetry`][isotile.enforce_symmetry]: for ``U`` and ``S`` the first half of the curve is kept and the
    second half generated, so a ``wave`` on a ``U`` edge becomes two bumps and a ``bump`` on an ``S`` edge becomes a
    bump followed by a dent. Whether a curve gives a valid prototile depends on the type and the amplitude; test it
    with [`check_prototile`][isotile.check_prototile].

    Parameters
    ----------
    name : str
        One of [`list_edge_curve_presets`][isotile.list_edge_curve_presets].
    shape : {"I", "J", "U", "S"}, default "J"
        Edge shape; ``"I"`` returns the segment.
    amplitude : float, default 0.1
        Largest distance of the curve from the segment, as a fraction of the edge length.

    Returns
    -------
    (m, 2) ndarray
        Control points from ``(0, 0)`` to ``(1, 0)`` (21 points, or 2 for ``"straight"`` and for ``"I"``).

    Raises
    ------
    ValueError
        If the name or shape is unknown.
    """
    if name not in _CURVES:
        raise ValueError(f"unknown edge curve preset {name!r}; available presets: {', '.join(_CURVES)}")
    x = np.linspace(0.0, 1.0, _N_CONTROL)
    if name == "straight":
        return enforce_symmetry(np.array([[0.0, 0.0], [1.0, 0.0]]), "I")
    if name in ("bump", "dent"):
        y = amplitude * np.sin(np.pi * x) * (1.0 if name == "bump" else -1.0)
    elif name == "wave":
        y = amplitude * np.sin(2.0 * np.pi * x)
    else:
        t = 2.0 * x - 1.0
        y = -amplitude * (3.0 * np.sqrt(3.0) / 2.0) * t * (1.0 - t * t)
    y[[0, -1]] = 0.0
    return enforce_symmetry(np.column_stack([x, y]), shape)


def edge_curves_from_preset(tiling_type: int | TilingType, name: str, amplitude: float = 0.1) -> dict[int, Array]:
    """The curve of a preset for every shape id of a type that can be curved, as the ``edge_curves`` argument.

    Parameters
    ----------
    tiling_type : int or TilingType
    name : str
        One of [`list_edge_curve_presets`][isotile.list_edge_curve_presets].
    amplitude : float, default 0.1
        See [`edge_curve_preset`][isotile.edge_curve_preset].

    Returns
    -------
    dict of int to (m, 2) ndarray
        Shape ids of ``J``, ``U`` and ``S`` edges; empty for a type with only ``I`` edges.
    """
    t = tiling_type if isinstance(tiling_type, TilingType) else get_type(tiling_type)
    used = sorted(set(t.edge_shape_ids.tolist()))
    return {i: edge_curve_preset(name, t.shape_types[i], amplitude) for i in used if t.shape_types[i] != "I"}
