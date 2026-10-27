"""Tiling type data: the 81 isohedral types, loaded from ``data/types.npz``.

The coefficients in the data file are derived from the tiling data of Tactile / tactile-js
(Copyright (c) 2018, Craig S. Kaplan, BSD 3-Clause; see ``data/LICENSE-tactile``).
"""

from __future__ import annotations

import functools
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

DATA_FILE = Path(__file__).parent / "data" / "types.npz"
EDGE_SHAPES = ("J", "U", "S", "I")  # index = code stored in the data file
KIND_EDGE, KIND_VERTEX = 0, 1  # kind column of the adjacency table: shared edge / shared vertex only


@dataclass(frozen=True, eq=False)
class TilingType:
    """Static data of one isohedral type. Geometry is ``coeffs @ [*params, 1]``.

    Attributes
    ----------
    vertex_coeffs : (n_vertices, 2, n_params + 1)
    aspect_coeffs : (n_aspects, 2, 3, n_params + 1)
        Affine 2x3 matrix of each aspect (tile orientation inside the translation unit).
    translation_coeffs : (2, 2, n_params + 1)
        Translation vectors t1, t2.
    edge_orientations : (n_vertices, 2)
        Flip and rotate flags of each edge (index into tactile's four orientation matrices).
    coloring : (19,)
        Tactile's coloring table (12 aspect/next-color entries, 3 + 3 transitions, period).
    adjacency : (k, 5)
        Rows ``(aspect, d_t1, d_t2, neighbor_aspect, kind)`` for a tile ``(t1, t2, aspect)``.
    """

    number: int
    edge_shapes: tuple[str, ...]
    edge_shape_ids: NDArray[np.int64]
    edge_orientations: NDArray[np.int64]
    default_params: NDArray[np.float64]
    vertex_coeffs: NDArray[np.float64]
    aspect_coeffs: NDArray[np.float64]
    translation_coeffs: NDArray[np.float64]
    coloring: NDArray[np.int64]
    adjacency: NDArray[np.int64]

    @property
    def n_params(self) -> int:
        return len(self.default_params)

    @property
    def n_vertices(self) -> int:
        return len(self.edge_shape_ids)

    @property
    def n_aspects(self) -> int:
        return len(self.aspect_coeffs)


@functools.cache
def _load() -> dict[int, TilingType]:
    with np.load(DATA_FILE) as z:
        types = {}
        for n in z["numbers"].tolist():
            get = lambda name, n=n: z[f"t{n}_{name}"]
            types[n] = TilingType(
                number=n,
                edge_shapes=tuple(EDGE_SHAPES[c] for c in get("edge_shapes")),
                edge_shape_ids=get("edge_shape_ids"),
                edge_orientations=get("edge_orientations"),
                default_params=get("default_params"),
                vertex_coeffs=get("vertex_coeffs"),
                aspect_coeffs=get("aspect_coeffs"),
                translation_coeffs=get("translation_coeffs"),
                coloring=get("coloring"),
                adjacency=get("adjacency"),
            )
    return types


def tiling_types() -> tuple[int, ...]:
    """The 81 valid type numbers."""
    return tuple(_load())


def get_type(number: int) -> TilingType:
    try:
        return _load()[number]
    except KeyError:
        raise ValueError(f"Invalid isohedral tiling type: {number}") from None
