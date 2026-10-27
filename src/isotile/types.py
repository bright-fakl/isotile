"""Tiling type data: the 81 isohedral types, loaded from ``data/types.npz``.

The coefficients in the data file are derived from the tiling data of Tactile / tactile-js
(Copyright (c) 2018, Craig S. Kaplan, BSD 3-Clause; see ``data/LICENSE-tactile``).
"""

from __future__ import annotations

import functools
from collections.abc import Collection, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

DATA_FILE = Path(__file__).parent / "data" / "types.npz"
EDGE_SHAPES = ("J", "U", "S", "I")  # index = code stored in the data file
KIND_EDGE, KIND_VERTEX = 0, 1  # kind column of the adjacency table: shared edge / shared vertex only

EDGE_DESCRIPTIONS = {
    "J": "Free curve",
    "S": "180° rotationally symmetric",
    "U": "Mirror symmetric",
    "I": "Straight (fixed)",
}
_AFFECTED_THRESHOLD = 1e-6  # smaller vertex coefficients count as "no effect"


@dataclass(frozen=True, eq=False)
class TilingType:
    """Static data of one isohedral type. Geometry is ``coeffs @ [*params, 1]``.

    Attributes
    ----------
    number : int
        Type number, 7 for IH7.
    shape_types : tuple of str
        Shape type (``"J"``, ``"U"``, ``"S"`` or ``"I"``) of each edge shape id.
    edge_shape_ids : (n_vertices,) int
        Shape id of each edge.
    edge_orientations : (n_vertices, 2) int
        Flip and rotate flags of each edge (index into the four orientation matrices).
    default_params : (n_params,) float
        Default parameters.
    vertex_coeffs : (n_vertices, 2, n_params + 1) float
        Coefficients of the prototile vertices; the last column is the constant term.
    aspect_coeffs : (n_aspects, 2, 3, n_params + 1) float
        Affine 2x3 matrix of each aspect (tile orientation inside the translation unit).
    translation_coeffs : (2, 2, n_params + 1) float
        Translation vectors ``translation_u`` and ``translation_v``; row k is vector k.
    coloring : (19,) int
        Coloring table: 12 aspect/next-color entries, 3 + 3 transitions, period.
    adjacency : (k, 5) int
        Rows ``(aspect, d_cell_u, d_cell_v, neighbor_aspect, kind)`` for a tile with cell ``(u, v)`` and the given
        aspect: the neighbor has the cell ``(u + d_cell_u, v + d_cell_v)``. ``kind`` is 0 for a shared edge and 1
        for a shared vertex only.
    safe_ranges : (n_params, 2) float
        Per free parameter, the interval ``[low, high]`` around the default value in which the straight prototile
        stays simple while all other parameters keep their defaults; an infinite end (``-inf`` or ``inf``) means
        no limit. A finite end is where the prototile becomes invalid (rounded inward to a multiple of 1e-6 of the
        distance from the default). Beyond a degenerate value the prototile can be valid again (a vertex passing
        through another one), but that is another component of the valid set.

        The intervals are one-parameter slices, not the valid region: combinations of parameters inside their
        intervals can still be invalid (use [`check_prototile`][isotile.check_prototile]), and values outside an
        interval can be valid when other parameters change.
    """

    number: int
    shape_types: tuple[str, ...]
    edge_shape_ids: NDArray[np.int64]
    edge_orientations: NDArray[np.int64]
    default_params: NDArray[np.float64]
    vertex_coeffs: NDArray[np.float64]
    aspect_coeffs: NDArray[np.float64]
    translation_coeffs: NDArray[np.float64]
    coloring: NDArray[np.int64]
    adjacency: NDArray[np.int64]
    safe_ranges: NDArray[np.float64]

    @property
    def n_params(self) -> int:
        """Number of free parameters."""
        return len(self.default_params)

    @property
    def n_vertices(self) -> int:
        """Number of prototile vertices (equal to the number of edges)."""
        return len(self.edge_shape_ids)

    @property
    def n_aspects(self) -> int:
        """Number of aspects, the tiles per cell."""
        return len(self.aspect_coeffs)

    @property
    def n_edge_shapes(self) -> int:
        """Number of distinct edge shape ids."""
        return len(set(self.edge_shape_ids.tolist()))

    @property
    def edge_shapes(self) -> tuple[str, ...]:
        """Shape type of each edge."""
        return tuple(self.shape_types[i] for i in self.edge_shape_ids.tolist())

    @property
    def has_reflections(self) -> bool:
        """True if an aspect is a reflection (negative determinant of its linear part at the defaults)."""
        aspects = self.aspect_coeffs @ np.append(self.default_params, 1.0)
        det = aspects[:, 0, 0] * aspects[:, 1, 1] - aspects[:, 0, 1] * aspects[:, 1, 0]
        return bool(np.any(det < 0))

    def info(self) -> dict[str, Any]:
        """Metadata of the type as a dict.

        Keys: ``num_parameters``, ``num_vertices``, ``num_aspects``, ``num_edge_shapes``, ``edge_shapes``
        (one name per edge), ``default_parameters``, ``edges`` (per edge: ``edge_index``, ``shape_id``,
        ``shape_type``, ``description``, ``customizable``), ``has_reflections``, ``customizable_shape_ids`` and
        ``parameters`` (per parameter: ``index``, ``default``, ``affected_vertices`` with the exact derivative
        ``dx_dp``, ``dy_dp`` of each vertex it moves, rounded to 4 decimals, and a ``description``).
        """
        edges = [
            {
                "edge_index": index,
                "shape_id": int(shape_id),
                "shape_type": shape,
                "description": EDGE_DESCRIPTIONS[shape],
                "customizable": shape != "I",
            }
            for index, (shape, shape_id) in enumerate(zip(self.edge_shapes, self.edge_shape_ids.tolist(), strict=True))
        ]
        parameters = []
        for index in range(self.n_params):
            derivative = self.vertex_coeffs[:, :, index]
            affected = [
                {"vertex": vertex, "dx_dp": round(float(dx), 4), "dy_dp": round(float(dy), 4)}
                for vertex, (dx, dy) in enumerate(derivative)
                if abs(dx) > _AFFECTED_THRESHOLD or abs(dy) > _AFFECTED_THRESHOLD
            ]
            parameters.append({
                "index": index,
                "default": float(self.default_params[index]),
                "affected_vertices": affected,
                "description": _describe_parameter(affected),
            })
        return {
            "num_parameters": self.n_params,
            "num_vertices": self.n_vertices,
            "num_aspects": self.n_aspects,
            "num_edge_shapes": self.n_edge_shapes,
            "edge_shapes": list(self.edge_shapes),
            "default_parameters": self.default_params.tolist(),
            "edges": edges,
            "has_reflections": self.has_reflections,
            "customizable_shape_ids": sorted({e["shape_id"] for e in edges if e["customizable"]}),
            "parameters": parameters,
        }

    def describe(self, parameters: ArrayLike | None = None) -> str:
        """Multi-line human-readable summary of the type.

        With ``parameters`` (``n_params`` values) each parameter line shows its value next to the default.
        """
        info = self.info()
        lines = [f"IH{self.number} — Isohedral Tiling Type {self.number}"]
        if info["num_parameters"] == 0:
            lines.append("  Parameters: none (fixed shape)")
        else:
            lines.append(f"  Parameters: {info['num_parameters']} free")
            values = None if parameters is None else np.asarray(parameters, dtype=float)
            for p in info["parameters"]:
                shown = (
                    f"default={p['default']:.4g}"
                    if values is None
                    else f"value={values[p['index']]:.4g} (default {p['default']:.4g})"
                )
                lines.append(f"    [{p['index']}] {shown}  — {p['description']}")
        lines.append(f"  Prototile: {info['num_vertices']} vertices, {info['num_vertices']} edges")
        reflections = "with reflections" if info["has_reflections"] else "no reflections"
        lines.append(f"  Aspects: {info['num_aspects']} ({reflections})")
        lines.append("  Edges:")
        count_of_id: dict[int, int] = {}
        for edge in info["edges"]:
            count_of_id[edge["shape_id"]] = count_of_id.get(edge["shape_id"], 0) + 1
        for edge in info["edges"]:
            shape_id = edge["shape_id"]
            shared = f", same shape as id {shape_id}" if count_of_id[shape_id] > 1 else ""
            lines.append(
                f"    Edge {edge['edge_index']} (shape_id={shape_id}): {edge['shape_type']} — {edge['description']}{shared}"
            )
        ids = info["customizable_shape_ids"]
        lines.append(
            f"  Customizable shape IDs: {{{', '.join(map(str, ids))}}}"
            if ids
            else "  No customizable edges (all straight)"
        )
        return "\n".join(lines)


def _describe_parameter(affected: list[dict[str, Any]]) -> str:
    if not affected:
        return "no visible effect on vertices"
    parts = []
    for a in affected:
        dx, dy = a["dx_dp"], a["dy_dp"]
        if abs(dy) < _AFFECTED_THRESHOLD:
            direction = "horizontally"
        elif abs(dx) < _AFFECTED_THRESHOLD:
            direction = "vertically"
        else:
            direction = "diagonally"
        parts.append(f"V{a['vertex']} {direction}")
    return "moves " + ", ".join(parts)


def type_from_arrays(arrays: Mapping[str, NDArray[Any]], number: int) -> TilingType:
    """Build the type ``number`` from the arrays of a data file (keys ``t{number}_{name}``)."""

    def get(name: str) -> NDArray[Any]:
        return arrays[f"t{number}_{name}"]

    return TilingType(
        number=number,
        shape_types=tuple(EDGE_SHAPES[c] for c in get("edge_shapes").tolist()),
        edge_shape_ids=get("edge_shape_ids"),
        edge_orientations=get("edge_orientations"),
        default_params=get("default_params"),
        vertex_coeffs=get("vertex_coeffs"),
        aspect_coeffs=get("aspect_coeffs"),
        translation_coeffs=get("translation_coeffs"),
        coloring=get("coloring"),
        adjacency=get("adjacency"),
        safe_ranges=get("safe_ranges"),
    )


@functools.cache
def _load() -> dict[int, TilingType]:
    with np.load(DATA_FILE) as z:
        arrays = {key: z[key] for key in z.files}
    return {n: type_from_arrays(arrays, n) for n in arrays["numbers"].tolist()}


def tiling_types() -> tuple[int, ...]:
    """The 81 valid type numbers."""
    return tuple(_load())


def get_type(number: int) -> TilingType:
    """The data of type ``number``; raises ``ValueError`` for a number that is not one of the 81 types."""
    try:
        return _load()[number]
    except KeyError:
        raise ValueError(f"Invalid isohedral tiling type: {number}") from None


def find_types(
    *,
    num_vertices: int | None = None,
    num_parameters: int | None = None,
    max_parameters: int | None = None,
    num_aspects: int | None = None,
    has_reflections: bool | None = None,
    edge_types: Collection[str] | None = None,
    all_edges: str | None = None,
    has_customizable_edges: bool | None = None,
) -> list[int]:
    """Numbers of the types that satisfy every given filter, sorted.

    Parameters
    ----------
    num_vertices : int, optional
        Exact number of prototile vertices (3, 4, 5 or 6).
    num_parameters : int, optional
        Exact number of free parameters.
    max_parameters : int, optional
        Maximum number of free parameters.
    num_aspects : int, optional
        Exact number of aspects.
    has_reflections : bool, optional
        True: only types with a reflected aspect; False: only types without.
    edge_types : collection of str, optional
        Edge shapes (``"J"``, ``"U"``, ``"S"``, ``"I"``) that the type must all contain.
    all_edges : str, optional
        Only types in which every edge has this shape.
    has_customizable_edges : bool, optional
        True: at least one edge is J, U or S; False: all edges are I.
    """
    required = None if edge_types is None else set(edge_types)
    matches = []
    for number in sorted(_load()):
        t = _load()[number]
        if num_vertices is not None and t.n_vertices != num_vertices:
            continue
        if num_parameters is not None and t.n_params != num_parameters:
            continue
        if max_parameters is not None and t.n_params > max_parameters:
            continue
        if num_aspects is not None and t.n_aspects != num_aspects:
            continue
        shapes = set(t.edge_shapes)
        if required is not None and not required <= shapes:
            continue
        if all_edges is not None and shapes != {all_edges}:
            continue
        if has_customizable_edges is not None and bool(shapes - {"I"}) != has_customizable_edges:
            continue
        if has_reflections is not None and t.has_reflections != has_reflections:
            continue
        matches.append(number)
    return matches
