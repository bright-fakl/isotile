"""Isohedral tilings in NumPy: the 81 isohedral types, tile generation and exact tile adjacency."""

from __future__ import annotations

from importlib.metadata import version as _version

from .adjacency import AdjacencyKind, geometric_adjacency, lattice_adjacency
from .curves import EdgeInfo
from .edges import EdgeShape, enforce_symmetry, interpolate, is_symmetric
from .presets import edge_curve_preset, edge_curves_from_preset, list_edge_curve_presets, list_presets
from .prototile import InvalidPrototileError, PrototileReport, check_prototile
from .tiling import IsohedralTiling, TileSet
from .types import TilingType, find_types, get_type, tiling_types

__version__ = _version("isotile")

__all__ = [
    "AdjacencyKind",
    "EdgeInfo",
    "EdgeShape",
    "InvalidPrototileError",
    "IsohedralTiling",
    "PrototileReport",
    "TileSet",
    "TilingType",
    "check_prototile",
    "edge_curve_preset",
    "edge_curves_from_preset",
    "enforce_symmetry",
    "find_types",
    "geometric_adjacency",
    "get_type",
    "interpolate",
    "is_symmetric",
    "lattice_adjacency",
    "list_edge_curve_presets",
    "list_presets",
    "tiling_types",
]
