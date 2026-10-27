"""Isohedral tilings in NumPy: the 81 isohedral types, tile generation and exact tile adjacency."""

from __future__ import annotations

from importlib.metadata import version as _version

from .adjacency import AdjacencyKind, geometric_adjacency, lattice_adjacency
from .prototile import PrototileReport, check_prototile
from .tiling import IsohedralTiling, TileSet
from .types import TilingType, find_types, get_type, tiling_types

__version__ = _version("isotile")

__all__ = [
    "AdjacencyKind",
    "IsohedralTiling",
    "PrototileReport",
    "TileSet",
    "TilingType",
    "check_prototile",
    "find_types",
    "geometric_adjacency",
    "get_type",
    "lattice_adjacency",
    "tiling_types",
]
