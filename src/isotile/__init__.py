"""Prototype of the isotile core: the 81 isohedral types, tile generation and lattice adjacency."""

from .adjacency import geometric_adjacency, lattice_adjacency
from .families import hexagon, quadrilateral, square, triangle
from .tiling import IsohedralTiling, TileSet
from .types import TilingType, get_type, tiling_types

__all__ = [
    "IsohedralTiling",
    "TileSet",
    "TilingType",
    "geometric_adjacency",
    "get_type",
    "hexagon",
    "lattice_adjacency",
    "quadrilateral",
    "square",
    "tiling_types",
    "triangle",
]
