# API reference

Every public name is importable from the top-level package `isotile`; the modules `isotile.io` and `isotile.plot` are imported explicitly.

| Page | Names |
|---|---|
| [Tilings and tile sets](tiling.md) | `IsohedralTiling`, `TileSet` |
| [Tiling types](types.md) | `TilingType`, `get_type`, `tiling_types`, `find_types` |
| [Prototile validity](prototile.md) | `check_prototile`, `PrototileReport` |
| [Edge curves](edges.md) | `EdgeShape`, `EdgeInfo`, `enforce_symmetry`, `is_symmetric`, `interpolate` |
| [Presets](presets.md) | `list_presets`, `list_edge_curve_presets`, `edge_curve_preset`, `edge_curves_from_preset` |
| [Serialization and plotting](io.md) | `isotile.io`, `isotile.plot` |
| [Adjacency](adjacency.md) | `AdjacencyKind`, `lattice_adjacency`, `geometric_adjacency` |

`isotile.__version__` holds the installed version.
