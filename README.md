# isotile

Isohedral tilings in NumPy: the 81 isohedral tiling types, vectorized tile generation and exact tile adjacency from
a lattice table. It replaces [tactile](https://github.com/isohedral/tactile) for these tasks and keeps its tiling
type numbers (IH1 to IH93) and parameter format.

- Tiles are generated for a rectangle or for a target tile count, as arrays of cells, aspects and affine transforms.
- Adjacency (shared edge, shared vertex, vertex only) comes from a table of the tiling type, as a scipy sparse
  boolean matrix. It does not depend on the parameters, the scale or the position of the tiling.
- Squares, hexagons, any triangle and any simple quadrilateral are named constructors of the same class.
- A parameter set that makes the prototile self-intersecting raises `ValueError`; `check_prototile` reports the
  edges at fault without constructing a tiling.

## Install

```
pip install isotile            # numpy and scipy
pip install "isotile[shapely]" # adds TileSet.polygons()
```

Python 3.10 or newer.

## Quickstart

```python
from isotile import IsohedralTiling, check_prototile, find_types

tiling = IsohedralTiling(7)  # type IH7 at its default parameters
print(tiling.describe())

tiles = tiling.fill_region((0, 0, 10, 6), n_tiles=200)  # tiles that cover the region
print(len(tiles), tiles.cell.shape, tiles.aspect.shape, tiles.transforms.shape)

polygons = tiles.vertices()  # (n, n_vertices, 2) tile outlines
edge = tiles.adjacency("edge")  # scipy.sparse.csr_array of bool, (n, n)
print(edge.sum(axis=1).max(), tiles.neighbors(0, "vertex"))

hexagons = IsohedralTiling.hexagon(pointy_top=False)
triangles = IsohedralTiling.triangle([(0, 0), (3, 0), (1, 2)])  # any triangle
print(find_types(num_vertices=3, all_edges="S"))
print(check_prototile(7, [0.25, -1.0]))  # PrototileReport(valid=False, ...)
```

## Concepts

A tiling is one of 81 types with free parameters that change the shape of the prototile. The lattice is spanned by
two translation vectors (`translation_u`, `translation_v`). A tile is identified by its cell `(u, v)` and its
aspect, the index of the tile within the cell. A tile set holds `cell` (n, 2), `aspect` (n,) and the affine
`transforms` (n, 2, 3) of the prototile onto each tile.

`IsohedralTiling(type, parameters, frame=...)` takes an affine `frame` that maps the type's native coordinates to
world coordinates (scale, rotation, reflection, position). `fill_region(bounds, n_tiles=... | tile_size=...,
margin=...)` computes the frame from a region and a tile count or size.

## License

MIT. The tiling type data is derived from Tactile / tactile-js by Craig S. Kaplan (BSD 3-Clause); see `NOTICE`.
