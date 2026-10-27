# isotile

Isohedral tilings in NumPy: the 81 isohedral tiling types, vectorized tile generation and exact tile adjacency from a
lattice table. isotile keeps the tiling type numbers (IH1 to IH93) and the parameter format of
[tactile](https://github.com/isohedral/tactile).

- Tiles are generated for a rectangle or for a target tile count, as arrays of cells, aspects and affine transforms.
- Adjacency (shared edge, shared vertex, vertex only) comes from a table of the tiling type, as a scipy sparse boolean
  matrix. It does not depend on the parameters, the scale or the position of the tiling.
- Squares, hexagons, any triangle and any simple quadrilateral are named constructors of the same class.
- A parameter set that makes the prototile self-intersecting raises `ValueError`; `check_prototile` reports the edges
  at fault without constructing a tiling.

## Install

```bash
pip install isotile            # numpy and scipy
pip install "isotile[shapely]" # adds TileSet.polygons()
```

Python 3.10 or newer.

## Example

Generate a tiling of type IH7 that covers a region, and read the adjacency of its tiles.

```python
from isotile import IsohedralTiling

tiling = IsohedralTiling(7)
tiles = tiling.fill_region((0, 0, 10, 6), n_tiles=200)
print(len(tiles), tiles.vertices().shape)

edge = tiles.adjacency("edge")  # scipy.sparse.csr_array of bool, (n, n)
print(edge.shape, edge.nnz)
print(tiles.neighbors(0, "edge"))  # indices of the tiles that share an edge with tile 0
```

Where to continue:

- [Quickstart](quickstart.md): the common calls, step by step.
- [Concepts](concepts.md): types, parameters, cells, aspects, frames and the tile selection.
- [Adjacency](adjacency.md): the three kinds of adjacency and how they are computed.
- [API reference](api/index.md): every public name.
