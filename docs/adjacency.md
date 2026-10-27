# Adjacency

Two tiles of a tiling are adjacent in one of three ways:

| Kind | Meaning |
|---|---|
| `"edge"` | The tiles share an edge. |
| `"vertex"` | The tiles share an edge or a vertex. |
| `"vertex-only"` | The tiles share exactly a vertex and no edge. |

`"vertex"` is the union of `"edge"` and `"vertex-only"`. The kinds are the type `AdjacencyKind`.

<figure markdown="span">
  ![Two panels. Squares: an amber tile with four blue edge neighbors and four green vertex-only neighbors. IH28: an amber pentagon with five blue edge neighbors and two green vertex-only neighbors.](assets/figures/adjacency-neighbors-light.svg#only-light)
  ![Two panels. Squares: an amber tile with four blue edge neighbors and four green vertex-only neighbors. IH28: an amber pentagon with five blue edge neighbors and two green vertex-only neighbors.](assets/figures/adjacency-neighbors-dark.svg#only-dark)
  <figcaption>One tile (amber), its edge neighbors (blue) and its vertex-only neighbors (green) in the square tiling and in IH28. The number of neighbors depends on the type and the aspect.</figcaption>
</figure>

## The lattice table

In an isohedral tiling the neighbors of a tile depend only on the aspect of the tile, not on its cell. Every type
therefore has a table (`TilingType.adjacency`) with one row per neighbor relation:

```
aspect, d_cell_u, d_cell_v, neighbor_aspect, kind
```

A tile with cell `(u, v)` and the given aspect has a neighbor with cell `(u + d_cell_u, v + d_cell_v)` and aspect
`neighbor_aspect`. `kind` is 0 if the two tiles share an edge and 1 if they share a vertex only. The table is symmetric:
if tile A lists tile B, tile B lists tile A.

<figure markdown="span">
  ![A pentagonal tiling with an amber tile labeled cell (0, 0), aspect 0, surrounded by its seven neighbors. Each neighbor is labeled with its cell offset and its aspect; five are blue and two are green.](assets/figures/adjacency-offsets-light.svg#only-light)
  ![A pentagonal tiling with an amber tile labeled cell (0, 0), aspect 0, surrounded by its seven neighbors. Each neighbor is labeled with its cell offset and its aspect; five are blue and two are green.](assets/figures/adjacency-offsets-dark.svg#only-dark)
  <figcaption>The rows of <code>get_type(28).adjacency</code> for aspect 0: each neighbor is labeled with <code>(d_cell_u, d_cell_v)</code> and <code>neighbor_aspect</code>. Edge neighbors (<code>kind</code> 0) are blue, vertex-only neighbors (<code>kind</code> 1) are green.</figcaption>
</figure>

The offsets are in cell coordinates and the aspect is an index, so the table does not depend on the parameters, the
scale or the frame of the tiling, or on where the tiles are. Two tilings of the same type with different parameters,
scales or rotations have the same adjacency for the same cells and aspects. The table needs no geometry, and it is exact:
there is no tolerance, and shared vertices that coincide up to rounding are never missed or merged.

For the square tiling (type 76) there is one aspect and the table has eight rows: the offsets `(+-1, 0)` and `(0, +-1)`
with kind 0 (four edge neighbors) and `(+-1, +-1)` with kind 1 (four vertex-only neighbors).

```python
from isotile import IsohedralTiling, get_type

print(get_type(76).adjacency)
```

`TileSet.adjacency(kind)` looks up, for every tile of the set and every row of the table for its aspect, whether the tile
at the neighboring cell and aspect is in the set. A tile that is not in the set is dropped, so tiles at the border of a
set have fewer neighbors. `lattice_adjacency(cell, aspect, tiling_type, kind)` does the same for arrays of cells and
aspects without a tiling object.

## The matrix format

The result is a `scipy.sparse.csr_array` of dtype `bool` and shape `(n, n)`, with `n` the number of tiles. Row and column
`i` refer to tile `i` of the tile set, in the order of `TileSet.cell`. The matrix is symmetric and its diagonal is empty.
The indices of the neighbors of tile `i` are `adjacency.indices[adjacency.indptr[i]:adjacency.indptr[i + 1]]`, which
`TileSet.neighbors(i, kind)` returns sorted.

## Worked example

Four squares in a 2 x 2 block and one square far away, as tiles of the square tiling.

```python
tiles = IsohedralTiling.square().tiles([[0, 0], [1, 0], [0, 1], [1, 1], [3, 3]], 0)

edge = tiles.adjacency("edge")
vertex = tiles.adjacency("vertex")
only = tiles.adjacency("vertex-only")

print(edge.toarray().astype(int))
print(only.toarray().astype(int))
```

<figure markdown="span">
  ![Left: five squares numbered 0 to 4; tiles 0 to 3 form a 2 by 2 block and tile 4 is apart. Right: the 5 by 5 matrix with entries 0-1, 0-2, 1-3 and 2-3 and their mirror entries blue for edge neighbors, entries 0-3 and 1-2 and their mirrors green for vertex-only, and row and column 4 empty.](assets/figures/adjacency-matrix-light.svg#only-light)
  ![Left: five squares numbered 0 to 4; tiles 0 to 3 form a 2 by 2 block and tile 4 is apart. Right: the 5 by 5 matrix with entries 0-1, 0-2, 1-3 and 2-3 and their mirror entries blue for edge neighbors, entries 0-3 and 1-2 and their mirrors green for vertex-only, and row and column 4 empty.](assets/figures/adjacency-matrix-dark.svg#only-dark)
  <figcaption>The tiles of the worked example and the matrices <code>edge</code> (blue) and <code>only</code> (green) as one grid. Row and column <code>i</code> are tile <code>i</code>; both matrices are symmetric with an empty diagonal.</figcaption>
</figure>

Tile 0 (cell `(0, 0)`) has edge neighbors 1 and 2 (cells `(1, 0)` and `(0, 1)`) and the vertex-only neighbor 3 (cell
`(1, 1)`). The tile at `(3, 3)` has no neighbor in the set. The same cells and aspects on a scaled and rotated tiling
give the same matrices.

```python
moved = IsohedralTiling.square().scaled(100).rotated(33).tiles(tiles.cell, tiles.aspect)

print(tiles.neighbors(0, "edge"), tiles.neighbors(0, "vertex-only"))  # [1 2] [3]
assert (moved.adjacency("vertex") != vertex).nnz == 0
assert (vertex != edge + only).nnz == 0
```

## The geometric method

`geometric_adjacency(polygons, tol=1e-6)` computes adjacency for any list of polygons, each given as a `(k, 2)` array of
vertices, and returns the pair `(edge, vertex_only)` of sparse boolean matrices of the same format. It identifies
vertices that are closer than `tol` (in the units of the coordinates) as one vertex, and counts for each pair of polygons
the vertices they share. Two polygons that share two or more vertices are edge neighbors; two that share exactly one are
vertex-only neighbors. Only the vertices are compared: a vertex of one polygon that lies on the interior of an edge of
another polygon does not count as shared. Choose `tol` smaller than the shortest edge and larger than the rounding error
of the coordinates.

```python
from isotile import geometric_adjacency

edge_g, only_g = geometric_adjacency(list(tiles.vertices()))
assert (edge_g != edge).nnz == 0
assert (only_g != only).nnz == 0
```

For the tiles of an `IsohedralTiling` the two methods give the same matrices. The lattice method is the one to use for
tiles of a tiling: it does not depend on a tolerance and does not compare coordinates. The geometric method serves polygons
that no tiling generated.
