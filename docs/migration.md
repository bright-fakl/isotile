# Migrating from tactile

isotile keeps the tiling type numbers (IH1 to IH93, 81 of them), the parameter format and the geometry of
[tactile](https://github.com/isohedral/tactile): vertices, aspects, translation vectors, edge transforms and the tile
order of the scanline fill agree with tactile for all types. What changes is the interface: results are NumPy arrays
instead of lists of points and generators, and tiles are returned as one `TileSet`.

The code blocks marked `py` use tactile and are not run; the `python` blocks use isotile and run with the pages.

## Create a tiling and set parameters

```py
import tactile

tiling = tactile.IsohedralTiling(7)
tiling.parameters = [0.5, 0.2]  # must be a Python list
print(tiling.num_parameters, tiling.vertices)  # list of Point
```

```python
from isotile import IsohedralTiling

tiling = IsohedralTiling(7)
tiling.parameters = [0.5, 0.2]  # any array-like, stored as float64
print(tiling.type.n_params, tiling.vertices.shape)  # (6, 2) array
```

The `parameters` setter checks the length and that the prototile is a simple polygon, and keeps the previous state when
it raises `InvalidPrototileError` (a `ValueError`). `with_parameters` returns a changed copy and leaves the tiling
unchanged. tactile does not check the prototile; use `check_prototile` to test a parameter set without constructing a
tiling.

## Geometry of the prototile

| tactile | isotile |
|---|---|
| `tactile.tiling_types` (list) | `tiling_types()` (tuple of the 81 numbers) |
| `tiling.num_parameters`, `num_vertices`, `num_aspects`, `num_edge_shapes` | `tiling.type.n_params`, `n_vertices`, `n_aspects`, `n_edge_shapes` |
| `tiling.vertices` (list of `Point`) | `tiling.vertices`, an `(n_vertices, 2)` array |
| `tiling.aspects` (list of 6-lists) | `tiling.aspects`, an `(n_aspects, 2, 3)` array of affine matrices |
| `tiling.t1`, `tiling.t2` | `tiling.translation_u`, `tiling.translation_v` |
| `tiling.edges`, `tiling.reversals` | `tiling.edge_transforms` `(n_vertices, 2, 3)`, `tiling.edge_reversed` |
| `tiling.shapes`, `tiling.parts` (generators of `Shape`) | `tiling.edges` (tuple of `EdgeInfo`), `tiling.edge_shape_ids`, `tiling.edge_shapes` |
| `tiling.edge_shapes` (`EdgeShape` enum) | `tiling.edge_shapes` (strings `"I"`, `"J"`, `"U"`, `"S"`) |

An affine matrix is `(2, 3)`: the linear part and the translation. tactile stores it as `[a, b, tx, c, d, ty]`.

## Fill a rectangle

```py
for shape in tiling.fill_region_bounds(0, 0, 10, 6):
    t, t1, t2, aspect = shape.T, shape.t1, shape.t2, shape.aspect
    color = tiling.get_color(t1, t2, aspect)
```

```python
tiles = tiling.fill((0, 0, 10, 6))  # the same tiles in the same order, without repeats
print(len(tiles), tiles.cell[0], tiles.aspect[0], tiles.transforms.shape, tiles.colors()[:3])
```

The tile `i` has the lattice cell `tiles.cell[i]` (tactile's `t1, t2`), the index `tiles.aspect[i]` and the affine
transform `tiles.transforms[i]` of the prototile onto the tile. `tiles.vertices()` gives the corners of all tiles and
`tiles.colors()` the colors of `get_color`. tactile's fill can return the same tile twice for some regions; `fill`
returns each tile once. `fill` and tactile's `fill_region_bounds` use the native coordinates of the type; to place tiles
in a region of your coordinates use `fill_region` (a tile count or a tile size) or set a `frame`.

## What tactile does not have

- Tile selection that covers a region (`cover`, `fill_region`) and tiles from lattice cells (`tiles`).
- Adjacency of tiles from a lattice table: `tiles.adjacency("edge")`, `neighbors`, `neighbor_lists`.
- A frame (scale, rotation, position), named constructors for squares, hexagons, triangles and quadrilaterals, and
  presets (`list_presets`).
- Edge curves with symmetry constraints and an exact validity test, serialization (`save`, `load`, `to_dict`) and
  GeoJSON export (`TileSet.to_geojson`).
- `find_types` and `TilingType.info` to select and describe types.

## Not carried over

The 12 types of tactile's numbering without an implementation (IH numbers missing from `tiling_types()`) are not
provided, as in tactile. The tactile package itself is only needed to rebuild `src/isotile/data/types.npz` and to run
the equivalence tests.
