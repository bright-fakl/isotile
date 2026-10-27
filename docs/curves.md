# Edge curves

The edges of the prototile can be replaced by curves. The copies of the prototile still fit together, because the
curve of an edge is the curve of the edge it is matched with. This page describes how curves are defined, how the
shape of an edge constrains them, how they are sampled, when a curved prototile is valid, and how to save and draw
a curved tiling.

## Defining curves

A curve is given on the canonical edge, the segment from `(0, 0)` to `(1, 0)`, by a list of control points that starts
at `(0, 0)` and ends at `(1, 0)`. The side `y > 0` is the left of the direction from `(0, 0)` to `(1, 0)`. A curve
belongs to an edge shape id (`TilingType.edge_shape_ids`): every edge with that id gets the curve, placed on the edge
by `IsohedralTiling.edge_transforms` and reversed where `IsohedralTiling.edge_reversed` is true. Edge shape ids without
a curve stay straight.

```python
import numpy as np

from isotile import IsohedralTiling, check_prototile, get_type

print(get_type(28).shape_types, get_type(28).edge_shape_ids.tolist())

curves = {
    0: [(0, 0), (0.25, 0.12), (0.5, 0), (0.75, -0.12), (1, 0)],
    1: [(0, 0), (0.3, 0.1), (0.7, 0.1), (1, 0)],
    2: [(0, 0), (0.35, -0.12), (0.6, -0.12), (1, 0)],
}
curved = IsohedralTiling(28, edge_curves=curves)
print(len(curved.prototile()), "points on the outline,", curved.area)
```

The shape ids of IH28 are 0 for the first edge and 1 and 2 for the two pairs of edges after it. The curves are
available as `edge_curves`, `edge_curve_modes` and, edge by edge, `edges`, a tuple of `EdgeInfo` with the index, shape
id, shape, reversal and curve of each edge. `with_edge_curves(curves)` returns a copy with other curves, and
`rotated`, `scaled` and `with_frame` keep the curves.

An edge shape id of a straight (`I`) edge cannot have a curve; giving one raises `ValueError`, as does a curve that
does not run from `(0, 0)` to `(1, 0)`.

## Edge shapes

The shape of an edge restricts its curve. A curve with `m` control points has the point `k` and the point `m - 1 - k`
paired.

| Shape | Constraint | Paired points |
|---|---|---|
| `I` | The segment from `(0, 0)` to `(1, 0)`. | |
| `J` | None. | |
| `U` | Mirror symmetric about the perpendicular bisector `x = 0.5`. | `(x, y)` and `(1 - x, y)` |
| `S` | Symmetric under a rotation by 180 degrees about `(0.5, 0)`. | `(x, y)` and `(1 - x, -y)` |

<figure markdown="span">
  ![Four curves on the canonical edge, one per shape. I is a straight line. J is a free wavy curve. U is a bump with a dashed vertical mirror line in the middle and two pairs of mirrored points joined by dashed lines. S is a curve that rises on the left and falls on the right, with a dot at the midpoint and two pairs of points joined by dashed lines through it.](assets/figures/curves-shapes-light.svg#only-light)
  ![Four curves on the canonical edge, one per shape. I is a straight line. J is a free wavy curve. U is a bump with a dashed vertical mirror line in the middle and two pairs of mirrored points joined by dashed lines. S is a curve that rises on the left and falls on the right, with a dot at the midpoint and two pairs of points joined by dashed lines through it.](assets/figures/curves-shapes-dark.svg#only-dark)
  <figcaption>One curve per edge shape on the canonical edge. Dots are control points; dashed lines join paired points.</figcaption>
</figure>

The pairing is by position in the list, not by coordinates, so a curve may turn back in `x`. `enforce_symmetry(points,
shape)` is the projection onto the curves of a shape: it keeps the first half of the points (`k < m // 2`), replaces
the second half by the images of the first half, and moves the middle point of an odd number of points onto the axis
(`U`: `x = 0.5`) or onto the center (`S`: `(0.5, 0)`). `J` returns the points, `I` the segment.
`is_symmetric(points, shape)` tests the constraint. The projection is idempotent: applying it to its own result
returns the same array, and a curve that already satisfies the constraint is returned unchanged.

```python
from isotile import enforce_symmetry, is_symmetric

raw = [(0, 0), (0.2, 0.1), (0.4, 0.3), (0.9, 0.9), (1, 0)]
u_curve = enforce_symmetry(raw, "U")
s_curve = enforce_symmetry(raw, "S")
print(u_curve.tolist())
print(s_curve.tolist())
assert is_symmetric(u_curve, "U") and not is_symmetric(u_curve, "S")
assert np.array_equal(enforce_symmetry(u_curve, "U"), u_curve)
```

The constructor and `check_prototile` apply `enforce_symmetry` to the curves they receive, so `edge_curves` always
holds curves that satisfy their shape. `EdgeShape.from_points(shape, points, mode)` does the same for a single curve.

## Interpolation

The control points are joined by one of two modes, set per shape id with `edge_curve_modes`:

- `"spline"` (default): a parametric natural cubic spline through the control points. The parameter is the cumulative
  chord length scaled to `[0, 1]`, so the curve can turn back in `x`. `interpolate(points, "spline", n_points)`
  evaluates it at `n_points` equally spaced parameters; the default is `max(100, 10 * m)` for `m` control points. The
  samples keep the symmetry of the control points. Consecutive control points that coincide count once.
- `"linear"`: the control points themselves, joined by straight segments.

<figure markdown="span">
  ![One set of control points drawn twice on the canonical edge: a smooth spline through the points and the polyline of the points.](assets/figures/curves-modes-light.svg#only-light)
  ![One set of control points drawn twice on the canonical edge: a smooth spline through the points and the polyline of the points.](assets/figures/curves-modes-dark.svg#only-dark)
  <figcaption>The same control points in the two modes.</figcaption>
</figure>

```python
from isotile import interpolate

points = [(0, 0), (0.3, 0.2), (0.7, -0.2), (1, 0)]
print(interpolate(points).shape, interpolate(points, n_points=20).shape, interpolate(points, "linear").shape)
```

## The curved prototile and its tiles

`IsohedralTiling.prototile()` returns the closed outline in world coordinates: the vertices with every curved edge
replaced by its samples. It starts at vertex 0 and runs along the edges in order, without repeating the first point.
`prototile(curved=False)` and `vertices` are the straight polygon; `edge_outlines()` returns one polyline per edge,
from vertex `i` to vertex `i + 1`. The number of samples of a spline edge is `n_points`, by default as in
`interpolate`. A frame maps the outline like the vertices. `area`, `centroid` and `inscribed_radius` refer to the
straight prototile; the curved prototile has the same area.

<figure markdown="span">
  ![Two panels. Left: the straight pentagon of type IH28. Right: the same pentagon with wavy edges and a dashed outline of the straight pentagon underneath.](assets/figures/curves-prototile-light.svg#only-light)
  ![Two panels. Left: the straight pentagon of type IH28. Right: the same pentagon with wavy edges and a dashed outline of the straight pentagon underneath.](assets/figures/curves-prototile-dark.svg#only-dark)
  <figcaption>The prototile of the example above without and with its edge curves. The dashed line is the straight prototile.</figcaption>
</figure>

The tiles of a `TileSet` follow the same rule. `TileSet.outlines()` returns an `(n, M, 2)` array with the outline of
every tile, and `TileSet.polygons()` the tiles as shapely polygons (shapely extra); both take `curved=False` for the
straight tiles. `TileSet.vertices()` is always the straight polygon. Curves do not change which tiles are adjacent:
`TileSet.adjacency` is the same for any valid curves.

```python
tiles = curved.fill_region((0, 0, 8, 6), n_tiles=30)
outlines = tiles.outlines()
print(outlines.shape, len(tiles.adjacency("edge").nonzero()[0]))
```

<figure markdown="span">
  ![Two tilings with curved edges, types IH28 and IH29. Each tile has wavy edges; neighboring tiles fit together without gaps. One tile of each tiling is amber.](assets/figures/curves-tiling-light.svg#only-light)
  ![Two tilings with curved edges, types IH28 and IH29. Each tile has wavy edges; neighboring tiles fit together without gaps. One tile of each tiling is amber.](assets/figures/curves-tiling-dark.svg#only-dark)
  <figcaption>Curved tilings of types IH28 and IH29.</figcaption>
</figure>

## Validity

A curved prototile is valid when

1. the straight prototile of the same parameters is valid,
2. its outline is a simple closed curve with positive area, and
3. parts of the outline that are not directly consecutive are at least `1e-6 * diameter` apart, where the diameter is
   the larger side of the bounding box of the outline in native coordinates.

Touching or nearly touching parts of the outline are therefore invalid. No other condition is needed: the curve of an edge is the curve of its partner edge,
so tiles that are adjacent in the straight tiling fit along every shared edge, they do not overlap, and the union of
the tiles is the plane. The area of the curved prototile equals the area of the straight prototile, and the
adjacency of the tiles is the one of the straight tiling.

The test is exact for the curve, not for a sampled polyline: a spline is tested as its cubic pieces and a linear curve
as its segments, so the result does not depend on `n_points`, which only sets how finely the outline is drawn. Pieces
that are directly consecutive may share their join; two of them are separated by their shape except near the join.
The outline is invalid if a piece turns its tangent by 180 degrees or more or has a vanishing tangent, if the tangent
turns by 180 degrees or more at a join (the curve folds back), if a piece has zero length, or if two other parts come
closer than the clearance. Validity does not depend on the frame; it is judged in native coordinates.

`check_prototile(tiling_type, parameters, edge_curves=..., edge_curve_modes=...)` returns a `PrototileReport` without
building a tiling: `valid`, the sorted `bad_edges` (edges whose curves cross, touch, come closer than the clearance or fold back) and a `message`. The
constructor and the `parameters` setter apply the same test and raise `ValueError` when it fails; the setter keeps the
previous state. Curves can be valid in the spline mode and invalid in the linear mode, and the other way round.

```python
bow = {0: [(0, 0), (0.8, 0.5), (0.2, 0.5), (1, 0)]}
report = check_prototile(41, edge_curves=bow, edge_curve_modes={0: "linear"})
print(report.valid, report.bad_edges)
print(report.message)

try:
    IsohedralTiling(41, edge_curves=bow, edge_curve_modes={0: "linear"})
except ValueError as error:
    print("rejected:", error)
```

<figure markdown="span">
  ![Two panels showing the outline of a four-sided prototile of type IH41. Left: a valid outline with a gentle bump on two opposite edges. Right: an invalid outline where the curve of two opposite edges crosses itself; the crossing edges are drawn in red.](assets/figures/curves-invalid-light.svg#only-light)
  ![Two panels showing the outline of a four-sided prototile of type IH41. Left: a valid outline with a gentle bump on two opposite edges. Right: an invalid outline where the curve of two opposite edges crosses itself; the crossing edges are drawn in red.](assets/figures/curves-invalid-dark.svg#only-dark)
  <figcaption>A valid outline and one that crosses itself. The edges that <code>bad_edges</code> names are red.</figcaption>
</figure>

## Presets

`edge_curve_preset(name, shape, amplitude)` returns the control points of a named curve for any edge shape, with the
constraint of the shape enforced. `list_edge_curve_presets()` lists the names: `straight`, `bump`, `dent`, `wave` and
`s_curve`. The curve is built for a free edge and then projected, so a `wave` on a `U` edge becomes two bumps.
`edge_curves_from_preset(tiling_type, name, amplitude)` returns the curve for every shape id of a type that can be
curved.

```python
from isotile import edge_curve_preset, edge_curves_from_preset, list_edge_curve_presets

print(list(list_edge_curve_presets()))
wavy = IsohedralTiling(66, edge_curves=edge_curves_from_preset(66, "wave", 0.06))
print(sorted(wavy.edge_curves), edge_curve_preset("bump", "U", 0.1).shape)
```

Whether a preset gives a valid prototile depends on the type and the amplitude; test it with `check_prototile`.

`list_presets()` lists 15 named tilings (type, parameters and curves) and `IsohedralTiling.from_preset(name)` builds
one; keyword arguments replace the preset's `parameters`, `edge_curves`, `edge_curve_modes` or `frame`.

```python
from isotile import list_presets

print(list(list_presets()))
flowers = IsohedralTiling.from_preset("flowers")
```

## Saving and loading

`IsohedralTiling.to_dict()` returns a JSON-compatible dict with the type, parameters, edge curves, edge curve modes and,
unless it is the identity, the frame. `from_dict` reads it back, `save(path)` writes it as JSON and `load(path)` reads
the file. The output is deterministic and a tiling read back has the same type, parameters, curves, modes and frame.
The format is described in `isotile.io`. Version 1.0 files, which have no modes, are read with the spline mode for
every curve, and unknown keys are ignored.

```python
data = curved.to_dict()
again = IsohedralTiling.from_dict(data)
assert np.array_equal(again.prototile(), curved.prototile())
print(list(data))
```

## Plotting

`isotile.plot` has two helpers for matplotlib, which is an optional dependency (`pip install isotile[plot]`) that
is imported when a function is called. `plot_prototile(tiling, ax=None, show_labels=True)` draws the outline, the
straight prototile dashed when there are curves, and the index and shape of every edge with an arrow in the direction
of the canonical edge. `plot_tiles(tiles, ax=None, color="coloring")` draws curved tiles colored by the coloring of the
type, by aspect or in one color.

```text
from isotile.plot import plot_prototile, plot_tiles

ax = plot_tiles(tiles)
plot_prototile(curved)
```
