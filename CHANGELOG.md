# Changelog

All notable changes to isotile are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [0.1.0] - Unreleased

First release. isotile keeps the tiling type numbers (IH1 to IH93) and the parameter format of
[tactile](https://github.com/isohedral/tactile); its type data is derived from tactile (see `NOTICE`).

### Added

- Tiling types: the 81 isohedral types of tactile as arrays (`TilingType`, `get_type`, `tiling_types`), with
  `find_types`, `TilingType.info()` and `describe()`. `TilingType.safe_ranges` holds per parameter the interval around
  the default in which the prototile stays valid while the other parameters keep their defaults (an infinite end means
  no limit).
- `IsohedralTiling(type, parameters, frame=)`: geometry of one type at given parameters, with a public affine `frame`
  (scale, rotation, reflection, position), `prototile(normalized=)`, `inscribed_radius`, `with_parameters`, `scaled`,
  `rotated` and `with_frame`. `repr(tiling)` is the constructor call as Python code that rebuilds an equal tiling.
- Named constructors `square()`, `hexagon(pointy_top=True)`, `triangle(vertices)` and `quadrilateral(vertices)`.
- Tile generation: `fill` (tactile's scanline fill, vectorized, without repeated tiles), `cover`, `tiles(cell, aspect)`
  and `fill_region(bounds, n_tiles= | tile_size=, margin=, offset=, outline=)`. `cover` includes a tile iff its polygon
  overlaps the region with positive area (touching at a point or along an edge does not count; a region of zero width
  or height raises `ValueError`) and measures the overlap relative to the center of the region, so it is exact far from
  the origin. `fill_region` anchors the lattice at the region center plus `u * translation_u + v * translation_v` of the
  placed tiling (`offset=(u, v)`), so a region expanded by the same amount on all sides keeps the lattice.
  `outline="curved"` selects tiles by the overlap of their curved polygon with the region and `"straight"` uses the
  straight polygon; the default `None` means `"curved"` if the tiling has an edge curve and `"straight"` otherwise.
- `TileSet` with cells, aspects, transforms, vertices, centroids, colors, `polygons()` (shapely extra), `outlines()` and
  sparse boolean adjacency of kinds `"edge"`, `"vertex"` and `"vertex-only"` from the lattice table, `neighbors(i)` and
  `neighbor_lists(kind)` (the sorted neighbors of all tiles from one matrix). `geometric_adjacency` computes the same
  matrices for arbitrary polygons.
- `TileSet.to_geojson(properties=None, curved=True, n_points=None)`: the tiles as a GeoJSON `FeatureCollection` dict
  (properties `cell`, `aspect`, `color`; rings counterclockwise); needs no shapely. `TileSet.save_geojson(path, ...)`
  writes it as a UTF-8 JSON file.
- Validity: `check_prototile` reports whether a parameter set gives a valid prototile and which edges are at fault. The
  constructor, `with_parameters`, `with_edge_curves` and the `parameters` setter raise `InvalidPrototileError`
  (a `ValueError` with the `PrototileReport` as `report`) for an invalid prototile or outline and keep the previous
  state. Malformed arguments raise a plain `ValueError`.
- Edge curves: `IsohedralTiling(..., edge_curves=, edge_curve_modes=)` puts curves on the canonical edge per edge shape
  id, with `edge_curves`, `edge_curve_modes`, `edges` (`EdgeInfo`), `with_edge_curves`, `prototile(curved=True,
  n_points=)`, `edge_outlines()` and `TileSet.polygons(curved=True)`. `EdgeShape`, `enforce_symmetry` (idempotent
  projection by index onto the `U` and `S` constraints), `is_symmetric` and `interpolate` (natural cubic spline on the
  chord-length parameter, or the control points).
- Validity of curves: the outline must be a simple closed curve with positive area whose parts that are not directly
  consecutive are at least `1e-6` times its diameter apart. The test is exact for the curve (spline pieces are tested
  as cubic Beziers by subdivision, a piece against its own halves included, so a hairpin within one piece is valid and
  a loop is not) and does not depend on the sampling.
- Presets: `list_presets`, `IsohedralTiling.from_preset`, `list_edge_curve_presets`, `edge_curve_preset` and
  `edge_curves_from_preset`. The `wavy_triangle` preset has the curve amplitude 0.15.
- `IsohedralTiling.to_dict`, `from_dict`, `save` and `load` (format version 1.1, with edge curve modes and the frame;
  version 1.0 is read) in `isotile.io`.
- `isotile.plot.plot_prototile` and `plot_tiles` (extra `plot`, matplotlib).
- Extras `shapely`, `plot` and `all`; typed package (`py.typed`, `mypy --strict` clean); `isotile.__version__`.

### Documentation

- Documentation site at https://bright-fakl.github.io/isotile/ (mkdocs-material, mkdocstrings): quickstart, concepts,
  catalog of the 81 types, edge curves, adjacency with a benchmark, migration notes from tactile, API reference and
  development guide. The tests run the code in the pages and check that every public name is in the API reference.
- Logo (mark, lockup, favicon, social preview) in `docs/assets/logo/`.
