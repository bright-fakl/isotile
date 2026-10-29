# Changelog

All notable changes to isotile are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Fixed

- `docs/curves.md` gave 0.1 as the default `margin` of `fill_region`; the default is 0.0.
- `with_frame` (and so `rotated`, `scaled` and `fill_region`) no longer checks the prototile again, because a frame does
  not change validity. `fill_region(outline="straight")` of the nine presets with curves takes a median of 2.1 ms for
  100 tiles instead of 7.0 ms; the tiles and the tiling are bit-identical.
- The `wavy_triangle` preset has the curve amplitude 0.15 again (0.12 before), as in the carto-flow preset of the same
  name; a test compares all 15 presets with `tests/data/carto_flow_presets.json`, which
  `scripts/make_carto_flow_presets_fixture.py` regenerates.
- `IsohedralTiling.cover` (and so `fill_region`) measures the overlap areas relative to the center of the region.
  Far from the origin the rounding error of the absolute coordinates could drop tiles whose overlap is real (holes in
  triangle tilings at a distance of 1e6 with tiles of size 0.25) or count a tile that only touches the region.

### Added

- `IsohedralTiling.with_parameters(parameters)`: a new tiling with the parameters replaced and the same checks as the
  `parameters` setter; the tiling is unchanged.
- `InvalidPrototileError(ValueError)` with the `PrototileReport` as `report`, raised by the constructor,
  `with_parameters`, `with_edge_curves` and the `parameters` setter when the prototile or its outline is not valid.
  Malformed arguments still raise a plain `ValueError`.
- `repr(tiling)` is the constructor call as Python code that rebuilds an equal tiling.
- `TileSet.to_geojson(properties=None, curved=True, n_points=None)`: the tiles as a GeoJSON `FeatureCollection` dict
  (properties `cell`, `aspect`, `color`; rings counterclockwise); needs no shapely.
- `TileSet.save_geojson(path, properties=None, curved=True, n_points=None)` writes `to_geojson` as a UTF-8 JSON file.
- `cover(bounds, outline=None)` and `fill_region(..., outline=None)`: `"curved"` selects tiles by the overlap of their
  curved polygon with the region, so the region is covered completely at `margin=0` for tilings with edge curves;
  `"straight"` is the former test. The default `None` means `"curved"` if the tiling has an edge curve (any entry of
  `edge_curves`) and `"straight"` otherwise. Behavior change: a curved tiling now also returns the tiles that reach
  into the region (or, with a positive margin, the margin band) only through their curves, for example 126 instead of
  112 tiles for `puzzle_piece` at `n_tiles=100`, `margin=0.1`, and the selection takes 0.5 to 1 s instead of 0.1 s for
  10,000 tiles. `outline="straight"` restores the former result.
- `TileSet.neighbor_lists(kind="edge")`: the sorted neighbors of every tile from one adjacency matrix; `neighbors(i)`
  builds the matrix on every call.

- `IsohedralTiling.fill_region(..., offset=(u, v))`: anchors the lattice at the region center plus
  `u * translation_u + v * translation_v` of the placed tiling, in one `cover` pass. The default `(0.0, 0.0)` gives the
  previous result.

### Documentation

- `fill_region` states that the lattice is anchored at the region center, so a region expanded by the same amount on
  all sides keeps the lattice.

### Edge curves

- `IsohedralTiling(..., edge_curves=, edge_curve_modes=)`: curves on the canonical edge per edge shape id, with
  `edge_curves`, `edge_curve_modes`, `edges` (`EdgeInfo`), `with_edge_curves`, `prototile(curved=True, n_points=)`
  and `edge_outlines()`; `TileSet.outlines()` and `TileSet.polygons(curved=True)`.
- `EdgeShape`, `enforce_symmetry` (idempotent projection by index onto the `U` and `S` constraints), `is_symmetric`
  and `interpolate` (natural cubic spline on the chord-length parameter, or the control points).
- Validity of curves: the outline must be a simple closed curve with positive area whose parts that are not directly
  consecutive are at least `1e-6` times its diameter apart. The test is exact for the curve (spline pieces are tested
  as cubic Beziers by subdivision, a piece against its own halves included, so a hairpin within one piece is valid
  and a loop is not) and does not depend on the sampling. `check_prototile` takes
  `edge_curves` and `edge_curve_modes`; the constructor and the `parameters` setter raise `ValueError` for an invalid
  outline and keep the previous state.
- Tiling presets (`list_presets`, `IsohedralTiling.from_preset`) and edge curve presets (`list_edge_curve_presets`,
  `edge_curve_preset`, `edge_curves_from_preset`).
- `IsohedralTiling.to_dict`, `from_dict`, `save` and `load` (format version 1.1, with edge curve modes and the frame;
  version 1.0 is read) in `isotile.io`.
- `isotile.plot.plot_prototile` and `plot_tiles` (extra `plot`, matplotlib).
- Page Edge curves, API pages, and figures of the edge shapes and curved tilings.

### Documentation

- Documentation site built with mkdocs-material and mkdocstrings (home, quickstart, concepts, adjacency, API
  reference, development, changelog), deployed to GitHub Pages at https://bright-fakl.github.io/isotile/. The tests run
  the code in the pages and check that every public name is in the API reference. Docstrings use the numpy style with
  cross-references.
- Dependency group `docs` (`uv sync --group docs`).
- Logo (mark, lockup, favicon, social preview) in `docs/assets/logo/`, generated by `scripts/make_logo_assets.py`; a
  Branding page in the documentation; the README header shows the logo.
- Figures in the documentation (home, quickstart, concepts, adjacency): tilings, cells and aspects, parameters,
  neighbors, lattice offsets and the adjacency matrix, as light and dark SVG files in `docs/assets/figures/` generated
  by `scripts/make_doc_figures.py` (matplotlib in the `docs` group).

### Added

- The 81 isohedral tiling types of tactile as arrays (`TilingType`, `get_type`, `tiling_types`), with `find_types`,
  `TilingType.info()` and `describe()`.
- `IsohedralTiling.cover` includes a tile iff its polygon overlaps the region with positive area (touching at a point or
  along an edge does not count; a region of zero width or height raises `ValueError`).
- `IsohedralTiling` with `fill`, `cover`, `tiles(cell, aspect)` and `fill_region` (tile count or tile size, bounds,
  margin), a public affine `frame`, `prototile(normalized=...)` and `inscribed_radius`.
- Named constructors `square()`, `hexagon(pointy_top=True)`, `triangle(vertices)` and `quadrilateral(vertices)`.
- `TileSet` with cells, aspects, transforms, vertices, centroids, colors, `polygons()` (shapely extra) and
  sparse boolean adjacency of kinds `"edge"`, `"vertex"` and `"vertex-only"` from the lattice table, plus
  `geometric_adjacency` for arbitrary polygons.
- `check_prototile` reports whether a parameter set gives a valid prototile and which edges are at fault; the
  constructor raises `ValueError` for an invalid prototile. `TilingType.safe_ranges` holds per parameter the
  interval around the default in which the prototile stays valid while the other parameters keep their defaults
  (an infinite end means no limit).
