# Changelog

All notable changes to isotile are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added

- The 81 isohedral tiling types of tactile as arrays (`TilingType`, `get_type`, `tiling_types`), with `find_types`,
  `TilingType.info()` and `describe()`.
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
