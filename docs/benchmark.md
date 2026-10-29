# Benchmark

Time of the main operations at several sizes. The script `benchmarks/bench.py` produces the numbers; it uses no random
numbers, so every run does the same work, and each time is the best of three runs.

```bash
uv run python benchmarks/bench.py --markdown                # the tables of this page
uv run python benchmarks/bench.py --markdown --tactile      # adds the comparison with tactile
uv run python benchmarks/bench.py --sizes 1000 --repeat 1   # a quick run, plain text
```

The tables were measured with Python 3.13, NumPy 2.5 and SciPy 1.18 on one machine (AMD Ryzen 7 5825U, Linux under WSL2; the
code is single-threaded), with other programs running. Absolute times differ between machines; the ratios and the growth
with the number of tiles are what the page is for. The benchmark is not part of the default CI.

## Operations

The cases are the types IH7 (hexagon, 3 aspects), IH46 (quadrilateral) and IH84 (triangle) at their default parameters
and the preset `puzzle_piece` (IH41 with curved edges). The target tile counts are 1,000, 10,000 and 100,000; the
column Tiles is the number of tiles returned, which also includes the tiles on the boundary of the region.

- `fill_region`: `IsohedralTiling.fill_region((0, 0, 1, 1), n_tiles=N)`, which sets the frame and selects the tiles
  that overlap the region.
- `vertices`: `TileSet.vertices()`, the `(n, n_vertices, 2)` array of corners.
- `adjacency`: `TileSet.adjacency("edge")`, the sparse matrix from the lattice table.
- `neighbor_lists`: `TileSet.neighbor_lists("edge")`, the sorted neighbors of every tile.
- `geometric_adjacency`: the method for arbitrary polygons, on the vertex arrays of the same tiles. The benchmark
  checks that its edge matrix equals the lattice matrix.

| Case | Tiles | fill_region | vertices | adjacency | neighbor_lists | geometric_adjacency |
|---|---:|---:|---:|---:|---:|---:|
| IH7 | 1,068 | 13.1 ms | 475 µs | 695 µs | 1.2 ms | 3.3 ms |
| IH7 | 10,300 | 119.0 ms | 5.0 ms | 2.5 ms | 9.2 ms | 31.8 ms |
| IH7 | 100,764 | 1.21 s | 47.2 ms | 23.2 ms | 84.9 ms | 348.5 ms |
| IH46 | 1,024 | 6.6 ms | 315 µs | 432 µs | 1.0 ms | 2.7 ms |
| IH46 | 10,000 | 67.5 ms | 3.4 ms | 1.7 ms | 7.8 ms | 23.5 ms |
| IH46 | 101,124 | 796.8 ms | 31.8 ms | 16.4 ms | 77.0 ms | 262.4 ms |
| IH84 | 1,118 | 5.9 ms | 264 µs | 331 µs | 1.1 ms | 2.7 ms |
| IH84 | 10,108 | 50.8 ms | 2.5 ms | 1.4 ms | 8.0 ms | 21.5 ms |
| IH84 | 101,398 | 586.1 ms | 23.8 ms | 12.8 ms | 72.7 ms | 229.7 ms |
| puzzle_piece | 1,024 | 84.6 ms | 314 µs | 294 µs | 908 µs | 2.6 ms |
| puzzle_piece | 10,400 | 610.3 ms | 3.2 ms | 1.4 ms | 7.3 ms | 23.2 ms |
| puzzle_piece | 101,124 | 2.14 s | 31.9 ms | 15.5 ms | 75.2 ms | 252.1 ms |

All times grow about linearly with the number of tiles. The lattice adjacency is 5 to 17 times faster than
`geometric_adjacency` for the same tiles, and it does not depend on a tolerance. `fill_region` of a tiling with curves
is slower than for straight tilings because the tiles are selected by the overlap of their curved polygons
(`outline="curved"`); `outline="straight"` gives the time of the straight tilings.

## Validity test

`check_prototile` for the default parameters of the types above and, for `puzzle_piece`, with its curves (the mean of
100 calls). The cost does not depend on the number of tiles.

| Case | check_prototile |
|---|---:|
| IH7 | 155 µs |
| IH46 | 154 µs |
| IH84 | 147 µs |
| puzzle_piece | 5.1 ms |

## Comparison with tactile

tactile has no adjacency and no region selection, so only the scanline fill is compared: `tactile.IsohedralTiling.fill_region_bounds`
(converted to a list) against `IsohedralTiling.fill`, on a square region in native coordinates that holds about N tiles. The
two return the same tiles; the count column shows it. Sizes above 10,000 tiles are not run with tactile.

| Type | Tiles (tactile) | tactile fill | Tiles (isotile) | isotile fill | Speedup |
|---|---:|---:|---:|---:|---:|
| IH7 | 1,083 | 2.7 ms | 1,083 | 368 µs | 7x |
| IH7 | 10,398 | 27.3 ms | 10,398 | 1.4 ms | 20x |
| IH46 | 1,122 | 3.1 ms | 1,122 | 346 µs | 9x |
| IH46 | 10,404 | 28.0 ms | 10,404 | 1.9 ms | 15x |
| IH84 | 1,144 | 3.1 ms | 1,144 | 385 µs | 8x |
| IH84 | 10,242 | 25.1 ms | 10,242 | 1.9 ms | 13x |

The speedup grows with the number of tiles because the fill of isotile runs in NumPy and tactile loops in Python.
