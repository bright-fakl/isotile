"""Benchmark: tile generation and adjacency, library vs carto-flow's current methods.

Run: ``PYTHONPATH=. uv run --project /home/fakl/carto-flow python bench.py [n_tiles ...]``
"""

from __future__ import annotations

import sys
import time

import numpy as np

from isotile import IsohedralTiling, geometric_adjacency

TYPES = (7, 46, 84)
HALF_WIDTH_FACTOR = 0.5  # half-width of the square region = factor * sqrt(n * tile area)


def best(fn, repeat: int = 3):
    times, out = [], None
    for _ in range(repeat):
        t0 = time.perf_counter()
        out = fn()
        times.append(time.perf_counter() - t0)
    return min(times), out


def run_case(number: int, n: int, carto_ih, carto_adjacency) -> None:
    ih = IsohedralTiling(number)
    half = HALF_WIDTH_FACTOR * np.sqrt(n * ih.area)
    bounds = (-half, -half, half, half)
    t_gen, tiles = best(lambda: ih.cover(bounds))
    t_poly, verts = best(tiles.vertices)
    t_lat, adj = best(lambda: tiles.adjacency("edge"))
    t_geo, (edge, _) = best(lambda: geometric_adjacency(verts, tol=1e-6), repeat=2)
    if (edge != adj).nnz:
        raise RuntimeError("lattice and geometric adjacency differ")
    # carto-flow with the same number of tiles: its generate (shapely polygons, O(n^2) deduplication) and adjacency
    t0 = time.perf_counter()
    res = carto_ih(number).generate(bounds=(-1, -1, 1, 1), n_tiles=len(tiles), margin=0.0, compute_adjacency=False)
    t_cgen = time.perf_counter() - t0
    t0 = time.perf_counter()
    carto_adjacency(res.polygons, tol=res.tile_size * 0.01)
    t_cgeo = time.perf_counter() - t0
    print(
        f"IH{number:<3} {len(tiles):>7} | {t_gen + t_poly:>8.4f}s {t_lat:>10.4f}s {t_geo:>13.4f}s | "
        f"{t_cgen:>13.2f}s {t_cgeo:>13.2f}s | lattice {t_cgeo / t_lat:>6.0f}x, "
        f"vectorized geometric {t_cgeo / t_geo:>5.0f}x  (carto tiles {len(res.polygons)})",
        flush=True,
    )


def main(sizes: list[int]) -> None:
    from carto_flow.symbol_cartogram.tiling import IsohedralTiling as CartoIH
    from carto_flow.symbol_cartogram.tiling import _compute_adjacency_matrices

    print(
        f"{'type':>5} {'tiles':>7} | {'generate':>9} {'lattice adj':>11} {'geom adj (new)':>14} | "
        f"{'carto generate':>14} {'carto geom adj':>14} | speedup of the adjacency step"
    )
    for number in TYPES:
        for n in sizes:
            run_case(number, n, CartoIH, _compute_adjacency_matrices)


if __name__ == "__main__":
    main([int(a) for a in sys.argv[1:]] or [1000, 10000])
