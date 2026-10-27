"""Benchmark: tile generation, lattice adjacency and geometric adjacency.

Run: ``uv run python benchmarks/bench.py [n_tiles ...]``
"""

from __future__ import annotations

import sys
import time
from collections.abc import Callable
from typing import TypeVar

from isotile import IsohedralTiling, geometric_adjacency

T = TypeVar("T")
TYPES = (7, 46, 84)


def best(fn: Callable[[], T], repeat: int = 3) -> tuple[float, T]:
    """Smallest wall-clock time of ``repeat`` calls and the last result."""
    times = []
    out = None
    for _ in range(repeat):
        t0 = time.perf_counter()
        out = fn()
        times.append(time.perf_counter() - t0)
    return min(times), out  # type: ignore[return-value]


def run_case(number: int, n: int) -> None:
    tiling = IsohedralTiling(number)
    t_tiles, tiles = best(lambda: tiling.fill_region((0.0, 0.0, 1.0, 1.0), n_tiles=n))
    t_vertices, vertices = best(tiles.vertices)
    t_lattice, lattice = best(lambda: tiles.adjacency("edge"))
    t_geometric, (edge, _) = best(lambda: geometric_adjacency(vertices, tol=1e-9), repeat=2)
    if (edge != lattice).nnz:
        raise RuntimeError("lattice and geometric adjacency differ")
    print(
        f"IH{number:<3} {len(tiles):>7} | {t_tiles:>9.4f}s {t_vertices:>9.4f}s {t_lattice:>9.4f}s {t_geometric:>9.4f}s"
        f" | lattice is {t_geometric / t_lattice:>5.1f}x faster than geometric",
        flush=True,
    )


def main(sizes: list[int]) -> None:
    print(f"{'type':>5} {'tiles':>7} | {'tiles':>10} {'vertices':>10} {'lattice':>10} {'geometric':>10}")
    for number in TYPES:
        for n in sizes:
            run_case(number, n)


if __name__ == "__main__":
    main([int(a) for a in sys.argv[1:]] or [1000, 10000])
