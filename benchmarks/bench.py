"""Benchmark of the main operations of isotile at several sizes, optionally against tactile.

Run: ``uv run python benchmarks/bench.py [--sizes 1000 10000 100000] [--repeat 3] [--tactile] [--markdown]``

Operations, for the types IH7 (hexagon, 3 aspects), IH46 (quadrilateral) and IH84 (triangle) and for the preset
``puzzle_piece`` (curved edges):

- ``fill_region``: tiles for a region with about N tiles;
- ``vertices``: the (n, nv, 2) vertex array of those tiles;
- ``adjacency``: the sparse edge adjacency from the lattice table;
- ``neighbor_lists``: the sorted neighbors of every tile;
- ``geometric``: ``geometric_adjacency`` of the vertex arrays (the method for polygons that no tiling generated);
- ``check_prototile``: the validity test of the prototile (independent of the number of tiles).

With ``--tactile`` (needs the ``tactile`` package) the scanline fill of tactile (``fill_region_bounds``) is timed on
the same region in native coordinates, up to ``--tactile-max`` tiles. The work is deterministic (no random numbers);
the times depend on the machine. Each time is the best of ``--repeat`` runs.
"""

from __future__ import annotations

import argparse
import platform
import sys
import time
from collections.abc import Callable
from typing import Any, TypeVar

import numpy as np
import scipy

import isotile
from isotile import IsohedralTiling, check_prototile, geometric_adjacency

T = TypeVar("T")
CASES: tuple[tuple[str, int | str], ...] = (("IH7", 7), ("IH46", 46), ("IH84", 84), ("puzzle_piece", "puzzle_piece"))


def best(fn: Callable[[], T], repeat: int) -> tuple[float, T]:
    """Smallest wall-clock time of ``repeat`` calls and the last result."""
    times = []
    out: Any = None
    for _ in range(repeat):
        t0 = time.perf_counter()
        out = fn()
        times.append(time.perf_counter() - t0)
    return min(times), out


def make(case: int | str) -> IsohedralTiling:
    return IsohedralTiling.from_preset(case) if isinstance(case, str) else IsohedralTiling(case)


def run_case(name: str, case: int | str, n: int, repeat: int, tactile_max: int | None) -> dict[str, Any]:
    tiling = make(case)
    row: dict[str, Any] = {"case": name, "n_requested": n}
    row["fill_region"], tiles = best(lambda: tiling.fill_region((0.0, 0.0, 1.0, 1.0), n_tiles=n), repeat)
    row["n_tiles"] = len(tiles)
    row["vertices"], vertices = best(tiles.vertices, repeat)
    row["adjacency"], edge = best(lambda: tiles.adjacency("edge"), repeat)
    row["neighbor_lists"], _ = best(lambda: tiles.neighbor_lists("edge"), repeat)
    row["geometric"], (edge_g, _) = best(lambda: geometric_adjacency(vertices, tol=1e-9), max(1, repeat - 1))
    if (edge_g != edge).nnz:
        raise RuntimeError(f"{name}: lattice and geometric adjacency differ")
    if tactile_max is not None and isinstance(case, int) and n <= tactile_max:
        row.update(tactile_fill(case, n, repeat))
    return row


def tactile_fill(number: int, n: int, repeat: int) -> dict[str, Any]:
    """tactile and isotile ``fill`` on the same native-coordinate rectangle with about ``n`` tiles."""
    import tactile

    old = tactile.IsohedralTiling(number)
    new = IsohedralTiling(number)
    side = float(np.sqrt(n * new.area))
    t_old, shapes = best(lambda: list(old.fill_region_bounds(0.0, 0.0, side, side)), max(1, repeat - 1))
    t_new, tiles = best(lambda: new.fill((0.0, 0.0, side, side)), repeat)
    return {
        "tactile_fill": t_old,
        "tactile_tiles": len(shapes),
        "isotile_fill": t_new,
        "isotile_fill_tiles": len(tiles),
    }


def prototile_rows(repeat: int) -> list[dict[str, Any]]:
    rows = []
    calls = 100
    for name, case in CASES:
        tiling = make(case)
        number = case if isinstance(case, int) else tiling.type.number
        curves = tiling.edge_curves or None
        modes = tiling.edge_curve_modes or None

        def run(
            number: int = number, tiling: IsohedralTiling = tiling, curves: Any = curves, modes: Any = modes
        ) -> None:
            for _ in range(calls):
                check_prototile(number, tiling.parameters, edge_curves=curves, edge_curve_modes=modes)

        t, _ = best(run, repeat)
        rows.append({"case": name, "check_prototile": t / calls})
    return rows


def fmt(seconds: float) -> str:
    if seconds < 1e-3:
        return f"{seconds * 1e6:.0f} µs"
    if seconds < 1.0:
        return f"{seconds * 1e3:.1f} ms"
    return f"{seconds:.2f} s"


def print_markdown(rows: list[dict[str, Any]], protos: list[dict[str, Any]], with_tactile: bool) -> None:
    print("| Case | Tiles | fill_region | vertices | adjacency | neighbor_lists | geometric_adjacency |")
    print("|---|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        cells = [fmt(r[k]) for k in ("fill_region", "vertices", "adjacency", "neighbor_lists", "geometric")]
        print(f"| {r['case']} | {r['n_tiles']:,} | " + " | ".join(cells) + " |")
    print()
    print("| Case | check_prototile |")
    print("|---|---:|")
    for p in protos:
        print(f"| {p['case']} | {fmt(p['check_prototile'])} |")
    if with_tactile:
        print()
        print("| Type | Tiles (tactile) | tactile fill | Tiles (isotile) | isotile fill | Speedup |")
        print("|---|---:|---:|---:|---:|---:|")
        for r in rows:
            if "tactile_fill" in r:
                print(
                    f"| {r['case']} | {r['tactile_tiles']:,} | {fmt(r['tactile_fill'])} | {r['isotile_fill_tiles']:,}"
                    f" | {fmt(r['isotile_fill'])} | {r['tactile_fill'] / r['isotile_fill']:.0f}x |"
                )


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--sizes", type=int, nargs="+", default=[1000, 10000, 100000], help="target tile counts")
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--tactile", action="store_true", help="also time tactile's fill (needs tactile)")
    ap.add_argument("--tactile-max", type=int, default=10000, help="largest tile count run with tactile")
    ap.add_argument("--markdown", action="store_true", help="print Markdown tables")
    args = ap.parse_args(argv)

    print(
        f"isotile {isotile.__version__}, Python {platform.python_version()}, numpy {np.__version__}, "
        f"scipy {scipy.__version__}, {platform.system()} {platform.machine()}",
        file=sys.stderr,
    )
    tactile_max = args.tactile_max if args.tactile else None
    rows = [run_case(name, case, n, args.repeat, tactile_max) for name, case in CASES for n in args.sizes]
    protos = prototile_rows(args.repeat)
    if args.markdown:
        print_markdown(rows, protos, args.tactile)
        return
    for r in rows:
        print(
            f"{r['case']:<13}{r['n_tiles']:>8} | fill_region {fmt(r['fill_region']):>9} | vertices {fmt(r['vertices']):>9}"
            f" | adjacency {fmt(r['adjacency']):>9} | neighbor_lists {fmt(r['neighbor_lists']):>9}"
            f" | geometric {fmt(r['geometric']):>9}",
            flush=True,
        )
    for p in protos:
        print(f"{p['case']:<13} check_prototile {fmt(p['check_prototile'])}")


if __name__ == "__main__":
    main()
