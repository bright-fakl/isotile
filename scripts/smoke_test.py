"""Smoke test of an installed isotile: import, version, the 81 types, tiles and adjacency.

Run it with the interpreter of a clean environment that has the wheel installed, from a directory that is not the
checkout: ``python scripts/smoke_test.py [--version X.Y.Z]``. It fails if ``isotile`` resolves to a source checkout.
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import isotile


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", help="the version the installed package must report")
    args = parser.parse_args()

    here = pathlib.Path(isotile.__file__).resolve()
    if "site-packages" not in here.parts:
        sys.exit(f"isotile resolved to {here}, not to an installed package")
    if args.version is not None and isotile.__version__ != args.version:
        sys.exit(f"isotile.__version__ is {isotile.__version__}, expected {args.version}")
    if len(isotile.tiling_types()) != 81:
        sys.exit("expected 81 tiling types")
    for number in isotile.tiling_types():
        info = isotile.get_type(number)
        if info.safe_ranges.shape != (info.n_params, 2):
            sys.exit(f"IH{number}: safe_ranges has the wrong shape")
        tiles = isotile.IsohedralTiling(number).fill_region((0, 0, 4, 3), n_tiles=50)
        if len(tiles) < 50 or tiles.adjacency("edge").nnz == 0:
            sys.exit(f"IH{number}: fill_region or adjacency gave no result")
    print(f"isotile {isotile.__version__}: 81 types generated tiles from {here}")


if __name__ == "__main__":
    main()
