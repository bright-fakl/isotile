"""Export the 81 isohedral types from tactile into ``data/types.npz`` and add the lattice adjacency tables.

Development script. Needs the ``tactile`` package.

The exported coefficients are derived from Tactile / tactile-js:

    BSD 3-Clause License

    Copyright (c) 2018, Craig S. Kaplan
    All rights reserved.

    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are met:

    * Redistributions of source code must retain the above copyright notice, this
      list of conditions and the following disclaimer.

    * Redistributions in binary form must reproduce the above copyright notice,
      this list of conditions and the following disclaimer in the documentation
      and/or other materials provided with the distribution.

    * Neither the name of the copyright holder nor the names of its
      contributors may be used to endorse or promote products derived from
      this software without specific prior written permission.

    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
    AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
    IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
    DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
    FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
    DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
    SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
    CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
    OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
    OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

The adjacency tables are computed by shared-vertex counting on a patch of tiles. The safe parameter ranges are
found per parameter, with the other parameters at their defaults, from the real roots of the orientation
determinants of the prototile vertices (see ``critical_values``): the validity of the straight prototile is
constant between consecutive roots. The range ends at the first root, going outward from the default, where the
prototile is invalid at the root or just beyond it, or at a distance of 5 from the default. The distance is
rounded down to a multiple of 1e-6 (and kept at least 1e-9 inside).

The file is written deterministically (arrays sorted by key, fixed dtypes, uncompressed, fixed time stamps), and
``--check`` rebuilds the arrays and fails on any difference from the stored file.

Usage: ``python -m isotile._build_data [--check]``.
"""

from __future__ import annotations

import argparse
import itertools
import math
import sys
import zipfile
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from .adjacency import geometric_adjacency
from .prototile import check_polygon
from .tiling import IsohedralTiling
from .types import DATA_FILE, EDGE_SHAPES, KIND_EDGE, KIND_VERTEX, TilingType, type_from_arrays

NOTICE = (
    "Derived from Tactile / tactile-js, Copyright (c) 2018, Craig S. Kaplan, BSD 3-Clause License. "
    "See LICENSE-tactile in this directory."
)
PATCH_RADIUS = 5  # patch of (2R + 1)^2 cells around the origin
SCAN_LIMIT = 5.0
RANGE_GRID = 1e-6
FIXED_TIME = (1980, 1, 1, 0, 0, 0)


def export_tactile() -> dict[str, NDArray[np.generic]]:
    """Arrays of all valid types, keyed ``t{number}_{name}``; adjacency and safe ranges are placeholders."""
    from tactile import EdgeShape, tiling_types
    from tactile.tiling_data import TilingTypeData

    shape_code = {EdgeShape[name]: code for code, name in enumerate(EDGE_SHAPES)}
    out: dict[str, NDArray[np.generic]] = {
        "numbers": np.array(tiling_types, dtype=np.int64),
        "notice": np.array(NOTICE),
    }
    for n in tiling_types:
        d = TilingTypeData.get_data(n)
        npar, nv, na = d.num_params, d.num_vertices, d.num_aspects
        out[f"t{n}_edge_shapes"] = np.array([shape_code[s] for s in d.edge_shapes], dtype=np.int64)
        out[f"t{n}_edge_shape_ids"] = np.array(d.edge_shape_ids, dtype=np.int64)
        out[f"t{n}_edge_orientations"] = np.array(d.edge_orientations, dtype=np.int64).reshape(nv, 2)
        out[f"t{n}_default_params"] = np.array(d.default_params, dtype=np.float64).reshape(npar)
        out[f"t{n}_vertex_coeffs"] = np.array(d.vertex_coeffs, dtype=np.float64).reshape(nv, 2, npar + 1)
        out[f"t{n}_aspect_coeffs"] = np.array(d.aspect_coeffs, dtype=np.float64).reshape(na, 2, 3, npar + 1)
        out[f"t{n}_translation_coeffs"] = np.array(d.translation_coeffs, dtype=np.float64).reshape(2, 2, npar + 1)
        out[f"t{n}_coloring"] = np.array(d.coloring, dtype=np.int64)
        out[f"t{n}_adjacency"] = np.empty((0, 5), dtype=np.int64)
        out[f"t{n}_safe_ranges"] = np.empty((npar, 2), dtype=np.float64)
    return out


def adjacency_table(tiling: IsohedralTiling) -> NDArray[np.int64]:
    """Rows ``(aspect, d_cell_u, d_cell_v, neighbor_aspect, kind)`` of the tiles around the origin, by shared vertices."""
    r = PATCH_RADIUS
    cells = np.array([(i, j) for i in range(-r, r + 1) for j in range(-r, r + 1)])
    n_aspects = tiling.type.n_aspects
    tiles = tiling.tiles(np.repeat(cells, n_aspects, axis=0), np.tile(np.arange(n_aspects), len(cells)))
    edge, vertex = geometric_adjacency(tiles.vertices(), tol=1e-6)
    rows = []
    for adj, kind in ((edge, KIND_EDGE), (vertex, KIND_VERTEX)):
        coo = adj.tocoo()
        for i, j in zip(coo.row.tolist(), coo.col.tolist(), strict=True):
            if tuple(tiles.cell[i]) == (0, 0):  # tiles of the central cell
                d_cell_u, d_cell_v = tiles.cell[j] - tiles.cell[i]
                rows.append((tiles.aspect[i], d_cell_u, d_cell_v, tiles.aspect[j], kind))
    table = np.array(sorted(rows), dtype=np.int64)
    if np.abs(table[:, 1:3]).max() > 3:
        raise RuntimeError("cell offset beyond the lookup margin")
    return table


def critical_values(tiling_type: TilingType, index: int) -> NDArray[np.float64]:
    """Values of parameter ``index`` (the others at their defaults) where validity of the prototile can change.

    The vertices are affine in the parameter, so every orientation determinant of three vertices (which decides
    crossings, touching, folds and the area) is a quadratic polynomial, and every edge component is linear. Validity
    is constant between the real roots of these polynomials; at a root it can differ (a degenerate prototile).
    """
    coeffs = tiling_type.vertex_coeffs
    slope = coeffs[:, :, index]
    p = tiling_type.default_params.copy()
    p[index] = 0.0
    offset = coeffs[:, :, :-1] @ p + coeffs[:, :, -1]  # vertices at parameter value 0
    roots: list[float] = []

    def add_linear(c0: float, c1: float) -> None:
        if abs(c1) > 1e-12:
            roots.append(-c0 / c1)

    def add_quadratic(c0: float, c1: float, c2: float) -> None:
        if abs(c2) < 1e-12:
            add_linear(c0, c1)
            return
        disc = c1 * c1 - 4.0 * c2 * c0
        if disc >= 0.0:
            root = math.sqrt(disc)
            roots.extend([(-c1 - root) / (2.0 * c2), (-c1 + root) / (2.0 * c2)])

    n = tiling_type.n_vertices
    for k in range(n):
        for axis in range(2):
            add_linear(offset[(k + 1) % n, axis] - offset[k, axis], slope[(k + 1) % n, axis] - slope[k, axis])
    for a, b, c in itertools.combinations(range(n), 3):
        u0, u1 = offset[b] - offset[a], slope[b] - slope[a]
        w0, w1 = offset[c] - offset[a], slope[c] - slope[a]
        add_quadratic(
            u0[0] * w0[1] - u0[1] * w0[0],
            u0[0] * w1[1] + u1[0] * w0[1] - u0[1] * w1[0] - u1[1] * w0[0],
            u1[0] * w1[1] - u1[1] * w1[0],
        )
    unique: NDArray[np.float64] = np.unique(np.array(roots))
    return unique


def safe_ranges(tiling_type: TilingType) -> NDArray[np.float64]:
    """(n_params, 2) interval per parameter around its default in which the straight prototile is valid."""
    defaults = tiling_type.default_params

    def valid(index: int, value: float) -> bool:
        p = defaults.copy()
        p[index] = value
        return check_polygon(tiling_type.vertex_coeffs @ np.append(p, 1.0)).valid

    out = np.empty((tiling_type.n_params, 2))
    for index in range(tiling_type.n_params):
        default = float(defaults[index])
        if not valid(index, default):
            raise RuntimeError(f"IH{tiling_type.number}: the default parameters are invalid")
        roots = critical_values(tiling_type, index)
        for side, sign in enumerate((-1.0, 1.0)):
            distances = np.sort(sign * (roots[sign * (roots - default) > 1e-9] - default))
            distances = distances[distances <= SCAN_LIMIT]
            bounds = [*distances.tolist(), SCAN_LIMIT]
            limit = None  # distance of the first invalid value
            for k, distance in enumerate(bounds[:-1]):
                middle = (distance + bounds[k + 1]) / 2.0
                if not valid(index, default + sign * distance) or not valid(index, default + sign * middle):
                    limit = distance
                    break
            if limit is None:
                out[index, side] = default + sign * SCAN_LIMIT
            else:
                steps = max(math.floor((limit - 1e-9) / RANGE_GRID), 0)
                out[index, side] = default + sign * (steps * RANGE_GRID)
    return out


def build_arrays() -> dict[str, NDArray[np.generic]]:
    """All arrays of the data file, computed from tactile and this package's own geometry."""
    arrays = export_tactile()
    for n in arrays["numbers"].tolist():
        tiling_type = type_from_arrays(arrays, n)
        arrays[f"t{n}_adjacency"] = adjacency_table(IsohedralTiling(tiling_type))
        arrays[f"t{n}_safe_ranges"] = safe_ranges(tiling_type)
    return arrays


def write_arrays(path: Path, arrays: dict[str, NDArray[np.generic]]) -> None:
    """Write an uncompressed ``.npz`` with sorted keys and fixed time stamps (byte-identical for equal arrays)."""
    with zipfile.ZipFile(path, "w", zipfile.ZIP_STORED) as archive:
        for key in sorted(arrays):
            info = zipfile.ZipInfo(f"{key}.npy", date_time=FIXED_TIME)
            info.compress_type = zipfile.ZIP_STORED
            info.external_attr = 0o644 << 16
            with archive.open(info, "w", force_zip64=True) as f:
                np.save(f, np.require(arrays[key], requirements="C"), allow_pickle=False)


def differences(built: dict[str, NDArray[np.generic]], stored: dict[str, NDArray[np.generic]]) -> list[str]:
    """Keys whose dtype, shape or values differ between two sets of arrays (including missing keys)."""
    out = [f"only in one of the two: {key}" for key in sorted(built.keys() ^ stored.keys())]
    for key in sorted(built.keys() & stored.keys()):
        a, b = built[key], stored[key]
        if a.dtype != b.dtype or a.shape != b.shape or not np.array_equal(a, b):
            out.append(f"differs: {key}")
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0] if __doc__ else None)
    parser.add_argument("--check", action="store_true", help="rebuild the arrays and fail on any difference")
    args = parser.parse_args(argv)
    arrays = build_arrays()
    if args.check:
        with np.load(DATA_FILE) as z:
            stored = {key: z[key] for key in z.files}
        problems = differences(arrays, stored)
        if problems:
            print(f"{DATA_FILE} is not up to date:\n  " + "\n  ".join(problems), file=sys.stderr)
            return 1
        print(f"{DATA_FILE} is up to date ({len(arrays['numbers'])} types)")
        return 0
    write_arrays(DATA_FILE, arrays)
    print(f"wrote {DATA_FILE} ({len(arrays['numbers'])} types)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
