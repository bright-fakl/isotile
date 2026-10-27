"""Export the 81 isohedral types from tactile into ``data/types.npz`` and add the lattice adjacency tables.

Development script, not part of the installed package. Needs the ``tactile`` package.

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

Usage: ``python -m isotile_proto._build_data``.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from . import types as types_module
from .adjacency import geometric_adjacency
from .tiling import IsohedralTiling
from .types import DATA_FILE, EDGE_SHAPES, KIND_EDGE, KIND_VERTEX

NOTICE = (
    "Derived from Tactile / tactile-js, Copyright (c) 2018, Craig S. Kaplan, BSD 3-Clause License. "
    "See LICENSE-tactile in this directory."
)
PATCH_RADIUS = 5  # patch of (2R + 1)^2 translation cells around the origin


def export_tactile() -> dict[str, NDArray]:
    """Arrays of all valid types, keyed ``t{number}_{name}``; the adjacency tables start empty."""
    from tactile import EdgeShape, tiling_types
    from tactile.tiling_data import TilingTypeData

    shape_code = {EdgeShape[name]: code for code, name in enumerate(EDGE_SHAPES)}
    out: dict[str, NDArray] = {"numbers": np.array(tiling_types), "notice": np.array(NOTICE)}
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
    return out


def adjacency_table(tiling: IsohedralTiling) -> NDArray[np.int64]:
    """Rows ``(aspect, d_t1, d_t2, neighbor_aspect, kind)`` of the tiles around the origin, by shared vertices."""
    r = PATCH_RADIUS
    cells = np.array([(i, j) for i in range(-r, r + 1) for j in range(-r, r + 1)])
    tiles = tiling.tiles(
        np.column_stack([
            np.repeat(cells, tiling.type.n_aspects, axis=0),
            np.tile(np.arange(tiling.type.n_aspects), len(cells)),
        ])
    )
    edge, vertex = geometric_adjacency(tiles.vertices(), tol=1e-6)
    rows = []
    for adj, kind in ((edge, KIND_EDGE), (vertex, KIND_VERTEX)):
        coo = adj.tocoo()
        for i, j in zip(coo.row, coo.col, strict=True):
            if tuple(tiles.lattice[i, :2]) == (0, 0):  # tiles of the central cell
                da, db = tiles.lattice[j, :2] - tiles.lattice[i, :2]
                rows.append((tiles.lattice[i, 2], da, db, tiles.lattice[j, 2], kind))
    table = np.array(sorted(rows), dtype=np.int64)
    if np.abs(table[:, 1:3]).max() > 3:
        raise RuntimeError("lattice offset beyond the lookup margin")
    return table


def main() -> None:
    arrays = export_tactile()
    np.savez_compressed(DATA_FILE, **arrays)
    types_module._load.cache_clear()
    for n in arrays["numbers"].tolist():
        arrays[f"t{n}_adjacency"] = adjacency_table(IsohedralTiling(n))
    np.savez_compressed(DATA_FILE, **arrays)
    types_module._load.cache_clear()
    print(f"wrote {DATA_FILE} ({len(arrays['numbers'])} types)")


if __name__ == "__main__":
    main()
