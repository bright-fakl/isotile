"""Adjacency of tiles: from the lattice table of a type (exact, no geometry) or from shared vertices."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import coo_array, csr_array
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from .types import KIND_EDGE, KIND_VERTEX, TilingType

if TYPE_CHECKING:
    from .tiling import AdjacencyKind


def lattice_adjacency(lattice: NDArray[np.int64], tiling_type: TilingType, kind: AdjacencyKind) -> csr_array:
    """Adjacency of tiles given by ``(t1, t2, aspect)`` rows, from the type's table (independent of parameters)."""
    kinds = {"edge": (KIND_EDGE,), "vertex": (KIND_EDGE, KIND_VERTEX), "vertex-only": (KIND_VERTEX,)}
    try:
        wanted = kinds[kind]
    except KeyError:
        raise ValueError(f"kind must be 'edge', 'vertex' or 'vertex-only', got {kind!r}") from None
    n = len(lattice)
    if n == 0:
        return csr_array((0, 0), dtype=bool)
    na = tiling_type.n_aspects
    lo = lattice[:, :2].min(axis=0) - 3
    width = lattice[:, 1].max() - lo[1] + 4

    def key(t1, t2, a):
        return ((t1 - lo[0]) * width + (t2 - lo[1])) * na + a

    keys = key(lattice[:, 0], lattice[:, 1], lattice[:, 2])
    order = np.argsort(keys)
    sorted_keys = keys[order]
    rows, cols = [], []
    for a, da, db, a2, k in tiling_type.adjacency:
        if k not in wanted:
            continue
        src = np.flatnonzero(lattice[:, 2] == a)
        target = key(lattice[src, 0] + da, lattice[src, 1] + db, a2)
        pos = np.minimum(np.searchsorted(sorted_keys, target), n - 1)
        found = sorted_keys[pos] == target
        rows.append(src[found])
        cols.append(order[pos[found]])
    r, c = (np.concatenate(x) for x in (rows, cols)) if rows else (np.empty(0, int), np.empty(0, int))
    return coo_array((np.ones(len(r), dtype=bool), (r, c)), shape=(n, n)).tocsr()


def geometric_adjacency(polygons: Sequence[NDArray[np.float64]], tol: float = 1e-6) -> tuple[csr_array, csr_array]:
    """Edge and vertex-only adjacency of polygons given as (k, 2) vertex arrays, by counting shared vertices.

    Vertices closer than ``tol`` are one vertex. Two polygons share an edge if they share 2 or more vertices and
    are vertex-only neighbors if they share exactly one.
    """
    pts = np.concatenate(polygons)
    owner = np.repeat(np.arange(len(polygons)), [len(p) for p in polygons])
    pairs = cKDTree(pts).query_pairs(tol, output_type="ndarray")
    m = len(pts)
    graph = coo_array((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(m, m))
    _, vid = connected_components(graph, directed=False)
    inc = coo_array((np.ones(m), (owner, vid))).tocsr()
    inc.data[:] = 1.0  # duplicate (polygon, vertex) entries were summed
    shared = (inc @ inc.T).tocoo()
    off = shared.row != shared.col
    r, c, s = shared.row[off], shared.col[off], shared.data[off]
    n = len(polygons)

    def build(mask):
        return coo_array((np.ones(mask.sum(), dtype=bool), (r[mask], c[mask])), shape=(n, n)).tocsr()

    return build(s >= 2), build(s == 1)
