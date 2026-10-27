"""Adjacency of tiles: from the lattice table of a type (exact, no geometry) or from shared vertices."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import coo_array, csr_array
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from .types import KIND_EDGE, KIND_VERTEX, TilingType

AdjacencyKind = Literal["edge", "vertex", "vertex-only"]

_KINDS = {"edge": (KIND_EDGE,), "vertex": (KIND_EDGE, KIND_VERTEX), "vertex-only": (KIND_VERTEX,)}


def lattice_adjacency(
    cell: NDArray[np.int64], aspect: NDArray[np.int64], tiling_type: TilingType, kind: AdjacencyKind
) -> csr_array:
    """Adjacency of tiles given by their cells (n, 2) and aspects (n,), from the type's table.

    The table does not depend on the parameters, the frame or the position of the tiling. Pairs where one tile is
    not in the set are dropped. ``kind``: ``"edge"`` (shared edge), ``"vertex"`` (shared edge or vertex),
    ``"vertex-only"`` (shared vertex, no shared edge).
    """
    try:
        wanted = _KINDS[kind]
    except KeyError:
        raise ValueError(f"kind must be 'edge', 'vertex' or 'vertex-only', got {kind!r}") from None
    n = len(cell)
    if n == 0:
        return csr_array((0, 0), dtype=bool)
    n_aspects = tiling_type.n_aspects
    low = cell.min(axis=0) - 3
    width = cell[:, 1].max() - low[1] + 4

    def key(u: NDArray[np.int64], v: NDArray[np.int64], a: int | NDArray[np.int64]) -> NDArray[np.int64]:
        combined: NDArray[np.int64] = ((u - low[0]) * width + (v - low[1])) * n_aspects + a
        return combined

    keys = key(cell[:, 0], cell[:, 1], aspect)
    order = np.argsort(keys)
    sorted_keys = keys[order]
    rows: list[NDArray[np.int64]] = []
    cols: list[NDArray[np.int64]] = []
    for from_aspect, d_cell_u, d_cell_v, neighbor_aspect, k in tiling_type.adjacency.tolist():
        if k not in wanted:
            continue
        source = np.flatnonzero(aspect == from_aspect)
        target = key(cell[source, 0] + d_cell_u, cell[source, 1] + d_cell_v, neighbor_aspect)
        position = np.minimum(np.searchsorted(sorted_keys, target), n - 1)
        found = sorted_keys[position] == target
        rows.append(source[found])
        cols.append(order[position[found]])
    row = np.concatenate(rows) if rows else np.empty(0, dtype=np.int64)
    col = np.concatenate(cols) if cols else np.empty(0, dtype=np.int64)
    return coo_array((np.ones(len(row), dtype=bool), (row, col)), shape=(n, n)).tocsr()


def geometric_adjacency(
    polygons: Sequence[NDArray[np.float64]] | NDArray[np.float64], tol: float = 1e-6
) -> tuple[csr_array, csr_array]:
    """Edge and vertex-only adjacency of polygons given as (k, 2) vertex arrays, by counting shared vertices.

    Vertices closer than ``tol`` are one vertex. Two polygons share an edge if they share 2 or more vertices and
    are vertex-only neighbors if they share exactly one.
    """
    pts = np.concatenate(polygons)
    owner = np.repeat(np.arange(len(polygons)), [len(p) for p in polygons])
    pairs = cKDTree(pts).query_pairs(tol, output_type="ndarray")
    m = len(pts)
    graph = coo_array((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(m, m))
    _, vertex_id = connected_components(graph, directed=False)
    incidence = coo_array((np.ones(m), (owner, vertex_id))).tocsr()
    incidence.data[:] = 1.0  # duplicate (polygon, vertex) entries were summed
    shared = (incidence @ incidence.T).tocoo()
    off_diagonal = shared.row != shared.col
    row, col, count = shared.row[off_diagonal], shared.col[off_diagonal], shared.data[off_diagonal]
    n = len(polygons)

    def build(mask: NDArray[np.bool_]) -> csr_array:
        return coo_array((np.ones(int(mask.sum()), dtype=bool), (row[mask], col[mask])), shape=(n, n)).tocsr()

    return build(count >= 2), build(count == 1)
