"""Certified test that a closed outline of cubic Bezier pieces is a simple curve with clearance.

The outline is a closed chain of cubic Bezier pieces (straight pieces are degenerate Beziers). With ``tol`` the
clearance, the test certifies that

- every piece is a simple arc: the tangent turns by less than 180 degrees across it and does not vanish;
- at every join of consecutive pieces the tangent turns by less than 180 degrees (no fold-back);
- parts of the outline that are not directly consecutive pieces are at least ``tol`` apart, and parts of two
  consecutive pieces are at least ``tol`` apart except where they contain the join.

Pairs of pieces are separated by recursive subdivision (de Casteljau): two pieces whose control-point boxes, each
grown by ``tol / 2``, are disjoint are at least ``tol`` apart; otherwise the one with the larger extent is split.
Pieces that deviate from their chord by at most ``tol / 4`` are compared as segments. The subdivision of all
candidate pairs runs level by level on numpy arrays.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]
IntArray = NDArray[np.int64]
Mask = NDArray[np.bool_]

_GAUSS = ((np.polynomial.legendre.leggauss(4)[0] + 1) / 2, np.polynomial.legendre.leggauss(4)[1] / 2)
_MAX_LEVELS = 80
_FOLD = np.pi - 1e-9  # a turn of at least this much is a fold-back
_SMOOTH = 3 * np.pi / 4  # consecutive pieces whose tangents stay within this arc form a graph over a direction


def split(p: Array) -> tuple[Array, Array]:
    """Halves of cubic Bezier pieces ``(..., 4, 2)`` at the parameter 1/2."""
    p0, p1, p2, p3 = p[..., 0, :], p[..., 1, :], p[..., 2, :], p[..., 3, :]
    m = (p0 + 3 * p1 + 3 * p2 + p3) / 8
    left = np.stack([p0, (p0 + p1) / 2, (p0 + 2 * p1 + p2) / 4, m], axis=-2)
    right = np.stack([m, (p1 + 2 * p2 + p3) / 4, (p2 + p3) / 2, p3], axis=-2)
    return left, right


def _extent(p: Array) -> Array:
    """Larger side of the box of the control points."""
    result: Array = np.ptp(p, axis=-2).max(axis=-1)
    return result


def _flat(p: Array, tol: float) -> Mask:
    """Pieces within ``tol / 4`` of their chord (or no larger than that)."""
    chord = p[..., 3, :] - p[..., 0, :]
    length = np.linalg.norm(chord, axis=-1)
    safe = np.where(length > 0, length, 1.0)
    unit = chord / safe[..., None]
    dev = np.zeros(p.shape[:-2])
    for k in (1, 2):
        v = p[..., k, :] - p[..., 0, :]
        line = np.abs(v[..., 0] * unit[..., 1] - v[..., 1] * unit[..., 0])
        dev = np.maximum(dev, np.where(length > 0, line, np.linalg.norm(v, axis=-1)))
    flat: Mask = (dev <= tol / 4) | (_extent(p) <= tol / 4)
    return flat


def _point_segment(p: Array, a: Array, b: Array) -> Array:
    ab = b - a
    denom = np.maximum(np.sum(ab * ab, axis=-1), 1e-300)
    t = np.clip(np.sum((p - a) * ab, axis=-1) / denom, 0.0, 1.0)
    distance: Array = np.linalg.norm(p - (a + t[..., None] * ab), axis=-1)
    return distance


def segment_distance(p: Array, q: Array, r: Array, s: Array) -> Array:
    """Distance between the segments ``pq`` and ``rs`` (arrays of shape ``(..., 2)``)."""

    def cross(a: Array, b: Array) -> Array:
        c: Array = a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]
        return c

    o1, o2 = cross(q - p, r - p), cross(q - p, s - p)
    o3, o4 = cross(s - r, p - r), cross(s - r, q - r)
    crossing = (o1 * o2 < 0) & (o3 * o4 < 0)
    d = np.minimum.reduce([
        _point_segment(p, r, s),
        _point_segment(q, r, s),
        _point_segment(r, p, q),
        _point_segment(s, p, q),
    ])
    distance: Array = np.where(crossing, 0.0, d)
    return distance


def _spread(vectors: Array) -> Array:
    """Angle of the smallest arc that holds the directions of ``(..., k, 2)`` vectors; infinity if one is zero."""
    angle = np.arctan2(vectors[..., 1], vectors[..., 0])
    rel = (angle - angle[..., :1] + np.pi) % (2 * np.pi) - np.pi
    spread: Array = rel.max(axis=-1) - rel.min(axis=-1)
    zero = (np.abs(vectors).max(axis=-1) == 0).any(axis=-1)
    return np.where(zero, np.inf, spread)


def bounding_box(pieces: Array) -> tuple[Array, Array]:
    """Exact lower and upper corner of the box of the pieces ``(n, 4, 2)`` (derivative roots included)."""
    d = np.diff(pieces, axis=1)  # hodograph / 3: a (1-u)^2 + 2 b u (1-u) + c u^2
    qa = d[:, 0] - 2 * d[:, 1] + d[:, 2]
    qb = 2 * (d[:, 1] - d[:, 0])
    qc = d[:, 0]
    pts = [pieces[:, 0], pieces[:, 3]]
    disc = qb**2 - 4 * qa * qc
    for sign in (1.0, -1.0):
        with np.errstate(divide="ignore", invalid="ignore"):
            quad = (-qb + sign * np.sqrt(np.maximum(disc, 0.0))) / (2 * qa)
            lin = -qc / qb
        u = np.where(np.abs(qa) > 1e-14 * np.maximum(np.abs(qb), 1e-300), quad, lin)
        u = np.where((disc >= 0) & np.isfinite(u) & (u > 0) & (u < 1), u, 0.0)  # (n, 2): per coordinate
        w = 1 - u
        c = pieces
        value = (
            w**3 * c[:, 0] + 3 * w**2 * u * c[:, 1] + 3 * w * u**2 * c[:, 2] + u**3 * c[:, 3]
        )  # each coordinate evaluated at its own root
        pts.append(value)
    stack = np.stack(pts)
    return stack.min(axis=(0, 1)), stack.max(axis=(0, 1))


def signed_area(pieces: Array) -> float:
    """Exact signed area of the closed chain of pieces (Green's theorem, Gauss-Legendre of degree 7)."""
    u, w = _GAUSS
    b = np.stack([(1 - u) ** 3, 3 * (1 - u) ** 2 * u, 3 * (1 - u) * u**2, u**3], axis=-1)  # (4, 4)
    db = 3 * np.stack([-((1 - u) ** 2), (1 - u) ** 2 - 2 * (1 - u) * u, 2 * (1 - u) * u - u**2, u**2], axis=-1)
    pos = np.einsum("qk,nkc->nqc", b, pieces)
    vel = np.einsum("qk,nkc->nqc", db, pieces)
    integrand = pos[..., 0] * vel[..., 1] - pos[..., 1] * vel[..., 0]
    return float(0.5 * np.sum(integrand * w))


def _turning(pieces: Array, tol: float) -> tuple[Array, Mask]:
    """Total tangent turning of each piece and whether its tangent vanishes (cusp)."""
    n = len(pieces)
    total = np.zeros(n)
    cusp = np.zeros(n, dtype=bool)
    current: Array = pieces
    ids: IntArray = np.arange(n)
    for _ in range(_MAX_LEVELS):
        d = np.diff(current, axis=1)
        ok = _spread(d) < np.pi
        angle = np.arctan2(d[..., 1], d[..., 0])
        turn = (angle[:, 2] - angle[:, 0] + np.pi) % (2 * np.pi) - np.pi
        np.add.at(total, ids[ok], turn[ok])
        current, ids = current[~ok], ids[~ok]
        if not len(ids):
            return total, cusp
        tiny = _extent(current) <= tol / 8
        cusp[ids[tiny]] = True
        current, ids = current[~tiny], ids[~tiny]
        left, right = split(current)
        current, ids = np.concatenate([left, right]), np.concatenate([ids, ids])
    cusp[ids] = True
    return total, cusp


def _end_tangent(p: Array) -> Array:
    """Direction at the end of pieces: the first nonzero of the last three control point differences."""
    cands = np.stack([p[:, 3] - p[:, 2], p[:, 3] - p[:, 1], p[:, 3] - p[:, 0]], axis=1)
    first = np.argmax(np.abs(cands).max(axis=2) > 0, axis=1)
    result: Array = cands[np.arange(len(p)), first]
    return result


def _start_tangent(p: Array) -> Array:
    cands = np.stack([p[:, 1] - p[:, 0], p[:, 2] - p[:, 0], p[:, 3] - p[:, 0]], axis=1)
    first = np.argmax(np.abs(cands).max(axis=2) > 0, axis=1)
    result: Array = cands[np.arange(len(p)), first]
    return result


def _refine(a: Array, b: Array, ta: Mask, tb: Mask, pid: IntArray, n_pairs: int, tol: float) -> Mask:
    """Pairs (of ``n_pairs``) that contain two parts closer than ``tol``; ``ta`` and ``tb`` mark parts at the join."""
    hit = np.zeros(n_pairs, dtype=bool)
    pad = tol / 2
    for _ in range(_MAX_LEVELS):
        alive = ~hit[pid]
        a, b, ta, tb, pid = a[alive], b[alive], ta[alive], tb[alive], pid[alive]
        if not len(pid):
            return hit
        apart = ((a.min(axis=1) - pad > b.max(axis=1) + pad) | (b.min(axis=1) - pad > a.max(axis=1) + pad)).any(axis=1)
        a, b, ta, tb, pid = a[~apart], b[~apart], ta[~apart], tb[~apart], pid[~apart]
        if not len(pid):
            return hit
        flat_a, flat_b = _flat(a, tol), _flat(b, tol)
        both = flat_a & flat_b
        if both.any():
            i = np.flatnonzero(both)
            d = segment_distance(a[i, 0], a[i, 3], b[i, 0], b[i, 3])
            hit[pid[i[(d < tol) & ~(ta[i] & tb[i])]]] = True
        rest = ~both
        a, b, ta, tb, pid = a[rest], b[rest], ta[rest], tb[rest], pid[rest]
        flat_a, flat_b = flat_a[rest], flat_b[rest]
        split_a = ~flat_a & (flat_b | (_extent(a) >= _extent(b)))
        a1, a2 = split(a)
        b1, b2 = split(b)
        sa = split_a[:, None, None]
        a = np.concatenate([np.where(sa, a1, a), np.where(sa, a2, a)])
        b = np.concatenate([np.where(sa, b, b1), np.where(sa, b, b2)])
        ta = np.concatenate([np.where(split_a, False, ta), ta])  # the part at the end of A is its second half
        tb = np.concatenate([tb, np.where(split_a, tb, False)])  # the part at the start of B is its first half
        pid = np.concatenate([pid, pid])
    hit[pid] = True  # not resolved: cannot be certified
    return hit


def check_pieces(pieces: Array, owner: IntArray, n_edges: int, tol: float) -> tuple[tuple[int, ...], list[str]]:
    """Certified test of the closed chain of ``pieces`` ``(N, 4, 2)``, ``owner[k]`` the edge of piece ``k``.

    Returns the sorted edges at fault and a message per kind of fault (empty lists if the outline is valid). The
    area of the outline is not tested here.
    """
    n = len(pieces)
    bad: set[int] = set()
    problems: list[str] = []

    def report(message: str, mask_or_pieces: Mask | IntArray) -> None:
        idx = np.flatnonzero(mask_or_pieces) if mask_or_pieces.dtype == bool else mask_or_pieces
        edges = sorted(set(owner[idx].tolist()))
        bad.update(edges)
        problems.append(f"{message}: {edges}")

    length = np.abs(pieces - pieces[:, :1]).max(axis=(1, 2))
    zero = length <= 1e-12 * max(float(np.ptp(pieces.reshape(-1, 2), axis=0).max()), 1e-300)
    if zero.any():
        report("zero-length pieces in edges", zero)
    usable = ~zero

    total, cusp = _turning(pieces[usable], tol)
    full = np.zeros(n)
    full[usable] = total
    cusp_full = np.zeros(n, dtype=bool)
    cusp_full[usable] = cusp
    folded = usable & (cusp_full | (np.abs(full) >= _FOLD))
    nxt = np.roll(np.arange(n), -1)
    t_end, t_start = _end_tangent(pieces), _start_tangent(pieces[nxt])
    turn = np.arctan2(t_end[:, 0] * t_start[:, 1] - t_end[:, 1] * t_start[:, 0], np.sum(t_end * t_start, axis=1))
    join_fold = usable & usable[nxt] & (np.abs(turn) >= _FOLD)
    if folded.any() or join_fold.any():
        idx = np.concatenate([np.flatnonzero(folded), np.flatnonzero(join_fold), nxt[join_fold]])
        report("the curve folds back (a turn of 180 degrees or more) in edges", idx)

    # candidate pairs: all pairs of pieces that are not consecutive, and the consecutive pairs that need a test
    lo, hi = pieces.min(axis=1) - tol / 2, pieces.max(axis=1) + tol / 2
    close = np.ones((n, n), dtype=bool)
    for axis in (0, 1):
        close &= (lo[:, None, axis] <= hi[None, :, axis]) & (lo[None, :, axis] <= hi[:, None, axis])
    candidate = np.triu(close, 2)
    candidate[0, n - 1] = False
    i, j = np.nonzero(candidate)
    keep = usable[i] & usable[j]
    i, j = i[keep], j[keep]
    flags = np.zeros(len(i), dtype=bool)

    k = np.flatnonzero(usable & usable[nxt])
    d = np.diff(pieces, axis=1)
    smooth = _spread(np.concatenate([d[k], d[nxt[k]]], axis=1)) <= _SMOOTH
    k = k[~smooth]
    ia, ib = np.concatenate([i, k]), np.concatenate([j, nxt[k]])
    flags = np.concatenate([flags, np.ones(len(k), dtype=bool)])
    if len(ia):
        hit = _refine(pieces[ia], pieces[ib], flags, flags, np.arange(len(ia)), len(ia), tol)
        if hit.any():
            edges = np.concatenate([ia[hit], ib[hit]])
            report("edges cross, touch or are closer than the clearance", edges)
    return tuple(sorted(bad)), problems
