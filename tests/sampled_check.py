"""The sampled polyline test that the library used before the certified clearance test (kept for comparisons)."""

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]
IntArray = NDArray[np.int64]

_ORIENT_TOL = 1e-12
_LENGTH_TOL = 1e-12
_AREA_TOL = 1e-9


def check_outline(points: Array, starts: IntArray) -> tuple[tuple[int, ...], str]:
    """Test that the closed polyline is simple and has positive area.

    Returns the sorted edges at fault (edge ``i`` owns the segments ``starts[i]`` to ``starts[i + 1] - 1``) and the
    message; the message is ``"valid"`` and the edges empty if there is no fault. Faults: segments of zero length;
    two segments that are not neighbors and cross, touch or overlap; neighboring segments that fold back onto each
    other; an area that is not positive (up to a relative ``1e-9`` of the squared extent).
    """
    n = len(points)
    nxt = np.roll(points, -1, axis=0)
    seg = nxt - points
    length = np.linalg.norm(seg, axis=1)
    extent = float(np.ptp(points, axis=0).max())
    owner = np.repeat(np.arange(len(starts) - 1), np.diff(starts))
    problems: list[str] = []
    bad: set[int] = set()

    short = np.flatnonzero(length <= _LENGTH_TOL * extent)
    if len(short):
        bad.update(owner[short].tolist())
        problems.append(f"zero-length segments in edges {sorted(set(owner[short].tolist()))}")

    tol = _ORIENT_TOL * extent**2

    def side(p: Array, q: Array, r: Array) -> Array:
        c = (q[:, 0] - p[:, 0]) * (r[:, 1] - p[:, 1]) - (q[:, 1] - p[:, 1]) * (r[:, 0] - p[:, 0])
        signs: Array = np.where(np.abs(c) <= tol, 0.0, np.sign(c))
        return signs

    def on_segment(p: Array, q: Array, r: Array) -> NDArray[np.bool_]:
        """r lies in the bounding box of pq (r is known to be collinear with p and q)."""
        inside: NDArray[np.bool_] = (
            (np.minimum(p[:, 0], q[:, 0]) <= r[:, 0])
            & (r[:, 0] <= np.maximum(p[:, 0], q[:, 0]))
            & (np.minimum(p[:, 1], q[:, 1]) <= r[:, 1])
            & (r[:, 1] <= np.maximum(p[:, 1], q[:, 1]))
        )
        return inside

    lo, hi = np.minimum(points, nxt), np.maximum(points, nxt)
    eps = 1e-9 * extent
    close = np.ones((n, n), dtype=bool)
    for axis in (0, 1):
        close &= (lo[:, None, axis] <= hi[None, :, axis] + eps) & (lo[None, :, axis] <= hi[:, None, axis] + eps)
    candidate = np.triu(close, 2)
    candidate[0, n - 1] = False
    i, j = np.nonzero(candidate)
    p, q, r, s = points[i], nxt[i], points[j], nxt[j]
    d1, d2, d3, d4 = side(p, q, r), side(p, q, s), side(r, s, p), side(r, s, q)
    hit = (
        ((d1 * d2 < 0) & (d3 * d4 < 0))
        | ((d1 == 0) & on_segment(p, q, r))
        | ((d2 == 0) & on_segment(p, q, s))
        | ((d3 == 0) & on_segment(r, s, p))
        | ((d4 == 0) & on_segment(r, s, q))
    )
    if hit.any():
        pairs = sorted({(int(owner[a]), int(owner[b])) for a, b in zip(i[hit], j[hit], strict=True)})
        bad.update(x for pair in pairs for x in pair)
        problems.append(
            "edges cross or touch: " + ", ".join(f"{a}-{b}" if a != b else f"{a} (itself)" for a, b in pairs)
        )

    following = np.roll(seg, -1, axis=0)
    cross = seg[:, 0] * following[:, 1] - seg[:, 1] * following[:, 0]
    folded = np.flatnonzero((np.abs(cross) <= tol) & (np.sum(seg * following, axis=1) < 0))
    if len(folded):
        edges = sorted(set(owner[folded].tolist()) | set(owner[(folded + 1) % n].tolist()))
        bad.update(edges)
        problems.append(f"the curve folds back onto itself in edges {edges}")

    area = 0.5 * float(np.sum(points[:, 0] * nxt[:, 1] - nxt[:, 0] * points[:, 1]))
    if not area > _AREA_TOL * extent**2:
        if not bad:
            bad.update(range(len(starts) - 1))
        problems.append("the outline does not have positive area")

    if not problems:
        return (), "valid"
    return tuple(sorted(bad)), "; ".join(problems)
