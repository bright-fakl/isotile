"""Time of the validity test of curved prototiles: typical valid curves, near collisions and invalid curves.

uv run python benchmarks/bench_curves.py
"""

from __future__ import annotations

import time

import numpy as np

from isotile import check_prototile, enforce_symmetry, get_type
from isotile.curves import check_curves, normalize_curves


def milliseconds(function, repeat: int = 30) -> float:
    function()
    start = time.perf_counter()
    for _ in range(repeat):
        function()
    return (time.perf_counter() - start) / repeat * 1e3


def random_curves(number: int, scale: float, seed: int) -> dict[int, list[tuple[float, float]]]:
    """Curves with 5 inner control points on every curvable shape id; ``scale`` multiplies the heights."""
    t = get_type(number)
    rng = np.random.default_rng(seed)
    curves = {}
    for i in sorted(set(t.edge_shape_ids.tolist())):
        if t.shape_types[i] == "I":
            continue
        x = np.linspace(0, 1, 7)
        y = scale * rng.uniform(-1, 1, 7)
        y[[0, -1]] = 0.0
        curves[i] = enforce_symmetry(np.column_stack([x, y]), t.shape_types[i]).tolist()
    return curves


def threshold(number: int, seed: int) -> tuple[float, float]:
    """The largest valid and the smallest invalid scale of the random curves (bisection)."""
    low, high = 0.0, 4.0
    for _ in range(40):
        mid = (low + high) / 2
        if check_prototile(number, edge_curves=random_curves(number, mid, seed)).valid:
            low = mid
        else:
            high = mid
    return low, high


def cases() -> list[tuple[str, int, dict[int, list[tuple[float, float]]]]]:
    out = []
    for number in (28, 4, 41, 66, 13):
        out.append(("typical", number, random_curves(number, 0.05, number)))
        low, high = threshold(number, number)
        out.append(("near collision", number, random_curves(number, low, number)))
        out.append(("just invalid", number, random_curves(number, high, number)))
        out.append(("invalid", number, random_curves(number, 3.0, number)))
    return out


def main() -> None:
    print(f"{'case':16} {'type':>5} {'valid':>6} {'check_prototile':>16} {'check_curves':>13}")
    for name, number, curves in cases():
        t = get_type(number)
        report = check_prototile(number, edge_curves=curves)
        normalized = normalize_curves(t, curves, None)
        a = milliseconds(lambda n=number, c=curves: check_prototile(n, edge_curves=c))
        b = milliseconds(lambda t=t, c=normalized: check_curves(t, t.default_params, c))
        print(f"{name:16} {'IH' + str(number):>5} {report.valid!s:>6} {a:13.2f} ms {b:10.2f} ms")


if __name__ == "__main__":
    main()
