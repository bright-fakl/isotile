import numpy as np
import pytest

from isotile_proto import IsohedralTiling, tiling_types

ALL_TYPES = tiling_types()


def random_tilings(number: int, n: int, spread: float, seed: int) -> list[IsohedralTiling]:
    """Up to ``n`` tilings of a type with parameters = defaults + uniform(-spread, spread); invalid draws are skipped."""
    rng = np.random.default_rng(seed + number)
    base = IsohedralTiling(number)
    out = []
    for _ in range(4 * n):
        if len(out) == n:
            break
        try:
            out.append(IsohedralTiling(number, base.parameters + rng.uniform(-spread, spread, len(base.parameters))))
        except ValueError:
            continue
    return out


@pytest.fixture(params=ALL_TYPES, ids=[f"IH{n}" for n in ALL_TYPES])
def number(request) -> int:
    return request.param
