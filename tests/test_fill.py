"""fill without tactile: no repeats and a deterministic order."""

import numpy as np
import pytest

from isotile import IsohedralTiling

from .conftest import random_tilings


@pytest.mark.parametrize("seed", range(2))
def test_fill_has_no_repeats(number, seed):
    rng = np.random.default_rng(10 * seed + number)
    tilings = [IsohedralTiling(number), *random_tilings(number, 1, 0.1, seed=seed)]
    for tiling in tilings:
        x0, y0 = rng.uniform(-5, 5, 2)
        width, height = rng.uniform(3, 10, 2)
        tiles = tiling.fill((x0, y0, x0 + width, y0 + height))
        assert len(tiles) > 0
        assert len(np.unique(np.column_stack([tiles.cell, tiles.aspect]), axis=0)) == len(tiles)


def test_fill_is_deterministic():
    a = IsohedralTiling(7).fill((-1.5, -2.5, 6.0, 4.0))
    b = IsohedralTiling(7).fill((-1.5, -2.5, 6.0, 4.0))
    np.testing.assert_array_equal(a.cell, b.cell)
    np.testing.assert_array_equal(a.aspect, b.aspect)


def test_tiles_validates_the_aspect():
    tiling = IsohedralTiling(7)
    with pytest.raises(ValueError, match="aspect"):
        tiling.tiles([(0, 0)], [tiling.type.n_aspects])
    with pytest.raises(ValueError, match="aspect"):
        tiling.tiles([(0, 0)], [-1])
    assert len(tiling.tiles([(0, 0), (1, 2)], 1)) == 2
    assert len(tiling.tiles(np.empty((0, 2), dtype=int), 0)) == 0
