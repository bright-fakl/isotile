"""Results do not depend on the iteration order of dicts and sets (hash randomization)."""

import os
import subprocess
import sys
from pathlib import Path

from isotile import IsohedralTiling, find_types

SCRIPT = """
import hashlib, json
import numpy as np
from isotile import IsohedralTiling, find_types, get_type, tiling_types

digest = hashlib.sha256()
def feed(obj):
    digest.update(json.dumps(obj, sort_keys=False).encode())

feed(find_types(edge_types={"J", "S", "U"}))
feed(find_types(edge_types={"S", "I"}, has_reflections=True))
feed([get_type(n).info() for n in tiling_types()])
feed([get_type(n).describe() for n in (7, 20, 46, 84)])
for n in tiling_types():
    tiles = IsohedralTiling(n).fill_region((0, 0, 3, 3), n_tiles=60)
    for kind in ("edge", "vertex", "vertex-only"):
        adj = tiles.adjacency(kind)
        digest.update(adj.indptr.tobytes()); digest.update(adj.indices.tobytes())
    digest.update(tiles.transforms.tobytes()); digest.update(tiles.colors().tobytes())
print(digest.hexdigest())
"""


def run(hash_seed: str) -> str:
    env = {**os.environ, "PYTHONHASHSEED": hash_seed}
    root = Path(__file__).parent.parent
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(root / "src"), env.get("PYTHONPATH", "")]))
    out = subprocess.run([sys.executable, "-c", SCRIPT], env=env, capture_output=True, text=True, check=True)
    return out.stdout.strip()


def test_results_are_identical_under_different_hash_seeds():
    digests = {run(seed) for seed in ("0", "1", "12345")}
    assert len(digests) == 1


def test_edge_type_filter_ignores_set_order():
    shapes = ["J", "U", "S", "I"]
    results = {tuple(find_types(edge_types=set(order))) for order in (shapes, shapes[::-1], shapes[1:] + shapes[:1])}
    assert len(results) == 1


def test_neighbors_are_sorted():
    tiles = IsohedralTiling(7).cover((-6, -6, 6, 6))
    for i in range(0, len(tiles), 17):
        n = tiles.neighbors(i, "vertex")
        assert list(n) == sorted(n)
