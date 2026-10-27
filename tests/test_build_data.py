"""The stored data file is what `_build_data` computes from tactile, byte for byte."""

import numpy as np
import pytest

from isotile import _build_data
from isotile.types import DATA_FILE

pytest.importorskip("tactile")

pytestmark = pytest.mark.tactile


@pytest.fixture(scope="module")
def built():
    return _build_data.build_arrays()


@pytest.fixture(scope="module")
def stored():
    with np.load(DATA_FILE) as z:
        return {key: z[key] for key in z.files}


def test_regeneration_reproduces_the_stored_arrays(built, stored):
    assert _build_data.differences(built, stored) == []


def test_regeneration_is_deterministic_and_byte_identical(built, tmp_path):
    first, second = tmp_path / "first.npz", tmp_path / "second.npz"
    _build_data.write_arrays(first, built)
    _build_data.write_arrays(second, dict(reversed(list(built.items()))))  # insertion order does not matter
    assert first.read_bytes() == second.read_bytes()
    assert first.read_bytes() == DATA_FILE.read_bytes()


def test_dtypes_are_fixed(stored):
    assert stored["numbers"].dtype == np.int64
    for key, array in stored.items():
        if key in ("notice", "numbers"):
            continue
        name = key.split("_", 1)[1]
        expected = (
            np.float64
            if name in {"default_params", "vertex_coeffs", "aspect_coeffs", "translation_coeffs", "safe_ranges"}
            else np.int64
        )
        assert array.dtype == expected, key


def test_differences_reports_changes(stored):
    changed = dict(stored)
    changed["t7_safe_ranges"] = stored["t7_safe_ranges"] + 1e-12
    del changed["t1_coloring"]
    assert _build_data.differences(changed, stored) == [
        "only in one of the two: t1_coloring",
        "differs: t7_safe_ranges",
    ]


def test_check_mode_succeeds(capsys, monkeypatch, built):
    monkeypatch.setattr(_build_data, "build_arrays", lambda: built)
    assert _build_data.main(["--check"]) == 0
    broken = dict(built)
    broken["t7_adjacency"] = built["t7_adjacency"][:-1]
    monkeypatch.setattr(_build_data, "build_arrays", lambda: broken)
    assert _build_data.main(["--check"]) == 1
    assert "t7_adjacency" in capsys.readouterr().err
