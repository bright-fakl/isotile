"""benchmarks/bench.py and the catalog hook of the documentation keep working."""

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent


def load(path: Path, name: str):
    if not path.exists():
        pytest.skip("not a source checkout")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_benchmark_runs_at_a_tiny_size(capsys):
    load(ROOT / "benchmarks" / "bench.py", "bench").main(["--sizes", "60", "--repeat", "1", "--markdown"])
    out = capsys.readouterr().out
    assert "| IH7 |" in out
    assert "check_prototile" in out


def test_catalog_page_lists_all_81_types():
    pytest.importorskip("matplotlib")
    from isotile import tiling_types

    page = load(ROOT / "docs" / "hooks.py", "docs_hooks").catalog_page()
    for number in tiling_types():
        assert f"**IH{number}**" in page
    assert page.count("| **IH") == 81
