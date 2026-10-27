"""The Python code blocks of the documentation run, and the API reference lists every public name."""

import re
from pathlib import Path

import pytest

import isotile

DOCS = Path(__file__).parent.parent / "docs"
PAGES = ["index.md", "quickstart.md", "concepts.md", "adjacency.md", "development.md"]
FENCE = re.compile(r"^```(\w*)[^\n]*\n(.*?)^```", re.S | re.M)


def python_blocks(path: Path) -> list[str]:
    return [body for language, body in FENCE.findall(path.read_text(encoding="utf-8")) if language == "python"]


@pytest.mark.parametrize("page", PAGES)
def test_python_blocks_run(page, tmp_path, monkeypatch):
    path = DOCS / page
    if not path.exists():
        pytest.skip("not a source checkout")
    matplotlib = pytest.importorskip("matplotlib") if "matplotlib" in path.read_text(encoding="utf-8") else None
    if matplotlib is not None:
        matplotlib.use("Agg")
    monkeypatch.chdir(tmp_path)
    namespace: dict[str, object] = {"__name__": "__docs__"}
    for number, block in enumerate(python_blocks(path), start=1):
        exec(compile(block, f"{page}, block {number}", "exec"), namespace)  # noqa: S102


def test_pages_with_code_are_listed():
    """A page with a Python block must be one of the tested pages."""
    if not DOCS.exists():
        pytest.skip("not a source checkout")
    untested = [
        str(p.relative_to(DOCS))
        for p in DOCS.rglob("*.md")
        if p.relative_to(DOCS).as_posix() not in PAGES and python_blocks(p)
    ]
    assert not untested, f"pages with Python blocks that no test runs: {untested}"


def test_api_reference_lists_every_public_name():
    api = DOCS / "api"
    if not api.exists():
        pytest.skip("not a source checkout")
    documented = [
        name
        for page in sorted(api.glob("*.md"))
        for name in re.findall(r"^::: isotile\.(\w+)\s*$", page.read_text(encoding="utf-8"), re.M)
    ]
    assert sorted(documented) == sorted(isotile.__all__)


def test_public_names_are_importable():
    assert isotile.__all__
    for name in isotile.__all__:
        assert hasattr(isotile, name), name
