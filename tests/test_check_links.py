"""scripts/check_links.py: link collection, ignore rules and the result of a check (no network)."""

import importlib.util
import urllib.error
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parent.parent / "scripts" / "check_links.py"


@pytest.fixture
def links():
    if not SCRIPT.exists():
        pytest.skip("not a source checkout")
    spec = importlib.util.spec_from_file_location("check_links", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_main_reports_broken_links_and_honors_ignores(links, tmp_path, monkeypatch, capsys):
    page = tmp_path / "page.md"
    page.write_text(
        '[ok](https://example.org/ok), <a href="https://example.org/gone">x</a>, https://example.org/later.\n'
        "http://localhost:8000/x https://example.org/skipped/1 (https://example.org/paren)\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(links, "check", lambda url: "HTTP 404" if url.endswith("gone") else None)
    status = links.main([str(page), str(tmp_path / "missing.md"), "--ignore", "skipped"])
    out = capsys.readouterr().out
    assert status == 1
    assert "BROKEN HTTP 404: https://example.org/gone" in out
    assert "https://example.org/later" not in out
    assert "4 links checked, 2 skipped, 1 broken" in out


def test_check_accepts_429_and_retries_server_errors(links, monkeypatch):
    calls = []

    def urlopen(request, timeout):
        calls.append(request.full_url)
        raise urllib.error.HTTPError(request.full_url, 429 if "limited" in request.full_url else 503, "x", {}, None)  # type: ignore[arg-type]

    monkeypatch.setattr(links.urllib.request, "urlopen", urlopen)
    monkeypatch.setattr(links.time, "sleep", lambda _: None)
    assert links.check("https://example.org/limited") is None
    assert links.check("https://example.org/down") == "HTTP 503"
    assert len(calls) == 1 + 4
