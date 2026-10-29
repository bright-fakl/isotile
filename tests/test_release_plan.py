"""scripts/release_plan.py: which index a publish run targets and when it refuses a tag."""

import importlib.util
import urllib.error
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parent.parent / "scripts" / "release_plan.py"


@pytest.fixture
def plan(monkeypatch, capsys):
    if not SCRIPT.exists():
        pytest.skip("not a source checkout")
    spec = importlib.util.spec_from_file_location("release_plan", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module.urllib.request, "urlopen", _not_found)

    def run(event, ref_type="", ref_name="", version="0.1.0"):
        monkeypatch.setenv("EVENT", event)
        monkeypatch.setenv("REF_TYPE", ref_type)
        monkeypatch.setenv("REF_NAME", ref_name)
        monkeypatch.setattr(module.tomllib, "loads", lambda _: {"project": {"version": version}})
        module.main()
        return dict(line.split("=") for line in capsys.readouterr().out.split())

    return run


def _not_found(url, timeout):
    raise urllib.error.HTTPError(url, 404, "Not Found", {}, None)  # type: ignore[arg-type]


def test_final_tag_goes_to_pypi(plan):
    assert plan("push", "tag", "v0.1.0") == {"version": "0.1.0", "target": "pypi", "published": "false"}


def test_prerelease_tag_goes_to_testpypi_and_normalizes(plan):
    assert plan("release", "tag", "v0.2.0-rc1", version="0.2.0rc1")["target"] == "testpypi"


def test_manual_run_goes_to_testpypi(plan):
    assert plan("workflow_dispatch", "branch", "main", version="0.1.0.dev0")["target"] == "testpypi"


def test_tag_must_equal_the_version(plan):
    with pytest.raises(SystemExit, match="not the package version"):
        plan("push", "tag", "v0.1.1")


def test_branch_is_refused(plan):
    with pytest.raises(SystemExit, match="not a tag"):
        plan("push", "branch", "main")


def test_published_version_is_reported(plan, monkeypatch):
    import urllib.request

    monkeypatch.setattr(urllib.request, "urlopen", lambda url, timeout: None)
    assert plan("push", "tag", "v0.1.0")["published"] == "true"
