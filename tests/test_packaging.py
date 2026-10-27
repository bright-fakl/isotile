"""Data files and license texts are part of the package and of the built wheel and sdist."""

import importlib.resources
import sys
import tarfile
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
DATA_FILES = {"types.npz", "LICENSE-tactile"}


def test_data_files_are_installed_with_the_package():
    data = importlib.resources.files("isotile") / "data"
    for name in DATA_FILES:
        assert (data / name).is_file(), name
    assert (importlib.resources.files("isotile") / "py.typed").is_file()
    assert "Craig S. Kaplan" in (data / "LICENSE-tactile").read_text()


def test_notice_carries_the_tactile_license():
    notice = ROOT / "NOTICE"
    if not notice.exists():
        pytest.skip("not a source checkout")
    text = notice.read_text()
    license_text = (importlib.resources.files("isotile") / "data" / "LICENSE-tactile").read_text()
    assert license_text.strip() in text


@pytest.mark.skipif(sys.platform == "win32", reason="builds are checked on Linux")
def test_built_wheel_and_sdist_contain_the_data(tmp_path, monkeypatch):
    build = pytest.importorskip("hatchling.build")
    if not (ROOT / "pyproject.toml").exists():
        pytest.skip("not a source checkout")
    monkeypatch.chdir(ROOT)
    wheel = tmp_path / build.build_wheel(str(tmp_path))
    sdist = tmp_path / build.build_sdist(str(tmp_path))
    with zipfile.ZipFile(wheel) as zf:
        names = set(zf.namelist())
    for name in DATA_FILES:
        assert f"isotile/data/{name}" in names
    assert "isotile/py.typed" in names
    assert any(n.endswith(".dist-info/licenses/LICENSE") for n in names)
    assert any(n.endswith(".dist-info/licenses/NOTICE") for n in names)
    assert not any(n.startswith(("tests/", "benchmarks/")) for n in names)
    with tarfile.open(sdist) as tf:
        sdist_names = tf.getnames()
    for name in DATA_FILES:
        assert any(n.endswith(f"src/isotile/data/{name}") for n in sdist_names), name
    for name in ("LICENSE", "NOTICE", "README.md", "pyproject.toml"):
        assert any(n.endswith(f"/{name}") for n in sdist_names), name
