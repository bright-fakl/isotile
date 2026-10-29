"""Check the built distributions of isotile before they are published.

Run: ``python scripts/check_dist.py DIST_DIR [--version X.Y.Z]``

Checks the wheel and the sdist in ``DIST_DIR``: both exist and have the same version (and the version given), the data
files of ``src/isotile/data`` and ``py.typed`` are in the wheel, the sdist and a wheel built from the sdist, and the
license texts (MIT and the notice of Craig S. Kaplan) are in the wheel and the sdist. Needs ``uv`` on the path for the
wheel built from the sdist.
"""

from __future__ import annotations

import argparse
import pathlib
import subprocess
import sys
import tarfile
import tempfile
import zipfile

ROOT = pathlib.Path(__file__).resolve().parent.parent


def fail(message: str) -> None:
    sys.exit(f"check_dist: {message}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("dist", type=pathlib.Path)
    parser.add_argument("--version", help="the version the distributions must have")
    args = parser.parse_args()

    wheels = sorted(args.dist.glob("isotile-*-py3-none-any.whl"))
    sdists = sorted(args.dist.glob("isotile-*.tar.gz"))
    if len(wheels) != 1 or len(sdists) != 1:
        fail(f"expected one wheel and one sdist in {args.dist}, found {[p.name for p in wheels + sdists]}")
    wheel, sdist = wheels[0], sdists[0]
    version = wheel.name.split("-")[1]
    if sdist.name != f"isotile-{version}.tar.gz":
        fail(f"the sdist {sdist.name} and the wheel {wheel.name} have different versions")
    if args.version is not None and version != args.version:
        fail(f"the distributions have version {version}, expected {args.version}")

    expected = {p.name for p in (ROOT / "src/isotile/data").iterdir() if p.is_file() and p.suffix != ".py"}
    if not expected:
        fail("no data files found in src/isotile/data; the check would be vacuous")

    def wheel_names(path: pathlib.Path) -> set[str]:
        with zipfile.ZipFile(path) as zf:
            return set(zf.namelist())

    with tarfile.open(sdist) as tf:
        sdist_names = set(tf.getnames())
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run(["uv", "build", "--wheel", "--out-dir", tmp, str(sdist)], check=True)  # noqa: S603, S607
        rebuilt = wheel_names(next(pathlib.Path(tmp).glob("isotile-*.whl")))

    built = wheel_names(wheel)
    for label, names in (("wheel", built), ("wheel built from the sdist", rebuilt)):
        for name in sorted(expected):
            if f"isotile/data/{name}" not in names:
                fail(f"data file {name} is not in the {label}")
        if "isotile/py.typed" not in names:
            fail(f"py.typed is not in the {label}")
        for required in ("LICENSE", "NOTICE"):
            if not any(n.endswith(f".dist-info/licenses/{required}") for n in names):
                fail(f"{required} is not in the {label}")
    prefix = f"isotile-{version}/"
    for name in sorted(expected):
        if f"{prefix}src/isotile/data/{name}" not in sdist_names:
            fail(f"data file {name} is not in the sdist")
    for required in ("LICENSE", "NOTICE", "README.md", "CHANGELOG.md", "pyproject.toml"):
        if prefix + required not in sdist_names:
            fail(f"{required} is not in the sdist")
    print(
        f"isotile {version}: wheel, sdist and the wheel built from the sdist contain {sorted(expected)}, py.typed and the licenses"
    )


if __name__ == "__main__":
    main()
