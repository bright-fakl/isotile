"""Decide what the publish workflow does and print it as ``key=value`` lines for ``$GITHUB_OUTPUT``.

Environment: ``EVENT`` (the GitHub event name), ``REF_TYPE`` and ``REF_NAME`` (the ref that triggered the run).

- ``version``: the version in ``pyproject.toml``.
- ``target``: ``testpypi`` for a manual run and for a pre-release or development version, otherwise ``pypi``.
- ``published``: ``true`` if that version is already on the target index (a version can never be replaced).

For a tag run the tag must be ``v`` followed by the package version (PEP 440 normalization applies, so ``v0.1.0-rc1``
equals ``0.1.0rc1``); otherwise the script exits with an error.
"""

from __future__ import annotations

import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

import tomllib
from packaging.version import Version


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    version = Version(tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))["project"]["version"])
    event, ref_type, ref_name = os.environ["EVENT"], os.environ.get("REF_TYPE", ""), os.environ.get("REF_NAME", "")

    if event == "workflow_dispatch":
        target = "testpypi"
    else:
        if ref_type != "tag" or not ref_name.startswith("v"):
            sys.exit(f"the ref {ref_type} {ref_name!r} is not a tag v*")
        if Version(ref_name[1:]) != version:
            sys.exit(f"the tag {ref_name} is not the package version {version}")
        target = "testpypi" if version.is_prerelease or version.is_devrelease else "pypi"

    host = "https://pypi.org" if target == "pypi" else "https://test.pypi.org"
    try:
        urllib.request.urlopen(f"{host}/pypi/isotile/{version}/json", timeout=30)  # noqa: S310
        published = True
    except urllib.error.HTTPError as error:
        if error.code != 404:
            raise
        published = False
    print(f"version={version}")
    print(f"target={target}")
    print(f"published={str(published).lower()}")


if __name__ == "__main__":
    main()
