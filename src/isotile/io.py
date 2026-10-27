"""Serialization of a tiling to a JSON-compatible dict and to JSON files.

The dict has the keys

``version``
    ``"1.1"``.
``tiling_type``
    Type number.
``parameters``
    List of ``n_params`` floats.
``edge_curves``
    Object that maps a shape id (as a string) to the list of ``[x, y]`` control points of its curve, with the symmetry
    of the shape enforced. Absent if the tiling has no curves.
``edge_curve_modes``
    Object that maps a shape id to ``"spline"`` or ``"linear"``, for every id in ``edge_curves``.
``frame``
    ``[[a, b, c], [d, e, f]]``, the affine frame. Absent if it is the identity.

Writing is deterministic: the same tiling gives the same dict and the same file, and reading it back gives a tiling with
the same type, parameters, curves, modes and frame. Version ``"1.0"`` (no modes, no frame) is read, and keys that are
not listed are ignored.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from .tiling import IsohedralTiling

FORMAT_VERSION = "1.1"
_READABLE_MAJOR = "1"


def to_dict(tiling: IsohedralTiling) -> dict[str, Any]:
    """The tiling as a JSON-compatible dict, see the module description for the keys."""
    data: dict[str, Any] = {
        "version": FORMAT_VERSION,
        "tiling_type": int(tiling.type.number),
        "parameters": [float(x) for x in tiling.parameters],
    }
    curves, modes = tiling.edge_curves, tiling.edge_curve_modes
    if curves:
        data["edge_curves"] = {str(k): [[float(x), float(y)] for x, y in curves[k]] for k in sorted(curves)}
        data["edge_curve_modes"] = {str(k): modes[k] for k in sorted(curves)}
    frame = tiling.frame
    if not np.array_equal(frame, np.eye(3)[:2]):
        data["frame"] = [[float(x) for x in row] for row in frame]
    return data


def from_dict(data: Mapping[str, Any]) -> IsohedralTiling:
    """The tiling that [`to_dict`][isotile.io.to_dict] describes.

    Raises
    ------
    ValueError
        If the version is not 1.x, a required key is missing, a shape id is not an integer, or the content is not a
        valid tiling (see [`IsohedralTiling`][isotile.IsohedralTiling]).
    """
    from .tiling import IsohedralTiling

    version = str(data.get("version", "1.0"))
    if version.split(".")[0] != _READABLE_MAJOR:
        raise ValueError(f"unsupported tiling format version {version!r}")
    if "tiling_type" not in data:
        raise ValueError("the tiling data has no 'tiling_type'")

    def ids(key: str) -> dict[int, Any]:
        try:
            return {int(k): v for k, v in (data.get(key) or {}).items()}
        except ValueError:
            raise ValueError(f"the keys of {key!r} must be shape ids (integers)") from None

    curves, modes = ids("edge_curves"), ids("edge_curve_modes")
    return IsohedralTiling(
        data["tiling_type"],
        data.get("parameters"),
        edge_curves=curves or None,
        edge_curve_modes=modes or None,
        frame=data.get("frame"),
    )


def save(tiling: IsohedralTiling, path: str | Path) -> None:
    """Write the tiling as JSON (UTF-8, indented by 2, ending with a newline); an existing file is overwritten."""
    Path(path).write_text(json.dumps(to_dict(tiling), indent=2) + "\n", encoding="utf-8")


def load(path: str | Path) -> IsohedralTiling:
    """Read a tiling from a JSON file written by [`save`][isotile.io.save]."""
    return from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
