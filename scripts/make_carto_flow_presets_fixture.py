"""Regenerate tests/data/carto_flow_presets.json from the preset table of carto-flow.

The fixture holds, for each of the 15 tiling presets of carto-flow 2.0.0 (``IsohedralTiling._PRESETS``), the type,
the parameters, the edge curves and the edge curve modes. ``tests/test_presets.py`` compares the presets of isotile
with it.

Run with an environment that has carto-flow installed:

    uv run --with "carto-flow==2.0.0" python scripts/make_carto_flow_presets_fixture.py
"""

from __future__ import annotations

import json
from pathlib import Path

from carto_flow.symbol_cartogram.tiling import IsohedralTiling

OUT = Path(__file__).resolve().parents[1] / "tests" / "data" / "carto_flow_presets.json"


def main() -> None:
    data = {}
    for name, p in IsohedralTiling._PRESETS.items():
        curves = p.get("edge_curves") or {}
        modes = p.get("edge_curve_modes") or {}
        data[name] = {
            "type": int(p["tiling_type"]),
            "parameters": [float(x) for x in p["parameters"]] if p.get("parameters") is not None else None,
            "edge_curves": {str(k): [[float(x), float(y)] for x, y in v] for k, v in sorted(curves.items())},
            "edge_curve_modes": {str(k): str(v) for k, v in sorted(modes.items())},
        }
    OUT.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
