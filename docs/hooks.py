"""mkdocs hook: the catalog page and its figures are generated at build time from the type data.

``catalog.md`` and ``assets/catalog/ih<N>-light.svg`` / ``-dark.svg`` are added to the site as generated files; nothing
is written to the ``docs`` directory. The drawing helpers come from ``scripts/make_doc_figures.py``.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
GROUPS = {3: "Triangles", 4: "Quadrilaterals", 5: "Pentagons", 6: "Hexagons"}


def _figures() -> Any:
    """The module ``scripts/make_doc_figures.py``, imported by path."""
    name = "make_doc_figures"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / "make_doc_figures.py")
    assert spec is not None and spec.loader is not None  # noqa: S101
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


CURVES = {
    "J": [
        [(0, 0), (0.15, 0.14), (0.45, 0.04), (0.7, 0.17), (1, 0)],
        [(0, 0), (0.3, -0.15), (0.6, 0.0), (0.85, -0.12), (1, 0)],
        [(0, 0), (0.2, 0.2), (0.4, 0.2), (1, 0)],
    ],
    "U": [
        [(0, 0), (0.2, 0.17), (0.8, 0.17), (1, 0)],
        [(0, 0), (0.2, -0.15), (0.35, 0.1), (0.65, 0.1), (0.8, -0.15), (1, 0)],
    ],
    "S": [[(0, 0), (0.25, 0.16), (0.75, -0.16), (1, 0)], [(0, 0), (0.2, -0.16), (0.8, 0.16), (1, 0)]],
}
PARAMETER_STEP = 0.4
FALLBACK_STEP = 0.15


def example(number: int) -> tuple[Any, bool, bool]:
    """The tiling drawn for a type, and whether its parameters and its curves differ from the default straight one.

    Each parameter moves 40 percent of the way from its default to the nearer finite end of its safe range (by 0.15 if
    both ends are infinite). Every edge of shape J, U or S gets a fixed curve of its shape, chosen by the edge shape id so that edges
    with different ids differ (J: a wave; U: a bump or a double bump; S: an S or its mirror); edges of shape I stay straight. If the prototile is invalid, the parameter step and the curve heights are
    halved up to four times; then the default value or the straight edge is used.
    """
    from isotile import InvalidPrototileError, IsohedralTiling, get_type

    t = get_type(number)
    default = np.asarray(t.default_params, dtype=float)
    target = default.copy()
    for i, (lo, hi) in enumerate(t.safe_ranges):
        ends = [e for e in (lo, hi) if np.isfinite(e)]
        if ends:
            end = min(ends, key=lambda e: abs(e - default[i]))
            target[i] = default[i] + PARAMETER_STEP * (end - default[i])
        else:
            target[i] = default[i] + FALLBACK_STEP
    shape_of_id = {i: t.shape_types[i] for i in range(t.n_edge_shapes)}
    for k in range(5):
        scale = 0.5**k
        params = default + scale * (target - default)
        curves = {
            i: [(x, y * scale) for x, y in CURVES[s][i % len(CURVES[s])]] for i, s in shape_of_id.items() if s in CURVES
        }
        try:
            return (
                IsohedralTiling(number, params, edge_curves=curves or None),
                bool(np.any(params != default)),
                bool(curves),
            )
        except InvalidPrototileError:
            continue
    return IsohedralTiling(number), False, False


def type_figure(number: int):
    """Draw function ``theme -> figure`` of one type: a patch of tiles with the middle tile in amber."""
    figures = _figures()

    def draw_figure(th: Any):
        import matplotlib.pyplot as plt

        tiling = figures.unit(example(number)[0])
        center = tiling.centroid
        tiles = figures.window_tiles(tiling, center, (2.2, 2.2))
        fig, ax = plt.subplots(figsize=(1.9, 1.9))
        fig.subplots_adjust(0.01, 0.01, 0.99, 0.99)
        ax.set_xlim(center[0] - 1.7, center[0] + 1.7)
        ax.set_ylim(center[1] - 1.7, center[1] + 1.7)
        ax.set_aspect("equal")
        ax.axis("off")
        figures.draw(ax, tiles, figures.color_faces(tiles, th, figures.central(tiles, center)), th)
        return fig

    return draw_figure


def catalog_page() -> str:
    """The Markdown of the catalog page."""
    from isotile import get_type, tiling_types

    lines = [
        "# Catalog of the 81 types",
        "",
        "Each figure shows a patch of the type, with the tile colors of the type; the middle tile is amber. The page is",
        "generated from the type data when the documentation is built. So that types of the same default shape can be",
        "told apart, the figures do not use the default parameters and straight edges. Each parameter is moved 40 percent",
        "of the way from its default to the nearer finite end of its safe range (by 0.15 if both ends are infinite), and",
        "every edge of shape `J` (free), `U` (mirror symmetric) or `S` (rotation symmetric) has a fixed curve of that",
        "shape (a wave, a bump, an S), and edges with different edge shape ids get different curves. Edges of shape `I` stay straight. If the prototile is not valid, the",
        "parameter step and the curve heights are halved, up to four times; the figure then falls back to the default",
        "value or to a straight edge. The figures illustrate the symmetry of the type, not a recommended shape.",
        "`Aspects` is the number of tiles per lattice cell, `Parameters` the number of free parameters, `Edges` the",
        "shape of each edge of the prototile (`I` straight, `J` free curve, `U` mirror symmetric, `S` rotation",
        "symmetric) and `Edge ids` the number of distinct edge shape ids, which is the number of curves a tiling can carry.",
        "`Reflections` says whether some aspects are mirror images. `find_types` selects types by these properties.",
        "",
    ]
    by_vertices: dict[int, list[int]] = {}
    for number in tiling_types():
        by_vertices.setdefault(get_type(number).n_vertices, []).append(number)
    for n_vertices in sorted(by_vertices):
        title = GROUPS.get(n_vertices, f"{n_vertices}-gons")
        lines += [
            f"## {title}",
            "",
            "| Tiling | Type | Aspects | Parameters | Edges | Edge ids | Reflections |",
            "|---|---|---:|---:|---|---:|---|",
        ]
        for number in by_vertices[n_vertices]:
            t = get_type(number)
            alt = f"A patch of tiling IH{number} with the middle tile in amber."
            image = (
                f"![{alt}](assets/catalog/ih{number}-light.svg#only-light){{ width=120 }}"
                f"![{alt}](assets/catalog/ih{number}-dark.svg#only-dark){{ width=120 }}"
            )
            lines.append(
                f"| {image} | **IH{number}** | {t.n_aspects} | {t.n_params} | `{' '.join(t.edge_shapes)}` |"
                f" {t.n_edge_shapes} | {'yes' if t.has_reflections else 'no'} |"
            )
        lines.append("")
    return "\n".join(lines)


def on_files(files: Any, config: Any) -> Any:
    from mkdocs.structure.files import File

    from isotile import tiling_types

    figures = _figures()
    files.append(File.generated(config, "catalog.md", content=catalog_page()))
    for number in tiling_types():
        draw_figure = type_figure(number)
        for th in figures.THEMES:
            content = figures.render(draw_figure, th)
            files.append(File.generated(config, f"assets/catalog/ih{number}-{th.name}.svg", content=content))
    return files


__all__ = ["catalog_page", "example", "on_files", "type_figure"]
