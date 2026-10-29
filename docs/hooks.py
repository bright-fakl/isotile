"""mkdocs hook: the catalog page and its figures are generated at build time from the type data.

``catalog.md`` and ``assets/catalog/ih<N>-light.svg`` / ``-dark.svg`` are added to the site as generated files; nothing
is written to the ``docs`` directory. The drawing helpers come from ``scripts/make_doc_figures.py``.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

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


def type_figure(number: int):
    """Draw function ``theme -> figure`` of one type: a patch of tiles with the middle tile in amber."""
    figures = _figures()

    def draw_figure(th: Any):
        import matplotlib.pyplot as plt

        tiling = figures.unit(figures.IsohedralTiling(number))
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
        "Each figure shows a patch of the type at its default parameters, with the tile colors of the type; the middle tile",
        "is amber. The page is generated from the type data when the documentation is built.",
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


__all__ = ["catalog_page", "on_files", "type_figure"]
