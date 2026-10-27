"""Generate the figures of the documentation in ``docs/assets/figures/``.

Every figure is drawn twice, as ``<name>-light.svg`` and ``<name>-dark.svg`` on a transparent background, and the pages
show the variant of the active color scheme (``#only-light`` and ``#only-dark`` image URLs, a Material convention). The
colors are those of the logo (see ``docs/branding.md``). The tiles come from isotile; where a figure illustrates a code
block of a page, the script runs the blocks of that page up to the block and draws the objects that the block created,
so the figure always shows what the code on the page produces.

Run from the repository root (matplotlib comes from the docs group)::

    uv run --group docs python scripts/make_doc_figures.py

``--out`` changes the output directory. ``--check`` regenerates the files in memory and exits with status 1 if they
differ from the files on disk. The output is deterministic: text is converted to paths, the SVG hash salt is fixed and
the date and creator metadata are removed.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.patches import FancyArrowPatch, Polygon, Rectangle

from isotile import IsohedralTiling, TileSet, get_type

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs"
FENCE = re.compile(r"^```(\w*)[^\n]*\n(.*?)^```", re.S | re.M)
# Width of the content column of the pages is about 780 px; figures stay below that (1 in = 96 px).
WIDTH = 7.2


@dataclass(frozen=True)
class Theme:
    name: str
    surface: str  # surface of the docs, only used to mix tints
    text: str
    blue: str
    amber: str
    green: str
    plum: str
    on_color: str  # text on a filled tile

    def mix(self, color: str, weight: float) -> str:
        """``color`` at ``weight`` over the surface, as an opaque hex color."""
        c, s = np.array(matplotlib.colors.to_rgb(color)), np.array(matplotlib.colors.to_rgb(self.surface))
        return matplotlib.colors.to_hex(weight * c + (1 - weight) * s)

    @property
    def muted(self) -> str:
        return self.mix(self.blue, 0.16)

    @property
    def line(self) -> str:
        return self.mix(self.text, 0.55)

    @property
    def faint(self) -> str:
        return self.mix(self.text, 0.35)

    @property
    def aspects(self) -> list[str]:
        return [self.blue, self.amber, self.green, self.plum]

    @property
    def tints(self) -> list[str]:
        return [self.mix(self.blue, w) for w in (0.9, 0.5, 0.28, 0.7)]


LIGHT = Theme("light", "#ffffff", "#1b1f24", "#2c6e9e", "#c27400", "#2a7d4f", "#9b4a8f", "#ffffff")
DARK = Theme("dark", "#1e2029", "#e8edf2", "#85b7e0", "#facc38", "#86d9a0", "#e0a3d6", "#1b1f24")
THEMES = (LIGHT, DARK)

# ---- Code of the pages --------------------------------------------------------------------------------------


def doc_namespace(page: str, until: str) -> dict[str, Any]:
    """Run the Python blocks of ``docs/<page>`` in order up to the first block that contains ``until``."""
    namespace: dict[str, Any] = {"__name__": "__docs__"}
    blocks = [
        body for language, body in FENCE.findall((DOCS / page).read_text(encoding="utf-8")) if language == "python"
    ]
    for body in blocks:
        with contextlib.redirect_stdout(io.StringIO()):
            exec(compile(body, page, "exec"), namespace)  # noqa: S102
        if until in body:
            return namespace
    raise ValueError(f"no block of {page} contains {until!r}")


# ---- Drawing helpers, built on tile polygons and adjacency only -----------------------------------------------


def unit(tiling: IsohedralTiling) -> IsohedralTiling:
    """The tiling scaled so that a tile has area 1."""
    return tiling.scaled(1 / np.sqrt(tiling.area))


def window_tiles(tiling: IsohedralTiling, center: Any, half: tuple[float, float]) -> TileSet:
    cx, cy = center
    return tiling.cover((cx - half[0], cy - half[1], cx + half[0], cy + half[1]))


def new_axes(n: int, aspect: float = 1.0, wspace: float = 0.06, width: float = WIDTH):
    """A row of ``n`` panels; ``aspect`` is height over width of one panel."""
    panel = width / (n + (n - 1) * wspace)
    fig, axes = plt.subplots(
        1,
        n,
        figsize=(width, panel * aspect / 0.78),
        gridspec_kw={"wspace": wspace, "left": 0.005, "right": 0.995, "bottom": 0.1, "top": 0.88},
    )
    return fig, np.atleast_1d(axes)


def frame(ax, center, half, th: Theme, title: str | None = None) -> None:
    """Fix the window of a panel, with a thin frame, and title it."""
    cx, cy = center
    ax.set_xlim(cx - half[0], cx + half[0])
    ax.set_ylim(cy - half[1], cy + half[1])
    ax.set_aspect("equal")
    ax.axis("off")
    ax.add_patch(
        Rectangle((cx - half[0], cy - half[1]), 2 * half[0], 2 * half[1], fill=False, ec=th.faint, lw=0.8, zorder=9)
    )
    if title:
        ax.set_title(title, fontsize=9, color=th.text, pad=5)


def draw(ax, tiles: TileSet, faces, th: Theme, lw: float = 0.9, zorder: float = 1, edge: str | None = None) -> None:
    """Draw the tiles that reach into the window of the axes (set the window first, see ``frame``)."""
    polygons = tiles.vertices()
    (x0, x1), (y0, y1) = ax.get_xlim(), ax.get_ylim()
    keep = np.flatnonzero(
        (polygons[..., 0].max(axis=1) >= x0)
        & (polygons[..., 0].min(axis=1) <= x1)
        & (polygons[..., 1].max(axis=1) >= y0)
        & (polygons[..., 1].min(axis=1) <= y1)
    )
    ax.add_collection(
        PolyCollection(
            polygons[keep],
            facecolors=[faces[k] for k in keep],
            edgecolors=edge or th.line,
            linewidths=lw,
            joinstyle="round",
            zorder=zorder,
        )
    )


def label(
    ax, xy, text, th: Theme, color: str | None = None, size: float = 9, weight: str = "bold", zorder: float = 6
) -> None:
    ax.text(
        *xy,
        text,
        ha="center",
        va="center",
        fontsize=size,
        color=color or th.text,
        fontweight=weight,
        zorder=zorder,
        linespacing=1.1,
    )


def arrow(ax, p, q, th: Theme, color: str, lw: float = 2.0) -> None:
    ax.add_patch(
        FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=12, lw=lw, color=color, shrinkA=0, shrinkB=0, zorder=8)
    )


def dot(ax, p, th: Theme, color: str, size: float = 4.5) -> None:
    ax.plot(*p, "o", ms=size, color=color, mec=th.surface, mew=0.8, zorder=8)


def color_faces(tiles: TileSet, th: Theme, highlight: int | None = None):
    faces = [th.tints[c % len(th.tints)] for c in tiles.colors()]
    if highlight is not None:
        faces[highlight] = th.amber
    return faces


def central(tiles: TileSet, point) -> int:
    return int(np.argmin(np.linalg.norm(tiles.centroids - np.asarray(point), axis=1)))


def tile_origin(tiles: TileSet, i: int) -> np.ndarray:
    """Image of the native origin under the transform of tile ``i``."""
    return tiles.transforms[i, :, 2]


def caption_box(fig, th: Theme, items: list[tuple[str, str]], y: float = -0.02) -> None:
    """A legend row below the panels: colored squares with text."""
    n = len(items)
    x = 0.5 - 0.5 * sum(0.045 + 0.0105 * len(t) for _, t in items)
    for color, text in items:
        fig.patches.append(
            Rectangle((x, y), 0.022, 0.06, transform=fig.transFigure, fc=color, ec=th.line, lw=0.7, clip_on=False)
        )
        fig.text(x + 0.03, y + 0.03, text, fontsize=8.5, color=th.text, va="center", ha="left")
        x += 0.045 + 0.0105 * len(text)
    del n


# ---- Figures --------------------------------------------------------------------------------------------------


def fig_hero(th: Theme):
    """Four of the types side by side, the middle tile of each in amber."""
    fig, axes = new_axes(4, aspect=1.0, wspace=0.05)
    for ax, number in zip(axes, (28, 21, 29, 84), strict=True):
        tiling = unit(IsohedralTiling(number))
        tiles = window_tiles(tiling, tiling.centroid, (2.0, 2.0))
        frame(ax, tiling.centroid, (1.6, 1.6), th, f"IH{number}")
        draw(ax, tiles, color_faces(tiles, th, central(tiles, tiling.centroid)), th)
    return fig


def cell_extent(tiling: IsohedralTiling):
    """Center and half-size of a window that holds the cell ``(0, 0)``: its tiles and its parallelogram."""
    home = tiling.tiles([[0, 0]] * tiling.type.n_aspects, range(tiling.type.n_aspects))
    o, u, v = tile_origin(home, 0), tiling.translation_u, tiling.translation_v
    points = np.vstack([home.vertices().reshape(-1, 2), o, o + u, o + v, o + u + v])
    lo, hi = points.min(axis=0), points.max(axis=0)
    return home, (lo + hi) / 2, (hi - lo) / 2


def cell_outline(ax, tiling: IsohedralTiling, origin, th: Theme) -> None:
    u, v = tiling.translation_u, tiling.translation_v
    ax.add_patch(
        Polygon(
            [origin, origin + u, origin + u + v, origin + v], fill=False, ec=th.text, lw=1.1, ls=(0, (4, 3)), zorder=7
        )
    )


def note(ax, xy, text, th: Theme, color: str, size: float = 8.5) -> None:
    ax.text(
        *xy,
        text,
        ha="center",
        va="center",
        fontsize=size,
        color=color,
        zorder=10,
        bbox={"boxstyle": "round,pad=0.18", "fc": th.surface, "ec": "none", "alpha": 0.9},
    )


def lattice_dots(ax, tiling: IsohedralTiling, th: Theme, size: float = 3.2) -> None:
    """The lattice points ``u * translation_u + v * translation_v`` (translates of the origin) inside the window."""
    o = tile_origin(tiling.tiles([[0, 0]], 0), 0)
    (x0, x1), (y0, y1) = ax.get_xlim(), ax.get_ylim()
    k = np.arange(-12, 13)
    pts = o + k[:, None, None] * tiling.translation_u + k[None, :, None] * tiling.translation_v
    pts = pts.reshape(-1, 2)
    pts = pts[(pts[:, 0] >= x0) & (pts[:, 0] <= x1) & (pts[:, 1] >= y0) & (pts[:, 1] <= y1)]
    ax.plot(pts[:, 0], pts[:, 1], "o", ms=size, color=th.text, mec=th.surface, mew=0.6, zorder=8, ls="none")


def cell_tiles(tiling: IsohedralTiling, cells) -> TileSet:
    """The tiles of the given cells, all aspects."""
    n = tiling.type.n_aspects
    cells = np.asarray(cells).reshape(-1, 2)
    return tiling.tiles(np.repeat(cells, n, axis=0), np.tile(np.arange(n), len(cells)))


def fig_lattice(th: Theme):
    """IH28: the lattice points as dots, one cell outlined, its tiles in color."""
    tiling = unit(IsohedralTiling(28))
    home, center, half = cell_extent(tiling)
    half = 1.18 * half + 0.35
    half = (max(half) * 1.3, max(half))
    tiles = window_tiles(tiling, center, half)
    fig, (ax,) = new_axes(1, aspect=half[1] / half[0], width=WIDTH * 0.75)
    frame(ax, center, half, th)
    draw(ax, tiles, [th.muted] * len(tiles), th)
    draw(ax, home, [th.aspects[a] for a in range(len(home))], th, lw=1.1, zorder=3)
    origin = tile_origin(home, 0)
    u, v = tiling.translation_u, tiling.translation_v
    cell_outline(ax, tiling, origin, th)
    lattice_dots(ax, tiling, th)
    arrow(ax, origin, origin + u, th, th.blue)
    arrow(ax, origin, origin + v, th, th.green)
    dot(ax, origin, th, th.text, 6.5)
    note(ax, origin + 0.5 * u + 0.17 * np.array([-v[1], v[0]]) / np.linalg.norm(v) * 1.2, "translation_u", th, th.blue)
    note(ax, origin + 0.5 * v + np.array([-0.75, 0.2]), "translation_v", th, th.green)
    note(ax, origin + np.array([0.55, -0.3]), "origin", th, th.text, 8)
    return fig


def fig_aspects(th: Theme):
    """The tiles of the cell (0, 0), one color per aspect, in a type without and a type with reflections."""
    numbers = (28, 29)
    extents = [cell_extent(unit(IsohedralTiling(n))) for n in numbers]
    half = 1.12 * max(float(e[2].max()) for e in extents) + 0.3
    fig, axes = new_axes(2, aspect=1.0, wspace=0.08)
    for ax, number, (_, center, _) in zip(axes, numbers, extents, strict=True):
        t = get_type(number)
        tiling = unit(IsohedralTiling(number))
        home = cell_tiles(tiling, [[0, 0]])
        around = window_tiles(tiling, center, (half, half))
        reflect = "with reflections" if t.has_reflections else "no reflections"
        frame(ax, center, (half, half), th, f"IH{number}: {t.n_aspects} aspects, {reflect}")
        draw(ax, around, [th.muted] * len(around), th, lw=0.6, edge=th.faint)
        draw(ax, home, [th.aspects[a] for a in range(t.n_aspects)], th, lw=1.1, zorder=3)
        cell_outline(ax, tiling, tile_origin(home, 0), th)
        lattice_dots(ax, tiling, th)
        o = tile_origin(home, 0)
        for a, p in enumerate(home.centroids):
            label(ax, p + 0.3 * (p - o), str(a), th, th.on_color, 12)
        dot(ax, o, th, th.text, 6.5)
        note(ax, o + (np.array([0.6, -0.32]) if number == 28 else np.array([-0.55, 0.0])), "origin", th, th.text, 8)
    return fig


def group_outline(ax, tiles: TileSet, th: Theme) -> None:
    """Thick lines on the tile edges that separate two cells or border the set (edges are matched by their ends)."""
    polygons = np.round(tiles.vertices(), 6)
    owners: dict[tuple, list[int]] = {}
    for i, poly in enumerate(polygons):
        for p, q in zip(poly, np.roll(poly, -1, axis=0), strict=True):
            owners.setdefault(tuple(sorted((tuple(p), tuple(q)))), []).append(i)
    lines = [key for key, ids in owners.items() if len(ids) == 1 or len({tuple(tiles.cell[i]) for i in ids}) > 1]
    ax.add_collection(LineCollection(lines, colors=th.text, linewidths=2.2, capstyle="round", zorder=5))


def fig_cells(th: Theme):
    """IH29: the tiles of four neighboring cells, each cell a translate of the cell (0, 0)."""
    tiling = unit(IsohedralTiling(29))
    cells = [(0, 0), (1, 0), (0, 1), (1, 1)]
    group = cell_tiles(tiling, cells)
    u, v = tiling.translation_u, tiling.translation_v
    o = tile_origin(group, 0)
    corners = np.array([o, o + 2 * u, o + 2 * v, o + 2 * u + 2 * v])
    pts = np.vstack([group.vertices().reshape(-1, 2), corners])
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    center, half = (lo + hi) / 2, float(max(hi - lo) / 2) * 1.06
    fig, (ax,) = new_axes(1, aspect=1.0, width=WIDTH * 0.8)
    frame(ax, center, (half, half), th)
    around = window_tiles(tiling, center, (half, half))
    draw(ax, around, [th.muted] * len(around), th, lw=0.6, edge=th.faint)
    draw(ax, group, [th.aspects[a] for a in group.aspect], th, lw=0.8, zorder=3)
    group_outline(ax, group, th)
    cell_outline(ax, tiling, o, th)
    for cell in cells:
        mask = (group.cell == cell).all(axis=1)
        note(ax, group.centroids[mask].mean(axis=0) + np.array([0.0, 0.75]), f"cell {cell}", th, th.text, 8.5)
    lattice_dots(ax, tiling, th)
    return fig


def fig_families(th: Theme, ns: dict[str, Any] | None = None):
    """The four named constructors, with the type numbers."""
    ns = ns or doc_namespace("quickstart.md", "quadrilateral = ")
    rows = [
        ("square", ns["square"], "IH76 square"),
        ("hexagon", ns["hexagon"], "IH20 hexagon"),
        ("triangle", ns["triangle"], "IH84 triangle"),
        ("quadrilateral", ns["quadrilateral"], "IH46 quadrilateral"),
    ]
    fig, axes = new_axes(4, aspect=1.0, wspace=0.05)
    for ax, (_, tiling, title) in zip(axes, rows, strict=True):
        tiling = unit(tiling)
        c = tiling.centroid
        tiles = window_tiles(tiling, c, (1.8, 1.8))
        frame(ax, c, (1.8, 1.8), th, title)
        draw(ax, tiles, color_faces(tiles, th, central(tiles, c)), th)
    return fig


def fig_parameters(th: Theme):
    """Type IH28 at four parameter sets."""
    sets = [None, [0.8, 0.5], [0.2, 0.15], [0.45, 0.9]]
    fig, axes = new_axes(4, aspect=1.0, wspace=0.05)
    for ax, params in zip(axes, sets, strict=True):
        tiling = unit(IsohedralTiling(28, params))
        c = tiling.centroid
        tiles = window_tiles(tiling, c, (1.8, 1.8))
        p = tiling.parameters
        frame(ax, c, (1.8, 1.8), th, f"[{p[0]:.2f}, {p[1]:.2f}]")
        draw(ax, tiles, color_faces(tiles, th, central(tiles, c)), th)
    return fig


def neighbor_panel(ax, tiling: IsohedralTiling, th: Theme, title: str, half: float = 1.9):
    tiling = unit(tiling)
    c = tiling.centroid
    tiles = window_tiles(tiling, c, (half + 1.5, half + 1.5))
    i = central(tiles, c)
    edge, only = tiles.neighbors(i, "edge"), tiles.neighbors(i, "vertex-only")
    faces = [th.muted] * len(tiles)
    for j in edge:
        faces[j] = th.blue
    for j in only:
        faces[j] = th.green
    faces[i] = th.amber
    frame(ax, tiles.centroids[i], (half, half), th, f"{title}: {len(edge)} edge, {len(only)} vertex-only")
    draw(ax, tiles, faces, th)


def fig_neighbors(th: Theme):
    fig, axes = new_axes(2, aspect=1.0, wspace=0.08)
    neighbor_panel(axes[0], IsohedralTiling.square(), th, "Square (IH76)")
    neighbor_panel(axes[1], IsohedralTiling(28), th, "IH28", half=2.1)
    caption_box(fig, th, [(th.amber, "tile"), (th.blue, "edge neighbor"), (th.green, "vertex-only neighbor")], y=0.03)
    return fig


def fig_offsets(th: Theme):
    """The neighbor relations of one tile of IH28: cell offset and aspect of each neighbor."""
    tiling = unit(IsohedralTiling(28))
    t = tiling.type
    aspect = 0
    rows = t.adjacency[t.adjacency[:, 0] == aspect]
    cells = [(int(du), int(dv)) for du, dv in rows[:, 1:3]]
    tiles = window_tiles(tiling, tiling.centroid, (3.6, 3.2))
    home = int(np.flatnonzero((tiles.cell == (0, 0)).all(axis=1) & (tiles.aspect == aspect))[0])
    index = {(tuple(c), int(a)): j for j, (c, a) in enumerate(zip(tiles.cell, tiles.aspect, strict=True))}
    faces = [th.muted] * len(tiles)
    faces[home] = th.amber
    for du, dv, a, kind in rows[:, 1:]:
        faces[index[((int(du), int(dv)), int(a))]] = th.blue if kind == 0 else th.green
    fig, (ax,) = new_axes(1, aspect=1.0, width=WIDTH * 0.8)
    frame(ax, tiles.centroids[home], (2.3, 2.3), th)
    draw(ax, tiles, faces, th)
    for (du, dv), a in zip(cells, rows[:, 3], strict=True):
        j = index[((du, dv), int(a))]
        label(ax, tiles.centroids[j], f"({du}, {dv})\naspect {int(a)}", th, th.on_color, 7.5)
    label(ax, tiles.centroids[home], f"cell (0, 0)\naspect {aspect}", th, th.on_color, 7.5)
    caption_box(fig, th, [(th.amber, "tile"), (th.blue, "kind 0: edge"), (th.green, "kind 1: vertex only")], y=0.03)
    return fig


def fig_matrix(th: Theme):
    """The five squares of the worked example of the Adjacency page and their adjacency matrices."""
    ns = doc_namespace("adjacency.md", "only = tiles.adjacency")
    tiles, edge, only = ns["tiles"], ns["edge"].toarray(), ns["only"].toarray()
    n = len(tiles)
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH * 0.85, 3.0), gridspec_kw={"wspace": 0.12, "width_ratios": [1.1, 1]})
    ax = axes[0]
    ax.set_xlim(-0.3, 4.3)
    ax.set_ylim(-0.3, 4.3)
    draw(ax, tiles, [th.muted] * n, th, lw=1.0)
    for i, p in enumerate(tiles.centroids):
        label(ax, p, str(i), th, th.text, 12)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("tile index", fontsize=9, color=th.text, pad=5)
    ax.set_anchor("N")
    ax = axes[1]
    for i in range(n):
        for j in range(n):
            fc = th.blue if edge[i, j] else th.green if only[i, j] else th.surface
            ax.add_patch(
                Rectangle(
                    (j, n - 1 - i), 1, 1, fc=fc, ec=th.faint, lw=0.8, alpha=1.0 if (edge[i, j] or only[i, j]) else 0.0
                )
            )
            ax.add_patch(Rectangle((j, n - 1 - i), 1, 1, fill=False, ec=th.faint, lw=0.8))
    for i in range(n):
        ax.text(-0.3, n - 1 - i + 0.5, str(i), ha="right", va="center", fontsize=9, color=th.text)
        ax.text(i + 0.5, n + 0.25, str(i), ha="center", va="bottom", fontsize=9, color=th.text)
    ax.text(-0.9, n / 2, "row i", rotation=90, ha="center", va="center", fontsize=8.5, color=th.text)
    ax.text(n / 2, n + 0.8, "column j", ha="center", va="bottom", fontsize=8.5, color=th.text)
    ax.set_xlim(-1.2, n + 0.2)
    ax.set_ylim(-0.2, n + 1.4)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_anchor("N")
    ax.set_title("adjacency matrix", fontsize=9, color=th.text, pad=5)
    caption_box(fig, th, [(th.blue, "edge"), (th.green, "vertex-only")], y=-0.03)
    return fig


# Quickstart


def region_rect(ax, bounds, th: Theme) -> None:
    x0, y0, x1, y1 = bounds
    ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec=th.text, lw=1.5, ls=(0, (5, 3)), zorder=7))


def tile_view(ax, tiles: TileSet, bounds, th: Theme, faces, title: str | None = None) -> None:
    x0, y0, x1, y1 = bounds
    pad = 0.04 * (x1 - x0)
    ax.set_xlim(x0 - pad, x1 + pad)
    ax.set_ylim(y0 - pad, y1 + pad)
    draw(ax, tiles, faces, th, lw=0.7)
    region_rect(ax, bounds, th)
    ax.set_aspect("equal")
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=9, color=th.text, pad=5)


def fig_q_create(th: Theme):
    ns = doc_namespace("quickstart.md", "custom = ")
    fig, axes = new_axes(2, aspect=0.8, wspace=0.08)
    for ax, key, title in zip(
        axes, ("tiling", "custom"), ("IsohedralTiling(7)", "IsohedralTiling(7, [0.7, 0.3])"), strict=True
    ):
        tiling = unit(ns[key])
        c = tiling.centroid
        tiles = window_tiles(tiling, c, (2.6, 2.1))
        frame(ax, c, (2.6, 2.1), th, title)
        draw(ax, tiles, color_faces(tiles, th, central(tiles, c)), th)
        ax.add_patch(Polygon(tiling.prototile(), fill=False, ec=th.text, lw=1.6, zorder=5))
    return fig


def fig_q_fill(th: Theme):
    ns = doc_namespace("quickstart.md", "tiles = tiling.fill_region")
    bounds = (0, 0, 10, 6)
    tiles = ns["tiles"]
    fig, (ax,) = new_axes(1, aspect=0.6, width=WIDTH * 0.9)
    tile_view(ax, tiles, bounds, th, color_faces(tiles, th))
    return fig


def fig_q_select(th: Theme):
    ns = doc_namespace("quickstart.md", "inside = ")
    tiles, inside = ns["tiles"], ns["inside"]
    selected = {(*tuple(c), int(a)) for c, a in zip(inside.cell, inside.aspect, strict=True)}
    faces = [
        th.amber if (*tuple(c), int(a)) in selected else th.muted for c, a in zip(tiles.cell, tiles.aspect, strict=True)
    ]
    fig, (ax,) = new_axes(1, aspect=0.6, width=WIDTH * 0.9)
    tile_view(ax, tiles, (0, 0, 10, 6), th, faces)
    return fig


def fig_q_adjacency(th: Theme):
    ns = doc_namespace("quickstart.md", "grid = ")
    tiles, grid = ns["tiles"], ns["grid"]
    fig, axes = new_axes(2, aspect=0.9, wspace=0.08)
    ax = axes[0]
    nb = tiles.neighbors(0, "edge")
    faces = [th.muted] * len(tiles)
    for j in nb:
        faces[j] = th.blue
    faces[0] = th.amber
    corners = tiles.vertices().reshape(-1, 2)
    c = np.clip(tiles.centroids[0], corners.min(axis=0) + 1.7, corners.max(axis=0) - 1.7)
    frame(ax, c, (1.7, 1.7), th, "tiles.neighbors(0, 'edge')")
    draw(ax, tiles, faces, th)
    ax = axes[1]
    i = int(np.argmin(np.linalg.norm(grid.centroids - grid.centroids.mean(axis=0), axis=1)))
    faces = [th.muted] * len(grid)
    for j in grid.neighbors(i, "vertex-only"):
        faces[j] = th.green
    for j in grid.neighbors(i, "edge"):
        faces[j] = th.blue
    faces[i] = th.amber
    frame(ax, grid.centroids.mean(axis=0), (2.4, 2.4), th, "square grid, an inner tile")
    draw(ax, grid, faces, th, lw=1.0)
    caption_box(fig, th, [(th.amber, "tile"), (th.blue, "edge"), (th.green, "vertex-only")], y=0.03)
    return fig


def fig_q_frames(th: Theme):
    ns = doc_namespace("quickstart.md", "rotated_tiles = ")
    tiles = ns["rotated_tiles"]
    fig, (ax,) = new_axes(1, aspect=0.6, width=WIDTH * 0.9)
    tile_view(ax, tiles, (0, 0, 10, 6), th, color_faces(tiles, th))
    return fig


FIGURES = {
    "hero": fig_hero,
    "concepts-lattice": fig_lattice,
    "concepts-aspects": fig_aspects,
    "concepts-cells": fig_cells,
    "concepts-families": fig_families,
    "concepts-parameters": fig_parameters,
    "adjacency-neighbors": fig_neighbors,
    "adjacency-offsets": fig_offsets,
    "adjacency-matrix": fig_matrix,
    "quickstart-create": fig_q_create,
    "quickstart-fill": fig_q_fill,
    "quickstart-select": fig_q_select,
    "quickstart-adjacency": fig_q_adjacency,
    "quickstart-frames": fig_q_frames,
}

# ---- Output ---------------------------------------------------------------------------------------------------


def shorten(path_data: str) -> str:
    """Path data with coordinates rounded to 0.1 pt (the glyph outlines are integers and stay as they are)."""
    return re.sub(r"-?\d+\.\d+", lambda m: f"{float(m.group()):.1f}".replace("-0.0", "0"), path_data)


def compact(text: str) -> str:
    """Smaller SVG: compact path data; clip path and stroke of a tile collection move from the paths to their group."""
    text = re.sub(
        r'\bd="([^"]*)"', lambda m: 'd="' + re.sub(r"\s*([MLCQz])\s*", r"\1", shorten(m.group(1)).strip()) + '"', text
    )

    def collection(m: re.Match[str]) -> str:
        head, body = m.group(1), m.group(2)
        clips = set(re.findall(r' clip-path="([^"]*)"', body))
        strokes = set(re.findall(r'; (stroke: [^"]*)"', body))
        if len(clips) != 1 or len(strokes) != 1:
            return m.group(0)
        (clip,), (stroke,) = clips, strokes
        body = body.replace(f' clip-path="{clip}"', "").replace(f"; {stroke}", "")
        body = re.sub(r'style="fill: (#[0-9a-f]+)"', r'fill="\1"', body)
        return f'{head[:-1]} clip-path="{clip}" style="{stroke}">{body}</g>'

    return re.sub(r'(<g id="PolyCollection_\d+">)(.*?)</g>', collection, text, flags=re.S)


def render(draw_figure, th: Theme) -> str:
    """The SVG text of one figure; fixed hash salt, no date and creator, text as paths."""
    with matplotlib.rc_context({
        "svg.hashsalt": "isotile-doc-figures",
        "svg.fonttype": "path",
        "font.family": "DejaVu Sans",
        "figure.facecolor": "none",
        "axes.facecolor": "none",
    }):
        fig = draw_figure(th)
        buffer = io.BytesIO()
        fig.savefig(
            buffer,
            format="svg",
            transparent=True,
            bbox_inches="tight",
            pad_inches=0.06,
            metadata={"Date": None, "Creator": None, "Format": None, "Type": None},
        )
        plt.close(fig)
    text = compact(buffer.getvalue().decode("utf-8"))
    return "\n".join(line.rstrip() for line in text.splitlines()) + "\n"


def svg_files() -> dict[str, str]:
    return {f"{name}-{th.name}.svg": render(draw_figure, th) for name, draw_figure in FIGURES.items() for th in THEMES}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", type=Path, default=ROOT / "docs" / "assets" / "figures")
    ap.add_argument("--check", action="store_true", help="compare the SVG files with the files on disk")
    args = ap.parse_args()
    files = svg_files()
    out: Path = args.out
    if args.check:
        stale = [
            n for n, text in files.items() if not (out / n).exists() or (out / n).read_text(encoding="utf-8") != text
        ]
        extra = sorted(p.name for p in out.glob("*.svg") if p.name not in files) if out.exists() else []
        if stale or extra:
            print("stale figure files:", ", ".join(stale + extra))
            sys.exit(1)
        print("figure files are up to date")
        return
    out.mkdir(parents=True, exist_ok=True)
    for name, text in files.items():
        (out / name).write_text(text, encoding="utf-8")
    print(f"wrote {len(files)} files, {sum(len(t.encode()) for t in files.values()) / 1024:.0f} KiB")


if __name__ == "__main__":
    main()
