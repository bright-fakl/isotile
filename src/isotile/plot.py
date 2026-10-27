"""Minimal matplotlib helpers to draw prototiles and tiles (needs the ``plot`` extra: ``pip install isotile[plot]``).

matplotlib is imported when a function is called, never when ``isotile`` is imported.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from .tiling import IsohedralTiling, TileSet

_PALETTE = ("#9ecae1", "#fdd49e", "#a1d99b", "#d4b9da", "#fcbba1", "#c7e9c0", "#bdbdbd", "#dadaeb")


def _pyplot() -> Any:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("isotile.plot needs matplotlib: pip install isotile[plot]") from None
    return plt


def _axes(ax: Any) -> Any:
    if ax is not None:
        return ax
    ax = _pyplot().figure().add_subplot()
    ax.set_aspect("equal")
    return ax


def plot_prototile(tiling: IsohedralTiling, ax: Any = None, show_labels: bool = True) -> Any:
    """Draw the prototile with its curves.

    The outline is filled; with curves, the straight prototile is drawn dashed. With ``show_labels`` each edge is
    labeled with its index and shape (for example ``"2 J"``), and an arrow on the edge shows the direction of the
    canonical edge (from ``(0, 0)`` to ``(1, 0)``), which is against the order of the outline for a reversed edge.

    Parameters
    ----------
    tiling : IsohedralTiling
    ax : matplotlib.axes.Axes, optional
        Axes to draw into; a new figure with equal axis scaling when omitted.
    show_labels : bool, default True

    Returns
    -------
    matplotlib.axes.Axes

    Raises
    ------
    ImportError
        If matplotlib is not installed.
    """
    _pyplot()
    from matplotlib.patches import Polygon

    ax = _axes(ax)
    outline = tiling.prototile()
    ax.add_patch(Polygon(outline, closed=True, fc=_PALETTE[0], ec="#2c6e9e", lw=1.5, joinstyle="round", zorder=2))
    if tiling.edge_curves:
        ax.add_patch(Polygon(tiling.vertices, closed=True, fill=False, ec="#555555", lw=0.8, ls="--", zorder=3))
    if show_labels:
        centroid = tiling.centroid
        for info, line in zip(tiling.edges, tiling.edge_outlines(), strict=True):
            a, b = (line[-1], line[0]) if info.reversed else (line[0], line[-1])
            mid = (a + b) / 2
            outward = mid - centroid
            outward /= np.linalg.norm(outward) or 1.0
            ax.annotate(
                "",
                xy=mid + 0.08 * (b - a),
                xytext=mid - 0.08 * (b - a),
                arrowprops={"arrowstyle": "-|>", "lw": 0.8},
                zorder=4,
            )
            text = f"{info.index} {info.shape}"
            ax.annotate(
                text,
                mid,
                xytext=0.12 * np.linalg.norm(b - a) * outward + mid,
                ha="center",
                va="center",
                fontsize=8,
                zorder=5,
            )
    ax.autoscale_view()
    ax.set_aspect("equal")
    return ax


def plot_tiles(tiles: TileSet, ax: Any = None, color: str = "coloring") -> Any:
    """Draw tiles with their curved edges.

    Parameters
    ----------
    tiles : TileSet
    ax : matplotlib.axes.Axes, optional
        Axes to draw into; a new figure with equal axis scaling when omitted.
    color : str, default "coloring"
        ``"coloring"`` colors the tiles by the coloring of the type (tiles of one color are never adjacent),
        ``"aspect"`` by aspect, anything else is a matplotlib color for all tiles.

    Returns
    -------
    matplotlib.axes.Axes

    Raises
    ------
    ImportError
        If matplotlib is not installed.
    """
    _pyplot()
    from matplotlib.collections import PolyCollection

    ax = _axes(ax)
    if color == "coloring":
        faces: Any = [_PALETTE[int(c) % len(_PALETTE)] for c in tiles.colors()]
    elif color == "aspect":
        faces = [_PALETTE[int(a) % len(_PALETTE)] for a in tiles.aspect]
    else:
        faces = color
    ax.add_collection(
        PolyCollection(
            list(tiles.outlines()), facecolors=faces, edgecolors="#444444", linewidths=0.8, joinstyle="round"
        )
    )
    ax.autoscale_view()
    ax.set_aspect("equal")
    return ax
