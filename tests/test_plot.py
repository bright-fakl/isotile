"""Plot helpers; matplotlib stays optional."""

import subprocess
import sys

import pytest

from isotile import IsohedralTiling


def test_importing_isotile_does_not_import_matplotlib():
    code = "import sys, isotile, isotile.plot; assert 'matplotlib' not in sys.modules"
    subprocess.run([sys.executable, "-c", code], check=True)


def test_plot_functions():
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    from isotile.plot import plot_prototile, plot_tiles

    tiling = IsohedralTiling.from_preset("flowers")
    ax = plot_prototile(tiling)
    assert len(ax.patches) == 2  # the outline and the dashed straight prototile
    assert any("S" in t.get_text() for t in ax.texts)
    assert len(plot_prototile(IsohedralTiling(7), show_labels=False).patches) == 1
    tiles = tiling.fill_region((0, 0, 3, 3), n_tiles=20)
    for color in ("coloring", "aspect", "tab:red"):
        ax = plot_tiles(tiles, color=color)
        assert len(ax.collections) == 1
        assert len(ax.collections[0].get_paths()) == len(tiles)
    same_axes = plot_tiles(tiles, ax=ax)
    assert same_axes is ax
    matplotlib.pyplot.close("all")
