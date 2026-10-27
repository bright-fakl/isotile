"""Generate the isotile logo assets in ``docs/assets/logo/``.

The mark is a patch of a pentagonal tiling (isohedral type 28) made of tiles placed by isotile: the seven tiles
nearest a four-valent vertex, one amber and six blue in three tints, each inset by a gap so that the background shows
through. The favicon uses a variant with the four tiles around the vertex and wider gaps. The text of the lockup is
converted to outlined paths from DejaVu Sans Bold, so the SVG files contain no live text and need no font.

Run from the repository root (shapely comes from the dev group; the other packages are only needed here)::

    uv run --with cairosvg --with pillow --with fonttools python scripts/make_logo_assets.py

``--out`` changes the output directory. ``--check`` regenerates the SVG files in memory and exits with status 1 if
they differ from the files on disk (the PNG files are not compared, because rasterizers differ between versions).
The script prints the WCAG contrast ratios of the logo colors against the docs surfaces.
"""

from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path

import numpy as np
import shapely

from isotile import IsohedralTiling

LIGHT = {"blue": "#2c6e9e", "amber": "#c27400", "text": "#1b1f24", "mode": "light"}
DARK = {"blue": "#85b7e0", "amber": "#facc38", "text": "#e8edf2", "mode": "dark"}
MONO = {"blue": "currentColor", "amber": "currentColor", "text": "currentColor", "mode": "light"}
# Docs surfaces: Material default scheme (white) and slate scheme (hsl(232, 15%, 14%)), headers white and black.
SURFACES = {"light page": "#ffffff", "light header": "#ffffff", "dark page": "#1e2029", "dark header": "#000000"}
DARK_BG = "#1b1f24"  # surface of the social preview
SIZE = 96.0
TAGLINE = "Isohedral tilings in NumPy"
FONT_DIRS = ("/usr/share/fonts/truetype/dejavu", "/usr/share/fonts/dejavu", "/usr/share/fonts/TTF")

# ---- Geometry (from the isotile logo exploration, marks.py) ------------------------------------------------


def pth(points, close=True):
    d = "M" + " L".join(f"{x:.2f},{y:.2f}" for x, y in points)
    return d + (" Z" if close else "")


def poly_path(p):
    """SVG path of a shapely polygon (exterior only; all shapes here are simple)."""
    return pth(list(p.exterior.coords)[:-1])


def fit(polys, pad=7.0, extra=()):
    """Scale and center shapely polygons so that their joint bounds fill the mark box. Returns (polys, fn)."""
    x0, y0, x1, y1 = shapely.union_all(list(polys) + list(extra)).bounds
    s = (SIZE - 2 * pad) / max(x1 - x0, y1 - y0)
    ox = SIZE / 2 - s * (x0 + x1) / 2
    oy = SIZE / 2 - s * (y0 + y1) / 2

    def fn(p):
        return np.asarray(p, dtype=float) * s + [ox, oy]

    return [shapely.polygons(fn(p.exterior.coords)) for p in polys], fn, s


def shrink(p, gap):
    """Inset by half a gap with softly rounded corners."""
    q = p.buffer(-gap / 2 - 0.6, join_style="round").buffer(0.6, join_style="round")
    return q


def levels(pal, light=(1.0, 0.62, 0.34), dark=(1.0, 0.74, 0.5)):
    """Opacity levels of the blue tiles: lighter surfaces take lower levels, dark ones stay brighter."""
    return light if pal.get("mode", "light") == "light" else dark


def fill_attr(col, op):
    return f'fill="{col}"' + (f' fill-opacity="{op}"' if op < 1 else "")


def nearest(ts, k, center=None):
    c = ts.centroids
    if center is None:
        center = c.mean(axis=0)
    order = np.argsort(np.linalg.norm(c - center, axis=1))
    return order[:k]


def vertex_center(ts, valence):
    """A point of the patch where ``valence`` tiles meet (the vertex closest to the middle of the region)."""
    V = np.round(ts.vertices().reshape(-1, 2), 6)
    keys, cnt = np.unique(V, axis=0, return_counts=True)
    cand = keys[cnt == valence]
    return cand[np.argmin(np.linalg.norm(cand - 0.5, axis=1))]


def patch(pal, ttype, k=8, tints=None, gap=2.6, n_tiles=40, rotate=0.0, pad=6.0, params=None, valence=None, amber=None):
    """A small patch of an actual tiling: the k tiles nearest a center (the middle, or a vertex where ``valence``
    tiles meet), one amber."""
    tints = tints or levels(pal)
    t = IsohedralTiling(ttype, params)
    if rotate:
        t = t.rotated(rotate)
    ts = t.fill_region((0, 0, 1, 1), n_tiles=n_tiles)
    c0 = np.array([0.5, 0.5]) if valence is None else vertex_center(ts, valence)
    sel = nearest(ts, k, c0)
    ts = ts.select(sel)
    polys = ts.polygons()
    col = ts.colors()
    centre = int(np.argmin(np.linalg.norm(ts.centroids - c0, axis=1))) if amber is None else amber
    fp, _, _ = fit(polys, pad)
    out = ""
    for i, p in enumerate(fp):
        q = shrink(p, gap)
        if i == centre:
            out += f'<path d="{poly_path(q)}" fill="{pal["amber"]}"/>'
        else:
            out += f'<path d="{poly_path(q)}" {fill_attr(pal["blue"], tints[col[i] % len(tints)])}/>'
    return out


def mark_body(pal):
    return patch(pal, 28, k=7, params=[0.3153, 0.5], valence=4)


def favicon_body(pal):
    return patch(
        pal, 28, k=5, params=[0.3153, 0.5], valence=4, gap=4.4, pad=4.0, tints=levels(pal, (1.0, 0.55), (1.0, 0.65))
    )


# ---- SVG documents ----------------------------------------------------------------------------------------


def svg(body, w=96.0, h=96.0, title="isotile", width=None, height=None):
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w:.2f} {h:.2f}" '
        f'width="{width or w:.0f}" height="{height or h:.0f}" role="img" aria-label="{title}">'
        f"<title>{title}</title>{body}</svg>\n"
    )


def mark_svg(pal, mono=False):
    body = mark_body(pal)
    return svg(f'<g color="#000000">{body}</g>' if mono else body)


def favicon_svg():
    """Favicon mark with classes for the fills; the colors follow ``prefers-color-scheme``."""
    marker = {"blue": "__b__", "amber": "__a__", "mode": "light"}
    body = favicon_body(marker).replace('fill="__b__"', 'class="b"').replace('fill="__a__"', 'class="a"')
    style = (
        f"<style>.b{{fill:{LIGHT['blue']}}}.a{{fill:{LIGHT['amber']}}}"
        f"@media (prefers-color-scheme:dark){{.b{{fill:{DARK['blue']}}}.a{{fill:{DARK['amber']}}}}}</style>"
    )
    return svg(style + body)


# ---- Text outlines ----------------------------------------------------------------------------------------


def find_font(name):
    for d in FONT_DIRS:
        f = Path(d) / name
        if f.exists():
            return str(f)
    import matplotlib

    return str(Path(matplotlib.get_data_path()) / "fonts" / "ttf" / name)


def text_paths(text, font_file, size, x, baseline, fill):
    """Outlined text: one <path> per glyph. Returns the markup and the end x."""
    from fontTools.pens.svgPathPen import SVGPathPen
    from fontTools.pens.transformPen import TransformPen
    from fontTools.ttLib import TTFont

    font = TTFont(font_file)
    gs = font.getGlyphSet()
    cmap = font.getBestCmap()
    scale = size / font["head"].unitsPerEm
    out = ""
    for ch in text:
        name = cmap[ord(ch)]
        if ch != " ":
            pen = SVGPathPen(gs, ntos=lambda v: f"{v:.2f}")
            gs[name].draw(TransformPen(pen, (scale, 0, 0, -scale, x, baseline)))
            out += f'<path fill="{fill}" d="{pen.getCommands()}"/>'
        x += font["hmtx"][name][0] * scale
    return out, x


def lockup_svg(pal, size=50.0, gap=14.0):
    """Horizontal lockup: the 96 x 96 mark, then the name as outlines (x-height centered on the mark)."""
    glyphs, end = text_paths(
        "isotile", find_font("DejaVuSans-Bold.ttf"), size, SIZE + gap, 48 + 0.36 * size, pal["text"]
    )
    return svg(mark_body(pal) + glyphs, w=end, h=SIZE)


def social_svg(pal, bg):
    """1280 x 640 card: lockup scaled up, tagline below, on a plain surface."""
    wm = lockup_svg(pal)
    inner = wm[wm.index("</title>") + len("</title>") : -len("</svg>\n")]
    w = float(wm.split('viewBox="0 0 ')[1].split()[0])
    scale = 760 / w
    wm_h = SIZE * scale
    tag, tag_end = text_paths(TAGLINE, find_font("DejaVuSans.ttf"), 34, 0, 0, pal["text"])
    top = (640 - (wm_h + 70 + 34)) / 2
    return (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1280 640" width="1280" height="640">'
        f'<rect width="1280" height="640" fill="{bg}"/>'
        f'<g transform="translate({(1280 - 760) / 2:.1f},{top:.1f}) scale({scale:.4f})">{inner}</g>'
        f'<g transform="translate({(1280 - tag_end) / 2:.1f},{top + wm_h + 70:.1f})" opacity="0.85">{tag}</g></svg>\n'
    )


# ---- Contrast ---------------------------------------------------------------------------------------------


def luminance(hex_color):
    c = [int(hex_color[i : i + 2], 16) / 255 for i in (1, 3, 5)]
    c = [v / 12.92 if v <= 0.03928 else ((v + 0.055) / 1.055) ** 2.4 for v in c]
    return 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]


def contrast(a, b):
    hi, lo = sorted((luminance(a), luminance(b)), reverse=True)
    return (hi + 0.05) / (lo + 0.05)


def contrast_report():
    rows = []
    for variant, pal, surfaces in (
        ("light", LIGHT, ("light page", "light header")),
        ("dark", DARK, ("dark page", "dark header")),
    ):
        for s in surfaces:
            for role in ("blue", "amber", "text"):
                rows.append(
                    f"{variant:5s} {role:5s} {pal[role]} on {s:12s} {SURFACES[s]}: {contrast(pal[role], SURFACES[s]):.2f}"
                )
    return "\n".join(rows)


# ---- Rasterizing ------------------------------------------------------------------------------------------


def to_png(svg_text, path, width, height):
    import cairosvg

    cairosvg.svg2png(bytestring=svg_text.encode(), write_to=str(path), output_width=width, output_height=height)


def centered_png(svg_text, path, canvas, mark_px, background):
    """Mark of ``mark_px`` pixels centered on an opaque ``canvas`` x ``canvas`` background."""
    import cairosvg
    from PIL import Image

    buf = io.BytesIO()
    cairosvg.svg2png(bytestring=svg_text.encode(), write_to=buf, output_width=mark_px, output_height=mark_px)
    im = Image.open(buf).convert("RGBA")
    base = Image.new("RGBA", (canvas, canvas), background)
    off = (canvas - mark_px) // 2
    base.alpha_composite(im, (off, off))
    base.convert("RGB").save(path)


def svg_files():
    return {
        "mark-light.svg": mark_svg(LIGHT),
        "mark-dark.svg": mark_svg(DARK),
        "mark-mono.svg": mark_svg(MONO, mono=True),
        "lockup-light.svg": lockup_svg(LIGHT),
        "lockup-dark.svg": lockup_svg(DARK),
        "favicon.svg": favicon_svg(),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", type=Path, default=Path("docs/assets/logo"))
    ap.add_argument("--check", action="store_true", help="compare the SVG files with the files on disk")
    args = ap.parse_args()
    out = args.out
    files = svg_files()
    if args.check:
        stale = [n for n, text in files.items() if not (out / n).exists() or (out / n).read_text() != text]
        if stale:
            print("stale logo files:", ", ".join(stale))
            sys.exit(1)
        print("logo files are up to date")
        return
    out.mkdir(parents=True, exist_ok=True)
    for name, text in files.items():
        (out / name).write_text(text)
    fav = favicon_svg()
    to_png(fav, out / "favicon-32.png", 32, 32)
    centered_png(mark_svg(LIGHT), out / "apple-touch-icon.png", 180, 136, "#ffffff")
    to_png(social_svg(DARK, DARK_BG), out / "social-preview.png", 1280, 640)
    print(contrast_report())


if __name__ == "__main__":
    main()
