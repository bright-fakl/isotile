# Logo and branding

The logo files are in `docs/assets/logo/`.

## The mark

The mark is a patch of a pentagonal tiling: the seven tiles nearest a four-valent vertex of isohedral type 28, placed by
isotile itself. One tile is amber and the other six are blue in three tints. Each tile is inset, so the background shows
through the gaps and the mark works on any surface.

<div class="it-surfaces">
  <div class="it-surface it-surface--light"><img src="../assets/logo/lockup-light.svg" alt="isotile lockup, light variant" height="64"></div>
  <div class="it-surface it-surface--dark"><img src="../assets/logo/lockup-dark.svg" alt="isotile lockup, dark variant" height="64"></div>
</div>

## Files

| File | Use |
|---|---|
| `mark-light.svg`, `mark-dark.svg` | Mark for light and dark surfaces |
| `mark-mono.svg` | One color (black); amber and blue tiles are both drawn in it, the tints keep the tiles apart |
| `lockup-light.svg`, `lockup-dark.svg` | Mark and the name `isotile`, text as outlined paths |
| `favicon.svg` | Mark of four tiles with wider gaps, for 16 px; switches colors with `prefers-color-scheme` |
| `favicon-32.png`, `apple-touch-icon.png` | Raster icons (32 px with a transparent background; 180 px with an opaque white background) |
| `social-preview.png` | Repository social preview, 1280 x 640 px |

<div class="it-surfaces">
  <div class="it-surface it-surface--light">
    <img src="../assets/logo/mark-light.svg" alt="Mark, light variant" height="96">
    <img src="../assets/logo/mark-mono.svg" alt="Mark, one color" height="96">
    <img src="../assets/logo/favicon-32.png" alt="Favicon, 32 px" height="32">
    <img src="../assets/logo/apple-touch-icon.png" alt="Apple touch icon, 180 px" height="96">
  </div>
  <div class="it-surface it-surface--dark">
    <img src="../assets/logo/mark-dark.svg" alt="Mark, dark variant" height="96">
  </div>
</div>

<img src="../assets/logo/social-preview.png" alt="Social preview, 1280 x 640 px" width="640">

## Colors

| Variant | Blue | Amber | Text | Use on |
|---|---|---|---|---|
| Light | `#2c6e9e` | `#c27400` | `#1b1f24` | white and light surfaces |
| Dark | `#85b7e0` | `#facc38` | `#e8edf2` | dark surfaces, e.g. `#1e2029` (docs, slate scheme) and `#000000` (docs header) |

The three blue tints are fixed opacities of the blue: 1.0, 0.62 and 0.34 in the light variant, 1.0, 0.74 and 0.5 in the
dark variant. The social preview uses the dark variant on `#1b1f24`.

WCAG contrast ratios against the docs surfaces:

| Color | On `#ffffff` | On `#1e2029` | On `#000000` |
|---|---|---|---|
| Light blue `#2c6e9e` | 5.48 | not used | not used |
| Light amber `#c27400` | 3.63 | not used | not used |
| Light text `#1b1f24` | 16.56 | not used | not used |
| Dark blue `#85b7e0` | not used | 7.61 | 9.85 |
| Dark amber `#facc38` | not used | 10.63 | 13.76 |
| Dark text `#e8edf2` | not used | 13.78 | 17.83 |

The amber is told apart from the blue by hue; the two are only 1.4 to 1.5 apart in lightness.

## Size and clear space

- Keep a clear space of 25% of the mark height on every side.
- Use the mark at 32 px high or more. At that size the gaps are about one pixel and the tints separate the tiles. Below
  32 px, down to 16 px, use `favicon.svg`: it has four tiles and wider gaps, and stays a cluster of four shapes with one
  amber shape.
- Use the lockup at 32 px high or more.

## Do and do not

- Use the light variant on light surfaces and the dark variant on dark surfaces.
- Do not recolor the mark or the lockup, except for `mark-mono.svg`.
- Do not stretch or rotate the files, and do not change the gap width.
- Do not put the light variant on a dark surface or the dark variant on a light surface.
- Do not retype the name in another font; use the files.

## License

The mark and the lockup layout are original work, released under the project's
[MIT license](https://github.com/bright-fakl/isotile/blob/main/LICENSE). The letters of the name are outlines of DejaVu
Sans Bold (the tagline in the social preview is DejaVu Sans). DejaVu fonts are based on Bitstream Vera and are distributed
under a free license that permits use, embedding and redistribution; see
[dejavu-fonts.github.io/License.html](https://dejavu-fonts.github.io/License.html).

## Regenerating

`scripts/make_logo_assets.py` generates every file in `docs/assets/logo/`. The tile geometry comes from isotile, so run it
in the development environment:

```bash
uv run --with cairosvg --with pillow --with fonttools python scripts/make_logo_assets.py
```

The command writes the SVG files, `favicon-32.png`, `apple-touch-icon.png` and `social-preview.png`. With `--check` it
regenerates the SVG files in memory and fails if they differ from the stored files:

```bash
uv run --with cairosvg --with pillow --with fonttools python scripts/make_logo_assets.py --check
```
