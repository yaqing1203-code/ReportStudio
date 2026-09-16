"""Generate the ReportStudio app icon: white "RS" on the brand orange.

Produces packaging/reportstudio.ico with the standard Windows icon sizes
(16/24/32/48/64/128/256). Run once (or after changing the look):

    conda run -n py311 python packaging/make_icon.py

The exe picks it up via the `icon=` argument in ReportStudio.spec.
"""
from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

# Brand orange used by the in-app "RS" mark (styles.css --accent).
BG = (201, 100, 66, 255)      # #c96442
FG = (255, 255, 255, 255)     # white text
SIZES = [16, 24, 32, 48, 64, 128, 256]

HERE = Path(__file__).resolve().parent
OUT = HERE / "reportstudio.ico"


def _load_font(px: int) -> ImageFont.FreeTypeFont:
    """A bold sans font at the requested pixel size, trying common Windows faces
    before falling back to Pillow's bundled default."""
    for name in ("arialbd.ttf", "segoeuib.ttf", "ariblk.ttf", "arial.ttf"):
        try:
            return ImageFont.truetype(name, px)
        except OSError:
            continue
    return ImageFont.load_default()


def _render(size: int) -> Image.Image:
    # Supersample for crisp edges, then downscale.
    scale = 4
    S = size * scale
    img = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)

    # Rounded-square background (radius ~22% like the in-app mark).
    radius = int(S * 0.22)
    d.rounded_rectangle([0, 0, S - 1, S - 1], radius=radius, fill=BG)

    # Centered "RS". Pick the largest font that fits ~72% of the width.
    text = "RS"
    target_w = S * 0.72
    px = int(S * 0.5)
    font = _load_font(px)
    for _ in range(24):
        box = d.textbbox((0, 0), text, font=font)
        w = box[2] - box[0]
        if w <= target_w or px <= 6:
            break
        px = int(px * target_w / max(w, 1))
        font = _load_font(px)

    box = d.textbbox((0, 0), text, font=font)
    w, h = box[2] - box[0], box[3] - box[1]
    x = (S - w) / 2 - box[0]
    y = (S - h) / 2 - box[1]
    d.text((x, y), text, font=font, fill=FG)

    return img.resize((size, size), Image.LANCZOS)


def main() -> None:
    base = _render(256)
    frames = [_render(s) for s in SIZES]
    base.save(OUT, format="ICO", sizes=[(s, s) for s in SIZES], append_images=frames)
    print(f"wrote {OUT} ({OUT.stat().st_size} bytes; sizes {SIZES})")


if __name__ == "__main__":
    main()
