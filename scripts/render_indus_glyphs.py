#!/usr/bin/env python3
"""Render NFM Indus Script font glyphs as PNG images.

The NFM font encodes Indus signs as PUA codepoints starting at U+E000.
Codepoint U+E000+N-1 renders sign N (1-indexed). We name output images
indus_M001.png … indus_M419.png, treating PUA-offset as Mahadevan number.
This is an approximation (Parpola and Mahadevan numbering differ), but
is sufficient for the KS-test statistical analysis in cross_script_similarity.py.

Usage:
    python scripts/render_indus_glyphs.py --font /path/to/NFM-IndusScript.ttf \
        --out-dir data/glyphs/indus --size 64 [--max-sign 419]
"""

import argparse
import logging
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

PUA_BASE = 0xE000  # U+E000 = sign index 1


def render_sign(font: ImageFont.FreeTypeFont, codepoint: int, size: int) -> Image.Image:
    """Render a single codepoint onto a white square canvas."""
    char = chr(codepoint)
    # Determine actual bounding box of the glyph
    bbox = font.getbbox(char)  # (left, top, right, bottom)
    if bbox is None or (bbox[2] - bbox[0]) == 0 or (bbox[3] - bbox[1]) == 0:
        return None

    glyph_w = bbox[2] - bbox[0]
    glyph_h = bbox[3] - bbox[1]

    # Render on a slightly larger canvas then crop to a square
    canvas_size = max(glyph_w, glyph_h) + 16
    img = Image.new("RGB", (canvas_size, canvas_size), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)

    x = (canvas_size - glyph_w) // 2 - bbox[0]
    y = (canvas_size - glyph_h) // 2 - bbox[1]
    draw.text((x, y), char, font=font, fill=(0, 0, 0))

    # Crop to content with small padding
    img = img.crop((
        max(0, (canvas_size - glyph_w) // 2 - 4),
        max(0, (canvas_size - glyph_h) // 2 - 4),
        min(canvas_size, (canvas_size + glyph_w) // 2 + 4),
        min(canvas_size, (canvas_size + glyph_h) // 2 + 4),
    ))

    img = img.resize((size, size), Image.LANCZOS)
    return img


def main() -> None:
    parser = argparse.ArgumentParser(description="Render NFM Indus script glyphs to PNGs")
    parser.add_argument("--font", type=Path, required=True,
                        help="Path to NFM-IndusScript.ttf")
    parser.add_argument("--out-dir", type=Path, default=Path("data/glyphs/indus"),
                        help="Output directory for PNG files")
    parser.add_argument("--size", type=int, default=64,
                        help="Output image size in pixels (square)")
    parser.add_argument("--max-sign", type=int, default=419,
                        help="Highest sign number to render (default 419 = full Mahadevan list)")
    parser.add_argument("--font-size", type=int, default=48,
                        help="Font point size used for rendering (default 48)")
    args = parser.parse_args()

    if not args.font.exists():
        raise FileNotFoundError(f"Font not found: {args.font}")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    try:
        pil_font = ImageFont.truetype(str(args.font), size=args.font_size)
    except Exception as e:
        raise RuntimeError(f"Failed to load font: {e}")

    rendered = 0
    skipped = 0
    already = 0

    for sign_num in range(1, args.max_sign + 1):
        out_path = args.out_dir / f"indus_M{sign_num:03d}.png"
        if out_path.exists():
            already += 1
            continue

        codepoint = PUA_BASE + (sign_num - 1)
        img = render_sign(pil_font, codepoint, args.size)
        if img is None:
            log.debug("Sign M%03d (U+%04X): empty glyph, skipping", sign_num, codepoint)
            skipped += 1
            continue

        img.save(out_path)
        rendered += 1

    log.info(
        "Done. Rendered: %d  Skipped (empty): %d  Already present: %d → %s",
        rendered, skipped, already, args.out_dir,
    )


if __name__ == "__main__":
    main()
