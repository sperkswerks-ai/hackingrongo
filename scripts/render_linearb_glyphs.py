"""
render_linearb_glyphs.py

Renders Linear B syllabograms (Unicode U+10000–U+1007F) as PNG images using
PIL and a Unicode-capable font.  Generates the control image set used by
cross_script_similarity.py (same era as rongorongo, no proposed connection).

Usage:
    python scripts/render_linearb_glyphs.py [--out-dir data/glyphs/linear_b]
                                             [--size 64]
                                             [--font /path/to/font.ttf]
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Linear B Syllabary: U+10000–U+1007F (syllabograms only; ideograms at U+10080+)
# We render the syllabograms since they are the closest structural equivalent
# to rongorongo signs (phonetic signs in an undeciphered script context).
_LINEAR_B_SYLLABARY_RANGE = (0x10000, 0x1007F + 1)

# Candidate font paths (checked in order; first match is used)
_FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/noto/NotoSansLinearB-Regular.ttf",
    "/usr/local/share/fonts/NotoSansLinearB-Regular.ttf",
    "/System/Library/Fonts/Supplemental/NotoSansLinearB-Regular.ttf",
    # Noto Sans might cover supplementary planes; fall back to it
    "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
]

# Signs with no glyph in most fonts (unassigned or reserved codepoints)
_SKIP_CODEPOINTS: set[int] = {
    0x1000C, 0x10027, 0x1003B, 0x1003E, 0x1004E, 0x1004F,
    0x1005E, 0x1005F, 0x10060, 0x10061, 0x10062, 0x10063,
    0x10064, 0x10065, 0x10066, 0x10067, 0x10068, 0x10069,
    0x1006A, 0x1006B, 0x1006C, 0x1006D, 0x1006E, 0x1006F,
    0x10070, 0x10071, 0x10072, 0x10073, 0x10074, 0x10075,
    0x10076, 0x10077, 0x10078, 0x10079, 0x1007A, 0x1007B,
    0x1007C, 0x1007D, 0x1007E, 0x1007F,
}


def _find_font(font_path: str | None) -> str | None:
    if font_path:
        if Path(font_path).exists():
            return font_path
        log.warning("Specified font not found: %s — searching fallbacks", font_path)
    for candidate in _FONT_CANDIDATES:
        if Path(candidate).exists():
            return candidate
    return None


def _render_glyph(char: str, size: int, font) -> "Image.Image":
    from PIL import Image, ImageDraw

    img = Image.new("L", (size, size), color=255)
    draw = ImageDraw.Draw(img)
    # Centre the glyph
    try:
        bbox = font.getbbox(char)
        gw = bbox[2] - bbox[0]
        gh = bbox[3] - bbox[1]
        x = (size - gw) // 2 - bbox[0]
        y = (size - gh) // 2 - bbox[1]
    except Exception:
        x, y = size // 8, size // 8
    draw.text((x, y), char, fill=0, font=font)
    return img


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Render Linear B syllabogram PNG images")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "glyphs" / "linear_b",
    )
    parser.add_argument("--size", type=int, default=64)
    parser.add_argument("--font", type=str, default=None)
    args = parser.parse_args(argv)

    try:
        from PIL import Image, ImageFont  # noqa: F401
    except ImportError:
        log.error("Pillow is required: pip install Pillow")
        sys.exit(1)

    font_path = _find_font(args.font)
    if font_path is None:
        log.warning(
            "No Unicode font found for Linear B. Glyphs will render as boxes. "
            "Install NotoSansLinearB: https://fonts.google.com/noto/specimen/Noto+Sans+Linear+B"
        )
        font = ImageFont.load_default()
    else:
        log.info("Using font: %s", font_path)
        font_size = max(8, int(args.size * 0.7))
        font = ImageFont.truetype(font_path, size=font_size)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    n_rendered = n_skipped = 0
    for cp in range(*_LINEAR_B_SYLLABARY_RANGE):
        if cp in _SKIP_CODEPOINTS:
            continue
        char = chr(cp)
        filename = f"linearb_U{cp:05X}.png"
        out_path = args.out_dir / filename

        if out_path.exists():
            n_skipped += 1
            continue

        img = _render_glyph(char, args.size, font)
        img.save(out_path)
        n_rendered += 1

    log.info(
        "Done. Rendered: %d  Skipped (already present): %d → %s",
        n_rendered, n_skipped, args.out_dir,
    )


if __name__ == "__main__":
    main()
