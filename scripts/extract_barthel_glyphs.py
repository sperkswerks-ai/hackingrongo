"""extract_barthel_glyphs.py — Extract individual glyph images from Barthel's PDF plates.

Two extraction modes:

  formentafeln
      Barthel_Formentafeln.pdf (8 pages, A4 landscape).
      Reference sign tables showing canonical drawings of each Barthel sign
      type (Kennziffern 1–999).  Each entry = printed number label + glyph
      drawing.  Decade-grouped rows: row 1 = codes 1 + 10-19,
      row 2 = codes 2 + 20-29, …  For pages ≥ 2 each row = one decade.
      Output dir : data/glyphs/barthel_ref/
      source_quality: "barthel_reference"
      Filename: {barthel_code}_barthel_{page}_{cell:03d}.png
        e.g. 076_barthel_2_004.png

  tafeln
      Barthel_Tafeln.pdf (56 pages).  Pages 1–3 are tablet schematics
      (skipped).  Pages 4–56 are transcriptions: labelled glyph rows with
      left-margin line codes (e.g. Аа1 = tablet A, side a, line 1).
      Glyph position within each labelled line is linked to corpus JSON to
      resolve Barthel codes.
      Output dir : data/glyphs/barthel_corpus/
      source_quality: "barthel_scan"
      Filename: {barthel_code}_barthel_{page}_{seq:03d}.png
        e.g. 200_barthel_5_017.png
        Unlabelled glyphs use: unk_{tablet}{side}{line}s{seq}_barthel_{page}_{seq}.png

Usage::

    python scripts/extract_barthel_glyphs.py --source formentafeln \\
        [--pdf-dir data/barthel_pdfs] [--out-dir data/glyphs] \\
        [--corpus-dir data/corpus] [--dpi 200] [--glyph-size 64] [--dry-run]

    python scripts/extract_barthel_glyphs.py --source tafeln \\
        [--pdf-dir data/barthel_pdfs] [--out-dir data/glyphs] \\
        [--corpus-dir data/corpus]

    python scripts/extract_barthel_glyphs.py --source both

Dependencies::

    pip install pdf2image pillow scipy pytesseract
    brew install tesseract          # macOS — tesseract OCR engine (optional)
    brew install poppler            # required by pdf2image (pdftoppm)

Notes:

* Tesseract is used to OCR Formentafeln numeral labels and Tafeln page
  headers.  If unavailable the script falls back to positional estimation
  and emits a warning; labels may be less accurate.
* The outlined glyph style in Tafeln means strokes are thin rings; a small
  dilation bridges intra-glyph gaps before connected-component labelling.
  Adjacent glyphs that happen to touch will produce merged blobs — these are
  flagged ``merge_suspect=True`` in the catalog.
* Output PNGs are 64×64 grayscale (or the size set by --glyph-size), white
  background, glyph centred with padding.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import warnings
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image
from scipy import ndimage

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional OCR
# ---------------------------------------------------------------------------

try:
    import pytesseract
    _TESSERACT_AVAILABLE = True
except ImportError:
    _TESSERACT_AVAILABLE = False


def _ocr_digits(pil_crop: Image.Image) -> str:
    """Return digit string from a small PIL image, or '' on failure."""
    if not _TESSERACT_AVAILABLE:
        return ""
    cfg = "--psm 7 --oem 3 -c tessedit_char_whitelist=0123456789abcdefghijklmnopqrstuvwxyz"
    try:
        text = pytesseract.image_to_string(pil_crop, config=cfg).strip()
        return re.sub(r"[^0-9a-z]", "", text.lower())
    except Exception:
        return ""


def _ocr_header(pil_crop: Image.Image) -> str:
    """Return raw OCR text from a page header strip.

    Uses the ``eng`` language pack which is universally available with
    Tesseract.  The Barthel Tafeln headers contain only ASCII letters and
    digits so English OCR is sufficient; ``deu`` added no benefit and
    caused hard failures when the German language pack was absent.
    """
    if not _TESSERACT_AVAILABLE:
        return ""
    cfg = "--psm 6 --oem 3"
    try:
        return pytesseract.image_to_string(pil_crop, config=cfg).strip()
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# Image utilities
# ---------------------------------------------------------------------------

def _binarize(img_gray: np.ndarray, threshold: int = 128) -> np.ndarray:
    """Return boolean array True where pixel is dark (ink/stroke)."""
    return img_gray < threshold


def _find_blobs(binary: np.ndarray,
                dilation: int = 2,
                min_area: int = 50,
                max_area: int = 200_000) -> list[tuple[int, int, int, int, int]]:
    """Return list of (y0, x0, y1, x1, area) for connected components.

    Parameters
    ----------
    binary : bool ndarray H×W where True = ink
    dilation : morphological dilation iterations before labelling
    min_area / max_area : filter by pixel count
    """
    struct = ndimage.generate_binary_structure(2, 2)
    dilated = ndimage.binary_dilation(binary, structure=struct,
                                      iterations=dilation)
    labeled, _ = ndimage.label(dilated, structure=struct)
    # Recover original (un-dilated) bounding boxes from labeled regions
    blobs = []
    for region_id, size in enumerate(np.bincount(labeled.ravel())[1:], 1):
        if size < min_area or size > max_area:
            continue
        ys, xs = np.where(labeled == region_id)
        blobs.append((int(ys.min()), int(xs.min()),
                      int(ys.max()), int(xs.max()), int(size)))
    return blobs


def _pad_to_square_patch(pil_img: Image.Image,
                          y0: int, x0: int, y1: int, x1: int,
                          size: int = 64,
                          padding_frac: float = 0.1) -> Image.Image:
    """Crop a bounding box from pil_img, pad to square, resize to size×size."""
    h, w = y1 - y0, x1 - x0
    pad = int(max(h, w) * padding_frac)
    # Expand bbox with padding, clamp to image
    img_h, img_w = pil_img.height, pil_img.width
    y0c = max(0, y0 - pad)
    x0c = max(0, x0 - pad)
    y1c = min(img_h, y1 + pad)
    x1c = min(img_w, x1 + pad)
    crop = pil_img.crop((x0c, y0c, x1c, y1c))
    # Pad short axis to square on white background
    cw, ch = crop.width, crop.height
    side = max(cw, ch)
    square = Image.new("L", (side, side), color=255)
    square.paste(crop, ((side - cw) // 2, (side - ch) // 2))
    return square.resize((size, size), Image.LANCZOS)


# ---------------------------------------------------------------------------
# Corpus index (shared with link_svg_to_corpus)
# ---------------------------------------------------------------------------

_SIDE_TO_AB: dict[str, str] = {"r": "a", "v": "b", "a": "a", "b": "b"}


def _build_corpus_index(corpus_dir: Path) -> dict[str, dict]:
    """Return {pos_key: corpus_fields}.  Key: {TABLET}{SIDE_ab}{LINE}-{SEQ:03d}."""
    index: dict[str, dict] = {}
    for path in sorted(corpus_dir.glob("[A-Z].json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        tid = data["tablet_id"]
        cluster = data.get("cluster", "unknown")
        groups: dict[tuple[str, str], list] = defaultdict(list)
        for g in sorted(data["glyphs"], key=lambda g: g.get("position", 0)):
            side_ab = _SIDE_TO_AB.get(g.get("side", "a"), "a")
            line = g.get("line", "01")
            groups[(side_ab, line)].append(g)
        for (side_ab, line), glyphs in groups.items():
            for seq, g in enumerate(glyphs, 1):
                key = f"{tid}{side_ab}{line}-{seq:03d}"
                index[key] = {
                    "barthel_code": g.get("barthel_code"),
                    "horley_code": g.get("horley_code"),
                    "horley_components": g.get("horley_components"),
                    "inverted": g.get("inverted", False),
                    "uncertain": g.get("uncertain", False),
                    "cluster": cluster,
                }
    return index


# ---------------------------------------------------------------------------
# Formentafeln extraction
# ---------------------------------------------------------------------------

# Each PDF page header says "Formentafel N (Kennziffern X—Y)".
# Known page ranges (1-indexed):
_FORMENTAFELN_PAGE_RANGES = {
    1: (1, 99),
    2: (100, 196),
    3: (200, 299),
    4: (300, 399),
    5: (400, 499),
    6: (500, 599),
    7: (600, 799),
    8: (700, 999),
}

# Blobs above this area (px²) at 200dpi are glyph drawings; below = numeral labels.
_FORMEN_GLYPH_MIN_AREA = 350    # px² at 200dpi (applied to dilation=2 blobs)
_FORMEN_NUMERAL_MAX_AREA = 640  # px² at 200dpi (applied to raw undilated digit blobs, max observed ~531)
_FORMEN_NOISE_MIN_AREA = 20     # below this = noise


@dataclass
class _FormenEntry:
    barthel_code: str
    glyph_bbox: tuple[int, int, int, int]   # y0, x0, y1, x1
    page: int
    numeral_ocr: str = ""
    confidence: str = "positional"  # "ocr" | "positional"


def _cluster_numeral_blobs(
    numeral_blobs: list[tuple[int, int, int, int, int]],
    x_gap: int = 20,
    y_gap: int = 14,
) -> list[tuple[int, int, int, int]]:
    """Group individual digit blobs into per-numeral bounding boxes.

    Multi-digit numbers like "076" split into three separate connected
    components.  This function merges blobs that are horizontally adjacent
    (x_gap) and at similar vertical positions (y_gap) into one bbox.
    Returns list of merged (y0, x0, y1, x1) bounding boxes.
    """
    if not numeral_blobs:
        return []
    # Sort left-to-right so the greedy merge naturally assembles digits in order
    sorted_blobs = sorted(numeral_blobs, key=lambda b: b[1])  # by x0
    clusters: list[list[tuple]] = [[sorted_blobs[0]]]
    for blob in sorted_blobs[1:]:
        by0, bx0, by1, bx1, _ = blob
        b_cy = (by0 + by1) / 2
        prev = clusters[-1]
        prev_x1 = max(b[3] for b in prev)
        prev_cy = sum((b[0] + b[2]) / 2 for b in prev) / len(prev)
        if bx0 - prev_x1 <= x_gap and abs(b_cy - prev_cy) <= y_gap:
            clusters[-1].append(blob)
        else:
            clusters.append([blob])
    merged = []
    for cluster in clusters:
        ys0 = min(b[0] for b in cluster)
        xs0 = min(b[1] for b in cluster)
        ys1 = max(b[2] for b in cluster)
        xs1 = max(b[3] for b in cluster)
        merged.append((ys0, xs0, ys1, xs1))
    return merged


def _raw_digit_blobs(
    binary: np.ndarray,
    min_area: int = 20,
    max_area: int = 600,
) -> list[tuple[int, int, int, int, int]]:
    """Return (y0, x0, y1, x1, area) blobs from undilated binary image.

    Using scipy.ndimage.label directly avoids the scipy quirk where
    binary_dilation(iterations=0) runs until convergence instead of
    performing zero dilation iterations.
    """
    struct = ndimage.generate_binary_structure(2, 2)
    labeled, _ = ndimage.label(binary, structure=struct)
    blobs = []
    counts = np.bincount(labeled.ravel())
    for region_id, size in enumerate(counts[1:], 1):
        if size < min_area or size > max_area:
            continue
        ys, xs = np.where(labeled == region_id)
        blobs.append((int(ys.min()), int(xs.min()),
                      int(ys.max()), int(xs.max()), int(size)))
    return blobs


def _pair_numerals_and_glyphs(
    glyph_blobs: list[tuple[int, int, int, int, int]],
    binary: np.ndarray,
    page_width: int,
) -> list[tuple[tuple, tuple]]:
    """Pair each glyph blob with the numeral cluster immediately to its left.

    Uses raw (undilated) connected components for digit detection so that
    individual digit strokes of multi-digit numerals remain separate blobs
    that can then be clustered into per-numeral bounding boxes.

    Returns list of (numeral_cluster_bbox_or_None, glyph_bbox) in reading order.
    """
    glyphs = [(y0, x0, y1, x1) for y0, x0, y1, x1, a in glyph_blobs
              if a >= _FORMEN_GLYPH_MIN_AREA]
    # Collect all small raw blobs that could be digit strokes
    all_digit_blobs = _raw_digit_blobs(binary,
                                        min_area=_FORMEN_NOISE_MIN_AREA,
                                        max_area=_FORMEN_NUMERAL_MAX_AREA)
    pairs = []
    for gy0, gx0, gy1, gx1 in glyphs:
        g_cy = (gy0 + gy1) / 2
        gh = gy1 - gy0
        # Digit blobs must be left of the glyph and at similar y-centre
        candidates = [
            b for b in all_digit_blobs
            if b[3] <= gx0 and abs((b[0] + b[2]) / 2 - g_cy) < gh * 0.8
        ]
        if not candidates:
            pairs.append((None, (gy0, gx0, gy1, gx1)))
            continue
        # Cluster remaining candidates into per-numeral merged bboxes
        clusters = _cluster_numeral_blobs(candidates)
        # Nearest cluster = rightmost right-edge (closest to glyph left)
        nearest = max(clusters, key=lambda c: c[3])
        pairs.append((nearest, (gy0, gx0, gy1, gx1)))
    # Reading order: row band (80px tolerance) then left-to-right
    pairs.sort(key=lambda p: (p[1][0] // 80, p[1][1]))
    return pairs


def extract_formentafeln(
    pdf_path: Path,
    out_dir: Path,
    dpi: int = 200,
    glyph_size: int = 64,
    dry_run: bool = False,
) -> list[dict]:
    """Extract canonical sign prototypes from Barthel_Formentafeln.pdf."""
    from pdf2image import convert_from_path

    log.info("Formentafeln: rendering %s at %d dpi …", pdf_path.name, dpi)
    pages = convert_from_path(str(pdf_path), dpi=dpi)

    out_dir.mkdir(parents=True, exist_ok=True)
    catalog: list[dict] = []

    for page_num, page_img in enumerate(pages, 1):
        page_range = _FORMENTAFELN_PAGE_RANGES.get(page_num)
        log.info("  Page %d / %d  (Kennziffern %s)",
                 page_num, len(pages),
                 f"{page_range[0]}–{page_range[1]}" if page_range else "?")

        gray = np.array(page_img.convert("L"))
        binary = _binarize(gray)
        blobs = _find_blobs(binary, dilation=2,
                            min_area=_FORMEN_NOISE_MIN_AREA,
                            max_area=50_000)

        pairs = _pair_numerals_and_glyphs(blobs, binary, gray.shape[1])
        log.debug("    %d glyph blobs found", len(pairs))

        for numeral_bbox, glyph_bbox in pairs:
            # --- determine Barthel code ---
            code = ""
            confidence = "positional"
            if numeral_bbox and _TESSERACT_AVAILABLE:
                # OCR the tight bounding box of the already-clustered numeral.
                # _cluster_numeral_blobs merged individual digit blobs into one
                # bbox per number, so this crop contains exactly one numeral.
                ny0, nx0, ny1, nx1 = numeral_bbox
                pad = 5
                crop = page_img.convert("L").crop((
                    max(0, nx0 - pad), max(0, ny0 - pad),
                    nx1 + pad, ny1 + pad,
                ))
                # Upscale small crops for better tesseract accuracy
                min_side = 36
                if crop.height < min_side:
                    scale = max(2, min_side // crop.height)
                    crop = crop.resize(
                        (crop.width * scale, crop.height * scale),
                        Image.NEAREST,
                    )
                raw = _ocr_digits(crop)
                # _ocr_digits already returns lowercase [0-9a-z]; split into
                # a numeric prefix and an optional variant-letter suffix so
                # we can validate the number against the page range while
                # still preserving codes like "400b", "440a", "421d".
                _lm = re.match(r'^([0-9]+)([a-z]*)$', raw)
                ocr_digits  = _lm.group(1) if _lm else re.sub(r"[^0-9]", "", raw)
                ocr_suffix  = _lm.group(2) if _lm else ""
                # Validate against known page range to reject merged/garbage reads
                code = ""
                if ocr_digits and page_range:
                    try:
                        val = int(ocr_digits)
                        if page_range[0] <= val <= page_range[1]:
                            code = ocr_digits + ocr_suffix
                    except ValueError:
                        pass
                elif ocr_digits:
                    code = ocr_digits + ocr_suffix  # no range known, accept as-is
                confidence = "ocr" if code else "positional"

            # Positional fallback: rank within the page × known range
            if not code and page_range:
                rank = pairs.index((numeral_bbox, glyph_bbox))
                # We don't know exact gaps, so just note the rank
                code = f"{page_range[0]}+{rank}"
                confidence = "positional"

            gy0, gx0, gy1, gx1 = glyph_bbox
            cell = pairs.index((numeral_bbox, glyph_bbox)) + 1
            safe_code = code.replace('+', '_') if code else f"unk_p{page_num}c{cell}"
            fname = f"{safe_code}_barthel_{page_num}_{cell:03d}.png"
            out_path = out_dir / fname

            if not dry_run:
                patch = _pad_to_square_patch(page_img.convert("L"),
                                             gy0, gx0, gy1, gx1, size=glyph_size)
                patch.save(str(out_path))

            catalog.append({
                "source": "barthel_formentafeln",
                "source_quality": "barthel_reference",
                "barthel_code": code if confidence == "ocr" else None,
                "barthel_code_estimate": code,
                "code_confidence": confidence,
                "page": page_num,
                "bbox": list(glyph_bbox),
                "path": str(out_path.relative_to(out_dir.parent)),
                "merge_suspect": False,
            })

    log.info("Formentafeln: %d glyphs extracted", len(catalog))
    return catalog


# ---------------------------------------------------------------------------
# Tafeln extraction
# ---------------------------------------------------------------------------

_TAFELN_FIRST_TRANSCRIPTION_PAGE = 4   # pages 1-3 are schematics
_TAFELN_MARGIN_X_FRAC = 0.13           # left margin fraction where labels live
_TAFELN_LABEL_MIN_AREA = 150           # px² at 200dpi
_TAFELN_LABEL_MAX_AREA = 8_000
_TAFELN_GLYPH_MIN_AREA = 500
_TAFELN_GLYPH_MAX_AREA = 12_000
_TAFELN_MERGE_AREA_THRESHOLD = 7_000   # blobs larger than this may be merges

# Regex to parse page header text from OCR (e.g. "Tafel „Tahua" (Aa 1—Aa 3)").
# Barthel's Tafeln uses mixed side-notation: some tablets use a/b (recto/verso),
# others use r/v.  Both are accepted; callers normalise to a/b via _SIDE_TO_AB.
_HEADER_RE = re.compile(
    r"\(([A-Z])\s*([abrv])\s*(\d+)\s*[-\u2013\u2014]\s*[A-Z]\s*[abrv]\s*(\d+)\)",
    re.IGNORECASE,
)

# Map raw OCR side codes to the canonical a/b convention.
_SIDE_TO_AB: dict[str, str] = {"a": "a", "r": "a", "b": "b", "v": "b"}

# Fallback page index if OCR is unavailable or the header does not match.
# Derived from Barthel (1958) Tafeln structure, verified against OCR sampling.
# Format: page_num → (tablet_id, side, lines_on_page)
# Side: 'a' = recto, 'b' = verso.
_TAFELN_PAGE_INDEX: dict[int, tuple[str, str, list[int]]] = {
    # ── Tablet A — Tahua (recto=8 lines, verso=8 lines) ──────────────────────
    4:  ("A", "a", [1, 2, 3]),
    5:  ("A", "a", [4, 5, 6]),
    6:  ("A", "a", [7, 8]),
    7:  ("A", "b", [1, 2, 3]),
    8:  ("A", "b", [4, 5, 6]),
    9:  ("A", "b", [7, 8]),
    # ── Tablet B — Aruku-Kurenga (recto=10, verso=12) ────────────────────────
    # OCR confirmed: page 11 = Br 8-10, page 13 = Bv 8-12
    10: ("B", "a", [1, 2, 3, 4, 5, 6, 7]),
    11: ("B", "a", [8, 9, 10]),
    12: ("B", "b", [1, 2, 3, 4, 5, 6, 7]),
    13: ("B", "b", [8, 9, 10, 11, 12]),
    # ── Tablet C — Mamari (recto=14, verso=14) ───────────────────────────────
    # OCR confirmed: page 15 = Ca 8-14
    14: ("C", "a", [1, 2, 3, 4, 5, 6, 7]),
    15: ("C", "a", [8, 9, 10, 11, 12, 13, 14]),
    16: ("C", "b", [1, 2, 3, 4, 5, 6, 7]),
    17: ("C", "b", [8, 9, 10, 11, 12, 13, 14]),
    # ── Tablet D — Échancrée (recto=7, verso=6) ──────────────────────────────
    18: ("D", "a", [1, 2, 3, 4, 5, 6, 7]),
    # ── Tablet E — Keiti (recto=9, verso=8) ──────────────────────────────────
    # OCR identifies pages 19-23 as "Keiti" (= tablet E)
    19: ("E", "a", [1, 2, 3]),
    20: ("E", "a", [4, 5, 6]),
    21: ("E", "a", [7, 8, 9]),
    22: ("E", "b", [1, 2, 3, 4]),
    23: ("E", "b", [5, 6, 7, 8]),
    # ── Tablet G — Small Santiago (recto=8, verso=8) ─────────────────────────
    # OCR identifies pages 24-25 as "Kleine Santiagotafel" (= tablet G)
    24: ("G", "a", [1, 2, 3, 4, 5, 6, 7, 8]),
    25: ("G", "b", [1, 2, 3, 4, 5, 6, 7, 8]),
    # ── Tablet H — Great Santiago (recto=12, verso=12) ───────────────────────
    # OCR confirmed: page 26 = Hr 1-5, page 27 = Hr 6-10
    26: ("H", "a", [1, 2, 3, 4, 5]),
    27: ("H", "a", [6, 7, 8, 9, 10]),
    28: ("H", "a", [11, 12]),
    29: ("H", "b", [1, 2, 3, 4, 5]),
    30: ("H", "b", [6, 7, 8, 9, 10]),
    31: ("H", "b", [11, 12]),
    # ── Tablet I — Santiago Staff (recto=14 lines, no verso) ─────────────────
    # OCR identifies pages 32-38 as "Santiagostab" (= tablet I)
    32: ("I", "a", [1, 2]),
    33: ("I", "a", [3, 4]),
    34: ("I", "a", [5, 6]),
    35: ("I", "a", [7, 8, 9]),
    36: ("I", "a", [10, 11]),
    37: ("I", "a", [12, 13]),
    38: ("I", "a", [14]),
    # ── Tablet M — Great Vienna (recto=9 lines, no verso) ────────────────────
    # OCR identifies page 39 as "Große Wientafel" (= tablet M)
    39: ("M", "a", [1, 2, 3, 4, 5, 6, 7, 8, 9]),
    # ── Tablet N — Small Vienna (recto=5, verso=5) ───────────────────────────
    # OCR identifies page 40 as "Kleine Wientafel" (= tablet N)
    40: ("N", "a", [1, 2, 3, 4, 5]),
    # ── Tablet O — Boomerang / Berlintafel (recto=7 lines, no verso) ─────────
    # OCR confirmed: page 41 = "(O 1—O 7)" exactly matching O's 7 recto lines
    41: ("O", "a", [1, 2, 3, 4, 5, 6, 7]),
    # ── Tablet P — Great St. Petersburg (recto=11, verso=11) ─────────────────
    # OCR identifies pages 42-45 as "Große Leningradtafel" (= tablet P)
    42: ("P", "a", [1, 2, 3, 4, 5, 6]),
    43: ("P", "a", [7, 8, 9, 10, 11]),
    44: ("P", "b", [1, 2, 3, 4, 5, 6]),
    45: ("P", "b", [7, 8, 9, 10, 11]),
    # ── Tablet K — Small London (recto=5, verso=5) ───────────────────────────
    46: ("K", "a", [1, 2, 3, 4, 5]),
    47: ("K", "b", [1, 2, 3, 4, 5]),
    # ── Tablet Q — Small St. Petersburg (recto=9, verso=9) ───────────────────
    # OCR identifies page 49 as "Kleine Leningradtafel" (= tablet Q)
    48: ("Q", "a", [1, 2, 3, 4, 5, 6, 7, 8, 9]),
    49: ("Q", "b", [1, 2, 3, 4, 5, 6, 7, 8, 9]),
    # ── Tablet R — Atua-Mata-Riri (recto=8, verso=8) ─────────────────────────
    # OCR identifies pages 50-51 as "Atua-mata-riri" (= tablet R)
    50: ("R", "a", [1, 2, 3, 4, 5, 6, 7, 8]),
    51: ("R", "b", [1, 2, 3, 4, 5, 6, 7, 8]),
    # ── Tablet S — Great Washington (recto=8, verso=8) ───────────────────────
    # OCR confirmed: page 52 = Sa 1-8
    52: ("S", "a", [1, 2, 3, 4, 5, 6, 7, 8]),
    53: ("S", "b", [1, 2, 3, 4, 5, 6, 7, 8]),
    # ── Small tablets: T, U, V, W, X, Y (Honolulu, Tangata Manu, Snuff Box) ──
    54: ("T", "a", [1, 2, 3, 4, 5, 6]),
    55: ("T", "a", [7, 8, 9, 10, 11]),
    56: ("Y", "a", [1, 2, 3]),
}


def _detect_sections(binary: np.ndarray,
                     margin_x: int,
                     dilation: int = 3) -> list[int]:
    """Return sorted list of Y coordinates marking the start of each line section.

    Looks for isolated blobs in the left margin (x < margin_x).
    """
    margin_strip = binary[:, :margin_x]
    struct = ndimage.generate_binary_structure(2, 2)
    dilated = ndimage.binary_dilation(margin_strip, structure=struct,
                                      iterations=dilation)
    labeled, _ = ndimage.label(dilated, structure=struct)
    section_ys = []
    for region_id, size in enumerate(np.bincount(labeled.ravel())[1:], 1):
        if size < _TAFELN_LABEL_MIN_AREA or size > _TAFELN_LABEL_MAX_AREA:
            continue
        ys, _ = np.where(labeled == region_id)
        section_ys.append(int(ys.min()))
    return sorted(section_ys)


def _segment_glyphs_in_section(
    binary: np.ndarray,
    y_top: int,
    y_bot: int,
    x_start: int,
) -> list[tuple[int, int, int, int, int, bool]]:
    """Return (y0, x0, y1, x1, area, merge_suspect) for each glyph in section."""
    region = binary[y_top:y_bot, x_start:]
    struct = ndimage.generate_binary_structure(2, 2)
    dilated = ndimage.binary_dilation(region, structure=struct, iterations=3)
    labeled, _ = ndimage.label(dilated, structure=struct)
    glyphs = []
    for region_id, size in enumerate(np.bincount(labeled.ravel())[1:], 1):
        if size < _TAFELN_GLYPH_MIN_AREA or size > _TAFELN_GLYPH_MAX_AREA:
            continue
        ys, xs = np.where(labeled == region_id)
        glyphs.append((
            int(ys.min()) + y_top,
            int(xs.min()) + x_start,
            int(ys.max()) + y_top,
            int(xs.max()) + x_start,
            int(size),
            size > _TAFELN_MERGE_AREA_THRESHOLD,
        ))
    # Sort: top-to-bottom by row band (50px), then left-to-right
    glyphs.sort(key=lambda g: (g[0] // 50, g[1]))
    return glyphs


def _parse_header_for_line_info(
    page_img: Image.Image,
    page_num: int,
    img_height: int,
) -> Optional[tuple[str, str, int, int]]:
    """Return (tablet, side, start_line, end_line) from page header OCR or index."""
    # 1. Try OCR on the top 150px header strip
    if _TESSERACT_AVAILABLE:
        header_height = min(150, img_height // 8)
        header_crop = page_img.convert("L").crop(
            (0, 0, page_img.width, header_height))
        text = _ocr_header(header_crop)
        m = _HEADER_RE.search(text)
        if m:
            tablet = m.group(1).upper()
            # Normalise r/v (recto/verso) → a/b
            side = _SIDE_TO_AB.get(m.group(2).lower(), m.group(2).lower())
            return tablet, side, int(m.group(3)), int(m.group(4))

    # 2. Fall back to hard-coded page index
    entry = _TAFELN_PAGE_INDEX.get(page_num)
    if entry:
        tablet, side, lines = entry
        return tablet, side, lines[0], lines[-1]

    log.warning("    Page %d: could not determine tablet/line info", page_num)
    return None


def extract_tafeln(
    pdf_path: Path,
    out_dir: Path,
    corpus_dir: Path,
    dpi: int = 200,
    glyph_size: int = 64,
    first_page: int = _TAFELN_FIRST_TRANSCRIPTION_PAGE,
    dry_run: bool = False,
) -> list[dict]:
    """Extract glyph instances from Barthel_Tafeln.pdf transcription pages."""
    from pdf2image import convert_from_path

    log.info("Tafeln: building corpus index from %s …", corpus_dir)
    corpus_index = _build_corpus_index(corpus_dir)
    log.info("  Corpus index: %d keys", len(corpus_index))

    log.info("Tafeln: rendering %s pages %d– at %d dpi …",
             pdf_path.name, first_page, dpi)
    pages = convert_from_path(str(pdf_path), dpi=dpi,
                               first_page=first_page)

    catalog: list[dict] = []

    for page_offset, page_img in enumerate(pages):
        page_num = first_page + page_offset
        log.info("  Page %d / 56", page_num)

        gray = np.array(page_img.convert("L"))
        binary = _binarize(gray)
        img_h, img_w = gray.shape
        margin_x = int(img_w * _TAFELN_MARGIN_X_FRAC)

        # --- detect section boundaries ---
        section_ys = _detect_sections(binary, margin_x)
        if not section_ys:
            log.warning("    No sections detected on page %d — skipping", page_num)
            continue
        # Add page bottom as sentinel
        section_ys.append(img_h)
        log.debug("    %d sections at y=%s", len(section_ys) - 1, section_ys[:-1])

        # --- determine tablet / side / line range for this page ---
        line_info = _parse_header_for_line_info(page_img, page_num, img_h)
        if line_info:
            tablet, side, start_line, end_line = line_info
            n_sections = len(section_ys) - 1
            line_numbers = list(range(start_line, end_line + 1))
            # If detected sections ≠ expected lines, log a warning but continue
            if len(line_numbers) != n_sections:
                log.warning(
                    "    Page %d: expected %d sections for lines %d-%d, "
                    "found %d — assignments may be off",
                    page_num, len(line_numbers), start_line, end_line,
                    n_sections,
                )
        else:
            tablet, side = "?", "a"
            line_numbers = list(range(1, len(section_ys)))

        # --- per-section glyph extraction ---
        for sec_idx, (y_top, y_bot) in enumerate(
            zip(section_ys[:-1], section_ys[1:])
        ):
            line_num = line_numbers[sec_idx] if sec_idx < len(line_numbers) else sec_idx + 1
            line_str = f"{line_num:02d}"

            glyphs = _segment_glyphs_in_section(binary, y_top, y_bot, margin_x)
            log.debug("      Section %d (line %s): %d glyphs", sec_idx + 1,
                      line_str, len(glyphs))

            # Build section output dir
            tab_dir = out_dir / tablet
            if not dry_run:
                tab_dir.mkdir(parents=True, exist_ok=True)

            for seq, (gy0, gx0, gy1, gx1, area, merge_suspect) in enumerate(glyphs, 1):
                pos_key = f"{tablet}{side}{line_str}-{seq:03d}"
                corpus_entry = corpus_index.get(pos_key, {})
                barthel_code = corpus_entry.get("barthel_code")

                safe_code = barthel_code if barthel_code else f"unk_{tablet}{side}{line_str}s{seq:03d}"
                fname = f"{safe_code}_barthel_{page_num}_{seq:03d}.png"
                out_path = tab_dir / fname

                if not dry_run:
                    patch = _pad_to_square_patch(page_img.convert("L"),
                                                 gy0, gx0, gy1, gx1,
                                                 size=glyph_size)
                    patch.save(str(out_path))

                catalog.append({
                    "source": "barthel_tafeln",
                    "source_quality": "barthel_scan",
                    "tablet": tablet,
                    "side": side,
                    "line": line_str,
                    "seq_on_line": seq,
                    "barthel_code": barthel_code,
                    "horley_code": corpus_entry.get("horley_code"),
                    "horley_components": corpus_entry.get("horley_components"),
                    "inverted": corpus_entry.get("inverted", False),
                    "uncertain": corpus_entry.get("uncertain", False),
                    "cluster": corpus_entry.get("cluster", "unknown"),
                    "corpus_key": pos_key,
                    "page": page_num,
                    "bbox": [gy0, gx0, gy1, gx1],
                    "path": str(out_path.relative_to(out_dir.parent))
                            if not dry_run else str(out_path),
                    "merge_suspect": merge_suspect,
                })

    linked = sum(1 for r in catalog if r.get("barthel_code"))
    log.info("Tafeln: %d glyphs extracted, %d linked to corpus (%.1f%%)",
             len(catalog), linked,
             100 * linked / len(catalog) if catalog else 0)
    return catalog


# ---------------------------------------------------------------------------
# Combined catalog writer
# ---------------------------------------------------------------------------

def _write_catalog(records: list[dict], out_path: Path) -> None:
    import numpy as np

    def _default(obj):
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")

    existing: list[dict] = []
    if out_path.exists():
        raw = json.loads(out_path.read_text(encoding="utf-8"))
        existing = raw.get("records", raw) if isinstance(raw, dict) else raw
    merged = existing + records
    out_path.write_text(
        json.dumps({"records": merged}, ensure_ascii=False, indent=2, default=_default),
        encoding="utf-8",
    )
    log.info("Catalog written: %s (%d records)", out_path, len(merged))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--source", choices=["formentafeln", "tafeln", "both"],
                   default="both",
                   help="Which PDF to process (default: both)")
    p.add_argument("--pdf-dir", default="data/barthel_pdfs",
                   help="Directory containing the Barthel PDFs")
    p.add_argument("--out-dir", default="data/glyphs",
                   help="Root output directory (default: data/glyphs)")
    p.add_argument("--corpus-dir", default="data/corpus",
                   help="Corpus JSON directory (default: data/corpus)")
    p.add_argument("--dpi", type=int, default=200,
                   help="Rasterisation DPI (default: 200)")
    p.add_argument("--glyph-size", type=int, default=64,
                   help="Output PNG size in pixels (default: 64)")
    p.add_argument("--dry-run", action="store_true",
                   help="Detect and count glyphs without writing files")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(message)s",
    )

    if not _TESSERACT_AVAILABLE:
        warnings.warn(
            "pytesseract not importable — OCR disabled.  "
            "Install with: pip install pytesseract && brew install tesseract",
            UserWarning,
            stacklevel=1,
        )

    pdf_dir = Path(args.pdf_dir)
    out_dir = Path(args.out_dir)
    corpus_dir = Path(args.corpus_dir)

    all_records: list[dict] = []

    # --- Formentafeln ---
    if args.source in ("formentafeln", "both"):
        pdf = pdf_dir / "Barthel_Formentafeln.pdf"
        if not pdf.exists():
            log.error("Not found: %s", pdf)
        else:
            records = extract_formentafeln(
                pdf_path=pdf,
                out_dir=out_dir / "barthel_ref",
                dpi=args.dpi,
                glyph_size=args.glyph_size,
                dry_run=args.dry_run,
            )
            all_records.extend(records)
            coded = sum(1 for r in records if r.get("barthel_code"))
            print(f"Formentafeln: {len(records)} glyphs extracted, "
                  f"{coded} with confirmed Barthel code")

    # --- Tafeln ---
    if args.source in ("tafeln", "both"):
        pdf = pdf_dir / "Barthel_Tafeln.pdf"
        if not pdf.exists():
            log.error("Not found: %s", pdf)
        else:
            records = extract_tafeln(
                pdf_path=pdf,
                out_dir=out_dir / "barthel_corpus",
                corpus_dir=corpus_dir,
                dpi=args.dpi,
                glyph_size=args.glyph_size,
                dry_run=args.dry_run,
            )
            all_records.extend(records)
            linked = sum(1 for r in records if r.get("barthel_code"))
            suspect = sum(1 for r in records if r.get("merge_suspect"))
            print(f"Tafeln: {len(records)} glyphs extracted, "
                  f"{linked} linked to corpus, "
                  f"{suspect} merge-suspect")

    # --- Catalog ---
    if not args.dry_run and all_records:
        cat_path = out_dir / "barthel_catalog.json"
        _write_catalog(all_records, cat_path)
        print(f"Catalog written: {cat_path} ({len(all_records)} records)")
    elif args.dry_run:
        print(f"(dry-run) Total glyphs detected: {len(all_records)}")


if __name__ == "__main__":
    main()
