#!/usr/bin/env python3
"""
Glyph segmentation from rendered 3D tablet views.

Takes PNG screenshots produced by render_tablet_views.py and extracts
per-glyph crops using CLAHE + Canny + connected-component analysis.
Assigns corpus glyph IDs by matching boustrophedon sort order to the
corpus JSON line/segment structure.

No pixel-coordinate metadata exists in the corpus — assignment is
positional only (order within a line). Confidence scores flag cases
where the segmented count diverges from the corpus expected count.

Pipeline:
    render_tablet_views.py → 512×512 (or larger) PNG screenshots
        ↓  this script
    data/glyphs/3d_crops/tablet_X/side_Y/  (per-glyph PNG crops)
    data/glyphs/3d_crops/tablet_X/manifest.json

Usage:
    python segment_3d_glyphs.py --tablet B --side r \\
        --renders data/glyphs/synthetic_views/tablet_B/ \\
        --corpus  data/corpus/B.json \\
        --output  data/glyphs/3d_crops/

    # Override crop size and segmentation sensitivity:
    python segment_3d_glyphs.py --tablet D --side a \\
        --renders data/glyphs/synthetic_views/tablet_D/ \\
        --corpus  data/corpus/D.json \\
        --output  data/glyphs/3d_crops/ \\
        --crop-size 96 --canny-low 20 --canny-high 60

Notes:
  • Render at >= 1024×1024 (--width 2048 --height 2048 in render_tablet_views.py)
    for reliable segmentation; 512px gives ~10 px/glyph, which is too coarse.
  • The best view (most face-on to the tablet surface) is selected automatically
    by maximising edge density in the central 60% of the image.
  • Only one face (side) is visible per render session. Re-run with the other
    --side after flipping the 3D model (elevation +180° in 3DHOP) for the verso.
"""

import argparse
import json
import logging
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

log = logging.getLogger(__name__)


# ── Tunable defaults ──────────────────────────────────────────────────────────

_CANNY_LOW  = 30    # lower Canny hysteresis threshold (carved glyphs are subtle)
_CANNY_HIGH = 80    # upper Canny hysteresis threshold
_MORPH_CLOSE_PX = 5  # close kernel — merges broken glyph stroke segments
_MORPH_OPEN_PX  = 2  # open kernel — removes isolated noise dots
_MIN_AREA_FRAC  = 0.00015  # minimum blob area as fraction of image area
_MAX_AREA_FRAC  = 0.04     # maximum blob area (filters out full-tablet edges)
_CROP_PADDING   = 0.20     # fractional padding around each glyph bounding box
_DEFAULT_CROP_SIZE = 64    # output crop size in pixels (matches GlyphPreprocessor)


# ── Best-view selection ───────────────────────────────────────────────────────

def _edge_density_central(png_path: Path, centre_frac: float = 0.60) -> float:
    """
    Return the Canny edge density in the central `centre_frac` window.
    Higher = more face-on view of the tablet surface.
    """
    img = cv2.imread(str(png_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return 0.0
    h, w = img.shape
    y0 = int(h * (1 - centre_frac) / 2)
    y1 = int(h * (1 + centre_frac) / 2)
    x0 = int(w * (1 - centre_frac) / 2)
    x1 = int(w * (1 + centre_frac) / 2)
    roi = img[y0:y1, x0:x1]
    edges = cv2.Canny(roi, _CANNY_LOW, _CANNY_HIGH)
    return float(edges.sum()) / (roi.size + 1)


def select_best_view(render_dir: Path) -> Path:
    """
    Pick the rendered PNG with the highest central edge density.
    Falls back to the first PNG if none found.
    """
    pngs = sorted(render_dir.glob("*.png"))
    if not pngs:
        raise FileNotFoundError(f"No PNG files found in {render_dir}")
    if len(pngs) == 1:
        return pngs[0]
    scored = [(p, _edge_density_central(p)) for p in pngs]
    best = max(scored, key=lambda x: x[1])
    log.info("Best view: %s (edge density %.4f)", best[0].name, best[1])
    return best[0]


# ── Image enhancement for incised carvings ───────────────────────────────────

def enhance_for_incisions(bgr: np.ndarray) -> np.ndarray:
    """
    Enhance local contrast on a BGR render to make carved incisions visible.

    Steps:
      1. Convert to LAB colour space
      2. Apply CLAHE on the L channel (amplifies local contrast → reveals carvings)
      3. Back to BGR, then to grayscale for Canny
    """
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l_eq = clahe.apply(l)
    lab_eq = cv2.merge([l_eq, a, b])
    bgr_eq = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)
    gray = cv2.cvtColor(bgr_eq, cv2.COLOR_BGR2GRAY)
    return gray


# ── Connected-component segmentation ─────────────────────────────────────────

def find_glyph_candidates(
    gray: np.ndarray,
    canny_low: int = _CANNY_LOW,
    canny_high: int = _CANNY_HIGH,
) -> list[dict]:
    """
    Run Canny + morphological close/open + connected-component analysis.
    Returns list of dicts with keys: x, y, w, h, area, cx, cy.
    Filtered to plausible glyph sizes; not yet sorted.
    """
    h, w = gray.shape
    img_area = h * w
    min_area = _MIN_AREA_FRAC * img_area
    max_area = _MAX_AREA_FRAC * img_area

    edges = cv2.Canny(gray, canny_low, canny_high)

    close_k = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (_MORPH_CLOSE_PX, _MORPH_CLOSE_PX)
    )
    open_k = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (_MORPH_OPEN_PX, _MORPH_OPEN_PX)
    )
    mask = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, close_k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_k)

    n_labels, _, stats, centroids = cv2.connectedComponentsWithStats(
        mask, connectivity=8
    )

    candidates: list[dict] = []
    for i in range(1, n_labels):  # skip background label 0
        x, y, bw, bh, area = (
            stats[i, cv2.CC_STAT_LEFT],
            stats[i, cv2.CC_STAT_TOP],
            stats[i, cv2.CC_STAT_WIDTH],
            stats[i, cv2.CC_STAT_HEIGHT],
            stats[i, cv2.CC_STAT_AREA],
        )
        if not (min_area <= area <= max_area):
            continue
        aspect = max(bw, bh) / max(min(bw, bh), 1)
        if aspect > 8:  # very thin horizontal/vertical lines — tablet edges, not glyphs
            continue
        candidates.append(
            dict(
                x=int(x), y=int(y), w=int(bw), h=int(bh),
                area=int(area),
                cx=float(centroids[i, 0]),
                cy=float(centroids[i, 1]),
            )
        )
    return candidates


# ── Boustrophedon ordering ────────────────────────────────────────────────────

def cluster_into_lines(candidates: list[dict], n_lines: int) -> list[list[dict]]:
    """
    Cluster glyph candidates into `n_lines` horizontal bands using 1-D k-means
    on the centroid y-coordinate. Returns a list of lists, one per line,
    sorted top-to-bottom by median y.
    """
    if not candidates:
        return []

    ys = np.array([[c["cy"]] for c in candidates], dtype=np.float32)
    n_clusters = min(n_lines, len(candidates))

    _, labels, _ = cv2.kmeans(
        ys,
        n_clusters,
        None,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2),
        attempts=10,
        flags=cv2.KMEANS_PP_CENTERS,
    )

    line_map: dict[int, list[dict]] = defaultdict(list)
    for glyph, label in zip(candidates, labels.flatten()):
        line_map[int(label)].append(glyph)

    # Sort clusters top-to-bottom by their median y
    sorted_lines = sorted(line_map.values(), key=lambda gs: np.median([g["cy"] for g in gs]))
    return sorted_lines


def sort_boustrophedon(candidates: list[dict], n_lines: int) -> list[dict]:
    """
    Sort candidates into boustrophedon (alternating L→R / R→L) reading order.

    Line 0 (topmost): left → right
    Line 1: right → left
    Line 2: left → right
    ...etc.
    """
    lines = cluster_into_lines(candidates, n_lines)
    ordered: list[dict] = []
    for line_idx, line in enumerate(lines):
        reverse = (line_idx % 2 == 1)
        sorted_glyphs = sorted(line, key=lambda g: g["cx"], reverse=reverse)
        for pos_in_line, g in enumerate(sorted_glyphs):
            g = dict(g, line_idx=line_idx, pos_in_line=pos_in_line, reverse=reverse)
            ordered.append(g)
    return ordered


# ── Corpus glyph ID assignment ────────────────────────────────────────────────

def load_corpus_glyphs(corpus_json: Path, side: str) -> list[dict]:
    """
    Load glyphs for one side from a corpus JSON file.
    Returns list sorted by (line, glyph_num).
    """
    data = json.loads(corpus_json.read_text(encoding="utf-8"))
    glyphs = data.get("glyphs", [])
    side_glyphs = [g for g in glyphs if g.get("side") == side]
    # Sort by line (numerically) then glyph_num
    side_glyphs.sort(key=lambda g: (int(g["line"]), int(g["glyph_num"])))
    return side_glyphs


def assign_corpus_ids(
    ordered_candidates: list[dict],
    corpus_glyphs: list[dict],
) -> list[dict]:
    """
    Assign corpus glyph IDs to segmented candidates by positional order.

    Positional assignment is the only option without 3D→pixel projection data.
    Confidence is set to 'high' when segmented count matches corpus count for
    that line; 'low' when there's a mismatch.
    """
    # Group corpus glyphs by line
    corpus_by_line: dict[str, list[dict]] = defaultdict(list)
    for g in corpus_glyphs:
        corpus_by_line[g["line"]].append(g)
    corpus_lines = sorted(corpus_by_line.keys(), key=int)

    # Group segmented candidates by line_idx
    seg_by_line: dict[int, list[dict]] = defaultdict(list)
    for c in ordered_candidates:
        seg_by_line[c["line_idx"]].append(c)
    seg_lines = sorted(seg_by_line.keys())

    assigned: list[dict] = []
    for seg_line_idx, corpus_line_key in zip(seg_lines, corpus_lines):
        seg_glyphs   = seg_by_line[seg_line_idx]
        corp_glyphs  = corpus_by_line[corpus_line_key]
        count_match  = len(seg_glyphs) == len(corp_glyphs)
        confidence   = "high" if count_match else "low"

        for pos, seg in enumerate(seg_glyphs):
            corp = corp_glyphs[pos] if pos < len(corp_glyphs) else None
            entry: dict[str, Any] = dict(seg)
            if corp:
                entry["glyph_id"]     = corp.get("position")
                entry["barthel_code"] = corp.get("barthel_code")
                entry["horley_code"]  = corp.get("horley_code")
                entry["corpus_line"]  = corpus_line_key
                entry["corpus_glyph_num"] = corp.get("glyph_num")
            else:
                entry["glyph_id"]     = None
                entry["barthel_code"] = None
                entry["horley_code"]  = None
                entry["corpus_line"]  = corpus_line_key
                entry["corpus_glyph_num"] = None
            entry["id_confidence"] = confidence
            assigned.append(entry)

    # Any candidates beyond the corpus lines get no ID
    for extra_line_idx in seg_lines[len(corpus_lines):]:
        for seg in seg_by_line[extra_line_idx]:
            entry = dict(seg, glyph_id=None, barthel_code=None,
                         horley_code=None, corpus_line=None,
                         corpus_glyph_num=None, id_confidence="none")
            assigned.append(entry)

    return assigned


# ── Crop extraction ───────────────────────────────────────────────────────────

def extract_crops(
    bgr: np.ndarray,
    assigned: list[dict],
    output_dir: Path,
    crop_size: int = _DEFAULT_CROP_SIZE,
) -> list[dict]:
    """
    Crop each glyph from the BGR image with padding, resize to crop_size × crop_size,
    save as PNG, and return updated manifest entries.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    h_img, w_img = bgr.shape[:2]
    manifest_entries: list[dict] = []

    for i, entry in enumerate(assigned):
        bx, by, bw, bh = entry["x"], entry["y"], entry["w"], entry["h"]
        pad_x = int(bw * _CROP_PADDING)
        pad_y = int(bh * _CROP_PADDING)
        x0 = max(0, bx - pad_x)
        y0 = max(0, by - pad_y)
        x1 = min(w_img, bx + bw + pad_x)
        y1 = min(h_img, by + bh + pad_y)

        crop_bgr = bgr[y0:y1, x0:x1]
        if crop_bgr.size == 0:
            continue

        # Resize to square output size
        crop_resized = cv2.resize(
            crop_bgr, (crop_size, crop_size), interpolation=cv2.INTER_AREA
        )

        # Filename: line_pos_barthel.png, or just sequential index if no ID
        barthel = entry.get("barthel_code") or "unk"
        line_key = entry.get("corpus_line") or f"seg{entry.get('line_idx', 0):02d}"
        pos = entry.get("corpus_glyph_num") or f"{entry.get('pos_in_line', i):03d}"
        fname = f"L{line_key}_G{pos}_{barthel}.png"
        out_path = output_dir / fname

        cv2.imwrite(str(out_path), crop_resized)

        manifest_entries.append(
            {
                "file": fname,
                "glyph_id":          entry.get("glyph_id"),
                "barthel_code":      entry.get("barthel_code"),
                "horley_code":       entry.get("horley_code"),
                "corpus_line":       entry.get("corpus_line"),
                "corpus_glyph_num":  entry.get("corpus_glyph_num"),
                "id_confidence":     entry.get("id_confidence", "none"),
                "bbox_original":     [bx, by, bw, bh],
                "crop_size":         crop_size,
                "line_idx":          entry.get("line_idx"),
                "pos_in_line":       entry.get("pos_in_line"),
            }
        )

    return manifest_entries


# ── Diagnostics ───────────────────────────────────────────────────────────────

def _count_per_line(assigned: list[dict]) -> dict[int, int]:
    counts: dict[int, int] = defaultdict(int)
    for e in assigned:
        counts[e.get("line_idx", -1)] += 1
    return dict(counts)


def print_summary(
    tablet: str,
    side: str,
    best_view: Path,
    n_found: int,
    n_corpus: int,
    n_high_conf: int,
    output_dir: Path,
) -> None:
    sep = "─" * 60
    print(f"\n{sep}")
    print(f"Tablet {tablet}  side={side}  source: {best_view.name}")
    print(f"  Segmented: {n_found} candidates  |  Corpus expected: {n_corpus}")
    match_pct = 100 * n_found / max(n_corpus, 1)
    print(f"  Match rate: {match_pct:.0f}%  |  High-confidence IDs: {n_high_conf}/{n_found}")
    if match_pct < 50:
        print("  WARNING: < 50% match — check render resolution or --canny-* thresholds")
    elif match_pct < 80:
        print("  NOTE: < 80% match — some glyphs may be missed or merged")
    print(f"  Crops → {output_dir}")
    print(sep)


# ── Main ──────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Segment glyph instances from rendered 3D tablet views"
    )
    p.add_argument("--tablet",   required=True, help="Tablet ID (B, C, D, …)")
    p.add_argument(
        "--side", required=True,
        help="Tablet side to segment: 'r'/'v' (recto/verso) or 'a'/'b'"
    )
    p.add_argument(
        "--renders", type=Path, required=True,
        help="Directory of PNG screenshots from render_tablet_views.py"
    )
    p.add_argument(
        "--corpus", type=Path, default=None,
        help="Corpus JSON for this tablet (enables ID assignment). "
             "Defaults to data/corpus/<TABLET>.json"
    )
    p.add_argument(
        "--output", type=Path, default=Path("data/glyphs/3d_crops"),
        help="Root output directory (default: data/glyphs/3d_crops)"
    )
    p.add_argument(
        "--crop-size", type=int, default=_DEFAULT_CROP_SIZE,
        help=f"Output crop size in pixels (default: {_DEFAULT_CROP_SIZE})"
    )
    p.add_argument(
        "--n-lines", type=int, default=None,
        help="Expected number of text lines on this side. "
             "Inferred from corpus if omitted."
    )
    p.add_argument("--canny-low",  type=int, default=_CANNY_LOW)
    p.add_argument("--canny-high", type=int, default=_CANNY_HIGH)
    p.add_argument(
        "--best-view", type=Path, default=None,
        help="Use this specific PNG instead of auto-selecting the best view"
    )
    p.add_argument(
        "--roi", type=str, default=None,
        help="Restrict segmentation to a sub-region: 'x0,y0,x1,y1' in pixels. "
             "Useful to exclude viewer chrome (e.g. INSCRIBE sidebar/header). "
             "Example: --roi 65,70,480,455"
    )
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    # ── Resolve corpus path ──────────────────────────────────────────────────
    corpus_path = args.corpus
    if corpus_path is None:
        corpus_path = Path(f"data/corpus/{args.tablet}.json")
    if not corpus_path.exists():
        log.warning("Corpus file not found: %s — ID assignment disabled", corpus_path)
        corpus_path = None

    # ── Load corpus glyphs for this side ────────────────────────────────────
    corpus_glyphs: list[dict] = []
    n_lines_expected = args.n_lines
    if corpus_path:
        corpus_glyphs = load_corpus_glyphs(corpus_path, args.side)
        if not corpus_glyphs:
            log.warning(
                "No glyphs found for side=%r in %s", args.side, corpus_path
            )
        else:
            from collections import Counter
            line_counts = Counter(g["line"] for g in corpus_glyphs)
            if n_lines_expected is None:
                n_lines_expected = len(line_counts)
            log.info(
                "Corpus: %d glyphs across %d lines (side=%s)",
                len(corpus_glyphs), n_lines_expected, args.side
            )

    if n_lines_expected is None:
        n_lines_expected = 10  # safe fallback
        log.warning("No corpus — assuming %d lines", n_lines_expected)

    # ── Select best view ─────────────────────────────────────────────────────
    render_dir: Path = args.renders
    if args.best_view:
        best_view = args.best_view
        if not best_view.exists():
            raise FileNotFoundError(f"--best-view path not found: {best_view}")
    else:
        best_view = select_best_view(render_dir)

    # ── Load and enhance image ───────────────────────────────────────────────
    bgr = cv2.imread(str(best_view))
    if bgr is None:
        raise IOError(f"Could not read image: {best_view}")
    log.info("Image: %s  size=%dx%d", best_view.name, bgr.shape[1], bgr.shape[0])

    if bgr.shape[0] < 512 or bgr.shape[1] < 512:
        log.warning(
            "Image is only %dx%d — render at >= 1024×1024 for reliable segmentation",
            bgr.shape[1], bgr.shape[0],
        )

    # ── Apply ROI crop to exclude viewer chrome ────────────────────────────
    if args.roi:
        try:
            rx0, ry0, rx1, ry1 = [int(v) for v in args.roi.split(",")]
            bgr = bgr[ry0:ry1, rx0:rx1]
            log.info("ROI crop applied: (%d,%d)→(%d,%d)  result=%dx%d",
                     rx0, ry0, rx1, ry1, bgr.shape[1], bgr.shape[0])
        except ValueError:
            log.warning("Invalid --roi format %r — expected x0,y0,x1,y1; skipping", args.roi)

    gray = enhance_for_incisions(bgr)

    # ── Segment ──────────────────────────────────────────────────────────────
    candidates = find_glyph_candidates(gray, args.canny_low, args.canny_high)
    log.info("Raw candidates: %d", len(candidates))

    if not candidates:
        print("No glyph candidates found. Try lowering --canny-low or "
              "using a higher-resolution render.")
        return

    ordered = sort_boustrophedon(candidates, n_lines_expected)
    log.debug("After boustrophedon sort: %d candidates", len(ordered))

    # ── Assign corpus IDs ────────────────────────────────────────────────────
    if corpus_glyphs:
        assigned = assign_corpus_ids(ordered, corpus_glyphs)
    else:
        assigned = [
            dict(e, glyph_id=None, barthel_code=None, horley_code=None,
                 corpus_line=None, corpus_glyph_num=None, id_confidence="none")
            for e in ordered
        ]

    # ── Extract and save crops ───────────────────────────────────────────────
    out_dir = args.output / f"tablet_{args.tablet}" / f"side_{args.side}"
    manifest_entries = extract_crops(bgr, assigned, out_dir, args.crop_size)

    # ── Write manifest ───────────────────────────────────────────────────────
    manifest: dict[str, Any] = {
        "tablet":        args.tablet,
        "side":          args.side,
        "source_view":   best_view.name,
        "render_dir":    str(render_dir),
        "crop_size":     args.crop_size,
        "n_segmented":   len(manifest_entries),
        "n_corpus":      len(corpus_glyphs),
        "canny_low":     args.canny_low,
        "canny_high":    args.canny_high,
        "glyphs":        manifest_entries,
    }
    manifest_path = args.output / f"tablet_{args.tablet}" / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    existing: dict = {}
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
    existing[args.side] = manifest
    manifest_path.write_text(json.dumps(existing, indent=2), encoding="utf-8")

    # ── Summary ──────────────────────────────────────────────────────────────
    n_high = sum(1 for e in manifest_entries if e["id_confidence"] == "high")
    print_summary(
        tablet=args.tablet,
        side=args.side,
        best_view=best_view,
        n_found=len(manifest_entries),
        n_corpus=len(corpus_glyphs),
        n_high_conf=n_high,
        output_dir=out_dir,
    )
    print(f"Manifest → {manifest_path}")


if __name__ == "__main__":
    main()
