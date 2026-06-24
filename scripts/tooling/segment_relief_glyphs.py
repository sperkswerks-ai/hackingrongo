#!/usr/bin/env python3
"""
segment_relief_glyphs.py — corpus-GUIDED segmentation of high-def 3D relief maps
into per-glyph crops, labelled by corpus position.

Replaces the old web-viewer path (segment_3d_glyphs.py: 128px cap, ~33% under-
segmentation, unlabelled). Key idea: the corpus JSON already tells us, for each
tablet side, how many glyphs are on each line and in what order. We use that
COUNT as ground truth to drive the split — so we stop guessing how many glyphs a
line has and stop under-segmenting.

Pipeline per relief image (one tablet face):
  1. Foreground "glyph-mass" map: |relief - mid| blurred + thresholded.
  2. Horizontal projection → line bands; reconcile band count to the corpus's
     number of lines for that side.
  3. Per band: vertical projection valleys → glyph cuts; reconcile the cut count
     to the corpus glyph count for that line (merge shallowest / split widest to
     hit the target N).
  4. Crop each glyph at NATIVE resolution (padded), label with the corpus code
     in position order, write a crop + a row in the manifest.
  5. Write a DEBUG OVERLAY (boxes + assigned codes drawn on the relief) so we can
     visually audit alignment and tune — this first run is calibration.

NOTE: not testable on the dev Mac (relief PNGs + libs live on Azure). Run on one
face first, eyeball the *_debug.png overlay, and we tune from what it shows.

Requires: numpy, pillow, scipy (optional, falls back), opencv (optional).
Usage:
    python scripts/tooling/segment_relief_glyphs.py \
        --relief data/glyphs/highres_views/tablet_c_mamari/tablet_c_mamari_recto_relief.png \
        --tablet C --side-letters a,r \
        --out-dir data/glyphs/glyph_crops/C_recto
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

REPO = Path(__file__).resolve().parents[2]


def corpus_lines(tablet: str, side_letters: list[str]) -> dict[str, list[dict]]:
    """Return {line_id: [glyph dicts in position order]} for the requested side(s)."""
    data = json.loads((REPO / "data" / "corpus" / f"{tablet}.json").read_text())
    by_line: dict[str, list[dict]] = {}
    for g in data.get("glyphs", []):
        if str(g.get("side", "")).lower() not in side_letters:
            continue
        by_line.setdefault(str(g.get("line", "")), []).append(g)
    for k in by_line:
        by_line[k].sort(key=lambda g: int(g.get("position", 0)))
    # order lines by their first glyph's position
    return dict(sorted(by_line.items(), key=lambda kv: int(kv[1][0].get("position", 0))))


def _blur(a: np.ndarray, sigma: float) -> np.ndarray:
    try:
        from scipy.ndimage import gaussian_filter
        return gaussian_filter(a, sigma)
    except Exception:
        im = Image.fromarray((a * 255).clip(0, 255).astype("uint8"), "L")
        from PIL import ImageFilter
        return np.asarray(im.filter(ImageFilter.GaussianBlur(sigma)), np.float32) / 255.0


def foreground_mass(gray: np.ndarray, sigma: float) -> np.ndarray:
    """Glyph-mass map: deviation of relief from its surface mid-tone, blurred."""
    g = gray.astype(np.float32) / 255.0
    med = float(np.median(g[g > 0])) if (g > 0).any() else 0.5
    mass = np.abs(g - med)
    mass[gray == 0] = 0.0                       # ignore background
    mass = _blur(mass, sigma)
    if mass.max() > 0:
        mass /= mass.max()
    return mass


def split_to_count(profile: np.ndarray, target: int, lo_frac: float = 0.15) -> list[tuple[int, int]]:
    """Split a 1-D mass profile into `target` segments using the deepest valleys.
    Reconciles detected structure to the known count: find candidate valleys, keep
    the `target-1` deepest, return contiguous [start,end) spans."""
    n = len(profile)
    if target <= 1 or n == 0:
        return [(0, n)]
    thr = profile.max() * lo_frac
    # candidate cut points = local minima below threshold
    valleys = [i for i in range(1, n - 1)
               if profile[i] <= thr and profile[i] <= profile[i - 1] and profile[i] <= profile[i + 1]]
    if len(valleys) < target - 1:
        # not enough real valleys → fall back to even spacing (honest: low confidence)
        cuts = [round(n * k / target) for k in range(1, target)]
    else:
        valleys.sort(key=lambda i: profile[i])           # shallowest..deepest
        cuts = sorted(valleys[:target - 1])
    bounds = [0] + cuts + [n]
    return [(bounds[i], bounds[i + 1]) for i in range(len(bounds) - 1)]


def segment(relief_path: Path, tablet: str, side_letters: list[str], out_dir: Path,
            pad: int = 6, mass_sigma_frac: float = 0.004):
    gray = np.asarray(Image.open(relief_path).convert("L"))
    H, W = gray.shape
    lines = corpus_lines(tablet, side_letters)
    if not lines:
        raise SystemExit(f"No corpus glyphs for tablet {tablet} side in {side_letters}")
    out_dir.mkdir(parents=True, exist_ok=True)
    mass = foreground_mass(gray, sigma=max(1.0, mass_sigma_frac * W))

    # 1) line bands via horizontal projection, reconciled to corpus line count
    row_profile = mass.sum(axis=1)
    bands = split_to_count(row_profile, target=len(lines))
    print(f"  image {W}×{H} · corpus lines={len(lines)} · detected bands={len(bands)}")

    overlay = Image.open(relief_path).convert("RGB")
    draw = ImageDraw.Draw(overlay)
    manifest = []
    n_crops = 0
    for (line_id, glyphs), (y0, y1) in zip(lines.items(), bands):
        band = mass[y0:y1, :]
        col_profile = band.sum(axis=0)
        spans = split_to_count(col_profile, target=len(glyphs))
        for g, (x0, x1) in zip(glyphs, spans):
            cx0, cy0 = max(0, x0 - pad), max(0, y0 - pad)
            cx1, cy1 = min(W, x1 + pad), min(H, y1 + pad)
            code = str(g.get("barthel_code", "?"))
            pos = int(g.get("position", 0))
            crop = Image.fromarray(gray[cy0:cy1, cx0:cx1]).convert("L")
            safe = code.replace(":", "-").replace("/", "-").replace("?", "Q")
            fname = f"L{line_id}_P{pos:04d}_{safe}.png"
            crop.save(out_dir / fname)
            draw.rectangle([cx0, cy0, cx1, cy1], outline=(0, 200, 0), width=2)
            draw.text((cx0 + 1, cy0 + 1), code, fill=(255, 80, 80))
            manifest.append({"tablet": tablet, "line": line_id, "position": pos,
                             "barthel_code": code, "bbox": [cx0, cy0, cx1, cy1],
                             "file": fname, "uncertain": bool(g.get("uncertain"))})
            n_crops += 1

    overlay.save(out_dir / "_debug_overlay.png")
    (out_dir / "manifest.json").write_text(json.dumps(
        {"relief": str(relief_path), "tablet": tablet, "n_corpus_glyphs": sum(len(v) for v in lines.values()),
         "n_crops": n_crops, "crops": manifest}, indent=1))
    print(f"  ✓ {n_crops} crops (corpus expected {sum(len(v) for v in lines.values())}) → {out_dir}")
    print(f"  → audit {out_dir/'_debug_overlay.png'} : boxes should sit on glyphs, red codes in reading order")


def main() -> None:
    ap = argparse.ArgumentParser(description="Corpus-guided segmentation of 3D relief maps into labelled glyph crops.")
    ap.add_argument("--relief", type=Path, required=True, help="a *_relief.png from render_mesh_highres.py")
    ap.add_argument("--tablet", required=True, help="tablet letter, e.g. C")
    ap.add_argument("--side-letters", default="a,r", help="corpus 'side' values for THIS face (e.g. a,r for recto)")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--pad", type=int, default=6)
    args = ap.parse_args()
    segment(args.relief, args.tablet.upper(),
            [s.strip().lower() for s in args.side_letters.split(",")], args.out_dir, args.pad)


if __name__ == "__main__":
    main()
