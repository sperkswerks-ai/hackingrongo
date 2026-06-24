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


def _load_font(size: int):
    """A readable TrueType font for overlay labels (PIL's default is ~10px,
    invisible on a 4096px image). Falls back to the bitmap default."""
    from PIL import ImageFont
    for name in ("DejaVuSans-Bold.ttf", "DejaVuSans.ttf",
                 "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                 "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except Exception:
            continue
    return ImageFont.load_default()


def tablet_bbox(gray: np.ndarray, bg_thresh: int = 8) -> tuple[int, int, int, int]:
    """Bounding box of the non-background (tablet) region: (x0, y0, x1, y1)."""
    nonbg = gray > bg_thresh
    ys, xs = np.where(nonbg)
    if len(xs) == 0:
        return 0, 0, gray.shape[1], gray.shape[0]
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def foreground_mass(gray: np.ndarray, sigma: float, edge_trim: int = 0) -> np.ndarray:
    """Glyph-mass map from EDGE ENERGY (gradient magnitude), blurred.

    Brightness-deviation fails on this relief because it's blown out to a bimodal
    0/255 image (median 255), so 'deviation from mid-tone' highlights the wrong
    pixels. Edge energy is polarity- and saturation-independent: glyph STROKES
    have many edges; the smooth surface (white, grey, or black) has none. This is
    what actually localises carvings across the 14 rows.

    `edge_trim` erodes the tablet rim so its huge step edge doesn't dominate.
    """
    g = gray.astype(np.float32) / 255.0
    fg = gray > 8
    gx = np.zeros_like(g); gy = np.zeros_like(g)
    gx[:, 1:] = np.abs(np.diff(g, axis=1))
    gy[1:, :] = np.abs(np.diff(g, axis=0))
    mass = np.hypot(gx, gy)
    mass[~fg] = 0.0
    if edge_trim > 0:
        # Suppress only the tablet RIM, which (after bbox-crop) sits at the image
        # borders. Do NOT erode the fg mask — fg is sparse glyph strokes, and
        # eroding it deletes the carvings (the bug that left a single stripe).
        mass[:edge_trim, :] = 0.0
        mass[-edge_trim:, :] = 0.0
        mass[:, :edge_trim] = 0.0
        mass[:, -edge_trim:] = 0.0
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
    min_w = max(1, int((n / target) * 0.5))     # cuts ≥ half a band-width apart
    valleys = [i for i in range(1, n - 1)
               if profile[i] <= profile[i - 1] and profile[i] <= profile[i + 1]]
    valleys.sort(key=lambda i: profile[i])      # deepest (lowest) first
    chosen: list[int] = []
    for v in valleys:
        if v < min_w or v > n - min_w:
            continue
        if all(abs(v - c) >= min_w for c in chosen):
            chosen.append(v)
        if len(chosen) == target - 1:
            break
    if len(chosen) < target - 1:                # not enough separated valleys → even spacing
        cuts = [round(n * k / target) for k in range(1, target)]
    else:
        cuts = sorted(chosen)
    bounds = [0] + cuts + [n]
    return [(bounds[i], bounds[i + 1]) for i in range(len(bounds) - 1)]


def segment(relief_path: Path, tablet: str, side_letters: list[str], out_dir: Path,
            pad: int = 6, mass_sigma_frac: float = 0.004, edge_trim_frac: float = 0.02,
            diagnose: bool = False):
    gray = np.asarray(Image.open(relief_path).convert("L"))
    H, W = gray.shape
    lines = corpus_lines(tablet, side_letters)
    if not lines:
        raise SystemExit(f"No corpus glyphs for tablet {tablet} side in {side_letters}")
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- DIAGNOSTICS: characterise the image numerically (so we tune from data) ---
    tx0, ty0, tx1, ty1 = tablet_bbox(gray)
    nz = gray[gray > 8]
    print("── DIAGNOSTICS ──────────────────────────────────────────")
    print(f"  image: {W}×{H}  dtype={gray.dtype}")
    print(f"  intensity: min={gray.min()} max={gray.max()} mean={gray.mean():.1f}")
    print(f"  pixels==0 (black bg): {100*np.mean(gray==0):.1f}%  | non-bg: {100*np.mean(gray>8):.1f}%")
    if nz.size:
        pcts = np.percentile(nz, [5, 25, 50, 75, 95])
        print(f"  non-bg intensity pctiles [5/25/50/75/95]: {[int(p) for p in pcts]}")
        print(f"  non-bg std (glyph-signal strength): {nz.std():.1f}")
    print(f"  TABLET bbox: x[{tx0}:{tx1}] y[{ty0}:{ty1}]  = {tx1-tx0}×{ty1-ty0}  "
          f"({100*(tx1-tx0)*(ty1-ty0)/(W*H):.0f}% of image)")

    # --- crop to the tablet, work in tablet-local coords ---
    g_t = gray[ty0:ty1, tx0:tx1]
    Wt, Ht = g_t.shape[1], g_t.shape[0]
    edge_trim = max(1, int(edge_trim_frac * min(Wt, Ht)))
    mass = foreground_mass(g_t, sigma=max(1.0, mass_sigma_frac * Wt), edge_trim=edge_trim)

    row_profile = mass.sum(axis=1)
    # --- decisive: is row-mass SPREAD across rows, or jammed in one stripe? ---
    rp = row_profile
    if rp.sum() > 0:
        binned = [rp[i * Ht // 14:(i + 1) * Ht // 14].sum() for i in range(14)]
        pct = [round(100 * b / rp.sum()) for b in binned]
        cum = np.cumsum(rp) / rp.sum()
        y_lo, y_hi = int(np.searchsorted(cum, 0.05)), int(np.searchsorted(cum, 0.95))
        print(f"  row-mass by 14ths (%): {pct}")
        print(f"  central 90% of row-mass spans y_local [{y_lo}:{y_hi}] of {Ht}  "
              f"({100*(y_hi-y_lo)/Ht:.0f}% of tablet height)")
    bands = split_to_count(row_profile, target=len(lines))
    top_rows = np.argsort(row_profile)[-5:][::-1]
    print(f"  edge_trim={edge_trim}px · row-mass peaks at y(local)={sorted(int(r) for r in top_rows)}")
    print(f"  corpus lines={len(lines)} · bands={len(bands)} · band heights={[b[1]-b[0] for b in bands]}")
    print("─────────────────────────────────────────────────────────")
    if diagnose:
        # save a heat preview of the mass map + exit (no crops)
        Image.fromarray((mass * 255).astype("uint8")).save(out_dir / "_mass_preview.png")
        print(f"  [diagnose] wrote {out_dir/'_mass_preview.png'} — no crops written")
        return

    overlay = Image.open(relief_path).convert("RGB")
    draw = ImageDraw.Draw(overlay)
    font = _load_font(30)
    manifest = []
    n_crops = 0
    for (line_id, glyphs), (y0, y1) in zip(lines.items(), bands):
        band = mass[y0:y1, :]
        col_profile = band.sum(axis=0)
        spans = split_to_count(col_profile, target=len(glyphs))
        for g, (x0, x1) in zip(glyphs, spans):
            # tighten the grid cell to the actual glyph content (bright pixels),
            # so boxes hug glyphs instead of forming a rigid rectangle and stop
            # swallowing edge-damage noise.
            cell = g_t[y0:y1, x0:x1]
            ys2, xs2 = np.where(cell > 8)
            if xs2.size > 20:
                ax0, ay0 = x0 + int(xs2.min()), y0 + int(ys2.min())
                ax1, ay1 = x0 + int(xs2.max()) + 1, y0 + int(ys2.max()) + 1
            else:
                ax0, ay0, ax1, ay1 = x0, y0, x1, y1     # empty cell → keep grid box
            # offset tablet-local coords back to full image
            cx0, cy0 = max(0, tx0 + ax0 - pad), max(0, ty0 + ay0 - pad)
            cx1, cy1 = min(W, tx0 + ax1 + pad), min(H, ty0 + ay1 + pad)
            code = str(g.get("barthel_code", "?"))
            pos = int(g.get("position", 0))
            crop = Image.fromarray(gray[cy0:cy1, cx0:cx1]).convert("L")
            safe = code.replace(":", "-").replace("/", "-").replace("?", "Q")
            fname = f"L{line_id}_P{pos:04d}_{safe}.png"
            crop.save(out_dir / fname)
            draw.rectangle([cx0, cy0, cx1, cy1], outline=(0, 220, 0), width=3)
            draw.text((cx0 + 2, max(0, cy0 - 32)), code, fill=(255, 60, 60), font=font)
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
    ap.add_argument("--edge-trim-frac", type=float, default=0.02,
                    help="fraction of tablet size to erode inward, to suppress the bright rim")
    ap.add_argument("--diagnose", action="store_true",
                    help="print image stats + write _mass_preview.png, write NO crops")
    args = ap.parse_args()
    segment(args.relief, args.tablet.upper(),
            [s.strip().lower() for s in args.side_letters.split(",")], args.out_dir, args.pad,
            edge_trim_frac=args.edge_trim_frac, diagnose=args.diagnose)


if __name__ == "__main__":
    main()
