"""
hackingrongo.results.reconstruction_report
==========================================

Generates an HTML gallery report from a directory of glyph reconstruction
outputs produced by ``scripts/reconstruct_glyph.py``.

Each entry shows the reconstruction strip (original | masked | decoded | error,
optionally KNN columns) alongside MSE / SSIM metrics for the full image and
masked region.  A summary table at the top ranks entries by masked-region SSIM.

Inputs
------
    outputs/reconstruction/     — directory of *_metrics.json + *_reconstruction.png
                                  files from reconstruct_glyph.py

Output
------
    outputs/reconstruction/reconstruction_report.html

CLI
---
    python -m hackingrongo.results.reconstruction_report \\
        --input  outputs/reconstruction/ \\
        --output outputs/reconstruction/reconstruction_report.html
"""

from __future__ import annotations

import argparse
import base64
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------

_CSS = """
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
    background: #0f0f14;
    color: #d4cfc9;
    font-family: 'Segoe UI', system-ui, sans-serif;
    font-size: 14px;
    line-height: 1.5;
}
header {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    border-bottom: 1px solid #2a2a4a;
    padding: 24px 32px;
}
header h1 { font-size: 1.6rem; color: #e8e0d8; font-weight: 300; letter-spacing: 0.05em; }
header p  { color: #888; margin-top: 4px; font-size: 0.85rem; }
nav { padding: 12px 32px; background: #14141c; border-bottom: 1px solid #1e1e30; }
nav a { color: #7a9ccf; text-decoration: none; margin-right: 20px; font-size: 0.85rem; }
nav a:hover { color: #aac4f0; }
.container { max-width: 1400px; margin: 0 auto; padding: 24px 32px; }
h2 { color: #c8c0b8; font-size: 1.1rem; font-weight: 400;
     border-bottom: 1px solid #2a2a40; padding-bottom: 8px; margin-bottom: 16px; }
/* ── summary table ── */
.summary-table { width: 100%; border-collapse: collapse; margin-bottom: 40px; font-size: 0.82rem; }
.summary-table th { background: #1a1a30; color: #9090b0; font-weight: 500;
                    padding: 8px 12px; text-align: left; border-bottom: 1px solid #2a2a44; }
.summary-table td { padding: 7px 12px; border-bottom: 1px solid #1c1c28; }
.summary-table tr:hover td { background: #18182a; }
.metric-good  { color: #6abf69; }
.metric-mid   { color: #e6c96e; }
.metric-bad   { color: #e07070; }
/* ── entry cards ── */
.entry { background: #14141e; border: 1px solid #222236;
         border-radius: 6px; margin-bottom: 28px; overflow: hidden; }
.entry-header { background: #18182c; padding: 12px 18px;
                border-bottom: 1px solid #222236; display: flex; align-items: baseline;
                gap: 16px; }
.entry-title  { font-size: 1rem; color: #c8c0b8; }
.entry-meta   { font-size: 0.78rem; color: #666; }
.strip-wrap   { padding: 16px 18px; }
.strip-img    { max-width: 100%; height: auto; display: block; image-rendering: pixelated; }
.metrics-row  { display: flex; flex-wrap: wrap; gap: 20px;
                padding: 12px 18px 16px; border-top: 1px solid #1e1e30; }
.metric-block { display: flex; flex-direction: column; }
.metric-label { font-size: 0.72rem; color: #666; text-transform: uppercase; letter-spacing: 0.06em; }
.metric-value { font-size: 1rem; font-weight: 500; }
.knn-neighbors { padding: 8px 18px 14px; font-size: 0.78rem; color: #666; }
.knn-neighbors span { color: #9aaccc; margin-right: 8px; }
footer { text-align: center; padding: 32px; color: #444; font-size: 0.8rem;
         border-top: 1px solid #1e1e30; }
"""

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ssim_colour(v: float | None) -> str:
    if v is None:
        return ""
    if v >= 0.80:
        return "metric-good"
    if v >= 0.55:
        return "metric-mid"
    return "metric-bad"


def _mse_colour(v: float | None) -> str:
    if v is None:
        return ""
    if v <= 0.02:
        return "metric-good"
    if v <= 0.08:
        return "metric-mid"
    return "metric-bad"


def _fmt(v: float | None, decimals: int = 4) -> str:
    return f"{v:.{decimals}f}" if v is not None else "—"


def _embed_image(path: Path) -> str:
    """Return a data-URI string for inline embedding."""
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{data}"


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------

def build_reconstruction_report(metrics_dir: Path) -> str:
    """Read all *_metrics.json files in *metrics_dir* and return HTML string."""
    entries = []
    for mpath in sorted(metrics_dir.glob("*_metrics.json")):
        try:
            m = json.loads(mpath.read_text(encoding="utf-8"))
        except Exception:
            logger.warning("Skipping unreadable metrics file: %s", mpath)
            continue

        strip_path = mpath.with_name(mpath.stem.replace("_metrics", "_reconstruction") + ".png")
        if not strip_path.exists():
            logger.warning("Strip image not found, skipping: %s", strip_path)
            continue

        entries.append((mpath.stem.replace("_metrics", ""), m, strip_path))

    if not entries:
        raise ValueError(f"No valid entries found in {metrics_dir}")

    # Sort by masked-region SSIM descending (fall back to full-image SSIM).
    def _sort_key(e):
        m = e[1]
        return -(m.get("ssim_masked") or m.get("ssim_full") or 0.0)

    entries.sort(key=_sort_key)

    generated = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    # ── summary table ────────────────────────────────────────────────────────
    rows = []
    for prefix, m, _strip in entries:
        ssim_m = m.get("ssim_masked")
        ssim_f = m.get("ssim_full")
        mse_m  = m.get("mse_masked")
        mse_f  = m.get("mse_full")
        knn_sm = m.get("knn_ssim_masked")
        knn_sf = m.get("knn_ssim_full")
        rows.append(
            f"<tr>"
            f"<td><a href='#{prefix}'>{prefix}</a></td>"
            f"<td class='{_mse_colour(mse_f)}'>{_fmt(mse_f)}</td>"
            f"<td class='{_ssim_colour(ssim_f)}'>{_fmt(ssim_f, 3)}</td>"
            f"<td class='{_mse_colour(mse_m)}'>{_fmt(mse_m)}</td>"
            f"<td class='{_ssim_colour(ssim_m)}'>{_fmt(ssim_m, 3)}</td>"
            f"<td class='{_ssim_colour(knn_sf)}'>{_fmt(knn_sf, 3)}</td>"
            f"<td class='{_ssim_colour(knn_sm)}'>{_fmt(knn_sm, 3)}</td>"
            f"</tr>"
        )
    summary_table = (
        "<table class='summary-table'>"
        "<thead><tr>"
        "<th>Glyph</th>"
        "<th>MSE (full)</th><th>SSIM (full)</th>"
        "<th>MSE (masked)</th><th>SSIM (masked)</th>"
        "<th>KNN SSIM (full)</th><th>KNN SSIM (masked)</th>"
        "</tr></thead>"
        "<tbody>" + "\n".join(rows) + "</tbody>"
        "</table>"
    )

    # ── entry cards ──────────────────────────────────────────────────────────
    cards = []
    for prefix, m, strip_path in entries:
        img_uri = _embed_image(strip_path)
        neighbors = m.get("knn_neighbors")

        metric_pairs = [
            ("MSE full",      m.get("mse_full"),      _mse_colour(m.get("mse_full"))),
            ("SSIM full",     m.get("ssim_full"),      _ssim_colour(m.get("ssim_full"))),
            ("MSE masked",    m.get("mse_masked"),     _mse_colour(m.get("mse_masked"))),
            ("SSIM masked",   m.get("ssim_masked"),    _ssim_colour(m.get("ssim_masked"))),
            ("KNN SSIM full", m.get("knn_ssim_full"),  _ssim_colour(m.get("knn_ssim_full"))),
            ("KNN SSIM mask", m.get("knn_ssim_masked"),_ssim_colour(m.get("knn_ssim_masked"))),
        ]
        metric_blocks = "".join(
            f"<div class='metric-block'>"
            f"<span class='metric-label'>{label}</span>"
            f"<span class='metric-value {cls}'>{_fmt(val, 4 if 'MSE' in label else 3)}</span>"
            f"</div>"
            for label, val, cls in metric_pairs
            if val is not None
        )

        ckpt = Path(m.get("checkpoint", "")).name
        image_name = Path(m.get("image", "")).name
        knn_block = ""
        if neighbors:
            spans = "".join(f"<span>{c}</span>" for c in neighbors)
            knn_block = f"<div class='knn-neighbors'>KNN neighbours: {spans}</div>"

        mask_rect = m.get("mask_rect")
        mask_info = (
            f"mask y0={mask_rect[0]}px x0={mask_rect[1]}px "
            f"h={mask_rect[2]}px w={mask_rect[3]}px"
            if mask_rect else "no mask"
        )

        cards.append(
            f"<div class='entry' id='{prefix}'>"
            f"<div class='entry-header'>"
            f"<span class='entry-title'>{prefix}</span>"
            f"<span class='entry-meta'>{image_name} · {ckpt} · {mask_info}</span>"
            f"</div>"
            f"<div class='strip-wrap'>"
            f"<img class='strip-img' src='{img_uri}' alt='{prefix} reconstruction strip'>"
            f"</div>"
            f"<div class='metrics-row'>{metric_blocks}</div>"
            f"{knn_block}"
            f"</div>"
        )

    cards_html = "\n".join(cards)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Reconstruction Report — hackingrongo</title>
<style>{_CSS}</style>
</head>
<body>
<header>
  <h1>Glyph Reconstruction Report</h1>
  <p>Fill-the-gap: autoencoder + KNN inpainting of damaged rongorongo signs &nbsp;·&nbsp; {generated}</p>
</header>
<nav>
  <a href="#summary">Summary</a>
  {''.join(f'<a href="#{p}">{p}</a>' for p, _, _ in entries)}
</nav>
<div class="container">
  <h2 id="summary">Summary</h2>
  {summary_table}
  <h2>Reconstructions</h2>
  {cards_html}
</div>
<footer>
  hackingrongo · fill-the-gap reconstruction report · {generated}
</footer>
</body>
</html>"""


def save_reconstruction_report(metrics_dir: Path, output_path: Path) -> None:
    html = build_reconstruction_report(metrics_dir)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    logger.info("Report saved: %s  (%d KB)", output_path, len(html) // 1024)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
    p = argparse.ArgumentParser(
        description="Generate an HTML gallery from reconstruct_glyph.py outputs."
    )
    p.add_argument(
        "--input", default=None,
        help="Directory containing *_metrics.json + *_reconstruction.png files.  "
             "Default: outputs/reconstruction/",
    )
    p.add_argument(
        "--output", default=None,
        help="Output HTML path.  Default: <input>/reconstruction_report.html",
    )
    args = p.parse_args()

    project_root = Path(__file__).resolve().parent.parent.parent
    metrics_dir = Path(args.input) if args.input else project_root / "outputs" / "reconstruction"
    output_path = (
        Path(args.output) if args.output
        else metrics_dir / "reconstruction_report.html"
    )

    save_reconstruction_report(metrics_dir, output_path)
    print(f"Report: {output_path}")


if __name__ == "__main__":
    main()
