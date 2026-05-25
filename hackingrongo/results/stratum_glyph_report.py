"""
hackingrongo.results.stratum_glyph_report
==========================================

HTML report showing which Barthel-coded glyphs are unique to the
pre-contact stratum, unique to the post-contact stratum, or span both.

For each unique sign the report shows:
  - One representative SVG image (first recorded occurrence)
  - Barthel code
  - Barthel taxonomy category (derived from code range)
  - Known name / scholarly reading (where available)
  - Which tablets it occurs on, with occurrence count

Usage
-----
    python -m hackingrongo.results.stratum_glyph_report \\
        --catalog   data/glyphs/svg/catalog.json \\
        --tablets   data/metadata/tablets.json \\
        --sign-meta data/catalog/sign_metadata.json \\
        --glyphs-dir data/glyphs \\
        --output    outputs/analysis/stratum_glyph_report.html
"""

from __future__ import annotations

import argparse
import base64
import logging
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
import json

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Barthel taxonomy — derived from the 100-block code ranges in
# Barthel (1958) "Grundlagen zur Entzifferung der Osterinselschrift".
# ---------------------------------------------------------------------------

_TAXONOMY: list[tuple[int, int, str]] = [
    (1,   99,  "Anthropomorphic / human figures"),
    (100, 199, "Birds & animals"),
    (200, 299, "Fish & marine creatures"),
    (300, 399, "Plants & botanical forms"),
    (400, 499, "Celestial & abstract"),
    (500, 599, "Composite / compound glyphs"),
    (600, 699, "Ligature marks & determinatives"),
    (700, 999, "Extended catalogue"),
]

_STRATUM_LABEL = {
    "pre_contact":  "Pre-contact",
    "post_contact": "Post-contact",
}

_SECTION_ORDER = ["pre_contact", "post_contact", "both"]
_SECTION_TITLE = {
    "pre_contact": "Pre-contact only",
    "post_contact": "Post-contact only",
    "both": "Spans both strata",
}
_SECTION_DESC = {
    "pre_contact": (
        "Signs attested exclusively on tablets assigned to the pre-contact stratum "
        "(Tablet D and equivalents dated before European contact)."
    ),
    "post_contact": (
        "Signs attested exclusively on tablets assigned to the post-contact stratum."
    ),
    "both": (
        "Signs that appear on tablets in both strata — these are the most "
        "chronologically persistent and are strong candidates for core vocabulary."
    ),
}
_SECTION_COLOUR = {
    "pre_contact":  "#2d6a4f",
    "post_contact": "#1d3557",
    "both":         "#7b2d8b",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _barthel_taxonomy(code: str) -> str:
    """Return the taxonomy label for a Barthel code string."""
    try:
        numeric = int("".join(c for c in code if c.isdigit()))
    except ValueError:
        return "Unknown"
    for lo, hi, label in _TAXONOMY:
        if lo <= numeric <= hi:
            return label
    return "Unknown"


def _load_svg(path: Path) -> str:
    """Return SVG content as a data URI, or a placeholder if file is missing."""
    if not path.exists():
        return ""
    try:
        raw = path.read_bytes()
        b64 = base64.b64encode(raw).decode("ascii")
        return f"data:image/svg+xml;base64,{b64}"
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# Core: build glyph inventory grouped by stratum bucket
# ---------------------------------------------------------------------------


def _build_inventory(
    catalog: list[dict],
    sign_meta: dict[str, dict],
    tablets_meta: dict[str, dict],
    glyphs_dir: Path,
) -> dict[str, list[dict]]:
    """Return {bucket: [glyph_entry, ...]} where bucket ∈ pre_contact | post_contact | both."""

    # Per Barthel code: track strata set, tablet set, first svg path, occurrence count.
    by_code: dict[str, dict] = {}

    for record in catalog:
        code = str(record.get("barthel_code", "")).strip()
        if not code or code in ("?", ""):
            continue
        cluster = str(record.get("cluster", "")).strip()
        if cluster not in ("pre_contact", "post_contact"):
            continue
        tablet = str(record.get("tablet", record.get("tablet_id", ""))).strip()
        svg_rel = str(record.get("svg_path", "")).strip()

        if code not in by_code:
            by_code[code] = {
                "barthel_code": code,
                "strata": set(),
                "tablets": defaultdict(int),   # tablet_id → count
                "svg_path": svg_rel,           # first occurrence
                "count": 0,
            }

        entry = by_code[code]
        entry["strata"].add(cluster)
        entry["tablets"][tablet] += 1
        entry["count"] += 1
        # Prefer a pre-contact SVG as the canonical image (cleaner provenance).
        if cluster == "pre_contact" and not entry["svg_path"].startswith("svg/D"):
            entry["svg_path"] = svg_rel

    # Build final records with resolved metadata.
    buckets: dict[str, list[dict]] = {"pre_contact": [], "post_contact": [], "both": []}

    for code, entry in sorted(by_code.items(), key=lambda x: _sort_key(x[0])):
        strata = entry["strata"]
        if len(strata) == 2:
            bucket = "both"
        else:
            bucket = next(iter(strata))

        # Tablet list: sorted by tablet ID, with per-tablet counts.
        tablet_items = sorted(entry["tablets"].items())
        tablet_strs: list[str] = []
        for tid, cnt in tablet_items:
            tab_name = tablets_meta.get(tid, {}).get("name", "")
            label = f"{tid} ({tab_name})" if tab_name else tid
            tablet_strs.append(f"{label} ×{cnt}")

        # Per-tablet stratum labels for the tooltip.
        tablet_strata: list[str] = []
        for tid, _ in tablet_items:
            tab_cluster = tablets_meta.get(tid, {}).get(
                "date_distribution", {}
            ).get("type", "")
            tablet_strata.append(
                "pre" if "pre" in tab_cluster or tid == "D" else "post"
            )

        meta = sign_meta.get(code, {})
        known_as = meta.get("known_as", "")

        svg_uri = ""
        if entry["svg_path"]:
            svg_uri = _load_svg(glyphs_dir / entry["svg_path"])

        buckets[bucket].append({
            "barthel_code": code,
            "taxonomy": _barthel_taxonomy(code),
            "known_as": known_as,
            "strata": sorted(strata),
            "tablets": tablet_strs,
            "total_count": entry["count"],
            "svg_uri": svg_uri,
        })

    return buckets


def _sort_key(code: str) -> tuple:
    """Sort by numeric part, then suffix."""
    digits = "".join(c for c in code if c.isdigit())
    suffix = "".join(c for c in code if not c.isdigit())
    try:
        return (int(digits), suffix)
    except ValueError:
        return (9999, code)


# ---------------------------------------------------------------------------
# HTML rendering
# ---------------------------------------------------------------------------

_CSS = """
:root {
  --bg: #fafaf8;
  --card-bg: #ffffff;
  --border: #e0dcd6;
  --text: #1a1a1a;
  --muted: #666;
  --accent-pre:  #2d6a4f;
  --accent-post: #1d3557;
  --accent-both: #7b2d8b;
  --radius: 10px;
  --shadow: 0 1px 4px rgba(0,0,0,.08);
  font-size: 15px;
}
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
body { background: var(--bg); color: var(--text); font-family: -apple-system, "Segoe UI", Helvetica, Arial, sans-serif; padding: 32px 24px; }
h1 { font-size: 28px; font-weight: 700; margin-bottom: 6px; }
.subtitle { color: var(--muted); font-size: 15px; margin-bottom: 32px; }
.toc { display: flex; gap: 12px; flex-wrap: wrap; margin-bottom: 40px; }
.toc a { padding: 6px 14px; border-radius: 20px; font-size: 13px; font-weight: 600; text-decoration: none; color: #fff; }
.toc-pre  { background: var(--accent-pre); }
.toc-post { background: var(--accent-post); }
.toc-both { background: var(--accent-both); }
section { margin-bottom: 56px; }
.section-header { border-left: 5px solid currentColor; padding-left: 14px; margin-bottom: 10px; }
.section-header h2 { font-size: 22px; font-weight: 700; }
.section-header .desc { font-size: 13px; color: var(--muted); margin-top: 4px; }
.section-header .count { font-size: 13px; font-weight: 600; margin-top: 2px; }
.pre-contact  .section-header { color: var(--accent-pre); }
.post-contact .section-header { color: var(--accent-post); }
.both         .section-header { color: var(--accent-both); }
.glyph-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(160px, 1fr));
  gap: 14px;
  margin-top: 16px;
}
.glyph-card {
  background: var(--card-bg);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  box-shadow: var(--shadow);
  padding: 14px 10px 10px;
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 6px;
}
.glyph-img {
  width: 80px; height: 80px;
  object-fit: contain;
  background: #f5f3ef;
  border-radius: 6px;
  padding: 6px;
}
.glyph-img-placeholder {
  width: 80px; height: 80px;
  background: #f0ede8;
  border-radius: 6px;
  display: flex;
  align-items: center;
  justify-content: center;
  color: #bbb;
  font-size: 11px;
}
.glyph-code { font-family: "JetBrains Mono", "Fira Code", monospace; font-size: 14px; font-weight: 700; letter-spacing: .02em; }
.glyph-name { font-size: 12px; color: var(--muted); font-style: italic; text-align: center; }
.glyph-tax  { font-size: 11px; color: #888; text-align: center; line-height: 1.3; }
.glyph-tablets { font-size: 11px; color: var(--muted); text-align: center; line-height: 1.5; }
.glyph-count { font-size: 11px; color: #aaa; }
footer { margin-top: 40px; padding-top: 20px; border-top: 1px solid var(--border); font-size: 12px; color: var(--muted); }
"""


def _card_html(entry: dict) -> str:
    if entry["svg_uri"]:
        img_html = f'<img class="glyph-img" src="{entry["svg_uri"]}" alt="{entry["barthel_code"]}">'
    else:
        img_html = f'<div class="glyph-img-placeholder">no image</div>'

    name_html = f'<div class="glyph-name">{entry["known_as"]}</div>' if entry["known_as"] else ""
    tablets_joined = "<br>".join(entry["tablets"])

    return f"""
<div class="glyph-card">
  {img_html}
  <div class="glyph-code">{entry["barthel_code"]}</div>
  {name_html}
  <div class="glyph-tax">{entry["taxonomy"]}</div>
  <div class="glyph-tablets">{tablets_joined}</div>
  <div class="glyph-count">n={entry["total_count"]}</div>
</div>"""


def _section_html(bucket: str, entries: list[dict]) -> str:
    colour = _SECTION_COLOUR[bucket]
    css_class = bucket.replace("_", "-")
    cards = "\n".join(_card_html(e) for e in entries)
    return f"""
<section class="{css_class}" style="--section-colour:{colour}">
  <div class="section-header">
    <h2>{_SECTION_TITLE[bucket]}</h2>
    <div class="desc">{_SECTION_DESC[bucket]}</div>
    <div class="count">{len(entries)} unique sign(s)</div>
  </div>
  <div class="glyph-grid">
    {cards}
  </div>
</section>"""


def build_stratum_glyph_report(
    catalog_path: Path,
    tablets_path: Path,
    sign_meta_path: Path,
    glyphs_dir: Path,
) -> str:
    """Build and return the full HTML string."""
    catalog: list[dict] = json.loads(catalog_path.read_text(encoding="utf-8"))
    if isinstance(catalog, dict):
        catalog = catalog.get("records", [])

    tablets_meta: dict = json.loads(tablets_path.read_text(encoding="utf-8"))

    sign_meta: dict = {}
    if sign_meta_path.exists():
        raw = json.loads(sign_meta_path.read_text(encoding="utf-8"))
        for v in raw.values():
            if isinstance(v, dict) and "barthel_code" in v:
                sign_meta[v["barthel_code"]] = v

    buckets = _build_inventory(catalog, sign_meta, tablets_meta, glyphs_dir)

    toc_links = "".join(
        f'<a class="toc-{b.split("_")[0]}" href="#{b}">'
        f'{_SECTION_TITLE[b]} ({len(buckets[b])})</a>'
        for b in _SECTION_ORDER
    )

    sections = "\n".join(
        f'<div id="{b}">' + _section_html(b, buckets[b]) + "</div>"
        for b in _SECTION_ORDER
    )

    total = sum(len(v) for v in buckets.values())
    generated = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Rongorongo — Stratum Glyph Report</title>
<style>{_CSS}</style>
</head>
<body>
<h1>Rongorongo — Stratum Glyph Inventory</h1>
<p class="subtitle">
  {total} unique Barthel-coded signs classified by stratigraphic occurrence.
  Generated {generated}.
</p>
<nav class="toc">{toc_links}</nav>
{sections}
<footer>
  Glyph images from Fischer / Barthel SVG corpus.
  Stratum assignments from radiocarbon-calibrated tablet metadata.
  Taxonomy ranges after Barthel (1958).
</footer>
</body>
</html>"""


def save_stratum_glyph_report(
    catalog_path: Path,
    tablets_path: Path,
    sign_meta_path: Path,
    glyphs_dir: Path,
    output_path: Path,
) -> None:
    html = build_stratum_glyph_report(catalog_path, tablets_path, sign_meta_path, glyphs_dir)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    logger.info("Stratum glyph report → %s  (%d bytes)", output_path, len(html))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

    p = argparse.ArgumentParser(description="Generate stratum glyph HTML report.")
    p.add_argument("--catalog",   required=True, type=Path, help="data/glyphs/svg/catalog.json")
    p.add_argument("--tablets",   required=True, type=Path, help="data/metadata/tablets.json")
    p.add_argument("--sign-meta", required=True, type=Path, help="data/catalog/sign_metadata.json")
    p.add_argument("--glyphs-dir", required=True, type=Path, help="data/glyphs/")
    p.add_argument("--output",    required=True, type=Path, help="Output HTML path")
    args = p.parse_args()

    save_stratum_glyph_report(
        catalog_path=args.catalog,
        tablets_path=args.tablets,
        sign_meta_path=args.sign_meta,
        glyphs_dir=args.glyphs_dir,
        output_path=args.output,
    )
