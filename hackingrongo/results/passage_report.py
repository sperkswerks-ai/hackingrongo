"""
hackingrongo.results.passage_report
=====================================

Scholar-facing HTML report for parallel passage alignment analysis.

Two output modes
----------------
``build_passage_report(passages_json, ...)``
    Returns an HTML string for the *diachronic cross-passage summary page* —
    a single document that aggregates all passages, ranks them by interest
    score, and surfaces holy-grail candidates and family-crossing changes.
    This is the entry point Ferrara and Horley should see first.

``build_single_passage_report(passage, ...)``
    Returns an HTML string for one passage's detail card.

``save_passage_report(passages_json, output_path, ...)``
    Writes the summary report.

``PassageReportGenerator``
    Backward-compatible class interface wrapping the functions above.
    ``generate_report()`` writes per-passage HTML files *and* a summary
    index.  ``render_passage()`` returns the per-passage HTML string.

Design language
---------------
Matches compound_report, divergence_report, and decipherment_report:
  * Light background — CSS variables --bg / --surface / --surface2
  * Cormorant Garamond (body) + JetBrains Mono (code / metadata)
  * Accent colour --accent = #c4a96d (gold)
  * No Jinja2 — pure Python f-strings, no external template dependency
"""

from __future__ import annotations

import argparse
import base64
import json
import logging
import re
import html as _html
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent

# ---------------------------------------------------------------------------
# Glyph catalog loading (SVG + PNG fallback, same pattern as compound_report)
# ---------------------------------------------------------------------------

def _load_svg_catalog(catalog_path: Path | None = None) -> dict[str, list[Path]]:
    if catalog_path is None:
        catalog_path = _REPO_ROOT / "data" / "glyphs" / "svg" / "catalog.json"
    if not catalog_path.exists():
        logger.warning("SVG catalog not found: %s — glyphs will not render.", catalog_path)
        return {}

    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    svg_dir = catalog_path.parent

    exact: dict[str, list[Path]] = defaultdict(list)
    base_map: dict[str, list[Path]] = defaultdict(list)

    for r in catalog.get("records", []):
        code = str(r.get("barthel_code", "")).strip()
        if not code:
            continue
        rel = str(r.get("svg_path", "")).replace("svg/", "", 1)
        full = svg_dir / rel
        if not full.exists():
            continue
        exact[code].append(full)
        base = re.sub(r'[!?()\s]+$', '', code).strip()
        if base != code:
            base_map[base].append(full)
        numeric_base = re.sub(r'[a-zA-Z!?()\s].*$', '', code).strip()
        if numeric_base and numeric_base != code and numeric_base != base:
            base_map[numeric_base].append(full)

    merged: dict[str, list[Path]] = dict(exact)
    for base_code, paths in base_map.items():
        if base_code not in merged:
            merged[base_code] = paths

    # PNG fallback
    bc_path = catalog_path.parent.parent / "barthel_catalog.json"
    if bc_path.exists():
        glyph_dir = catalog_path.parent.parent
        bc_records = json.loads(bc_path.read_text(encoding="utf-8")).get("records", [])
        png_stage: dict[str, Path] = {}
        for source_pref in ("barthel_formentafeln", "barthel_tafeln"):
            for r in bc_records:
                if r.get("source") != source_pref:
                    continue
                code = str(r.get("barthel_code") or "").strip()
                png_rel = r.get("path", "")
                if not code or not png_rel or not png_rel.endswith(".png"):
                    continue
                png_full = glyph_dir / png_rel
                if png_full.exists():
                    png_stage[code] = png_full
        for code, png_path in png_stage.items():
            if code not in merged:
                merged[code] = [png_path]
            numeric_base = re.sub(r'[a-zA-Z!?()\s].*$', '', code).strip()
            if numeric_base and numeric_base != code and numeric_base not in merged:
                merged[numeric_base] = [png_path]

    return merged


def _normalise_svg(svg_text: str, size: int = 60) -> str:
    svg = svg_text.strip()
    svg = re.sub(r'width="[^"]*"',  f'width="{size}"',  svg)
    svg = re.sub(r'height="[^"]*"', f'height="{size}"', svg)
    svg = re.sub(
        r"<path ",
        '<path fill="none" stroke="currentColor" stroke-width="1.5" '
        'stroke-linecap="round" stroke-linejoin="round" ',
        svg,
    )
    return svg


def _get_glyph_html(code: str, catalog: dict[str, list[Path]], size: int = 60) -> str | None:
    """Return inline SVG or base64 img HTML for a Barthel code. None if unavailable."""
    instances = catalog.get(code, [])
    if not instances:
        base = re.sub(r'[!?()\s]+$', '', code).strip()
        instances = catalog.get(base, [])
    if not instances:
        numeric_base = re.sub(r'[a-zA-Z!?()\s].*$', '', code).strip()
        if numeric_base and numeric_base != code:
            instances = catalog.get(numeric_base, [])
    if not instances:
        return None
    path = instances[0]
    try:
        if path.suffix.lower() == ".png":
            b64 = base64.b64encode(path.read_bytes()).decode()
            return (
                f'<img src="data:image/png;base64,{b64}" '
                f'style="max-width:{size}px;max-height:{size}px;display:block;margin:auto;" '
                f'alt="Barthel {code}">'
            )
        return _normalise_svg(path.read_text(encoding="utf-8"), size=size)
    except Exception:
        return None


def _glyph_cell(code: str, catalog: dict[str, list[Path]], size: int = 56,
                cls: str = "", label_override: str | None = None) -> str:
    """Return a single glyph cell div (image + code label)."""
    glyph_html = _get_glyph_html(code, catalog, size)
    label = label_override if label_override is not None else code
    if glyph_html:
        return (
            f'<div class="g-cell {cls}">'
            f'<div class="g-img" style="color:var(--accent)">{glyph_html}</div>'
            f'<div class="g-code">{label}</div>'
            f'</div>'
        )
    return (
        f'<div class="g-cell g-missing {cls}">'
        f'<div class="g-img">?</div>'
        f'<div class="g-code">{label}</div>'
        f'</div>'
    )


def _glyph_strip(codes: list[str], catalog: dict[str, list[Path]],
                 highlight_pos: int | None = None, size: int = 52) -> str:
    """Return a horizontal strip of glyph cells for a sequence of Barthel codes."""
    if not codes or not catalog:
        return ""
    cells = []
    for i, code in enumerate(codes):
        cls = "g-hl" if i == highlight_pos else ""
        cells.append(_glyph_cell(str(code), catalog, size=size, cls=cls))
    return f'<div class="g-strip">{"".join(cells)}</div>'

# ---------------------------------------------------------------------------
# Shared CSS  (mirrors decipherment_report / divergence_report variables)
# ---------------------------------------------------------------------------

_CSS = """
:root {
  --bg: #fefefe; --surface: #f7f7fa; --surface2: #eeeef4;
  --border: #dddde8; --text: #1a1a2e; --muted: #6b7280;
  --accent: #c4a96d; --accent2: #7b9ee0;
  --pre: #1d4ed8; --post: #6d28d9; --undated: #6b7280;
  --holy: #d97706; --cross: #b91c1c;
  --cm: #f0fdf4; --cmb: #86efac; --cmt: #14532d;
  --cs: #fef3c7; --csb: #fcd34d; --cst: #78350f;
  --cg: #f8fafc; --cgb: #cbd5e1; --cgt: #9ca3af;
}
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
  background: var(--bg); color: var(--text);
  font-family: 'Cormorant Garamond', 'Palatino Linotype', Georgia, serif;
  font-size: 16px; line-height: 1.65;
}
.wrap { max-width: 1200px; margin: 0 auto; padding: 52px 28px 80px; }
.mono { font-family: 'JetBrains Mono', 'Fira Mono', monospace; }
.muted { color: var(--muted); }
.small { font-size: 11px; }

/* ── Report header ── */
.report-header { border-bottom: 1px solid var(--border);
                 padding-bottom: 44px; margin-bottom: 44px; }
.report-title { font-size: 38px; font-weight: 600; color: #000;
                letter-spacing: -0.5px; line-height: 1.2; }
.report-subtitle { font-size: 17px; color: var(--accent); font-style: italic;
                   margin-top: 8px; }
.report-meta { margin-top: 20px; font-family: 'JetBrains Mono', monospace;
               font-size: 11px; color: var(--muted); line-height: 2.4; }
.report-meta b { color: #333; }
.abstract { margin-top: 22px; font-size: 14.5px; color: #333;
            max-width: 840px; line-height: 1.9;
            border-left: 3px solid var(--accent); padding-left: 18px; }
.abstract p + p { margin-top: 12px; }

/* ── Summary stats ── */
.stats-row { display: flex; flex-wrap: wrap; gap: 14px; margin-bottom: 44px; }
.stat-card { background: var(--surface); border: 1px solid var(--border);
             border-radius: 8px; padding: 18px 24px; min-width: 110px;
             text-align: center; transition: box-shadow .15s; }
.stat-card:hover { box-shadow: 0 2px 12px rgba(0,0,0,.07); }
.stat-value { font-family: 'JetBrains Mono', monospace; font-size: 30px;
              font-weight: 500; color: var(--accent); line-height: 1; margin-bottom: 6px; }
.stat-label { font-size: 9.5px; color: var(--muted); margin-top: 4px;
              font-family: 'JetBrains Mono', monospace;
              text-transform: uppercase; letter-spacing: .05em; }
.stat-card.holy .stat-value { color: var(--holy); }
.stat-card.cross .stat-value { color: var(--cross); }
.stat-card.pre .stat-value { color: var(--pre); }

/* ── Section label ── */
.section-label { font-family: 'JetBrains Mono', monospace; font-size: 9px;
                 color: var(--muted); letter-spacing: 0.1em;
                 text-transform: uppercase; margin-bottom: 12px; }

/* ── Holy Grail Spotlight ── */
.hg-section { background: linear-gradient(135deg, #fffbf0, #fff8e8);
              border: 1px solid #f6c549; border-left: 4px solid var(--holy);
              border-radius: 8px; padding: 28px 32px; margin-bottom: 44px; }
.hg-title { font-family: 'JetBrains Mono', monospace; font-size: 10px;
            font-weight: 600; color: var(--holy); text-transform: uppercase;
            letter-spacing: .12em; margin-bottom: 6px; }
.hg-sub { font-size: 13.5px; color: #666; margin-bottom: 22px; font-style: italic; }
.hg-card { background: #fff; border: 1px solid #f6c549; border-radius: 6px;
           padding: 20px 24px; margin-bottom: 14px; }
.hg-card:last-child { margin-bottom: 0; }
.hg-head { display: flex; align-items: center; gap: 12px;
           margin-bottom: 16px; flex-wrap: wrap; }
.hg-pid { font-family: 'JetBrains Mono', monospace; font-size: 13px;
          font-weight: 500; color: var(--holy); }
.hg-pos { font-family: 'JetBrains Mono', monospace; font-size: 11px; color: var(--muted); }
.hg-diff { display: flex; align-items: center; gap: 10px; flex-wrap: wrap; margin-top: 4px; }
.hg-sign-grp { display: flex; flex-direction: column; gap: 4px; }
.hg-sign-lbl { font-family: 'JetBrains Mono', monospace; font-size: 8px;
               text-transform: uppercase; letter-spacing: .06em; color: var(--muted); }
.hg-sign { font-family: 'JetBrains Mono', monospace; font-size: 18px;
           font-weight: 500; padding: 7px 14px; border-radius: 5px; display: inline-block; }
.hg-sign.pre-s { background: #dbeafe; color: var(--pre); border: 1px solid #bfdbfe; }
.hg-sign.post-s { background: #ede9fe; color: var(--post); border: 1px solid #ddd6fe; }
.hg-arrow { font-size: 22px; color: var(--muted); align-self: flex-end; padding-bottom: 8px; }
.hg-tablets { font-family: 'JetBrains Mono', monospace; font-size: 10px;
              color: var(--muted); margin-top: 10px; }
.hg-tablets b { color: #333; }
.hg-canon { display: flex; gap: 4px; margin-top: 10px; flex-wrap: wrap; align-items: center; }
.hg-canon-lbl { font-family: 'JetBrains Mono', monospace; font-size: 9px; color: var(--muted);
                text-transform: uppercase; letter-spacing: .05em; margin-right: 4px; }
.hg-chip { font-family: 'JetBrains Mono', monospace; font-size: 10px; padding: 2px 7px;
           border-radius: 3px; border: 1px solid var(--border);
           background: var(--surface2); color: #333; }
.hg-chip.hl { background: var(--cs); border-color: var(--csb); color: var(--cst); font-weight: 700; }

/* ── Filter bar ── */
.filter-bar { position: sticky; top: 0; background: rgba(254,254,254,.94);
              backdrop-filter: blur(8px); -webkit-backdrop-filter: blur(8px);
              border-bottom: 1px solid var(--border); padding: 10px 0 9px;
              margin: 0 0 24px; display: flex; align-items: center;
              flex-wrap: wrap; gap: 8px; z-index: 100; }
.fb-search { border: 1px solid var(--border); border-radius: 5px;
             padding: 6px 12px; font-family: 'JetBrains Mono', monospace;
             font-size: 11px; background: var(--surface); width: 200px;
             outline: none; transition: border-color .12s; }
.fb-search:focus { border-color: var(--accent); }
.filter-chips { display: flex; gap: 5px; flex-wrap: wrap; }
.chip { font-family: 'JetBrains Mono', monospace; font-size: 9px; padding: 4px 11px;
        border-radius: 20px; border: 1px solid var(--border);
        background: var(--surface); color: var(--muted); cursor: pointer;
        text-transform: uppercase; letter-spacing: .05em; transition: all .12s; }
.chip:hover { border-color: var(--accent); color: var(--accent); }
.chip.active { background: var(--accent); color: #fff; border-color: var(--accent); }
.chip.c-holy.active { background: var(--holy); border-color: var(--holy); }
.chip.c-cross.active { background: var(--cross); border-color: var(--cross); }
.fb-right { display: flex; align-items: center; gap: 8px; margin-left: auto; }
.fb-sort { font-family: 'JetBrains Mono', monospace; font-size: 10px;
           border: 1px solid var(--border); border-radius: 5px; padding: 5px 8px;
           background: var(--surface); cursor: pointer; }
.fb-count { font-family: 'JetBrains Mono', monospace; font-size: 10px;
            color: var(--muted); white-space: nowrap; }
.fb-btn { font-family: 'JetBrains Mono', monospace; font-size: 9px;
          padding: 4px 9px; border-radius: 4px; border: 1px solid var(--border);
          background: var(--surface); color: var(--muted); cursor: pointer;
          text-transform: uppercase; letter-spacing: .04em; }
.fb-btn:hover { background: var(--surface2); }

/* ── Passage cards (interactive accordion) ── */
.pc { border: 1px solid var(--border); border-radius: 8px;
      margin-bottom: 10px; overflow: hidden; transition: box-shadow .15s; }
.pc:hover { box-shadow: 0 2px 10px rgba(0,0,0,.06); }
.pc.holy-p { border-color: #f6c549; border-left: 3px solid var(--holy); }
.pc.dia-p:not(.holy-p) { border-left: 3px solid var(--pre); }
.pc-summary { padding: 12px 18px; display: flex; align-items: center;
              gap: 12px; cursor: pointer; background: var(--bg);
              flex-wrap: wrap; user-select: none; }
.pc-summary:hover { background: var(--surface); }
.pc.open .pc-summary { background: var(--surface); border-bottom: 1px solid var(--border); }
.pc-toggle { font-size: 9px; color: var(--muted); width: 14px; text-align: center;
             flex-shrink: 0; transition: transform .15s;
             font-family: 'JetBrains Mono', monospace; }
.pc.open .pc-toggle { transform: rotate(90deg); }
.pc-id { font-family: 'JetBrains Mono', monospace; font-size: 11.5px;
         color: var(--accent); font-weight: 500; flex-shrink: 0; min-width: 70px; }
.pc-badges { display: flex; gap: 4px; align-items: center; flex-shrink: 0; }
.bd-holy { background: #fff3cd; color: var(--holy); border-color: #fde68a; }
.bd-cross { background: #fde8e8; color: var(--cross); border-color: #fca5a5; }
.bd-dia { background: #f0fdf4; color: #15803d; border-color: #86efac; }
.pc-canonical { display: flex; flex-wrap: wrap; gap: 3px;
                align-items: center; flex: 1; }
.s-chip { font-family: 'JetBrains Mono', monospace; font-size: 9.5px;
          padding: 2px 6px; background: var(--surface2);
          border: 1px solid var(--border); border-radius: 3px; color: #333; }
.pc-right { display: flex; align-items: center; gap: 14px;
            margin-left: auto; flex-shrink: 0; }
.pc-tablet-list { font-family: 'JetBrains Mono', monospace;
                  font-size: 9.5px; color: var(--muted); }
.pc-score { font-family: 'JetBrains Mono', monospace; font-size: 10.5px; color: #555; }
.pc-detail { padding: 22px 22px 26px; background: var(--bg); }
.pc-meta-row { display: flex; gap: 18px; flex-wrap: wrap;
               font-family: 'JetBrains Mono', monospace; font-size: 10px;
               color: var(--muted); margin-bottom: 16px; }
.pc-meta-row b { color: #444; }

/* ── Alignment grid ── */
.ag-wrap { overflow-x: auto; margin-bottom: 22px; -webkit-overflow-scrolling: touch; }
.ag-table { border-collapse: separate; border-spacing: 2px;
            font-family: 'JetBrains Mono', monospace; font-size: 10px;
            min-width: max-content; }
.ag-corner { font-size: 8px; color: var(--muted); text-transform: uppercase;
             letter-spacing: .05em; padding: 3px 10px 3px 4px;
             text-align: right; vertical-align: bottom; white-space: nowrap; }
.ag-pos-h { font-size: 8px; color: var(--muted); text-align: center;
            padding: 3px 2px; vertical-align: bottom; width: 54px; min-width: 54px; }
.ag-lbl { text-align: right; padding: 0 10px 0 4px;
          vertical-align: middle; white-space: nowrap; }
.ag-lbl-in { display: flex; align-items: center; justify-content: flex-end;
             gap: 5px; height: 36px; }
.ag-tid { font-weight: 600; font-size: 12px; color: #333; }
.ag-dot { width: 7px; height: 7px; border-radius: 50%;
          display: inline-block; flex-shrink: 0; }
.ag-dot.pre  { background: var(--pre); }
.ag-dot.post { background: var(--post); }
.ag-dot.und  { background: var(--undated); }
.ag-dot.exc  { background: #d1d5db; }
.ag-pre-row .ag-lbl { border-left: 3px solid var(--pre);
                       background: rgba(29,78,216,.03); }
.ag-cell { width: 54px; min-width: 54px; height: 34px; text-align: center;
           vertical-align: middle; padding: 2px; }
.ag-cell-in { display: flex; align-items: center; justify-content: center;
              height: 100%; font-size: 10px; font-weight: 500; border-radius: 3px;
              padding: 0 3px; border: 1px solid transparent; flex-direction: column; }
.ag-can .ag-cell-in { background: var(--surface2); border-color: var(--border);
                       color: #333; font-weight: 600; }
.ag-mat .ag-cell-in { background: var(--cm); border-color: var(--cmb); color: var(--cmt); }
.ag-sub .ag-cell-in { background: var(--cs); border-color: var(--csb);
                       color: var(--cst); font-weight: 700; }
.ag-gap .ag-cell-in { background: var(--cg); border-color: var(--cgb);
                       border-style: dashed; color: var(--cgt); font-style: italic; }
.ag-nocc { font-size: 8px; color: var(--muted); font-weight: normal; }
.ag-cons-row .ag-lbl { font-size: 8px; color: var(--muted); font-style: italic; }
.ag-cons-cell { width: 54px; min-width: 54px; padding: 2px;
                vertical-align: middle; text-align: center; }
.ag-cbar-w { width: 46px; height: 8px; background: var(--surface2);
             border-radius: 3px; overflow: hidden; margin: 0 auto 2px; }
.ag-cbar { height: 100%; background: var(--cmb); border-radius: 3px; }
.ag-cpct { font-size: 8px; color: var(--muted); }

/* ── Inline changes ── */
.chg-wrap { margin-top: 6px; }
.chg-lbl { font-family: 'JetBrains Mono', monospace; font-size: 9px;
           color: var(--muted); text-transform: uppercase;
           letter-spacing: .08em; margin-bottom: 8px; }
.chg-item { display: flex; align-items: flex-start; gap: 10px;
            padding: 11px 14px; background: var(--surface);
            border: 1px solid var(--border); border-left: 3px solid var(--accent2);
            border-radius: 4px; margin-bottom: 7px; flex-wrap: wrap; }
.chg-item.holy  { border-left-color: var(--holy);  background: #fffbf0; }
.chg-item.cross { border-left-color: var(--cross); background: #fff8f8; }
.chg-pos { font-family: 'JetBrains Mono', monospace; font-size: 9px;
           color: var(--muted); min-width: 56px; padding-top: 2px; }
.chg-signs { display: flex; align-items: center; gap: 8px; }
.chg-sign { font-family: 'JetBrains Mono', monospace; font-size: 13px;
            padding: 4px 10px; border-radius: 4px; }
.chg-sign.pre-s  { background: #dbeafe; color: var(--pre);  border: 1px solid #bfdbfe; }
.chg-sign.post-s { background: #ede9fe; color: var(--post); border: 1px solid #ddd6fe; }
.chg-arr { color: var(--muted); font-size: 14px; }
.chg-meta { font-size: 11px; color: var(--muted);
            display: flex; flex-direction: column; gap: 2px; padding-top: 2px; }
.chg-flags { display: flex; gap: 4px; flex-wrap: wrap; margin-top: 4px; }

/* ── Passage table (summary / individual pages) ── */
.passage-table-wrap { overflow-x: auto; margin-bottom: 44px; }
.passage-table { width: 100%; border-collapse: collapse; font-size: 13px; }
.passage-table thead th { font-family: 'JetBrains Mono', monospace;
                          font-size: 9px; color: var(--muted); font-weight: 600;
                          text-transform: uppercase; letter-spacing: 0.08em;
                          padding: 6px 10px; border-bottom: 1px solid var(--border);
                          text-align: left; }
.passage-table tbody td { padding: 7px 10px;
                           border-bottom: 1px solid rgba(221,221,232,.6); }
.passage-table tbody tr:last-child td { border-bottom: none; }
.passage-table tbody tr:hover { background: var(--surface); }
.pt-id { font-family: 'JetBrains Mono', monospace; font-size: 11.5px;
         color: var(--accent); text-decoration: none; }
.pt-id:hover { text-decoration: underline; }
.score-val { font-family: 'JetBrains Mono', monospace; font-size: 11.5px; }
.badge { display: inline-block; font-family: 'JetBrains Mono', monospace;
         font-size: 8.5px; border-radius: 3px; padding: 2px 6px;
         border: 1px solid transparent; white-space: nowrap; }
.badge-pre  { color: var(--pre);  background: #dbeafe; border-color: #bfdbfe; }
.badge-post { color: var(--post); background: #ede9fe; border-color: #ddd6fe; }
.badge-none { color: var(--muted); background: var(--surface2);
              border-color: var(--border); }
.tag-holy  { display: inline-block; font-family: 'JetBrains Mono', monospace;
             font-size: 8px; background: #fff3cd; color: var(--holy);
             border: 1px solid #ffe08a; border-radius: 3px; padding: 1px 5px; }
.tag-cross { display: inline-block; font-family: 'JetBrains Mono', monospace;
             font-size: 8px; background: #fde8e8; color: var(--cross);
             border: 1px solid #f5b7b1; border-radius: 3px; padding: 1px 5px; }
.tag-anchor { display: inline-block; font-family: 'JetBrains Mono', monospace;
              font-size: 8px; background: #dbeafe; color: var(--pre);
              border: 1px solid #bfdbfe; border-radius: 3px; padding: 1px 5px;
              margin-left: 6px; }
.row-anchor > td { background: rgba(29,78,216,0.03); }
.row-anchor > td:first-child { border-left: 3px solid var(--pre); padding-left: 7px; }

/* ── Passage detail card (individual pages) ── */
.passage-card { background: var(--surface); border: 1px solid var(--border);
                border-radius: 8px; margin-bottom: 36px; overflow: hidden; }
.passage-card-header { padding: 16px 22px 12px; border-bottom: 1px solid var(--border);
                       display: flex; align-items: baseline; gap: 14px; flex-wrap: wrap; }
.passage-id { font-size: 15px; color: var(--accent); font-weight: 500;
              font-family: 'JetBrains Mono', monospace; }
.passage-score { font-family: 'JetBrains Mono', monospace; font-size: 12px;
                 color: var(--muted); }
.passage-body { padding: 20px 22px; }

/* ── Canonical sequence ── */
.sign-seq { display: flex; flex-wrap: wrap; gap: 5px; margin: 10px 0 18px; }
.sign-chip { font-family: 'JetBrains Mono', monospace; font-size: 10.5px;
             background: var(--surface2); border: 1px solid var(--border);
             border-radius: 3px; padding: 3px 7px; color: #333; }

/* ── Attestation table ── */
.attest-table { width: 100%; border-collapse: collapse; font-size: 12px;
                margin-bottom: 20px; }
.attest-table th { font-family: 'JetBrains Mono', monospace; font-size: 9px;
                   font-weight: 600; color: var(--muted); text-transform: uppercase;
                   letter-spacing: 0.06em; padding: 5px 8px;
                   border-bottom: 1px solid var(--border); text-align: left; }
.attest-table td { padding: 5px 8px; border-bottom: 1px solid rgba(221,221,232,.5);
                   vertical-align: top; }
.attest-table tr:last-child td { border-bottom: none; }
.attest-seq { font-family: 'JetBrains Mono', monospace; font-size: 10px;
              color: #333; }
.attest-ed  { font-family: 'JetBrains Mono', monospace; font-size: 11px;
              color: var(--muted); text-align: center; }

/* ── Alignment row (legacy per-attestation view) ── */
.align-row { display: flex; flex-wrap: wrap; gap: 2px; margin: 4px 0 8px; }
.align-cell { width: 26px; height: 26px; display: flex; align-items: center;
              justify-content: center; border-radius: 3px; font-size: 9px;
              font-family: 'JetBrains Mono', monospace; border: 1px solid transparent; }
.align-match { background: #d1fae5; border-color: #6ee7b7; color: #065f46; }
.align-sub   { background: #fef3c7; border-color: #fcd34d; color: #92400e; }
.align-gap   { background: #fee2e2; border-color: #fca5a5; color: #991b1b; }

/* ── Change cards (individual page) ── */
.changes-section { margin-top: 18px; }
.change-card { background: var(--bg); border: 1px solid var(--border);
               border-left: 3px solid var(--accent2);
               border-radius: 4px; padding: 12px 16px; margin-bottom: 10px; }
.change-card.holy  { border-left-color: var(--holy);  background: #fffbf0; }
.change-card.cross { border-left-color: var(--cross); background: #fff8f8; }
.change-head { display: flex; align-items: center; gap: 10px; margin-bottom: 8px;
               flex-wrap: wrap; }
.change-type-tag { font-family: 'JetBrains Mono', monospace; font-size: 8.5px;
                   padding: 2px 7px; border-radius: 3px; font-weight: 600;
                   text-transform: uppercase; }
.ct-sub  { background: #dbeafe; color: #1d4ed8; }
.ct-ins  { background: #d1fae5; color: #065f46; }
.ct-del  { background: #fee2e2; color: #991b1b; }
.change-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(130px, 1fr));
               gap: 8px; font-size: 12px; }
.change-field-label { font-family: 'JetBrains Mono', monospace; font-size: 8.5px;
                      color: var(--muted); margin-bottom: 2px; }
.change-field-val { font-family: 'JetBrains Mono', monospace; font-size: 12px; }

/* ── Report footer ── */
.report-footer { border-top: 1px solid var(--border); margin-top: 56px;
                 padding-top: 26px; font-size: 12px; color: var(--muted);
                 line-height: 2.0; }
.report-footer a { color: var(--accent); text-decoration: none; }
.page-nav { font-family: 'JetBrains Mono', monospace; font-size: 11px;
            color: var(--accent); margin-bottom: 32px; text-decoration: none;
            display: inline-block; }
.page-nav:hover { text-decoration: underline; }

/* ── Glyph strips ── */
.g-strip { display: flex; flex-wrap: wrap; gap: 6px; margin: 10px 0; align-items: flex-end; }
.g-cell { display: flex; flex-direction: column; align-items: center; gap: 3px;
          background: var(--surface); border: 1px solid var(--border);
          border-radius: 5px; padding: 6px 5px 4px; min-width: 52px; }
.g-cell.g-hl { background: var(--cs); border-color: var(--csb); }
.g-cell.g-missing .g-img { width: 52px; height: 52px; display: flex; align-items: center;
                            justify-content: center; color: var(--muted); font-size: 20px; }
.g-img { display: flex; align-items: center; justify-content: center; }
.g-img svg { display: block; }
.g-code { font-family: 'JetBrains Mono', monospace; font-size: 8px; color: var(--muted);
           text-align: center; line-height: 1; }
/* ── Sign-pair glyph display (holy grail / change cards) ── */
.sign-pair { display: flex; align-items: center; gap: 14px; margin: 10px 0; }
.sign-slot { display: flex; flex-direction: column; align-items: center; gap: 4px; }
.sign-slot-lbl { font-family: 'JetBrains Mono', monospace; font-size: 8px;
                 color: var(--muted); text-transform: uppercase; letter-spacing:.05em; }
.sign-glyph-box { background: var(--surface); border: 1px solid var(--border);
                  border-radius: 6px; padding: 8px 10px;
                  display: flex; align-items: center; justify-content: center; }
.sign-glyph-box.pre-box  { border-color: #93c5fd; background: #eff6ff; }
.sign-glyph-box.post-box { border-color: #c4b5fd; background: #f5f3ff; }
.sign-glyph-box svg { color: #1d4ed8; }
.sign-glyph-box.post-box svg { color: #6d28d9; }
.sign-arrow { font-size: 20px; color: var(--muted); padding-bottom: 10px; }
/* ── Hacker callout box ── */
.hack-box { background: #0f172a; color: #94a3b8; border-radius: 8px;
            padding: 18px 24px; margin-bottom: 28px; font-family: 'JetBrains Mono', monospace;
            font-size: 12px; line-height: 1.8; }
.hack-box .hack-title { color: #38bdf8; font-weight: 500; margin-bottom: 6px;
                        font-size: 11px; text-transform: uppercase; letter-spacing: .08em; }
.hack-box b { color: #e2e8f0; }
.hack-box .hack-hl { color: #fbbf24; }
"""

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_STRATUM_LABEL: dict[str, str] = {
    "pre_contact":  "Pre-contact",
    "post_contact": "Post-contact",
    "undated":      "Undated",
    "unknown":      "Unknown",
}


def _stratum_badge(stratum: str) -> str:
    css = {
        "pre_contact":  "badge-pre",
        "post_contact": "badge-post",
    }.get(stratum, "badge-none")
    label = _STRATUM_LABEL.get(stratum, stratum)
    return f'<span class="badge {css}">{label}</span>'


def _sign_seq_html(codes: list[str]) -> str:
    chips = "".join(f'<span class="sign-chip">{c}</span>' for c in codes)
    return f'<div class="sign-seq">{chips}</div>'


def _change_type_tag(change_type: str) -> str:
    css = {"substitution": "ct-sub", "insertion": "ct-ins", "deletion": "ct-del"}.get(
        change_type, "ct-sub"
    )
    return f'<span class="change-type-tag {css}">{change_type}</span>'


def _alignment_html(alignment: list[dict]) -> str:
    if not alignment:
        return ""
    cells = []
    for cell in alignment:
        mt = cell.get("match_type", "match")
        q, c = cell.get("query_code", ""), cell.get("corpus_code", "")
        tip = f"{q}→{c}" if q and c else (q or c or "—")
        css = {"match": "align-match", "substitution": "align-sub"}.get(mt, "align-gap")
        sym = {"match": "=", "substitution": "S", "insertion": "I", "deletion": "D"}.get(mt, "?")
        cells.append(f'<div class="align-cell {css}" title="{tip}">{sym}</div>')
    return f'<div class="align-row">{"".join(cells)}</div>'


# ---------------------------------------------------------------------------
# Interactive JavaScript for single-page summary
# ---------------------------------------------------------------------------

_JS = """
(function(){
  var _pcs = Array.from(document.querySelectorAll('.pc'));
  var _filter = 'all', _sort = 'score', _search = '';

  function _refresh() {
    var shown = 0;
    _pcs.forEach(function(pc) {
      var s = pc.dataset, ok = true;
      if (_search) ok = (s.pid||'').toLowerCase().indexOf(_search)>=0 ||
                        (s.canonical||'').toLowerCase().indexOf(_search)>=0;
      if (ok && _filter==='diachronic') ok = s.diachronic==='1';
      else if (ok && _filter==='holygrail') ok = s.holy==='1';
      else if (ok && _filter==='familycross') ok = s.cross==='1';
      pc.style.display = ok ? '' : 'none';
      if (ok) shown++;
    });
    var container = document.getElementById('passages');
    if (container) {
      var vis = _pcs.filter(function(pc){ return pc.style.display!=='none'; });
      vis.sort(function(a,b){
        if (_sort==='score')   return parseFloat(b.dataset.score||0)  - parseFloat(a.dataset.score||0);
        if (_sort==='tablets') return parseInt(b.dataset.tablets||0)  - parseInt(a.dataset.tablets||0);
        if (_sort==='changes') return parseInt(b.dataset.changes||0)  - parseInt(a.dataset.changes||0);
        if (_sort==='pid')     return (a.dataset.pid||'').localeCompare(b.dataset.pid||'');
        return 0;
      });
      vis.forEach(function(pc){ container.appendChild(pc); });
    }
    var el = document.getElementById('fb-count');
    if (el) el.textContent = 'Showing '+shown+' of '+_pcs.length;
  }

  window.togglePassage = function(pid) {
    var pc = document.getElementById('pc-'+pid);
    var det = document.getElementById('det-'+pid);
    if (!pc || !det) return;
    if (pc.classList.contains('open')) {
      det.style.display='none'; pc.classList.remove('open');
    } else {
      det.style.display='block'; pc.classList.add('open');
    }
  };
  window.setFilter = function(f, el) {
    _filter = f;
    document.querySelectorAll('.chip').forEach(function(c){ c.classList.remove('active'); });
    if (el) el.classList.add('active');
    _refresh();
  };
  window.setSort = function(v) { _sort=v; _refresh(); };
  window.searchPassages = function(q) { _search=q.toLowerCase().trim(); _refresh(); };
  window.expandAll = function() {
    _pcs.forEach(function(pc){
      if (pc.style.display==='none') return;
      var det = document.getElementById('det-'+pc.dataset.pid);
      if (det){ det.style.display='block'; pc.classList.add('open'); }
    });
  };
  window.collapseAll = function() {
    _pcs.forEach(function(pc){
      var det = document.getElementById('det-'+pc.dataset.pid);
      if (det){ det.style.display='none'; pc.classList.remove('open'); }
    });
  };
  // Auto-expand holy-grail passages on load
  _pcs.forEach(function(pc){
    if (pc.dataset.holy!=='1') return;
    var det = document.getElementById('det-'+pc.dataset.pid);
    if (det){ det.style.display='block'; pc.classList.add('open'); }
  });
})();
"""

# ---------------------------------------------------------------------------
# Tablet name lookup
# ---------------------------------------------------------------------------

_TABLET_NAMES: dict[str, str] = {
    "A": "Keiti",        "B": "Mamari",       "C": "Échancrée C",
    "D": "Échancrée D",  "E": "Tablet E",     "F": "Tablet F",
    "G": "Tablet G",     "H": "Tablet H",     "I": "Tablet I",
    "J": "Tablet J",     "K": "Tablet K",     "L": "Tablet L",
    "M": "Tablet M",     "N": "Tablet N",     "O": "Tablet O",
    "P": "Tablet P",     "Q": "Tablet Q",     "R": "Tablet R",
    "S": "Tablet S",     "T": "Tablet T",     "U": "Tablet U",
    "V": "Tablet V",     "W": "Tablet W",     "X": "Tablet X",
    "Y": "Tablet Y",
}


# ---------------------------------------------------------------------------
# Alignment grid helpers
# ---------------------------------------------------------------------------

def _build_tablet_consensus(attestations: list[dict]) -> list[dict]:
    """One consensus row per tablet: majority-vote form, sorted pre → post → undated."""
    from collections import Counter as _Counter
    by_tablet: dict[str, list] = {}
    for att in attestations:
        tid = att.get("tablet", "?")
        by_tablet.setdefault(tid, []).append(att)

    rows = []
    for tid, atts in sorted(by_tablet.items()):
        forms = [tuple(a.get("form") or []) for a in atts]
        majority = list(_Counter(forms).most_common(1)[0][0])
        stratum = next(
            (a["stratum"] for a in atts if a.get("stratum") not in ("", None)),
            "unknown",
        )
        rows.append({"tablet": tid, "stratum": stratum, "form": majority, "n_occ": len(atts)})

    _order = {"pre_contact": 0, "post_contact": 1, "undated": 2, "unknown": 3}
    rows.sort(key=lambda r: (_order.get(r["stratum"], 4), r["tablet"]))
    return rows


def _render_alignment_grid(canonical_form: list[str], attestations: list[dict]) -> str:
    """Render a colour-coded per-tablet alignment grid."""
    if not attestations:
        return '<p class="muted small">No attestations.</p>'

    tablet_rows = _build_tablet_consensus(attestations)
    n_pos = len(canonical_form)

    # Column headers
    pos_headers = "".join(
        f'<th class="ag-pos-h">{i + 1}</th>' for i in range(n_pos)
    )
    # Canonical row
    can_cells = "".join(
        f'<td class="ag-cell ag-can"><div class="ag-cell-in">{_html.escape(str(code))}</div></td>'
        for code in canonical_form
    )

    # Per-position consistency across non-excluded tablets
    relevant = [r for r in tablet_rows if r["stratum"] != "excluded"]
    consistency: list[float] = []
    for i in range(n_pos):
        if not relevant:
            consistency.append(0.0)
            continue
        matches = sum(
            1 for r in relevant
            if i < len(r["form"]) and r["form"][i] == canonical_form[i]
        )
        consistency.append(matches / len(relevant))

    # Tablet data rows
    att_rows_html: list[str] = []
    for row in tablet_rows:
        tid = row["tablet"]
        stratum = row["stratum"]
        form = row["form"]
        n_occ = row["n_occ"]

        dot_cls = {"pre_contact": "pre", "post_contact": "post", "undated": "und"}.get(
            stratum, "exc"
        )
        row_cls = "ag-pre-row" if stratum == "pre_contact" else ""
        name = _TABLET_NAMES.get(tid, tid)
        occ_html = f'<span class="ag-nocc">×{n_occ}</span>' if n_occ > 1 else ""

        lbl_html = (
            f'<td class="ag-lbl">'
            f'<div class="ag-lbl-in">'
            f'<span class="ag-dot {dot_cls}"></span>'
            f'<span class="ag-tid" title="{_html.escape(name, quote=True)}">{_html.escape(tid)}</span>'
            f'{occ_html}</div></td>'
        )

        cells = []
        for i, canon_code in enumerate(canonical_form):
            if i < len(form):
                css = "ag-mat" if form[i] == canon_code else "ag-sub"
                content = _html.escape(str(form[i]))
            else:
                css = "ag-gap"
                content = "—"
            cells.append(
                f'<td class="ag-cell {css}"><div class="ag-cell-in">{content}</div></td>'
            )
        att_rows_html.append(f'<tr class="{row_cls}">{lbl_html}{"".join(cells)}</tr>')

    # Consistency bar row
    cons_cells = "".join(
        f'<td class="ag-cons-cell">'
        f'<div class="ag-cbar-w"><div class="ag-cbar" style="width:{c*100:.0f}%"></div></div>'
        f'<div class="ag-cpct">{c*100:.0f}%</div></td>'
        for c in consistency
    )

    return (
        f'<div class="ag-wrap"><table class="ag-table"><thead>'
        f'<tr><th class="ag-corner">Tablet</th>{pos_headers}</tr>'
        f'<tr><td class="ag-corner" style="font-style:italic">Canonical</td>{can_cells}</tr>'
        f'</thead><tbody>{"".join(att_rows_html)}'
        f'<tr class="ag-cons-row">'
        f'<td class="ag-lbl" style="font-size:8px;font-style:italic;text-align:right;padding-right:10px">Consistency</td>'
        f'{cons_cells}</tr>'
        f'</tbody></table></div>'
    )


def _render_changes_inline(changes: list[dict]) -> str:
    """Render inline diachronic change items for a passage card."""
    if not changes:
        return (
            '<p class="muted small" style="font-style:italic;margin-top:4px">'
            'No diachronic changes detected — passage may lack co-occurrence of '
            'pre- and post-contact attestations at a consistent position.</p>'
        )
    items: list[str] = []
    for c in changes:
        is_holy = bool(c.get("is_holy_grail_candidate"))
        is_cross = bool(c.get("crosses_barthel_family"))
        item_cls = "holy" if is_holy else ("cross" if is_cross else "")
        pos = c.get("position", -1)
        pos_label = f"pos {pos + 1}" if pos >= 0 else "—"
        pre = c.get("pre_contact_sign", "—")
        post_sign = c.get("post_contact_sign", "—")
        n_cons = c.get("n_tablets_consistent", 0)
        ct = c.get("change_type", "substitution")

        flags: list[str] = []
        if is_holy:
            flags.append('<span class="badge bd-holy">★ Holy Grail</span>')
        if is_cross:
            flags.append('<span class="badge bd-cross">↕ Family-Crossing</span>')
        flags_html = (
            f'<div class="chg-flags">{"".join(flags)}</div>' if flags else ""
        )

        items.append(
            f'<div class="chg-item {item_cls}">'
            f'<div class="chg-pos">{_html.escape(str(pos_label))}<br>'
            f'<span style="font-size:8px">{_html.escape(str(ct))}</span></div>'
            f'<div class="chg-signs">'
            f'<span class="chg-sign pre-s">{_html.escape(str(pre))}</span>'
            f'<span class="chg-arr">→</span>'
            f'<span class="chg-sign post-s">{_html.escape(str(post_sign))}</span>'
            f'</div>'
            f'<div class="chg-meta">'
            f'<span><b>{n_cons}</b> post-contact tablet{"s" if n_cons != 1 else ""} consistent</span>'
            f'{flags_html}</div></div>'
        )
    return (
        f'<div class="chg-wrap">'
        f'<div class="chg-lbl">Diachronic changes ({len(changes)})</div>'
        f'{"".join(items)}</div>'
    )


def _render_passage_card(passage: dict, idx: int) -> str:
    """Render a collapsible accordion card for one passage."""
    pid = passage.get("passage_id", f"P{idx:03d}")
    canonical_form = passage.get("canonical_form") or []
    attestations = passage.get("attestations", [])
    changes = passage.get("diachronic_changes", [])
    score = passage.get("interest_score", 0.0)

    tablets = sorted({str(a.get("tablet", "")) for a in attestations if a.get("tablet")})
    strata = {a.get("stratum", "") for a in attestations}
    is_diachronic = "pre_contact" in strata and "post_contact" in strata
    n_holy = sum(1 for c in changes if c.get("is_holy_grail_candidate"))
    n_cross = sum(1 for c in changes if c.get("crosses_barthel_family"))

    card_cls = "holy-p" if n_holy else ("dia-p" if is_diachronic else "")

    badges: list[str] = []
    if n_holy:
        badges.append('<span class="badge bd-holy">★ Holy Grail</span>')
    if n_cross:
        badges.append('<span class="badge bd-cross">↕ Family-Crossing</span>')
    if is_diachronic and not n_holy:
        badges.append('<span class="badge bd-dia">⚓ Diachronic</span>')
    badges_html = (
        f'<div class="pc-badges">{"".join(badges)}</div>' if badges else ""
    )

    chips = "".join(f'<span class="s-chip">{c}</span>' for c in canonical_form)
    tablet_str = " · ".join(tablets)

    pre_count = sum(1 for a in attestations if a.get("stratum") == "pre_contact")
    post_count = sum(1 for a in attestations if a.get("stratum") == "post_contact")

    grid_html = _render_alignment_grid(canonical_form, attestations)
    changes_html = _render_changes_inline(changes)

    pid_attr = _html.escape(pid, quote=True)
    pid_js   = pid.replace("\\", "\\\\").replace("'", "\\'")
    return (
        f'<div class="pc {card_cls}" id="pc-{pid_attr}"'
        f' data-pid="{pid_attr}" data-score="{score}" data-tablets="{len(tablets)}"'
        f' data-changes="{len(changes)}" data-holy="{1 if n_holy else 0}"'
        f' data-cross="{1 if n_cross else 0}" data-diachronic="{1 if is_diachronic else 0}"'
        f' data-canonical="{_html.escape(" ".join(str(c) for c in canonical_form), quote=True)}">'
        f"<div class=\"pc-summary\" onclick=\"togglePassage('{pid_js}')\">"
        f'<span class="pc-toggle">▶</span>'
        f'<span class="pc-id">{pid_attr}</span>'
        f'{badges_html}'
        f'<div class="pc-canonical">{chips}</div>'
        f'<div class="pc-right">'
        f'<span class="pc-tablet-list">{tablet_str}</span>'
        f'<span class="pc-score">{score:.2f}</span>'
        f'</div></div>'
        f'<div class="pc-detail" id="det-{pid_attr}" style="display:none">'
        f'<div class="pc-meta-row">'
        f'<span><b>Tablets:</b> {len(tablets)} ({", ".join(tablets)})</span>'
        f'<span><b>Attestations:</b> {len(attestations)}</span>'
        f'<span><b>Pre-contact:</b> {pre_count}</span>'
        f'<span><b>Post-contact:</b> {post_count}</span>'
        f'<span><b>Canonical length:</b> {len(canonical_form)}</span>'
        f'</div>'
        f'<div class="section-label">Attestation alignment — one row per tablet'
        f' (majority form; ×N = repeated occurrences on that tablet)</div>'
        f'{grid_html}'
        f'{changes_html}'
        f'</div></div>'
    )


def _render_holy_grail_spotlight(passages: list[dict], catalog: dict | None = None) -> str:
    """Render the highlighted holy-grail candidates section."""
    catalog = catalog or {}
    cards: list[str] = []
    for p in sorted(passages, key=lambda x: x.get("interest_score", 0), reverse=True):
        pid = p.get("passage_id", "?")
        canonical = p.get("canonical_form") or []
        attestations = p.get("attestations", [])
        holy_changes = [
            c for c in p.get("diachronic_changes", [])
            if c.get("is_holy_grail_candidate")
        ]
        if not holy_changes:
            continue

        pre_tablets = sorted({
            str(a.get("tablet")) for a in attestations
            if a.get("stratum") == "pre_contact"
        })
        post_tablets = sorted({
            str(a.get("tablet")) for a in attestations
            if a.get("stratum") == "post_contact"
        })

        for change in holy_changes:
            pos = change.get("position", -1)
            pre_sign = change.get("pre_contact_sign", "—")
            post_sign = change.get("post_contact_sign", "—")
            n_cons = change.get("n_tablets_consistent", 0)
            is_cross = bool(change.get("crosses_barthel_family"))

            cross_badge = (
                ' <span class="badge bd-cross">↕ Family-Crossing</span>' if is_cross else ""
            )
            pre_label = f'Pre-contact ({", ".join(pre_tablets) or "—"})'
            post_suffix = "…" if len(post_tablets) > 4 else ""
            post_label = f'Post-contact ({", ".join(post_tablets[:4])}{post_suffix})'

            # Glyph display for the sign pair
            pre_glyph = _get_glyph_html(str(pre_sign), catalog, size=72) if catalog else None
            post_glyph = _get_glyph_html(str(post_sign), catalog, size=72) if catalog else None
            pre_glyph_html = pre_glyph or f'<span class="hg-sign pre-s">{pre_sign}</span>'
            post_glyph_html = post_glyph or f'<span class="hg-sign post-s">{post_sign}</span>'

            sign_pair_html = (
                f'<div class="sign-pair">'
                f'<div class="sign-slot">'
                f'<div class="sign-slot-lbl">{pre_label}</div>'
                f'<div class="sign-glyph-box pre-box" style="color:#1d4ed8">{pre_glyph_html}</div>'
                f'<div class="g-code">{pre_sign}</div>'
                f'</div>'
                f'<div class="sign-arrow">→</div>'
                f'<div class="sign-slot">'
                f'<div class="sign-slot-lbl">{post_label}</div>'
                f'<div class="sign-glyph-box post-box" style="color:#6d28d9">{post_glyph_html}</div>'
                f'<div class="g-code">{post_sign}</div>'
                f'</div>'
                f'</div>'
            )

            # Canonical sequence glyph strip with highlighted position
            canon_strip = _glyph_strip(
                [str(c) for c in canonical], catalog, highlight_pos=pos, size=44
            ) if catalog else ""
            canon_chips = "".join(
                f'<span class="hg-chip{"  hl" if i == pos else ""}">{code}</span>'
                for i, code in enumerate(canonical)
            )

            cards.append(
                f'<div class="hg-card">'
                f'<div class="hg-head">'
                f'<span class="hg-pid">{pid}</span>'
                f'<span class="hg-pos">Position {pos + 1 if pos >= 0 else "?"}</span>'
                f'<span class="badge bd-holy">★ Holy Grail candidate</span>'
                f'{cross_badge}</div>'
                f'{sign_pair_html}'
                f'<div class="hg-tablets" style="margin-top:8px"><b>{n_cons}</b> post-contact tablet'
                f'{"s" if n_cons != 1 else ""} show this substitution consistently.</div>'
                f'<div style="margin-top:12px">'
                f'<div class="section-label" style="margin-bottom:4px">Canonical sequence (highlighted = changed position)</div>'
                f'{canon_strip}'
                f'<div class="hg-canon" style="margin-top:4px">{canon_chips}</div>'
                f'</div>'
                f'</div>'
            )

    if not cards:
        return ""

    return (
        f'<div class="hg-section">'
        f'<div class="hg-title">★ Holy Grail Candidates</div>'
        f'<div class="hg-sub">Sign substitutions consistent across ≥ 2 post-contact tablets — '
        f'same passage slot, different glyph. Pre-contact → post-contact. '
        f'This is your strongest computational evidence for systematic change.</div>'
        f'{"".join(cards)}</div>'
    )


# ---------------------------------------------------------------------------
# Change card
# ---------------------------------------------------------------------------

def _render_change_card(change: dict, catalog: dict | None = None) -> str:
    catalog = catalog or {}
    is_holy = bool(change.get("is_holy_grail_candidate"))
    is_cross = bool(change.get("crosses_barthel_family"))
    card_cls = " holy" if is_holy else (" cross" if is_cross else "")

    tags = []
    if is_holy:
        tags.append('<span class="tag-holy">Holy Grail candidate</span>')
    if is_cross:
        tags.append('<span class="tag-cross">Family-Crossing</span>')
    if change.get("is_known_allograph"):
        tags.append('<span class="badge badge-none">Known allograph</span>')

    tag_html = " ".join(tags)
    ct = change.get("change_type", "substitution")
    pos = change.get("position", -1)
    pos_label = f"Position {pos + 1}" if pos >= 0 else "—"
    pre_sign = str(change.get("pre_contact_sign", "—"))
    post_sign = str(change.get("post_contact_sign", "—"))
    n_cons = change.get("n_tablets_consistent", 0)

    pre_glyph = _get_glyph_html(pre_sign, catalog, size=64) if catalog else None
    post_glyph = _get_glyph_html(post_sign, catalog, size=64) if catalog else None
    _pre_fallback = f'<span class="chg-sign pre-s">{pre_sign}</span>'
    _post_fallback = f'<span class="chg-sign post-s">{post_sign}</span>'
    n_tablets_str = f'{n_cons} post-contact tablet{"s" if n_cons != 1 else ""}'

    sign_pair_html = (
        f'<div class="sign-pair">'
        f'<div class="sign-slot">'
        f'<div class="sign-slot-lbl">Pre-contact</div>'
        f'<div class="sign-glyph-box pre-box" style="color:#1d4ed8">'
        f'{pre_glyph or _pre_fallback}</div>'
        f'<div class="g-code">{pre_sign}</div>'
        f'</div>'
        f'<div class="sign-arrow">→</div>'
        f'<div class="sign-slot">'
        f'<div class="sign-slot-lbl">Post-contact</div>'
        f'<div class="sign-glyph-box post-box" style="color:#6d28d9">'
        f'{post_glyph or _post_fallback}</div>'
        f'<div class="g-code">{post_sign}</div>'
        f'</div>'
        f'<div style="margin-left:16px;align-self:center;font-size:12px;color:var(--muted)">'
        f'<div class="change-field-label">Consistent across</div>'
        f'<div class="change-field-val">{n_tablets_str}</div>'
        f'</div>'
        f'</div>'
    )

    return f"""<div class="change-card{card_cls}">
  <div class="change-head">
    {_change_type_tag(ct)}
    <span class="muted small">{pos_label}</span>
    {tag_html}
  </div>
  {sign_pair_html}
</div>"""


# ---------------------------------------------------------------------------
# Attestation table
# ---------------------------------------------------------------------------

def _render_attestation_table(attestations: list[dict], catalog: dict | None = None) -> str:
    catalog = catalog or {}
    if not attestations:
        return '<p class="muted small">No attestations recorded.</p>'
    rows = []
    for att in sorted(attestations, key=lambda a: a.get("tablet", "")):
        tablet = att.get("tablet", "—")
        tablet_name = att.get("tablet_name", "")
        stratum = att.get("stratum", "unknown")
        date_range = att.get("date_range", "—")
        seq = att.get("form") or att.get("sequence") or att.get("glyphs") or []
        ed = att.get("edit_distance", "—")
        align = att.get("alignment", [])

        seq_codes = [str(c) for c in seq]
        seq_text = '<span class="attest-seq">' + " ".join(seq_codes) + "</span>" if seq else "—"
        glyph_row = _glyph_strip(seq_codes, catalog, size=40) if catalog and seq else ""
        align_html = _alignment_html(align) if align else ""

        rows.append(f"""<tr>
  <td><b>{_html.escape(str(tablet))}</b> <span class="muted small">{_html.escape(str(tablet_name))}</span></td>
  <td>{_stratum_badge(stratum)}</td>
  <td class="muted small">{_html.escape(str(date_range))}</td>
  <td>{glyph_row}{seq_text}<br>{align_html}</td>
  <td class="attest-ed">{_html.escape(str(ed))}</td>
</tr>""")

    return f"""<table class="attest-table">
<thead><tr>
  <th>Tablet</th><th>Stratum</th><th>Date range</th>
  <th>Sequence / alignment</th><th>Edit dist.</th>
</tr></thead>
<tbody>{"".join(rows)}</tbody>
</table>"""


# ---------------------------------------------------------------------------
# Single-passage detail page
# ---------------------------------------------------------------------------

def _holy_signal_line(n_holy: int) -> str:
    if not n_holy:
        return ""
    subs = "substitutions" if n_holy != 1 else "substitution"
    return (
        f'<br><span class="hack-hl">⚡ {n_holy} holy-grail {subs} detected</span>'
        " — same position, different glyph, consistent across ≥ 2 post-contact tablets."
    )


def _render_passage_page(passage: dict, catalog: dict | None = None) -> str:
    """Full standalone HTML for a single passage."""
    catalog = catalog or {}
    pid = passage.get("passage_id", "unknown")
    score = passage.get("interest_score", 0.0)
    canonical_seq = passage.get("canonical_sequence") or passage.get("canonical_form", [])
    canonical_tablet = passage.get("canonical_tablet") or passage.get("source_tablet") or ""
    canonical_stratum = passage.get("canonical_stratum") or ""
    if not canonical_tablet:
        _atts = passage.get("attestations", [])
        _pre = [a for a in _atts if a.get("stratum") == "pre_contact"]
        if _pre:
            canonical_tablet = str(_pre[0].get("tablet") or "")
            canonical_stratum = canonical_stratum or "pre_contact"
        elif _atts:
            _tab_counts: dict[str, int] = {}
            for _a in _atts:
                _t = _a.get("tablet")
                if _t:
                    _tab_counts[str(_t)] = _tab_counts.get(str(_t), 0) + 1
            if _tab_counts:
                canonical_tablet = max(_tab_counts, key=_tab_counts.__getitem__)
                _src = next((a for a in _atts if str(a.get("tablet")) == canonical_tablet), {})
                canonical_stratum = canonical_stratum or _src.get("stratum", "")
    canonical_tablet = canonical_tablet or "—"
    canonical_stratum = canonical_stratum or "unknown"
    attestations = passage.get("attestations", [])
    changes = passage.get("diachronic_changes", [])

    pre_count = sum(1 for a in attestations if a.get("stratum") == "pre_contact")
    post_count = sum(1 for a in attestations if a.get("stratum") == "post_contact")
    n_tablets = len({a.get("tablet") for a in attestations})
    n_holy = sum(1 for c in changes if c.get("is_holy_grail_candidate"))
    n_cross = sum(1 for c in changes if c.get("crosses_barthel_family"))

    has_diachronic = pre_count > 0 and post_count > 0
    diachronic_badge = (
        '<span class="badge badge-pre">Pre ✓</span> '
        '<span class="badge badge-post">Post ✓</span>'
        if has_diachronic else
        '<span class="badge badge-none">Single stratum</span>'
    )

    # Attestation / change sections
    attest_html = _render_attestation_table(attestations, catalog)

    holy_html = cross_html = other_html = ""
    holy_changes = [c for c in changes if c.get("is_holy_grail_candidate")]
    cross_changes = [c for c in changes if c.get("crosses_barthel_family") and not c.get("is_holy_grail_candidate")]
    other_changes = [c for c in changes if not c.get("is_holy_grail_candidate") and not c.get("crosses_barthel_family")]

    if holy_changes:
        holy_html = (
            '<div class="section-label" style="margin-top:20px;color:var(--holy)">'
            'Holy Grail candidates — consistent substitutions across ≥ 2 post-contact tablets</div>'
            + "".join(_render_change_card(c, catalog) for c in holy_changes)
        )
    if cross_changes:
        cross_html = (
            '<div class="section-label" style="margin-top:16px;color:var(--cross)">'
            'Family-Crossing changes — substitutions spanning Barthel century blocks</div>'
            + "".join(_render_change_card(c, catalog) for c in cross_changes)
        )
    if other_changes:
        other_html = (
            '<div class="section-label" style="margin-top:16px">'
            'Other diachronic changes</div>'
            + "".join(_render_change_card(c, catalog) for c in other_changes)
        )

    no_changes_note = (
        '<p class="muted small" style="margin-top:16px">No diachronic changes detected '
        'in this passage — either the stratum signal is absent or all attestations '
        'match the canonical form.</p>'
        if not changes else ""
    )

    generated = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>hackingrongo — Passage {pid}</title>
<link href="https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,400;0,600;1,400&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>{_CSS}</style>
</head>
<body>
<div class="wrap">

<div class="report-header">
  <div class="report-title">Passage {pid}</div>
  <div class="report-subtitle">Parallel passage alignment — diachronic analysis</div>
  <div class="report-meta">
    <b>Canonical source:</b> Tablet {canonical_tablet}
    ({_STRATUM_LABEL.get(canonical_stratum, canonical_stratum)})
    &nbsp;·&nbsp;
    <b>Interest score:</b> {score:.2f}
    &nbsp;·&nbsp;
    <b>Attestations:</b> {len(attestations)} forms across {n_tablets} tablet{"s" if n_tablets != 1 else ""}
    &nbsp;·&nbsp;
    <b>Diachronic signal:</b> {diachronic_badge}
    &nbsp;·&nbsp;
    <b>Generated:</b> {generated}
  </div>
</div>

<div class="hack-box">
  <div class="hack-title">// what you're looking at</div>
  This glyph sequence appears in <b>{n_tablets} tablet{"s" if n_tablets != 1 else ""}</b> — the algorithm found it by aligning rongorongo tablets pairwise and clustering matching subsequences.
  Each row in the attestation table below is one tablet's version of the same passage.
  {_holy_signal_line(n_holy)}
  {('<br>Pre-contact tablets (' + str(pre_count) + ' attestations) and post-contact tablets (' + str(post_count) + ' attestations) are compared at each position.') if has_diachronic else ''}
</div>

<div class="stats-row">
  <div class="stat-card">
    <div class="stat-value">{len(attestations)}</div>
    <div class="stat-label">attestations</div>
  </div>
  <div class="stat-card">
    <div class="stat-value">{pre_count}</div>
    <div class="stat-label">pre-contact</div>
  </div>
  <div class="stat-card">
    <div class="stat-value">{post_count}</div>
    <div class="stat-label">post-contact</div>
  </div>
  <div class="stat-card">
    <div class="stat-value">{len(canonical_seq)}</div>
    <div class="stat-label">canonical length</div>
  </div>
  <div class="stat-card holy">
    <div class="stat-value">{n_holy}</div>
    <div class="stat-label">holy-grail candidates</div>
  </div>
  <div class="stat-card cross">
    <div class="stat-value">{n_cross}</div>
    <div class="stat-label">family-crossing changes</div>
  </div>
</div>

<div class="section-label">Canonical sequence (from Tablet {canonical_tablet})</div>
{_glyph_strip([str(c) for c in canonical_seq], catalog)}
{_sign_seq_html(canonical_seq)}

<div class="section-label" style="margin-top:24px">All attestations</div>
{attest_html}

<div class="changes-section">
  <div class="section-label">Diachronic change analysis</div>
  {holy_html}
  {cross_html}
  {other_html}
  {no_changes_note}
</div>

<div class="report-footer">
  <p><a href="index.html">← Back to passage summary</a></p>
  <p><b>hackingrongo</b> · Parallel passage alignment report · MIT License</p>
  <p>Alignment: Needleman-Wunsch global alignment, diagonal-first tie-breaking.
  Diachronic analysis: consensus sign per stratum; holy-grail criterion requires
  non-allographic substitution consistent across ≥ 2 post-contact tablets.
  Family-Crossing: pre- and post-contact consensus signs in different Barthel
  century blocks (1–199, 200–299, …, 700–799).</p>
</div>

</div>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Diachronic cross-passage summary page (the main report)
# ---------------------------------------------------------------------------

def _render_summary_page(passages: list[dict], meta: dict[str, Any], catalog: dict | None = None) -> str:
    """Render the interactive single-page diachronic passage summary."""
    catalog = catalog or {}
    generated = meta.get(
        "generated", datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    )
    source_file = meta.get("source_file", "—")

    n_passages = len(passages)
    all_attestations = [a for p in passages for a in p.get("attestations", [])]
    all_changes = [c for p in passages for c in p.get("diachronic_changes", [])]
    n_diachronic = sum(
        1 for p in passages
        if any(a.get("stratum") == "pre_contact" for a in p.get("attestations", []))
        and any(a.get("stratum") == "post_contact" for a in p.get("attestations", []))
    )
    n_holy = sum(1 for c in all_changes if c.get("is_holy_grail_candidate"))
    n_cross = sum(1 for c in all_changes if c.get("crosses_barthel_family"))
    all_tablets = sorted({a.get("tablet", "") for a in all_attestations if a.get("tablet")})

    # ── Stats cards ──────────────────────────────────────────────────────────
    stats_html = (
        f'<div class="stats-row">'
        f'<div class="stat-card"><div class="stat-value">{n_passages}</div>'
        f'<div class="stat-label">Parallel passages</div></div>'
        f'<div class="stat-card pre"><div class="stat-value">{n_diachronic}</div>'
        f'<div class="stat-label">With pre+post signal</div></div>'
        f'<div class="stat-card"><div class="stat-value">{len(all_tablets)}</div>'
        f'<div class="stat-label">Tablets covered</div></div>'
        f'<div class="stat-card"><div class="stat-value">{len(all_changes)}</div>'
        f'<div class="stat-label">Diachronic changes</div></div>'
        f'<div class="stat-card holy"><div class="stat-value">{n_holy}</div>'
        f'<div class="stat-label">Holy-Grail candidates</div></div>'
        f'<div class="stat-card cross"><div class="stat-value">{n_cross}</div>'
        f'<div class="stat-label">Family-Crossing changes</div></div>'
        f'</div>'
    )

    # ── Holy-grail spotlight ─────────────────────────────────────────────────
    hg_html = _render_holy_grail_spotlight(passages, catalog)

    # ── Filter bar ───────────────────────────────────────────────────────────
    n_dia_chip = n_diachronic
    n_holy_chip = sum(
        1 for p in passages
        if any(c.get("is_holy_grail_candidate") for c in p.get("diachronic_changes", []))
    )
    n_cross_chip = sum(
        1 for p in passages
        if any(c.get("crosses_barthel_family") for c in p.get("diachronic_changes", []))
    )
    filter_bar = (
        f'<div class="filter-bar">'
        f'<input type="search" class="fb-search" placeholder="Search passages…"'
        f' oninput="searchPassages(this.value)">'
        f'<div class="filter-chips">'
        f'<button class="chip active" onclick="setFilter(\'all\',this)">All ({n_passages})</button>'
        f'<button class="chip" onclick="setFilter(\'diachronic\',this)">Diachronic ({n_dia_chip})</button>'
        f'<button class="chip c-holy" onclick="setFilter(\'holygrail\',this)">Holy Grail ({n_holy_chip})</button>'
        f'<button class="chip c-cross" onclick="setFilter(\'familycross\',this)">Family-Crossing ({n_cross_chip})</button>'
        f'</div>'
        f'<div class="fb-right">'
        f'<select class="fb-sort" onchange="setSort(this.value)">'
        f'<option value="score">↓ Interest score</option>'
        f'<option value="tablets">↓ Tablet count</option>'
        f'<option value="changes">↓ Changes</option>'
        f'<option value="pid">↑ Passage ID</option>'
        f'</select>'
        f'<span id="fb-count" class="fb-count">Showing {n_passages} of {n_passages}</span>'
        f'<button class="fb-btn" onclick="expandAll()">Expand all</button>'
        f'<button class="fb-btn" onclick="collapseAll()">Collapse all</button>'
        f'</div></div>'
    )

    # ── Passage cards ────────────────────────────────────────────────────────
    sorted_passages = sorted(
        passages, key=lambda p: p.get("interest_score", 0), reverse=True
    )
    cards = "".join(
        _render_passage_card(p, i) for i, p in enumerate(sorted_passages, 1)
    )

    # ── Abstract ─────────────────────────────────────────────────────────────
    abstract = (
        f'<div class="abstract">'
        f'<p><b>TL;DR:</b> {n_passages} glyph sequences appear verbatim across multiple rongorongo tablets. '
        f'Align them, compare pre-contact (Tablet D, ~1500 CE) vs post-contact tablets, '
        f'and some positions consistently show a <em>different</em> glyph — that\'s a Holy Grail candidate. '
        f'We found <b>{n_holy}</b>.</p>'
        f'<p><b>Holy Grail candidate:</b> same passage position, different glyph, consistent across '
        f'≥ 2 independent post-contact tablets. Idiosyncratic scribal error can\'t explain it. '
        f'<b>Family-Crossing:</b> the pre- and post-contact glyphs are in different Barthel century '
        f'blocks — e.g. a bird-headed sign replaced by an object. Iconographically surprising. '
        f'There are <b>{n_cross}</b> such changes.</p>'
        f'<p>Click any passage row to expand its attestation alignment. '
        f'The <span style="color:var(--holy)">★ Holy Grail</span> section above shows glyphs.</p>'
        f'</div>'
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>hackingrongo — Parallel Passage Analysis</title>
<link href="https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,400;0,600;1,400&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>{_CSS}</style>
</head>
<body>
<div class="wrap">

<div class="report-header">
  <div class="report-title">Parallel Passage<br>Diachronic Analysis</div>
  <div class="report-subtitle">Cross-contact sign changes in repeated rongorongo sequences</div>
  <div class="report-meta">
    <b>Passages:</b> {n_passages} &nbsp;·&nbsp;
    <b>With diachronic signal:</b> {n_diachronic} &nbsp;·&nbsp;
    <b>Holy-Grail candidates:</b> {n_holy} &nbsp;·&nbsp;
    <b>Tablets covered:</b> {len(all_tablets)} ({", ".join(all_tablets)}) &nbsp;·&nbsp;
    <b>Source:</b> {_html.escape(str(source_file))} &nbsp;·&nbsp;
    <b>Generated:</b> {generated}
  </div>
  {abstract}
</div>

{stats_html}

{hg_html}

<div class="section-label">All parallel passages</div>
{filter_bar}

<div id="passages">
{cards}
</div>

<div class="report-footer">
  <p><b>hackingrongo</b> · Parallel passage alignment report · MIT License</p>
  <p>Alignment: position-based with majority-vote consensus per tablet.
  Levenshtein threshold = 1 for passage matching against corpus sequences.</p>
  <p>Pre-contact anchor: Tablet D (radiocarbon 1493–1509 CE, Ferrara et al. 2024).
  Post-contact tablets: B, C, E, G, H, I, K, P, Q, S and others (RC min ≥ 1650 CE).
  Holy-Grail criterion: non-allographic substitution consistent in ≥ 2 post-contact
  tablets at the same canonical position (Barthel 1958 allograph catalog).
  Family-Crossing: pre/post consensus signs in different Barthel century blocks.</p>
  <p>This is a computational hypothesis report. All candidates require expert
  epigraphic and linguistic review before any interpretive claim.</p>
  <p><b>SperksWerks LLC</b> ·
  <a href="https://sperkswerks.ai" target="_blank">sperkswerks.ai</a></p>
</div>

</div>
<script>{_JS}</script>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Public API — module-level functions
# ---------------------------------------------------------------------------

def build_passage_report(
    passages_json: Path,
    filter_interest_score: float = 0.0,
) -> str:
    """Build the diachronic cross-passage summary report HTML.

    Parameters
    ----------
    passages_json : Path
        JSON file written by ``passage_alignment.py`` (top-level dict with
        ``"passages"`` key, or a bare list of passage dicts).
    filter_interest_score : float
        Only include passages with interest_score >= this value.

    Returns
    -------
    str
        Complete HTML document string for the summary page.
    """
    raw = json.loads(passages_json.read_text(encoding="utf-8"))
    if isinstance(raw, list):
        passages = raw
    else:
        passages = raw.get("passages", raw.get("alignments", []))

    if filter_interest_score > 0.0:
        passages = [p for p in passages if p.get("interest_score", 0) >= filter_interest_score]

    if not passages:
        logger.warning(
            "passages list is empty in %s. "
            "Check that cross_reference_parallels.py completed successfully "
            "and wrote schema_version 2.0. File size: %d bytes.",
            passages_json.name,
            passages_json.stat().st_size,
        )

    catalog = _load_svg_catalog()
    meta = {
        "generated": datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "source_file": passages_json.name,
    }
    logger.info("Building passage summary report: %d passages.", len(passages))
    return _render_summary_page(passages, meta, catalog)


def build_single_passage_report(passage: dict) -> str:
    """Build the detail-page HTML for a single passage dict.

    Parameters
    ----------
    passage : dict
        One passage dict as produced by ``passage_alignment.py``.

    Returns
    -------
    str
        Complete HTML document string.
    """
    catalog = _load_svg_catalog()
    return _render_passage_page(passage, catalog)


def save_passage_report(
    passages_json: Path,
    output_path: Path,
    filter_interest_score: float = 0.0,
) -> None:
    """Generate and write the diachronic cross-passage summary report.

    Parameters
    ----------
    passages_json : Path
        Input JSON file.
    output_path : Path
        Destination HTML file.  Parent directories are created if needed.
    filter_interest_score : float
        Only include passages at or above this interest score.
    """
    html = build_passage_report(passages_json, filter_interest_score=filter_interest_score)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    logger.info("Passage summary report written: %s (%d bytes).", output_path, len(html))


# ---------------------------------------------------------------------------
# Backward-compatible class interface
# ---------------------------------------------------------------------------

class PassageReportGenerator:
    """Backward-compatible interface.

    Writes per-passage HTML files and a summary index (``index.html``).
    The index is now the full diachronic cross-passage summary report.
    """

    def render_passage(self, passage: dict) -> str:
        """Return the detail-page HTML for a single passage dict."""
        return _render_passage_page(passage, _load_svg_catalog())

    def generate_report(
        self,
        passages_json: Path,
        output_dir: Path,
        filter_interest_score: float = 0.0,
        individual_files: bool = True,
    ) -> None:
        """Generate HTML reports from the passages JSON file.

        Writes ``{output_dir}/index.html`` (diachronic summary) and,
        when ``individual_files=True``, one HTML file per passage.
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        raw = json.loads(passages_json.read_text(encoding="utf-8"))
        if isinstance(raw, list):
            all_passages = raw
        else:
            all_passages = raw.get("passages", raw.get("alignments", []))

        filtered = [
            p for p in all_passages
            if p.get("interest_score", 0) >= filter_interest_score
        ]
        logger.info(
            "Loaded %d passages, filtered to %d (interest >= %.2f).",
            len(all_passages), len(filtered), filter_interest_score,
        )
        if not all_passages:
            logger.warning(
                "passages list is empty in %s. "
                "Check that cross_reference_parallels.py completed successfully "
                "and wrote schema_version 2.0. File size: %d bytes.",
                passages_json.name,
                passages_json.stat().st_size,
            )

        catalog = _load_svg_catalog()

        if individual_files:
            for passage in filtered:
                pid = passage.get("passage_id", "unknown")
                html = _render_passage_page(passage, catalog)
                out = output_dir / f"{pid}.html"
                out.write_text(html, encoding="utf-8")
                logger.info("  %s → %s", pid, out.name)

        meta = {
            "generated": datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
            "source_file": passages_json.name,
        }
        index_html = _render_summary_page(filtered, meta, catalog)
        index_file = output_dir / "index.html"
        index_file.write_text(index_html, encoding="utf-8")
        logger.info("Summary index → %s", index_file)


# ---------------------------------------------------------------------------
# CLI entry point  (python -m hackingrongo.results.passage_report)
# ---------------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

    p = argparse.ArgumentParser(
        description="Generate parallel passage HTML reports from parallel_variants JSON."
    )
    p.add_argument(
        "--input",
        type=Path,
        required=True,
        metavar="JSON",
        help="Input JSON produced by cross_reference_parallels.py.",
    )
    p.add_argument(
        "--output",
        type=Path,
        required=True,
        metavar="DIR",
        help="Output directory; receives index.html and one HTML per passage.",
    )
    p.add_argument(
        "--filter-score",
        type=float,
        default=0.0,
        metavar="SCORE",
        help="Exclude passages below this interest_score (default: 0 = show all).",
    )
    args = p.parse_args()

    if not args.input.exists():
        logger.error("Input not found: %s", args.input)
        sys.exit(1)

    PassageReportGenerator().generate_report(
        args.input,
        args.output,
        filter_interest_score=args.filter_score,
    )


if __name__ == "__main__":
    main()
