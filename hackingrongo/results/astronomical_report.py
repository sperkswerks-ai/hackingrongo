"""
hackingrongo.results.astronomical_report
==========================================

Scholar-facing HTML report for the Zone B astronomical hypothesis tests.

Shows the top-ranked sign candidates with their glyph drawings, per-test
scores, and Polynesian star name correspondences from the Dietrich lookup
table.

Public API
----------
``build_astronomical_report(candidates_path, svg_catalog_path)`` → HTML str
``save_astronomical_report(candidates_path, svg_catalog_path, output_path)``

Design language
---------------
Matches compound_report, divergence_report, passage_report:
  * Light background — CSS variables --bg / --surface / --surface2
  * Cormorant Garamond (body) + JetBrains Mono (code / metadata)
  * Accent colour --accent = #c4a96d (gold)
"""

from __future__ import annotations

import base64
import json
import logging
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Shared CSS
# ---------------------------------------------------------------------------

_CSS = """\
:root {
  --bg: #ffffff; --surface: #f8f8fa; --surface2: #f0f0f5;
  --border: #d0d0dd; --text: #1a1a1a; --muted: #666666;
  --accent: #c4a96d; --accent2: #7b9ee0;
  --star: #e8a838; --astro: #5b8dd9;
}
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
  background: var(--bg); color: var(--text);
  font-family: 'Cormorant Garamond', 'Palatino Linotype', Georgia, serif;
  font-size: 16px; line-height: 1.65;
}
.wrap { max-width: 1100px; margin: 0 auto; padding: 52px 28px; }

.report-header { border-bottom: 1px solid var(--border);
                 padding-bottom: 38px; margin-bottom: 48px; }
.report-title { font-size: 34px; font-weight: 600; color: #000; letter-spacing: -0.3px; }
.report-subtitle { font-size: 17px; color: var(--accent); font-style: italic; margin-top: 6px; }
.report-meta { margin-top: 22px; font-family: 'JetBrains Mono', monospace;
               font-size: 11px; color: var(--muted); line-height: 2.2; }
.report-meta b { color: #333; }
.abstract { margin-top: 22px; font-size: 14.5px; color: #333333;
            max-width: 760px; line-height: 1.85; }
.abstract p + p { margin-top: 10px; }

.legend { display: flex; flex-wrap: wrap; gap: 10px;
          margin: 28px 0 44px; align-items: center; }
.legend-label { font-size: 12px; color: var(--muted); margin-right: 4px; }
.legend-chip { font-family: 'JetBrains Mono', monospace; font-size: 10px;
               border-radius: 3px; padding: 3px 9px; }

.entry { background: var(--surface); border: 1px solid var(--border);
         border-radius: 8px; margin-bottom: 32px; overflow: hidden; }
.entry-header { padding: 16px 22px 10px; display: flex; align-items: center;
                gap: 14px; flex-wrap: wrap; }
.rank-badge { font-family: 'JetBrains Mono', monospace; font-size: 11px;
              color: var(--muted); min-width: 28px; }
.entry-title { display: flex; align-items: center; gap: 10px; flex: 1; }
.code-label { font-family: 'JetBrains Mono', monospace; font-size: 14px;
              color: var(--accent); font-weight: 500; }
.tier-tag { font-family: 'JetBrains Mono', monospace; font-size: 9.5px;
            border-radius: 3px; padding: 2px 8px; white-space: nowrap; }
.score-block { display: flex; align-items: center; gap: 10px; }
.score-value { font-family: 'JetBrains Mono', monospace; font-size: 18px;
               color: #000; font-weight: 500; min-width: 48px; }
.score-bar { font-family: 'JetBrains Mono', monospace; font-size: 10px;
             color: var(--accent); letter-spacing: -1px; }
.methods-tag { font-family: 'JetBrains Mono', monospace; font-size: 9.5px;
               color: var(--muted); }
.bar-accent { height: 3px; }

.entry-body { display: grid; grid-template-columns: 200px 260px 1fr; }
.glyph-col, .dietrich-col, .tests-col {
  padding: 20px 22px; border-right: 1px solid var(--border);
}
.tests-col { border-right: none; }
.section-label { font-family: 'JetBrains Mono', monospace; font-size: 9px;
                 color: var(--muted); letter-spacing: 0.1em;
                 text-transform: uppercase; margin-bottom: 10px; }

.main-svg { display: flex; align-items: center; justify-content: center;
            background: var(--surface2); border: 1px solid var(--border);
            border-radius: 5px; padding: 10px; min-height: 108px;
            color: var(--accent); }
.main-missing { color: var(--muted); font-size: 11px; }
.glyph-code { font-family: 'JetBrains Mono', monospace; font-size: 11px;
              color: #333; margin-top: 8px; }
.glyph-freq { font-size: 11px; color: var(--muted); margin-top: 2px; }

.dietrich-item { margin-bottom: 10px; }
.dietrich-label { font-family: 'JetBrains Mono', monospace; font-size: 9px;
                  color: var(--muted); letter-spacing: 0.08em;
                  text-transform: uppercase; }
.dietrich-value { font-size: 13px; color: #222; margin-top: 1px; }
.dietrich-star { color: var(--star); font-style: italic; }
.dietrich-none { color: var(--muted); font-size: 12px; font-style: italic; margin-top: 6px; }
.dietrich-conf { font-family: 'JetBrains Mono', monospace; font-size: 9px;
                 border-radius: 3px; padding: 2px 7px; }

.test-row { display: flex; align-items: center; gap: 10px;
            margin-bottom: 10px; flex-wrap: wrap; }
.test-name { font-family: 'JetBrains Mono', monospace; font-size: 10px;
             color: var(--muted); width: 200px; flex-shrink: 0; }
.test-bar-wrap { flex: 1; background: var(--surface2); border: 1px solid var(--border);
                 border-radius: 3px; height: 8px; min-width: 80px; overflow: hidden; }
.test-bar-fill { height: 100%; border-radius: 3px; }
.test-score { font-family: 'JetBrains Mono', monospace; font-size: 10px;
              color: #333; min-width: 36px; text-align: right; }
.test-na { font-family: 'JetBrains Mono', monospace; font-size: 10px;
           color: var(--muted); font-style: italic; }
.test-detail { font-size: 11.5px; color: #444; margin-top: 2px;
               padding-left: 212px; line-height: 1.6; margin-bottom: 6px; }
.test-detail code { font-family: 'JetBrains Mono', monospace; font-size: 10px;
                    background: var(--surface2); border: 1px solid var(--border);
                    border-radius: 2px; padding: 1px 4px; }

.report-footer { border-top: 1px solid var(--border); margin-top: 52px;
                 padding-top: 26px; font-size: 12px; color: var(--muted); line-height: 2.0; }
.report-footer a { color: var(--accent); text-decoration: none; }

@media (max-width: 820px) {
  .entry-body { grid-template-columns: 1fr; }
  .glyph-col, .dietrich-col {
    border-right: none; border-bottom: 1px solid var(--border);
  }
  .test-name { width: 140px; }
  .test-detail { padding-left: 0; }
}
"""

# ---------------------------------------------------------------------------
# Test metadata (label, short description for tooltip / detail line)
# ---------------------------------------------------------------------------

_TEST_META: list[tuple[str, str, str]] = [
    ("test1_positional_entropy",   "T1 · positional entropy",
     "Constrained position in sequence → proper name / astronomical marker"),
    ("test2_calendar_anchor",       "T2 · Mamari calendar",
     "Prevalence in Mamari Ca6–Ca9 (the known lunar calendar section)"),
    ("test3_tablet_stability",      "T3 · cross-tablet stability",
     "Low Jensen-Shannon divergence of positional distribution across tablets"),
    ("test4_dietrich_match",        "T4 · Dietrich table",
     "Match in the Dietrich (2007) / Fischer (1997) star-name correspondence table"),
    ("test5_tablet_d_specificity",  "T5 · Tablet D specificity",
     "Overrepresentation on Tablet D (oldest pre-contact tablet)"),
]

# ---------------------------------------------------------------------------
# SVG helpers — same pattern as compound_report.py
# ---------------------------------------------------------------------------


def _load_svg_catalog(catalog_path: Path) -> dict[str, list[Path]]:
    """Return {barthel_code: [svg_path, ...]} for all valid SVG files.

    Lookup is attempted in order:
    1. Exact code match (e.g. ``'661!'``)
    2. Base code — trailing variant/modifier chars stripped (e.g. ``'661'``)
    """
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

    merged: dict[str, list[Path]] = dict(exact)
    for base_code, paths in base_map.items():
        if base_code not in merged:
            merged[base_code] = paths

    # PNG fallback: barthel_catalog.json covers codes not in the SVG-scraped tablets.
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
        logger.info(
            "barthel_catalog fallback: %d PNG codes added.",
            sum(1 for c in png_stage if c not in exact),
        )
    else:
        logger.warning("barthel_catalog.json not found — PNG fallback disabled.")

    return merged


def _normalise_svg(svg_text: str, size: int = 88) -> str:
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


def _get_svg(code: str, catalog: dict[str, list[Path]], size: int = 88) -> str | None:
    """Return an HTML fragment (inline SVG or base64 img). Tries exact code, modifier-stripped base, then all-alpha-stripped numeric base."""
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
                f'style="max-width:{size}px;max-height:{size}px;'
                f'display:block;margin:auto;" '
                f'alt="Barthel {code}">'
            )
        return _normalise_svg(path.read_text(encoding="utf-8"), size=size)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Score bar rendering
# ---------------------------------------------------------------------------


def _score_colour(score: float) -> str:
    if score >= 0.75:
        return "#4caf7d"
    if score >= 0.55:
        return "#8bc34a"
    if score >= 0.35:
        return "#d4a817"
    return "#e07b54"


def _tier_tag(n_methods: int, overall: float) -> tuple[str, str]:
    """Return (label, colour) for the confidence tier chip."""
    if n_methods >= 4:
        return "STRONG — 4-5 tests", "#4caf7d"
    if n_methods == 3:
        return "MODERATE — 3 tests", "#8bc34a"
    if n_methods == 2:
        return "TENTATIVE — 2 tests", "#d4a817"
    return "WEAK — 1 test", "#e07b54"


def _render_test_row(key: str, label: str, desc: str, test_data: dict[str, Any]) -> str:
    score = test_data.get("score")
    if score is None:
        reason = test_data.get("reason", "—")
        return f"""
<div class="test-row">
  <span class="test-name">{label}</span>
  <span class="test-na">n/a — {reason}</span>
</div>"""

    pct = int(score * 100)
    colour = _score_colour(score)
    bar_fill = f'<div class="test-bar-fill" style="width:{pct}%;background:{colour}"></div>'

    # Build a one-line detail from available keys
    detail_parts: list[str] = []
    if key == "test1_positional_entropy":
        h = test_data.get("positional_entropy_bits")
        bm = test_data.get("baseline_mean_bits")
        z = test_data.get("z_score")
        n = test_data.get("n_occurrences")
        if h is not None:
            detail_parts.append(f"H={h:.3f} bits (baseline {bm:.3f}) · z={z:+.2f} · n={n}")
    elif key == "test2_calendar_anchor":
        nc = test_data.get("n_calendar", 0)
        nn = test_data.get("n_non_calendar", 0)
        frac = test_data.get("calendar_fraction", 0.0)
        excl = test_data.get("calendar_exclusive", False)
        detail_parts.append(
            f"cal={nc} non-cal={nn} ({frac*100:.0f}%)"
            + (" · calendar-exclusive" if excl else "")
        )
    elif key == "test3_tablet_stability":
        nt = test_data.get("n_tablets", 0)
        js = test_data.get("mean_js_divergence")
        if js is not None:
            detail_parts.append(f"{nt} tablets · mean JS={js:.3f}")
    elif key == "test4_dietrich_match":
        if test_data.get("in_dietrich_table"):
            ref = test_data.get("proposed_referent", "")
            poly = test_data.get("polynesian_name", "")
            src = test_data.get("source", "")
            detail_parts.append(f'“{ref}” · {poly} · {src}')
    elif key == "test5_tablet_d_specificity":
        fd = test_data.get("f_tablet_d", 0)
        ft = test_data.get("f_total", 0)
        ratio = test_data.get("ratio")
        if ratio is not None:
            detail_parts.append(f"f_D={fd} / f_total={ft} · ratio={ratio:.2f}×")

    detail_html = ""
    if detail_parts:
        detail_html = (
            f'<div class="test-detail"><code>{desc}</code> — {detail_parts[0]}</div>'
        )

    return f"""
<div class="test-row">
  <span class="test-name">{label}</span>
  <div class="test-bar-wrap">{bar_fill}</div>
  <span class="test-score">{score:.2f}</span>
</div>{detail_html}"""


def _render_entry(
    rank: int,
    cand: dict[str, Any],
    catalog: dict[str, list[Path]],
    freq_lookup: dict[str, int],
) -> str:
    code = cand["barthel_code"]
    overall = cand.get("overall_score", 0.0)
    n_methods = cand.get("n_methods_flagged", 0)
    dietrich = cand.get("dietrich_entry")

    tier_label, tier_colour = _tier_tag(n_methods, overall)
    bar_filled = "█" * int(overall * 10)
    bar_empty = "░" * (10 - len(bar_filled))

    svg = _get_svg(code, catalog)
    glyph_html = (
        f'<div class="main-svg">{svg}</div>'
        if svg
        else '<div class="main-svg"><span class="main-missing">no SVG</span></div>'
    )
    freq = freq_lookup.get(code, 0)

    # Dietrich column
    if dietrich:
        conf = dietrich.get("confidence", "speculative")
        conf_colours = {"high": "#4caf7d", "medium": "#d4a817", "low": "#e07b54"}
        conf_colour = conf_colours.get(conf, "#888")
        dietrich_html = f"""
<div class="dietrich-item">
  <div class="dietrich-label">Proposed referent</div>
  <div class="dietrich-value">{dietrich.get('proposed_referent', '—')}</div>
</div>
<div class="dietrich-item">
  <div class="dietrich-label">Polynesian name</div>
  <div class="dietrich-value dietrich-star">{dietrich.get('polynesian_name', '—')}</div>
</div>
<div class="dietrich-item">
  <div class="dietrich-label">Western equivalent</div>
  <div class="dietrich-value">{dietrich.get('western_equivalent', '—')}</div>
</div>
<div class="dietrich-item">
  <div class="dietrich-label">Source</div>
  <div class="dietrich-value" style="font-size:12px;color:var(--muted)">{dietrich.get('source', '—')}</div>
</div>
<div style="margin-top:8px">
  <span class="dietrich-conf"
        style="background:{conf_colour}22;color:{conf_colour};border:1px solid {conf_colour}55">
    {conf}
  </span>
</div>"""
    else:
        dietrich_html = '<p class="dietrich-none">Not in Dietrich correspondence table</p>'

    # Test rows
    tests_html = "".join(
        _render_test_row(key, label, desc, cand.get(key, {}))
        for key, label, desc in _TEST_META
    )

    bar_colour = _score_colour(overall)
    return f"""
<div class="entry">
  <div class="entry-header">
    <span class="rank-badge">#{rank:02d}</span>
    <div class="entry-title">
      <span class="code-label">{code}</span>
      <span class="tier-tag"
            style="background:{tier_colour}22;color:{tier_colour};border:1px solid {tier_colour}55">
        {tier_label}
      </span>
    </div>
    <div class="score-block">
      <span class="score-value">{overall:.2f}</span>
      <span class="score-bar">{bar_filled}{bar_empty}</span>
      <span class="methods-tag">{n_methods}/5 tests flagged</span>
    </div>
  </div>
  <div class="bar-accent" style="background:{bar_colour};opacity:0.7"></div>
  <div class="entry-body">
    <div class="glyph-col">
      <div class="section-label">Glyph</div>
      {glyph_html}
      <div class="glyph-code">{code}</div>
      <div class="glyph-freq">corpus freq: {freq}</div>
    </div>
    <div class="dietrich-col">
      <div class="section-label">Astronomical correspondence</div>
      {dietrich_html}
    </div>
    <div class="tests-col">
      <div class="section-label">Test scores</div>
      {tests_html}
    </div>
  </div>
</div>"""


# ---------------------------------------------------------------------------
# Full HTML document
# ---------------------------------------------------------------------------


def _render_html(
    candidates: list[dict[str, Any]],
    catalog: dict[str, list[Path]],
    report_meta: dict[str, Any],
) -> str:
    n_total = report_meta.get("n_candidates", len(candidates))
    generated = report_meta.get("generated", "—")

    # Build corpus frequency lookup from all_evaluated list
    freq_lookup: dict[str, int] = {}
    for c in report_meta.get("all_evaluated", []):
        t5 = c.get("test5_tablet_d_specificity", {})
        f = t5.get("f_total")
        if f is not None:
            freq_lookup[c["barthel_code"]] = int(f)

    entries_html = "".join(
        _render_entry(rank, cand, catalog, freq_lookup)
        for rank, cand in enumerate(candidates, start=1)
    )

    legend_chips = "".join(
        f'<span class="legend-chip" '
        f'style="background:{colour}22;color:{colour};border:1px solid {colour}55">'
        f'{label}</span>'
        for label, colour in [
            ("STRONG ≥ 4 tests", "#4caf7d"),
            ("MODERATE — 3 tests", "#8bc34a"),
            ("TENTATIVE — 2 tests", "#d4a817"),
        ]
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>hackingrongo — Astronomical Sign Candidates</title>
<link href="https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,400;0,600;1,400&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>{_CSS}</style>
</head>
<body>
<div class="wrap">

<div class="report-header">
  <div class="report-title">hackingrongo<br>Astronomical Sign Candidates</div>
  <div class="report-subtitle">
    Statistical evidence for astronomical referents in the rongorongo corpus
  </div>
  <div class="report-meta">
    <b>Candidates:</b> {n_total} &nbsp;·&nbsp;
    <b>Tests run:</b> 5 &nbsp;·&nbsp;
    <b>Generated:</b> {generated}
  </div>
  <div class="abstract">
    <p>The <em>astronomical hypothesis</em> proposes that a subset of rongorongo signs
    encode star names, constellation markers, and navigational references used in
    traditional Polynesian wayfinding. The central claim, articulated by Dietrich (2007)
    and Fischer (1997), is that bird-headed signs (Barthel 600–699) correspond to
    bird constellations (Manu, Keoe, Humu) and that the Mamari tablet's Ca6–Ca9
    section encodes a lunar calendar — making those lines the most astronomically
    dense passage in the corpus.</p>
    <p>This report presents five independent statistical tests applied to all
    bird-headed signs and Dietrich table entries. No test alone is diagnostic;
    convergence across tests is the signal. <b>Signs flagged by three or more
    independent tests should be prioritised for epigraphic review.</b></p>
    <p><b>Caution:</b> all correspondences are speculative. The statistical tests
    establish structural regularity, not meaning. Every candidate requires
    expert review by an epigraphist familiar with the Polynesian astronomical
    tradition.</p>
  </div>
</div>

<div class="legend">
  <span class="legend-label">Confidence tier:</span>
  {legend_chips}
</div>

{entries_html}

<div class="report-footer">
  <p><b>hackingrongo</b> · Astronomical hypothesis tests · MIT License</p>
  <p>Tests: (T1) positional entropy · (T2) Mamari Ca6–Ca9 calendar anchor ·
  (T3) cross-tablet Jensen-Shannon stability · (T4) Dietrich correspondence table ·
  (T5) Tablet D pre-contact overrepresentation.</p>
  <p>References: Barthel (1958) <em>Grundlagen zur Entzifferung der Osterinselschrift</em> ·
  Dietrich (2007) "Star Names in Rongorongo" · Fischer (1997) <em>Rongorongo: The Easter
  Island Script</em> · Wieczorek (2016) "Astronomical Encodings in Easter Island Script".</p>
  <p>This is a computational hypothesis report, not a decipherment claim.</p>
</div>

</div>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_astronomical_report(
    candidates_path: Path,
    svg_catalog_path: Path,
    max_candidates: int = 30,
) -> str:
    """Build the astronomical candidates report HTML.

    Parameters
    ----------
    candidates_path : Path
        ``astronomical_candidates.json`` written by
        ``zone_b.astronomical_analysis.run_all_tests()``.
    svg_catalog_path : Path
        ``data/glyphs/svg/catalog.json``.
    max_candidates : int
        Maximum candidates to render (ranked by n_methods_flagged, overall_score).

    Returns
    -------
    str
        Complete HTML document.
    """
    if not candidates_path.exists():
        raise FileNotFoundError(
            f"Astronomical candidates file not found: {candidates_path}\n"
            "Run zone_b.astronomical_analysis first."
        )

    data = json.loads(candidates_path.read_text(encoding="utf-8"))
    candidates = sorted(
        data.get("candidates", []),
        key=lambda c: (-c.get("n_methods_flagged", 0), -c.get("overall_score", 0)),
    )[:max_candidates]

    catalog = _load_svg_catalog(svg_catalog_path)
    logger.info(
        "Building astronomical report: %d candidates, %d SVG codes.",
        len(candidates), len(catalog),
    )

    report_meta = {**data, "candidates": candidates}
    return _render_html(candidates, catalog, report_meta)


def save_astronomical_report(
    candidates_path: Path,
    svg_catalog_path: Path,
    output_path: Path,
    max_candidates: int = 30,
) -> None:
    """Generate and write the astronomical report to an HTML file.

    Parameters
    ----------
    candidates_path : Path
    svg_catalog_path : Path
    output_path : Path
        Destination ``.html`` file.  Parent directories are created.
    max_candidates : int
    """
    html = build_astronomical_report(
        candidates_path, svg_catalog_path, max_candidates=max_candidates
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    logger.info(
        "Astronomical report written: %s (%d bytes).", output_path, len(html)
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args():
    import argparse

    p = argparse.ArgumentParser(
        description="Generate the astronomical sign candidates HTML report."
    )
    p.add_argument(
        "--candidates",
        type=Path,
        required=True,
        metavar="PATH",
        help="Path to astronomical_candidates.json.",
    )
    p.add_argument(
        "--svg-catalog",
        type=Path,
        required=True,
        metavar="PATH",
        help="Path to data/glyphs/svg/catalog.json.",
    )
    p.add_argument(
        "--output",
        type=Path,
        required=True,
        metavar="PATH",
        help="Destination HTML file.",
    )
    p.add_argument(
        "--max-candidates",
        type=int,
        default=30,
        help="Maximum number of candidates to render (default: 30).",
    )
    return p.parse_args()


if __name__ == "__main__":
    import logging as _logging
    _logging.basicConfig(level=_logging.INFO, format="%(levelname)s  %(message)s")
    args = _parse_args()
    save_astronomical_report(
        candidates_path=args.candidates,
        svg_catalog_path=args.svg_catalog,
        output_path=args.output,
        max_candidates=args.max_candidates,
    )
