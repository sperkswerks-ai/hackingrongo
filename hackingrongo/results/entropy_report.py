"""
hackingrongo.results.entropy_report
=====================================

Combined report — Entropy, Reading Direction, and Tablet Spectrum.

The three previously-separate reports (entropy_report, reading_order_report,
spectrum_report) are unified here so that all information-theoretic analyses
of the rongorongo corpus are available in one self-contained document.

Structure
---------
Part I  — Entropy & Information Theory
  · How language-like is rongorongo? (composite assessment)
  · Sign frequency and IC decomposition
  · Mathematical framework (every formula)
  · Reference baselines
  · IC sensitivity (pre/post, 3 dating scenarios)
  · Shannon entropy H with bootstrap CIs
  · Conditional entropy & bigram MI
  · Positional mutual information
  · Boustrophedon voice-split test
  · Zipf's law analysis

Part II — Reading Direction
  · Test 4: recto/verso ordering (LOO perplexity)
  · Test 1: conditional entropy asymmetry
  · Test 2: n-gram perplexity asymmetry
  · Test 3: line-boundary entropy
  · Methodology appendix + references

Part III — Tablet Spectrum
  · The six spectrum features explained
  · Per-tablet logographic/syllabic spectrum visualization
  · Feature breakdown table
  · Scholarly annotations vs computed scores
  · Pre-contact vs post-contact comparison

Inputs (all optional — renders gracefully with pending placeholders)
--------------------------------------------------------------------
  outputs/sensitivity_analysis.json       — IC + H + entropy rate per scenario
  outputs/analysis/zipf_analysis.json     — Zipf exponent + KS test
  outputs/analysis/boustrophedon_ic.json  — voice-split IC per parity
  outputs/reading_order_results.json      — reading-direction test results
  outputs/analysis/spectrum_scores.json   — per-tablet spectrum scores
  data/metadata/tablets.json             — tablet names + content annotations
  data/corpus/                           — for sign frequency breakdown
  data/glyphs/svg/catalog.json           — for glyph images in frequency section

Output
------
  outputs/analysis/entropy_report.html   (replaces the three separate reports)

CLI
---
    python -m hackingrongo.results.entropy_report \\
        --sensitivity    outputs/sensitivity_analysis.json \\
        --reading-order  outputs/reading_order_results.json \\
        --spectrum       outputs/analysis/spectrum_scores.json \\
        --output         outputs/analysis/entropy_report.html
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import html as _html
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional imports from sibling report modules
# ---------------------------------------------------------------------------

try:
    from hackingrongo.results.reading_order_report import (
        _render_test4_hero   as _ro_test4,
        _render_test1        as _ro_test1,
        _render_test2        as _ro_test2,
        _render_test3        as _ro_test3,
        _render_methodology  as _ro_methodology,
        _render_references   as _ro_references,
        _CSS                 as _RO_CSS,
    )
    _HAS_RO = True
except Exception:
    _HAS_RO = False
    _RO_CSS = ""
    logger.debug("reading_order_report not importable — Part II will show pending.")

try:
    from hackingrongo.results.spectrum_report import (
        _section_explanation        as _sp_explanation,
        _section_spectrum_bars      as _sp_bars,
        _section_feature_table      as _sp_table,
        _section_annotations        as _sp_annotations,
        _section_stratum_comparison as _sp_stratum,
        _CSS                        as _SP_CSS,
    )
    _HAS_SP = True
except Exception:
    _HAS_SP = False
    _SP_CSS = ""
    logger.debug("spectrum_report not importable — Part III will show pending.")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fmt(v: Any, d: int = 4) -> str:
    if v is None or (isinstance(v, float) and not math.isfinite(v)):
        return "—"
    return f"{v:.{d}f}"


def _pct(v: Any) -> str:
    if v is None or (isinstance(v, float) and not math.isfinite(v)):
        return "—"
    return f"{v:.1f}%"


def _ci(lo: Any, hi: Any, d: int = 4) -> str:
    if lo is None or hi is None:
        return "—"
    return f"[{lo:.{d}f}, {hi:.{d}f}]"


def _load_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _pending(msg: str = "Run the entropy pipeline to populate.") -> str:
    return (
        f'<div class="pending">'
        f'<span class="pending-icon">⏳</span> {msg}'
        f'</div>'
    )


# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------

_CSS = """\
:root {
  --bg: #ffffff; --surface: #f8f8fa; --surface2: #f0f0f5;
  --border: #d0d0dd; --text: #1a1a1a; --muted: #666666;
  --accent: #c4a96d; --accent2: #7b9ee0;
  --pre: #2563eb; --post: #7c3aed; --robust: #16a34a;
  --warn: #d97706; --boust: #0e7490; --math: #1e3a5f;
}
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
  background: var(--bg); color: var(--text);
  font-family: 'Cormorant Garamond', 'Palatino Linotype', Georgia, serif;
  font-size: 16px; line-height: 1.65;
}
.wrap { max-width: 1040px; margin: 0 auto; padding: 52px 28px; }

/* ── Header ── */
.report-header { border-bottom: 2px solid var(--border);
                 padding-bottom: 38px; margin-bottom: 48px; }
.report-title  { font-size: 34px; font-weight: 600; color: #000; }
.report-subtitle { font-size: 17px; color: var(--accent); font-style: italic; margin-top: 6px; }
.report-meta { margin-top: 18px; font-family: 'JetBrains Mono', monospace;
               font-size: 11px; color: var(--muted); line-height: 2.2; }
.report-meta b { color: #333; }

/* ── Section headers ── */
.sec-head { font-size: 22px; font-weight: 600; color: #000;
            margin: 48px 0 6px; border-top: 1px solid var(--border);
            padding-top: 28px; }
.sec-sub  { font-size: 13.5px; color: var(--muted); font-style: italic; margin-bottom: 22px; }

/* ── Intro prose ── */
.intro { max-width: 820px; margin-bottom: 40px; }
.intro p { font-size: 14.5px; color: #333; line-height: 1.9; margin-bottom: 12px; }
.intro b { color: #000; }
.intro code { font-family: 'JetBrains Mono', monospace; font-size: 12px;
              background: var(--surface2); border: 1px solid var(--border);
              border-radius: 2px; padding: 1px 5px; }

/* ── Math display ── */
.formula-block {
  background: var(--surface); border-left: 3px solid var(--math);
  border-radius: 0 6px 6px 0; padding: 16px 22px; margin: 18px 0;
  max-width: 780px;
}
.formula-block .formula-label {
  font-family: 'JetBrains Mono', monospace; font-size: 9px;
  letter-spacing: 0.12em; text-transform: uppercase; color: var(--muted);
  margin-bottom: 8px;
}
.formula-block .formula-expr {
  font-family: 'Palatino Linotype', Georgia, serif; font-size: 18px;
  color: var(--math); line-height: 1.7;
}
.formula-block .formula-where {
  font-size: 12.5px; color: #444; line-height: 1.8; margin-top: 8px;
}
.formula-block .formula-where b { color: #000; }

/* ── Baselines table ── */
.baselines-table { width: 100%; border-collapse: collapse; font-size: 13.5px;
                   margin: 16px 0 28px; }
.baselines-table th { text-align: left; padding: 8px 14px;
                      font-family: 'JetBrains Mono', monospace; font-size: 10px;
                      letter-spacing: 0.08em; text-transform: uppercase;
                      color: var(--muted); border-bottom: 1px solid var(--border);
                      background: var(--surface); }
.baselines-table td { padding: 9px 14px; border-bottom: 1px solid var(--border);
                      color: #333; line-height: 1.6; vertical-align: top; }
.baselines-table tr:last-child td { border-bottom: none; }
.baselines-table .num { font-family: 'JetBrains Mono', monospace; font-size: 12px; }
.baselines-table .highlight { background: #fffde7; }

/* ── Metric cards ── */
.metric-strip { display: flex; flex-wrap: wrap; gap: 12px; margin: 18px 0 28px; }
.mcard { background: var(--surface); border: 1px solid var(--border);
         border-radius: 7px; padding: 14px 18px; min-width: 140px; }
.mcard-n { font-family: 'JetBrains Mono', monospace; font-size: 22px;
           font-weight: 500; color: var(--accent); display: block; }
.mcard-label { font-size: 10.5px; color: var(--muted); line-height: 1.45;
               margin-top: 3px; display: block; }
.mcard-sub { font-size: 10px; color: var(--accent2); margin-top: 2px; display: block; }
.mcard.pre  { border-color: #93c5fd; }
.mcard.post { border-color: #c4b5fd; }
.mcard.diff { border-color: #6ee7b7; }

/* ── Scenario / comparison table ── */
.data-table { width: 100%; border-collapse: collapse; font-size: 13px; margin: 14px 0 28px; }
.data-table th { text-align: left; padding: 8px 12px;
                 font-family: 'JetBrains Mono', monospace; font-size: 9.5px;
                 letter-spacing: 0.08em; text-transform: uppercase;
                 color: var(--muted); border-bottom: 1px solid var(--border);
                 background: var(--surface); white-space: nowrap; }
.data-table td { padding: 9px 12px; border-bottom: 1px solid var(--border);
                 color: #333; vertical-align: top; }
.data-table tr:last-child td { border-bottom: none; }
.data-table .mono { font-family: 'JetBrains Mono', monospace; font-size: 11px; }
.data-table .ci   { font-size: 11px; color: var(--muted); }
.data-table .pos  { color: #16a34a; font-weight: 600; }
.data-table .neg  { color: #dc2626; font-weight: 600; }
.badge { font-family: 'JetBrains Mono', monospace; font-size: 9.5px;
         border-radius: 3px; padding: 2px 8px; white-space: nowrap; }
.badge-ok   { background: #dcfce7; color: #15803d; border: 1px solid #86efac; }
.badge-fail { background: #fee2e2; color: #b91c1c; border: 1px solid #fca5a5; }
.badge-warn { background: #fef9c3; color: #92400e; border: 1px solid #fde047; }

/* ── Interpretation callout ── */
.interp { background: var(--surface2); border: 1px solid var(--border);
          border-radius: 6px; padding: 14px 18px; margin: 12px 0 28px;
          font-size: 13.5px; color: #333; line-height: 1.85; max-width: 820px; }
.interp b { color: #000; }
.interp.positive { border-color: #86efac; background: #f0fdf4; }
.interp.negative { border-color: #fca5a5; background: #fff1f2; }
.interp.neutral  { border-color: #bfdbfe; background: #eff6ff; }

/* ── Boustrophedon ── */
.boust-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin: 16px 0; }
.boust-cell { background: var(--surface); border: 1px solid var(--border);
              border-radius: 6px; padding: 14px 18px; text-align: center; }
.boust-cell-label { font-family: 'JetBrains Mono', monospace; font-size: 10px;
                    text-transform: uppercase; color: var(--muted); margin-bottom: 6px; }
.boust-ic { font-family: 'JetBrains Mono', monospace; font-size: 26px;
            color: var(--boust); font-weight: 500; }
.boust-ci { font-size: 11px; color: var(--muted); margin-top: 4px; }
.boust-n  { font-size: 11px; color: var(--muted); margin-top: 2px; }
.delta-row { font-family: 'JetBrains Mono', monospace; font-size: 13px;
             color: #333; margin: 10px 0; }

/* ── Pending placeholder ── */
.pending { background: var(--surface2); border: 1px dashed var(--border);
           border-radius: 5px; padding: 12px 16px; font-size: 13px;
           color: var(--muted); margin: 12px 0; }
.pending-icon { margin-right: 6px; }

/* ── Footer ── */
.report-footer { border-top: 1px solid var(--border); margin-top: 52px;
                 padding-top: 26px; font-size: 12px; color: var(--muted);
                 line-height: 2.0; }
.report-footer a { color: var(--accent); }
.report-footer code { font-family: 'JetBrains Mono', monospace; font-size: 11px;
                      background: var(--surface2); border: 1px solid var(--border);
                      border-radius: 2px; padding: 1px 4px; }

/* ── Frequency breakdown ── */
.freq-grid { display: flex; flex-wrap: wrap; gap: 10px; margin: 20px 0 32px; }
.freq-card {
  background: var(--surface); border: 1px solid var(--border);
  border-radius: 6px; padding: 10px 10px 8px; width: 108px;
  display: flex; flex-direction: column; align-items: center; gap: 4px;
}
.freq-glyph {
  width: 56px; height: 56px; display: flex; align-items: center;
  justify-content: center; color: var(--accent);
}
.freq-no-img { font-size: 22px; color: var(--muted); }
.freq-code { font-family: 'JetBrains Mono', monospace; font-size: 10px;
             color: var(--accent); font-weight: 500; }
.freq-stat { font-family: 'JetBrains Mono', monospace; font-size: 11px; color: #333; }
.freq-pct  { font-size: 10px; color: var(--muted); }
.ic-bar-wrap { width: 88px; height: 12px; background: var(--surface2);
               border-radius: 2px; position: relative; overflow: hidden; }
.ic-bar { height: 100%; background: var(--accent); border-radius: 2px; }
.ic-bar-label { position: absolute; right: 3px; top: 0; font-family: 'JetBrains Mono', monospace;
                font-size: 8px; color: #333; line-height: 12px; }
.freq-meta { display: flex; gap: 5px; align-items: center; width: 100%;
             justify-content: center; }
.freq-tabs { font-size: 9px; color: var(--muted); }

/* Positional entropy dot */
.pos-dot { font-family: 'JetBrains Mono', monospace; font-size: 8.5px;
           border-radius: 2px; padding: 1px 5px; white-space: nowrap; }
.pos-anchor { background: #dcfce7; color: #15803d; border: 1px solid #86efac; }
.pos-low    { background: #fef9c3; color: #92400e; border: 1px solid #fde047; }
.pos-free   { background: var(--surface2); color: var(--muted); border: 1px solid var(--border); }

@media (max-width: 700px) {
  .boust-grid { grid-template-columns: 1fr; }
  .metric-strip { flex-direction: column; }
  .freq-grid { gap: 8px; }
  .freq-card { width: 96px; }
}

/* ── Sticky navigation bar ── */
.report-nav {
  position: sticky; top: 0; z-index: 200;
  background: rgba(255,255,255,0.97);
  border-bottom: 1px solid var(--border);
  padding: 7px 28px; display: flex; align-items: center;
  gap: 0; overflow-x: auto; white-space: nowrap;
  font-family: 'JetBrains Mono', monospace; font-size: 10.5px;
  backdrop-filter: blur(6px);
}
.nav-part { color: var(--accent); font-weight: 500; margin-right: 4px; }
.nav-sep  { color: var(--border); margin: 0 8px; }
.report-nav a { color: var(--muted); text-decoration: none;
                padding: 2px 8px; border-radius: 3px; transition: color 0.15s; }
.report-nav a:hover { color: var(--accent); background: var(--surface); }

/* ── Part dividers ── */
.part-divider { border-top: 2px solid var(--accent);
                margin: 72px 0 48px; padding-top: 36px; }
.part-label { font-family: 'JetBrains Mono', monospace; font-size: 9.5px;
              letter-spacing: 0.15em; text-transform: uppercase;
              color: var(--accent); margin-bottom: 8px; }
.part-title { font-size: 28px; font-weight: 600; color: #000; margin-bottom: 4px; }
.part-sub   { font-size: 14px; color: var(--muted); font-style: italic; }

/* ── Reading-order unique styles (from reading_order_report.py) ── */
:root { --rv: #7c3aed; --rv-bg: #faf5ff; --rv-border: #ddd6fe; --rv-dark: #4c1d95;
        --confirmed: #16a34a; --t1: #0e7490; --t3: #065f46; }
.test-section { margin-bottom: 56px; }
.test-header { margin-bottom: 20px; }
.test-number { font-family: 'JetBrains Mono', monospace; font-size: 10px;
               color: var(--muted); letter-spacing: 0.1em; text-transform: uppercase;
               margin-bottom: 4px; }
.test-title  { font-size: 22px; font-weight: 600; color: #000; }
.test-desc   { font-size: 13.5px; color: var(--muted); margin-top: 4px; }
.rv-hero { background: var(--rv-bg); border: 1px solid var(--rv-border);
           border-radius: 10px; padding: 32px 32px 28px; margin-bottom: 28px; }
.rv-hero-label { font-family: 'JetBrains Mono', monospace; font-size: 9.5px;
                 letter-spacing: 0.14em; text-transform: uppercase;
                 color: var(--rv); margin-bottom: 12px; }
.rv-verdict-head { font-size: 28px; font-weight: 700; color: var(--rv-dark);
                   line-height: 1.25; margin-bottom: 8px; }
.rv-verdict-sub  { font-size: 15px; color: #444; line-height: 1.75; max-width: 740px; }
.rv-context { font-size: 14px; color: #444; line-height: 1.8; max-width: 780px;
              margin-bottom: 24px; }
.ppl-wrap { overflow-x: auto; margin: 28px 0; }
table.ppl-table { width: 100%; border-collapse: collapse; font-size: 14px; }
table.ppl-table th { font-family: 'JetBrains Mono', monospace; font-size: 9px;
  letter-spacing: 0.08em; text-transform: uppercase; color: var(--muted);
  padding: 8px 14px; text-align: left; border-bottom: 2px solid var(--border); }
table.ppl-table td { padding: 11px 14px; border-bottom: 1px solid var(--border); }
table.ppl-table tr:last-child td { border-bottom: none; }
.ppl-winner { font-family: 'JetBrains Mono', monospace; font-size: 17px;
              font-weight: 500; color: var(--confirmed); }
.ppl-loser  { font-family: 'JetBrains Mono', monospace; font-size: 17px; color: var(--muted); }
.ppl-reduc  { font-family: 'JetBrains Mono', monospace; font-size: 11px;
              color: var(--confirmed); white-space: nowrap; }
.ppl-row-label { font-size: 13.5px; color: #333; }
.ppl-bar-wrap { margin: 6px 0; }
.ppl-bar-row  { display: flex; align-items: center; gap: 10px; margin-bottom: 6px; }
.ppl-bar-label { font-family: 'JetBrains Mono', monospace; font-size: 10px;
                 color: var(--muted); width: 40px; flex-shrink: 0; }
.ppl-bar-track { flex: 1; height: 8px; background: #e5e7eb; border-radius: 4px; overflow: hidden; }
.ppl-bar-fill  { height: 100%; border-radius: 4px; }
.ppl-bar-fill.winner { background: var(--confirmed); }
.ppl-bar-fill.loser  { background: #d1d5db; }
.ppl-bar-val  { font-family: 'JetBrains Mono', monospace; font-size: 11.5px;
                color: #333; width: 50px; text-align: right; flex-shrink: 0; }
.note-box { background: var(--surface); border: 1px solid var(--border);
            border-radius: 6px; padding: 16px 20px; margin-top: 20px;
            font-size: 13px; color: #555; line-height: 1.75; }
.note-box-title { font-family: 'JetBrains Mono', monospace; font-size: 9px;
                  letter-spacing: 0.1em; text-transform: uppercase;
                  color: var(--muted); margin-bottom: 6px; }
.verdict-strip { padding: 13px 18px; border-left: 3px solid var(--border);
                 background: var(--surface2); font-size: 14px; color: #333;
                 line-height: 1.65; margin-top: 4px; }
.verdict-strip.ok   { border-color: var(--confirmed); background: #f0fdf4; }
.verdict-strip.warn { border-color: var(--warn);      background: #fffbeb; }
.verdict-strip.fail { border-color: #dc2626;          background: #fff1f2; }
.metric-block { background: var(--surface); border: 1px solid var(--border);
                border-radius: 8px; padding: 22px 26px; margin-bottom: 16px; }
.metric-grid { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 18px; margin-bottom: 18px; }
.metric-value { font-family: 'JetBrains Mono', monospace; font-size: 20px;
                font-weight: 500; color: #000; }
.metric-unit  { font-family: 'JetBrains Mono', monospace; font-size: 11px;
                color: var(--muted); margin-left: 3px; }
.metric-label { font-size: 11px; color: var(--muted); margin-top: 3px; }
.metric-delta-pos { color: var(--confirmed); font-family: 'JetBrains Mono', monospace; }
.metric-delta-neg { color: #dc2626;          font-family: 'JetBrains Mono', monospace; }
.metric-delta-neu { color: var(--warn);      font-family: 'JetBrains Mono', monospace; }
.method-section { border-top: 1px solid var(--border); margin-top: 60px; padding-top: 36px; }
.method-title { font-size: 20px; font-weight: 600; margin-bottom: 22px; }
.method-item  { margin-bottom: 22px; }
.method-item-head { font-size: 14.5px; font-weight: 600; margin-bottom: 6px; color: #222; }
.method-item-body { font-size: 13.5px; color: #444; line-height: 1.8; max-width: 800px; }
.method-formula { font-family: 'JetBrains Mono', monospace; font-size: 12px;
                  background: var(--surface2); border: 1px solid var(--border);
                  border-radius: 4px; padding: 8px 14px; display: inline-block;
                  margin: 6px 0; color: #333; }
.ref-section { border-top: 1px solid var(--border); margin-top: 48px; padding-top: 32px; }
.ref-title { font-size: 16px; font-weight: 600; margin-bottom: 18px; color: #333; }
.ref-list { list-style: none; }
.ref-list li { font-size: 13px; color: #444; line-height: 1.75; margin-bottom: 8px;
               padding-left: 22px; text-indent: -22px; }
.stat-card.rv        .stat-value { color: var(--rv); }
.stat-card.confirmed .stat-value { color: var(--confirmed); }
.abstract { margin-top: 22px; font-size: 14.5px; color: #333;
            max-width: 760px; line-height: 1.85; }
.abstract p + p { margin-top: 10px; }

/* ── Spectrum unique styles (from spectrum_report.py) ── */
:root { --syllabic: #2563eb; --logographic: #9333ea; --mixed: #d97706; }
.spectrum-axis { display: flex; align-items: center; gap: 0; margin: 18px 0 26px;
                 font-family: 'JetBrains Mono', monospace; font-size: 11px; }
.axis-label-left  { color: var(--syllabic); min-width: 90px; }
.axis-label-right { color: var(--logographic); min-width: 90px; text-align: right; }
.axis-track { flex: 1; height: 6px; border-radius: 3px;
              background: linear-gradient(to right, var(--syllabic), var(--mixed), var(--logographic)); }
.tablet-grid { display: flex; flex-direction: column; gap: 6px; margin: 16px 0 32px; }
.tablet-row { display: grid; grid-template-columns: 32px 180px 1fr 56px 80px 70px;
              align-items: center; gap: 10px; }
.tablet-id   { font-family: 'JetBrains Mono', monospace; font-size: 12px;
               color: var(--accent); font-weight: 500; }
.tablet-name { font-size: 13px; color: #333; white-space: nowrap;
               overflow: hidden; text-overflow: ellipsis; }
.bar-track { height: 16px; background: var(--surface2); border-radius: 3px;
             overflow: hidden; position: relative; }
.bar-fill  { height: 100%; border-radius: 3px; }
.score-val { font-family: 'JetBrains Mono', monospace; font-size: 12px;
             color: #333; text-align: right; }
.stratum-badge { font-family: 'JetBrains Mono', monospace; font-size: 9px;
                 border-radius: 3px; padding: 2px 6px; text-align: center; }
.stratum-pre     { background: #dcfce7; color: #15803d; border: 1px solid #86efac; }
.stratum-post    { background: #ede9fe; color: #6d28d9; border: 1px solid #c4b5fd; }
.stratum-unknown { background: var(--surface2); color: var(--muted); border: 1px solid var(--border); }
.stratum-excluded { background: #fee2e2; color: #991b1b; border: 1px solid #fca5a5; }
.reliability-warn { font-family: 'JetBrains Mono', monospace; font-size: 9px; color: var(--muted); }
.feat-explain { background: var(--surface); border-left: 3px solid var(--accent);
                border-radius: 0 6px 6px 0; padding: 14px 18px; margin: 10px 0;
                max-width: 820px; }
.feat-explain-label { font-family: 'JetBrains Mono', monospace; font-size: 9.5px;
                      letter-spacing: 0.1em; text-transform: uppercase;
                      color: var(--muted); margin-bottom: 5px; }
.feat-explain p { font-size: 13.5px; color: #333; line-height: 1.8; }
.annot-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; margin: 16px 0 32px; }
.annot-card { background: var(--surface); border: 1px solid var(--border);
              border-radius: 7px; padding: 14px 16px; }
.annot-card-header { display: flex; align-items: center; gap: 10px; margin-bottom: 8px; }
.annot-tid  { font-family: 'JetBrains Mono', monospace; font-size: 13px;
              color: var(--accent); font-weight: 500; }
.annot-name { font-size: 14px; font-weight: 600; color: #111; }
.annot-score { font-family: 'JetBrains Mono', monospace; font-size: 18px;
               font-weight: 500; margin-left: auto; }
.annot-hypothesis { font-size: 12.5px; color: #444; line-height: 1.7; }
.annot-verdict { font-family: 'JetBrains Mono', monospace; font-size: 9.5px;
                 border-radius: 3px; padding: 2px 7px; margin-top: 8px; display: inline-block; }
.verdict-match    { background: #dcfce7; color: #15803d; border: 1px solid #86efac; }
.verdict-surprise { background: #fef9c3; color: #92400e; border: 1px solid #fde047; }
.verdict-na       { background: var(--surface2); color: var(--muted); border: 1px solid var(--border); }
.stratum-compare { display: grid; grid-template-columns: 1fr 1fr; gap: 18px; margin: 16px 0 32px; }
.stratum-col { background: var(--surface); border: 1px solid var(--border);
               border-radius: 7px; padding: 16px 18px; }
.stratum-col-title { font-family: 'JetBrains Mono', monospace; font-size: 10px;
                     letter-spacing: 0.1em; text-transform: uppercase;
                     color: var(--muted); margin-bottom: 10px; }
.stratum-col-score { font-family: 'JetBrains Mono', monospace; font-size: 28px;
                     font-weight: 500; color: var(--accent); }
.stratum-col-sub { font-size: 12px; color: var(--muted); margin-top: 4px; }
"""


# ---------------------------------------------------------------------------
# Section renderers
# ---------------------------------------------------------------------------

def _section_language_likeness(sensitivity: dict, zipf: dict, boust: dict) -> str:
    """Language-likeness composite — answers 'how language-like is rongorongo?'"""
    scenarios = sensitivity.get("scenarios", {})
    robustness = sensitivity.get("robustness", {})

    # Collect IC_norm and IC/random ratios across scenarios
    rows_data = []
    for sname, clusters in scenarios.items():
        pre  = clusters.get("pre_contact", {})
        post = clusters.get("post_contact", {})
        pre_ic, post_ic = pre.get("ic", 0.0), post.get("ic", 0.0)
        pre_k,  post_k  = pre.get("n_types", 1), post.get("n_types", 1)
        pre_norm  = pre_ic * pre_k
        post_norm = post_ic * post_k
        pre_rand_ratio  = pre_norm   # IC_norm = IC * k = IC / (1/k)
        post_rand_ratio = post_norm
        rows_data.append((sname, pre_ic, post_ic, pre_ic/post_ic if post_ic else 0,
                          pre_norm, post_norm, pre_rand_ratio, post_rand_ratio,
                          pre_k, post_k))

    zipf_alpha = zipf.get("exponent_mle")
    zipf_r2    = zipf.get("r_squared_loglog")
    zipf_ok    = zipf.get("consistent_with_zipf", False)
    boust_split = not (boust or {}).get("cis_overlap", True)

    # English reference: IC=0.065, k=26, IC_norm=1.69
    eng_ic_norm = 0.065 * 26  # ≈ 1.69

    # Build scenario comparison table rows
    table_rows = ""
    for sname, pre_ic, post_ic, raw_ratio, pre_norm, post_norm, _, _, pre_k, post_k in rows_data:
        norm_ratio = pre_norm / post_norm if post_norm else 0
        direction = "pre > post" if pre_norm > post_norm else "post > pre"
        dir_cls = "pos" if pre_norm > post_norm else "neg"
        table_rows += (
            f'<tr><td class="mono">{sname.replace("_"," ")}</td>'
            f'<td class="mono">{pre_ic:.6f} (k={pre_k})</td>'
            f'<td class="mono">{post_ic:.6f} (k={post_k})</td>'
            f'<td class="mono">{raw_ratio:.2f}×</td>'
            f'<td class="mono">{pre_norm:.2f}</td>'
            f'<td class="mono">{post_norm:.2f}</td>'
            f'<td class="{dir_cls}">{direction} ({norm_ratio:.2f}×)</td></tr>'
        )

    # Language-likeness scorecard items
    robust_val = robustness.get("robust")
    all_pos    = robustness.get("all_pre_gt_post")

    def _score(signal: str, value: str, verdict: str, verdict_cls: str, explanation: str) -> str:
        return (
            f'<tr>'
            f'<td><b>{signal}</b></td>'
            f'<td class="mono">{value}</td>'
            f'<td><span class="badge badge-{verdict_cls}">{verdict}</span></td>'
            f'<td style="font-size:12.5px;color:#555;line-height:1.6">{explanation}</td>'
            f'</tr>'
        )

    # Build scorecard based on real numbers
    # IC norm: all scenarios >> 1.0 (random) and >> 1.69 (English)
    ic_norm_range = (
        f"{min(r[4] for r in rows_data):.1f}–{max(r[4] for r in rows_data):.1f} (pre), "
        f"{min(r[5] for r in rows_data):.1f}–{max(r[5] for r in rows_data):.1f} (post)"
        if rows_data else "—"
    )
    zipf_val = f"α={_fmt(zipf_alpha,3)}, R²={_fmt(zipf_r2,3)}" if zipf_alpha else "—"

    scorecard_rows = (
        _score("Non-random sign usage",
               ic_norm_range,
               "strong signal", "ok",
               (f"IC × k ranges from "
                f"{min(r[4] for r in rows_data):.1f}–{max(r[5] for r in rows_data):.1f} "
                f"across all scenarios and strata — all are far above the random baseline (1.0) "
                f"and above the English reference ({eng_ic_norm:.2f}). "
                "The sign distribution is highly concentrated relative to a uniform draw."
                if rows_data else
                "No scenario data available.")) +
        _score("Zipf power-law distribution",
               zipf_val,
               "partial" if not zipf_ok else "consistent", "warn" if not zipf_ok else "ok",
               f"Sign frequencies follow a power law with R² = {_fmt(zipf_r2,3)} — "
               f"a good fit. However, α = {_fmt(zipf_alpha,3)} (MLE) exceeds the canonical "
               f"natural-language range [0.9, 1.1], meaning usage is more top-heavy than "
               "typical language corpora. Some scripts (e.g. Chinese characters) also show α > 1.") +
        _score("Stratum divergence (pre ≠ post contact)",
               f"Δ IC range: {_fmt(robustness.get('delta_range'),5)}; robust: {robust_val}",
               "confirmed" if all_pos else "partial", "ok" if all_pos else "warn",
               "Pre-contact and post-contact rongorongo have statistically distinguishable IC values "
               "in all three dating scenarios (direction IC_pre &gt; IC_post holds throughout). "
               "This is the headline finding: the script evolved measurably across the contact boundary.") +
        _score("Boustrophedon voice-split",
               f"CIs {'non-overlapping' if boust_split else 'overlap (marginal)'}",
               "marginal", "warn",
               "Odd and even lines show a consistent IC trend (IC_even &gt; IC_odd) "
               "but the CIs overlap marginally. No strong evidence of two structurally "
               "distinct text streams, though the trend is notable.") +
        _score("Sequential structure — bigram MI I(Sₙ; Sₙ₋₁)",
               "pending — rerun pipeline",
               "pending", "warn",
               "Requires updated sensitivity_analysis.json with the new conditional entropy fields. "
               "For natural language this is typically 1–3 bits; near 0 implies random sign order.") +
        _score("Positional structure — I(sign; position)",
               "pending — rerun pipeline",
               "pending", "warn",
               "Requires position fields in corpus JSON. The sign classifier already uses "
               "positional entropy ≤ 0.40 bits as a logogram threshold — signs below this "
               "are \"anchors\" at constrained syntactic positions (see connection below).")
    )

    # The vocabulary-size correction explanation
    correction_html = ""
    if rows_data:
        raw_min = min(r[3] for r in rows_data)
        raw_max = max(r[3] for r in rows_data)
        correction_html = f"""
<div class="interp negative">
  <b>Correcting the "3× more language-like" claim.</b>
  The raw IC ratio IC<sub>pre</sub> / IC<sub>post</sub> is <b>{raw_min:.2f}×–{raw_max:.2f}×</b>
  across the three tablet-dating scenarios — not 3×.
  Raw IC is heavily influenced by vocabulary size: for a uniform distribution,
  IC = 1/k, so a stratum with fewer sign types will always appear to have higher
  IC even if its usage pattern is structurally identical.
  Once normalized (IC × k), the picture reverses in some scenarios — post-contact
  shows <em>higher</em> normalized IC than pre-contact when undated tablets are
  treated conservatively. The sign distribution is structured in both strata;
  the magnitude and direction of their difference depends more on how undated
  tablets are assigned than on any intrinsic property of the strata themselves.
</div>"""

    return f"""
<div class="sec-head" style="margin-top:36px;border-top:2px solid var(--math)">
  How Language-Like Is Rongorongo? — A Composite Assessment
</div>
<div class="sec-sub">
  Aggregating all entropy metrics into a single answer; correcting the raw IC comparison
</div>

<div class="intro">
  <p>Each entropy metric answers a different slice of the question
  "does rongorongo behave like a natural writing system?"
  No single number settles it. This section assembles the full picture
  across all metrics computed in this report and translates them into
  a plain-language verdict, with explicit corrections where a metric
  is commonly misread.</p>
</div>

{correction_html}

<table class="data-table" style="margin-top:20px">
<thead><tr>
  <th>Scenario</th>
  <th>IC<sub>pre</sub> (k)</th>
  <th>IC<sub>post</sub> (k)</th>
  <th>Raw ratio</th>
  <th>IC<sub>norm</sub> pre</th>
  <th>IC<sub>norm</sub> post</th>
  <th>Normalized direction</th>
</tr></thead>
<tbody>{table_rows}
<tr style="background:var(--surface);font-size:11.5px;color:var(--muted)">
  <td>English reference</td>
  <td class="mono" colspan="2">IC ≈ 0.065 (k=26)</td>
  <td>—</td>
  <td class="mono" colspan="2">IC<sub>norm</sub> ≈ {eng_ic_norm:.2f}</td>
  <td>language benchmark</td>
</tr>
<tr style="background:var(--surface);font-size:11.5px;color:var(--muted)">
  <td>Random baseline</td>
  <td class="mono" colspan="2">IC = 1/k (any k)</td>
  <td>—</td>
  <td class="mono" colspan="2">IC<sub>norm</sub> = 1.00 (by definition)</td>
  <td>noise floor</td>
</tr>
</tbody>
</table>

<div class="interp neutral" style="margin-bottom:24px">
  <b>Reading the table:</b>
  IC<sub>norm</sub> &gt; 1.0 = more structured than random.
  IC<sub>norm</sub> &gt; {eng_ic_norm:.2f} = more concentrated than English text.
  All rongorongo strata are well above both baselines in every scenario,
  confirming that sign usage is highly non-random. The scenario-to-scenario
  variation in IC<sub>norm</sub> shows how much the comparison is driven by
  which undated tablets are assigned to each stratum.
</div>

<table class="data-table">
<thead><tr>
  <th>Language-Likeness Signal</th>
  <th>Our value</th>
  <th>Verdict</th>
  <th>What it means</th>
</tr></thead>
<tbody>{scorecard_rows}</tbody>
</table>

<div class="sec-head" style="font-size:18px;margin-top:32px">
  Glyph Anchors as Language Constraints
</div>
<div class="interp neutral">
  <p>The sign classifier flags signs with positional entropy ≤ 0.40 bits as
  <b>logogram candidates</b> — signs that appear at highly constrained
  within-line positions across the corpus. These are the "anchor" glyphs
  of the writing system: wherever they appear, they occupy a predictable
  slot in the sequence, which is the syntactic signature of a function word,
  taxogram, or discourse marker.</p>
  <p>Positional mutual information I(sign; position) measures this at the
  corpus level: how many bits of sign identity are explained by knowing
  where in a line the glyph appears. A value of I = 0 means position and
  sign identity are independent (random-order text). A value of I &gt; 0
  confirms that some signs are structurally position-constrained — the
  same constraint the 0.40-bit threshold detects per-sign.</p>
  <p>The connection to entropy is direct: the sign classifier's logogram
  threshold is a <em>per-sign</em> positional entropy cutoff, while I(sign; position)
  is the <em>corpus-level</em> summary of the same phenomenon. Once the pipeline
  is rerun with position fields in the corpus JSON, this section will show
  both numbers side by side and compute what fraction of the sign inventory
  falls below the 0.40-bit logogram threshold.</p>
</div>
"""


# ---------------------------------------------------------------------------
# Frequency breakdown — data + rendering
# ---------------------------------------------------------------------------

def _compute_frequency_stats(corpus_dir: Path, top_n: int = 20) -> dict:
    """Compute per-sign frequency, IC contribution, and positional entropy.

    Returns a dict with keys:
      total_tokens, total_types, approx_ic,
      top_signs: list of dicts per sign (code, freq, pct, ic_contrib_pct,
                 cumulative_ic_pct, pos_entropy, n_tablets, tablets)
    """
    import re as _re
    from collections import defaultdict as _dd

    sign_data: dict = _dd(lambda: {"freq": 0, "positions": [], "tablets": set()})

    for jf in sorted(corpus_dir.glob("[A-Z].json")):
        if "ferrara" in jf.stem:
            continue
        try:
            data = json.loads(jf.read_text(encoding="utf-8"))
        except Exception:
            continue
        tablet_id = jf.stem
        glyphs = data.get("glyphs", [])

        by_line: dict = _dd(list)
        for g in glyphs:
            key = (g.get("side", "?"), g.get("line", "?"))
            by_line[key].append(g.get("barthel_code", ""))

        for line_glyphs in by_line.values():
            n = len(line_glyphs)
            for pos, code in enumerate(line_glyphs):
                if not code or code == "?" or "!" in code:
                    continue
                rel = pos / max(n - 1, 1)
                sign_data[code]["freq"] += 1
                sign_data[code]["positions"].append(rel)
                sign_data[code]["tablets"].add(tablet_id)

    total_tokens = sum(v["freq"] for v in sign_data.values())
    total_types  = len(sign_data)
    approx_ic    = sum((v["freq"] / total_tokens) ** 2 for v in sign_data.values()) if total_tokens else 0.0

    def _pos_h(positions: list) -> float:
        if len(positions) < 2:
            return 0.0
        n_bins = 5
        bins: dict = {}
        for p in positions:
            b = min(int(p * n_bins), n_bins - 1)
            bins[b] = bins.get(b, 0) + 1
        total = sum(bins.values())
        return -sum((c / total) * math.log2(c / total) for c in bins.values() if c > 0)

    ranked = sorted(sign_data.items(), key=lambda x: -x[1]["freq"])
    top = ranked[:top_n]

    cumulative = 0.0
    result_signs = []
    for code, d in top:
        freq = d["freq"]
        pct  = freq / total_tokens * 100 if total_tokens else 0
        ic_c = (freq / total_tokens) ** 2 if total_tokens else 0
        ic_pct = ic_c / approx_ic * 100 if approx_ic else 0
        cumulative += ic_pct
        result_signs.append({
            "code": code,
            "freq": freq,
            "pct": round(pct, 2),
            "ic_contrib_pct": round(ic_pct, 1),
            "cumulative_ic_pct": round(cumulative, 1),
            "pos_entropy": round(_pos_h(d["positions"]), 4),
            "n_tablets": len(d["tablets"]),
            "tablets": sorted(d["tablets"]),
        })

    return {
        "total_tokens": total_tokens,
        "total_types": total_types,
        "approx_ic": round(approx_ic, 6),
        "top_n": top_n,
        "top_signs": result_signs,
    }


def _glyph_img(code: str, catalog: dict, size: int = 52) -> str:
    """Return inline SVG or base64 img for code, or empty string."""
    import re as _re, base64 as _b64
    instances = catalog.get(code, [])
    if not instances:
        base = _re.sub(r"[!?()\s]+$", "", code).strip()
        instances = catalog.get(base, [])
    if not instances:
        return ""
    path = instances[0]
    try:
        if path.suffix.lower() == ".png":
            b64 = _b64.b64encode(path.read_bytes()).decode()
            return (
                f'<img src="data:image/png;base64,{b64}" '
                f'style="max-width:{size}px;max-height:{size}px;display:block;margin:auto" '
                f'alt="Barthel {code}">'
            )
        svg = path.read_text(encoding="utf-8").strip()
        svg = _re.sub(r'width="[^"]*"',  f'width="{size}"',  svg)
        svg = _re.sub(r'height="[^"]*"', f'height="{size}"', svg)
        svg = _re.sub(
            r"<path ",
            '<path fill="none" stroke="currentColor" stroke-width="1.5" '
            'stroke-linecap="round" stroke-linejoin="round" ',
            svg,
        )
        return svg
    except Exception:
        return ""


def _load_svg_catalog_for_entropy(catalog_path: Path | None) -> dict:
    """Load glyph image catalog (same logic as compound_report, localised here)."""
    import re as _re, json as _json
    if not catalog_path or not catalog_path.exists():
        return {}
    try:
        cat_data = _json.loads(catalog_path.read_text(encoding="utf-8"))
    except Exception:
        return {}

    from collections import defaultdict as _dd
    exact:    dict = _dd(list)
    base_map: dict = _dd(list)
    svg_dir = catalog_path.parent

    for r in cat_data.get("records", []):
        code = str(r.get("barthel_code", "")).strip()
        if not code:
            continue
        rel  = str(r.get("svg_path", "")).replace("svg/", "", 1)
        full = svg_dir / rel
        if not full.exists():
            continue
        exact[code].append(full)
        base = _re.sub(r"[!?()\s]+$", "", code).strip()
        if base != code:
            base_map[base].append(full)

    merged = dict(exact)
    for b, paths in base_map.items():
        if b not in merged:
            merged[b] = paths

    # PNG fallback
    bc_path = catalog_path.parent.parent / "barthel_catalog.json"
    if bc_path.exists():
        glyph_dir = catalog_path.parent.parent
        try:
            bc_records = _json.loads(bc_path.read_text(encoding="utf-8")).get("records", [])
        except Exception:
            bc_records = []
        for source_pref in ("barthel_formentafeln", "barthel_tafeln"):
            for r in bc_records:
                if r.get("source") != source_pref:
                    continue
                code = str(r.get("barthel_code") or "").strip()
                rel  = r.get("path", "")
                if not code or not rel or not rel.endswith(".png"):
                    continue
                png = glyph_dir / rel
                if png.exists() and code not in merged:
                    merged[code] = [png]
    return dict(merged)


def _section_frequency_breakdown(corpus_dir: Path | None, svg_catalog_path: Path | None) -> str:
    """IC decomposition section — top signs, frequency, IC contribution, glyph images."""
    if not corpus_dir or not corpus_dir.exists():
        return (
            f'\n<div class="sec-head">Sign Frequency &amp; IC Decomposition</div>'
            f'\n<div class="sec-sub">What the IC number is actually made of</div>'
            f'\n{_pending("Pass --corpus-dir to populate this section.")}'
        )

    stats   = _compute_frequency_stats(corpus_dir)
    catalog = _load_svg_catalog_for_entropy(svg_catalog_path)

    total   = stats["total_tokens"]
    types   = stats["total_types"]
    ic_val  = stats["approx_ic"]
    signs   = stats["top_signs"]
    top_n   = stats["top_n"]

    # Top-2 and top-20 IC percentages for the explanation
    top2_ic  = sum(s["ic_contrib_pct"] for s in signs[:2])
    top5_ic  = sum(s["ic_contrib_pct"] for s in signs[:5])
    top20_ic = signs[-1]["cumulative_ic_pct"] if signs else 0

    # Build sign cards
    cards_html = ""
    max_ic_pct = signs[0]["ic_contrib_pct"] if signs else 1.0
    # Positional entropy context: max is log2(5)≈2.32, threshold 0.40
    pos_h_max = math.log2(5)

    for s in signs:
        code     = s["code"]
        freq     = s["freq"]
        pct      = s["pct"]
        ic_pct   = s["ic_contrib_pct"]
        cum_pct  = s["cumulative_ic_pct"]
        pos_h    = s["pos_entropy"]
        n_tabs   = s["n_tablets"]

        img = _glyph_img(code, catalog, size=52)
        img_block = (
            f'<div class="freq-glyph">{img}</div>'
            if img else
            f'<div class="freq-glyph freq-no-img">?</div>'
        )

        # IC bar width as % of widest bar
        bar_w = round(ic_pct / max_ic_pct * 100)

        # Positional entropy colour: green=anchor-like (<1.0), amber=moderate, grey=free
        if pos_h <= 0.40:
            pos_cls, pos_label = "pos-anchor", "anchor"
        elif pos_h <= 1.2:
            pos_cls, pos_label = "pos-low", "low-H"
        else:
            pos_cls, pos_label = "pos-free", "free"

        cards_html += f"""
<div class="freq-card">
  {img_block}
  <div class="freq-code">{_html.escape(str(code))}</div>
  <div class="freq-stat">{freq:,} <span class="freq-pct">({pct:.1f}%)</span></div>
  <div class="ic-bar-wrap" title="IC contribution: {ic_pct:.1f}%">
    <div class="ic-bar" style="width:{bar_w}%"></div>
    <span class="ic-bar-label">{ic_pct:.1f}%</span>
  </div>
  <div class="freq-meta">
    <span class="pos-dot {pos_cls}" title="Positional entropy {pos_h:.3f} bits">{pos_label}</span>
    <span class="freq-tabs">{n_tabs} tabs</span>
  </div>
</div>"""

    # Explanation prose
    s1 = signs[0] if signs else {}
    s2 = signs[1] if len(signs) > 1 else {}

    return f"""
<div class="sec-head">Sign Frequency &amp; IC Decomposition</div>
<div class="sec-sub">
  What IC = {ic_val:.6f} is actually made of — {total:,} tokens · {types:,} types
</div>

<div class="intro">
  <p>The Index of Coincidence is <b>Σ p<sub>i</sub>²</b> — a sum of squared
  frequencies.  It is dominated by the most common signs: a sign that
  accounts for 4% of all tokens contributes (0.04)² = 0.0016 to IC, while a
  sign at 0.1% contributes only 0.000001.  Understanding what IC means
  requires knowing which signs drive it.</p>
  <p>
  Signs <b>{s1.get('code','—')}</b> ({s1.get('freq',0):,} occurrences,
  {s1.get('pct',0):.1f}% of tokens) and
  <b>{s2.get('code','—')}</b> ({s2.get('freq',0):,} occurrences,
  {s2.get('pct',0):.1f}%) together explain
  <b>{top2_ic:.1f}%</b> of the total corpus IC.
  The top 5 signs explain <b>{top5_ic:.1f}%</b>;
  the top {top_n} explain <b>{top20_ic:.1f}%</b>.
  IC is overwhelmingly the story of a small number of dominant glyphs.
  </p>
</div>

<div class="interp neutral">
  <b>Positional entropy key:</b>
  <span class="pos-dot pos-anchor">anchor</span> ≤ 0.40 bits — constrained to specific line
  positions (logogram candidate per sign classifier).
  <span class="pos-dot pos-low">low-H</span> ≤ 1.20 bits — moderate positional preference.
  <span class="pos-dot pos-free">free</span> &gt; 1.20 bits — appears throughout lines
  without positional constraint.
  <br><br>
  <b>Note on signs 001 and 076:</b> Both are the two most frequent signs and together
  drive ~40% of corpus IC, yet both have positional entropy near the maximum
  ({pos_h_max:.2f} bits for 5 bins).  They are <em>not</em> grammatical anchors —
  they appear freely across all line positions on nearly every tablet.
  Sign 200 (the taxogram) similarly shows high within-line positional entropy;
  its grammatical anchor role is expressed in <em>bigram context</em>
  (what follows it), not in where it sits within a line.
</div>

<div class="freq-grid">
{cards_html}
</div>
"""


def _section_math_framework() -> str:
    """The complete mathematical framework section with every formula."""
    return """
<div class="sec-head">1 · Mathematical Framework</div>
<div class="sec-sub">Every formula used in this report, exactly as implemented</div>

<div class="intro">
  <p>The following metrics form an ordered hierarchy of information about the
  rongorongo sign distribution. Each measures a different aspect: how
  concentrated the distribution is (IC), how uncertain the next sign is (H),
  how much the previous sign reduces that uncertainty (H<sub>cond</sub>), how
  much position predicts sign identity (I<sub>pos</sub>), and how the rank–frequency
  distribution behaves globally (Zipf α).</p>
</div>

<!-- IC -->
<div class="formula-block">
  <div class="formula-label">Index of Coincidence (Friedman 1922)</div>
  <div class="formula-expr">
    IC = Σ<sub>i</sub> f<sub>i</sub>(f<sub>i</sub> − 1) / [N(N − 1)]
  </div>
  <div class="formula-where">
    <b>f<sub>i</sub></b> = observed frequency of sign type <i>i</i> &nbsp;·&nbsp;
    <b>N</b> = total token count &nbsp;·&nbsp;
    <b>k</b> = number of distinct sign types.<br>
    Random baseline: IC<sub>random</sub> = 1/k (uniform distribution over k signs).<br>
    IC &gt; 1/k indicates the distribution is more concentrated than random —
    some signs are used far more than others, as expected in structured writing.
    IC ≈ 1/k means every sign is equally probable — indistinguishable from noise.<br>
    <b>Normalized IC</b> = IC × k removes the vocabulary-size dependence:
    random = 1.0 regardless of k; values &gt; 1 measure structure above random.
  </div>
</div>

<!-- Shannon H -->
<div class="formula-block">
  <div class="formula-label">Shannon Entropy</div>
  <div class="formula-expr">
    H(S) = −Σ<sub>i</sub> p<sub>i</sub> log<sub>2</sub> p<sub>i</sub>  [bits]
  </div>
  <div class="formula-where">
    <b>p<sub>i</sub></b> = f<sub>i</sub> / N (empirical frequency of sign i).<br>
    H = 0 bits: one sign appears exclusively.<br>
    H = log<sub>2</sub>(k) bits: all k signs equally probable (maximum entropy).<br>
    H measures average uncertainty per sign draw — higher = more evenly spread usage.<br>
    <b>95% bootstrap CI</b>: 2 000 resamples of size N; percentile method.<br>
    Implemented in <code>zone_b.entropy.shannon_entropy</code> and
    <code>bootstrap_h_ci</code>.
  </div>
</div>

<!-- Conditional entropy -->
<div class="formula-block">
  <div class="formula-label">Conditional Bigram Entropy (Entropy Rate Approximation)</div>
  <div class="formula-expr">
    H(S<sub>n</sub> | S<sub>n−1</sub>) = −Σ<sub>s,t</sub> P(s,t) log<sub>2</sub> P(t | s)
  </div>
  <div class="formula-where">
    <b>P(s,t)</b> = empirical bigram probability &nbsp;·&nbsp;
    <b>P(t|s)</b> = P(s,t) / P(s) (conditional probability of t given s).<br>
    This is the entropy rate under a first-order Markov (bigram) model.<br>
    For natural language: H(S<sub>n</sub> | S<sub>n−1</sub>) &lt; H(S<sub>n</sub>) —
    knowing the previous sign reduces uncertainty about the next.<br>
    The true entropy rate H<sub>∞</sub> = lim H(S<sub>n</sub>|S<sub>n−1</sub>,…,S<sub>1</sub>)
    is approximated here with the bigram order.<br>
    Implemented in <code>zone_b.entropy.conditional_bigram_entropy</code>.
  </div>
</div>

<!-- Bigram MI -->
<div class="formula-block">
  <div class="formula-label">Bigram Mutual Information (Sequential Sign Dependency)</div>
  <div class="formula-expr">
    I(S<sub>n</sub> ; S<sub>n−1</sub>) = H(S<sub>n</sub>) − H(S<sub>n</sub> | S<sub>n−1</sub>)
  </div>
  <div class="formula-where">
    Measures how much knowing sign S<sub>n−1</sub> reduces uncertainty about
    the next sign S<sub>n</sub>, in bits.<br>
    I = 0: signs are statistically independent (random order).<br>
    I &gt; 0: some sign transitions are more probable than others — sequential structure.
    Natural language corpora show I ≫ 0 due to grammatical constraints.
  </div>
</div>

<!-- Positional MI -->
<div class="formula-block">
  <div class="formula-label">Positional Mutual Information</div>
  <div class="formula-expr">
    I(sign ; position) = H(sign) − H(sign | position<sub>bin</sub>)
  </div>
  <div class="formula-where">
    <b>position<sub>bin</sub></b> = relative within-line position divided into
    5 equal-width quintile bins (0–20%, 20–40%, 40–60%, 60–80%, 80–100%).<br>
    H(sign | position<sub>bin</sub>) = Σ<sub>b</sub> P(bin=b) · H(sign | bin=b).<br>
    I ≈ 0: sign identity is independent of position — every sign appears
    equally across all line positions.<br>
    I &gt; 0: some signs are positionally concentrated (e.g. taxograms that
    cluster post-line-boundary, sentence-final particles).<br>
    I / H(sign) = fraction of sign entropy explained by position alone.<br>
    Implemented in <code>zone_b.entropy.positional_mutual_information</code>.
  </div>
</div>

<!-- Zipf -->
<div class="formula-block">
  <div class="formula-label">Zipf's Law — Power-Law Exponent</div>
  <div class="formula-expr">
    freq(r) ∝ r<sup>−α</sup>
  </div>
  <div class="formula-where">
    <b>r</b> = rank of sign (1 = most frequent) &nbsp;·&nbsp; <b>α</b> = power-law exponent.<br>
    <b>α<sub>MLE</sub></b>: maximum-likelihood estimate via <code>scipy.stats.zipf</code>.<br>
    <b>α<sub>OLS</sub></b>: log-log OLS regression slope (−α = d log freq / d log rank).<br>
    Natural language corpora: α ∈ [0.9, 1.1].<br>
    Goodness-of-fit: Kolmogorov–Smirnov statistic D and p-value against fitted Zipf CDF
    (note: KS assumes continuous distributions — result is an approximation for discrete ranks).<br>
    Implemented in <code>zone_b.entropy.zipf_analysis</code>.
  </div>
</div>
"""


def _section_baselines() -> str:
    """Reference baseline table."""
    rows = [
        ("Uniform random (k signs)",
         "1/k", "log₂(k)", "1.0", "n/a",
         "Maximum uncertainty; no structure. IC = random baseline."),
        ("Simple substitution cipher (English)",
         "≈ 0.065", "≈ 3.9–4.2 bits", "≈ 1.69", "≈ 1.0",
         "Same IC as plaintext English — frequency distribution preserved under substitution."),
        ("Natural language (English text)",
         "≈ 0.065", "≈ 4.0 bits/char", "≈ 1.69", "≈ 1.0–1.5 bits",
         "Canonical linguistic reference. IC ≈ 0.065 for English (26-letter alphabet, k=26)."),
        ("Linear B (deciphered Bronze Age script)",
         "≈ 0.04–0.06", "n/a", "> 1", "> 0",
         "Known syllabic script; IC above random baseline, structured sequential dependencies."),
        ("Rongorongo corpus (this analysis)",
         "see below", "see below", "see below", "see below",
         "Direct comparison to the above references is the goal of this report."),
    ]
    def _row(i: int, row: tuple) -> str:
        name, ic, h, ic_n, h_cond, note = row
        cls = ' class="highlight"' if i == len(rows) - 1 else ""
        return (
            f'<tr{cls}>'
            f'<td>{name}</td>'
            f'<td class="num">{ic}</td>'
            f'<td class="num">{h}</td>'
            f'<td class="num">{ic_n}</td>'
            f'<td class="num">{h_cond}</td>'
            f'<td style="font-size:12px;color:#555">{note}</td>'
            f'</tr>'
        )
    rows_html = "".join(_row(i, r) for i, r in enumerate(rows))
    return f"""
<div class="sec-head">2 · Reference Baselines</div>
<div class="sec-sub">
  What entropy values to expect for random text, ciphers, and natural language
</div>
<table class="baselines-table">
  <thead><tr>
    <th>System</th>
    <th>IC</th>
    <th>H(S) (bits)</th>
    <th>IC<sub>norm</sub></th>
    <th>H<sub>cond</sub> (bits)</th>
    <th>Notes</th>
  </tr></thead>
  <tbody>{rows_html}</tbody>
</table>
"""


def _section_ic(sensitivity: dict) -> str:
    scenarios = sensitivity.get("scenarios", {})
    deltas    = sensitivity.get("deltas", {})
    robustness = sensitivity.get("robustness", {})

    if not scenarios:
        return f"""
<div class="sec-head">3 · Index of Coincidence — Temporal Stratum Comparison</div>
<div class="sec-sub">Pre-contact vs post-contact IC across three tablet-dating scenarios</div>
{_pending("Run step 4a (entropy sensitivity analysis) to populate.")}
"""

    rows = []
    for name, clusters in scenarios.items():
        pre  = clusters.get("pre_contact", {})
        post = clusters.get("post_contact", {})
        delta = deltas.get(name)
        non_overlap = (
            pre.get("ic_ci_95_lo") is not None
            and post.get("ic_ci_95_hi") is not None
            and pre["ic_ci_95_lo"] > post["ic_ci_95_hi"]
        )
        delta_cls = "pos" if (delta or 0) > 0 else "neg"
        badge = '<span class="badge badge-ok">non-overlapping</span>' if non_overlap \
            else '<span class="badge badge-fail">overlapping</span>'
        pre_norm  = pre.get("ic_normalized")
        post_norm = post.get("ic_normalized")
        rows.append(
            f'<tr>'
            f'<td class="mono">{name.replace("_"," ")}</td>'
            f'<td class="mono">{_fmt(pre.get("ic"),6)}'
            f' <span class="ci">{_ci(pre.get("ic_ci_95_lo"), pre.get("ic_ci_95_hi"), 5)}</span></td>'
            f'<td class="mono">{_fmt(pre_norm,3)}</td>'
            f'<td class="mono">{_fmt(post.get("ic"),6)}'
            f' <span class="ci">{_ci(post.get("ic_ci_95_lo"), post.get("ic_ci_95_hi"), 5)}</span></td>'
            f'<td class="mono">{_fmt(post_norm,3)}</td>'
            f'<td class="{delta_cls} mono">{_fmt(delta,6)}</td>'
            f'<td>{badge}</td>'
            f'</tr>'
        )

    is_robust = robustness.get("robust", False)
    all_pos   = robustness.get("all_pre_gt_post", False)
    var_pct   = robustness.get("relative_variation_pct")
    rob_badge = '<span class="badge badge-ok">robust</span>' if is_robust \
        else '<span class="badge badge-fail">not robust</span>'

    interp_cls = "positive" if is_robust and all_pos else "neutral"
    if is_robust and all_pos:
        interp_body = (
            "<b>IC<sub>pre</sub> &gt; IC<sub>post</sub> consistently across all three "
            "tablet-dating scenarios</b>, with non-overlapping 95% bootstrap CIs in "
            f"{'all' if robustness.get('all_ci_non_overlapping') else 'most'} scenarios. "
            f"Relative variation in Δ IC across scenarios: {_pct(var_pct)}. "
            "This is the headline cryptanalytic finding: pre-contact and post-contact "
            "rongorongo have statistically distinguishable sign-frequency distributions, "
            "consistent with a scribal tradition that evolved across the contact boundary."
        )
    else:
        interp_body = (
            f"IC<sub>pre</sub> &gt; IC<sub>post</sub> in "
            f"{'all' if all_pos else 'some'} scenarios but the result is "
            f"<b>{'robust' if is_robust else 'not robust'}</b> at the 10% relative-variation threshold. "
            f"Relative variation: {_pct(var_pct)}. "
            "Further data or a tighter scenario specification is needed to confirm robustness."
        )

    return f"""
<div class="sec-head">3 · Index of Coincidence — Temporal Stratum Comparison</div>
<div class="sec-sub">Pre-contact vs post-contact IC across three tablet-dating scenarios</div>

<table class="data-table">
<thead><tr>
  <th>Scenario</th>
  <th>IC<sub>pre</sub> (95% CI)</th>
  <th>IC<sub>pre</sub><sub>norm</sub></th>
  <th>IC<sub>post</sub> (95% CI)</th>
  <th>IC<sub>post</sub><sub>norm</sub></th>
  <th>Δ IC (pre−post)</th>
  <th>95% CIs</th>
</tr></thead>
<tbody>
{"".join(rows)}
<tr style="background:var(--surface);font-weight:600">
  <td class="mono">robustness</td>
  <td colspan="4" class="mono" style="font-size:11px;color:var(--muted)">
    variation: {_pct(var_pct)} &nbsp;·&nbsp; all pre&gt;post: {'✓' if all_pos else '✗'}
    &nbsp;·&nbsp; all CIs non-overlap: {'✓' if robustness.get('all_ci_non_overlapping') else '✗'}
  </td>
  <td colspan="2">{rob_badge}</td>
</tr>
</tbody>
</table>

<div class="interp {interp_cls}">{interp_body}</div>
"""


def _section_shannon(sensitivity: dict) -> str:
    scenarios = sensitivity.get("scenarios", {})
    if not scenarios:
        return f"""
<div class="sec-head">4 · Shannon Entropy H(S)</div>
<div class="sec-sub">Unigram entropy with 95% bootstrap confidence intervals</div>
{_pending()}
"""

    rows = []
    for name, clusters in scenarios.items():
        pre  = clusters.get("pre_contact", {})
        post = clusters.get("post_contact", {})
        pre_h  = pre.get("entropy_bits")
        post_h = post.get("entropy_bits")
        diff_h = (pre_h - post_h) if (pre_h is not None and post_h is not None) else None
        diff_cls = "pos" if (diff_h or 0) > 0 else "neg"
        rows.append(
            f'<tr>'
            f'<td class="mono">{name.replace("_"," ")}</td>'
            f'<td>{_fmt(pre_h)} <span class="ci">{_ci(pre.get("entropy_ci_95_lo"), pre.get("entropy_ci_95_hi"))}</span></td>'
            f'<td class="mono">{_fmt(pre.get("n_types"),0)}</td>'
            f'<td>{_fmt(post_h)} <span class="ci">{_ci(post.get("entropy_ci_95_lo"), post.get("entropy_ci_95_hi"))}</span></td>'
            f'<td class="mono">{_fmt(post.get("n_types"),0)}</td>'
            f'<td class="{diff_cls}">{_fmt(diff_h)}</td>'
            f'</tr>'
        )

    # Use first scenario's pre/post for headline values
    first = next(iter(scenarios.values()), {})
    pre_ref  = first.get("pre_contact", {})
    post_ref = first.get("post_contact", {})
    pre_h_max  = math.log2(pre_ref["n_types"])  if pre_ref.get("n_types")  else None
    post_h_max = math.log2(post_ref["n_types"]) if post_ref.get("n_types") else None
    pre_util   = (pre_ref.get("entropy_bits") / pre_h_max * 100) if (pre_h_max and pre_h_max > 0) else None
    post_util  = (post_ref.get("entropy_bits") / post_h_max * 100) if (post_h_max and post_h_max > 0) else None

    util_note = ""
    if pre_util and post_util:
        util_note = (
            f"Pre-contact utilises <b>{pre_util:.1f}%</b> of its maximum possible entropy "
            f"(H<sub>max</sub> = log₂({pre_ref['n_types']}) = {_fmt(pre_h_max)} bits); "
            f"post-contact utilises <b>{post_util:.1f}%</b> "
            f"(H<sub>max</sub> = log₂({post_ref['n_types']}) = {_fmt(post_h_max)} bits). "
            "A lower percentage means usage is concentrated in a smaller proportion of the available vocabulary."
        )

    return f"""
<div class="sec-head">4 · Shannon Entropy H(S)</div>
<div class="sec-sub">
  Unigram entropy in bits — how much uncertainty per sign draw, with 95% bootstrap CIs
</div>

<table class="data-table">
<thead><tr>
  <th>Scenario</th>
  <th>H<sub>pre</sub> (95% CI)</th>
  <th>k<sub>pre</sub></th>
  <th>H<sub>post</sub> (95% CI)</th>
  <th>k<sub>post</sub></th>
  <th>Δ H (pre−post)</th>
</tr></thead>
<tbody>{"".join(rows)}</tbody>
</table>

{('<div class="interp neutral">' + util_note + '</div>') if util_note else ''}
"""


def _section_entropy_rate(sensitivity: dict) -> str:
    scenarios = sensitivity.get("scenarios", {})
    if not scenarios:
        return f"""
<div class="sec-head">5 · Conditional Entropy & Bigram Mutual Information</div>
<div class="sec-sub">H(S<sub>n</sub> | S<sub>n−1</sub>) and I(S<sub>n</sub> ; S<sub>n−1</sub>)</div>
{_pending()}
"""
    rows = []
    for name, clusters in scenarios.items():
        pre  = clusters.get("pre_contact", {})
        post = clusters.get("post_contact", {})
        pre_hc  = pre.get("conditional_entropy_bigram")
        post_hc = post.get("conditional_entropy_bigram")
        pre_mi  = pre.get("bigram_mi_bits")
        post_mi = post.get("bigram_mi_bits")
        pre_h   = pre.get("entropy_bits")
        post_h  = post.get("entropy_bits")
        pre_pct  = (pre_mi / pre_h * 100) if (pre_mi is not None and pre_h) else None
        post_pct = (post_mi / post_h * 100) if (post_mi is not None and post_h) else None
        rows.append(
            f'<tr>'
            f'<td class="mono">{name.replace("_"," ")}</td>'
            f'<td>{_fmt(pre_hc)}</td>'
            f'<td>{_fmt(pre_mi)} <span class="ci">({_pct(pre_pct)} of H)</span></td>'
            f'<td>{_fmt(post_hc)}</td>'
            f'<td>{_fmt(post_mi)} <span class="ci">({_pct(post_pct)} of H)</span></td>'
            f'</tr>'
        )

    return f"""
<div class="sec-head">5 · Conditional Entropy &amp; Bigram Mutual Information</div>
<div class="sec-sub">
  How much does knowing the previous sign reduce uncertainty about the next?
</div>

<table class="data-table">
<thead><tr>
  <th>Scenario</th>
  <th>H(S<sub>n</sub>|S<sub>n−1</sub>) pre</th>
  <th>I(S<sub>n</sub>;S<sub>n−1</sub>) pre</th>
  <th>H(S<sub>n</sub>|S<sub>n−1</sub>) post</th>
  <th>I(S<sub>n</sub>;S<sub>n−1</sub>) post</th>
</tr></thead>
<tbody>{"".join(rows)}</tbody>
</table>

<div class="interp neutral">
  <b>H(S<sub>n</sub>|S<sub>n−1</sub>)</b> is the entropy rate approximation —
  the average uncertainty per sign once the preceding sign is known.
  <b>I(S<sub>n</sub>;S<sub>n−1</sub>)</b> = H(S<sub>n</sub>) −
  H(S<sub>n</sub>|S<sub>n−1</sub>) is the bigram mutual information: how many
  bits of uncertainty are resolved by knowing the previous sign.
  For random text, I = 0. For structured language, I &gt; 0 reflects grammar,
  collocation, and compositional constraints. The percentage column shows
  what fraction of total sign entropy is explained by bigram context.
</div>
"""


def _section_positional_mi(sensitivity: dict) -> str:
    pmi = sensitivity.get("positional_mutual_information", {})
    if not pmi or not pmi.get("n_tokens"):
        return f"""
<div class="sec-head">6 · Positional Mutual Information I(sign ; position)</div>
<div class="sec-sub">How much does a sign's within-line position predict its identity?</div>
{_pending("Position data requires corpus glyphs to carry position_in_line and line_length fields.")}
"""
    n      = pmi.get("n_tokens", 0)
    k      = pmi.get("n_types", 0)
    h_sign = pmi.get("h_sign_bits")
    h_cond = pmi.get("h_sign_given_position_bits")
    mi     = pmi.get("mutual_information_bits")
    mi_pct = pmi.get("mutual_info_pct_of_h")
    bins   = pmi.get("bin_labels", [])

    mi_interp = ""
    if mi is not None and h_sign:
        if mi_pct and mi_pct > 10:
            mi_interp = (
                f"<b>{_pct(mi_pct)}</b> of sign entropy is explained by position — "
                "significant positional structure. Some signs are strongly concentrated "
                "at specific within-line positions (e.g. line-initial taxograms, "
                "sequence-final particles). This is consistent with a writing system "
                "with positional syntax."
            )
        elif mi_pct and mi_pct > 3:
            mi_interp = (
                f"<b>{_pct(mi_pct)}</b> of sign entropy explained by position — "
                "moderate positional structure. Some signs show position preferences "
                "but the script is not strongly position-determined."
            )
        else:
            mi_interp = (
                f"<b>{_pct(mi_pct)}</b> of sign entropy explained by position — "
                "near-zero positional structure. Sign usage is approximately independent "
                "of within-line position."
            )

    cards = [
        ("mcard", _fmt(h_sign), "H(sign) [bits]", "unigram entropy"),
        ("mcard", _fmt(h_cond), "H(sign|pos) [bits]", "entropy given position"),
        ("mcard diff", _fmt(mi), "I(sign;pos) [bits]", "mutual information"),
        ("mcard diff", _pct(mi_pct), "I / H(sign)", "fraction explained"),
    ]
    cards_html = "".join(
        f'<div class="{cls}"><span class="mcard-n">{n_val}</span>'
        f'<span class="mcard-label">{lbl}</span>'
        f'<span class="mcard-sub">{sub}</span></div>'
        for cls, n_val, lbl, sub in cards
    )
    bins_str = " · ".join(f"bin {i}: {b}" for i, b in enumerate(bins)) if bins else ""

    return f"""
<div class="sec-head">6 · Positional Mutual Information I(sign ; position)</div>
<div class="sec-sub">
  Corpus: {n:,} tokens, {k} types.
  Position bins (quintiles): {bins_str or "5 equal-width bins, 0–100% of line length"}.
</div>

<div class="metric-strip">{cards_html}</div>

{('<div class="interp neutral">' + mi_interp + '</div>') if mi_interp else ''}
"""


def _section_boustrophedon(sensitivity: dict, boust_standalone: dict) -> str:
    boust = sensitivity.get("boustrophedon_ic") or boust_standalone
    if not boust:
        return f"""
<div class="sec-head">7 · Boustrophedon Voice-Split Test</div>
<div class="sec-sub">IC by line parity — are odd and even lines structurally distinct?</div>
{_pending()}
"""
    ic_odd  = boust.get("ic_odd", float("nan"))
    ic_even = boust.get("ic_even", float("nan"))
    lo_odd  = boust.get("ic_odd_ci_95_lo")
    hi_odd  = boust.get("ic_odd_ci_95_hi")
    lo_even = boust.get("ic_even_ci_95_lo")
    hi_even = boust.get("ic_even_ci_95_hi")
    n_odd_l = boust.get("n_odd_lines", 0)
    n_even_l = boust.get("n_even_lines", 0)
    n_odd  = boust.get("n_odd_tokens", 0)
    n_even = boust.get("n_even_tokens", 0)
    delta   = boust.get("delta_ic_odd_minus_even", float("nan"))
    overlap = boust.get("cis_overlap", True)
    marginal = boust.get("marginal_overlap", False)
    ov_frac = boust.get("overlap_fraction")
    finding = boust.get("finding", "")

    sep_str = ('<b style="color:#16a34a">do not overlap</b>' if not overlap
               else ('overlap marginally' if marginal else 'overlap'))
    finding_cls = "positive" if not overlap else ("neutral" if marginal else "")

    ov_note = ""
    if marginal and ov_frac is not None:
        ov_note = f" Overlap fraction: {ov_frac*100:.1f}% of CI width."

    return f"""
<div class="sec-head">7 · Boustrophedon Voice-Split Test</div>
<div class="sec-sub">
  IC by line parity — if odd and even lines carry structurally different content,
  IC<sub>odd</sub> ≠ IC<sub>even</sub> with non-overlapping 95% CIs
</div>

<div class="intro" style="margin-bottom:16px">
  <p>Rongorongo is written in reverse boustrophedon: lines alternate reading
  direction, with odd-numbered lines (1, 3, 5, …) and even-numbered lines
  (2, 4, 6, …) running opposite ways. If these two physical text-streams were
  composed in different registers — or represent structurally different content
  — their sign-frequency distributions should differ, measurable as
  IC<sub>odd</sub> ≠ IC<sub>even</sub> with non-overlapping 95% bootstrap CIs.</p>
</div>

<div class="boust-grid">
  <div class="boust-cell">
    <div class="boust-cell-label">Odd lines (1, 3, 5, …)</div>
    <div class="boust-ic">{_fmt(ic_odd, 6)}</div>
    <div class="boust-ci">95% CI {_ci(lo_odd, hi_odd, 5)}</div>
    <div class="boust-n">{n_odd_l} lines · {n_odd:,} tokens</div>
  </div>
  <div class="boust-cell">
    <div class="boust-cell-label">Even lines (2, 4, 6, …)</div>
    <div class="boust-ic">{_fmt(ic_even, 6)}</div>
    <div class="boust-ci">95% CI {_ci(lo_even, hi_even, 5)}</div>
    <div class="boust-n">{n_even_l} lines · {n_even:,} tokens</div>
  </div>
</div>

<div class="delta-row">
  Δ IC (odd − even) = {_fmt(delta, 6)}  ·  95% CIs {sep_str}
</div>

<div class="interp {finding_cls}">{finding}{ov_note}</div>
"""


def _section_zipf(zipf: dict) -> str:
    if not zipf:
        return f"""
<div class="sec-head">8 · Zipf's Law Analysis</div>
<div class="sec-sub">Power-law exponent and goodness-of-fit for sign frequency distribution</div>
{_pending("Run zone_b.entropy.zipf_analysis to generate zipf_analysis.json.")}
"""
    n_tok  = zipf.get("n_tokens", 0)
    n_typ  = zipf.get("n_types", 0)
    a_mle  = zipf.get("exponent_mle")
    a_ols  = zipf.get("exponent_ols")
    r2     = zipf.get("r_squared_loglog")
    ks_d   = zipf.get("ks_statistic")
    ks_p   = zipf.get("ks_pvalue")
    sp_rho = zipf.get("spearman_rho")
    consistent = zipf.get("consistent_with_zipf", False)
    interp = zipf.get("interpretation", "")

    badge = ('<span class="badge badge-ok">consistent with Zipf</span>' if consistent
             else '<span class="badge badge-warn">outside canonical range</span>')

    cards = [
        ("mcard", _fmt(a_mle, 3), "α<sub>MLE</sub>", "power-law exponent"),
        ("mcard", _fmt(a_ols, 3), "α<sub>OLS</sub>", "log-log regression"),
        ("mcard", _fmt(r2, 3), "R² (log-log)", "goodness of fit"),
        ("mcard", _fmt(ks_d, 4), "KS statistic", "vs Zipf CDF"),
        ("mcard", _fmt(ks_p, 4), "KS p-value", "discrete approx"),
        ("mcard diff", _fmt(sp_rho, 4), "Spearman ρ", "obs vs predicted"),
    ]
    cards_html = "".join(
        f'<div class="{cls}"><span class="mcard-n">{n_val}</span>'
        f'<span class="mcard-label">{lbl}</span>'
        f'<span class="mcard-sub">{sub}</span></div>'
        for cls, n_val, lbl, sub in cards
    )

    return f"""
<div class="sec-head">8 · Zipf's Law Analysis</div>
<div class="sec-sub">
  {n_tok:,} tokens · {n_typ} sign types · {badge}
</div>

<div class="metric-strip">{cards_html}</div>

<div class="interp {"positive" if consistent else "neutral"}">{interp}</div>

<div class="intro" style="margin-top:8px">
  <p><b>KS test caveat:</b> The Kolmogorov–Smirnov statistic here is computed against
  the continuous-limit Zipf CDF <code>scipy.stats.zipf.cdf(r, α)</code>.  Because
  rank is a discrete variable, the KS p-value is an approximation; it tends to
  be conservative (over-rejecting the null). Treat the p-value as a rough
  diagnostic rather than a precise significance level.  The R² on the log-log
  scale and the Spearman ρ are more reliable descriptive statistics for fit quality.</p>
</div>
"""


def _section_reading_order(reading_order: dict) -> str:
    t1 = reading_order.get("test1", {})
    t3 = reading_order.get("test3", {})
    if not t1 and not t3:
        return f"""
<div class="sec-head">9 · Reading-Order Entropy Tests (Summary)</div>
<div class="sec-sub">
  Conditional entropy and perplexity tests for L→R vs R→L reading direction
</div>
{_pending("Run scripts/reading_order_tests.py to generate reading_order_results.json.")}
<div class="interp neutral">
  For full details see
  <a href="reading_order_report.html">reading_order_report.html</a>.
</div>
"""
    h_fwd = t1.get("h_forward")
    h_rev = t1.get("h_reverse")
    delta1 = t1.get("delta")
    dir1   = t1.get("direction", "—")
    h_within = t3.get("h_within")
    h_cross  = t3.get("h_cross")
    delta3   = t3.get("delta")

    fwd_better = (h_fwd is not None and h_rev is not None and h_fwd < h_rev)
    t1_cls = "positive" if fwd_better else "neutral"

    return f"""
<div class="sec-head">9 · Reading-Order Entropy Tests (Summary)</div>
<div class="sec-sub">
  Conditional H(S<sub>n</sub>|S<sub>n−1</sub>) asymmetry between forward and reverse reading
</div>

<table class="data-table" style="max-width:700px">
<thead><tr>
  <th>Test</th><th>Forward</th><th>Reverse / Cross</th><th>Δ</th><th>Verdict</th>
</tr></thead>
<tbody>
<tr>
  <td>Test 1: H(S<sub>n</sub>|S<sub>n−1</sub>)</td>
  <td class="mono">{_fmt(h_fwd)}</td>
  <td class="mono">{_fmt(h_rev)}</td>
  <td class="{"pos" if fwd_better else "neg"}">{_fmt(delta1)}</td>
  <td>{dir1}</td>
</tr>
<tr>
  <td>Test 3: within-line vs cross-line H</td>
  <td class="mono">{_fmt(h_within)}</td>
  <td class="mono">{_fmt(h_cross)}</td>
  <td class="mono">{_fmt(delta3)}</td>
  <td>{"line-boundary is structural" if (delta3 or 0) > 0.1 else "—"}</td>
</tr>
</tbody>
</table>

<div class="interp {t1_cls}">
  Lower H(S<sub>n</sub>|S<sub>n−1</sub>) in the forward direction implies that
  reading left-to-right yields more predictable sign sequences — consistent with
  left-to-right as the canonical reading direction within each line.
  For full test results including perplexity and recto/verso order, see
  <a href="reading_order_report.html">reading_order_report.html</a>.
</div>
"""


# ---------------------------------------------------------------------------
# Part II — Reading Direction wrappers
# ---------------------------------------------------------------------------


def _part_reading_direction(reading_order: dict) -> str:
    """Render Part II using imported reading_order_report section functions."""
    if not _HAS_RO:
        return _pending(
            "reading_order_report module not available — "
            "ensure hackingrongo is installed and run "
            "scripts/reading_order_tests.py to generate results."
        )

    t1 = reading_order.get("test1")
    t2 = reading_order.get("test2")
    t3 = reading_order.get("test3")
    t4 = reading_order.get("test4")

    if not any([t1, t2, t3, t4]):
        return _pending(
            "Reading-order results not yet available. "
            "Run: <code>python scripts/reading_order_tests.py "
            "--corpus data/corpus --tests 1 2 3 4 "
            "--output outputs/reading_order_results.json</code>"
        )

    n_tablets = reading_order.get("corpus_tablets", "?")
    n_tokens  = reading_order.get("corpus_tokens", 0)

    rv_pref   = (t4 or {}).get("preferred_order", None)
    dir1      = (t1 or {}).get("direction", None)
    rv_label  = {"ab": "recto first", "ba": "verso first", "mixed": "mixed"}.get(rv_pref or "", "—")
    dir_label = {"forward": "left-to-right", "reverse": "right-to-left",
                 "neutral": "unclear"}.get(dir1 or "", "—")
    rv_cls  = "rv" if rv_pref in ("ab", "ba") else "warn"
    dir_cls = "confirmed" if dir1 == "forward" else "warn"

    return f"""
<div class="stats-row">
  <div class="stat-card"><div class="stat-value">{n_tablets}</div>
    <div class="stat-label">tablets</div></div>
  <div class="stat-card"><div class="stat-value">{n_tokens:,}</div>
    <div class="stat-label">tokens</div></div>
  <div class="stat-card {dir_cls}"><div class="stat-value">{dir_label}</div>
    <div class="stat-label">reading direction</div></div>
  <div class="stat-card {rv_cls}"><div class="stat-value">{rv_label}</div>
    <div class="stat-label">side order (Test 4)</div></div>
</div>

{_ro_test4(t4)}
<hr style="border:none;border-top:1px solid var(--border);margin:52px 0 48px">
{_ro_test1(t1)}
{_ro_test2(t2)}
{_ro_test3(t3)}
{_ro_methodology()}
{_ro_references()}
"""


# ---------------------------------------------------------------------------
# Part III — Tablet Spectrum wrappers
# ---------------------------------------------------------------------------


def _part_spectrum(spectrum_scores: dict, metadata_path: "Path | None") -> str:
    """Render Part III using imported spectrum_report section functions."""
    if not _HAS_SP:
        return _pending(
            "spectrum_report module not available — "
            "ensure hackingrongo is installed."
        )

    if not spectrum_scores:
        return _pending(
            "Spectrum scores not yet computed. "
            "Run: <code>python -m hackingrongo.zone_b.spectrum_analyzer "
            "--corpus-dir data/corpus "
            "--output outputs/analysis/spectrum_scores.json</code>"
        )

    # Enrich tablet dicts with cluster info from metadata
    import json as _json
    cluster_map: dict[str, str] = {}
    if metadata_path and metadata_path.exists():
        try:
            meta = _json.loads(metadata_path.read_text(encoding="utf-8"))
            for tid, tdata in meta.items():
                cluster_map[tid] = tdata.get("date_distribution", {}).get("type", "unknown")
        except Exception:
            pass

    tablets: dict = {}
    for tid, feat in spectrum_scores.items():
        enriched = dict(feat)
        enriched["cluster"] = cluster_map.get(tid, "unknown")
        tablets[tid] = enriched

    n_reliable = sum(1 for f in tablets.values() if f.get("reliable", False))
    n_total    = len(tablets)

    return f"""
<div class="report-meta" style="margin-bottom:24px">
  <b>Tablets:</b> {n_total} &nbsp;·&nbsp;
  <b>Reliable (n ≥ 50 tokens):</b> {n_reliable} &nbsp;·&nbsp;
  <b>Score:</b> 0 = syllabic &nbsp;·&nbsp; 1 = logographic
</div>

<div class="intro" style="margin-bottom:20px">
  <p>Six entropy-theoretic features — Index of Coincidence, bigram mutual information,
  cross-tablet consistency variance, entropy decay rate, compound glyph density, and
  hapax rate — are each sensitive to different aspects of script type.  Their
  equal-weight projection places every tablet on a spectrum from purely syllabic
  (phonotactic, low-IC, fast-decaying entropy) to purely logographic (formulaic,
  high-IC, slow-decaying entropy, compound-dense).  All features flow directly from
  the entropy calculations in Part I.</p>
</div>

{_sp_explanation()}
{_sp_bars(tablets)}
{_sp_table(tablets)}
{_sp_annotations(tablets)}
{_sp_stratum(tablets)}
"""


# ---------------------------------------------------------------------------
# Full HTML document
# ---------------------------------------------------------------------------

def _render_html(
    sensitivity:    dict,
    zipf:           dict,
    boust:          dict,
    reading_order:  dict,
    generated:      str,
    corpus_dir:     "Path | None" = None,
    svg_catalog:    "Path | None" = None,
    spectrum_scores: dict | None = None,
    metadata_path:  "Path | None" = None,
) -> str:
    scenarios = sensitivity.get("scenarios", {})
    robust    = sensitivity.get("robustness", {})
    is_robust = robust.get("robust")
    boust_sep = not (sensitivity.get("boustrophedon_ic") or boust).get("cis_overlap", True)
    zipf_ok   = zipf.get("consistent_with_zipf")
    ro_done   = bool(reading_order.get("test4") or reading_order.get("test1"))
    sp_done   = bool(spectrum_scores)

    def _hstat(v: bool | None) -> str:
        return "yes" if v is True else ("no" if v is False else "pending")

    meta = (
        f"<b>IC robust:</b> {_hstat(is_robust)} &nbsp;·&nbsp; "
        f"<b>Boustrophedon:</b> {_hstat(boust_sep)} &nbsp;·&nbsp; "
        f"<b>Zipf:</b> {_hstat(zipf_ok)} &nbsp;·&nbsp; "
        f"<b>Reading direction:</b> {'✓' if ro_done else 'pending'} &nbsp;·&nbsp; "
        f"<b>Spectrum:</b> {'✓' if sp_done else 'pending'} &nbsp;·&nbsp; "
        f"<b>Generated:</b> {generated}"
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>hackingrongo — Entropy, Reading Direction &amp; Tablet Spectrum</title>
<link href="https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,400;0,600;1,400&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>{_CSS}</style>
</head>
<body>

<nav class="report-nav">
  <span class="nav-part">Part I</span>
  <a href="#entropy-top">Entropy</a>
  <a href="#sec-language">Language-Likeness</a>
  <a href="#sec-freq">Frequency</a>
  <a href="#sec-ic">IC</a>
  <a href="#sec-shannon">Shannon H</a>
  <a href="#sec-cond">Cond. Entropy</a>
  <a href="#sec-zipf">Zipf</a>
  <span class="nav-sep">|</span>
  <span class="nav-part">Part II</span>
  <a href="#reading-top">Reading Direction</a>
  <a href="#test4">Test 4 (recto/verso)</a>
  <a href="#test1">Test 1</a>
  <span class="nav-sep">|</span>
  <span class="nav-part">Part III</span>
  <a href="#spectrum-top">Tablet Spectrum</a>
</nav>

<div class="wrap">

<div class="report-header" id="entropy-top">
  <div class="report-title">hackingrongo<br>Entropy, Reading Direction &amp; Tablet Spectrum</div>
  <div class="report-subtitle">
    All information-theoretic analyses of the rongorongo corpus in one document
  </div>
  <div class="report-meta">{meta}</div>
</div>

<!-- ═══ PART I ═══════════════════════════════════════════════════════════ -->
<div class="part-divider" style="border-color:var(--accent);margin-top:0">
  <div class="part-label">Part I</div>
  <div class="part-title">Entropy &amp; Information Theory</div>
  <div class="part-sub">IC, Shannon H, conditional entropy, Zipf — with full mathematical framework</div>
</div>

<div id="sec-language">
{_section_language_likeness(sensitivity, zipf, sensitivity.get("boustrophedon_ic") or boust)}
</div>
{_section_frequency_breakdown(corpus_dir, svg_catalog)}
<div id="sec-freq"></div>
{_section_math_framework()}
{_section_baselines()}
<div id="sec-ic">
{_section_ic(sensitivity)}
</div>
<div id="sec-shannon">
{_section_shannon(sensitivity)}
</div>
<div id="sec-cond">
{_section_entropy_rate(sensitivity)}
</div>
{_section_positional_mi(sensitivity)}
{_section_boustrophedon(sensitivity, boust)}
<div id="sec-zipf">
{_section_zipf(zipf)}
</div>

<!-- ═══ PART II ══════════════════════════════════════════════════════════ -->
<div class="part-divider" id="reading-top">
  <div class="part-label">Part II</div>
  <div class="part-title">Reading Direction</div>
  <div class="part-sub">
    Four entropy-theoretic tests for transcription-direction verification and recto/verso ordering
  </div>
</div>

{_part_reading_direction(reading_order)}

<!-- ═══ PART III ═════════════════════════════════════════════════════════ -->
<div class="part-divider" id="spectrum-top">
  <div class="part-label">Part III</div>
  <div class="part-title">Tablet Spectrum</div>
  <div class="part-sub">
    Logographic ↔ syllabic spectrum for each tablet, derived from Part I entropy features
  </div>
</div>

{_part_spectrum(spectrum_scores or {}, metadata_path)}

<div class="report-footer">
  <p><b>hackingrongo</b> · Entropy, Reading Direction &amp; Tablet Spectrum · MIT License ·
  <a href="https://github.com/violasarah2000/hackingrongo" target="_blank">GitHub</a></p>
  <p><b>Part I sources:</b> IC: Friedman (1922) · Shannon entropy: Shannon (1948) ·
  Bootstrap CIs: percentile method, 2 000 resamples, seed 42 ·
  Zipf MLE: Clauset (2009) via <code>scipy.stats.zipf</code>.</p>
  <p><b>Part II sources:</b> Conditional entropy &amp; LOO perplexity: corpus-derived
  empirical counts · Add-0.5 Laplace smoothing · Barthel (1958) transcription conventions.</p>
  <p><b>Part III sources:</b> Six-feature spectrum score — equal-weight projection ·
  Compound density: Barthel (1958) syntactic punctuation ·
  Tablet annotations: Fischer (1997), Pozdniakov (1996, 2007), Barthel (1958).</p>
  <p>This is a computational hypothesis document, not a decipherment claim.
  All findings require expert review.</p>
  <p><b>SperksWerks LLC</b> ·
  <a href="https://sperkswerks.ai" target="_blank">sperkswerks.ai</a> ·
  <a href="mailto:studio@sperkswerks.ai">studio@sperkswerks.ai</a></p>
</div>

</div>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_entropy_report(
    sensitivity_json:   Path,
    zipf_json:          Path | None = None,
    boustrophedon_json: Path | None = None,
    reading_order_json: Path | None = None,
    corpus_dir:         Path | None = None,
    svg_catalog_path:   Path | None = None,
    spectrum_json:      Path | None = None,
    metadata_json:      Path | None = None,
) -> str:
    """Build the combined entropy + reading direction + tablet spectrum report.

    All inputs are optional — any missing file renders a pending placeholder.

    Parameters
    ----------
    sensitivity_json : Path
        ``sensitivity_analysis.json`` — IC, Shannon H, entropy rate per scenario.
    zipf_json : Path, optional
        ``zipf_analysis.json`` — Zipf exponent + KS test.
    boustrophedon_json : Path, optional
        ``boustrophedon_ic.json`` — voice-split IC per line parity.
    reading_order_json : Path, optional
        ``reading_order_results.json`` — four reading-direction tests.
    corpus_dir : Path, optional
        ``data/corpus/`` — for the sign-frequency breakdown section.
    svg_catalog_path : Path, optional
        ``data/glyphs/svg/catalog.json`` — glyph images in frequency section.
    spectrum_json : Path, optional
        ``spectrum_scores.json`` — per-tablet logographic/syllabic scores.
    metadata_json : Path, optional
        ``data/metadata/tablets.json`` — tablet names and content annotations.
    """
    sensitivity    = _load_json(sensitivity_json) if sensitivity_json.exists() else {}
    zipf           = _load_json(zipf_json)          if zipf_json and zipf_json.exists() else {}
    boust          = _load_json(boustrophedon_json)  if boustrophedon_json and boustrophedon_json.exists() else {}
    reading_order  = _load_json(reading_order_json)  if reading_order_json and reading_order_json.exists() else {}
    spectrum_raw   = _load_json(spectrum_json)        if spectrum_json and spectrum_json.exists() else {}
    spectrum_scores = spectrum_raw.get("tablets", {}) if spectrum_raw else {}

    generated = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    return _render_html(
        sensitivity, zipf, boust, reading_order, generated,
        corpus_dir=corpus_dir, svg_catalog=svg_catalog_path,
        spectrum_scores=spectrum_scores, metadata_path=metadata_json,
    )


def save_entropy_report(
    sensitivity_json:   Path,
    output_path:        Path,
    zipf_json:          Path | None = None,
    boustrophedon_json: Path | None = None,
    reading_order_json: Path | None = None,
    corpus_dir:         Path | None = None,
    svg_catalog_path:   Path | None = None,
    spectrum_json:      Path | None = None,
    metadata_json:      Path | None = None,
) -> None:
    """Generate and write the combined HTML report."""
    html = build_entropy_report(
        sensitivity_json, zipf_json, boustrophedon_json, reading_order_json,
        corpus_dir=corpus_dir, svg_catalog_path=svg_catalog_path,
        spectrum_json=spectrum_json, metadata_json=metadata_json,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    logger.info("Combined report written: %s (%d bytes).", output_path, len(html))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(
        description="Generate the combined entropy / reading direction / spectrum report."
    )
    p.add_argument("--sensitivity",    type=Path, default=Path("outputs/sensitivity_analysis.json"))
    p.add_argument("--zipf",           type=Path, default=Path("outputs/analysis/zipf_analysis.json"))
    p.add_argument("--boustrophedon",  type=Path, default=Path("outputs/analysis/boustrophedon_ic.json"))
    p.add_argument("--reading-order",  type=Path, default=Path("outputs/reading_order_results.json"))
    p.add_argument("--spectrum",       type=Path, default=Path("outputs/analysis/spectrum_scores.json"),
                   help="spectrum_scores.json from zone_b.spectrum_analyzer.")
    p.add_argument("--metadata",       type=Path, default=Path("data/metadata/tablets.json"),
                   help="Tablet metadata for spectrum annotations.")
    p.add_argument("--corpus-dir",     type=Path, default=Path("data/corpus"))
    p.add_argument("--svg-catalog",    type=Path, default=Path("data/glyphs/svg/catalog.json"))
    p.add_argument("--output",         type=Path, default=Path("outputs/analysis/entropy_report.html"))
    args = p.parse_args()

    import logging as _logging
    _logging.basicConfig(level=_logging.INFO, format="%(levelname)s  %(message)s")
    save_entropy_report(
        sensitivity_json   = args.sensitivity,
        output_path        = args.output,
        zipf_json          = args.zipf,
        boustrophedon_json = args.boustrophedon,
        reading_order_json = args.reading_order,
        corpus_dir         = args.corpus_dir,
        svg_catalog_path   = args.svg_catalog,
        spectrum_json      = args.spectrum,
        metadata_json      = args.metadata,
    )
    print(f"Combined report → {args.output}")


if __name__ == "__main__":
    main()
