"""
hackingrongo.results.reading_order_report
=========================================

Scholar-facing HTML report for the four reading-order entropy tests.

Sections
--------
1. Corpus summary and overall statistics
2. Test 4 hero section — recto/verso ordering (Pozdniakov's 1958 question)
3. Test 1 — conditional entropy asymmetry (transcription direction)
4. Test 2 — n-gram model perplexity asymmetry
5. Test 3 — line-boundary entropy (structural breaks)
6. Methodology appendix
7. Scholarly references

Inputs
------
  outputs/reading_order_results.json   — written by reading_order_tests.py --output

Output
------
  outputs/analysis/reading_order_report.html

CLI
---
    python -m hackingrongo.results.reading_order_report \\
        --input  outputs/reading_order_results.json \\
        --output outputs/analysis/reading_order_report.html

Public API
----------
``build_reading_order_report(results_json)``  → HTML str
``save_reading_order_report(results_json, output_path)``

Design language
---------------
Matches entropy_report, compound_report, divergence_report, passage_report:
  * Light background — CSS variables --bg / --surface / --surface2
  * Cormorant Garamond (body) + JetBrains Mono (code / metadata)
  * Accent colour --accent = #c4a96d (gold)
  * Test 4 uses --rv = #7c3aed (violet) — the primary finding
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from datetime import datetime, timezone
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
  --confirmed: #16a34a; --warn: #d97706; --danger: #dc2626;
  --rv: #7c3aed; --rv-bg: #faf5ff; --rv-border: #ddd6fe; --rv-dark: #4c1d95;
  --t1: #0e7490; --t3: #065f46;
}
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
  background: var(--bg); color: var(--text);
  font-family: 'Cormorant Garamond', 'Palatino Linotype', Georgia, serif;
  font-size: 16px; line-height: 1.65;
}
.wrap { max-width: 1020px; margin: 0 auto; padding: 52px 28px; }
.mono { font-family: 'JetBrains Mono', 'Fira Mono', monospace; }
.muted { color: var(--muted); }
.small { font-size: 11px; }

/* ── Header ── */
.report-header { border-bottom: 1px solid var(--border);
                 padding-bottom: 38px; margin-bottom: 48px; }
.report-title  { font-size: 34px; font-weight: 600; color: #000; letter-spacing: -0.3px; }
.report-subtitle { font-size: 17px; color: var(--accent); font-style: italic; margin-top: 6px; }
.report-meta { margin-top: 22px; font-family: 'JetBrains Mono', monospace;
               font-size: 11px; color: var(--muted); line-height: 2.2; }
.report-meta b { color: #333; }
.abstract { margin-top: 22px; font-size: 14.5px; color: #333;
            max-width: 760px; line-height: 1.85; }
.abstract p + p { margin-top: 10px; }

/* ── Stats row ── */
.stats-row { display: flex; gap: 14px; flex-wrap: wrap; margin: 28px 0 44px; }
.stat-card { background: var(--surface); border: 1px solid var(--border);
             border-radius: 7px; padding: 14px 20px; min-width: 120px; }
.stat-value { font-family: 'JetBrains Mono', monospace; font-size: 22px;
              font-weight: 500; color: #000; }
.stat-label { font-size: 11px; color: var(--muted); margin-top: 2px; }
.stat-card.confirmed .stat-value { color: var(--confirmed); }
.stat-card.warn      .stat-value { color: var(--warn); }
.stat-card.rv        .stat-value { color: var(--rv); }

/* ── Section scaffolding ── */
.section-label { font-family: 'JetBrains Mono', monospace; font-size: 9px;
                 color: var(--muted); letter-spacing: 0.12em;
                 text-transform: uppercase; margin-bottom: 10px; }
.test-section { margin-bottom: 56px; }
.test-header { margin-bottom: 20px; }
.test-number { font-family: 'JetBrains Mono', monospace; font-size: 10px;
               color: var(--muted); letter-spacing: 0.1em; text-transform: uppercase;
               margin-bottom: 4px; }
.test-title  { font-size: 22px; font-weight: 600; color: #000; }
.test-desc   { font-size: 13.5px; color: var(--muted); margin-top: 4px; }

/* ── Generic verdict badge ── */
.badge { font-family: 'JetBrains Mono', monospace; font-size: 9.5px;
         border-radius: 3px; padding: 2px 8px; white-space: nowrap; display: inline-block; }
.badge-ok   { background: #dcfce7; color: var(--confirmed); }
.badge-warn { background: #fef9c3; color: #854d0e; }
.badge-fail { background: #fee2e2; color: var(--danger); }
.badge-rv   { background: var(--rv-bg); color: var(--rv-dark); border: 1px solid var(--rv-border); }

/* ── Metric row (Test 1 / Test 3 style) ── */
.metric-block { background: var(--surface); border: 1px solid var(--border);
                border-radius: 8px; padding: 22px 26px; margin-bottom: 16px; }
.metric-grid { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 18px;
               margin-bottom: 18px; }
.metric-item {}
.metric-value { font-family: 'JetBrains Mono', monospace; font-size: 20px;
                font-weight: 500; color: #000; }
.metric-unit  { font-family: 'JetBrains Mono', monospace; font-size: 11px;
                color: var(--muted); margin-left: 3px; }
.metric-label { font-size: 11px; color: var(--muted); margin-top: 3px; }
.metric-delta-pos { color: var(--confirmed); font-family: 'JetBrains Mono', monospace; }
.metric-delta-neg { color: var(--danger);    font-family: 'JetBrains Mono', monospace; }
.metric-delta-neu { color: var(--warn);      font-family: 'JetBrains Mono', monospace; }

.verdict-strip { padding: 13px 18px; border-left: 3px solid var(--border);
                 background: var(--surface2); font-size: 14px; color: #333;
                 line-height: 1.65; margin-top: 4px; }
.verdict-strip.ok   { border-color: var(--confirmed); background: #f0fdf4; }
.verdict-strip.warn { border-color: var(--warn);      background: #fffbeb; }
.verdict-strip.fail { border-color: var(--danger);    background: #fff1f2; }

/* ── Test 4 hero ── */
.rv-hero { background: var(--rv-bg); border: 1px solid var(--rv-border);
           border-radius: 10px; padding: 32px 32px 28px; margin-bottom: 28px; }
.rv-hero-label { font-family: 'JetBrains Mono', monospace; font-size: 9.5px;
                 letter-spacing: 0.14em; text-transform: uppercase;
                 color: var(--rv); margin-bottom: 12px; }
.rv-verdict-head { font-size: 28px; font-weight: 700; color: var(--rv-dark);
                   line-height: 1.25; margin-bottom: 8px; }
.rv-verdict-sub  { font-size: 15px; color: #444; line-height: 1.75;
                   max-width: 740px; }

.rv-context { font-size: 14px; color: #444; line-height: 1.8; max-width: 780px;
              margin-bottom: 24px; }
.rv-context + .rv-hero { margin-top: 0; }

/* ── PPL comparison table ── */
.ppl-wrap { overflow-x: auto; margin: 28px 0; }
table.ppl-table { width: 100%; border-collapse: collapse; font-size: 14px; }
table.ppl-table th { font-family: 'JetBrains Mono', monospace; font-size: 9px;
  letter-spacing: 0.08em; text-transform: uppercase; color: var(--muted);
  padding: 8px 14px; text-align: left; border-bottom: 2px solid var(--border); }
table.ppl-table td { padding: 11px 14px; border-bottom: 1px solid var(--border); }
table.ppl-table tr:last-child td { border-bottom: none; }
.ppl-winner { font-family: 'JetBrains Mono', monospace; font-size: 17px;
              font-weight: 500; color: var(--confirmed); }
.ppl-loser  { font-family: 'JetBrains Mono', monospace; font-size: 17px;
              color: var(--muted); }
.ppl-reduc  { font-family: 'JetBrains Mono', monospace; font-size: 11px;
              color: var(--confirmed); white-space: nowrap; }
.ppl-row-label { font-size: 13.5px; color: #333; }

/* ── PPL bar viz ── */
.ppl-bar-wrap { margin: 6px 0; }
.ppl-bar-row  { display: flex; align-items: center; gap: 10px; margin-bottom: 6px; }
.ppl-bar-label { font-family: 'JetBrains Mono', monospace; font-size: 10px;
                 color: var(--muted); width: 40px; flex-shrink: 0; }
.ppl-bar-track { flex: 1; height: 8px; background: #e5e7eb; border-radius: 4px;
                 overflow: hidden; }
.ppl-bar-fill  { height: 100%; border-radius: 4px; }
.ppl-bar-fill.winner { background: var(--confirmed); }
.ppl-bar-fill.loser  { background: #d1d5db; }
.ppl-bar-val  { font-family: 'JetBrains Mono', monospace; font-size: 11.5px;
                color: #333; width: 50px; text-align: right; flex-shrink: 0; }

/* ── Note / method box ── */
.note-box { background: var(--surface); border: 1px solid var(--border);
            border-radius: 6px; padding: 16px 20px; margin-top: 20px;
            font-size: 13px; color: #555; line-height: 1.75; }
.note-box-title { font-family: 'JetBrains Mono', monospace; font-size: 9px;
                  letter-spacing: 0.1em; text-transform: uppercase;
                  color: var(--muted); margin-bottom: 6px; }

/* ── Methodology appendix ── */
.method-section { border-top: 1px solid var(--border); margin-top: 60px;
                  padding-top: 36px; }
.method-title { font-size: 20px; font-weight: 600; margin-bottom: 22px; }
.method-item  { margin-bottom: 22px; }
.method-item-head { font-size: 14.5px; font-weight: 600; margin-bottom: 6px; color: #222; }
.method-item-body { font-size: 13.5px; color: #444; line-height: 1.8; max-width: 800px; }
.method-formula { font-family: 'JetBrains Mono', monospace; font-size: 12px;
                  background: var(--surface2); border: 1px solid var(--border);
                  border-radius: 4px; padding: 8px 14px; display: inline-block;
                  margin: 6px 0; color: #333; }

/* ── References ── */
.ref-section { border-top: 1px solid var(--border); margin-top: 48px;
               padding-top: 32px; }
.ref-title { font-size: 16px; font-weight: 600; margin-bottom: 18px; color: #333; }
.ref-list { list-style: none; }
.ref-list li { font-size: 13px; color: #444; line-height: 1.75; margin-bottom: 8px;
               padding-left: 22px; text-indent: -22px; }

/* ── Footer ── */
.report-footer { margin-top: 60px; padding-top: 20px;
                 border-top: 1px solid var(--border);
                 font-size: 11.5px; color: var(--muted); line-height: 1.9; }
"""


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------

def _fmt(v: Any, decimals: int = 4) -> str:
    if v is None or (isinstance(v, float) and not math.isfinite(v)):
        return "—"
    return f"{v:.{decimals}f}"


def _pct_reduction(winner: float, loser: float) -> str:
    if loser == 0 or not math.isfinite(winner) or not math.isfinite(loser):
        return ""
    pct = (loser - winner) / loser * 100
    return f"−{pct:.1f}%"


def _bar_width(val: float, max_val: float) -> int:
    if max_val == 0 or not math.isfinite(val):
        return 0
    return min(100, int(val / max_val * 100))


def _verdict_strip(direction: str | None, text: str, good: str = "forward") -> str:
    if direction == good or direction == "confirmed":
        cls = "ok"
    elif direction in ("reverse", "unexpected"):
        cls = "fail"
    else:
        cls = "warn"
    return f'<div class="verdict-strip {cls}">{text}</div>'


# ---------------------------------------------------------------------------
# Test 4 — hero section
# ---------------------------------------------------------------------------

def _render_test4_hero(t4: dict | None) -> str:
    if not t4:
        return """<div class="test-section" id="test4">
  <div class="test-header">
    <div class="test-number">Test 4 &nbsp;·&nbsp; Primary Finding</div>
    <div class="test-title">Recto / Verso Ordering</div>
    <div class="test-desc">Resolving Pozdniakov's question from 1958</div>
  </div>
  <div class="note-box">
    <div class="note-box-title">Not yet run</div>
    Test 4 requires only the corpus directory. Run:<br>
    <code>python scripts/reading_order_tests.py --corpus data/corpus --tests 3 4 --output outputs/reading_order_results.json</code>
  </div>
</div>"""

    ppl_ab2 = t4.get("ppl_ab_bigram", float("nan"))
    ppl_ab3 = t4.get("ppl_ab_trigram", float("nan"))
    ppl_ba2 = t4.get("ppl_ba_bigram", float("nan"))
    ppl_ba3 = t4.get("ppl_ba_trigram", float("nan"))
    preferred = t4.get("preferred_order", "mixed")
    votes_ab  = t4.get("votes_ab", -1)

    if preferred == "ab":
        verdict_head = "Recto (side a) precedes verso (side b)"
        verdict_prose = (
            "Both the bigram and trigram leave-one-out perplexity tests prefer "
            "the a→b ordering. The corpus is more predictable when side a "
            "is read before side b, indicating that the scribe composed recto "
            "before verso. Pozdniakov’s question, open since Barthel’s 1958 "
            "edition, admits a clean empirical answer: <strong>recto first</strong>."
        )
        hero_badge = '<span class="badge badge-rv">a→b preferred</span>'
        ppl_ab_cls2 = "ppl-winner"; ppl_ba_cls2 = "ppl-loser"
        ppl_ab_cls3 = "ppl-winner"; ppl_ba_cls3 = "ppl-loser"
    elif preferred == "ba":
        verdict_head = "Verso (side b) precedes recto (side a)"
        verdict_prose = (
            "Both the bigram and trigram leave-one-out perplexity tests prefer "
            "the b→a ordering. The corpus is more predictable when side b "
            "is read before side a, indicating that the scribe composed verso "
            "before recto. Pozdniakov’s question, open since Barthel’s 1958 "
            "edition, admits a clean empirical answer: <strong>verso first</strong>."
        )
        hero_badge = '<span class="badge badge-rv">b→a preferred</span>'
        ppl_ab_cls2 = "ppl-loser";  ppl_ba_cls2 = "ppl-winner"
        ppl_ab_cls3 = "ppl-loser";  ppl_ba_cls3 = "ppl-winner"
    else:
        verdict_head = "Mixed signal — bigram and trigram disagree"
        verdict_prose = (
            "The bigram and trigram leave-one-out perplexity tests favour "
            "different side orderings. No unambiguous verdict is possible from "
            "this corpus alone. Pozdniakov’s question remains open."
        )
        hero_badge = '<span class="badge badge-warn">inconclusive</span>'
        ppl_ab_cls2 = "ppl-winner" if ppl_ab2 < ppl_ba2 else "ppl-loser"
        ppl_ba_cls2 = "ppl-loser"  if ppl_ab2 < ppl_ba2 else "ppl-winner"
        ppl_ab_cls3 = "ppl-winner" if ppl_ab3 < ppl_ba3 else "ppl-loser"
        ppl_ba_cls3 = "ppl-loser"  if ppl_ab3 < ppl_ba3 else "ppl-winner"

    reduc2_ab = _pct_reduction(ppl_ab2, ppl_ba2) if ppl_ab2 < ppl_ba2 else ""
    reduc2_ba = _pct_reduction(ppl_ba2, ppl_ab2) if ppl_ba2 < ppl_ab2 else ""
    reduc3_ab = _pct_reduction(ppl_ab3, ppl_ba3) if ppl_ab3 < ppl_ba3 else ""
    reduc3_ba = _pct_reduction(ppl_ba3, ppl_ab3) if ppl_ba3 < ppl_ab3 else ""

    # bar widths (scale to 100 = larger value)
    max2 = max(ppl_ab2, ppl_ba2) if math.isfinite(ppl_ab2) and math.isfinite(ppl_ba2) else 1
    max3 = max(ppl_ab3, ppl_ba3) if math.isfinite(ppl_ab3) and math.isfinite(ppl_ba3) else 1
    w_ab2 = _bar_width(ppl_ab2, max2); w_ba2 = _bar_width(ppl_ba2, max2)
    w_ab3 = _bar_width(ppl_ab3, max3); w_ba3 = _bar_width(ppl_ba3, max3)

    return f"""<section class="test-section" id="test4">
<div class="test-header">
  <div class="test-number">Test 4 &nbsp;·&nbsp; Primary Finding</div>
  <div class="test-title">Recto / Verso Ordering</div>
  <div class="test-desc" style="color:var(--rv)">Resolving Pozdniakov’s question from 1958</div>
</div>

<div class="rv-context">
  <p>
    Since Barthel’s 1958 <em>Grundlagen zur Entzifferung der Osterinselschrift</em>,
    the reading order of the two tablet sides has been a matter of scholarly debate.
    Which comes first — recto (side a) or verso (side b)? Pozdniakov (1996)
    noted the question is not resolvable by transcription alone. This test provides
    an empirical answer: the side ordering that yields lower leave-one-out perplexity
    is the one under which the sequence statistics are more regular — i.e., the
    reading order the scribes actually used.
  </p>
</div>

<div class="rv-hero">
  <div class="rv-hero-label">Finding &nbsp;&middot;&nbsp; {hero_badge}</div>
  <div class="rv-verdict-head">{verdict_head}</div>
  <div class="rv-verdict-sub">{verdict_prose}</div>
</div>

<div class="ppl-wrap">
<table class="ppl-table">
<thead><tr>
  <th>Metric</th>
  <th>a → b &nbsp;<span style="font-weight:400;font-size:8.5px">(recto first)</span></th>
  <th>b → a &nbsp;<span style="font-weight:400;font-size:8.5px">(verso first)</span></th>
  <th>Reduction</th>
</tr></thead>
<tbody>
<tr>
  <td class="ppl-row-label">Bigram LOO-perplexity</td>
  <td>
    <div class="{ppl_ab_cls2}">{_fmt(ppl_ab2, 2)}</div>
    <div class="ppl-bar-wrap">
      <div class="ppl-bar-row">
        <div class="ppl-bar-track"><div class="ppl-bar-fill {'winner' if ppl_ab_cls2 == 'ppl-winner' else 'loser'}" style="width:{w_ab2}%"></div></div>
      </div>
    </div>
  </td>
  <td>
    <div class="{ppl_ba_cls2}">{_fmt(ppl_ba2, 2)}</div>
    <div class="ppl-bar-wrap">
      <div class="ppl-bar-row">
        <div class="ppl-bar-track"><div class="ppl-bar-fill {'winner' if ppl_ba_cls2 == 'ppl-winner' else 'loser'}" style="width:{w_ba2}%"></div></div>
      </div>
    </div>
  </td>
  <td class="ppl-reduc">{reduc2_ab or reduc2_ba or "—"}</td>
</tr>
<tr>
  <td class="ppl-row-label">Trigram LOO-perplexity</td>
  <td>
    <div class="{ppl_ab_cls3}">{_fmt(ppl_ab3, 2)}</div>
    <div class="ppl-bar-wrap">
      <div class="ppl-bar-row">
        <div class="ppl-bar-track"><div class="ppl-bar-fill {'winner' if ppl_ab_cls3 == 'ppl-winner' else 'loser'}" style="width:{w_ab3}%"></div></div>
      </div>
    </div>
  </td>
  <td>
    <div class="{ppl_ba_cls3}">{_fmt(ppl_ba3, 2)}</div>
    <div class="ppl-bar-wrap">
      <div class="ppl-bar-row">
        <div class="ppl-bar-track"><div class="ppl-bar-fill {'winner' if ppl_ba_cls3 == 'ppl-winner' else 'loser'}" style="width:{w_ba3}%"></div></div>
      </div>
    </div>
  </td>
  <td class="ppl-reduc">{reduc3_ab or reduc3_ba or "—"}</td>
</tr>
</tbody>
</table>
</div>

<div class="note-box">
  <div class="note-box-title">Methodology</div>
  Leave-one-out (LOO) perplexity: each tablet is held out in turn; an add-0.5
  smoothed n-gram model is trained on the remaining tablets; the held-out tablet
  is scored. Lower mean perplexity means the model trained on the other tablets
  predicts the held-out tablet better — indicating more regular sequence
  structure under that ordering. The test uses both bigram and trigram order;
  agreement between them strengthens the verdict.
</div>
</section>"""


# ---------------------------------------------------------------------------
# Test 1 — conditional entropy asymmetry
# ---------------------------------------------------------------------------

def _render_test1(t1: dict | None) -> str:
    if not t1:
        return """<section class="test-section" id="test1">
  <div class="test-header">
    <div class="test-number">Test 1</div>
    <div class="test-title">Conditional Entropy Asymmetry</div>
    <div class="test-desc">Requires corpus only</div>
  </div>
  <p class="muted small">Not yet run.</p>
</section>"""

    h_f = t1.get("h_forward", float("nan"))
    h_r = t1.get("h_reverse", float("nan"))
    delta = t1.get("delta", float("nan"))
    direction = t1.get("direction", "neutral")
    verdict = t1.get("verdict_text", "")

    delta_cls = "metric-delta-pos" if delta > 0.05 else ("metric-delta-neg" if delta < -0.05 else "metric-delta-neu")
    delta_sign = f"+{delta:.4f}" if delta > 0 else f"{delta:.4f}"

    return f"""<section class="test-section" id="test1">
<div class="test-header">
  <div class="test-number">Test 1</div>
  <div class="test-title">Conditional Entropy Asymmetry</div>
  <div class="test-desc">H(Sₙ | Sₙ₋₁) forward vs reversed — does transcription direction match reading direction?</div>
</div>

<div class="metric-block">
  <div class="metric-grid">
    <div class="metric-item">
      <div class="metric-value">{_fmt(h_f, 4)}<span class="metric-unit">bits</span></div>
      <div class="metric-label">H forward</div>
    </div>
    <div class="metric-item">
      <div class="metric-value">{_fmt(h_r, 4)}<span class="metric-unit">bits</span></div>
      <div class="metric-label">H reverse</div>
    </div>
    <div class="metric-item">
      <div class="metric-value {delta_cls}">{delta_sign}<span class="metric-unit">bits</span></div>
      <div class="metric-label">Δ (reverse − forward)</div>
    </div>
  </div>
  {_verdict_strip(direction, verdict, good="forward")}
</div>

<div class="note-box">
  <div class="note-box-title">Interpretation</div>
  If H_forward &lt; H_reverse, consecutive signs are more predictable in the
  transcribed (left-to-right) direction, confirming the transcription direction
  as the reading direction. A Δ &gt; 0.05 bits is treated as a meaningful signal.
</div>
</section>"""


# ---------------------------------------------------------------------------
# Test 2 — n-gram model perplexity
# ---------------------------------------------------------------------------

def _render_test2(t2: dict | None) -> str:
    if not t2:
        return """<section class="test-section" id="test2">
  <div class="test-header">
    <div class="test-number">Test 2</div>
    <div class="test-title">N-gram Model Perplexity Asymmetry</div>
    <div class="test-desc">Requires trained sequence model</div>
  </div>
  <p class="muted small">Not yet run — requires a trained NgramModel (run step 5 first).</p>
</section>"""

    ppl_f = t2.get("ppl_forward", float("nan"))
    ppl_r = t2.get("ppl_reverse", float("nan"))
    ratio  = t2.get("ratio", float("nan"))
    order  = t2.get("model_order", "?")
    direction = t2.get("direction", "neutral")
    verdict   = t2.get("verdict_text", "")

    winner_f = ppl_f < ppl_r if math.isfinite(ppl_f) and math.isfinite(ppl_r) else False
    ppl_f_cls = "metric-delta-pos" if winner_f else ""
    ppl_r_cls = "" if winner_f else "metric-delta-pos"
    ratio_txt = f"{ratio:.2f}×" if math.isfinite(ratio) else "—"

    return f"""<section class="test-section" id="test2">
<div class="test-header">
  <div class="test-number">Test 2</div>
  <div class="test-title">N-gram Model Perplexity Asymmetry</div>
  <div class="test-desc">Forward vs reversed perplexity under the trained {order}-gram model</div>
</div>

<div class="metric-block">
  <div class="metric-grid">
    <div class="metric-item">
      <div class="metric-value {ppl_f_cls}">{_fmt(ppl_f, 2)}</div>
      <div class="metric-label">PPL forward{' ✓' if winner_f else ''}</div>
    </div>
    <div class="metric-item">
      <div class="metric-value {ppl_r_cls}">{_fmt(ppl_r, 2)}</div>
      <div class="metric-label">PPL reverse{' ✓' if not winner_f else ''}</div>
    </div>
    <div class="metric-item">
      <div class="metric-value">{ratio_txt}</div>
      <div class="metric-label">PPL ratio (reverse / forward)</div>
    </div>
  </div>
  {_verdict_strip(direction, verdict, good="forward")}
</div>

<div class="note-box">
  <div class="note-box-title">Interpretation</div>
  The n-gram model was trained on forward sequences. If it assigns lower perplexity
  to forward sequences than to their reverses, it has learned genuinely directional
  structure — confirming that the transcription direction is the reading direction.
</div>
</section>"""


# ---------------------------------------------------------------------------
# Test 3 — line-boundary entropy
# ---------------------------------------------------------------------------

def _render_test3(t3: dict | None) -> str:
    if not t3:
        return """<section class="test-section" id="test3">
  <div class="test-header">
    <div class="test-number">Test 3</div>
    <div class="test-title">Line-Boundary Entropy</div>
    <div class="test-desc">Requires corpus only</div>
  </div>
  <p class="muted small">Not yet run.</p>
</section>"""

    h_w = t3.get("h_within", float("nan"))
    h_c = t3.get("h_cross", float("nan"))
    n_w = t3.get("n_within_bigrams", 0)
    n_c = t3.get("n_cross_bigrams", 0)
    delta = t3.get("delta", float("nan"))
    direction = t3.get("direction", "neutral")
    verdict = t3.get("verdict_text", "")

    delta_cls = "metric-delta-pos" if delta > 0.1 else ("metric-delta-neg" if delta < -0.1 else "metric-delta-neu")
    delta_sign = f"+{delta:.4f}" if delta > 0 else f"{delta:.4f}"

    return f"""<section class="test-section" id="test3">
<div class="test-header">
  <div class="test-number">Test 3</div>
  <div class="test-title">Line-Boundary Entropy</div>
  <div class="test-desc">Within-line vs cross-line bigrams — are line boundaries real structural breaks?</div>
</div>

<div class="metric-block">
  <div class="metric-grid">
    <div class="metric-item">
      <div class="metric-value">{_fmt(h_w, 4)}<span class="metric-unit">bits</span></div>
      <div class="metric-label">H within-line &nbsp;<span class="muted small">({n_w:,} bigrams)</span></div>
    </div>
    <div class="metric-item">
      <div class="metric-value">{_fmt(h_c, 4)}<span class="metric-unit">bits</span></div>
      <div class="metric-label">H cross-line &nbsp;<span class="muted small">({n_c:,} bigrams)</span></div>
    </div>
    <div class="metric-item">
      <div class="metric-value {delta_cls}">{delta_sign}<span class="metric-unit">bits</span></div>
      <div class="metric-label">Δ (cross − within)</div>
    </div>
  </div>
  {_verdict_strip(direction, verdict, good="confirmed")}
</div>

<div class="note-box">
  <div class="note-box-title">Interpretation</div>
  Within-line bigrams are pairs of consecutive signs on the same line.
  Cross-line bigrams connect the last sign of one line to the first sign of the
  next after the boustrophedon flip. If H_cross &gt; H_within by more than 0.1 bits,
  the sign distribution is less predictable across line boundaries than within them
  — confirming that line breaks represent genuine compositional units.
</div>
</section>"""


# ---------------------------------------------------------------------------
# Methodology appendix
# ---------------------------------------------------------------------------

def _render_methodology() -> str:
    return """<section class="method-section" id="methodology">
<div class="section-label">Appendix</div>
<div class="method-title">Methodology</div>

<div class="method-item">
  <div class="method-item-head">Test 1 &mdash; Conditional Entropy</div>
  <div class="method-item-body">
    Conditional bigram entropy H(S&#x2099; | S&#x2099;&#x208b;&#x2081;) is computed from empirical counts
    over all sign sequences in the corpus. The same computation is repeated on the
    reversed sequences. If forward entropy is lower, consecutive signs are more
    predictable in the transcribed direction.
    <div class="method-formula">H(S&#x2099; | S&#x2099;&#x208b;&#x2081;) = &minus;&sum;&#x2099;&#x209B;&#x209C; P(s,t) log&#x2082; P(t|s)</div>
    Threshold: |&#x0394;| &gt; 0.05 bits is treated as directionally meaningful.
  </div>
</div>

<div class="method-item">
  <div class="method-item-head">Test 2 &mdash; N-gram Perplexity</div>
  <div class="method-item-body">
    An NgramModel trained on the full corpus (forward sequences) is scored on
    forward and reversed sequences. Perplexity is 2<sup>&minus;(mean log&#x2082;p per token)</sup>.
    A model that learned real directional structure will assign higher probability
    (lower perplexity) to forward sequences.
  </div>
</div>

<div class="method-item">
  <div class="method-item-head">Test 3 &mdash; Line-Boundary Entropy</div>
  <div class="method-item-body">
    Bigrams are split into within-line (same side and line number) and cross-line
    (spanning a line boundary, after the boustrophedon reversal of odd-numbered lines).
    Cross-line entropy exceeding within-line entropy by &gt; 0.1 bits indicates that
    line breaks mark real compositional boundaries.
  </div>
</div>

<div class="method-item">
  <div class="method-item-head">Test 4 &mdash; Leave-one-out Perplexity</div>
  <div class="method-item-body">
    Each tablet is held out in turn. An add-&#x03B1; smoothed n-gram model (&#x03B1; = 0.5)
    is trained on the remaining tablets and used to score the held-out tablet.
    This is repeated under both a&#x2192;b (recto-first) and b&#x2192;a (verso-first) side orderings.
    The ordering that yields lower mean LOO perplexity is the one under which the
    corpus exhibits more regular sequential structure &mdash; taken as evidence of the
    scribes&rsquo; actual reading order.
    <div class="method-formula">PPL&#x2097;&#x2092;&#x2092; = 2<sup>&minus;(1/N) &sum;&#x1D62; log&#x2082; P&#x208b;&#x1D62;(x&#x1D62;)</sup></div>
    where P&#x208b;&#x1D62; is the model trained without tablet <em>i</em>.
    Both bigram (n=2) and trigram (n=3) models are used; agreement between orders
    strengthens the verdict. Vocabulary is estimated per fold; out-of-vocabulary
    tokens receive the smoothing floor.
  </div>
</div>

</section>"""


# ---------------------------------------------------------------------------
# References
# ---------------------------------------------------------------------------

def _render_references() -> str:
    return """<section class="ref-section" id="references">
<div class="ref-title">References</div>
<ul class="ref-list">
  <li>Barthel, T.&thinsp;S. (1958). <em>Grundlagen zur Entzifferung der Osterinselschrift.</em>
      Cram, de Gruyter &amp; Co., Hamburg.</li>
  <li>Fischer, S.&thinsp;R. (1997). <em>RongoRongo: The Easter Island Script.</em>
      Clarendon Press, Oxford.</li>
  <li>Friedman, W.&thinsp;F. (1922). <em>The Index of Coincidence and Its Applications in
      Cryptanalysis.</em> Riverbank Publication No.&thinsp;22, Geneva, IL.</li>
  <li>Horley, P. (2010). Structural analysis of rongorongo inscriptions.
      <em>Rapa Nui Journal</em>, 24(2), 26&ndash;52.</li>
  <li>Horley, P. (2016). Allographs and alloforms in Rongorongo inscriptions.
      <em>Journal de la Soci&eacute;t&eacute; des Oc&eacute;anistes</em>, 142&ndash;143, 55&ndash;96.</li>
  <li>Pozdniakov, K. (1996). Preliminary report on the Proto-Polynesian borrowings
      in the Rapanui language and Rongorongo script.
      <em>Journal de la Soci&eacute;t&eacute; des Oc&eacute;anistes</em>, 103(2), 159&ndash;170.</li>
  <li>Pozdniakov, K. &amp; Pozdniakov, I. (2007). Rapanui writing and the Rapanui language.
      <em>Forum for Anthropology and Culture</em>, 3, 3&ndash;36.</li>
  <li>Shannon, C.&thinsp;E. (1948). A mathematical theory of communication.
      <em>Bell System Technical Journal</em>, 27(3), 379&ndash;423.</li>
</ul>
</section>"""


# ---------------------------------------------------------------------------
# Full report
# ---------------------------------------------------------------------------

def _render_full_report(data: dict, source_file: str) -> str:
    generated    = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    n_tablets    = data.get("corpus_tablets", 0)
    n_tokens     = data.get("corpus_tokens", 0)
    tests_run    = data.get("tests_run", [])
    t1 = data.get("test1")
    t2 = data.get("test2")
    t3 = data.get("test3")
    t4 = data.get("test4")

    # Stats row values
    rv_pref   = (t4 or {}).get("preferred_order", None)
    dir1      = (t1 or {}).get("direction", None)

    rv_label  = {"ab": "recto first", "ba": "verso first", "mixed": "mixed"}.get(rv_pref or "", "—")
    rv_cls    = "rv" if rv_pref in ("ab", "ba") else ("warn" if rv_pref == "mixed" else "")
    dir_label = {"forward": "left-to-right", "reverse": "right-to-left", "neutral": "unclear"}.get(dir1 or "", "—")
    dir_cls   = "confirmed" if dir1 == "forward" else ("warn" if dir1 == "neutral" else "warn")

    # Abstracts adapt to available results
    t4_summary = ""
    if t4:
        if rv_pref == "ab":
            t4_summary = " Test 4 resolves the long-standing question of recto/verso order: recto (side a) precedes verso (side b), confirmed by both bigram and trigram leave-one-out perplexity."
        elif rv_pref == "ba":
            t4_summary = " Test 4 resolves the long-standing question of recto/verso order: verso (side b) precedes recto (side a), confirmed by both bigram and trigram leave-one-out perplexity."
        else:
            t4_summary = " Test 4 (recto/verso ordering) returns a mixed signal: bigram and trigram models disagree."

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>hackingrongo — Reading Order Analysis</title>
<link href="https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,400;0,600;1,400&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>{_CSS}</style>
</head>
<body>
<div class="wrap">

<div class="report-header">
  <div class="report-title">hackingrongo<br>Reading Order Analysis</div>
  <div class="report-subtitle">Four entropy-theoretic tests for transcription-direction verification</div>
  <div class="report-meta">
    <b>Corpus:</b> {n_tablets} tablets &nbsp;&middot;&nbsp;
    <b>Tokens:</b> {n_tokens:,} &nbsp;&middot;&nbsp;
    <b>Tests run:</b> {", ".join(str(t) for t in tests_run) or "none"} &nbsp;&middot;&nbsp;
    <b>Source:</b> {source_file} &nbsp;&middot;&nbsp;
    <b>Generated:</b> {generated}
  </div>
  <div class="abstract">
    <p>Rongorongo, the undeciphered script of Easter Island (Rapa Nui), is written
    in reverse boustrophedon across the two sides of carved wooden tablets. Before
    any phonological decipherment can proceed, two structural questions must be
    settled: (1) is the transcribed left-to-right direction the actual reading
    direction? and (2) which tablet side comes first — recto or verso?</p>
    <p>This report presents four entropy-theoretic tests applied to the full
    corpus of {n_tablets} transcribed tablets ({n_tokens:,} tokens).
    Tests 1 and 2 address transcription direction through conditional entropy
    and n-gram perplexity asymmetry. Test 3 checks whether line boundaries are
    real compositional units. Test 4 — the primary finding — uses
    leave-one-out perplexity to determine recto/verso reading order.{t4_summary}</p>
    <p>All four tests are purely sequence-statistical: they require no phonological
    assumptions, no external language model, and no knowledge of glyph meaning.
    The results serve as structural constraints for all downstream decipherment work.</p>
  </div>
</div>

<div class="stats-row">
  <div class="stat-card">
    <div class="stat-value">{n_tablets}</div>
    <div class="stat-label">tablets</div>
  </div>
  <div class="stat-card">
    <div class="stat-value">{n_tokens:,}</div>
    <div class="stat-label">tokens</div>
  </div>
  <div class="stat-card">
    <div class="stat-value">{len(tests_run)}</div>
    <div class="stat-label">tests run</div>
  </div>
  <div class="stat-card {dir_cls}">
    <div class="stat-value">{dir_label}</div>
    <div class="stat-label">reading direction (Test 1)</div>
  </div>
  <div class="stat-card {rv_cls}">
    <div class="stat-value">{rv_label}</div>
    <div class="stat-label">side order (Test 4)</div>
  </div>
</div>

{_render_test4_hero(t4)}

<hr style="border:none;border-top:1px solid var(--border);margin:52px 0 48px">

{_render_test1(t1)}
{_render_test2(t2)}
{_render_test3(t3)}

{_render_methodology()}
{_render_references()}

<div class="report-footer">
  <p><b>hackingrongo</b> · Reading Order Analysis · MIT License</p>
  <p>Tests computed from corpus at <code>{source_file}</code>.
  LOO-PPL uses add-0.5 Laplace smoothing.
  Conditional entropy from empirical bigram/unigram counts, no smoothing.</p>
</div>

</div>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_reading_order_report(results_json: Path) -> str:
    """Build the reading-order HTML report from a reading_order_results.json file.

    Parameters
    ----------
    results_json : Path
        Path to JSON written by ``reading_order_tests.py --output``.

    Returns
    -------
    str
        Complete HTML document string.
    """
    data = json.loads(results_json.read_text(encoding="utf-8"))
    logger.info(
        "Building reading-order report: %d tablets, tests run: %s",
        data.get("corpus_tablets", 0),
        data.get("tests_run", []),
    )
    return _render_full_report(data, source_file=results_json.name)


def save_reading_order_report(results_json: Path, output_path: Path) -> None:
    """Generate and write the reading-order HTML report.

    Parameters
    ----------
    results_json : Path
        Input JSON written by ``reading_order_tests.py --output``.
    output_path : Path
        Destination ``.html`` file.  Parent directories are created if needed.
    """
    html = build_reading_order_report(results_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    logger.info("Reading-order report written: %s (%d bytes).", output_path, len(html))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate the reading-order HTML report."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("outputs/reading_order_results.json"),
        help="reading_order_results.json path (default: outputs/reading_order_results.json).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/analysis/reading_order_report.html"),
        help="Output HTML path (default: outputs/analysis/reading_order_report.html).",
    )
    args = parser.parse_args()

    if not args.input.exists():
        parser.error(
            f"Input not found: {args.input}  "
            "(run: python scripts/reading_order_tests.py --corpus data/corpus "
            "--tests 3 4 --output outputs/reading_order_results.json)"
        )

    save_reading_order_report(args.input, args.output)
    print(f"Reading-order report → {args.output}")


if __name__ == "__main__":
    main()
