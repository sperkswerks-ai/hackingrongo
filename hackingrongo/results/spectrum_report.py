"""
hackingrongo.results.spectrum_report
======================================

HTML report showing the per-tablet logographic/syllabic spectrum scores
alongside the six underlying features, known content-type annotations,
and temporal stratum.

Sections
--------
1. What the spectrum measures — explanation of the six features
2. Tablet spectrum visualization — horizontal bars per tablet
3. Feature breakdown table — all six scores per tablet
4. Scholarly annotation — known content types vs computed scores
5. Pre-contact vs post-contact spectrum comparison
6. Sign type classification summary (when mixed decoder has run)

Inputs
------
  outputs/analysis/spectrum_scores.json  — from spectrum_analyzer.py
  data/metadata/tablets.json             — tablet names + notes

Output
------
  outputs/analysis/spectrum_report.html

CLI
---
    python -m hackingrongo.results.spectrum_report \\
        --scores  outputs/analysis/spectrum_scores.json \\
        --output  outputs/analysis/spectrum_report.html
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

# Known content-type annotations from scholarship
_TABLET_ANNOTATIONS: dict[str, dict[str, str]] = {
    "C": {
        "hypothesis": "Lunar calendar",
        "scholar": "Barthel (1958); Pozdniakov (1996)",
        "expected_spectrum": "logographic",
        "note": "Contains repetitive numerical sequences for lunar months. "
                "High formulaic structure should show high bigram MI.",
    },
    "I": {
        "hypothesis": "Procreation chant / genealogy",
        "scholar": "Fischer (1997)",
        "expected_spectrum": "mixed → syllabic",
        "note": "Fischer proposed this is a phonological chant. "
                "Largest sign count (2431); long repeated parallel sequences.",
    },
    "X": {
        "hypothesis": "Bird-man (Tangata Manu) ritual text",
        "scholar": "Barthel (1958)",
        "expected_spectrum": "logographic",
        "note": "Bird-man motif tablet; likely iconographic/ceremonial content.",
    },
    "H": {
        "hypothesis": "Major narrative / chant",
        "scholar": "Multiple",
        "expected_spectrum": "syllabic",
        "note": "Parallel passages with P and Q; high sign reuse across tablets.",
    },
    "P": {
        "hypothesis": "Parallel to H and Q",
        "scholar": "Multiple",
        "expected_spectrum": "syllabic",
        "note": "Shares large passages with H and Q.",
    },
    "Q": {
        "hypothesis": "Parallel to H and P",
        "scholar": "Multiple",
        "expected_spectrum": "syllabic",
        "note": "Shares passages with H and P.",
    },
    "D": {
        "hypothesis": "Pre-contact inscription",
        "scholar": "Temporal dating",
        "expected_spectrum": "uncertain",
        "note": "Only confirmed pre-contact tablet. Small corpus (234 signs).",
    },
    "G": {
        "hypothesis": "Parallel to K",
        "scholar": "Multiple",
        "expected_spectrum": "syllabic",
        "note": "Parallel passages with Small London (K).",
    },
}

_TABLET_NAMES: dict[str, str] = {
    "A": "Tahua", "B": "Aruku-Kurenga", "C": "Mamari",
    "D": "Échancrée", "E": "Keiti", "F": "Stephen-Chauvet Fragment",
    "G": "Small Santiago", "H": "Great Santiago", "I": "Santiago Staff",
    "J": "Reimiro 1", "K": "Small London", "L": "Reimiro 2",
    "M": "Great Vienna", "N": "Small Vienna", "O": "Boomerang",
    "P": "Great St. Petersburg", "Q": "Small St. Petersburg",
    "R": "Atua-Mata-Riri", "S": "Great Washington",
    "T": "Honolulu 1", "U": "Honolulu 2", "V": "Honolulu 3",
    "W": "Honolulu 4", "X": "Tangata Manu", "Y": "Snuff Box",
}


# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------

_CSS = """\
:root {
  --bg: #ffffff; --surface: #f8f8fa; --surface2: #f0f0f5;
  --border: #d0d0dd; --text: #1a1a1a; --muted: #666666;
  --accent: #c4a96d; --accent2: #7b9ee0;
  --syllabic: #2563eb; --logographic: #9333ea; --mixed: #d97706;
  --pre: #16a34a; --post: #7c3aed; --unknown: #9ca3af;
}
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
  background: var(--bg); color: var(--text);
  font-family: 'Cormorant Garamond', 'Palatino Linotype', Georgia, serif;
  font-size: 16px; line-height: 1.65;
}
.wrap { max-width: 1100px; margin: 0 auto; padding: 52px 28px; }

.report-header { border-bottom: 2px solid var(--border);
                 padding-bottom: 38px; margin-bottom: 44px; }
.report-title  { font-size: 34px; font-weight: 600; color: #000; }
.report-subtitle { font-size: 17px; color: var(--accent); font-style: italic; margin-top: 6px; }
.report-meta { margin-top: 16px; font-family: 'JetBrains Mono', monospace;
               font-size: 11px; color: var(--muted); line-height: 2.2; }
.report-meta b { color: #333; }

.sec-head { font-size: 22px; font-weight: 600; color: #000;
            margin: 44px 0 6px; border-top: 1px solid var(--border); padding-top: 26px; }
.sec-sub  { font-size: 13.5px; color: var(--muted); font-style: italic; margin-bottom: 20px; }

.intro p { font-size: 14.5px; color: #333; line-height: 1.9; margin-bottom: 12px;
           max-width: 820px; }
.intro b { color: #000; }

/* ── Spectrum axis legend ── */
.spectrum-axis { display: flex; align-items: center; gap: 0; margin: 18px 0 26px;
                 font-family: 'JetBrains Mono', monospace; font-size: 11px; }
.axis-label-left  { color: var(--syllabic); min-width: 90px; }
.axis-label-right { color: var(--logographic); min-width: 90px; text-align: right; }
.axis-track { flex: 1; height: 6px; border-radius: 3px;
              background: linear-gradient(to right, var(--syllabic), var(--mixed), var(--logographic)); }

/* ── Per-tablet spectrum bar ── */
.tablet-grid { display: flex; flex-direction: column; gap: 6px; margin: 16px 0 32px; }
.tablet-row { display: grid; grid-template-columns: 32px 180px 1fr 56px 80px 70px;
              align-items: center; gap: 10px; }
.tablet-id  { font-family: 'JetBrains Mono', monospace; font-size: 12px;
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
.reliability-warn { font-family: 'JetBrains Mono', monospace; font-size: 9px;
                    color: var(--muted); }

/* ── Feature table ── */
.feat-table { width: 100%; border-collapse: collapse; font-size: 12.5px;
              margin: 14px 0 32px; }
.feat-table th { text-align: left; padding: 7px 10px;
                 font-family: 'JetBrains Mono', monospace; font-size: 9px;
                 letter-spacing: 0.08em; text-transform: uppercase;
                 color: var(--muted); border-bottom: 1px solid var(--border);
                 background: var(--surface); white-space: nowrap; }
.feat-table td { padding: 7px 10px; border-bottom: 1px solid var(--border);
                 color: #333; vertical-align: top; }
.feat-table tr:last-child td { border-bottom: none; }
.feat-table .mono { font-family: 'JetBrains Mono', monospace; font-size: 11px; }
.feat-table .hi { color: var(--logographic); font-weight: 600; }
.feat-table .lo { color: var(--syllabic); font-weight: 600; }
.feat-table .mid { color: #555; }

/* ── Annotation cards ── */
.annot-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 14px;
              margin: 16px 0 32px; }
.annot-card { background: var(--surface); border: 1px solid var(--border);
              border-radius: 7px; padding: 14px 16px; }
.annot-card-header { display: flex; align-items: center; gap: 10px;
                     margin-bottom: 8px; }
.annot-tid  { font-family: 'JetBrains Mono', monospace; font-size: 13px;
              color: var(--accent); font-weight: 500; }
.annot-name { font-size: 14px; font-weight: 600; color: #111; }
.annot-score { font-family: 'JetBrains Mono', monospace; font-size: 18px;
               font-weight: 500; margin-left: auto; }
.annot-hypothesis { font-size: 12.5px; color: #444; line-height: 1.7; }
.annot-hypothesis b { color: #000; }
.annot-verdict { font-family: 'JetBrains Mono', monospace; font-size: 9.5px;
                 border-radius: 3px; padding: 2px 7px; margin-top: 8px;
                 display: inline-block; }
.verdict-match    { background: #dcfce7; color: #15803d; border: 1px solid #86efac; }
.verdict-surprise { background: #fef9c3; color: #92400e; border: 1px solid #fde047; }
.verdict-na       { background: var(--surface2); color: var(--muted);
                    border: 1px solid var(--border); }

/* ── Feature explanation boxes ── */
.feat-explain { background: var(--surface); border-left: 3px solid var(--accent);
                border-radius: 0 6px 6px 0; padding: 14px 18px; margin: 10px 0;
                max-width: 820px; }
.feat-explain-label { font-family: 'JetBrains Mono', monospace; font-size: 9.5px;
                      letter-spacing: 0.1em; text-transform: uppercase;
                      color: var(--muted); margin-bottom: 5px; }
.feat-explain p { font-size: 13.5px; color: #333; line-height: 1.8; }
.feat-explain b { color: #000; }
.feat-explain code { font-family: 'JetBrains Mono', monospace; font-size: 11px;
                     background: var(--surface2); border: 1px solid var(--border);
                     border-radius: 2px; padding: 1px 5px; }

/* ── Stratum comparison ── */
.stratum-compare { display: grid; grid-template-columns: 1fr 1fr; gap: 18px;
                   margin: 16px 0 32px; }
.stratum-col { background: var(--surface); border: 1px solid var(--border);
               border-radius: 7px; padding: 16px 18px; }
.stratum-col-title { font-family: 'JetBrains Mono', monospace; font-size: 10px;
                     letter-spacing: 0.1em; text-transform: uppercase;
                     color: var(--muted); margin-bottom: 10px; }
.stratum-col-score { font-family: 'JetBrains Mono', monospace; font-size: 28px;
                     font-weight: 500; color: var(--accent); }
.stratum-col-sub { font-size: 12px; color: var(--muted); margin-top: 4px; }

.report-footer { border-top: 1px solid var(--border); margin-top: 52px;
                 padding-top: 26px; font-size: 12px; color: var(--muted);
                 line-height: 2.0; }
.report-footer a { color: var(--accent); text-decoration: none; }

@media (max-width: 800px) {
  .annot-grid { grid-template-columns: 1fr; }
  .stratum-compare { grid-template-columns: 1fr; }
  .tablet-row { grid-template-columns: 28px 130px 1fr 50px; }
}
"""


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------

def _spectrum_colour(score: float) -> str:
    """CSS colour interpolated from syllabic (blue) through mixed (amber) to logographic (purple)."""
    if score < 0.5:
        # blue → amber
        t = score * 2
        r = int(37 + t * (217 - 37))
        g = int(99 + t * (119 - 99))
        b = int(235 + t * (6 - 235))
    else:
        # amber → purple
        t = (score - 0.5) * 2
        r = int(217 + t * (147 - 217))
        g = int(119 + t * (51 - 119))
        b = int(6 + t * (234 - 6))
    return f"rgb({r},{g},{b})"


def _verdict(tablet_id: str, score: float) -> tuple[str, str]:
    """(verdict label, CSS class) comparing computed score to expected."""
    annot = _TABLET_ANNOTATIONS.get(tablet_id, {})
    expected = annot.get("expected_spectrum", "")
    if not expected or expected == "uncertain":
        return "no prior hypothesis", "verdict-na"
    if expected == "logographic" and score >= 0.45:
        return "matches hypothesis", "verdict-match"
    if expected == "syllabic" and score <= 0.45:
        return "matches hypothesis", "verdict-match"
    if expected == "mixed → syllabic" and score <= 0.55:
        return "consistent", "verdict-match"
    return "surprising — investigate", "verdict-surprise"


def _stratum_badge(cluster: str) -> str:
    cls = {
        "pre_contact": "stratum-pre",
        "post_contact": "stratum-post",
        "excluded": "stratum-excluded",
    }.get(cluster, "stratum-unknown")
    label = {
        "pre_contact": "pre-contact",
        "post_contact": "post-contact",
        "excluded": "excluded",
    }.get(cluster, "unknown")
    return f'<span class="stratum-badge {cls}">{label}</span>'


def _feat_cls(val: float, lo_good: bool = False) -> str:
    """CSS class for a feature value — hi/lo/mid relative to [0,1]."""
    if lo_good:
        return "lo" if val < 0.3 else ("hi" if val > 0.7 else "mid")
    return "hi" if val > 0.7 else ("lo" if val < 0.3 else "mid")


# ---------------------------------------------------------------------------
# Section renderers
# ---------------------------------------------------------------------------

def _section_explanation() -> str:
    features = [
        ("mean_ic",
         "Index of Coincidence",
         "IC = Σ fᵢ(fᵢ−1) / [N(N−1)]. Measures how concentrated the sign-frequency "
         "distribution is on this tablet. High IC → a few signs dominate (logographic); "
         "moderate IC → distributed usage (syllabic)."),
        ("mean_bigram_mi",
         "Mean Bigram Mutual Information",
         "I(Sₙ; Sₙ₋₁) per sign, averaged across all signs on the tablet. "
         "High MI → signs strongly predict their successors (formulaic, logographic). "
         "Low MI → signs follow freely-varying phonotactics (syllabic)."),
        ("consistency_variance",
         "Cross-tablet Consistency Variance",
         "Variance of each sign's cross-tablet consistency score. "
         "High variance → some signs are universal while others are tablet-specific "
         "(different content domains = logographic). Low variance → same signs everywhere (syllabic)."),
        ("entropy_decay_rate",
         "Entropy Decay Rate H₁→H₂",
         "(H₁ − H₂) / H₁: fraction of unigram entropy removed by bigram context. "
         "High rate → phonotactics tightly constrain the next sign (syllabic). "
         "Low rate → sequences are semantically but not phonotactically driven (logographic). "
         "<b>Note: this feature is inverted in the spectrum score</b> — slow decay → more logographic."),
        ("compound_density",
         "Compound Glyph Density",
         "Fraction of glyph positions occupied by explicitly compound-marked glyphs. "
         "Compound glyphs encode compositional semantics (morpheme₁ + morpheme₂ = concept). "
         "High density → logographic; absent → consistent with syllabic."),
        ("hapax_rate",
         "Hapax Legomena Rate",
         "Fraction of sign types appearing exactly once on this tablet. "
         "High hapax → unique domain-specific content (logographic); "
         "Low hapax → the same small syllable set recycled throughout (syllabic)."),
    ]

    blocks = "".join(
        f'<div class="feat-explain">'
        f'<div class="feat-explain-label">{name}</div>'
        f'<p><b>{label}.</b> {desc}</p>'
        f'</div>'
        for name, label, desc in features
    )

    return f"""
<div class="sec-head">1 · The Six Spectrum Features</div>
<div class="sec-sub">What each feature measures and which direction is logographic</div>
<div class="intro">
  <p>The spectrum score is an equal-weight projection of six entropy-theoretic
  features onto [0 = syllabic, 1 = logographic]. Each feature is normalised
  by corpus-wide min/max before projection, so a score of 0.5 means a tablet
  sits at the corpus median — not that it is equally logographic and syllabic
  in absolute terms.</p>
</div>
{blocks}
"""


def _section_spectrum_bars(tablets: dict[str, Any]) -> str:
    # Sort by spectrum score descending (most logographic first)
    sorted_tabs = sorted(
        tablets.items(),
        key=lambda x: x[1].get("spectrum_score", 0),
        reverse=True,
    )

    rows_html = ""
    for tid, feat in sorted_tabs:
        score = feat.get("spectrum_score", 0.0)
        name  = feat.get("tablet_name", tid)
        cluster = feat.get("cluster", "unknown")
        reliable = feat.get("reliable", True)
        n_tokens = feat.get("n_tokens", 0)
        colour = _spectrum_colour(score)
        bar_pct = round(score * 100)

        badge = _stratum_badge(cluster)
        rel_note = (
            f'<span class="reliability-warn">n={n_tokens}</span>'
            if not reliable else ""
        )

        # Annotation dot
        annot = _TABLET_ANNOTATIONS.get(tid, {})
        hyp = annot.get("hypothesis", "")
        hyp_note = f'<span style="font-size:10px;color:var(--muted)"> — {hyp}</span>' if hyp else ""

        rows_html += f"""
<div class="tablet-row">
  <div class="tablet-id">{tid}</div>
  <div class="tablet-name">{name}{hyp_note}</div>
  <div class="bar-track">
    <div class="bar-fill" style="width:{bar_pct}%;background:{colour}"></div>
  </div>
  <div class="score-val">{score:.3f}</div>
  {badge}
  {rel_note}
</div>"""

    return f"""
<div class="sec-head">2 · Tablet Spectrum Visualization</div>
<div class="sec-sub">Each bar shows position on the syllabic ↔ logographic axis</div>

<div class="spectrum-axis">
  <div class="axis-label-left">0.0 syllabic</div>
  <div class="axis-track"></div>
  <div class="axis-label-right">logographic 1.0</div>
</div>

<div class="tablet-grid">
{rows_html}
</div>
"""


def _section_feature_table(tablets: dict[str, Any]) -> str:
    sorted_tabs = sorted(
        tablets.items(),
        key=lambda x: x[1].get("spectrum_score", 0),
        reverse=True,
    )

    rows = ""
    for tid, feat in sorted_tabs:
        if not feat.get("reliable", True):
            continue
        name   = feat.get("tablet_name", tid)
        score  = feat.get("spectrum_score", 0.0)
        ic     = feat.get("mean_ic", 0.0)
        mi     = feat.get("mean_bigram_mi", 0.0)
        cv     = feat.get("consistency_variance", 0.0)
        ed     = feat.get("entropy_decay_rate", 0.0)
        cd     = feat.get("compound_density", 0.0)
        hr     = feat.get("hapax_rate", 0.0)

        def _cell(v: float, lo_good: bool = False) -> str:
            cls = _feat_cls(v, lo_good)
            return f'<td class="mono {cls}">{v:.4f}</td>'

        rows += (
            f'<tr>'
            f'<td class="mono">{tid}</td>'
            f'<td style="font-size:12px">{name}</td>'
            f'<td class="mono" style="color:var(--accent);font-weight:600">{score:.3f}</td>'
            + _cell(ic)
            + _cell(mi)
            + _cell(cv)
            + _cell(ed, lo_good=True)  # low decay rate = logographic = "hi" colour
            + _cell(cd)
            + _cell(hr)
            + f'</tr>'
        )

    return f"""
<div class="sec-head">3 · Feature Breakdown (reliable tablets only)</div>
<div class="sec-sub">
  <span style="color:var(--logographic);font-weight:600">Purple = high (logographic direction)</span>
  &nbsp;·&nbsp;
  <span style="color:var(--syllabic);font-weight:600">Blue = low (syllabic direction)</span>
  &nbsp;·&nbsp; entropy decay rate is inverted (low = logographic)
</div>
<table class="feat-table">
<thead><tr>
  <th>ID</th><th>Name</th><th>Score</th>
  <th>IC</th><th>Bigram MI</th><th>Consist. Var</th>
  <th>Entropy Decay</th><th>Compound Dens.</th><th>Hapax Rate</th>
</tr></thead>
<tbody>{rows}</tbody>
</table>
"""


def _section_annotations(tablets: dict[str, Any]) -> str:
    cards = ""
    for tid, annot in _TABLET_ANNOTATIONS.items():
        feat = tablets.get(tid, {})
        score = feat.get("spectrum_score", None)
        name  = feat.get("tablet_name", _TABLET_NAMES.get(tid, tid))
        hyp   = annot.get("hypothesis", "")
        scholar = annot.get("scholar", "")
        note  = annot.get("note", "")
        expected = annot.get("expected_spectrum", "")

        score_html = (
            f'<span class="annot-score" style="color:{_spectrum_colour(score)}">'
            f'{score:.3f}</span>'
            if score is not None else
            '<span class="annot-score" style="color:var(--muted)">—</span>'
        )

        v_label, v_cls = _verdict(tid, score or 0.5)

        cards += f"""
<div class="annot-card">
  <div class="annot-card-header">
    <span class="annot-tid">{tid}</span>
    <span class="annot-name">{name}</span>
    {score_html}
  </div>
  <div class="annot-hypothesis">
    <b>Hypothesis:</b> {hyp}<br>
    <b>Scholar:</b> {scholar}<br>
    <b>Expected spectrum:</b> {expected}<br>
    {note}
  </div>
  <span class="annot-verdict {v_cls}">{v_label}</span>
</div>"""

    return f"""
<div class="sec-head">4 · Scholarly Annotations vs Computed Scores</div>
<div class="sec-sub">
  Comparing computed spectrum positions to prior content-type hypotheses
</div>
<div class="annot-grid">
{cards}
</div>
"""


def _section_stratum_comparison(tablets: dict[str, Any]) -> str:
    pre_scores  = [f["spectrum_score"] for f in tablets.values()
                   if f.get("cluster") == "pre_contact" and f.get("reliable")]
    post_scores = [f["spectrum_score"] for f in tablets.values()
                   if f.get("cluster") == "post_contact" and f.get("reliable")]

    def _mean(vals: list[float]) -> str:
        return f"{sum(vals)/len(vals):.3f}" if vals else "—"

    def _range(vals: list[float]) -> str:
        return f"[{min(vals):.3f}, {max(vals):.3f}]" if vals else "—"

    pre_tids  = [tid for tid, f in tablets.items()
                 if f.get("cluster") == "pre_contact" and f.get("reliable")]
    post_tids = [tid for tid, f in tablets.items()
                 if f.get("cluster") == "post_contact" and f.get("reliable")]

    pre_list  = " · ".join(sorted(pre_tids))
    post_list = " · ".join(sorted(post_tids))

    return f"""
<div class="sec-head">5 · Pre-contact vs Post-contact Spectrum</div>
<div class="sec-sub">
  Does script type correlate with temporal stratum?
</div>

<div class="stratum-compare">
  <div class="stratum-col">
    <div class="stratum-col-title">Pre-contact tablets ({len(pre_scores)} reliable)</div>
    <div class="stratum-col-score">{_mean(pre_scores)}</div>
    <div class="stratum-col-sub">mean spectrum score</div>
    <div class="stratum-col-sub" style="margin-top:6px">Range: {_range(pre_scores)}</div>
    <div class="stratum-col-sub" style="margin-top:8px;font-size:11px">
      Tablets: {pre_list or "—"}</div>
  </div>
  <div class="stratum-col">
    <div class="stratum-col-title">Post-contact tablets ({len(post_scores)} reliable)</div>
    <div class="stratum-col-score">{_mean(post_scores)}</div>
    <div class="stratum-col-sub">mean spectrum score</div>
    <div class="stratum-col-sub" style="margin-top:6px">Range: {_range(post_scores)}</div>
    <div class="stratum-col-sub" style="margin-top:8px;font-size:11px">
      Tablets: {post_list or "—"}</div>
  </div>
</div>

<div style="max-width:820px;font-size:14px;color:#444;line-height:1.85">
  <p><b>Interpretation caveat:</b> With only one confirmed pre-contact tablet (D) in the
  reliable set, the pre-contact mean is a single data point. The stratum comparison becomes
  statistically meaningful only after Zone A embeddings enable temporal re-classification
  of the 19 undated tablets under the three dating scenarios. These spectrum scores are
  computed independently of temporal dating and can be re-stratified once dating improves.</p>
  <p style="margin-top:10px">For the full entropy methodology behind IC, bigram MI, and
  entropy decay, see <a href="entropy_report.html">entropy_report.html</a>.</p>
</div>
"""


# ---------------------------------------------------------------------------
# Full HTML document
# ---------------------------------------------------------------------------


def _render_html(
    tablets: dict[str, Any],
    generated: str,
    n_reliable: int,
) -> str:
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>hackingrongo — Tablet Spectrum Analysis</title>
<link href="https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,400;0,600;1,400&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>{_CSS}</style>
</head>
<body>
<div class="wrap">

<div class="report-header">
  <div class="report-title">hackingrongo<br>Tablet Spectrum Analysis</div>
  <div class="report-subtitle">
    Logographic ↔ Syllabic spectrum for each rongorongo tablet
  </div>
  <div class="report-meta">
    <b>Tablets:</b> {len(tablets)} &nbsp;·&nbsp;
    <b>Reliable (n ≥ 50 tokens):</b> {n_reliable} &nbsp;·&nbsp;
    <b>Score range:</b> 0 = syllabic · 1 = logographic &nbsp;·&nbsp;
    <b>Generated:</b> {generated}
  </div>
  <div class="intro" style="margin-top:18px">
    <p>This report answers: <b>if you knew nothing about what rongorongo says,
    what would the statistical shape of each tablet tell you about how it works?</b>
    Six entropy-theoretic features — Index of Coincidence, bigram mutual information,
    cross-tablet consistency variance, entropy decay rate, compound glyph density, and
    hapax rate — are each sensitive to different aspects of script type. Their
    equal-weight projection places every tablet on a spectrum from purely syllabic
    (phonotactic, low-IC, fast-decaying entropy) to purely logographic (formulaic,
    high-IC, slow-decaying entropy, compound-dense).</p>
    <p>This is a computational hypothesis, not a decipherment claim.
    All scores require expert epigraphic interpretation.</p>
  </div>
</div>

{_section_explanation()}
{_section_spectrum_bars(tablets)}
{_section_feature_table(tablets)}
{_section_annotations(tablets)}
{_section_stratum_comparison(tablets)}

<div class="report-footer">
  <p><b>hackingrongo</b> · Tablet Spectrum Analysis · MIT License ·
  <a href="https://github.com/violasarah2000/hackingrongo">GitHub</a></p>
  <p>Spectrum features: IC (Friedman 1922), bigram MI, entropy decay rate (Shannon 1948),
  compound density (Barthel 1958), hapax rate. Equal-weight projection.</p>
  <p>Related reports:
  <a href="entropy_report.html">entropy_report.html</a> (full entropy methodology) ·
  <a href="compound_report.html">compound_report.html</a> (compound glyph analysis) ·
  <a href="divergence_report.html">divergence_report.html</a> (Zone A cluster divergence)</p>
  <p>This is a computational hypothesis document. All findings require expert review.</p>
  <p><b>SperksWerks LLC</b> ·
  <a href="https://sperkswerks.ai">sperkswerks.ai</a> ·
  <a href="mailto:studio@sperkswerks.ai">studio@sperkswerks.ai</a></p>
</div>

</div>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_spectrum_report(
    scores_path: Path,
    metadata_path: Path | None = None,
) -> str:
    """Build the spectrum HTML report from spectrum_scores.json."""
    if not scores_path.exists():
        raise FileNotFoundError(
            f"Spectrum scores not found: {scores_path}\n"
            "Run python -m hackingrongo.zone_b.spectrum_analyzer first."
        )
    data = json.loads(scores_path.read_text(encoding="utf-8"))
    tablets_raw: dict[str, Any] = data.get("tablets", {})

    # Enrich with cluster from metadata
    cluster_map: dict[str, str] = {}
    if metadata_path and metadata_path.exists():
        try:
            meta = json.loads(metadata_path.read_text(encoding="utf-8"))
            for tid, tdata in meta.items():
                cluster_map[tid] = tdata.get("date_distribution", {}).get("type", "unknown")
        except Exception:
            pass

    tablets: dict[str, Any] = {}
    for tid, feat in tablets_raw.items():
        enriched = dict(feat)
        enriched["cluster"] = cluster_map.get(tid, "unknown")
        tablets[tid] = enriched

    n_reliable = data.get("n_reliable", sum(1 for f in tablets.values() if f.get("reliable")))
    generated  = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    return _render_html(tablets, generated, n_reliable)


def save_spectrum_report(
    scores_path: Path,
    output_path: Path,
    metadata_path: Path | None = None,
) -> None:
    html = build_spectrum_report(scores_path, metadata_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    logger.info("Spectrum report written: %s (%d bytes).", output_path, len(html))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    import logging as _logging
    _logging.basicConfig(level=_logging.INFO, format="%(levelname)s  %(message)s")
    p = argparse.ArgumentParser(
        description="Generate the tablet logographic/syllabic spectrum HTML report."
    )
    p.add_argument("--scores",   type=Path, default=Path("outputs/analysis/spectrum_scores.json"))
    p.add_argument("--metadata", type=Path, default=Path("data/metadata/tablets.json"))
    p.add_argument("--output",   type=Path, default=Path("outputs/analysis/spectrum_report.html"))
    args = p.parse_args()
    save_spectrum_report(args.scores, args.output, args.metadata)
    print(f"Spectrum report → {args.output}")


if __name__ == "__main__":
    main()
