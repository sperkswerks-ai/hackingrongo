"""
hackingrongo.results.decipherment_report
=========================================

Generates a scholar-facing HTML report of Zone C decipherment hypotheses.

Each hypothesis card shows:
  - Overall LM score, MCMC log-posterior, beam score, and run provenance
  - Full phoneme assignment table with per-sign confidence and evidence count
  - Per-stratum parallel-passage alignment scores (consistency, LM mean ± std,
    languages that score above the random baseline)
  - A score-band bar situating the hypothesis within the full ranking

Inputs
------
  outputs/decipherment/ranking.json   — HypothesisRanking (required)

Output
------
  outputs/decipherment/decipherment_report.html

CLI
---
    python -m hackingrongo.results.decipherment_report \\
        --ranking outputs/decipherment/ranking.json \\
        --output  outputs/decipherment/decipherment_report.html \\
        [--top-n 20]

Public API
----------
``build_decipherment_report``   → HTML string
``save_decipherment_report``    → writes HTML file
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from datetime import datetime, timezone
from pathlib import Path

from hackingrongo.results.schema import (
    DecryptionHypothesis,
    HypothesisRanking,
    PhonemeAssignment,
    StratumScore,
    load_ranking,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Shared metadata
# ---------------------------------------------------------------------------

_STRATUM_LABELS: dict[str, str] = {
    "pre_contact":  "Pre-contact",
    "post_contact": "Post-contact",
    "excluded":     "Excluded",
    "unknown":      "Unknown",
}

_TYPE_LABELS: dict[str, str] = {
    "syllabic":       "Syllabic",
    "logographic":    "Logographic",
    "semasiographic": "Semasiographic",
}

# ---------------------------------------------------------------------------
# Confidence helpers
# ---------------------------------------------------------------------------


def _conf_colour(confidence: float) -> str:
    if confidence >= 0.70:
        return "#4caf7d"
    if confidence >= 0.40:
        return "#d4a817"
    if confidence > 0.0:
        return "#e07b54"
    return "#888888"  # beam-only (confidence == 0)


def _conf_bar(confidence: float, width: int = 16) -> str:
    filled = round(confidence * width)
    colour = _conf_colour(confidence)
    return (
        f'<span class="cbar" style="color:{colour}">'
        + "█" * filled + '<span style="opacity:0.2">' + "█" * (width - filled) + "</span>"
        + "</span>"
    )


def _pct_bar(fraction: float, colour: str = "#7b9ee0", width: int = 12) -> str:
    filled = round(max(0.0, min(1.0, fraction)) * width)
    return (
        f'<span class="cbar" style="color:{colour}">'
        + "█" * filled + '<span style="opacity:0.2">' + "█" * (width - filled) + "</span>"
        + "</span>"
    )


def _lang_chips(languages: list[str]) -> str:
    if not languages:
        return '<span class="muted">—</span>'
    chips = "".join(
        f'<span class="lang-chip">{lang}</span>'
        for lang in languages
    )
    return f'<span class="lang-row">{chips}</span>'


# ---------------------------------------------------------------------------
# Phoneme assignment table
# ---------------------------------------------------------------------------


def _render_phoneme_table(assignments: list[PhonemeAssignment]) -> str:
    if not assignments:
        return '<p class="muted small">No phoneme assignments recorded.</p>'

    # Sort by descending confidence, then sign code for stable tie-breaking.
    sorted_a = sorted(assignments, key=lambda a: (-a.confidence, a.sign_code))

    rows = []
    for a in sorted_a:
        beam_only = a.confidence == 0.0
        conf_cls = "beam-only" if beam_only else ""
        beam_tag = (
            '<span class="beam-tag" title="Beam-refined; differs from all MCMC samples">beam</span>'
            if beam_only else ""
        )
        colour = _conf_colour(a.confidence)
        rows.append(
            f'<tr class="{conf_cls}">'
            f'<td class="mono sign-code">{a.sign_code}</td>'
            f'<td class="mono phoneme-val">{a.phoneme}</td>'
            f'<td class="conf-cell" style="color:{colour}">'
            f'  {a.confidence:.3f} {_conf_bar(a.confidence, 10)}{beam_tag}'
            f'</td>'
            f'<td class="evid-cell">{a.evidence_count:,}</td>'
            f'</tr>'
        )

    rows_html = "\n".join(rows)
    return f"""
<div class="table-scroll">
  <table class="assign-table">
    <thead>
      <tr>
        <th>Sign</th><th>Phoneme</th>
        <th>Confidence</th><th>Evidence</th>
      </tr>
    </thead>
    <tbody>
      {rows_html}
    </tbody>
  </table>
</div>"""


# ---------------------------------------------------------------------------
# Stratum / parallel-passage alignment panel
# ---------------------------------------------------------------------------


def _render_stratum_scores(stratum_scores: list[StratumScore]) -> str:
    if not stratum_scores:
        return '<p class="muted small">No stratum scores recorded.</p>'

    rows = []
    for s in stratum_scores:
        label = _STRATUM_LABELS.get(s.stratum, s.stratum)
        cons_bar = _pct_bar(s.consistency_score, colour="#7b9ee0")
        lm_str = f"{s.lm_score_mean:.2f} ± {s.lm_score_std:.2f}"
        rows.append(
            f'<tr>'
            f'<td class="stratum-name">{label}</td>'
            f'<td class="cons-cell">{s.consistency_score:.3f} {cons_bar}</td>'
            f'<td class="lm-cell mono">{lm_str}</td>'
            f'<td class="pass-cell">{s.n_passages}</td>'
            f'<td class="lang-cell">{_lang_chips(s.languages_above_baseline)}</td>'
            f'</tr>'
        )

    rows_html = "\n".join(rows)
    return f"""
<table class="stratum-table">
  <thead>
    <tr>
      <th>Stratum</th>
      <th>Consistency</th>
      <th>LM score (bits)</th>
      <th>Passages</th>
      <th>Languages ≥ baseline</th>
    </tr>
  </thead>
  <tbody>
    {rows_html}
  </tbody>
</table>"""


# ---------------------------------------------------------------------------
# Score-band bar
# ---------------------------------------------------------------------------


def _score_band(
    rank: int,
    n_total: int,
    overall_lm: float,
    best: float,
    worst: float,
    null_baseline: float | None = None,
) -> str:
    """Horizontal bar showing where this hypothesis sits in the full ranking.

    Parameters
    ----------
    null_baseline : float, optional
        OOV-floor ensemble score used as a null hypothesis reference line.
        When provided, a thin vertical marker is drawn at the corresponding
        position on the score track labelled "OOV floor".
    """
    span = best - worst if best != worst else 1.0
    pct = max(0.0, min(1.0, (overall_lm - worst) / span)) * 100
    quantile_pct = max(0.0, min(100.0, (n_total - rank) / max(n_total - 1, 1) * 100))

    baseline_marker = ""
    baseline_label = ""
    if null_baseline is not None:
        bl_pct = max(0.0, min(100.0, (null_baseline - worst) / span * 100))
        baseline_marker = (
            f'<div class="score-baseline" style="left:{bl_pct:.1f}%" '
            f'title="OOV floor: {null_baseline:.2f} bits"></div>'
        )
        baseline_label = (
            f'<span class="muted small" '
            f'style="position:absolute;left:{bl_pct:.1f}%;'
            f'transform:translateX(-50%);top:5px;white-space:nowrap;font-size:9px;">'
            f'OOV&nbsp;floor</span>'
        )

    return f"""
<div class="score-band">
  <div class="score-band-label">
    Rank {rank}/{n_total} &mdash; LM score {overall_lm:.4f} bits
    &nbsp;<span class="muted small">(percentile {quantile_pct:.0f}%)</span>
  </div>
  <div class="score-track" style="position:relative;">
    <div class="score-fill" style="width:{pct:.1f}%"></div>
    <div class="score-marker" style="left:{pct:.1f}%"></div>
    {baseline_marker}
  </div>
  <div class="score-track-labels" style="position:relative;height:18px;">
    <span class="muted small">worst&nbsp;{worst:.2f}</span>
    {baseline_label}
    <span class="muted small">best&nbsp;{best:.2f}</span>
  </div>
</div>"""


# ---------------------------------------------------------------------------
# Hypothesis type badge
# ---------------------------------------------------------------------------


def _type_badge(hypothesis_type: str) -> str:
    colours: dict[str, tuple[str, str]] = {
        "syllabic":       ("#2563eb", "#dbeafe"),
        "logographic":    ("#7c3aed", "#ede9fe"),
        "semasiographic": ("#059669", "#d1fae5"),
    }
    text_col, bg_col = colours.get(hypothesis_type, ("#666", "#eee"))
    label = _TYPE_LABELS.get(hypothesis_type, hypothesis_type)
    return (
        f'<span class="type-badge" '
        f'style="color:{text_col};background:{bg_col};border-color:{text_col}44">'
        f'{label}</span>'
    )


# ---------------------------------------------------------------------------
# Per-hypothesis card
# ---------------------------------------------------------------------------


def _render_card(
    rank: int,
    hyp: DecryptionHypothesis,
    n_total: int,
    best_lm: float,
    worst_lm: float,
    null_baseline: float | None = None,
) -> str:
    score_section = _score_band(rank, n_total, hyp.overall_lm_score, best_lm, worst_lm, null_baseline)
    assign_section = _render_phoneme_table(hyp.assignments)
    stratum_section = _render_stratum_scores(hyp.stratum_scores)

    # Provenance metadata block
    cfg_abbrev = (hyp.config_hash[:12] + "…") if hyp.config_hash else "—"
    run_abbrev = (hyp.run_id[:16] + "…") if len(hyp.run_id) > 16 else hyp.run_id or "—"
    beam_flag = hyp.beam_score != 0.0

    prov_html = f"""
<table class="prov-table">
  <tr><td class="prov-key">Run ID</td>
      <td class="prov-val mono" title="{hyp.run_id}">{run_abbrev}</td></tr>
  <tr><td class="prov-key">Config hash</td>
      <td class="prov-val mono" title="{hyp.config_hash}">{cfg_abbrev}</td></tr>
  <tr><td class="prov-key">MCMC log-post.</td>
      <td class="prov-val mono">{hyp.mcmc_log_posterior:.4f}</td></tr>
  <tr><td class="prov-key">Beam score</td>
      <td class="prov-val mono">{"—" if not beam_flag else f"{hyp.beam_score:.4f}"}</td></tr>
  <tr><td class="prov-key">Generated</td>
      <td class="prov-val mono">{hyp.created_at[:19].replace("T", " ")}</td></tr>
  <tr><td class="prov-key">Assignments</td>
      <td class="prov-val mono">{len(hyp.assignments)}</td></tr>
</table>"""

    # Beam-only count note
    n_beam_only = sum(1 for a in hyp.assignments if a.confidence == 0.0)
    beam_note = (
        f'<p class="beam-note">↑ {n_beam_only} assignment{"s" if n_beam_only != 1 else ""} '
        f'marked <span class="beam-tag">beam</span> were refined by beam search and '
        f'have no MCMC posterior support — confidence is legitimately 0.</p>'
        if n_beam_only > 0 else ""
    )

    return f"""
<div class="card" id="{hyp.hypothesis_id}">

  <div class="card-header">
    <div class="rank-badge">#{rank}</div>
    <div class="card-title">
      <span class="hyp-id mono">{hyp.hypothesis_id}</span>
      {_type_badge(hyp.hypothesis_type)}
    </div>
    <div class="overall-score">
      <span class="score-val">{hyp.overall_lm_score:.4f}</span>
      <span class="score-unit">bits</span>
    </div>
  </div>

  {score_section}

  <div class="card-body">

    <div class="col-assign">
      <div class="section-label">Phoneme assignments</div>
      {assign_section}
      {beam_note}
    </div>

    <div class="col-right">
      <div class="section-label">Parallel-passage alignment</div>
      {stratum_section}

      <div class="section-label" style="margin-top:20px">Provenance</div>
      {prov_html}
    </div>

  </div>
</div>"""


# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------

_CSS = """\
:root {
  --bg: #ffffff; --surface: #f8f8fa; --surface2: #f0f0f5;
  --border: #d0d0dd; --text: #1a1a1a; --muted: #666666;
  --accent: #c4a96d; --accent2: #7b9ee0;
}
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
  background: var(--bg); color: var(--text);
  font-family: 'Cormorant Garamond', 'Palatino Linotype', Georgia, serif;
  font-size: 16px; line-height: 1.65;
}
.wrap { max-width: 1140px; margin: 0 auto; padding: 52px 28px; }
.mono { font-family: 'JetBrains Mono', 'Fira Mono', monospace; }
.muted { color: var(--muted); }
.small { font-size: 11px; }

/* ── Report header ── */
.report-header { border-bottom: 1px solid var(--border);
                 padding-bottom: 38px; margin-bottom: 44px; }
.report-title { font-size: 34px; font-weight: 600; color: #000; letter-spacing: -0.3px; }
.report-subtitle { font-size: 17px; color: var(--accent); font-style: italic; margin-top: 6px; }
.report-meta { margin-top: 20px; font-family: 'JetBrains Mono', monospace;
               font-size: 11px; color: var(--muted); line-height: 2.2; }
.report-meta b { color: #333; }
.abstract { margin-top: 20px; font-size: 14px; color: #333; max-width: 800px; line-height: 1.85; }

/* ── TOC ── */
.toc { margin: 0 0 44px; }
.toc-title { font-family: 'JetBrains Mono', monospace; font-size: 9px;
             text-transform: uppercase; letter-spacing: 0.1em;
             color: var(--muted); margin-bottom: 10px; }
.toc-grid { display: flex; flex-wrap: wrap; gap: 6px; }
.toc-chip { font-family: 'JetBrains Mono', monospace; font-size: 10px;
            background: var(--surface); border: 1px solid var(--border);
            border-radius: 3px; padding: 3px 9px; color: var(--accent);
            text-decoration: none; }
.toc-chip:hover { background: var(--surface2); }

/* ── Hypothesis card ── */
.card { background: var(--surface); border: 1px solid var(--border);
        border-radius: 8px; margin-bottom: 36px; overflow: hidden; }
.card-header { padding: 16px 22px 10px; display: flex; align-items: center;
               gap: 14px; flex-wrap: wrap; }
.rank-badge { font-family: 'JetBrains Mono', monospace; font-size: 11px;
              color: var(--muted); min-width: 32px; }
.card-title { display: flex; align-items: center; gap: 10px; flex: 1; }
.hyp-id { font-size: 15px; color: var(--accent); font-weight: 500; }
.type-badge { font-family: 'JetBrains Mono', monospace; font-size: 9.5px;
              border-radius: 3px; padding: 2px 8px; border: 1px solid transparent;
              white-space: nowrap; }
.overall-score { display: flex; align-items: baseline; gap: 5px; }
.score-val { font-family: 'JetBrains Mono', monospace; font-size: 22px;
             color: #000; font-weight: 500; }
.score-unit { font-family: 'JetBrains Mono', monospace; font-size: 10px; color: var(--muted); }

/* ── Score band ── */
.score-band { padding: 10px 22px 12px; border-bottom: 1px solid var(--border);
              background: var(--surface2); }
.score-band-label { font-family: 'JetBrains Mono', monospace; font-size: 10.5px;
                    color: #444; margin-bottom: 7px; }
.score-track { position: relative; height: 6px; background: var(--border);
               border-radius: 3px; margin-bottom: 4px; }
.score-fill { height: 100%; background: linear-gradient(90deg, #7b9ee0, #4caf7d);
              border-radius: 3px; }
.score-marker { position: absolute; top: -3px; width: 2px; height: 12px;
                background: #000; border-radius: 1px; transform: translateX(-50%); }
.score-track-labels { display: flex; justify-content: space-between; }
.score-baseline { position: absolute; top: -3px; width: 1px; height: 12px;
                  background: #e07b54; border-radius: 1px;
                  transform: translateX(-50%); opacity: 0.85; }

/* ── Card body ── */
.card-body { display: grid; grid-template-columns: 1fr 380px; gap: 0; }
.col-assign { padding: 20px 22px; border-right: 1px solid var(--border); min-width: 0; }
.col-right { padding: 20px 22px; }
.section-label { font-family: 'JetBrains Mono', monospace; font-size: 9px;
                 color: var(--muted); letter-spacing: 0.1em; text-transform: uppercase;
                 margin-bottom: 10px; }

/* ── Phoneme assignment table ── */
.table-scroll { max-height: 440px; overflow-y: auto;
                border: 1px solid var(--border); border-radius: 4px; }
.assign-table { width: 100%; border-collapse: collapse; font-size: 12px; }
.assign-table thead { position: sticky; top: 0; background: var(--surface2);
                       z-index: 1; }
.assign-table th { padding: 6px 10px; text-align: left; font-weight: 600;
                   font-family: 'JetBrains Mono', monospace; font-size: 9.5px;
                   color: var(--muted); border-bottom: 1px solid var(--border); }
.assign-table td { padding: 4px 10px; border-bottom: 1px solid var(--border)40; }
.assign-table tr:last-child td { border-bottom: none; }
.assign-table tr.beam-only { opacity: 0.55; }
.assign-table tr:hover { background: var(--surface2); }
.sign-code { color: var(--accent); }
.phoneme-val { color: #2563eb; font-size: 13px; }
.conf-cell { white-space: nowrap; }
.evid-cell { color: var(--muted); text-align: right; }
.cbar { font-size: 8px; letter-spacing: -1.5px; vertical-align: middle; }
.beam-tag { font-family: 'JetBrains Mono', monospace; font-size: 8px;
            background: #88888822; border: 1px solid #88888844; color: #888;
            border-radius: 2px; padding: 1px 4px; margin-left: 4px; vertical-align: middle; }
.beam-note { font-size: 11px; color: var(--muted); margin-top: 8px; line-height: 1.6; }

/* ── Stratum / alignment table ── */
.stratum-table { width: 100%; border-collapse: collapse; font-size: 12px; }
.stratum-table th { padding: 5px 8px; text-align: left; font-weight: 600;
                    font-family: 'JetBrains Mono', monospace; font-size: 9px;
                    color: var(--muted); border-bottom: 1px solid var(--border); }
.stratum-table td { padding: 5px 8px; border-bottom: 1px solid var(--border)40; }
.stratum-table tr:last-child td { border-bottom: none; }
.stratum-name { font-weight: 500; }
.cons-cell { white-space: nowrap; }
.lm-cell { font-size: 11px; }
.pass-cell { text-align: center; color: var(--muted); }
.lang-row { display: inline-flex; flex-wrap: wrap; gap: 3px; }
.lang-chip { font-family: 'JetBrains Mono', monospace; font-size: 8.5px;
             background: var(--accent2)22; border: 1px solid var(--accent2)55;
             color: var(--accent2); border-radius: 2px; padding: 1px 5px; }

/* ── Provenance table ── */
.prov-table { width: 100%; border-collapse: collapse; font-size: 11.5px; }
.prov-key { color: var(--muted); width: 100px; padding: 3px 8px 3px 0;
            vertical-align: top; }
.prov-val { color: #333; word-break: break-all; padding: 3px 0; }

/* ── Footer ── */
.report-footer { border-top: 1px solid var(--border); margin-top: 56px;
                 padding-top: 26px; font-size: 12px; color: var(--muted); line-height: 2.0; }
.report-footer a { color: var(--accent); text-decoration: none; }
.report-footer code { background: var(--surface2); border: 1px solid var(--border);
                      border-radius: 2px; padding: 1px 5px;
                      font-family: 'JetBrains Mono', monospace; }

/* ── Quantum Analysis section ── */
.quantum-section { margin-top: 56px; }
.quantum-heading {
  font-family: 'JetBrains Mono', monospace; font-size: 11px;
  text-transform: uppercase; letter-spacing: 0.12em; color: var(--muted);
  margin-bottom: 18px; padding-bottom: 8px; border-bottom: 1px solid var(--border);
}
.quantum-heading span { color: #7c3aed; margin-right: 6px; }
.q-card {
  background: var(--surface); border: 1px solid var(--border);
  border-radius: 8px; padding: 24px 26px; margin-bottom: 24px;
}
.q-card-title {
  font-family: 'JetBrains Mono', monospace; font-size: 12px; font-weight: 600;
  color: #333; margin-bottom: 14px;
}
.q-meta { font-size: 12px; color: var(--muted); margin-bottom: 14px; }
.q-placeholder {
  background: var(--surface); border: 1px dashed var(--border);
  border-radius: 6px; padding: 22px 26px; color: var(--muted);
  font-size: 13px; line-height: 1.9;
}
.q-placeholder code { background: var(--surface2); border: 1px solid var(--border);
  border-radius: 2px; padding: 1px 5px; font-family: 'JetBrains Mono', monospace;
  font-size: 11px; }
.q-interp {
  margin-top: 16px; font-size: 13px; color: #333; line-height: 1.85;
  max-width: 820px; background: var(--surface2); border-left: 3px solid #7c3aed44;
  padding: 10px 14px; border-radius: 0 4px 4px 0;
}

/* ── Quantum hardness table ── */
.hardness-table { width: 100%; border-collapse: collapse; font-size: 12.5px; margin-bottom: 4px; }
.hardness-table th {
  padding: 6px 12px; text-align: left; font-weight: 600;
  font-family: 'JetBrains Mono', monospace; font-size: 9.5px; color: var(--muted);
  border-bottom: 1px solid var(--border); white-space: nowrap;
}
.hardness-table td { padding: 7px 12px; border-bottom: 1px solid var(--border)40; white-space: nowrap; }
.hardness-table tr:last-child td { border-bottom: none; }
.hardness-table tr:hover { background: var(--surface2); }
.h-tau { font-family: 'JetBrains Mono', monospace; color: var(--accent); font-weight: 500; }
.h-pgood { font-family: 'JetBrains Mono', monospace; font-size: 12px; }
.h-num { font-family: 'JetBrains Mono', monospace; font-size: 12px; text-align: right; }
.speedup-strong { color: #4caf7d; font-weight: 600; }
.speedup-mid    { color: #d4a817; font-weight: 500; }
.speedup-low    { color: var(--muted); }

/* ── Comparison table ── */
.compare-table { width: 100%; border-collapse: collapse; font-size: 12.5px; }
.compare-table th {
  padding: 6px 12px; text-align: left; font-weight: 600;
  font-family: 'JetBrains Mono', monospace; font-size: 9.5px; color: var(--muted);
  border-bottom: 1px solid var(--border);
}
.compare-table td { padding: 7px 12px; border-bottom: 1px solid var(--border)40; }
.compare-table tr:last-child td { border-bottom: none; }
.compare-table tr:hover { background: var(--surface2); }
.compare-table tr.qubo-row { background: #7c3aed0a; }
.compare-table tr.qubo-row:hover { background: #7c3aed15; }
.qubo-badge {
  font-family: 'JetBrains Mono', monospace; font-size: 9.5px;
  color: #7c3aed; background: #7c3aed18; border: 1px solid #7c3aed44;
  border-radius: 3px; padding: 2px 7px; white-space: nowrap;
}
.mcmc-rank { font-family: 'JetBrains Mono', monospace; font-size: 10px; color: var(--muted); }
.delta-pos { color: #4caf7d; font-family: 'JetBrains Mono', monospace; font-size: 11px; }
.delta-neg { color: #e07b54; font-family: 'JetBrains Mono', monospace; font-size: 11px; }

@media (max-width: 860px) {
  .card-body { grid-template-columns: 1fr; }
  .col-assign { border-right: none; border-bottom: 1px solid var(--border); }
}
"""

# ---------------------------------------------------------------------------
# Quantum Analysis section
# ---------------------------------------------------------------------------


def _fmt_sci(val: float) -> str:
    """Format a small probability as HTML scientific notation, e.g. 1.2×10<sup>-3</sup>."""
    if not math.isfinite(val) or val <= 0:
        return "0"
    exp = int(math.floor(math.log10(val)))
    mantissa = val / (10 ** exp)
    return f"{mantissa:.1f}&times;10<sup>{exp}</sup>"


def _render_hardness_table(pgood_data: dict) -> str:
    n_samples = pgood_data.get("n_samples", 0)
    dist      = pgood_data.get("score_distribution", {})
    thresholds = pgood_data.get("thresholds", [])
    interp     = pgood_data.get("interpretation", "")

    rows = []
    for t in thresholds:
        tau   = t.get("tau", 0)
        pg    = t.get("p_good", 0.0)
        gc    = t.get("grover_oracle_calls", -1)
        cc    = t.get("classical_random_calls", -1)
        sp    = t.get("quantum_speedup_ratio")
        mg    = t.get("mcmc_vs_grover_ratio")

        gc_str = f"{gc:,}"   if isinstance(gc, int) and gc > 0 else "N/A"
        cc_str = f"{cc:,}"   if isinstance(cc, int) and cc > 0 else "N/A"
        mg_str = f"{mg:.1f}&times;" if mg is not None else "—"

        if sp is None:
            sp_html = '<span class="speedup-low">—</span>'
        elif sp > 100:
            sp_html = f'<span class="speedup-strong">{sp:.1f}&times;</span>'
        elif sp > 10:
            sp_html = f'<span class="speedup-mid">{sp:.1f}&times;</span>'
        else:
            sp_html = f'<span class="speedup-low">{sp:.1f}&times;</span>'

        rows.append(
            f"<tr>"
            f'<td class="h-tau">&tau;={tau:.2f}</td>'
            f'<td class="h-pgood">{_fmt_sci(pg)}</td>'
            f'<td class="h-num">{gc_str}</td>'
            f'<td class="h-num">{cc_str}</td>'
            f'<td class="h-num">{sp_html}</td>'
            f'<td class="h-num">{mg_str}</td>'
            f"</tr>"
        )

    rows_html = "\n".join(rows)
    dist_mean = dist.get("mean", 0)
    dist_std  = dist.get("std", 0)

    return f"""
<div class="q-card">
  <div class="q-card-title">Quantum Hardness Analysis</div>
  <p class="q-meta">
    {n_samples:,} random sign&rarr;phoneme assignments sampled &middot;
    Score distribution: mean&nbsp;{dist_mean:.2f}, std&nbsp;{dist_std:.2f}
    (mean per-token log&sub;2;p)
  </p>
  <table class="hardness-table">
    <thead>
      <tr>
        <th>Threshold</th>
        <th>p<sub>good</sub></th>
        <th style="text-align:right">Grover calls</th>
        <th style="text-align:right">Classical</th>
        <th style="text-align:right">Speedup</th>
        <th style="text-align:right">MCMC / Grover</th>
      </tr>
    </thead>
    <tbody>
      {rows_html}
    </tbody>
  </table>
  <p class="q-interp">{interp}</p>
</div>"""


def _render_comparison_table(ranking: "HypothesisRanking", qubo_data: dict | None) -> str:
    hypotheses = ranking.hypotheses
    if not hypotheses and qubo_data is None:
        return ""

    rows = []

    for rank, hyp in enumerate(hypotheses[:10], start=1):
        score_str = f"{hyp.overall_lm_score:.4f}"
        hyp_type  = _TYPE_LABELS.get(hyp.hypothesis_type, hyp.hypothesis_type or "—")
        rows.append(
            f"<tr>"
            f'<td><span class="mcmc-rank">#{rank}</span></td>'
            f'<td><a class="toc-chip" href="#{hyp.hypothesis_id}" '
            f'style="font-size:11px">{hyp.hypothesis_id}</a></td>'
            f'<td class="h-num">{score_str}</td>'
            f'<td><span class="lang-chip" style="font-size:9px">{hyp_type}</span></td>'
            f'<td></td>'
            f"</tr>"
        )

    if qubo_data is not None:
        qubo_lm   = qubo_data.get("best_lm_score")
        solver    = qubo_data.get("solver", "?")
        n_reads   = qubo_data.get("n_reads", 0)
        energy    = qubo_data.get("best_energy")
        delta     = qubo_data.get("improvement_over_mcmc")

        lm_str = f"{qubo_lm:.4f}" if qubo_lm is not None else "—"
        energy_str = f"{energy:.1f}" if energy is not None else "—"

        if delta is not None:
            delta_cls  = "delta-pos" if delta >= 0 else "delta-neg"
            delta_html = f'<span class="{delta_cls}">{delta:+.4f}</span>'
        else:
            delta_html = '<span class="speedup-low">—</span>'

        detail = (
            f'<span class="muted" style="font-size:10px">'
            f'{solver} &middot; {n_reads:,} reads &middot; energy {energy_str}'
            f'</span>'
        )
        rows.append(
            f'<tr class="qubo-row">'
            f'<td><span class="qubo-badge">&#9883; QUBO</span></td>'
            f'<td><span class="mono" style="font-size:11px;color:#7c3aed">qubo_result</span></td>'
            f'<td class="h-num">{lm_str}</td>'
            f'<td>{detail}</td>'
            f'<td>{delta_html}</td>'
            f"</tr>"
        )

    rows_html = "\n".join(rows)
    return f"""
<div class="q-card">
  <div class="q-card-title">Score Comparison: MCMC vs QUBO</div>
  <table class="compare-table">
    <thead>
      <tr>
        <th>Source</th>
        <th>Hypothesis</th>
        <th style="text-align:right">LM Score (bits)</th>
        <th>Details</th>
        <th style="text-align:right">&Delta; vs MCMC #1</th>
      </tr>
    </thead>
    <tbody>
      {rows_html}
    </tbody>
  </table>
</div>"""


def _render_quantum_section(
    ranking: "HypothesisRanking",
    pgood_data: dict | None,
    qubo_data: dict | None,
) -> str:
    if pgood_data is None and qubo_data is None:
        return """
<div class="quantum-section" id="quantum-analysis">
  <div class="quantum-heading"><span>&#9883;</span>Quantum Analysis</div>
  <div class="q-placeholder">
    Quantum hardness analysis not yet run.<br>
    See <code>scripts/measure_pgood.py</code> (Layer 5Q) and
    <code>scripts/run_qubo_decipherment.py</code> (Layer 4Q).
  </div>
</div>"""

    parts: list[str] = []
    if qubo_data is not None or ranking.hypotheses:
        parts.append(_render_comparison_table(ranking, qubo_data))
    if pgood_data is not None:
        parts.append(_render_hardness_table(pgood_data))

    inner = "\n".join(parts)
    return f"""
<div class="quantum-section" id="quantum-analysis">
  <div class="quantum-heading"><span>&#9883;</span>Quantum Analysis</div>
  {inner}
</div>"""


# ---------------------------------------------------------------------------
# Full HTML document
# ---------------------------------------------------------------------------


def _render_html(
    ranking: HypothesisRanking,
    top_n: int,
    null_baseline: float | None = None,
    pgood_data: dict | None = None,
    qubo_data: dict | None = None,
) -> str:
    hypotheses = ranking.top_n(top_n)
    n_total = len(ranking.hypotheses)
    n_shown = len(hypotheses)

    if not hypotheses:
        return "<p>No hypotheses found in ranking.</p>"

    scores = [h.overall_lm_score for h in ranking.hypotheses]
    best_lm = max(scores)
    worst_lm = min(scores)

    generated = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    run_id_summary = hypotheses[0].run_id if hypotheses else "—"
    cfg_hash_summary = (hypotheses[0].config_hash[:12] + "…") if hypotheses[0].config_hash else "—"

    # Table of contents
    toc_chips = "".join(
        f'<a class="toc-chip" href="#{h.hypothesis_id}">{h.hypothesis_id}</a>'
        for h in hypotheses
    )
    toc_chips += '<a class="toc-chip" href="#quantum-analysis" style="color:#7c3aed">&#9883; Quantum</a>'

    # Hypothesis cards
    cards_html = "\n".join(
        _render_card(rank, hyp, n_total, best_lm, worst_lm, null_baseline)
        for rank, hyp in enumerate(hypotheses, start=1)
    )

    quantum_section = _render_quantum_section(ranking, pgood_data, qubo_data)

    n_assignments_top = len(hypotheses[0].assignments) if hypotheses else 0

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>hackingrongo — Zone C Decipherment Hypotheses</title>
<link href="https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,400;0,600;1,400&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>{_CSS}</style>
</head>
<body>
<div class="wrap">

<div class="report-header">
  <div class="report-title">hackingrongo<br>Zone C Decipherment Hypotheses</div>
  <div class="report-subtitle">Ranked phoneme-assignment hypotheses from MCMC + beam search — for scholar review</div>
  <div class="report-meta">
    <b>Showing:</b> {n_shown} of {n_total} hypotheses &nbsp;&middot;&nbsp;
    <b>Ranking metric:</b> {ranking.ranking_metric} &nbsp;&middot;&nbsp;
    <b>LM score range:</b> {worst_lm:.4f} &ndash; {best_lm:.4f} bits &nbsp;&middot;&nbsp;
    <b>Run ID:</b> {run_id_summary} &nbsp;&middot;&nbsp;
    <b>Config:</b> <span title="{hypotheses[0].config_hash if hypotheses else ''}">{cfg_hash_summary}</span> &nbsp;&middot;&nbsp;
    <b>Generated:</b> {generated}
  </div>
  <div class="abstract">
    <p>Each card represents one phoneme-assignment hypothesis: a complete mapping of
    rongorongo sign codes to proposed phoneme or syllable values. Hypotheses are ranked
    by overall language-model log-probability across all tablet strata. Higher (less
    negative) scores indicate that the proposed phoneme sequence reads more like a known
    Polynesian language.</p>
    <p>The <b>parallel-passage alignment</b> panel shows how consistently the hypothesis
    decodes parallel passages in each temporal stratum, and which reference languages
    score above the random baseline. The <b>phoneme assignment</b> table lists every
    sign in the active inventory, sorted by posterior confidence. Assignments marked
    <span class="beam-tag">beam</span> were refined by beam search after MCMC and have
    no direct MCMC posterior support; their confidence is legitimately&nbsp;0.
    <b>We invite rongorongo scholars to review these hypotheses and advise on
    linguistic plausibility.</b></p>
  </div>
</div>

<div class="toc">
  <div class="toc-title">Jump to hypothesis</div>
  <div class="toc-grid">{toc_chips}</div>
</div>

{cards_html}

{quantum_section}

<div class="report-footer">
  <p><b>hackingrongo</b> &middot; Zone C MCMC + beam-search decipherment pipeline &middot; MIT License</p>
  <p>Hypotheses generated by <code>scripts/run_decipherment.py</code> using
  <code>hackingrongo.zone_c.mcmc.MCMCSampler</code> (Metropolis-Hastings, {n_total} chains ×
  configurable iterations) and <code>hackingrongo.zone_c.beam_search.BeamSearchDecoder</code>.
  Language models: <code>hackingrongo.zone_c.lm_scoring.LMScorer</code>
  (Polynesian n-gram LMs from ABVD + Hawaiian corpus).</p>
  <p>Sign inventory: Horley (2010) coding system. Barthel (1958) base codes.
  Corpus: {n_assignments_top}-sign active inventory across 26 tablets ({15273} total glyphs).</p>
  <p>This is a computational hypothesis report, not a decipherment claim.
  All hypotheses require expert linguistic and epigraphic review.</p>
  <p><b>SperksWerks LLC</b> &middot;
  <a href="https://sperkswerks.ai" target="_blank">sperkswerks.ai</a> &middot;
  <a href="mailto:studio@sperkswerks.ai">studio@sperkswerks.ai</a></p>
</div>

</div>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_decipherment_report(
    ranking_path: Path,
    top_n: int = 20,
    null_baseline: float | None = None,
    pgood_path: Path | None = None,
    qubo_path: Path | None = None,
) -> str:
    """Build the decipherment hypothesis report HTML.

    Parameters
    ----------
    ranking_path : Path
        Path to a ``ranking.json`` file written by ``run_decipherment.py``.
    top_n : int
        Maximum number of hypotheses to include, best-first.
    null_baseline : float, optional
        OOV-floor ensemble score for the corpus — shown as a reference line
        on each hypothesis score-band bar.  Pass
        ``cfg.zone_c.lm_scoring.oov_log_prob_per_order[max_order]`` (e.g. −20.0)
        from the run config.  When omitted the reference line is not drawn.
    pgood_path : Path, optional
        Path to ``pgood_analysis.json`` from ``measure_pgood.py``.  When
        provided, renders the Quantum Hardness Analysis table.
    qubo_path : Path, optional
        Path to ``qubo_result.json`` from ``run_qubo_decipherment.py``.  When
        provided, adds a QUBO row to the score comparison table.

    Returns
    -------
    str
        Complete HTML document as a string.
    """
    ranking = load_ranking(ranking_path)
    logger.info(
        "Building decipherment report: %d total hypotheses, showing top %d.",
        len(ranking.hypotheses), top_n,
    )

    pgood_data: dict | None = None
    if pgood_path is not None and pgood_path.exists():
        try:
            pgood_data = json.loads(pgood_path.read_text(encoding="utf-8"))
            logger.info("Loaded pgood data from %s.", pgood_path)
        except Exception as exc:
            logger.warning("Could not load %s: %s", pgood_path, exc)

    qubo_data: dict | None = None
    if qubo_path is not None and qubo_path.exists():
        try:
            qubo_data = json.loads(qubo_path.read_text(encoding="utf-8"))
            logger.info("Loaded QUBO data from %s.", qubo_path)
        except Exception as exc:
            logger.warning("Could not load %s: %s", qubo_path, exc)

    return _render_html(
        ranking, top_n,
        null_baseline=null_baseline,
        pgood_data=pgood_data,
        qubo_data=qubo_data,
    )


def save_decipherment_report(
    ranking_path: Path,
    output_path: Path,
    top_n: int = 20,
    null_baseline: float | None = None,
    pgood_path: Path | None = None,
    qubo_path: Path | None = None,
) -> None:
    """Generate and write the decipherment report to an HTML file.

    Parameters
    ----------
    ranking_path : Path
        Path to ``ranking.json``.
    output_path : Path
        Destination ``.html`` file.  Parent directories are created if needed.
    top_n : int
        Maximum number of hypotheses to include.
    null_baseline : float, optional
        OOV-floor score — see :func:`build_decipherment_report`.
    pgood_path : Path, optional
        Path to ``pgood_analysis.json`` — see :func:`build_decipherment_report`.
    qubo_path : Path, optional
        Path to ``qubo_result.json`` — see :func:`build_decipherment_report`.
    """
    html = build_decipherment_report(
        ranking_path, top_n=top_n, null_baseline=null_baseline,
        pgood_path=pgood_path, qubo_path=qubo_path,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    logger.info("Decipherment report written: %s (%d bytes).", output_path, len(html))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate scholar-facing HTML report of Zone C decipherment hypotheses."
    )
    p.add_argument(
        "--ranking", type=Path,
        default=Path("outputs/decipherment/ranking.json"),
        help="ranking.json from run_decipherment.py (default: outputs/decipherment/ranking.json).",
    )
    p.add_argument(
        "--output", type=Path, default=None,
        help="Output HTML path (default: <ranking dir>/decipherment_report.html).",
    )
    p.add_argument(
        "--top-n", type=int, default=20,
        help="Number of top hypotheses to include (default: 20).",
    )
    p.add_argument(
        "--pgood", type=Path, default=None, metavar="JSON",
        help="pgood_analysis.json from measure_pgood.py (optional).",
    )
    p.add_argument(
        "--qubo", type=Path, default=None, metavar="JSON",
        help="qubo_result.json from run_qubo_decipherment.py (optional).",
    )
    return p.parse_args()


def main() -> None:
    import logging as _logging
    _logging.basicConfig(level=_logging.INFO, format="%(levelname)s  %(message)s")
    args = _parse_args()

    # Auto-discover sibling outputs in the same directory as ranking.json
    ranking_dir = args.ranking.parent
    pgood_path = args.pgood or (ranking_dir.parent / "zone_b" / "pgood_analysis.json")
    qubo_path  = args.qubo  or (ranking_dir / "qubo_result.json")

    output = args.output or (ranking_dir / "decipherment_report.html")
    save_decipherment_report(
        ranking_path=args.ranking,
        output_path=output,
        top_n=args.top_n,
        pgood_path=pgood_path if pgood_path.exists() else None,
        qubo_path=qubo_path   if qubo_path.exists()  else None,
    )
    print(f"Report written to: {output}")


if __name__ == "__main__":
    main()
