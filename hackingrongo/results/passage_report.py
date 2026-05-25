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
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Shared CSS  (mirrors decipherment_report / divergence_report variables)
# ---------------------------------------------------------------------------

_CSS = """
:root {
  --bg: #ffffff; --surface: #f8f8fa; --surface2: #f0f0f5;
  --border: #d0d0dd; --text: #1a1a1a; --muted: #666666;
  --accent: #c4a96d; --accent2: #7b9ee0;
  --pre: #2563eb; --post: #7c3aed; --undated: #888888;
  --holy: #d4860a; --cross: #c0392b;
}
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
  background: var(--bg); color: var(--text);
  font-family: 'Cormorant Garamond', 'Palatino Linotype', Georgia, serif;
  font-size: 16px; line-height: 1.65;
}
.wrap { max-width: 1100px; margin: 0 auto; padding: 52px 28px; }
.mono { font-family: 'JetBrains Mono', 'Fira Mono', monospace; }
.muted { color: var(--muted); }
.small { font-size: 11px; }

/* ── Report header ── */
.report-header { border-bottom: 1px solid var(--border);
                 padding-bottom: 38px; margin-bottom: 44px; }
.report-title { font-size: 34px; font-weight: 600; color: #000;
                letter-spacing: -0.3px; }
.report-subtitle { font-size: 17px; color: var(--accent); font-style: italic;
                   margin-top: 6px; }
.report-meta { margin-top: 20px; font-family: 'JetBrains Mono', monospace;
               font-size: 11px; color: var(--muted); line-height: 2.2; }
.report-meta b { color: #333; }
.abstract { margin-top: 20px; font-size: 14px; color: #333;
            max-width: 800px; line-height: 1.85; }
.abstract p + p { margin-top: 12px; }

/* ── Summary stats ── */
.stats-row { display: flex; flex-wrap: wrap; gap: 14px; margin-bottom: 40px; }
.stat-card { background: var(--surface); border: 1px solid var(--border);
             border-radius: 6px; padding: 16px 22px; min-width: 110px;
             text-align: center; }
.stat-value { font-family: 'JetBrains Mono', monospace; font-size: 28px;
              font-weight: 500; color: var(--accent); }
.stat-label { font-size: 11px; color: var(--muted); margin-top: 4px;
              font-family: 'JetBrains Mono', monospace; }
.stat-card.holy .stat-value { color: var(--holy); }
.stat-card.cross .stat-value { color: var(--cross); }

/* ── Section label ── */
.section-label { font-family: 'JetBrains Mono', monospace; font-size: 9px;
                 color: var(--muted); letter-spacing: 0.1em;
                 text-transform: uppercase; margin-bottom: 12px; }

/* ── Passage table (summary) ── */
.passage-table-wrap { overflow-x: auto; margin-bottom: 44px; }
.passage-table { width: 100%; border-collapse: collapse; font-size: 13px; }
.passage-table thead th { font-family: 'JetBrains Mono', monospace;
                          font-size: 9px; color: var(--muted); font-weight: 600;
                          text-transform: uppercase; letter-spacing: 0.08em;
                          padding: 6px 10px; border-bottom: 1px solid var(--border);
                          text-align: left; }
.passage-table tbody td { padding: 7px 10px; border-bottom: 1px solid var(--border)60; }
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
.row-anchor > td { background: rgba(37,99,235,0.03); }
.row-anchor > td:first-child { border-left: 3px solid var(--pre); padding-left: 7px; }

/* ── Passage detail card ── */
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
.attest-table td { padding: 5px 8px; border-bottom: 1px solid var(--border)50;
                   vertical-align: top; }
.attest-table tr:last-child td { border-bottom: none; }
.attest-seq { font-family: 'JetBrains Mono', monospace; font-size: 10px;
              color: #333; }
.attest-ed  { font-family: 'JetBrains Mono', monospace; font-size: 11px;
              color: var(--muted); text-align: center; }

/* ── Alignment row ── */
.align-row { display: flex; flex-wrap: wrap; gap: 2px; margin: 4px 0 8px; }
.align-cell { width: 26px; height: 26px; display: flex; align-items: center;
              justify-content: center; border-radius: 3px; font-size: 9px;
              font-family: 'JetBrains Mono', monospace; border: 1px solid transparent; }
.align-match { background: #d1fae5; border-color: #6ee7b7; color: #065f46; }
.align-sub   { background: #fef3c7; border-color: #fcd34d; color: #92400e; }
.align-gap   { background: #fee2e2; border-color: #fca5a5; color: #991b1b; }

/* ── Change cards ── */
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
# Change card
# ---------------------------------------------------------------------------

def _render_change_card(change: dict) -> str:
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
    pre_sign = change.get("pre_contact_sign", "—")
    post_sign = change.get("post_contact_sign", "—")
    n_cons = change.get("n_tablets_consistent", 0)

    return f"""<div class="change-card{card_cls}">
  <div class="change-head">
    {_change_type_tag(ct)}
    <span class="muted small">{pos_label}</span>
    {tag_html}
  </div>
  <div class="change-grid">
    <div>
      <div class="change-field-label">Pre-contact sign</div>
      <div class="change-field-val">{pre_sign}</div>
    </div>
    <div>
      <div class="change-field-label">Post-contact sign</div>
      <div class="change-field-val">{post_sign}</div>
    </div>
    <div>
      <div class="change-field-label">Consistent across</div>
      <div class="change-field-val">{n_cons} post-contact tablet{"s" if n_cons != 1 else ""}</div>
    </div>
  </div>
</div>"""


# ---------------------------------------------------------------------------
# Attestation table
# ---------------------------------------------------------------------------

def _render_attestation_table(attestations: list[dict]) -> str:
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

        seq_html = (
            '<span class="attest-seq">' + " ".join(str(c) for c in seq) + "</span>"
        ) if seq else "—"

        align_html = _alignment_html(align) if align else ""

        rows.append(f"""<tr>
  <td><b>{tablet}</b> <span class="muted small">{tablet_name}</span></td>
  <td>{_stratum_badge(stratum)}</td>
  <td class="muted small">{date_range}</td>
  <td>{seq_html}<br>{align_html}</td>
  <td class="attest-ed">{ed}</td>
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

def _render_passage_page(passage: dict) -> str:
    """Full standalone HTML for a single passage."""
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
    attest_html = _render_attestation_table(attestations)

    holy_html = cross_html = other_html = ""
    holy_changes = [c for c in changes if c.get("is_holy_grail_candidate")]
    cross_changes = [c for c in changes if c.get("crosses_barthel_family") and not c.get("is_holy_grail_candidate")]
    other_changes = [c for c in changes if not c.get("is_holy_grail_candidate") and not c.get("crosses_barthel_family")]

    if holy_changes:
        holy_html = (
            '<div class="section-label" style="margin-top:20px;color:var(--holy)">'
            'Holy Grail candidates — consistent substitutions across ≥ 2 post-contact tablets</div>'
            + "".join(_render_change_card(c) for c in holy_changes)
        )
    if cross_changes:
        cross_html = (
            '<div class="section-label" style="margin-top:16px;color:var(--cross)">'
            'Family-Crossing changes — substitutions spanning Barthel century blocks</div>'
            + "".join(_render_change_card(c) for c in cross_changes)
        )
    if other_changes:
        other_html = (
            '<div class="section-label" style="margin-top:16px">'
            'Other diachronic changes</div>'
            + "".join(_render_change_card(c) for c in other_changes)
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

def _render_summary_page(passages: list[dict], meta: dict[str, Any]) -> str:
    """Render the diachronic cross-passage summary — the primary scholar view."""
    generated = meta.get("generated", datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC"))
    source_file = meta.get("source_file", "—")

    n_passages = len(passages)
    n_with_diachronic = sum(
        1 for p in passages
        if any(a.get("stratum") == "pre_contact" for a in p.get("attestations", []))
        and any(a.get("stratum") == "post_contact" for a in p.get("attestations", []))
    )
    all_changes = [c for p in passages for c in p.get("diachronic_changes", [])]
    n_holy = sum(1 for c in all_changes if c.get("is_holy_grail_candidate"))
    n_cross = sum(1 for c in all_changes if c.get("crosses_barthel_family"))

    # Summary stats
    stats_html = f"""<div class="stats-row">
  <div class="stat-card">
    <div class="stat-value">{n_passages}</div>
    <div class="stat-label">parallel passages</div>
  </div>
  <div class="stat-card">
    <div class="stat-value">{n_with_diachronic}</div>
    <div class="stat-label">with diachronic signal</div>
  </div>
  <div class="stat-card">
    <div class="stat-value">{len(all_changes)}</div>
    <div class="stat-label">total changes detected</div>
  </div>
  <div class="stat-card holy">
    <div class="stat-value">{n_holy}</div>
    <div class="stat-label">holy-grail candidates</div>
  </div>
  <div class="stat-card cross">
    <div class="stat-value">{n_cross}</div>
    <div class="stat-label">family-crossing changes</div>
  </div>
</div>"""

    # Holy-grail summary (top of report — most scientifically interesting)
    holy_rows = []
    for p in sorted(passages, key=lambda x: x.get("interest_score", 0), reverse=True):
        pid = p.get("passage_id", "—")
        for c in p.get("diachronic_changes", []):
            if not c.get("is_holy_grail_candidate"):
                continue
            pre = c.get("pre_contact_sign", "—")
            post = c.get("post_contact_sign", "—")
            pos = c.get("position", -1)
            n_cons = c.get("n_tablets_consistent", 0)
            ct = c.get("change_type", "substitution")
            holy_rows.append(
                f'<tr>'
                f'<td><a class="pt-id" href="{pid}.html">{pid}</a></td>'
                f'<td>{_change_type_tag(ct)}</td>'
                f'<td><span class="mono" style="font-size:11px">{pre} → {post}</span></td>'
                f'<td class="muted small" style="text-align:center">'
                f'{pos + 1 if pos >= 0 else "—"}</td>'
                f'<td class="muted small" style="text-align:center">{n_cons}</td>'
                f'</tr>'
            )

    holy_section = ""
    if holy_rows:
        holy_section = f"""
<div class="section-label" style="color:var(--holy);margin-bottom:10px">
  Holy Grail candidates — cross-contact substitutions consistent across ≥ 2 post-contact tablets
</div>
<div class="passage-table-wrap">
<table class="passage-table">
<thead><tr>
  <th>Passage</th><th>Type</th><th>Pre → Post sign</th>
  <th style="text-align:center">Position</th>
  <th style="text-align:center">Tablets consistent</th>
</tr></thead>
<tbody>{"".join(holy_rows)}</tbody>
</table>
</div>"""

    # Full passage table — all passages, pre-contact anchors highlighted
    def _row(p: dict) -> str:
        pid = p.get("passage_id", "—")
        score = p.get("interest_score", 0.0)
        atts = p.get("attestations", [])
        n_tabs = len({a.get("tablet") for a in atts})
        pre_c = sum(1 for a in atts if a.get("stratum") == "pre_contact")
        post_c = sum(1 for a in atts if a.get("stratum") == "post_contact")
        changes = p.get("diachronic_changes", [])
        ph = sum(1 for c in changes if c.get("is_holy_grail_candidate"))
        pc = sum(1 for c in changes if c.get("crosses_barthel_family"))

        is_anchor = pre_c > 0
        row_class = ' class="row-anchor"' if is_anchor else ""
        anchor_tag = '<span class="tag-anchor">⚓ pre-contact</span>' if is_anchor else ""

        diachronic_cell = (
            f'<span class="badge badge-pre">{pre_c} pre</span> '
            f'<span class="badge badge-post">{post_c} post</span>'
            if pre_c and post_c else
            f'<span class="badge badge-none">{"pre only" if pre_c else "post only" if post_c else "—"}</span>'
        )
        holy_tag = f' <span class="tag-holy">★ {ph} holy</span>' if ph else ""
        cross_tag = f' <span class="tag-cross">↕ {pc} cross</span>' if pc else ""
        return (
            f'<tr{row_class}>'
            f'<td><a class="pt-id" href="{pid}.html">{pid}</a> {anchor_tag}</td>'
            f'<td class="score-val">{score:.2f}</td>'
            f'<td style="text-align:center" class="muted small">{n_tabs}</td>'
            f'<td>{diachronic_cell}</td>'
            f'<td class="muted small" style="text-align:center">{len(changes)}{holy_tag}{cross_tag}</td>'
            f'</tr>'
        )

    sorted_passages = sorted(passages, key=lambda p: p.get("interest_score", 0), reverse=True)
    table_rows = "".join(_row(p) for p in sorted_passages)

    passage_table = f"""
<div class="section-label" style="margin-bottom:10px">All passages — ranked by interest score</div>
<div class="passage-table-wrap">
<table class="passage-table">
<thead><tr>
  <th>Passage ID</th>
  <th>Interest score</th>
  <th style="text-align:center">Tablets</th>
  <th>Strata</th>
  <th>Changes</th>
</tr></thead>
<tbody>{table_rows}</tbody>
</table>
</div>"""

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>hackingrongo — Diachronic Passage Analysis</title>
<link href="https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,400;0,600;1,400&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>{_CSS}</style>
</head>
<body>
<div class="wrap">

<div class="report-header">
  <div class="report-title">hackingrongo<br>Diachronic Passage Analysis</div>
  <div class="report-subtitle">Cross-contact sign changes in repeated passage sequences — for scholar review</div>
  <div class="report-meta">
    <b>Passages:</b> {n_passages} &nbsp;·&nbsp;
    <b>With pre/post signal:</b> {n_with_diachronic} &nbsp;·&nbsp;
    <b>Total changes detected:</b> {len(all_changes)} &nbsp;·&nbsp;
    <b>Source:</b> {source_file} &nbsp;·&nbsp;
    <b>Generated:</b> {generated}
  </div>
  <div class="abstract">
    <p>Rongorongo contains at least {n_passages} passages that appear on multiple tablets.
    Where the same passage occurs on both a pre-contact tablet (Tablet D, radiocarbon-dated
    to before 1722 CE; Ferrara et al. 2024) and one or more post-contact tablets, sign
    substitutions at consistent positions across multiple post-contact attestations provide
    the strongest available evidence for systematic scribal or linguistic change across
    the contact boundary.</p>
    <p>A <b>holy-grail candidate</b> is a non-allographic substitution that recurs at
    the same canonical position in ≥ 2 independent post-contact tablets, making
    idiosyncratic scribal error unlikely. There are {n_holy} such candidates in this
    dataset. A <b>family-crossing change</b> involves pre- and post-contact consensus
    signs from different Barthel century blocks (e.g. 200-series → 700-series), which
    is iconographically surprising and may reflect sign innovation, semantic shift, or
    scribal tradition discontinuity. There are {n_cross} family-crossing changes.</p>
    <p>Each passage links to a detail page with the full attestation table, glyph-level
    alignment visualisation, and per-change analysis. <b>We invite Prof. Ferrara,
    Dr. Horley, and other rongorongo scholars to review these candidates and advise
    on their linguistic and epigraphic interpretation.</b></p>
  </div>
</div>

{stats_html}

{holy_section}

{passage_table}

<div class="report-footer">
  <p><b>hackingrongo</b> · Parallel passage alignment · MIT License ·
  <a href="https://github.com/violasarah2000/hackingrongo" target="_blank">GitHub</a></p>
  <p>Alignment method: Needleman-Wunsch global alignment with diagonal-first tie-breaking.
  Diachronic analysis: majority-vote consensus sign per stratum, then cross-stratum comparison.
  Holy-grail criterion: non-allographic substitution consistent in ≥ 2 post-contact tablets
  at the same canonical position (Barthel 1958 allograph catalog).
  Family-Crossing: pre/post consensus signs in different Barthel century blocks.</p>
  <p>Pre-contact anchor: Tablet D (radiocarbon 1390–1520 CE, Ferrara et al. 2024).
  Post-contact tablets: H, G, P, Q, and others per Barthel (1958) / Fischer (1997) dating.</p>
  <p>This is a computational hypothesis report. All change candidates require expert
  epigraphic and linguistic review before any interpretive claim.</p>
  <p><b>SperksWerks LLC</b> ·
  <a href="https://sperkswerks.ai" target="_blank">sperkswerks.ai</a> ·
  <a href="mailto:studio@sperkswerks.ai">studio@sperkswerks.ai</a></p>
</div>

</div>
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

    meta = {
        "generated": datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "source_file": passages_json.name,
    }
    logger.info("Building passage summary report: %d passages.", len(passages))
    return _render_summary_page(passages, meta)


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
    return _render_passage_page(passage)


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
        return _render_passage_page(passage)

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

        if individual_files:
            for passage in filtered:
                pid = passage.get("passage_id", "unknown")
                html = _render_passage_page(passage)
                out = output_dir / f"{pid}.html"
                out.write_text(html, encoding="utf-8")
                logger.info("  %s → %s", pid, out.name)

        meta = {
            "generated": datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
            "source_file": passages_json.name,
        }
        index_html = _render_summary_page(filtered, meta)
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
