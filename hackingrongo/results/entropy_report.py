"""
hackingrongo.results.entropy_report
=====================================

Scholar-facing HTML report for the IC sensitivity and boustrophedon
voice-split analysis.

Sections
--------
1. IC pre/post comparison across the three tablet-dating scenarios
2. Robustness summary
3. Boustrophedon voice-split test — IC by odd vs even lines

Inputs
------
  outputs/sensitivity_analysis.json   — written by ``sensitivity_analysis()``

Output
------
  outputs/analysis/entropy_report.html

CLI
---
    python -m hackingrongo.results.entropy_report \\
        --input  outputs/sensitivity_analysis.json \\
        --output outputs/analysis/entropy_report.html

Public API
----------
``build_entropy_report(sensitivity_json)``  → HTML str
``save_entropy_report(sensitivity_json, output_path)``

Design language
---------------
Matches compound_report, divergence_report, passage_report, astronomical_report:
  * Light background — CSS variables --bg / --surface / --surface2
  * Cormorant Garamond (body) + JetBrains Mono (code / metadata)
  * Accent colour --accent = #c4a96d (gold)
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
  --pre: #2563eb; --post: #7c3aed; --robust: #16a34a;
  --warn: #d97706; --boust: #0e7490;
}
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
  background: var(--bg); color: var(--text);
  font-family: 'Cormorant Garamond', 'Palatino Linotype', Georgia, serif;
  font-size: 16px; line-height: 1.65;
}
.wrap { max-width: 1000px; margin: 0 auto; padding: 52px 28px; }
.mono { font-family: 'JetBrains Mono', 'Fira Mono', monospace; }
.muted { color: var(--muted); }
.small { font-size: 11px; }

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

.section-label { font-family: 'JetBrains Mono', monospace; font-size: 9px;
                 color: var(--muted); letter-spacing: 0.1em;
                 text-transform: uppercase; margin-bottom: 10px; }

/* ── Stats row ── */
.stats-row { display: flex; gap: 16px; flex-wrap: wrap; margin: 28px 0 38px; }
.stat-card { background: var(--surface); border: 1px solid var(--border);
             border-radius: 7px; padding: 14px 20px; min-width: 130px; }
.stat-value { font-family: 'JetBrains Mono', monospace; font-size: 22px;
              font-weight: 500; color: #000; }
.stat-label { font-size: 11px; color: var(--muted); margin-top: 2px; }
.stat-card.robust .stat-value { color: var(--robust); }
.stat-card.warn   .stat-value { color: var(--warn); }

/* ── Scenario table ── */
.scenario-wrap { overflow-x: auto; margin-bottom: 40px; }
table.scenario-table { width: 100%; border-collapse: collapse;
                       font-size: 13.5px; }
table.scenario-table th { font-family: 'JetBrains Mono', monospace;
  font-size: 9px; letter-spacing: 0.08em; text-transform: uppercase;
  color: var(--muted); padding: 8px 12px; text-align: left;
  border-bottom: 2px solid var(--border); }
table.scenario-table td { padding: 9px 12px; border-bottom: 1px solid var(--border); }
table.scenario-table tr:last-child td { border-bottom: none; }
.ci-str { font-family: 'JetBrains Mono', monospace; font-size: 10.5px;
          color: var(--muted); }
.delta-pos { color: var(--robust); font-family: 'JetBrains Mono', monospace; }
.delta-neg { color: #dc2626; font-family: 'JetBrains Mono', monospace; }
.badge-ok   { background: #dcfce7; color: var(--robust); }
.badge-fail { background: #fee2e2; color: #dc2626; }
.badge { font-family: 'JetBrains Mono', monospace; font-size: 9.5px;
         border-radius: 3px; padding: 2px 8px; white-space: nowrap; }

/* ── Boustrophedon section ── */
.boust-section { background: #f0f9ff; border: 1px solid #bae6fd;
                 border-radius: 8px; padding: 28px 28px 24px; margin-top: 44px; }
.boust-title { font-size: 18px; font-weight: 600; color: var(--boust);
               margin-bottom: 6px; }
.boust-subtitle { font-size: 13px; color: var(--muted); margin-bottom: 20px; }
.boust-grid { display: grid; grid-template-columns: 1fr 1fr;
              gap: 20px; margin: 20px 0; }
.boust-cell { background: #fff; border: 1px solid #bae6fd;
              border-radius: 6px; padding: 16px 18px; }
.boust-cell-label { font-family: 'JetBrains Mono', monospace; font-size: 9px;
                    letter-spacing: 0.1em; text-transform: uppercase;
                    color: var(--boust); margin-bottom: 8px; }
.boust-ic  { font-family: 'JetBrains Mono', monospace; font-size: 22px;
             font-weight: 500; color: #000; }
.boust-ci  { font-family: 'JetBrains Mono', monospace; font-size: 10.5px;
             color: var(--muted); margin-top: 3px; }
.boust-n   { font-size: 11px; color: var(--muted); margin-top: 4px; }
.finding-box { margin-top: 18px; padding: 14px 18px;
               border-left: 3px solid var(--boust);
               background: rgba(14,116,144,0.05); }
.finding-box.overlap { border-color: var(--muted); background: var(--surface); }
.finding-text { font-size: 14px; color: #333; line-height: 1.7; }
.delta-boust { font-family: 'JetBrains Mono', monospace; font-size: 13px;
               margin-top: 10px; color: #333; }

/* ── Footer ── */
.report-footer { margin-top: 64px; padding-top: 22px;
                 border-top: 1px solid var(--border);
                 font-size: 11.5px; color: var(--muted); line-height: 1.9; }
"""


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------

def _fmt(v: Any, decimals: int = 6) -> str:
    if v is None or (isinstance(v, float) and not math.isfinite(v)):
        return "—"
    return f"{v:.{decimals}f}"


def _ci_str(lo: Any, hi: Any) -> str:
    if lo is None or hi is None:
        return "—"
    return f"[{lo:.5f}, {hi:.5f}]"


def _render_scenario_table(scenarios: dict, deltas: dict, robustness: dict) -> str:
    rows = []
    for name, clusters in scenarios.items():
        pre  = clusters.get("pre_contact",  {})
        post = clusters.get("post_contact", {})
        delta = deltas.get(name, float("nan"))
        non_overlap = (
            pre.get("ic_ci_95_lo") is not None
            and post.get("ic_ci_95_hi") is not None
            and pre["ic_ci_95_lo"] > post["ic_ci_95_hi"]
        )
        delta_cls = "delta-pos" if delta > 0 else "delta-neg"
        badge_cls = "badge badge-ok" if non_overlap else "badge badge-fail"
        badge_txt = "non-overlapping" if non_overlap else "overlapping"
        rows.append(f"""<tr>
  <td class="mono small">{name}</td>
  <td>{_fmt(pre.get("ic"), 6)} <span class="ci-str">{_ci_str(pre.get("ic_ci_95_lo"), pre.get("ic_ci_95_hi"))}</span></td>
  <td>{_fmt(post.get("ic"), 6)} <span class="ci-str">{_ci_str(post.get("ic_ci_95_lo"), post.get("ic_ci_95_hi"))}</span></td>
  <td class="{delta_cls}">{delta:+.6f}</td>
  <td><span class="{badge_cls}">{badge_txt}</span></td>
</tr>""")

    is_robust = robustness.get("robust", False)
    robust_cls = "robust" if is_robust else "warn"
    robust_txt = "robust" if is_robust else "not robust"
    var_pct = robustness.get("relative_variation_pct", float("nan"))
    summary_row = f"""<tr style="background:var(--surface);font-weight:600">
  <td class="mono small">robustness</td>
  <td colspan="2" class="muted small">Δ range: [{_fmt(robustness.get("delta_range",0)/2,6)}, …]
    &nbsp;·&nbsp; variation: {_fmt(var_pct,1)}%</td>
  <td colspan="2">
    <span class="badge badge-{'ok' if is_robust else 'fail'}">{robust_txt}</span>
    &nbsp;all pre&gt;post: {'✓' if robustness.get('all_pre_gt_post') else '✗'}
    &nbsp;all CIs non-overlap: {'✓' if robustness.get('all_ci_non_overlapping') else '✗'}
  </td>
</tr>"""

    return f"""<div class="scenario-wrap">
<table class="scenario-table">
<thead><tr>
  <th>Scenario</th>
  <th>IC pre-contact (95% CI)</th>
  <th>IC post-contact (95% CI)</th>
  <th>Δ IC (pre − post)</th>
  <th>95% CIs</th>
</tr></thead>
<tbody>
{"".join(rows)}
{summary_row}
</tbody>
</table>
</div>"""


def _render_boustrophedon_section(boust: dict) -> str:
    if not boust:
        no_data = (
            '<p class="muted small" style="margin-top:12px">'
            'Boustrophedon IC data not yet available — re-run step 4a to populate.</p>'
        )
        return f"""<div class="boust-section">
  <div class="boust-title">Boustrophedon Voice-Split Test</div>
  <div class="boust-subtitle">IC by odd vs even lines — are the two alternating text streams structurally distinct?</div>
  {no_data}
</div>"""

    ic_odd  = boust.get("ic_odd",  float("nan"))
    ic_even = boust.get("ic_even", float("nan"))
    lo_odd  = boust.get("ic_odd_ci_95_lo")
    hi_odd  = boust.get("ic_odd_ci_95_hi")
    lo_even = boust.get("ic_even_ci_95_lo")
    hi_even = boust.get("ic_even_ci_95_hi")
    n_odd   = boust.get("n_odd_tokens",  0)
    n_even  = boust.get("n_even_tokens", 0)
    n_odd_l = boust.get("n_odd_lines",   0)
    n_even_l= boust.get("n_even_lines",  0)
    delta   = boust.get("delta_ic_odd_minus_even", float("nan"))
    overlap = boust.get("cis_overlap", True)
    marginal= boust.get("marginal_overlap", False)
    ov_frac = boust.get("overlap_fraction")
    finding = boust.get("finding", "")

    finding_cls = "" if not overlap else "overlap"

    ov_note = ""
    if overlap and marginal and ov_frac is not None:
        ov_note = f'<p class="muted small" style="margin-top:6px">CIs overlap by {ov_frac*100:.1f}% of CI width — marginal trend.</p>'

    return f"""<div class="boust-section">
  <div class="boust-title">Boustrophedon Voice-Split Test</div>
  <div class="boust-subtitle">IC by odd vs even lines — are the two alternating text streams structurally distinct?</div>

  <p style="font-size:14px;color:#444;line-height:1.8;max-width:780px">
    Rongorongo is written in reverse boustrophedon: lines alternate direction,
    with odd-numbered lines (1, 3, 5, …) running one way and even-numbered lines
    (2, 4, 6, …) the other.  If the two physical streams were composed in different
    registers — or by different hands — their sign-frequency distributions should
    differ, detectable as IC_odd ≠ IC_even with non-overlapping 95% bootstrap CIs.
  </p>

  <div class="boust-grid">
    <div class="boust-cell">
      <div class="boust-cell-label">Odd lines (1, 3, 5, …)</div>
      <div class="boust-ic">{_fmt(ic_odd, 6)}</div>
      <div class="boust-ci">95% CI {_ci_str(lo_odd, hi_odd)}</div>
      <div class="boust-n">{n_odd_l} lines · {n_odd} tokens</div>
    </div>
    <div class="boust-cell">
      <div class="boust-cell-label">Even lines (2, 4, 6, …)</div>
      <div class="boust-ic">{_fmt(ic_even, 6)}</div>
      <div class="boust-ci">95% CI {_ci_str(lo_even, hi_even)}</div>
      <div class="boust-n">{n_even_l} lines · {n_even} tokens</div>
    </div>
  </div>

  <div class="delta-boust">Δ IC (odd − even) = {delta:+.6f}
    &nbsp;·&nbsp; 95% CIs {'<b style="color:#dc2626">do not overlap</b>' if not overlap else 'overlap'}
  </div>

  <div class="finding-box {finding_cls}">
    <div class="finding-text">{finding}</div>
    {ov_note}
  </div>
</div>"""


def _render_full_report(data: dict, source_file: str) -> str:
    generated = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    scenarios   = data.get("scenarios", {})
    deltas      = data.get("deltas", {})
    robustness  = data.get("robustness", {})
    boust       = data.get("boustrophedon_ic", {})

    n_scenarios = len(scenarios)
    is_robust   = robustness.get("robust", False)
    all_pos     = robustness.get("all_pre_gt_post", False)
    boust_sep   = not boust.get("cis_overlap", True)

    scenario_html   = _render_scenario_table(scenarios, deltas, robustness) if scenarios else \
        '<p class="muted small">No scenario data found.</p>'
    boust_html      = _render_boustrophedon_section(boust)

    stat_robust_cls = "robust" if is_robust else "warn"
    stat_robust_val = "yes" if is_robust else "no"
    stat_boust_cls  = "robust" if boust_sep else ""
    stat_boust_val  = "distinct" if boust_sep else "overlapping"

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>hackingrongo — IC / Entropy Analysis</title>
<link href="https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,400;0,600;1,400&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>{_CSS}</style>
</head>
<body>
<div class="wrap">

<div class="report-header">
  <div class="report-title">hackingrongo<br>IC / Entropy Analysis</div>
  <div class="report-subtitle">Index of Coincidence sensitivity and boustrophedon voice-split test</div>
  <div class="report-meta">
    <b>Scenarios:</b> {n_scenarios} &nbsp;·&nbsp;
    <b>IC_pre &gt; IC_post robust:</b> {stat_robust_val} &nbsp;·&nbsp;
    <b>Boustrophedon CIs:</b> {stat_boust_val} &nbsp;·&nbsp;
    <b>Source:</b> {source_file} &nbsp;·&nbsp;
    <b>Generated:</b> {generated}
  </div>
  <div class="abstract">
    <p>The Index of Coincidence (Friedman 1922) measures how concentrated a
    sign-frequency distribution is relative to a uniform baseline.  IC &gt; 1/k
    indicates structure; a statistically significant IC_pre &gt; IC_post difference
    (non-overlapping 95% bootstrap CIs) is the headline cryptanalytic finding —
    implying pre- and post-contact rongorongo have different underlying distributions,
    consistent with scribal tradition that evolved across the contact boundary.</p>
    <p>Results are presented under three tablet-dating scenarios to test robustness
    against dating uncertainty.  A separate boustrophedon voice-split test checks
    whether odd- and even-numbered lines — which alternate direction in reverse
    boustrophedon — carry structurally different sign distributions.</p>
  </div>
</div>

<div class="stats-row">
  <div class="stat-card">
    <div class="stat-value">{n_scenarios}</div>
    <div class="stat-label">dating scenarios</div>
  </div>
  <div class="stat-card {'robust' if all_pos else 'warn'}">
    <div class="stat-value">{'all' if all_pos else 'not all'}</div>
    <div class="stat-label">pre &gt; post IC</div>
  </div>
  <div class="stat-card {stat_robust_cls}">
    <div class="stat-value">{stat_robust_val}</div>
    <div class="stat-label">robust across scenarios</div>
  </div>
  <div class="stat-card {stat_boust_cls}">
    <div class="stat-value">{stat_boust_val}</div>
    <div class="stat-label">odd/even line CIs</div>
  </div>
</div>

<div class="section-label" style="margin-bottom:14px">IC pre vs post — sensitivity across dating scenarios</div>
{scenario_html}

{boust_html}

<div class="report-footer">
  <p><b>hackingrongo</b> · IC / Entropy Analysis · MIT License</p>
  <p>IC computed as Σ f_i(f_i−1) / [N(N−1)].  95% CIs via 2 000 bootstrap resamples.
  Robustness threshold: max allowed relative variation in Δ IC across scenarios.
  Boustrophedon test: IC split by line parity across all tablets.</p>
</div>

</div>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_entropy_report(sensitivity_json: Path) -> str:
    """Build the IC / entropy HTML report from a sensitivity_analysis.json file.

    Parameters
    ----------
    sensitivity_json : Path
        Path to ``sensitivity_analysis.json`` as written by
        ``sensitivity_analysis()`` in ``zone_b.entropy``.

    Returns
    -------
    str
        Complete HTML document string.
    """
    data = json.loads(sensitivity_json.read_text(encoding="utf-8"))
    logger.info(
        "Building entropy report: %d scenarios, boustrophedon_ic present: %s",
        len(data.get("scenarios", {})),
        "boustrophedon_ic" in data,
    )
    return _render_full_report(data, source_file=sensitivity_json.name)


def save_entropy_report(sensitivity_json: Path, output_path: Path) -> None:
    """Generate and write the IC / entropy HTML report.

    Parameters
    ----------
    sensitivity_json : Path
        Input JSON written by ``sensitivity_analysis()``.
    output_path : Path
        Destination ``.html`` file.  Parent directories are created if needed.
    """
    html = build_entropy_report(sensitivity_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    logger.info("Entropy report written: %s (%d bytes).", output_path, len(html))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate the IC / entropy sensitivity HTML report."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("outputs/sensitivity_analysis.json"),
        help="sensitivity_analysis.json path (default: outputs/sensitivity_analysis.json).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/analysis/entropy_report.html"),
        help="Output HTML path (default: outputs/analysis/entropy_report.html).",
    )
    args = parser.parse_args()

    if not args.input.exists():
        parser.error(f"Input not found: {args.input}  (run step 4a first)")

    save_entropy_report(args.input, args.output)
    print(f"Entropy report → {args.output}")


if __name__ == "__main__":
    main()
