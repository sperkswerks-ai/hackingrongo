"""hackingrongo.results.pozdniakov_report
========================================

Standalone HTML summary for the Pozdniakov hypothesis tests.

Input
-----
outputs/analysis/pozdniakov_hypothesis_tests.json

Output
------
outputs/analysis/pozdniakov_hypothesis_report.html

CLI
---
    python -m hackingrongo.results.pozdniakov_report \
        --input outputs/analysis/pozdniakov_hypothesis_tests.json \
        --output outputs/analysis/pozdniakov_hypothesis_report.html
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_CSS = """
:root {
  --bg: #f4efe6;
  --panel: rgba(255, 252, 246, 0.92);
  --panel-strong: #fff9ef;
  --ink: #1f1a17;
  --muted: #6f655b;
  --accent: #4a7c59;
  --accent-2: #8c5a3c;
  --line: rgba(80, 63, 49, 0.16);
  --shadow: 0 16px 45px rgba(52, 39, 29, 0.12);
}
* { box-sizing: border-box; }
html, body { margin: 0; padding: 0; }
body {
  font-family: Georgia, 'Times New Roman', serif;
  color: var(--ink);
  background:
    radial-gradient(circle at 12% 18%, rgba(74, 124, 89, 0.12), transparent 26%),
    radial-gradient(circle at 82% 10%, rgba(140, 90, 60, 0.12), transparent 24%),
    linear-gradient(180deg, #f8f3ea 0%, var(--bg) 100%);
}
.wrap {
  max-width: 1180px;
  margin: 0 auto;
  padding: 34px 24px 56px;
}
.hero {
  position: relative;
  overflow: hidden;
  background: linear-gradient(145deg, rgba(255, 250, 241, 0.96), rgba(244, 237, 225, 0.88));
  border: 1px solid var(--line);
  border-radius: 28px;
  box-shadow: var(--shadow);
  padding: 30px 30px 24px;
}
.hero::after {
  content: '';
  position: absolute;
  inset: auto -120px -110px auto;
  width: 280px;
  height: 280px;
  border-radius: 50%;
  background: radial-gradient(circle, rgba(74, 124, 89, 0.14), transparent 68%);
  pointer-events: none;
}
.kicker {
  font-family: 'SFMono-Regular', Consolas, 'Liberation Mono', Menlo, monospace;
  text-transform: uppercase;
  letter-spacing: 0.18em;
  font-size: 0.74rem;
  color: var(--accent-2);
  margin-bottom: 12px;
}
.title {
  font-size: clamp(2.2rem, 4vw, 3.8rem);
  line-height: 0.95;
  margin: 0;
  font-weight: 700;
}
.subtitle {
  margin: 14px 0 0;
  max-width: 880px;
  font-size: 1.05rem;
  line-height: 1.65;
  color: var(--muted);
}
.meta-row {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 14px;
  margin-top: 22px;
}
.stat {
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: 18px;
  padding: 16px 18px;
  box-shadow: 0 8px 24px rgba(52, 39, 29, 0.06);
}
.stat-value {
  font-size: 1.55rem;
  font-weight: 700;
  line-height: 1;
}
.stat-label {
  margin-top: 6px;
  font-size: 0.82rem;
  color: var(--muted);
  text-transform: uppercase;
  letter-spacing: 0.08em;
}
.section {
  margin-top: 24px;
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: 24px;
  box-shadow: var(--shadow);
  padding: 24px;
}
.section h2 {
  margin: 0 0 16px;
  font-size: 1.6rem;
}
.section p {
  margin: 0 0 12px;
  line-height: 1.7;
}
.grid {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 16px;
}
.card {
  background: var(--panel-strong);
  border: 1px solid var(--line);
  border-radius: 18px;
  padding: 18px;
}
.card h3 {
  margin: 0 0 12px;
  font-size: 1.08rem;
}
.metric-row {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 12px;
}
.metric {
  background: rgba(255,255,255,0.6);
  border: 1px solid rgba(80, 63, 49, 0.1);
  border-radius: 14px;
  padding: 12px;
}
.metric .label {
  font-size: 0.75rem;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  color: var(--muted);
}
.metric .value {
  margin-top: 6px;
  font-size: 1.2rem;
  font-weight: 700;
}
.table-wrap {
  overflow-x: auto;
}
.table {
  width: 100%;
  border-collapse: collapse;
  margin-top: 8px;
  font-size: 0.96rem;
}
.table th, .table td {
  border-bottom: 1px solid var(--line);
  padding: 10px 8px;
  text-align: left;
  vertical-align: top;
}
.table th {
  font-size: 0.75rem;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  color: var(--muted);
}
.badge {
  display: inline-block;
  padding: 4px 10px;
  border-radius: 999px;
  font-size: 0.74rem;
  font-family: 'SFMono-Regular', Consolas, 'Liberation Mono', Menlo, monospace;
  background: rgba(74, 124, 89, 0.12);
  color: var(--accent);
}
.badge.warn { background: rgba(140, 90, 60, 0.12); color: var(--accent-2); }
.footer {
  margin-top: 18px;
  color: var(--muted);
  font-size: 0.9rem;
  line-height: 1.6;
}
code {
  font-family: 'SFMono-Regular', Consolas, 'Liberation Mono', Menlo, monospace;
  font-size: 0.92em;
}
@media (max-width: 900px) {
  .meta-row, .grid, .metric-row { grid-template-columns: 1fr; }
}
"""


def _fmt_num(value: Any, digits: int = 4) -> str:
    if value is None:
        return "—"
    try:
        if isinstance(value, bool):
            return "yes" if value else "no"
        if isinstance(value, int):
            return f"{value:,}"
        value = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not (value == value):
        return "—"
    return f"{value:.{digits}f}"


def _metric(label: str, value: Any, digits: int = 4) -> str:
    return f"""
    <div class="metric">
      <div class="label">{label}</div>
      <div class="value">{_fmt_num(value, digits)}</div>
    </div>
    """


def _render_test1(data: dict) -> str:
    test1 = data.get("test1") or {}
    pre_ci = test1.get("pre_ci") or [None, None]
    post_ci = test1.get("post_ci") or [None, None]
    return f"""
    <div class="card">
      <h3>Test 1. Matched-size frequency correlation</h3>
      <div class="metric-row">
        {_metric("Pre-contact rho", test1.get("pre_rho"))}
        {_metric("Post-contact rho", test1.get("post_full_rho"))}
        {_metric("P(post bootstrap ≥ pre rho)", test1.get("p_post_boot_ge_pre"), 6)}
        {_metric("P(pre bootstrap ≥ post rho)", test1.get("p_pre_boot_ge_post_full"), 6)}
      </div>
      <p>Pre-contact data were downsampled to the same token count as the requested sample size, then compared against bootstrap resamples of the post-contact corpus. This is the direct test of whether the syllabic correlation is special to the pre-contact layer or broadly shared by the later corpus.</p>
      <p><span class="badge">pre 95% CI {_fmt_num(pre_ci[0])} to {_fmt_num(pre_ci[1])}</span> <span class="badge">post 95% CI {_fmt_num(post_ci[0])} to {_fmt_num(post_ci[1])}</span></p>
    </div>
    """


def _render_test2(data: dict) -> str:
    test2 = data.get("test2") or {}
    rows = []
    for label in ("pre_contact", "post_contact", "unknown"):
        vals = test2.get(label) or {}
        rows.append(
            f"<tr><td>{label.replace('_', ' ')}</td><td>{_fmt_num(vals.get('n_tokens'), 0)}</td><td>{_fmt_num(vals.get('n_types'), 0)}</td><td>{_fmt_num(vals.get('n_hapax'), 0)}</td><td>{_fmt_num(vals.get('hapax_rate'))}</td></tr>"
        )
    return f"""
    <div class="card">
      <h3>Test 2. Hapax legomena by stratum</h3>
      <div class="table-wrap">
        <table class="table">
          <thead><tr><th>Stratum</th><th>Tokens</th><th>Types</th><th>Hapax</th><th>Hapax / type</th></tr></thead>
          <tbody>{''.join(rows)}</tbody>
        </table>
      </div>
      <p>Higher hapax density is a crude signal for innovation, noise, or late-stage lexical turnover. The table is intended to show whether the contact-era material is lexically more volatile than the pre-contact corpus.</p>
    </div>
    """


def _render_test3(data: dict) -> str:
    test3 = data.get("test3") or {}
    return f"""
    <div class="card">
      <h3>Test 3. Passage stability</h3>
      <div class="metric-row">
        {_metric("Passages with pre/post", test3.get("n_passages_with_pre_and_post"), 0)}
        {_metric("Mean pre-post edit distance", test3.get("mean_pre_post_edit_distance"))}
        {_metric("Mean post-post edit distance", test3.get("mean_post_post_edit_distance"))}
        {_metric("Non-singleton passages", len(test3.get('rows') or []), 0)}
      </div>
      <p>This section tracks whether parallel attestations of the same passage remain structurally stable across strata. Lower normalized edit distance means the passage form is more conserved; higher distance means it is more divergent or more contaminated by local variation.</p>
    </div>
    """


def _render_test4(data: dict) -> str:
    test4 = data.get("test4") or {}
    tablets = test4.get("tablets") or []
    top = max(tablets, key=lambda r: r.get("lm_score", float("-inf")), default=None)
    bottom = min(tablets, key=lambda r: r.get("lm_score", float("inf")), default=None)
    table_rows = []
    for row in tablets[:12]:
        table_rows.append(
            "<tr>"
            f"<td>{row.get('tablet_id', '—')}</td>"
            f"<td>{row.get('stratum', '—')}</td>"
            f"<td>{_fmt_num(row.get('date_midpoint'))}</td>"
            f"<td>{_fmt_num(row.get('n_tokens'), 0)}</td>"
            f"<td>{_fmt_num(row.get('lm_score'))}</td>"
            "</tr>"
        )
    top_line = f"Best-scoring tablet: <code>{top.get('tablet_id', '—')}</code> ({_fmt_num(top.get('lm_score'))})" if top else "Best-scoring tablet: —"
    bottom_line = f"Worst-scoring tablet: <code>{bottom.get('tablet_id', '—')}</code> ({_fmt_num(bottom.get('lm_score'))})" if bottom else "Worst-scoring tablet: —"
    return f"""
    <div class="card">
      <h3>Test 4. LM score by tablet under H0001</h3>
      <div class="metric-row">
        {_metric("Spearman rho(date, score)", test4.get("spearman_rho_date_score"))}
        <div class="metric"><div class="label">Best tablet</div><div class="value">{top_line}</div></div>
        <div class="metric"><div class="label">Worst tablet</div><div class="value">{bottom_line}</div></div>
        <div class="metric"><div class="label">Tablets scored</div><div class="value">{_fmt_num(len(tablets), 0)}</div></div>
      </div>
      <p>The score-vs-date trend is a coarse check for whether the hypothesis degrades or improves over time. A strong monotonic slope would suggest the fit is not random with respect to chronology.</p>
      <div class="table-wrap">
        <table class="table">
          <thead><tr><th>Tablet</th><th>Stratum</th><th>Date midpoint</th><th>Tokens</th><th>LM score</th></tr></thead>
          <tbody>{''.join(table_rows)}</tbody>
        </table>
      </div>
    </div>
    """


def _render_test5(data: dict) -> str:
    test5 = data.get("test5") or {}
    return f"""
    <div class="card">
      <h3>Test 5. Zipf null model</h3>
      <div class="metric-row">
        {_metric("Zipf alpha", test5.get("alpha_zipf_post_contact"))}
        {_metric("Observed post rho", test5.get("observed_post_rho"))}
        {_metric("Null mean rho", test5.get("null_mean_rho"))}
        {_metric("P(null ≥ observed)", test5.get("p_null_ge_observed"), 6)}
      </div>
      <p>This is the most important control: it asks whether the observed post-contact correlation stands out from a synthetic world that matches only a Zipf-like frequency law and random phoneme assignment. If the observed value sits inside the null cloud, the syllabic signal is not very discriminating.</p>
      <p><span class="badge warn">95% null CI {_fmt_num((test5.get('null_ci') or [None, None])[0])} to {_fmt_num((test5.get('null_ci') or [None, None])[1])}</span></p>
    </div>
    """


def _render_full_report(data: dict, source_file: str) -> str:
    generated = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    artifacts = data.get("artifacts") or {}
    test1 = data.get("test1") or {}
    test5 = data.get("test5") or {}
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Pozdniakov Hypothesis Report</title>
<style>{_CSS}</style>
</head>
<body>
<div class="wrap">
  <div class="hero">
    <div class="kicker">hackingrongo / analysis</div>
    <h1 class="title">Pozdniakov Hypothesis Report</h1>
    <p class="subtitle">A compact HTML summary of the notebook-generated hypothesis tests. The focus is whether the syllabic-only signal is actually distinguishable from a mixed or null explanation, with special attention to the matched-size frequency test and the Zipf null control.</p>
    <div class="meta-row">
      <div class="stat"><div class="stat-value">{data.get('hypothesis_id', 'H0001')}</div><div class="stat-label">hypothesis</div></div>
      <div class="stat"><div class="stat-value">{_fmt_num(data.get('sample_size'), 0)}</div><div class="stat-label">matched sample size</div></div>
      <div class="stat"><div class="stat-value">{_fmt_num(data.get('n_bootstrap'), 0)}</div><div class="stat-label">bootstrap draws</div></div>
      <div class="stat"><div class="stat-value">{_fmt_num(data.get('n_null'), 0)}</div><div class="stat-label">null draws</div></div>
    </div>
    <div class="footer">
      Source JSON: <code>{source_file}</code><br>
      Generated: {generated}<br>
      Artifacts: <code>{artifacts.get('summary_plot', '—')}</code> and <code>{artifacts.get('tablet_score_plot', '—')}</code>
    </div>
  </div>

  <div class="section">
    <h2>Executive Summary</h2>
    <div class="grid">
      <div class="card">
        <h3>Test 1 takeaway</h3>
        <p>{'The pre-contact matched-size correlation is ' + ('stronger than' if (test1.get('pre_rho') or 0) > (test1.get('post_full_rho') or 0) else 'not stronger than') + ' the post-contact score in the current run.'}</p>
      </div>
      <div class="card">
        <h3>Test 5 takeaway</h3>
        <p>{'The observed post-contact rho is ' + ('outside' if ((test5.get('p_null_ge_observed') or 1.0) < 0.05) else 'inside') + ' the Zipf null tail in the current run.'}</p>
      </div>
    </div>
  </div>

  <div class="section">
    <h2>Detailed Results</h2>
    <div class="grid">
      {_render_test1(data)}
      {_render_test2(data)}
      {_render_test3(data)}
      {_render_test5(data)}
    </div>
  </div>

  <div class="section">
    <h2>Tablet Score Trace</h2>
    {_render_test4(data)}
  </div>

  <div class="section">
    <h2>Interpretation</h2>
    <p>The notebook’s JSON output is now mirrored by this HTML page, so the Pozdniakov tests can be inspected without opening the notebook. The two decisive checks are the matched-size frequency comparison and the Zipf null control: if those are weak, the syllabic-only interpretation is probably not buying much beyond a generic frequency law.</p>
    <p>The page is intentionally separate from the main decipherment report because it is an analytical sidecar rather than part of the core hypothesis ranking. If you want, it can be linked from the existing decipherment report or from the notebook outputs index.</p>
  </div>
</div>
</body>
</html>"""


def build_pozdniakov_report(results_json: Path) -> str:
    data = json.loads(results_json.read_text(encoding="utf-8"))
    logger.info("Building Pozdniakov report from %s", results_json)
    return _render_full_report(data, source_file=results_json.name)


def save_pozdniakov_report(results_json: Path, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    html = build_pozdniakov_report(results_json)
    output_path.write_text(html, encoding="utf-8")
    logger.info("Pozdniakov report written: %s (%d bytes)", output_path, len(html.encode("utf-8")))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build the Pozdniakov HTML summary report.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("outputs/analysis/pozdniakov_hypothesis_tests.json"),
        help="Input JSON written by the notebook cell.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/analysis/pozdniakov_hypothesis_report.html"),
        help="Output HTML path.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Logging level.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    logging.basicConfig(level=getattr(logging, str(args.log_level).upper(), logging.INFO), format="[%(levelname)s] %(message)s")
    if not args.input.exists():
        raise FileNotFoundError(f"Missing input JSON: {args.input}")
    save_pozdniakov_report(args.input, args.output)
    print(str(args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
