#!/usr/bin/env python3
"""
run_network_centrality.py — Layer 4R: Bigram network centrality analysis.

Builds directed PMI-weighted sign bigram graphs for four corpus variants
(aggregated, per-tablet, pre-contact, post-contact) and computes a full
suite of centrality measures.  Optionally runs quantum PageRank and quantum
Fiedler estimation on IBM Qiskit simulators.

Usage
-----
    python scripts/run_network_centrality.py
    python scripts/run_network_centrality.py --quantum-pagerank
    python scripts/run_network_centrality.py --quantum-fiedler
    python scripts/run_network_centrality.py --quantum-pagerank --quantum-fiedler \\
        --top-signs 64 --pmi-floor 0.0
    python scripts/run_network_centrality.py --smoke-test
"""

from __future__ import annotations

import argparse
import html as _html
import json
import logging
import math
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from hackingrongo.zone_b.network_analysis import (  # noqa: E402
    build_pmi_graph,
    compute_centralities,
    determinative_candidates,
    diachronic_shift,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

_OUTPUTS_DIR = PROJECT_ROOT / "outputs" / "network"


# ---------------------------------------------------------------------------
# Corpus loading
# ---------------------------------------------------------------------------

def _load_corpus_by_stratum(
    corpus_dir: Path,
    use_barthel: bool = True,
) -> dict[str, Any]:
    """Return sequences grouped by stratum and per-tablet.

    Returns:
      {
        "all":         list[list[str]],
        "pre_contact": list[list[str]],
        "post_contact":list[list[str]],
        "per_tablet":  dict[str, list[list[str]]],
        "tablet_ids":  list[str],
      }
    """
    all_seqs: list[list[str]] = []
    pre_seqs: list[list[str]] = []
    post_seqs: list[list[str]] = []
    per_tablet: dict[str, list[list[str]]] = {}

    key = "barthel_code" if use_barthel else "horley_code"

    for path in sorted(corpus_dir.glob("[A-Z].json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        tablet_id = path.stem
        glyphs    = data.get("glyphs", [])
        seq       = [str(g[key]) for g in glyphs if g.get(key)]
        if len(seq) < 3:
            continue
        cluster = data.get("cluster", "unknown")

        all_seqs.append(seq)
        per_tablet.setdefault(tablet_id, []).append(seq)
        if cluster == "pre_contact":
            pre_seqs.append(seq)
        elif cluster == "post_contact":
            post_seqs.append(seq)
        else:
            # Unknown: include in both strata (probabilistic_weighted 50/50)
            n = len(seq)
            pre_seqs.append(seq[: n // 2])
            post_seqs.append(seq[n // 2:])

    return {
        "all":          all_seqs,
        "pre_contact":  pre_seqs,
        "post_contact": post_seqs,
        "per_tablet":   per_tablet,
        "tablet_ids":   sorted(per_tablet.keys()),
    }


# ---------------------------------------------------------------------------
# HTML rendering
# ---------------------------------------------------------------------------

_CSS = """\
:root {
  --bg: #ffffff; --surface: #f8f8fa; --surface2: #f0f0f5;
  --border: #d0d0dd; --text: #1a1a1a; --muted: #666666;
  --accent: #c4a96d; --accent2: #7b9ee0;
  --pre: #2563eb; --post: #7c3aed; --quantum: #0e7490;
  --warn: #d97706; --math: #1e3a5f;
}
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
  background: var(--bg); color: var(--text);
  font-family: 'Cormorant Garamond', 'Palatino Linotype', Georgia, serif;
  font-size: 16px; line-height: 1.65;
}
.wrap { max-width: 1040px; margin: 0 auto; padding: 52px 28px; }
.report-header { border-bottom: 2px solid var(--border);
                 padding-bottom: 38px; margin-bottom: 48px; }
.report-title  { font-size: 34px; font-weight: 600; color: #000; }
.report-subtitle { font-size: 17px; color: var(--accent); font-style: italic; margin-top: 6px; }
.report-meta { margin-top: 18px; font-family: 'JetBrains Mono', monospace;
               font-size: 11px; color: var(--muted); line-height: 2.2; }
.report-meta b { color: #333; }
.sec-head { font-size: 22px; font-weight: 600; color: #000;
            margin: 48px 0 6px; border-top: 1px solid var(--border);
            padding-top: 28px; }
.sec-sub  { font-size: 13.5px; color: var(--muted); font-style: italic; margin-bottom: 22px; }
.intro { max-width: 820px; margin-bottom: 40px; }
.intro p { font-size: 14.5px; color: #333; line-height: 1.9; margin-bottom: 12px; }
.intro b { color: #000; }
.metric-strip { display: flex; flex-wrap: wrap; gap: 12px; margin: 18px 0 28px; }
.mcard { background: var(--surface); border: 1px solid var(--border);
         border-radius: 7px; padding: 14px 18px; min-width: 140px; }
.mcard-label { font-family: 'JetBrains Mono', monospace; font-size: 9.5px;
               letter-spacing: 0.1em; text-transform: uppercase;
               color: var(--muted); margin-bottom: 6px; }
.mcard-n { font-family: 'JetBrains Mono', monospace; font-size: 22px;
           font-weight: 700; color: #000; }
.mcard-sub { font-size: 11px; color: var(--muted); margin-top: 4px; }
.data-table { width: 100%; border-collapse: collapse; font-size: 13px;
              margin: 16px 0 28px; }
.data-table th { text-align: left; padding: 8px 12px;
                 font-family: 'JetBrains Mono', monospace; font-size: 9.5px;
                 letter-spacing: 0.08em; text-transform: uppercase;
                 color: var(--muted); border-bottom: 1px solid var(--border);
                 background: var(--surface); }
.data-table td { padding: 8px 12px; border-bottom: 1px solid var(--border);
                 color: #333; vertical-align: top; }
.data-table tr:last-child td { border-bottom: none; }
.data-table .mono { font-family: 'JetBrains Mono', monospace; font-size: 11px; }
.data-table tr.highlight td { background: #fffde7; }
.badge { display: inline-block; font-family: 'JetBrains Mono', monospace;
         font-size: 9.5px; padding: 2px 7px; border-radius: 3px; }
.badge-det  { background: #fef9c3; color: #92400e; border: 1px solid #fde047; }
.badge-key  { background: #fce7f3; color: #9d174d; border: 1px solid #f9a8d4; }
.badge-qpr  { background: #e0f2fe; color: #0369a1; border: 1px solid #7dd3fc; }
.badge-up   { background: #dcfce7; color: #15803d; border: 1px solid #86efac; }
.badge-down { background: #fee2e2; color: #b91c1c; border: 1px solid #fca5a5; }
.two-col { display: grid; grid-template-columns: 1fr 1fr; gap: 24px; margin-bottom: 28px; }
.community-box { background: var(--surface); border: 1px solid var(--border);
                 border-radius: 6px; padding: 16px; }
.community-box h4 { font-family: 'JetBrains Mono', monospace; font-size: 11px;
                    letter-spacing: 0.08em; text-transform: uppercase;
                    color: var(--muted); margin-bottom: 10px; }
.sign-cloud { font-family: 'JetBrains Mono', monospace; font-size: 11px;
              color: #444; line-height: 2; }
.sign-cloud .s { display: inline-block; background: var(--surface2);
                 border: 1px solid var(--border); border-radius: 3px;
                 padding: 1px 6px; margin: 2px; }
.formula-block { background: var(--surface); border-left: 3px solid var(--math);
                 border-radius: 0 6px 6px 0; padding: 16px 22px; margin: 18px 0;
                 max-width: 780px; }
.formula-block .formula-label { font-family: 'JetBrains Mono', monospace;
                 font-size: 9px; letter-spacing: 0.12em; text-transform: uppercase;
                 color: var(--muted); margin-bottom: 8px; }
.formula-block .formula-expr { font-family: 'Palatino Linotype', Georgia, serif;
                 font-size: 18px; color: var(--math); line-height: 1.7; }
"""

_HTML_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Rongorongo · Network Centrality Report</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,400;0,600;1,400&family=JetBrains+Mono:wght@400;700&display=swap" rel="stylesheet">
<style>{css}</style>
</head>
<body>
<div class="wrap">
{body}
</div>
</body>
</html>
"""


def _fmt(v: float, d: int = 4) -> str:
    return f"{v:.{d}f}" if math.isfinite(v) else "—"


def _sign_tag(s: str) -> str:
    return f'<span class="s">{_html.escape(s)}</span>'


def _top10_table(
    measure: str,
    centrality: dict[str, float],
    label: str,
    badge_class: str = "",
) -> str:
    top = sorted(centrality, key=lambda s: -centrality[s])[:10]
    rows = ""
    for rank, s in enumerate(top, 1):
        badge = f' <span class="badge {badge_class}">{label}</span>' if badge_class else ""
        rows += (
            f'<tr><td class="mono">{rank}</td>'
            f'<td class="mono">{_html.escape(s)}</td>'
            f'<td class="mono">{_fmt(centrality[s])}</td>'
            f'<td>{badge}</td></tr>\n'
        )
    return (
        f'<table class="data-table">'
        f'<thead><tr><th>#</th><th>Sign</th><th>{_html.escape(measure)}</th><th></th></tr></thead>'
        f'<tbody>{rows}</tbody></table>'
    )


def _render_html(result: dict[str, Any]) -> str:
    now_str = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    c = result.get("corpus_graph", {})
    pre = result.get("pre_graph", {})
    post = result.get("post_graph", {})
    c_cen = c.get("centralities", {})

    parts: list[str] = []

    # ── Header ────────────────────────────────────────────────────────────────
    parts.append(f"""
<div class="report-header">
  <div class="report-title">Rongorongo · Network Centrality</div>
  <div class="report-subtitle">Bigram graph analysis — PMI-weighted directed graph</div>
  <div class="report-meta">
    <b>Generated</b> {now_str}<br>
    <b>Nodes (corpus graph)</b> {c.get("n_nodes", "—")}&ensp;
    <b>Edges</b> {c.get("n_edges", "—")}&ensp;
    <b>Density</b> {_fmt(c.get("density", float("nan")), 4)}<br>
    <b>Pre-contact nodes</b> {pre.get("n_nodes", "—")}&ensp;
    <b>Post-contact nodes</b> {post.get("n_nodes", "—")}
  </div>
</div>""")

    # ── Overview metrics ──────────────────────────────────────────────────────
    parts.append('<h2 class="sec-head">Overview</h2>')
    parts.append('<p class="sec-sub">Graph statistics across the four corpus variants.</p>')

    per_tab = result.get("per_tablet_graphs", {})
    parts.append(f"""<div class="metric-strip">
  <div class="mcard">
    <div class="mcard-label">Corpus nodes</div>
    <div class="mcard-n">{c.get("n_nodes", "—")}</div>
    <div class="mcard-sub">unique sign types</div>
  </div>
  <div class="mcard">
    <div class="mcard-label">PMI edges</div>
    <div class="mcard-n">{c.get("n_edges", "—")}</div>
    <div class="mcard-sub">bigram pairs, PMI ≥ 0</div>
  </div>
  <div class="mcard">
    <div class="mcard-label">Tablets analysed</div>
    <div class="mcard-n">{len(per_tab)}</div>
    <div class="mcard-sub">individual graphs</div>
  </div>
  <div class="mcard">
    <div class="mcard-label">Det. candidates</div>
    <div class="mcard-n">{len(result.get("determinative_candidates", []))}</div>
    <div class="mcard-sub">high btwn, low freq</div>
  </div>
</div>""")

    # ── Formula ───────────────────────────────────────────────────────────────
    parts.append("""
<div class="formula-block">
  <div class="formula-label">Edge weight</div>
  <div class="formula-expr">PMI(sᵢ, sⱼ) = log P(sᵢ, sⱼ) / P(sᵢ)·P(sⱼ)</div>
</div>""")

    # ── Betweenness top-10 ────────────────────────────────────────────────────
    parts.append('<h2 class="sec-head">Betweenness Centrality</h2>')
    parts.append('<p class="sec-sub">Signs that bridge distinct sub-communities in the bigram flow (Brandes algorithm, normalised).</p>')
    if c_cen.get("betweenness"):
        parts.append(_top10_table("Betweenness", c_cen["betweenness"], "bridge", "badge-det"))

    # ── PageRank top-10 ───────────────────────────────────────────────────────
    parts.append('<h2 class="sec-head">PageRank  (α = 0.85)</h2>')
    parts.append('<p class="sec-sub">Signs receiving the most context-weighted attentional flow across the corpus graph.</p>')
    if c_cen.get("pagerank"):
        parts.append(_top10_table("PageRank", c_cen["pagerank"], "", ""))

    # ── HITS ──────────────────────────────────────────────────────────────────
    parts.append('<h2 class="sec-head">HITS — Hub and Authority Scores</h2>')
    parts.append('<p class="sec-sub">Hub: signs that introduce diverse continuations.  Authority: signs reached from many hubs.</p>')
    if c_cen.get("hits_hub"):
        parts.append("<h3 style='font-size:15px;margin:18px 0 8px'>Top-10 Hubs</h3>")
        parts.append(_top10_table("Hub score", c_cen["hits_hub"], "hub", ""))
    if c_cen.get("hits_authority"):
        parts.append("<h3 style='font-size:15px;margin:18px 0 8px'>Top-10 Authorities</h3>")
        parts.append(_top10_table("Authority score", c_cen["hits_authority"], "authority", ""))

    # ── Determinative candidates ──────────────────────────────────────────────
    parts.append('<h2 class="sec-head">Determinative Candidates</h2>')
    parts.append('<p class="sec-sub">Signs with betweenness &gt; 2 × mean AND frequency &lt; median — structural bridges despite rarity, consistent with grammatical/determinative function.</p>')
    det = result.get("determinative_candidates", [])
    if det:
        rows = "".join(
            f'<tr class="{"highlight" if i < 3 else ""}">'
            f'<td class="mono">{_html.escape(d["sign"])}</td>'
            f'<td class="mono">{_fmt(d["betweenness"])}</td>'
            f'<td class="mono">{d["freq"]}</td>'
            f'<td class="mono">{_fmt(d["pagerank"])}</td>'
            f'<td><span class="badge badge-det">determinative?</span></td></tr>\n'
            for i, d in enumerate(det[:20])
        )
        parts.append(
            '<table class="data-table"><thead><tr>'
            '<th>Sign</th><th>Betweenness</th><th>Freq</th><th>PageRank</th><th></th>'
            '</tr></thead><tbody>' + rows + '</tbody></table>'
        )
    else:
        parts.append('<p class="sec-sub" style="color:var(--warn)">No determinative candidates found with current thresholds.</p>')

    # ── Diachronic shift ──────────────────────────────────────────────────────
    parts.append('<h2 class="sec-head">Diachronic Centrality Shift</h2>')
    parts.append('<p class="sec-sub">Δ betweenness = betweenness_post − betweenness_pre.  Signs with |Δ| &gt; 1 std are key-change candidates.</p>')
    shift = result.get("diachronic_shift", {})
    kc = shift.get("key_change_candidates", [])
    if kc:
        rows = "".join(
            f'<tr><td class="mono">{_html.escape(d["sign"])}</td>'
            f'<td class="mono">{_fmt(d["pre_betweenness"])}</td>'
            f'<td class="mono">{_fmt(d["post_betweenness"])}</td>'
            f'<td class="mono" style="color:{"#15803d" if d["delta_betweenness"]>0 else "#b91c1c"}">'
            f'{"+" if d["delta_betweenness"]>0 else ""}{_fmt(d["delta_betweenness"])}</td>'
            f'<td class="mono">{_fmt(d["delta_pagerank"])}</td>'
            f'<td><span class="badge badge-key">key change</span></td></tr>\n'
            for d in kc[:15]
        )
        parts.append(
            '<table class="data-table"><thead><tr>'
            '<th>Sign</th><th>Pre btwn</th><th>Post btwn</th>'
            '<th>Δ btwn</th><th>Δ PageRank</th><th></th>'
            '</tr></thead><tbody>' + rows + '</tbody></table>'
        )
    else:
        parts.append('<p class="sec-sub">No key-change candidates (pre/post strata may be too small).</p>')

    # ── Quantum PageRank ──────────────────────────────────────────────────────
    qpr = result.get("quantum_pagerank")
    if qpr and "error" not in qpr:
        parts.append('<h2 class="sec-head">Quantum PageRank</h2>')
        parts.append(
            f'<p class="sec-sub">Szegedy discrete-time quantum walk · '
            f'{qpr["n_qubits"]}-qubit state space · '
            f't = {qpr["n_steps"]} steps · '
            f'L₁ divergence from classical = <b>{_fmt(qpr["l1_divergence"], 4)}</b></p>'
        )
        parts.append("""
<div class="formula-block">
  <div class="formula-label">Walk operator</div>
  <div class="formula-expr">W = SWAP · (2Π_T − I),&ensp; QPageRank(i) = ⟨|i⟩⟨i| ⊗ I⟩_T</div>
</div>""")
        top_div = qpr.get("top_divergent", [])
        if top_div:
            rows = "".join(
                f'<tr><td class="mono">{_html.escape(d["sign"])}</td>'
                f'<td class="mono">{d["classical_rank"]}</td>'
                f'<td class="mono">{d["quantum_rank"]}</td>'
                f'<td class="mono" style="color:{"#15803d" if d["rank_delta"]<0 else "#b91c1c"}">'
                f'{"+" if d["rank_delta"]>0 else ""}{d["rank_delta"]}</td>'
                f'<td class="mono">{_fmt(d["classical_pr"], 5)}</td>'
                f'<td class="mono">{_fmt(d["quantum_pr"], 5)}</td>'
                f'<td><span class="badge badge-qpr">latent hub?</span></td></tr>\n'
                for d in top_div
            )
            parts.append(
                '<table class="data-table"><thead><tr>'
                '<th>Sign</th><th>Classical rank</th><th>Quantum rank</th>'
                '<th>Δ rank</th><th>Classical PR</th><th>Quantum PR</th><th></th>'
                '</tr></thead><tbody>' + rows + '</tbody></table>'
            )

    # ── Quantum Fiedler ───────────────────────────────────────────────────────
    qf = result.get("quantum_fiedler")
    if qf and "error" not in qf:
        parts.append('<h2 class="sec-head">Quantum Fiedler Estimation</h2>')
        parts.append(
            f'<p class="sec-sub">QPE on normalised graph Laplacian · '
            f'{qf["n_qubits_total"]}-qubit circuit · '
            f'Fiedler value λ₂ (classical) = <b>{_fmt(qf["fiedler_value_classical"], 4)}</b> · '
            f'QPE estimate = <b>{_fmt(qf["fiedler_value_qpe"], 4)}</b> · '
            f'N = {qf["n_nodes_used"]} signs</p>'
        )
        ca = qf.get("community_a", [])
        cb = qf.get("community_b", [])
        parts.append(f"""<div class="two-col">
  <div class="community-box">
    <h4>Community A ({len(ca)} signs)</h4>
    <div class="sign-cloud">{"".join(_sign_tag(s) for s in sorted(ca)[:40])}</div>
  </div>
  <div class="community-box">
    <h4>Community B ({len(cb)} signs)</h4>
    <div class="sign-cloud">{"".join(_sign_tag(s) for s in sorted(cb)[:40])}</div>
  </div>
</div>""")

    body = "\n".join(parts)
    return _HTML_TEMPLATE.format(css=_CSS, body=body)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Rongorongo bigram network centrality analysis.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--corpus-dir", type=Path, default=None)
    p.add_argument("--output-dir", type=Path, default=_OUTPUTS_DIR)
    p.add_argument("--pmi-floor",  type=float, default=0.0,
                   help="Minimum PMI to retain an edge (default: 0.0).")
    p.add_argument("--min-cofreq", type=int, default=2,
                   help="Minimum bigram co-occurrence count (default: 2).")
    p.add_argument("--top-signs",  type=int, default=64,
                   help="Signs passed to quantum routines (default: 64).")
    p.add_argument("--quantum-pagerank", action="store_true",
                   help="Run quantum PageRank via Szegedy walk (Qiskit).")
    p.add_argument("--quantum-fiedler",  action="store_true",
                   help="Run quantum Fiedler estimation via QPE (Qiskit).")
    p.add_argument("--n-walk-steps", type=int, default=20,
                   help="Quantum walk steps (default: 20).")
    p.add_argument("--smoke-test", action="store_true",
                   help="Use top-8 signs, 4 walk steps, skip QPE for fast validation.")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = _parse_args()

    corpus_dir = args.corpus_dir
    if corpus_dir is None:
        try:
            from omegaconf import OmegaConf
            cfg = OmegaConf.load(PROJECT_ROOT / "conf" / "config.yaml")
            corpus_dir = PROJECT_ROOT / cfg.paths.corpus_dir
        except Exception:
            corpus_dir = PROJECT_ROOT / "data" / "corpus"

    if not corpus_dir.exists():
        log.error("Corpus directory not found: %s", corpus_dir)
        sys.exit(1)

    if args.smoke_test:
        args.top_signs    = 8
        args.n_walk_steps = 4
        args.quantum_fiedler = False
        log.info("Smoke test: top_signs=8, walk_steps=4, no QPE.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_out = args.output_dir / "centrality_report.json"
    html_out = args.output_dir / "centrality_report.html"

    # ── Load corpus ───────────────────────────────────────────────────────────
    log.info("Loading corpus from %s …", corpus_dir)
    corpus = _load_corpus_by_stratum(corpus_dir)
    log.info(
        "  %d tablets, %d sequences total, %d pre-contact, %d post-contact.",
        len(corpus["tablet_ids"]),
        len(corpus["all"]),
        len(corpus["pre_contact"]),
        len(corpus["post_contact"]),
    )

    # ── Build graphs ──────────────────────────────────────────────────────────
    def _build(seqs: list[list[str]], label: str) -> dict[str, Any]:
        import networkx as nx
        G = build_pmi_graph(seqs, min_cofreq=args.min_cofreq, pmi_floor=args.pmi_floor)
        return {
            "n_nodes":   G.number_of_nodes(),
            "n_edges":   G.number_of_edges(),
            "density":   float(nx.density(G)),
            "centralities": compute_centralities(G),
            "_graph":    G,
            "label":     label,
        }

    log.info("Building corpus-wide graph …")
    corp_data = _build(corpus["all"], "corpus")

    log.info("Building pre-contact graph …")
    pre_data  = _build(corpus["pre_contact"],  "pre_contact")

    log.info("Building post-contact graph …")
    post_data = _build(corpus["post_contact"], "post_contact")

    log.info("Building per-tablet graphs …")
    per_tablet_data: dict[str, dict[str, Any]] = {}
    for tid, seqs in corpus["per_tablet"].items():
        per_tablet_data[tid] = _build(seqs, tid)

    # ── Determinative candidates & diachronic shift ───────────────────────────
    det_cands = determinative_candidates(
        corp_data["_graph"], corp_data["centralities"]
    )
    shift = diachronic_shift(pre_data["centralities"], post_data["centralities"])

    # ── Quantum PageRank ──────────────────────────────────────────────────────
    qpr_result: dict[str, Any] | None = None
    if args.quantum_pagerank:
        log.info("Running quantum PageRank (n_steps=%d, top=%d) …",
                 args.n_walk_steps, args.top_signs)
        try:
            from hackingrongo.zone_b.network_analysis import quantum_pagerank
            qpr_result = quantum_pagerank(
                corp_data["_graph"],
                n_steps=args.n_walk_steps,
                n_pos=max(1, math.ceil(math.log2(args.top_signs + 1))),
            )
            log.info("  L₁ divergence: %.4f", qpr_result.get("l1_divergence", 0))
        except Exception as exc:
            log.error("Quantum PageRank failed: %s", exc)
            qpr_result = {"error": str(exc)}

    # ── Quantum Fiedler ───────────────────────────────────────────────────────
    qf_result: dict[str, Any] | None = None
    if args.quantum_fiedler:
        log.info("Running quantum Fiedler (top_n=%d) …", args.top_signs)
        try:
            from hackingrongo.zone_b.network_analysis import quantum_fiedler
            qf_result = quantum_fiedler(
                corp_data["_graph"],
                top_n=args.top_signs,
                n_pos=6,
                n_qpe=4,
            )
            log.info(
                "  Fiedler λ₂: classical=%.4f, QPE=%.4f",
                qf_result.get("fiedler_value_classical", 0),
                qf_result.get("fiedler_value_qpe", 0),
            )
        except Exception as exc:
            log.error("Quantum Fiedler failed: %s", exc)
            qf_result = {"error": str(exc)}

    # ── Assemble result dict ──────────────────────────────────────────────────
    def _strip_graph(d: dict) -> dict:
        return {k: v for k, v in d.items() if k != "_graph"}

    result: dict[str, Any] = {
        "corpus_graph":            _strip_graph(corp_data),
        "pre_graph":               _strip_graph(pre_data),
        "post_graph":              _strip_graph(post_data),
        "per_tablet_graphs":       {tid: _strip_graph(v) for tid, v in per_tablet_data.items()},
        "determinative_candidates": det_cands,
        "diachronic_shift":        shift,
    }
    if qpr_result is not None:
        result["quantum_pagerank"] = qpr_result
    if qf_result is not None:
        result["quantum_fiedler"] = qf_result

    # ── Save JSON ─────────────────────────────────────────────────────────────
    json_out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info("JSON report written to %s", json_out)

    # ── Save HTML ─────────────────────────────────────────────────────────────
    html_out.write_text(_render_html(result), encoding="utf-8")
    log.info("HTML report written to %s", html_out)

    # ── MLflow tracking ───────────────────────────────────────────────────────
    try:
        import mlflow as _mlflow
        from datetime import datetime as _dt, timezone as _tz
        _mlflow.set_tracking_uri(
            __import__("os").environ.get(
                "MLFLOW_TRACKING_URI",
                f"file://{(PROJECT_ROOT / 'outputs' / 'mlruns').resolve()}",
            )
        )
        _mlflow.set_experiment("rongorongo_network")
        with _mlflow.start_run(run_name=f"centrality-{_dt.now(tz=_tz.utc).strftime('%Y%m%d-%H%M')}"):
            _mlflow.log_params({
                "pmi_floor":       args.pmi_floor,
                "min_cofreq":      args.min_cofreq,
                "top_signs":       args.top_signs,
                "quantum_pagerank": args.quantum_pagerank,
                "quantum_fiedler":  args.quantum_fiedler,
            })
            _mlflow.log_metrics({
                "n_nodes":      corp_data["n_nodes"],
                "n_edges":      corp_data["n_edges"],
                "n_det_cands":  len(det_cands),
                "n_key_changes": len(shift.get("key_change_candidates", [])),
            })
            if qpr_result and "l1_divergence" in qpr_result:
                _mlflow.log_metric("quantum_l1_divergence", qpr_result["l1_divergence"])
            if qf_result and "fiedler_value_classical" in qf_result:
                _mlflow.log_metric("fiedler_classical", qf_result["fiedler_value_classical"])
                _mlflow.log_metric("fiedler_qpe",       qf_result["fiedler_value_qpe"])
            _mlflow.log_artifact(str(json_out), artifact_path="network")
            _mlflow.log_artifact(str(html_out), artifact_path="network")
    except ImportError:
        pass

    # ── Console summary ───────────────────────────────────────────────────────
    c = corp_data
    print(f"\n{'═' * 62}")
    print(f"  Network Centrality — Rongorongo Layer 4R")
    print(f"{'═' * 62}")
    print(f"  Corpus graph   : {c['n_nodes']} nodes, {c['n_edges']} edges (density {c['density']:.4f})")
    print(f"  Det. candidates: {len(det_cands)}")
    print(f"  Key changes    : {len(shift.get('key_change_candidates', []))}")
    if qpr_result and "l1_divergence" in qpr_result:
        print(f"  Quantum L₁     : {qpr_result['l1_divergence']:.4f}")
    if qf_result and "fiedler_value_classical" in qf_result:
        print(f"  Fiedler λ₂     : {qf_result['fiedler_value_classical']:.4f} (QPE: {qf_result['fiedler_value_qpe']:.4f})")
    if det_cands:
        print(f"\n  Top determinative candidates:")
        for d in det_cands[:5]:
            print(f"    {d['sign']:>8}  btwn={d['betweenness']:.4f}  freq={d['freq']}")
    print(f"\n  Output: {html_out}")
    print()


if __name__ == "__main__":
    main()
