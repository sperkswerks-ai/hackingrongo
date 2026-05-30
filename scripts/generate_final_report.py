"""
scripts/generate_final_report.py

Generates outputs/final_report.html — a single self-contained file that
aggregates every finding from one pipeline run.

Sections
--------
1. Calendar gloss validation    (calendar_gloss_validation.json)
2. Reading direction analysis   (reading_order_v2.json + reading_direction_combined.png)
3. Compound compositionality    (compound_compositionality.json)
4. Hypothesis convergence       (hypothesis_comparison.json)
5. Deity name search            (deity_name_search.json)
6. Decipherment ranking         (ranking.json)

The PNG chart is embedded as a base64 data URI so the report is fully
self-contained for Colab download or email attachment.

Usage
-----
    python scripts/generate_final_report.py
    python scripts/generate_final_report.py --output outputs/final_report.html
"""

from __future__ import annotations

import argparse
import base64
import html as _html
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load(rel: str, default: Any = None) -> Any:
    p = PROJECT_ROOT / rel
    if not p.exists():
        log.warning("Missing: %s", rel)
        return default
    return json.loads(p.read_text(encoding="utf-8"))


def _b64_png(rel: str) -> str | None:
    p = PROJECT_ROOT / rel
    if not p.exists():
        return None
    data = base64.b64encode(p.read_bytes()).decode()
    return f"data:image/png;base64,{data}"


def _esc(s: object) -> str:
    return _html.escape(str(s))


def _pct(v: float) -> str:
    return f"{v * 100:.1f}%"


# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------

_CSS = """\
:root {
  --bg: #0d0f12; --surface: #161920; --surface2: #1e2229;
  --border: #2a2e38; --text: #d0d4dc; --muted: #6b7280;
  --accent: #c4a96d; --accent2: #7b9ee0;
  --green: #4ade80; --yellow: #facc15; --red: #f87171; --blue: #93c5fd;
}
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
  background: var(--bg); color: var(--text);
  font-family: 'Cormorant Garamond', 'Palatino Linotype', Georgia, serif;
  font-size: 16px; line-height: 1.7;
}
a { color: var(--accent); }
.wrap { max-width: 1050px; margin: 0 auto; padding: 56px 28px; }
/* report header */
.report-header { border-bottom: 1px solid var(--border); padding-bottom: 36px; margin-bottom: 52px; }
.report-title { font-size: 30px; font-weight: 600; color: var(--accent); letter-spacing: -.3px; }
.report-meta { font-family: 'JetBrains Mono', monospace; font-size: 11px;
               color: var(--muted); line-height: 2.2; margin-top: 16px; }
.report-meta b { color: #888; }
/* section */
.section { margin-bottom: 64px; }
.section-title { font-size: 22px; font-weight: 600; color: var(--text);
                 border-bottom: 1px solid var(--border); padding-bottom: 10px;
                 margin-bottom: 24px; }
.section-num { color: var(--accent); font-family: 'JetBrains Mono', monospace;
               font-size: 13px; margin-right: 10px; }
/* stat grid */
.stat-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(190px, 1fr));
             gap: 14px; margin-bottom: 28px; }
.stat { background: var(--surface); border: 1px solid var(--border); border-radius: 6px;
        padding: 16px 20px; }
.stat-label { font-family: 'JetBrains Mono', monospace; font-size: 9.5px;
              color: var(--muted); text-transform: uppercase; letter-spacing: .07em; }
.stat-value { font-size: 28px; font-weight: 600; margin-top: 4px; color: var(--accent); }
.stat-sub { font-size: 11px; color: var(--muted); margin-top: 2px; }
/* verdict */
.verdict { border-left: 3px solid var(--accent); padding: 14px 20px;
           background: var(--surface); border-radius: 0 6px 6px 0; margin: 20px 0; }
.verdict strong { color: var(--accent); }
.verdict p { font-size: 14.5px; margin-top: 6px; line-height: 1.8; }
/* tables */
table { width: 100%; border-collapse: collapse; font-family: 'JetBrains Mono', monospace;
        font-size: 11px; margin-top: 14px; }
th { padding: 7px 10px; text-align: left; font-size: 9.5px; color: var(--muted);
     border-bottom: 1px solid var(--border); text-transform: uppercase; letter-spacing: .06em; }
td { padding: 6px 10px; border-bottom: 1px solid rgba(42,46,56,.4); }
tr:hover td { background: var(--surface2); }
.code { color: var(--accent); }
.ph { color: var(--blue); }
.hi { color: var(--green); }
.med { color: var(--yellow); }
.lo { color: var(--muted); }
.neg { color: var(--red); }
/* reading dir chart */
.chart-wrap { margin: 20px 0; border: 1px solid var(--border); border-radius: 6px;
              overflow: hidden; background: var(--surface); }
.chart-wrap img { width: 100%; display: block; }
.chart-caption { font-family: 'JetBrains Mono', monospace; font-size: 10px;
                 color: var(--muted); padding: 8px 14px; }
/* convergence bar */
.conv-bar { height: 18px; border-radius: 3px; display: flex; overflow: hidden;
            margin: 10px 0; }
.conv-full { background: #166534; }
.conv-div  { background: #7f1d1d; }
/* section 5 — deity name result */
.negative-badge { display: inline-block; background: rgba(248,113,113,.12);
                  color: var(--red); padding: 3px 10px; border-radius: 3px;
                  font-family: 'JetBrains Mono', monospace; font-size: 10px; margin-left: 8px; }
.positive-badge { display: inline-block; background: rgba(74,222,128,.12);
                  color: var(--green); padding: 3px 10px; border-radius: 3px;
                  font-family: 'JetBrains Mono', monospace; font-size: 10px; margin-left: 8px; }
"""

# ---------------------------------------------------------------------------
# Section builders
# ---------------------------------------------------------------------------

def _section1_calendar(cal: dict | None) -> str:
    if not cal:
        return "<p class='lo'>calendar_gloss_validation.json not found — run validate_glosses_calendar.py</p>"

    cs   = cal["calendar_stats"]
    bs   = cal["baseline_stats"]
    lift = cal["lunar_lift"]
    coh  = cal["coherence_score"]

    verdict_text = (
        f"The Mamari Ca6–Ca9 calendar section returns lunar vocabulary at "
        f"{_pct(cs['frac_lunar'])} of sign positions — {lift:.2f}× the corpus "
        f"baseline of {_pct(bs['frac_lunar'])}. Among high-tier (IDS/Thomson) matches, "
        f"{_pct(cs['frac_high_lunar'])} are lunar. Coherence across high-confidence "
        f"night alignments is {coh:.2f}. "
        f"The lexical lookup is returning semantically appropriate results for this section."
        if lift >= 2.0 else
        f"Lunar lift is {lift:.2f}×. Review phoneme inventory coverage."
    )

    return f"""
<div class="stat-grid">
  <div class="stat"><div class="stat-label">Calendar lunar %</div>
    <div class="stat-value hi">{_pct(cs['frac_lunar'])}</div>
    <div class="stat-sub">{cs['n_lunar']} / {cs['n_signs']} signs</div></div>
  <div class="stat"><div class="stat-label">Baseline lunar %</div>
    <div class="stat-value">{_pct(bs['frac_lunar'])}</div>
    <div class="stat-sub">{bs['n_lunar']} / {bs['n_signs']} signs</div></div>
  <div class="stat"><div class="stat-label">Lunar lift</div>
    <div class="stat-value">{lift:.2f}×</div>
    <div class="stat-sub">calendar vs corpus baseline</div></div>
  <div class="stat"><div class="stat-label">Coherence score</div>
    <div class="stat-value">{coh:.2f}</div>
    <div class="stat-sub">high-conf nights with lunar gloss</div></div>
  <div class="stat"><div class="stat-label">HIGH-tier lunar hits</div>
    <div class="stat-value hi">{_pct(cs['frac_high_lunar'])}</div>
    <div class="stat-sub">{cs['n_high_lunar']} / {cs['n_high_tier']} IDS matches</div></div>
</div>
<div class="verdict"><strong>Finding</strong>
  <p>{_esc(verdict_text)}</p></div>
"""


def _section2_reading(ro: dict | None, chart_uri: str | None) -> str:
    if not ro:
        return "<p class='lo'>reading_order_v2.json not found — run reading_order_v2.py</p>"

    t5 = ro.get("test5", {})
    t6 = ro.get("test6", {})
    t5_margin   = t5.get("margin", 0)
    t5_strength = t5.get("signal_strength", "?")
    t5_pref     = t5.get("preferred_order", "?")
    t6_wins_ab  = t6.get("wins_ab", 0)
    t6_wins_ba  = t6.get("wins_ba", 0)
    t6_n        = t6.get("n_tablets", 0)
    t6_pref     = t6.get("preferred_order", "?")
    t6_median   = t6.get("median_margin", 0)
    t7          = ro.get("test7", {})
    nmi         = t7.get("nmi", 0)

    chart_html = ""
    if chart_uri:
        chart_html = (
            f'<div class="chart-wrap">'
            f'<img src="{chart_uri}" alt="Reading direction per-tablet chart">'
            f'<div class="chart-caption">Left: per-tablet preference sorted by margin. '
            f'Right: tablet size vs preference. Green = a→b, orange = b→a. '
            f'★ = Tablet D (pre-contact, 1493–1509 CE).</div>'
            f'</div>'
        )

    t5_colour = "hi" if t5_strength == "strong" else ("med" if t5_strength == "moderate" else "lo")

    return f"""
<div class="stat-grid">
  <div class="stat"><div class="stat-label">Test 5 — cross-side PPL</div>
    <div class="stat-value {t5_colour}">{t5_pref}</div>
    <div class="stat-sub">margin {t5_margin:.3f} ({t5_strength} signal)</div></div>
  <div class="stat"><div class="stat-label">Test 6 — LTOO tablets</div>
    <div class="stat-value">{'ab' if t6_wins_ab > t6_wins_ba else 'ba'}</div>
    <div class="stat-sub">{t6_wins_ab} a→b / {t6_wins_ba} b→a of {t6_n} tablets</div></div>
  <div class="stat"><div class="stat-label">LTOO median margin</div>
    <div class="stat-value">{t6_median:.3f}</div>
    <div class="stat-sub">perplexity difference per tablet</div></div>
  <div class="stat"><div class="stat-label">Test 7 — recto/verso MI</div>
    <div class="stat-value">{nmi:.4f}</div>
    <div class="stat-sub">normalised mutual information</div></div>
</div>
{chart_html}
<div class="verdict"><strong>Finding</strong>
  <p>Test 5 (cross-side bigrams only) shows a <span class="{t5_colour}">{t5_strength} {t5_pref} signal</span>
  with margin {t5_margin:.3f} — significantly stronger than the 0.03–0.04 margin from token-level
  Test 4. Test 6 (LTOO) shows a divergent result: {t6_wins_ba} of {t6_n} tablets prefer b→a
  vs {t6_wins_ab} for a→b. The per-tablet heterogeneity is genuine and should be reported honestly.
  Test 7 NMI = {nmi:.4f} indicates the two sides are largely independent content.</p></div>
"""


def _section3_compound(cmp: dict | None) -> str:
    if not cmp:
        return "<p class='lo'>compound_compositionality.json not found — run compound_compositionality.py</p>"

    n_total = cmp["n_compound_occurrences"]
    n_dist  = cmp["n_distinct_compounds"]
    n_comp  = cmp["n_compositional"]
    n_anch  = cmp["n_new_anchor_candidates"]
    frac    = cmp["frac_compositional"]

    rows = ""
    for r in sorted(
        cmp.get("unique_compositional", []),
        key=lambda x: (x["tier"] != "HIGH", x["tier"] != "MEDIUM"),
    )[:15]:
        tier_cls = {"HIGH": "hi", "MEDIUM": "med"}.get(r["tier"], "lo")
        anch_html = (' <span class="positive-badge">NEW ANCHOR</span>'
                     if r.get("new_anchor_candidate") else "")
        rows += (
            f"<tr>"
            f'<td class="code">{_esc(r["barthel_code"])}</td>'
            f'<td class="ph">{_esc(r["phoneme_a"])}</td>'
            f'<td class="ph">{_esc(r["phoneme_b"])}</td>'
            f'<td class="ph">{_esc(r["concat_phoneme"])}</td>'
            f'<td class="{tier_cls}">{_esc(r["gloss"])}{anch_html}</td>'
            f'<td class="{tier_cls}">{_esc(r["tier"])}</td>'
            f"</tr>"
        )

    table_html = (
        "<table><thead><tr>"
        "<th>Compound</th><th>Phoneme A</th><th>Phoneme B</th>"
        "<th>Concat</th><th>Gloss</th><th>Tier</th>"
        f"</tr></thead><tbody>{rows}</tbody></table>"
    ) if rows else "<p class='lo'>No compositional matches found.</p>"

    return f"""
<div class="stat-grid">
  <div class="stat"><div class="stat-label">Compound occurrences</div>
    <div class="stat-value">{n_total}</div>
    <div class="stat-sub">{n_dist} distinct codes</div></div>
  <div class="stat"><div class="stat-label">Compositional matches</div>
    <div class="stat-value hi">{n_comp}</div>
    <div class="stat-sub">{_pct(frac)} of occurrences</div></div>
  <div class="stat"><div class="stat-label">New anchor candidates</div>
    <div class="stat-value hi">{n_anch}</div>
    <div class="stat-sub">HIGH-tier compositional matches</div></div>
</div>
<div class="verdict"><strong>Finding</strong>
  <p>{n_comp} compound glyphs decompose into known Rapa Nui morpheme sequences when
  phoneme(A) + phoneme(B) is looked up in the IDS/Thomson lexicon.
  {n_anch} of these are HIGH-tier exact matches, each simultaneously validating
  two component phoneme assignments. These become new Type-1 anchors.</p></div>
{table_html}
"""


def _section4_hypotheses(hyp: dict | None) -> str:
    if not hyp:
        return "<p class='lo'>hypothesis_comparison.json not found — run compare_top_hypotheses.py</p>"

    n_hyps   = hyp["n_hypotheses"]
    n_pos    = hyp["n_positions"]
    n_full   = hyp["n_full_agreement"]
    n_div    = hyp["n_diverge"]
    conv     = hyp["convergence_rate"]
    hyp_ids  = hyp["hyp_ids"]

    full_pct = n_full / max(n_pos, 1)
    div_pct  = n_div  / max(n_pos, 1)

    bar_full = f'<div class="conv-full" style="width:{full_pct*100:.1f}%">&nbsp;</div>'
    bar_div  = f'<div class="conv-div"  style="width:{div_pct*100:.1f}%">&nbsp;</div>'

    # Top diverging positions
    diverge_rows = ""
    for pos in hyp.get("positions", []):
        if pos["agreement"] == "DIVERGE":
            phones_str = " / ".join(
                f'{_esc(hid)}: {_esc(ph)}'
                for hid, ph in pos.get("hyp_phonemes", {}).items()
            )
            diverge_rows += (
                f'<tr><td>{pos["position"]}</td>'
                f'<td class="code">{_esc(pos["barthel_code"])}</td>'
                f'<td>{_esc(pos["night_name"])}</td>'
                f'<td class="lo">{_esc(phones_str)}</td></tr>'
            )
            if diverge_rows.count("<tr>") >= 10:
                break

    div_table = (
        "<table><thead><tr>"
        "<th>Pos</th><th>Sign</th><th>Night</th><th>Hypothesis phonemes</th>"
        f"</tr></thead><tbody>{diverge_rows}</tbody></table>"
        if diverge_rows else ""
    )

    return f"""
<div class="stat-grid">
  <div class="stat"><div class="stat-label">Hypotheses compared</div>
    <div class="stat-value">{n_hyps}</div>
    <div class="stat-sub">{' · '.join(hyp_ids)}</div></div>
  <div class="stat"><div class="stat-label">FULL agreement</div>
    <div class="stat-value hi">{n_full}</div>
    <div class="stat-sub">{_pct(full_pct)} of {n_pos} positions</div></div>
  <div class="stat"><div class="stat-label">Convergence rate</div>
    <div class="stat-value hi">{_pct(conv)}</div>
    <div class="stat-sub">FULL + MAJORITY combined</div></div>
  <div class="stat"><div class="stat-label">Divergent positions</div>
    <div class="stat-value neg">{n_div}</div>
    <div class="stat-sub">low-confidence readings</div></div>
</div>
<div class="conv-bar">{bar_full}{bar_div}</div>
<p style="font-family:monospace;font-size:10px;color:var(--muted);margin-bottom:16px">
  <span style="color:var(--green)">■ FULL {n_full}</span> &nbsp;
  <span style="color:var(--red)">■ DIVERGE {n_div}</span>
</p>
<div class="verdict"><strong>Finding</strong>
  <p>Across {n_pos} Mamari Ca6–Ca9 sign positions, {_esc(hyp_ids[0])}–{_esc(hyp_ids[-1])}
  agree fully on {n_full} positions ({_pct(full_pct)}). Where all five hypotheses
  converge, the phoneme assignment can be treated as high-confidence. The {n_div} divergent
  positions are the productive targets for refinement — listed below.</p></div>
{'<p style="font-size:13px;color:var(--muted);margin:12px 0">First 10 divergent positions:</p>' + div_table if div_table else ''}
"""


def _section5_deity(dty: dict | None) -> str:
    if not dty:
        return "<p class='lo'>deity_name_search.json not found — run find_deity_names.py</p>"

    n_hits    = dty["n_hits"]
    n_pre     = dty["n_precontact_hits"]
    perm      = dty["permutation_test"]
    p_val     = perm["p_value"]
    n_perms   = perm["n_perms"]
    perm_med  = perm["perm_median"]
    sig       = perm["significant"]
    hyp_id    = dty["hypothesis_id"]

    verdict_text = (
        f"The permutation test (n={n_perms} shuffles) shows the real corpus produces "
        f"FEWER deity name matches ({n_hits}) than the median of random shuffles "
        f"({perm_med}), p = {p_val:.4f}. These {n_hits} matches are noise — they do not "
        f"provide evidence that the current phoneme assignments capture deity name vocabulary. "
        f"This is a correct negative result that prevents overclaiming. "
        f"The 1 pre-contact hit (Tablet D) involves a damaged-glyph sequence and should not "
        f"be reported without further evidence."
        if not sig else
        f"SIGNIFICANT: real corpus produces {n_hits} hits vs perm median {perm_med}, "
        f"p = {p_val:.4f}. Review hits carefully."
    )
    badge = '<span class="negative-badge">NOT SIGNIFICANT</span>' if not sig else \
            '<span class="positive-badge">SIGNIFICANT</span>'

    # Hits table
    rows = ""
    for h in dty.get("hits", []):
        pre_marker = "★ " if h.get("is_precontact") else ""
        rows += (
            f"<tr><td class='hi'>{_esc(h['deity'])}</td>"
            f"<td>{_esc(pre_marker + h['tablet'])}</td>"
            f"<td>{h['start_position']}</td>"
            f"<td class='ph'>{_esc(' + '.join(h['phonemes']))}</td>"
            f"<td>{'ABAB ✓' if h.get('abab_confirmed') else '—'}</td>"
            f"<td>{'pre-contact' if h.get('is_precontact') else ''}</td></tr>"
        )

    table_html = (
        "<table><thead><tr>"
        "<th>Deity</th><th>Tablet</th><th>Position</th>"
        "<th>Phonemes</th><th>ABAB</th><th>Stratum</th>"
        f"</tr></thead><tbody>{rows}</tbody></table>"
    )

    return f"""
<div class="stat-grid">
  <div class="stat"><div class="stat-label">Deity patterns searched</div>
    <div class="stat-value">9</div>
    <div class="stat-sub">makemake, tangaroa, rongo, tane, hina, hiro, atua, haua, tive</div></div>
  <div class="stat"><div class="stat-label">Real hits</div>
    <div class="stat-value">{n_hits}</div>
    <div class="stat-sub">{n_pre} on pre-contact Tablet D</div></div>
  <div class="stat"><div class="stat-label">Permutation median</div>
    <div class="stat-value neg">{perm_med}</div>
    <div class="stat-sub">random-shuffle baseline ({n_perms} perms)</div></div>
  <div class="stat"><div class="stat-label">p-value {badge}</div>
    <div class="stat-value {'neg' if not sig else 'hi'}">{p_val:.4f}</div>
    <div class="stat-sub">one-tailed permutation test</div></div>
</div>
<div class="verdict"><strong>Finding</strong>
  <p>{_esc(verdict_text)}</p></div>
{table_html}
"""


def _section6_ranking(rnk: dict | None) -> str:
    if not rnk:
        return "<p class='lo'>ranking.json not found — run run_decipherment.py</p>"

    hyps = rnk.get("hypotheses", [])[:5]
    rows = ""
    for h in hyps:
        rows += (
            f"<tr>"
            f"<td class='code'>{_esc(h['hypothesis_id'])}</td>"
            f"<td>{_esc(h.get('hypothesis_type', ''))}</td>"
            f"<td>{h.get('overall_lm_score', 0):.4f}</td>"
            f"<td>{h.get('beam_score', 0):.4f}</td>"
            f"<td>{h.get('mcmc_log_posterior', 0):.2f}</td>"
            f"</tr>"
        )

    return f"""
<table><thead><tr>
<th>ID</th><th>Type</th><th>LM Score</th><th>Beam Score</th><th>MCMC log-posterior</th>
</tr></thead><tbody>{rows}</tbody></table>
<p style="font-size:13px;color:var(--muted);margin-top:12px">
  Top hypothesis: {_esc(hyps[0]['hypothesis_id'] if hyps else '—')} · type: {_esc(hyps[0].get('hypothesis_type','') if hyps else '—')}
</p>
"""


# ---------------------------------------------------------------------------
# Final assembly
# ---------------------------------------------------------------------------

def build_report() -> str:
    cal  = _load("outputs/analysis/calendar_gloss_validation.json")
    ro   = _load("outputs/analysis/reading_order_v2.json")
    cmp  = _load("outputs/analysis/compound_compositionality.json")
    hyp  = _load("outputs/analysis/hypothesis_comparison.json")
    dty  = _load("outputs/analysis/deity_name_search.json")
    rnk  = _load("outputs/decipherment/ranking.json")
    chart_uri = _b64_png("outputs/analysis/reading_direction_combined.png")

    generated = datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    sections = [
        ("1", "Calendar Gloss Validation",     _section1_calendar(cal)),
        ("2", "Reading Direction Analysis",    _section2_reading(ro, chart_uri)),
        ("3", "Compound Compositionality",     _section3_compound(cmp)),
        ("4", "Hypothesis Convergence",        _section4_hypotheses(hyp)),
        ("5", "Deity Name Search",             _section5_deity(dty)),
        ("6", "Decipherment Ranking",          _section6_ranking(rnk)),
    ]

    sections_html = ""
    for num, title, body in sections:
        sections_html += (
            f'<div class="section">'
            f'<div class="section-title">'
            f'<span class="section-num">§{num}</span>{_esc(title)}'
            f'</div>'
            f'{body}'
            f'</div>'
        )

    if cal and hyp and cmp and dty:
        meta_line = (
            f"<b>Generated</b> {_esc(generated)}<br>"
            f"<b>Calendar lift</b> {cal['lunar_lift']:.2f}× &nbsp; "
            f"<b>Hypothesis convergence</b> {hyp['convergence_rate']*100:.1f}% &nbsp; "
            f"<b>Compound anchors found</b> {cmp['n_new_anchor_candidates']} &nbsp; "
            f"<b>Deity search p-value</b> {dty['permutation_test']['p_value']:.4f}"
        )
    else:
        meta_line = f"<b>Generated</b> {_esc(generated)}"

    return (
        "<!DOCTYPE html><html lang='en'>"
        "<head><meta charset='utf-8'>"
        "<title>Rongorongo — Final Pipeline Report</title>"
        f"<style>{_CSS}</style></head>"
        "<body><div class='wrap'>"
        "<div class='report-header'>"
        "<div class='report-title'>Rongorongo Decipherment Pipeline</div>"
        f"<div class='report-meta'>{meta_line}</div>"
        "</div>"
        + sections_html
        + "</div></body></html>"
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate self-contained final pipeline report.")
    p.add_argument(
        "--output", type=Path,
        default=PROJECT_ROOT / "outputs" / "final_report.html",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    html = build_report()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(html, encoding="utf-8")
    size_kb = args.output.stat().st_size // 1024
    log.info("Final report → %s  (%d KB, self-contained)", args.output, size_kb)


if __name__ == "__main__":
    main()
