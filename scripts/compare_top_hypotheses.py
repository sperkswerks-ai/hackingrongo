"""
scripts/compare_top_hypotheses.py

Side-by-side comparison of the top five decipherment hypotheses at each
Mamari calendar (Ca6–Ca9) sign position.

Convergence = signal.  If all five hypotheses assign the same gloss to a
calendar position, that position is a high-confidence reading.  Divergence
flags low-confidence positions worth diagnosing.

Outputs
-------
1. Markdown table: one row per calendar position, columns H0001–H0005.
   Cells show "phoneme (tier)" with AGREE/DIVERGE annotation.

2. HTML report: the same table with colour coding.
   Green cells = all five hypotheses agree on the gloss.
   Yellow cells = majority agree (3–4/5).
   Red cells    = strong disagreement (≤ 2/5 agree).

3. JSON: machine-readable per-position comparison data.

Usage
-----
    python scripts/compare_top_hypotheses.py
    python scripts/compare_top_hypotheses.py --top 3
    python scripts/compare_top_hypotheses.py --smoke-test
"""

from __future__ import annotations

import argparse
import html as _html
import json
import logging
import sys
from collections import Counter
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

from scripts.gloss_hypotheses import (  # noqa: E402
    _load_lexicon,
    _lookup_window,
    TIER_HIGH,
    TIER_MEDIUM,
    TIER_NONE,
)

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_all_phoneme_maps(
    ranking_path: Path,
    top: int = 5,
) -> list[tuple[str, dict[str, str]]]:
    """Return [(hyp_id, {sign_code: phoneme}), ...] for the top hypotheses."""
    data = json.loads(ranking_path.read_text(encoding="utf-8"))
    hyps = data.get("hypotheses", [])[:top]
    return [
        (h["hypothesis_id"], {a["sign_code"]: a["phoneme"] for a in h["assignments"]})
        for h in hyps
    ]


def load_calendar_positions(alignment_path: Path) -> list[dict[str, Any]]:
    """Expand the alignment anchors into a flat per-sign position list."""
    data = json.loads(alignment_path.read_text(encoding="utf-8"))
    positions: list[dict[str, Any]] = []
    for night_name, entry in data["anchors"].items():
        span     = entry.get("span", {})
        signs    = entry.get("sign_sequence", [])
        start    = span.get("start_pos", 0)
        night_num = entry.get("night_num", 0)
        phase    = entry.get("phase", "")
        ambiguous = entry.get("ambiguous", False)
        for i, code in enumerate(signs):
            positions.append({
                "position":    start + i,
                "barthel_code": code,
                "night_name":  night_name,
                "night_num":   night_num,
                "phase":       phase,
                "ambiguous":   ambiguous,
            })
    positions.sort(key=lambda x: x["position"])
    return positions


# ---------------------------------------------------------------------------
# Comparison logic
# ---------------------------------------------------------------------------

def compare_position(
    pos: dict[str, Any],
    hyp_maps: list[tuple[str, dict[str, str]]],
    high_forms: set[str],
    all_forms: set[str],
) -> dict[str, Any]:
    """For one sign position, get phoneme + gloss from every hypothesis."""
    code = pos["barthel_code"]
    cells: list[dict] = []

    for hyp_id, pmap in hyp_maps:
        phoneme = pmap.get(code, "<UNK>")
        gloss, tier = _lookup_window([phoneme], high_forms, all_forms)
        cells.append({"hyp_id": hyp_id, "phoneme": phoneme, "gloss": gloss, "tier": tier})

    # Agreement analysis
    phonemes = [c["phoneme"] for c in cells]
    glosses  = [c["gloss"]   for c in cells]
    ph_counts = Counter(phonemes)
    gl_counts = Counter(glosses)

    top_ph, top_ph_n = ph_counts.most_common(1)[0]
    top_gl, top_gl_n = gl_counts.most_common(1)[0]
    n_hyps = len(cells)

    if top_ph_n == n_hyps:
        agreement = "FULL"
    elif top_ph_n >= (n_hyps * 2 / 3):
        agreement = "MAJORITY"
    else:
        agreement = "DIVERGE"

    return {
        **pos,
        "cells": cells,
        "agreement": agreement,
        "top_phoneme": top_ph,
        "top_phoneme_count": top_ph_n,
        "top_gloss": top_gl,
        "top_gloss_count": top_gl_n,
        "n_hyps": n_hyps,
    }


# ---------------------------------------------------------------------------
# Markdown output
# ---------------------------------------------------------------------------

def build_markdown(
    comparisons: list[dict],
    hyp_ids: list[str],
) -> str:
    lines: list[str] = []
    lines.append("# Hypothesis comparison — Mamari Ca6–Ca9 calendar positions")
    lines.append("")
    lines.append("Columns: sign position | night | " +
                 " | ".join(hyp_ids) + " | Agreement")
    lines.append("")

    # Header
    sep = "|".join(["---"] * (4 + len(hyp_ids)))
    header = "| Pos | Night | " + " | ".join(hyp_ids) + " | Agreement |"
    lines.append(header)
    lines.append("| " + sep + " |")

    prev_night = None
    for c in comparisons:
        nn = c["night_name"]
        if nn != prev_night:
            lines.append(f"| **{nn}** |  | " + " | " * len(hyp_ids) + " |  |")
            prev_night = nn

        cells_str = " | ".join(
            f"{cell['phoneme']} *({cell['tier'][:1]})*"
            for cell in c["cells"]
        )
        agree_sym = {
            "FULL": "✓ AGREE",
            "MAJORITY": "≈ MAJORITY",
            "DIVERGE": "✗ DIVERGE",
        }.get(c["agreement"], c["agreement"])

        lines.append(
            f"| {c['position']} | {c['barthel_code']} | {cells_str} | {agree_sym} |"
        )

    lines.append("")
    n_full     = sum(1 for c in comparisons if c["agreement"] == "FULL")
    n_majority = sum(1 for c in comparisons if c["agreement"] == "MAJORITY")
    n_diverge  = sum(1 for c in comparisons if c["agreement"] == "DIVERGE")
    lines.append(
        f"**Summary**: {len(comparisons)} positions — "
        f"FULL: {n_full}, MAJORITY: {n_majority}, DIVERGE: {n_diverge}"
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# HTML report
# ---------------------------------------------------------------------------

_CSS = """\
:root{--bg:#0d0f12;--surface:#161920;--surface2:#1e2229;--border:#2a2e38;
      --text:#d0d4dc;--muted:#6b7280;--accent:#c4a96d;
      --full:#166534;--full-text:#4ade80;
      --majority:#713f12;--majority-text:#fde68a;
      --diverge:#7f1d1d;--diverge-text:#f87171;}
*{box-sizing:border-box;margin:0;padding:0;}
body{background:var(--bg);color:var(--text);
     font-family:'JetBrains Mono',monospace;font-size:11px;line-height:1.55;}
.wrap{max-width:1200px;margin:0 auto;padding:44px 24px;}
h1{font-size:19px;color:var(--accent);margin-bottom:4px;}
.sub{color:var(--muted);font-size:10px;margin-bottom:28px;}
.legend{display:flex;gap:16px;margin-bottom:20px;font-size:10px;}
.leg{padding:3px 10px;border-radius:3px;}
.leg-full{background:var(--full);color:var(--full-text);}
.leg-majority{background:var(--majority);color:var(--majority-text);}
.leg-diverge{background:var(--diverge);color:var(--diverge-text);}
table{width:100%;border-collapse:collapse;}
th{padding:6px 10px;text-align:left;font-size:9px;color:var(--muted);
   border-bottom:1px solid var(--border);text-transform:uppercase;
   position:sticky;top:0;background:var(--bg);}
td{padding:5px 8px;border-bottom:1px solid rgba(42,46,56,.3);vertical-align:top;}
.code{color:var(--accent);}
.ph{color:#93c5fd;}
.tier-H{color:#4ade80;font-size:9px;}
.tier-M{color:#facc15;font-size:9px;}
.tier-N{color:#374151;font-size:9px;}
.cell-full{background:var(--full);}
.cell-majority{background:var(--majority);}
.cell-diverge{background:var(--diverge);}
.night-row td{background:var(--surface);color:var(--accent);font-weight:600;}
.agree-full{background:var(--full);color:var(--full-text);padding:2px 6px;border-radius:2px;}
.agree-majority{background:var(--majority);color:var(--majority-text);padding:2px 6px;border-radius:2px;}
.agree-diverge{background:var(--diverge);color:var(--diverge-text);padding:2px 6px;border-radius:2px;}
.summary{margin-top:20px;padding:14px 18px;background:var(--surface);border-radius:5px;font-size:11px;}
"""


def _tier_abbr(tier: str) -> str:
    return tier[0] if tier else "N"


def _agree_html(agreement: str) -> str:
    cls = {"FULL": "agree-full", "MAJORITY": "agree-majority", "DIVERGE": "agree-diverge"}.get(
        agreement, ""
    )
    return f'<span class="{cls}">{agreement}</span>'


def build_html(
    comparisons: list[dict],
    hyp_ids: list[str],
) -> str:
    n_full     = sum(1 for c in comparisons if c["agreement"] == "FULL")
    n_majority = sum(1 for c in comparisons if c["agreement"] == "MAJORITY")
    n_diverge  = sum(1 for c in comparisons if c["agreement"] == "DIVERGE")
    n_total    = len(comparisons)

    legend_html = (
        '<div class="legend">'
        '<span class="leg leg-full">FULL agreement — high-confidence reading</span>'
        '<span class="leg leg-majority">MAJORITY (≥2/3) — moderate confidence</span>'
        '<span class="leg leg-diverge">DIVERGE — low confidence</span>'
        "</div>"
    )

    # Table header
    hyp_headers = "".join(f"<th>{_html.escape(h)}</th>" for h in hyp_ids)
    thead = (
        f"<thead><tr><th>Pos</th><th>Sign</th><th>Night</th>"
        f"{hyp_headers}<th>Agreement</th></tr></thead>"
    )

    rows: list[str] = []
    prev_night = None
    for c in comparisons:
        nn = c["night_name"]
        if nn != prev_night:
            prev_night = nn
            night_label = f"Night {c['night_num']} — {nn} ({c['phase']})"
            if c["ambiguous"]:
                night_label += " ⚠ ambiguous alignment"
            n_cols = 4 + len(hyp_ids)
            rows.append(
                f'<tr class="night-row">'
                f'<td colspan="{n_cols}">{_html.escape(night_label)}</td>'
                f'</tr>'
            )

        # Colour each cell by whether this hyp agrees with the majority
        cell_tds = ""
        for cell in c["cells"]:
            agree_with_top = (cell["phoneme"] == c["top_phoneme"])
            row_cls = "" if agree_with_top else (
                "cell-majority" if c["agreement"] == "MAJORITY" else "cell-diverge"
            )
            tier_abbr = _tier_abbr(cell["tier"])
            cell_tds += (
                f'<td class="{row_cls}">'
                f'<span class="ph">{_html.escape(cell["phoneme"])}</span> '
                f'<span class="tier-{tier_abbr}">[{tier_abbr}]</span>'
                f"</td>"
            )

        rows.append(
            f"<tr>"
            f"<td>{c['position']}</td>"
            f'<td class="code">{_html.escape(c["barthel_code"])}</td>'
            f"<td>{_html.escape(nn)}</td>"
            f"{cell_tds}"
            f"<td>{_agree_html(c['agreement'])}</td>"
            f"</tr>"
        )

    summary_html = (
        f'<div class="summary">'
        f"<strong>Convergence summary</strong> — {n_total} sign positions, "
        f"{len(hyp_ids)} hypotheses compared<br>"
        f'<span class="agree-full">FULL {n_full}</span>  '
        f'<span class="agree-majority">MAJORITY {n_majority}</span>  '
        f'<span class="agree-diverge">DIVERGE {n_diverge}</span>  '
        f"| convergence rate: {100*(n_full+n_majority)/max(n_total,1):.1f}%"
        f"</div>"
    )

    return (
        "<!DOCTYPE html><html lang='en'>"
        "<head><meta charset='utf-8'>"
        "<title>Hypothesis Comparison — Mamari Calendar</title>"
        f"<style>{_CSS}</style></head>"
        "<body><div class='wrap'>"
        f"<h1>Top-{len(hyp_ids)} Hypothesis Comparison — Mamari Ca6–Ca9</h1>"
        f"<div class='sub'>Each cell: phoneme assigned by that hypothesis · "
        "[H=HIGH, M=MEDIUM, N=NONE] lexicon tier · "
        "Green rows = all hypotheses agree</div>"
        + legend_html
        + f"<table>{thead}<tbody>"
        + "".join(rows)
        + f"</tbody></table>"
        + summary_html
        + "</div></body></html>"
    )


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

def _smoke_test() -> None:
    hyp_maps = [
        ("H0001", {"040": "kokore", "152": "omotohi", "001": "a"}),
        ("H0002", {"040": "kokore", "152": "omotohi", "001": "b"}),
        ("H0003", {"040": "kokore", "152": "omotohi", "001": "a"}),
    ]
    high = {"kokore", "omotohi"}
    all_f = high | {"ao", "manu"}
    pos = {"position": 1, "barthel_code": "040",
           "night_name": "Korekore-i", "night_num": 26,
           "phase": "old_moon", "ambiguous": False}
    result = compare_position(pos, hyp_maps, high, all_f)
    assert result["agreement"] == "FULL", f"Expected FULL: {result}"
    assert result["top_phoneme"] == "kokore"
    log.info("Smoke test passed.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compare top hypotheses side-by-side at Mamari calendar positions."
    )
    p.add_argument("--ranking", type=Path,
                   default=PROJECT_ROOT / "outputs" / "decipherment" / "ranking.json")
    p.add_argument("--alignment", type=Path,
                   default=PROJECT_ROOT / "outputs" / "analysis" / "mamari_calendar_alignment.json")
    p.add_argument("--top", type=int, default=5, help="Number of hypotheses to compare.")
    p.add_argument("--output-md", type=Path,
                   default=PROJECT_ROOT / "outputs" / "analysis" / "hypothesis_comparison.md")
    p.add_argument("--output-html", type=Path,
                   default=PROJECT_ROOT / "outputs" / "analysis" / "hypothesis_comparison.html")
    p.add_argument("--output-json", type=Path,
                   default=PROJECT_ROOT / "outputs" / "analysis" / "hypothesis_comparison.json")
    p.add_argument("--smoke-test", action="store_true")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    if args.smoke_test:
        _smoke_test()
        return

    high_forms, all_forms = _load_lexicon(PROJECT_ROOT)
    hyp_maps = load_all_phoneme_maps(args.ranking, args.top)
    hyp_ids  = [hid for hid, _ in hyp_maps]
    log.info("Loaded %d hypotheses: %s", len(hyp_maps), hyp_ids)

    positions = load_calendar_positions(args.alignment)
    log.info("Calendar positions: %d signs across 30 night names.", len(positions))

    comparisons = [
        compare_position(pos, hyp_maps, high_forms, all_forms)
        for pos in positions
    ]

    n_full     = sum(1 for c in comparisons if c["agreement"] == "FULL")
    n_majority = sum(1 for c in comparisons if c["agreement"] == "MAJORITY")
    n_diverge  = sum(1 for c in comparisons if c["agreement"] == "DIVERGE")
    log.info(
        "Agreement: FULL=%d, MAJORITY=%d, DIVERGE=%d / %d positions",
        n_full, n_majority, n_diverge, len(comparisons),
    )

    for path, content in [
        (args.output_md,   build_markdown(comparisons, hyp_ids)),
        (args.output_html, build_html(comparisons, hyp_ids)),
    ]:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        log.info("%s → %s", path.suffix.upper(), path)

    json_out = {
        "n_hypotheses": len(hyp_maps),
        "hyp_ids": hyp_ids,
        "n_positions": len(comparisons),
        "n_full_agreement": n_full,
        "n_majority_agreement": n_majority,
        "n_diverge": n_diverge,
        "convergence_rate": round((n_full + n_majority) / max(len(comparisons), 1), 3),
        "positions": [
            {k: v for k, v in c.items() if k != "cells"}
            | {"hyp_phonemes": {cell["hyp_id"]: cell["phoneme"] for cell in c["cells"]},
               "hyp_glosses":  {cell["hyp_id"]: cell["gloss"]   for cell in c["cells"]}}
            for c in comparisons
        ],
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(json_out, indent=2, ensure_ascii=False))
    log.info("JSON → %s", args.output_json)

    print(
        f"\nHypothesis comparison ({len(hyp_ids)} hypotheses, {len(positions)} positions):"
        f"\n  FULL agreement:     {n_full} positions ({100*n_full/max(len(comparisons),1):.1f}%)"
        f"\n  MAJORITY agreement: {n_majority} positions"
        f"\n  DIVERGE:            {n_diverge} positions"
        f"\n  Convergence rate:   {100*(n_full+n_majority)/max(len(comparisons),1):.1f}%"
    )


if __name__ == "__main__":
    main()
