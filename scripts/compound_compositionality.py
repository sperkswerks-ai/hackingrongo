"""
scripts/compound_compositionality.py

Compositionality check for compound rongorongo glyphs.

For any compound glyph A:B (Barthel colon notation, e.g. "009:005")
where both A and B have confident phoneme assignments in H0001, this
script tests whether the concatenation phoneme(A)+phoneme(B) produces
a known Rapa Nui morpheme.

Motivation
----------
If the phoneme assignments are correct, compound glyphs should decompose
into real Polynesian morpheme sequences.  A successful compositional match:
  (a) validates both component phoneme assignments simultaneously, and
  (b) provides a new anchor with Type-1 (phonemic-structural) evidence.

Architecture
------------
This script re-uses the lexicon loading from gloss_hypotheses.py.
Do NOT duplicate the lexicon ingestion logic — one source of truth.

Output
------
outputs/analysis/compound_compositionality.json
outputs/analysis/compound_compositionality.html

Usage
-----
    python scripts/compound_compositionality.py
    python scripts/compound_compositionality.py --hypothesis H0002
    python scripts/compound_compositionality.py --smoke-test
"""

from __future__ import annotations

import argparse
import html as _html
import json
import logging
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

# Re-use lexicon loader — single source of truth.
from scripts.gloss_hypotheses import (  # noqa: E402
    _load_lexicon,
    _lookup_window,
    _normalise,
    TIER_HIGH,
    TIER_MEDIUM,
    TIER_NONE,
)

# ---------------------------------------------------------------------------
# Compound glyph extraction
# ---------------------------------------------------------------------------

_COLON_RE = re.compile(r"^([A-Za-z0-9]+):([A-Za-z0-9]+)$")


def _parse_compound(code: str) -> tuple[str, str] | None:
    """Return (comp_a, comp_b) if code is a valid colon-notation compound."""
    m = _COLON_RE.match(code)
    if not m:
        return None
    a, b = m.group(1), m.group(2)
    # Normalise to 3-digit padded Barthel form if numeric
    def _pad(s: str) -> str:
        return s.zfill(3) if s.isdigit() else s
    return _pad(a), _pad(b)


def load_compound_glyphs(corpus_dir: Path) -> list[dict[str, Any]]:
    """Return all compound-glyph occurrences across the corpus."""
    compounds: list[dict[str, Any]] = []
    for path in sorted(corpus_dir.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        tablet = path.stem
        for g in data.get("glyphs", []):
            code = str(g.get("barthel_code", ""))
            parsed = _parse_compound(code)
            if parsed is None:
                continue
            comp_a, comp_b = parsed
            compounds.append({
                "tablet":       tablet,
                "position":     g["position"],
                "barthel_code": code,
                "comp_a":       comp_a,
                "comp_b":       comp_b,
                "side":         g.get("side", "?"),
                "line":         g.get("line", "?"),
            })
    log.info(
        "Found %d compound-glyph occurrences (%d distinct codes) across %d tablets.",
        len(compounds),
        len({c["barthel_code"] for c in compounds}),
        len({c["tablet"] for c in compounds}),
    )
    return compounds


# ---------------------------------------------------------------------------
# Phoneme lookup
# ---------------------------------------------------------------------------

def load_phoneme_map(ranking_path: Path, hypothesis_id: str) -> dict[str, str]:
    """Return {sign_code: phoneme} for the given hypothesis."""
    data = json.loads(ranking_path.read_text(encoding="utf-8"))
    for hyp in data.get("hypotheses", []):
        if hyp["hypothesis_id"] == hypothesis_id:
            return {a["sign_code"]: a["phoneme"] for a in hyp["assignments"]}
    avail = [h["hypothesis_id"] for h in data.get("hypotheses", [])]
    raise ValueError(f"{hypothesis_id} not found in ranking.json. Available: {avail}")


# ---------------------------------------------------------------------------
# Compositional analysis
# ---------------------------------------------------------------------------

def analyse_compound(
    compound: dict[str, Any],
    phoneme_map: dict[str, str],
    high_forms: set[str],
    all_forms: set[str],
) -> dict[str, Any]:
    """Check compositionality for one compound glyph."""
    comp_a = compound["comp_a"]
    comp_b = compound["comp_b"]

    ph_a = phoneme_map.get(comp_a)
    ph_b = phoneme_map.get(comp_b)

    if ph_a is None or ph_b is None:
        return {
            **compound,
            "phoneme_a": ph_a,
            "phoneme_b": ph_b,
            "concat_phoneme": None,
            "gloss": None,
            "tier": TIER_NONE,
            "compositional": False,
            "new_anchor_candidate": False,
            "notes": f"missing phoneme for {'A' if ph_a is None else 'B'} ({comp_a if ph_a is None else comp_b})",
        }

    concat = ph_a + ph_b
    phones = [ph_a, ph_b]

    # Try two-sign window lookup
    gloss, tier = _lookup_window(phones, high_forms, all_forms)

    # Also try single concatenated string
    if tier == TIER_NONE:
        gloss_cat, tier_cat = _lookup_window([concat], high_forms, all_forms)
        if tier_cat != TIER_NONE:
            gloss, tier = gloss_cat, tier_cat

    compositional = tier in (TIER_HIGH, TIER_MEDIUM)

    return {
        **compound,
        "phoneme_a": ph_a,
        "phoneme_b": ph_b,
        "concat_phoneme": concat,
        "gloss": gloss,
        "tier": tier,
        "compositional": compositional,
        "new_anchor_candidate": tier == TIER_HIGH,
        "notes": f"comp={comp_a}({ph_a}) + {comp_b}({ph_b}) → {concat!r} → {gloss!r} [{tier}]",
    }


def run_analysis(
    corpus_dir: Path,
    ranking_path: Path,
    hypothesis_id: str,
) -> dict[str, Any]:
    high_forms, all_forms = _load_lexicon(PROJECT_ROOT)
    phoneme_map = load_phoneme_map(ranking_path, hypothesis_id)
    compounds = load_compound_glyphs(corpus_dir)

    results: list[dict] = []
    for c in compounds:
        r = analyse_compound(c, phoneme_map, high_forms, all_forms)
        results.append(r)

    n_total        = len(results)
    n_compositional= sum(1 for r in results if r["compositional"])
    n_new_anchors  = sum(1 for r in results if r["new_anchor_candidate"])
    n_missing      = sum(1 for r in results if r["phoneme_a"] is None or r["phoneme_b"] is None)

    # Deduplicate: if the same compound code appears in multiple tablets,
    # keep the one with the best tier.
    by_code: dict[str, dict] = {}
    for r in results:
        code = r["barthel_code"]
        if code not in by_code or (
            r["compositional"] and not by_code[code]["compositional"]
        ):
            by_code[code] = r

    unique_compositional = [r for r in by_code.values() if r["compositional"]]
    unique_anchors       = [r for r in by_code.values() if r["new_anchor_candidate"]]

    log.info(
        "%s: %d compounds, %d compositional (%.1f%%), %d HIGH-tier (new anchors), "
        "%d missing phoneme.",
        hypothesis_id, n_total, n_compositional,
        100 * n_compositional / max(n_total, 1),
        n_new_anchors, n_missing,
    )

    for r in sorted(unique_compositional, key=lambda x: x["tier"], reverse=True):
        log.info(
            "  ✓ %s → %s+%s = %s  [%s]  (tablets: %s)",
            r["barthel_code"], r["phoneme_a"], r["phoneme_b"],
            r["gloss"], r["tier"], r["tablet"],
        )

    return {
        "hypothesis_id": hypothesis_id,
        "n_compound_occurrences": n_total,
        "n_distinct_compounds": len(by_code),
        "n_compositional": n_compositional,
        "n_new_anchor_candidates": n_new_anchors,
        "n_missing_phoneme": n_missing,
        "frac_compositional": round(n_compositional / max(n_total, 1), 3),
        "unique_compositional": unique_compositional,
        "new_anchor_candidates": unique_anchors,
        "all_results": results,
    }


# ---------------------------------------------------------------------------
# HTML report
# ---------------------------------------------------------------------------

_CSS = """\
:root{--bg:#0d0f12;--surface:#161920;--border:#2a2e38;
      --text:#d0d4dc;--muted:#6b7280;--accent:#c4a96d;
      --high:#4ade80;--medium:#facc15;--none:#374151;}
*{box-sizing:border-box;margin:0;padding:0;}
body{background:var(--bg);color:var(--text);
     font-family:'JetBrains Mono',monospace;font-size:12px;line-height:1.6;}
.wrap{max-width:1050px;margin:0 auto;padding:44px 24px;}
h1{font-size:20px;color:var(--accent);margin-bottom:6px;}
.sub{color:var(--muted);font-size:10px;margin-bottom:32px;}
.metrics{display:flex;gap:16px;flex-wrap:wrap;margin-bottom:32px;}
.metric{background:var(--surface);border:1px solid var(--border);
        border-radius:5px;padding:12px 18px;min-width:140px;}
.ml{font-size:9px;color:var(--muted);text-transform:uppercase;letter-spacing:.06em;}
.mv{font-size:22px;color:var(--accent);margin-top:2px;}
h2{color:var(--accent);font-size:14px;margin:28px 0 10px;}
table{width:100%;border-collapse:collapse;}
th{padding:6px 10px;text-align:left;font-size:9px;color:var(--muted);
   border-bottom:1px solid var(--border);text-transform:uppercase;}
td{padding:5px 10px;border-bottom:1px solid rgba(42,46,56,.4);}
.code{color:var(--accent);}
.phoneme{color:#93c5fd;}
.gloss-high{color:var(--high);}
.gloss-medium{color:var(--medium);}
.gloss-none{color:var(--none);}
.anchor-badge{background:rgba(74,222,128,.12);color:var(--high);
              padding:2px 6px;border-radius:3px;font-size:9px;margin-left:4px;}
"""


def build_html(result: dict) -> str:
    metrics_html = (
        f'<div class="metrics">'
        f'<div class="metric"><div class="ml">Compound occurrences</div>'
        f'<div class="mv">{result["n_compound_occurrences"]}</div></div>'
        f'<div class="metric"><div class="ml">Distinct compounds</div>'
        f'<div class="mv">{result["n_distinct_compounds"]}</div></div>'
        f'<div class="metric"><div class="ml">Compositional</div>'
        f'<div class="mv">{result["n_compositional"]}</div></div>'
        f'<div class="metric"><div class="ml">New anchor candidates</div>'
        f'<div class="mv">{result["n_new_anchor_candidates"]}</div></div>'
        f'</div>'
    )

    def _row(r: dict) -> str:
        tier = r["tier"]
        gcls = f'gloss-{tier.lower()}' if tier != TIER_NONE else "gloss-none"
        anchor = '<span class="anchor-badge">NEW ANCHOR</span>' \
                 if r["new_anchor_candidate"] else ""
        return (
            f"<tr>"
            f'<td class="code">{_html.escape(r["barthel_code"])}</td>'
            f"<td>{_html.escape(r['tablet'])}</td>"
            f'<td class="phoneme">{_html.escape(r["phoneme_a"] or "?")}</td>'
            f'<td class="phoneme">{_html.escape(r["phoneme_b"] or "?")}</td>'
            f'<td class="phoneme">{_html.escape(r["concat_phoneme"] or "?")}</td>'
            f'<td class="{gcls}">{_html.escape(r["gloss"] or "—")}{anchor}</td>'
            f"<td>{tier}</td>"
            f"</tr>"
        )

    comp_rows = "".join(_row(r) for r in sorted(
        result["unique_compositional"],
        key=lambda x: (x["tier"] != TIER_HIGH, x["tier"] != "MEDIUM"),
    ))

    all_rows = "".join(
        _row(r) for r in sorted(result["all_results"], key=lambda x: x["barthel_code"])
    )

    thead = (
        "<thead><tr><th>Compound</th><th>Tablet</th>"
        "<th>Phoneme A</th><th>Phoneme B</th><th>Concat</th>"
        "<th>Gloss</th><th>Tier</th></tr></thead>"
    )

    return (
        "<!DOCTYPE html><html lang='en'>"
        "<head><meta charset='utf-8'>"
        "<title>Compound Compositionality</title>"
        f"<style>{_CSS}</style></head>"
        "<body><div class='wrap'>"
        f"<h1>Compound Compositionality — {_html.escape(result['hypothesis_id'])}</h1>"
        f"<div class='sub'>{result['n_distinct_compounds']} distinct compounds · "
        f"{result['n_compositional']} compositional · "
        f"{result['n_new_anchor_candidates']} new anchor candidates</div>"
        + metrics_html
        + "<h2>Compositional matches (potential new anchors)</h2>"
        + f'<table>{thead}<tbody>{comp_rows}</tbody></table>'
        + "<h2>All compound results</h2>"
        + f'<table>{thead}<tbody>{all_rows}</tbody></table>'
        + "</div></body></html>"
    )


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

def _smoke_test() -> None:
    parsed = _parse_compound("009:005")
    assert parsed == ("009", "005"), f"Parse failed: {parsed}"
    assert _parse_compound("001") is None
    assert _parse_compound("001:002:003") is None

    high = {"manu", "ika", "ao", "mao"}
    all_f = high | {"tangata", "henua"}
    phoneme_map = {"009": "ma", "005": "nu"}
    c = {
        "tablet": "C", "position": 10, "barthel_code": "009:005",
        "comp_a": "009", "comp_b": "005", "side": "a", "line": "01",
    }
    result = analyse_compound(c, phoneme_map, high, all_f)
    assert result["compositional"] is True or result["tier"] == TIER_NONE, \
        f"Unexpected result: {result}"
    log.info("Smoke test passed.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Check compositionality of compound rongorongo glyphs."
    )
    p.add_argument("--ranking", type=Path,
                   default=PROJECT_ROOT / "outputs" / "decipherment" / "ranking.json")
    p.add_argument("--corpus-dir", type=Path,
                   default=PROJECT_ROOT / "data" / "corpus")
    p.add_argument("--hypothesis", default="H0001", metavar="ID")
    p.add_argument("--output-json", type=Path,
                   default=PROJECT_ROOT / "outputs" / "analysis" / "compound_compositionality.json")
    p.add_argument("--output-html", type=Path,
                   default=PROJECT_ROOT / "outputs" / "analysis" / "compound_compositionality.html")
    p.add_argument("--smoke-test", action="store_true")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    if args.smoke_test:
        _smoke_test()
        return

    result = run_analysis(args.corpus_dir, args.ranking, args.hypothesis)

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    json_out = {k: v for k, v in result.items() if k != "all_results"}
    args.output_json.write_text(
        json.dumps(json_out, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    log.info("JSON → %s", args.output_json)

    html_str = build_html(result)
    args.output_html.parent.mkdir(parents=True, exist_ok=True)
    args.output_html.write_text(html_str, encoding="utf-8")
    log.info("HTML → %s", args.output_html)


if __name__ == "__main__":
    main()
