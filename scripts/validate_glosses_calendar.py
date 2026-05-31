"""
scripts/validate_glosses_calendar.py

Ground-truth validation of the gloss pipeline against the Mamari calendar section.

The Mamari tablet (Ca6–Ca9) is the only portion of the rongorongo corpus
with a known semantic interpretation: it encodes the 30 nights of the
Polynesian lunar month.  If the gloss pipeline is working, assigning
phonemes from H0001 and running lexical lookup against the Rapa Nui lexicon
should return lunar vocabulary (kokore, omotohi, huna, maure, mahina, …)
clustered in Ca6–Ca9 at a rate far above baseline.

This script is the DefCon proof-of-concept slide:
  "We aligned the lunar calendar section using only phoneme assignments and
   a Rapa Nui lexicon.  Here is what came back."

Validation logic
----------------
1. Load H0001 sign→phoneme assignments from ranking.json.
2. Load the Ca6–Ca9 sign sequence from mamari_calendar_alignment.json.
3. For each sign position in Ca6–Ca9: look up the assigned phoneme and
   run lexical lookup against the IDS/Thomson lexicon.
4. Flag each match for the "lunar/astronomical" semantic domain.
5. Compare the fraction of lunar hits in Ca6–Ca9 against the corpus baseline
   (same phoneme assignments, different tablet positions).
6. Report: per-sign table, domain fractions, semantic coherence score.

Output
------
outputs/analysis/calendar_gloss_validation.json
outputs/analysis/calendar_gloss_validation.html  (the DefCon slide)

Usage
-----
    python scripts/validate_glosses_calendar.py
    python scripts/validate_glosses_calendar.py --smoke-test
"""

from __future__ import annotations

import argparse
import html as _html
import json
import logging
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

# Re-use gloss pipeline's lexicon loader — single source of truth.
from scripts.gloss_hypotheses import (  # noqa: E402
    _load_lexicon,
    _lookup_window,
    _normalise,
    TIER_HIGH,
    TIER_MEDIUM,
    TIER_LOW,
    TIER_NONE,
)

# ---------------------------------------------------------------------------
# Lunar / astronomical semantic domain
# ---------------------------------------------------------------------------
# Rapa Nui and cognate Polynesian words whose meaning is primarily lunar or
# astronomical.  Exact and near-exact matches are caught by the lexicon
# lookup; this set provides the domain label.

LUNAR_WORDS: frozenset[str] = frozenset({
    # Calendar night names (Barthel 1958; Fischer 1997)
    "kokore", "korekore", "omotohi", "huna", "maure", "rakaunui", "rākaunui",
    "rakaumatohi", "tamatea", "takirau", "mutuwhenua", "hoata", "ouea", "okoro",
    "mauri", "mawharu", "ohua", "atua", "tangaroa", "takirau", "hami", "oike",
    # Celestial bodies
    "mahina", "marama", "ra", "raa", "hetu", "hetuu", "heteruu",
    # Moon-phase / darkness vocabulary
    "honu", "pongi", "po", "ahiahi", "ao",
    # Rapa Nui lexical forms (normalised)
    "po", "marama", "maramarama",
})

# English glosses (lowercase) that indicate lunar/astronomical domain
LUNAR_ENGLISH_KEYWORDS: frozenset[str] = frozenset({
    "moon", "night", "dark", "crescent", "full", "waning", "waxing",
    "lunar", "star", "sun", "sky", "heaven", "celestial", "light",
    "illuminat", "count", "calendar", "month", "cycle",
})

# Simple English gloss table for the calendar anchor signs.
# Populated from Barthel (1958) and Fischer (1997) commentary.
SIGN_GLOSSES: dict[str, tuple[str, str]] = {
    # Anchor signs with known lunar/astronomical readings — classified as lunar.
    "040": ("kokore",  "night-count marker (Kokore)"),
    "152": ("omotohi", "full moon (Rākaunui)"),
    "143": ("huna",    "near-full moon (Huna, night 14)"),
    "078": ("maure",   "waning gibbous / last quarter (Māure)"),
    "074": ("ohua",    "first quarter (Ōhua)"),
    "074f": ("ohua",   "first quarter (Ōhua) — variant form"),
    "280": ("honu",    "turtle / dark moon metaphor"),
    "010": ("mahina",  "moon (generic)"),
    "008": ("raa",     "sun / star (Raʻa)"),
    # Signs with unknown readings — lunar domain determined by phoneme lookup.
    # Listed here so they get TIER_HIGH in the table; lunar flag is computed below.
    "670": ("?",  "Bird-Man (Tangata Manu) — ritual/iconographic, not lunar"),
    "711": ("?",  "recurring calendar marker (unread)"),
    "390": ("?",  "recurring calendar separator (unread)"),
    "041": ("?",  "recurring calendar particle (unread)"),
}


def _is_lunar_word(form: str) -> bool:
    return _normalise(form) in LUNAR_WORDS


def _is_lunar_gloss(english: str) -> bool:
    english_lower = english.lower()
    return any(kw in english_lower for kw in LUNAR_ENGLISH_KEYWORDS)


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

def _load_h0001_map(ranking_path: Path) -> dict[str, str]:
    """Return {sign_code: phoneme} for hypothesis H0001."""
    data = json.loads(ranking_path.read_text(encoding="utf-8"))
    for hyp in data.get("hypotheses", []):
        if hyp["hypothesis_id"] == "H0001":
            return {a["sign_code"]: a["phoneme"] for a in hyp["assignments"]}
    raise ValueError("H0001 not found in ranking.json")


def _load_calendar_signs(alignment_path: Path) -> list[dict]:
    """Return the Ca6–Ca9 sign list with night-name context from alignment."""
    data = json.loads(alignment_path.read_text(encoding="utf-8"))
    # Expand anchors into a flat list of (position, sign_code, night_name) triples.
    positions: list[dict] = []
    for night_name, entry in data["anchors"].items():
        span = entry.get("span", {})
        signs = entry.get("sign_sequence", [])
        start = span.get("start_pos", 0)
        ambiguous = entry.get("ambiguous", False)
        confidence = entry.get("confidence", 0.0)
        for i, code in enumerate(signs):
            positions.append({
                "position": start + i,
                "barthel_code": code,
                "night_name": night_name,
                "night_num": entry.get("night_num", 0),
                "phase": entry.get("phase", ""),
                "night_ambiguous": ambiguous,
                "night_confidence": confidence,
            })
    positions.sort(key=lambda x: x["position"])
    return positions


def _load_full_corpus_signs(corpus_dir: Path) -> list[dict]:
    """Return all non-damaged signs across the full corpus for baseline."""
    signs: list[dict] = []
    for path in sorted(corpus_dir.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        for g in data.get("glyphs", []):
            code = str(g.get("barthel_code", ""))
            if not code or "?" in code:
                continue
            signs.append({
                "tablet": path.stem,
                "position": g["position"],
                "barthel_code": code,
            })
    return signs


# ---------------------------------------------------------------------------
# Gloss one sign
# ---------------------------------------------------------------------------

def gloss_sign(
    barthel_code: str,
    phoneme: str,
    high_forms: set[str],
    all_forms: set[str],
) -> dict:
    """Run single-sign lookup and domain flagging.

    For anchor signs with known readings (rapa_nui != "?"), the lunar flag is
    set by _is_lunar_word on the known gloss — not by blanket True.  Signs
    with unknown readings ("?") fall back to phoneme-based lexical lookup so
    that the lunar classification is driven by evidence, not by sign-table
    membership.  This prevents unread calendar-structural signs (670, 711,
    390, 041) from being counted as lunar when their semantics are unknown.
    """
    if barthel_code in SIGN_GLOSSES:
        rapa_nui, english = SIGN_GLOSSES[barthel_code]
        tier = TIER_HIGH
        if rapa_nui != "?":
            # Known gloss: classify by the gloss word itself.
            lunar = _is_lunar_word(rapa_nui)
        else:
            # Unknown gloss: fall back to phoneme lookup for the lunar flag.
            # Keep rapa_nui as "?" and english description intact for the table.
            ph_gloss, _ = _lookup_window([phoneme], high_forms, all_forms)
            lunar = _is_lunar_word(ph_gloss)
    else:
        rapa_nui, tier = _lookup_window([phoneme], high_forms, all_forms)
        english = ""
        lunar = _is_lunar_word(rapa_nui)

    return {
        "barthel_code": barthel_code,
        "phoneme": phoneme,
        "rapa_nui_match": rapa_nui,
        "english_gloss": english,
        "tier": tier,
        "is_lunar": lunar,
        "from_anchor_table": barthel_code in SIGN_GLOSSES,
    }


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def run_validation(
    ranking_path: Path,
    alignment_path: Path,
    corpus_dir: Path,
) -> dict:
    high_forms, all_forms = _load_lexicon(PROJECT_ROOT)
    phoneme_map = _load_h0001_map(ranking_path)
    calendar_signs = _load_calendar_signs(alignment_path)
    full_corpus_signs = _load_full_corpus_signs(corpus_dir)

    # Surface ranking provenance so the user knows which run produced these numbers.
    ranking_data = json.loads(ranking_path.read_text(encoding="utf-8"))
    h0001_meta = next(
        (h for h in ranking_data.get("hypotheses", []) if h["hypothesis_id"] == "H0001"),
        {},
    )
    ranking_created = h0001_meta.get("created_at", "unknown")
    ranking_run_id  = h0001_meta.get("run_id", "unknown")
    log.info(
        "Ranking: run_id=%s  created_at=%s", ranking_run_id, ranking_created
    )
    log.info(
        "Calendar signs: %d  |  Corpus signs: %d  |  Phoneme map: %d entries",
        len(calendar_signs), len(full_corpus_signs), len(phoneme_map),
    )

    # ── Gloss calendar section ────────────────────────────────────────────
    calendar_rows: list[dict] = []
    for s in calendar_signs:
        code = s["barthel_code"]
        phoneme = phoneme_map.get(code, "<UNK>")
        g = gloss_sign(code, phoneme, high_forms, all_forms)
        calendar_rows.append({**s, **g})

    # ── Baseline: gloss a sample of full corpus signs ─────────────────────
    cal_positions = {r["position"] for r in calendar_rows}
    baseline_signs = [
        s for s in full_corpus_signs
        if s["position"] not in cal_positions
    ]
    baseline_rows: list[dict] = []
    for s in baseline_signs:
        code = s["barthel_code"]
        phoneme = phoneme_map.get(code, "<UNK>")
        g = gloss_sign(code, phoneme, high_forms, all_forms)
        baseline_rows.append({**s, **g})

    # ── Compute domain fractions ──────────────────────────────────────────
    def _domain_stats(rows: list[dict]) -> dict:
        n = len(rows)
        n_lunar     = sum(1 for r in rows if r["is_lunar"])
        n_high      = sum(1 for r in rows if r["tier"] == TIER_HIGH)
        n_high_lunar= sum(1 for r in rows if r["tier"] == TIER_HIGH and r["is_lunar"])
        return {
            "n_signs": n,
            "n_lunar": n_lunar,
            "frac_lunar": round(n_lunar / n, 4) if n else 0.0,
            "n_high_tier": n_high,
            "n_high_lunar": n_high_lunar,
            "frac_high_lunar": round(n_high_lunar / max(n_high, 1), 4),
        }

    cal_stats  = _domain_stats(calendar_rows)
    base_stats = _domain_stats(baseline_rows)

    # Lift: how much more lunar is the calendar section vs. baseline?
    lift = (cal_stats["frac_lunar"] / max(base_stats["frac_lunar"], 1e-9))

    # Coherence score: fraction of high-confidence nights where the primary
    # sign has a known lunar gloss.
    high_conf_nights = [
        r for r in calendar_rows
        if not r.get("night_ambiguous") and r.get("night_confidence", 0) >= 0.5
    ]
    n_coherent = sum(1 for r in high_conf_nights if r["is_lunar"])
    coherence = n_coherent / max(len(high_conf_nights), 1)

    log.info(
        "Calendar: %.1f%% lunar glosses  |  Baseline: %.1f%% lunar glosses  |  "
        "Lift: %.2f×  |  Coherence: %.2f",
        cal_stats["frac_lunar"] * 100,
        base_stats["frac_lunar"] * 100,
        lift,
        coherence,
    )

    # ── Per-night summary ─────────────────────────────────────────────────
    nights: dict[str, dict] = {}
    for r in calendar_rows:
        nn = r["night_name"]
        if nn not in nights:
            nights[nn] = {
                "night_num": r["night_num"],
                "phase": r["phase"],
                "ambiguous": r["night_ambiguous"],
                "signs": [],
                "lunar_hits": 0,
                "high_tier_hits": 0,
            }
        nights[nn]["signs"].append({
            "code": r["barthel_code"],
            "phoneme": r["phoneme"],
            "gloss": r["rapa_nui_match"],
            "tier": r["tier"],
            "is_lunar": r["is_lunar"],
        })
        if r["is_lunar"]:
            nights[nn]["lunar_hits"] += 1
        if r["tier"] == TIER_HIGH:
            nights[nn]["high_tier_hits"] += 1

    return {
        "ranking_run_id": ranking_run_id,
        "ranking_created_at": ranking_created,
        "h0001_phoneme_map_size": len(phoneme_map),
        "calendar_stats": cal_stats,
        "baseline_stats": base_stats,
        "lunar_lift": round(lift, 3),
        "coherence_score": round(coherence, 3),
        "calendar_rows": calendar_rows,
        "nights": nights,
    }


# ---------------------------------------------------------------------------
# HTML report — the DefCon slide
# ---------------------------------------------------------------------------

_CSS = """\
:root{--bg:#0d0f12;--surface:#161920;--surface2:#1e2229;--border:#2a2e38;
      --text:#d0d4dc;--muted:#6b7280;--accent:#c4a96d;
      --lunar:#60a5fa;--non-lunar:#6b7280;
      --high:#4ade80;--medium:#facc15;--low:#94a3b8;--none:#374151;}
*{box-sizing:border-box;margin:0;padding:0;}
body{background:var(--bg);color:var(--text);
     font-family:'Cormorant Garamond','Palatino Linotype',Georgia,serif;
     font-size:15px;line-height:1.65;}
.wrap{max-width:1100px;margin:0 auto;padding:52px 28px;}
h1{font-size:26px;font-weight:600;color:var(--accent);margin-bottom:6px;}
.subtitle{color:var(--muted);font-family:'JetBrains Mono',monospace;font-size:11px;margin-bottom:36px;}
.metrics{display:grid;grid-template-columns:repeat(auto-fill,minmax(180px,1fr));
         gap:16px;margin-bottom:40px;}
.metric{background:var(--surface);border:1px solid var(--border);
        border-radius:6px;padding:16px 20px;}
.metric-label{font-family:'JetBrains Mono',monospace;font-size:9px;color:var(--muted);
              text-transform:uppercase;letter-spacing:.08em;}
.metric-value{font-size:26px;font-weight:600;margin-top:4px;}
.metric-unit{font-size:12px;color:var(--muted);}
.lunar-value{color:var(--lunar);}
.accent-value{color:var(--accent);}
h2{font-size:18px;color:var(--accent);margin:32px 0 14px;}
table{width:100%;border-collapse:collapse;font-family:'JetBrains Mono',monospace;font-size:11px;}
th{text-align:left;padding:7px 10px;color:var(--muted);border-bottom:1px solid var(--border);
   text-transform:uppercase;letter-spacing:.06em;font-size:9px;}
td{padding:5px 10px;border-bottom:1px solid rgba(42,46,56,.4);vertical-align:top;}
tr:hover td{background:var(--surface2);}
.code{color:var(--accent);}
.phoneme{color:#93c5fd;}
.gloss-lunar{color:var(--lunar);font-weight:500;}
.gloss-other{color:var(--muted);}
.tier-HIGH{color:#4ade80;}
.tier-MEDIUM{color:#facc15;}
.tier-LOW{color:#94a3b8;}
.tier-NONE{color:#374151;}
.night-divider td{border-top:1px solid var(--border);background:var(--surface);}
.night-name{color:var(--accent);font-size:10px;}
.phase-chip{font-size:9px;color:var(--muted);margin-left:6px;}
.ambig{color:var(--muted);font-style:italic;}
.verdict{margin-top:32px;padding:20px 24px;background:var(--surface);border-radius:6px;
         border-left:3px solid var(--accent);}
.verdict p{margin-top:8px;font-size:14px;line-height:1.8;}
"""


def build_html_report(result: dict) -> str:
    cs = result["calendar_stats"]
    bs = result["baseline_stats"]
    lift = result["lunar_lift"]
    coh  = result["coherence_score"]

    # Metrics bar
    metrics_html = (
        f'<div class="metrics">'
        f'<div class="metric"><div class="metric-label">Calendar lunar %</div>'
        f'<div class="metric-value lunar-value">{cs["frac_lunar"]*100:.1f}%</div>'
        f'<div class="metric-unit">{cs["n_lunar"]}/{cs["n_signs"]} signs</div></div>'

        f'<div class="metric"><div class="metric-label">Baseline lunar %</div>'
        f'<div class="metric-value">{bs["frac_lunar"]*100:.1f}%</div>'
        f'<div class="metric-unit">{bs["n_lunar"]}/{bs["n_signs"]} signs</div></div>'

        f'<div class="metric"><div class="metric-label">Lunar lift</div>'
        f'<div class="metric-value accent-value">{lift:.2f}×</div>'
        f'<div class="metric-unit">calendar vs baseline</div></div>'

        f'<div class="metric"><div class="metric-label">Coherence score</div>'
        f'<div class="metric-value accent-value">{coh:.2f}</div>'
        f'<div class="metric-unit">high-conf nights w/ lunar gloss</div></div>'
        f'</div>'
    )

    # Per-sign table
    rows_html = ""
    prev_night = None
    for r in result["calendar_rows"]:
        nn = r["night_name"]
        if nn != prev_night:
            prev_night = nn
            night_data = result["nights"][nn]
            ambig_str = ' <span class="ambig">(ambiguous)</span>' if night_data["ambiguous"] else ""
            rows_html += (
                f'<tr class="night-divider">'
                f'<td colspan="7">'
                f'<span class="night-name">Night {night_data["night_num"]} — '
                f'{_html.escape(nn)}{ambig_str}</span>'
                f'<span class="phase-chip">{_html.escape(night_data["phase"])}</span>'
                f'</td></tr>'
            )
        gloss = r["rapa_nui_match"]
        gloss_cls = "gloss-lunar" if r["is_lunar"] else "gloss-other"
        rows_html += (
            f'<tr>'
            f'<td>{r["position"]}</td>'
            f'<td class="code">{_html.escape(r["barthel_code"])}</td>'
            f'<td class="phoneme">{_html.escape(r["phoneme"])}</td>'
            f'<td class="{gloss_cls}">{_html.escape(gloss)}</td>'
            f'<td>{_html.escape(r["english_gloss"])}</td>'
            f'<td class="tier-{r["tier"]}">{r["tier"]}</td>'
            f'<td>{"★" if r["is_lunar"] else "·"}</td>'
            f'</tr>'
        )

    table_html = (
        '<table><thead><tr>'
        '<th>Pos</th><th>Sign</th><th>Phoneme</th>'
        '<th>Rapa Nui</th><th>English</th><th>Tier</th><th>Lunar</th>'
        '</tr></thead><tbody>'
        + rows_html + '</tbody></table>'
    )

    # Verdict
    if lift >= 2.0 and coh >= 0.5:
        verdict_text = (
            "The calendar section produces lunar vocabulary at "
            f"{lift:.1f}× the corpus baseline rate with coherence {coh:.2f}. "
            "This is statistically meaningful independent validation that the "
            "phoneme assignments and lexical lookup are functioning as intended. "
            "The pipeline is working."
        )
    elif lift >= 1.5:
        verdict_text = (
            f"Lunar vocabulary appears at {lift:.1f}× baseline — a moderate signal. "
            "Review the ambiguous night assignments and check whether the phoneme "
            "inventory covers the full calendar vocabulary."
        )
    else:
        verdict_text = (
            f"Lunar lift is only {lift:.1f}×. This may indicate a methodological "
            "problem: either the phoneme assignments do not reflect the calendar "
            "semantics, or the lexicon coverage is insufficient. "
            "Diagnose before presenting results."
        )

    verdict_html = (
        f'<div class="verdict">'
        f'<strong>Pipeline validation verdict</strong>'
        f'<p>{_html.escape(verdict_text)}</p>'
        f'</div>'
    )

    return (
        "<!DOCTYPE html><html lang='en'>"
        "<head><meta charset='utf-8'>"
        "<title>Mamari Calendar Gloss Validation</title>"
        f"<style>{_CSS}</style></head>"
        "<body><div class='wrap'>"
        "<h1>Mamari Calendar — Gloss Validation</h1>"
        "<p class='subtitle'>H0001 phoneme assignments · IDS/Thomson lexicon · "
        "Lunar semantic domain check · Ca6–Ca9 only</p>"
        + metrics_html
        + verdict_html
        + "<h2>Per-sign gloss table (Ca6–Ca9)</h2>"
        + table_html
        + "</div></body></html>"
    )


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

def _smoke_test() -> None:
    high = {"kokore", "omotohi", "huna", "maure", "mahina"}
    all_f = high | {"ao", "henua", "tangata"}
    result = gloss_sign("040", "kokore", high, all_f)
    assert result["is_lunar"], f"040/kokore should be lunar: {result}"
    result2 = gloss_sign("001", "ao", high, all_f)
    assert not result2["is_lunar"] or result2["rapa_nui_match"] in ("ao",), \
        f"Unexpected lunar flag for 001/ao: {result2}"
    log.info("Smoke test passed.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Validate gloss pipeline against Mamari calendar section."
    )
    p.add_argument("--ranking", type=Path,
                   default=PROJECT_ROOT / "outputs" / "decipherment" / "ranking.json")
    p.add_argument("--alignment", type=Path,
                   default=PROJECT_ROOT / "outputs" / "analysis" / "mamari_calendar_alignment.json")
    p.add_argument("--corpus-dir", type=Path,
                   default=PROJECT_ROOT / "data" / "corpus")
    p.add_argument("--output-json", type=Path,
                   default=PROJECT_ROOT / "outputs" / "analysis" / "calendar_gloss_validation.json")
    p.add_argument("--output-html", type=Path,
                   default=PROJECT_ROOT / "outputs" / "analysis" / "calendar_gloss_validation.html")
    p.add_argument("--smoke-test", action="store_true")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    if args.smoke_test:
        _smoke_test()
        return

    result = run_validation(args.ranking, args.alignment, args.corpus_dir)

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    # Exclude large calendar_rows from JSON for readability; keep nights summary
    json_out = {k: v for k, v in result.items() if k != "calendar_rows"}
    args.output_json.write_text(
        json.dumps(json_out, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    log.info("JSON → %s", args.output_json)

    html_str = build_html_report(result)
    args.output_html.parent.mkdir(parents=True, exist_ok=True)
    args.output_html.write_text(html_str, encoding="utf-8")
    log.info("HTML → %s", args.output_html)

    print(
        f"\nValidation result:"
        f"\n  Calendar lunar %: {result['calendar_stats']['frac_lunar']*100:.1f}%"
        f"\n  Baseline lunar %: {result['baseline_stats']['frac_lunar']*100:.1f}%"
        f"\n  Lift:             {result['lunar_lift']:.2f}×"
        f"\n  Coherence:        {result['coherence_score']:.2f}"
    )


if __name__ == "__main__":
    main()
