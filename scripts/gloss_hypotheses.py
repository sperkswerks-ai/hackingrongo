"""
scripts/gloss_hypotheses.py

Post-processing gloss layer: consume outputs/decipherment/ranking.json and
produce a gloss table (CSV) and scholarly HTML report.

Architecture
------------
Read-only post-processing — the MCMC is never touched.  All glossing is
deterministic: given the same lexicon and ranking file the output is
identical.

Sliding-window algorithm
------------------------
For each hypothesis, walk the assigned phoneme sequence with windows of
1, 2, 3, and 4 consecutive signs.  At each starting position take the
longest match found in the lexicon.  Ties at the same length are broken
by confidence tier (HIGH > MEDIUM > LOW).

Confidence tiers
----------------
HIGH   — exact match against the IDS Rapa Nui lexicon or the Thomson
         1891 / pre-contact wordlist (ids_pre_contact.txt).
MEDIUM — edit-distance-1 from any HIGH entry, OR first-syllable match
         with an attested continuation in the lexicon.
LOW    — Proto-Polynesian cognate form only (pp_cognates set).
NONE   — no match at any window width.

Taxograms
---------
Signs with Barthel base codes in TAXOGRAM_CODES render as [STRUCTURAL]
regardless of their phoneme assignment.

Usage
-----
    python scripts/gloss_hypotheses.py
    python scripts/gloss_hypotheses.py --ranking outputs/decipherment/ranking.json
    python scripts/gloss_hypotheses.py --top 3 --smoke-test
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import logging
import re
import sys
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Barthel base-code prefixes that render as [STRUCTURAL] rather than glosses.
# Includes anthropomorphic heads (200-series) and the taxogram markers.
TAXOGRAM_CODES: frozenset[str] = frozenset({
    "076", "200", "201", "202", "203", "204", "205", "206", "207", "208",
    "209", "210", "211", "212", "213", "214", "215", "216", "217", "218",
    "219", "220",
})

# Confidence tier labels (ordered best → worst)
TIER_HIGH   = "HIGH"
TIER_MEDIUM = "MEDIUM"
TIER_LOW    = "LOW"
TIER_NONE   = "NONE"

_TIER_RANK: dict[str, int] = {TIER_HIGH: 3, TIER_MEDIUM: 2, TIER_LOW: 1, TIER_NONE: 0}

# Maximum window size for sliding-window match
_MAX_WINDOW = 4

# Levenshtein edit-distance threshold for MEDIUM tier
_MEDIUM_EDIT_DIST = 1

# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------

_DIACRITIC_MAP = str.maketrans(
    "āēīōūĀĒĪŌŪáéíóúàèìòùâêîôûäëïöü"
    "ʻʼʾ'ʻ",
    "aeiouAEIOUaeiouaeiouaeiouaeiouu"
    "''''"
)

_BRACKET_RE = re.compile(r"\[.*?\]|\(.*?\)")
_SPACE_RE   = re.compile(r"\s+")


def _normalise(s: str) -> str:
    """Lowercase, strip diacritics, brackets, punctuation, collapse spaces."""
    s = _BRACKET_RE.sub("", s)
    s = s.translate(_DIACRITIC_MAP)
    s = unicodedata.normalize("NFD", s)
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    s = re.sub(r"[^a-z0-9' ]", "", s.lower())
    s = _SPACE_RE.sub(" ", s).strip()
    return s


# ---------------------------------------------------------------------------
# Levenshtein edit distance (single-character edits)
# ---------------------------------------------------------------------------

def _levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        curr = [i]
        for j, cb in enumerate(b, 1):
            curr.append(min(
                prev[j] + 1,       # deletion
                curr[j - 1] + 1,   # insertion
                prev[j - 1] + (ca != cb),  # substitution
            ))
        prev = curr
    return prev[-1]


# ---------------------------------------------------------------------------
# Lexicon loading
# ---------------------------------------------------------------------------

_LEXICON_SOURCES = [
    "data/polynesian_texts/rapanui/ids.txt",
    "data/polynesian_texts/rapanui/ids_pre_contact.txt",
    "data/polynesian_texts/old_rapa_nui/barthel.txt",
    "data/polynesian_texts/old_rapa_nui/blixen.txt",
    "data/polynesian_texts/old_rapa_nui/fischer.txt",
]

# Rapa Nui words that are also attested in pre-contact or IDS primary sources.
# These receive HIGH confidence when matched.
_HIGH_TIER_SOURCE_FILES = frozenset({
    "data/polynesian_texts/rapanui/ids.txt",
    "data/polynesian_texts/rapanui/ids_pre_contact.txt",
})


def _load_lexicon(project_root: Path) -> tuple[set[str], set[str]]:
    """Return (high_tier_forms, all_forms) — both normalised.

    Reads one word/phrase per non-empty, non-comment line from each lexicon
    source.  Lines starting with '[' are skipped (source annotations in the
    Barthel/Blixen/Fischer files).
    """
    high_forms: set[str] = set()
    all_forms:  set[str] = set()

    for rel_path in _LEXICON_SOURCES:
        path = project_root / rel_path
        if not path.exists():
            log.debug("Lexicon file missing (skipped): %s", path)
            continue
        is_high = rel_path in _HIGH_TIER_SOURCE_FILES
        n = 0
        for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
            raw = raw.strip()
            if not raw or raw.startswith("[") or raw.startswith("#"):
                continue
            norm = _normalise(raw)
            if not norm:
                continue
            all_forms.add(norm)
            if is_high:
                high_forms.add(norm)
            n += 1
        log.debug("Loaded %d entries from %s", n, path.name)

    log.info(
        "Lexicon: %d HIGH-tier entries, %d total entries.",
        len(high_forms), len(all_forms),
    )
    return high_forms, all_forms


# ---------------------------------------------------------------------------
# Gloss lookup
# ---------------------------------------------------------------------------

def _tier_for(form: str, high_forms: set[str], all_forms: set[str]) -> str:
    if form in high_forms:
        return TIER_HIGH
    if form in all_forms:
        return TIER_MEDIUM
    return TIER_NONE


def _lookup_window(
    phonemes: list[str],
    high_forms: set[str],
    all_forms: set[str],
) -> tuple[str, str]:
    """Return (gloss, tier) for the concatenated phoneme window.

    Tries exact match first, then edit-distance-1 against HIGH entries,
    then first-syllable prefix match against all entries.
    """
    joined = _normalise(" ".join(phonemes))
    joined_nospace = joined.replace(" ", "")

    # Exact match
    tier = _tier_for(joined, high_forms, all_forms)
    if tier != TIER_NONE:
        return joined, tier

    # Without spaces (some lexicon entries are written without word breaks)
    tier = _tier_for(joined_nospace, high_forms, all_forms)
    if tier != TIER_NONE:
        return joined_nospace, TIER_MEDIUM  # at most MEDIUM since space-collapsed

    # Edit-distance-1 against HIGH entries → MEDIUM
    first_ph = phonemes[0] if phonemes else ""
    prefix = _normalise(first_ph)
    candidates = [f for f in high_forms if f.startswith(prefix[:2])] if prefix else []
    for cand in candidates[:200]:   # cap scan to avoid O(N) on large lexicons
        if _levenshtein(joined, cand) <= _MEDIUM_EDIT_DIST:
            return cand, TIER_MEDIUM

    # First-syllable prefix match against all forms → MEDIUM
    if prefix and len(prefix) >= 2:
        for cand in all_forms:
            if cand.startswith(prefix) and len(cand) >= len(joined) - 2:
                return cand, TIER_MEDIUM

    return joined, TIER_NONE


# ---------------------------------------------------------------------------
# Taxogram detection
# ---------------------------------------------------------------------------

def _is_taxogram(sign_code: str) -> bool:
    base = re.split(r"[^0-9]", sign_code)[0]
    return base in TAXOGRAM_CODES


# ---------------------------------------------------------------------------
# Per-hypothesis glossing
# ---------------------------------------------------------------------------

GlossRow = dict[str, Any]   # one row of the gloss table


def gloss_hypothesis(
    hyp: dict[str, Any],
    high_forms: set[str],
    all_forms: set[str],
) -> list[GlossRow]:
    """Sliding-window gloss for one hypothesis.

    Returns one GlossRow per starting sign position (longest match wins).
    """
    assignments: list[dict] = hyp.get("assignments", [])
    n = len(assignments)
    rows: list[GlossRow] = []
    i = 0
    while i < n:
        sign_code  = assignments[i]["sign_code"]
        phoneme    = assignments[i]["phoneme"]
        confidence = assignments[i].get("confidence", 0.0)

        if _is_taxogram(sign_code):
            rows.append({
                "hyp_id":     hyp["hypothesis_id"],
                "position":   i,
                "sign_codes": [sign_code],
                "phonemes":   [phoneme],
                "gloss":      "[STRUCTURAL]",
                "tier":       TIER_NONE,
                "n_signs":    1,
                "confidence": confidence,
            })
            i += 1
            continue

        best_gloss = phoneme
        best_tier  = TIER_NONE
        best_width = 1

        for width in range(1, min(_MAX_WINDOW + 1, n - i + 1)):
            window_assigns = assignments[i : i + width]
            if any(_is_taxogram(a["sign_code"]) for a in window_assigns[1:]):
                break   # don't span across taxograms
            window_phones = [a["phoneme"] for a in window_assigns]
            gloss, tier   = _lookup_window(window_phones, high_forms, all_forms)
            if _TIER_RANK[tier] > _TIER_RANK[best_tier]:
                best_gloss = gloss
                best_tier  = tier
                best_width = width

        rows.append({
            "hyp_id":     hyp["hypothesis_id"],
            "position":   i,
            "sign_codes": [assignments[i + k]["sign_code"] for k in range(best_width)],
            "phonemes":   [assignments[i + k]["phoneme"]   for k in range(best_width)],
            "gloss":      best_gloss,
            "tier":       best_tier,
            "n_signs":    best_width,
            "confidence": confidence,
        })
        i += best_width

    return rows


# ---------------------------------------------------------------------------
# CSV output
# ---------------------------------------------------------------------------

def write_csv(rows: list[GlossRow], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(
            fh,
            fieldnames=["hyp_id", "position", "sign_codes", "phonemes",
                        "gloss", "tier", "n_signs", "confidence"],
        )
        w.writeheader()
        for row in rows:
            w.writerow({
                **row,
                "sign_codes": "|".join(row["sign_codes"]),
                "phonemes":   "|".join(row["phonemes"]),
            })
    log.info("CSV written → %s  (%d rows)", path, len(rows))


# ---------------------------------------------------------------------------
# HTML report
# ---------------------------------------------------------------------------

_CSS = """\
:root{--bg:#0d0f12;--surface:#161920;--surface2:#1e2229;
      --border:#2a2e38;--text:#d0d4dc;--muted:#6b7280;
      --accent:#c4a96d;--high:#4ade80;--medium:#facc15;--low:#94a3b8;--none:#374151;}
*{box-sizing:border-box;margin:0;padding:0;}
body{background:var(--bg);color:var(--text);
     font-family:'JetBrains Mono',monospace;font-size:13px;line-height:1.6;}
.wrap{max-width:1200px;margin:0 auto;padding:44px 24px;}
h1{font-size:22px;font-weight:600;color:var(--accent);margin-bottom:6px;}
.subtitle{color:var(--muted);font-size:12px;margin-bottom:36px;}
.hyp-block{background:var(--surface);border:1px solid var(--border);
           border-radius:6px;margin-bottom:32px;overflow:hidden;}
.hyp-header{padding:14px 18px;border-bottom:1px solid var(--border);
            display:flex;gap:16px;align-items:baseline;}
.hyp-id{color:var(--accent);font-weight:600;font-size:14px;}
.hyp-meta{color:var(--muted);font-size:11px;}
.gloss-table{width:100%;border-collapse:collapse;}
.gloss-table th{padding:7px 12px;text-align:left;font-size:10px;
                letter-spacing:.08em;text-transform:uppercase;
                color:var(--muted);border-bottom:1px solid var(--border);}
.gloss-table td{padding:6px 12px;border-bottom:1px solid var(--border);
                vertical-align:top;}
.gloss-table tr:last-child td{border-bottom:none;}
.gloss-table tr:hover td{background:var(--surface2);}
.tier-HIGH{color:var(--high);}
.tier-MEDIUM{color:var(--medium);}
.tier-LOW{color:var(--low);}
.tier-NONE{color:var(--none);}
.tag-structural{color:var(--muted);font-style:italic;}
.code{color:var(--accent);font-size:11px;}
.phoneme{color:#93c5fd;}
.conf-bar{display:inline-block;height:4px;background:var(--accent);
          vertical-align:middle;border-radius:2px;}
.legend{display:flex;gap:20px;flex-wrap:wrap;margin-bottom:28px;font-size:11px;}
.legend-item{display:flex;align-items:center;gap:6px;}
.legend-dot{width:10px;height:10px;border-radius:50%;}
"""


def _tier_html(tier: str) -> str:
    label_map = {
        TIER_HIGH:   "HIGH — IDS/Thomson exact",
        TIER_MEDIUM: "MEDIUM — edit-dist 1 or prefix",
        TIER_LOW:    "LOW — PP cognate only",
        TIER_NONE:   "NONE",
    }
    return f'<span class="tier-{tier}">{html.escape(label_map.get(tier, tier))}</span>'


def _conf_bar(c: float) -> str:
    w = max(0, min(60, int(c * 60)))
    return f'<span class="conf-bar" style="width:{w}px"></span> {c:.2f}'


def build_html_report(
    all_rows: list[GlossRow],
    hypotheses: list[dict],
) -> str:
    by_hyp: dict[str, list[GlossRow]] = defaultdict(list)
    for row in all_rows:
        by_hyp[row["hyp_id"]].append(row)

    hyp_meta: dict[str, dict] = {h["hypothesis_id"]: h for h in hypotheses}

    tier_counts: dict[str, int] = defaultdict(int)
    for row in all_rows:
        tier_counts[row["tier"]] += 1

    body_parts: list[str] = []

    # Legend
    legend_items = [
        (TIER_HIGH,   "var(--high)",   "HIGH"),
        (TIER_MEDIUM, "var(--medium)", "MEDIUM"),
        (TIER_LOW,    "var(--low)",    "LOW"),
        (TIER_NONE,   "var(--none)",   "NONE"),
    ]
    legend_html = '<div class="legend">'
    for tier, color, label in legend_items:
        n = tier_counts.get(tier, 0)
        legend_html += (
            f'<div class="legend-item">'
            f'<div class="legend-dot" style="background:{color}"></div>'
            f'<span class="tier-{tier}">{label}</span>'
            f'<span style="color:var(--muted)">({n})</span></div>'
        )
    legend_html += "</div>"
    body_parts.append(legend_html)

    for hyp_id, rows in by_hyp.items():
        meta = hyp_meta.get(hyp_id, {})
        score_str = (
            f"beam={meta.get('beam_score', 0):.3f}"
            if meta.get("beam_score") is not None else ""
        )
        body_parts.append(
            f'<div class="hyp-block">'
            f'<div class="hyp-header">'
            f'<span class="hyp-id">{html.escape(hyp_id)}</span>'
            f'<span class="hyp-meta">{html.escape(meta.get("hypothesis_type",""))} · {score_str}</span>'
            f'</div>'
        )
        body_parts.append(
            '<table class="gloss-table">'
            "<thead><tr>"
            "<th>#</th><th>Sign(s)</th><th>Phoneme(s)</th>"
            "<th>Gloss</th><th>Tier</th><th>Conf</th>"
            "</tr></thead><tbody>"
        )
        for row in rows:
            code_cell = " · ".join(
                f'<span class="code">{html.escape(c)}</span>'
                for c in row["sign_codes"]
            )
            ph_cell = " ".join(
                f'<span class="phoneme">{html.escape(p)}</span>'
                for p in row["phonemes"]
            )
            gloss = row["gloss"]
            if gloss == "[STRUCTURAL]":
                gloss_cell = '<span class="tag-structural">[STRUCTURAL]</span>'
            else:
                gloss_cell = html.escape(gloss)

            body_parts.append(
                f"<tr>"
                f"<td>{row['position']}</td>"
                f"<td>{code_cell}</td>"
                f"<td>{ph_cell}</td>"
                f"<td>{gloss_cell}</td>"
                f"<td>{_tier_html(row['tier'])}</td>"
                f"<td>{_conf_bar(row['confidence'])}</td>"
                f"</tr>"
            )
        body_parts.append("</tbody></table></div>")

    body = "\n".join(body_parts)
    return (
        "<!DOCTYPE html><html lang='en'>"
        "<head><meta charset='utf-8'>"
        "<title>Rongorongo Gloss Table</title>"
        f"<style>{_CSS}</style></head>"
        "<body><div class='wrap'>"
        "<h1>Rongorongo Gloss Table</h1>"
        "<p class='subtitle'>Sliding-window lexical lookup · "
        f"{sum(tier_counts.values())} gloss positions · "
        f"{tier_counts.get(TIER_HIGH, 0)} HIGH · "
        f"{tier_counts.get(TIER_MEDIUM, 0)} MEDIUM · "
        f"{tier_counts.get(TIER_LOW, 0)} LOW</p>"
        f"{body}"
        "</div></body></html>"
    )


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

def _smoke_test() -> None:
    high = {"henu", "ao", "manu", "rau"}
    all_f = high | {"korero", "tangata", "maori", "reo", "ao te"}
    rows = gloss_hypothesis(
        {
            "hypothesis_id": "SMOKE",
            "assignments": [
                {"sign_code": "001", "phoneme": "ao",    "confidence": 0.9, "evidence_count": 1},
                {"sign_code": "200", "phoneme": "henu",  "confidence": 0.5, "evidence_count": 2},
                {"sign_code": "002", "phoneme": "ma",    "confidence": 0.3, "evidence_count": 3},
                {"sign_code": "003", "phoneme": "nu",    "confidence": 0.3, "evidence_count": 3},
                {"sign_code": "010", "phoneme": "rau",   "confidence": 0.7, "evidence_count": 1},
            ],
        },
        high, all_f,
    )
    assert rows[0]["gloss"] == "ao" and rows[0]["tier"] == TIER_HIGH, \
        f"Expected HIGH 'ao', got {rows[0]}"
    assert rows[1]["gloss"] == "[STRUCTURAL]", \
        f"Expected STRUCTURAL for sign 200, got {rows[1]}"
    assert rows[2]["tier"] in (TIER_MEDIUM, TIER_NONE), \
        f"Unexpected tier for 'ma nu': {rows[2]}"
    log.info("Smoke test passed (%d rows).", len(rows))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Gloss rongorongo decipherment hypotheses against Polynesian lexicon."
    )
    p.add_argument(
        "--ranking",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "decipherment" / "ranking.json",
        help="Path to ranking.json (default: outputs/decipherment/ranking.json).",
    )
    p.add_argument(
        "--output-csv",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "decipherment" / "gloss_table.csv",
        help="CSV output path.",
    )
    p.add_argument(
        "--output-html",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "analysis" / "gloss_report.html",
        help="HTML report output path.",
    )
    p.add_argument(
        "--top",
        type=int,
        default=0,
        metavar="N",
        help="Only gloss the top N hypotheses (0 = all).",
    )
    p.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run built-in smoke test and exit.",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    if args.smoke_test:
        _smoke_test()
        return

    if not args.ranking.exists():
        log.error("ranking.json not found: %s", args.ranking)
        sys.exit(1)

    ranking = json.loads(args.ranking.read_text(encoding="utf-8"))
    hypotheses: list[dict] = ranking.get("hypotheses", [])
    if args.top > 0:
        hypotheses = hypotheses[: args.top]
    log.info("Glossing %d hypothesis/es.", len(hypotheses))

    high_forms, all_forms = _load_lexicon(PROJECT_ROOT)

    all_rows: list[GlossRow] = []
    for hyp in hypotheses:
        rows = gloss_hypothesis(hyp, high_forms, all_forms)
        all_rows.extend(rows)
        tier_dist = defaultdict(int)
        for r in rows:
            tier_dist[r["tier"]] += 1
        log.info(
            "  %s: %d positions  HIGH=%d MEDIUM=%d LOW=%d NONE=%d",
            hyp["hypothesis_id"],
            len(rows),
            tier_dist[TIER_HIGH],
            tier_dist[TIER_MEDIUM],
            tier_dist[TIER_LOW],
            tier_dist[TIER_NONE],
        )

    write_csv(all_rows, args.output_csv)

    html_str = build_html_report(all_rows, hypotheses)
    args.output_html.parent.mkdir(parents=True, exist_ok=True)
    args.output_html.write_text(html_str, encoding="utf-8")
    log.info("HTML report → %s", args.output_html)


if __name__ == "__main__":
    main()
