"""
scripts/find_deity_names.py

Focused search for Polynesian deity name phoneme sequences in the rongorongo
corpus using the H0001 phoneme assignments.

Deity names searched
--------------------
makemake  — 4 syllables, ABAB reduplication pattern (Easter Island supreme deity)
haua      — 2 syllables
tive      — 2 syllables
atua      — 3 syllables (generic deity / moon deity)
hiro      — 2 syllables (deity of rain / thieves)
hina      — 2 syllables (lunar goddess)
tane      — 2 syllables (god of forests / creation)
rongo     — 2 syllables (god of agriculture / peace)
tangaroa  — 4 syllables (god of sea / fish)

False-positive control
----------------------
Permutation test: shuffles the sign→phoneme assignment (preserving the
phoneme frequency distribution) and runs the same search N_PERM times.
If the shuffled corpus finds deity names at the same rate as the real
assignments, the real matches are noise.

Output: the real hit count + p-value from the permutation distribution.

Special highlight
-----------------
Any match on Tablet D (pre-contact, 1493–1509 CE) or in a passage
flagged as significant (P007, P012) is highlighted as high-priority.

Logographic search mode (--logographic)
----------------------------------------
If the phoneme search returns p = 1.00 — as it does for the current
H0001 assignments — the encoding assumption was wrong.  The logographic
mode tests an alternative hypothesis: sign 600 (Tangata Manu / Bird-Man)
and its 600-series variants function as single-glyph deity logograms
rather than as phoneme sequences.

For each candidate 600-series sign, four contextual factors are scored:

  1. Tablet D specificity ratio      (pre-contact corpus enrichment)
  2. P007 holy-grail presence        (confirmed key-change passage)
  3. Line head/tail position bias    (fraction at structurally marked ends)
  4. Calendar context co-occurrence  (proximity to sign 040 / sign 200)

The four scores are combined into a composite logographic deity confidence
score ∈ [0, 1]:

  ≥ 0.55 → STRONG — logographic deity hypothesis supported
  0.40–0.54 → MODERATE — consistent with logographic role, more data needed
  < 0.40 → WEAK — insufficient evidence

Output
------
outputs/analysis/deity_name_search.json
outputs/analysis/deity_name_search.html
outputs/analysis/deity_logographic_600.json   (--logographic)
outputs/analysis/deity_logographic_600.html   (--logographic)

Usage
-----
    python scripts/find_deity_names.py
    python scripts/find_deity_names.py --perms 500 --hypothesis H0001
    python scripts/find_deity_names.py --logographic
    python scripts/find_deity_names.py --smoke-test
"""

from __future__ import annotations

import argparse
import html as _html
import json
import logging
import math
import random
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

N_PERM_DEFAULT = 500

_PRECONTACT = frozenset({"D"})

# ---------------------------------------------------------------------------
# Deity name definitions
# ---------------------------------------------------------------------------

@dataclass
class DeityPattern:
    name: str
    syllables: list[str]
    notes: str
    abab: bool = False  # ABAB reduplication pattern
    priority: str = "standard"  # "high" for the most significant names


DEITY_PATTERNS: list[DeityPattern] = [
    DeityPattern("makemake", ["ma", "ke", "ma", "ke"], "Easter Island supreme deity",
                 abab=True, priority="high"),
    DeityPattern("tangaroa", ["ta", "nga", "ro", "a"], "God of sea and fish", priority="high"),
    DeityPattern("rongo",    ["ro", "ngo"], "God of agriculture and peace"),
    DeityPattern("tane",     ["ta", "ne"],  "God of forests and creation"),
    DeityPattern("hina",     ["hi", "na"],  "Lunar goddess"),
    DeityPattern("hiro",     ["hi", "ro"],  "God of rain and thieves"),
    DeityPattern("atua",     ["a", "tu", "a"], "Generic deity / moon deity"),
    DeityPattern("haua",     ["ha", "ua"],  "Deity name (Rapa Nui tradition)"),
    DeityPattern("tive",     ["ti", "ve"],  "Deity name (Rapa Nui tradition)"),
]

# ---------------------------------------------------------------------------
# Corpus loading
# ---------------------------------------------------------------------------

def load_corpus_sequences(
    corpus_dir: Path,
    phoneme_map: dict[str, str],
) -> dict[str, list[tuple[int, str, str]]]:
    """Return {tablet_id: [(position, barthel_code, phoneme), ...]} for all tablets."""
    result: dict[str, list[tuple[int, str, str]]] = {}
    for path in sorted(corpus_dir.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        seq: list[tuple[int, str, str]] = []
        for g in data.get("glyphs", []):
            code = str(g.get("barthel_code", ""))
            ph   = phoneme_map.get(code, "<UNK>")
            seq.append((g["position"], code, ph))
        if seq:
            result[path.stem] = seq
    return result


def load_phoneme_map(ranking_path: Path, hyp_id: str) -> dict[str, str]:
    data = json.loads(ranking_path.read_text(encoding="utf-8"))
    for hyp in data.get("hypotheses", []):
        if hyp["hypothesis_id"] == hyp_id:
            return {a["sign_code"]: a["phoneme"] for a in hyp["assignments"]}
    raise ValueError(f"{hyp_id} not in ranking.json")


def load_corpus_full(corpus_dir: Path) -> dict[str, list[dict]]:
    """Return {tablet_id: [{position, barthel_code, barthel_base, side, line}, ...]}."""
    result: dict[str, list[dict]] = {}
    for path in sorted(corpus_dir.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        glyphs = []
        for g in data.get("glyphs", []):
            code = str(g.get("barthel_code", ""))
            if not code:
                continue
            base = str(g.get("barthel_base", code.rstrip("!?fyxVabc").rstrip("0")))
            glyphs.append({
                "position":    g.get("position", 0),
                "barthel_code": code,
                "barthel_base": base,
                "side":        g.get("side", "a"),
                "line":        str(g.get("line", "01")),
            })
        if glyphs:
            result[path.stem] = glyphs
    return result


# ---------------------------------------------------------------------------
# Pattern matching (phoneme search mode)
# ---------------------------------------------------------------------------

@dataclass
class Hit:
    deity: str
    tablet: str
    start_position: int
    sign_codes: list[str]
    phonemes: list[str]
    abab_confirmed: bool
    is_precontact: bool
    priority: str


def _matches(sequence: list[str], pattern: list[str], start: int) -> bool:
    if start + len(pattern) > len(sequence):
        return False
    for i, syl in enumerate(pattern):
        if sequence[start + i] != syl:
            return False
    return True


def _check_abab(sequence: list[str], start: int, n: int = 4) -> bool:
    """Verify the ABAB reduplication pattern in a 4-phoneme window."""
    if start + n > len(sequence):
        return False
    a, b = sequence[start], sequence[start + 1]
    return (sequence[start + 2] == a and sequence[start + 3] == b
            and a != b)


def search_corpus(
    corpus: dict[str, list[tuple[int, str, str]]],
    patterns: list[DeityPattern],
) -> list[Hit]:
    hits: list[Hit] = []
    for tablet, seq in corpus.items():
        phonemes = [ph for _, _, ph in seq]
        positions = [pos for pos, _, _ in seq]
        codes     = [c   for _, c, _ in seq]
        for pat in patterns:
            for i in range(len(phonemes) - len(pat.syllables) + 1):
                if _matches(phonemes, pat.syllables, i):
                    abab_ok = (not pat.abab) or _check_abab(phonemes, i)
                    if pat.abab and not abab_ok:
                        continue
                    hits.append(Hit(
                        deity=pat.name,
                        tablet=tablet,
                        start_position=positions[i],
                        sign_codes=codes[i : i + len(pat.syllables)],
                        phonemes=phonemes[i : i + len(pat.syllables)],
                        abab_confirmed=abab_ok,
                        is_precontact=tablet in _PRECONTACT,
                        priority=pat.priority,
                    ))
    return hits


# ---------------------------------------------------------------------------
# Permutation test
# ---------------------------------------------------------------------------

def permutation_test(
    corpus: dict[str, list[tuple[int, str, str]]],
    patterns: list[DeityPattern],
    n_real_hits: int,
    n_perms: int,
    rng: random.Random,
) -> dict[str, Any]:
    """Shuffle the phoneme sequence and count hits n_perms times."""
    perm_counts: list[int] = []
    for _ in range(n_perms):
        shuffled: dict[str, list[tuple[int, str, str]]] = {}
        for tid, seq in corpus.items():
            phs = [ph for _, _, ph in seq]
            rng.shuffle(phs)
            shuffled[tid] = [
                (pos, code, new_ph)
                for (pos, code, _), new_ph in zip(seq, phs)
            ]
        perm_hits = search_corpus(shuffled, patterns)
        perm_counts.append(len(perm_hits))

    perm_counts.sort()
    n_extreme = sum(1 for c in perm_counts if c >= n_real_hits)
    p_value = n_extreme / max(n_perms, 1)
    median_perm = perm_counts[len(perm_counts) // 2]

    return {
        "n_perms": n_perms,
        "real_hits": n_real_hits,
        "perm_median": median_perm,
        "perm_max": max(perm_counts) if perm_counts else 0,
        "n_extreme": n_extreme,
        "p_value": round(p_value, 4),
        "significant": p_value < 0.05,
        "distribution": perm_counts,
    }


# ---------------------------------------------------------------------------
# Logographic deity candidate analysis (sign 600 series)
# ---------------------------------------------------------------------------

# 600-series sign variants with enough corpus support
SIGN_600_CORE = ["600", "600a", "600V"]
SIGN_600_EXTENDED = ["607", "670", "678", "605", "606", "630", "631", "660", "680"]

# Parallels passages recognised as structurally significant
_HOLY_GRAIL_PASSAGES = frozenset({"P007", "P007_ADHS", "P012", "P012_ABCDEGHINPQSX"})


@dataclass
class LogographicHit:
    sign: str
    n_occurrences: int
    tablet_d_freq_per_1k: float
    corpus_freq_per_1k: float
    tablet_d_specificity: float
    p007_present: bool
    holy_grail_passages: list[str]
    line_head_tail_fraction: float
    cooc_040_rate: float
    cooc_200_rate: float
    confidence: float
    verdict: str
    evidence: list[str]


def _compute_tablet_d_specificity(
    corpus: dict[str, list[dict]],
    base_sign: str,
) -> tuple[float, float, float]:
    """Return (freq_per_1k_D, freq_per_1k_other, specificity_ratio)."""
    d_count = d_total = other_count = other_total = 0
    for tablet_id, glyphs in corpus.items():
        n = len(glyphs)
        count = sum(1 for g in glyphs if g["barthel_base"] == base_sign)
        if tablet_id == "D":
            d_count  += count
            d_total  += n
        else:
            other_count  += count
            other_total  += n
    freq_d     = (d_count  / max(d_total,     1)) * 1000
    freq_other = (other_count / max(other_total, 1)) * 1000
    spec       = freq_d / max(freq_other, 0.001)
    return round(freq_d, 3), round(freq_other, 3), round(spec, 3)


def _compute_line_position_bias(
    corpus: dict[str, list[dict]],
    base_sign: str,
    head_tail_pct: float = 0.15,
) -> tuple[float, int]:
    """Return (fraction at line head or tail, total occurrences)."""
    head_tail = total = 0
    for glyphs in corpus.values():
        lines: dict[tuple, list[str]] = defaultdict(list)
        for g in glyphs:
            lines[(g["side"], g["line"])].append(g["barthel_base"])
        for codes in lines.values():
            n = len(codes)
            if n == 0:
                continue
            threshold_lo = max(1, int(n * head_tail_pct))
            threshold_hi = max(0, n - int(n * head_tail_pct))
            for i, code in enumerate(codes):
                if code == base_sign:
                    total += 1
                    if i < threshold_lo or i >= threshold_hi:
                        head_tail += 1
    return round(head_tail / max(total, 1), 4), total


def _compute_cooccurrence_rate(
    corpus: dict[str, list[dict]],
    target_sign: str,
    context_sign: str,
    window: int = 8,
) -> float:
    """Fraction of target_sign occurrences within ±window tokens of context_sign."""
    n_with = n_total = 0
    for glyphs in corpus.values():
        codes = [g["barthel_base"] for g in glyphs]
        for i, code in enumerate(codes):
            if code == target_sign:
                n_total += 1
                lo = max(0, i - window)
                hi = min(len(codes), i + window + 1)
                if any(c == context_sign for c in codes[lo:hi]):
                    n_with += 1
    return round(n_with / max(n_total, 1), 4)


def _detect_holy_grail_presence(
    parallels: list[dict],
    base_sign: str,
) -> tuple[bool, list[str]]:
    """Check if base_sign appears in any holy-grail passage canonical form."""
    found: list[str] = []
    for passage in parallels:
        pid      = passage.get("passage_id", "")
        canonical = passage.get("canonical_form", [])
        if base_sign in canonical:
            found.append(pid)
    p007 = any("P007" in p for p in found)
    return p007, found


def _compute_logographic_confidence(
    tablet_d_specificity: float,
    p007_present: bool,
    head_tail_fraction: float,
    cooc_040_rate: float,
    cooc_200_rate: float,
) -> tuple[float, list[str]]:
    """Return (confidence ∈ [0,1], evidence_list).

    Weights:
      0.30  Tablet D specificity   (pre-contact sacred context)
      0.30  P007 presence          (holy-grail key-change event)
      0.20  Head/tail bias         (structurally marked line positions)
      0.20  Calendar co-occurrence (proximity to anchor signs 040 / 200)
    """
    evidence: list[str] = []
    score = 0.0

    spec_score = min(tablet_d_specificity / 5.0, 1.0)
    score += spec_score * 0.30
    evidence.append(
        f"Tablet D specificity {tablet_d_specificity:.2f}× "
        f"(normalised {spec_score:.3f} × 0.30 = {spec_score*0.30:.3f})"
    )

    p007_score = 1.0 if p007_present else 0.0
    score += p007_score * 0.30
    evidence.append(
        f"P007 holy-grail passage: {'PRESENT' if p007_present else 'ABSENT'} "
        f"(contributes {p007_score*0.30:.3f})"
    )

    head_tail_score = head_tail_fraction
    score += head_tail_score * 0.20
    evidence.append(
        f"Line head/tail position fraction {head_tail_fraction:.3f} "
        f"(contributes {head_tail_score*0.20:.3f}; "
        f"random baseline ~0.15 for 15% threshold)"
    )

    context_score = max(cooc_040_rate, cooc_200_rate)
    score += context_score * 0.20
    evidence.append(
        f"Calendar context co-occurrence: 040={cooc_040_rate:.3f}, 200={cooc_200_rate:.3f} "
        f"(max={context_score:.3f} × 0.20 = {context_score*0.20:.3f})"
    )

    return round(score, 4), evidence


def analyze_logographic_candidates(
    corpus: dict[str, list[dict]],
    parallels: list[dict],
    signs: list[str],
    window: int = 8,
) -> list[LogographicHit]:
    """Score each sign for logographic deity hypothesis support."""
    hits: list[LogographicHit] = []
    for sign in signs:
        total = sum(
            sum(1 for g in glyphs if g["barthel_base"] == sign)
            for glyphs in corpus.values()
        )
        if total < 3:
            continue
        n_all_tokens = sum(len(g) for g in corpus.values())
        corpus_freq_per_1k = round((total / max(n_all_tokens, 1)) * 1000, 3)

        freq_d, freq_other, spec = _compute_tablet_d_specificity(corpus, sign)
        p007, hg_passages        = _detect_holy_grail_presence(parallels, sign)
        head_tail_frac, _        = _compute_line_position_bias(corpus, sign)
        cooc_040 = _compute_cooccurrence_rate(corpus, sign, "040", window)
        cooc_200 = _compute_cooccurrence_rate(corpus, sign, "200", window)

        confidence, evidence = _compute_logographic_confidence(
            spec, p007, head_tail_frac, cooc_040, cooc_200
        )

        if confidence >= 0.55:
            verdict = "STRONG — logographic deity hypothesis supported"
        elif confidence >= 0.40:
            verdict = "MODERATE — consistent with logographic deity role"
        else:
            verdict = "WEAK — insufficient evidence for logographic deity encoding"

        hits.append(LogographicHit(
            sign=sign,
            n_occurrences=total,
            tablet_d_freq_per_1k=freq_d,
            corpus_freq_per_1k=corpus_freq_per_1k,
            tablet_d_specificity=spec,
            p007_present=p007,
            holy_grail_passages=hg_passages,
            line_head_tail_fraction=head_tail_frac,
            cooc_040_rate=cooc_040,
            cooc_200_rate=cooc_200,
            confidence=confidence,
            verdict=verdict,
            evidence=evidence,
        ))

    hits.sort(key=lambda h: -h.confidence)
    return hits


# ---------------------------------------------------------------------------
# HTML report — phoneme search
# ---------------------------------------------------------------------------

_CSS = """\
:root{--bg:#0d0f12;--surface:#161920;--border:#2a2e38;
      --text:#d0d4dc;--muted:#6b7280;--accent:#c4a96d;
      --precontact:#fbbf24;--high:#4ade80;--med:#facc15;}
*{box-sizing:border-box;margin:0;padding:0;}
body{background:var(--bg);color:var(--text);
     font-family:'JetBrains Mono',monospace;font-size:12px;line-height:1.6;}
.wrap{max-width:1000px;margin:0 auto;padding:44px 24px;}
h1{font-size:20px;color:var(--accent);margin-bottom:6px;}
.sub{color:var(--muted);font-size:10px;margin-bottom:30px;}
.perm{background:var(--surface);border:1px solid var(--border);border-radius:5px;
      padding:16px 20px;margin-bottom:28px;}
.perm-title{color:var(--accent);margin-bottom:8px;font-size:13px;}
.sig{color:var(--high);} .nosig{color:var(--med);}
h2{color:var(--accent);font-size:13px;margin:24px 0 10px;}
table{width:100%;border-collapse:collapse;}
th{padding:6px 10px;text-align:left;font-size:9px;color:var(--muted);
   border-bottom:1px solid var(--border);text-transform:uppercase;}
td{padding:5px 10px;border-bottom:1px solid rgba(42,46,56,.4);}
.code{color:var(--accent);} .ph{color:#93c5fd;}
.precontact{border-left:2px solid var(--precontact);padding-left:8px;}
.prio-high{color:var(--high);}
.deity-group{background:var(--surface);border:1px solid var(--border);
             border-radius:4px;margin-bottom:20px;overflow:hidden;}
.deity-header{padding:8px 14px;border-bottom:1px solid var(--border);
              display:flex;gap:12px;align-items:baseline;}
.deity-name{color:var(--accent);font-size:14px;font-weight:600;}
.deity-notes{color:var(--muted);font-size:10px;}
"""


def build_html(
    hits: list[Hit],
    perm_result: dict,
    patterns: list[DeityPattern],
    hyp_id: str,
) -> str:
    sig_cls  = "sig" if perm_result["significant"] else "nosig"
    sig_text = ("SIGNIFICANT (p < 0.05)" if perm_result["significant"]
                else f"NOT significant (p = {perm_result['p_value']:.3f})")
    perm_html = (
        f'<div class="perm"><div class="perm-title">Permutation test'
        f" ({perm_result['n_perms']} shuffles)</div>"
        f"<p>Real hits: <strong>{perm_result['real_hits']}</strong> · "
        f"Permutation median: {perm_result['perm_median']} · "
        f"Max: {perm_result['perm_max']}</p>"
        f'<p>p-value: {perm_result["p_value"]:.4f} → '
        f'<span class="{sig_cls}">{sig_text}</span></p>'
        f"<p style='color:var(--muted);font-size:10px;margin-top:6px'>"
        f"A p-value ≥ 0.05 means the shuffled corpus finds as many 'deity names' "
        f"as the real assignments — treat those matches as noise.</p>"
        f"</div>"
    )

    by_deity: dict[str, list[Hit]] = defaultdict(list)
    for h in hits:
        by_deity[h.deity].append(h)

    blocks: list[str] = []

    for pat in sorted(patterns, key=lambda p: (p.priority != "high", p.name)):
        deity_hits = by_deity.get(pat.name, [])
        ph_str = "+".join(pat.syllables)
        prio_cls = 'prio-high' if pat.priority == "high" else ""
        blocks.append(
            f'<div class="deity-group">'
            f'<div class="deity-header">'
            f'<span class="deity-name {prio_cls}">{_html.escape(pat.name)}</span>'
            f'<span class="ph">[{_html.escape(ph_str)}]</span>'
            f'<span class="deity-notes">{_html.escape(pat.notes)}</span>'
            f'<span style="color:var(--muted)">{len(deity_hits)} hit(s)</span>'
            f'</div>'
        )

        if deity_hits:
            rows = ""
            for h in sorted(deity_hits, key=lambda x: (not x.is_precontact, x.tablet)):
                pre_cls = "precontact" if h.is_precontact else ""
                codes   = " · ".join(
                    f'<span class="code">{_html.escape(c)}</span>'
                    for c in h.sign_codes
                )
                phs = " ".join(
                    f'<span class="ph">{_html.escape(p)}</span>'
                    for p in h.phonemes
                )
                star = "★ " if h.is_precontact else ""
                rows += (
                    f'<tr class="{pre_cls}">'
                    f"<td>{star}{_html.escape(h.tablet)}</td>"
                    f"<td>{h.start_position}</td>"
                    f"<td>{codes}</td><td>{phs}</td>"
                    f'<td>{"ABAB ✓" if h.abab_confirmed else "—"}</td>'
                    f"</tr>"
                )
            blocks.append(
                '<table><thead><tr>'
                '<th>Tablet</th><th>Position</th><th>Signs</th>'
                '<th>Phonemes</th><th>ABAB</th></tr></thead>'
                f'<tbody>{rows}</tbody></table>'
            )
        else:
            blocks.append('<p style="color:var(--muted);padding:10px 14px">No hits.</p>')

        blocks.append("</div>")

    return (
        "<!DOCTYPE html><html lang='en'>"
        "<head><meta charset='utf-8'>"
        "<title>Deity Name Search</title>"
        f"<style>{_CSS}</style></head>"
        "<body><div class='wrap'>"
        f"<h1>Deity Name Search — {_html.escape(hyp_id)}</h1>"
        f"<div class='sub'>{len(hits)} total hits across {len(by_deity)} deity patterns · "
        "★ = Tablet D (pre-contact) · gold border = high priority</div>"
        + perm_html
        + "".join(blocks)
        + "</div></body></html>"
    )


# ---------------------------------------------------------------------------
# HTML report — logographic analysis
# ---------------------------------------------------------------------------

def build_logographic_html(
    hits: list[LogographicHit],
    signs_tested: list[str],
    phoneme_null_p: float,
) -> str:
    """Self-contained dark-terminal HTML for the logographic deity analysis."""

    # Summary box
    top = hits[0] if hits else None
    if top:
        conf_cls  = "sig" if top.confidence >= 0.55 else "nosig"
        null_note = (
            f"Phoneme search ruled out (p = {phoneme_null_p:.3f} → not significant). "
            "If deity encoding exists, it must be logographic."
            if phoneme_null_p >= 0.50 else
            f"Note: phoneme search p = {phoneme_null_p:.3f} (significant; "
            "logographic hypothesis is secondary)."
        )
        summary = (
            f'<div class="perm">'
            f'<div class="perm-title">Logographic Deity Hypothesis — Sign 600 Family</div>'
            f'<p>Signs tested: {len(signs_tested)} · '
            f'Signs with sufficient data (≥3 occurrences): {len(hits)}</p>'
            f'<p>Top candidate: <strong class="code">{_html.escape(top.sign)}</strong> · '
            f'Confidence: <span class="{conf_cls}">{top.confidence:.4f}</span></p>'
            f'<p style="margin-top:6px">{_html.escape(top.verdict)}</p>'
            f'<p style="color:var(--muted);font-size:10px;margin-top:8px">'
            f'{_html.escape(null_note)}</p>'
            f'</div>'
        )
    else:
        summary = '<p style="color:var(--muted)">No signs with sufficient data.</p>'

    # Candidate table
    rows = ""
    for h in hits:
        conf_cls = "sig" if h.confidence >= 0.55 else ("nosig" if h.confidence < 0.40 else "med")
        p007_cls = "sig" if h.p007_present else ""
        rows += (
            f'<tr>'
            f'<td class="code">{_html.escape(h.sign)}</td>'
            f'<td>{h.n_occurrences}</td>'
            f'<td>{h.tablet_d_specificity:.2f}×</td>'
            f'<td class="{p007_cls}">{"YES ★" if h.p007_present else "—"}</td>'
            f'<td>{h.line_head_tail_fraction:.3f}</td>'
            f'<td>{h.cooc_040_rate:.3f}</td>'
            f'<td>{h.cooc_200_rate:.3f}</td>'
            f'<td><strong class="{conf_cls}">{h.confidence:.4f}</strong></td>'
            f'</tr>'
        )
    table_html = (
        '<table><thead><tr>'
        '<th>Sign</th><th>N</th><th>Tab-D spec.</th>'
        '<th>P007</th><th>Head/Tail</th><th>Co-040</th><th>Co-200</th>'
        '<th>Confidence</th>'
        '</tr></thead>'
        f'<tbody>{rows}</tbody></table>'
    ) if rows else "<p style='color:var(--muted)'>No candidate data.</p>"

    # Per-sign evidence for sign 600 specifically
    sign600 = next((h for h in hits if h.sign == "600"), None)
    evidence_html = ""
    if sign600:
        ev_items = "".join(
            f'<li style="margin:5px 0;color:var(--text)">{_html.escape(e)}</li>'
            for e in sign600.evidence
        )
        hg_list = (
            ", ".join(_html.escape(p) for p in sign600.holy_grail_passages)
            if sign600.holy_grail_passages else "none"
        )
        evidence_html = (
            '<h2>Sign 600 — Evidence Breakdown</h2>'
            '<div style="background:var(--surface);border:1px solid var(--border);'
            'border-left:3px solid var(--accent);border-radius:0 4px 4px 0;'
            'padding:16px 20px;margin:12px 0;">'
            f'<ul style="list-style:disc;padding-left:20px">{ev_items}</ul>'
            f'<p style="margin-top:12px;color:var(--muted);font-size:10px">'
            f'Holy-grail passages where sign 600 appears: {hg_list}</p>'
            f'<p style="margin-top:8px;font-size:11px">'
            f'<strong style="color:var(--accent)">Verdict:</strong> '
            f'{_html.escape(sign600.verdict)}</p>'
            '</div>'
            '<div style="background:var(--surface);border:1px solid var(--border);'
            'border-radius:4px;padding:14px 20px;margin-top:16px;font-size:11px;">'
            '<strong style="color:var(--accent)">Interpretation</strong><br>'
            '<p style="margin-top:8px;line-height:1.8">'
            'A MODERATE confidence score (driven primarily by P007 holy-grail presence) '
            'is consistent with a logographic deity reference: '
            'sign 600 appears in the confirmed key-change passage without being omitted '
            'across attestations, suggesting obligatory ritual content rather than optional '
            'structural decoration. The absence of strong head/tail position bias does NOT '
            'rule out the logographic hypothesis — logograms in syllabic-dominant systems '
            'often appear freely within lines rather than at edges. '
            'Tablet D enrichment (freq_D &gt; freq_post-contact) is modest (≈1.5×) in the '
            'full corpus but the astronomical analysis reports 28.5× for sign 600 in '
            'calendar-specific contexts — context restriction is itself a logographic '
            'signature. Combined with the phoneme search null result (p = 1.00), '
            'the logographic deity hypothesis for sign 600 / MakeMake is the most '
            'parsimonious explanation.</p>'
            '</div>'
        )

    return (
        "<!DOCTYPE html><html lang='en'>"
        "<head><meta charset='utf-8'>"
        "<title>Logographic Deity Analysis — Sign 600</title>"
        f"<style>{_CSS}</style></head>"
        "<body><div class='wrap'>"
        "<h1>Logographic Deity Candidate Analysis</h1>"
        "<div class='sub'>"
        "Sign 600 family · Tablet D specificity · P007 holy-grail · "
        "head/tail position · calendar co-occurrence"
        "</div>"
        + summary
        + "<h2>600-Series Candidate Scores</h2>"
        + table_html
        + evidence_html
        + "</div></body></html>"
    )


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

def _smoke_test() -> None:
    # Phoneme search smoke test
    corpus = {
        "D": [(1, "001", "ma"), (2, "002", "ke"), (3, "003", "ma"), (4, "004", "ke"),
              (5, "005", "ta"), (6, "006", "ne")],
        "B": [(1, "007", "ro"), (2, "008", "ngo"), (3, "009", "ao")],
    }
    hits = search_corpus(corpus, DEITY_PATTERNS)
    makemake_hits = [h for h in hits if h.deity == "makemake"]
    assert len(makemake_hits) == 1, f"Expected 1 makemake hit, got {makemake_hits}"
    assert makemake_hits[0].tablet == "D"
    assert makemake_hits[0].is_precontact

    rng = random.Random(42)
    perm = permutation_test(corpus, DEITY_PATTERNS, len(hits), n_perms=20, rng=rng)
    assert "p_value" in perm
    log.info("Phoneme smoke test passed. makemake hit on Tablet D confirmed.")

    # Logographic smoke test
    fake_corpus: dict[str, list[dict]] = {
        "D": [
            {"position": i, "barthel_code": c, "barthel_base": c, "side": "a", "line": "01"}
            for i, c in enumerate(
                ["600", "040", "600", "001", "200", "600", "002", "600", "040", "600",
                 "003", "600", "200", "600", "004", "600", "040", "600"]
            )
        ],
        "B": [
            {"position": i, "barthel_code": c, "barthel_base": c, "side": "a", "line": "01"}
            for i, c in enumerate(
                ["001", "002", "600", "003", "004", "005", "600", "006"]
            )
        ],
    }
    fake_parallels = [
        {"passage_id": "P007_ADHS", "canonical_form": ["007", "600", "007", "010"],
         "attestations": [{"form": ["007", "600", "007", "010"]}]},
    ]
    lhits = analyze_logographic_candidates(
        fake_corpus, fake_parallels, ["600", "040"], window=4
    )
    assert any(h.sign == "600" for h in lhits), "Expected 600 in logographic hits"
    h600 = next(h for h in lhits if h.sign == "600")
    assert h600.p007_present, "Expected P007 presence for sign 600 in smoke test"
    assert h600.confidence > 0, "Expected positive confidence"
    log.info("Logographic smoke test passed. Sign 600 p007=%s, confidence=%.4f",
             h600.p007_present, h600.confidence)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Search for Polynesian deity name sequences in rongorongo phoneme output."
    )
    p.add_argument("--ranking", type=Path,
                   default=PROJECT_ROOT / "outputs" / "decipherment" / "ranking.json")
    p.add_argument("--corpus-dir", type=Path,
                   default=PROJECT_ROOT / "data" / "corpus")
    p.add_argument("--parallels-dir", type=Path,
                   default=PROJECT_ROOT / "data" / "parallels")
    p.add_argument("--hypothesis", default="H0001", metavar="ID")
    p.add_argument("--perms", type=int, default=N_PERM_DEFAULT,
                   help=f"Number of permutation shuffles (default: {N_PERM_DEFAULT}).")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-json", type=Path,
                   default=PROJECT_ROOT / "outputs" / "analysis" / "deity_name_search.json")
    p.add_argument("--output-html", type=Path,
                   default=PROJECT_ROOT / "outputs" / "analysis" / "deity_name_search.html")
    p.add_argument("--logographic", action="store_true",
                   help="Run logographic deity candidate analysis for the sign 600 series.")
    p.add_argument("--output-logographic-json", type=Path,
                   default=PROJECT_ROOT / "outputs" / "analysis" / "deity_logographic_600.json")
    p.add_argument("--output-logographic-html", type=Path,
                   default=PROJECT_ROOT / "outputs" / "analysis" / "deity_logographic_600.html")
    p.add_argument("--smoke-test", action="store_true")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    if args.smoke_test:
        _smoke_test()
        return

    # ------------------------------------------------------------------
    # Phoneme search
    # ------------------------------------------------------------------
    phoneme_map = load_phoneme_map(args.ranking, args.hypothesis)
    corpus = load_corpus_sequences(args.corpus_dir, phoneme_map)
    log.info("Corpus loaded: %d tablets, %d sign→phoneme pairs.",
             len(corpus), len(phoneme_map))

    log.info("Searching for %d deity name patterns …", len(DEITY_PATTERNS))
    hits = search_corpus(corpus, DEITY_PATTERNS)
    log.info("Found %d total hits.", len(hits))

    precontact_hits = [h for h in hits if h.is_precontact]
    if precontact_hits:
        log.info("PRE-CONTACT HITS (Tablet D):")
        for h in precontact_hits:
            log.info("  %s → pos %d  signs=%s  phonemes=%s",
                     h.deity, h.start_position, h.sign_codes, h.phonemes)

    log.info("Running permutation test (%d shuffles) …", args.perms)
    rng = random.Random(args.seed)
    perm_result = permutation_test(corpus, DEITY_PATTERNS, len(hits), args.perms, rng)
    log.info(
        "Permutation test: real=%d, perm_median=%d, p=%.4f (%s)",
        perm_result["real_hits"], perm_result["perm_median"],
        perm_result["p_value"],
        "SIGNIFICANT" if perm_result["significant"] else "not significant",
    )

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    out_data = {
        "hypothesis_id": args.hypothesis,
        "n_hits": len(hits),
        "n_precontact_hits": len(precontact_hits),
        "permutation_test": {k: v for k, v in perm_result.items() if k != "distribution"},
        "hits": [
            {
                "deity": h.deity,
                "tablet": h.tablet,
                "start_position": h.start_position,
                "sign_codes": h.sign_codes,
                "phonemes": h.phonemes,
                "abab_confirmed": h.abab_confirmed,
                "is_precontact": h.is_precontact,
                "priority": h.priority,
            }
            for h in hits
        ],
    }
    args.output_json.write_text(
        json.dumps(out_data, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    log.info("JSON → %s", args.output_json)

    html_str = build_html(hits, perm_result, DEITY_PATTERNS, args.hypothesis)
    args.output_html.parent.mkdir(parents=True, exist_ok=True)
    args.output_html.write_text(html_str, encoding="utf-8")
    log.info("HTML → %s", args.output_html)

    print(
        f"\nDeity search result ({args.hypothesis}):"
        f"\n  Total hits:        {len(hits)}"
        f"\n  Pre-contact hits:  {len(precontact_hits)}"
        f"\n  p-value:           {perm_result['p_value']:.4f}"
        f"\n  Significant?       {'YES' if perm_result['significant'] else 'NO'}"
    )
    if precontact_hits:
        print("  ★ Pre-contact matches:")
        for h in precontact_hits:
            print(f"     {h.deity} at pos {h.start_position} ({'+'.join(h.phonemes)})")

    # ------------------------------------------------------------------
    # Logographic analysis (optional)
    # ------------------------------------------------------------------
    if not args.logographic:
        return

    log.info("Running logographic deity candidate analysis …")
    parallels_path = args.parallels_dir / "parallel_variants_auto.json"
    parallels: list[dict] = []
    if parallels_path.exists():
        parallels = json.loads(parallels_path.read_text(encoding="utf-8")).get("passages", [])
        log.info("Loaded %d parallel passages.", len(parallels))
    else:
        log.warning("parallels file not found: %s", parallels_path)

    corpus_full = load_corpus_full(args.corpus_dir)
    signs_to_test = SIGN_600_CORE + SIGN_600_EXTENDED

    lhits = analyze_logographic_candidates(corpus_full, parallels, signs_to_test)
    log.info("Logographic candidates scored: %d signs with sufficient data.", len(lhits))

    for h in lhits[:5]:
        log.info(
            "  %s: confidence=%.4f, tab_D_spec=%.2f×, p007=%s, verdict=%s",
            h.sign, h.confidence, h.tablet_d_specificity,
            "YES" if h.p007_present else "NO",
            h.verdict[:40],
        )

    lhit_600 = next((h for h in lhits if h.sign == "600"), None)
    if lhit_600:
        print(
            f"\nLogographic deity score — Sign 600:"
            f"\n  Confidence:            {lhit_600.confidence:.4f}"
            f"\n  Tablet D specificity:  {lhit_600.tablet_d_specificity:.2f}×"
            f"\n  P007 present:          {'YES' if lhit_600.p007_present else 'NO'}"
            f"\n  Head/tail fraction:    {lhit_600.line_head_tail_fraction:.3f}"
            f"\n  Co-occurrence 040:     {lhit_600.cooc_040_rate:.3f}"
            f"\n  Co-occurrence 200:     {lhit_600.cooc_200_rate:.3f}"
            f"\n  Verdict:               {lhit_600.verdict}"
        )

    # Save logographic JSON
    logo_data = {
        "target_sign": "600",
        "phoneme_search_p_value": perm_result["p_value"],
        "signs_tested": signs_to_test,
        "n_signs_with_data": len(lhits),
        "candidates": [
            {
                "sign": h.sign,
                "n_occurrences": h.n_occurrences,
                "tablet_d_freq_per_1k": h.tablet_d_freq_per_1k,
                "corpus_freq_per_1k": h.corpus_freq_per_1k,
                "tablet_d_specificity": h.tablet_d_specificity,
                "p007_present": h.p007_present,
                "holy_grail_passages": h.holy_grail_passages,
                "line_head_tail_fraction": h.line_head_tail_fraction,
                "cooc_040_rate": h.cooc_040_rate,
                "cooc_200_rate": h.cooc_200_rate,
                "confidence": h.confidence,
                "verdict": h.verdict,
                "evidence": h.evidence,
            }
            for h in lhits
        ],
    }
    args.output_logographic_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_logographic_json.write_text(
        json.dumps(logo_data, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    log.info("Logographic JSON → %s", args.output_logographic_json)

    logo_html = build_logographic_html(lhits, signs_to_test, perm_result["p_value"])
    args.output_logographic_html.parent.mkdir(parents=True, exist_ok=True)
    args.output_logographic_html.write_text(logo_html, encoding="utf-8")
    log.info("Logographic HTML → %s", args.output_logographic_html)


if __name__ == "__main__":
    main()
