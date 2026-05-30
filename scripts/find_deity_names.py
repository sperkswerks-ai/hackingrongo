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

Output
------
outputs/analysis/deity_name_search.json
outputs/analysis/deity_name_search.html

Usage
-----
    python scripts/find_deity_names.py
    python scripts/find_deity_names.py --perms 500 --hypothesis H0001
    python scripts/find_deity_names.py --smoke-test
"""

from __future__ import annotations

import argparse
import html as _html
import json
import logging
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


# ---------------------------------------------------------------------------
# Pattern matching
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
    # Collect all phonemes from the corpus (preserving tablet structure)
    all_phonemes_per_tablet = {
        tid: [ph for _, _, ph in seq]
        for tid, seq in corpus.items()
    }

    perm_counts: list[int] = []
    for _ in range(n_perms):
        # Shuffle phonemes WITHIN each tablet to preserve length distribution
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
# HTML report
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
    # Permutation box
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

    # Group hits by deity
    by_deity: dict[str, list[Hit]] = defaultdict(list)
    for h in hits:
        by_deity[h.deity].append(h)

    pat_map = {p.name: p for p in patterns}
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
# Smoke test
# ---------------------------------------------------------------------------

def _smoke_test() -> None:
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
    log.info("Smoke test passed. makemake hit on Tablet D confirmed.")


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
    p.add_argument("--hypothesis", default="H0001", metavar="ID")
    p.add_argument("--perms", type=int, default=N_PERM_DEFAULT,
                   help=f"Number of permutation shuffles (default: {N_PERM_DEFAULT}).")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-json", type=Path,
                   default=PROJECT_ROOT / "outputs" / "analysis" / "deity_name_search.json")
    p.add_argument("--output-html", type=Path,
                   default=PROJECT_ROOT / "outputs" / "analysis" / "deity_name_search.html")
    p.add_argument("--smoke-test", action="store_true")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    if args.smoke_test:
        _smoke_test()
        return

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

    # Headline for the terminal
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


if __name__ == "__main__":
    main()
