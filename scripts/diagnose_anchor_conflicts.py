"""
scripts/diagnose_anchor_conflicts.py

Focused diagnostic for sign 600's phoneme assignment instability and its
relationship to the new calendar anchors.

Three hypotheses under test
---------------------------
A  Logographic — sign 600 (Bird-Man) is genuinely logographic / has no
   stable phoneme reading; its posterior is flat by nature.
B  Occupancy conflict — a new calendar anchor displaced 600 from a phoneme
   it previously held with high confidence.
C  Missing anchor — 600 has a correct phoneme but no known-plaintext
   evidence, so it drifts freely across chains.

Tests run
---------
1. Anchor activation audit
   For each CALENDAR_ANCHORS_HARD entry: was the sign actually in the
   corpus used for this run (sign_ids)?  Three anchors (143/huna,
   152/omotohi, 280/honu) are absent from Tablet D (smoke-test corpus)
   and were therefore never activated as cribs.

2. Occupancy pressure map
   For each anchor phoneme, compute the occupancy count in H0001 and
   flag any phoneme above max_signs_per_phoneme=4.  'a' currently has
   13 signs — a severe violation that drives 600 toward 'a' despite no
   evidence.

3. Sign 600 cross-hypothesis variance
   Report 600's phoneme assignment across H0001–H0005 and compute the
   entropy of the distribution.  High entropy = flat posterior (Hyp A/C).
   Perfect split (some chains 'a', some 'po') = conflict signal (Hyp B).

4. Phoneme displacement check
   For each anchor phoneme, list every sign that shares it in H0001.
   If a sign that previously had high confidence on a phoneme now shares
   it with an anchor, the anchor may have created an overcrowded slot.

5. Pinned LM score test  (the decisive test)
   Run the LMScorer on the full Tablet D sequence under three maps:
     (a) H0001 baseline (600 → 'a')
     (b) 600 pinned to TARGET_PHONEME (default 'manu', the iconographic
         reading; pass --pin-phoneme to override)
     (c) 600 pinned to 'i' (alternative phoneme proposed in the analysis)
   Compare ensemble log-probabilities.
   If (b) or (c) > (a): the displacement hypothesis (B) is supported —
     the baseline assignment is wrong because the chain was pushed away.
   If (a) ≥ (b) and (a) ≥ (c): 600 genuinely belongs at 'a' or the LM
     is agnostic (Hyp A or C).

Output
------
outputs/analysis/anchor_conflict_diagnosis.json
stdout: human-readable summary suitable for copy-paste into the talk notes

Usage
-----
    python scripts/diagnose_anchor_conflicts.py
    python scripts/diagnose_anchor_conflicts.py --pin-phoneme manu
    python scripts/diagnose_anchor_conflicts.py --smoke-test
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
_SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(_SCRIPTS_DIR))

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants — imported from run_decipherment to avoid divergence
# ---------------------------------------------------------------------------

from run_decipherment import CALENDAR_ANCHORS_HARD, CALENDAR_ANCHORS_SOFT  # noqa: E402

TARGET_SIGN         = "600"
DEFAULT_PIN_PHONEME = "manu"    # iconographic reading (Barthel / Metoro)
ALT_PIN_PHONEME     = "i"       # phonemic candidate from earlier run
OCCUPANCY_CAP       = 4         # max_signs_per_phoneme from config


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_ranking(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_corpus_tablet(corpus_dir: Path, tablet_id: str) -> list[str]:
    """Return the glyph sequence for one tablet as a list of Barthel codes."""
    p = corpus_dir / f"{tablet_id}.json"
    if not p.exists():
        raise FileNotFoundError(p)
    data = json.loads(p.read_text(encoding="utf-8"))
    return [g["barthel_code"] for g in data.get("glyphs", [])
            if g.get("barthel_code") and g["barthel_code"] != "?"]


def _build_phoneme_map(hypothesis: dict[str, Any]) -> dict[str, str]:
    return {a["sign_code"]: a["phoneme"] for a in hypothesis["assignments"]}


# ---------------------------------------------------------------------------
# Test 1: Anchor activation audit
# ---------------------------------------------------------------------------

def test1_anchor_activation(
    sign_ids: set[str],
) -> list[dict[str, Any]]:
    results = []
    all_anchors = dict(CALENDAR_ANCHORS_HARD)
    all_anchors.update({s: ph for s, (ph, _) in CALENDAR_ANCHORS_SOFT.items()})

    for sign, phoneme in sorted(all_anchors.items()):
        activated = sign in sign_ids
        anchor_type = ("HARD" if sign in CALENDAR_ANCHORS_HARD
                       else f"SOFT(w={CALENDAR_ANCHORS_SOFT[sign][1]})")
        results.append({
            "sign": sign,
            "pinned_phoneme": phoneme,
            "anchor_type": anchor_type,
            "in_corpus": activated,
            "status": "ACTIVE" if activated else "SILENTLY_SKIPPED",
            "note": (
                "Crib applied — sign is present in the run's corpus."
                if activated else
                "Sign absent from the run corpus (smoke-test = Tablet D only). "
                "Crib was built but never applied."
            ),
        })
    return results


# ---------------------------------------------------------------------------
# Test 2: Occupancy pressure map
# ---------------------------------------------------------------------------

def test2_occupancy(phoneme_map: dict[str, str]) -> dict[str, Any]:
    ph_to_signs: dict[str, list[str]] = {}
    for sign, ph in phoneme_map.items():
        ph_to_signs.setdefault(ph, []).append(sign)

    anchor_phonemes = set(CALENDAR_ANCHORS_HARD.values()) | {
        ph for ph, _ in CALENDAR_ANCHORS_SOFT.values()
    }

    rows = []
    for ph, signs in sorted(ph_to_signs.items(), key=lambda kv: -len(kv[1])):
        is_anchor = ph in anchor_phonemes
        over_cap = len(signs) > OCCUPANCY_CAP
        rows.append({
            "phoneme": ph,
            "n_signs": len(signs),
            "signs": sorted(signs),
            "is_anchor_phoneme": is_anchor,
            "over_cap": over_cap,
            "pressure": "HIGH" if over_cap else ("MODERATE" if len(signs) == OCCUPANCY_CAP else "OK"),
        })

    target_phoneme = phoneme_map.get(TARGET_SIGN, "<UNK>")
    target_row = next((r for r in rows if r["phoneme"] == target_phoneme), None)

    n_over = sum(1 for r in rows if r["over_cap"])
    return {
        "n_distinct_phonemes": len(rows),
        "n_over_cap": n_over,
        "occupancy_cap": OCCUPANCY_CAP,
        "target_sign_phoneme": target_phoneme,
        "target_phoneme_occupancy": target_row,
        "all_rows": rows,
    }


# ---------------------------------------------------------------------------
# Test 3: Sign 600 cross-hypothesis variance
# ---------------------------------------------------------------------------

def test3_sign600_variance(hypotheses: list[dict[str, Any]]) -> dict[str, Any]:
    assignments = []
    for h in hypotheses:
        for a in h["assignments"]:
            if a["sign_code"] == TARGET_SIGN:
                assignments.append({
                    "hyp_id": h["hypothesis_id"],
                    "phoneme": a["phoneme"],
                    "confidence": a["confidence"],
                    "evidence_count": a["evidence_count"],
                })

    phoneme_counts = Counter(a["phoneme"] for a in assignments)
    n_hyps = len(assignments)

    if n_hyps == 0:
        return {
            "target_sign": TARGET_SIGN,
            "n_hypotheses": 0,
            "phoneme_distribution": {},
            "cross_hyp_entropy_bits": 0.0,
            "normalised_entropy": 0.0,
            "interpretation": "ABSENT — sign not found in any hypothesis",
            "per_hypothesis": [],
        }

    # Entropy of the cross-hypothesis distribution
    entropy = 0.0
    for ph, cnt in phoneme_counts.items():
        p = cnt / n_hyps
        entropy -= p * math.log2(p)

    max_entropy = math.log2(n_hyps) if n_hyps > 1 else 1.0
    normalised_entropy = entropy / max_entropy if max_entropy > 0 else 0.0

    # Interpret
    if normalised_entropy < 0.1:
        interpretation = "STABLE — all hypotheses agree; high confidence reading"
    elif normalised_entropy < 0.5:
        interpretation = "MODERATE VARIANCE — partial agreement; 2–3 distinct readings"
    else:
        interpretation = "HIGH VARIANCE — flat posterior; sign is genuinely unconstrained"

    return {
        "target_sign": TARGET_SIGN,
        "n_hypotheses": n_hyps,
        "phoneme_distribution": dict(phoneme_counts),
        "cross_hyp_entropy_bits": round(entropy, 4),
        "normalised_entropy": round(normalised_entropy, 4),
        "interpretation": interpretation,
        "per_hypothesis": assignments,
    }


# ---------------------------------------------------------------------------
# Test 4: Displacement check — which signs share each anchor phoneme
# ---------------------------------------------------------------------------

def test4_displacement(
    phoneme_map: dict[str, str],
    hypothesis: dict[str, Any],
) -> list[dict[str, Any]]:
    conf_map = {a["sign_code"]: a["confidence"] for a in hypothesis["assignments"]}
    ph_to_signs: dict[str, list[str]] = {}
    for sign, ph in phoneme_map.items():
        ph_to_signs.setdefault(ph, []).append(sign)

    results = []
    anchor_phonemes = {
        **CALENDAR_ANCHORS_HARD,
        **{s: ph for s, (ph, _) in CALENDAR_ANCHORS_SOFT.items()},
    }

    for anchor_sign, anchor_ph in sorted(anchor_phonemes.items()):
        co_occupants = [
            {"sign": s, "confidence": conf_map.get(s, 0.0)}
            for s in ph_to_signs.get(anchor_ph, [])
            if s != anchor_sign
        ]
        high_conf_co = [c for c in co_occupants if c["confidence"] >= 0.7]
        results.append({
            "anchor_sign": anchor_sign,
            "anchor_phoneme": anchor_ph,
            "anchor_type": "HARD" if anchor_sign in CALENDAR_ANCHORS_HARD else "SOFT",
            "n_co_occupants": len(co_occupants),
            "co_occupants": co_occupants,
            "high_conf_co_occupants": high_conf_co,
            "conflict": len(co_occupants) > 0,
            "serious_conflict": len(high_conf_co) > 0,
        })
    return results


# ---------------------------------------------------------------------------
# Test 5: Pinned LM score test
# ---------------------------------------------------------------------------

def test5_pinned_lm_score(
    phoneme_map: dict[str, str],
    corpus_sequences: list[list[str]],
    pin_phonemes: list[str],
) -> dict[str, Any]:
    """Score the corpus under H0001 baseline and with 600 pinned to each candidate."""
    try:
        from omegaconf import OmegaConf
        from hackingrongo.zone_c.lm_scoring import LMScorer
    except ImportError as e:
        return {"error": str(e), "skipped": True}

    try:
        cfg = OmegaConf.load(PROJECT_ROOT / "conf" / "config.yaml")
        scorer = LMScorer(cfg, PROJECT_ROOT)
        if not scorer.languages_available:
            return {
                "error": "No language models loaded — run scripts/build_language_models.py first.",
                "skipped": True,
            }
    except Exception as e:
        return {"error": f"LMScorer init failed: {e}", "skipped": True}

    def _score_map(pmap: dict[str, str]) -> float:
        total = 0.0
        for seq in corpus_sequences:
            ph_seq = [pmap.get(code, "<UNK>") for code in seq]
            result = scorer.score(ph_seq)
            if math.isfinite(result.ensemble_log_prob):
                total += result.ensemble_log_prob
            else:
                total -= 1000.0
        return total

    baseline_lp = _score_map(phoneme_map)
    log.info("Baseline LM score (600→%r): %.4f", phoneme_map.get(TARGET_SIGN), baseline_lp)

    pinned_results = []
    for ph in pin_phonemes:
        pinned_map = dict(phoneme_map)
        pinned_map[TARGET_SIGN] = ph
        lp = _score_map(pinned_map)
        delta = lp - baseline_lp
        log.info("  600→%r:  LM score = %.4f  Δ = %+.4f", ph, lp, delta)
        pinned_results.append({
            "pin_phoneme": ph,
            "lm_log_prob": round(lp, 4),
            "delta_vs_baseline": round(delta, 4),
            "interpretation": (
                "BETTER than baseline — baseline assignment is displaced/wrong"
                if delta > 0.1 else
                ("COMPARABLE — no clear preference (flat posterior for this sign)"
                 if abs(delta) <= 0.1 else
                 "WORSE — baseline assignment is LM-preferred")
            ),
        })

    # best_overall includes the baseline so we compare apples to apples
    all_candidates = [
        {"pin_phoneme": phoneme_map.get(TARGET_SIGN, "<UNK>"),
         "lm_log_prob": baseline_lp, "delta_vs_baseline": 0.0}
    ] + pinned_results
    best_overall = max(all_candidates, key=lambda r: r["lm_log_prob"])
    best_pinned  = max(pinned_results, key=lambda r: r["lm_log_prob"])

    baseline_entry = {
        "pin_phoneme": phoneme_map.get(TARGET_SIGN, "<UNK>"),
        "lm_log_prob": round(baseline_lp, 4),
        "delta_vs_baseline": 0.0,
        "interpretation": "baseline (H0001)",
    }

    # Hypothesis determination
    all_scores = [r["lm_log_prob"] for r in all_candidates]
    score_range = max(all_scores) - min(all_scores)

    if score_range < 0.5:
        hypothesis_supported = "C"
        hypothesis_text = (
            "All phoneme assignments produce near-identical LM scores "
            f"(range {score_range:.4f}). Sign 600 is LM-agnostic — the LM has "
            "no preference and the chain assigns it arbitrarily. Adding a crib "
            "would stabilise it without hurting the score (Hypothesis C)."
        )
    elif best_overall["pin_phoneme"] != phoneme_map.get(TARGET_SIGN):
        # A pinned alternative beats the baseline
        delta = best_pinned["delta_vs_baseline"]
        hypothesis_supported = "B"
        hypothesis_text = (
            f"Pinning 600 to '{best_pinned['pin_phoneme']}' produces a HIGHER LM "
            f"score (Δ={delta:+.4f}) than the H0001 baseline '{phoneme_map.get(TARGET_SIGN)}'. "
            "The current assignment is displaced — the occupancy pressure on the "
            "overloaded 'a' slot pushed 600 away from its correct phoneme "
            "(Hypothesis B). Recommend adding 600 as a soft crib."
        )
    else:
        # Baseline is best
        hypothesis_supported = "A"
        hypothesis_text = (
            f"The H0001 baseline phoneme '{phoneme_map.get(TARGET_SIGN)}' produces "
            f"the highest LM score (score range {score_range:.4f}). "
            "Sign 600 may be LM-correct at its current assignment, or may be "
            "genuinely logographic with no meaningful phoneme reading. "
            "No occupancy displacement is indicated (Hypothesis A)."
        )

    return {
        "baseline": baseline_entry,
        "pinned": pinned_results,
        "score_range": round(score_range, 4),
        "best_overall_phoneme": best_overall["pin_phoneme"],
        "baseline_is_best": best_overall["pin_phoneme"] == phoneme_map.get(TARGET_SIGN),
        "hypothesis_supported": hypothesis_supported,
        "hypothesis_text": hypothesis_text,
    }


# ---------------------------------------------------------------------------
# Summary printer
# ---------------------------------------------------------------------------

_SEP = "─" * 64


def _print_report(
    t1: list[dict],
    t2: dict,
    t3: dict,
    t4: list[dict],
    t5: dict,
    sign_600_current: str,
) -> None:
    print(f"\n{'═' * 64}")
    print("  Anchor Conflict Diagnostic — Sign 600")
    print(f"{'═' * 64}")

    # Test 1
    print(f"\n{_SEP}")
    print("TEST 1 — Anchor activation audit")
    print(_SEP)
    n_skipped = sum(1 for r in t1 if r["status"] == "SILENTLY_SKIPPED")
    for r in t1:
        icon = "✓" if r["in_corpus"] else "✗"
        print(f"  {icon} {r['anchor_type']:12} {r['sign']:5} → {r['pinned_phoneme']:12}  {r['status']}")
    if n_skipped:
        print(f"\n  ⚠ {n_skipped} anchor(s) silently skipped because their signs are absent from")
        print("    the run corpus.  Full-corpus run required to activate them.")

    # Test 2
    print(f"\n{_SEP}")
    print("TEST 2 — Occupancy pressure map")
    print(_SEP)
    print(f"  Occupancy cap: {t2['occupancy_cap']}  |  "
          f"Distinct phonemes used: {t2['n_distinct_phonemes']}  |  "
          f"Phonemes over cap: {t2['n_over_cap']}")
    print(f"  Sign 600 current phoneme: {sign_600_current!r}")
    if t2["target_phoneme_occupancy"]:
        row = t2["target_phoneme_occupancy"]
        print(f"  Occupancy of {row['phoneme']!r}: {row['n_signs']} signs  [{row['pressure']}]")
        if row["over_cap"]:
            print(f"  → Phoneme {row['phoneme']!r} is {row['n_signs'] - t2['occupancy_cap']} "
                  f"sign(s) over cap. Occupancy penalty is actively penalising this slot.")
    print("\n  Top phonemes by occupancy (anchor phonemes marked ★):")
    for row in t2["all_rows"][:8]:
        star = " ★" if row["is_anchor_phoneme"] else ""
        print(f"    {row['phoneme']:12} {row['n_signs']:3} signs  [{row['pressure']}]{star}")

    # Test 3
    print(f"\n{_SEP}")
    print("TEST 3 — Sign 600 cross-hypothesis variance")
    print(_SEP)
    print(f"  Distribution: {t3['phoneme_distribution']}")
    print(f"  Entropy: {t3['cross_hyp_entropy_bits']:.4f} bits  "
          f"(normalised: {t3['normalised_entropy']:.4f})")
    print(f"  → {t3['interpretation']}")
    print("\n  Per-hypothesis assignments:")
    for row in t3["per_hypothesis"]:
        print(f"    {row['hyp_id']}: {row['phoneme']:12}  conf={row['confidence']:.3f}")

    # Test 4
    print(f"\n{_SEP}")
    print("TEST 4 — Displacement check (which signs share anchor phonemes)")
    print(_SEP)
    conflicts = [r for r in t4 if r["conflict"]]
    serious   = [r for r in t4 if r["serious_conflict"]]
    print(f"  Anchor phonemes with co-occupants: {len(conflicts)}/{len(t4)}")
    print(f"  Serious conflicts (co-occupant conf ≥ 0.70): {len(serious)}")
    for r in t4:
        if r["co_occupants"]:
            co_str = ", ".join(
                f"{c['sign']}(conf={c['confidence']:.2f})" for c in r["co_occupants"]
            )
            serious_flag = " ← SERIOUS CONFLICT" if r["serious_conflict"] else ""
            print(f"  {r['anchor_type']:5} {r['anchor_sign']:5}→{r['anchor_phoneme']:12}  "
                  f"co-occupants: {co_str}{serious_flag}")
        else:
            print(f"  {r['anchor_type']:5} {r['anchor_sign']:5}→{r['anchor_phoneme']:12}  "
                  f"no conflict")

    # Test 5
    print(f"\n{_SEP}")
    print("TEST 5 — Pinned LM score comparison (decisive test)")
    print(_SEP)
    if t5.get("skipped"):
        print(f"  SKIPPED: {t5.get('error', 'unknown error')}")
    else:
        print(f"  Baseline (H0001, 600→{t5['baseline']['pin_phoneme']!r}): "
              f"log-prob = {t5['baseline']['lm_log_prob']:.4f}")
        for r in t5["pinned"]:
            better = " ← BETTER" if r["delta_vs_baseline"] > 0.1 else ""
            print(f"  600→{r['pin_phoneme']:12}  log-prob = {r['lm_log_prob']:.4f}  "
                  f"Δ = {r['delta_vs_baseline']:+.4f}{better}")
        print(f"\n  Score range: {t5['score_range']:.4f}")
        best_label = "baseline" if t5["baseline_is_best"] else "pinned alternative"
        print(f"  Best overall phoneme: {t5['best_overall_phoneme']!r}  ({best_label})")
        print(f"\n  SUPPORTED HYPOTHESIS: {t5['hypothesis_supported']}")
        print(f"  {t5['hypothesis_text']}")

    print(f"\n{'═' * 64}\n")


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

def _smoke_test() -> None:
    phoneme_map = {
        "600": "a", "040": "kokore", "001": "a", "002": "u",
        "003": "a", "004": "hi", "005": "a", "006": "a",
    }
    sign_ids = set(phoneme_map.keys())

    t1 = test1_anchor_activation(sign_ids)
    assert any(r["status"] == "SILENTLY_SKIPPED" for r in t1), "Expected skipped anchors"
    assert any(r["status"] == "ACTIVE" for r in t1), "Expected active anchors"

    t2 = test2_occupancy(phoneme_map)
    assert t2["n_over_cap"] > 0, "Expected overcrowded phoneme 'a'"

    hyps_mock = [
        {"hypothesis_id": f"H000{i}", "assignments": [
            {"sign_code": "600", "phoneme": ph, "confidence": 0.0, "evidence_count": 1}
        ]}
        for i, ph in enumerate(["a", "a", "po", "po", "u"], 1)
    ]
    t3 = test3_sign600_variance(hyps_mock)
    assert t3["normalised_entropy"] > 0, "Expected non-zero entropy"

    log.info("Smoke test passed.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Diagnose anchor conflicts around sign 600."
    )
    p.add_argument("--ranking", type=Path,
                   default=PROJECT_ROOT / "outputs" / "decipherment" / "ranking.json")
    p.add_argument("--corpus-dir", type=Path,
                   default=PROJECT_ROOT / "data" / "corpus")
    p.add_argument("--tablet", default="D",
                   help="Tablet used for the LM score test (default: D, the smoke-test tablet).")
    p.add_argument("--pin-phoneme", default=DEFAULT_PIN_PHONEME,
                   help=f"Primary phoneme to test for sign 600 (default: {DEFAULT_PIN_PHONEME!r}).")
    p.add_argument("--output", type=Path,
                   default=PROJECT_ROOT / "outputs" / "analysis" / "anchor_conflict_diagnosis.json")
    p.add_argument("--smoke-test", action="store_true")
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    if args.smoke_test:
        _smoke_test()
        return

    ranking = _load_ranking(args.ranking)
    hypotheses = ranking["hypotheses"]
    h1_map = _build_phoneme_map(hypotheses[0])
    sign_ids = set(h1_map.keys())
    sign_600_current = h1_map.get(TARGET_SIGN, "<UNK>")

    log.info("Loaded %d hypotheses, %d distinct signs.", len(hypotheses), len(sign_ids))
    log.info("Sign 600 in H0001: %r", sign_600_current)

    # Load corpus for the LM score test
    try:
        corpus_seq = _load_corpus_tablet(args.corpus_dir, args.tablet)
        log.info("Corpus tablet %s: %d glyphs.", args.tablet, len(corpus_seq))
    except FileNotFoundError:
        log.error("Tablet %s not found in %s.", args.tablet, args.corpus_dir)
        corpus_seq = []

    pin_phonemes = [args.pin_phoneme]
    if ALT_PIN_PHONEME not in pin_phonemes:
        pin_phonemes.append(ALT_PIN_PHONEME)
    if "manu" not in pin_phonemes:
        pin_phonemes.append("manu")

    t1 = test1_anchor_activation(sign_ids)
    t2 = test2_occupancy(h1_map)
    t3 = test3_sign600_variance(hypotheses)
    t4 = test4_displacement(h1_map, hypotheses[0])
    t5 = test5_pinned_lm_score(h1_map, [corpus_seq] if corpus_seq else [], pin_phonemes)

    _print_report(t1, t2, t3, t4, t5, sign_600_current)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    out = {
        "sign_under_test": TARGET_SIGN,
        "h0001_assignment": sign_600_current,
        "test1_anchor_activation": t1,
        "test2_occupancy": {k: v for k, v in t2.items() if k != "all_rows"},
        "test2_top_occupancy": t2["all_rows"][:12],
        "test3_sign600_variance": t3,
        "test4_displacement": t4,
        "test5_pinned_lm": t5,
    }
    args.output.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info("Diagnosis written → %s", args.output)


if __name__ == "__main__":
    main()
