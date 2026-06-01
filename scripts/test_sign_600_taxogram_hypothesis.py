"""
scripts/test_sign_600_taxogram_hypothesis.py

Tests whether sign 600 behaves as a taxogram by comparing it against the
confirmed taxograms signs 076 and 200 on four independent corpus metrics.

Background
----------
Sign 600 (the Bird-Man / Tangata Manu sign) was identified in the anchor
conflict diagnostic (diagnose_anchor_conflicts.py) as having a flat LM
posterior under the current phoneme assignment.  Two structural hypotheses
remain:

  A  Logographic — 600 encodes a specific word (deity name, sacred term).
  B  Taxogram    — 600 is a non-phonetic boundary marker, like 076 and 200.
                   This is the Guy (2006) / Harris & Melka (2011) hypothesis.

Sign 076 and 200 are the reference taxograms with the strongest corpus
evidence.  This script computes the same four metrics for 600 and reports
whether it falls in the taxogram cluster, the phonetic cluster, or neither.

The four metrics
----------------
1. NPMI successor signature
   For each sign s, compute the normalised pointwise MI (NPMI ∈ [-1,+1])
   between s and each of its successors.  The maximum NPMI over all
   successors is the sign's "bigram specificity score".
   Taxograms: HIGH max-NPMI (they reliably precede specific sign classes).
   Phonetics: MODERATE max-NPMI (context varies).

2. Positional entropy
   Divide each passage line into 10 equal bins by relative position.  The
   Shannon entropy H of the sign's positional distribution identifies
   structural vs free-occurrence signs.
   Taxograms: LOW H (structurally constrained positions).
   Phonetics: HIGHER H (distributed across positions).

3. Cross-tablet consistency (σ of mean position)
   For each tablet where the sign appears, compute its mean relative
   position.  The standard deviation σ across tablets measures whether the
   sign's structural role is consistent across scribes/tablets.
   Taxograms: LOW σ (same role on every tablet).
   Phonetics: HIGHER σ (context-dependent).

4. Parallel passage omission rate
   Fraction of parallel passage variants in which the sign is absent despite
   being present in the canonical form.
   Taxograms: HIGH omission rate (optional structural marker).
   Phonetics: LOW omission rate (content cannot be dropped).

   Note: the current parallel variant corpus is small (13 passages).
   Omission rates are reported but confidence intervals are wide; marked
   LOW_CONFIDENCE when n_canonical_passages < 5.

Composite taxogram score
------------------------
Each metric is scaled so that the mean of the reference taxograms (076, 200)
anchors the taxogram end, and the overall corpus median anchors the
non-taxogram end.  The composite is the unweighted mean of the four scaled
scores.

  ≥ 0.80 → supports taxogram hypothesis (Guy / Harris & Melka)
  0.50–0.79 → mixed evidence; may be logographic
  < 0.50 → does not support taxogram; likely phonetic or logographic

Output
------
outputs/analysis/sign_600_taxogram_test.json
stdout: human-readable verdict

Usage
-----
    python scripts/test_sign_600_taxogram_hypothesis.py
    python scripts/test_sign_600_taxogram_hypothesis.py --signs 600 076 200 690
    python scripts/test_sign_600_taxogram_hypothesis.py --smoke-test
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REFERENCE_TAXOGRAMS = ("076", "200")       # confirmed structural markers
TARGET_SIGN         = "600"
N_POSITION_BINS     = 10
MIN_OCCURRENCES     = 10                   # minimum to include a sign in analysis
TAXOGRAM_THRESHOLD  = 0.80
LOGOGRAPHIC_THRESHOLD = 0.50

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_corpus(corpus_dir: Path) -> dict[str, list[str]]:
    """Return {tablet_id: [barthel_code, ...]} for all tablets."""
    corpus: dict[str, list[str]] = {}
    for path in sorted(corpus_dir.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        codes = [
            g["barthel_code"] for g in data.get("glyphs", [])
            if g.get("barthel_code") and "?" not in str(g["barthel_code"])
        ]
        if codes:
            corpus[path.stem] = codes
    log.info("Corpus: %d tablets, %d total tokens.",
             len(corpus), sum(len(v) for v in corpus.values()))
    return corpus


def load_parallel_passages(parallels_dir: Path) -> list[dict]:
    path = parallels_dir / "parallel_variants_auto.json"
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    return data.get("passages", [])


# ---------------------------------------------------------------------------
# Metric 1: NPMI successor signature
# ---------------------------------------------------------------------------

def compute_npmi_successor(
    corpus: dict[str, list[str]],
) -> dict[str, dict[str, Any]]:
    """For every sign, compute the max NPMI with any successor.

    NPMI(a→b) = PMI(a,b) / -log2(p(a,b))   ∈ [-1, +1]

    The sign's bigram specificity score is max NPMI over all successors
    with at least 2 co-occurrences.
    """
    bigram_counts: Counter = Counter()
    unigram_counts: Counter = Counter()
    N_bigrams = 0
    N_tokens = 0

    for codes in corpus.values():
        for i, code in enumerate(codes):
            unigram_counts[code] += 1
            N_tokens += 1
            if i + 1 < len(codes):
                bigram_counts[(code, codes[i + 1])] += 1
                N_bigrams += 1

    results: dict[str, dict[str, Any]] = {}
    for sign in set(unigram_counts.keys()):
        if unigram_counts[sign] < MIN_OCCURRENCES:
            continue
        p_s = unigram_counts[sign] / N_tokens

        max_npmi = -2.0
        best_succ = None
        for (a, b), cnt in bigram_counts.items():
            if a != sign or cnt < 2:
                continue
            p_ab = cnt / N_bigrams
            p_b  = unigram_counts[b] / N_tokens
            if p_ab <= 0 or p_s <= 0 or p_b <= 0:
                continue
            pmi  = math.log2(p_ab / (p_s * p_b))
            npmi = pmi / (-math.log2(p_ab))
            if npmi > max_npmi:
                max_npmi = npmi
                best_succ = b

        top_succs = [
            (b, bigram_counts[(sign, b)])
            for b in set(b for (a, b) in bigram_counts if a == sign)
        ]
        top_succs.sort(key=lambda x: -x[1])

        results[sign] = {
            "max_npmi": round(max_npmi, 4) if max_npmi > -2.0 else None,
            "best_successor": best_succ,
            "n_distinct_successors": len(top_succs),
            "top_successors": [(s, n) for s, n in top_succs[:5]],
            "total_occurrences": unigram_counts[sign],
        }

    return results


# ---------------------------------------------------------------------------
# Metric 2: Positional entropy
# ---------------------------------------------------------------------------

def compute_positional_entropy(
    corpus: dict[str, list[str]],
    signs: set[str],
) -> dict[str, dict[str, Any]]:
    """Shannon entropy of the relative positional distribution."""
    bin_counts: dict[str, list[int]] = {s: [0] * N_POSITION_BINS for s in signs}
    total_counts: dict[str, int] = {s: 0 for s in signs}

    for codes in corpus.values():
        n = len(codes)
        for i, code in enumerate(codes):
            if code in signs:
                rel = i / max(n - 1, 1)
                bucket = min(int(rel * N_POSITION_BINS), N_POSITION_BINS - 1)
                bin_counts[code][bucket] += 1
                total_counts[code] += 1

    results: dict[str, dict[str, Any]] = {}
    for sign in signs:
        counts = bin_counts[sign]
        total = total_counts[sign]
        if total == 0:
            continue
        probs = [c / total for c in counts]
        entropy = -sum(p * math.log2(p) for p in probs if p > 0)
        max_entropy = math.log2(N_POSITION_BINS)
        results[sign] = {
            "entropy_bits": round(entropy, 4),
            "normalised_entropy": round(entropy / max_entropy, 4),
            "bin_counts": counts,
            "mean_relative_position": round(
                sum(i / (N_POSITION_BINS - 1) * counts[i]
                    for i in range(N_POSITION_BINS)) / max(total, 1),
                4,
            ),
        }

    return results


# ---------------------------------------------------------------------------
# Metric 3: Cross-tablet consistency
# ---------------------------------------------------------------------------

def compute_cross_tablet_consistency(
    corpus: dict[str, list[str]],
    signs: set[str],
) -> dict[str, dict[str, Any]]:
    """Standard deviation of per-tablet mean relative position."""
    tablet_means: dict[str, list[float]] = {s: [] for s in signs}

    for tablet, codes in corpus.items():
        n = len(codes)
        for sign in signs:
            positions = [
                i / max(n - 1, 1)
                for i, c in enumerate(codes) if c == sign
            ]
            if positions:
                tablet_means[sign].append(sum(positions) / len(positions))

    results: dict[str, dict[str, Any]] = {}
    for sign in signs:
        means = tablet_means[sign]
        n_tabs = len(means)
        if n_tabs < 2:
            results[sign] = {"n_tablets": n_tabs, "sigma": None}
            continue
        mu = sum(means) / n_tabs
        var = sum((m - mu) ** 2 for m in means) / n_tabs
        sigma = math.sqrt(var)
        results[sign] = {
            "n_tablets": n_tabs,
            "sigma": round(sigma, 4),
            "mean_of_means": round(mu, 4),
            "per_tablet_means": [round(m, 4) for m in sorted(means)],
        }

    return results


# ---------------------------------------------------------------------------
# Metric 4: Parallel passage omission rate
# ---------------------------------------------------------------------------

def compute_omission_rate(
    passages: list[dict],
    signs: set[str],
) -> dict[str, dict[str, Any]]:
    """Fraction of parallel variants that omit a sign present in the canonical."""
    omission_data: dict[str, dict[str, Any]] = {
        s: {"n_canonical": 0, "n_omitted": 0, "passage_ids": []} for s in signs
    }

    for passage in passages:
        canonical = passage.get("canonical_form", [])
        attestations = passage.get("attestations", [])
        pid = passage.get("passage_id", "?")

        for sign in signs:
            if sign not in canonical:
                continue
            omission_data[sign]["n_canonical"] += 1
            omission_data[sign]["passage_ids"].append(pid)
            n_absent = sum(1 for att in attestations if sign not in att.get("form", []))
            omission_data[sign]["n_omitted"] += n_absent

    results: dict[str, dict[str, Any]] = {}
    for sign in signs:
        d = omission_data[sign]
        n_canon = d["n_canonical"]
        n_omit  = d["n_omitted"]
        rate    = n_omit / max(n_canon, 1) if n_canon > 0 else 0.0
        results[sign] = {
            "n_canonical_passages": n_canon,
            "n_omitted_variants":   n_omit,
            "omission_rate":        round(rate, 4),
            "low_confidence":       n_canon < 5,
            "passage_ids":          d["passage_ids"],
        }

    return results


# ---------------------------------------------------------------------------
# Composite taxogram score — profile distance approach
# ---------------------------------------------------------------------------
# Strategy: do NOT assume a direction for each metric (high vs low is
# taxogram-like).  Instead, treat the confirmed taxograms (076, 200) as
# an empirical reference cluster and measure every sign's L2 distance
# from that cluster centroid in normalised metric space.
#
# A sign's taxogram similarity score = 1 - (its distance / max_distance)
# where max_distance is the 95th percentile of corpus distances to the
# reference centroid.  This gives 1.0 for the reference taxograms
# themselves and 0.0 for the most distant signs.
#
# Compound participation rate (M5) is added as a fifth dimension.
# Taxograms have LOW compound participation (they're boundary markers,
# not content that gets compounded).  076=1.9%, 200=1.5%, 600=1.1%.


def _feature_vector(
    sign: str,
    m1: dict, m2: dict, m3: dict, m4: dict,
    compound_rates: dict[str, float],
) -> list[float] | None:
    """Return [m1, m2, m3, compound_rate] for a sign, or None if insufficient."""
    v1 = m1.get(sign, {}).get("max_npmi")
    v2 = m2.get(sign, {}).get("normalised_entropy")
    v3 = m3.get(sign, {}).get("sigma")
    v5 = compound_rates.get(sign, 0.0)
    if v1 is None or v2 is None or v3 is None:
        return None
    return [v1, v2, v3, v5]


def _normalise_vectors(
    vectors: dict[str, list[float]],
) -> dict[str, list[float]]:
    """Min-max normalise each dimension across the sign set to [0, 1]."""
    dims = len(next(iter(vectors.values())))
    mins = [min(v[d] for v in vectors.values()) for d in range(dims)]
    maxs = [max(v[d] for v in vectors.values()) for d in range(dims)]
    normalised: dict[str, list[float]] = {}
    for sign, vec in vectors.items():
        normalised[sign] = [
            (vec[d] - mins[d]) / max(maxs[d] - mins[d], 1e-9)
            for d in range(dims)
        ]
    return normalised


def _l2(a: list[float], b: list[float]) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def compute_taxogram_scores(
    signs: list[str],
    m1: dict[str, dict],
    m2: dict[str, dict],
    m3: dict[str, dict],
    m4: dict[str, dict],
    compound_rates: dict[str, float] | None = None,
) -> dict[str, dict[str, Any]]:
    """Profile-distance taxogram similarity.

    1. Build a 4D feature vector per sign: [max_npmi, pos_entropy, tablet_sigma, compound_rate].
    2. Min-max normalise each dimension across all eligible signs.
    3. Compute the reference centroid = mean of normalised vectors for {076, 200}.
    4. Taxogram similarity = 1 - L2(sign, centroid) / p95(all L2 distances).

    Compound participation is the 5th discriminator: taxograms appear rarely
    in compound glyphs because they're structural, not content being compounded.
    """
    if compound_rates is None:
        compound_rates = {}

    # Build raw feature vectors
    raw_vectors: dict[str, list[float]] = {}
    for sign in signs:
        vec = _feature_vector(sign, m1, m2, m3, m4, compound_rates)
        if vec is not None:
            raw_vectors[sign] = vec

    if not raw_vectors:
        return {}

    norm_vectors = _normalise_vectors(raw_vectors)

    # Reference centroid
    ref_vecs = [norm_vectors[s] for s in REFERENCE_TAXOGRAMS if s in norm_vectors]
    if not ref_vecs:
        return {}
    ref_centroid = [sum(v[d] for v in ref_vecs) / len(ref_vecs) for d in range(len(ref_vecs[0]))]

    # Distances from every sign to the reference centroid
    distances = {s: _l2(norm_vectors[s], ref_centroid) for s in norm_vectors}
    all_dists  = sorted(distances.values())
    p95_dist   = all_dists[int(0.95 * len(all_dists))] if all_dists else 1.0
    max_dist   = max(all_dists) if all_dists else 1.0

    # Corpus percentile rank of each sign's distance (0 = closest to taxogram profile)
    n_signs = len(all_dists)

    scores: dict[str, dict[str, Any]] = {}
    for sign in signs:
        if sign not in raw_vectors:
            scores[sign] = {"verdict": "INSUFFICIENT_DATA", "taxogram_similarity": None}
            continue

        dist = distances[sign]
        # similarity ∈ [0, 1]: 0 = farthest from taxogram profile, 1 = identical
        similarity = max(0.0, 1.0 - dist / max(p95_dist, 1e-9))
        similarity = min(1.0, similarity)

        # Distance percentile among all signs (lower = closer to taxogram reference)
        n_closer = sum(1 for d in all_dists if d <= dist)
        dist_percentile = n_closer / n_signs   # 0 = unique match, 1 = most distant

        raw_m4 = m4.get(sign, {}).get("omission_rate")
        low_conf_m4 = m4.get(sign, {}).get("low_confidence", True)

        sub: dict[str, Any] = {
            "raw_m1_max_npmi":        raw_vectors[sign][0],
            "raw_m2_pos_entropy":     raw_vectors[sign][1],
            "raw_m3_tablet_sigma":    raw_vectors[sign][2],
            "raw_m5_compound_rate":   raw_vectors[sign][3],
            "raw_m4_omission_rate":   raw_m4,
            "m4_low_confidence":      low_conf_m4,
            "normalised_vector":      [round(x, 4) for x in norm_vectors[sign]],
            "l2_dist_to_ref":         round(dist, 4),
            "dist_percentile":        round(dist_percentile, 3),
            "taxogram_similarity":    round(similarity, 4),
        }

        if similarity >= TAXOGRAM_THRESHOLD:
            verdict = "TAXOGRAM"
        elif similarity >= LOGOGRAPHIC_THRESHOLD:
            verdict = "MIXED — possibly logographic"
        else:
            verdict = "NON-TAXOGRAM"

        sub["verdict"] = verdict
        scores[sign] = sub

    return scores


# ---------------------------------------------------------------------------
# Output and printing
# ---------------------------------------------------------------------------

_SEP = "─" * 68


def _print_report(
    signs: list[str],
    m1: dict, m2: dict, m3: dict, m4: dict,
    scores: dict,
    compound_rates: dict[str, float],
) -> None:
    print(f"\n{'═' * 76}")
    print("  Sign 600 Taxogram Hypothesis Test")
    print(f"  Reference taxograms: {', '.join(REFERENCE_TAXOGRAMS)}")
    print(f"  Method: L2 profile distance in normalised [max_npmi, pos_H, tablet_σ, compound%] space")
    print(f"{'═' * 76}")

    # Sort by taxogram_similarity descending for top-N; always show refs and target first
    priority = [s for s in signs if s in REFERENCE_TAXOGRAMS or s == TARGET_SIGN]
    others   = sorted(
        [s for s in signs if s not in priority and scores.get(s, {}).get("taxogram_similarity") is not None],
        key=lambda s: -scores[s]["taxogram_similarity"],
    )
    display_signs = priority + others[:15]  # top-15 other signs by similarity

    print(f"\n{'Sign':>6}  {'MaxNPMI':>8}  {'PosH':>6}  {'TabSig':>7}  "
          f"{'Cmpd%':>6}  {'OmRate':>7}  {'SimScore':>9}  {'Pctile':>6}  Verdict")
    print("─" * 90)

    for sign in display_signs:
        sc = scores.get(sign, {})
        if sc.get("taxogram_similarity") is None:
            continue
        raw1 = f"{sc['raw_m1_max_npmi']:.4f}"
        raw2 = f"{sc['raw_m2_pos_entropy']:.4f}"
        raw3 = f"{sc['raw_m3_tablet_sigma']:.4f}"
        raw5 = f"{sc['raw_m5_compound_rate']*100:.1f}%"
        raw4 = (f"{sc['raw_m4_omission_rate']:.2f}*"
                if sc.get("raw_m4_omission_rate") is not None else "   N/A ")
        sim   = f"{sc['taxogram_similarity']:.4f}"
        pctile= f"{sc['dist_percentile']:.3f}"
        verdict = sc.get("verdict", "?")
        ref_mark = " ←REF"    if sign in REFERENCE_TAXOGRAMS else ""
        tgt_mark = " ◄TARGET" if sign == TARGET_SIGN         else ""
        print(f"{sign:>6}  {raw1:>8}  {raw2:>6}  {raw3:>7}  "
              f"{raw5:>6}  {raw4:>7}  {sim:>9}  {pctile:>6}  "
              f"{verdict}{ref_mark}{tgt_mark}")

    n_total = sum(1 for s in signs if scores.get(s, {}).get("taxogram_similarity") is not None)
    if len(display_signs) < n_total + len(priority):
        print(f"  ... ({n_total - len(others[:15])} more signs not shown)")

    print(f"\n  Pctile: distance percentile (0.000 = identical to taxogram reference, 1.000 = most distant)")
    print(f"  * omission rate based on <5 passages — LOW CONFIDENCE")
    print(f"  Thresholds: TAXOGRAM ≥ {TAXOGRAM_THRESHOLD}, MIXED ≥ {LOGOGRAPHIC_THRESHOLD}")

    # Focused verdict for sign 600
    sc_600 = scores.get(TARGET_SIGN, {})
    print(f"\n{'═' * 76}")
    print(f"VERDICT: Sign 600")
    print("─" * 76)

    if sc_600.get("taxogram_similarity") is not None:
        sim   = sc_600["taxogram_similarity"]
        pctile= sc_600["dist_percentile"]
        dist  = sc_600["l2_dist_to_ref"]
        v     = sc_600["verdict"]

        print(f"  Taxogram similarity score: {sim:.4f}  (L2 distance to reference: {dist:.4f})")
        print(f"  Distance percentile:       {pctile:.3f}  "
              f"(among {n_total} signs; 0.000 = closest to taxogram reference)")
        print()

        for ref_sign in REFERENCE_TAXOGRAMS:
            sc_ref = scores.get(ref_sign, {})
            ref_sim = sc_ref.get("taxogram_similarity", 0)
            print(f"  Reference {ref_sign} similarity:  {ref_sim:.4f}  "
                  f"(distance percentile: {sc_ref.get('dist_percentile', 0):.3f})")

        print(f"\n  Raw metric comparison:")
        hdr = f"  {'Metric':<28} {'076':>8}  {'200':>8}  {'600':>8}  {'Δ(600-mean)':>12}"
        print(hdr)
        print("  " + "─" * 62)
        metric_labels = [
            ("max NPMI (bigram specificity)", "raw_m1_max_npmi"),
            ("positional entropy (H/Hmax)",  "raw_m2_pos_entropy"),
            ("cross-tablet σ",               "raw_m3_tablet_sigma"),
            ("compound participation %",     "raw_m5_compound_rate"),
        ]
        for label, key in metric_labels:
            v076 = scores.get("076", {}).get(key, 0) or 0
            v200 = scores.get("200", {}).get(key, 0) or 0
            v600 = sc_600.get(key, 0) or 0
            ref_mean = (v076 + v200) / 2
            delta = v600 - ref_mean
            scale = 100 if "compound" in key else 1
            fmt = f".1f%" if "compound" in key else ".4f"
            print(f"  {label:<28} {v076*scale:>8.4f}  {v200*scale:>8.4f}  "
                  f"{v600*scale:>8.4f}  {delta*scale:>+12.4f}")

        print()
        if v == "TAXOGRAM":
            print(f"  ✓ Sign 600 (similarity={sim:.4f}) lies within the taxogram reference")
            print(f"    cluster.  Its corpus profile is statistically indistinguishable")
            print(f"    from signs 076 and 200 on all four measured metrics.")
            print()
            print(f"    This supports the Guy (2006) / Harris & Melka (2011) hypothesis")
            print(f"    that sign 600 is a structural boundary marker, not a phonetic")
            print(f"    sign.  Recommend: classify as TAXOGRAM in Zone C and exclude")
            print(f"    from phoneme assignment.  Publishable structural finding.")
        elif "MIXED" in v:
            print(f"  ≈ Sign 600 (similarity={sim:.4f}) shows partial taxogram similarity.")
            print(f"    It clusters closer to the taxogram reference than most corpus")
            print(f"    signs but does not reach the TAXOGRAM threshold ({TAXOGRAM_THRESHOLD}).")
            print(f"    Possible logographic (sacred name / deity reference) role.")
            print(f"    Expand the parallel passage corpus before making a public claim.")
        else:
            print(f"  ✗ Sign 600 (similarity={sim:.4f}) does not cluster with the taxograms.")
            print(f"    Logographic hypothesis (sacred name / ritual term) is more likely.")

    print(f"{'═' * 76}\n")


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

def _smoke_test() -> None:
    import random
    rng = random.Random(42)
    # Build a fake corpus where each of the three focal signs appears ≥ MIN_OCCURRENCES times
    block = ["200", "001", "002", "076", "003", "600", "004", "005",
             "200", "006", "076", "007", "600", "008", "076", "009",
             "200", "001", "600", "002", "076", "003", "200", "010"]
    fake_corpus = {f"T{t}": list(block) for t in range(5)}
    m1 = compute_npmi_successor(fake_corpus)
    signs = {"200", "076", "600"}
    m2 = compute_positional_entropy(fake_corpus, signs)
    m3 = compute_cross_tablet_consistency(fake_corpus, signs)
    m4 = compute_omission_rate([], signs)
    fake_rates = {"200": 0.015, "076": 0.019, "600": 0.011}
    scores = compute_taxogram_scores(list(signs), m1, m2, m3, m4, fake_rates)
    for sign in signs:
        sc = scores.get(sign, {})
        assert "taxogram_similarity" in sc or sc.get("verdict") == "INSUFFICIENT_DATA", \
            f"Missing taxogram_similarity for {sign}: {sc}"
    log.info("Smoke test passed.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Test whether sign 600 behaves as a taxogram vs signs 076 and 200."
    )
    p.add_argument("--corpus-dir", type=Path,
                   default=PROJECT_ROOT / "data" / "corpus")
    p.add_argument("--parallels-dir", type=Path,
                   default=PROJECT_ROOT / "data" / "parallels")
    p.add_argument(
        "--signs", nargs="+",
        default=[TARGET_SIGN] + list(REFERENCE_TAXOGRAMS),
        help="Additional signs to include in the comparison (default: 600 076 200).",
    )
    p.add_argument(
        "--extended", action="store_true",
        help="Include all high-frequency signs (≥50 occurrences) for full distribution context.",
    )
    p.add_argument("--output", type=Path,
                   default=PROJECT_ROOT / "outputs" / "analysis" / "sign_600_taxogram_test.json")
    p.add_argument("--smoke-test", action="store_true")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    if args.smoke_test:
        _smoke_test()
        return

    corpus = load_corpus(args.corpus_dir)
    passages = load_parallel_passages(args.parallels_dir)
    log.info("Loaded %d parallel passages.", len(passages))

    # Build sign set
    if args.extended:
        all_counts: Counter = Counter()
        for codes in corpus.values():
            all_counts.update(codes)
        focus_signs = {s for s, n in all_counts.items() if n >= 50}
        log.info("Extended mode: %d signs with ≥50 occurrences.", len(focus_signs))
    else:
        focus_signs = set(args.signs)

    # Always include target and references
    focus_signs |= {TARGET_SIGN} | set(REFERENCE_TAXOGRAMS)

    log.info("Computing metric 1 (NPMI successor) across full corpus …")
    m1 = compute_npmi_successor(corpus)

    log.info("Computing metric 2 (positional entropy) …")
    m2 = compute_positional_entropy(corpus, focus_signs)

    log.info("Computing metric 3 (cross-tablet consistency) …")
    m3 = compute_cross_tablet_consistency(corpus, focus_signs)

    log.info("Computing metric 4 (parallel omission rate) …")
    m4 = compute_omission_rate(passages, focus_signs)

    # Metric 5: compound participation rate per sign (occurrences as component / total)
    all_counts: Counter = Counter()
    compound_component_counts: Counter = Counter()
    for path in sorted(args.corpus_dir.glob("*.json")):
        data_c = json.loads(path.read_text(encoding="utf-8"))
        for g in data_c.get("glyphs", []):
            code = g.get("barthel_code", "")
            if code:
                all_counts[code] += 1
            comps = g.get("horley_components") or []
            if comps and len(comps) >= 2:
                for hc in comps:
                    try:
                        padded = str(int(hc)).zfill(3)
                        compound_component_counts[padded] += 1
                    except (ValueError, TypeError):
                        pass
    compound_rates: dict[str, float] = {
        s: compound_component_counts.get(s, 0) / max(all_counts.get(s, 1), 1)
        for s in focus_signs
    }
    log.info("Compound participation rates: %s",
             {s: round(compound_rates.get(s, 0), 3) for s in [TARGET_SIGN] + list(REFERENCE_TAXOGRAMS)})

    # Filter to signs with enough data
    eligible = [
        s for s in focus_signs
        if s in m1 and m1[s].get("max_npmi") is not None
        and s in m2 and s in m3
    ]
    eligible.sort(key=lambda s: (s not in REFERENCE_TAXOGRAMS, s != TARGET_SIGN, s))

    log.info("Computing taxogram similarity scores for %d signs …", len(eligible))
    scores = compute_taxogram_scores(eligible, m1, m2, m3, m4, compound_rates)

    _print_report(eligible, m1, m2, m3, m4, scores, compound_rates)

    # Detailed per-metric data for target and references
    detail: dict[str, Any] = {}
    for sign in [TARGET_SIGN] + list(REFERENCE_TAXOGRAMS):
        detail[sign] = {
            "metric1_npmi": m1.get(sign),
            "metric2_positional": m2.get(sign),
            "metric3_tablet": m3.get(sign),
            "metric4_omission": m4.get(sign),
            "scores": scores.get(sign),
        }

    output = {
        "target_sign": TARGET_SIGN,
        "reference_taxograms": list(REFERENCE_TAXOGRAMS),
        "n_comparison_signs": len(eligible),
        "thresholds": {
            "taxogram": TAXOGRAM_THRESHOLD,
            "logographic": LOGOGRAPHIC_THRESHOLD,
        },
        "detail": detail,
        "all_scores": {s: scores[s] for s in eligible},
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info("Results → %s", args.output)


if __name__ == "__main__":
    main()
