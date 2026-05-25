"""
Zone B astronomical analysis — five independent statistical tests of the
hypothesis that certain rongorongo signs encode astronomical referents
(stars, constellations, the Polynesian Navigator's Triangle).

Five tests (each scores 0.0–1.0):
  1. Positional entropy       — structured signs appear in constrained positions
  2. Mamari calendar anchor   — signs exclusive or predominant in Ca6–Ca9
  3. Cross-tablet stability   — low Jensen-Shannon divergence across tablets
  4. Dietrich correspondence  — lookup-table match + statistical validation
  5. Tablet D specificity     — overrepresentation on the oldest pre-contact tablet

Data sources:
  corpus_dir/   — per-tablet JSON corpus files (required)
  embeddings    — Zone A autoencoder embeddings cache (optional)

Output: outputs/zone_b/astronomical_candidates.json
"""

from __future__ import annotations

import json
import logging
import math
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Astronomical sign correspondence table
# ---------------------------------------------------------------------------
# Proposed meanings from Barthel (1958), Guy, Metoro/Jaussen recitations,
# and recent synthesis.  All entries are speculative — not validated
# decipherments.  Source note preserved verbatim from the project catalog.

DIETRICH_TABLE: list[dict[str, str]] = [
    {
        "barthel_code": "008",
        "proposed_referent": "sun/star (raʻa/hetuʻu)",
        "polynesian_name": "raʻa / hetuʻu",
        "western_equivalent": "sun or generic star",
        "source": "Barthel (1958); Guy; Metoro/Jaussen recitations",
        "confidence": "medium",
        "type": "celestial_body",
    },
    {
        "barthel_code": "010",
        "proposed_referent": "moon (mahina)",
        "polynesian_name": "mahina",
        "western_equivalent": "Moon",
        "source": "Barthel (1958); Guy; Metoro/Jaussen recitations",
        "confidence": "high",
        "type": "celestial_body",
    },
    {
        "barthel_code": "040",
        "proposed_referent": "night-count marker (Kokore)",
        "polynesian_name": "Kokore",
        "western_equivalent": "Lunar month night marker",
        "source": "Barthel (1958) calendar commentary",
        "confidence": "high",
        "type": "calendar",
    },
    {
        "barthel_code": "074",
        "proposed_referent": "first quarter moon (Hua)",
        "polynesian_name": "Hua",
        "western_equivalent": "Lunar month first quarter",
        "source": "Barthel (1958); Guy; Metoro/Jaussen recitations",
        "confidence": "medium",
        "type": "calendar",
    },
    {
        "barthel_code": "078",
        "proposed_referent": "waning gibbous (Maure)",
        "polynesian_name": "Maure",
        "western_equivalent": "Lunar month waning gibbous",
        "source": "Barthel (1958); Guy; Metoro/Jaussen recitations",
        "confidence": "medium",
        "type": "calendar",
    },
    {
        "barthel_code": "143",
        "proposed_referent": "night before full moon (Rakau)",
        "polynesian_name": "Rakau",
        "western_equivalent": "Lunar month — night 14",
        "source": "Barthel (1958); Guy; Metoro/Jaussen recitations",
        "confidence": "medium",
        "type": "calendar",
    },
    {
        "barthel_code": "152",
        "proposed_referent": "full moon (Omotohi)",
        "polynesian_name": "Omotohi",
        "western_equivalent": "Full moon",
        "source": "Barthel (1958); Guy; Metoro/Jaussen recitations",
        "confidence": "high",
        "type": "celestial_body",
    },
    {
        "barthel_code": "280",
        "proposed_referent": "turtle (honu) = dark/new moon",
        "polynesian_name": "honu",
        "western_equivalent": "New moon (disappearing moon metaphor)",
        "source": "Metoro identification; recent synthesis",
        "confidence": "high",
        "type": "celestial_body",
        "note": "Turtle metaphor for disappearing moon (Metoro recitation)",
    },
    {
        "barthel_code": "385",
        "proposed_referent": "waning crescent",
        "polynesian_name": "—",
        "western_equivalent": "Lunar month waning crescent",
        "source": "Barthel (1958); Guy; Metoro/Jaussen recitations",
        "confidence": "low",
        "type": "calendar",
    },
    {
        "barthel_code": "600",
        "proposed_referent": "bird/frigatebird (manu)",
        "polynesian_name": "manu",
        "western_equivalent": "Bird-named constellations (via Dietrich)",
        "source": "Metoro recitation; Dietrich (2007)",
        "confidence": "high",
        "type": "iconographic",
        "note": "Metoro recitation; may also map to bird-named constellations via Dietrich",
    },
    {
        "barthel_code": "690",
        "proposed_referent": "Bird-Man (Tangata Manu)",
        "polynesian_name": "Tangata Manu",
        "western_equivalent": "Easter Island Bird-Man cult figure",
        "source": "Barthel (1958); recent synthesis",
        "confidence": "high",
        "type": "iconographic",
    },
]

# Barthel codes for the bird-headed family (600–699 base codes only)
_BIRD_HEADED_PREFIX = range(600, 700)

# Mamari calendar lines: side 'a', lines 06–09
_MAMARI_TABLET_ID = "C"
_CALENDAR_SIDE = "a"
_CALENDAR_LINES = {"06", "07", "08", "09"}

# Minimum occurrences for a sign to be evaluated
_MIN_OCCURRENCES = 5


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class TestResult:
    score: float | None  # 0.0–1.0; None = test could not run
    detail: dict[str, Any] = field(default_factory=dict)


@dataclass
class AstronomicalCandidate:
    barthel_code: str
    n_methods_flagged: int
    overall_score: float
    test1_positional_entropy: TestResult
    test2_calendar_anchor: TestResult
    test3_tablet_stability: TestResult
    test4_dietrich_match: TestResult
    test5_tablet_d_specificity: TestResult
    dietrich_entry: dict[str, str] | None = None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        # Flatten TestResult objects for JSON readability
        for key in (
            "test1_positional_entropy",
            "test2_calendar_anchor",
            "test3_tablet_stability",
            "test4_dietrich_match",
            "test5_tablet_d_specificity",
        ):
            d[key] = {"score": d[key]["score"], **d[key]["detail"]}
        return d


# ---------------------------------------------------------------------------
# Corpus loading helpers
# ---------------------------------------------------------------------------


def _load_corpus(corpus_dir: Path) -> dict[str, list[dict[str, Any]]]:
    """Return {tablet_id: [glyph_dict, ...]} for every single-letter tablet."""
    corpus: dict[str, list[dict[str, Any]]] = {}
    for path in sorted(corpus_dir.glob("[A-Z].json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        tablet_id = data.get("tablet_id", path.stem)
        glyphs = [
            g for g in data.get("glyphs", [])
            if g.get("barthel_code") and g["barthel_code"] != "?"
        ]
        if glyphs:
            corpus[tablet_id] = glyphs
    log.info("Loaded %d tablets.", len(corpus))
    return corpus


def _barthel_base_num(code: str) -> int | None:
    """Return the integer prefix of a Barthel code, or None if unparseable."""
    digits = "".join(c for c in code.split(".")[0] if c.isdigit())
    try:
        return int(digits) if digits else None
    except ValueError:
        return None


def _is_bird_headed(code: str) -> bool:
    num = _barthel_base_num(code)
    return num is not None and 600 <= num <= 699


# ---------------------------------------------------------------------------
# Test 1: Positional entropy
# ---------------------------------------------------------------------------


def _positional_entropy(
    corpus: dict[str, list[dict[str, Any]]],
    code: str,
    n_bins: int = 10,
) -> float | None:
    """Shannon entropy (bits) of the relative positional distribution of `code`."""
    positions: list[float] = []
    for glyphs in corpus.values():
        seq_codes = [g["barthel_code"] for g in glyphs]
        n = len(seq_codes)
        for i, c in enumerate(seq_codes):
            if c == code:
                positions.append(i / max(n - 1, 1))
    if len(positions) < _MIN_OCCURRENCES:
        return None
    counts = np.zeros(n_bins)
    for p in positions:
        bucket = min(int(p * n_bins), n_bins - 1)
        counts[bucket] += 1
    counts = counts / counts.sum()
    h = -float(np.sum(counts[counts > 0] * np.log2(counts[counts > 0])))
    return h


def test_positional_entropy(
    corpus: dict[str, list[dict[str, Any]]],
    target_codes: list[str],
    n_bins: int = 10,
) -> dict[str, TestResult]:
    """Test 1: compare positional entropy of each target sign to the corpus baseline."""
    # Baseline: mean positional entropy of all signs with ≥ MIN_OCCURRENCES
    all_codes: Counter[str] = Counter()
    for glyphs in corpus.values():
        all_codes.update(g["barthel_code"] for g in glyphs)
    baseline_codes = [c for c, n in all_codes.items() if n >= _MIN_OCCURRENCES]

    baseline_entropies: list[float] = []
    for c in baseline_codes:
        h = _positional_entropy(corpus, c, n_bins)
        if h is not None:
            baseline_entropies.append(h)

    baseline_mean = float(np.mean(baseline_entropies)) if baseline_entropies else math.log2(n_bins)
    baseline_std = float(np.std(baseline_entropies)) if len(baseline_entropies) > 1 else 1.0
    log.info("Test 1 baseline: mean H=%.3f bits, std=%.3f, n=%d signs.",
             baseline_mean, baseline_std, len(baseline_entropies))

    results: dict[str, TestResult] = {}
    for code in target_codes:
        h = _positional_entropy(corpus, code, n_bins)
        if h is None:
            results[code] = TestResult(score=None, detail={"reason": "too few occurrences"})
            continue
        # z-score: how many SDs below the baseline mean?
        z = (baseline_mean - h) / max(baseline_std, 1e-9)
        # Sigmoid mapping: z=0 → 0.5, z=2 → 0.88, z=-2 → 0.12
        score = float(1.0 / (1.0 + math.exp(-z)))
        results[code] = TestResult(
            score=score,
            detail={
                "positional_entropy_bits": round(h, 4),
                "baseline_mean_bits": round(baseline_mean, 4),
                "z_score": round(z, 3),
                "n_occurrences": int(all_codes[code]),
            },
        )
        log.info("  Test 1 [%s]: H=%.3f bits  z=%.2f  score=%.3f", code, h, z, score)
    return results


# ---------------------------------------------------------------------------
# Test 2: Mamari calendar anchor
# ---------------------------------------------------------------------------


def test_mamari_calendar(
    corpus: dict[str, list[dict[str, Any]]],
    target_codes: list[str],
) -> dict[str, TestResult]:
    """Test 2: fraction of Mamari appearances that fall in the calendar section (Ca6–Ca9)."""
    mamari = corpus.get(_MAMARI_TABLET_ID, [])
    if not mamari:
        log.warning("Test 2: Mamari (tablet C) not found in corpus — skipping.")
        return {c: TestResult(score=None, detail={"reason": "Mamari not in corpus"})
                for c in target_codes}

    cal_codes: Counter[str] = Counter()
    non_cal_codes: Counter[str] = Counter()
    for g in mamari:
        code = g["barthel_code"]
        side = str(g.get("side", "a"))
        line = str(g.get("line", "00")).zfill(2)
        if side == _CALENDAR_SIDE and line in _CALENDAR_LINES:
            cal_codes[code] += 1
        else:
            non_cal_codes[code] += 1

    calendar_only = {c for c in cal_codes if c not in non_cal_codes}
    log.info("Test 2: %d calendar glyphs, %d calendar-only codes.",
             sum(cal_codes.values()), len(calendar_only))

    results: dict[str, TestResult] = {}
    for code in target_codes:
        n_cal = cal_codes[code]
        n_non = non_cal_codes[code]
        total_mamari = n_cal + n_non
        if total_mamari == 0:
            results[code] = TestResult(score=0.0, detail={
                "n_calendar": 0, "n_non_calendar": 0, "calendar_fraction": 0.0,
            })
            continue
        frac = n_cal / total_mamari
        # Score: calendar-exclusive = 1.0, calendar-absent = 0.0
        score = frac
        results[code] = TestResult(
            score=score,
            detail={
                "n_calendar": n_cal,
                "n_non_calendar": n_non,
                "calendar_fraction": round(frac, 4),
                "calendar_exclusive": code in calendar_only,
            },
        )
        log.info("  Test 2 [%s]: cal=%d non_cal=%d frac=%.3f score=%.3f",
                 code, n_cal, n_non, frac, score)
    return results


# ---------------------------------------------------------------------------
# Test 3: Cross-tablet stability (Jensen-Shannon divergence)
# ---------------------------------------------------------------------------


def _positional_dist(
    glyphs: list[dict[str, Any]],
    code: str,
    n_bins: int = 10,
) -> np.ndarray | None:
    """Positional distribution of `code` within one tablet's glyph list."""
    n = len(glyphs)
    counts = np.zeros(n_bins)
    for i, g in enumerate(glyphs):
        if g["barthel_code"] == code:
            bucket = min(int(i / max(n - 1, 1) * n_bins), n_bins - 1)
            counts[bucket] += 1
    if counts.sum() == 0:
        return None
    return counts / counts.sum()


def _js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """Jensen-Shannon divergence (base-2); always in [0, 1]."""
    m = 0.5 * (p + q)
    eps = 1e-12

    def kl(a: np.ndarray, b: np.ndarray) -> float:
        mask = a > eps
        return float(np.sum(a[mask] * np.log2(a[mask] / b[mask])))

    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def test_tablet_stability(
    corpus: dict[str, list[dict[str, Any]]],
    target_codes: list[str],
    min_tablets: int = 3,
    n_bins: int = 10,
) -> dict[str, TestResult]:
    """Test 3: mean pairwise JS divergence of positional distributions across tablets."""
    results: dict[str, TestResult] = {}
    for code in target_codes:
        dists: list[tuple[str, np.ndarray]] = []
        for tablet_id, glyphs in corpus.items():
            dist = _positional_dist(glyphs, code, n_bins)
            if dist is not None:
                dists.append((tablet_id, dist))

        n_tablets = len(dists)
        if n_tablets < min_tablets:
            results[code] = TestResult(score=None, detail={
                "n_tablets": n_tablets,
                "reason": f"appears on fewer than {min_tablets} tablets",
            })
            continue

        js_values: list[float] = []
        for i in range(n_tablets):
            for j in range(i + 1, n_tablets):
                js_values.append(_js_divergence(dists[i][1], dists[j][1]))

        mean_js = float(np.mean(js_values))
        # score: JS=0 (identical distributions) → 1.0; JS=1 (max divergence) → 0.0
        score = 1.0 - mean_js
        results[code] = TestResult(
            score=score,
            detail={
                "n_tablets": n_tablets,
                "tablets": [t for t, _ in dists],
                "mean_js_divergence": round(mean_js, 4),
                "n_pairs": len(js_values),
            },
        )
        log.info("  Test 3 [%s]: %d tablets  mean_JS=%.3f  score=%.3f",
                 code, n_tablets, mean_js, score)
    return results


# ---------------------------------------------------------------------------
# Test 4: Dietrich lookup matching
# ---------------------------------------------------------------------------


def test_dietrich_match(
    target_codes: list[str],
) -> dict[str, TestResult]:
    """Test 4: is this sign in the Dietrich correspondence table?"""
    lookup = {entry["barthel_code"]: entry for entry in DIETRICH_TABLE}
    results: dict[str, TestResult] = {}
    for code in target_codes:
        entry = lookup.get(code)
        if entry is None:
            results[code] = TestResult(score=0.0, detail={"in_dietrich_table": False})
        else:
            conf_scores = {"high": 0.8, "medium": 0.5, "low": 0.2}
            score = conf_scores.get(entry["confidence"], 0.3)
            results[code] = TestResult(
                score=score,
                detail={
                    "in_dietrich_table": True,
                    "proposed_referent": entry["proposed_referent"],
                    "polynesian_name": entry["polynesian_name"],
                    "western_equivalent": entry["western_equivalent"],
                    "source": entry["source"],
                    "confidence": entry["confidence"],
                },
            )
            log.info("  Test 4 [%s]: %s  score=%.2f", code,
                     entry["proposed_referent"], score)
    return results


# ---------------------------------------------------------------------------
# Test 5: Tablet D specificity
# ---------------------------------------------------------------------------


def test_tablet_d_specificity(
    corpus: dict[str, list[dict[str, Any]]],
    target_codes: list[str],
    tablet_d_id: str = "D",
) -> dict[str, TestResult]:
    """Test 5: overrepresentation of each sign on Tablet D (oldest pre-contact tablet)."""
    if tablet_d_id not in corpus:
        log.warning("Test 5: Tablet D not found in corpus — skipping.")
        return {c: TestResult(score=None, detail={"reason": "Tablet D not in corpus"})
                for c in target_codes}

    d_counts: Counter[str] = Counter(g["barthel_code"] for g in corpus[tablet_d_id])
    n_d = sum(d_counts.values())

    all_counts: Counter[str] = Counter()
    n_total = 0
    for glyphs in corpus.values():
        all_counts.update(g["barthel_code"] for g in glyphs)
        n_total += len(glyphs)

    log.info("Test 5: Tablet D has %d tokens (%.1f%% of corpus).",
             n_d, 100.0 * n_d / max(n_total, 1))

    results: dict[str, TestResult] = {}
    for code in target_codes:
        f_d = d_counts[code]
        f_total = all_counts[code]
        if f_total == 0:
            results[code] = TestResult(score=0.0, detail={
                "f_tablet_d": 0, "f_total": 0, "ratio": 0.0,
            })
            continue

        rate_d = f_d / max(n_d, 1)
        rate_total = f_total / max(n_total, 1)
        ratio = rate_d / max(rate_total, 1e-12)

        # Score: ratio=1 (corpus-average) → 0.0; ratio=5 → 1.0; clipped
        score = float(min(1.0, max(0.0, (ratio - 1.0) / 4.0)))
        results[code] = TestResult(
            score=score,
            detail={
                "f_tablet_d": int(f_d),
                "f_total": int(f_total),
                "rate_tablet_d_per_k": round(rate_d * 1000, 3),
                "rate_corpus_per_k": round(rate_total * 1000, 3),
                "ratio": round(ratio, 3),
            },
        )
        log.info("  Test 5 [%s]: f_D=%d f_total=%d ratio=%.2f score=%.3f",
                 code, f_d, f_total, ratio, score)
    return results


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def run_all_tests(
    corpus_dir: Path,
    embeddings_path: Path | None = None,
    min_methods: int = 2,
    output_path: Path | None = None,
) -> dict[str, Any]:
    """Run all five astronomical tests and return the result dict.

    Parameters
    ----------
    corpus_dir : Path
        Directory containing per-tablet corpus JSON files.
    embeddings_path : Path or None
        Optional Zone A embeddings cache.  Currently unused but reserved
        for future embedding-space checks.
    min_methods : int
        Minimum number of tests (score > 0.5) for a sign to appear in the
        candidate list.
    output_path : Path or None
        If given, write the JSON result here.

    Returns
    -------
    dict with keys ``generated``, ``n_candidates``, ``candidates``, ``tests``.
    """
    corpus = _load_corpus(corpus_dir)

    # Candidate pool: Dietrich signs + all bird-headed signs with ≥ MIN_OCCURRENCES
    all_counts: Counter[str] = Counter()
    for glyphs in corpus.values():
        all_counts.update(g["barthel_code"] for g in glyphs)

    dietrich_codes = {e["barthel_code"] for e in DIETRICH_TABLE}
    bird_codes = {
        c for c, n in all_counts.items()
        if _is_bird_headed(c) and n >= _MIN_OCCURRENCES
    }
    target_codes = sorted(dietrich_codes | bird_codes)
    log.info("Candidate pool: %d codes (%d Dietrich + %d bird-headed).",
             len(target_codes), len(dietrich_codes), len(bird_codes))

    # Run tests
    t1 = test_positional_entropy(corpus, target_codes)
    t2 = test_mamari_calendar(corpus, target_codes)
    t3 = test_tablet_stability(corpus, target_codes)
    t4 = test_dietrich_match(target_codes)
    t5 = test_tablet_d_specificity(corpus, target_codes)

    dietrich_lookup = {e["barthel_code"]: e for e in DIETRICH_TABLE}

    candidates: list[AstronomicalCandidate] = []
    for code in target_codes:
        scores = [
            r.score for r in (t1[code], t2[code], t3[code], t4[code], t5[code])
            if r.score is not None
        ]
        n_flagged = sum(1 for s in scores if s > 0.5)
        overall = float(np.mean(scores)) if scores else 0.0

        cand = AstronomicalCandidate(
            barthel_code=code,
            n_methods_flagged=n_flagged,
            overall_score=round(overall, 4),
            test1_positional_entropy=t1[code],
            test2_calendar_anchor=t2[code],
            test3_tablet_stability=t3[code],
            test4_dietrich_match=t4[code],
            test5_tablet_d_specificity=t5[code],
            dietrich_entry=dietrich_lookup.get(code),
        )
        candidates.append(cand)

    candidates.sort(key=lambda c: (-c.n_methods_flagged, -c.overall_score))
    top = [c for c in candidates if c.n_methods_flagged >= min_methods]
    log.info("Candidates: %d total, %d with ≥%d methods flagged.",
             len(candidates), len(top), min_methods)

    from datetime import datetime, timezone
    result: dict[str, Any] = {
        "generated": datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "n_candidates": len(top),
        "n_evaluated": len(candidates),
        "min_methods_threshold": min_methods,
        "tests": {
            "test1": "positional_entropy — constrained position vs corpus baseline",
            "test2": "mamari_calendar_anchor — sign prevalence in Ca6–Ca9",
            "test3": "tablet_stability — Jensen-Shannon divergence across tablets",
            "test4": "dietrich_match — correspondence table (Dietrich 2007 / Fischer 1997)",
            "test5": "tablet_d_specificity — overrepresentation on oldest pre-contact tablet",
        },
        "candidates": [c.to_dict() for c in top],
        "all_evaluated": [c.to_dict() for c in candidates],
    }

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False),
                               encoding="utf-8")
        log.info("Astronomical candidates written: %d → %s", len(top), output_path)

    return result


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def _parse_args():
    import argparse

    p = argparse.ArgumentParser(
        description="Run Zone B astronomical analysis on the rongorongo corpus."
    )
    p.add_argument(
        "--corpus-dir",
        type=Path,
        default=None,
        help="Path to corpus JSON directory (default: from config).",
    )
    p.add_argument(
        "--embeddings",
        type=Path,
        default=None,
        metavar="PATH",
        help="Zone A embeddings cache .pt file (optional).",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSON path (default: outputs/zone_b/astronomical_candidates.json).",
    )
    p.add_argument(
        "--min-methods",
        type=int,
        default=2,
        metavar="N",
        help="Minimum test methods flagged (score > 0.5) for inclusion (default: 2).",
    )
    return p.parse_args()


def main() -> None:
    import logging as _logging
    _logging.basicConfig(level=_logging.INFO, format="%(levelname)s  %(message)s")

    from omegaconf import OmegaConf
    args = _parse_args()

    cfg = OmegaConf.load(PROJECT_ROOT / "conf" / "config.yaml")
    corpus_dir = args.corpus_dir or (PROJECT_ROOT / cfg.paths.corpus_dir)
    out_path = args.output or (
        PROJECT_ROOT / cfg.paths.outputs_dir / "zone_b" / "astronomical_candidates.json"
    )

    run_all_tests(
        corpus_dir=corpus_dir,
        embeddings_path=args.embeddings,
        min_methods=args.min_methods,
        output_path=out_path,
    )


if __name__ == "__main__":
    main()
