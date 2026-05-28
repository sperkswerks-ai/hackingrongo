"""
hackingrongo.zone_b.spectrum_analyzer
======================================

Per-tablet logographic/syllabic spectrum analysis.

Theory
------
A **purely syllabic** tablet encodes phonology: signs map to syllables,
combinations are phonotactically driven, and the same syllables appear
across all contexts. Its statistical signature:

  - Moderate IC (syllable frequencies follow a natural Zipf distribution)
  - Low bigram MI (phonotactics allow many continuations)
  - Low consistency variance (same syllables used everywhere)
  - Fast entropy decay H₁ → H₂ → H₃ (phonotactic constraints are tight)
  - Rare compound glyphs
  - Low hapax rate (the same ~45 syllables keep recurring)

A **purely logographic** tablet encodes semantics: signs map to morphemes
or concepts, sequences are formulaic or domain-specific, and different
tablets cover different content domains. Its statistical signature:

  - High IC (a few dominant morphemes monopolise the text)
  - High bigram MI (morpheme sequences are highly formulaic)
  - High consistency variance (domain-specific signs only here)
  - Slow entropy decay (semantic associations are longer-range)
  - Frequent compound glyphs (compositional semantics)
  - High hapax rate (unique morphemes appear once per domain)

Six features are computed per tablet and projected onto a single
spectrum score in [0, 1] where 0 = maximally syllabic and 1 = maximally
logographic.  Weights are equal initially (exploratory measurement).

Public API
----------
``TabletSpectrumFeatures``
    Per-tablet feature vector.
``CorpusSpectrumNorms``
    Min/max normalization parameters derived from the full corpus.
``SpectrumAnalyzer``
    Main class.  ``analyze(corpus_dir) → dict[str, TabletSpectrumFeatures]``

CLI
---
    python -m hackingrongo.zone_b.spectrum_analyzer \\
        --corpus-dir data/corpus \\
        --output outputs/analysis/spectrum_scores.json
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Tablets with too few tokens for reliable statistics.
MIN_TOKENS_FOR_ANALYSIS: int = 50


# ---------------------------------------------------------------------------
# Feature dataclass
# ---------------------------------------------------------------------------


@dataclass
class TabletSpectrumFeatures:
    """Six entropy features + spectrum score for one tablet.

    Attributes
    ----------
    tablet_id : str
    tablet_name : str
    n_tokens : int
        Total sign tokens (non-compound positions counted individually).
    n_types : int
        Distinct sign types.
    mean_ic : float
        Index of Coincidence for this tablet's token sequence.
    mean_bigram_mi : float
        Mean per-sign bigram MI-as-predictor across signs on this tablet.
        Uses pre-computed :class:`~hackingrongo.zone_b.priors.CorpusSignStats`
        values so single-tablet bigrams are not too sparse.
    consistency_variance : float
        Variance of cross-tablet consistency scores for signs on this tablet.
        High variance = some signs are universal, others are tablet-specific.
    entropy_decay_rate : float
        (H₁ − H₂) / H₁ where H₁ = unigram entropy, H₂ = conditional bigram
        entropy.  High = phonotactically tight (syllabic); Low = loose (logographic).
    compound_density : float
        Fraction of glyph positions occupied by explicitly-marked compound glyphs.
    hapax_rate : float
        Fraction of sign *types* that appear exactly once on this tablet.
    spectrum_score : float
        [0 = syllabic, 1 = logographic].  Equal-weight projection of the
        six features (after per-corpus min-max normalisation).
    reliable : bool
        False for tablets with fewer than ``MIN_TOKENS_FOR_ANALYSIS`` tokens.
    """

    tablet_id: str
    tablet_name: str
    n_tokens: int
    n_types: int
    mean_ic: float
    mean_bigram_mi: float
    consistency_variance: float
    entropy_decay_rate: float
    compound_density: float
    hapax_rate: float
    spectrum_score: float = 0.0
    reliable: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CorpusSpectrumNorms:
    """Min-max normalisation parameters derived from the full corpus.

    Each key is a feature name; values are (min, max) tuples.
    The ``logographic_direction`` dict records whether higher feature
    values point toward logographic (+1) or syllabic (−1 → invert).
    """

    ranges: dict[str, tuple[float, float]] = field(default_factory=dict)
    logographic_direction: dict[str, int] = field(default_factory=dict)

    @classmethod
    def from_features(
        cls,
        features: list[TabletSpectrumFeatures],
    ) -> "CorpusSpectrumNorms":
        """Derive norms from a list of (reliable) tablet features."""
        norms = cls(
            logographic_direction={
                "mean_ic":              +1,  # high IC → logographic
                "mean_bigram_mi":       +1,  # high bigram MI → logographic
                "consistency_variance": +1,  # high variance → logographic
                "entropy_decay_rate":   -1,  # high decay rate → syllabic (invert)
                "compound_density":     +1,  # high compounds → logographic
                "hapax_rate":           +1,  # high hapax → logographic
            }
        )
        feature_names = [
            "mean_ic", "mean_bigram_mi", "consistency_variance",
            "entropy_decay_rate", "compound_density", "hapax_rate",
        ]
        for fname in feature_names:
            vals = [getattr(f, fname) for f in features if f.reliable]
            if not vals:
                norms.ranges[fname] = (0.0, 1.0)
                continue
            lo, hi = min(vals), max(vals)
            if hi - lo < 1e-9:
                lo, hi = lo - 0.5, lo + 0.5
            norms.ranges[fname] = (lo, hi)
        return norms

    def normalise(self, fname: str, value: float) -> float:
        """Map a raw feature value to [0, 1] in the logographic direction."""
        lo, hi = self.ranges.get(fname, (0.0, 1.0))
        norm = (value - lo) / max(hi - lo, 1e-9)
        norm = float(max(0.0, min(1.0, norm)))
        direction = self.logographic_direction.get(fname, +1)
        return norm if direction == +1 else 1.0 - norm


# ---------------------------------------------------------------------------
# SpectrumAnalyzer
# ---------------------------------------------------------------------------


class SpectrumAnalyzer:
    """Compute per-tablet logographic/syllabic spectrum scores.

    Parameters
    ----------
    corpus_dir : Path
        ``data/corpus/`` directory.
    metadata_path : Path | None
        Optional ``data/metadata/tablets.json`` for tablet names.
    corpus_stats : CorpusSignStats | None
        Pre-computed sign statistics from
        :func:`~hackingrongo.zone_b.priors.compute_corpus_sign_stats`.
        If None, mean_bigram_mi and consistency_variance fall back to 0.
    """

    def __init__(
        self,
        corpus_dir: Path,
        metadata_path: Path | None = None,
        corpus_stats: "Any | None" = None,
    ) -> None:
        self._corpus_dir = corpus_dir
        self._corpus_stats = corpus_stats

        # Tablet names from metadata
        self._tablet_names: dict[str, str] = {}
        if metadata_path and metadata_path.exists():
            try:
                meta = json.loads(metadata_path.read_text(encoding="utf-8"))
                self._tablet_names = {k: v.get("name", k) for k, v in meta.items()}
            except Exception as exc:
                logger.warning("Could not load tablet metadata: %s", exc)

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def analyze(self) -> dict[str, TabletSpectrumFeatures]:
        """Compute spectrum features for all tablets.

        Returns
        -------
        dict[str, TabletSpectrumFeatures]
            Keyed by tablet ID.  Spectrum scores are calibrated against
            corpus-wide min/max after all tablets are computed.
        """
        raw: dict[str, TabletSpectrumFeatures] = {}

        for jf in sorted(self._corpus_dir.glob("[A-Z].json")):
            if "ferrara" in jf.stem:
                continue
            try:
                data = json.loads(jf.read_text(encoding="utf-8"))
            except Exception as exc:
                logger.warning("Could not read %s: %s", jf, exc)
                continue

            tablet_id = jf.stem
            name = self._tablet_names.get(tablet_id, tablet_id)
            features = self._compute_features(tablet_id, name, data)
            raw[tablet_id] = features
            logger.debug(
                "Tablet %s: n=%d  ic=%.4f  decay=%.4f  compound_d=%.4f  "
                "hapax=%.3f  bigram_mi=%.4f",
                tablet_id,
                features.n_tokens,
                features.mean_ic,
                features.entropy_decay_rate,
                features.compound_density,
                features.hapax_rate,
                features.mean_bigram_mi,
            )

        # Calibrate spectrum scores across all reliable tablets
        reliable = [f for f in raw.values() if f.reliable]
        if reliable:
            norms = CorpusSpectrumNorms.from_features(reliable)
            for feat in raw.values():
                feat.spectrum_score = self._project(feat, norms)
        else:
            logger.warning("No reliable tablets found — spectrum scores set to 0.5.")

        logger.info(
            "SpectrumAnalyzer: %d tablets, %d reliable.",
            len(raw),
            len(reliable),
        )
        return raw

    # ------------------------------------------------------------------
    # Per-tablet feature computation
    # ------------------------------------------------------------------

    def _compute_features(
        self,
        tablet_id: str,
        name: str,
        data: dict[str, Any],
    ) -> TabletSpectrumFeatures:
        """Extract six features from a single tablet's corpus data."""
        glyphs = data.get("glyphs", [])

        # ── Token sequence ────────────────────────────────────────────────
        COMPOUND_SEPS = (":", ".", "-", "'")
        tokens: list[str] = []
        n_compound_positions = 0

        for g in glyphs:
            bc = str(g.get("barthel_code") or "").strip()
            if not bc or bc == "?":
                continue
            if any(sep in bc for sep in COMPOUND_SEPS) and "!" not in bc:
                n_compound_positions += 1
            if bc and "!" not in bc:
                tokens.append(bc)

        n_tokens = len(tokens)
        reliable = n_tokens >= MIN_TOKENS_FOR_ANALYSIS

        if not tokens:
            return TabletSpectrumFeatures(
                tablet_id=tablet_id, tablet_name=name,
                n_tokens=0, n_types=0,
                mean_ic=0.0, mean_bigram_mi=0.0,
                consistency_variance=0.0, entropy_decay_rate=0.0,
                compound_density=0.0, hapax_rate=0.0,
                spectrum_score=0.5, reliable=False,
            )

        counts = Counter(tokens)
        n_types = len(counts)

        # ── Feature 1: IC ─────────────────────────────────────────────────
        mean_ic = self._ic(tokens)

        # ── Feature 2: mean bigram MI (from corpus-level stats) ──────────
        if self._corpus_stats is not None:
            mi_vals = [
                self._corpus_stats.bigram_mi(code)
                for code in counts
            ]
            mean_bigram_mi = sum(mi_vals) / len(mi_vals) if mi_vals else 0.0
        else:
            mean_bigram_mi = self._bigram_mi_local(tokens)

        # ── Feature 3: consistency variance ──────────────────────────────
        if self._corpus_stats is not None:
            cons_vals = [
                self._corpus_stats.consistency(code)
                for code in counts
            ]
            consistency_variance = (
                float(_variance(cons_vals)) if len(cons_vals) >= 2 else 0.0
            )
        else:
            consistency_variance = 0.0

        # ── Feature 4: entropy decay rate ─────────────────────────────────
        entropy_decay_rate = self._entropy_decay(tokens)

        # ── Feature 5: compound density ──────────────────────────────────
        n_total_positions = len(glyphs)
        compound_density = (
            n_compound_positions / n_total_positions
            if n_total_positions > 0 else 0.0
        )

        # ── Feature 6: hapax rate ─────────────────────────────────────────
        n_hapax = sum(1 for c in counts.values() if c == 1)
        hapax_rate = n_hapax / n_types if n_types > 0 else 0.0

        return TabletSpectrumFeatures(
            tablet_id=tablet_id,
            tablet_name=name,
            n_tokens=n_tokens,
            n_types=n_types,
            mean_ic=round(mean_ic, 6),
            mean_bigram_mi=round(mean_bigram_mi, 6),
            consistency_variance=round(consistency_variance, 6),
            entropy_decay_rate=round(entropy_decay_rate, 6),
            compound_density=round(compound_density, 6),
            hapax_rate=round(hapax_rate, 6),
            reliable=reliable,
        )

    # ------------------------------------------------------------------
    # Statistical helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _ic(tokens: list[str]) -> float:
        n = len(tokens)
        if n < 2:
            return 0.0
        counts = Counter(tokens)
        return sum(f * (f - 1) for f in counts.values()) / (n * (n - 1))

    @staticmethod
    def _bigram_mi_local(tokens: list[str]) -> float:
        """Compute bigram MI from tablet-local counts (fallback when no corpus_stats)."""
        if len(tokens) < 3:
            return 0.0
        bigrams: Counter = Counter(zip(tokens[:-1], tokens[1:]))
        unigrams: Counter = Counter(tokens)
        n_bg = sum(bigrams.values())
        n = len(tokens)
        mi = 0.0
        for (s, t), cnt in bigrams.items():
            p_st = cnt / n_bg
            p_s = unigrams[s] / n
            p_t = unigrams[t] / n
            if p_s > 0 and p_t > 0:
                p_t_given_s = cnt / unigrams[s]
                if p_t_given_s > 0:
                    mi += p_st * math.log2(p_t_given_s / p_t)
        return max(0.0, mi)

    @staticmethod
    def _entropy_decay(tokens: list[str]) -> float:
        """(H₁ − H₂) / H₁ — fraction of unigram entropy removed by bigram context."""
        if len(tokens) < 3:
            return 0.0
        n = len(tokens)
        counts = Counter(tokens)
        h1 = -sum((c / n) * math.log2(c / n) for c in counts.values() if c > 0)
        if h1 < 1e-9:
            return 0.0
        bigrams: Counter = Counter(zip(tokens[:-1], tokens[1:]))
        n_bg = sum(bigrams.values())
        h2 = 0.0
        for (s, t), cnt in bigrams.items():
            p_st = cnt / n_bg
            p_t_given_s = cnt / counts[s]
            if p_t_given_s > 0:
                h2 -= p_st * math.log2(p_t_given_s)
        return max(0.0, (h1 - h2) / h1)

    # ------------------------------------------------------------------
    # Spectrum projection
    # ------------------------------------------------------------------

    @staticmethod
    def _project(
        features: TabletSpectrumFeatures,
        norms: CorpusSpectrumNorms,
        weights: dict[str, float] | None = None,
    ) -> float:
        """Project a feature vector onto [0 = syllabic, 1 = logographic].

        Uses equal weights by default.  Override ``weights`` to apply a
        custom weighting scheme.
        """
        feature_names = [
            "mean_ic", "mean_bigram_mi", "consistency_variance",
            "entropy_decay_rate", "compound_density", "hapax_rate",
        ]
        if weights is None:
            weights = {f: 1.0 / len(feature_names) for f in feature_names}

        score = 0.0
        total_w = sum(weights.get(f, 0.0) for f in feature_names)
        if total_w < 1e-9:
            return 0.5
        for fname in feature_names:
            raw_val = getattr(features, fname)
            normed = norms.normalise(fname, raw_val)
            score += normed * weights.get(fname, 0.0)
        return round(score / total_w, 4)


# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------


def _variance(vals: list[float]) -> float:
    if len(vals) < 2:
        return 0.0
    mean = sum(vals) / len(vals)
    return sum((v - mean) ** 2 for v in vals) / len(vals)


def save_spectrum_scores(
    scores: dict[str, TabletSpectrumFeatures],
    output_path: Path,
) -> None:
    payload = {
        "n_tablets": len(scores),
        "n_reliable": sum(1 for f in scores.values() if f.reliable),
        "tablets": {tid: f.to_dict() for tid, f in scores.items()},
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    logger.info("Spectrum scores written: %s", output_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compute per-tablet logographic/syllabic spectrum scores."
    )
    p.add_argument("--corpus-dir", type=Path, default=Path("data/corpus"))
    p.add_argument("--metadata", type=Path, default=Path("data/metadata/tablets.json"))
    p.add_argument("--output", type=Path, default=Path("outputs/analysis/spectrum_scores.json"))
    return p.parse_args()


def main() -> None:
    import logging as _logging
    _logging.basicConfig(level=_logging.INFO, format="%(levelname)s  %(message)s")
    args = _parse_args()

    try:
        from hackingrongo.zone_b.priors import compute_corpus_sign_stats
        corpus_stats = compute_corpus_sign_stats(args.corpus_dir)
    except Exception as exc:
        logger.warning("Could not load corpus stats: %s — using local fallback.", exc)
        corpus_stats = None

    analyzer = SpectrumAnalyzer(args.corpus_dir, args.metadata, corpus_stats)
    scores = analyzer.analyze()
    save_spectrum_scores(scores, args.output)

    print(f"\n── Tablet Spectrum Scores ({'syllabic=0  logographic=1'})")
    print(f"  {'Tablet':<4} {'Name':<26} {'Score':>6}  {'n':>5}  {'Reliable'}")
    print(f"  {'-'*4} {'-'*26} {'-'*6}  {'-'*5}  {'-'*8}")
    for tid, f in sorted(scores.items(), key=lambda x: -x[1].spectrum_score):
        rel = "✓" if f.reliable else "⚠ small"
        print(f"  {tid:<4} {f.tablet_name:<26} {f.spectrum_score:>6.3f}  "
              f"{f.n_tokens:>5}  {rel}")
    print(f"\nFull results: {args.output}")


if __name__ == "__main__":
    main()
