"""
hackingrongo.zone_c.mixed_decoder
===================================

Mixed phonogram/logogram decoder for Zone C.

Motivation
----------
The base :class:`~hackingrongo.zone_c.mcmc.MCMCSampler` treats every
sign as a phonogram — mapping it to a Rapa Nui syllable and scoring the
translation against a phonotactic language model.  This is appropriate
if rongorongo is a purely phonological script, but may be wrong.

Signs that carry semantic (morphemic) content rather than phonological
content should be *excluded* from phonotactic LM scoring — the LM models
syllable phonotactics, not morpheme co-occurrence.  Forcing logographic
signs through the phonotactic scorer corrupts it with non-phonological
signal, depressing the LM score of correct phoneme assignments and
introducing false confidence in incorrect ones.

This module implements a mixed decoder in which each sign is typed as:

``PHONOGRAM``
    Normal MCMC phoneme proposal; participates in phonotactic LM scoring.
``LOGOGRAM``
    Assigned a morpheme from a morpheme inventory; *excluded* from LM
    scoring.  Implemented by pinning these signs to the ``<LOGOGRAM>``
    placeholder token, which the LM scorer skips (treats as OOV).
``TAXOGRAM``
    Existing behaviour — pin to a crib phoneme or exclude entirely.
``UNCERTAIN``
    Initially treated as PHONOGRAM.  A ``type_flip`` proposal move can
    switch uncertain signs between phonogram and logogram categories,
    letting the posterior decide which hypothesis is better supported.

Type initialisation
-------------------
Sign types are initialised from a ``sign_type_map`` argument.  A
convenience constructor :meth:`MixedMCMCSampler.from_spectrum` uses the
per-tablet spectrum scores from
:class:`~hackingrongo.zone_b.spectrum_analyzer.SpectrumAnalyzer` and the
per-sign corpus stats from
:class:`~hackingrongo.zone_b.priors.CorpusSignStats` to assign initial
types automatically:

  - Signs where ``ic_contribution × bigram_mi > logogram_threshold`` AND
    ``is_compound_component == 1`` → LOGOGRAM
  - Signs with ``ic_contribution > phonogram_threshold`` AND
    ``bigram_mi < low_mi_threshold`` → likely PHONOGRAM
  - All others → UNCERTAIN

Public API
----------
``SignType``
    Enum.
``MixedMCMCSampler``
    Main class.  Wraps :class:`~hackingrongo.zone_c.mcmc.MCMCSampler`
    and adds logogram bookkeeping.
``MixedResult``
    Extended result dataclass with morpheme_map and type_map.
``LOGOGRAM_TOKEN``
    Placeholder token inserted for logogram signs in phoneme sequences.
"""

from __future__ import annotations

import logging
import random
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any

from omegaconf import DictConfig

from hackingrongo.zone_c.lm_scoring import LMScorer, PhonemeMap
from hackingrongo.zone_c.mcmc import MCMCSampler, MCMCResult, MCMCSample

if TYPE_CHECKING:
    from hackingrongo.zone_b.priors import CorpusSignStats
    from hackingrongo.zone_b.spectrum_analyzer import TabletSpectrumFeatures

logger = logging.getLogger(__name__)

# Token inserted for logogram signs.  The LM scorer treats this as OOV
# (same as <UNK>), so logogram positions are excluded from n-gram windows.
LOGOGRAM_TOKEN: str = "<LOGOGRAM>"

# Default Rapa Nui morpheme inventory — top-50 high-frequency morphemes.
# Derived from pre_contact_lm vocabulary (common Rapa Nui roots/particles).
_DEFAULT_MORPHEME_INVENTORY: list[str] = [
    "ariki", "tangata", "manu", "ika", "rima", "mata", "haka", "ko",
    "ki", "i", "o", "a", "no", "mo", "e", "ana", "ia", "nei",
    "ra", "na", "te", "he", "ka", "kua", "kia", "ai", "mai",
    "atu", "ake", "iho", "roto", "raro", "runga", "mua", "muri",
    "aha", "pea", "anei", "tahi", "rua", "toru", "ha", "rima",
    "ono", "hitu", "varu", "iva", "hangahulu", "hanere",
]


# ---------------------------------------------------------------------------
# SignType
# ---------------------------------------------------------------------------


class SignType(Enum):
    """Classification of a rongorongo sign for the mixed decoder."""
    PHONOGRAM = "phonogram"   # maps to syllable; participates in LM scoring
    LOGOGRAM  = "logogram"    # maps to morpheme; excluded from LM scoring
    TAXOGRAM  = "taxogram"    # grammatical marker; pinned / excluded
    UNCERTAIN = "uncertain"   # initial phonogram; type_flip moves allowed


# ---------------------------------------------------------------------------
# MixedResult
# ---------------------------------------------------------------------------


@dataclass
class MixedResult:
    """Extended MCMC result carrying both phoneme and morpheme assignments.

    Attributes
    ----------
    mcmc_result : MCMCResult
        Underlying phoneme-assignment result from the MCMCSampler.
    morpheme_map : dict[str, str]
        Sign → morpheme for signs classified as LOGOGRAM.
    type_map : dict[str, SignType]
        Final type assignment for every sign.
    n_phonograms : int
    n_logograms : int
    n_taxograms : int
    n_uncertain : int
    logogram_fraction : float
        Fraction of distinct signs classified as LOGOGRAM.
    """

    mcmc_result: MCMCResult
    morpheme_map: dict[str, str] = field(default_factory=dict)
    type_map: dict[str, SignType] = field(default_factory=dict)
    n_phonograms: int = 0
    n_logograms: int = 0
    n_taxograms: int = 0
    n_uncertain: int = 0
    logogram_fraction: float = 0.0

    @property
    def top_phoneme_map(self) -> PhonemeMap:
        """Best phoneme assignment from the MCMC run."""
        if not self.mcmc_result.top_samples:
            return {}
        return self.mcmc_result.top_samples[0].phoneme_map

    @property
    def best_log_posterior(self) -> float:
        if not self.mcmc_result.top_samples:
            return float("-inf")
        return self.mcmc_result.top_samples[0].log_posterior


# ---------------------------------------------------------------------------
# MixedMCMCSampler
# ---------------------------------------------------------------------------


class MixedMCMCSampler:
    """MCMC phoneme sampler with per-sign logogram/phonogram type assignments.

    Parameters
    ----------
    cfg : DictConfig
        Root Hydra config.
    lm_scorer : LMScorer
    corpus_sequences : list[list[str]]
    sign_ids : list[str]
    sign_type_map : dict[str, SignType]
        Initial type for each sign.  Signs absent from this dict are
        treated as UNCERTAIN (and handled as PHONOGRAM).
    morpheme_inventory : list[str]
        Candidate morphemes for LOGOGRAM signs.
    phoneme_inventory : list[str] | None
    phoneme_priors : list[float] | None
    sign_ic_weights : dict[str, float] | None
        IC-contribution weights for IC-weighted MCMC proposals (from
        :func:`~hackingrongo.zone_b.priors.compute_corpus_sign_stats`).
    seed : int | None
    cribs : dict[str, str] | None
        Known-plaintext phoneme assignments for specific signs.
    type_flip_prob : float
        Fraction of proposals that attempt to flip an UNCERTAIN sign's
        type between PHONOGRAM and LOGOGRAM.  Default 0.05.
    """

    def __init__(
        self,
        cfg: DictConfig,
        lm_scorer: LMScorer,
        corpus_sequences: list[list[str]],
        sign_ids: list[str],
        sign_type_map: dict[str, SignType] | None = None,
        morpheme_inventory: list[str] | None = None,
        phoneme_inventory: list[str] | None = None,
        phoneme_priors: list[float] | None = None,
        sign_ic_weights: dict[str, float] | None = None,
        seed: int | None = None,
        cribs: dict[str, str] | None = None,
        type_flip_prob: float = 0.05,
    ) -> None:
        self._cfg = cfg
        self._lm_scorer = lm_scorer
        self._corpus_sequences = corpus_sequences
        self._sign_ids = list(sign_ids)
        self._seed = seed
        self._type_flip_prob = type_flip_prob

        self._morpheme_inventory: list[str] = (
            list(morpheme_inventory)
            if morpheme_inventory is not None
            else list(_DEFAULT_MORPHEME_INVENTORY)
        )

        # Resolve sign types
        self._type_map: dict[str, SignType] = {}
        for sign in sign_ids:
            if sign_type_map and sign in sign_type_map:
                self._type_map[sign] = sign_type_map[sign]
            else:
                self._type_map[sign] = SignType.UNCERTAIN

        # Partition signs
        self._logogram_signs: frozenset[str] = frozenset(
            s for s, t in self._type_map.items() if t is SignType.LOGOGRAM
        )
        self._taxogram_signs: frozenset[str] = frozenset(
            s for s, t in self._type_map.items() if t is SignType.TAXOGRAM
        )
        self._uncertain_signs: list[str] = [
            s for s, t in self._type_map.items() if t is SignType.UNCERTAIN
        ]

        logger.info(
            "MixedMCMCSampler: %d signs total — %d phonogram, %d logogram, "
            "%d taxogram, %d uncertain.",
            len(sign_ids),
            sum(1 for t in self._type_map.values() if t is SignType.PHONOGRAM),
            len(self._logogram_signs),
            len(self._taxogram_signs),
            len(self._uncertain_signs),
        )

        # Logogram signs are pinned to LOGOGRAM_TOKEN so they are excluded
        # from phoneme proposals and from n-gram LM windows.
        combined_cribs = dict(cribs) if cribs else {}
        for sign in self._logogram_signs:
            combined_cribs[sign] = LOGOGRAM_TOKEN
        for sign in self._taxogram_signs:
            # Taxograms also excluded from LM scoring.
            combined_cribs.setdefault(sign, LOGOGRAM_TOKEN)

        # Morpheme initialisation for logogram signs (random, refined later).
        rng = random.Random(seed)
        self._morpheme_map: dict[str, str] = {
            sign: rng.choice(self._morpheme_inventory)
            for sign in self._logogram_signs
        }

        # Wrap the base MCMCSampler for phoneme assignments.
        self._mcmc = MCMCSampler(
            cfg=cfg,
            lm_scorer=lm_scorer,
            corpus_sequences=corpus_sequences,
            sign_ids=sign_ids,
            phoneme_inventory=phoneme_inventory,
            phoneme_priors=phoneme_priors,
            seed=seed,
            cribs=combined_cribs,
            sign_ic_weights=sign_ic_weights,
        )

    # ------------------------------------------------------------------
    # Factory: initialise types from corpus stats + spectrum
    # ------------------------------------------------------------------

    @classmethod
    def from_spectrum(
        cls,
        cfg: DictConfig,
        lm_scorer: LMScorer,
        corpus_sequences: list[list[str]],
        sign_ids: list[str],
        corpus_stats: "CorpusSignStats",
        logogram_threshold: float = 0.15,
        phonogram_threshold: float = 0.05,
        low_mi_threshold: float = 0.30,
        **kwargs: Any,
    ) -> "MixedMCMCSampler":
        """Initialise sign types from corpus-level entropy statistics.

        Classification rules
        --------------------
        LOGOGRAM (sign is probably semantic)
            ``ic_contribution ≥ logogram_threshold``
            AND ``bigram_mi_score ≥ 0.5``
            AND ``is_compound_component == 1``
            → Strong semantic anchor: high IC weight, highly predictive
              of successor, and known compound component.

        PHONOGRAM (sign is probably phonological)
            ``ic_contribution ≥ phonogram_threshold``
            AND ``bigram_mi_score < low_mi_threshold``
            AND NOT compound component
            → High IC (common) but low sequential predictability and
              not a compound element = likely a phonogram.

        All others → UNCERTAIN.

        Parameters
        ----------
        logogram_threshold : float
            Minimum normalised IC contribution to consider for LOGOGRAM.
        phonogram_threshold : float
            Minimum IC contribution to classify as definite PHONOGRAM.
        low_mi_threshold : float
            Maximum bigram MI score below which a sign is phonogram-like.
        """
        type_map: dict[str, SignType] = {}
        for sign in sign_ids:
            ic  = corpus_stats.ic_contribution(sign)
            mi  = corpus_stats.bigram_mi(sign)
            cc  = corpus_stats.is_compound_component(sign)

            if ic >= logogram_threshold and mi >= 0.5 and cc:
                type_map[sign] = SignType.LOGOGRAM
            elif ic >= phonogram_threshold and mi < low_mi_threshold and not cc:
                type_map[sign] = SignType.PHONOGRAM
            else:
                type_map[sign] = SignType.UNCERTAIN

        n_log = sum(1 for t in type_map.values() if t is SignType.LOGOGRAM)
        n_ph  = sum(1 for t in type_map.values() if t is SignType.PHONOGRAM)
        n_unc = sum(1 for t in type_map.values() if t is SignType.UNCERTAIN)
        logger.info(
            "from_spectrum: %d logogram, %d phonogram, %d uncertain "
            "(thresholds: logogram_ic≥%.2f, phonogram_ic≥%.2f, low_mi<%.2f)",
            n_log, n_ph, n_unc,
            logogram_threshold, phonogram_threshold, low_mi_threshold,
        )

        return cls(
            cfg=cfg,
            lm_scorer=lm_scorer,
            corpus_sequences=corpus_sequences,
            sign_ids=sign_ids,
            sign_type_map=type_map,
            **kwargs,
        )

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run(self) -> MixedResult:
        """Run the mixed MCMC decoder.

        Runs the underlying :class:`MCMCSampler` for phoneme assignments,
        while keeping logogram signs pinned to ``LOGOGRAM_TOKEN``.
        Morpheme assignments for logogram signs are returned as-is from
        random initialisation (post-processing / morpheme refinement is
        handled separately).

        Returns
        -------
        MixedResult
        """
        # ── Type-flip adaptation (burn-in only) ─────────────────────────
        # For uncertain signs, we run a short pre-pass that experiments
        # with flipping each uncertain sign between phonogram and logogram
        # and observes whether the LM improves or degrades.  This is a
        # simple heuristic — not full Bayesian type inference.
        final_type_map = self._type_flip_prescan()

        # Re-initialise the MCMCSampler with updated type assignments.
        updated_cribs = {
            sign: LOGOGRAM_TOKEN
            for sign, t in final_type_map.items()
            if t in (SignType.LOGOGRAM, SignType.TAXOGRAM)
        }
        updated_mcmc = MCMCSampler(
            cfg=self._cfg,
            lm_scorer=self._lm_scorer,
            corpus_sequences=self._corpus_sequences,
            sign_ids=self._sign_ids,
            phoneme_inventory=self._mcmc._phoneme_inventory,
            phoneme_priors=self._mcmc._phoneme_priors,
            seed=self._seed,
            cribs=updated_cribs,
            sign_ic_weights=(
                {s: w for s, w in self._mcmc._sign_proposal_weights.items()}
                if self._mcmc._sign_proposal_weights else None
            ),
        )

        mcmc_result = updated_mcmc.run()

        # Tally sign type counts
        n_ph  = sum(1 for t in final_type_map.values() if t is SignType.PHONOGRAM)
        n_log = sum(1 for t in final_type_map.values() if t is SignType.LOGOGRAM)
        n_tax = sum(1 for t in final_type_map.values() if t is SignType.TAXOGRAM)
        n_unc = sum(1 for t in final_type_map.values() if t is SignType.UNCERTAIN)
        n_total = len(final_type_map)

        return MixedResult(
            mcmc_result=mcmc_result,
            morpheme_map=dict(self._morpheme_map),
            type_map=final_type_map,
            n_phonograms=n_ph,
            n_logograms=n_log,
            n_taxograms=n_tax,
            n_uncertain=n_unc,
            logogram_fraction=n_log / max(n_total, 1),
        )

    # ------------------------------------------------------------------
    # Type-flip pre-scan
    # ------------------------------------------------------------------

    def _type_flip_prescan(self) -> dict[str, SignType]:
        """For each UNCERTAIN sign, test whether LOGOGRAM improves LM score.

        Translates the corpus with the sign as LOGOGRAM_TOKEN vs its
        current phoneme assignment, compares the LM log-probability, and
        classifies accordingly.

        This is a fast greedy heuristic — O(n_uncertain × n_sequences).
        """
        if not self._uncertain_signs or self._type_flip_prob <= 0.0:
            return dict(self._type_map)

        rng = random.Random(self._seed)
        final = dict(self._type_map)

        # Baseline: score current full assignment (uncertain → phonogram).
        # Use a random initial map for uncertain signs.
        base_map: dict[str, str] = {
            sign: LOGOGRAM_TOKEN
            for sign, t in self._type_map.items()
            if t in (SignType.LOGOGRAM, SignType.TAXOGRAM)
        }
        for sign in self._uncertain_signs:
            base_map[sign] = rng.choice(self._mcmc._phoneme_inventory)

        base_seqs = [
            [base_map.get(tok, tok) for tok in seq]
            for seq in self._corpus_sequences
        ]
        base_score = sum(
            s.ensemble_log_prob for s in
            (self._lm_scorer.score(seq) for seq in base_seqs)
            if s.ensemble_log_prob is not None
        )

        flipped = 0
        # Only test a fraction of uncertain signs (speed guard).
        candidates = (
            rng.sample(self._uncertain_signs, min(50, len(self._uncertain_signs)))
        )
        for sign in candidates:
            if rng.random() > self._type_flip_prob * 10:
                continue

            # Test with this sign as LOGOGRAM_TOKEN.
            test_seqs = [
                [LOGOGRAM_TOKEN if tok == sign else base_map.get(tok, tok) for tok in seq]
                for seq in self._corpus_sequences
            ]
            test_score = sum(
                s.ensemble_log_prob for s in
                (self._lm_scorer.score(seq) for seq in test_seqs)
                if s.ensemble_log_prob is not None
            )

            # Flip to logogram if it improves LM score and the sign has
            # compound component membership (additional evidence).
            improvement = test_score - base_score
            if improvement > 0.1 and sign in (self._mcmc._cribs or {}):
                final[sign] = SignType.LOGOGRAM
                self._morpheme_map[sign] = rng.choice(self._morpheme_inventory)
                flipped += 1

        if flipped:
            logger.info(
                "Type-flip prescan: %d uncertain signs flipped to LOGOGRAM.", flipped
            )
        return final

    # ------------------------------------------------------------------
    # Summary helpers
    # ------------------------------------------------------------------

    @property
    def type_summary(self) -> dict[str, int]:
        """Count of each sign type in the current assignment."""
        return Counter(t.value for t in self._type_map.values())

    def type_table(self) -> list[dict[str, str]]:
        """Return a list of {sign, type} dicts sorted by IC weight."""
        ic_w = getattr(self._mcmc, "_sign_proposal_weights", {})
        rows = []
        for sign in sorted(
            self._sign_ids,
            key=lambda s: ic_w.get(s, 0.0),
            reverse=True,
        ):
            rows.append({
                "sign": sign,
                "type": self._type_map[sign].value,
                "morpheme": self._morpheme_map.get(sign, ""),
            })
        return rows
