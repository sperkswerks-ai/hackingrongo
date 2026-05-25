"""
hackingrongo.zone_c.beam_search
================================

Deterministic bounded-width beam search over phoneme assignment maps.

Unlike the MCMC sampler (which explores the full posterior), beam search
greedily extends a partial assignment one sign at a time.  It is faster
but biased toward locally optimal phoneme choices; use it as a warm-start
complement to MCMC rather than a replacement.

Algorithm
---------
1. Start with ``beam_width`` copies of an empty map (or maps seeded from
   MCMC top samples if provided).
2. At each step, pick the next unassigned sign (ordered by decreasing
   corpus frequency so high-frequency signs are committed early).
3. Expand every beam item by trying every phoneme in the inventory.
4. Score each expanded item (partial sequence treated as a complete
   sequence; unassigned signs are masked with ``<MASK>`` tokens and
   excluded from n-gram scoring using ``skip_unk`` logic in
   :meth:`LMScorer.score`).
5. Prune back to ``beam_width`` (by log-score normalised for current
   depth, using the length-normalisation alpha).
6. Apply the early-stopping criterion: if the best score has not improved
   by ``min_improvement`` for ``patience`` consecutive steps, terminate.
7. Return the top-K complete assignments sorted by final score.

Partial scoring
---------------
Because ``LMScorer`` filters out ``<UNK>`` in n-gram windows, a partial
map scores only the n-grams that are fully resolved.  This under-estimates
the final score but gives a consistent partial ordering.

Public API
----------
``BeamSearchDecoder``
    ``decode(sign_ids, corpus_sequences) -> BeamSearchResult``
``BeamSearchResult``
    Dataclass holding ranked hypotheses and diagnostics.
``BeamHypothesis``
    Single beam item with its current partial/complete assignment and score.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field

from omegaconf import DictConfig

from hackingrongo.zone_c.lm_scoring import LMScorer, LMScoringResult, PhonemeMap
from hackingrongo.zone_c.mcmc import _DEFAULT_PHONEME_INVENTORY, MCMCSample

logger = logging.getLogger(__name__)

_MASK_TOKEN: str = "<MASK>"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class BeamHypothesis:
    """A partial or complete phoneme assignment hypothesis in the beam.

    Attributes
    ----------
    phoneme_map : PhonemeMap
        Current (possibly partial) sign → phoneme assignment.
    log_score : float
        Raw (unnormalised) log₂-probability accumulated so far.
    depth : int
        Number of signs assigned so far.
    """

    phoneme_map: PhonemeMap
    log_score: float
    depth: int = 0
    phoneme_seqs: list[list[str]] = field(default_factory=list)

    def normalised_score(self, alpha: float) -> float:
        """Length-normalised score: ``log_score / depth^alpha``.

        Uses ``depth`` as the length; ``alpha=0`` returns the raw score.
        """
        if self.depth <= 0:
            return -math.inf
        return self.log_score / (self.depth ** alpha)


@dataclass
class BeamSearchResult:
    """Output from a completed beam search.

    Attributes
    ----------
    top_hypotheses : list[BeamHypothesis]
        Top-K complete assignments by normalised score (descending).
    n_signs : int
        Total number of distinct signs that were assigned.
    n_steps : int
        Number of beam search steps taken.
    early_stopped : bool
        Whether early stopping fired before exhausting all signs.
    """

    top_hypotheses: list[BeamHypothesis]
    n_signs: int = 0
    n_steps: int = 0
    early_stopped: bool = False


# ---------------------------------------------------------------------------
# BeamSearchDecoder
# ---------------------------------------------------------------------------


class BeamSearchDecoder:
    """Deterministic bounded beam search over phoneme assignments.

    Parameters
    ----------
    cfg : DictConfig
        Root Hydra config.  Reads ``cfg.zone_c.beam_search``.
    lm_scorer : LMScorer
        Loaded :class:`~hackingrongo.zone_c.lm_scoring.LMScorer`.
    phoneme_inventory : list[str] | None
        Candidate phonemes.  Defaults to the 45-token Rapa Nui inventory.
    """

    def __init__(
        self,
        cfg: DictConfig,
        lm_scorer: LMScorer,
        phoneme_inventory: list[str] | None = None,
    ) -> None:
        self._scorer = lm_scorer
        self._phoneme_inventory = (
            list(phoneme_inventory)
            if phoneme_inventory is not None
            else list(_DEFAULT_PHONEME_INVENTORY)
        )

        bsc = cfg.zone_c.beam_search
        self._beam_width: int = int(bsc.beam_width)
        self._max_depth: int = int(bsc.max_depth)
        self._alpha: float = float(bsc.length_penalty_alpha)
        self._prune_threshold: float = float(bsc.prune_threshold)
        self._patience: int = int(bsc.early_stopping_patience)
        self._min_improvement: float = float(bsc.min_improvement)
        self._top_k: int = int(bsc.top_k)

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def decode(
        self,
        sign_ids: list[str],
        corpus_sequences: list[list[str]],
        seed_hypotheses: list[MCMCSample] | None = None,
    ) -> BeamSearchResult:
        """Run beam search over phoneme assignments for ``sign_ids``.

        Parameters
        ----------
        sign_ids : list[str]
            Ordered list of distinct sign identifiers.  Signs are
            committed in corpus-frequency order (most frequent first).
        corpus_sequences : list[list[str]]
            Glyph-token sequences against which assignments are scored.
        seed_hypotheses : list[MCMCSample] | None
            Optional MCMC samples used to seed the initial beam (instead
            of empty maps).  Only the first ``beam_width`` seeds are used.

        Returns
        -------
        BeamSearchResult
        """
        if not sign_ids:
            return BeamSearchResult(top_hypotheses=[], n_signs=0)

        # Order signs by descending frequency in corpus.
        ordered_signs = self._order_signs_by_frequency(sign_ids, corpus_sequences)

        # Precompute sign → per-sequence position index for incremental scoring.
        # sign_positions[sign][seq_idx] = list of 0-based positions in that sequence.
        sign_positions: dict[str, list[list[int]]] = {s: [] for s in sign_ids}
        for seq in corpus_sequences:
            pos_in_seq: dict[str, list[int]] = {}
            for pos, token in enumerate(seq):
                pos_in_seq.setdefault(token, []).append(pos)
            for s in sign_ids:
                sign_positions[s].append(pos_in_seq.get(s, []))

        # Initialise beam with per-hypothesis translated sequences.
        beam = self._init_beam(seed_hypotheses, corpus_sequences)

        best_norm_score = -math.inf
        no_improve_steps = 0
        n_steps = 0
        early_stopped = False

        for step, sign in enumerate(ordered_signs[:self._max_depth]):
            beam = self._expand(beam, sign, sign_positions)
            if not beam:
                break
            n_steps = step + 1

            current_best = beam[0].normalised_score(self._alpha)
            if current_best - best_norm_score > self._min_improvement:
                best_norm_score = current_best
                no_improve_steps = 0
            else:
                no_improve_steps += 1

            if no_improve_steps >= self._patience:
                early_stopped = True
                logger.info(
                    "Beam search early stop at step %d (patience=%d).",
                    step, self._patience,
                )
                break

        # Fill any unassigned signs (signs not yet given a real phoneme in this beam item).
        # Skip signs that all beam items already have a concrete assignment for — this
        # happens when MCMC seeds provide a complete map and max_depth < n_signs.
        assigned_set = set(ordered_signs[:n_steps])
        remaining = [
            s for s in ordered_signs
            if s not in assigned_set
            and any(hyp.phoneme_map.get(s) is None for hyp in beam)
        ]
        if remaining:
            beam = self._fill_remaining(beam, remaining, sign_positions)

        top_k = sorted(beam, key=lambda h: h.normalised_score(self._alpha), reverse=True)[
            : self._top_k
        ]

        logger.info(
            "Beam search complete: %d steps, best norm score=%.4f, early_stop=%s",
            n_steps, top_k[0].normalised_score(self._alpha) if top_k else -math.inf,
            early_stopped,
        )

        return BeamSearchResult(
            top_hypotheses=top_k,
            n_signs=len(sign_ids),
            n_steps=n_steps,
            early_stopped=early_stopped,
        )

    # ------------------------------------------------------------------
    # Core beam operations
    # ------------------------------------------------------------------

    def _expand(
        self,
        beam: list[BeamHypothesis],
        sign: str,
        sign_positions: dict[str, list[list[int]]],
    ) -> list[BeamHypothesis]:
        """Expand every beam item by assigning each phoneme to ``sign``.

        Uses incremental :meth:`~LMScorer.score_delta` instead of full-corpus
        rescoring.  Phoneme sequences are materialised only for the surviving
        beam members, keeping allocation proportional to ``beam_width`` rather
        than ``beam_width × |phoneme_inventory|``.
        """
        positions_per_seq = sign_positions[sign]

        # Phase 1: score all (hyp, phoneme) pairs via cheap delta computation.
        # Each entry: (beam_idx, phoneme, new_log_score, normalised_score).
        candidates: list[tuple[int, str, float, float]] = []

        for hyp_idx, hyp in enumerate(beam):
            old_ph = hyp.phoneme_map.get(sign, _MASK_TOKEN)
            for phoneme in self._phoneme_inventory:
                if phoneme == old_ph:
                    delta = 0.0
                else:
                    delta = sum(
                        self._scorer.score_delta(
                            hyp.phoneme_seqs[seq_idx],
                            positions,
                            [old_ph] * len(positions),
                            [phoneme] * len(positions),
                        )
                        for seq_idx, positions in enumerate(positions_per_seq)
                        if positions
                    )
                new_lp = hyp.log_score + delta
                if new_lp < self._prune_threshold:
                    continue
                new_depth = hyp.depth + 1
                norm = new_lp / (new_depth ** self._alpha) if new_depth > 0 else -math.inf
                candidates.append((hyp_idx, phoneme, new_lp, norm))

        if not candidates:
            # Fallback: accept best phoneme per hypothesis regardless of threshold.
            for hyp_idx, hyp in enumerate(beam):
                old_ph = hyp.phoneme_map.get(sign, _MASK_TOKEN)
                best_phoneme: str | None = None
                best_lp = -math.inf
                for phoneme in self._phoneme_inventory:
                    if phoneme == old_ph:
                        delta = 0.0
                    else:
                        delta = sum(
                            self._scorer.score_delta(
                                hyp.phoneme_seqs[seq_idx],
                                positions,
                                [old_ph] * len(positions),
                                [phoneme] * len(positions),
                            )
                            for seq_idx, positions in enumerate(positions_per_seq)
                            if positions
                        )
                    new_lp = hyp.log_score + delta
                    if new_lp > best_lp:
                        best_lp = new_lp
                        best_phoneme = phoneme
                if best_phoneme is None:
                    # Every phoneme scored -inf; assign first arbitrarily.
                    best_phoneme = self._phoneme_inventory[0]
                new_depth = hyp.depth + 1
                norm = best_lp / (new_depth ** self._alpha) if new_depth > 0 else -math.inf
                candidates.append((hyp_idx, best_phoneme, best_lp, norm))

        # Phase 2: select top beam_width and materialise phoneme_seqs.
        candidates.sort(key=lambda c: c[3], reverse=True)
        survivors: list[BeamHypothesis] = []
        for hyp_idx, phoneme, new_lp, _ in candidates:
            if len(survivors) >= self._beam_width:
                break
            hyp = beam[hyp_idx]
            new_map = dict(hyp.phoneme_map)
            new_map[sign] = phoneme
            # Copy only the sequences that contain this sign.
            new_seqs = [list(seq) for seq in hyp.phoneme_seqs]
            for seq_idx, positions in enumerate(positions_per_seq):
                for pos in positions:
                    new_seqs[seq_idx][pos] = phoneme
            survivors.append(BeamHypothesis(
                phoneme_map=new_map,
                log_score=new_lp,
                depth=hyp.depth + 1,
                phoneme_seqs=new_seqs,
            ))

        return survivors

    def _partial_log_score(
        self,
        phoneme_map: PhonemeMap,
        corpus_sequences: list[list[str]],
    ) -> float:
        """Score a (possibly partial) assignment against all corpus sequences.

        Unmapped signs are translated to ``<MASK>`` tokens; these are
        treated as OOV by the LM scorer (floor-penalised).
        """
        total = 0.0
        for seq in corpus_sequences:
            translated = [phoneme_map.get(g, _MASK_TOKEN) for g in seq]
            result = self._scorer.score(translated)
            if math.isfinite(result.ensemble_log_prob):
                total += result.ensemble_log_prob
            else:
                total -= 100.0
        return total

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _init_beam(
        self,
        seed_hypotheses: list[MCMCSample] | None,
        corpus_sequences: list[list[str]],
    ) -> list[BeamHypothesis]:
        n_seqs = len(corpus_sequences)
        seq_lengths = [len(s) for s in corpus_sequences]

        def _masked_seqs() -> list[list[str]]:
            return [[_MASK_TOKEN] * seq_lengths[i] for i in range(n_seqs)]

        if seed_hypotheses:
            beam: list[BeamHypothesis] = []
            for s in seed_hypotheses[: self._beam_width]:
                seqs = [
                    [s.phoneme_map.get(tok, _MASK_TOKEN) for tok in seq]
                    for seq in corpus_sequences
                ]
                beam.append(BeamHypothesis(
                    phoneme_map=dict(s.phoneme_map),
                    log_score=s.log_posterior,
                    depth=len(s.phoneme_map),
                    phoneme_seqs=seqs,
                ))
            if len(beam) < self._beam_width:
                beam += [
                    BeamHypothesis(phoneme_map={}, log_score=0.0, depth=0,
                                   phoneme_seqs=_masked_seqs())
                    for _ in range(self._beam_width - len(beam))
                ]
            return beam

        return [
            BeamHypothesis(phoneme_map={}, log_score=0.0, depth=0,
                           phoneme_seqs=_masked_seqs())
            for _ in range(self._beam_width)
        ]

    @staticmethod
    def _order_signs_by_frequency(
        sign_ids: list[str],
        corpus_sequences: list[list[str]],
    ) -> list[str]:
        """Return ``sign_ids`` sorted by descending corpus frequency."""
        freq: dict[str, int] = {s: 0 for s in sign_ids}
        for seq in corpus_sequences:
            for token in seq:
                if token in freq:
                    freq[token] += 1
        return sorted(sign_ids, key=lambda s: freq[s], reverse=True)

    def _fill_remaining(
        self,
        beam: list[BeamHypothesis],
        remaining_signs: list[str],
        sign_positions: dict[str, list[list[int]]],
    ) -> list[BeamHypothesis]:
        """Greedily assign remaining signs (best phoneme per sign, per hypothesis)."""
        for sign in remaining_signs:
            positions_per_seq = sign_positions[sign]
            updated: list[BeamHypothesis] = []
            for hyp in beam:
                old_ph = hyp.phoneme_map.get(sign, _MASK_TOKEN)
                best_phoneme: str | None = None
                best_lp = -math.inf
                for phoneme in self._phoneme_inventory:
                    if phoneme == old_ph:
                        new_lp = hyp.log_score
                    else:
                        delta = sum(
                            self._scorer.score_delta(
                                hyp.phoneme_seqs[seq_idx],
                                positions,
                                [old_ph] * len(positions),
                                [phoneme] * len(positions),
                            )
                            for seq_idx, positions in enumerate(positions_per_seq)
                            if positions
                        )
                        new_lp = hyp.log_score + delta
                    if new_lp > best_lp:
                        best_lp = new_lp
                        best_phoneme = phoneme
                if best_phoneme is None:
                    # Every phoneme scored -inf; assign first arbitrarily.
                    best_phoneme = self._phoneme_inventory[0]
                new_map = dict(hyp.phoneme_map)
                new_map[sign] = best_phoneme
                new_seqs = [list(seq) for seq in hyp.phoneme_seqs]
                for seq_idx, positions in enumerate(positions_per_seq):
                    for pos in positions:
                        new_seqs[seq_idx][pos] = best_phoneme
                updated.append(BeamHypothesis(
                    phoneme_map=new_map,
                    log_score=best_lp,
                    depth=hyp.depth + 1,
                    phoneme_seqs=new_seqs,
                ))
            beam = updated
        return beam
