"""
hackingrongo.zone_c.lm_scoring
============================

External Polynesian language model scoring for Zone C.

This module is deliberately kept outside the trained pipeline.  The LMs
are pre-built and fixed so that (a) a wrong LM does not produce
confidently wrong training gradients, and (b) you can swap languages,
re-score without retraining, and run adversarial multi-hypothesis scoring
across all four Polynesian variants simultaneously.

A phoneme assignment map π : {sign_id → phoneme} translates a glyph-token
sequence to a phoneme/syllable sequence, which is scored by each LM at the
configured n-gram orders.  An ensemble score is computed as a
weight-normalised sum of the best-available-order log-probabilities.

**Syllable tokenisation**: when ``cfg.data.lm_tokenization_level == "syllable"``
(matching the level used by ``build_all_lms``), each phoneme token is split into
CV syllables before n-gram lookup.  A sign assigned "ariki" is scored as
["a", "ri", "ki"], not ["ariki"].  This is the correct granularity for LMs
trained at the syllable level.

Language model files are JSON, written by
:meth:`~hackingrongo.data.rapa_nui_corpus.NGramLM.save`.  Missing files
are logged as warnings and skipped; scoring proceeds on whatever languages
are available.

Public API
----------
``PhonemeMap``
    Type alias: ``dict[str, str]`` mapping sign ID → phoneme/syllable.

``LMScoringResult``
    Dataclass holding per-language scores, ensemble log-probability,
    OOV count, and coverage rate.

``LMScorer``
    Primary interface.  ``score(phoneme_sequence)`` and
    ``score_with_map(glyph_sequence, phoneme_map)`` are the main entry
    points.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from omegaconf import DictConfig, OmegaConf

if TYPE_CHECKING:
    from hackingrongo.data.rapa_nui_corpus import NGramLM

logger = logging.getLogger(__name__)

# Type alias: sign_id → phoneme/syllable string
PhonemeMap = dict[str, str]


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class LMScoringResult:
    """Scoring result for one phoneme-assignment hypothesis.

    Attributes
    ----------
    sequence : list[str]
        Phoneme sequence produced by applying the assignment map.
    per_language : dict[str, dict[str, float]]
        ``{language: {order_str: log2_prob}}`` for each configured
        language and n-gram order.
    ensemble_log_prob : float
        Ensemble log₂-probability: weight-normalised sum of the
        best-available-order per-language scores.
    n_oov : int
        Total number of n-grams that fell back to the OOV floor across
        all languages and orders.
    coverage : float
        Fraction of scored n-grams that were found in at least one LM
        (1.0 − oov_rate, averaged across languages).
    """

    sequence: list[str]
    per_language: dict[str, dict[str, float]] = field(default_factory=dict)
    ensemble_log_prob: float = -math.inf
    n_oov: int = 0
    coverage: float = 0.0


# ---------------------------------------------------------------------------
# LMScorer
# ---------------------------------------------------------------------------


class LMScorer:
    """Loads and queries Polynesian language models for hypothesis scoring.

    Parameters
    ----------
    cfg : DictConfig
        Root Hydra config.  Reads ``cfg.zone_c.lm_scoring``.
    data_root : Path
        Project data root (``hydra.utils.get_original_cwd()``).  LM JSON
        files are resolved relative to this.
    """

    def __init__(self, cfg: DictConfig, data_root: Path) -> None:
        from hackingrongo.data.rapa_nui_corpus import NGramLM  # local import

        lm_cfg = cfg.zone_c.lm_scoring
        self._tokenization_level: str = str(
            OmegaConf.select(cfg, "data.lm_tokenization_level", default="word")
        )
        self._orders: list[int] = [int(o) for o in lm_cfg.ngram_orders]
        self._oov_log_prob: dict[int, float] = {
            int(k): float(v) for k, v in lm_cfg.oov_log_prob_per_order.items()
        }
        self._ensemble_weights: dict[str, float] = dict(lm_cfg.ensemble_weights)
        self._languages: list[str] = list(lm_cfg.lms)

        # Validate ensemble weights at construction time — a missing key silently
        # zero-weights the language during scoring, which skews normalisation and
        # produces wrong scores hours into an MCMC run.
        for _lang in self._languages:
            if _lang not in self._ensemble_weights:
                raise ValueError(
                    f"Language '{_lang}' is listed in lm_scoring.lms but has no "
                    f"entry in lm_scoring.ensemble_weights. Add it to config.yaml "
                    f"or remove it from lms. Current weights: "
                    f"{dict(self._ensemble_weights)}"
                )
            if self._ensemble_weights[_lang] == 0.0:
                logger.warning(
                    "Language '%s' has ensemble_weight=0.0 — it will not "
                    "contribute to scoring. Intentional?", _lang,
                )

        # _lms[language][order] = NGramLM | None
        self._lms: dict[str, dict[int, NGramLM | None]] = {}

        primary_path_tmpl: dict[str, Path] = {
            lang: data_root / str(lm_cfg.lm_files[lang])
            for lang in self._languages
        }

        max_order = max(self._orders)
        for lang in self._languages:
            self._lms[lang] = {}
            primary = primary_path_tmpl[lang]

            for order in self._orders:
                if order == max_order:
                    lm_path = primary
                else:
                    lm_path = primary.parent / (
                        primary.stem + f"_order{order}" + primary.suffix
                    )

                if not lm_path.exists():
                    logger.warning(
                        "LM file not found for '%s' order %d: %s — "
                        "this language/order will use the OOV floor.",
                        lang, order, lm_path,
                    )
                    self._lms[lang][order] = None
                    continue

                try:
                    self._lms[lang][order] = NGramLM.load(lm_path)
                    logger.info(
                        "Loaded LM '%s' order %d from %s.", lang, order, lm_path
                    )
                except Exception as exc:
                    logger.error(
                        "Failed to load LM '%s' order %d from %s: %s",
                        lang, order, lm_path, exc,
                    )
                    self._lms[lang][order] = None

        # Audit: warn loudly if every order for a language failed to load.
        # Silent fallback to the OOV floor makes that LM's ensemble weight
        # meaningless — the caller needs to know about this.
        missing_entirely = [
            lang for lang in self._languages
            if all(v is None for v in self._lms[lang].values())
        ]
        if missing_entirely:
            weight_lost = sum(
                self._ensemble_weights.get(lang, 0.0) for lang in missing_entirely
            )
            logger.error(
                "LMScorer: ALL orders missing for language(s) %s "
                "(%.0f%% of ensemble weight will use the OOV floor). "
                "Run `python scripts/build_language_models.py` to build the missing LMs.",
                missing_entirely,
                weight_lost * 100.0,
            )

    # ------------------------------------------------------------------
    # Primary scoring interface
    # ------------------------------------------------------------------

    def score(self, phoneme_sequence: list[str]) -> LMScoringResult:
        """Score a phoneme sequence against all configured LMs.

        Parameters
        ----------
        phoneme_sequence : list[str]
            Sequence of phoneme/syllable tokens to score.

        Returns
        -------
        LMScoringResult
        """
        result = LMScoringResult(sequence=list(phoneme_sequence))
        if not phoneme_sequence:
            return result

        # Expand word-level tokens to CV syllables when the LM was trained at
        # syllable level.  result.sequence retains the original word-level tokens.
        expanded_sequence, _ = self._expand_tokens(phoneme_sequence)

        total_ngrams = 0
        total_oov = 0
        ensemble_lp = 0.0
        total_weight = 0.0

        for lang in self._languages:
            lang_scores: dict[str, float] = {}
            best_lp: float | None = None
            best_found_order: int | None = None

            for order in self._orders:
                lm = self._lms[lang].get(order)
                oov_floor = self._oov_log_prob.get(order, -20.0)
                if lm is None:
                    lang_scores[str(order)] = oov_floor * len(expanded_sequence)
                    continue

                lp, n_ngrams, n_oov = self._score_lm(lm, expanded_sequence, order)
                lang_scores[str(order)] = lp

                # Track the highest-order LM that is actually loaded.
                # Only accumulate n-grams/OOV counts for the best (highest) order
                # so they are not double-counted across orders for the same language.
                if best_found_order is None or order > best_found_order:
                    best_lp = lp
                    best_found_order = order
                    best_n_ngrams = n_ngrams
                    best_n_oov = n_oov

            result.per_language[lang] = lang_scores

            if best_lp is not None:
                weight = self._ensemble_weights.get(lang, 0.0)
                ensemble_lp += weight * best_lp
                total_weight += weight
                total_ngrams += best_n_ngrams
                total_oov += best_n_oov

        if total_weight > 0.0:
            result.ensemble_log_prob = ensemble_lp / total_weight

        result.n_oov = total_oov
        result.coverage = (
            1.0 - total_oov / total_ngrams if total_ngrams > 0 else 0.0
        )
        return result

    def score_with_map(
        self,
        glyph_sequence: list[str],
        phoneme_map: PhonemeMap,
    ) -> LMScoringResult:
        """Apply a phoneme assignment map to a glyph sequence and score it.

        Signs not present in ``phoneme_map`` are mapped to ``"<UNK>"``.

        Parameters
        ----------
        glyph_sequence : list[str]
            Sequence of sign/glyph identifiers (e.g. Horley codes).
        phoneme_map : PhonemeMap
            Partial or complete assignment ``{sign_id: phoneme}``.

        Returns
        -------
        LMScoringResult
        """
        phoneme_seq = [phoneme_map.get(g, "<UNK>") for g in glyph_sequence]
        return self.score(phoneme_seq)

    def score_delta(
        self,
        phoneme_seq: list[str],
        changed_positions: list[int],
        old_phonemes: list[str],
        new_phonemes: list[str],
    ) -> float:
        """Compute the change in weighted ensemble log-prob from updating
        specific positions in a phoneme sequence.

        Only n-grams that overlap a changed position are re-evaluated;
        all others cancel out in the delta.  This is the key primitive
        for incremental MCMC and beam-search scoring.

        Parameters
        ----------
        phoneme_seq : list[str]
            Current (pre-change) phoneme sequence.
        changed_positions : list[int]
            0-based indices into ``phoneme_seq`` that will change.
        old_phonemes : list[str]
            Phoneme currently at each position (same length as
            ``changed_positions``).
        new_phonemes : list[str]
            Replacement phoneme for each position.

        Returns
        -------
        float
            Δ ensemble log₂-probability (positive = proposal is better).
        """
        if not changed_positions:
            return 0.0

        if self._tokenization_level == "syllable":
            # Expand only the changed tokens (O(1) per change) to check
            # whether syllable counts match.  For the default single-syllable
            # Rapa Nui inventory every phoneme maps to exactly one syllable,
            # so this is a no-op and the O(k) incremental window path is used.
            # Variable-length expansion (e.g. "ariki"→3 vs "manu"→2) falls
            # back to full rescoring (rare / never for the default inventory).
            old_syls_list: list[list[str]] = []
            new_syls_list: list[list[str]] = []
            for old_ph, new_ph in zip(old_phonemes, new_phonemes):
                old_exp, _ = self._expand_tokens([old_ph])
                new_exp, _ = self._expand_tokens([new_ph])
                old_syls_list.append(old_exp)
                new_syls_list.append(new_exp)

            if any(len(o) != len(n) for o, n in zip(old_syls_list, new_syls_list)):
                # Variable-length: positions shift — must full-rescore.
                new_seq = list(phoneme_seq)
                for pos, new_ph in zip(changed_positions, new_phonemes):
                    new_seq[pos] = new_ph
                old_result = self.score(phoneme_seq)
                new_result = self.score(new_seq)
                return new_result.ensemble_log_prob - old_result.ensemble_log_prob

            is_one_to_one = all(len(s) == 1 for s in old_syls_list)
            if is_one_to_one:
                # Syllable positions == word positions and phoneme_seq IS the
                # expanded sequence.  Skip full-sequence expansion entirely.
                expanded_seq: list[str] = phoneme_seq  # read-only alias
                syl_positions = changed_positions
                syl_old = old_phonemes
                syl_new = new_phonemes
            else:
                # Same count but multi-syllable tokens: expand the full
                # sequence once to get the correct syllable positions.
                expanded_seq, ranges = self._expand_tokens(phoneme_seq)
                syl_positions = []
                syl_old = []
                syl_new = []
                for pos, o_syls, n_syls in zip(
                    changed_positions, old_syls_list, new_syls_list
                ):
                    start = ranges[pos][0]
                    for i in range(len(o_syls)):
                        syl_positions.append(start + i)
                        syl_old.append(o_syls[i])
                        syl_new.append(n_syls[i])

            # Incremental window scoring on the expanded sequence — O(k).
            ensemble_delta = 0.0
            total_weight = 0.0
            for lang in self._languages:
                best_lm = None
                best_order = None
                for order in sorted(self._orders, reverse=True):
                    lm = self._lms[lang].get(order)
                    if lm is not None:
                        best_lm = lm
                        best_order = order
                        break
                if best_lm is None or best_order is None:
                    continue
                weight = self._ensemble_weights.get(lang, 0.0)
                if weight == 0.0:
                    continue
                oov_floor = self._oov_log_prob.get(best_order, -20.0)
                lang_delta = self._score_delta_lm(
                    best_lm, expanded_seq,
                    syl_positions, syl_old, syl_new,
                    best_order, oov_floor,
                )
                ensemble_delta += weight * lang_delta
                total_weight += weight
            if total_weight > 0.0:
                ensemble_delta /= total_weight
            return ensemble_delta

        ensemble_delta = 0.0
        total_weight = 0.0

        for lang in self._languages:
            # Use the highest-order LM available for this language.
            best_lm = None
            best_order = None
            for order in sorted(self._orders, reverse=True):
                lm = self._lms[lang].get(order)
                if lm is not None:
                    best_lm = lm
                    best_order = order
                    break

            if best_lm is None:
                continue

            weight = self._ensemble_weights.get(lang, 0.0)
            if weight == 0.0:
                continue

            oov_floor = self._oov_log_prob.get(best_order, -20.0)
            lang_delta = self._score_delta_lm(
                best_lm, phoneme_seq,
                changed_positions, old_phonemes, new_phonemes,
                best_order, oov_floor,
            )
            ensemble_delta += weight * lang_delta
            total_weight += weight

        if total_weight > 0.0:
            ensemble_delta /= total_weight
        return ensemble_delta

    def _score_delta_lm(
        self,
        lm: "NGramLM",
        phoneme_seq: list[str],
        changed_positions: list[int],
        old_phonemes: list[str],
        new_phonemes: list[str],
        order: int,
        oov_floor: float,
    ) -> float:
        """Compute log-prob delta for one LM from replacing phonemes at
        specific positions.

        Identifies the minimal set of affected n-gram windows (those that
        overlap at least one changed position), evaluates old and new
        log-probs, and returns their sum-of-differences.
        """
        padded_offset = order - 1
        padded = ["<s>"] * padded_offset + phoneme_seq + ["</s>"]
        padded_len = len(padded)

        # Map padded position → new phoneme for fast lookup.
        pos_to_new: dict[int, str] = {
            pos + padded_offset: new_ph
            for pos, new_ph in zip(changed_positions, new_phonemes)
        }

        # Collect every n-gram start index that overlaps a changed token.
        affected: set[int] = set()
        for pos in changed_positions:
            padded_pos = pos + padded_offset
            lo = max(0, padded_pos - (order - 1))
            hi = min(padded_pos + 1, padded_len - order + 1)
            for start in range(lo, hi):
                affected.add(start)

        delta = 0.0
        for start in affected:
            ngram_old = tuple(padded[start : start + order])
            ngram_new = tuple(
                pos_to_new.get(start + i, padded[start + i])
                for i in range(order)
            )
            lp_old = lm.log_prob(ngram_old)
            lp_new = lm.log_prob(ngram_new)
            delta += (
                (lp_new if math.isfinite(lp_new) else oov_floor)
                - (lp_old if math.isfinite(lp_old) else oov_floor)
            )

        return delta

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    @property
    def languages_available(self) -> list[str]:
        """Languages for which at least one LM order was successfully loaded."""
        return [
            lang for lang in self._languages
            if any(lm is not None for lm in self._lms[lang].values())
        ]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _expand_tokens(
        self, phoneme_sequence: list[str]
    ) -> tuple[list[str], list[tuple[int, int]]]:
        """Expand phoneme tokens into CV syllables when tokenization_level='syllable'.

        Returns ``(expanded_tokens, ranges)`` where ``ranges[i]`` is the
        ``[start, end)`` slice in ``expanded_tokens`` corresponding to
        ``phoneme_sequence[i]``.  In word mode both are identity mappings.
        """
        if self._tokenization_level != "syllable":
            return (
                list(phoneme_sequence),
                [(i, i + 1) for i in range(len(phoneme_sequence))],
            )

        from hackingrongo.data.rapa_nui_corpus import tokenize_text  # local import

        expanded: list[str] = []
        ranges: list[tuple[int, int]] = []
        for token in phoneme_sequence:
            start = len(expanded)
            if token.startswith("<") and token.endswith(">"):
                # Keep special tokens (e.g. "<UNK>") as-is — they are not words.
                expanded.append(token)
            else:
                syls = tokenize_text(token, "syllable")
                expanded.extend(syls if syls else [token])
            ranges.append((start, len(expanded)))
        return expanded, ranges

    def _score_lm(
        self,
        lm: "NGramLM",
        tokens: list[str],
        order: int,
    ) -> tuple[float, int, int]:
        """Score ``tokens`` under ``lm``, returning (total_log2_prob, n_ngrams, n_oov).

        Uses ``lm.score_sequence`` which handles boundary padding internally.
        OOV n-grams (those returning ``-inf``) are replaced by the configured
        per-order floor and counted separately.
        """
        oov_floor = self._oov_log_prob.get(order, -20.0)

        # score_sequence returns sum of log2 probs including boundary tokens.
        # We need per-ngram scores to count OOV; iterate manually.
        padded = ["<s>"] * (order - 1) + tokens + ["</s>"]
        n_ngrams = len(padded) - (order - 1)
        total_lp = 0.0
        n_oov = 0

        for i in range(order - 1, len(padded)):
            ngram = tuple(padded[i - order + 1 : i + 1])
            lp = lm.log_prob(ngram)

            if not math.isfinite(lp):
                total_lp += oov_floor
                n_oov += 1
            else:
                total_lp += lp

        return total_lp, n_ngrams, n_oov
