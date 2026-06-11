"""
hackingrongo.data.rapa_nui_corpus
============================

Loads, cleans, and structures the Polynesian text corpora for language
model training.  This is the bridge between the raw text files in
``data/polynesian_texts/`` and the serialized n-gram LM pickles that
Zone C's :mod:`~hackingrongo.zone_c.lm_scoring` module consumes.

Design
------
* Bracket annotations ``[like this]`` mark scholarly reconstructions.
  Whether they are included in LM training is controlled by
  ``cfg.data.include_reconstructions`` (default ``false``).  Unlike
  Souza (2022), this decision is explicit and the bracketed lines are
  never silently discarded — they are always loaded and always tagged.

* Each language directory under ``data/polynesian_texts/`` must contain
  a ``metadata.json`` file listing source files with genre and citation.

* The :class:`NGramLM` uses interpolated modified Kneser-Ney discounts
  (Chen & Goodman 1999) with MLE unigram base.  KN continuation counts
  have been removed pending a larger training corpus; the model currently
  serialises only raw n-gram counts and discount parameters.

Public API
----------
``TextRecord``
    Cleaned text record with provenance metadata.

``NGramLM``
    Serializable n-gram language model.  ``log_prob(ngram)`` and
    ``score_sequence(tokens)`` are the primary query methods.

``load_text_corpus``
    Load all text files for one language from the directory tree.

``tokenize_text``
    Split a text string into character, syllable, or word tokens.

``build_ngram_lm``
    Build one :class:`NGramLM` from a list of :class:`TextRecord` objects.

``build_all_lms``
    Build and serialize all LMs specified in ``cfg.zone_c.lm_scoring``.
"""

from __future__ import annotations

import json
import logging
import math
import re
import unicodedata
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from omegaconf import DictConfig

logger = logging.getLogger(__name__)

# Regex matching a bracketed scholarly reconstruction.
_BRACKET_RE: re.Pattern[str] = re.compile(r"\[([^\]]*)\]")

# Diacritic normalisation applied when ingesting pre-built n-gram frequency
# tables (e.g. the Hawaiian Corpus Project word-frequency files).  Maps
# Hawaiian macron vowels, okina variants, and common Unicode diacritics to
# their plain ASCII equivalents so that the smoothing LM tokens are
# consistent with the rest of the pipeline.
_FREQ_TABLE_NORMALISE_MAP: dict[str, str] = {
    "\u0101": "a", "\u0113": "e", "\u012b": "i", "\u014d": "o", "\u016b": "u",  # macrons
    "\u0103": "a", "\u0115": "e", "\u012d": "i", "\u014f": "o", "\u016d": "u",  # breves
    "\u00e1": "a", "\u00e9": "e", "\u00ed": "i", "\u00f3": "o", "\u00fa": "u",  # acutes
    "\u00e0": "a", "\u00e8": "e", "\u00ec": "i", "\u00f2": "o", "\u00f9": "u",  # graves
    "\u00e2": "a", "\u00ea": "e", "\u00ee": "i", "\u00f4": "o", "\u00fb": "u",  # circumflex
    "\u02bc": "",  # ʼ MODIFIER LETTER APOSTROPHE (okina variant)
    "\u02bb": "",  # ʻ MODIFIER LETTER TURNED COMMA (Hawaiian okina)
    "\u0027": "",  # plain apostrophe used as okina in some entries
    "\u2018": "",  # LEFT SINGLE QUOTATION MARK
    "\u2019": "",  # RIGHT SINGLE QUOTATION MARK
}

# First-line pattern that identifies a pre-built n-gram frequency table:
#   <integer> <TAB> <token> [<COMMA> <token> ...]
_FREQ_TABLE_RE: re.Pattern[str] = re.compile(r"^\d+\t\S")

# ---------------------------------------------------------------------------
# Per-language phonological validation for syllable-token filtering
# ---------------------------------------------------------------------------

# Structural (C)V validation lives in hackingrongo.data.phoneme_inventory —
# the single source of truth for the sign→phoneme search space.  The old
# character-set membership check admitted consonant clusters ("gra", "nta",
# "tto") whose characters are individually legal; structural validation
# rejects them.
from hackingrongo.data.phoneme_inventory import (  # noqa: E402
    clean_syllables,
    is_valid_syllable as _is_valid_syllable,
)

# Maps LM era names (from config) to the language whose inventory to use.
_ERA_LANGUAGE: dict[str, str] = {
    "pre_contact":  "rapanui",
    "post_contact": "old_rapa_nui",
    "smoothing":    "hawaiian",
}


# ---------------------------------------------------------------------------
# TextRecord
# ---------------------------------------------------------------------------


@dataclass
class TextRecord:
    """A single cleaned source text with full provenance.

    Attributes
    ----------
    text_id : str
        Unique identifier (typically ``<language>/<filename_stem>``).
    language : str
        ISO-like language label matching ``cfg.zone_c.lm_scoring.languages``
        (e.g. ``"old_rapa_nui"``, ``"maori"``).
    genre : str
        Genre label from ``metadata.json`` (e.g. ``"kaikai"``,
        ``"recitation"``).
    source : str
        Citation string from ``metadata.json``.
    lines : list[str]
        Cleaned text lines with brackets stripped or included according
        to ``include_reconstructions``.
    bracketed_lines : list[str]
        Lines that originally contained bracket annotations, retained
        verbatim for auditing regardless of ``include_reconstructions``.
    """

    text_id: str
    language: str
    genre: str
    source: str
    lines: list[str]
    bracketed_lines: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# NGramLM helpers (module-level for picklability)
# ---------------------------------------------------------------------------


def _defaultdict_int() -> defaultdict:
    """Module-level factory for picklable defaultdict(int)."""
    return defaultdict(int)


def _json_to_ngram_table(entries: list) -> dict:
    """
    Reconstruct an n-gram count table from its JSON-serialised form.

    Module-level (not nested) so the function is picklable by
    multiprocessing — required for parallel MCMC chain execution.

    Parameters
    ----------
    entries : list
        List of [context_list, counter_dict] pairs as written by
        NGramLM.save().

    Returns
    -------
    dict
        defaultdict mapping context tuples to defaultdict(int) counters.
    """
    table: dict = defaultdict(_defaultdict_int)
    for ctx_list, ctr in entries:
        table[tuple(ctx_list)] = defaultdict(int, ctr)
    return table


# ---------------------------------------------------------------------------
# NGramLM
# ---------------------------------------------------------------------------


class NGramLM:
    """Interpolated n-gram language model with modified Kneser-Ney discounts.

    Implements the discount estimation of Chen & Goodman (1999), computing
    three discount values (D1, D2, D3+) from one- and two-count statistics.
    The lower-order base distribution uses Laplace-smoothed MLE unigrams.

    KN continuation counts (``_cont``) have been removed pending a larger
    training corpus.  At the current scale (~1,345 pre-contact forms) the
    difference between full KN and interpolated add-α smoothing is negligible
    relative to OOV rate.  They will be re-introduced if the corpus grows
    substantially.

    Parameters
    ----------
    order : int
        N-gram order (e.g. ``3`` for trigram).
    language : str
        Language label for logging and serialisation metadata.

    Notes
    -----
    The model is trained incrementally via :meth:`update` and finalised
    via :meth:`finalise` (which computes discounts and caches interpolated
    unigrams).  Do not call :meth:`log_prob` before :meth:`finalise`.
    """

    def __init__(self, order: int, language: str) -> None:
        self.order: int = order
        self.language: str = language
        self._finalised: bool = False

        # Raw n-gram counts for each sub-order k = 1 … order.
        # _counts[k][context_tuple][word] = count
        self._counts: dict[int, dict[tuple, dict[str, int]]] = {
            k: defaultdict(_defaultdict_int) for k in range(1, order + 1)
        }

        # Discount parameters per order, set in finalise().
        self._discounts: dict[int, tuple[float, float, float]] = {}

        # Vocabulary
        self._vocab: set[str] = set()

        # Cache: unigram log probs (KN continuation at order 1)
        self._unigram_log_prob: dict[str, float] = {}

        # Cache: linear unigram probabilities for O(1) base-case lookup in _kn_prob.
        # Avoids the O(vocab) sum() that the recursive version recomputed every call.
        self._unigram_linear: dict[str, float] = {}
        self._unigram_linear_unk: float = 0.0

        # Precomputed per-context aggregate statistics (populated by _precompute_ctx_stats).
        # _ctx_stats[k][context] = (ctx_total, n1_types, n2_types, n3plus_types)
        self._ctx_stats: dict[int, dict[tuple, tuple[int, int, int, int]]] = {}

        # Inference-time log_prob result cache.  Populated lazily on first call;
        # never persisted (not included in save/load).  Bounded in practice by
        # vocab^order — for the 50-phoneme MCMC inventory at order 5 this is at
        # most ~184 M entries but in practice only the n-grams that actually
        # appear in generated sequences accumulate, which is much smaller.
        self._lp_cache: dict[tuple[str, ...], float] = {}

        # Optional cross-lingual backoff for unseen high-order contexts.
        # Set via set_backoff_lm(); used in _kn_prob for orders >= _backoff_from_order.
        # The path is persisted in save() so load() can restore the reference.
        self._backoff_lm: "NGramLM | None" = None
        self._backoff_from_order: int = 4
        self._backoff_alpha: float = 0.15
        self._backoff_lm_path: "str | None" = None

        # Permanent memo for backoff-LM probabilities, keyed by
        # (word, truncated_context).  Unlike _lp_cache this is never
        # cleared between scoring passes: the backoff LM is immutable
        # after finalise() and its probabilities do not depend on any
        # sign→phoneme mapping, so entries stay valid for the process
        # lifetime.  Bounded by (backoff vocab)^(backoff order) — tens of
        # thousands of entries for syllable alphabets.  This matters in
        # hot loops (measure_pgood samples 10K+ assignments): for k ≥
        # _backoff_from_order the same (word, 2-token-context) backoff
        # walk would otherwise be recomputed at every order level of
        # every window.
        self._backoff_memo: dict[tuple, float] = {}

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def update(self, tokens: list[str]) -> None:
        """Incorporate a token sequence into the model's counts.

        Parameters
        ----------
        tokens : list[str]
            Tokenized text line.  The method adds ``<s>`` padding at
            the start and ``</s>`` at the end.
        """
        if self._finalised:
            raise RuntimeError(
                "Cannot update a finalised NGramLM."
            )
        padded = (
            ["<s>"] * (self.order - 1)
            + tokens
            + ["</s>"]
        )
        self._vocab.update(tokens)
        self._vocab.update({"<s>", "</s>"})

        for i in range(self.order - 1, len(padded)):
            word = padded[i]
            for k in range(1, self.order + 1):
                context = tuple(padded[i - k + 1 : i])
                self._counts[k][context][word] += 1

    def ingest_ngram_table(self, path: Path) -> int:
        """Load a pre-built n-gram frequency table directly into ``_counts``.

        Expects the format produced by the Hawaiian Corpus Project
        (https://github.com/dohliam/hawaiian-corpus) and compatible tools::

            <count> TAB <w1>[,<w2>[,...,<wk>]]

        The n-gram order ``k`` is inferred from the number of
        comma-separated tokens on each line.  Only orders ≤ ``self.order``
        are loaded; higher-order entries are silently skipped (they cannot
        be stored in the current model's ``_counts`` structure).

        Tokens are normalised with :data:`_FREQ_TABLE_NORMALISE_MAP`
        (macrons, okina variants, and other Polynesian diacritics → ASCII)
        before insertion.

        ``_cont`` continuation counts are not populated or serialised; this
        does not affect scoring correctness.

        Parameters
        ----------
        path : Path
            Path to the frequency-table text file.

        Returns
        -------
        int
            Number of entries successfully loaded.

        Raises
        ------
        RuntimeError
            If called after :meth:`finalise`.
        """
        if self._finalised:
            raise RuntimeError("Cannot ingest into a finalised NGramLM.")

        def _norm(token: str) -> str:
            token = unicodedata.normalize("NFC", token.lower().strip())
            return "".join(_FREQ_TABLE_NORMALISE_MAP.get(ch, ch) for ch in token)

        n_loaded = 0
        n_skipped_order = 0
        raw_text = path.read_text(encoding="utf-8")
        for raw_line in raw_text.splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            tab_pos = line.find("\t")
            if tab_pos < 0:
                continue
            try:
                count = int(line[:tab_pos])
            except ValueError:
                continue
            tokens = [_norm(t) for t in line[tab_pos + 1:].split(",") if t.strip()]
            if not tokens:
                continue
            k = len(tokens)
            if k > self.order:
                n_skipped_order += 1
                continue
            context = tuple(tokens[:-1])
            word = tokens[-1]
            self._counts[k][context][word] += count
            self._vocab.update(tokens)
            n_loaded += 1

        if n_skipped_order:
            logger.debug(
                "ingest_ngram_table(%s): skipped %d entries with order > %d.",
                path.name, n_skipped_order, self.order,
            )
        logger.info(
            "ingest_ngram_table: loaded %d entries from %s into NGramLM[%s, order=%d].",
            n_loaded, path.name, self.language, self.order,
        )
        return n_loaded

    def finalise(self) -> None:
        """Estimate discount parameters and precompute unigram probs.

        Must be called once after all :meth:`update` calls and before
        any :meth:`log_prob` calls.
        """
        for k in range(1, self.order + 1):
            self._discounts[k] = self._estimate_discounts(k)

        # Precompute unigram probabilities — log form for scoring, linear for interpolation.
        total_unigrams = sum(self._counts[1][()].values())
        vocab_size = len(self._vocab)
        self._unigram_linear_unk = 1.0 / (total_unigrams + vocab_size)
        for word in self._vocab:
            cnt = self._counts[1][()].get(word, 0)
            p = (cnt + 1.0) / (total_unigrams + vocab_size)
            self._unigram_log_prob[word] = math.log(p)
            self._unigram_linear[word] = p

        self._precompute_ctx_stats()
        self._finalised = True
        logger.debug(
            "NGramLM[%s, order=%d] finalised: vocab=%d, discounts=%s.",
            self.language,
            self.order,
            len(self._vocab),
            self._discounts,
        )

    def _estimate_discounts(self, k: int) -> tuple[float, float, float]:
        """Estimate modified KN discount values for sub-order k.

        Uses Chen & Goodman (1999) formula:

        ``Y = n1 / (n1 + 2 * n2)``
        ``D_i = i - (i + 1) * Y * (n_{i+1} / n_i)`` for i = 1, 2, 3+

        where ``n_i`` = number of n-grams with exactly ``i`` occurrences.

        Returns (D1, D2, D3+) clamped to [0, 1].
        """
        n1 = n2 = n3 = n4 = 0
        for ctx_dict in self._counts[k].values():
            for cnt in ctx_dict.values():
                if cnt == 1:
                    n1 += 1
                elif cnt == 2:
                    n2 += 1
                elif cnt == 3:
                    n3 += 1
                elif cnt == 4:
                    n4 += 1

        if n1 == 0 or n2 == 0:
            # Fallback: simple absolute discounting with d=0.75
            return (0.75, 0.75, 0.75)

        Y = n1 / (n1 + 2.0 * n2)
        d1 = max(0.0, min(1.0, 1 - 2.0 * Y * (n2 / n1)))
        d2_raw = 2.0 - 3.0 * Y * (n3 / n2) if n3 > 0 and n2 > 0 else d1
        d2 = max(0.0, min(1.0, d2_raw))
        d3_raw = 3.0 - 4.0 * Y * (n4 / n3) if n4 > 0 and n3 > 0 else d2
        d3 = max(0.0, min(1.0, d3_raw))
        return (d1, d2, d3)

    def _precompute_ctx_stats(self) -> None:
        """Cache per-context aggregate statistics used by :meth:`_kn_prob`.

        Eliminates four O(vocab) passes per n-gram lookup in the scoring
        hot-path.  Called by :meth:`finalise` and :meth:`load`.
        """
        self._ctx_stats = {}
        for k in range(2, self.order + 1):
            stats: dict[tuple, tuple[int, int, int, int]] = {}
            for context, word_counts in self._counts[k].items():
                total = 0
                n1 = n2 = n3plus = 0
                for c in word_counts.values():
                    total += c
                    if c == 1:
                        n1 += 1
                    elif c == 2:
                        n2 += 1
                    else:
                        n3plus += 1
                stats[context] = (total, n1, n2, n3plus)
            self._ctx_stats[k] = stats

    def _precompute_unigram_cache(self) -> None:
        """Cache linear unigram probabilities for the _kn_prob base case.

        Called by load() after _counts is restored from JSON.  finalise()
        populates the same fields inline during its unigram loop.
        """
        uni = self._counts[1][()]
        total = sum(uni.values())
        V = len(self._vocab)
        self._unigram_linear_unk = 1.0 / (total + V)
        self._unigram_linear = {
            word: (uni.get(word, 0) + 1.0) / (total + V)
            for word in self._vocab
        }

    def set_backoff_lm(
        self,
        lm: "NGramLM",
        lm_path: "str | Path | None" = None,
        from_order: int = 4,
        alpha: float = 0.15,
    ) -> None:
        """Attach a cross-lingual backoff LM for unseen high-order contexts.

        When ``_kn_prob`` encounters an unseen context at order k ≥
        ``from_order``, it interpolates the lower-order probability with
        the backoff LM's probability at weight ``alpha``:

            P_final = (1 - alpha) * P_primary + alpha * P_backoff

        The typical use-case is attaching the Hawaiian smoothing LM to a
        5-gram Rapa Nui model so that unseen 4/5-gram contexts fall back
        to well-estimated Hawaiian probabilities rather than pure lower-
        order KN.

        The ``_lp_cache`` is cleared so previously cached values computed
        without backoff are invalidated.

        Parameters
        ----------
        lm : NGramLM
            Finalised backoff model.
        lm_path : str or Path, optional
            Path to the backoff model's JSON file.  Stored in save() so
            load() can restore the reference automatically.
        from_order : int
            Minimum order at which to apply backoff.  Default 4.
        alpha : float
            Interpolation weight for the backoff LM (0 < alpha < 1).
        """
        if not lm._finalised:
            raise RuntimeError("Backoff LM must be finalised before attaching.")
        self._backoff_lm = lm
        self._backoff_from_order = from_order
        self._backoff_alpha = alpha
        self._backoff_lm_path = str(lm_path) if lm_path is not None else None
        self._lp_cache.clear()  # invalidate cached values computed without backoff
        self._backoff_memo.clear()  # memo entries belong to the previous backoff LM
        logger.info(
            "NGramLM[%s, order=%d]: backoff LM attached (language=%s, from_order=%d, alpha=%.2f).",
            self.language, self.order, lm.language, from_order, alpha,
        )

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def log_prob(self, ngram: tuple[str, ...]) -> float:
        """Return the log₂ probability of the last element given context.

        Parameters
        ----------
        ngram : tuple[str, ...]
            Length ``order`` n-gram.  The last element is the predicted
            word; the preceding elements are the context.

        Returns
        -------
        float
            Log₂ probability (always ≤ 0).  Returns the configured OOV
            floor for words outside the vocabulary.

        Raises
        ------
        RuntimeError
            If called before :meth:`finalise`.
        """
        if not self._finalised:
            raise RuntimeError("Call finalise() before log_prob().")
        if len(ngram) != self.order:
            raise ValueError(
                f"ngram length {len(ngram)} != model order {self.order}."
            )
        cached = self._lp_cache.get(ngram)
        if cached is not None:
            return cached
        prob = self._kn_prob(ngram[-1], tuple(ngram[:-1]))
        result = math.log2(max(prob, 1e-300))
        self._lp_cache[ngram] = result
        return result

    def score_sequence(self, tokens: list[str]) -> float:
        """Sum of log₂ probs for all n-grams in the sequence.

        Parameters
        ----------
        tokens : list[str]
            Tokenized text.

        Returns
        -------
        float
            Total log₂ probability (always ≤ 0).
        """
        padded = ["<s>"] * (self.order - 1) + tokens + ["</s>"]
        total = 0.0
        for i in range(self.order - 1, len(padded)):
            ngram = tuple(padded[i - self.order + 1 : i + 1])
            total += self.log_prob(ngram)
        return total

    def _kn_prob(self, word: str, context: tuple[str, ...]) -> float:
        """Iteratively compute KN probability P_KN(word | context).

        Replaces the previous recursive implementation.  The recursion
        descended from order k down to 1 on every call, recomputing
        sum(_counts[1][()].values()) — O(vocab) — at the base case each
        time.  This iterative version:

        * reads the cached _unigram_linear table for the base case — O(1)
        * avoids Python function-call overhead at every order level
        * is semantically identical: unseen context at order k leaves the
          accumulated lower-order probability unchanged (equivalent to the
          recursive back-off)
        """
        # Base: Laplace-smoothed unigram (O(1) from cache).
        prob = self._unigram_linear.get(word, self._unigram_linear_unk)

        # Walk up from bigram to full order, interpolating when context is seen.
        for k in range(2, self.order + 1):
            ctx = context[-(k - 1):]
            ctx_stat = self._ctx_stats.get(k, {}).get(ctx)
            if ctx_stat is None:
                # Unseen context: back off to lower-order probability.
                # For high orders (>= _backoff_from_order) also blend in the
                # cross-lingual backoff LM if one has been attached.
                if (
                    self._backoff_lm is not None
                    and k >= self._backoff_from_order
                    and self._backoff_lm._finalised
                ):
                    # Truncate context to what the backoff LM can handle.
                    bo_ctx = ctx[-(self._backoff_lm.order - 1):]
                    memo_key = (word, bo_ctx)
                    bo_prob = self._backoff_memo.get(memo_key)
                    if bo_prob is None:
                        bo_prob = self._backoff_lm._kn_prob(word, bo_ctx)
                        self._backoff_memo[memo_key] = bo_prob
                    prob = (1.0 - self._backoff_alpha) * prob + self._backoff_alpha * bo_prob
                continue
            ctx_total, n1_types, n2_types, n3plus_types = ctx_stat
            d1, d2, d3 = self._discounts[k]
            raw_cnt = self._counts[k][ctx].get(word, 0)
            if raw_cnt == 1:
                d = d1
            elif raw_cnt == 2:
                d = d2
            else:
                d = d3
            numerator = max(raw_cnt - d, 0.0)
            lam = (d1 * n1_types + d2 * n2_types + d3 * n3plus_types) / ctx_total
            prob = numerator / ctx_total + lam * prob

        return prob

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def save(self, path: Path) -> None:
        """Serialise the model to a JSON file.

        Parameters
        ----------
        path : Path
            Destination file path.  Parent directories are created if
            they do not exist.
        """
        def _table_to_json(table: dict) -> list:
            return [[list(ctx), dict(ctr)] for ctx, ctr in table.items()]

        payload = {
            "order": self.order,
            "language": self.language,
            "finalised": self._finalised,
            "vocab": sorted(self._vocab),
            "unigram_log_prob": self._unigram_log_prob,
            "discounts": {str(k): list(v) for k, v in self._discounts.items()},
            "counts": {str(n): _table_to_json(t) for n, t in self._counts.items()},
            "backoff_lm_path": self._backoff_lm_path,
            "backoff_from_order": self._backoff_from_order,
            "backoff_alpha": self._backoff_alpha,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        logger.info(
            "NGramLM[%s, order=%d] saved to %s.", self.language, self.order, path
        )

    @classmethod
    def load(cls, path: Path) -> "NGramLM":
        """Load a serialised :class:`NGramLM` from a JSON file.

        Parameters
        ----------
        path : Path
            Path to the JSON file written by :meth:`save`.

        Returns
        -------
        NGramLM

        Raises
        ------
        FileNotFoundError
            If the file does not exist.
        """
        if not path.exists():
            raise FileNotFoundError(f"NGramLM JSON not found: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))

        model = cls(order=payload["order"], language=payload["language"])
        model._finalised = payload["finalised"]
        model._vocab = set(payload["vocab"])
        model._unigram_log_prob = payload["unigram_log_prob"]
        model._discounts = {
            int(k): tuple(v) for k, v in payload["discounts"].items()
        }
        model._counts = {
            int(n): _json_to_ngram_table(entries)
            for n, entries in payload["counts"].items()
        }
        # "cont" key in older JSON files is intentionally ignored — continuation
        # counts were removed; see NGramLM class docstring.
        model._precompute_ctx_stats()
        model._precompute_unigram_cache()

        # Restore optional cross-lingual backoff reference.
        backoff_path_str: str | None = payload.get("backoff_lm_path")
        if backoff_path_str is not None:
            model._backoff_lm_path = backoff_path_str
            model._backoff_from_order = int(payload.get("backoff_from_order", 4))
            model._backoff_alpha = float(payload.get("backoff_alpha", 0.15))
            bo_path = Path(backoff_path_str)
            if bo_path.exists():
                try:
                    model._backoff_lm = cls.load(bo_path)
                    logger.info(
                        "NGramLM[%s, order=%d]: backoff LM auto-loaded from %s.",
                        model.language, model.order, bo_path,
                    )
                except Exception as exc:
                    logger.warning(
                        "NGramLM[%s, order=%d]: failed to load backoff LM from %s: %s",
                        model.language, model.order, bo_path, exc,
                    )
            else:
                logger.warning(
                    "NGramLM[%s, order=%d]: backoff_lm_path %s not found — backoff disabled.",
                    model.language, model.order, bo_path,
                )

        logger.info(
            "NGramLM[%s, order=%d] loaded from %s.",
            model.language, model.order, path,
        )
        return model


# ---------------------------------------------------------------------------
# Text loading
# ---------------------------------------------------------------------------


def load_text_corpus(
    texts_dir: Path,
    language: str,
    cfg: DictConfig,
) -> list[TextRecord]:
    """Load all text files for one language from the directory tree.

    Parameters
    ----------
    texts_dir : Path
        Absolute path to ``data/polynesian_texts/<language>/``.
    language : str
        Language label (used in :attr:`TextRecord.language`).
    cfg : DictConfig
        Root Hydra config.  Reads ``cfg.data.include_reconstructions``
        to decide whether bracketed lines are included in
        :attr:`TextRecord.lines`.

    Returns
    -------
    list[TextRecord]
        One record per source text file found under ``texts_dir``.
        Returns an empty list with a WARNING if ``texts_dir`` does not
        exist or contains no ``.txt`` files.

    Notes
    -----
    Lines beginning with ``#`` are treated as comments and skipped.
    Empty lines are also skipped.  Bracket handling is explicit — lines
    containing brackets are always preserved in
    :attr:`TextRecord.bracketed_lines` regardless of the config flag.
    """
    include_reconstructions: bool = bool(cfg.data.include_reconstructions)

    if not texts_dir.is_dir():
        logger.warning(
            "Polynesian text directory not found for language '%s': %s",
            language,
            texts_dir,
        )
        return []

    # Load source metadata if available.
    metadata_path = texts_dir / "metadata.json"
    source_meta: dict[str, dict[str, Any]] = {}
    if metadata_path.exists():
        with metadata_path.open("r", encoding="utf-8") as fh:
            raw_meta: dict[str, Any] = json.load(fh)
        for entry in raw_meta.get("sources", []):
            source_meta[entry["file"]] = entry

    txt_files = sorted(texts_dir.glob("*.txt"))
    if not txt_files:
        logger.warning(
            "No .txt files found for language '%s' in %s.", language, texts_dir
        )
        return []

    records: list[TextRecord] = []

    for txt_path in txt_files:
        meta = source_meta.get(txt_path.name, {})
        genre: str = str(meta.get("genre", ""))
        source: str = str(meta.get("citation", str(txt_path)))
        text_id: str = f"{language}/{txt_path.stem}"

        raw_lines = txt_path.read_text(encoding="utf-8").splitlines()
        clean_lines: list[str] = []
        bracketed_lines: list[str] = []

        for raw in raw_lines:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue

            has_bracket = bool(_BRACKET_RE.search(line))
            if has_bracket:
                bracketed_lines.append(line)
                if include_reconstructions:
                    # Include line with brackets stripped.
                    clean_lines.append(_BRACKET_RE.sub(r"\1", line).strip())
                # else: bracketed lines are logged but not included in clean_lines
            else:
                clean_lines.append(line)

        records.append(
            TextRecord(
                text_id=text_id,
                language=language,
                genre=genre,
                source=source,
                lines=clean_lines,
                bracketed_lines=bracketed_lines,
            )
        )
        logger.debug(
            "Loaded text '%s': %d clean lines, %d bracketed lines.",
            text_id,
            len(clean_lines),
            len(bracketed_lines),
        )

    logger.info(
        "Loaded %d text record(s) for language '%s' (%d total clean lines).",
        len(records),
        language,
        sum(len(r.lines) for r in records),
    )
    return records


# ---------------------------------------------------------------------------
# Tokenisation
# ---------------------------------------------------------------------------

# Syllable-splitting heuristic for Polynesian languages (CV or V syllables).
_VOWELS: frozenset[str] = frozenset("aeiouāēīōū")
_CV_RE: re.Pattern[str] = re.compile(r"[^aeiouāēīōū]*[aeiouāēīōū]", re.IGNORECASE)


def _normalize_for_syllables(text: str) -> str:
    """Fold a source line to plain ASCII before CV syllable splitting.

    Decomposes to NFD and drops combining marks (so ``hā`` → ``ha``
    instead of leaving a combining macron stranded inside a "consonant"
    cluster), then applies the same diacritic/okina map used for
    pre-built frequency tables.  Vowel length is deliberately folded:
    the historical wordlists mark it too inconsistently to model.
    """
    decomposed = unicodedata.normalize("NFD", text.lower())
    stripped = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    return "".join(_FREQ_TABLE_NORMALISE_MAP.get(ch, ch) for ch in stripped)


def tokenize_text(text: str, level: str) -> list[str]:
    """Split a text string into tokens at the requested granularity.

    Parameters
    ----------
    text : str
        A single text line.
    level : str
        Tokenisation granularity:

        ``"char"``
            One token per character (excluding spaces).
        ``"syllable"``
            Heuristic CV/V syllable splitting based on Polynesian
            phonotactics.  Consonant clusters are left together with
            the following vowel.
        ``"word"``
            Split on whitespace.

    Returns
    -------
    list[str]
        Non-empty token strings.

    Raises
    ------
    ValueError
        If ``level`` is not one of the three supported values.
    """
    if level == "char":
        return [ch for ch in text if not ch.isspace()]
    if level == "word":
        return text.split()
    if level == "syllable":
        # Split on whitespace first so that spaces are never consumed as
        # consonant onset material by the CV regex.  Without this, a line
        # like "rakau jf rau" would produce the token " jf ra" (space + jf
        # treated as onset consonants of the 'ra' nucleus).
        syllables: list[str] = []
        for word in _normalize_for_syllables(text).split():
            syllables.extend(_CV_RE.findall(word))
        return syllables
    raise ValueError(
        f"Unsupported tokenisation level '{level}'. "
        "Choose char | syllable | word."
    )


# ---------------------------------------------------------------------------
# LM building
# ---------------------------------------------------------------------------


def build_ngram_lm(
    texts: list[TextRecord],
    order: int,
    language: str,
    tokenization_level: str = "syllable",
) -> NGramLM:
    """Build one :class:`NGramLM` from a list of text records.

    Parameters
    ----------
    texts : list[TextRecord]
        Loaded text records for the target language.
    order : int
        N-gram order.
    language : str
        Language label embedded in the returned model.
    tokenization_level : str
        Passed to :func:`tokenize_text`.

    Returns
    -------
    NGramLM
        Finalised model ready for :meth:`~NGramLM.log_prob` calls.
    """
    model = NGramLM(order=order, language=language)
    n_lines = 0
    n_tokens_rejected = 0
    for record in texts:
        for line in record.lines:
            raw_tokens = tokenize_text(line, tokenization_level)
            if tokenization_level == "syllable":
                tokens = clean_syllables(raw_tokens, language)
                n_tokens_rejected += len(raw_tokens) - len(tokens)
            else:
                tokens = raw_tokens
            if tokens:
                model.update(tokens)
                n_lines += 1
    model.finalise()
    logger.info(
        "Built NGramLM[%s, order=%d] from %d lines (%d syllable tokens rejected).",
        language, order, n_lines, n_tokens_rejected,
    )
    return model


def build_all_lms(cfg: DictConfig, project_root: Path) -> None:
    """Build and serialise era-stratified language models from lm_sources config.

    Iterates over ``cfg.zone_c.lm_scoring.lms`` (e.g. ``pre_contact``,
    ``post_contact``, ``smoothing``), loads the source files listed under
    ``cfg.zone_c.lm_scoring.lm_sources[era]``, builds one
    :class:`NGramLM` per (era, order) pair, and serialises each to the
    path in ``cfg.zone_c.lm_scoring.lm_files[era]``.

    Source specs ending with ``'/'`` are treated as directory globs
    (all ``*.txt`` files inside).  Files containing ``cognate_neighbour``
    in the name are loaded at ``cfg.zone_c.lm_scoring.cognate_neighbour_weight``
    via Bernoulli-sampling (seed 42 for reproducibility).

    Parameters
    ----------
    cfg : DictConfig
        Root Hydra config.
    project_root : Path
        Absolute path to the repository root
        (``hydra.utils.get_original_cwd()``).
    """
    import random as _random

    lm_cfg = cfg.zone_c.lm_scoring
    tokenization_level: str = str(cfg.data.lm_tokenization_level)
    include_reconstructions: bool = bool(cfg.data.include_reconstructions)
    texts_root: Path = project_root / cfg.paths.polynesian_texts_dir
    orders: list[int] = [int(o) for o in lm_cfg.ngram_orders]
    max_order = max(orders)

    try:
        cognate_weight: float = float(lm_cfg.cognate_neighbour_weight)
    except Exception:
        cognate_weight = 0.5

    rng = _random.Random(42)

    for era in lm_cfg.lms:
        source_specs = list(lm_cfg.lm_sources[era])
        primary_path = project_root / str(lm_cfg.lm_files[era])

        # Resolve each spec to (Path, weight) pairs.
        source_files: list[tuple[Path, float]] = []
        for spec in source_specs:
            spec_str = str(spec)
            weight = cognate_weight if "cognate_neighbour" in spec_str else 1.0
            if spec_str.endswith("/"):
                dir_path = texts_root / spec_str.rstrip("/")
                if dir_path.is_dir():
                    for p in sorted(dir_path.glob("*.txt")):
                        source_files.append((p, weight))
                else:
                    logger.warning(
                        "LM '%s': source directory not found: %s", era, dir_path
                    )
            else:
                p = texts_root / spec_str
                if p.exists():
                    source_files.append((p, weight))
                else:
                    logger.warning(
                        "LM '%s': source file not found: %s — skipping.", era, p
                    )

        if not source_files:
            logger.warning(
                "No source files found for LM '%s'; skipping build.", era
            )
            continue

        primary_path.parent.mkdir(parents=True, exist_ok=True)
        era_language = _ERA_LANGUAGE.get(era, era)

        for order in orders:
            model = NGramLM(order=order, language=era)
            n_lines = 0
            n_tokens_rejected = 0

            for file_path, file_weight in source_files:
                # Auto-detect pre-built frequency tables by peeking at the
                # first non-comment, non-empty line.  Format: "<count>\t<w1>,..."
                # If detected, ingest directly into _counts via the dedicated
                # method rather than training from raw text lines.
                is_freq_table = False
                try:
                    for raw_peek in file_path.read_text(encoding="utf-8").splitlines():
                        peek = raw_peek.strip()
                        if peek and not peek.startswith("#"):
                            is_freq_table = bool(_FREQ_TABLE_RE.match(peek))
                            break
                except OSError:
                    pass

                if is_freq_table:
                    loaded = model.ingest_ngram_table(file_path)
                    n_lines += loaded
                    continue

                raw_lines = file_path.read_text(encoding="utf-8").splitlines()
                for raw in raw_lines:
                    line = raw.strip()
                    if not line or line.startswith("#"):
                        continue
                    if _BRACKET_RE.search(line):
                        if not include_reconstructions:
                            continue
                        line = _BRACKET_RE.sub(r"\1", line).strip()
                    if file_weight < 1.0 and rng.random() > file_weight:
                        continue
                    raw_tokens = tokenize_text(line, tokenization_level)
                    if tokenization_level == "syllable":
                        tokens = clean_syllables(raw_tokens, era_language)
                        n_tokens_rejected += len(raw_tokens) - len(tokens)
                    else:
                        tokens = raw_tokens
                    if tokens:
                        model.update(tokens)
                        n_lines += 1

            model.finalise()
            logger.info(
                "Built NGramLM[%s, order=%d] from %d lines (%d syllable tokens rejected).",
                era, order, n_lines, n_tokens_rejected,
            )

            if order == max_order:
                out_path = primary_path
            else:
                out_path = primary_path.parent / (
                    primary_path.stem + f"_order{order}" + primary_path.suffix
                )
            model.save(out_path)

    # -----------------------------------------------------------------------
    # Attach Hawaiian smoothing LM as backoff for unseen 4/5-gram contexts
    # in the primary (pre_contact / post_contact) models.
    # -----------------------------------------------------------------------
    smoothing_era = "smoothing"
    if smoothing_era in lm_cfg.lms and smoothing_era in lm_cfg.lm_files:
        smoothing_primary_path = project_root / str(lm_cfg.lm_files[smoothing_era])
        backoff_orders = [o for o in orders if o >= 4]
        primary_eras = [e for e in lm_cfg.lms if e != smoothing_era]
        if backoff_orders and primary_eras:
            for era in primary_eras:
                for order in backoff_orders:
                    primary_path = project_root / str(lm_cfg.lm_files[era])
                    if order == max_order:
                        model_path = primary_path
                    else:
                        model_path = primary_path.parent / (
                            primary_path.stem + f"_order{order}" + primary_path.suffix
                        )
                    # Determine backoff LM path (use order-3 smoothing if available).
                    bo_order = min(3, max(orders))
                    if bo_order == max_order:
                        bo_path = smoothing_primary_path
                    else:
                        bo_path = smoothing_primary_path.parent / (
                            smoothing_primary_path.stem
                            + f"_order{bo_order}"
                            + smoothing_primary_path.suffix
                        )
                    if not model_path.exists() or not bo_path.exists():
                        logger.debug(
                            "Backoff attachment skipped: model=%s bo=%s (one or both missing).",
                            model_path, bo_path,
                        )
                        continue
                    try:
                        primary_model = NGramLM.load(model_path)
                        bo_model = NGramLM.load(bo_path)
                        primary_model.set_backoff_lm(
                            bo_model, lm_path=bo_path, from_order=4, alpha=0.15
                        )
                        primary_model.save(model_path)  # re-save with backoff_lm_path
                        logger.info(
                            "Attached Hawaiian backoff (order=%d) to NGramLM[%s, order=%d].",
                            bo_order, era, order,
                        )
                    except Exception as exc:
                        logger.warning(
                            "Could not attach backoff LM to NGramLM[%s, order=%d]: %s",
                            era, order, exc,
                        )
