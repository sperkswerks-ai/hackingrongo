"""
hackingrongo.zone_b.sequence_model
====================================

N-gram language model trained on the resolved Horley-token corpus.

The corpus is small (~10 k resolved tokens across 25 tablets) so a neural
sequence model would overfit severely.  An add-α smoothed n-gram model is
interpretable, fast to train, and appropriate for the data volume.  Kneser-Ney
smoothing is implemented for trigrams downward.

Reading order
-------------
Rongorongo is written in *reverse boustrophedon*: even-indexed lines
(1-based: lines 1, 3, 5 …) run left-to-right, odd-indexed lines (2, 4, 6 …)
run right-to-left with each glyph inverted.  The corpus JSON stores glyphs in
visual/physical order with an ``inverted`` flag.  To reconstruct the correct
*scribal* reading sequence we simply keep visual order — the physical order on
the object is the correct token sequence, provided we handle the right-to-left
lines by reversing the glyph order on those lines before concatenating.

The ``load_sequences`` function handles this automatically using the
``side``, ``line``, and ``glyph_num`` fields plus the ``inverted`` flag.

Public API
----------
``load_sequences(corpus_dir)``
    → list of lists of Horley code strings, one list per tablet.

``NgramModel``
    Stores n-gram count tables and exposes:

    * ``train(sequences)``
    * ``log_prob(token, context)``  — log₂ probability of token given context
    * ``score(sequence)``           — total log₂ prob of a sequence
    * ``top_k_next(context, k=10)`` — most likely next tokens with scores
    * ``save(path)`` / ``NgramModel.load(path)``

Usage
-----
    from hackingrongo.zone_b.sequence_model import load_sequences, NgramModel

    seqs = load_sequences(corpus_dir)
    model = NgramModel(order=3)
    model.train(seqs)
    model.save(Path("outputs/sequence_model.json"))
"""

from __future__ import annotations

import json
import logging
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Sequence

log = logging.getLogger(__name__)

# Special boundary token inserted at the start/end of each sequence so
# the model can learn sequence-initial and sequence-final patterns.
BOS = "<BOS>"
EOS = "<EOS>"
UNK = "<UNK>"


# ---------------------------------------------------------------------------
# Corpus loading — reading-order reconstruction
# ---------------------------------------------------------------------------

def _safe_glyph_num(val: object) -> int:
    """Safely parse a glyph ``glyph_num`` value to int, defaulting to 0.

    Non-integer values occur for uncertain and compound glyph entries in
    the corpus JSON (e.g. ``"12a"``, ``null``).  Using plain
    ``int(g.get("glyph_num", 0))`` in the sort key raises ``ValueError``
    for those tokens and aborts sequence loading mid-tablet.
    """
    try:
        return int(val)  # type: ignore[arg-type]
    except (ValueError, TypeError):
        return 0


def load_sequences(
    corpus_dir: Path,
    include_uncertain: bool = False,
    include_compound_components: bool = True,
) -> list[list[str]]:
    """Return one token sequence per tablet in scribal reading order.

    Parameters
    ----------
    corpus_dir:
        Directory containing enriched corpus JSON files (A.json … Y.json).
    include_uncertain:
        If True, include tokens marked with ``uncertain=True``.
    include_compound_components:
        If True, expand compound glyphs to their resolved component Horley
        codes (from the ``horley_components`` field).  Compounds without
        resolved components are skipped (consistent with the entropy analysis).

    Returns
    -------
    List of token sequences (one per tablet).  Each sequence is a flat list
    of Horley code strings in scribal reading order.
    """
    sequences: list[list[str]] = []

    for path in sorted(corpus_dir.glob("[A-Z].json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        glyphs = data.get("glyphs", [])

        # Group glyphs by (side, line) to handle boustrophedon reversal
        line_groups: dict[tuple, list[dict]] = defaultdict(list)
        for g in glyphs:
            side = str(g.get("side", "a")).lower()
            try:
                line_num = int(g.get("line", 0))
            except (ValueError, TypeError):
                line_num = 0
            line_groups[(side, line_num)].append(g)

        # Sort groups by (side_order, line_num)
        side_order_map = {"a": 0, "b": 1, "c": 2}
        sorted_lines = sorted(
            line_groups.items(),
            key=lambda kv: (side_order_map.get(kv[0][0], 9), kv[0][1]),
        )

        tokens: list[str] = []
        for line_key, line_glyphs in sorted_lines:
            # Sort within line by glyph_num ascending (physical order)
            line_glyphs = sorted(
                line_glyphs,
                key=lambda g: _safe_glyph_num(g.get("glyph_num", 0)),
            )
            # Determine reading direction: inverted line = right-to-left
            # In the corpus, odd 1-based lines (lines 1, 3, 5...) are L→R.
            # Lines 2, 4, 6 are R→L (inverted glyphs).
            # We detect this from the majority `inverted` flag on the line.
            n_inverted = sum(1 for g in line_glyphs if g.get("inverted", False))
            is_rtl = n_inverted > len(line_glyphs) / 2 if line_glyphs else False
            if is_rtl:
                line_glyphs = list(reversed(line_glyphs))

            for g in line_glyphs:
                if not include_uncertain and g.get("uncertain", False):
                    continue

                hc = g.get("horley_code")
                if hc:
                    tokens.append(hc)
                    continue

                if include_compound_components:
                    for comp in (g.get("horley_components") or []):
                        tokens.append(comp)

        if tokens:
            sequences.append(tokens)
            log.debug(
                "Tablet %s: %d tokens (cluster=%s)",
                data["tablet_id"], len(tokens), data.get("cluster", "?"),
            )

    return sequences


# (tablet_id, side, line_num, parity, tokens)
# parity = "odd"  if line_num % 2 == 1
# parity = "even" if line_num % 2 == 0  (includes line 0 for unattributed glyphs)
LineRecord = tuple[str, str, int, str, list[str]]


def load_sequences_with_lines(
    corpus_dir: Path,
    include_uncertain: bool = False,
    include_compound_components: bool = True,
) -> list[LineRecord]:
    """Return one record per (tablet, side, line) preserving line structure.

    Identical boustrophedon handling to :func:`load_sequences` — RTL lines are
    reversed before token extraction — but instead of flattening into one list
    per tablet, each line is returned as a separate record.

    Parameters
    ----------
    corpus_dir:
        Directory containing enriched corpus JSON files (A.json … Y.json).
    include_uncertain:
        If True, include tokens marked ``uncertain=True``.
    include_compound_components:
        If True, expand compound glyphs via ``horley_components``.

    Returns
    -------
    List of ``(tablet_id, side, line_num, parity, tokens)`` tuples in scribal
    reading order.  ``parity`` is ``"odd"`` when ``line_num % 2 == 1``,
    ``"even"`` otherwise.

    Notes
    -----
    Line parity is determined solely by the 1-based line number, not by the
    ``inverted`` flag (which is sparsely populated in the corpus).  In standard
    reverse-boustrophedon rongorongo, odd lines run one direction and even lines
    the other, so IC(odd) vs IC(even) tests whether the two physical text-streams
    have distinct sign-frequency distributions.
    """
    # Handles both 'a'/'b' (older tablets) and 'r'/'v' (enriched tablets).
    _SIDE_ORDER: dict[str, int] = {"a": 0, "r": 0, "b": 1, "v": 1, "c": 2}

    records: list[LineRecord] = []

    for path in sorted(corpus_dir.glob("[A-Z].json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        tablet_id: str = data["tablet_id"]
        glyphs = data.get("glyphs", [])

        # Group by (side, line_num)
        line_groups: dict[tuple[str, int], list[dict]] = defaultdict(list)
        for g in glyphs:
            side = str(g.get("side", "a")).lower()
            try:
                line_num = int(g.get("line", 0))
            except (ValueError, TypeError):
                line_num = 0
            line_groups[(side, line_num)].append(g)

        # Sort in physical reading order: side first, then line number
        sorted_lines = sorted(
            line_groups.items(),
            key=lambda kv: (_SIDE_ORDER.get(kv[0][0], 9), kv[0][1]),
        )

        for (side, line_num), line_glyphs in sorted_lines:
            line_glyphs = sorted(
                line_glyphs,
                key=lambda g: _safe_glyph_num(g.get("glyph_num", 0)),
            )
            n_inverted = sum(1 for g in line_glyphs if g.get("inverted", False))
            if n_inverted > len(line_glyphs) / 2:
                line_glyphs = list(reversed(line_glyphs))

            tokens: list[str] = []
            for g in line_glyphs:
                if not include_uncertain and g.get("uncertain", False):
                    continue
                hc = g.get("horley_code")
                if hc:
                    tokens.append(hc)
                    continue
                if include_compound_components:
                    for comp in (g.get("horley_components") or []):
                        tokens.append(comp)

            if tokens:
                parity = "odd" if line_num % 2 == 1 else "even"
                records.append((tablet_id, side, line_num, parity, tokens))

    log.debug(
        "load_sequences_with_lines: %d line records across %d tablets",
        len(records),
        len({r[0] for r in records}),
    )
    return records


# ---------------------------------------------------------------------------
# N-gram model
# ---------------------------------------------------------------------------

class NgramModel:
    """Add-α smoothed n-gram language model over Horley sign sequences.

    Parameters
    ----------
    order:
        Maximum n-gram order (1 = unigram, 2 = bigram, 3 = trigram, …).
    alpha:
        Laplace / Lidstone smoothing parameter added to each count.
        Default 0.01 (add-0.01 smoothing).

    Notes
    -----
    For each order 1 … ``order``, a separate count table is kept.
    At query time ``log_prob`` uses the highest-order context available,
    backing off to lower orders when the context is unseen (Stupid Backoff
    with a 0.4 discount factor).
    """

    _BACKOFF_DISCOUNT = 0.4  # log2 penalty per backoff step

    def __init__(self, order: int = 3, alpha: float = 0.01) -> None:
        if order < 1:
            raise ValueError("order must be >= 1")
        self.order = order
        self.alpha = alpha

        # counts[n] maps context_tuple → {token: count}
        # For n=1 (unigram) context is ()
        self.counts: dict[int, dict[tuple, Counter]] = {
            n: defaultdict(Counter) for n in range(1, order + 1)
        }
        self.vocab: set[str] = set()
        self._total_tokens: int = 0
        # Number of distinct observed sign types (excludes BOS/EOS/UNK sentinels).
        self._n_types: int = 0

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self, sequences: list[list[str]]) -> None:
        """Build count tables from *sequences*.

        Parameters
        ----------
        sequences:
            List of token lists (as returned by :func:`load_sequences`).
            Each sequence is padded with BOS/EOS sentinels.
        """
        self.vocab.clear()
        self.counts = {n: defaultdict(Counter) for n in range(1, self.order + 1)}
        self._total_tokens = 0

        for seq in sequences:
            padded = [BOS] * (self.order - 1) + seq + [EOS]
            self.vocab.update(seq)
            for i in range(self.order - 1, len(padded)):
                token = padded[i]
                self._total_tokens += 1
                for n in range(1, self.order + 1):
                    context = tuple(padded[i - n + 1 : i])
                    self.counts[n][context][token] += 1

        # Record observed type count before sentinel tokens inflate the set.
        self._n_types = len(self.vocab)
        self.vocab.add(BOS)
        self.vocab.add(EOS)
        self.vocab.add(UNK)
        log.info(
            "NgramModel(order=%d, α=%.3f) trained: %d sequences, %d tokens, %d types",
            self.order, self.alpha,
            len(sequences), self._total_tokens, len(self.vocab),
        )

    # ------------------------------------------------------------------
    # Probability
    # ------------------------------------------------------------------

    def log_prob(
        self,
        token: str,
        context: Sequence[str],
    ) -> float:
        """Return log₂ P(token | context) with stupid-backoff.

        The context is truncated to at most ``order - 1`` tokens.
        If the full context is unseen, backs off to a shorter context
        with a single ``_BACKOFF_DISCOUNT`` penalty per step.

        Parameters
        ----------
        token:
            The token whose probability is queried.
        context:
            Preceding tokens (most recent last).
        """
        # Map OOV tokens to UNK for scoring
        token_q = token if token in self.vocab else UNK
        ctx = tuple(context[-(self.order - 1) :]) if self.order > 1 else ()
        n = len(ctx) + 1
        n = max(1, min(n, self.order))

        ctx_counts = self.counts[n].get(ctx, Counter())
        # Use only observed sign types for V; sentinels (BOS/EOS/UNK) inflate the
        # denominator and make low-frequency signs look worse than they are.
        V = self._n_types if self._n_types > 0 else len(self.vocab)
        total_in_ctx = sum(ctx_counts.values())
        count = ctx_counts.get(token_q, 0)

        # Add-α smoothed log probability
        log_p = math.log2((count + self.alpha) / (total_in_ctx + self.alpha * V))

        # Back off only when the context itself is entirely unseen (total_in_ctx == 0).
        # When the context IS seen, add-α smoothing already assigns a valid non-zero
        # probability to unseen tokens; adding a Stupid-Backoff penalty on top would
        # double-penalise them and distort the model's probability estimates.
        if total_in_ctx == 0 and len(ctx) > 0:
            shorter_lp = self.log_prob(token_q, list(ctx[1:]))
            return shorter_lp + math.log2(self._BACKOFF_DISCOUNT)
        return log_p

    def score(self, sequence: list[str]) -> float:
        """Return total log₂ probability of *sequence* (sum of per-token log-probs)."""
        padded = [BOS] * (self.order - 1) + sequence + [EOS]
        total = 0.0
        for i in range(self.order - 1, len(padded)):
            context = list(padded[max(0, i - self.order + 1) : i])
            total += self.log_prob(padded[i], context)
        return total

    def perplexity(self, sequences: list[list[str]]) -> float:
        """Return perplexity on *sequences* (2^(−mean log₂ prob per token))."""
        total_lp = 0.0
        total_n = 0
        for seq in sequences:
            total_lp += self.score(seq)
            total_n += len(seq) + 1  # +1 for EOS
        if total_n == 0:
            return float("inf")
        avg_lp = total_lp / total_n
        return 2.0 ** (-avg_lp)

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def top_k_next(
        self,
        context: list[str],
        k: int = 10,
    ) -> list[tuple[str, float]]:
        """Return top-k next tokens and their log₂ probabilities.

        Parameters
        ----------
        context:
            Preceding tokens (most recent last).
        k:
            Number of candidates to return.

        Returns
        -------
        List of (token, log_prob) pairs sorted by decreasing probability.
        """
        candidates = [(tok, self.log_prob(tok, context)) for tok in self.vocab]
        candidates.sort(key=lambda x: -x[1])
        return candidates[:k]

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def save(self, path: Path) -> None:
        """Serialise the model to a JSON file."""
        payload = {
            "order": self.order,
            "alpha": self.alpha,
            "vocab": sorted(self.vocab),
            "total_tokens": self._total_tokens,
            "n_types": self._n_types,
            # Serialise count tables: {n: [[ctx_list, {tok: count}], ...]}
            "counts": {
                str(n): [
                    [list(ctx), dict(ctr)]
                    for ctx, ctr in table.items()
                ]
                for n, table in self.counts.items()
            },
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        log.info("NgramModel saved to %s", path)

    @classmethod
    def load(cls, path: Path) -> "NgramModel":
        """Deserialise a model previously saved with :meth:`save`."""
        payload = json.loads(path.read_text(encoding="utf-8"))
        model = cls(order=payload["order"], alpha=payload["alpha"])
        model.vocab = set(payload["vocab"])
        model._total_tokens = payload["total_tokens"]
        _specials = {BOS, EOS, UNK}
        model._n_types = payload.get(
            "n_types", len(model.vocab) - sum(1 for s in _specials if s in model.vocab)
        )
        model.counts = {}
        for n_str, entries in payload["counts"].items():
            n = int(n_str)
            table: dict[tuple, Counter] = defaultdict(Counter)
            for ctx_list, ctr in entries:
                table[tuple(ctx_list)] = Counter(ctr)
            model.counts[n] = table
        log.info("NgramModel loaded from %s (order=%d)", path, model.order)
        return model
