"""
hackingrongo.zone_b.morpheme_segmentation
==========================================

Zellig Harris (1955) *successor entropy* boundary detection for
rongorongo sign sequences.

Theory
------
Harris observed that in a natural language the entropy of the next
symbol (the *successor entropy*) is relatively low inside morphemes and
spikes at morpheme boundaries.  For rongorongo, we treat each glyph code
as a "phoneme-level" unit and apply the same principle: high successor
entropy at position *n* → likely boundary between position *n* and *n+1*.

Given a corpus of sequences *S = [s₁, s₂, …, sₙ]* where each sᵢ is a
list of sign codes, the successor entropy of sign *x* is::

    H(successor | x) = − Σ_{y} P(y | x) · log₂ P(y | x)

where *P(y | x)* is estimated from bigram counts.

Positions with H(successor) > threshold are flagged as boundaries.  The
threshold can be:

* **auto** (``threshold=None``): mean + 1 × std of the distribution.
* **manual** (``threshold=float``): explicit cut-off in bits.

Usage
-----
    # Library:
    from hackingrongo.zone_b.morpheme_segmentation import (
        successor_entropy, morpheme_boundaries, segment_sequences,
    )

    # CLI:
    python scripts/segment_morphemes.py \\
        --corpus-dir data/corpus \\
        --output     outputs/morpheme_segments.json

    python scripts/segment_morphemes.py \\
        --corpus-dir data/corpus \\
        --threshold  1.5 \\
        --output     outputs/morpheme_segments.json \\
        --json
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Sequence

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core statistics
# ---------------------------------------------------------------------------

def successor_entropy(sequences: list[list[str]]) -> dict[str, float]:
    """Compute Harris successor entropy for each sign.

    For every sign *x* observed in the corpus, estimate::

        H(successor | x) = − Σ_{y} P(y | x) · log₂ P(y | x)

    where the sum is over all signs *y* observed to follow *x* in at
    least one sequence.  Sequence-final tokens have no successor and do
    not contribute to the bigram counts.

    Parameters
    ----------
    sequences : list[list[str]]
        Corpus as a list of sign-code sequences.  Each inner list is one
        inscription or passage.

    Returns
    -------
    dict[str, float]
        Mapping sign → H(successor | sign) in bits.  Signs that appear
        only at the end of sequences are assigned entropy 0.0.
    """
    # Bigram counts: bigrams[x][y] = count(x followed by y)
    bigrams: dict[str, Counter[str]] = defaultdict(Counter)
    for seq in sequences:
        for i in range(len(seq) - 1):
            bigrams[seq[i]][seq[i + 1]] += 1

    h: dict[str, float] = {}
    for sign, successors in bigrams.items():
        total = sum(successors.values())
        if total == 0:
            h[sign] = 0.0
        else:
            h[sign] = -sum(
                (c / total) * math.log2(c / total)
                for c in successors.values()
            )
    return h


def morpheme_boundaries(
    sequences: list[list[str]],
    threshold: float | None = None,
    successor_h: dict[str, float] | None = None,
) -> list[list[int]]:
    """Return boundary positions for each sequence using Harris criterion.

    A position *i* in sequence *seq* is a boundary if
    ``successor_h[seq[i]] > threshold``.  This means the boundary is
    *between* position *i* and position *i+1*.

    Parameters
    ----------
    sequences : list[list[str]]
        Corpus sequences.
    threshold : float | None
        Entropy threshold in bits.  If ``None``, the threshold is set to
        ``mean(H) + 1 × std(H)`` (one standard-deviation rule).
    successor_h : dict[str, float] | None
        Pre-computed successor entropy.  Computed automatically from
        *sequences* if not supplied.

    Returns
    -------
    list[list[int]]
        For each sequence, a (possibly empty) list of boundary positions
        *i* such that the boundary falls between *seq[i]* and *seq[i+1]*.
    """
    if successor_h is None:
        successor_h = successor_entropy(sequences)

    all_h = list(successor_h.values())
    if not all_h:
        return [[] for _ in sequences]

    if threshold is None:
        mu = sum(all_h) / len(all_h)
        variance = sum((v - mu) ** 2 for v in all_h) / len(all_h)
        sigma = variance ** 0.5
        threshold = mu + sigma
        log.info(
            "Auto threshold: mean=%.3f bits  std=%.3f bits  threshold=%.3f bits",
            mu, sigma, threshold,
        )

    boundaries: list[list[int]] = []
    for seq in sequences:
        seq_bounds: list[int] = [
            i
            for i in range(len(seq) - 1)
            if successor_h.get(seq[i], 0.0) > threshold
        ]
        boundaries.append(seq_bounds)
    return boundaries


def segment_sequences(
    sequences: list[list[str]],
    threshold: float | None = None,
    successor_h: dict[str, float] | None = None,
) -> list[list[list[str]]]:
    """Segment each sequence into morpheme-like chunks.

    Uses :func:`morpheme_boundaries` to find boundary positions, then
    splits the sequence at those positions.

    Parameters
    ----------
    sequences : list[list[str]]
        Corpus sequences.
    threshold : float | None
        Passed to :func:`morpheme_boundaries`.
    successor_h : dict[str, float] | None
        Pre-computed successor entropy.

    Returns
    -------
    list[list[list[str]]]
        For each sequence, a list of morpheme chunks (each chunk is a
        list of sign codes).

    Example
    -------
    ::

        seqs = [["001", "002", "003", "004", "005"]]
        boundaries = [[1, 3]]  # → boundaries after position 1 and 3
        # returns [[["001","002"], ["003","004"], ["005"]]]
    """
    bounds_list = morpheme_boundaries(sequences, threshold, successor_h)
    result: list[list[list[str]]] = []
    for seq, bounds in zip(sequences, bounds_list):
        if not bounds:
            result.append([list(seq)])
            continue
        chunks: list[list[str]] = []
        prev = 0
        for b in bounds:
            chunks.append(seq[prev : b + 1])
            prev = b + 1
        if prev < len(seq):
            chunks.append(seq[prev:])
        result.append(chunks)
    return result


# ---------------------------------------------------------------------------
# Convenience: load corpus tokens from directory
# ---------------------------------------------------------------------------

def _load_sequences(corpus_dir: Path) -> list[list[str]]:
    """Load sign sequences from JSON corpus files under *corpus_dir*."""
    sequences: list[list[str]] = []
    for path in sorted(corpus_dir.glob("**/*.json")):
        try:
            data = json.loads(path.read_text())
        except Exception:
            continue
        for key in ("tokens", "signs", "sequence", "text"):
            val = data.get(key)
            if val is None:
                continue
            if isinstance(val, list) and val:
                if isinstance(val[0], str):
                    sequences.append(val)
                elif isinstance(val[0], list):
                    for row in val:
                        if row and isinstance(row[0], str):
                            sequences.append(row)
            break
    log.info("Loaded %d sequences from %s", len(sequences), corpus_dir)
    return sequences


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Zellig Harris successor-entropy morpheme segmentation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--corpus-dir", type=Path, required=True,
        metavar="DIR", help="Directory of corpus JSON files.",
    )
    p.add_argument(
        "--output", type=Path, default=None,
        metavar="FILE",
        help="Output JSON path (default: outputs/morpheme_segments.json).",
    )
    p.add_argument(
        "--threshold", type=float, default=None,
        metavar="BITS",
        help="Entropy threshold in bits.  Defaults to mean + 1 SD (auto).",
    )
    p.add_argument(
        "--json", action="store_true",
        help="Print full result as JSON to stdout in addition to writing file.",
    )
    p.add_argument(
        "--seed", type=int, default=20260606, metavar="INT",
        help="Global RNG seed for reproducibility (default: 20260606).",
    )
    return p


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
    args = _build_parser().parse_args()
    from hackingrongo.repro import set_global_seed
    set_global_seed(args.seed)

    sequences = _load_sequences(args.corpus_dir)
    if not sequences:
        log.error("No sequences found under %s.", args.corpus_dir)
        sys.exit(1)

    h = successor_entropy(sequences)
    segmented = segment_sequences(sequences, threshold=args.threshold, successor_h=h)

    # Summary statistics.
    all_h = list(h.values())
    mu    = sum(all_h) / len(all_h) if all_h else 0.0
    var   = sum((v - mu) ** 2 for v in all_h) / len(all_h) if all_h else 0.0
    n_bounds = sum(len(b) for segs in segmented for b in [segs])
    # actually count morphemes
    n_morphemes = sum(len(segs) for segs in segmented)

    result = {
        "n_sequences":    len(sequences),
        "n_sign_types":   len(h),
        "entropy_stats": {
            "mean_bits":   round(mu, 4),
            "std_bits":    round(var ** 0.5, 4),
            "threshold_used": args.threshold if args.threshold is not None
                              else round(mu + var ** 0.5, 4),
        },
        "n_morpheme_chunks": n_morphemes,
        "mean_morpheme_length": round(
            sum(len(tok) for segs in segmented for tok in segs) / max(n_morphemes, 1), 2
        ),
        "top_boundary_signs": sorted(h, key=lambda s: -h[s])[:20],
        "successor_entropy":  {k: round(v, 4) for k, v in sorted(h.items())},
        "segmented_sequences": [
            {"sequence_index": i, "morphemes": segs}
            for i, segs in enumerate(segmented)
        ],
    }

    output = args.output
    if output is None:
        try:
            from omegaconf import OmegaConf
            cfg = OmegaConf.load(PROJECT_ROOT / "conf" / "config.yaml")
            outputs_dir = PROJECT_ROOT / cfg.paths.outputs_dir
        except Exception:
            outputs_dir = PROJECT_ROOT / "outputs"
        output = outputs_dir / "morpheme_segments.json"

    from hackingrongo.provenance import stamp
    stamp(result, seed=args.seed)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2))
    log.info("Morpheme segmentation written to %s", output)
    log.info(
        "  %d sequences → %d chunks, mean length %.2f signs",
        len(sequences), n_morphemes, result["mean_morpheme_length"],
    )
    log.info("  Top 5 boundary signs: %s", result["top_boundary_signs"][:5])

    if args.json:
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
