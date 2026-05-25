"""
Train the n-gram sequence model on the resolved Horley-token corpus.

Trains at multiple n-gram orders, reports per-order cross-validation
perplexity (leave-one-tablet-out), and saves the best model.

Usage
-----
    conda run -n base python scripts/train_sequence_model.py
    conda run -n base python scripts/train_sequence_model.py --order 3 --alpha 0.01
    conda run -n base python scripts/train_sequence_model.py --output outputs/sequence_model.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from omegaconf import OmegaConf  # noqa: E402

from hackingrongo.zone_b.sequence_model import (  # noqa: E402
    NgramModel,
    load_sequences,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)


def cross_validate(
    sequences: list[list[str]],
    order: int,
    alpha: float,
) -> float:
    """Leave-one-tablet-out perplexity estimate."""
    if len(sequences) < 2:
        return float("nan")
    ppl_values: list[float] = []
    for i in range(len(sequences)):
        train = sequences[:i] + sequences[i + 1 :]
        held_out = [sequences[i]]
        model = NgramModel(order=order, alpha=alpha)
        model.train(train)
        ppl = model.perplexity(held_out)
        ppl_values.append(ppl)
    return sum(ppl_values) / len(ppl_values)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train n-gram sequence model on resolved Horley tokens.",
    )
    parser.add_argument(
        "--order",
        type=int,
        default=3,
        help="N-gram order (default: 3).",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.01,
        help="Add-α smoothing coefficient (default: 0.01).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Path to write the trained model JSON (default: outputs/zone_b/sequence_model.json).",
    )
    parser.add_argument(
        "--cv",
        action="store_true",
        help="Run leave-one-tablet-out cross-validation before saving.",
    )
    parser.add_argument(
        "--include-uncertain",
        action="store_true",
        help="Include uncertain tokens in training sequences.",
    )
    args = parser.parse_args()

    cfg = OmegaConf.load(PROJECT_ROOT / "conf" / "config.yaml")
    corpus_dir = PROJECT_ROOT / cfg.paths.corpus_dir
    out_path = args.output or PROJECT_ROOT / cfg.paths.outputs_dir / "zone_b" / "sequence_model.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    log.info("Loading corpus from %s …", corpus_dir)
    sequences = load_sequences(corpus_dir, include_uncertain=args.include_uncertain)
    total_tokens = sum(len(s) for s in sequences)
    log.info(
        "%d sequences loaded, %d total tokens across %d tablets",
        len(sequences), total_tokens, len(sequences),
    )

    # Per-cluster breakdown
    for path in sorted(corpus_dir.glob("[A-Z].json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        glyphs = data["glyphs"]
        resolved = sum(1 for g in glyphs if g.get("horley_code"))
        log.debug("  %s  cluster=%-14s  resolved=%d", data["tablet_id"], data.get("cluster", "?"), resolved)

    # Optional cross-validation
    if args.cv:
        log.info("")
        log.info("Leave-one-tablet-out cross-validation …")
        for order in range(1, args.order + 1):
            ppl = cross_validate(sequences, order=order, alpha=args.alpha)
            log.info("  order=%d  α=%.3f  mean LOO perplexity = %.2f", order, args.alpha, ppl)
        log.info("")

    # Train final model on full corpus
    log.info("Training NgramModel(order=%d, α=%.3f) on full corpus …", args.order, args.alpha)
    model = NgramModel(order=args.order, alpha=args.alpha)
    model.train(sequences)

    full_ppl = model.perplexity(sequences)
    log.info("  Train perplexity = %.2f", full_ppl)

    # Show top-10 bigrams and most common signs as a sanity check
    log.info("")
    log.info("Top 10 most probable next tokens after BOS (sequence-initial signs):")
    bos_context = ["<BOS>"] * (args.order - 1)
    for tok, lp in model.top_k_next(bos_context, k=10):
        if tok not in ("<BOS>", "<EOS>", "<UNK>"):
            log.info("  %-12s  log₂p = %+.3f  (p ≈ %.4f)", tok, lp, 2.0 ** lp)

    # Most frequent bigrams (top 10)
    if args.order >= 2:
        bigram_table = model.counts[2]
        all_bigrams: list[tuple[tuple, str, int]] = []
        for ctx, ctr in bigram_table.items():
            for tok, cnt in ctr.items():
                if tok not in ("<BOS>", "<EOS>", "<UNK>") and ctx:
                    all_bigrams.append((ctx, tok, cnt))
        all_bigrams.sort(key=lambda x: -x[2])
        log.info("")
        log.info("Top 10 most frequent bigrams:")
        for ctx, tok, cnt in all_bigrams[:10]:
            log.info("  %s → %s  (count=%d)", ctx[0], tok, cnt)

    model.save(out_path)
    log.info("Model saved to %s", out_path)


if __name__ == "__main__":
    main()
