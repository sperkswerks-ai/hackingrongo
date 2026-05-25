"""
reading_order_tests.py — four entropy tests for rongorongo reading direction.

Tests
-----
1. Conditional entropy asymmetry
   H(Sₙ | Sₙ₋₁) forward vs reverse across the full corpus.
   If H_forward < H_reverse, consecutive signs are more predictable in the
   transcribed direction — left-to-right reading confirmed.

2. N-gram model perplexity asymmetry
   Existing NgramModel (trained forward) scored on forward vs reversed sequences.
   Lower perplexity on forward sequences means the model learned real structure
   — the transcription direction is the reading direction.

3. Line-boundary entropy
   Within-line bigrams vs cross-line bigrams (last sign of line N → first sign
   of line N+1 after boustrophedon flip). Cross-line entropy should exceed
   within-line entropy if line boundaries are real structural breaks.

4. Recto/verso ordering
   Leave-one-out perplexity under a→b vs b→a side ordering.
   The ordering that produces lower held-out perplexity is the reading order,
   resolving Pozdniakov's unresolved question from 1958.

Usage
-----
    # All four tests (requires trained model + corpus)
    python scripts/reading_order_tests.py \\
        --model outputs/zone_b/sequence_model.json \\
        --corpus data/corpus

    # Just the recto/verso test (no model needed)
    python scripts/reading_order_tests.py --corpus data/corpus --tests 3 4
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from hackingrongo.zone_b.sequence_model import NgramModel  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

_SIDE_AB: dict[str, int] = {"a": 0, "b": 1, "c": 2}
_SIDE_BA: dict[str, int] = {"b": 0, "a": 1, "c": 2}

TabletRow = tuple[str, int, str]  # (side, line_num, token)


# ── Corpus loading ────────────────────────────────────────────────────────────

def _safe_int(v: object) -> int:
    try:
        return int(v)  # type: ignore[arg-type]
    except (ValueError, TypeError):
        return 0


def _glyph_tokens(g: dict) -> list[str]:
    hc = g.get("horley_code")
    if hc:
        return [hc]
    comps = g.get("horley_components") or []
    return list(comps) if comps else []


def load_tablet(path: Path, side_order: dict[str, int]) -> list[TabletRow]:
    data = json.loads(path.read_text(encoding="utf-8"))
    glyphs = data.get("glyphs", [])
    line_groups: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for g in glyphs:
        side = str(g.get("side", "a")).lower()
        line_groups[(side, _safe_int(g.get("line", 0)))].append(g)
    sorted_lines = sorted(
        line_groups.items(),
        key=lambda kv: (side_order.get(kv[0][0], 9), kv[0][1]),
    )
    rows: list[TabletRow] = []
    for (side, line_num), line_glyphs in sorted_lines:
        line_glyphs = sorted(line_glyphs, key=lambda g: _safe_int(g.get("glyph_num", 0)))
        n_inv = sum(1 for g in line_glyphs if g.get("inverted", False))
        if line_glyphs and n_inv > len(line_glyphs) / 2:
            line_glyphs = list(reversed(line_glyphs))
        for g in line_glyphs:
            for tok in _glyph_tokens(g):
                rows.append((side, line_num, tok))
    return rows


def load_corpus(
    corpus_dir: Path,
    side_order: dict[str, int],
) -> dict[str, list[TabletRow]]:
    result: dict[str, list[TabletRow]] = {}
    for path in sorted(corpus_dir.glob("*.json")):
        rows = load_tablet(path, side_order)
        if rows:
            result[path.stem] = rows
    return result


def _seqs(corpus: dict[str, list[TabletRow]]) -> list[list[str]]:
    return [[tok for _, _, tok in rows] for rows in corpus.values()]


# ── Entropy primitives ────────────────────────────────────────────────────────

def _conditional_entropy(bigrams: Counter, unigrams: Counter) -> float:
    """H(Sₙ | Sₙ₋₁) from empirical bigram/unigram counts."""
    total = sum(bigrams.values())
    if total == 0:
        return 0.0
    h = 0.0
    for (s, t), cnt in bigrams.items():
        p_st = cnt / total
        p_t_given_s = cnt / unigrams[s]
        h -= p_st * math.log2(p_t_given_s)
    return h


def _bigram_counts(sequences: list[list[str]]) -> tuple[Counter, Counter]:
    bigrams: Counter = Counter()
    unigrams: Counter = Counter()
    for seq in sequences:
        for i in range(len(seq) - 1):
            bigrams[(seq[i], seq[i + 1])] += 1
            unigrams[seq[i]] += 1
    return bigrams, unigrams


# ── Perplexity helpers ────────────────────────────────────────────────────────

def ngram_perplexity(model: NgramModel, sequences: list[list[str]]) -> float:
    """2^(-mean log₂p per token) using the trained NgramModel."""
    total_lp = 0.0
    total_n = 0
    for seq in sequences:
        if len(seq) < model.order:
            continue
        total_lp += model.score(seq)
        total_n += len(seq)
    return 2.0 ** (-total_lp / total_n) if total_n else float("inf")


def loo_perplexity(sequences: list[list[str]], order: int = 2, alpha: float = 0.5) -> float:
    """Leave-one-out add-alpha perplexity (unbiased; used for Test 4)."""
    vocab: set[str] = set()
    for seq in sequences:
        vocab.update(seq)
    V = len(vocab) + 2  # +BOS +EOS

    def _count(seqs: list[list[str]]) -> tuple[Counter, Counter]:
        counts: Counter = Counter()
        ctx_counts: Counter = Counter()
        for seq in seqs:
            padded = ["<BOS>"] * (order - 1) + seq + ["<EOS>"]
            for i in range(order - 1, len(padded)):
                ng = tuple(padded[i - order + 1 : i + 1])
                counts[ng] += 1
                ctx_counts[ng[:-1]] += 1
        return counts, ctx_counts

    def _score(seq: list[str], counts: Counter, ctx_counts: Counter) -> tuple[float, int]:
        padded = ["<BOS>"] * (order - 1) + seq + ["<EOS>"]
        lp = 0.0
        for i in range(order - 1, len(padded)):
            ng = tuple(padded[i - order + 1 : i + 1])
            ctx = ng[:-1]
            lp += math.log2((counts[ng] + alpha) / (ctx_counts[ctx] + alpha * V))
        return lp, len(padded) - (order - 1)

    total_lp = 0.0
    total_n = 0
    for i in range(len(sequences)):
        train = [s for j, s in enumerate(sequences) if j != i]
        counts, ctx_counts = _count(train)
        lp, n = _score(sequences[i], counts, ctx_counts)
        total_lp += lp
        total_n += n
    return 2.0 ** (-total_lp / total_n) if total_n else float("inf")


# ── The four tests ────────────────────────────────────────────────────────────

def test1_conditional_entropy(sequences: list[list[str]]) -> dict:
    print("\n── Test 1: Conditional entropy asymmetry ────────────────────────────────")
    rev = [list(reversed(s)) for s in sequences]
    bg_f, uni_f = _bigram_counts(sequences)
    bg_r, uni_r = _bigram_counts(rev)
    h_f = _conditional_entropy(bg_f, uni_f)
    h_r = _conditional_entropy(bg_r, uni_r)
    delta = h_r - h_f
    print(f"  H(Sₙ | Sₙ₋₁) forward : {h_f:.4f} bits")
    print(f"  H(Sₙ | Sₙ₋₁) reverse : {h_r:.4f} bits")
    if delta > 0.05:
        direction = "forward"
        verdict = f"✓ forward entropy lower by {delta:.4f} bits — left-to-right reading confirmed"
    elif delta < -0.05:
        direction = "reverse"
        verdict = f"✗ reverse entropy lower by {-delta:.4f} bits — right-to-left suggested"
    else:
        direction = "neutral"
        verdict = f"≈ no directional preference (Δ = {delta:+.4f} bits)"
    print(f"  {verdict}")
    return {"h_forward": h_f, "h_reverse": h_r, "delta": delta,
            "direction": direction, "verdict_text": verdict}


def test2_perplexity(model: NgramModel, sequences: list[list[str]]) -> dict:
    print("\n── Test 2: N-gram model perplexity asymmetry ────────────────────────────")
    rev = [list(reversed(s)) for s in sequences]
    ppl_f = ngram_perplexity(model, sequences)
    ppl_r = ngram_perplexity(model, rev)
    ratio = ppl_r / ppl_f if ppl_f > 0 else float("inf")
    print(f"  Perplexity (forward) : {ppl_f:.2f}")
    print(f"  Perplexity (reverse) : {ppl_r:.2f}")
    if ppl_f < ppl_r:
        direction = "forward"
        verdict = (
            f"✓ model prefers forward sequences ({ratio:.2f}× lower PPL)"
            " — transcription direction confirmed"
        )
    else:
        direction = "reverse"
        verdict = "✗ model prefers reversed sequences — transcription direction may be wrong"
    print(f"  {verdict}")
    return {"ppl_forward": ppl_f, "ppl_reverse": ppl_r, "ratio": ratio,
            "model_order": model.order, "direction": direction, "verdict_text": verdict}


def test3_line_boundary(corpus: dict[str, list[TabletRow]]) -> dict:
    print("\n── Test 3: Line-boundary entropy ────────────────────────────────────────")
    within_bg: Counter = Counter()
    within_uni: Counter = Counter()
    cross_bg: Counter = Counter()
    cross_uni: Counter = Counter()
    for rows in corpus.values():
        for i in range(len(rows) - 1):
            s0, l0, tok0 = rows[i]
            s1, l1, tok1 = rows[i + 1]
            if s0 == s1 and l0 == l1:
                within_bg[(tok0, tok1)] += 1
                within_uni[tok0] += 1
            else:
                cross_bg[(tok0, tok1)] += 1
                cross_uni[tok0] += 1
    h_w = _conditional_entropy(within_bg, within_uni)
    h_c = _conditional_entropy(cross_bg, cross_uni)
    n_w = sum(within_bg.values())
    n_c = sum(cross_bg.values())
    delta = h_c - h_w
    print(f"  Within-line bigrams : {n_w:>6,}   H = {h_w:.4f} bits")
    print(f"  Cross-line bigrams  : {n_c:>6,}   H = {h_c:.4f} bits")
    if delta > 0.1:
        direction = "confirmed"
        verdict = (
            f"✓ cross-line entropy higher by {delta:.4f} bits"
            " — line boundaries are real structural breaks"
        )
    elif delta < -0.1:
        direction = "unexpected"
        verdict = f"✗ within-line entropy unexpectedly higher (Δ = {delta:.4f} bits)"
    else:
        direction = "neutral"
        verdict = f"≈ no significant entropy jump at line boundaries (Δ = {delta:+.4f} bits)"
    print(f"  {verdict}")
    return {"n_within_bigrams": n_w, "n_cross_bigrams": n_c,
            "h_within": h_w, "h_cross": h_c, "delta": delta,
            "direction": direction, "verdict_text": verdict}


def test4_recto_verso(corpus_dir: Path) -> dict:
    print("\n── Test 4: Recto/verso ordering ─────────────────────────────────────────")
    corp_ab = load_corpus(corpus_dir, _SIDE_AB)
    corp_ba = load_corpus(corpus_dir, _SIDE_BA)
    seqs_ab = _seqs(corp_ab)
    seqs_ba = _seqs(corp_ba)

    log.info("LOO perplexity a→b order (bigram) …")
    ppl_ab2 = loo_perplexity(seqs_ab, order=2)
    log.info("LOO perplexity a→b order (trigram) …")
    ppl_ab3 = loo_perplexity(seqs_ab, order=3)
    log.info("LOO perplexity b→a order (bigram) …")
    ppl_ba2 = loo_perplexity(seqs_ba, order=2)
    log.info("LOO perplexity b→a order (trigram) …")
    ppl_ba3 = loo_perplexity(seqs_ba, order=3)

    print(f"  Bigram  LOO-PPL   a→b : {ppl_ab2:.2f}   b→a : {ppl_ba2:.2f}")
    print(f"  Trigram LOO-PPL   a→b : {ppl_ab3:.2f}   b→a : {ppl_ba3:.2f}")
    votes_ab = sum([ppl_ab2 < ppl_ba2, ppl_ab3 < ppl_ba3])
    if votes_ab == 2:
        preferred_order = "ab"
        verdict = "✓ a→b ordering preferred — recto (side a) precedes verso (side b)"
    elif votes_ab == 0:
        preferred_order = "ba"
        verdict = "✓ b→a ordering preferred — verso (side b) precedes recto (side a)"
    else:
        preferred_order = "mixed"
        verdict = "≈ mixed signal — bigram and trigram disagree"
    print(f"  {verdict}")
    return {"ppl_ab_bigram": ppl_ab2, "ppl_ab_trigram": ppl_ab3,
            "ppl_ba_bigram": ppl_ba2, "ppl_ba_trigram": ppl_ba3,
            "votes_ab": votes_ab, "preferred_order": preferred_order,
            "verdict_text": verdict}


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Four entropy tests for rongorongo reading direction.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--model", type=Path, default=None,
        help="Path to trained NgramModel JSON (required for Tests 1 and 2).",
    )
    p.add_argument(
        "--corpus", type=Path, default=None,
        help="Path to corpus directory with tablet JSONs.",
    )
    p.add_argument(
        "--tests", nargs="+", type=int, default=[1, 2, 3, 4],
        choices=[1, 2, 3, 4], metavar="N",
        help="Which tests to run (default: all four).",
    )
    p.add_argument(
        "--output", type=Path, default=None,
        help="Write structured JSON results to this path (for HTML report generation).",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    corpus_dir = args.corpus
    model_path = args.model

    if corpus_dir is None or model_path is None:
        try:
            from omegaconf import OmegaConf
            cfg = OmegaConf.load(PROJECT_ROOT / "conf" / "config.yaml")
            if corpus_dir is None:
                corpus_dir = PROJECT_ROOT / cfg.paths.corpus_dir
            if model_path is None:
                model_path = PROJECT_ROOT / cfg.paths.outputs_dir / "zone_b" / "sequence_model.json"
        except Exception:
            pass

    if corpus_dir is None or not corpus_dir.exists():
        log.error("Corpus directory not found. Pass --corpus <path>.")
        sys.exit(1)

    tests = set(args.tests)

    model: NgramModel | None = None
    if {1, 2} & tests:
        if model_path is None or not model_path.exists():
            log.error("Model not found. Train first or pass --model <path>.")
            sys.exit(1)
        log.info("Loading model from %s …", model_path)
        model = NgramModel.load(model_path)
        log.info("NgramModel  order=%d  vocab=%d", model.order, len(model.vocab))

    log.info("Loading corpus from %s …", corpus_dir)
    corpus = load_corpus(corpus_dir, _SIDE_AB)
    sequences = _seqs(corpus)
    total_tokens = sum(len(s) for s in sequences)

    print(f"\n{'═' * 60}")
    print("  Rongorongo Reading-Order Tests")
    print(f"  Corpus : {len(corpus)} tablets  |  {total_tokens:,} tokens")
    if model is not None:
        print(f"  Model  : order={model.order}  vocab={len(model.vocab):,}")
    print(f"{'═' * 60}")

    results: dict = {
        "corpus_tablets": len(corpus),
        "corpus_tokens": total_tokens,
        "tests_run": sorted(tests),
    }
    if model is not None:
        results["model_order"] = model.order
        results["model_vocab"] = len(model.vocab)

    if 1 in tests:
        results["test1"] = test1_conditional_entropy(sequences)
    if 2 in tests and model is not None:
        results["test2"] = test2_perplexity(model, sequences)
    if 3 in tests:
        results["test3"] = test3_line_boundary(corpus)
    if 4 in tests:
        results["test4"] = test4_recto_verso(corpus_dir)

    print()

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")
        log.info("Results written → %s", args.output)


if __name__ == "__main__":
    main()
