"""
complete_sequence.py — predict masked signs in rongorongo sequences.

Given a Barthel-code sequence with one [MASK] token, returns the top-k most
probable completions ranked by full-sequence log₂-probability (left and right
context, via the trained NgramModel).

Modes
-----
  --sequence  inline sequence with a [MASK] placeholder
  --tablet    load a tablet JSON and reconstruct all illegible (?) glyphs

Usage
-----
    # Inline single-mask
    conda run -n hackingrongo python scripts/complete_sequence.py \\
        --sequence 007 '[MASK]' 010 \\
        --model outputs/zone_b/sequence_model.json

    # Reconstruct Tablet F (81 glyphs, several uncertain readings)
    conda run -n hackingrongo python scripts/complete_sequence.py \\
        --tablet F \\
        --model outputs/zone_b/sequence_model.json \\
        --out outputs/zone_b/tablet_f_reconstruction.json

    # Any other damaged tablet
    conda run -n hackingrongo python scripts/complete_sequence.py \\
        --tablet G --top-k 5
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from omegaconf import OmegaConf  # noqa: E402

from hackingrongo.zone_b.sequence_model import NgramModel  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

MASK = "[MASK]"
_SKIP = {"<BOS>", "<EOS>", "<UNK>"}


def _safe_int(val: object) -> int:
    try:
        return int(val)  # type: ignore[arg-type]
    except (ValueError, TypeError):
        return 0


# ---------------------------------------------------------------------------
# Core prediction
# ---------------------------------------------------------------------------

def fill_mask(
    sequence: list[str],
    model: NgramModel,
    k: int = 10,
    pool: int | None = None,
) -> list[tuple[str, float, float]]:
    """Return top-k (sign, left_log2p, seq_log2p) for the single [MASK].

    Candidates are drawn from top_k_next on the left context, then re-ranked
    by the full-sequence score which incorporates right context too.

    Parameters
    ----------
    sequence:   Barthel/Horley codes with exactly one MASK token.
    model:      Trained NgramModel.
    k:          Number of results to return.
    pool:       Candidate pool size before right-context re-ranking (default 4k).

    Returns
    -------
    List of (sign, left_log2p, seq_log2p) sorted by seq_log2p descending.
    left_log2p  — log₂ P(sign | left n-gram context)
    seq_log2p   — log₂ P(full sequence with sign substituted)
    """
    if MASK not in sequence:
        raise ValueError("sequence contains no [MASK] token")
    mask_idx = sequence.index(MASK)
    left = sequence[:mask_idx]
    right = sequence[mask_idx + 1:]

    ctx = list(left[-(model.order - 1):])
    candidates = model.top_k_next(ctx, k=(pool or k * 4))

    scored: list[tuple[str, float, float]] = []
    for sign, left_lp in candidates:
        if sign in _SKIP:
            continue
        seq_lp = model.score(left + [sign] + right)
        scored.append((sign, left_lp, seq_lp))

    scored.sort(key=lambda x: -x[2])
    return scored[:k]


# ---------------------------------------------------------------------------
# Tablet loading with mask-site tracking
# ---------------------------------------------------------------------------

def load_tablet_sequence(
    tablet_path: Path,
) -> tuple[list[str | None], list[dict | None]]:
    """Load a tablet JSON and return (tokens, mask_info) in scribal reading order.

    Compound glyphs are expanded to their horley_components (matching the
    training corpus). Illegible glyphs (horley_code is None, no components)
    produce a None token with a mask_info dict recording their position.

    Returns
    -------
    tokens:     flat list; str for resolved signs, None for illegible glyphs.
    mask_info:  parallel list; dict for each illegible position, None otherwise.
    """
    data = json.loads(tablet_path.read_text(encoding="utf-8"))
    glyphs = data.get("glyphs", [])

    line_groups: dict[tuple, list[dict]] = defaultdict(list)
    for g in glyphs:
        side = str(g.get("side", "a")).lower()
        line_num = _safe_int(g.get("line", 0))
        line_groups[(side, line_num)].append(g)

    side_order = {"a": 0, "b": 1, "c": 2}
    sorted_lines = sorted(
        line_groups.items(),
        key=lambda kv: (side_order.get(kv[0][0], 9), kv[0][1]),
    )

    tokens: list[str | None] = []
    mask_info: list[dict | None] = []

    for _line_key, line_glyphs in sorted_lines:
        line_glyphs = sorted(line_glyphs, key=lambda g: _safe_int(g.get("glyph_num", 0)))
        n_inv = sum(1 for g in line_glyphs if g.get("inverted", False))
        if line_glyphs and n_inv > len(line_glyphs) / 2:
            line_glyphs = list(reversed(line_glyphs))

        for g in line_glyphs:
            hc = g.get("horley_code")
            comps = g.get("horley_components") or []

            if hc:
                tokens.append(hc)
                mask_info.append(None)
            elif comps:
                for comp in comps:
                    tokens.append(comp)
                    mask_info.append(None)
            elif g.get("barthel_code") == "?":
                # Genuinely illegible — Barthel himself could not read this sign
                tokens.append(None)
                mask_info.append({
                    "position": g.get("position", 0),
                    "side": str(g.get("side", "a")),
                    "line": _safe_int(g.get("line", 0)),
                    "glyph_num": str(g.get("glyph_num", "?")),
                    "barthel_raw": str(g.get("barthel_code", "?")),
                    "seq_index": len(tokens) - 1,
                })

    return tokens, mask_info


def reconstruct_tablet(
    tablet_path: Path,
    model: NgramModel,
    k: int = 10,
) -> list[dict]:
    """Run fill_mask on every illegible glyph in a tablet.

    Each masked glyph is predicted independently using all surrounding
    resolved tokens as context (other masks are skipped, not filled in).
    """
    tokens, mask_infos = load_tablet_sequence(tablet_path)
    results: list[dict] = []

    for i, info in enumerate(mask_infos):
        if info is None:
            continue

        left = [t for t in tokens[:i] if t is not None]
        right = [t for t in tokens[i + 1:] if t is not None]
        seq = left + [MASK] + right

        predictions = fill_mask(seq, model, k=k)
        results.append({
            **info,
            "left_context": left[-(model.order - 1):],
            "right_context": right[: model.order - 1],
            "predictions": [
                {
                    "rank": rank,
                    "sign": sign,
                    "left_log2p": round(left_lp, 4),
                    "seq_log2p": round(seq_lp, 4),
                    "confidence": round(2.0**left_lp, 4),
                }
                for rank, (sign, left_lp, seq_lp) in enumerate(predictions, 1)
            ],
        })

    return results


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------

def _print_inline(
    sequence: list[str],
    predictions: list[tuple[str, float, float]],
) -> None:
    mask_idx = sequence.index(MASK)
    left = sequence[:mask_idx]
    right = sequence[mask_idx + 1:]
    ctx_str = " ".join(left or ["(start)"])
    right_str = " ".join(right or ["(end)"])
    print(f"\nContext: {ctx_str}  [MASK]  {right_str}")
    print(f"Position {mask_idx + 1} of {len(sequence)}\n")
    print(f"{'Rank':>4}  {'Sign':<10}  {'left log₂p':>10}  {'seq log₂p':>10}  {'P(sign|ctx)':>12}")
    print("─" * 54)
    for rank, (sign, left_lp, seq_lp) in enumerate(predictions, 1):
        print(f"{rank:>4}  {sign:<10}  {left_lp:>10.3f}  {seq_lp:>10.3f}  {2.0**left_lp:>12.4f}")
    print()


def _print_tablet_report(tablet_id: str, results: list[dict], model_order: int) -> None:
    n = len(results)
    print(f"\nTablet {tablet_id} — {n} illegible glyph{'s' if n != 1 else ''} reconstructed")
    print(f"{'─' * 60}\n")
    for rec in results:
        loc = (
            f"side={rec['side']}  line={rec['line']}  "
            f"glyph={rec['glyph_num']}  pos={rec['position']}"
        )
        ctx_l = " ".join(rec["left_context"]) or "—"
        ctx_r = " ".join(rec["right_context"]) or "—"
        print(f"  [{loc}]")
        print(f"  Context: … {ctx_l}  [?]  {ctx_r} …")
        preds = rec["predictions"]
        for p in preds[:5]:
            bar = "█" * round(p["confidence"] * 20)
            print(f"    {p['rank']:>2}. {p['sign']:<8}  {p['confidence']:.4f}  {bar}")
        print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Predict masked signs in rongorongo sequences.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--sequence",
        nargs="+",
        metavar="CODE",
        help="Barthel/Horley codes with one [MASK] placeholder.",
    )
    mode.add_argument(
        "--tablet",
        metavar="ID",
        help="Tablet ID (e.g. F) — reconstruct all illegible glyphs.",
    )
    p.add_argument(
        "--model",
        type=Path,
        default=None,
        help="Path to trained NgramModel JSON (default: outputs/zone_b/sequence_model.json).",
    )
    p.add_argument(
        "--top-k",
        type=int,
        default=10,
        metavar="K",
        help="Predictions per mask position (default: 10).",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Write results as JSON to this path.",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    cfg = OmegaConf.load(PROJECT_ROOT / "conf" / "config.yaml")
    model_path: Path = args.model or (
        PROJECT_ROOT / cfg.paths.outputs_dir / "zone_b" / "sequence_model.json"
    )

    if not model_path.exists():
        log.error("Model not found: %s", model_path)
        log.error("Train first:  conda run -n hackingrongo python scripts/train_sequence_model.py")
        sys.exit(1)

    log.info("Loading model from %s …", model_path)
    model = NgramModel.load(model_path)
    log.info(
        "NgramModel  order=%d  vocab=%d  total_tokens=%d",
        model.order, len(model.vocab), model._total_tokens,
    )

    # --sequence mode --------------------------------------------------------
    if args.sequence is not None:
        sequence = args.sequence
        if MASK not in sequence:
            log.error("No [MASK] in --sequence. Use '[MASK]' (with quotes) as placeholder.")
            sys.exit(1)
        predictions = fill_mask(sequence, model, k=args.top_k)
        _print_inline(sequence, predictions)

        if args.out:
            payload = {
                "sequence": sequence,
                "mask_position": sequence.index(MASK),
                "predictions": [
                    {
                        "rank": r,
                        "sign": s,
                        "left_log2p": round(l, 4),
                        "seq_log2p": round(q, 4),
                        "confidence": round(2.0**l, 4),
                    }
                    for r, (s, l, q) in enumerate(predictions, 1)
                ],
            }
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(
                json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            log.info("Results written to %s", args.out)

    # --tablet mode ----------------------------------------------------------
    else:
        corpus_dir = PROJECT_ROOT / cfg.paths.corpus_dir
        tablet_path = corpus_dir / f"{args.tablet}.json"
        if not tablet_path.exists():
            log.error("Tablet file not found: %s", tablet_path)
            sys.exit(1)

        log.info("Reconstructing Tablet %s …", args.tablet)
        results = reconstruct_tablet(tablet_path, model, k=args.top_k)

        if not results:
            log.info("No illegible glyphs found in Tablet %s.", args.tablet)
        else:
            _print_tablet_report(args.tablet, results, model.order)
            log.info("%d mask site(s) reconstructed.", len(results))

        if args.out:
            payload = {
                "tablet_id": args.tablet,
                "model_path": str(model_path),
                "model_order": model.order,
                "n_illegible": len(results),
                "reconstructions": results,
            }
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(
                json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            log.info("Results written to %s", args.out)


if __name__ == "__main__":
    main()
