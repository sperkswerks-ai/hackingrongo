"""scripts/refine_assignments.py

Gumbel-Softmax warm-start gradient refinement of MCMC phoneme assignments.

Takes the top-K hypotheses from ranking.json, refines each with a short
gradient descent through a differentiable soft-LM scorer, then reads out
the refined hard assignment and re-scores with the real KN-smoothed LMScorer.

Design
------
* P matrix [n_signs × n_phonemes]: row-stochastic via softmax over logits.
  Warm-started from the MCMC hard assignment (near-one-hot logits).
* Calendar anchors (sign 040 = kokore, sign 152 = omotohi) plus any
  assignment with confidence ≥ 1.0 and evidence_count ≥ 2 are frozen:
  their logit rows are clamped to a sharp one-hot and receive zero gradient.
* Differentiable scorer uses add-α smoothed bigram log-probs:
    T[a, b] = log P_smooth(b | a)   (shape [V × V], V = phoneme count)
  Soft score for the full corpus:
    score = Σ_{i,j} C[i,j] · (P[i] @ T) · P[j]
          = (P @ T  *  C @ P).sum()         ← batched bilinear
  where C[i, j] = count of adjacent sign pair (i→j) in corpus.
* Gumbel-Softmax: at each step sample G ~ Gumbel(0,1); zero G for frozen
  rows; compute P = softmax((logits + G) / τ).
* Occupancy penalty: λ · Σ_p max(0, Σ_s P[s,p] − k_p)²  (k_p = 4.0).
* Schedules: τ 1.0→0.1 (linear), λ 0.0→λ_max (linear).
* After refinement: hard assignment = argmax(P), re-scored with LMScorer.
* Output: ranking.json with added fields per hypothesis:
    refined_assignments, refined_soft_score, refined_lm_score,
    refinement_delta_lm, refinement_meta.

Usage
-----
    python scripts/refine_assignments.py \\
        --ranking  outputs/decipherment/ranking.json \\
        --lm-dir   data/language_models \\
        --n-steps  100 \\
        --top-k    5 \\
        --output   outputs/decipherment/ranking.json
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Hard-pinned calendar anchors (Mamari calendar evidence, see run_decipherment.py)
# ---------------------------------------------------------------------------
ALWAYS_FROZEN: dict[str, str] = {
    "152": "omotohi",  # full moon — score 1.0, calendar-exclusive
    "040": "kokore",   # night count — score 0.62, calendar-dominant
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Corpus loading
# ---------------------------------------------------------------------------

def _load_corpus_sequences(
    project_root: Path,
    cfg_path: Path,
) -> list[list[str]]:
    """Load corpus using OmegaConf + hackingrongo data loader.

    Returns list of barthel-code sequences (same format as MCMCSampler).
    Falls back to an empty list if data loading fails.
    """
    try:
        from omegaconf import OmegaConf
        from hackingrongo.data.corpus import load_corpus

        cfg = OmegaConf.load(cfg_path)
        tablets = load_corpus(cfg, project_root)
        seqs = [[tok.barthel_code for tok in t.tokens] for t in tablets]
        log.info("Corpus: %d tablets, %d total tokens.",
                 len(seqs), sum(len(s) for s in seqs))
        return seqs
    except ImportError:
        raise  # misconfigured environment — not a graceful-degradation case
    except Exception as exc:
        log.warning("Could not load corpus via hackingrongo loader: %s", exc, exc_info=True)
        return []


# ---------------------------------------------------------------------------
# Co-occurrence matrix
# ---------------------------------------------------------------------------

def _build_cooccurrence(
    corpus_seqs: list[list[str]],
    sign_to_idx: dict[str, int],
) -> np.ndarray:
    """Build C[i, j] = count of adjacent pair (sign_i → sign_j) in corpus.

    Tokens not present in sign_to_idx are silently skipped.
    """
    n = len(sign_to_idx)
    C = np.zeros((n, n), dtype=np.float32)
    for seq in corpus_seqs:
        filtered = [s for s in seq if s in sign_to_idx]
        for t in range(1, len(filtered)):
            C[sign_to_idx[filtered[t - 1]], sign_to_idx[filtered[t]]] += 1.0
    return C


# ---------------------------------------------------------------------------
# Transition matrix (add-α smoothed bigram log-probs)
# ---------------------------------------------------------------------------

def _build_transition_matrix(
    lm_dir: Path,
    phoneme_inv: list[str],
    alpha: float = 0.01,
) -> np.ndarray:
    """Build T[a, b] = log P_smooth(phoneme_b | phoneme_a), natural log.

    Combines raw bigram counts from both pre- and post-contact LMs, then
    applies add-α Laplace smoothing restricted to the phoneme inventory so
    every cell is strictly positive (no -∞ gradient sinks).
    """
    try:
        from hackingrongo.data.rapa_nui_corpus import NGramLM
    except ImportError as exc:
        raise RuntimeError("hackingrongo package must be importable") from exc

    V = len(phoneme_inv)
    ph_to_idx = {ph: i for i, ph in enumerate(phoneme_inv)}
    raw = np.zeros((V, V), dtype=np.float64)

    for fname in ("pre_contact_lm.json", "post_contact_lm.json"):
        lm_path = lm_dir / fname
        if not lm_path.exists():
            log.warning("LM file not found, skipping: %s", lm_path)
            continue
        lm = NGramLM.load(lm_path)
        # _counts[2][(context_token,)][next_token] = raw count
        bigram_counts = lm._counts.get(2, {})
        for ctx_tuple, next_counts in bigram_counts.items():
            if len(ctx_tuple) == 1 and ctx_tuple[0] in ph_to_idx:
                a = ph_to_idx[ctx_tuple[0]]
                for word, count in next_counts.items():
                    if word in ph_to_idx:
                        raw[a, ph_to_idx[word]] += count

    smoothed = raw + alpha  # add-α: every cell > 0
    row_sums = smoothed.sum(axis=1, keepdims=True)
    T = np.log(smoothed / row_sums)  # natural log P(b | a)
    log.info(
        "Transition matrix: shape %s, range [%.3f, %.3f]",
        T.shape, T.min(), T.max(),
    )
    return T.astype(np.float32)


# ---------------------------------------------------------------------------
# Logits initialisation
# ---------------------------------------------------------------------------

def _init_logits(
    hypothesis: dict[str, Any],
    sign_to_idx: dict[str, int],
    ph_to_idx: dict[str, int],
    n_signs: int,
    n_phonemes: int,
    init_sharpness: float = 5.0,
) -> torch.Tensor:
    """Initialise logits near one-hot from the MCMC hard assignment."""
    logits = torch.zeros(n_signs, n_phonemes)
    for asgn in hypothesis.get("assignments", []):
        s, p = asgn.get("sign_code"), asgn.get("phoneme")
        if s in sign_to_idx and p in ph_to_idx:
            logits[sign_to_idx[s], ph_to_idx[p]] = init_sharpness
    return logits


# ---------------------------------------------------------------------------
# Frozen-sign mask
# ---------------------------------------------------------------------------

def _build_frozen_mask(
    hypothesis: dict[str, Any],
    sign_to_idx: dict[str, int],
    ph_to_idx: dict[str, int],
    n_signs: int,
    n_phonemes: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (frozen_rows [n_signs bool], anchor_one_hots [n_signs, n_phonemes]).

    Frozen sources (in priority order):
    1. ALWAYS_FROZEN calendar anchors (040, 152).
    2. High-confidence MCMC assignments: confidence >= 1.0 AND evidence_count >= 2.

    If a frozen sign's phoneme is outside ph_to_idx (shouldn't happen for the
    canonical inventory but can occur in edge cases), the sign is skipped.
    """
    frozen = torch.zeros(n_signs, dtype=torch.bool)
    one_hots = torch.zeros(n_signs, n_phonemes)

    # High-confidence MCMC assignments first (so ALWAYS_FROZEN overrides below)
    for asgn in hypothesis.get("assignments", []):
        s, p = asgn.get("sign_code"), asgn.get("phoneme")
        conf = float(asgn.get("confidence", 0.0))
        ev = int(asgn.get("evidence_count", 0))
        if s in sign_to_idx and p in ph_to_idx and conf >= 1.0 and ev >= 2:
            si = sign_to_idx[s]
            frozen[si] = True
            one_hots[si, ph_to_idx[p]] = 1.0

    # Calendar anchors always override
    for sign_code, phoneme in ALWAYS_FROZEN.items():
        if sign_code in sign_to_idx and phoneme in ph_to_idx:
            si = sign_to_idx[sign_code]
            pi = ph_to_idx[phoneme]
            frozen[si] = True
            one_hots[si] = 0.0
            one_hots[si, pi] = 1.0

    n_frozen = int(frozen.sum().item())
    log.debug("  Frozen signs: %d / %d", n_frozen, n_signs)
    return frozen, one_hots


# ---------------------------------------------------------------------------
# Core refinement loop
# ---------------------------------------------------------------------------

def _refine_hypothesis(
    hypothesis: dict[str, Any],
    C: torch.Tensor,         # [n_signs, n_signs] bigram co-occurrence
    T: torch.Tensor,         # [n_phonemes, n_phonemes] smoothed log-prob
    sign_to_idx: dict[str, int],
    ph_to_idx: dict[str, int],
    phoneme_inv: list[str],
    n_steps: int = 100,
    lr: float = 0.05,
    occ_cap: float = 4.0,
    lambda_occ_max: float = 0.5,
) -> dict[str, Any]:
    """Run gradient refinement on one hypothesis. Returns updated hypothesis dict."""

    n_signs = len(sign_to_idx)
    n_phonemes = len(ph_to_idx)

    logits = _init_logits(
        hypothesis, sign_to_idx, ph_to_idx, n_signs, n_phonemes
    )
    logits.requires_grad_(True)

    frozen, one_hots = _build_frozen_mask(
        hypothesis, sign_to_idx, ph_to_idx, n_signs, n_phonemes
    )

    # Pin frozen rows to sharp one-hot from the start
    with torch.no_grad():
        logits[frozen] = one_hots[frozen] * 10.0

    optimizer = torch.optim.Adam([logits], lr=lr)

    tau_start, tau_end = 1.0, 0.1
    best_score = -math.inf
    best_logits = logits.detach().clone()

    for step in range(n_steps):
        frac = step / max(n_steps - 1, 1)
        tau = tau_start + frac * (tau_end - tau_start)
        lambda_occ = lambda_occ_max * frac

        # Gumbel noise — zeroed for frozen signs
        eps = 1e-10
        u = torch.rand_like(logits).clamp(eps, 1.0 - eps)
        gumbel = -torch.log(-torch.log(u))
        gumbel = gumbel.detach()          # noise is not a learnable variable
        with torch.no_grad():
            gumbel[frozen] = 0.0

        P = F.softmax((logits + gumbel) / tau, dim=-1)

        # Override frozen rows with exact one-hots (detaches their contribution)
        P_eff = P.clone()
        with torch.no_grad():
            P_eff[frozen] = one_hots[frozen]

        # Differentiable score:
        #   score = Σ_{i,j} C[i,j] · (P[i] @ T) · P[j]
        #         = (P @ T  *  C @ P).sum()
        Q = P_eff @ T          # [n_signs, n_phonemes]
        score = (Q * (C @ P_eff)).sum()

        # Soft occupancy penalty
        expected_occ = P_eff.sum(dim=0)                         # [n_phonemes]
        excess = torch.clamp(expected_occ - occ_cap, min=0.0)
        penalty = lambda_occ * (excess ** 2).sum()

        loss = -(score - penalty)

        optimizer.zero_grad()
        loss.backward()

        # Ensure no gradient leaks into frozen rows
        with torch.no_grad():
            if logits.grad is not None:
                logits.grad[frozen] = 0.0

        optimizer.step()

        # Re-pin frozen rows (Adam step may have shifted them slightly)
        with torch.no_grad():
            logits[frozen] = one_hots[frozen] * 10.0

        if score.item() > best_score:
            best_score = score.item()
            best_logits = logits.detach().clone()

        if step % 20 == 0:
            log.debug(
                "    step %4d: score=%.4f  penalty=%.4f  τ=%.3f  λ=%.3f",
                step, score.item(), penalty.item(), tau, lambda_occ,
            )

    # ── Hard readout ──────────────────────────────────────────────────────────
    with torch.no_grad():
        # Sharp softmax of best logits for readout
        P_final = F.softmax(best_logits / 0.1, dim=-1)
        P_final[frozen] = one_hots[frozen]
        hard_idx = P_final.argmax(dim=-1)                         # [n_signs]
        # Per-sign confidence: 1 − normalised entropy
        entropy = -(P_final * (P_final.clamp(1e-12).log())).sum(dim=-1)
        max_ent = math.log(n_phonemes)
        confidence = (1.0 - entropy / max_ent).clamp(0.0, 1.0)

    idx_to_sign = {v: k for k, v in sign_to_idx.items()}
    refined_assignments: list[dict] = []
    for si in range(n_signs):
        refined_assignments.append({
            "sign_code": idx_to_sign[si],
            "phoneme": phoneme_inv[hard_idx[si].item()],
            "confidence": round(float(confidence[si].item()), 4),
            "refinement_entropy": round(float(entropy[si].item()), 4),
        })

    return {
        **hypothesis,
        "refined_assignments": refined_assignments,
        "refined_soft_score": round(float(best_score), 4),
        "refinement_meta": {
            "n_steps": n_steps,
            "tau_start": tau_start,
            "tau_end": tau_end,
            "lambda_occ_max": lambda_occ_max,
            "occ_cap": occ_cap,
            "n_frozen": int(frozen.sum().item()),
            "n_signs": n_signs,
            "n_phonemes": n_phonemes,
        },
    }


# ---------------------------------------------------------------------------
# LMScorer re-scoring (honest KN comparison)
# ---------------------------------------------------------------------------

def _build_lm_scorer(project_root: Path, cfg_path: Path) -> "LMScorer | None":
    """Build a KN-smoothed LMScorer once for reuse across all hypotheses."""
    try:
        from omegaconf import OmegaConf
        from hackingrongo.zone_c.lm_scoring import LMScorer
        cfg = OmegaConf.load(cfg_path)
        return LMScorer(cfg, project_root)
    except Exception as exc:
        log.warning("Could not build LMScorer: %s", exc, exc_info=True)
        return None


def _rescore_with_lm(
    refined_hyp: dict[str, Any],
    corpus_seqs: list[list[str]],
    scorer: "LMScorer | None",
) -> float | None:
    """Score the refined hard assignment with the KN-smoothed LMScorer.

    Returns mean ensemble log₂-prob across corpus tablets, or None on failure.
    """
    if scorer is None:
        return None
    try:
        import statistics
        phoneme_map: dict[str, str] = {
            a["sign_code"]: a["phoneme"]
            for a in refined_hyp.get("refined_assignments", [])
        }
        log_probs: list[float] = []
        for seq in corpus_seqs:
            phoneme_seq = [phoneme_map.get(tok, "<UNK>") for tok in seq]
            if not phoneme_seq:
                continue
            result = scorer.score(phoneme_seq)
            if math.isfinite(result.ensemble_log_prob):
                log_probs.append(result.ensemble_log_prob)
        if not log_probs:
            return None
        return round(statistics.mean(log_probs), 6)
    except Exception as exc:
        log.warning("LMScorer re-scoring failed: %s", exc, exc_info=True)
        return None


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Gumbel-Softmax gradient refinement of MCMC phoneme assignments",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--ranking", type=Path, required=True,
        help="Path to ranking.json produced by run_decipherment.py",
    )
    parser.add_argument(
        "--lm-dir", type=Path, required=True,
        help="Directory containing *_lm.json language model files",
    )
    parser.add_argument(
        "--conf", type=Path, default=Path("conf/config.yaml"),
        help="Path to Hydra config (relative to --project-root)",
    )
    parser.add_argument(
        "--project-root", type=Path, default=Path("."),
        help="Project root directory (parent of conf/ and data/)",
    )
    parser.add_argument(
        "--n-steps", type=int, default=100,
        help="Number of gradient steps per hypothesis",
    )
    parser.add_argument(
        "--top-k", type=int, default=5,
        help="Number of top hypotheses to refine",
    )
    parser.add_argument(
        "--lr", type=float, default=0.05,
        help="Adam learning rate",
    )
    parser.add_argument(
        "--occ-cap", type=float, default=4.0,
        help="Occupancy cap k_p (max expected assignments per phoneme)",
    )
    parser.add_argument(
        "--lambda-occ-max", type=float, default=0.5,
        help="Maximum occupancy penalty coefficient",
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Output path (defaults to overwriting --ranking)",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Enable DEBUG-level logging",
    )
    args = parser.parse_args(argv)

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # ── Load ranking.json ─────────────────────────────────────────────────────
    if not args.ranking.exists():
        log.error("ranking.json not found: %s", args.ranking)
        sys.exit(1)

    with open(args.ranking) as fh:
        ranking = json.load(fh)

    hypotheses = ranking.get("hypotheses", [])
    top_k = min(args.top_k, len(hypotheses))
    if top_k == 0:
        log.error("No hypotheses found in %s", args.ranking)
        sys.exit(1)

    log.info("Refining top %d / %d hypotheses.", top_k, len(hypotheses))

    # ── Build shared inventories ──────────────────────────────────────────────
    # Phoneme inventory: union of all phonemes across top-K hypotheses
    phoneme_inv = sorted({
        a["phoneme"]
        for h in hypotheses[:top_k]
        for a in h.get("assignments", [])
    })
    # Always include the anchor phonemes even if not in current hypotheses
    for ph in ALWAYS_FROZEN.values():
        if ph not in phoneme_inv:
            phoneme_inv.append(ph)
    phoneme_inv = sorted(set(phoneme_inv))
    ph_to_idx = {ph: i for i, ph in enumerate(phoneme_inv)}
    log.info("Phoneme inventory: %d tokens.", len(phoneme_inv))

    # Sign inventory: union of all sign codes across top-K hypotheses
    all_sign_codes = sorted({
        a["sign_code"]
        for h in hypotheses[:top_k]
        for a in h.get("assignments", [])
    })
    sign_to_idx = {s: i for i, s in enumerate(all_sign_codes)}
    log.info("Sign inventory: %d signs.", len(all_sign_codes))

    # ── Load corpus sequences ─────────────────────────────────────────────────
    cfg_path = args.project_root / args.conf
    corpus_seqs = _load_corpus_sequences(args.project_root, cfg_path)
    if not corpus_seqs:
        log.warning(
            "Corpus is empty — bigram co-occurrence matrix will be all zeros. "
            "Gradient signal will be flat; refinement is a no-op."
        )

    # ── Build tensors ─────────────────────────────────────────────────────────
    log.info("Building bigram co-occurrence matrix…")
    C_np = _build_cooccurrence(corpus_seqs, sign_to_idx)
    n_bigrams = int(C_np.sum())
    log.info("  %d bigrams involving hypothesis signs.", n_bigrams)
    C = torch.from_numpy(C_np)

    log.info("Building LM transition matrix (add-α smoothed)…")
    T_np = _build_transition_matrix(args.lm_dir, phoneme_inv, alpha=0.01)
    T = torch.from_numpy(T_np)

    # Build LMScorer once — not per hypothesis (loading all LM files is expensive)
    log.info("Building KN LMScorer for re-scoring…")
    lm_scorer = _build_lm_scorer(args.project_root, cfg_path)

    # ── Refine each hypothesis ────────────────────────────────────────────────
    refined_hypotheses: list[dict] = []

    for idx, hyp in enumerate(hypotheses[:top_k]):
        hyp_id = hyp.get("hypothesis_id", f"hyp_{idx}")
        orig_score = hyp.get("overall_lm_score", None)
        log.info(
            "─── Hypothesis %d/%d: %s  (original LM score: %s)",
            idx + 1, top_k, hyp_id,
            f"{orig_score:.4f}" if orig_score is not None else "N/A",
        )

        refined = _refine_hypothesis(
            hyp, C, T,
            sign_to_idx, ph_to_idx, phoneme_inv,
            n_steps=args.n_steps,
            lr=args.lr,
            occ_cap=args.occ_cap,
            lambda_occ_max=args.lambda_occ_max,
        )

        # Re-score with KN LMScorer for honest comparison
        log.info("  Re-scoring refined assignment with KN LMScorer…")
        refined_lm = _rescore_with_lm(refined, corpus_seqs, lm_scorer)
        refined["refined_lm_score"] = refined_lm

        if orig_score is not None and refined_lm is not None:
            delta = refined_lm - orig_score
            refined["refinement_delta_lm"] = round(delta, 6)
            log.info("  KN LM score: %.4f → %.4f  (Δ = %+.4f)", orig_score, refined_lm, delta)
        else:
            refined["refinement_delta_lm"] = None
            log.info("  KN LM score (refined): %s", refined_lm)

        refined_hypotheses.append(refined)

    # ── Merge back and save ───────────────────────────────────────────────────
    ranking["hypotheses"][:top_k] = refined_hypotheses

    import os, tempfile
    out_path = args.output or args.ranking
    tmp_fd, tmp_name = tempfile.mkstemp(dir=out_path.parent, suffix=".tmp")
    try:
        with os.fdopen(tmp_fd, "w", encoding="utf-8") as fh:
            json.dump(ranking, fh, indent=2)
        os.replace(tmp_name, out_path)
    except BaseException:
        os.unlink(tmp_name)
        raise

    log.info("Wrote refined ranking to %s", out_path)
    log.info(
        "Summary: %d hypotheses refined, %d bigrams used for gradient signal.",
        top_k, n_bigrams,
    )


if __name__ == "__main__":
    main()
