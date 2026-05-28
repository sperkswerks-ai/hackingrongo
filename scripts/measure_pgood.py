"""
measure_pgood.py — quantum hardness analysis for rongorongo decipherment.

Estimates p_good (fraction of random sign→phoneme assignments that score
above a threshold under the Rapa Nui LM), derives the theoretical Grover
oracle call count, and compares to MCMC iteration requirements.

The quantum speedup ratio is the core publishable result: it quantifies how
much faster Grover-based search would find a "good" phoneme assignment than
classical random search.

Thresholds are normalised: τ = 0.90 means the score is in the top (1-0.90)
= 10% of the observed random-sample score range.  Because Rapa Nui LM scores
are negative log-probabilities, the normalised score is
    s_norm = (s - s_min) / (s_max - s_min)
and "good" means s_norm >= τ.

Usage
-----
    python scripts/measure_pgood.py \\
        --corpus-dir data/corpus \\
        --lm-dir data/language_models \\
        --n-samples 10000 \\
        --thresholds 0.90,0.95,0.99 \\
        --mcmc-iterations 5000 \\
        --output outputs/zone_b/pgood_analysis.json

    # Fast smoke test (100 samples)
    python scripts/measure_pgood.py --smoke-test
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from hackingrongo.data.rapa_nui_corpus import NGramLM  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_barthel_sequences(corpus_dir: Path) -> list[list[str]]:
    sequences: list[list[str]] = []
    for path in sorted(corpus_dir.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        glyphs = data.get("glyphs", [])
        seq = [str(g["barthel_code"]) for g in glyphs if g.get("barthel_code")]
        if len(seq) >= 3:
            sequences.append(seq)
    return sequences


def _load_lms(lm_dir: Path) -> list[NGramLM]:
    lms: list[NGramLM] = []
    for name in ["pre_contact_lm", "post_contact_lm"]:
        path = lm_dir / f"{name}.json"
        if path.exists():
            log.info("Loading %s …", path.name)
            lms.append(NGramLM.load(path))
        else:
            log.warning("Not found, skipping: %s", path)
    if not lms:
        log.error("No LMs found in %s — run build_language_models.py first.", lm_dir)
        sys.exit(1)
    return lms


def _phoneme_inventory(lms: list[NGramLM]) -> list[str]:
    """Collect CV syllable tokens from LM vocabularies."""
    phonemes: set[str] = set()
    for lm in lms:
        vocab = lm._vocab if hasattr(lm, "_vocab") else set()
        for tok in vocab:
            if tok and not tok.startswith("<") and 1 <= len(tok) <= 6:
                phonemes.add(tok)
    return sorted(phonemes)


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def _score_assignment(
    phone_map: dict[str, str],
    corpus_seqs: list[list[str]],
    lms: list[NGramLM],
) -> float:
    """Mean per-token log-prob of the translated corpus under all LMs."""
    total_lp = 0.0
    total_n = 0
    for seq in corpus_seqs:
        translated = [phone_map.get(s, "<UNK>") for s in seq]
        for lm in lms:
            if len(translated) >= lm.order:
                total_lp += lm.score_sequence(translated)
                total_n += len(translated)
    return total_lp / total_n if total_n > 0 else -math.inf


def _sample_scores(
    corpus_seqs: list[list[str]],
    lms: list[NGramLM],
    signs: list[str],
    phonemes: list[str],
    n_samples: int,
    seed: int = 42,
) -> list[float]:
    rng = np.random.default_rng(seed)
    phonemes_arr = np.array(phonemes)
    n_phonemes = len(phonemes)
    n_signs = len(signs)
    log_interval = max(100, n_samples // 20)

    scores: list[float] = []
    t0 = time.perf_counter()

    for i in range(n_samples):
        if i > 0 and i % log_interval == 0:
            elapsed = time.perf_counter() - t0
            eta = elapsed / i * (n_samples - i)
            log.info("  %d/%d  (%.0fs elapsed, ETA %.0fs)", i, n_samples, elapsed, eta)

        # Clear LM n-gram caches between samples: each sample uses a different
        # phoneme mapping so cached n-gram lookups from prior samples are never
        # reused and would otherwise accumulate to hundreds of millions of entries
        # (up to V^order, e.g. 45^5 ≈ 184 M for order-5 LMs), exhausting RAM.
        for lm in lms:
            lm._lp_cache.clear()

        idx = rng.integers(0, n_phonemes, size=n_signs)
        phone_map = {sign: phonemes_arr[k] for sign, k in zip(signs, idx)}
        scores.append(_score_assignment(phone_map, corpus_seqs, lms))

    return scores


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def _analyze(
    scores: list[float],
    thresholds: list[float],
    mcmc_iterations: int,
    n_signs: int,
    n_phonemes: int,
) -> dict:
    finite = np.array([s for s in scores if math.isfinite(s)])
    s_min = float(np.min(finite))
    s_max = float(np.max(finite))
    score_range = s_max - s_min

    results: dict = {
        "n_samples":   len(scores),
        "n_finite":    len(finite),
        "n_signs":     n_signs,
        "n_phonemes":  n_phonemes,
        "score_distribution": {
            "mean": float(np.mean(finite)),
            "std":  float(np.std(finite)),
            "min":  s_min,
            "max":  s_max,
            "percentiles": {
                str(pct): float(np.percentile(finite, pct))
                for pct in [50, 90, 95, 99]
            },
        },
        "thresholds": [],
    }

    for tau in thresholds:
        cutoff = s_min + tau * score_range
        n_good = int(np.sum(finite >= cutoff))
        p_good = n_good / len(finite) if len(finite) > 0 else 0.0

        if p_good > 0:
            grover_calls   = math.ceil(math.pi / (4.0 * math.sqrt(p_good)))
            classical_calls = math.ceil(1.0 / p_good)
            speedup        = classical_calls / grover_calls
            mcmc_vs_grover = mcmc_iterations / grover_calls
        else:
            grover_calls    = -1
            classical_calls = -1
            speedup         = math.inf
            mcmc_vs_grover  = math.inf

        results["thresholds"].append({
            "tau":                    tau,
            "score_cutoff":           round(cutoff, 6),
            "n_good":                 n_good,
            "p_good":                 float(p_good),
            "grover_oracle_calls":    grover_calls,
            "classical_random_calls": classical_calls,
            "quantum_speedup_ratio":  round(speedup, 2) if math.isfinite(speedup) else None,
            "mcmc_iterations":        mcmc_iterations,
            "mcmc_vs_grover_ratio":   round(mcmc_vs_grover, 2) if math.isfinite(mcmc_vs_grover) else None,
        })

    # Interpretation text for the highest threshold
    best = results["thresholds"][-1] if results["thresholds"] else {}
    speedup_val = best.get("quantum_speedup_ratio") or 0
    tau_val = best.get("tau", 0.99)
    pg_val = best.get("p_good", 0)

    if speedup_val > 100:
        interp = (
            f"Strong quantum advantage at τ={tau_val:.2f}: Grover's algorithm achieves "
            f"a {speedup_val:.0f}× speedup over classical random search (p_good={pg_val:.2e}). "
            "The rongorongo sign→phoneme search space is hard enough that quantum amplitude "
            "amplification provides a meaningful practical speedup over random sampling."
        )
    elif speedup_val > 10:
        interp = (
            f"Moderate quantum advantage at τ={tau_val:.2f} ({speedup_val:.0f}× speedup). "
            "Grover's algorithm outperforms classical random search. Near-term hardware "
            "constraints may limit practical implementation."
        )
    elif speedup_val > 1:
        interp = (
            f"Limited quantum advantage at τ={tau_val:.2f} ({speedup_val:.1f}× speedup). "
            "High p_good suggests the search space is not as hard as expected — "
            "structured MCMC likely explores good regions far more efficiently than random search."
        )
    else:
        interp = (
            "No quantum advantage detected. The p_good values suggest that random assignments "
            "frequently achieve above-threshold scores, making both Grover and MCMC unnecessary."
        )

    results["interpretation"] = interp
    return results


# ---------------------------------------------------------------------------
# IQAE — Iterative Quantum Amplitude Estimation (classical simulation)
# ---------------------------------------------------------------------------

def _clopper_pearson(
    k: int, n: int, alpha: float = 0.05
) -> tuple[float, float]:
    """Exact Clopper-Pearson confidence interval for a binomial proportion.

    Parameters
    ----------
    k : int   Number of successes.
    n : int   Total trials.
    alpha : float  Significance level (two-sided CI: coverage = 1 - alpha).

    Returns
    -------
    tuple[float, float]   (lower, upper) bounds.
    """
    from scipy.stats import beta as sp_beta  # type: ignore

    lo: float = sp_beta.ppf(alpha / 2.0, k, n - k + 1) if k > 0 else 0.0
    hi: float = sp_beta.ppf(1.0 - alpha / 2.0, k + 1, n - k) if k < n else 1.0
    return lo, hi


def _iqae_estimate(
    corpus_seqs: list[list[str]],
    lms: list[NGramLM],
    signs: list[str],
    phonemes: list[str],
    cutoff: float,
    epsilon: float = 0.05,
    alpha: float = 0.05,
    seed: int = 42,
    max_oracle_calls: int = 200_000,
) -> dict:
    """Adaptive classical amplitude estimation for ``p_good``.

    Uses sequential sampling with exponentially growing batch sizes and
    exact Clopper-Pearson confidence intervals.  Stops as soon as the CI
    half-width falls below ``epsilon``.

    Note: this is *not* quantum IQAE. Actual IQAE (Grinko et al. 2021)
    achieves O(1/ε) oracle calls via quantum amplitude amplification on
    gate-model hardware (Qiskit/IBM Quantum).  Classical adaptive sampling
    still requires O(1/ε²) samples in the worst case; the adaptive stopping
    only helps when ``p_good`` is near 0 or 1 (where the CI tightens faster
    than the normal approximation predicts).  The ``iqae_vs_mc_speedup`` field
    reflects this early-stopping benefit relative to a fixed-p normal
    approximation, not Heisenberg scaling.

    Parameters
    ----------
    corpus_seqs, lms, signs, phonemes :
        Same as :func:`_sample_scores`.
    cutoff : float
        Score threshold; oracle returns 1 when ``score >= cutoff``.
    epsilon : float
        Half-width of the target CI (stop when CI half-width ≤ ε).
    alpha : float
        Significance level for the Clopper-Pearson CI.
    seed : int
        RNG seed for reproducibility.
    max_oracle_calls : int
        Hard cap on total oracle evaluations.

    Returns
    -------
    dict
        ``p_good_estimate``, ``lower_ci``, ``upper_ci``, ``n_oracle_calls``,
        ``mc_oracle_calls_equiv`` (MC calls needed for same ε), and
        ``iqae_vs_mc_speedup``.
    """
    rng = np.random.default_rng(seed)
    phonemes_arr = np.array(phonemes)
    n_ph = len(phonemes)
    n_sg = len(signs)

    total_calls: int = 0
    total_n: int = 0
    total_k: int = 0
    batch_size: int = 32  # initial batch; doubles each round

    lo, hi = 0.0, 1.0

    while total_calls < max_oracle_calls:
        # Sample a batch of random assignments.
        idxs = rng.integers(0, n_ph, size=(batch_size, n_sg))
        for row in idxs:
            for lm in lms:
                lm._lp_cache.clear()
            phone_map = {sign: phonemes_arr[k] for sign, k in zip(signs, row)}
            score = _score_assignment(phone_map, corpus_seqs, lms)
            if math.isfinite(score) and score >= cutoff:
                total_k += 1
            total_n += 1
            total_calls += 1
            if total_calls >= max_oracle_calls:
                break

        if total_n >= 20:  # need enough samples for a stable CI
            lo, hi = _clopper_pearson(total_k, total_n, alpha)
            if (hi - lo) / 2.0 <= epsilon:
                break

        batch_size = min(batch_size * 2, 4096)  # double batch size each round

    lo, hi = _clopper_pearson(total_k, max(total_n, 1), alpha)
    p_est = total_k / max(total_n, 1)

    # Equivalent MC calls for same half-width CI using normal approximation:
    # n_mc = z^2 * p*(1-p) / epsilon^2   (z = 1.96 for 95% CI)
    z = 1.96  # 95% CI
    p_for_mc = max(p_est, 1e-6)
    mc_equiv = math.ceil(z**2 * p_for_mc * (1.0 - p_for_mc) / (epsilon**2))
    speedup = mc_equiv / max(total_calls, 1)

    return {
        "p_good_estimate":      round(p_est, 8),
        "lower_ci":             round(lo, 8),
        "upper_ci":             round(hi, 8),
        "ci_half_width":        round((hi - lo) / 2.0, 8),
        "n_oracle_calls":       total_calls,
        "n_samples_evaluated":  total_n,
        "mc_oracle_calls_equiv": mc_equiv,
        "iqae_vs_mc_speedup":   round(speedup, 2),
        "epsilon":              epsilon,
        "alpha":                alpha,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Estimate p_good and Grover oracle calls for rongorongo decipherment.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--corpus-dir",       type=Path,  default=None, metavar="DIR")
    p.add_argument("--lm-dir",           type=Path,  default=None, metavar="DIR")
    p.add_argument("--n-samples",        type=int,   default=10_000, metavar="N")
    p.add_argument(
        "--thresholds", default="0.90,0.95,0.99", metavar="TAUS",
        help="Comma-separated normalised score thresholds (default: 0.90,0.95,0.99).",
    )
    p.add_argument(
        "--mcmc-iterations", type=int, default=5_000, metavar="N",
        help="Total MCMC iterations from Zone C run (for comparison table).",
    )
    p.add_argument("--output",     type=Path, default=None, metavar="JSON")
    p.add_argument("--seed",       type=int,  default=42)
    p.add_argument(
        "--smoke-test", action="store_true",
        help="Run 100 samples (fast end-to-end wiring check).",
    )
    p.add_argument(
        "--iqae", action="store_true",
        help="Run IQAE-style sequential CI estimation alongside Monte Carlo. "
             "Uses --iqae-epsilon and --iqae-alpha for stopping criteria.",
    )
    p.add_argument(
        "--iqae-epsilon", type=float, default=0.05, metavar="EPS",
        help="IQAE target CI half-width (default: 0.05).",
    )
    p.add_argument(
        "--iqae-alpha", type=float, default=0.05, metavar="ALPHA",
        help="IQAE significance level for Clopper-Pearson CI (default: 0.05).",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    if args.smoke_test:
        args.n_samples = 100
        log.info("Smoke-test mode: 100 samples.")

    corpus_dir = args.corpus_dir
    lm_dir     = args.lm_dir
    output     = args.output

    if corpus_dir is None or lm_dir is None or output is None:
        try:
            from omegaconf import OmegaConf
            cfg = OmegaConf.load(PROJECT_ROOT / "conf" / "config.yaml")
            if corpus_dir is None:
                corpus_dir = PROJECT_ROOT / cfg.paths.corpus_dir
            if lm_dir is None:
                lm_dir = PROJECT_ROOT / "data" / "language_models"
            if output is None:
                output = (
                    PROJECT_ROOT / cfg.paths.outputs_dir / "zone_b" / "pgood_analysis.json"
                )
        except Exception:
            pass

    if corpus_dir is None or not corpus_dir.exists():
        log.error("Corpus directory not found. Pass --corpus-dir.")
        sys.exit(1)
    if lm_dir is None or not lm_dir.exists():
        log.error("LM directory not found. Pass --lm-dir.")
        sys.exit(1)

    thresholds = [float(t.strip()) for t in args.thresholds.split(",")]

    # ── Load ──────────────────────────────────────────────────────────────────
    log.info("Loading corpus from %s …", corpus_dir)
    corpus_seqs = _load_barthel_sequences(corpus_dir)
    if not corpus_seqs:
        log.error("No corpus sequences found in %s.", corpus_dir)
        sys.exit(1)
    n_tokens = sum(len(s) for s in corpus_seqs)
    log.info("  %d tablets, %d total tokens.", len(corpus_seqs), n_tokens)

    lms = _load_lms(lm_dir)
    signs    = sorted({code for seq in corpus_seqs for code in seq})
    phonemes = _phoneme_inventory(lms)

    log.info("  Sign inventory  : %d signs", len(signs))
    log.info("  Phoneme inventory: %d phonemes", len(phonemes))

    if not phonemes:
        log.error("Phoneme inventory is empty — LM vocab may be absent or malformed.")
        sys.exit(1)

    # ── Sample ────────────────────────────────────────────────────────────────
    log.info("Sampling %d random assignments …", args.n_samples)
    t0 = time.perf_counter()
    scores = _sample_scores(corpus_seqs, lms, signs, phonemes, args.n_samples, args.seed)
    elapsed = time.perf_counter() - t0
    log.info("Sampling complete in %.1f s.", elapsed)

    # ── Analyse ───────────────────────────────────────────────────────────────
    results = _analyze(scores, thresholds, args.mcmc_iterations, len(signs), len(phonemes))
    results["sampling_time_seconds"] = round(elapsed, 2)

    # ── Print ─────────────────────────────────────────────────────────────────
    dist = results["score_distribution"]
    print(f"\n{'═' * 66}")
    print(f"  Quantum Hardness Analysis — Rongorongo Decipherment")
    print(f"  {len(corpus_seqs)} tablets · {n_tokens:,} tokens · "
          f"{len(signs)} signs · {len(phonemes)} phonemes")
    print(f"  {args.n_samples:,} random samples in {elapsed:.1f}s")
    print(f"{'═' * 66}")
    print(f"\n  Score distribution (mean per-token log-prob):")
    print(f"    mean = {dist['mean']:.4f}   std = {dist['std']:.4f}")
    print(f"    min  = {dist['min']:.4f}   max = {dist['max']:.4f}")
    print(f"    p50  = {dist['percentiles']['50']:.4f}   "
          f"p90 = {dist['percentiles']['90']:.4f}   "
          f"p99 = {dist['percentiles']['99']:.4f}")
    print()
    hdr = f"  {'τ':>5}  {'p_good':>10}  {'Grover':>10}  {'Classical':>12}  {'Speedup':>9}  {'MCMC/Grover':>12}"
    print(hdr)
    print(f"  {'─'*5}  {'─'*10}  {'─'*10}  {'─'*12}  {'─'*9}  {'─'*12}")
    for t in results["thresholds"]:
        sp  = f"{t['quantum_speedup_ratio']:.1f}×" if t["quantum_speedup_ratio"] else "∞"
        mg  = f"{t['mcmc_vs_grover_ratio']:.1f}×"  if t["mcmc_vs_grover_ratio"] else "∞"
        gc  = f"{t['grover_oracle_calls']:,}"        if t["grover_oracle_calls"] > 0 else "N/A"
        cc  = f"{t['classical_random_calls']:,}"     if t["classical_random_calls"] > 0 else "N/A"
        print(f"  {t['tau']:>5.2f}  {t['p_good']:>10.4e}  {gc:>10}  {cc:>12}  {sp:>9}  {mg:>12}")
    print()
    print(f"  {results['interpretation']}")
    print()

    # ── IQAE (optional) ───────────────────────────────────────────────────────
    if args.iqae:
        log.info("Running IQAE-style estimation (ε=%.3f, α=%.3f) …",
                 args.iqae_epsilon, args.iqae_alpha)
        iqae_results: list[dict] = []
        for t_entry in results["thresholds"]:
            tau = t_entry["tau"]
            cutoff_val = t_entry["score_cutoff"]
            log.info("  IQAE for τ=%.2f  cutoff=%.4f …", tau, cutoff_val)
            t_iqae = time.perf_counter()
            iq = _iqae_estimate(
                corpus_seqs=corpus_seqs,
                lms=lms,
                signs=signs,
                phonemes=phonemes,
                cutoff=cutoff_val,
                epsilon=args.iqae_epsilon,
                alpha=args.iqae_alpha,
                seed=args.seed,
            )
            iq["tau"] = tau
            iq["iqae_time_seconds"] = round(time.perf_counter() - t_iqae, 2)
            iqae_results.append(iq)
            log.info(
                "    τ=%.2f  p_good≈%.4e  CI=[%.4e, %.4e]  "
                "oracle_calls=%d  IQAE vs MC speedup=%.1f×",
                tau, iq["p_good_estimate"], iq["lower_ci"], iq["upper_ci"],
                iq["n_oracle_calls"], iq["iqae_vs_mc_speedup"],
            )
        results["iqae"] = iqae_results
        print(f"\n  IQAE Results (ε={args.iqae_epsilon}, α={args.iqae_alpha}):")
        hdr2 = f"  {'τ':>5}  {'p_good (IQAE)':>14}  {'CI lower':>10}  {'CI upper':>10}  {'Calls':>8}  {'Speedup vs MC':>14}"
        print(hdr2)
        print(f"  {'─'*5}  {'─'*14}  {'─'*10}  {'─'*10}  {'─'*8}  {'─'*14}")
        for iq in iqae_results:
            sp = f"{iq['iqae_vs_mc_speedup']:.1f}×"
            print(
                f"  {iq['tau']:>5.2f}  {iq['p_good_estimate']:>14.4e}  "
                f"{iq['lower_ci']:>10.4e}  {iq['upper_ci']:>10.4e}  "
                f"{iq['n_oracle_calls']:>8,}  {sp:>14}"
            )
        print()

    # ── Save ──────────────────────────────────────────────────────────────────
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        log.info("Results written to %s", output)


if __name__ == "__main__":
    main()
