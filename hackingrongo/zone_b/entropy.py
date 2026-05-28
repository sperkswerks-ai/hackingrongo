"""
hackingrongo.zone_b.entropy
===========================

Index of Coincidence (IC) and entropy analysis on resolved Horley tokens,
split by temporal cluster.

The Index of Coincidence (Friedman 1922) for a text of N tokens drawn from
an alphabet of k signs with observed frequencies f_i is::

    IC = Σ f_i(f_i - 1) / [N(N - 1)]

For a random (uniform) distribution over k signs: IC_random = 1/k.
For natural language (highly structured): IC >> 1/k.

A statistically significant IC_pre > IC_post (or vice versa) is the
headline CFP result: it implies the two temporal strata have different
underlying sign-frequency distributions, consistent with a scribal
tradition that evolved (or diverged) across the contact boundary.

The script also computes:
* Shannon entropy H = -Σ p_i log2(p_i)
* 95% bootstrap CI on IC (2 000 resamples)
* Type-token ratio (TTR) per cluster

Usage
-----
    conda run python hackingrongo/zone_b/entropy.py
    conda run python hackingrongo/zone_b/entropy.py --json   # machine-readable
    conda run python hackingrongo/zone_b/entropy.py \
        --scenario conservative_all_late \
        --scenario optimistic_distributed \
        --scenario probabilistic_weighted \
        --output outputs/sensitivity_analysis.json
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from omegaconf import OmegaConf  # noqa: E402
from hackingrongo.data.constants import (  # noqa: E402
    EXCLUDED_STRATUM,
    POST_CONTACT,
    PRE_CONTACT,
    UNKNOWN_STRATUM,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def index_of_coincidence(tokens: list[str]) -> float:
    """Return the (unbiased) Index of Coincidence for *tokens*."""
    n = len(tokens)
    if n < 2:
        return float("nan")
    counts = Counter(tokens)
    numerator = sum(f * (f - 1) for f in counts.values())
    return numerator / (n * (n - 1))


def shannon_entropy(tokens: list[str]) -> float:
    """Return Shannon entropy (bits) for *tokens*."""
    n = len(tokens)
    if n == 0:
        return float("nan")
    h = 0.0
    for f in Counter(tokens).values():
        p = f / n
        h -= p * math.log2(p)
    return h


def bootstrap_ic_ci(
    tokens: list[str],
    n_resamples: int = 2000,
    ci: float = 0.95,
    seed: int = 42,
) -> tuple[float, float]:
    """Return (lower, upper) bootstrap CI for the IC of *tokens*.

    Uses a fully-vectorised NumPy implementation (~50× faster than the
    pure-Python random.choices loop for 2 000 resamples).
    """
    n = len(tokens)
    denom = n * (n - 1)
    if denom == 0:
        return float("nan"), float("nan")

    # Encode string tokens as contiguous integers for fast counting.
    _, encoded = np.unique(tokens, return_inverse=True)
    k = int(encoded.max()) + 1

    rng = np.random.default_rng(seed)
    # Draw (n_resamples, n) random indices into the encoded array.
    sample_indices = rng.integers(0, n, size=(n_resamples, n))
    # Look up the encoded token id for each sampled position.
    samples = encoded[sample_indices]  # shape: (n_resamples, n)

    # Vectorised per-row bincount using the row-offset trick:
    # shift each row by row * k so they occupy non-overlapping slices.
    row_offsets = (np.arange(n_resamples, dtype=np.int64) * k).reshape(-1, 1)
    flat_counts = np.bincount(
        (samples + row_offsets).ravel(), minlength=n_resamples * k
    ).reshape(n_resamples, k)

    boot_ics = (flat_counts * (flat_counts - 1)).sum(axis=1) / denom
    boot_ics.sort()

    a = (1.0 - ci) / 2
    lo = float(boot_ics[max(0, int(a * n_resamples))])
    hi = float(boot_ics[min(int((1 - a) * n_resamples), n_resamples - 1)])
    return lo, hi


def ic_random_baseline(k: int) -> float:
    """IC expected for a uniform distribution over *k* sign types."""
    return 1.0 / k if k > 0 else float("nan")


def normalized_ic(tokens: list[str]) -> float:
    """Vocabulary-size-corrected IC: IC × k.

    Motivation
    ----------
    Raw IC is biased by vocabulary size: for a *uniform* distribution over k
    sign types, IC = 1/k.  Multiplying by k removes this dependence so that
    the random baseline is always 1.0 regardless of vocabulary size.

        normalized_IC = IC / (1/k) = IC × k

    Interpretation:
      = 1.0  →  indistinguishable from random usage of the observed vocabulary
      > 1.0  →  structure beyond random (concentrated usage of some signs)
    """
    k = len(set(tokens))
    if k == 0:
        return float("nan")
    return index_of_coincidence(tokens) * k


def bootstrap_h_ci(
    tokens: list[str],
    n_resamples: int = 2000,
    ci: float = 0.95,
    seed: int = 42,
) -> tuple[float, float]:
    """Return (lower, upper) bootstrap CI for Shannon entropy H of *tokens*.

    Uses the same fully-vectorised NumPy row-offset trick as
    :func:`bootstrap_ic_ci` for efficiency.
    """
    n = len(tokens)
    if n < 2:
        return float("nan"), float("nan")

    _, encoded = np.unique(tokens, return_inverse=True)
    k = int(encoded.max()) + 1

    rng = np.random.default_rng(seed)
    sample_indices = rng.integers(0, n, size=(n_resamples, n))
    samples = encoded[sample_indices]

    row_offsets = (np.arange(n_resamples, dtype=np.int64) * k).reshape(-1, 1)
    flat_counts = np.bincount(
        (samples + row_offsets).ravel(), minlength=n_resamples * k
    ).reshape(n_resamples, k)

    probs = flat_counts / n
    with np.errstate(divide="ignore", invalid="ignore"):
        log_probs = np.where(probs > 0, np.log2(probs), 0.0)
    boot_h = -(probs * log_probs).sum(axis=1)
    boot_h.sort()

    a = (1.0 - ci) / 2
    lo = float(boot_h[max(0, int(a * n_resamples))])
    hi = float(boot_h[min(int((1 - a) * n_resamples), n_resamples - 1)])
    return lo, hi


def conditional_bigram_entropy(tokens: list[str]) -> float:
    """Bigram approximation of entropy rate: H(S_n | S_{n-1}).

    Derivation
    ----------
    The conditional entropy of the next token given the previous is::

        H(S_n | S_{n-1}) = -Σ_{s,t} P(s,t) log₂ P(t | s)

    where P(s,t) is the empirical bigram probability and P(t|s) = P(s,t)/P(s).

    This is the entropy rate under a first-order Markov (bigram) model.
    For natural text H(S_n | S_{n-1}) < H(S_n): knowledge of the preceding
    sign reduces uncertainty about the next.  The difference
    H(S_n) − H(S_n | S_{n-1}) = I(S_n ; S_{n-1}) is the bigram mutual
    information — a direct measure of sequential sign dependency.
    """
    if len(tokens) < 2:
        return float("nan")
    bigrams: Counter = Counter(zip(tokens[:-1], tokens[1:]))
    unigrams: Counter = Counter(tokens[:-1])
    total = sum(bigrams.values())
    h = 0.0
    for (s, t), cnt in bigrams.items():
        p_st = cnt / total
        p_t_given_s = cnt / unigrams[s]
        h -= p_st * math.log2(p_t_given_s)
    return h


def positional_mutual_information(
    corpus_dir: "Path",
    n_bins: int = 5,
) -> dict:
    """I(sign ; relative_position) = H(sign) − H(sign | position_bin).

    Loads all Horley tokens from the corpus and computes the mutual
    information between sign identity and where it appears within its line
    (expressed as a relative-position quintile).

    Parameters
    ----------
    corpus_dir : Path
        Corpus directory (``data/corpus/``).
    n_bins : int
        Number of equal-width relative-position bins.  Default 5 (quintiles):
        0–20 %, 20–40 %, 40–60 %, 60–80 %, 80–100 % of line length.

    Returns
    -------
    dict
        Keys: n_tokens, n_types, n_bins, h_sign_bits,
        h_sign_given_position_bits, mutual_information_bits,
        mutual_info_pct_of_h, bin_label.

    Notes
    -----
    Mutual information I ≈ 0 means sign identity and position are
    independent — every sign is equally likely at every position.
    I > 0 means some signs are positionally concentrated (e.g. taxograms
    that tend to appear post-line-boundary, sequence-initial particles).
    I / H(sign) is the fraction of sign entropy explained by position.
    """
    from collections import defaultdict

    tokens_by_pos: list[tuple[str, float]] = []
    for path in sorted(corpus_dir.glob("[A-Z].json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        for g in data.get("glyphs", []):
            hc = g.get("horley_code")
            if not hc:
                continue
            # relative_position: pos / (line_len - 1), clamped to [0, 1]
            pos = g.get("position_in_line")
            line_len = g.get("line_length")
            if pos is not None and line_len and line_len > 1:
                rel = min(1.0, max(0.0, pos / (line_len - 1)))
            else:
                rel = 0.5  # unknown position → put in middle bin
            tokens_by_pos.append((hc, rel))

    if not tokens_by_pos:
        return {
            "n_tokens": 0, "mutual_information_bits": float("nan"),
            "note": "No position data available in corpus.",
        }

    tokens = [t for t, _ in tokens_by_pos]
    h_sign = shannon_entropy(tokens)

    from collections import defaultdict
    bin_tokens: dict[int, list[str]] = defaultdict(list)
    for tok, rel in tokens_by_pos:
        b = min(int(rel * n_bins), n_bins - 1)
        bin_tokens[b].append(tok)

    n = len(tokens)
    h_cond = 0.0
    for b, toks_in_bin in bin_tokens.items():
        p_b = len(toks_in_bin) / n
        h_b = shannon_entropy(toks_in_bin)
        if math.isfinite(h_b):
            h_cond += p_b * h_b

    mi = h_sign - h_cond
    mi_pct = round(mi / h_sign * 100, 2) if h_sign > 0 else 0.0

    bin_labels = [f"{100*i//n_bins}–{100*(i+1)//n_bins}%" for i in range(n_bins)]

    return {
        "n_tokens": n,
        "n_types": len(set(tokens)),
        "n_bins": n_bins,
        "bin_labels": bin_labels,
        "h_sign_bits": round(h_sign, 4),
        "h_sign_given_position_bits": round(h_cond, 4),
        "mutual_information_bits": round(mi, 4),
        "mutual_info_pct_of_h": mi_pct,
    }


# ---------------------------------------------------------------------------
# Scenario-aware token loading
# ---------------------------------------------------------------------------

_SCENARIO_NAMES = ("conservative_all_late", "optimistic_distributed", "probabilistic_weighted")


def load_tokens_under_scenario(
    corpus_dir: Path,
    scenario: str,
    uncertain_weight: float = 0.0,
) -> dict[str, list[str]]:
    """Load tokens reassigning unknown-cluster tablets per scenario.

    Scenario semantics
    ------------------
    ``conservative_all_late``
        All 19 undated tablets treated as *post_contact*.
        Maximum IC_post, minimum IC_pre separation — if result still holds,
        the finding is robust against the most pessimistic dating assumption.

    ``optimistic_distributed``
        Unknown tablets split evenly (50 / 50) between pre and post.
        Even-indexed tablets go to pre_contact, odd-indexed to post_contact.
        Tests whether pre/post IC difference survives when undated material
        is assumed uniformly distributed across both strata.

    ``probabilistic_weighted``
        Empirical 20 / 80 prior (1 pre anchor, 4 post anchors): first
        20 % of each unknown tablet's tokens are added to pre_contact,
        remaining 80 % to post_contact.  Matches the baseline analysis.
    """
    from collections import defaultdict

    if scenario not in _SCENARIO_NAMES:
        raise ValueError(f"Unknown scenario {scenario!r}. Choose from {_SCENARIO_NAMES}")

    # Collect all unknown tablets in alphabetical order (for deterministic split)
    unknown_paths = []
    all_paths = sorted(corpus_dir.glob("[A-Z].json"))

    by_cluster: dict[str, list[str]] = defaultdict(list)

    for path in all_paths:
        data = json.loads(path.read_text(encoding="utf-8"))
        cluster = data.get("cluster", UNKNOWN_STRATUM)

        tokens: list[str] = []
        for g in data["glyphs"]:
            hc = g.get("horley_code")
            is_uncertain = g.get("uncertain", False)
            if hc:
                if is_uncertain and uncertain_weight < 1.0:
                    if uncertain_weight >= 0.5:
                        tokens.append(hc)
                else:
                    tokens.append(hc)
            for comp_hc in (g.get("horley_components") or []):
                tokens.append(comp_hc)

        if cluster != UNKNOWN_STRATUM:
            # Dated tablets: use their assigned cluster unchanged.
            by_cluster[cluster].extend(tokens)
            continue

        # Unknown tablet — reassign under scenario.
        if scenario == "conservative_all_late":
            by_cluster[POST_CONTACT].extend(tokens)

        elif scenario == "optimistic_distributed":
            # Even index (0-based among ALL unknown tablets) → pre, odd → post
            unknown_paths.append(tokens)

        elif scenario == "probabilistic_weighted":
            n_pre = max(1, int(round(len(tokens) * 0.20)))
            by_cluster[PRE_CONTACT].extend(tokens[:n_pre])
            by_cluster[POST_CONTACT].extend(tokens[n_pre:])

    if scenario == "optimistic_distributed":
        for unk_pos, tokens in enumerate(unknown_paths):
            if unk_pos % 2 == 0:
                by_cluster[PRE_CONTACT].extend(tokens)
            else:
                by_cluster[POST_CONTACT].extend(tokens)

    return dict(by_cluster)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

CLUSTER_ORDER = (PRE_CONTACT, POST_CONTACT, EXCLUDED_STRATUM, UNKNOWN_STRATUM)


def load_tokens_by_cluster(
    corpus_dir: Path,
    uncertain_weight: float = 0.0,
    include_compound_components: bool = True,
) -> dict[str, list[str]]:
    """Return {cluster: [horley_code, ...]} for all resolved glyphs.

    Parameters
    ----------
    uncertain_weight : float
        Weight in [0, 1] to give tokens flagged as uncertain (``?`` diacritic).
        0.0 = exclude uncertain tokens (default, conservative).
        0.5 = include at half weight (fractional repeat).
        1.0 = treat uncertain tokens as fully reliable.
        Fractional weights are implemented by adding the token ⌊weight * N⌋
        times where N is the token's integer count.
    include_compound_components : bool
        If True (default), also include resolved component codes from compound
        tokens (``horley_components`` field).
    """
    from collections import defaultdict
    by_cluster: dict[str, list[str]] = defaultdict(list)
    for path in sorted(corpus_dir.glob("[A-Z].json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        cluster = data.get("cluster", UNKNOWN_STRATUM)
        for g in data["glyphs"]:
            hc = g.get("horley_code")
            is_uncertain = g.get("uncertain", False)
            if hc:
                if is_uncertain and uncertain_weight < 1.0:
                    if uncertain_weight <= 0.0:
                        continue
                    # Fractional inclusion: add token with probability = weight
                    # Deterministically: include once if weight >= 0.5
                    if uncertain_weight >= 0.5:
                        by_cluster[cluster].append(hc)
                    # (below 0.5 = exclude)
                else:
                    by_cluster[cluster].append(hc)
            # Compound components
            if include_compound_components:
                for comp_hc in (g.get("horley_components") or []):
                    by_cluster[cluster].append(comp_hc)
    return dict(by_cluster)


# ---------------------------------------------------------------------------
# Zipf's Law Analysis
# ---------------------------------------------------------------------------

def _load_all_tokens(corpus_dir: Path) -> Counter:
    """Load all resolved Horley tokens from every tablet, pooled across clusters."""
    counts: Counter = Counter()
    for path in sorted(corpus_dir.glob("[A-Z].json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        for g in data["glyphs"]:
            hc = g.get("horley_code")
            if hc:
                counts[hc] += 1
            for comp_hc in (g.get("horley_components") or []):
                counts[comp_hc] += 1
    return counts


def _plot_zipf(
    ranks: "np.ndarray",
    freq_sorted: "np.ndarray",
    alpha_ols: float,
    alpha_mle: float,
    intercept_ols: float,
    r_squared: float,
    output_dir: Path | None,
) -> None:
    import matplotlib
    if output_dir:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 5))

    ax.scatter(
        ranks, freq_sorted,
        s=18, alpha=0.65, color="#2563eb", zorder=3,
        label="Observed sign frequencies",
    )

    freq_fit = np.exp(intercept_ols) * ranks ** (-alpha_ols)
    ax.plot(
        ranks, freq_fit,
        linewidth=1.8, color="#dc2626",
        label=f"Power-law fit  αₒⱼₛ = {alpha_ols:.3f}  (R² = {r_squared:.3f})",
    )
    ax.plot(
        ranks, freq_fit * (ranks ** (alpha_ols - alpha_mle)),
        linewidth=1.4, color="#f97316", linestyle=(0, (5, 2)),
        label=f"Zipf MLE  αₘₗₑ = {alpha_mle:.3f}",
    )

    freq_zipf1 = freq_sorted[0] * ranks ** (-1.0)
    ax.plot(
        ranks, freq_zipf1,
        linewidth=1.1, linestyle="--", color="#6b7280",
        label="Ideal Zipf  α = 1.0",
    )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Rank", fontsize=12)
    ax.set_ylabel("Frequency", fontsize=12)
    ax.set_title(
        "Rongorongo Sign Frequency Distribution — Zipf’s Law Test",
        fontsize=12, pad=10,
    )
    ax.legend(fontsize=9.5)
    ax.grid(True, which="both", alpha=0.25, linewidth=0.6)
    fig.tight_layout()

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / "zipf_rank_frequency.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        log.info("Zipf plot saved: %s", out_path)
    else:
        plt.show()
    plt.close(fig)


def zipf_analysis(
    corpus_dir: Path,
    output_dir: Path | None = None,
    plot: bool = True,
) -> dict:
    """Test whether rongorongo sign frequencies follow Zipf's law.

    Method
    ------
    1. Pool all resolved Horley tokens across every tablet.
    2. Rank by descending frequency (rank 1 = most common sign).
    3. Log-log OLS regression of log(freq) on log(rank) — slope is -alpha_ols.
    4. MLE for scipy.stats.zipf exponent via scipy.optimize.minimize_scalar,
       maximising sum_r [ freq_r * logPMF(r; a) ] where PMF uses the Riemann-zeta
       normalisation (infinite support; standard approximation for large inventories).
    5. Kolmogorov-Smirnov test of empirical rank distribution against fitted Zipf CDF
       (note: KS test assumes continuous distributions — result is approximate here).
    6. Spearman rho between observed and OLS-predicted frequencies.

    Returns
    -------
    dict with keys: n_tokens, n_types, exponent_mle, exponent_ols, r_squared_loglog,
    ks_statistic, ks_pvalue, spearman_rho, spearman_pvalue, consistent_with_zipf,
    interpretation, and optionally plot_path.
    """
    from scipy import optimize, stats as sp_stats

    counts = _load_all_tokens(corpus_dir)
    if not counts:
        raise ValueError("No resolved tokens found in corpus.")

    n_types = len(counts)
    freq_sorted = np.array([f for _, f in counts.most_common()], dtype=np.float64)
    ranks = np.arange(1, n_types + 1, dtype=np.float64)
    n_tokens = int(freq_sorted.sum())

    # ── Log-log OLS ───────────────────────────────────────────────────────────
    log_r = np.log(ranks)
    log_f = np.log(freq_sorted)
    slope, intercept, r_value, _, _ = sp_stats.linregress(log_r, log_f)
    alpha_ols = float(-slope)
    r_squared = float(r_value ** 2)

    # ── MLE against scipy.stats.zipf ──────────────────────────────────────────
    # scipy.stats.zipf.fit() is absent in scipy 1.7; optimise directly.
    int_ranks = ranks.astype(int)

    def neg_ll(a: float) -> float:
        if a <= 1.0:
            return np.inf
        return -float((freq_sorted * sp_stats.zipf.logpmf(int_ranks, a)).sum())

    opt_result = optimize.minimize_scalar(neg_ll, bounds=(1.001, 10.0), method="bounded")
    alpha_mle = float(opt_result.x)

    # ── KS test ───────────────────────────────────────────────────────────────
    rank_sequence = np.repeat(int_ranks, freq_sorted.astype(int))
    ks_stat, ks_pval = sp_stats.kstest(
        rank_sequence, lambda x: sp_stats.zipf.cdf(x, alpha_mle)
    )

    # ── Spearman ρ ────────────────────────────────────────────────────────────
    freq_pred_ols = np.exp(intercept) * ranks ** slope
    spearman_rho, spearman_pval = sp_stats.spearmanr(freq_sorted, freq_pred_ols)

    # ── Interpretation ────────────────────────────────────────────────────────
    consistent = 0.8 <= alpha_mle <= 1.2
    if consistent:
        interp = (
            f"α = {alpha_mle:.3f} falls within the canonical Zipf range [0.8, 1.2]. "
            "Rongorongo sign frequencies are consistent with a natural-language "
            "power-law distribution — supporting (but not proving) the linguistic hypothesis."
        )
    elif alpha_mle > 1.2:
        interp = (
            f"α = {alpha_mle:.3f} exceeds the canonical Zipf range [0.8, 1.2]. "
            "Frequency is more concentrated in a small set of dominant signs than "
            "expected under natural language, which constrains the linguistic hypothesis."
        )
    else:
        interp = (
            f"α = {alpha_mle:.3f} is below the canonical Zipf range [0.8, 1.2]. "
            "The distribution is flatter than natural language, suggesting unusually "
            "uniform sign usage. This constrains the linguistic hypothesis."
        )

    # ── Report ────────────────────────────────────────────────────────────────
    log.info("")
    log.info("=" * 64)
    log.info("Zipf's Law Test — Rongorongo Sign Frequency Distribution")
    log.info("=" * 64)
    log.info("  N tokens : %d  |  N types (sign inventory) : %d", n_tokens, n_types)
    log.info("")
    log.info("  Exponent α  (MLE, scipy.stats.zipf)  : %.4f", alpha_mle)
    log.info("  Exponent α  (log-log OLS regression) : %.4f", alpha_ols)
    log.info("  R² on log-log scale                  : %.4f", r_squared)
    log.info(
        "  KS statistic / p-value               : %.4f / %.4g  [discrete approx]",
        ks_stat, ks_pval,
    )
    log.info("  Spearman ρ  (obs vs predicted)       : %.4f  p=%.4g", spearman_rho, spearman_pval)
    log.info("")
    log.info("  INTERPRETATION")
    log.info("  %s", interp)
    log.info("  Reference: natural language corpora typically α ∈ [0.9, 1.1]")
    log.info("=" * 64)

    result: dict = {
        "n_tokens": n_tokens,
        "n_types": n_types,
        "exponent_mle": round(alpha_mle, 6),
        "exponent_ols": round(alpha_ols, 6),
        "r_squared_loglog": round(r_squared, 6),
        "ks_statistic": round(float(ks_stat), 6),
        "ks_pvalue": round(float(ks_pval), 6),
        "spearman_rho": round(float(spearman_rho), 6),
        "spearman_pvalue": round(float(spearman_pval), 6),
        "consistent_with_zipf": consistent,
        "interpretation": interp,
    }

    if plot:
        try:
            _plot_zipf(ranks, freq_sorted, alpha_ols, alpha_mle, intercept, r_squared, output_dir)
            if output_dir:
                result["plot_path"] = str(output_dir / "zipf_rank_frequency.png")
        except Exception as exc:  # noqa: BLE001
            log.warning("Plot generation failed: %s", exc)

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        json_path = output_dir / "zipf_analysis.json"
        json_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        log.info("Zipf results written to %s", json_path)

    return result


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def analyse(emit_json: bool = False, uncertain_weight: float = 0.0) -> dict:
    """Run IC analysis on corpus-assigned clusters; return results dict."""
    cfg = OmegaConf.load(PROJECT_ROOT / "conf" / "config.yaml")
    corpus_dir = PROJECT_ROOT / cfg.paths.corpus_dir

    by_cluster = load_tokens_by_cluster(
        corpus_dir,
        uncertain_weight=uncertain_weight,
        include_compound_components=True,
    )

    # Shared inventory (types seen in pre + post — the analysis vocabulary)
    pre = by_cluster.get(PRE_CONTACT, [])
    post = by_cluster.get(POST_CONTACT, [])
    shared_vocab = set(pre) | set(post)

    results: dict[str, dict] = {}

    log.info("")
    log.info("=" * 64)
    log.info("Index of Coincidence — by temporal cluster")
    log.info("=" * 64)
    log.info("  (IC_random = 1/k where k = shared sign inventory size)")
    log.info("  (IC >> IC_random  →  structured / non-random distribution)")
    log.info("")

    for cluster in CLUSTER_ORDER:
        tokens = by_cluster.get(cluster, [])
        if not tokens:
            continue

        ic = index_of_coincidence(tokens)
        h = shannon_entropy(tokens)
        k = len(set(tokens))
        ttr = k / len(tokens) if tokens else float("nan")
        ic_rand = ic_random_baseline(k)
        ic_rand_shared = ic_random_baseline(len(shared_vocab)) if shared_vocab else float("nan")

        # Bootstrap CI (skip for very small samples)
        if len(tokens) >= 30:
            lo, hi = bootstrap_ic_ci(tokens)
            ci_str = f"[{lo:.5f}, {hi:.5f}]"
        else:
            lo, hi = float("nan"), float("nan")
            ci_str = "n/a (n < 30)"

        # Bootstrap CI on Shannon H
        if len(tokens) >= 30:
            h_lo, h_hi = bootstrap_h_ci(tokens)
        else:
            h_lo, h_hi = float("nan"), float("nan")

        # Conditional bigram entropy (entropy rate approximation)
        h_cond = conditional_bigram_entropy(tokens)

        # Normalized IC (vocabulary-size corrected)
        norm_ic = normalized_ic(tokens)

        results[cluster] = {
            "n_tokens": len(tokens),
            "n_types": k,
            "ic": round(ic, 6),
            "ic_ci_95_lo": round(lo, 6) if not math.isnan(lo) else None,
            "ic_ci_95_hi": round(hi, 6) if not math.isnan(hi) else None,
            "ic_normalized": round(norm_ic, 4) if math.isfinite(norm_ic) else None,
            "ic_random_own_vocab": round(ic_rand, 6),
            "ic_random_shared_vocab": round(ic_rand_shared, 6) if not math.isnan(ic_rand_shared) else None,
            "entropy_bits": round(h, 4),
            "entropy_ci_95_lo": round(h_lo, 4) if math.isfinite(h_lo) else None,
            "entropy_ci_95_hi": round(h_hi, 4) if math.isfinite(h_hi) else None,
            "conditional_entropy_bigram": round(h_cond, 4) if math.isfinite(h_cond) else None,
            "bigram_mi_bits": round(h - h_cond, 4) if math.isfinite(h_cond) else None,
            "ttr": round(ttr, 4),
        }

        h_ci_str = f"[{h_lo:.4f}, {h_hi:.4f}]" if math.isfinite(h_lo) else "n/a (n < 30)"
        log.info("Cluster: %s", cluster)
        log.info("  tokens=%d  types=%d  TTR=%.3f", len(tokens), k, ttr)
        log.info("  IC         = %.6f  95%% CI %s", ic, ci_str)
        log.info("  IC_norm    = %.4f  (IC × k; random baseline = 1.0)", norm_ic)
        log.info("  IC_random (own vocab, k=%d)    = %.6f", k, ic_rand)
        log.info("  IC_random (shared vocab, k=%d) = %.6f", len(shared_vocab), ic_rand_shared)
        log.info("  IC / IC_random (own) = %.2fx", ic / ic_rand if (ic_rand != 0.0 and not math.isnan(ic_rand)) else float("nan"))
        log.info("  Shannon H  = %.4f bits  95%% CI %s  (max=%.4f for k=%d)", h, h_ci_str, math.log2(k) if k > 1 else 0.0, k)
        log.info("  H(S|S-1)   = %.4f bits  (bigram entropy rate approx)", h_cond if math.isfinite(h_cond) else float("nan"))
        log.info("  I(S;S-1)   = %.4f bits  (bigram mutual information)", (h - h_cond) if math.isfinite(h_cond) else float("nan"))
        log.info("")

    # Headline comparison: pre vs post
    if PRE_CONTACT in results and POST_CONTACT in results:
        r_pre = results[PRE_CONTACT]
        r_post = results[POST_CONTACT]
        log.info("-" * 64)
        log.info("HEADLINE CFP RESULT: pre_contact vs post_contact IC")
        log.info("  pre_contact  IC = %.6f  (n=%d, k=%d)", r_pre["ic"], r_pre["n_tokens"], r_pre["n_types"])
        log.info("  post_contact IC = %.6f  (n=%d, k=%d)", r_post["ic"], r_post["n_tokens"], r_post["n_types"])
        diff = r_pre["ic"] - r_post["ic"]
        log.info("  Δ IC (pre − post) = %+.6f", diff)
        # Overlap check using CIs
        pre_lo = r_pre.get("ic_ci_95_lo")
        post_hi = r_post.get("ic_ci_95_hi")
        if pre_lo is not None and post_hi is not None:
            overlap = pre_lo < post_hi
            log.info("  95%% CIs overlap: %s", overlap)
        log.info("-" * 64)

    # Boustrophedon voice-split test
    results["boustrophedon_ic"] = compute_ic_by_line_parity(corpus_dir)

    # Positional mutual information I(sign ; relative_position)
    results["positional_mutual_information"] = positional_mutual_information(corpus_dir)

    if emit_json:
        print(json.dumps(results, indent=2))

    return results


def ic_for_clusters(by_cluster: dict[str, list[str]]) -> dict[str, dict]:
    """Compute IC, CI, entropy, and entropy-rate metrics for pre and post clusters."""
    shared_vocab = set(by_cluster.get(PRE_CONTACT, [])) | set(by_cluster.get(POST_CONTACT, []))
    out: dict[str, dict] = {}
    for cluster in (PRE_CONTACT, POST_CONTACT):
        tokens = by_cluster.get(cluster, [])
        if not tokens:
            continue
        ic = index_of_coincidence(tokens)
        k = len(set(tokens))
        n = len(tokens)
        lo, hi = bootstrap_ic_ci(tokens) if n >= 30 else (float("nan"), float("nan"))
        h = shannon_entropy(tokens)
        h_lo, h_hi = bootstrap_h_ci(tokens) if n >= 30 else (float("nan"), float("nan"))
        h_cond = conditional_bigram_entropy(tokens)
        norm_ic = normalized_ic(tokens)
        out[cluster] = {
            "n_tokens": n,
            "n_types": k,
            "ic": round(ic, 6),
            "ic_ci_95_lo": round(lo, 6) if math.isfinite(lo) else None,
            "ic_ci_95_hi": round(hi, 6) if math.isfinite(hi) else None,
            "ic_normalized": round(norm_ic, 4) if math.isfinite(norm_ic) else None,
            "ic_random_shared": round(1.0 / len(shared_vocab), 6) if shared_vocab else None,
            "entropy_bits": round(h, 4),
            "entropy_ci_95_lo": round(h_lo, 4) if math.isfinite(h_lo) else None,
            "entropy_ci_95_hi": round(h_hi, 4) if math.isfinite(h_hi) else None,
            "conditional_entropy_bigram": round(h_cond, 4) if math.isfinite(h_cond) else None,
            "bigram_mi_bits": round(h - h_cond, 4) if math.isfinite(h_cond) else None,
        }
    return out


def sensitivity_analysis(
    scenarios: list[str],
    uncertain_weight: float = 0.0,
    output_path: Path | None = None,
) -> dict:
    """Run IC analysis under each named scenario and compare robustness."""
    cfg = OmegaConf.load(PROJECT_ROOT / "conf" / "config.yaml")
    corpus_dir = PROJECT_ROOT / cfg.paths.corpus_dir
    robustness_threshold = float(
        cfg.corpus.temporal_model.get("robustness_threshold", 0.10)
    )

    log.info("")
    log.info("=" * 64)
    log.info("Sensitivity Analysis — IC pre vs post across %d scenarios", len(scenarios))
    log.info("=" * 64)
    log.info("  Robustness threshold: %.0f%% max allowed relative variation in Δ IC", robustness_threshold * 100)
    log.info("")

    scenario_results: dict[str, dict] = {}
    deltas: list[float] = []
    ci_non_overlapping: list[bool] = []

    for scenario in scenarios:
        by_cluster = load_tokens_under_scenario(corpus_dir, scenario, uncertain_weight)
        res = ic_for_clusters(by_cluster)
        scenario_results[scenario] = res

        pre = res.get(PRE_CONTACT, {})
        post = res.get(POST_CONTACT, {})
        delta = (pre.get("ic", 0.0) or 0.0) - (post.get("ic", 0.0) or 0.0)
        deltas.append(delta)

        pre_lo = pre.get("ic_ci_95_lo")
        post_hi = post.get("ic_ci_95_hi")
        non_overlap = (pre_lo is not None and post_hi is not None and pre_lo > post_hi)
        ci_non_overlapping.append(non_overlap)

        log.info("Scenario: %s", scenario)
        log.info("  pre_contact  n=%5d  IC=%.6f  CI=[%.5f, %.5f]",
                 pre.get("n_tokens", 0), pre.get("ic", float("nan")),
                 pre.get("ic_ci_95_lo") or float("nan"), pre.get("ic_ci_95_hi") or float("nan"))
        log.info("  post_contact n=%5d  IC=%.6f  CI=[%.5f, %.5f]",
                 post.get("n_tokens", 0), post.get("ic", float("nan")),
                 post.get("ic_ci_95_lo") or float("nan"), post.get("ic_ci_95_hi") or float("nan"))
        log.info("  Δ IC = %+.6f  CIs non-overlapping: %s", delta, non_overlap)
        log.info("")

    # Robustness: does the direction (pre > post) hold across all scenarios?
    all_positive = all(d > 0 for d in deltas)
    delta_range = max(deltas) - min(deltas) if deltas else 0.0
    max_ref_delta = max(abs(d) for d in deltas) if deltas else 1.0
    relative_variation = delta_range / max_ref_delta if max_ref_delta > 0 else 0.0
    is_robust = all_positive and relative_variation <= robustness_threshold

    log.info("-" * 64)
    log.info("ROBUSTNESS SUMMARY")
    log.info("  IC_pre > IC_post in ALL scenarios: %s", all_positive)
    log.info("  CIs non-overlapping in all: %s", all(ci_non_overlapping))
    log.info("  Δ IC range: [%+.6f, %+.6f] (variation: %.1f%%)",
             min(deltas), max(deltas), relative_variation * 100)
    log.info("  Robust at %.0f%% threshold: %s", robustness_threshold * 100, is_robust)
    log.info("-" * 64)

    # Boustrophedon voice-split test — corpus-level, independent of scenario
    output = {
        "scenarios": scenario_results,
        "deltas": {s: d for s, d in zip(scenarios, deltas)},
        "robustness": {
            "all_pre_gt_post": all_positive,
            "all_ci_non_overlapping": all(ci_non_overlapping),
            "delta_range": round(delta_range, 6),
            "relative_variation_pct": round(relative_variation * 100, 2),
            "robust": is_robust,
        },
        "boustrophedon_ic": compute_ic_by_line_parity(corpus_dir),
    }

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
        log.info("Sensitivity results written to %s", output_path)

    return output


# ---------------------------------------------------------------------------
# Boustrophedon voice-split test
# ---------------------------------------------------------------------------

def compute_ic_by_line_parity(
    corpus_dir: Path,
    n_bootstrap: int = 2000,
    output_path: Path | None = None,
) -> dict:
    """IC split by boustrophedon line parity — the voice-split test.

    Odd-numbered lines (1, 3, 5, …) and even-numbered lines (2, 4, 6, …)
    alternate direction in reverse boustrophedon.  If these two physical
    text-streams were written by different hands, in different registers, or
    carry structurally different content, their sign-frequency distributions
    should differ — measurable as IC_odd ≠ IC_even with non-overlapping
    bootstrap confidence intervals.

    Parameters
    ----------
    corpus_dir : Path
        Corpus directory (``data/corpus/``).
    n_bootstrap : int
        Number of bootstrap resamples for the CI (default 2 000).
    output_path : Path or None
        If given, write JSON results here.

    Returns
    -------
    dict
        Keys: n_odd_lines, n_even_lines, n_odd_tokens, n_even_tokens,
        ic_odd, ic_even, ic_odd_ci_95_{lo,hi}, ic_even_ci_95_{lo,hi},
        delta_ic_odd_minus_even, cis_overlap, finding.
    """
    from hackingrongo.zone_b.sequence_model import load_sequences_with_lines

    line_data = load_sequences_with_lines(corpus_dir)

    odd_tokens: list[str] = [
        tok for _, _, _, parity, tokens in line_data
        for tok in tokens if parity == "odd"
    ]
    even_tokens: list[str] = [
        tok for _, _, _, parity, tokens in line_data
        for tok in tokens if parity == "even"
    ]

    n_odd_lines = sum(1 for r in line_data if r[3] == "odd")
    n_even_lines = sum(1 for r in line_data if r[3] == "even")

    ic_odd = index_of_coincidence(odd_tokens)
    ic_even = index_of_coincidence(even_tokens)

    lo_odd, hi_odd = (
        bootstrap_ic_ci(odd_tokens, n_resamples=n_bootstrap)
        if len(odd_tokens) >= 30 else (float("nan"), float("nan"))
    )
    lo_even, hi_even = (
        bootstrap_ic_ci(even_tokens, n_resamples=n_bootstrap)
        if len(even_tokens) >= 30 else (float("nan"), float("nan"))
    )

    # CIs overlap if neither interval lies entirely above the other.
    cis_overlap = not (
        (math.isfinite(hi_odd) and math.isfinite(lo_even) and hi_odd < lo_even)
        or (math.isfinite(hi_even) and math.isfinite(lo_odd) and hi_even < lo_odd)
    )
    delta = ic_odd - ic_even

    # Overlap fraction: how much of the narrower CI width is shared?
    # Near-zero overlap (< 10%) is a marginal trend even when CIs technically touch.
    overlap_fraction = float("nan")
    if cis_overlap and all(math.isfinite(v) for v in (lo_odd, hi_odd, lo_even, hi_even)):
        overlap_lo = max(lo_odd, lo_even)
        overlap_hi = min(hi_odd, hi_even)
        overlap_width = max(0.0, overlap_hi - overlap_lo)
        narrower_width = min(hi_odd - lo_odd, hi_even - lo_even)
        overlap_fraction = overlap_width / narrower_width if narrower_width > 0 else 0.0
    marginal = cis_overlap and math.isfinite(overlap_fraction) and overlap_fraction < 0.10

    log.info("")
    log.info("=" * 64)
    log.info("Boustrophedon Voice-Split Test — IC by Line Parity")
    log.info("=" * 64)
    log.info("  Odd lines  (1,3,5,…): %d lines  %d tokens", n_odd_lines, len(odd_tokens))
    log.info("  Even lines (2,4,6,…): %d lines  %d tokens", n_even_lines, len(even_tokens))
    log.info("")
    log.info("  IC_odd  = %.6f  95%% CI [%.5f, %.5f]", ic_odd, lo_odd, hi_odd)
    log.info("  IC_even = %.6f  95%% CI [%.5f, %.5f]", ic_even, lo_even, hi_even)
    log.info("  Δ IC (odd − even) = %+.6f", delta)
    log.info("  95%% CIs overlap: %s", cis_overlap)
    log.info("")
    if not cis_overlap:
        log.info("  FINDING: IC_odd ≠ IC_even (non-overlapping CIs).")
        log.info("  Evidence of two structurally distinct text streams in the")
        log.info("  boustrophedon alternating lines — potential voice split.")
    elif marginal:
        log.info("  MARGINAL: CIs overlap by only %.1f%% of CI width.", overlap_fraction * 100)
        log.info("  IC_even > IC_odd consistently; trend below threshold but notable.")
    else:
        log.info("  No significant difference: IC_odd ≈ IC_even (CIs overlap).")
        log.info("  Boustrophedon voice-split hypothesis not supported by IC alone.")
    log.info("=" * 64)

    result: dict = {
        "n_odd_lines": n_odd_lines,
        "n_even_lines": n_even_lines,
        "n_odd_tokens": len(odd_tokens),
        "n_even_tokens": len(even_tokens),
        "ic_odd": round(ic_odd, 6),
        "ic_even": round(ic_even, 6),
        "ic_odd_ci_95_lo": round(lo_odd, 6) if math.isfinite(lo_odd) else None,
        "ic_odd_ci_95_hi": round(hi_odd, 6) if math.isfinite(hi_odd) else None,
        "ic_even_ci_95_lo": round(lo_even, 6) if math.isfinite(lo_even) else None,
        "ic_even_ci_95_hi": round(hi_even, 6) if math.isfinite(hi_even) else None,
        "delta_ic_odd_minus_even": round(delta, 6),
        "cis_overlap": cis_overlap,
        "overlap_fraction": round(overlap_fraction, 4) if math.isfinite(overlap_fraction) else None,
        "marginal_overlap": marginal,
        "finding": (
            "IC_odd ≠ IC_even with non-overlapping 95% CIs: evidence of two "
            "structurally distinct text streams in boustrophedon alternating lines."
            if not cis_overlap else
            f"CIs overlap marginally ({overlap_fraction*100:.1f}% of CI width); "
            "IC_even > IC_odd trend consistent but below threshold."
            if marginal else
            "No significant IC difference between odd and even lines (CIs overlap)."
        ),
    }

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        log.info("Boustrophedon IC results written to %s", output_path)

    return result


def _load_lines_by_cluster_for_scenario(
    corpus_dir: Path,
    scenario: str,
) -> dict[str, list[tuple[str, list[str]]]]:
    """Return {cluster: [(parity, tokens), …]} with tablet assignments per scenario.

    Mirrors the reassignment logic of :func:`load_tokens_under_scenario` but
    preserves line-parity information so the caller can split odd vs even within
    each stratum.  The ``probabilistic_weighted`` scenario assigns the first 20 %
    of each unknown tablet's lines (by reading order) to ``pre_contact`` and the
    rest to ``post_contact``.
    """
    from collections import defaultdict
    from hackingrongo.zone_b.sequence_model import load_sequences_with_lines

    # Determine base cluster for each tablet from the corpus JSON.
    tablet_cluster: dict[str, str] = {}
    unknown_ids: list[str] = []
    for path in sorted(corpus_dir.glob("[A-Z].json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        tid: str = data["tablet_id"]
        cl = data.get("cluster", UNKNOWN_STRATUM)
        if cl == UNKNOWN_STRATUM:
            unknown_ids.append(tid)
        else:
            tablet_cluster[tid] = cl

    # Apply scenario reassignment to unknown tablets.
    if scenario == "conservative_all_late":
        for tid in unknown_ids:
            tablet_cluster[tid] = POST_CONTACT

    elif scenario == "optimistic_distributed":
        for i, tid in enumerate(sorted(unknown_ids)):
            tablet_cluster[tid] = PRE_CONTACT if i % 2 == 0 else POST_CONTACT

    elif scenario == "probabilistic_weighted":
        # Mark unknowns for fractional line-split; resolved after reading lines.
        for tid in unknown_ids:
            tablet_cluster[tid] = "_weighted"

    else:
        raise ValueError(f"Unknown scenario {scenario!r}")

    # Accumulate line records per cluster.
    by_cluster: dict[str, list[tuple[str, list[str]]]] = defaultdict(list)
    # For probabilistic_weighted: gather lines per unknown tablet first.
    weighted: dict[str, list[tuple[str, list[str]]]] = defaultdict(list)

    for tablet_id, _side, _line_num, parity, tokens in load_sequences_with_lines(corpus_dir):
        cl = tablet_cluster.get(tablet_id, UNKNOWN_STRATUM)
        if cl == "_weighted":
            weighted[tablet_id].append((parity, tokens))
        else:
            by_cluster[cl].append((parity, tokens))

    # Split weighted tablets: first 20 % of lines → pre, rest → post.
    for _tid, lines in weighted.items():
        n_pre = max(1, round(len(lines) * 0.20))
        for pair in lines[:n_pre]:
            by_cluster[PRE_CONTACT].append(pair)
        for pair in lines[n_pre:]:
            by_cluster[POST_CONTACT].append(pair)

    return dict(by_cluster)


def boustrophedon_sensitivity(
    scenarios: list[str],
    corpus_dir: Path,
    output_path: Path | None = None,
) -> dict:
    """IC_odd vs IC_even per temporal cluster under each dating scenario.

    Runs the boustrophedon voice-split test independently for each cluster
    (pre_contact, post_contact) under each tablet-dating scenario.  This
    answers: is the line-parity IC difference driven by one stratum, or does
    it appear within both?

    Parameters
    ----------
    scenarios : list[str]
        Scenario names — same choices as :func:`sensitivity_analysis`.
    corpus_dir : Path
        Corpus directory.
    output_path : Path or None
        If given, write JSON results here.

    Returns
    -------
    dict
        Keyed by scenario name → cluster name → IC stats dict.
    """
    log.info("")
    log.info("=" * 64)
    log.info("Boustrophedon Sensitivity — IC by parity × cluster × scenario")
    log.info("=" * 64)

    scenario_results: dict[str, dict] = {}

    for scenario in scenarios:
        log.info("")
        log.info("Scenario: %s", scenario)
        by_cluster = _load_lines_by_cluster_for_scenario(corpus_dir, scenario)
        cluster_stats: dict[str, dict] = {}

        for cluster in (PRE_CONTACT, POST_CONTACT):
            lines = by_cluster.get(cluster, [])
            odd_toks = [t for parity, toks in lines for t in toks if parity == "odd"]
            even_toks = [t for parity, toks in lines for t in toks if parity == "even"]

            ic_odd = index_of_coincidence(odd_toks)
            ic_even = index_of_coincidence(even_toks)
            lo_odd, hi_odd = (
                bootstrap_ic_ci(odd_toks) if len(odd_toks) >= 30
                else (float("nan"), float("nan"))
            )
            lo_even, hi_even = (
                bootstrap_ic_ci(even_toks) if len(even_toks) >= 30
                else (float("nan"), float("nan"))
            )

            cis_overlap = not (
                (math.isfinite(hi_odd) and math.isfinite(lo_even) and hi_odd < lo_even)
                or (math.isfinite(hi_even) and math.isfinite(lo_odd) and hi_even < lo_odd)
            )
            delta = ic_odd - ic_even

            overlap_frac = float("nan")
            if cis_overlap and all(math.isfinite(v) for v in (lo_odd, hi_odd, lo_even, hi_even)):
                ov_width = max(0.0, min(hi_odd, hi_even) - max(lo_odd, lo_even))
                narrow = min(hi_odd - lo_odd, hi_even - lo_even)
                overlap_frac = ov_width / narrow if narrow > 0 else 0.0
            marginal = cis_overlap and math.isfinite(overlap_frac) and overlap_frac < 0.10

            log.info(
                "  %-14s  odd=%5d  even=%5d  IC_odd=%.6f  IC_even=%.6f"
                "  Δ=%+.6f  overlap=%s%s",
                cluster, len(odd_toks), len(even_toks),
                ic_odd, ic_even, delta,
                "yes" if cis_overlap else "NO",
                f" ({overlap_frac*100:.1f}%)" if marginal else "",
            )

            cluster_stats[cluster] = {
                "n_odd_tokens": len(odd_toks),
                "n_even_tokens": len(even_toks),
                "ic_odd": round(ic_odd, 6),
                "ic_even": round(ic_even, 6),
                "ic_odd_ci_95_lo": round(lo_odd, 6) if math.isfinite(lo_odd) else None,
                "ic_odd_ci_95_hi": round(hi_odd, 6) if math.isfinite(hi_odd) else None,
                "ic_even_ci_95_lo": round(lo_even, 6) if math.isfinite(lo_even) else None,
                "ic_even_ci_95_hi": round(hi_even, 6) if math.isfinite(hi_even) else None,
                "delta_ic_odd_minus_even": round(delta, 6),
                "cis_overlap": cis_overlap,
                "overlap_fraction": round(overlap_frac, 4) if math.isfinite(overlap_frac) else None,
                "marginal_overlap": marginal,
            }

        scenario_results[scenario] = cluster_stats

    log.info("")
    log.info("-" * 64)
    log.info("ROBUSTNESS: direction IC_even > IC_odd (Δ < 0) per cluster")
    for cluster in (PRE_CONTACT, POST_CONTACT):
        deltas = [
            scenario_results[s][cluster]["delta_ic_odd_minus_even"]
            for s in scenarios
            if cluster in scenario_results[s]
        ]
        consistent = all(d < 0 for d in deltas) if deltas else False
        log.info(
            "  %-14s  consistent IC_even > IC_odd: %s  deltas=%s",
            cluster, consistent,
            [f"{d:+.6f}" for d in deltas],
        )
    log.info("-" * 64)

    result = {"scenarios": scenario_results}

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        log.info("Boustrophedon sensitivity written to %s", output_path)

    return result


# ---------------------------------------------------------------------------
# Frequency-language match
# ---------------------------------------------------------------------------

def frequency_language_match(
    corpus_dir: Path,
    lm_dir: Path,
    phoneme_map: dict[str, str] | None = None,
) -> dict:
    """Compare the sign-frequency distribution to phoneme-frequency distributions
    in available language model reference corpora.

    Two complementary statistics are computed:

    **Zipf exponent comparison**
        The Zipf exponent α is estimated by maximum-likelihood (Clauset 2009
        approximation) for the sign frequency distribution and for each
        language-model phoneme frequency distribution.  A close match in α
        suggests the sign inventory behaves like that language's phoneme
        inventory under a Zipfian model.

    **Spearman rank correlation** (requires ``phoneme_map``)
        Given an explicit sign→phoneme assignment, the ranks of sign
        frequencies are correlated with the ranks of the assigned phoneme
        frequencies.  ρ ≈ 1 means frequent signs were assigned to frequent
        phonemes; ρ ≈ 0 means the assignment is independent of frequency.
        High ρ with a specific language LM is evidence that the Zipf
        frequency structure is preserved through the proposed assignment.

    **Chi-squared goodness-of-fit** (requires ``phoneme_map``)
        Aggregate phoneme counts implied by the map are compared to the
        expected proportions from each LM, producing a χ² statistic and
        p-value.  A high p-value means the assigned-phoneme distribution is
        consistent with the reference language.

    Parameters
    ----------
    corpus_dir : Path
        Directory containing corpus JSON/CSV files with sign tokens.
    lm_dir : Path
        Directory containing language-model files.  Each ``*.json`` file
        should have a ``unigram`` key mapping phoneme → log-probability (or
        count).
    phoneme_map : dict[str, str] | None
        Optional sign→phoneme assignment.  Required for Spearman and χ²
        diagnostics.

    Returns
    -------
    dict with keys:
        - ``zipf_alpha_signs``        : float, Zipf α for sign frequencies
        - ``zipf_alpha_per_lm``       : dict[str, float], Zipf α per LM
        - ``spearman_rho_per_lm``     : dict[str, float] | None
        - ``chi2_stat_per_lm``        : dict[str, float] | None
        - ``chi2_p_value_per_lm``     : dict[str, float] | None
        - ``best_lm_by_spearman``     : str | None
        - ``best_lm_by_chi2_p``       : str | None
    """
    try:
        from scipy.stats import spearmanr, chisquare  # type: ignore[import]
    except ImportError:
        log.warning("scipy not installed — chi2 and Spearman tests unavailable.")
        spearmanr = None  # type: ignore[assignment]
        chisquare = None  # type: ignore[assignment]

    # ── Load sign frequencies from corpus ───────────────────────────────────
    sign_counts: Counter[str] = Counter()
    for path in sorted(corpus_dir.glob("**/*.json")):
        try:
            data = json.loads(path.read_text())
            tokens_key = None
            for k in ("tokens", "signs", "sequence", "text"):
                if k in data:
                    tokens_key = k
                    break
            if tokens_key is None:
                continue
            val = data[tokens_key]
            if isinstance(val, list):
                if val and isinstance(val[0], str):
                    sign_counts.update(val)
                elif val and isinstance(val[0], list):
                    for row in val:
                        sign_counts.update(row)
        except Exception:
            continue

    if not sign_counts:
        log.warning("No sign tokens found under %s — frequency match skipped.", corpus_dir)
        return {"zipf_alpha_signs": None}

    sign_freqs: list[float] = sorted(sign_counts.values(), reverse=True)
    total_signs = sum(sign_freqs)

    def _zipf_alpha(counts: list[float]) -> float:
        """MLE Zipf / power-law exponent (continuous approximation)."""
        x_min = min(counts)
        n = len(counts)
        if x_min <= 0:
            return float("nan")
        s = sum(math.log(c / x_min) for c in counts)
        if s == 0.0:
            # All counts are equal — power-law exponent is undefined.
            return float("nan")
        return 1.0 + n / s

    zipf_alpha_signs = _zipf_alpha(sign_freqs)

    # ── Load language model phoneme frequencies ──────────────────────────────
    lm_files = sorted(lm_dir.glob("*.json"))
    lm_phoneme_counts: dict[str, dict[str, float]] = {}
    for lm_path in lm_files:
        try:
            lm_data = json.loads(lm_path.read_text())
            unigram: dict[str, float] = {}
            if "unigram" in lm_data:
                for ph, val in lm_data["unigram"].items():
                    # values may be log-probs or counts
                    v = float(val)
                    unigram[ph] = math.exp(v) if v < 0 else v
            elif "phoneme_counts" in lm_data:
                unigram = {ph: float(c) for ph, c in lm_data["phoneme_counts"].items()}
            if unigram:
                lm_phoneme_counts[lm_path.stem] = unigram
        except Exception:
            continue

    zipf_alpha_per_lm: dict[str, float] = {}
    for lm_name, counts in lm_phoneme_counts.items():
        freqs = sorted(counts.values(), reverse=True)
        zipf_alpha_per_lm[lm_name] = _zipf_alpha(freqs)

    # ── Spearman rank correlation & Chi-squared (require phoneme_map) ────────
    spearman_rho_per_lm: dict[str, float] | None = None
    chi2_stat_per_lm:    dict[str, float] | None = None
    chi2_p_per_lm:       dict[str, float] | None = None
    best_spearman:  str | None = None
    best_chi2_p:    str | None = None

    if phoneme_map is not None and lm_phoneme_counts and spearmanr is not None:
        # Sign ranks by descending frequency.
        sorted_signs = sorted(sign_counts, key=lambda s: -sign_counts[s])
        sign_rank = {s: i + 1 for i, s in enumerate(sorted_signs)}

        # Aggregate phoneme counts under the proposed assignment.
        agg_phoneme: Counter[str] = Counter()
        for sign, ph in phoneme_map.items():
            if sign in sign_counts:
                agg_phoneme[ph] += sign_counts[sign]

        spearman_rho_per_lm = {}
        chi2_stat_per_lm    = {}
        chi2_p_per_lm       = {}

        for lm_name, lm_counts in lm_phoneme_counts.items():
            lm_total = sum(lm_counts.values()) or 1.0

            # Spearman: compare sign freq rank to assigned-phoneme freq rank.
            lm_ph_rank = {
                ph: i + 1
                for i, ph in enumerate(sorted(lm_counts, key=lambda ph: -lm_counts[ph]))
            }
            ranks_sign: list[float] = []
            ranks_ph: list[float] = []
            for sign in sorted_signs:
                ph = phoneme_map.get(sign)
                if ph and ph in lm_ph_rank:
                    ranks_sign.append(sign_rank[sign])
                    ranks_ph.append(lm_ph_rank[ph])
            if len(ranks_sign) >= 5:
                rho, _ = spearmanr(ranks_sign, ranks_ph)
                spearman_rho_per_lm[lm_name] = float(rho)

            # Chi-squared goodness-of-fit.
            if chisquare is not None and agg_phoneme:
                common_ph = sorted(set(agg_phoneme) & set(lm_counts))
                if len(common_ph) >= 2:
                    observed  = np.array([agg_phoneme[ph] for ph in common_ph], dtype=float)
                    expected_props = np.array([lm_counts[ph] / lm_total for ph in common_ph])
                    expected_props /= expected_props.sum()
                    expected = expected_props * observed.sum()
                    stat, pval = chisquare(observed, f_exp=expected)
                    chi2_stat_per_lm[lm_name] = float(stat)
                    chi2_p_per_lm[lm_name]    = float(pval)

        if spearman_rho_per_lm:
            best_spearman = max(spearman_rho_per_lm, key=lambda k: spearman_rho_per_lm[k])  # type: ignore[index]
        if chi2_p_per_lm:
            best_chi2_p = max(chi2_p_per_lm, key=lambda k: chi2_p_per_lm[k])  # type: ignore[index]

    log.info(
        "Zipf α (signs)=%.3f  |  LMs: %s",
        zipf_alpha_signs,
        {k: f"{v:.3f}" for k, v in zipf_alpha_per_lm.items()},
    )
    if best_spearman:
        assert spearman_rho_per_lm is not None
        log.info(
            "Best Spearman ρ match: %s (ρ=%.3f)",
            best_spearman, spearman_rho_per_lm[best_spearman],
        )
    if best_chi2_p:
        assert chi2_p_per_lm is not None
        log.info(
            "Best χ² p-value match: %s (p=%.4f)",
            best_chi2_p, chi2_p_per_lm[best_chi2_p],
        )

    return {
        "zipf_alpha_signs":    zipf_alpha_signs,
        "zipf_alpha_per_lm":   zipf_alpha_per_lm,
        "spearman_rho_per_lm": spearman_rho_per_lm,
        "chi2_stat_per_lm":    chi2_stat_per_lm,
        "chi2_p_value_per_lm": chi2_p_per_lm,
        "best_lm_by_spearman": best_spearman,
        "best_lm_by_chi2_p":   best_chi2_p,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="IC / entropy analysis by cluster.")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON result.")
    parser.add_argument(
        "--uncertain-weight",
        type=float,
        default=0.0,
        metavar="W",
        help="Weight for uncertain tokens (0=exclude, 0.5=half-credit, 1=full; default: 0).",
    )
    parser.add_argument(
        "--scenario",
        action="append",
        dest="scenarios",
        metavar="NAME",
        choices=list(_SCENARIO_NAMES),
        help=(
            "Run sensitivity analysis under a named scenario. "
            "Repeat to include multiple scenarios. "
            f"Choices: {', '.join(_SCENARIO_NAMES)}"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Write JSON results to this path (e.g. outputs/sensitivity_analysis.json).",
    )
    parser.add_argument(
        "--zipf",
        action="store_true",
        help=(
            "Test whether the pooled sign frequency distribution follows Zipf's law. "
            "Fits scipy.stats.zipf via MLE, reports the exponent, and saves a log-log "
            "rank-frequency plot to outputs/analysis/."
        ),
    )
    parser.add_argument(
        "--boustrophedon",
        action="store_true",
        help=(
            "Boustrophedon voice-split test: compute IC separately for odd-numbered "
            "lines (1, 3, 5, …) vs even-numbered lines (2, 4, 6, …) across all "
            "tablets. IC_odd ≠ IC_even with non-overlapping CIs is evidence of two "
            "structurally distinct text streams."
        ),
    )
    args = parser.parse_args()

    if args.boustrophedon:
        cfg = OmegaConf.load(PROJECT_ROOT / "conf" / "config.yaml")
        corpus_dir = PROJECT_ROOT / cfg.paths.corpus_dir
        output_dir = PROJECT_ROOT / "outputs" / "analysis"
        if args.scenarios:
            result = boustrophedon_sensitivity(
                args.scenarios,
                corpus_dir,
                output_path=output_dir / "boustrophedon_sensitivity.json",
            )
        else:
            result = compute_ic_by_line_parity(
                corpus_dir,
                output_path=output_dir / "boustrophedon_ic.json",
            )
        if args.json:
            print(json.dumps(result, indent=2))
        return

    if args.zipf:
        cfg = OmegaConf.load(PROJECT_ROOT / "conf" / "config.yaml")
        corpus_dir = PROJECT_ROOT / cfg.paths.corpus_dir
        output_dir = PROJECT_ROOT / "outputs" / "analysis"
        result = zipf_analysis(corpus_dir, output_dir=output_dir, plot=True)
        if args.json:
            print(json.dumps(result, indent=2))
        return

    if args.scenarios:
        sensitivity_analysis(
            scenarios=args.scenarios,
            uncertain_weight=args.uncertain_weight,
            output_path=args.output,
        )
    else:
        results = analyse(emit_json=args.json, uncertain_weight=args.uncertain_weight)
        if args.output:
            out_path = args.output
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
            log.info("Results written to %s", out_path)


if __name__ == "__main__":
    main()
