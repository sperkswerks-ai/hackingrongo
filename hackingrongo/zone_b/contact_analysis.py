"""
hackingrongo.zone_b.contact_analysis
=====================================

Behavioural sign partitioning across the contact boundary.

For each Horley-resolved sign that appears in both the pre_contact (Tablet D)
and post_contact (B, C, O, Q) clusters, we compute:

1. Relative frequency in each cluster (per 1 000 tokens).
2. Frequency ratio  r = freq_pre / freq_post.
3. Log-likelihood G² statistic (Dunning 1993) — sign-level test of whether
   the observed pre/post frequency difference is significant given corpus sizes.
4. Signed log-odds  ω = log( (f_pre / n_pre) / (f_post / n_post) )
   with Laplace smoothing (+0.5 each cell).

Signs are then partitioned into two behavioural classes:
* **pre_biased**  — signs significantly more frequent in pre_contact (G² > χ²_1, p<0.05 critical = 3.841)
* **post_biased** — signs significantly more frequent in post_contact
* **neutral**     — no significant directional bias

The pre/post partition is the CFP's second headline result: it identifies
which signs characterise early (pre-contact) rongorongo text vs. the
post-contact corpus.  Signs with opposite bias patterns are candidates for
scribal innovation (post) or archaism (pre).

Usage
-----
    conda run python hackingrongo/zone_b/contact_analysis.py
    conda run python hackingrongo/zone_b/contact_analysis.py --json
    conda run python hackingrongo/zone_b/contact_analysis.py --min-g2 6.63  # p<0.01
    conda run python hackingrongo/zone_b/contact_analysis.py \
        --output outputs/contact_partition.json \
        --plot outputs/contact_partition_bipartite.html
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from omegaconf import OmegaConf  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

# χ²_1 critical values
CHI2_P05 = 3.841   # p < 0.05
CHI2_P01 = 6.635   # p < 0.01
CHI2_P001 = 10.828  # p < 0.001

# Dating scenario names — mirror ``entropy._SCENARIO_NAMES``
_SCENARIO_NAMES = ("conservative_all_late", "optimistic_distributed", "probabilistic_weighted")


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def log_likelihood_g2(a: int, b: int, c: int, d: int) -> float:
    """G² (log-likelihood ratio) for a 2×2 contingency table.

    Table layout::

              sign   ¬sign
        pre   a      b
        post  c      d

    Returns G² ≥ 0; large values indicate significant departure from
    independence (i.e. the sign is *not* distributed uniformly across clusters).
    """
    n = a + b + c + d
    if n == 0:
        return 0.0

    def _xlogx(x: float) -> float:
        return x * math.log(x) if x > 0 else 0.0

    # observed
    g2 = 2.0 * (
        _xlogx(a) + _xlogx(b) + _xlogx(c) + _xlogx(d)
        - _xlogx(a + b) - _xlogx(a + c)
        - _xlogx(b + d) - _xlogx(c + d)
        + _xlogx(n)
    )
    return max(g2, 0.0)


def signed_log_odds(f_pre: int, n_pre: int, f_post: int, n_post: int) -> float:
    """Signed log-odds with Laplace (+0.5) smoothing."""
    p_pre = (f_pre + 0.5) / (n_pre + 1.0)
    p_post = (f_post + 0.5) / (n_post + 1.0)
    return math.log(p_pre / p_post)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_freq_by_cluster(
    corpus_dir: Path,
    target_clusters: tuple[str, ...] = ("pre_contact", "post_contact"),
    include_compound_components: bool = True,
) -> dict[str, Counter]:
    freqs: dict[str, Counter] = {c: Counter() for c in target_clusters}
    for path in sorted(corpus_dir.glob("[A-Z].json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        cluster = data.get("cluster", "unknown")
        if cluster not in target_clusters:
            continue
        for g in data["glyphs"]:
            hc = g.get("horley_code")
            if hc:
                freqs[cluster][hc] += 1
            if include_compound_components:
                for comp_hc in (g.get("horley_components") or []):
                    freqs[cluster][comp_hc] += 1
    return freqs


def load_freq_by_cluster_under_scenario(
    corpus_dir: Path,
    scenario: str,
    include_compound_components: bool = True,
) -> dict[str, Counter]:
    """Like ``load_freq_by_cluster`` but assigns unknown-cluster tablets
    according to *scenario* instead of excluding them.

    Scenario semantics
    ------------------
    ``conservative_all_late``
        All undated tablets treated as *post_contact*.  If the G² partition
        still holds, the result is robust to the most pessimistic dating.

    ``optimistic_distributed``
        Unknown tablets split evenly (50/50) between pre and post.
        Even-indexed unknown tablets (sorted order) → pre_contact,
        odd-indexed → post_contact.

    ``probabilistic_weighted``
        Empirical 20/80 prior matching Ferrara anchor tablets: first 20% of
        each unknown tablet's tokens go to pre_contact, remaining 80% to
        post_contact.
    """
    if scenario not in _SCENARIO_NAMES:
        raise ValueError(
            f"Unknown scenario {scenario!r}. Choose from {_SCENARIO_NAMES}"
        )
    freqs: dict[str, Counter] = {"pre_contact": Counter(), "post_contact": Counter()}
    unknown_tablets: list[list[str]] = []  # collected for deferred even/odd split

    for path in sorted(corpus_dir.glob("[A-Z].json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        cluster = data.get("cluster", "unknown")

        tokens: list[str] = []
        for g in data["glyphs"]:
            hc = g.get("horley_code")
            if hc:
                tokens.append(hc)
            if include_compound_components:
                for comp_hc in (g.get("horley_components") or []):
                    tokens.append(comp_hc)

        if cluster in ("pre_contact", "post_contact"):
            for t in tokens:
                freqs[cluster][t] += 1
            continue

        # Unknown tablet — assign per scenario.
        if scenario == "conservative_all_late":
            for t in tokens:
                freqs["post_contact"][t] += 1

        elif scenario == "optimistic_distributed":
            unknown_tablets.append(tokens)  # defer to even/odd logic below

        elif scenario == "probabilistic_weighted":
            n_pre = max(1, int(round(len(tokens) * 0.20)))
            for t in tokens[:n_pre]:
                freqs["pre_contact"][t] += 1
            for t in tokens[n_pre:]:
                freqs["post_contact"][t] += 1

    if scenario == "optimistic_distributed":
        for unk_pos, tokens in enumerate(unknown_tablets):
            target = "pre_contact" if unk_pos % 2 == 0 else "post_contact"
            for t in tokens:
                freqs[target][t] += 1

    return freqs


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def analyse(
    min_g2: float = CHI2_P05,
    emit_json: bool = False,
    scenario: str | None = None,
) -> list[dict]:
    """Run contact analysis; return per-sign records list.

    Parameters
    ----------
    min_g2 : float
        G² significance threshold.
    emit_json : bool
        Print full per-sign records as JSON to stdout.
    scenario : str | None
        If given, one of ``_SCENARIO_NAMES``; controls how tablets with
        ``cluster == "unknown"`` are assigned to pre/post strata.
        ``None`` (default) excludes unknown tablets (original behaviour).
    """
    cfg = OmegaConf.load(PROJECT_ROOT / "conf" / "config.yaml")
    corpus_dir = PROJECT_ROOT / cfg.paths.corpus_dir

    if scenario is not None:
        freqs = load_freq_by_cluster_under_scenario(corpus_dir, scenario)
    else:
        freqs = load_freq_by_cluster(corpus_dir)
    pre_freq = freqs["pre_contact"]
    post_freq = freqs["post_contact"]

    n_pre = sum(pre_freq.values())
    n_post = sum(post_freq.values())

    log.info("")
    log.info("=" * 64)
    log.info("Contact Analysis — sign-level pre vs post_contact partitioning")
    log.info("=" * 64)
    log.info("  pre_contact  corpus: %d tokens, %d types (Tablet D)", n_pre, len(pre_freq))
    log.info("  post_contact corpus: %d tokens, %d types (B, C, O, Q)", n_post, len(post_freq))
    log.info("  G² threshold: %.3f (p<0.05)", min_g2)
    log.info("")

    # All signs seen in either cluster
    all_signs = sorted(set(pre_freq) | set(post_freq))

    records: list[dict] = []
    for sign in all_signs:
        a = pre_freq[sign]   # pre, sign
        b = n_pre - a        # pre, ¬sign
        c = post_freq[sign]  # post, sign
        d = n_post - c       # post, ¬sign

        g2 = log_likelihood_g2(a, b, c, d)
        omega = signed_log_odds(a, n_pre, c, n_post)
        freq_pre_per_k = 1000.0 * a / n_pre if n_pre else 0.0
        freq_post_per_k = 1000.0 * c / n_post if n_post else 0.0

        if g2 >= min_g2:
            bias = "pre_biased" if omega > 0 else "post_biased"
        else:
            bias = "neutral"

        records.append({
            "sign": sign,
            "f_pre": a,
            "f_post": c,
            "freq_pre_per_1k": round(freq_pre_per_k, 2),
            "freq_post_per_1k": round(freq_post_per_k, 2),
            "g2": round(g2, 3),
            "log_odds": round(omega, 3),
            "bias": bias,
            "seen_in_both": a > 0 and c > 0,
        })

    records.sort(key=lambda r: -r["g2"])

    # Summary counts
    pre_biased = [r for r in records if r["bias"] == "pre_biased"]
    post_biased = [r for r in records if r["bias"] == "post_biased"]
    neutral = [r for r in records if r["bias"] == "neutral"]
    both = [r for r in records if r["seen_in_both"]]

    log.info("Sign partition (G² ≥ %.3f):", min_g2)
    log.info("  pre_biased:  %d signs", len(pre_biased))
    log.info("  post_biased: %d signs", len(post_biased))
    log.info("  neutral:     %d signs", len(neutral))
    log.info("  seen in both clusters: %d of %d total signs", len(both), len(all_signs))
    log.info("")

    # Top pre-biased signs
    log.info("Top pre_biased signs (G² descending):")
    log.info("  %-8s  %5s  %5s  %6s  %6s  %7s  %s",
             "sign", "f_pre", "f_post", "pre/k", "post/k", "G²", "log_odds")
    for r in pre_biased[:20]:
        log.info("  %-8s  %5d  %5d  %6.1f  %6.1f  %7.3f  %+.3f",
                 r["sign"], r["f_pre"], r["f_post"],
                 r["freq_pre_per_1k"], r["freq_post_per_1k"],
                 r["g2"], r["log_odds"])

    log.info("")
    log.info("Top post_biased signs (G² descending):")
    log.info("  %-8s  %5s  %5s  %6s  %6s  %7s  %s",
             "sign", "f_pre", "f_post", "pre/k", "post/k", "G²", "log_odds")
    for r in post_biased[:20]:
        log.info("  %-8s  %5d  %5d  %6.1f  %6.1f  %7.3f  %+.3f",
                 r["sign"], r["f_pre"], r["f_post"],
                 r["freq_pre_per_1k"], r["freq_post_per_1k"],
                 r["g2"], r["log_odds"])

    # G² significance thresholds summary
    log.info("")
    log.info("Significance threshold summary:")
    for label, thr in [("p<0.05", CHI2_P05), ("p<0.01", CHI2_P01), ("p<0.001", CHI2_P001)]:
        n_sig = sum(1 for r in records if r["g2"] >= thr)
        log.info("  %s (G² ≥ %.3f): %d signs", label, thr, n_sig)

    if emit_json:
        print(json.dumps(records, indent=2))

    return records


def contact_sensitivity_analysis(
    min_g2: float = CHI2_P05,
    output_path: Path | None = None,
) -> dict:
    """Run contact analysis under all three dating scenarios and assess robustness.

    For each sign that is significant (G² ≥ *min_g2*) in at least one
    scenario, checks whether the *bias direction* (pre vs post) is
    consistent across all three scenarios.  Also reports the Jaccard
    similarity of the pre-biased and post-biased core sets.

    Parameters
    ----------
    min_g2 : float
        G² significance threshold (applied uniformly across scenarios).
    output_path : Path | None
        If given, write a JSON sensitivity report here.

    Returns
    -------
    dict with keys:
        ``scenarios``       — per-scenario {n_pre_biased, n_post_biased, n_neutral}
        ``sign_stability``  — per-sign bias direction across scenarios
        ``robustness``      — aggregate stability metrics
    """
    log.info("")
    log.info("=" * 64)
    log.info("Contact Sensitivity Analysis — G² partition across 3 scenarios")
    log.info("=" * 64)

    scenario_records: dict[str, list[dict]] = {}
    scenario_summaries: dict[str, dict] = {}

    for scenario in _SCENARIO_NAMES:
        records = analyse(min_g2=min_g2, scenario=scenario)
        scenario_records[scenario] = records
        scenario_summaries[scenario] = {
            "n_pre_biased":  sum(1 for r in records if r["bias"] == "pre_biased"),
            "n_post_biased": sum(1 for r in records if r["bias"] == "post_biased"),
            "n_neutral":     sum(1 for r in records if r["bias"] == "neutral"),
            "n_total":       len(records),
        }
        log.info(
            "  %-32s  pre_biased=%d  post_biased=%d  neutral=%d",
            scenario,
            scenario_summaries[scenario]["n_pre_biased"],
            scenario_summaries[scenario]["n_post_biased"],
            scenario_summaries[scenario]["n_neutral"],
        )

    # Sign stability: is the bias direction consistent across all three scenarios?
    all_signs: set[str] = set()
    for recs in scenario_records.values():
        all_signs.update(r["sign"] for r in recs)

    sign_lookup: dict[str, dict[str, dict]] = {
        scenario: {r["sign"]: r for r in recs}
        for scenario, recs in scenario_records.items()
    }

    sign_stability: list[dict] = []
    stable_count = 0
    unstable_count = 0

    for sign in sorted(all_signs):
        directions = {
            scenario: sign_lookup[scenario].get(sign, {}).get("bias", "neutral")
            for scenario in _SCENARIO_NAMES
        }
        # Only assess stability for signs significant in at least one scenario.
        if all(d == "neutral" for d in directions.values()):
            continue
        is_stable = len(set(directions.values())) == 1
        if is_stable:
            stable_count += 1
        else:
            unstable_count += 1
        sign_stability.append({"sign": sign, "stable": is_stable, "directions": directions})

    total_sig = stable_count + unstable_count
    stable_fraction = stable_count / total_sig if total_sig > 0 else 1.0

    # Jaccard similarity of the pre/post-biased core sets across scenarios.
    pre_sets  = [frozenset(r["sign"] for r in scenario_records[s] if r["bias"] == "pre_biased")  for s in _SCENARIO_NAMES]
    post_sets = [frozenset(r["sign"] for r in scenario_records[s] if r["bias"] == "post_biased") for s in _SCENARIO_NAMES]

    pre_core  = pre_sets[0].intersection(*pre_sets[1:])
    post_core = post_sets[0].intersection(*post_sets[1:])
    pre_any   = pre_sets[0].union(*pre_sets[1:])
    post_any  = post_sets[0].union(*post_sets[1:])

    pre_jaccard  = len(pre_core)  / len(pre_any)  if pre_any  else 1.0
    post_jaccard = len(post_core) / len(post_any) if post_any else 1.0

    log.info("")
    log.info("ROBUSTNESS SUMMARY")
    log.info("  Signs significant in ≥1 scenario: %d", total_sig)
    log.info("  Direction stable (all 3):         %d (%.1f%%)", stable_count, stable_fraction * 100)
    log.info("  Direction unstable (flips):       %d", unstable_count)
    log.info("  Pre-biased  core  |  Jaccard: %d signs  J=%.3f", len(pre_core),  pre_jaccard)
    log.info("  Post-biased core  |  Jaccard: %d signs  J=%.3f", len(post_core), post_jaccard)
    log.info("-" * 64)

    result = {
        "scenarios": scenario_summaries,
        "sign_stability": sign_stability,
        "robustness": {
            "n_significant_any_scenario": total_sig,
            "n_direction_stable":   stable_count,
            "n_direction_unstable": unstable_count,
            "stable_fraction":      round(stable_fraction, 4),
            "pre_biased_core":      sorted(pre_core),
            "post_biased_core":     sorted(post_core),
            "pre_biased_jaccard":   round(pre_jaccard, 4),
            "post_biased_jaccard":  round(post_jaccard, 4),
        },
    }

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        log.info("Contact sensitivity results written to %s", output_path)

    return result


# ---------------------------------------------------------------------------
# Contact partition (structured output with Bonferroni correction)
# ---------------------------------------------------------------------------

def build_contact_partition(
    min_g2: float = CHI2_P05,
    scenario: str | None = None,
    output_path: Path | None = None,
    report_path: Path | None = None,
) -> dict:
    """Compute the contact partition with Bonferroni correction and bipartite edges.

    Extends :func:`analyse` with:

    * Bonferroni correction for 120 signs × 2 strata multiple comparisons.
    * Structured output with ``stratum_shifting_signs``, ``stable_signs``, and
      ``bipartite_edges`` keys (expected by ``generate_holy_grail_report.py``).
    * The per-sign record list (same schema as :func:`analyse`) is written to
      *output_path* if given, preserving backward compatibility with
      ``CONTACT_JSON`` consumers that iterate the list directly.
    * Optionally writes an HTML report to *report_path* when ``--report`` is used.

    Parameters
    ----------
    min_g2 :
        Unadjusted G² threshold for the uncorrected analysis (default p<0.05).
    scenario :
        Dating scenario for unknown-stratum tablets; ``None`` excludes them.
    output_path :
        Path to write ``contact_partition.json``.
    report_path :
        Path to write ``contact_partition_report.html``.

    Returns
    -------
    dict
        Keys: ``stratum_shifting_signs``, ``stable_signs``, ``bipartite_edges``,
        ``summary``, ``records`` (the per-sign list).
    """
    import scipy.stats as _stats  # type: ignore

    records = analyse(min_g2=0.0, emit_json=False, scenario=scenario)

    n_tests = len(records) * 2  # signs × strata
    # Bonferroni-adjusted alpha
    alpha_bonf = 0.05 / max(n_tests, 1)
    # Corresponding G²_1 critical value (chi-squared CDF inverse)
    chi2_bonf = _stats.chi2.ppf(1.0 - alpha_bonf, df=1)
    log.info(
        "Bonferroni correction: n_tests=%d, alpha_adj=%.2e, G²_threshold=%.3f",
        n_tests, alpha_bonf, chi2_bonf,
    )

    stratum_shifting: list[dict] = []
    stable: list[dict] = []

    for r in records:
        g2_val = r["g2"]
        # Raw p-value from chi-squared(1) distribution
        p_raw = float(1.0 - _stats.chi2.cdf(g2_val, df=1)) if g2_val > 0 else 1.0
        p_bonf = min(p_raw * n_tests, 1.0)
        r["p_value"] = round(p_raw, 6)
        r["p_bonferroni"] = round(p_bonf, 6)
        r["bonferroni_significant"] = p_bonf < 0.05

        if r["bonferroni_significant"]:
            stratum_shifting.append({
                "sign":        r["sign"],
                "g2":          r["g2"],
                "p_value":     r["p_value"],
                "p_bonferroni": r["p_bonferroni"],
                "direction":   r["bias"],   # "pre_biased" | "post_biased"
                "f_pre":       r["f_pre"],
                "f_post":      r["f_post"],
                "freq_pre_per_1k":  r["freq_pre_per_1k"],
                "freq_post_per_1k": r["freq_post_per_1k"],
                "log_odds":    r["log_odds"],
                "seen_in_both": r["seen_in_both"],
            })
        else:
            stable.append({
                "sign":    r["sign"],
                "g2":      r["g2"],
                "p_value": r["p_value"],
                "p_bonferroni": r["p_bonferroni"],
                "bias":    r["bias"],
                "seen_in_both": r["seen_in_both"],
            })

    log.info(
        "Bonferroni-significant stratum-shifting signs: %d / %d total",
        len(stratum_shifting), len(records),
    )

    # Build bipartite edges: each sign connects to itself on the other side
    # with weight = G².  High G² → strong stratum-shifting (thick edge).
    bipartite_edges: list[dict] = [
        {
            "sign":      r["sign"],
            "g2":        r["g2"],
            "direction": r["bias"],
            "is_shifting": r["bonferroni_significant"],
        }
        for r in records
        if r["g2"] > 0
    ]
    bipartite_edges.sort(key=lambda e: -e["g2"])

    summary = {
        "n_total_signs":        len(records),
        "n_stratum_shifting":   len(stratum_shifting),
        "n_stable":             len(stable),
        "bonferroni_alpha":     round(alpha_bonf, 8),
        "g2_bonferroni_threshold": round(chi2_bonf, 4),
        "n_pre_biased_shifting": sum(1 for s in stratum_shifting if s["direction"] == "pre_biased"),
        "n_post_biased_shifting": sum(1 for s in stratum_shifting if s["direction"] == "post_biased"),
    }

    result = {
        "stratum_shifting_signs": stratum_shifting,
        "stable_signs":           stable,
        "bipartite_edges":        bipartite_edges,
        "summary":                summary,
    }

    # Write the flat per-sign list (backward-compatible with CONTACT_JSON consumers)
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(records, indent=2), encoding="utf-8")
        log.info("Contact partition records → %s", output_path)

    if report_path is not None:
        _write_contact_partition_report(result, records, report_path)

    return result


def _write_contact_partition_report(
    partition: dict,
    records: list[dict],
    report_path: Path,
) -> None:
    """Write a self-contained HTML report for the contact partition analysis."""
    import html as _html_mod

    def _esc(s: object) -> str:
        return _html_mod.escape(str(s))

    summary = partition["summary"]
    shifting = partition["stratum_shifting_signs"]
    stable   = partition["stable_signs"]
    n_shift  = summary["n_stratum_shifting"]
    n_total  = summary["n_total_signs"]
    n_pre    = summary["n_pre_biased_shifting"]
    n_post   = summary["n_post_biased_shifting"]
    alpha    = summary["bonferroni_alpha"]
    g2_thr   = summary["g2_bonferroni_threshold"]

    css = """
:root{--bg:#0d0f12;--surface:#161920;--surface2:#1e2229;
      --border:#2a2e38;--text:#d0d4dc;--muted:#6b7280;
      --accent:#c4a96d;--green:#4ade80;--yellow:#facc15;
      --red:#f87171;--blue:#93c5fd;}
*{box-sizing:border-box;margin:0;padding:0;}
body{background:var(--bg);color:var(--text);
     font-family:'JetBrains Mono',monospace;font-size:13px;line-height:1.65;}
.wrap{max-width:1060px;margin:0 auto;padding:52px 28px 80px;}
h1{font-size:22px;color:var(--accent);margin-bottom:4px;}
.sub{color:var(--muted);font-size:11px;margin-bottom:36px;}
.section{margin-bottom:48px;}
.section-title{font-size:15px;font-weight:600;color:var(--text);
               border-bottom:1px solid var(--border);
               padding-bottom:8px;margin-bottom:16px;}
.stat-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(160px,1fr));
           gap:12px;margin-bottom:24px;}
.stat{background:var(--surface);border:1px solid var(--border);border-radius:5px;
      padding:14px 16px;}
.stat-label{font-size:9px;color:var(--muted);text-transform:uppercase;letter-spacing:.07em;}
.stat-value{font-size:26px;font-weight:600;margin-top:2px;color:var(--accent);}
.stat-sub{font-size:10px;color:var(--muted);margin-top:1px;}
table{width:100%;border-collapse:collapse;font-size:11px;margin-top:8px;}
th{padding:6px 10px;text-align:left;font-size:9px;color:var(--muted);
   border-bottom:1px solid var(--border);text-transform:uppercase;letter-spacing:.06em;}
td{padding:5px 10px;border-bottom:1px solid rgba(42,46,56,.4);}
tr:hover td{background:var(--surface2);}
.code{color:var(--accent);}  .hi{color:var(--green);}
.lo{color:var(--muted);}  .med{color:var(--yellow);}  .neg{color:var(--red);}
.pre{background:rgba(74,222,128,.1);color:var(--green);
     font-size:9px;padding:1px 6px;border-radius:3px;}
.post{background:rgba(147,197,253,.12);color:var(--blue);
      font-size:9px;padding:1px 6px;border-radius:3px;}
.neut{font-size:9px;color:var(--muted);}
.verdict{border-left:3px solid var(--accent);padding:14px 18px;
         background:var(--surface);border-radius:0 5px 5px 0;margin:20px 0;}
.verdict strong{color:var(--accent);}
"""

    stats_html = f"""
<div class="stat-grid">
  <div class="stat">
    <div class="stat-label">Total signs</div>
    <div class="stat-value">{n_total}</div>
  </div>
  <div class="stat">
    <div class="stat-label">Stratum-shifting</div>
    <div class="stat-value hi">{n_shift}</div>
    <div class="stat-sub">Bonferroni p &lt; 0.05</div>
  </div>
  <div class="stat">
    <div class="stat-label">Pre-biased shifts</div>
    <div class="stat-value">{n_pre}</div>
    <div class="stat-sub">enriched pre-contact</div>
  </div>
  <div class="stat">
    <div class="stat-label">Post-biased shifts</div>
    <div class="stat-value">{n_post}</div>
    <div class="stat-sub">enriched post-contact</div>
  </div>
  <div class="stat">
    <div class="stat-label">Stable signs</div>
    <div class="stat-value lo">{summary["n_stable"]}</div>
    <div class="stat-sub">no significant shift</div>
  </div>
  <div class="stat">
    <div class="stat-label">G² threshold (Bonf.)</div>
    <div class="stat-value med">{g2_thr:.2f}</div>
    <div class="stat-sub">&alpha;={alpha:.2e}</div>
  </div>
</div>
"""

    verdict_text = (
        f"{n_shift} of {n_total} signs show statistically significant frequency "
        f"shifts across the contact boundary (Bonferroni-corrected G² ≥ {g2_thr:.2f}). "
        f"{n_pre} are pre-biased (characteristic of Tablet D / pre-contact rongorongo); "
        f"{n_post} are post-biased (enriched in the post-contact corpus). "
        + ("The majority of signs are stable across the contact boundary, consistent "
           "with script continuity." if n_shift < n_total * 0.3 else
           "A large fraction of signs shift — suggesting substantial register change "
           "or scribal innovation at contact.")
    )

    def _shift_row(s: dict) -> str:
        dir_badge = (
            f'<span class="pre">PRE</span>' if s["direction"] == "pre_biased"
            else f'<span class="post">POST</span>'
        )
        return (
            f"<tr>"
            f'<td class="code">{_esc(s["sign"])}</td>'
            f"<td>{dir_badge}</td>"
            f"<td class=\"hi\">{s['g2']:.3f}</td>"
            f"<td>{s['p_value']:.2e}</td>"
            f"<td>{s['p_bonferroni']:.2e}</td>"
            f"<td>{s['f_pre']}</td>"
            f"<td>{s['f_post']}</td>"
            f"<td>{s['freq_pre_per_1k']:.1f}</td>"
            f"<td>{s['freq_post_per_1k']:.1f}</td>"
            f"<td>{s['log_odds']:+.3f}</td>"
            f"</tr>"
        )

    shift_rows = "".join(_shift_row(s) for s in shifting[:50])
    shift_table = (
        "<table><thead><tr>"
        "<th>Sign</th><th>Direction</th><th>G²</th>"
        "<th>p (raw)</th><th>p (Bonf.)</th>"
        "<th>f_pre</th><th>f_post</th>"
        "<th>pre/k</th><th>post/k</th><th>log-odds</th>"
        f"</tr></thead><tbody>{shift_rows}</tbody></table>"
    )

    from datetime import datetime, timezone
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    html = (
        "<!DOCTYPE html><html lang='en'>"
        "<head><meta charset='utf-8'>"
        "<title>Rongorongo — Contact Partition Analysis</title>"
        f"<style>{css}</style></head>"
        "<body><div class='wrap'>"
        "<h1>Contact Partition Analysis</h1>"
        f"<div class='sub'>G² sign-frequency partition: pre-contact vs post-contact · "
        f"Generated {_esc(generated)}</div>"
        f"<div class='section'><div class='section-title'>Summary</div>"
        f"{stats_html}"
        "<div class='verdict'><strong>Finding</strong>"
        f"<p style='font-size:12px;margin-top:6px'>{_esc(verdict_text)}</p></div>"
        "</div>"
        f"<div class='section'><div class='section-title'>"
        f"Stratum-Shifting Signs (Bonferroni p &lt; 0.05, top 50)</div>"
        f"{shift_table}</div>"
        "</div></body></html>"
    )

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(html, encoding="utf-8")
    log.info("Contact partition HTML report → %s", report_path)


# ---------------------------------------------------------------------------
# Bigram co-occurrence (for bipartite edge weights)
# ---------------------------------------------------------------------------

def load_cross_bigrams(
    corpus_dir: Path,
    pre_biased_signs: set[str],
    post_biased_signs: set[str],
    include_compound_components: bool = True,
) -> dict[tuple[str, str], int]:
    """Return bigram counts for adjacent (pre_biased, post_biased) sign pairs.

    Counts both (pre_sign, post_sign) and (post_sign, pre_sign) orderings;
    the key is always sorted (pre_biased_sign, post_biased_sign) so the
    caller gets a single count per pair regardless of ordering direction.
    """
    counts: Counter = Counter()
    for path in sorted(corpus_dir.glob("[A-Z].json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        tokens: list[str] = []
        for g in data["glyphs"]:
            hc = g.get("horley_code")
            if hc:
                tokens.append(hc)
            if include_compound_components:
                for comp_hc in (g.get("horley_components") or []):
                    tokens.append(comp_hc)
        for i in range(len(tokens) - 1):
            s1, s2 = tokens[i], tokens[i + 1]
            if s1 in pre_biased_signs and s2 in post_biased_signs:
                counts[(s1, s2)] += 1
            elif s1 in post_biased_signs and s2 in pre_biased_signs:
                counts[(s2, s1)] += 1  # normalise so key is always (pre, post)
    return dict(counts)


# ---------------------------------------------------------------------------
# Bipartite visualisation (plotly)
# ---------------------------------------------------------------------------

def _node_size(freq: int, scale: float = 14.0, min_size: float = 8.0, max_size: float = 40.0) -> float:
    import math as _math
    return min(max(min_size, scale * _math.sqrt(freq + 1)), max_size)


def write_bipartite_html(
    records: list[dict],
    cfg,
    output_path: Path,
    min_bigram_count: int = 1,
    max_neutral: int = 12,
) -> None:
    """Write a plotly bipartite HTML visualisation of the sign partition.

    Layout
    ------
    * Left column  (x = 0): pre_biased signs, sorted by G² desc
    * Right column (x = 1): post_biased signs, sorted by G² desc
    * Centre row   (x = 0.5, y below): top-N neutral high-frequency signs
    * Edges: cross-partition bigram co-occurrences; thickness ∝ log(count)
    """
    try:
        import plotly.graph_objects as go
    except ImportError:
        log.error("plotly is required for --plot. Install with: pip install plotly")
        raise

    corpus_dir = PROJECT_ROOT / cfg.paths.corpus_dir

    pre_b = sorted([r for r in records if r["bias"] == "pre_biased"], key=lambda r: -r["g2"])
    post_b = sorted([r for r in records if r["bias"] == "post_biased"], key=lambda r: -r["g2"])
    neutral_top = sorted(
        [r for r in records if r["bias"] == "neutral" and r["seen_in_both"]],
        key=lambda r: -(r["f_pre"] + r["f_post"]),
    )[:max_neutral]

    pre_set = {r["sign"] for r in pre_b}
    post_set = {r["sign"] for r in post_b}

    cross_bigrams = load_cross_bigrams(corpus_dir, pre_set, post_set)

    # Assign y-positions
    def _y_positions(n: int) -> list[float]:
        if n == 0:
            return []
        if n == 1:
            return [0.5]
        return [i / (n - 1) for i in range(n)]

    pre_ys = _y_positions(len(pre_b))
    post_ys = _y_positions(len(post_b))

    pos: dict[str, tuple[float, float]] = {}
    for r, y in zip(pre_b, pre_ys):
        pos[r["sign"]] = (0.0, y)
    for r, y in zip(post_b, post_ys):
        pos[r["sign"]] = (1.0, y)
    # Neutral signs: arrange in a row below the main columns
    n_neu = len(neutral_top)
    for k, r in enumerate(neutral_top):
        x_neu = k / max(n_neu - 1, 1)
        pos[r["sign"]] = (x_neu, -0.18)

    # Build edge traces
    edge_traces = []
    max_bg = max(cross_bigrams.values()) if cross_bigrams else 1
    for (pre_sign, post_sign), cnt in sorted(cross_bigrams.items(), key=lambda kv: -kv[1]):
        if cnt < min_bigram_count:
            continue
        if pre_sign not in pos or post_sign not in pos:
            continue
        x0, y0 = pos[pre_sign]
        x1, y1 = pos[post_sign]
        alpha = 0.15 + 0.65 * math.log1p(cnt) / math.log1p(max_bg)
        edge_traces.append(go.Scatter(
            x=[x0, x1, None], y=[y0, y1, None],
            mode="lines",
            line=dict(width=1.0 + 2.5 * math.log1p(cnt) / math.log1p(max_bg), color=f"rgba(150,150,150,{alpha:.2f})"),
            hoverinfo="none",
            showlegend=False,
        ))

    # Build node trace helper
    def _node_trace(sign_records, marker_color, group_label, symbol="circle"):
        xs, ys, sizes, labels, hover = [], [], [], [], []
        for r in sign_records:
            if r["sign"] not in pos:
                continue
            x, y = pos[r["sign"]]
            xs.append(x); ys.append(y)
            sizes.append(_node_size(max(r["f_pre"], r["f_post"])))
            labels.append(r["sign"])
            hover.append(
                f"{r['sign']}<br>G²={r['g2']:.1f}  ω={r['log_odds']:+.2f}"
                f"<br>pre={r['f_pre']} ({r['freq_pre_per_1k']:.1f}/k)"
                f"<br>post={r['f_post']} ({r['freq_post_per_1k']:.1f}/k)"
            )
        return go.Scatter(
            x=xs, y=ys, mode="markers+text",
            marker=dict(size=sizes, color=marker_color, symbol=symbol,
                        line=dict(width=1, color="white")),
            text=labels, textposition="middle right",
            textfont=dict(size=9),
            hovertext=hover, hoverinfo="text",
            name=group_label,
        )

    fig = go.Figure()
    for tr in edge_traces:
        fig.add_trace(tr)
    fig.add_trace(_node_trace(pre_b, "#D96A45", "pre-biased"))
    fig.add_trace(_node_trace(post_b, "#3E7DB5", "post-biased"))
    if neutral_top:
        fig.add_trace(_node_trace(neutral_top, "#888888", "neutral (top freq)", symbol="diamond"))

    # Column header annotations
    n_pre = sum(r["f_pre"] + r["f_post"] for r in records if r["bias"] == "pre_biased")
    n_post_total = sum(r["f_pre"] + r["f_post"] for r in records if r["bias"] == "post_biased")
    annotations = [
        dict(x=0.0, y=1.07, xref="x", yref="paper", text=f"<b>Pre-contact biased</b><br>({len(pre_b)} signs)",
             showarrow=False, font=dict(size=13, color="#D96A45"), xanchor="center"),
        dict(x=1.0, y=1.07, xref="x", yref="paper", text=f"<b>Post-contact biased</b><br>({len(post_b)} signs)",
             showarrow=False, font=dict(size=13, color="#3E7DB5"), xanchor="center"),
    ]
    if neutral_top:
        annotations.append(dict(
            x=0.5, y=-0.26, xref="x", yref="paper",
            text=f"<i>Top {len(neutral_top)} neutral signs (for context)</i>",
            showarrow=False, font=dict(size=10, color="#888888"), xanchor="center",
        ))

    fig.update_layout(
        title=dict(
            text="Rongorongo Sign Partition: Pre- vs Post-contact Bias<br>"
                 "<sup>Node size ∝ corpus frequency • Edges = bigram co-occurrences</sup>",
            x=0.5, xanchor="center", font=dict(size=15),
        ),
        xaxis=dict(visible=False, range=[-0.25, 1.25]),
        yaxis=dict(visible=False, range=[-0.35, 1.15]),
        plot_bgcolor="white",
        paper_bgcolor="white",
        showlegend=True,
        legend=dict(x=0.01, y=0.01, bgcolor="rgba(255,255,255,0.8)"),
        annotations=annotations,
        margin=dict(l=40, r=40, t=100, b=80),
        width=820, height=680,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(output_path), include_plotlyjs="cdn")
    log.info("Bipartite graph written to %s", output_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Contact analysis — pre/post sign frequency partitioning.",
    )
    parser.add_argument(
        "--min-g2",
        type=float,
        default=CHI2_P05,
        help=f"G² threshold for significance (default: {CHI2_P05}, p<0.05).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit full per-sign records as JSON.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Write per-sign records JSON to this path.",
    )
    parser.add_argument(
        "--plot",
        type=Path,
        default=None,
        metavar="PATH",
        help="Write plotly bipartite HTML visualisation to this path.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        metavar="PATH",
        help="Write structured contact partition + HTML report. "
             "Generates contact_partition_report.html alongside the JSON output. "
             "Overrides simple --output behaviour: writes structured JSON with "
             "stratum_shifting_signs / stable_signs / bipartite_edges keys plus "
             "the flat record list for backward compat.",
    )
    parser.add_argument(
        "--scenario",
        choices=[*_SCENARIO_NAMES, "all"],
        default=None,
        metavar="SCENARIO",
        help=(
            "Run under a named dating scenario for tablets with unknown cluster "
            "(conservative_all_late | optimistic_distributed | "
            "probabilistic_weighted | all). "
            "'all' runs the full three-scenario sensitivity analysis, writing "
            "results to --output (default: outputs/contact_sensitivity.json). "
            "Default: exclude unknown tablets (original behaviour)."
        ),
    )
    args = parser.parse_args()

    if args.scenario == "all":
        sensitivity_path = args.output or Path("outputs/contact_sensitivity.json")
        contact_sensitivity_analysis(min_g2=args.min_g2, output_path=sensitivity_path)
        return

    if args.report:
        # Structured mode: Bonferroni correction + HTML report
        try:
            build_contact_partition(
                min_g2=args.min_g2,
                scenario=args.scenario,
                output_path=args.output,
                report_path=args.report,
            )
        except ImportError:
            log.error(
                "scipy is required for --report (Bonferroni correction). "
                "Install with: pip install scipy"
            )
            sys.exit(1)
        return

    records = analyse(min_g2=args.min_g2, emit_json=args.json, scenario=args.scenario)

    if args.output:
        out_path = args.output
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(records, indent=2), encoding="utf-8")
        log.info("Contact partition records written to %s", out_path)

    if args.plot:
        cfg = OmegaConf.load(PROJECT_ROOT / "conf" / "config.yaml")
        write_bipartite_html(records, cfg, args.plot)


if __name__ == "__main__":
    main()
