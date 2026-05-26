"""Compute and render the Pozdniakov hypothesis summary report.

This script is self-contained so it can run even if the package-level
``hackingrongo.results`` import path is unavailable.

Outputs
-------
outputs/analysis/pozdniakov_hypothesis_tests.json
outputs/analysis/pozdniakov_hypothesis_report.html
outputs/analysis/pozdniakov_hypothesis_tests.png
outputs/analysis/pozdniakov_tablet_scores.png
"""

from __future__ import annotations

import importlib.util
import json
import math
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from omegaconf import OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "analysis"
RANKING_PATH = PROJECT_ROOT / "outputs" / "decipherment" / "ranking.json"
POST_LM_PATH = PROJECT_ROOT / "data" / "language_models" / "post_contact_lm.json"
REPORT_MODULE_PATH = PROJECT_ROOT / "hackingrongo" / "results" / "pozdniakov_report.py"
CFG = OmegaConf.load(PROJECT_ROOT / "conf" / "config.yaml")
RNG = np.random.default_rng(42)
SAMPLE_SIZE = 88
N_BOOT = 1000
N_NULL = 1000
HYPOTHESIS_ID = "H0001"


class TabletRecord:
    def __init__(self, tablet_id: str, stratum: str, date_midpoint: float, tokens: list[str]):
        self.tablet_id = tablet_id
        self.stratum = stratum
        self.date_midpoint = date_midpoint
        self.tokens = tokens


class CorpusRecord:
    def __init__(self, tablet_id: str, stratum: str, date_midpoint: float, tokens: list[Any]):
        self.tablet_id = tablet_id
        self.stratum = stratum
        self.date_midpoint = date_midpoint
        self.tokens = tokens


def _load_report_module():
    spec = importlib.util.spec_from_file_location("pozdniakov_report", REPORT_MODULE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load report module from {REPORT_MODULE_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _rankdata(values: list[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    order = np.argsort(arr, kind="mergesort")
    ranks = np.empty(len(arr), dtype=float)
    i = 0
    while i < len(arr):
        j = i
        while j + 1 < len(arr) and arr[order[j + 1]] == arr[order[i]]:
            j += 1
        ranks[order[i : j + 1]] = (i + j + 2) / 2.0
        i = j + 1
    return ranks


def spearman_corr(x: list[float], y: list[float]) -> float:
    if len(x) < 2 or len(y) < 2 or len(x) != len(y):
        return float("nan")
    rx = _rankdata(x)
    ry = _rankdata(y)
    if np.std(rx) == 0 or np.std(ry) == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def bootstrap_ci(values: list[float], alpha: float = 0.05) -> tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return (float("nan"), float("nan"))
    return (float(np.quantile(arr, alpha / 2.0)), float(np.quantile(arr, 1.0 - alpha / 2.0)))


def levenshtein_distance(a: list[str], b: list[str]) -> int:
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        curr = [i]
        for j, cb in enumerate(b, start=1):
            cost = 0 if ca == cb else 1
            curr.append(min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost))
        prev = curr
    return prev[-1]


def norm_levenshtein(a: list[str], b: list[str]) -> float:
    denom = max(len(a), len(b), 1)
    return levenshtein_distance(a, b) / denom


def fit_zipf_alpha(counts: list[int]) -> float:
    freqs = np.asarray(sorted(counts, reverse=True), dtype=float)
    freqs = freqs[freqs > 0]
    ranks = np.arange(1, len(freqs) + 1, dtype=float)
    if len(freqs) < 2:
        return float("nan")
    slope, _ = np.polyfit(np.log(ranks), np.log(freqs), 1)
    return float(-slope)


def _load_corpus():
    from hackingrongo.data.corpus import load_corpus, split_by_cluster
    from hackingrongo.data.rapa_nui_corpus import NGramLM
    from hackingrongo.zone_c.lm_scoring import LMScorer

    return load_corpus, split_by_cluster, NGramLM, LMScorer


def _extract_ranking() -> dict[str, str]:
    ranking = json.loads(RANKING_PATH.read_text(encoding="utf-8"))
    if isinstance(ranking, dict):
        hypotheses = ranking.get("hypotheses") or ranking.get("items") or []
    else:
        hypotheses = ranking
    hyp = None
    for candidate in hypotheses:
        if isinstance(candidate, dict) and candidate.get("hypothesis_id") == HYPOTHESIS_ID:
            hyp = candidate
            break
    if hyp is None and hypotheses:
        hyp = hypotheses[0]
    if hyp is None:
        raise RuntimeError(f"Could not find hypothesis {HYPOTHESIS_ID} in {RANKING_PATH}")
    assignments = hyp.get("assignments", [])
    phoneme_map = {}
    for assignment in assignments:
        if isinstance(assignment, dict):
            sign_code = str(assignment.get("sign_code", ""))
            phoneme = str(assignment.get("phoneme", ""))
            if sign_code and phoneme:
                phoneme_map[sign_code] = phoneme
    if not phoneme_map:
        raise RuntimeError(f"No assignments found in hypothesis {HYPOTHESIS_ID}")
    return phoneme_map


def _load_post_lm(NGramLM):
    post_lm = NGramLM.load(POST_LM_PATH)
    ref_counts = dict(post_lm._counts[1].get((), {}))
    ref_counts = {k: int(v) for k, v in ref_counts.items() if not str(k).startswith("<") and v > 0}
    if not ref_counts:
        raise RuntimeError("Could not extract unigram frequencies from post_contact LM")
    return ref_counts


def _tokens_by_stratum(all_tablets):
    by_stratum: dict[str, list[str]] = defaultdict(list)
    for tablet in all_tablets:
        for tok in tablet.tokens:
            code = getattr(tok, "barthel_code", None)
            if code is not None:
                by_stratum[tablet.stratum].append(str(code))
    return by_stratum


def _sign_to_phoneme_rho(tokens: list[str], phoneme_map: dict[str, str], ref_phoneme_freq: dict[str, int]) -> float:
    sign_counts = Counter(tokens)
    if not sign_counts:
        return float("nan")
    sign_order = sorted(sign_counts, key=lambda s: (-sign_counts[s], s))
    sign_rank = {s: i + 1 for i, s in enumerate(sign_order)}
    x: list[float] = []
    y: list[float] = []
    for sign in sign_order:
        ph = phoneme_map.get(sign)
        if ph is None or ph not in ref_phoneme_freq:
            continue
        x.append(sign_rank[sign])
        y.append(ref_phoneme_freq[ph])
    return spearman_corr(x, y) if len(x) >= 5 else float("nan")


def _lm_score_tablet(tablet, phoneme_map, lm_scorer):
    seq = [phoneme_map.get(getattr(tok, "barthel_code", ""), "<UNK>") for tok in tablet.tokens]
    return float(lm_scorer.score(seq).ensemble_log_prob)


def _parallel_passages_path() -> Path:
    candidates = [
        PROJECT_ROOT / "data" / "parallels" / "parallel_variants_auto.json",
        PROJECT_ROOT / "data" / "parallels" / "parallel_variants.json",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError("No parallel passage JSON found")


def compute_results() -> dict[str, Any]:
    os.chdir(PROJECT_ROOT)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    load_corpus, split_by_cluster, NGramLM, LMScorer = _load_corpus()
    phoneme_map = _extract_ranking()
    all_tablets = load_corpus(CFG, PROJECT_ROOT)
    by_stratum = split_by_cluster(all_tablets)
    ref_counts = _load_post_lm(NGramLM)
    ref_phoneme_order = [ph for ph, _ in sorted(ref_counts.items(), key=lambda kv: (-kv[1], kv[0]))]
    lm_scorer = LMScorer(CFG, PROJECT_ROOT)

    pre_tokens = [tok.barthel_code for tablet in by_stratum.get("pre_contact", []) for tok in tablet.tokens if getattr(tok, "barthel_code", None) is not None]
    post_tokens = [tok.barthel_code for tablet in by_stratum.get("post_contact", []) for tok in tablet.tokens if getattr(tok, "barthel_code", None) is not None]
    unknown_tokens = [tok.barthel_code for tablet in by_stratum.get("unknown", []) for tok in tablet.tokens if getattr(tok, "barthel_code", None) is not None]

    if len(pre_tokens) < SAMPLE_SIZE:
        raise RuntimeError(f"Need at least {SAMPLE_SIZE} pre-contact tokens, found {len(pre_tokens)}")
    if len(post_tokens) < SAMPLE_SIZE:
        raise RuntimeError(f"Need at least {SAMPLE_SIZE} post-contact tokens, found {len(post_tokens)}")

    pre_sample = pre_tokens[:SAMPLE_SIZE]
    pre_rho = _sign_to_phoneme_rho(pre_sample, phoneme_map, ref_counts)
    pre_boot = []
    for _ in range(N_BOOT):
        sample = RNG.choice(pre_tokens, size=SAMPLE_SIZE, replace=True).tolist()
        pre_boot.append(_sign_to_phoneme_rho(sample, phoneme_map, ref_counts))
    pre_boot = [v for v in pre_boot if np.isfinite(v)]
    pre_ci = bootstrap_ci(pre_boot)

    post_boot = []
    for _ in range(N_BOOT):
        sample = RNG.choice(post_tokens, size=SAMPLE_SIZE, replace=True).tolist()
        post_boot.append(_sign_to_phoneme_rho(sample, phoneme_map, ref_counts))
    post_boot = [v for v in post_boot if np.isfinite(v)]
    post_ci = bootstrap_ci(post_boot)
    post_full_rho = _sign_to_phoneme_rho(post_tokens, phoneme_map, ref_counts)
    pre_vs_post_p = float(np.mean(np.asarray(post_boot) >= pre_rho)) if pre_boot else float("nan")
    post_vs_pre_p = float(np.mean(np.asarray(pre_boot) >= post_full_rho)) if pre_boot else float("nan")

    def hapax_rate(tokens: list[str]) -> dict[str, float]:
        counts = Counter(tokens)
        n_types = len(counts)
        n_hapax = sum(1 for c in counts.values() if c == 1)
        return {
            "n_tokens": len(tokens),
            "n_types": n_types,
            "n_hapax": n_hapax,
            "hapax_rate": n_hapax / n_types if n_types else float("nan"),
        }

    hapax = {
        "pre_contact": hapax_rate(pre_tokens),
        "post_contact": hapax_rate(post_tokens),
        "unknown": hapax_rate(unknown_tokens),
    }

    parallel_path = _parallel_passages_path()
    parallel_data = json.loads(parallel_path.read_text(encoding="utf-8"))
    passages = parallel_data.get("passages", parallel_data if isinstance(parallel_data, list) else [])
    pre_post_dists: list[float] = []
    post_post_dists: list[float] = []
    passage_stability_rows: list[dict[str, Any]] = []
    for entry in passages:
        if not isinstance(entry, dict):
            continue
        attestations = entry.get("attestations", entry.get("variants", []))
        by = defaultdict(list)
        for att in attestations:
            if not isinstance(att, dict):
                continue
            form = att.get("form", att.get("glyphs", []))
            if not form:
                continue
            seq = [str(g) for g in form]
            stratum = att.get("stratum", "unknown")
            by[stratum].append(seq)
        pre_forms = by.get("pre_contact", [])
        post_forms = by.get("post_contact", [])
        if pre_forms and post_forms:
            pp = [norm_levenshtein(a, b) for a in pre_forms for b in post_forms]
            qq = [norm_levenshtein(a, b) for i, a in enumerate(post_forms) for b in post_forms[i + 1 :]]
            if pp:
                pre_post_dists.extend(pp)
            if qq:
                post_post_dists.extend(qq)
            passage_stability_rows.append(
                {
                    "passage_id": entry.get("passage_id", entry.get("id", "?")),
                    "n_pre": len(pre_forms),
                    "n_post": len(post_forms),
                    "pre_post_mean": float(np.mean(pp)) if pp else float("nan"),
                    "post_post_mean": float(np.mean(qq)) if qq else float("nan"),
                }
            )

    tablet_rows = []
    for tablet in all_tablets:
        if getattr(tablet, "stratum", None) == "excluded":
            continue
        score = _lm_score_tablet(tablet, phoneme_map, lm_scorer)
        tablet_rows.append(
            {
                "tablet_id": tablet.tablet_id,
                "stratum": tablet.stratum,
                "date_midpoint": float(tablet.date_midpoint),
                "n_tokens": len(tablet.tokens),
                "lm_score": score,
            }
        )
    tablet_rows.sort(key=lambda r: r["date_midpoint"])
    dates = [r["date_midpoint"] for r in tablet_rows]
    score_series = [r["lm_score"] for r in tablet_rows]
    score_by_date_rho = spearman_corr(dates, score_series)

    post_counts = Counter(post_tokens)
    alpha_zipf = fit_zipf_alpha(list(post_counts.values()))
    post_n_tokens = len(post_tokens)
    if not np.isfinite(alpha_zipf):
        raise RuntimeError("Could not fit Zipf alpha for post-contact corpus")

    ranks = np.arange(1, len(post_counts) + 1, dtype=float)
    zipf_probs = ranks ** (-alpha_zipf)
    zipf_probs = zipf_probs / zipf_probs.sum()
    ref_phoneme_freqs = np.array([ref_counts[ph] for ph in ref_phoneme_order], dtype=float)
    ref_phoneme_freqs = ref_phoneme_freqs / ref_phoneme_freqs.sum()

    null_rhos = []
    for _ in range(N_NULL):
        sampled_counts = RNG.multinomial(post_n_tokens, zipf_probs)
        sampled_counts = sampled_counts[sampled_counts > 0]
        sign_freqs = sampled_counts.tolist()
        perm = RNG.permutation(len(ref_phoneme_order))
        if len(sign_freqs) > len(ref_phoneme_order):
            ph_freqs = np.resize(ref_phoneme_freqs[perm], len(sign_freqs))
        else:
            ph_freqs = ref_phoneme_freqs[perm[: len(sign_freqs)]]
        rho = spearman_corr(sign_freqs, ph_freqs.tolist())
        if np.isfinite(rho):
            null_rhos.append(rho)

    observed_post_rho = post_full_rho
    null_mean = float(np.mean(null_rhos)) if null_rhos else float("nan")
    null_ci = bootstrap_ci(null_rhos)
    p_null_ge_obs = float(np.mean(np.asarray(null_rhos) >= observed_post_rho)) if null_rhos else float("nan")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    ax = axes[0, 0]
    ax.hist(post_boot, bins=40, alpha=0.75, color="#4C72B0", label="post bootstrap")
    ax.axvline(pre_rho, color="#C44E52", linestyle="--", linewidth=2, label="pre rho")
    ax.axvline(np.mean(post_boot), color="#55A868", linewidth=2, label="post mean")
    ax.set_title("Test 1: matched-size frequency correlation")
    ax.set_xlabel("Spearman rho")
    ax.set_ylabel("Count")
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    labels = list(hapax)
    rates = [hapax[k]["hapax_rate"] for k in labels]
    ax.bar(labels, rates, color=["#4C72B0", "#55A868", "#C44E52"])
    ax.set_ylim(0, 1)
    ax.set_title("Test 2: hapax rate by stratum")
    ax.tick_params(axis="x", rotation=20)

    ax = axes[1, 0]
    if pre_post_dists and post_post_dists:
        ax.hist(pre_post_dists, bins=20, alpha=0.7, label="pre-post", color="#C44E52")
        ax.hist(post_post_dists, bins=20, alpha=0.7, label="post-post", color="#4C72B0")
    ax.set_title("Test 3: passage stability")
    ax.set_xlabel("Normalized edit distance")
    ax.set_ylabel("Count")
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    if null_rhos:
        ax.hist(null_rhos, bins=40, alpha=0.75, color="#7B6FE8", label="Zipf null")
        ax.axvline(observed_post_rho, color="#C44E52", linewidth=2, label="observed post-contact")
    ax.set_title("Test 5: Zipf null model")
    ax.set_xlabel("Spearman rho")
    ax.set_ylabel("Count")
    ax.legend(fontsize=8)

    fig.tight_layout()
    fig_path = OUTPUT_DIR / "pozdniakov_hypothesis_tests.png"
    fig.savefig(fig_path, dpi=160, bbox_inches="tight")
    plt.close(fig)

    fig2, ax2 = plt.subplots(figsize=(12, 6))
    colors = [
        "#4C72B0" if r["stratum"] == "pre_contact" else "#55A868" if r["stratum"] == "post_contact" else "#888888"
        for r in tablet_rows
    ]
    ax2.scatter(dates, score_series, c=colors, s=60, alpha=0.85, edgecolor="white", linewidth=0.5)
    ax2.set_title("Test 4: LM score by tablet under H0001")
    ax2.set_xlabel("Radiocarbon midpoint (CE)")
    ax2.set_ylabel("H0001 LM score")
    ax2.grid(alpha=0.2)
    fig2.tight_layout()
    score_fig_path = OUTPUT_DIR / "pozdniakov_tablet_scores.png"
    fig2.savefig(score_fig_path, dpi=160, bbox_inches="tight")
    plt.close(fig2)

    results = {
        "hypothesis_id": HYPOTHESIS_ID,
        "sample_size": SAMPLE_SIZE,
        "n_bootstrap": N_BOOT,
        "n_null": N_NULL,
        "test1": {
            "pre_rho": pre_rho,
            "pre_ci": pre_ci,
            "post_boot_mean": float(np.mean(post_boot)) if post_boot else float("nan"),
            "post_ci": post_ci,
            "post_full_rho": post_full_rho,
            "p_post_boot_ge_pre": pre_vs_post_p,
            "p_pre_boot_ge_post_full": post_vs_pre_p,
        },
        "test2": hapax,
        "test3": {
            "n_passages_with_pre_and_post": len(passage_stability_rows),
            "mean_pre_post_edit_distance": float(np.mean(pre_post_dists)) if pre_post_dists else None,
            "mean_post_post_edit_distance": float(np.mean(post_post_dists)) if post_post_dists else None,
            "rows": passage_stability_rows,
        },
        "test4": {
            "spearman_rho_date_score": score_by_date_rho,
            "tablets": tablet_rows,
        },
        "test5": {
            "alpha_zipf_post_contact": alpha_zipf,
            "observed_post_rho": observed_post_rho,
            "null_mean_rho": null_mean,
            "null_ci": null_ci,
            "p_null_ge_observed": p_null_ge_obs,
        },
        "artifacts": {
            "summary_plot": str(fig_path),
            "tablet_score_plot": str(score_fig_path),
            "parallel_path": str(parallel_path),
        },
    }

    results_path = OUTPUT_DIR / "pozdniakov_hypothesis_tests.json"
    results_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    return results_path


def main() -> int:
    results_path = compute_results()
    module = _load_report_module()
    html_path = OUTPUT_DIR / "pozdniakov_hypothesis_report.html"
    module.save_pozdniakov_report(results_path, html_path)
    print(str(results_path))
    print(str(html_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
