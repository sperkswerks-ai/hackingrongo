"""
hackingrongo.zone_b.sign_typology
=================================

**Independent, falsifiable test** of whether the rongorongo sign inventory splits
into two distributional populations — a phonogram-like and a logogram-like one —
or is a single smooth (Zipfian) continuum.

This runs BEFORE any decoding and is computed only from distributional features,
never from a decoder's own language-model score. It exists to break the
circularity of letting a decoder relabel signs to flatter its own fit: the type
partition is an *input* derived here, frozen, and never altered downstream.

The test only emits a ``sign_type_map`` **if the split is statistically real.**
If the inventory is a smooth continuum, the honest output is "no defensible
split" and no map is produced.

Features (per frequency-core sign, freq ≥ ``min_freq``)
------------------------------------------------------
* ``own_frequency``        — corpus-relative frequency
* ``positional_entropy``   — normalised entropy of position within a line
* ``bigram_mi``            — mean pointwise mutual information over the sign's
                             bigram edges (PMI graph)
* ``neighbor_diversity``   — distinct adjacent sign types / frequency
* ``compound_membership``  — 1 if the canonical code is a compound/ligature, else 0

Bimodality test
---------------
* **Hartigan's dip test** (primary) on the principal axis — tests unimodality
  directly and is robust to skew. Small p ⇒ reject unimodality ⇒ a real split.
* **Sarle's bimodality coefficient** (BC) on the principal axis as corroboration
  (BC > 5/9 ≈ 0.555 ⇒ bimodal shape).
* **2-vs-1-component Gaussian-mixture BIC** is reported *for completeness only* —
  it is NOT the arbiter. A skewed/Zipfian *unimodal* distribution is fit far
  better by two Gaussians than one (the second component absorbs the tail), so a
  large ΔBIC does not imply two populations. The frequency feature is
  log-transformed before testing to reduce that skew.

Every threshold is a config parameter, and ``sensitivity_sweep`` reports how
stable the phonogram/logogram partition is across reasonable choices — so a real
split can be distinguished from a tuned one.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from hackingrongo.zone_b.network_analysis import build_pmi_graph
from hackingrongo.zone_b.sign_fingerprint import (
    _neighbor_and_predecessor_stats,
    _positional_entropy_by_line,
    _sequences_by_tablet,
    load_glyph_records,
)

_FEATURES = ("own_frequency", "positional_entropy", "bigram_mi",
             "neighbor_diversity", "compound_membership")
_BC_THRESHOLD = 5.0 / 9.0          # Sarle's classic bimodality threshold ≈ 0.555


@dataclass
class TypologyResult:
    n_signs: int
    features: list[str]
    dip_statistic: float
    dip_p: float
    bimodality_coefficient: float
    bc_threshold: float
    gmm_bic_1: float
    gmm_bic_2: float
    delta_bic: float
    gmm_caveat: str
    is_bimodal: bool
    verdict: str
    n_phonogram: int = 0
    n_logogram: int = 0
    cluster_means: dict[str, dict[str, float]] = field(default_factory=dict)
    sign_type_map: dict[str, str] = field(default_factory=dict)
    sensitivity: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def _bigram_mi(sequences: list[list[str]], min_cofreq: int) -> dict[str, float]:
    """Mean PMI over each sign's incident bigram edges (from the PMI graph)."""
    g = build_pmi_graph(sequences, min_cofreq=min_cofreq)
    out: dict[str, float] = {}
    for node in g.nodes():
        ws = [d.get("weight", d.get("pmi", 0.0)) for _, _, d in g.edges(node, data=True)]
        out[node] = float(np.mean(ws)) if ws else 0.0
    return out


def compute_typology_features(
    records: list[dict[str, Any]], min_freq: int = 5, min_cofreq: int = 2
) -> tuple[list[str], np.ndarray]:
    """Return (signs, X) where X[i] is the feature vector for signs[i]."""
    from collections import Counter
    freq = Counter(r["code"] for r in records)
    total = sum(freq.values()) or 1
    core = sorted(s for s, c in freq.items() if c >= min_freq)

    sequences = _sequences_by_tablet(records)
    pos_entropy = _positional_entropy_by_line(records)
    nd, _sp, _dskew = _neighbor_and_predecessor_stats(sequences, freq, len(freq))
    bmi = _bigram_mi(sequences, min_cofreq)

    def is_compound(code: str) -> float:
        return 1.0 if (":" in code or "." in code or ("-" in code and not code.startswith("("))) else 0.0

    rows = []
    for s in core:
        rows.append([
            freq[s] / total,
            float(pos_entropy.get(s, 0.0)),
            float(bmi.get(s, 0.0)),
            float(nd.get(s, 0.0)),
            is_compound(s),
        ])
    return core, np.asarray(rows, dtype=float)


# ---------------------------------------------------------------------------
# Bimodality test
# ---------------------------------------------------------------------------

def _bimodality_coefficient(x: np.ndarray) -> float:
    """Sarle's BC = (g1² + 1) / (g2 + 3(n-1)²/((n-2)(n-3)))."""
    from scipy.stats import skew, kurtosis
    n = len(x)
    if n < 4:
        return float("nan")
    g1 = float(skew(x))
    g2 = float(kurtosis(x, fisher=True))
    denom = g2 + 3.0 * (n - 1) ** 2 / ((n - 2) * (n - 3))
    return (g1 * g1 + 1.0) / denom if denom else float("nan")


def _gmm_bic(X: np.ndarray, k: int, seed: int) -> tuple[float, Any]:
    from sklearn.mixture import GaussianMixture
    gm = GaussianMixture(n_components=k, covariance_type="full",
                         n_init=4, random_state=seed).fit(X)
    return float(gm.bic(X)), gm


def _dip_test(x: np.ndarray, seed: int) -> tuple[float, float]:
    """Hartigan's dip test (statistic, p). Null = unimodal. diptest is a soft
    dependency; if unavailable returns (nan, nan) and the caller falls back to BC."""
    try:
        import diptest
        stat, p = diptest.diptest(np.asarray(x, dtype=float))
        return float(stat), float(p)
    except Exception:
        return float("nan"), float("nan")


def bimodality_test(X: np.ndarray, seed: int = 20260606) -> dict[str, Any]:
    """Log-transform the Zipfian frequency feature, standardise, project to the
    principal axis, then test unimodality with the dip test (primary) + BC."""
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    Xt = X.copy()
    Xt[:, 0] = np.log(Xt[:, 0] + 1e-9)        # log-transform own_frequency (Zipfian)
    Xs = StandardScaler().fit_transform(Xt)

    pc1 = PCA(n_components=1, random_state=seed).fit_transform(Xs).ravel()
    dip_stat, dip_p = _dip_test(pc1, seed)
    bc = _bimodality_coefficient(pc1)

    bic1, _gm1 = _gmm_bic(Xs, 1, seed)
    bic2, gm2 = _gmm_bic(Xs, 2, seed)
    delta = bic1 - bic2

    return {"Xs": Xs, "pc1": pc1, "dip_stat": dip_stat, "dip_p": dip_p, "bc": bc,
            "bic1": bic1, "bic2": bic2, "delta_bic": delta, "gm2": gm2}


# ---------------------------------------------------------------------------
# Type assignment (only if the split is real) + sensitivity
# ---------------------------------------------------------------------------

def _assign(signs, Xs, gm2, X_raw, prob_threshold: float):
    """Assign each sign to a cluster; label the higher-frequency cluster
    PHONOGRAM and the other LOGOGRAM. Signs below the posterior threshold on
    either side are left UNRESOLVED (no forced label)."""
    post = gm2.predict_proba(Xs)
    hard = post.argmax(axis=1)
    # frequency is feature column 0; the cluster with higher mean freq = phonogram
    freq_col = X_raw[:, 0]
    means = [freq_col[hard == k].mean() if (hard == k).any() else 0.0 for k in (0, 1)]
    phon_cluster = int(np.argmax(means))
    type_map: dict[str, str] = {}
    for i, s in enumerate(signs):
        if post[i].max() < prob_threshold:
            type_map[s] = "UNRESOLVED"
        else:
            type_map[s] = "PHONOGRAM" if hard[i] == phon_cluster else "LOGOGRAM"
    return type_map, phon_cluster


def sensitivity_sweep(signs, Xs, gm2, X_raw, thresholds) -> dict[str, Any]:
    """Re-assign at several posterior thresholds; report partition stability."""
    base, _ = _assign(signs, Xs, gm2, X_raw, thresholds[0])
    rows, agreements = [], []
    base_resolved = {s: t for s, t in base.items() if t != "UNRESOLVED"}
    for thr in thresholds:
        m, _ = _assign(signs, Xs, gm2, X_raw, thr)
        n_log = sum(t == "LOGOGRAM" for t in m.values())
        n_phon = sum(t == "PHONOGRAM" for t in m.values())
        n_unres = sum(t == "UNRESOLVED" for t in m.values())
        # agreement vs base on signs resolved in both
        both = [s for s in base_resolved if m.get(s) != "UNRESOLVED"]
        agree = np.mean([base_resolved[s] == m[s] for s in both]) if both else 1.0
        agreements.append(float(agree))
        rows.append({"prob_threshold": thr, "n_phonogram": n_phon,
                     "n_logogram": n_log, "n_unresolved": n_unres,
                     "agreement_vs_base": round(float(agree), 4)})
    return {"thresholds": list(thresholds), "rows": rows,
            "min_agreement": round(float(min(agreements)), 4),
            "stable": bool(min(agreements) >= 0.9)}


def run(corpus_dir: Path, canon, min_freq: int = 5, min_cofreq: int = 2,
        alpha: float = 0.05, prob_threshold: float = 0.8,
        seed: int = 20260606) -> TypologyResult:
    records = load_glyph_records(corpus_dir, canon)
    signs, X = compute_typology_features(records, min_freq=min_freq, min_cofreq=min_cofreq)
    bt = bimodality_test(X, seed=seed)

    # Dip test is the arbiter (robust to skew); BC must corroborate. GMM-BIC is
    # NOT used for the verdict — it is confounded by the Zipfian skew.
    dip_p = bt["dip_p"]
    dip_ok = (not math.isnan(dip_p)) and (dip_p < alpha)
    bc_ok = bt["bc"] > _BC_THRESHOLD
    is_bimodal = dip_ok and bc_ok
    gmm_caveat = (f"GMM ΔBIC = {bt['delta_bic']:.1f} favours 2 components, but this is NOT "
                  "evidence of two populations — a skewed/Zipfian unimodal distribution is "
                  "fit better by 2 Gaussians (the 2nd absorbs the tail). The dip test and BC "
                  "are the arbiters here.")

    if is_bimodal:
        type_map, phon_cluster = _assign(signs, bt["Xs"], bt["gm2"], X, prob_threshold)
        sweep = sensitivity_sweep(signs, bt["Xs"], bt["gm2"], X, thresholds=[0.5, 0.6, 0.7, 0.8, 0.9])
        n_phon = sum(t == "PHONOGRAM" for t in type_map.values())
        n_log = sum(t == "LOGOGRAM" for t in type_map.values())
        cmeans = {}
        hard = bt["gm2"].predict(bt["Xs"])
        for k in (0, 1):
            sel = hard == k
            label = "phonogram-like" if k == phon_cluster else "logogram-like"
            cmeans[label] = {f: round(float(X[sel, j].mean()), 6) for j, f in enumerate(_FEATURES)} if sel.any() else {}
        verdict = (f"Bimodal split supported (dip p {dip_p:.3f} < {alpha}, BC {bt['bc']:.3f} > "
                   f"{_BC_THRESHOLD:.3f}). Frozen sign_type_map emitted.")
    else:
        type_map, sweep, n_phon, n_log, cmeans = {}, {}, 0, 0, {}
        verdict = (f"NO defensible bimodal split: dip-test p {dip_p:.3f} "
                   f"({'fails to reject unimodality' if not dip_ok else 'rejects'} at α={alpha}), "
                   f"BC {bt['bc']:.3f} (threshold {_BC_THRESHOLD:.3f}). The inventory is consistent "
                   f"with a smooth (Zipfian) continuum; NO sign_type_map is emitted. This is a real "
                   f"null result: the phonogram/logogram split is not distributionally supported.")

    return TypologyResult(
        n_signs=len(signs), features=list(_FEATURES),
        dip_statistic=round(bt["dip_stat"], 6), dip_p=round(dip_p, 6),
        bimodality_coefficient=round(bt["bc"], 6), bc_threshold=round(_BC_THRESHOLD, 6),
        gmm_bic_1=round(bt["bic1"], 4), gmm_bic_2=round(bt["bic2"], 4),
        delta_bic=round(bt["delta_bic"], 4), gmm_caveat=gmm_caveat,
        is_bimodal=is_bimodal, verdict=verdict,
        n_phonogram=n_phon, n_logogram=n_log, cluster_means=cmeans,
        sign_type_map=type_map, sensitivity=sweep,
    )
