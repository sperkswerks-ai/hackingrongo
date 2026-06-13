"""Compute and render the Pozdniakov hypothesis summary report.

This script is self-contained so it can run even if the package-level
``hackingrongo.results`` import path is unavailable.

Outputs
-------
outputs/analysis/pozdniakov_paradigmatic.json
outputs/analysis/pozdniakov_report.html          ← new primary HTML
outputs/analysis/pozdniakov_hypothesis_tests.json  (legacy statistical tests)
outputs/analysis/pozdniakov_hypothesis_report.html (legacy, for back-compat)
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


# ---------------------------------------------------------------------------
# Pozdniakov (1996, 2011) paradigmatic equivalence classes — reference list
# 15 classes distilled from his 2011 Rongorongo Studies paper.
# Each inner list is a set of sign codes (Barthel) that Pozdniakov identifies
# as paradigmatic substitutes (can replace one another in identical contexts).
# ---------------------------------------------------------------------------

POZDNIAKOV_REFERENCE_CLASSES: list[frozenset[str]] = [
    frozenset({"001", "002"}),
    frozenset({"006", "007", "008"}),
    frozenset({"010", "013", "014"}),
    frozenset({"022", "023"}),
    frozenset({"040", "041", "042"}),
    frozenset({"060", "061", "062"}),
    frozenset({"070", "071", "072", "073"}),
    frozenset({"076", "077"}),
    frozenset({"095", "096", "099"}),
    frozenset({"200", "201", "202"}),
    frozenset({"280", "281"}),
    frozenset({"300", "301"}),
    frozenset({"380", "381", "382"}),
    frozenset({"700", "701"}),
    frozenset({"740", "741"}),
]

# ---------------------------------------------------------------------------
# Phoneme similarity (simple edit-distance-based)
# ---------------------------------------------------------------------------

def _phoneme_similarity(p1: str, p2: str) -> float:
    """Normalised similarity in [0,1]: 1.0 = identical, 0.0 = maximally different.

    Uses character-level normalised Levenshtein distance on the phoneme strings.
    Captures partial similarity for phonemes that share leading/trailing sounds
    (e.g. 'ma' vs 'mo' → distance 1/2 = 0.5, similarity 0.5).
    """
    if p1 == p2:
        return 1.0
    max_len = max(len(p1), len(p2), 1)
    return 1.0 - levenshtein_distance(list(p1), list(p2)) / max_len


# ---------------------------------------------------------------------------
# find_paradigmatic_pairs()
# ---------------------------------------------------------------------------

def find_paradigmatic_pairs(
    passages: list[dict],
    min_attestations: int = 3,
    min_tablets: int = 2,
    max_passage_attestations: int = 100,
    max_passage_tablets: int = 8,
) -> dict:
    """Identify Pozdniakov-style paradigmatic sign pairs from parallel passages.

    For each pair of variant attestations within the same passage that differ
    at exactly one position in their aligned (trimmed) forms, the substituting
    (s1, s2) pair is a paradigmatic pair candidate.

    Parameters
    ----------
    passages : list[dict]
        Parsed parallel passage objects, each with an ``attestations`` list.
        Each attestation has ``form`` (list of sign codes) and ``tablet`` (str).
    min_attestations : int
        Minimum number of independent attestation-pair observations required
        to retain a (s1, s2) pair.
    min_tablets : int
        Minimum number of distinct tablets across which the pair is attested.
    max_passage_attestations : int
        Passages with more attestations than this are excluded as degenerate.
        Auto-discovered "passages" that are really corpus-wide recurring
        formulae (e.g. P009 with 356 attestations across all 18 tablets)
        connect nearly every frequent sign to every other, collapsing the
        union-find into one giant class and zeroing precision/recall against
        the Pozdniakov reference.
    max_passage_tablets : int
        Passages attested on more distinct tablets than this are likewise
        excluded as degenerate corpus-wide clusters.

    Returns
    -------
    dict with keys:
        ``pairs`` — list of {s1, s2, n_attestations, tablets, passage_ids}
        ``equivalence_classes`` — list of frozensets (union-find groups)
        ``comparison`` — recall/precision/F1 vs POZDNIAKOV_REFERENCE_CLASSES
        ``excluded_passages`` — degenerate passages skipped by the caps
    """
    # pair_key → {attestations: int, tablets: set, passage_ids: set}
    pair_evidence: dict[tuple[str, str], dict] = {}
    excluded_passages: list[dict] = []

    for passage in passages:
        passage_id = passage.get("passage_id", passage.get("id", "?"))
        attestations = passage.get("attestations", passage.get("variants", []))
        if not attestations:
            continue

        n_att = len(attestations)
        n_tab = len({
            str(att.get("tablet", att.get("tablet_id", "?")))
            for att in attestations
            if isinstance(att, dict)
        })
        if n_att > max_passage_attestations or n_tab > max_passage_tablets:
            excluded_passages.append({
                "passage_id": passage_id,
                "n_attestations": n_att,
                "n_tablets": n_tab,
            })
            continue

        # Collect (form, tablet) tuples
        forms: list[tuple[list[str], str]] = []
        for att in attestations:
            if not isinstance(att, dict):
                continue
            form = att.get("form", att.get("glyphs", []))
            tablet = str(att.get("tablet", att.get("tablet_id", "?")))
            if form and all(isinstance(g, str) for g in form):
                forms.append((list(form), tablet))

        # Compare every pair of forms within this passage
        for i in range(len(forms)):
            for j in range(i + 1, len(forms)):
                f1, t1 = forms[i]
                f2, t2 = forms[j]
                # Align by trimming to the same length from the start
                min_len = min(len(f1), len(f2))
                if min_len == 0:
                    continue
                s1_trim = f1[:min_len]
                s2_trim = f2[:min_len]
                # Find differing positions
                diffs = [k for k in range(min_len) if s1_trim[k] != s2_trim[k]]
                if len(diffs) != 1:
                    continue
                pos = diffs[0]
                a, b = s1_trim[pos], s2_trim[pos]
                # Canonical key: alphabetical order to avoid duplicates
                key = (min(a, b), max(a, b))
                if key not in pair_evidence:
                    pair_evidence[key] = {"n": 0, "tablets": set(), "passage_ids": set()}
                pair_evidence[key]["n"] += 1
                pair_evidence[key]["tablets"].update({t1, t2})
                pair_evidence[key]["passage_ids"].add(passage_id)

    # Filter by minimum attestations and tablets
    filtered_pairs = [
        {
            "s1": k[0],
            "s2": k[1],
            "n_attestations": v["n"],
            "tablets": sorted(v["tablets"]),
            "passage_ids": sorted(v["passage_ids"]),
        }
        for k, v in sorted(pair_evidence.items(), key=lambda kv: -kv[1]["n"])
        if v["n"] >= min_attestations and len(v["tablets"]) >= min_tablets
    ]

    # Build equivalence classes via union-find
    parent: dict[str, str] = {}

    def _find(x: str) -> str:
        # Walk to root without modifying parent
        root = x
        while root in parent:
            root = parent[root]
        # Path compression: point every node on the path directly at root
        while x in parent:
            next_x = parent[x]
            parent[x] = root
            x = next_x
        return root

    def _union(a: str, b: str) -> None:
        ra, rb = _find(a), _find(b)
        if ra != rb:
            parent[rb] = ra

    for p in filtered_pairs:
        _union(p["s1"], p["s2"])

    # Group signs into equivalence classes
    groups: dict[str, set[str]] = {}
    all_signs = {p["s1"] for p in filtered_pairs} | {p["s2"] for p in filtered_pairs}
    for sign in all_signs:
        root = _find(sign)
        groups.setdefault(root, set()).add(sign)

    equivalence_classes: list[frozenset[str]] = [
        frozenset(members) for members in groups.values() if len(members) >= 2
    ]

    # Compare to Pozdniakov reference
    comparison = _compare_to_reference(equivalence_classes, POZDNIAKOV_REFERENCE_CLASSES)

    return {
        "pairs": filtered_pairs,
        "equivalence_classes": [sorted(ec) for ec in equivalence_classes],
        "comparison": comparison,
        "n_pairs_found": len(filtered_pairs),
        "n_classes_found": len(equivalence_classes),
        "excluded_passages": excluded_passages,
        "parameters": {
            "min_attestations": min_attestations,
            "min_tablets": min_tablets,
            "max_passage_attestations": max_passage_attestations,
            "max_passage_tablets": max_passage_tablets,
        },
    }


def _compare_to_reference(
    recovered: list[frozenset[str]],
    reference: list[frozenset[str]],
) -> dict:
    """Compute precision, recall, and F1 against reference equivalence classes.

    A recovered class *matches* a reference class if their Jaccard similarity
    is > 0.5 (majority overlap).
    """
    def _jaccard(a: frozenset, b: frozenset) -> float:
        if not a and not b:
            return 1.0
        return len(a & b) / len(a | b)

    # For each reference class, does any recovered class match?
    matched_ref: list[bool] = []
    for ref_cls in reference:
        best_j = max((_jaccard(rec, ref_cls) for rec in recovered), default=0.0)
        matched_ref.append(best_j > 0.5)

    recall = sum(matched_ref) / len(reference) if reference else 0.0

    # For each recovered class, does it match any reference class?
    matched_rec: list[bool] = []
    for rec_cls in recovered:
        best_j = max((_jaccard(rec_cls, ref) for ref in reference), default=0.0)
        matched_rec.append(best_j > 0.5)

    precision = sum(matched_rec) / len(recovered) if recovered else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0 else 0.0
    )

    # Detail: which reference classes were recovered?
    recovered_details = []
    for i, ref_cls in enumerate(reference):
        best_match = None
        best_j = 0.0
        for rec in recovered:
            j = _jaccard(rec, ref_cls)
            if j > best_j:
                best_j = j
                best_match = sorted(rec)
        recovered_details.append({
            "reference_class": sorted(ref_cls),
            "matched": best_j > 0.5,
            "best_jaccard": round(best_j, 3),
            "best_matching_recovered": best_match,
        })

    return {
        "recall": round(recall, 4),
        "precision": round(precision, 4),
        "f1": round(f1, 4),
        "n_reference_classes": len(reference),
        "n_recovered_classes": len(recovered),
        "n_reference_matched": sum(matched_ref),
        "n_recovered_matching_reference": sum(matched_rec),
        "class_details": recovered_details,
    }


# ---------------------------------------------------------------------------
# MCMC cross-validation
# ---------------------------------------------------------------------------

def cross_validate_with_mcmc(
    pairs: list[dict],
    phoneme_map: dict[str, str],
    similarity_threshold: float = 0.5,
) -> dict:
    """Validate paradigmatic pairs against MCMC phoneme assignments.

    For each paradigmatic pair (s1, s2), computes the phoneme similarity
    between π(s1) and π(s2).  Pozdniakov predicts that members of the same
    equivalence class should have the same or phonetically similar phonemes.

    Returns fraction of pairs where similarity > threshold, plus per-pair details.
    """
    evaluated = []
    n_above = 0
    for p in pairs:
        s1, s2 = p["s1"], p["s2"]
        ph1 = phoneme_map.get(s1)
        ph2 = phoneme_map.get(s2)
        if ph1 is None or ph2 is None:
            sim = None
            above = None
        else:
            sim = round(_phoneme_similarity(ph1, ph2), 4)
            above = sim > similarity_threshold
            if above:
                n_above += 1
        evaluated.append({
            "s1": s1, "s2": s2,
            "phoneme_s1": ph1, "phoneme_s2": ph2,
            "similarity": sim,
            "above_threshold": above,
        })

    n_scored = sum(1 for e in evaluated if e["similarity"] is not None)
    fraction_above = n_above / n_scored if n_scored > 0 else None

    return {
        "similarity_threshold": similarity_threshold,
        "n_pairs_evaluated": len(evaluated),
        "n_pairs_scored": n_scored,
        "n_above_threshold": n_above,
        "fraction_above_threshold": round(fraction_above, 4) if fraction_above is not None else None,
        "pair_details": evaluated,
        "interpretation": (
            f"{n_above}/{n_scored} paradigmatic pairs have phoneme similarity "
            f"> {similarity_threshold} under the MCMC top hypothesis. "
            + ("This is consistent with Pozdniakov's structural hypothesis." if (fraction_above or 0) > 0.5
               else "Phoneme similarity below expectation — paradigmatic classes may need revision.")
        ),
    }


# ---------------------------------------------------------------------------
# HTML report builder
# ---------------------------------------------------------------------------

def _esc(s: object) -> str:
    import html as _html
    return _html.escape(str(s))


_PARA_CSS = """\
:root {
  --bg:#0d0f12; --surface:#161920; --surface2:#1e2229;
  --border:#2a2e38; --text:#d0d4dc; --muted:#6b7280;
  --accent:#c4a96d; --green:#4ade80; --yellow:#facc15;
  --red:#f87171; --blue:#93c5fd;
}
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
.code{color:var(--accent);}  .ph{color:var(--blue);}
.hi{color:var(--green);}  .lo{color:var(--muted);}
.med{color:var(--yellow);}
.badge-match{background:rgba(74,222,128,.12);color:var(--green);
             font-size:9px;padding:1px 6px;border-radius:3px;}
.badge-miss{background:rgba(248,113,113,.1);color:var(--red);
            font-size:9px;padding:1px 6px;border-radius:3px;}
.verdict{border-left:3px solid var(--accent);padding:14px 18px;
         background:var(--surface);border-radius:0 5px 5px 0;margin:20px 0;}
.verdict strong{color:var(--accent);}
"""


def _build_paradigmatic_html(
    paradigmatic: dict,
    mcmc_xval: dict | None,
    generated: str,
) -> str:
    comp = paradigmatic.get("comparison", {})
    recall    = comp.get("recall", 0.0)
    precision = comp.get("precision", 0.0)
    f1        = comp.get("f1", 0.0)
    n_pairs   = paradigmatic.get("n_pairs_found", 0)
    n_classes = paradigmatic.get("n_classes_found", 0)
    n_ref     = comp.get("n_reference_classes", 15)
    n_matched = comp.get("n_reference_matched", 0)

    # ── Summary stats ──────────────────────────────────────────────────────
    stats_html = f"""
<div class="stat-grid">
  <div class="stat">
    <div class="stat-label">Paradigmatic pairs</div>
    <div class="stat-value">{n_pairs}</div>
    <div class="stat-sub">≥ min attestations + tablets</div>
  </div>
  <div class="stat">
    <div class="stat-label">Equivalence classes</div>
    <div class="stat-value">{n_classes}</div>
    <div class="stat-sub">union-find groups</div>
  </div>
  <div class="stat">
    <div class="stat-label">Recall vs Pozdniakov</div>
    <div class="stat-value {'hi' if recall >= 0.6 else 'med' if recall >= 0.3 else 'lo'}">{recall:.1%}</div>
    <div class="stat-sub">{n_matched} / {n_ref} reference classes</div>
  </div>
  <div class="stat">
    <div class="stat-label">Precision</div>
    <div class="stat-value {'hi' if precision >= 0.6 else 'med' if precision >= 0.3 else 'lo'}">{precision:.1%}</div>
    <div class="stat-sub">recovered matching ref</div>
  </div>
  <div class="stat">
    <div class="stat-label">F1 score</div>
    <div class="stat-value {'hi' if f1 >= 0.6 else 'med' if f1 >= 0.3 else 'lo'}">{f1:.3f}</div>
    <div class="stat-sub">harmonic mean P/R</div>
  </div>
</div>
"""

    # ── Paradigmatic pairs table ───────────────────────────────────────────
    pair_rows = ""
    for p in paradigmatic.get("pairs", [])[:30]:
        pair_rows += (
            f"<tr>"
            f'<td class="code">{_esc(p["s1"])}</td>'
            f'<td class="code">{_esc(p["s2"])}</td>'
            f"<td>{p['n_attestations']}</td>"
            f"<td>{len(p['tablets'])}</td>"
            f'<td class="lo">{_esc(", ".join(p["tablets"][:6]))}</td>'
            f'<td class="lo">{_esc(", ".join(p["passage_ids"][:4]))}</td>'
            f"</tr>"
        )
    pairs_table = (
        "<table><thead><tr>"
        "<th>Sign 1</th><th>Sign 2</th><th>Attestations</th>"
        "<th>Tablets</th><th>Tablets (list)</th><th>Passages</th>"
        f"</tr></thead><tbody>{pair_rows}</tbody></table>"
        if pair_rows else "<p class='lo'>No paradigmatic pairs found with current filters.</p>"
    )

    # ── Equivalence classes ────────────────────────────────────────────────
    ec_rows = ""
    for i, cls in enumerate(paradigmatic.get("equivalence_classes", [])[:20], 1):
        ec_rows += (
            f"<tr>"
            f'<td class="lo">{i}</td>'
            f'<td class="code">{_esc(" ↔ ".join(cls))}</td>'
            f"<td>{len(cls)}</td>"
            f"</tr>"
        )
    ec_table = (
        "<table><thead><tr>"
        "<th>#</th><th>Signs</th><th>Size</th>"
        f"</tr></thead><tbody>{ec_rows}</tbody></table>"
        if ec_rows else "<p class='lo'>No equivalence classes.</p>"
    )

    # ── Reference comparison ───────────────────────────────────────────────
    ref_rows = ""
    for d in comp.get("class_details", []):
        badge = (
            '<span class="badge-match">MATCH</span>' if d["matched"]
            else '<span class="badge-miss">MISS</span>'
        )
        ref_rows += (
            f"<tr>"
            f"<td>{badge}</td>"
            f'<td class="code">{_esc(" ".join(d["reference_class"]))}</td>'
            f"<td>{d['best_jaccard']:.2f}</td>"
            f'<td class="lo">{_esc(" ".join(d["best_matching_recovered"] or ["-"]))}</td>'
            f"</tr>"
        )
    ref_table = (
        "<table><thead><tr>"
        "<th>Status</th><th>Reference class</th>"
        "<th>Jaccard</th><th>Best recovered match</th>"
        f"</tr></thead><tbody>{ref_rows}</tbody></table>"
        if ref_rows else ""
    )

    # ── MCMC cross-validation ──────────────────────────────────────────────
    xval_section = ""
    if mcmc_xval:
        frac = mcmc_xval.get("fraction_above_threshold")
        frac_str = f"{frac:.1%}" if frac is not None else "n/a"
        frac_cls = "hi" if (frac or 0) > 0.5 else "med" if (frac or 0) > 0.3 else "lo"
        xval_rows = ""
        for e in mcmc_xval.get("pair_details", [])[:20]:
            sim = e.get("similarity")
            above = e.get("above_threshold")
            sim_str = f"{sim:.3f}" if sim is not None else "?"
            sim_cls = "hi" if (sim or 0) > 0.7 else "med" if (sim or 0) > 0.4 else "lo"
            badge = (
                '<span class="badge-match">✓</span>' if above is True
                else '<span class="badge-miss">✗</span>' if above is False
                else "?"
            )
            xval_rows += (
                f"<tr>"
                f'<td class="code">{_esc(e["s1"])}</td>'
                f'<td class="ph">{_esc(e["phoneme_s1"] or "?")}</td>'
                f'<td class="code">{_esc(e["s2"])}</td>'
                f'<td class="ph">{_esc(e["phoneme_s2"] or "?")}</td>'
                f'<td class="{sim_cls}">{sim_str}</td>'
                f"<td>{badge}</td>"
                f"</tr>"
            )
        xval_section = f"""
<div class="section">
  <div class="section-title">MCMC Cross-Validation</div>
  <div class="stat-grid">
    <div class="stat">
      <div class="stat-label">Fraction above threshold</div>
      <div class="stat-value {frac_cls}">{frac_str}</div>
      <div class="stat-sub">phoneme sim &gt; {mcmc_xval["similarity_threshold"]}</div>
    </div>
    <div class="stat">
      <div class="stat-label">Pairs scored</div>
      <div class="stat-value">{mcmc_xval["n_pairs_scored"]}</div>
      <div class="stat-sub">of {mcmc_xval["n_pairs_evaluated"]} total</div>
    </div>
  </div>
  <div class="verdict"><strong>Interpretation</strong>
    <p style="font-size:12px;margin-top:6px">{_esc(mcmc_xval.get("interpretation",""))}</p>
  </div>
  <table><thead><tr>
    <th>Sign 1</th><th>Phoneme 1</th><th>Sign 2</th><th>Phoneme 2</th>
    <th>Similarity</th><th>&gt; threshold</th>
  </tr></thead><tbody>{xval_rows}</tbody></table>
</div>
"""

    verdict_text = (
        f"Paradigmatic analysis recovered {n_matched} of {n_ref} Pozdniakov (2011) "
        f"reference equivalence classes (recall {recall:.1%}, precision {precision:.1%}, "
        f"F1 {f1:.3f}). "
        + ("Replication quality is high — the structural signal in the corpus aligns with "
           "Pozdniakov's manual paradigmatic analysis." if f1 >= 0.5 else
           "Partial replication — the automated method recovers a subset of Pozdniakov's "
           "classes. Additional parallel passages or looser matching parameters may improve coverage.")
    )

    return (
        "<!DOCTYPE html><html lang='en'>"
        "<head><meta charset='utf-8'>"
        "<title>Rongorongo — Pozdniakov Paradigmatic Analysis</title>"
        f"<style>{_PARA_CSS}</style></head>"
        "<body><div class='wrap'>"
        "<h1>Pozdniakov Paradigmatic Analysis</h1>"
        f"<div class='sub'>Replication of Pozdniakov (1996, 2011) morpheme identification · "
        f"Generated {_esc(generated)}</div>"
        f"<div class='section'><div class='section-title'>Summary</div>"
        f"{stats_html}"
        "<div class='verdict'><strong>Finding</strong>"
        f"<p style='font-size:12px;margin-top:6px'>{_esc(verdict_text)}</p></div>"
        "</div>"
        f"<div class='section'><div class='section-title'>Paradigmatic Pairs (top 30)</div>"
        f"{pairs_table}</div>"
        f"<div class='section'><div class='section-title'>Recovered Equivalence Classes</div>"
        f"{ec_table}</div>"
        f"<div class='section'><div class='section-title'>Comparison to Pozdniakov (2011) Reference</div>"
        f"{ref_table}</div>"
        f"{xval_section}"
        "</div></body></html>"
    )


# ---------------------------------------------------------------------------
# compute_paradigmatic() — new primary entry point
# ---------------------------------------------------------------------------

def compute_paradigmatic(seed: int | None = None) -> dict:
    """Run the paradigmatic analysis and return results dict.

    Writes:
      outputs/analysis/pozdniakov_paradigmatic.json
      outputs/analysis/pozdniakov_report.html
    """
    from datetime import datetime, timezone

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    parallel_path = _parallel_passages_path()
    raw = json.loads(parallel_path.read_text(encoding="utf-8"))

    # Support both {passages: [...]} and plain list formats
    if isinstance(raw, dict) and "passages" in raw:
        passages = raw["passages"]
    elif isinstance(raw, list):
        passages = raw
    else:
        passages = list(raw.values()) if isinstance(raw, dict) else []

    paradigmatic = find_paradigmatic_pairs(passages)

    # Cross-validate with MCMC if ranking.json is available
    mcmc_xval: dict | None = None
    if RANKING_PATH.exists():
        try:
            phoneme_map = _extract_ranking()
            mcmc_xval = cross_validate_with_mcmc(paradigmatic["pairs"], phoneme_map)
            paradigmatic["mcmc_cross_validation"] = mcmc_xval
        except Exception as exc:
            import warnings
            warnings.warn(f"MCMC cross-validation skipped: {exc}")

    from hackingrongo.provenance import stamp
    stamp(paradigmatic, seed=seed)
    json_path = OUTPUT_DIR / "pozdniakov_paradigmatic.json"
    json_path.write_text(json.dumps(paradigmatic, indent=2, default=list), encoding="utf-8")

    generated = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    html = _build_paradigmatic_html(paradigmatic, mcmc_xval, generated)
    html_path = OUTPUT_DIR / "pozdniakov_report.html"
    html_path.write_text(html, encoding="utf-8")

    return paradigmatic


# ---------------------------------------------------------------------------
# Original statistical hypothesis tests (kept for backward compat)
# ---------------------------------------------------------------------------


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
    import argparse as _ap
    _p = _ap.ArgumentParser(description="Pozdniakov paradigmatic analysis and HTML report.")
    _p.add_argument("--seed", type=int, default=20260606, metavar="INT",
                    help="Global RNG seed for reproducibility (default: 20260606).")
    _args = _p.parse_args()
    from hackingrongo.repro import set_global_seed
    set_global_seed(_args.seed)

    # 1. Paradigmatic analysis (new primary output).  Needs only the
    # corpus and parallel passages — both Ring-1 data.
    compute_paradigmatic(seed=_args.seed)
    primary_html = OUTPUT_DIR / "pozdniakov_report.html"
    print(str(primary_html))

    # 2. Legacy statistical hypothesis tests (backward compat).  These
    # score a sign→phoneme assignment, so they need Zone C's
    # ranking.json (Step 5, which in turn needs the Ring-2 embeddings).
    # In a Ring-1 run that artifact legitimately doesn't exist yet —
    # skip the legacy tests instead of failing the whole step.
    if not RANKING_PATH.exists():
        print(
            f"ranking.json not found ({RANKING_PATH}) — phoneme-map "
            "hypothesis tests skipped (run Step 5 / Zone C decipherment "
            "to enable them). Paradigmatic report generated above."
        )
        return 0
    results_path = compute_results()
    module = _load_report_module()
    html_path = OUTPUT_DIR / "pozdniakov_hypothesis_report.html"
    module.save_pozdniakov_report(results_path, html_path)
    print(str(results_path))
    print(str(html_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
