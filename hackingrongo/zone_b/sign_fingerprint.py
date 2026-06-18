"""
hackingrongo/zone_b/sign_fingerprint.py
=======================================

Distributional "service discovery" for rongorongo signs.

Borrowing the network-pentest idea that an unknown host's role is inferred from
its *behaviour* (what ports it answers, who it talks to) rather than its
content, this module classifies each sign's **functional role from its
distributional fingerprint alone** — never from any assumed phonetic value.

It upgrades the single-signal heuristic in ``sign_classifier.py`` into a
multi-signal, *auditable* classifier (every assignment records the feature
values that produced it) and a *validated* one: roles are recomputed
independently on the pre- and post-contact strata, and the headline credibility
metric is how often a sign's role survives the contact boundary.

These roles are **distributional hypotheses, not confirmed linguistic
functions.**  A sign behaving like a determinative is not proof it is one.

Feature vector (per canonical sign, frequency >= ``min_freq``)
--------------------------------------------------------------
* ``betweenness`` / ``pagerank`` — from the corpus bigram-PMI graph
  (network_analysis.compute_centralities).
* ``positional_entropy`` — normalised entropy of the sign's position WITHIN A
  LINE (0 = fixed slot, 1 = appears anywhere).
* ``neighbor_diversity`` — distinct adjacent sign types / frequency
  (high = attaches broadly like a determinative).
* ``own_frequency`` — corpus-relative frequency.
* ``slot_predictability`` — 1 − normalised entropy of the sign's *predecessor*
  distribution (high = predictable grammatical slot).
* ``passage_anchor_score`` — fraction of occurrences at parallel-passage
  boundaries (start/end of detected passages).
"""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from hackingrongo.zone_b.network_analysis import build_pmi_graph, compute_centralities
from hackingrongo.zone_b.sign_classifier import SignClass

_POS_BINS = 10


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class SignFingerprint:
    """Auditable role assignment for one sign."""
    code: str
    frequency: int
    features: dict[str, float]
    role: str                       # SignClass value ("taxogram"/"logogram"/...)
    subtype: str | None             # "determinative" | "particle" | "anchor" | None
    rule: str                       # which threshold rule fired

    def as_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "frequency": self.frequency,
            "role": self.role,
            "subtype": self.subtype,
            "rule": self.rule,
            "features": {k: round(float(v), 6) for k, v in self.features.items()},
        }


# ---------------------------------------------------------------------------
# Corpus loading (per-glyph records, canonicalised, stratum-tagged)
# ---------------------------------------------------------------------------

def load_glyph_records(corpus_dir: Path, canon) -> list[dict[str, Any]]:
    """Return per-glyph records: {tablet, side, line, position, code, stratum}.

    *canon* is a ``SignCatalog.get_canonical_id`` callable (or identity).
    """
    records: list[dict[str, Any]] = []
    for path in sorted(corpus_dir.glob("[A-Z].json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        tablet = path.stem
        stratum = data.get("cluster", "unknown")
        for g in data.get("glyphs", []):
            raw = g.get("barthel_code")
            if not raw:
                continue
            records.append({
                "tablet":   tablet,
                "side":     str(g.get("side", "")),
                "line":     str(g.get("line", "")),
                "position": int(g.get("position", -1)),
                "code":     canon(str(raw)),
                "stratum":  stratum,
            })
    return records


def load_passage_boundaries(variants_path: Path) -> set[tuple[str, int]]:
    """Return {(tablet, position)} for the start AND end glyph of every
    parallel-passage attestation."""
    boundaries: set[tuple[str, int]] = set()
    if not variants_path.exists():
        return boundaries
    try:
        data = json.loads(variants_path.read_text(encoding="utf-8"))
    except Exception:
        return boundaries
    for p in data.get("passages", []):
        for att in p.get("attestations", []):
            tablet = str(att.get("tablet", ""))
            start = att.get("start_position")
            form = att.get("form", [])
            if tablet and isinstance(start, int) and form:
                boundaries.add((tablet, start))
            if tablet and isinstance(start, int) and form:
                boundaries.add((tablet, start + len(form) - 1))
    return boundaries


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def _sequences_by_tablet(records: list[dict[str, Any]]) -> list[list[str]]:
    by_tab: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in records:
        by_tab[r["tablet"]].append(r)
    seqs = []
    for rows in by_tab.values():
        rows.sort(key=lambda r: r["position"])
        seqs.append([r["code"] for r in rows])
    return seqs


def _positional_entropy_by_line(records: list[dict[str, Any]]) -> dict[str, float]:
    """Per sign: normalised entropy of position WITHIN a line (adapts
    astronomical_analysis._positional_entropy from per-tablet to per-line)."""
    by_line: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for r in records:
        by_line[(r["tablet"], r["side"], r["line"])].append(r)

    rel_pos: dict[str, list[float]] = defaultdict(list)
    for rows in by_line.values():
        rows.sort(key=lambda r: r["position"])
        n = len(rows)
        for i, r in enumerate(rows):
            rel_pos[r["code"]].append(i / max(n - 1, 1))

    out: dict[str, float] = {}
    denom = math.log2(_POS_BINS)
    for code, positions in rel_pos.items():
        counts = np.zeros(_POS_BINS)
        for p in positions:
            counts[min(int(p * _POS_BINS), _POS_BINS - 1)] += 1
        total = counts.sum()
        if total <= 0:
            out[code] = 0.0
            continue
        probs = counts[counts > 0] / total
        h = -float(np.sum(probs * np.log2(probs)))
        out[code] = h / denom if denom > 0 else 0.0   # normalised to [0, 1]
    return out


def _neighbor_and_predecessor_stats(
    sequences: list[list[str]],
    freq: dict[str, int],
    n_distinct: int,
) -> tuple[dict[str, float], dict[str, float]]:
    """Return (neighbor_diversity, slot_predictability) per sign."""
    neighbors: dict[str, set[str]] = defaultdict(set)
    predecessors: dict[str, Counter] = defaultdict(Counter)
    for seq in sequences:
        for i, s in enumerate(seq):
            if i > 0:
                neighbors[s].add(seq[i - 1])
                neighbors[seq[i - 1]].add(s)
                predecessors[s][seq[i - 1]] += 1
            if i < len(seq) - 1:
                neighbors[s].add(seq[i + 1])

    neighbor_diversity = {
        s: len(neighbors[s]) / freq[s] if freq.get(s) else 0.0
        for s in freq
    }
    # slot_predictability = 1 - H(predecessor dist) / log2(vocab) ∈ [0, 1]
    denom = math.log2(max(n_distinct, 2))
    slot_predictability: dict[str, float] = {}
    for s in freq:
        preds = predecessors.get(s)
        if not preds:
            slot_predictability[s] = 0.0
            continue
        total = sum(preds.values())
        probs = np.array([c / total for c in preds.values()])
        h = -float(np.sum(probs * np.log2(probs)))
        slot_predictability[s] = max(0.0, min(1.0, 1.0 - h / denom))
    return neighbor_diversity, slot_predictability


def compute_features(
    records: list[dict[str, Any]],
    boundaries: set[tuple[str, int]],
    min_freq: int = 5,
    min_cofreq: int = 2,
) -> tuple[dict[str, dict[str, float]], dict[str, int]]:
    """Return (features_by_sign, frequency_by_sign) for the freq>=min_freq core."""
    freq = Counter(r["code"] for r in records)
    total = sum(freq.values()) or 1
    core = {s for s, c in freq.items() if c >= min_freq}

    sequences = _sequences_by_tablet(records)
    pos_entropy = _positional_entropy_by_line(records)
    nd, sp = _neighbor_and_predecessor_stats(sequences, freq, len(freq))

    # Centralities on the bigram-PMI graph (signs below min_cofreq → 0).
    cen = compute_centralities(build_pmi_graph(sequences, min_cofreq=min_cofreq))
    betw = cen.get("betweenness", {})
    prank = cen.get("pagerank", {})

    # passage_anchor_score: fraction of a sign's occurrences at a boundary.
    boundary_hits: Counter = Counter()
    for r in records:
        if (r["tablet"], r["position"]) in boundaries:
            boundary_hits[r["code"]] += 1

    features: dict[str, dict[str, float]] = {}
    for s in sorted(core):
        features[s] = {
            "betweenness":          float(betw.get(s, 0.0)),
            "pagerank":             float(prank.get(s, 0.0)),
            "positional_entropy":   float(pos_entropy.get(s, 0.0)),
            "neighbor_diversity":   float(nd.get(s, 0.0)),
            "own_frequency":        freq[s] / total,
            "slot_predictability":  float(sp.get(s, 0.0)),
            "passage_anchor_score": boundary_hits.get(s, 0) / freq[s],
        }
    return features, {s: freq[s] for s in core}


# ---------------------------------------------------------------------------
# Role assignment (interpretable thresholds, not a black box)
# ---------------------------------------------------------------------------

def _threshold_stats(features: dict[str, dict[str, float]], anchor_thresh: float) -> dict[str, float]:
    if not features:
        return {"mean_betweenness": 0.0, "median_neighbor_diversity": 0.0,
                "median_positional_entropy": 0.0, "median_slot_predictability": 0.0,
                "anchor_thresh": anchor_thresh, "freq_hi_pct": 0.0}
    betw = np.array([f["betweenness"] for f in features.values()])
    nd = np.array([f["neighbor_diversity"] for f in features.values()])
    pe = np.array([f["positional_entropy"] for f in features.values()])
    sp = np.array([f["slot_predictability"] for f in features.values()])
    freqs = np.array([f["own_frequency"] for f in features.values()])
    return {
        "mean_betweenness":            float(betw.mean()),
        "median_neighbor_diversity":   float(np.median(nd)),
        "median_positional_entropy":   float(np.median(pe)),
        "median_slot_predictability":  float(np.median(sp)),
        "freq_hi":                     float(np.quantile(freqs, 0.75)),  # "high frequency"
        "anchor_thresh":               anchor_thresh,
    }


def assign_roles(
    features: dict[str, dict[str, float]],
    frequency: dict[str, int],
    anchor_thresh: float = 0.5,
) -> tuple[dict[str, SignFingerprint], dict[str, float]]:
    """Map each fingerprint to a SignClass + subtype via interpretable thresholds.

    Returns ``(fingerprints_by_sign, threshold_stats)``; the thresholds are
    returned so the report can show exactly which cutoffs were applied.
    """
    st = _threshold_stats(features, anchor_thresh)
    out: dict[str, SignFingerprint] = {}
    for s, f in features.items():
        anchor = f["passage_anchor_score"] >= st["anchor_thresh"]

        # 1. Determinative/classifier → TAXOGRAM
        if (f["betweenness"] > 2.0 * st["mean_betweenness"]
                and f["neighbor_diversity"] > st["median_neighbor_diversity"]
                and f["own_frequency"] < st["freq_hi"]):
            role, sub, rule = SignClass.TAXOGRAM, "determinative", "determinative"
        # 2. Grammatical particle → TAXOGRAM (subtype particle)
        elif (f["own_frequency"] >= st["freq_hi"]
                and f["slot_predictability"] > st["median_slot_predictability"]
                and f["positional_entropy"] < st["median_positional_entropy"]):
            role, sub, rule = SignClass.TAXOGRAM, "particle", "particle"
        # 3. Content sign/logogram → LOGOGRAM
        elif (f["positional_entropy"] > st["median_positional_entropy"]
                and f["slot_predictability"] < st["median_slot_predictability"]):
            role, sub, rule = SignClass.LOGOGRAM, None, "logogram"
        # 4. Default → PHONETIC
        else:
            role, sub, rule = SignClass.PHONETIC, None, "default"

        # Boundary marker is an orthogonal subtype tag (overrides finer subtype).
        if anchor:
            sub = "anchor"
            rule = rule + "+anchor"

        out[s] = SignFingerprint(
            code=s, frequency=frequency[s], features=f,
            role=role.value, subtype=sub, rule=rule,
        )
    return out, st


# ---------------------------------------------------------------------------
# Diachronic validation
# ---------------------------------------------------------------------------

def diachronic_stability(
    roles_pre: dict[str, SignFingerprint],
    roles_post: dict[str, SignFingerprint],
) -> dict[str, Any]:
    """role_stability = fraction of signs (attested in BOTH strata) whose role
    is identical pre vs post.  Role-changers are flagged."""
    shared = sorted(set(roles_pre) & set(roles_post))
    stable, changed = [], []
    for s in shared:
        pre, post = roles_pre[s].role, roles_post[s].role
        if pre == post:
            stable.append(s)
        else:
            changed.append({"code": s, "pre_role": pre, "post_role": post})
    n = len(shared)
    return {
        "n_signs_in_both_strata": n,
        "role_stability": (len(stable) / n) if n else None,
        "n_stable": len(stable),
        "n_changed": len(changed),
        "stable_signs": stable,
        "role_changes": changed,
    }
