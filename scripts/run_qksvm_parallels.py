#!/usr/bin/env python3
"""
run_qksvm_parallels.py — Projected Quantum Kernel SVM for soft parallel passage detection.

Trains a QK-SVM on the 13 confirmed cross-tablet parallel passages, then scores
every high-frequency cross-tablet position pair to find near-parallel sequences
missed by exact Barthel-code matching.

Background
----------
The existing Kasiski-equivalent matcher (cross_reference_parallels.py) finds
13 exact-match passages. It misses near-parallels where one sign is swapped for
a contextually related sign — the same low-confidence diachronic phenomenon as
the holy-grail passages, but at sub-threshold statistical evidence. A quantum
kernel trained on confirmed passages can score position pairs more sensitively.

Projected Quantum Kernel (PQK)
-------------------------------
K_PQK(x, y) = Σ_i Tr[ρ_i(x) · ρ_i(y)]
where ρ_i(x) is the reduced density matrix of qubit i after encoding x via
ZZFeatureMap. Requires ONE quantum circuit per data point (vs one per pair for
the full quantum kernel), giving O(N) circuits vs O(N²).

Feature vector (8 dimensions)
------------------------------
  0  delta_ic_contribution       abs(IC(sign_A) - IC(sign_B))        ∈ [0,1]
  1  delta_positional_entropy    abs(H_pos(sign_A) - H_pos(sign_B))  ∈ [0,1]
  2  delta_omission_rate         abs(omit(sign_A) - omit(sign_B))    ∈ [0,1]
  3  bigram_mi_left              MI(context_left, sign_A) in bits     ∈ [0,1]
  4  bigram_mi_right             MI(sign_A, context_right) in bits    ∈ [0,1]
  5  cross_tablet_freq_ratio     min(p_A,p_B)/max(p_A,p_B)           ∈ [0,1]
  6  context_embedding_sim       cosine(emb_A, emb_B)                 ∈ [0,1]
  7  positional_distance         |pos_A/len_A - pos_B/len_B|          ∈ [0,1]

Usage
-----
    python scripts/run_qksvm_parallels.py
    python scripts/run_qksvm_parallels.py --backend fake_brisbane
    python scripts/run_qksvm_parallels.py --backend ibmq --ibmq-token <T>
    python scripts/run_qksvm_parallels.py --inject-as-cribs
    python scripts/run_qksvm_parallels.py --score-threshold 0.7
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import pickle
import random
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
_VARIANTS_JSON      = PROJECT_ROOT / "data" / "parallels" / "parallel_variants_auto.json"
_CORPUS_DIR         = PROJECT_ROOT / "data" / "corpus"
_SENSITIVITY_JSON   = PROJECT_ROOT / "outputs" / "sensitivity_analysis.json"
_ZONE_B_CACHE       = PROJECT_ROOT / "outputs" / "zone_b_cache.pkl"
_EMBEDDINGS_CACHE   = PROJECT_ROOT / "outputs" / "embeddings_cache.pt"
_TRAINING_NPZ       = PROJECT_ROOT / "outputs" / "quantum" / "qksvm_training_data.npz"
_SOFT_PARALLELS_OUT = PROJECT_ROOT / "outputs" / "quantum" / "soft_parallels_qksvm.json"
_CRIB_OUT           = PROJECT_ROOT / "data" / "parallels" / "soft_parallel_candidates.json"

N_FEATURES          = 8
N_NEGATIVES         = 200
FREQ_THRESHOLD      = 0.005   # only score signs with p > this (~top 40)
POS_DIST_THRESHOLD  = 0.3     # |pos_A/len_A - pos_B/len_B| filter
SCORE_THRESHOLD     = 0.7     # SVM decision value for soft parallel
MAX_SCORING_PAIRS   = 8_000   # cap for statevector scoring; ibmq/fake use 1,000
N_FOLDS             = 5
RANDOM_SEED         = 42


# ─────────────────────────────────────────────────────────────────────────────
# Data types
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PositionPair:
    """A cross-tablet position pair — the unit of classification."""
    tablet_a:   str
    pos_a:      int     # 0-based absolute position in tablet_a glyph sequence
    sign_a:     str
    tablet_b:   str
    pos_b:      int
    sign_b:     str
    passage_id: str | None = None    # set for confirmed-passage pairs
    label:      int = -1             # 1 = positive, 0 = negative


@dataclass
class SoftParallelCandidate:
    tablet_pair:               tuple[str, str]
    position_range_a:          tuple[int, int]   # start, end positions in tablet_a
    position_range_b:          tuple[int, int]
    signs_tablet_a:            list[str]
    signs_tablet_b:            list[str]
    svm_scores:                list[float]
    feature_vectors:           list[list[float]]
    nearest_confirmed_passage: str
    mean_svm_score:            float


# ─────────────────────────────────────────────────────────────────────────────
# Corpus loading
# ─────────────────────────────────────────────────────────────────────────────

def load_corpus(corpus_dir: Path) -> dict[str, list[str]]:
    """Return {tablet_id: [sign, sign, …]} in glyph-sequence order."""
    tablet_seqs: dict[str, list[str]] = {}
    for path in sorted(corpus_dir.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        glyphs = data.get("glyphs", [])
        seq = [str(g["barthel_code"]) for g in glyphs if g.get("barthel_code")]
        if seq:
            tid = data.get("tablet_id") or path.stem
            tablet_seqs[tid] = seq
    return tablet_seqs


def load_parallel_passages(json_path: Path) -> list[dict]:
    """Load parallel_variants_auto.json without requiring SignCatalog."""
    data = json.loads(json_path.read_text(encoding="utf-8"))
    return data.get("passages", [])


# ─────────────────────────────────────────────────────────────────────────────
# Corpus statistics
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class CorpusStats:
    """Per-sign statistics derived from the full corpus."""
    freq:             dict[str, float]        # relative frequency p_i
    ic_contrib:       dict[str, float]        # p_i^2 / max(p_i^2), normalised
    positional_ent:   dict[str, float]        # H(position class), normalised
    omission_rate:    dict[str, float]        # from parallel passages
    bigram_counts:    dict[tuple[str,str], int]
    bigram_mi:        dict[tuple[str,str], float]  # mutual information in bits
    total_tokens:     int


def _compute_positional_entropy(
    tablet_seqs: dict[str, list[str]],
) -> dict[str, float]:
    """Position entropy: sign entropy over (begin / middle / end) thirds of lines."""
    pos_class_counts: defaultdict[str, Counter] = defaultdict(Counter)
    for seq in tablet_seqs.values():
        n = len(seq)
        if n == 0:
            continue
        for i, sign in enumerate(seq):
            frac = i / n
            cls = 0 if frac < 0.33 else (1 if frac < 0.67 else 2)
            pos_class_counts[sign][cls] += 1

    max_h = math.log2(3)  # maximum entropy = log2(3 classes)
    result: dict[str, float] = {}
    for sign, counts in pos_class_counts.items():
        total = sum(counts.values())
        probs = [c / total for c in counts.values()]
        h = -sum(p * math.log2(p) for p in probs if p > 0)
        result[sign] = h / max_h  # normalise to [0,1]
    return result


def _compute_omission_rates(
    passages: list[dict],
    tablet_seqs: dict[str, list[str]],
) -> dict[str, float]:
    """Omission rate: fraction of passages where a sign is absent from a variant
    that has the same canonical position occupied by a different sign."""
    omission_num: Counter[str] = Counter()
    omission_den: Counter[str] = Counter()
    for p in passages:
        canon = p["canonical_form"]
        atts  = p["attestations"]
        for i, canon_sign in enumerate(canon):
            present  = [a for a in atts if i < len(a["form"]) and a["form"][i] == canon_sign]
            absent   = [a for a in atts if i < len(a["form"]) and a["form"][i] != canon_sign]
            n_occ = len(present) + len(absent)
            if n_occ > 0:
                omission_num[canon_sign] += len(absent)
                omission_den[canon_sign] += n_occ
    return {
        sign: omission_num[sign] / omission_den[sign]
        for sign in omission_den
        if omission_den[sign] > 0
    }


def _compute_bigram_mi(
    tablet_seqs: dict[str, list[str]],
    freq: dict[str, float],
) -> dict[tuple[str, str], float]:
    """MI(A, B) = log2(P(A,B) / (P(A)*P(B))), normalised to [0,1]."""
    bigram_counts: Counter[tuple[str, str]] = Counter()
    total_bigrams = 0
    for seq in tablet_seqs.values():
        for a, b in zip(seq, seq[1:]):
            bigram_counts[(a, b)] += 1
            total_bigrams += 1
    if total_bigrams == 0:
        return {}

    mi: dict[tuple[str, str], float] = {}
    for (a, b), cnt in bigram_counts.items():
        p_ab = cnt / total_bigrams
        p_a  = freq.get(a, 1e-10)
        p_b  = freq.get(b, 1e-10)
        mi_val = math.log2(p_ab / (p_a * p_b + 1e-12) + 1e-12)
        mi[(a, b)] = max(0.0, mi_val)   # clip to non-negative

    max_mi = max(mi.values()) if mi else 1.0
    return {k: v / max_mi for k, v in mi.items()}


def build_corpus_stats(
    tablet_seqs: dict[str, list[str]],
    passages: list[dict],
) -> CorpusStats:
    all_tokens: list[str] = []
    for seq in tablet_seqs.values():
        all_tokens.extend(seq)
    total = len(all_tokens)

    freq_raw = Counter(all_tokens)
    freq = {s: c / total for s, c in freq_raw.items()}

    ic_raw = {s: (p ** 2) for s, p in freq.items()}
    max_ic = max(ic_raw.values()) if ic_raw else 1.0
    ic_contrib = {s: v / max_ic for s, v in ic_raw.items()}

    pos_ent   = _compute_positional_entropy(tablet_seqs)
    omit_rate = _compute_omission_rates(passages, tablet_seqs)
    bigram_mi = _compute_bigram_mi(tablet_seqs, freq)

    bigram_counts: dict[tuple[str, str], int] = {}
    for seq in tablet_seqs.values():
        for a, b in zip(seq, seq[1:]):
            bigram_counts[(a, b)] = bigram_counts.get((a, b), 0) + 1

    return CorpusStats(
        freq=freq,
        ic_contrib=ic_contrib,
        positional_ent=pos_ent,
        omission_rate=omit_rate,
        bigram_counts=bigram_counts,
        bigram_mi=bigram_mi,
        total_tokens=total,
    )


def load_embeddings(
    cache_path: Path,
) -> dict[str, np.ndarray] | None:
    """Load Zone A embeddings as {barthel_code: mean_embedding_vector}."""
    if not cache_path.exists():
        log.warning("Embeddings cache not found: %s — feature 6 = 0.0", cache_path)
        return None
    try:
        import torch
        data = torch.load(cache_path, weights_only=True)
        embs     = data["embeddings"].numpy()   # [N, D]
        codes    = list(data["barthel_codes"])
        # Average embeddings per sign code
        agg: defaultdict[str, list] = defaultdict(list)
        for code, vec in zip(codes, embs):
            agg[code].append(vec)
        return {code: np.mean(vecs, axis=0) for code, vecs in agg.items()}
    except Exception as exc:
        log.warning("Failed to load embeddings (%s) — feature 6 = 0.0", exc)
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Feature extraction
# ─────────────────────────────────────────────────────────────────────────────

def compute_feature_vector(
    pair: PositionPair,
    tablet_seqs: dict[str, list[str]],
    stats: CorpusStats,
    embeddings: dict[str, np.ndarray] | None,
) -> np.ndarray:
    """Return the 8-dimensional feature vector for a position pair."""
    sa, sb   = pair.sign_a, pair.sign_b
    seq_a    = tablet_seqs.get(pair.tablet_a, [])
    seq_b    = tablet_seqs.get(pair.tablet_b, [])
    len_a    = len(seq_a)
    len_b    = len(seq_b)

    # 0 — delta_ic_contribution
    ic_a = stats.ic_contrib.get(sa, 0.0)
    ic_b = stats.ic_contrib.get(sb, 0.0)
    f0 = abs(ic_a - ic_b)

    # 1 — delta_positional_entropy
    pe_a = stats.positional_ent.get(sa, 0.5)
    pe_b = stats.positional_ent.get(sb, 0.5)
    f1 = abs(pe_a - pe_b)

    # 2 — delta_omission_rate
    om_a = stats.omission_rate.get(sa, 0.0)
    om_b = stats.omission_rate.get(sb, 0.0)
    f2 = abs(om_a - om_b)

    # 3 — bigram_mi_left (MI of left context with sign_a)
    if pair.pos_a > 0 and seq_a:
        ctx_left = seq_a[pair.pos_a - 1] if pair.pos_a < len_a else ""
        f3 = stats.bigram_mi.get((ctx_left, sa), 0.0) if ctx_left else 0.0
    else:
        f3 = 0.0

    # 4 — bigram_mi_right (MI of sign_a with right context)
    if seq_a and pair.pos_a + 1 < len_a:
        ctx_right = seq_a[pair.pos_a + 1]
        f4 = stats.bigram_mi.get((sa, ctx_right), 0.0)
    else:
        f4 = 0.0

    # 5 — cross_tablet_freq_ratio
    p_a = stats.freq.get(sa, 1e-10)
    p_b = stats.freq.get(sb, 1e-10)
    f5 = min(p_a, p_b) / max(p_a, p_b) if max(p_a, p_b) > 0 else 0.0

    # 6 — context_embedding_similarity
    if embeddings is not None and sa in embeddings and sb in embeddings:
        ea, eb = embeddings[sa], embeddings[sb]
        denom = (np.linalg.norm(ea) * np.linalg.norm(eb))
        cosine = float(np.dot(ea, eb) / denom) if denom > 1e-9 else 0.0
        f6 = max(0.0, (cosine + 1.0) / 2.0)   # map [-1,1] → [0,1]
    else:
        f6 = 0.0

    # 7 — positional_distance
    norm_a = pair.pos_a / max(len_a - 1, 1)
    norm_b = pair.pos_b / max(len_b - 1, 1)
    f7 = abs(norm_a - norm_b)

    vec = np.array([f0, f1, f2, f3, f4, f5, f6, f7], dtype=np.float64)
    # Clip to [0, 1]
    return np.clip(vec, 0.0, 1.0)


def compute_feature_matrix(
    pairs: list[PositionPair],
    tablet_seqs: dict[str, list[str]],
    stats: CorpusStats,
    embeddings: dict[str, np.ndarray] | None,
) -> np.ndarray:
    return np.stack([
        compute_feature_vector(p, tablet_seqs, stats, embeddings)
        for p in pairs
    ])


# ─────────────────────────────────────────────────────────────────────────────
# Training set construction
# ─────────────────────────────────────────────────────────────────────────────

def build_positive_pairs(
    passages: list[dict],
    tablet_seqs: dict[str, list[str]],
) -> list[PositionPair]:
    """One positive pair per canonical position per attestation cross-pair."""
    positives: list[PositionPair] = []
    for p in passages:
        pid   = p["passage_id"]
        canon = p["canonical_form"]
        atts  = p["attestations"]
        for i, canon_sign in enumerate(canon):
            # Collect attestations that have sign at position i
            valid_atts = [
                a for a in atts
                if i < len(a["form"]) and a["form"][i] == canon_sign
            ]
            for ia in range(len(valid_atts)):
                for ib in range(ia + 1, len(valid_atts)):
                    a_att = valid_atts[ia]
                    b_att = valid_atts[ib]
                    # Skip same-tablet duplicates
                    if a_att["tablet"] == b_att["tablet"]:
                        continue
                    abs_pos_a = a_att.get("start_position", 0) + i
                    abs_pos_b = b_att.get("start_position", 0) + i
                    seq_a = tablet_seqs.get(a_att["tablet"])
                    seq_b = tablet_seqs.get(b_att["tablet"])
                    if seq_a is None or seq_b is None:
                        continue
                    positives.append(PositionPair(
                        tablet_a=a_att["tablet"],
                        pos_a=min(abs_pos_a, len(seq_a) - 1),
                        sign_a=canon_sign,
                        tablet_b=b_att["tablet"],
                        pos_b=min(abs_pos_b, len(seq_b) - 1),
                        sign_b=canon_sign,
                        passage_id=pid,
                        label=1,
                    ))
    return positives


def _confirmed_pair_set(positives: list[PositionPair]) -> set[tuple]:
    return {(p.tablet_a, p.pos_a, p.tablet_b, p.pos_b) for p in positives}


def sample_negatives(
    tablet_seqs: dict[str, list[str]],
    confirmed: set[tuple],
    n: int,
    rng: random.Random,
) -> list[PositionPair]:
    """Sample n cross-tablet position pairs not in any confirmed passage."""
    tablets = [t for t, seq in tablet_seqs.items() if len(seq) >= 5]
    negatives: list[PositionPair] = []
    attempts = 0
    while len(negatives) < n and attempts < n * 200:
        attempts += 1
        ta, tb = rng.sample(tablets, 2)
        seq_a  = tablet_seqs[ta]
        seq_b  = tablet_seqs[tb]
        pos_a  = rng.randrange(len(seq_a))
        pos_b  = rng.randrange(len(seq_b))
        key1   = (ta, pos_a, tb, pos_b)
        key2   = (tb, pos_b, ta, pos_a)
        if key1 in confirmed or key2 in confirmed:
            continue
        negatives.append(PositionPair(
            tablet_a=ta,
            pos_a=pos_a,
            sign_a=seq_a[pos_a],
            tablet_b=tb,
            pos_b=pos_b,
            sign_b=seq_b[pos_b],
            label=0,
        ))
    return negatives


# ─────────────────────────────────────────────────────────────────────────────
# Projected Quantum Kernel
# ─────────────────────────────────────────────────────────────────────────────

def build_feature_map(n_features: int, reps: int = 2):
    """ZZFeatureMap for n_features qubits and reps entanglement layers."""
    try:
        # Qiskit 2.1+ functional API (avoids DeprecationWarning on BlueprintCircuit)
        from qiskit.circuit.library import zz_feature_map
        return zz_feature_map(feature_dimension=n_features, reps=reps)
    except ImportError:
        from qiskit.circuit.library import ZZFeatureMap
        return ZZFeatureMap(feature_dimension=n_features, reps=reps)


def _rdms_statevector(
    X: np.ndarray,
    feature_map,
) -> list[list[np.ndarray]]:
    """Compute reduced density matrices for each row of X.

    Returns list[N] of list[n_qubits] of 2×2 numpy arrays.
    Each row requires ONE Statevector evaluation.
    """
    from qiskit.quantum_info import Statevector, partial_trace
    n_q = feature_map.num_qubits
    params = sorted(feature_map.parameters, key=lambda p: p.name)
    all_rdms: list[list[np.ndarray]] = []
    for x in X:
        bound = feature_map.assign_parameters(dict(zip(params, x)))
        sv    = Statevector(bound)
        rdms  = []
        for q in range(n_q):
            trace_out = [i for i in range(n_q) if i != q]
            rho_q = partial_trace(sv, trace_out)
            rdms.append(rho_q.data)
        all_rdms.append(rdms)
    return all_rdms


def _rdms_from_sampler(
    X: np.ndarray,
    feature_map,
    backend,
    shots: int = 1024,
) -> list[list[np.ndarray]]:
    """Estimate reduced density matrices via Pauli-basis measurements.

    Runs X, Y, Z measurement circuits for each qubit.
    Requires 3 × n_qubits circuits per data point.
    """
    from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
    from qiskit.compiler import transpile
    from qiskit_ibm_runtime import SamplerV2, Session

    n_q = feature_map.num_qubits
    params = sorted(feature_map.parameters, key=lambda p: p.name)
    all_rdms: list[list[np.ndarray]] = []

    for x in X:
        bound = feature_map.assign_parameters(dict(zip(params, x)))
        rdms = []
        for q in range(n_q):
            # Circuits for X, Y, Z measurements on qubit q
            bloch = np.zeros(3)  # [<X>, <Y>, <Z>]
            for basis_idx, (basis, pre_gates) in enumerate([
                ("Z", []),
                ("X", [("h", q)]),
                ("Y", [("sdg", q), ("h", q)]),
            ]):
                qc = QuantumCircuit(n_q, 1)
                qc.compose(bound, inplace=True)
                for gate_name, target in pre_gates:
                    getattr(qc, gate_name)(target)
                qc.measure(q, 0)
                t_qc = transpile(qc, backend=backend, optimization_level=1)
                with Session(backend=backend) as session:
                    sampler = SamplerV2(mode=session)
                    job = sampler.run([t_qc], shots=shots)
                    counts = job.result()[0].data.c.get_counts()
                n_0 = counts.get("0", 0)
                n_1 = counts.get("1", 0)
                exp_z = (n_0 - n_1) / shots
                if basis == "Z":
                    bloch[2] = exp_z
                elif basis == "X":
                    bloch[0] = exp_z
                else:
                    bloch[1] = exp_z

            # Reconstruct 2x2 density matrix from Bloch vector
            rho = 0.5 * (
                np.eye(2)
                + bloch[0] * np.array([[0, 1], [1, 0]])
                + bloch[1] * np.array([[0, -1j], [1j, 0]])
                + bloch[2] * np.array([[1, 0], [0, -1]])
            )
            rdms.append(rho)
        all_rdms.append(rdms)
    return all_rdms


def pqk_matrix(
    X_a: np.ndarray,
    X_b: np.ndarray,
    feature_map,
    backend: str = "simulator",
    ibmq_token: str | None = None,
    ibmq_instance: str = "ibm-q/open/main",
) -> tuple[np.ndarray, int]:
    """Compute the projected quantum kernel matrix K[i,j] = Σ_q Tr[ρ_q(x_a_i) ρ_q(x_b_j)].

    Returns (K, n_circuits) where n_circuits = len(X_a) + len(X_b).
    """
    n_q = feature_map.num_qubits

    if backend == "simulator":
        rdms_a = _rdms_statevector(X_a, feature_map)
        rdms_b = _rdms_statevector(X_b, feature_map)
        n_circuits = len(X_a) + len(X_b)
    elif backend == "fake_brisbane":
        from qiskit_ibm_runtime.fake_provider import FakeBrisbane
        bknd = FakeBrisbane()
        rdms_a = _rdms_from_sampler(X_a, feature_map, bknd)
        rdms_b = _rdms_from_sampler(X_b, feature_map, bknd)
        n_circuits = 3 * n_q * (len(X_a) + len(X_b))
    elif backend == "ibmq":
        from qiskit_ibm_runtime import QiskitRuntimeService
        service = QiskitRuntimeService(
            channel="ibm_quantum",
            token=ibmq_token,
            instance=ibmq_instance,
        )
        bknd = service.least_busy(
            operational=True, simulator=False, min_num_qubits=n_q,
        )
        log.info("IBM Quantum backend: %s", bknd.name)
        rdms_a = _rdms_from_sampler(X_a, feature_map, bknd)
        rdms_b = _rdms_from_sampler(X_b, feature_map, bknd)
        n_circuits = 3 * n_q * (len(X_a) + len(X_b))
    else:
        raise ValueError(f"Unknown backend: {backend!r}")

    K = np.zeros((len(X_a), len(X_b)))
    for i, rdm_row_a in enumerate(rdms_a):
        for j, rdm_row_b in enumerate(rdms_b):
            K[i, j] = sum(
                np.real(np.trace(rdm_row_a[q] @ rdm_row_b[q]))
                for q in range(n_q)
            )
    # Normalise to [0, 1]
    K /= n_q
    return K, n_circuits


def kernel_alignment(K: np.ndarray, y: np.ndarray) -> float:
    """Frobenius kernel alignment A(K, K_ideal) = <K, K_ideal>_F / (‖K‖_F ‖K_ideal‖_F)."""
    y_pm = 2 * y.astype(float) - 1          # {0,1} → {-1,+1}
    K_ideal = np.outer(y_pm, y_pm)
    num = np.sum(K * K_ideal)
    den = np.linalg.norm(K, "fro") * np.linalg.norm(K_ideal, "fro")
    return float(num / den) if den > 1e-10 else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# SVM training & evaluation
# ─────────────────────────────────────────────────────────────────────────────

def _svm_metrics(y_true, y_pred, y_score) -> dict[str, float]:
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
    )
    # Guard for single-class folds
    if len(set(y_true)) < 2:
        return dict(accuracy=float(accuracy_score(y_true, y_pred)),
                    precision=0.0, recall=0.0, f1=0.0, auc_roc=0.5)
    return dict(
        accuracy=float(accuracy_score(y_true, y_pred)),
        precision=float(precision_score(y_true, y_pred, zero_division=0)),
        recall=float(recall_score(y_true, y_pred, zero_division=0)),
        f1=float(f1_score(y_true, y_pred, zero_division=0)),
        auc_roc=float(roc_auc_score(y_true, y_score)),
    )


def cross_validate_qksvm(
    X: np.ndarray,
    y: np.ndarray,
    feature_map,
    backend: str = "simulator",
    ibmq_token: str | None = None,
    ibmq_instance: str = "ibm-q/open/main",
) -> tuple[dict[str, Any], "object"]:
    """5-fold stratified CV; returns (cv_results, fitted_svm_on_full_data)."""
    from sklearn.model_selection import StratifiedKFold
    from sklearn.svm import SVC

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    fold_metrics: list[dict] = []
    total_circuits = 0

    log.info("Pre-computing full kernel matrix (%d × %d) …", len(X), len(X))
    t0 = time.monotonic()
    K_full, n_c = pqk_matrix(X, X, feature_map, backend, ibmq_token, ibmq_instance)
    total_circuits += n_c
    log.info("  Done in %.1fs  (%d circuits)", time.monotonic() - t0, n_c)
    align_full = kernel_alignment(K_full, y)
    log.info("  Full-data kernel alignment: %.4f", align_full)

    for fold_i, (tr_idx, va_idx) in enumerate(skf.split(X, y)):
        K_tr = K_full[np.ix_(tr_idx, tr_idx)]
        K_va = K_full[np.ix_(va_idx, tr_idx)]

        svm = SVC(kernel="precomputed", probability=True, C=1.0, random_state=RANDOM_SEED)
        svm.fit(K_tr, y[tr_idx])
        y_pred  = svm.predict(K_va)
        y_score = svm.predict_proba(K_va)[:, 1]
        metrics = _svm_metrics(y[va_idx], y_pred, y_score)
        metrics["kernel_alignment"] = kernel_alignment(K_tr, y[tr_idx])
        fold_metrics.append(metrics)
        log.info(
            "  fold %d/%d  acc=%.3f  f1=%.3f  auc=%.3f  ka=%.3f",
            fold_i + 1, N_FOLDS,
            metrics["accuracy"], metrics["f1"],
            metrics["auc_roc"], metrics["kernel_alignment"],
        )

    # Fit on full data
    svm_final = SVC(kernel="precomputed", probability=True, C=1.0, random_state=RANDOM_SEED)
    svm_final.fit(K_full, y)

    summary: dict[str, Any] = {
        "n_samples":       len(X),
        "n_positives":     int(y.sum()),
        "n_negatives":     int((y == 0).sum()),
        "n_folds":         N_FOLDS,
        "total_circuits":  total_circuits,
        "kernel_alignment_full": round(align_full, 4),
        "K_full":          K_full,            # keep for scorer
        "svm_final":       svm_final,
    }
    for metric_name in ["accuracy", "precision", "recall", "f1", "auc_roc", "kernel_alignment"]:
        vals = [fm[metric_name] for fm in fold_metrics]
        summary[f"mean_{metric_name}"] = round(float(np.mean(vals)), 4)
        summary[f"std_{metric_name}"]  = round(float(np.std(vals)), 4)

    return summary, svm_final


def cross_validate_rbf(X: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    """Classical RBF-kernel SVC baseline; same 5-fold CV."""
    from sklearn.model_selection import StratifiedKFold
    from sklearn.svm import SVC
    from sklearn.preprocessing import StandardScaler

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    fold_metrics: list[dict] = []

    for tr_idx, va_idx in skf.split(X, y):
        sc  = StandardScaler()
        Xtr = sc.fit_transform(X[tr_idx])
        Xva = sc.transform(X[va_idx])
        svm = SVC(kernel="rbf", probability=True, C=1.0, random_state=RANDOM_SEED)
        svm.fit(Xtr, y[tr_idx])
        y_pred  = svm.predict(Xva)
        y_score = svm.predict_proba(Xva)[:, 1]
        fold_metrics.append(_svm_metrics(y[va_idx], y_pred, y_score))

    summary: dict[str, Any] = {"model": "rbf_classical"}
    for metric_name in ["accuracy", "precision", "recall", "f1", "auc_roc"]:
        vals = [fm[metric_name] for fm in fold_metrics]
        summary[f"mean_{metric_name}"] = round(float(np.mean(vals)), 4)
        summary[f"std_{metric_name}"]  = round(float(np.std(vals)), 4)
    return summary


# ─────────────────────────────────────────────────────────────────────────────
# All-pairs scoring
# ─────────────────────────────────────────────────────────────────────────────

def enumerate_scoring_pairs(
    tablet_seqs: dict[str, list[str]],
    stats: CorpusStats,
    confirmed: set[tuple],
    max_pairs: int = MAX_SCORING_PAIRS,
) -> list[PositionPair]:
    """Enumerate cross-tablet position pairs meeting frequency + distance filters."""
    # Pre-index high-frequency sign positions per tablet
    high_freq_signs = {s for s, p in stats.freq.items() if p > FREQ_THRESHOLD}
    tablet_hf_positions: dict[str, list[tuple[int, str]]] = {}
    tablets = list(tablet_seqs.keys())
    for tab in tablets:
        seq = tablet_seqs[tab]
        pos_signs = [(i, s) for i, s in enumerate(seq) if s in high_freq_signs]
        if pos_signs:
            tablet_hf_positions[tab] = pos_signs

    hf_tablets = list(tablet_hf_positions.keys())
    pairs: list[PositionPair] = []
    rng = random.Random(RANDOM_SEED + 1)

    # Shuffle tablet pairs to avoid systematic bias when capping
    tablet_pairs = [(hf_tablets[i], hf_tablets[j])
                    for i in range(len(hf_tablets))
                    for j in range(i + 1, len(hf_tablets))]
    rng.shuffle(tablet_pairs)

    for ta, tb in tablet_pairs:
        if len(pairs) >= max_pairs:
            break
        pos_a_list = tablet_hf_positions[ta]
        pos_b_list = tablet_hf_positions[tb]
        len_a = len(tablet_seqs[ta])
        len_b = len(tablet_seqs[tb])
        for (pos_a, sa) in pos_a_list:
            for (pos_b, sb) in pos_b_list:
                norm_a = pos_a / max(len_a - 1, 1)
                norm_b = pos_b / max(len_b - 1, 1)
                if abs(norm_a - norm_b) > POS_DIST_THRESHOLD:
                    continue
                key1 = (ta, pos_a, tb, pos_b)
                key2 = (tb, pos_b, ta, pos_a)
                if key1 in confirmed or key2 in confirmed:
                    continue
                pairs.append(PositionPair(
                    tablet_a=ta, pos_a=pos_a, sign_a=sa,
                    tablet_b=tb, pos_b=pos_b, sign_b=sb,
                ))
            if len(pairs) >= max_pairs:
                break

    log.info("Enumerated %d scoring candidate pairs", len(pairs))
    return pairs


def score_pairs(
    pairs: list[PositionPair],
    X_train: np.ndarray,
    train_rdms: list[list[np.ndarray]],
    svm_final,
    feature_map,
    tablet_seqs: dict[str, list[str]],
    stats: CorpusStats,
    embeddings: dict[str, np.ndarray] | None,
) -> list[tuple[PositionPair, float]]:
    """Score candidate pairs. Returns (pair, decision_value) list, threshold-filtered."""
    if not pairs:
        return []
    X_score = compute_feature_matrix(pairs, tablet_seqs, stats, embeddings)

    log.info("Computing PQK for %d scoring pairs …", len(pairs))
    t0 = time.monotonic()
    rdms_score = _rdms_statevector(X_score, feature_map)
    n_q = feature_map.num_qubits

    # K_score[i, j] = PQK(x_score_i, x_train_j)
    K_score = np.zeros((len(pairs), len(X_train)))
    for i, rdms_i in enumerate(rdms_score):
        for j, rdms_j in enumerate(train_rdms):
            K_score[i, j] = sum(
                np.real(np.trace(rdms_i[q] @ rdms_j[q]))
                for q in range(n_q)
            ) / n_q

    log.info("  PQK scoring done in %.1fs", time.monotonic() - t0)

    # Decision function using support vectors
    scores = svm_final.decision_function(K_score)
    return [(pair, float(score)) for pair, score in zip(pairs, scores)]


def group_into_soft_passages(
    scored_pairs: list[tuple[PositionPair, float]],
    threshold: float,
) -> list[SoftParallelCandidate]:
    """Chain adjacent position pairs on the same tablet pair into candidate passages."""
    # Filter by threshold
    hits = [(p, s) for p, s in scored_pairs if s >= threshold]
    if not hits:
        return []

    # Group by tablet pair
    by_tablet: defaultdict[tuple[str, str], list[tuple[PositionPair, float]]] = defaultdict(list)
    for pair, score in hits:
        key = (min(pair.tablet_a, pair.tablet_b), max(pair.tablet_a, pair.tablet_b))
        by_tablet[key].append((pair, score))

    candidates: list[SoftParallelCandidate] = []
    for (ta, tb), group in by_tablet.items():
        # Sort by position in tablet_a (normalized)
        group.sort(key=lambda ps: ps[0].pos_a)
        # Greedy chaining: gap ≤ 3 positions
        chains: list[list[tuple[PositionPair, float]]] = []
        current: list[tuple[PositionPair, float]] = []
        for ps in group:
            if not current or ps[0].pos_a - current[-1][0].pos_a <= 3:
                current.append(ps)
            else:
                if len(current) >= 2:
                    chains.append(current)
                current = [ps]
        if len(current) >= 2:
            chains.append(current)

        for chain in chains:
            ps_list, scores = zip(*chain)
            signs_a = [p.sign_a for p in ps_list]
            signs_b = [p.sign_b for p in ps_list]
            features = [
                compute_feature_vector(p, {}, CorpusStats({}, {}, {}, {}, {}, {}, 0), None).tolist()
                for p in ps_list
            ]
            candidates.append(SoftParallelCandidate(
                tablet_pair=(ta, tb),
                position_range_a=(ps_list[0].pos_a, ps_list[-1].pos_a),
                position_range_b=(ps_list[0].pos_b, ps_list[-1].pos_b),
                signs_tablet_a=signs_a,
                signs_tablet_b=signs_b,
                svm_scores=list(scores),
                feature_vectors=features,
                nearest_confirmed_passage="",   # filled in below
                mean_svm_score=float(np.mean(scores)),
            ))

    candidates.sort(key=lambda c: -c.mean_svm_score)
    return candidates


def _find_nearest_passage(
    candidate: SoftParallelCandidate,
    passages: list[dict],
) -> str:
    """Return passage_id of the confirmed passage most similar by tablet overlap."""
    ca_tablets = set(candidate.tablet_pair)
    best_pid   = ""
    best_score = -1
    for p in passages:
        att_tablets = {a["tablet"] for a in p["attestations"]}
        overlap = len(ca_tablets & att_tablets) / max(len(ca_tablets | att_tablets), 1)
        if overlap > best_score:
            best_score = overlap
            best_pid   = p["passage_id"]
    return best_pid


# ─────────────────────────────────────────────────────────────────────────────
# Output writing
# ─────────────────────────────────────────────────────────────────────────────

def write_soft_parallels(
    candidates: list[SoftParallelCandidate],
    passages: list[dict],
    output_path: Path,
    training_summary: dict,
    rbf_summary: dict,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    serialized = []
    for c in candidates:
        nearest = _find_nearest_passage(c, passages)
        serialized.append({
            "tablet_pair":               list(c.tablet_pair),
            "position_range":            {
                "tablet_a": list(c.position_range_a),
                "tablet_b": list(c.position_range_b),
            },
            "signs_tablet_a":            c.signs_tablet_a,
            "signs_tablet_b":            c.signs_tablet_b,
            "svm_score":                 round(c.mean_svm_score, 4),
            "svm_scores_per_position":   [round(s, 4) for s in c.svm_scores],
            "feature_vector":            [round(v, 5) for v in (
                c.feature_vectors[0] if c.feature_vectors else []
            )],
            "nearest_confirmed_passage": nearest,
        })
    output = {
        "generated_by":   "run_qksvm_parallels.py",
        "n_candidates":   len(candidates),
        "score_threshold": SCORE_THRESHOLD,
        "cv_quantum":     {k: v for k, v in training_summary.items()
                           if not isinstance(v, (np.ndarray, object.__class__))
                           and k not in ("K_full", "svm_final")},
        "cv_classical":   rbf_summary,
        "candidates":     serialized,
    }
    output_path.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info("Soft parallels written to %s  (%d candidates)", output_path, len(candidates))


def write_cribs(
    top5: list[SoftParallelCandidate],
    passages: list[dict],
    crib_path: Path,
) -> None:
    """Write top-5 candidates in parallel_variants_auto.json format for MCMC cribs."""
    crib_path.parent.mkdir(parents=True, exist_ok=True)
    crib_passages = []
    for rank, c in enumerate(top5[:5]):
        pid = f"SOFT_P{rank+1:03d}_{c.tablet_pair[0]}{c.tablet_pair[1]}"
        n_pos = len(c.signs_tablet_a)
        # Build synthetic attestations
        atts = [
            {
                "tablet":         c.tablet_pair[0],
                "form":           c.signs_tablet_a,
                "stratum":        "undated",
                "start_position": c.position_range_a[0],
            },
            {
                "tablet":         c.tablet_pair[1],
                "form":           c.signs_tablet_b,
                "stratum":        "undated",
                "start_position": c.position_range_b[0],
            },
        ]
        nearest = _find_nearest_passage(c, passages)
        crib_passages.append({
            "passage_id":             pid,
            "canonical_form":         c.signs_tablet_a,
            "n_tablets":              2,
            "attestations":           atts,
            "diachronic_changes":     [],
            "interest_score":         round(c.mean_svm_score, 4),
            "soft_parallel":          True,
            "nearest_confirmed":      nearest,
            "svm_score":              round(c.mean_svm_score, 4),
        })
    crib_path.write_text(
        json.dumps({"passages": crib_passages, "_source": "qksvm_soft_parallels"}, indent=2),
        encoding="utf-8",
    )
    log.info("Crib file written to %s  (%d passages)", crib_path, len(crib_passages))


# ─────────────────────────────────────────────────────────────────────────────
# Redteam agent tool handler
# ─────────────────────────────────────────────────────────────────────────────

def handle_find_soft_parallels_tool(
    svm_score_threshold: float = SCORE_THRESHOLD,
    backend: str = "simulator",
) -> dict[str, Any]:
    """Entry point called by redteam_agent.py tool dispatcher."""
    result = main(
        score_threshold=svm_score_threshold,
        backend=backend,
        inject_as_cribs=False,
        return_result=True,
    )
    candidates = result.get("candidates", [])
    return {
        "n_soft_parallels": len(candidates),
        "top_5": candidates[:5],
        "cv_f1_quantum": result.get("cv_quantum", {}).get("mean_f1"),
        "cv_f1_classical": result.get("cv_classical", {}).get("mean_f1"),
        "output_path": str(_SOFT_PARALLELS_OUT),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main(
    corpus_dir:        Path = _CORPUS_DIR,
    variants_json:     Path = _VARIANTS_JSON,
    output_path:       Path = _SOFT_PARALLELS_OUT,
    crib_path:         Path = _CRIB_OUT,
    backend:           str  = "simulator",
    score_threshold:   float = SCORE_THRESHOLD,
    inject_as_cribs:   bool = False,
    ibmq_token:        str | None = None,
    ibmq_instance:     str = "ibm-q/open/main",
    ibmq_backend_name: str | None = None,
    n_negatives:       int = N_NEGATIVES,
    return_result:     bool = False,
) -> dict[str, Any]:

    rng = random.Random(RANDOM_SEED)

    # ── Load data ─────────────────────────────────────────────────────────────
    log.info("Loading corpus from %s …", corpus_dir)
    tablet_seqs = load_corpus(corpus_dir)
    if not tablet_seqs:
        raise FileNotFoundError(f"No corpus JSONs in {corpus_dir}")
    log.info("  %d tablets loaded", len(tablet_seqs))

    log.info("Loading parallel passages from %s …", variants_json)
    passages = load_parallel_passages(variants_json)
    log.info("  %d confirmed passages", len(passages))

    log.info("Building corpus statistics …")
    stats = build_corpus_stats(tablet_seqs, passages)
    log.info(
        "  %d tokens, %d sign types, max_MI=%.3f",
        stats.total_tokens,
        len(stats.freq),
        max(stats.bigram_mi.values()) if stats.bigram_mi else 0,
    )

    log.info("Loading embeddings …")
    embeddings = load_embeddings(_EMBEDDINGS_CACHE)

    # ── Build training set ────────────────────────────────────────────────────
    log.info("Building training set …")
    positives = build_positive_pairs(passages, tablet_seqs)
    log.info("  %d positive pairs from %d passages", len(positives), len(passages))
    confirmed = _confirmed_pair_set(positives)
    negatives = sample_negatives(tablet_seqs, confirmed, n_negatives, rng)
    log.info("  %d negative pairs sampled", len(negatives))

    all_pairs  = positives + negatives
    labels     = np.array([p.label for p in all_pairs], dtype=int)
    log.info("  Training set: %d total (%.1f%% positive)", len(all_pairs), 100 * labels.mean())

    X_all = compute_feature_matrix(all_pairs, tablet_seqs, stats, embeddings)
    log.info("  Feature matrix: %s  finite=%s", X_all.shape, np.isfinite(X_all).all())

    # Save training data
    _TRAINING_NPZ.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(_TRAINING_NPZ), X=X_all, y=labels)
    log.info("  Training data saved to %s", _TRAINING_NPZ)

    # ── Build quantum feature map ─────────────────────────────────────────────
    log.info("Building ZZFeatureMap(dim=%d, reps=2) …", N_FEATURES)
    feature_map = build_feature_map(N_FEATURES, reps=2)
    log.info("  %d qubits, circuit depth %d", feature_map.num_qubits, feature_map.depth())

    # ── Quantum CV ────────────────────────────────────────────────────────────
    log.info("Running %d-fold stratified CV — QK-SVM (backend=%s) …", N_FOLDS, backend)
    cv_quantum, svm_final = cross_validate_qksvm(
        X_all, labels, feature_map,
        backend=backend,
        ibmq_token=ibmq_token,
        ibmq_instance=ibmq_instance,
    )
    log.info(
        "  QK-SVM  acc=%.3f±%.3f  f1=%.3f±%.3f  auc=%.3f±%.3f  "
        "ka=%.3f  circuits=%d",
        cv_quantum["mean_accuracy"], cv_quantum["std_accuracy"],
        cv_quantum["mean_f1"], cv_quantum["std_f1"],
        cv_quantum["mean_auc_roc"], cv_quantum["std_auc_roc"],
        cv_quantum["kernel_alignment_full"],
        cv_quantum["total_circuits"],
    )

    # ── Classical RBF baseline ────────────────────────────────────────────────
    log.info("Running %d-fold CV — classical RBF SVM …", N_FOLDS)
    cv_rbf = cross_validate_rbf(X_all, labels)
    log.info(
        "  RBF SVM  acc=%.3f±%.3f  f1=%.3f±%.3f  auc=%.3f±%.3f",
        cv_rbf["mean_accuracy"], cv_rbf["std_accuracy"],
        cv_rbf["mean_f1"], cv_rbf["std_f1"],
        cv_rbf["mean_auc_roc"], cv_rbf["std_auc_roc"],
    )

    # ── Score all candidate pairs ─────────────────────────────────────────────
    cap = MAX_SCORING_PAIRS if backend == "simulator" else 1_000
    log.info("Enumerating scoring candidate pairs (cap=%d) …", cap)
    scoring_pairs = enumerate_scoring_pairs(tablet_seqs, stats, confirmed, max_pairs=cap)

    # Pre-compute training RDMs once (used for K(score, train))
    log.info("Pre-computing training RDMs for scoring …")
    train_rdms = _rdms_statevector(X_all, feature_map)

    log.info("Scoring %d candidate pairs …", len(scoring_pairs))
    scored = score_pairs(
        scoring_pairs, X_all, train_rdms,
        svm_final, feature_map,
        tablet_seqs, stats, embeddings,
    )
    above_threshold = [(p, s) for p, s in scored if s >= score_threshold]
    log.info(
        "  %d / %d pairs above threshold %.2f",
        len(above_threshold), len(scored), score_threshold,
    )

    # ── Group into passages ───────────────────────────────────────────────────
    candidates = group_into_soft_passages(scored, score_threshold)
    log.info("  %d soft parallel candidates found", len(candidates))

    # ── Print summary ─────────────────────────────────────────────────────────
    _print_summary(cv_quantum, cv_rbf, candidates)

    # ── Write outputs ─────────────────────────────────────────────────────────
    write_soft_parallels(candidates, passages, output_path, cv_quantum, cv_rbf)
    if inject_as_cribs:
        write_cribs(candidates[:5], passages, crib_path)

    result = {
        "cv_quantum":  {k: v for k, v in cv_quantum.items()
                        if k not in ("K_full", "svm_final")},
        "cv_classical": cv_rbf,
        "n_scoring_pairs": len(scoring_pairs),
        "n_above_threshold": len(above_threshold),
        "candidates": [
            {
                "tablet_pair":               list(c.tablet_pair),
                "signs_a":                   c.signs_tablet_a,
                "signs_b":                   c.signs_tablet_b,
                "mean_svm_score":            round(c.mean_svm_score, 4),
                "nearest_confirmed_passage": _find_nearest_passage(c, passages),
            }
            for c in candidates
        ],
    }
    if return_result:
        return result
    return result


def _print_summary(cv_q, cv_rbf, candidates) -> None:
    print(f"\n{'═' * 70}")
    print("  QK-SVM Soft Parallel Detection — Rongorongo")
    print(f"{'═' * 70}")
    print(f"\n  Quantum kernel (PQK):")
    print(f"    accuracy  {cv_q['mean_accuracy']:.3f} ± {cv_q['std_accuracy']:.3f}")
    print(f"    F1        {cv_q['mean_f1']:.3f} ± {cv_q['std_f1']:.3f}")
    print(f"    AUC-ROC   {cv_q['mean_auc_roc']:.3f} ± {cv_q['std_auc_roc']:.3f}")
    print(f"    ker-align {cv_q['kernel_alignment_full']:.4f}")
    print(f"    circuits  {cv_q['total_circuits']}")
    print(f"\n  Classical RBF kernel:")
    print(f"    accuracy  {cv_rbf['mean_accuracy']:.3f} ± {cv_rbf['std_accuracy']:.3f}")
    print(f"    F1        {cv_rbf['mean_f1']:.3f} ± {cv_rbf['std_f1']:.3f}")
    print(f"    AUC-ROC   {cv_rbf['mean_auc_roc']:.3f} ± {cv_rbf['std_auc_roc']:.3f}")
    if candidates:
        print(f"\n  Top soft parallel candidates:")
        for c in candidates[:5]:
            print(f"    [{c.tablet_pair[0]}↔{c.tablet_pair[1]}] "
                  f"pos_a={c.position_range_a}  pos_b={c.position_range_b}  "
                  f"score={c.mean_svm_score:.3f}  "
                  f"signs={'·'.join(c.signs_tablet_a)}")
    else:
        print(f"\n  No soft parallels found above threshold {SCORE_THRESHOLD}.")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Projected QK-SVM soft parallel passage detector.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--corpus-dir",    type=Path, default=_CORPUS_DIR)
    p.add_argument("--variants-file", type=Path, default=_VARIANTS_JSON)
    p.add_argument("--output",        type=Path, default=_SOFT_PARALLELS_OUT)
    p.add_argument("--backend",
                   choices=["simulator", "fake_brisbane", "ibmq"],
                   default="simulator")
    p.add_argument("--ibmq-token",    default=None)
    p.add_argument("--ibmq-instance", default="ibm-q/open/main")
    p.add_argument("--ibmq-backend",  default=None)
    p.add_argument("--score-threshold", type=float, default=SCORE_THRESHOLD)
    p.add_argument("--n-negatives",   type=int, default=N_NEGATIVES)
    p.add_argument("--inject-as-cribs", action="store_true",
                   help="Write top-5 candidates to data/parallels/soft_parallel_candidates.json")
    return p.parse_args()


if __name__ == "__main__":
    import os
    args = _parse_args()
    main(
        corpus_dir=args.corpus_dir,
        variants_json=args.variants_file,
        output_path=args.output,
        backend=args.backend,
        score_threshold=args.score_threshold,
        inject_as_cribs=args.inject_as_cribs,
        ibmq_token=args.ibmq_token or os.environ.get("IBMQ_TOKEN"),
        ibmq_instance=args.ibmq_instance,
        ibmq_backend_name=args.ibmq_backend,
        n_negatives=args.n_negatives,
    )
