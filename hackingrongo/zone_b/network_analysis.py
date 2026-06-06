"""
hackingrongo.zone_b.network_analysis
======================================

Network centrality analysis for the rongorongo bigram graph.

Nodes are sign types; directed edges s_i → s_j carry weight = PMI(s_i, s_j).
Four graph variants are supported: corpus-wide, per-tablet, pre-contact
stratum, and post-contact stratum.

Classical centrality measures
------------------------------
  in_degree, out_degree, betweenness (Brandes), closeness, PageRank (α=0.85),
  eigenvector centrality, HITS hub/authority scores.

Determinative candidates
-------------------------
  Signs with betweenness > 2 × mean AND frequency < median.

Diachronic shift
-----------------
  Δ betweenness and Δ PageRank between pre- and post-contact graphs.

Quantum PageRank  (optional, requires qiskit)
----------------------------------------------
  Szegedy discrete-time quantum walk on the Google matrix.  State space:
  C^M ⊗ C^M where M = 2^n_pos (position register of n_pos qubits).
  Walk operator W = SWAP · (2Π_T − I).  Quantum PageRank = time-averaged
  first-register marginal over t walk steps.

Quantum Fiedler  (optional, requires qiskit + scipy)
------------------------------------------------------
  QPE on the normalised graph Laplacian restricted to top-64 signs.
  Circuit: n_qpe ancilla qubits + n_pos position qubits.
  Unitary U = exp(2πi L_norm / Λ); Fiedler value extracted from peak phase.
"""

from __future__ import annotations

import math
import logging
from collections import Counter
from typing import Any

import numpy as np

log = logging.getLogger(__name__)

try:
    import networkx as nx
    _NX = True
except ImportError:
    _NX = False
    log.warning("networkx not installed — graph functions unavailable.")

try:
    import scipy.sparse as _sp
    import scipy.sparse.linalg as _spla
    from scipy.linalg import expm as _expm
    _SCIPY = True
except ImportError:
    _SCIPY = False


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def build_pmi_graph(
    sequences: list[list[str]],
    min_cofreq: int = 2,
    pmi_floor: float = 0.0,
) -> "nx.DiGraph":
    """Build a directed bigram graph weighted by pointwise mutual information.

    PMI(i,j) = log P(i,j) / (P(i)·P(j)).  Only edges with PMI ≥ pmi_floor
    and co-occurrence count ≥ min_cofreq are retained.

    Node attribute ``freq`` stores the raw unigram count.
    Edge attribute ``count`` stores the raw bigram count.
    """
    if not _NX:
        raise ImportError("networkx is required.")

    unigram: Counter[str] = Counter()
    bigram:  Counter[tuple[str, str]] = Counter()

    for seq in sequences:
        for tok in seq:
            unigram[tok] += 1
        for i in range(len(seq) - 1):
            bigram[(seq[i], seq[i + 1])] += 1

    total_u = sum(unigram.values())
    total_b = sum(bigram.values())

    if total_u == 0 or total_b == 0:
        return nx.DiGraph()

    G = nx.DiGraph()
    for sign, cnt in unigram.items():
        G.add_node(sign, freq=cnt)

    for (si, sj), cnt in bigram.items():
        if cnt < min_cofreq:
            continue
        p_i  = unigram[si] / total_u
        p_j  = unigram[sj] / total_u
        p_ij = cnt / total_b
        pmi  = math.log(p_ij / (p_i * p_j))
        if pmi >= pmi_floor:
            G.add_edge(si, sj, weight=float(pmi), count=int(cnt))

    log.info(
        "PMI graph: %d nodes, %d edges (min_cofreq=%d, pmi_floor=%.1f).",
        G.number_of_nodes(), G.number_of_edges(), min_cofreq, pmi_floor,
    )
    return G


# ---------------------------------------------------------------------------
# Classical centrality
# ---------------------------------------------------------------------------

def compute_centralities(G: "nx.DiGraph") -> dict[str, dict[str, float]]:
    """Compute all centrality measures for a directed weighted graph.

    Returns a dict mapping measure name → {node: float}.
    """
    if not _NX:
        raise ImportError("networkx is required.")
    if G.number_of_nodes() == 0:
        return {}

    result: dict[str, dict[str, float]] = {}

    result["in_degree"]  = dict(nx.in_degree_centrality(G))
    result["out_degree"] = dict(nx.out_degree_centrality(G))

    result["betweenness"] = nx.betweenness_centrality(
        G, weight="weight", normalized=True,
    )

    # Closeness on directed graph: use out-going direction (default).
    # networkx uses distance = 1/weight for closeness; we pass the
    # reciprocal as the distance attribute.
    G2 = G.copy()
    for u, v, d in G2.edges(data=True):
        d["distance"] = 1.0 / max(d.get("weight", 1.0), 1e-9)
    result["closeness"] = dict(nx.closeness_centrality(G2, distance="distance"))

    result["pagerank"] = nx.pagerank(G, alpha=0.85, weight="weight")

    try:
        result["eigenvector"] = nx.eigenvector_centrality(
            G, weight="weight", max_iter=1000, tol=1e-6,
        )
    except nx.PowerIterationFailedConvergence:
        log.warning("Eigenvector centrality did not converge — returning zeros.")
        result["eigenvector"] = {n: 0.0 for n in G.nodes()}

    try:
        hubs, authorities = nx.hits(G, max_iter=1000, tol=1e-6)
        result["hits_hub"]       = hubs
        result["hits_authority"] = authorities
    except nx.PowerIterationFailedConvergence:
        log.warning("HITS did not converge.")
        result["hits_hub"]       = {n: 0.0 for n in G.nodes()}
        result["hits_authority"] = {n: 0.0 for n in G.nodes()}

    return result


def determinative_candidates(
    G: "nx.DiGraph",
    centralities: dict[str, dict[str, float]],
) -> list[dict[str, Any]]:
    """Return signs that are structural bridges but not high-frequency.

    Criterion: betweenness > 2 × mean(betweenness) AND
               freq < median(freq).

    These are signs that control information flow in the graph despite
    being rare — a hallmark of grammatical/determinative function.
    """
    btwn  = centralities.get("betweenness", {})
    freqs = dict(G.nodes(data="freq", default=1))
    if not btwn:
        return []

    btwn_vals  = np.array(list(btwn.values()), dtype=float)
    freq_vals  = np.array([freqs.get(n, 1) for n in btwn], dtype=float)
    btwn_mean  = float(np.mean(btwn_vals))
    freq_median = float(np.median(freq_vals))

    candidates = []
    for node in btwn:
        b = btwn[node]
        f = freqs.get(node, 1)
        if b > 2.0 * btwn_mean and f < freq_median:
            candidates.append({
                "sign":        node,
                "betweenness": round(b, 6),
                "freq":        f,
                "pagerank":    round(centralities.get("pagerank", {}).get(node, 0.0), 6),
            })

    candidates.sort(key=lambda x: -x["betweenness"])
    log.info(
        "Determinative candidates: %d (btwn > %.4f, freq < %.0f).",
        len(candidates), 2.0 * btwn_mean, freq_median,
    )
    return candidates


# ---------------------------------------------------------------------------
# Diachronic shift
# ---------------------------------------------------------------------------

def diachronic_shift(
    pre_centralities: dict[str, dict[str, float]],
    post_centralities: dict[str, dict[str, float]],
) -> dict[str, Any]:
    """Report per-sign Δ betweenness and Δ PageRank between strata.

    Signs with |Δ betweenness| > 1 std are flagged as key-change candidates.
    """
    pre_b  = pre_centralities.get("betweenness", {})
    post_b = post_centralities.get("betweenness", {})
    pre_pr = pre_centralities.get("pagerank", {})
    post_pr = post_centralities.get("pagerank", {})

    all_signs = set(pre_b) | set(post_b)
    deltas: list[dict[str, Any]] = []
    for sign in sorted(all_signs):
        db  = post_b.get(sign, 0.0) - pre_b.get(sign, 0.0)
        dpr = post_pr.get(sign, 0.0) - pre_pr.get(sign, 0.0)
        deltas.append({
            "sign":               sign,
            "delta_betweenness":  round(db, 6),
            "delta_pagerank":     round(dpr, 6),
            "pre_betweenness":    round(pre_b.get(sign, 0.0), 6),
            "post_betweenness":   round(post_b.get(sign, 0.0), 6),
        })

    if deltas:
        db_arr = np.array([d["delta_betweenness"] for d in deltas])
        std    = float(np.std(db_arr))
        for d in deltas:
            d["key_change_candidate"] = abs(d["delta_betweenness"]) > std
    else:
        std = 0.0

    key_changes = [d for d in deltas if d.get("key_change_candidate")]
    key_changes.sort(key=lambda x: -abs(x["delta_betweenness"]))
    log.info(
        "Diachronic shift: %d signs, std(Δ btwn)=%.4f, %d key-change candidates.",
        len(deltas), std, len(key_changes),
    )
    return {
        "per_sign":          deltas,
        "key_change_candidates": key_changes,
        "delta_betweenness_std": round(std, 6),
    }


# ---------------------------------------------------------------------------
# Helpers for quantum routines
# ---------------------------------------------------------------------------

def _google_matrix(
    G: "nx.DiGraph",
    nodes: list[str],
    damping: float = 0.85,
) -> np.ndarray:
    """Build column-stochastic Google matrix for PageRank walk.

    G_mat[:,i] = damping * T[:,i] + (1-damping)/N * ones
    where T[:,i] = out-edge weights from i, normalised.
    Dangling nodes (no out-edges) teleport uniformly.
    """
    N = len(nodes)
    idx = {n: i for i, n in enumerate(nodes)}
    T = np.zeros((N, N), dtype=float)

    for i, node in enumerate(nodes):
        out_edges = [(idx[v], d["weight"]) for v, d in G[node].items() if v in idx]
        if out_edges:
            total = sum(w for _, w in out_edges)
            for j, w in out_edges:
                T[j, i] = w / total
        else:
            T[:, i] = 1.0 / N  # dangling: uniform teleport

    return damping * T + (1.0 - damping) / N * np.ones((N, N), dtype=float)


def _szegedy_walk_matrix(
    T: np.ndarray,
    M: int,
) -> "_sp.csr_matrix":
    """Build sparse Szegedy walk operator W = SWAP · (2Π_T − I).

    State space: C^M ⊗ C^M.  Flat index k = i*M + j → |i⟩|j⟩.

    Π_T is block-diagonal:
      Π_T[(i,b),(i,d)] = √T[b][i] · √T[d][i]
    """
    if not _SCIPY:
        raise ImportError("scipy is required for quantum walk.")

    N = T.shape[0]
    sqT = np.sqrt(np.clip(T, 0, None))

    # Build R = 2Π_T − I (block-diagonal over source nodes)
    rows, cols, vals = [], [], []
    for i in range(N):
        v = sqT[:, i]       # shape (N,) — neighbours of source i
        outer = 2.0 * np.outer(v, v)  # (N,N)
        for b in range(N):
            for d in range(N):
                val = outer[b, d] - (1.0 if b == d else 0.0)
                if abs(val) > 1e-12:
                    rows.append(i * M + b)
                    cols.append(i * M + d)
                    vals.append(val)
    R = _sp.csr_matrix((vals, (rows, cols)), shape=(M * M, M * M), dtype=complex)

    # Build SWAP permutation: |i,j⟩ → |j,i⟩
    sw_rows, sw_cols = [], []
    for i in range(M):
        for j in range(M):
            sw_cols.append(i * M + j)
            sw_rows.append(j * M + i)
    S = _sp.csr_matrix(
        ([1.0] * (M * M), (sw_rows, sw_cols)),
        shape=(M * M, M * M), dtype=complex,
    )

    return S @ R


def _initial_walk_state(T: np.ndarray, M: int) -> np.ndarray:
    """Prepared initial state: 1/√N Σ_i |φ_i⟩, |φ_i⟩ = Σ_j √T[j][i] |i,j⟩."""
    N = T.shape[0]
    sqT = np.sqrt(np.clip(T, 0, None))
    psi = np.zeros(M * M, dtype=complex)
    for i in range(N):
        for j in range(N):
            psi[i * M + j] += sqT[j, i]
    norm = np.linalg.norm(psi)
    return psi / norm if norm > 0 else psi


# ---------------------------------------------------------------------------
# Quantum PageRank — Szegedy walk
# ---------------------------------------------------------------------------

def quantum_pagerank(
    G: "nx.DiGraph",
    n_steps: int = 20,
    n_pos:   int = 11,
    damping: float = 0.85,
) -> dict[str, Any]:
    """Quantum PageRank via Szegedy discrete-time quantum walk.

    Position register: n_pos qubits (addresses up to 2^n_pos node slots).
    State space: C^M ⊗ C^M where M = 2^n_pos.
    Walk operator: W = SWAP · (2Π_T − I).
    Quantum PageRank(i) = time-averaged first-register marginal over t steps.

    Returns a dict with:
      quantum_pagerank   — {sign: float}
      classical_pagerank — {sign: float}
      l1_divergence      — float
      top_divergent      — list[{sign, classical_rank, quantum_rank, delta}]
      n_nodes_used       — int (capped at 2^n_pos)
      n_qubits           — 2*ceil(log2(N)) for actual N used
    """
    if not _SCIPY:
        raise ImportError("scipy is required for quantum walk.")

    nodes = sorted(G.nodes())
    max_n = 2 ** n_pos
    if len(nodes) > max_n:
        # Use top nodes by frequency
        freq = dict(G.nodes(data="freq", default=0))
        nodes = sorted(nodes, key=lambda s: -freq.get(s, 0))[:max_n]
    N = len(nodes)
    if N < 2:
        return {"error": "Graph too small for quantum walk (need ≥ 2 nodes)."}

    # Pad to next power of two
    n_qubits_pos = max(1, math.ceil(math.log2(N + 1)))
    M = 2 ** n_qubits_pos
    total_qubits = 2 * n_qubits_pos

    log.info(
        "Quantum PageRank: N=%d nodes, M=%d (pad), %d qubits, t=%d steps.",
        N, M, total_qubits, n_steps,
    )

    T   = _google_matrix(G, nodes, damping)    # (N,N) padded implicitly via M
    # Pad T to M×M (extra rows/cols remain zero → dangling)
    T_pad = np.zeros((M, M), dtype=float)
    T_pad[:N, :N] = T
    for i in range(N, M):
        T_pad[:M, i] = 1.0 / M  # dangling

    W   = _szegedy_walk_matrix(T_pad, M)
    psi = _initial_walk_state(T_pad, M)

    # Time-average the first-register marginal
    qpr = np.zeros(M, dtype=float)
    for _ in range(n_steps):
        psi = W @ psi
        prob = np.abs(psi) ** 2
        for i in range(M):
            qpr[i] += np.sum(prob[i * M:(i + 1) * M])
    qpr /= n_steps
    qpr_nodes = {nodes[i]: float(qpr[i]) for i in range(N)}
    # Normalise to sum-1 over active nodes
    total_qpr = sum(qpr_nodes.values())
    if total_qpr > 0:
        qpr_nodes = {s: v / total_qpr for s, v in qpr_nodes.items()}

    # Classical PageRank for comparison
    cpr_full = nx.pagerank(G, alpha=damping, weight="weight")
    cpr_nodes = {s: cpr_full.get(s, 0.0) for s in nodes}
    total_cpr = sum(cpr_nodes.values())
    if total_cpr > 0:
        cpr_nodes = {s: v / total_cpr for s, v in cpr_nodes.items()}

    # L1 divergence
    l1 = float(sum(abs(qpr_nodes[s] - cpr_nodes[s]) for s in nodes))

    # Per-sign rank difference
    qpr_sorted = sorted(qpr_nodes, key=lambda s: -qpr_nodes[s])
    cpr_sorted = sorted(cpr_nodes, key=lambda s: -cpr_nodes[s])
    q_rank = {s: i + 1 for i, s in enumerate(qpr_sorted)}
    c_rank = {s: i + 1 for i, s in enumerate(cpr_sorted)}
    divergent = sorted(
        nodes, key=lambda s: -abs(q_rank[s] - c_rank[s])
    )[:10]
    top_divergent = [
        {
            "sign":          s,
            "classical_rank": c_rank[s],
            "quantum_rank":  q_rank[s],
            "rank_delta":    q_rank[s] - c_rank[s],
            "quantum_pr":    round(qpr_nodes[s], 6),
            "classical_pr":  round(cpr_nodes[s], 6),
        }
        for s in divergent
    ]

    return {
        "quantum_pagerank":   {s: round(v, 6) for s, v in qpr_nodes.items()},
        "classical_pagerank": {s: round(v, 6) for s, v in cpr_nodes.items()},
        "l1_divergence":      round(l1, 6),
        "top_divergent":      top_divergent,
        "n_nodes_used":       N,
        "n_qubits":           total_qubits,
        "n_steps":            n_steps,
    }


# ---------------------------------------------------------------------------
# Quantum Fiedler — QPE on normalised Laplacian
# ---------------------------------------------------------------------------

def _normalised_laplacian(
    G: "nx.DiGraph",
    nodes: list[str],
) -> np.ndarray:
    """Symmetric normalised Laplacian L_sym = D^(-1/2) L D^(-1/2) for the
    undirected version of G (symmetrised by averaging edge weights).
    """
    N = len(nodes)
    idx = {n: i for i, n in enumerate(nodes)}
    A = np.zeros((N, N), dtype=float)
    for u, v, d in G.edges(data=True):
        if u in idx and v in idx:
            w = d.get("weight", 1.0)
            i, j = idx[u], idx[v]
            A[i, j] += w
            A[j, i] += w          # symmetrise
    A /= 2.0

    d_vec = A.sum(axis=1)
    D_inv_sqrt = np.where(d_vec > 0, 1.0 / np.sqrt(d_vec), 0.0)
    L = np.diag(d_vec) - A
    return D_inv_sqrt[:, None] * L * D_inv_sqrt[None, :]


def quantum_fiedler(
    G: "nx.DiGraph",
    top_n: int = 64,
    n_pos:  int = 6,
    n_qpe:  int = 4,
    ic_freq: dict[str, int] | None = None,
) -> dict[str, Any]:
    """Estimate the Fiedler value via quantum phase estimation.

    Circuit: n_qpe QPE ancilla qubits + n_pos position qubits = n_qpe + n_pos total.
    Restricts to the top-min(top_n, 2^n_pos) signs by IC (or frequency).

    Unitary: U = exp(2πi L_norm / Λ) with Λ = 4 (maps eigenvalues [0,2] to phases
    [0, π/2]).  QPE returns phase ϕ ≈ λ/Λ → Fiedler value ≈ Λ · ϕ_peak.

    Returns:
      fiedler_value_classical  — exact (numpy eigh)
      fiedler_value_qpe        — estimated from QPE peak
      fiedler_vector           — {sign: float}
      community_A              — list of signs with Fiedler vector component > 0
      community_B              — list of signs with Fiedler vector component ≤ 0
      n_nodes_used             — int
      n_qubits_total           — int
      qpe_counts               — {bitstring: int}  (raw QPE measurement counts)
    """
    if not _SCIPY:
        raise ImportError("scipy is required for QPE simulation.")
    try:
        from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
        from qiskit.circuit.library import QFT
        from qiskit.quantum_info import Statevector, Operator
    except ImportError as exc:
        raise ImportError("qiskit required for quantum Fiedler.") from exc

    all_nodes = sorted(G.nodes())
    M_pos = 2 ** n_pos
    N = min(top_n, len(all_nodes), M_pos)

    # Select top-N by IC (log₂(total/freq)) or plain frequency
    freq = dict(G.nodes(data="freq", default=1))
    total_freq = sum(freq.values())
    ic = {s: -math.log2(freq.get(s, 1) / total_freq) if freq.get(s, 1) > 0 else 99.0
          for s in all_nodes}
    if ic_freq:
        ic.update({s: -math.log2(ic_freq[s] / sum(ic_freq.values()))
                   for s in ic_freq if ic_freq[s] > 0})
    nodes = sorted(all_nodes, key=lambda s: ic.get(s, 99.0))[:N]
    N = len(nodes)
    if N < 2:
        return {"error": "Too few nodes for QPE."}

    log.info(
        "Quantum Fiedler: N=%d nodes, n_pos=%d, n_qpe=%d (%d total qubits).",
        N, n_pos, n_qpe, n_pos + n_qpe,
    )

    # Build normalised Laplacian
    L_norm = _normalised_laplacian(G, nodes)

    # Classical eigendecomposition (exact reference)
    evals, evecs = np.linalg.eigh(L_norm)
    sort_idx     = np.argsort(evals)
    evals        = evals[sort_idx]
    evecs        = evecs[:, sort_idx]

    fiedler_val_classical = float(evals[1]) if len(evals) > 1 else 0.0
    fiedler_vec           = evecs[:, 1] if evecs.shape[1] > 1 else evecs[:, 0]

    # Pad L_norm to M_pos × M_pos
    L_pad = np.zeros((M_pos, M_pos), dtype=float)
    L_pad[:N, :N] = L_norm

    # Build unitary U = exp(2πi L_pad / Λ) with Λ=4
    LAMBDA = 4.0
    U_mat = _expm(2j * np.pi * L_pad / LAMBDA)

    # ── QPE circuit ──────────────────────────────────────────────────────────
    qpe_reg  = QuantumRegister(n_qpe, name="qpe")
    pos_reg  = QuantumRegister(n_pos, name="pos")
    c_reg    = ClassicalRegister(n_qpe, name="c")
    qc = QuantumCircuit(qpe_reg, pos_reg, c_reg)

    # Hadamard on all QPE qubits
    for k in range(n_qpe):
        qc.h(qpe_reg[k])

    # Initialize position register to Fiedler vector (padded)
    fv_pad = np.zeros(M_pos, dtype=complex)
    fv_pad[:N] = fiedler_vec.astype(complex)
    fv_pad /= np.linalg.norm(fv_pad)
    qc.initialize(fv_pad, pos_reg)

    # Controlled-U^(2^k) for each QPE qubit k
    for k in range(n_qpe):
        reps = 2 ** k
        # Compute U^reps = (expm)^reps — use matrix power
        U_k = np.linalg.matrix_power(U_mat, reps) if reps <= 8 else \
              _expm(2j * np.pi * reps * L_pad / LAMBDA)
        gate = Operator(U_k)
        qc.append(gate.to_instruction().control(1), [qpe_reg[k]] + list(pos_reg))

    # Inverse QFT on QPE register
    iqft = QFT(n_qpe, inverse=True, do_swaps=True)
    qc.append(iqft, qpe_reg)

    # Measure QPE register
    qc.measure(qpe_reg, c_reg)

    # Run on Statevector simulator then sample
    sv = Statevector.from_instruction(qc.remove_final_measurements(inplace=False))
    # Sample from statevector to get QPE measurement counts
    counts = sv.sample_counts(shots=8192, qargs=list(range(n_qpe)))

    # Convert raw integer keys to zero-padded bitstrings
    counts_bs: dict[str, int] = {}
    for key, cnt in counts.items():
        bs = format(int(key), f"0{n_qpe}b") if isinstance(key, int) else str(key).zfill(n_qpe)
        counts_bs[bs] = counts_bs.get(bs, 0) + cnt

    # Most frequent bitstring → phase → Fiedler estimate
    peak_bs  = max(counts_bs, key=counts_bs.__getitem__)
    peak_int = int(peak_bs, 2)
    phase    = peak_int / (2 ** n_qpe)
    fiedler_val_qpe = float(phase * LAMBDA)

    # Community bisection from Fiedler vector
    community_a = [nodes[i] for i in range(N) if fiedler_vec[i] > 0]
    community_b = [nodes[i] for i in range(N) if fiedler_vec[i] <= 0]

    log.info(
        "Quantum Fiedler: λ₂ (classical)=%.4f, QPE estimate=%.4f; "
        "community A=%d, B=%d signs.",
        fiedler_val_classical, fiedler_val_qpe, len(community_a), len(community_b),
    )

    return {
        "fiedler_value_classical": round(fiedler_val_classical, 6),
        "fiedler_value_qpe":       round(fiedler_val_qpe, 6),
        "fiedler_vector":          {nodes[i]: round(float(fiedler_vec[i]), 6) for i in range(N)},
        "community_a":             sorted(community_a),
        "community_b":             sorted(community_b),
        "n_nodes_used":            N,
        "n_qubits_total":          n_pos + n_qpe,
        "qpe_counts":              counts_bs,
        "eigenvalues_classical":   [round(float(e), 6) for e in evals[:min(10, len(evals))]],
    }
