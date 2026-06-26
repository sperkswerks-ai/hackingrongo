#!/usr/bin/env python3
# ============================================================================
# DEPRECATED — SYLLABIC SUBSTITUTION-CIPHER TRACK (set down 2026-06, in place).
# Part of the sign→phoneme substitution-cipher hypothesis, which was tested and
# set down as a recorded NEGATIVE RESULT — preserved as an archive, NOT fixed,
# tuned, or deleted. Do not extend this module. The structural/logographic track
# supersedes it. Full rationale + on-disk numbers: DEPRECATED_SYLLABIC.md (root).
# ============================================================================
"""
run_qaoa_decipherment.py — Layer 4Q-QAOA: Quantum Approximate Optimization
Algorithm for rongorongo sign→phoneme assignment.

Frames the decipherment as a QUBO over the top-N most-frequent signs and
top-M highest-LM phonemes, converts it to an Ising cost Hamiltonian, and
optimises the QAOAAnsatz parameters via classical COBYLA.  The QAOA result
for the subproblem is merged with the best MCMC hypothesis (if --init-from
is provided) to produce a full hybrid phoneme map.

Approach
--------
1. Select the top-N most-frequent signs and top-M highest-unigram phonemes.
2. Build a QUBO for that subproblem (same penalty structure as run_qubo_decipherment).
3. Convert QUBO → Ising  H = Σ_i h_i Z_i + Σ_{i<j} J_ij Z_i Z_j.
4. Build QAOAAnsatz(H, reps=p); optimise 2p parameters via COBYLA.
5. Sample the optimal circuit → most-frequent bitstring → sub sign→phoneme map.
6. Merge with MCMC ranking.json (if provided) for the remaining signs.
7. Score the hybrid map against the full corpus with both LMs.

Usage
-----
    python scripts/run_qaoa_decipherment.py --backend simulator
    python scripts/run_qaoa_decipherment.py --backend fake_brisbane
    python scripts/run_qaoa_decipherment.py --backend ibmq \\
        --reps 1 --top-signs 4 --top-phonemes 4
    python scripts/run_qaoa_decipherment.py \\
        --init-from outputs/decipherment/ranking.json
    python scripts/run_qaoa_decipherment.py --smoke-test
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from hackingrongo.data.rapa_nui_corpus import NGramLM  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

_TOP_SIGNS_DEFAULT    = 4
_TOP_PHONEMES_DEFAULT = 4
_REPS_DEFAULT         = 1
_SHOTS_DEFAULT        = 1024
_MAX_ITER_DEFAULT     = 200


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_corpus(corpus_dir: Path) -> list[list[str]]:
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
            lms.append(NGramLM.load(path))
        else:
            log.warning("LM not found, skipping: %s", path)
    if not lms:
        log.error("No LMs in %s — run build_language_models.py first.", lm_dir)
        sys.exit(1)
    return lms


def _phoneme_inventory(lms: list[NGramLM]) -> list[str]:
    """Canonical Rapa Nui syllable inventory (shared with MCMC and p_good).

    See hackingrongo.data.phoneme_inventory — the QAOA search space must
    match the inventory the classical solvers explore for the comparison
    to be meaningful.  The *lms* argument is kept for signature
    compatibility but is no longer consulted.
    """
    from hackingrongo.data.phoneme_inventory import RAPA_NUI_SYLLABLES
    return list(RAPA_NUI_SYLLABLES)


def _unigram_score(phoneme: str, lms: list[NGramLM]) -> float:
    total, n = 0.0, 0
    for lm in lms:
        lp = lm.score_sequence([phoneme])
        if math.isfinite(lp):
            total += lp
            n += 1
    return total / n if n > 0 else -20.0


def _score_assignment(
    phone_map: dict[str, str],
    corpus_seqs: list[list[str]],
    lms: list[NGramLM],
) -> float:
    total_lp, total_n = 0.0, 0
    for seq in corpus_seqs:
        translated = [phone_map.get(s, "<UNK>") for s in seq]
        for lm in lms:
            if len(translated) >= lm.order:
                total_lp += lm.score_sequence(translated)
                total_n += len(translated)
    return total_lp / total_n if total_n > 0 else -math.inf


# ---------------------------------------------------------------------------
# Subproblem selection
# ---------------------------------------------------------------------------

def _select_top_signs(
    corpus_seqs: list[list[str]],
    all_signs: list[str],
    top_n: int,
) -> list[str]:
    freq: dict[str, int] = {}
    for seq in corpus_seqs:
        for s in seq:
            freq[s] = freq.get(s, 0) + 1
    return sorted(all_signs, key=lambda s: -freq.get(s, 0))[:top_n]


def _select_top_phonemes(
    all_phonemes: list[str],
    lms: list[NGramLM],
    top_m: int,
) -> list[str]:
    return sorted(all_phonemes, key=lambda p: -_unigram_score(p, lms))[:top_m]


# ---------------------------------------------------------------------------
# QUBO construction (subproblem)
# ---------------------------------------------------------------------------

def _var(s_idx: int, p_idx: int, n_phonemes: int) -> int:
    return s_idx * n_phonemes + p_idx


def _build_subproblem_qubo(
    signs: list[str],
    phonemes: list[str],
    lms: list[NGramLM],
    lambda1: float = 10.0,
    lambda2: float = 5.0,
    sparse_penalties: bool = False,
) -> dict[tuple[int, int], float]:
    n_s, n_p = len(signs), len(phonemes)
    Q: dict[tuple[int, int], float] = {}

    def _add(i: int, j: int, val: float) -> None:
        if i > j:
            i, j = j, i
        Q[(i, j)] = Q.get((i, j), 0.0) + val

    # Objective: maximise LM unigram score (minimise negative score)
    for s_idx in range(n_s):
        for p_idx, phoneme in enumerate(phonemes):
            v = _var(s_idx, p_idx, n_p)
            _add(v, v, -_unigram_score(phoneme, lms))

    # One-hot: each sign maps to exactly one phoneme
    for s_idx in range(n_s):
        for p_idx in range(n_p):
            _add(_var(s_idx, p_idx, n_p), _var(s_idx, p_idx, n_p), -lambda1)
        if sparse_penalties:
            # Adjacent-only couplings: O(M×N) ZZ terms instead of O(M²×N).
            # Reduces SWAP overhead on heavy-hex topology at the cost of a
            # softer one-hot constraint (non-adjacent multi-hot states penalised
            # less strongly — acceptable when lambda1 >> objective scale).
            for p_idx in range(n_p - 1):
                _add(_var(s_idx, p_idx, n_p), _var(s_idx, p_idx + 1, n_p), 2.0 * lambda1)
        else:
            for p_idx in range(n_p):
                for q_idx in range(p_idx + 1, n_p):
                    _add(_var(s_idx, p_idx, n_p), _var(s_idx, q_idx, n_p), 2.0 * lambda1)

    # Capacity: discourage too many signs per phoneme
    for p_idx in range(n_p):
        for s_idx in range(n_s):
            for t_idx in range(s_idx + 1, n_s):
                _add(_var(s_idx, p_idx, n_p), _var(t_idx, p_idx, n_p), 2.0 * lambda2)

    log.info("Subproblem QUBO: %d vars, %d couplings.", n_s * n_p,
             sum(1 for (i, j) in Q if i != j))
    return Q


# ---------------------------------------------------------------------------
# QUBO → Ising → SparsePauliOp
# ---------------------------------------------------------------------------

def _qubo_to_ising(
    Q: dict[tuple[int, int], float],
    n: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Upper-triangular QUBO Q → Ising (h, J, offset).

    x_i = (1 - z_i) / 2,  z_i ∈ {-1, +1}
    H = Σ h_i z_i + Σ_{i<j} J_ij z_i z_j + offset
    """
    h      = np.zeros(n, dtype=np.float64)
    J      = np.zeros((n, n), dtype=np.float64)
    offset = 0.0

    for (i, j), val in Q.items():
        if i == j:
            h[i]   -= val / 2.0
            offset += val / 2.0
        else:          # i < j guaranteed by _add()
            J[i, j] += val / 4.0
            h[i]    -= val / 4.0
            h[j]    -= val / 4.0
            offset  += val / 4.0

    return h, J, offset


def _ising_to_sparse_pauli_op(
    h: np.ndarray,
    J: np.ndarray,
    n: int,
) -> Any:
    """Build SparsePauliOp from Ising (h, J).

    Qiskit Pauli string convention: rightmost character = qubit 0.
    Qubit i → string position n - 1 - i from the left.
    """
    from qiskit.quantum_info import SparsePauliOp

    pauli_list: list[str] = []
    coeffs: list[float]   = []

    for i in range(n):
        if abs(h[i]) < 1e-12:
            continue
        chars = ["I"] * n
        chars[n - 1 - i] = "Z"
        pauli_list.append("".join(chars))
        coeffs.append(float(h[i]))

    for i in range(n):
        for j in range(i + 1, n):
            if abs(J[i, j]) < 1e-12:
                continue
            chars = ["I"] * n
            chars[n - 1 - i] = "Z"
            chars[n - 1 - j] = "Z"
            pauli_list.append("".join(chars))
            coeffs.append(float(J[i, j]))

    if not pauli_list:
        return SparsePauliOp(["I" * n], [0.0])
    return SparsePauliOp(pauli_list, coeffs)


# ---------------------------------------------------------------------------
# Expectation value from bitstring counts
# ---------------------------------------------------------------------------

def _compute_expectation(
    counts: dict[str, int],
    h: np.ndarray,
    J: np.ndarray,
    n: int,
) -> float:
    """⟨H_ising⟩ from bitstring counts.  Bitstring bit b_i → z_i = 1 - 2*b_i."""
    total_shots = sum(counts.values())
    if total_shots == 0:
        return float("inf")
    total_energy = 0.0
    for bitstring, count in counts.items():
        padded = bitstring.zfill(n)
        # Qiskit: rightmost char = qubit 0 → qubit i is at index n-1-i
        z = np.array([1 - 2 * int(padded[n - 1 - i]) for i in range(n)], dtype=np.float64)
        total_energy += count * (float(h @ z) + float(z @ J @ z))
    return total_energy / total_shots


def _get_counts(result: Any, n: int) -> dict[str, int]:
    """Extract zero-padded counts from a SamplerV2 PrimitiveResult."""
    raw: dict[str, int] = result[0].data.meas.get_counts()
    return {k.zfill(n): v for k, v in raw.items()}


# ---------------------------------------------------------------------------
# Backend primitives (no Session — Open Plan job mode)
# ---------------------------------------------------------------------------

def _run_statevector(qc: Any, shots: int, n: int) -> dict[str, int]:
    """Sample bitstring counts from the pure statevector.

    Two pitfalls this avoids at n=16 (both allocate a dense 2ⁿ×2ⁿ ≈ 64 GiB
    complex matrix and OOM):
      * StatevectorSampler builds a density matrix for the measured circuit.
      * Statevector on an undecomposed QAOAAnsatz matrix-exponentiates the
        PauliEvolutionGate as a dense operator (the splu/spsolve path).
    Decomposing first turns the evolution into elementary gates, so Statevector
    evolves gate-by-gate on an O(2ⁿ) *vector* (~1 MB at n=16, scales to ~26 q).
    """
    from qiskit.quantum_info import Statevector
    qc_decomposed = qc.remove_final_measurements(inplace=False).decompose(reps=3)
    counts = Statevector(qc_decomposed).sample_counts(shots)
    return {k.zfill(n): int(v) for k, v in counts.items()}


def _make_ibmq_backend(min_qubits: int) -> Any:
    from qiskit_ibm_runtime import QiskitRuntimeService
    service = QiskitRuntimeService(
        channel="ibm_quantum_platform",
        token=os.environ.get("IBMQ_TOKEN"),
        instance=os.environ.get("IBMQ_INSTANCE"),
    )
    backend = service.least_busy(
        operational=True, simulator=False, min_num_qubits=min_qubits,
    )
    log.info("IBMQ backend: %s (%d qubits)", backend.name, backend.num_qubits)
    return backend


# ---------------------------------------------------------------------------
# QAOA optimisation
# ---------------------------------------------------------------------------

def _run_qaoa(
    ansatz: Any,
    t_ansatz: Any | None,
    h: np.ndarray,
    J: np.ndarray,
    n_qubits: int,
    shots: int,
    backend: str,
    ibmq_backend: Any | None,
    max_iter: int,
    reps: int,
    seed: int = 20260606,
) -> tuple[dict[str, int], float, dict[str, float], int, np.ndarray, list[str]]:
    """Classical-quantum optimisation loop via COBYLA.

    Returns (best_counts, best_objective, optimal_params_dict, n_evaluations,
             opt_x, ibmq_job_ids) where opt_x is the converged parameter
             vector and ibmq_job_ids is the list of every IBM job ID submitted
             (empty for non-IBMQ backends).
    Transpiled circuit (t_ansatz) is used for IBMQ / FakeBrisbane to avoid
    re-routing on every optimizer step.
    """
    from scipy.optimize import minimize
    from qiskit_ibm_runtime import SamplerV2

    n_params = ansatz.num_parameters
    rng = np.random.default_rng(seed)
    x0  = rng.uniform(0, np.pi, n_params)
    log.info("QAOA initial_point (seed=%d): %s", seed, np.round(x0, 4).tolist())

    best_energy: list[float]          = [float("inf")]
    best_counts: list[dict[str, int]] = [{}]
    n_evals: list[int]                = [0]
    ibmq_job_ids: list[str]           = []   # accumulated for provenance

    fake_backend: Any | None = None
    if backend == "fake_brisbane":
        from qiskit_ibm_runtime.fake_provider import FakeBrisbane
        fake_backend = FakeBrisbane()

    def _sample(params: np.ndarray) -> dict[str, int]:
        bound = (t_ansatz if t_ansatz is not None else ansatz).assign_parameters(params)
        if backend in ("statevector", "simulator"):
            # Run the logical ansatz on StatevectorSampler — no device transpile,
            # no Aer dependency, no 24-qubit BasicSimulator cap.
            return _run_statevector(bound, shots, n_qubits)
        elif backend == "fake_brisbane":
            result = SamplerV2(mode=fake_backend).run([bound], shots=shots).result()
            return _get_counts(result, n_qubits)
        else:  # ibmq
            sampler = SamplerV2(mode=ibmq_backend)
            sampler.options.dynamical_decoupling.enable = True
            sampler.options.default_shots = shots
            job = sampler.run([bound], shots=shots)
            ibmq_job_ids.append(job.job_id())
            log.info("  Job submitted: %s", job.job_id())
            return _get_counts(job.result(), n_qubits)

    def _objective(params: np.ndarray) -> float:
        n_evals[0] += 1
        counts = _sample(params)
        energy = _compute_expectation(counts, h, J, n_qubits)
        if energy < best_energy[0]:
            best_energy[0] = energy
            best_counts[0] = counts
        if n_evals[0] % 20 == 0:
            log.info("  COBYLA eval %d: E=%.4f", n_evals[0], energy)
        return energy

    opt_result = minimize(_objective, x0, method="COBYLA",
                          options={"maxiter": max_iter, "rhobeg": 0.5})
    opt_x      = opt_result.x

    opt_params = {f"param_{i}": round(float(opt_x[i]), 6) for i in range(n_params)}

    return best_counts[0], best_energy[0], opt_params, n_evals[0], opt_x, ibmq_job_ids


# ---------------------------------------------------------------------------
# Assignment extraction from bitstring
# ---------------------------------------------------------------------------

def _extract_assignment(
    counts: dict[str, int],
    signs: list[str],
    phonemes: list[str],
) -> dict[str, str]:
    """Most-frequent bitstring → sign→phoneme map.

    Variable layout: var(s_idx, p_idx) = s_idx * n_phonemes + p_idx.
    Qiskit bitstring: rightmost char = qubit 0, so qubit v = bitstring[n-1-v].
    """
    if not counts:
        return {s: phonemes[0] for s in signs}
    n_p = len(phonemes)
    n   = len(signs) * n_p
    best_bs = max(counts, key=counts.__getitem__).zfill(n)
    phone_map: dict[str, str] = {}
    for s_idx, sign in enumerate(signs):
        chosen: str | None = None
        for p_idx in range(n_p):
            v = _var(s_idx, p_idx, n_p)
            if int(best_bs[n - 1 - v]) == 1:
                chosen = phonemes[p_idx]
                break
        phone_map[sign] = chosen if chosen is not None else phonemes[0]
    return phone_map


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def _print_diagnostics(
    *,
    backend: str,
    reps: int,
    shots: int,
    sub_signs: list[str],
    sub_phonemes: list[str],
    n_qubits: int,
    n_z_terms: int,
    n_zz_terms: int,
    offset: float,
    ansatz: Any,
    t_ansatz: Any | None,
    all_signs: list[str],
    max_iter: int,
) -> None:
    bar = "═" * 66

    # ── Circuit resource estimate ─────────────────────────────────────────────
    # For fake_brisbane we need to count two-qubit gates from the transpiled circuit.
    cx_count: int | None = None
    if t_ansatz is not None:
        ops = t_ansatz.count_ops()
        cx_count = ops.get("cx", ops.get("ecr", ops.get("cz", 0)))

    print(f"\n{bar}")
    print(f"  QAOA Diagnostics — Rongorongo Layer 4Q-QAOA")
    print(f"  Backend : {backend}   Reps (p) : {reps}   Shots : {shots}")
    print(bar)

    print(f"\n  ── Subproblem ──────────────────────────────────────────────")
    print(f"  Top signs       : {len(sub_signs)}   Total signs in corpus : {len(all_signs)}")
    print(f"  Top phonemes    : {len(sub_phonemes)}")
    print(f"  Decision vars   : {len(sub_signs)} × {len(sub_phonemes)} = {n_qubits} binary variables")
    print(f"  Signs selected  : {', '.join(sub_signs)}")
    print(f"  Phonemes        : {', '.join(sub_phonemes)}")

    print(f"\n  ── Ising Hamiltonian ───────────────────────────────────────")
    print(f"  Z terms (local fields h_i)    : {n_z_terms}")
    print(f"  ZZ terms (couplings J_ij)     : {n_zz_terms}")
    print(f"  Constant offset               : {offset:.6f}")
    pauli_count = n_z_terms + n_zz_terms
    print(f"  Total Pauli terms in H        : {pauli_count}")

    print(f"\n  ── Ansatz (pre-transpile) ──────────────────────────────────")
    print(f"  Qubits                        : {ansatz.num_qubits}")
    print(f"  Depth                         : {ansatz.depth()}")
    print(f"  Optimiser parameters (2p)     : {ansatz.num_parameters}")
    print(f"  Num gates                     : {sum(ansatz.count_ops().values())}")
    ops_pre = ansatz.count_ops()
    for gate, cnt in sorted(ops_pre.items()):
        print(f"    {gate:<22} : {cnt}")

    if t_ansatz is not None:
        print(f"\n  ── Ansatz (post-transpile, {backend}) ───────────────────")
        print(f"  Qubits                        : {t_ansatz.num_qubits}")
        print(f"  Depth                         : {t_ansatz.depth()}")
        print(f"  Optimiser parameters (2p)     : {t_ansatz.num_parameters}")
        print(f"  Num gates                     : {sum(t_ansatz.count_ops().values())}")
        ops_post = t_ansatz.count_ops()
        for gate, cnt in sorted(ops_post.items()):
            print(f"    {gate:<22} : {cnt}")
        if cx_count is not None:
            print(f"\n  Two-qubit gate count          : {cx_count}")

    print(f"\n  ── Cost estimate (if you proceed) ──────────────────────────")
    print(f"  Max COBYLA iterations         : {max_iter}")
    print(f"  Shots per eval                : {shots}")
    print(f"  Max circuit executions        : {max_iter * shots:,}")
    if backend == "fake_brisbane":
        print(f"  Estimated wall-clock          : ~{max_iter * shots / 2_000:.0f}–{max_iter * shots / 500:.0f} s  (fake_brisbane is ~500–2k shots/s)")
    elif backend == "ibmq":
        print(f"  Estimated IBM Quantum jobs    : {max_iter} (one per COBYLA step)")

    print(f"\n{bar}\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="QAOA hybrid decipherment for rongorongo sign→phoneme assignment.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--corpus-dir",   type=Path, default=None)
    p.add_argument("--lm-dir",       type=Path, default=None)
    p.add_argument("--init-from",    type=Path, default=None, metavar="JSON",
                   help="ranking.json from run_decipherment.py (MCMC warm-start / hybrid merge).")
    p.add_argument("--backend", choices=["simulator", "statevector", "fake_brisbane", "ibmq"],
                   default="simulator")
    p.add_argument("--reps",         type=int, default=_REPS_DEFAULT,
                   help=f"QAOA depth p (default: {_REPS_DEFAULT}).")
    p.add_argument("--shots",        type=int, default=_SHOTS_DEFAULT,
                   help=f"Shots per circuit evaluation (default: {_SHOTS_DEFAULT}).")
    p.add_argument("--top-signs",    type=int, default=_TOP_SIGNS_DEFAULT,
                   help=f"Highest-frequency signs for subproblem (default: {_TOP_SIGNS_DEFAULT}).")
    p.add_argument("--top-phonemes", type=int, default=_TOP_PHONEMES_DEFAULT,
                   help=f"Top-LM phonemes for subproblem (default: {_TOP_PHONEMES_DEFAULT}).")
    p.add_argument("--max-iter",     type=int, default=_MAX_ITER_DEFAULT,
                   help=f"COBYLA max iterations (default: {_MAX_ITER_DEFAULT}).")
    p.add_argument("--output",       type=Path, default=None, metavar="JSON")
    p.add_argument("--smoke-test",   action="store_true",
                   help="3×3=9 qubits, 1 rep, 100 shots, 10 COBYLA iter.")
    p.add_argument("--post-process-oneshot", action="store_true",
                   help="After COBYLA converges, run one dedicated forward pass at the "
                        "optimal parameters and extract the assignment from the most-frequent "
                        "bitstring in that sample.  Less sensitive to shot noise than taking "
                        "the lowest-energy counts from mid-optimisation.")
    p.add_argument("--score-subset", action="store_true",
                   help="Score the hybrid LM only on corpus sequences that contain at least "
                        "one scoped sign.  Makes the hybrid score directly comparable to the "
                        "QAOA subproblem score instead of being diluted by unscoped sequences.")
    p.add_argument("--sparse-penalties", action="store_true",
                   help="One-hot ZZ couplings between adjacent phonemes only (O(M×N) terms "
                        "instead of O(M²×N)).  Reduces SWAP overhead on heavy-hex topology.")
    p.add_argument("--seed", type=int, default=20260606, metavar="INT",
                   help="Global RNG seed for reproducibility and COBYLA initial point (default: 20260606).")
    p.add_argument("--diagnostics-only", action="store_true",
                   help="Build Ising Hamiltonian and ansatz, print diagnostics, then exit "
                        "without running the optimiser.  Useful for circuit-resource checks "
                        "before submitting hardware jobs.")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = _parse_args()
    from hackingrongo.repro import set_global_seed
    set_global_seed(args.seed)

    if args.smoke_test:
        args.top_signs    = 3
        args.top_phonemes = 3
        args.reps         = 1
        args.shots        = 100
        args.max_iter     = 10
        log.info("Smoke-test: 3×3=9 qubits, 1 rep, 100 shots, 10 COBYLA iter.")

    corpus_dir = args.corpus_dir
    lm_dir     = args.lm_dir
    output     = args.output

    if corpus_dir is None or lm_dir is None or output is None:
        try:
            from omegaconf import OmegaConf
            cfg = OmegaConf.load(PROJECT_ROOT / "conf" / "config.yaml")
            corpus_dir = corpus_dir or (PROJECT_ROOT / cfg.paths.corpus_dir)
            lm_dir     = lm_dir     or (PROJECT_ROOT / "data" / "language_models")
            output     = output     or (
                PROJECT_ROOT / cfg.paths.outputs_dir / "decipherment" / "qaoa_result.json"
            )
        except Exception:
            pass

    if not corpus_dir or not corpus_dir.exists():
        log.error("Corpus directory not found. Pass --corpus-dir.")
        sys.exit(1)
    if not lm_dir or not lm_dir.exists():
        log.error("LM directory not found. Pass --lm-dir.")
        sys.exit(1)

    # ── Load ──────────────────────────────────────────────────────────────────
    corpus_seqs  = _load_corpus(corpus_dir)
    if not corpus_seqs:
        log.error("No corpus sequences found.")
        sys.exit(1)
    lms          = _load_lms(lm_dir)
    all_signs    = sorted({code for seq in corpus_seqs for code in seq})
    all_phonemes = _phoneme_inventory(lms)
    if not all_phonemes:
        log.error("Phoneme inventory empty — LM vocab absent or malformed.")
        sys.exit(1)
    log.info("Corpus: %d tablets, %d signs, %d phonemes.",
             len(corpus_seqs), len(all_signs), len(all_phonemes))

    # ── MCMC warm-start ───────────────────────────────────────────────────────
    mcmc_phone_map: dict[str, str]  = {}
    mcmc_best_lm_score: float | None = None

    if args.init_from and args.init_from.exists():
        try:
            ranking = json.loads(args.init_from.read_text(encoding="utf-8"))
            hyps    = ranking.get("hypotheses", [])
            if hyps:
                mcmc_best_lm_score = hyps[0].get("overall_lm_score")
                mcmc_phone_map     = {a["sign_code"]: a["phoneme"]
                                      for a in hyps[0].get("assignments", [])}
                log.info("MCMC warm-start: %d assignments, best_lm=%.4f.",
                         len(mcmc_phone_map), mcmc_best_lm_score or 0.0)
        except Exception as exc:
            log.warning("Could not load MCMC warm-start: %s", exc)

    # ── Subproblem selection ──────────────────────────────────────────────────
    sub_signs    = _select_top_signs(corpus_seqs, all_signs, args.top_signs)
    sub_signs_set = frozenset(sub_signs)
    sub_phonemes = _select_top_phonemes(all_phonemes, lms, args.top_phonemes)
    n_qubits     = len(sub_signs) * len(sub_phonemes)

    log.info("QAOA subproblem: %d signs × %d phonemes = %d qubits, reps=%d.",
             len(sub_signs), len(sub_phonemes), n_qubits, args.reps)

    # ── QUBO → Ising → PauliOp ───────────────────────────────────────────────
    Q              = _build_subproblem_qubo(sub_signs, sub_phonemes, lms,
                                            sparse_penalties=args.sparse_penalties)
    h, J, offset   = _qubo_to_ising(Q, n_qubits)
    cost_op        = _ising_to_sparse_pauli_op(h, J, n_qubits)
    n_z_terms      = int(np.count_nonzero(np.abs(h) > 1e-12))
    n_zz_terms     = int(np.count_nonzero(np.triu(np.abs(J), 1) > 1e-12))
    log.info("Ising: %d Z terms, %d ZZ terms, offset=%.4f.", n_z_terms, n_zz_terms, offset)

    # ── QAOA ansatz ───────────────────────────────────────────────────────────
    from qiskit.circuit.library import QAOAAnsatz
    ansatz = QAOAAnsatz(cost_operator=cost_op, reps=args.reps)
    ansatz.measure_all()
    log.info("Ansatz: %d qubits, depth %d, %d parameters.",
             ansatz.num_qubits, ansatz.depth(), ansatz.num_parameters)

    # Pre-transpile once for hardware backends (parameters are preserved)
    ibmq_backend: Any | None = None
    t_ansatz: Any | None     = None

    if args.backend == "ibmq":
        from qiskit.compiler import transpile
        ibmq_backend = _make_ibmq_backend(min_qubits=n_qubits)
        t_ansatz     = transpile(ansatz, backend=ibmq_backend, optimization_level=3)
        log.info("Transpiled (IBMQ): %d qubits, depth %d.",
                 t_ansatz.num_qubits, t_ansatz.depth())
    elif args.backend == "fake_brisbane":
        from qiskit.compiler import transpile
        from qiskit_ibm_runtime.fake_provider import FakeBrisbane
        t_ansatz = transpile(ansatz, backend=FakeBrisbane(), optimization_level=3)
        log.info("Transpiled (FakeBrisbane): %d qubits, depth %d.",
                 t_ansatz.num_qubits, t_ansatz.depth())

    # ── Diagnostics-only: print and exit ─────────────────────────────────────
    if args.diagnostics_only:
        _print_diagnostics(
            backend=args.backend,
            reps=args.reps,
            shots=args.shots,
            sub_signs=sub_signs,
            sub_phonemes=sub_phonemes,
            n_qubits=n_qubits,
            n_z_terms=n_z_terms,
            n_zz_terms=n_zz_terms,
            offset=offset,
            ansatz=ansatz,
            t_ansatz=t_ansatz,
            all_signs=all_signs,
            max_iter=args.max_iter,
        )
        return

    # ── QAOA optimisation ─────────────────────────────────────────────────────
    log.info("Running QAOA optimisation (backend=%s, max_iter=%d) …",
             args.backend, args.max_iter)
    t0 = time.perf_counter()
    best_counts, best_objective, optimal_params, n_evals, opt_x, _cobyla_job_ids = _run_qaoa(
        ansatz=ansatz,
        t_ansatz=t_ansatz,
        h=h, J=J,
        n_qubits=n_qubits,
        shots=args.shots,
        backend=args.backend,
        ibmq_backend=ibmq_backend,
        max_iter=args.max_iter,
        reps=args.reps,
        seed=args.seed,
    )
    qaoa_elapsed = time.perf_counter() - t0
    log.info("QAOA done: %.1f s, %d evals, best_objective=%.4f.",
             qaoa_elapsed, n_evals, best_objective)

    # ── Optional one-shot re-sample at converged parameters ───────────────────
    extraction_counts = best_counts
    if args.post_process_oneshot:
        log.info("--post-process-oneshot: sampling at converged params …")
        from qiskit_ibm_runtime import SamplerV2
        bound = (t_ansatz if t_ansatz is not None else ansatz).assign_parameters(opt_x)
        if args.backend == "statevector":
            extraction_counts = _run_statevector(bound, args.shots, n_qubits)
        elif args.backend == "fake_brisbane":
            from qiskit_ibm_runtime.fake_provider import FakeBrisbane
            extraction_counts = _get_counts(
                SamplerV2(mode=FakeBrisbane()).run([bound], shots=args.shots).result(),
                n_qubits,
            )
        else:  # ibmq
            sampler = SamplerV2(mode=ibmq_backend)
            sampler.options.dynamical_decoupling.enable = True
            sampler.options.default_shots = args.shots
            job = sampler.run([bound], shots=args.shots)
            log.info("  One-shot job submitted: %s", job.job_id())
            _cobyla_job_ids.append(job.job_id())
            extraction_counts = _get_counts(job.result(), n_qubits)
        log.info("One-shot top bitstring count: %d/%d.",
                 max(extraction_counts.values(), default=0), args.shots)

    # ── Extract subproblem assignment ─────────────────────────────────────────
    qaoa_map     = _extract_assignment(extraction_counts, sub_signs, sub_phonemes)
    sub_lm_score = _score_assignment(qaoa_map, corpus_seqs, lms)
    log.info("QAOA subproblem LM score: %.4f", sub_lm_score)

    # ── Hybrid: merge QAOA (top-N signs) + MCMC (remaining) ──────────────────
    # Signs outside both QAOA and MCMC coverage are omitted — no hardcoded
    # default so unassigned signs don't pollute the LM score or output.
    hybrid_map: dict[str, str] = {}
    for sign in all_signs:
        assigned = qaoa_map.get(sign) or mcmc_phone_map.get(sign)
        if assigned:
            hybrid_map[sign] = assigned

    score_seqs = (
        [seq for seq in corpus_seqs if any(s in sub_signs_set for s in seq)]
        if args.score_subset
        else corpus_seqs
    )
    if args.score_subset:
        log.info("--score-subset: %d/%d sequences contain a scoped sign.",
                 len(score_seqs), len(corpus_seqs))
    hybrid_lm_score       = _score_assignment(hybrid_map, score_seqs, lms)
    # Apples-to-apples baseline: score the MCMC map with the SAME per-token-mean
    # scorer.  mcmc_best_lm_score (ranking.json overall_lm_score) is a
    # corpus-level TOTAL on a different scale, so hybrid − mcmc_best produced a
    # meaningless Δ (e.g. +3119).  Only mcmc_baseline_lm_score is comparable.
    mcmc_baseline_lm_score: float | None = None
    improvement_over_mcmc: float | None = None
    if mcmc_phone_map:
        mcmc_baseline_lm_score = _score_assignment(mcmc_phone_map, score_seqs, lms)
        if math.isfinite(hybrid_lm_score) and math.isfinite(mcmc_baseline_lm_score):
            improvement_over_mcmc = round(hybrid_lm_score - mcmc_baseline_lm_score, 6)

    log.info("Hybrid LM score: %.4f%s", hybrid_lm_score,
             f"  (Δ vs MCMC baseline: {improvement_over_mcmc:+.4f})"
             if improvement_over_mcmc is not None else "")

    # ── Print summary ─────────────────────────────────────────────────────────
    print(f"\n{'═' * 64}")
    print(f"  QAOA Decipherment — Rongorongo Layer 4Q-QAOA")
    print(f"  Backend : {args.backend}   Reps (p) : {args.reps}")
    print(f"  Subproblem: {len(sub_signs)} signs × {len(sub_phonemes)} phonemes = {n_qubits} qubits")
    print(f"  Shots: {args.shots}   COBYLA evals: {n_evals}   Time: {qaoa_elapsed:.1f}s")
    print(f"{'═' * 64}")
    print(f"\n  QAOA objective (Ising ⟨H⟩)        : {best_objective:.4f}")
    print(f"  QAOA subproblem LM score           : {sub_lm_score:.4f}")
    print(f"  Hybrid LM score (full corpus)      : {hybrid_lm_score:.4f}")
    if mcmc_baseline_lm_score is not None:
        sym = "▲" if (improvement_over_mcmc or 0) > 0 else "▼"
        print(f"  MCMC baseline (same scorer)        : {mcmc_baseline_lm_score:.4f}")
        print(f"  Δ hybrid vs MCMC baseline          : {(improvement_over_mcmc or 0):+.4f} {sym}")
    if mcmc_best_lm_score is not None:
        print(f"  MCMC ranking total (diff. scale)   : {mcmc_best_lm_score:.4f}  (not comparable to the per-token scores above)")
    print()
    print("  QAOA subproblem assignments:")
    for sign, phoneme in sorted(qaoa_map.items()):
        print(f"    {sign:>8} → {phoneme}")
    print()

    # ── Output dict ───────────────────────────────────────────────────────────
    result_dict: dict[str, Any] = {
        "backend":                  args.backend,
        "reps":                     args.reps,
        "shots":                    args.shots,
        "top_signs":                args.top_signs,
        "top_phonemes":             args.top_phonemes,
        "n_qubits":                 n_qubits,
        "qaoa_objective":           round(best_objective, 6) if math.isfinite(best_objective) else None,
        "qaoa_subproblem_lm_score": round(sub_lm_score, 6) if math.isfinite(sub_lm_score) else None,
        "hybrid_lm_score":          round(hybrid_lm_score, 6) if math.isfinite(hybrid_lm_score) else None,
        # improvement_over_mcmc compares hybrid vs mcmc_baseline_lm_score — BOTH
        # per-token-mean scores from _score_assignment, so the Δ is meaningful.
        "improvement_over_mcmc":    improvement_over_mcmc,
        "mcmc_baseline_lm_score":   (round(mcmc_baseline_lm_score, 6)
                                     if mcmc_baseline_lm_score is not None else None),
        # mcmc_best_lm_score is the ranking.json overall_lm_score: a corpus-level
        # TOTAL on a different scale — NOT comparable to the per-token scores; kept
        # for provenance only, not used in improvement_over_mcmc.
        "mcmc_ranking_total_lm_score": (round(mcmc_best_lm_score, 6)
                                        if mcmc_best_lm_score is not None else None),
        "qaoa_elapsed_seconds":     round(qaoa_elapsed, 2),
        "n_cobyla_evaluations":     n_evals,
        "optimal_params":           optimal_params,
        "sub_signs":                sub_signs,
        "sub_phonemes":             sub_phonemes,
        "ising_offset":             round(float(offset), 6),
        "circuit_stats": {
            "n_qubits":                  ansatz.num_qubits,
            "depth_pre_transpile":       ansatz.depth(),
            "depth_post_transpile":      t_ansatz.depth() if t_ansatz is not None else None,
            "n_parameters":              ansatz.num_parameters,
            "n_ising_z_terms":           n_z_terms,
            "n_ising_zz_terms":          n_zz_terms,
        },
        "phoneme_assignments": [
            {
                "sign_code": s,
                "phoneme":   hybrid_map[s],
                "source":    "qaoa" if s in qaoa_map else "mcmc",
            }
            for s in sorted(all_signs)
            if s in hybrid_map
        ],
    }

    # ── Hardware provenance ───────────────────────────────────────────────────
    if args.backend == "ibmq" and _cobyla_job_ids:
        from hackingrongo.quantum_provenance import collect_multi_job_provenance
        result_dict["hardware_provenance"] = collect_multi_job_provenance(
            _cobyla_job_ids, ibmq_backend
        )

    # ── Save ──────────────────────────────────────────────────────────────────
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(result_dict, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        log.info("Result written to %s", output)
    if result_dict.get("hardware_provenance"):
        from hackingrongo.quantum_provenance import write_versioned_result
        run_key = f"{args.backend}_p{args.reps}_s{args.top_signs}x{args.top_phonemes}"
        write_versioned_result(result_dict, "qaoa", run_key)

    # ── MLflow tracking ───────────────────────────────────────────────────────
    try:
        import mlflow as _mlflow
        from datetime import datetime as _dt, timezone as _tz
        _mlflow.set_tracking_uri(os.environ.get(
            "MLFLOW_TRACKING_URI",
            f"file://{(PROJECT_ROOT / 'outputs' / 'mlruns').resolve()}",
        ))
        _mlflow.set_experiment("rongorongo_qaoa")
        with _mlflow.start_run(run_name=f"qaoa-{args.backend}-{_dt.now(tz=_tz.utc).strftime('%Y%m%d-%H%M')}"):
            _mlflow.log_params({
                "backend":      args.backend,
                "reps":         args.reps,
                "shots":        args.shots,
                "top_signs":    args.top_signs,
                "top_phonemes": args.top_phonemes,
                "n_qubits":     n_qubits,
            })
            _metrics: dict[str, float] = {"qaoa_elapsed_seconds": qaoa_elapsed}
            if math.isfinite(best_objective):
                _metrics["qaoa_objective"]            = best_objective
            if math.isfinite(sub_lm_score):
                _metrics["qaoa_subproblem_lm_score"]  = sub_lm_score
            if math.isfinite(hybrid_lm_score):
                _metrics["hybrid_lm_score"]           = hybrid_lm_score
            if improvement_over_mcmc is not None:
                _metrics["improvement_over_mcmc"]     = improvement_over_mcmc
            _mlflow.log_metrics(_metrics)
            if output and output.exists():
                _mlflow.log_artifact(str(output), artifact_path="qaoa")
    except ImportError:
        pass


if __name__ == "__main__":
    main()
