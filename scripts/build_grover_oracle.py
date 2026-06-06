#!/usr/bin/env python3
"""
build_grover_oracle.py — Grover search oracle for rongorongo decipherment.

Constructs the phase-kickback oracle  O_f |π⟩ = (-1)^{f(π)} |π⟩  where
f(π)=1 iff the one-hot sign→phoneme assignment π achieves a unigram LM
score above threshold τ.

Register layout
---------------
  q_data[k*M .. k*M+M-1]   one-hot phoneme indicator for sign k  (K*M)
  q_phase                   phase-kickback ancilla, init |->          (1)
  q_valid[0..K-1]           per-sign "exactly-one-phoneme" flags      (K)
  q_adder_carry + ctrl      WeightedAdder ancilla                     (*)
  q_sum[0..S-1]             S-bit weighted-sum output                 (S)
  q_cmp[0..C-1]             IntegerComparator ancilla; q_cmp[-1]=geq  (C)

  Total ≈ K*M + 1 + K + (S-1) + WeightedAdder.ancillas
        + IntegerComparator.num_qubits

Usage
-----
    python scripts/build_grover_oracle.py
    python scripts/build_grover_oracle.py --k 4 --m 4
    python scripts/build_grover_oracle.py --draw --simulate
    python scripts/build_grover_oracle.py --oracle-output outputs/quantum/oracle_stats.json
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
from pathlib import Path
from typing import NamedTuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

# Quantization scale: unigram log-probs scaled to integers in [0, WEIGHT_SCALE]
WEIGHT_SCALE = 128
# Surface code distance for physical qubit estimate
SURFACE_CODE_DISTANCE = 3


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


def _load_lms(lm_dir: Path):
    from hackingrongo.data.rapa_nui_corpus import NGramLM
    lms = []
    for name in ["pre_contact_lm", "post_contact_lm"]:
        p = lm_dir / f"{name}.json"
        if p.exists():
            lms.append(NGramLM.load(p))
    return lms


# ---------------------------------------------------------------------------
# Sign and phoneme selection
# ---------------------------------------------------------------------------

def _sign_frequency_entropy(corpus_seqs: list[list[str]]) -> dict[str, float]:
    """Per-sign (frequency × bigram entropy) — proxy for IC contribution."""
    from collections import Counter, defaultdict
    freq: Counter[str] = Counter()
    bigram_counts: defaultdict[str, Counter] = defaultdict(Counter)
    for seq in corpus_seqs:
        for tok in seq:
            freq[tok] += 1
        for a, b in zip(seq, seq[1:]):
            bigram_counts[a][b] += 1

    scores: dict[str, float] = {}
    for sign, cnt in freq.items():
        followers = bigram_counts[sign]
        total = sum(followers.values())
        if total == 0:
            entropy = 0.0
        else:
            probs = [c / total for c in followers.values()]
            entropy = -sum(p * math.log2(p) for p in probs if p > 0)
        scores[sign] = cnt * entropy
    return scores


def select_top_k_signs(corpus_seqs: list[list[str]], k: int) -> list[str]:
    """Top-K signs by (frequency × bigram entropy), descending."""
    ic = _sign_frequency_entropy(corpus_seqs)
    return [s for s, _ in sorted(ic.items(), key=lambda x: -x[1])][:k]


def _phoneme_unigram_lp(lms) -> dict[str, float]:
    """Average unigram log-prob (log₂) for each phoneme across all LMs."""
    vocab: set[str] = set()
    for lm in lms:
        for tok in lm._vocab:
            if tok and not tok.startswith("<") and 1 <= len(tok) <= 6:
                vocab.add(tok)

    scores: dict[str, float] = {}
    for ph in vocab:
        lps = [lm._unigram_log_prob.get(ph, -20.0) for lm in lms]
        scores[ph] = float(np.mean(lps))
    return scores


def select_top_m_phonemes(lms, m: int) -> list[str]:
    """Top-M phonemes by mean unigram log-prob (highest = most likely), descending."""
    lp = _phoneme_unigram_lp(lms)
    return [ph for ph, _ in sorted(lp.items(), key=lambda x: -x[1])][:m]


# ---------------------------------------------------------------------------
# Weight quantization
# ---------------------------------------------------------------------------

class OracleWeights(NamedTuple):
    phonemes: list[str]
    raw_lp: list[float]         # unigram log₂-prob
    w_int: list[int]            # quantized to [0, WEIGHT_SCALE]
    threshold_int: int          # integer threshold for ≥ check
    threshold_tau: float        # normalised τ ∈ (0,1)
    k_signs: int
    m_phonemes: int


def compute_oracle_weights(
    phonemes: list[str],
    lms,
    k: int,
    tau: float = 0.90,
) -> OracleWeights:
    """Quantize LM unigram scores to non-negative integers for the oracle."""
    lp_map = _phoneme_unigram_lp(lms)
    raw = [lp_map.get(ph, -20.0) for ph in phonemes]
    m = len(phonemes)

    lp_min = min(raw)
    lp_max = max(raw)
    span = max(lp_max - lp_min, 1e-9)

    w_int = [round((v - lp_min) / span * WEIGHT_SCALE) for v in raw]

    # Threshold: τ fraction of maximum possible total score (K signs × WEIGHT_SCALE)
    threshold_int = math.ceil(tau * k * WEIGHT_SCALE)

    return OracleWeights(
        phonemes=phonemes,
        raw_lp=raw,
        w_int=w_int,
        threshold_int=threshold_int,
        threshold_tau=tau,
        k_signs=k,
        m_phonemes=m,
    )


# ---------------------------------------------------------------------------
# Oracle circuit construction
# ---------------------------------------------------------------------------

def _one_hot_check_group(
    qc,
    group_qubits: list,
    valid_ancilla: int,
) -> None:
    """Flip valid_ancilla for each "exactly phoneme-m" pattern in group_qubits.

    For the one-hot state where only qubit group_qubits[m] is 1, this fires
    the MCX for m and flips valid_ancilla from 0→1.  The patterns are mutually
    exclusive, so valid_ancilla = 1 iff exactly one qubit in the group is set.
    """
    m_count = len(group_qubits)
    for m in range(m_count):
        # Invert all qubits except m so the pattern "only m is 1" becomes "all 1s"
        for j in range(m_count):
            if j != m:
                qc.x(group_qubits[j])
        # MCX: fires iff all group qubits are now 1 (i.e. original state had only qubit m set)
        qc.mcx(group_qubits, valid_ancilla)
        # Uninvert
        for j in range(m_count):
            if j != m:
                qc.x(group_qubits[j])


def build_grover_oracle(
    k: int,
    m: int,
    weights: OracleWeights,
) -> tuple:
    """Build the full phase-kickback Grover oracle circuit.

    Returns
    -------
    (circuit, qubit_map)
        circuit     : QuantumCircuit encoding O_f
        qubit_map   : dict with named qubit index ranges
    """
    from qiskit import QuantumCircuit, QuantumRegister
    from qiskit.circuit.library import WeightedAdder, IntegerComparator

    # ── Registers ────────────────────────────────────────────────────────────
    reg_data    = QuantumRegister(k * m,  "data")   # one-hot assignment bits
    reg_phase   = QuantumRegister(1,      "phase")  # phase-kickback ancilla |->
    reg_valid   = QuantumRegister(k,      "valid")  # per-sign one-hot check

    # WeightedAdder: weights repeated for each sign (unigram is sign-independent)
    flat_weights = weights.w_int * k  # length K*M
    wa = WeightedAdder(num_state_qubits=k * m, weights=flat_weights)
    reg_wa_sum   = QuantumRegister(wa.num_sum_qubits,   "sum")
    reg_wa_carry = QuantumRegister(wa.num_carry_qubits, "carry")
    reg_wa_ctrl  = QuantumRegister(1,                   "wa_ctrl")

    # IntegerComparator on sum register
    ic = IntegerComparator(
        num_state_qubits=wa.num_sum_qubits,
        value=weights.threshold_int,
        geq=True,
    )
    # ic uses num_state_qubits (sum) + ancilla qubits; last ancilla = geq flag
    n_cmp_ancilla = ic.num_qubits - ic.num_state_qubits
    reg_cmp = QuantumRegister(n_cmp_ancilla, "cmp")  # last qubit = geq output

    qc = QuantumCircuit(
        reg_data, reg_phase, reg_valid,
        reg_wa_sum, reg_wa_carry, reg_wa_ctrl,
        reg_cmp,
        name="grover_oracle",
    )

    # ── Phase ancilla init: |0⟩ → |−⟩ ────────────────────────────────────────
    qc.x(reg_phase[0])
    qc.h(reg_phase[0])

    # ── One-hot check (forward) ────────────────────────────────────────────
    for ki in range(k):
        group = [reg_data[ki * m + mi] for mi in range(m)]
        _one_hot_check_group(qc, group, reg_valid[ki])

    # ── WeightedAdder: enable (wa_ctrl must be |1⟩ to activate) ──────────────
    qc.x(reg_wa_ctrl[0])
    wa_qubits = (
        reg_data[:] + reg_wa_sum[:] + reg_wa_carry[:] + reg_wa_ctrl[:]
    )
    qc.append(wa, wa_qubits)

    # ── IntegerComparator: sum ≥ threshold → reg_cmp[-1] ─────────────────────
    ic_qubits = reg_wa_sum[:] + reg_cmp[:]
    qc.append(ic, ic_qubits)

    # ── Phase kickback: flip phase ancilla iff all valid AND threshold met ────
    # IntegerComparator qubit layout: [state..., compare(geq), ancilla...]
    # The geq output is the first qubit of reg_cmp (maps to ic's 'compare' register).
    geq_qubit = reg_cmp[0]
    control_qubits = list(reg_valid) + [geq_qubit]
    qc.mcx(control_qubits, reg_phase[0])

    # ── Uncompute IntegerComparator ───────────────────────────────────────────
    qc.append(ic.inverse(), ic_qubits)

    # ── Uncompute WeightedAdder ───────────────────────────────────────────────
    qc.append(wa.inverse(), wa_qubits)
    qc.x(reg_wa_ctrl[0])

    # ── Uncompute one-hot check (self-inverse) ────────────────────────────────
    for ki in range(k):
        group = [reg_data[ki * m + mi] for mi in range(m)]
        _one_hot_check_group(qc, group, reg_valid[ki])

    # reg_cmp[0] = IntegerComparator 'compare' (geq) output
    # reg_cmp[1:] = comparator ancilla bits
    cmp_start = qc.num_qubits - n_cmp_ancilla
    qubit_map = {
        "data":     (0, k * m),
        "phase":    (k * m, k * m + 1),
        "valid":    (k * m + 1, k * m + 1 + k),
        "sum":      (k * m + 1 + k, k * m + 1 + k + wa.num_sum_qubits),
        "carry":    None,
        "cmp":      (cmp_start, qc.num_qubits),
        "cmp_geq":  cmp_start,  # reg_cmp[0] = geq bit
    }
    return qc, qubit_map


# ---------------------------------------------------------------------------
# Circuit statistics
# ---------------------------------------------------------------------------

class CircuitStats(NamedTuple):
    num_qubits:             int
    depth:                  int
    num_nonlocal_gates:     int
    t_gate_count_approx:    int   # num_nonlocal_gates * 7
    surface_code_d3_phys:   int   # (2*d^2 - 1) * num_qubits  at d=3


def compute_stats(qc) -> CircuitStats:
    d = SURFACE_CODE_DISTANCE
    phys_per_logical = 2 * d * d - 1
    n_nonlocal = qc.num_nonlocal_gates()
    return CircuitStats(
        num_qubits=qc.num_qubits,
        depth=qc.depth(),
        num_nonlocal_gates=n_nonlocal,
        t_gate_count_approx=n_nonlocal * 7,
        surface_code_d3_phys=phys_per_logical * qc.num_qubits,
    )


def print_stats(stats: CircuitStats, weights: OracleWeights) -> None:
    print(f"\n{'═' * 60}")
    print(f"  Grover Oracle — Circuit Statistics")
    print(f"  K={weights.k_signs} signs × M={weights.m_phonemes} phonemes "
          f"= {weights.k_signs * weights.m_phonemes} variables")
    print(f"  τ={weights.threshold_tau:.2f}  threshold_int={weights.threshold_int}")
    print(f"{'═' * 60}")
    print(f"  logical qubits        : {stats.num_qubits:>8,}")
    print(f"  circuit depth         : {stats.depth:>8,}")
    print(f"  non-local gates       : {stats.num_nonlocal_gates:>8,}")
    print(f"  T-gate count (≈×7)    : {stats.t_gate_count_approx:>8,}")
    print(f"  surface code d=3 phys : {stats.surface_code_d3_phys:>8,}")
    print(f"{'─' * 60}")
    print(f"  Top phonemes (unigram weight):")
    for ph, w in zip(weights.phonemes, weights.w_int):
        print(f"    {ph:<12s}  w_int={w:>4d}")
    print()


# ---------------------------------------------------------------------------
# Grover diffuser
# ---------------------------------------------------------------------------

def build_diffuser(n_data: int) -> object:
    """Standard Grover diffuser on n_data qubits: 2|s><s| - I."""
    from qiskit import QuantumCircuit
    qc = QuantumCircuit(n_data, name="diffuser")
    qc.h(range(n_data))
    qc.x(range(n_data))
    qc.h(n_data - 1)
    qc.mcx(list(range(n_data - 1)), n_data - 1)
    qc.h(n_data - 1)
    qc.x(range(n_data))
    qc.h(range(n_data))
    return qc


# ---------------------------------------------------------------------------
# Simulation (reduced scale)
# ---------------------------------------------------------------------------

def run_grover_simulation(
    k_sim: int,
    m_sim: int,
    lms,
    corpus_seqs: list[list[str]],
    tau: float = 0.50,
    sim_weight_scale: int = 4,
) -> dict:
    """Run one Grover iteration on a k_sim × m_sim oracle using Statevector.

    Uses a reduced K,M and small integer weights to keep the state space
    tractable (NumPy statevector limit ≈ 2^26 amplitudes).  Reports whether
    the highest-amplitude data state after one iteration is a good assignment.

    Parameters
    ----------
    sim_weight_scale : int
        Max integer weight for phonemes in the simulation oracle.  Smaller
        values reduce the WeightedAdder carry register and total qubit count.
    """
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Statevector

    signs_sim    = select_top_k_signs(corpus_seqs, k_sim)
    phonemes_sim = select_top_m_phonemes(lms, m_sim)

    # Use a reduced weight scale for simulation so the WeightedAdder carry
    # register stays small (carry width ∝ log2(max_sum)).
    lp_map  = _phoneme_unigram_lp(lms)
    raw     = [lp_map.get(ph, -20.0) for ph in phonemes_sim]
    lp_min  = min(raw)
    lp_max  = max(raw)
    span    = max(lp_max - lp_min, 1e-9)
    w_int   = [round((v - lp_min) / span * sim_weight_scale) for v in raw]
    thresh  = math.ceil(tau * k_sim * sim_weight_scale)

    w_sim = OracleWeights(
        phonemes=phonemes_sim,
        raw_lp=raw,
        w_int=w_int,
        threshold_int=thresh,
        threshold_tau=tau,
        k_signs=k_sim,
        m_phonemes=m_sim,
    )

    log.info(
        "Simulation: K=%d × M=%d  threshold_int=%d  τ=%.2f  weight_scale=%d",
        k_sim, m_sim, w_sim.threshold_int, tau, sim_weight_scale,
    )

    oracle_qc, qmap = build_grover_oracle(k_sim, m_sim, w_sim)

    if oracle_qc.num_qubits > 26:
        raise RuntimeError(
            f"Simulation oracle has {oracle_qc.num_qubits} qubits — "
            f"exceeds NumPy statevector limit of 26.  "
            f"Reduce --sim-k/--sim-m or decrease sim_weight_scale."
        )
    n_data = k_sim * m_sim

    # ── Full Grover iteration circuit ─────────────────────────────────────────
    n_total = oracle_qc.num_qubits
    n_ancilla = n_total - n_data - 1  # -1 for phase ancilla

    grover = QuantumCircuit(n_total, n_data)

    # Init data register in uniform superposition
    grover.h(range(n_data))

    # Phase ancilla: init at |0⟩ — oracle itself applies X+H
    # One Grover iteration = oracle + diffuser on data
    grover.compose(oracle_qc, inplace=True)

    # Diffuser (on data qubits only)
    diff = build_diffuser(n_data)
    grover.compose(diff, qubits=range(n_data), inplace=True)

    # Measure data qubits
    grover.measure(range(n_data), range(n_data))

    # Statevector before measurement (drop the measurement layer)
    grover_no_meas = QuantumCircuit(n_total)
    grover_no_meas.h(range(n_data))
    grover_no_meas.compose(oracle_qc, inplace=True)
    grover_no_meas.compose(diff, qubits=range(n_data), inplace=True)

    sv = Statevector(grover_no_meas)

    # Marginalise over ancilla qubits to get data-register probabilities
    # Data qubits are [0 .. n_data-1] in the statevector (LSB = qubit 0)
    amps = sv.data
    n2 = 2 ** n_total
    n_data_states = 2 ** n_data
    n_anc_states  = 2 ** (n_total - n_data)

    # Sum over ancilla states for each data state
    probs_data = np.zeros(n_data_states)
    for anc_idx in range(n_anc_states):
        for d_idx in range(n_data_states):
            # In Qiskit convention: qubit 0 is the least-significant bit
            full_idx = d_idx + anc_idx * n_data_states
            if full_idx < n2:
                probs_data[d_idx] += abs(amps[full_idx]) ** 2

    best_idx = int(np.argmax(probs_data))
    best_prob = float(probs_data[best_idx])
    best_bits = format(best_idx, f"0{n_data}b")[::-1]  # LSB first

    # Decode assignment from best bits
    assignment = {}
    for ki in range(k_sim):
        group_bits = best_bits[ki * m_sim:(ki + 1) * m_sim]
        set_positions = [j for j, b in enumerate(group_bits) if b == "1"]
        if len(set_positions) == 1:
            assignment[f"sign_{ki}"] = w_sim.phonemes[set_positions[0]]
        else:
            assignment[f"sign_{ki}"] = f"<invalid:{group_bits}>"

    # Compute score of best assignment
    score_int = 0
    is_valid_one_hot = True
    for ki in range(k_sim):
        group_bits = best_bits[ki * m_sim:(ki + 1) * m_sim]
        set_pos = [j for j, b in enumerate(group_bits) if b == "1"]
        if len(set_pos) == 1:
            score_int += w_sim.w_int[set_pos[0]]
        else:
            is_valid_one_hot = False

    is_good = is_valid_one_hot and score_int >= w_sim.threshold_int

    log.info(
        "Best data state: %s  prob=%.4f  score_int=%d  threshold=%d  "
        "one_hot=%s  good=%s",
        best_bits, best_prob, score_int, w_sim.threshold_int,
        is_valid_one_hot, is_good,
    )

    return {
        "k_sim": k_sim,
        "m_sim": m_sim,
        "tau": tau,
        "threshold_int": w_sim.threshold_int,
        "best_data_bits": best_bits,
        "best_amplitude_prob": round(best_prob, 6),
        "assignment": assignment,
        "score_int": score_int,
        "is_valid_one_hot": is_valid_one_hot,
        "is_good_state": is_good,
        "target_has_highest_amplitude": is_good,
        "oracle_num_qubits": oracle_qc.num_qubits,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build and analyse the Grover oracle for rongorongo.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--k",          type=int,   default=8,  metavar="K",
                   help="Top-K signs by IC contribution (default 8).")
    p.add_argument("--m",          type=int,   default=8,  metavar="M",
                   help="Top-M phonemes by unigram score (default 8).")
    p.add_argument("--tau",        type=float, default=0.90, metavar="TAU",
                   help="Normalised score threshold (default 0.90).")
    p.add_argument("--corpus-dir", type=Path,  default=None, metavar="DIR")
    p.add_argument("--lm-dir",     type=Path,  default=None, metavar="DIR")
    p.add_argument(
        "--draw", action="store_true",
        help="Save circuit diagram to outputs/quantum/grover_oracle_circuit.png.",
    )
    p.add_argument(
        "--simulate", action="store_true",
        help="Run one Grover iteration (reduced K=2,M=2) on StatevectorSampler.",
    )
    p.add_argument(
        "--sim-k", type=int, default=2, metavar="K",
        help="K for simulation (default 2 — must be small enough to simulate).",
    )
    p.add_argument(
        "--sim-m", type=int, default=2, metavar="M",
        help="M for simulation (default 2).",
    )
    p.add_argument(
        "--oracle-output", type=Path, default=None, metavar="JSON",
        help="Save circuit_stats dict to this JSON file.",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    corpus_dir = args.corpus_dir
    lm_dir     = args.lm_dir

    if corpus_dir is None or lm_dir is None:
        try:
            from omegaconf import OmegaConf
            cfg = OmegaConf.load(PROJECT_ROOT / "conf" / "config.yaml")
            if corpus_dir is None:
                corpus_dir = PROJECT_ROOT / cfg.paths.corpus_dir
            if lm_dir is None:
                lm_dir = PROJECT_ROOT / "data" / "language_models"
        except Exception:
            pass

    if corpus_dir is None or not corpus_dir.exists():
        log.error("Corpus directory not found.  Pass --corpus-dir.")
        sys.exit(1)
    if lm_dir is None or not lm_dir.exists():
        log.error("LM directory not found.  Pass --lm-dir.")
        sys.exit(1)

    log.info("Loading corpus from %s …", corpus_dir)
    corpus_seqs = _load_barthel_sequences(corpus_dir)
    if not corpus_seqs:
        log.error("No corpus sequences found in %s.", corpus_dir)
        sys.exit(1)
    log.info("  %d tablets loaded.", len(corpus_seqs))

    log.info("Loading language models from %s …", lm_dir)
    lms = _load_lms(lm_dir)
    if not lms:
        log.error("No LMs found in %s — run build_language_models.py first.", lm_dir)
        sys.exit(1)

    # ── Select signs and phonemes ─────────────────────────────────────────────
    signs    = select_top_k_signs(corpus_seqs, args.k)
    phonemes = select_top_m_phonemes(lms, args.m)
    log.info("Top-%d signs: %s", args.k, signs)
    log.info("Top-%d phonemes: %s", args.m, phonemes)

    weights = compute_oracle_weights(phonemes, lms, args.k, tau=args.tau)

    # ── Build oracle ──────────────────────────────────────────────────────────
    log.info("Building oracle circuit (K=%d, M=%d) …", args.k, args.m)
    t0 = time.monotonic()
    oracle_qc, qubit_map = build_grover_oracle(args.k, args.m, weights)
    elapsed = time.monotonic() - t0
    log.info("Circuit built in %.1f s.", elapsed)

    # ── Compute and print statistics ──────────────────────────────────────────
    stats = compute_stats(oracle_qc)
    print_stats(stats, weights)

    circuit_stats = {
        "k_signs":             args.k,
        "m_phonemes":          args.m,
        "n_variables":         args.k * args.m,
        "tau":                 args.tau,
        "threshold_int":       weights.threshold_int,
        "top_signs":           signs,
        "top_phonemes":        phonemes,
        "phoneme_w_int":       weights.w_int,
        "num_qubits":          stats.num_qubits,
        "depth":               stats.depth,
        "num_nonlocal_gates":  stats.num_nonlocal_gates,
        "t_gate_count_approx": stats.t_gate_count_approx,
        "surface_code_d3_phys": stats.surface_code_d3_phys,
        "build_time_seconds":  round(elapsed, 2),
    }

    # ── Draw ──────────────────────────────────────────────────────────────────
    if args.draw:
        out_dir = PROJECT_ROOT / "outputs" / "quantum"
        out_dir.mkdir(parents=True, exist_ok=True)
        png_path = out_dir / "grover_oracle_circuit.png"
        log.info("Drawing circuit to %s …", png_path)
        fig = oracle_qc.draw(output="mpl", fold=-1)
        fig.savefig(png_path, dpi=150, bbox_inches="tight")
        log.info("Saved %s", png_path)
        circuit_stats["diagram_path"] = str(png_path)

    # ── Simulate ──────────────────────────────────────────────────────────────
    if args.simulate:
        log.info(
            "Running Grover simulation (K=%d, M=%d) — "
            "full K=%d,M=%d requires fault-tolerant hardware …",
            args.sim_k, args.sim_m, args.k, args.m,
        )
        sim_result = run_grover_simulation(
            k_sim=args.sim_k,
            m_sim=args.sim_m,
            lms=lms,
            corpus_seqs=corpus_seqs,
            tau=args.tau,
        )
        circuit_stats["simulation"] = sim_result
        print(f"\n  Grover simulation (K={args.sim_k}, M={args.sim_m}):")
        print(f"    best state  : {sim_result['best_data_bits']}")
        print(f"    amplitude²  : {sim_result['best_amplitude_prob']:.4f}")
        print(f"    assignment  : {sim_result['assignment']}")
        print(f"    score_int   : {sim_result['score_int']}  (threshold={sim_result['threshold_int']})")
        print(f"    is_good     : {sim_result['is_good_state']}")
        print(f"    target has highest amplitude: {sim_result['target_has_highest_amplitude']}")
        print()

    # ── Save ──────────────────────────────────────────────────────────────────
    if args.oracle_output:
        args.oracle_output.parent.mkdir(parents=True, exist_ok=True)
        args.oracle_output.write_text(
            json.dumps(circuit_stats, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        log.info("Circuit stats written to %s", args.oracle_output)

    return circuit_stats


# ---------------------------------------------------------------------------
# Public API (imported by measure_pgood.py via --oracle-circuit)
# ---------------------------------------------------------------------------

def build_oracle_stats(
    corpus_dir: Path,
    lm_dir: Path,
    k: int = 8,
    m: int = 8,
    tau: float = 0.90,
) -> dict:
    """Build the oracle and return circuit_stats dict.  No side effects."""
    corpus_seqs = _load_barthel_sequences(corpus_dir)
    lms         = _load_lms(lm_dir)
    signs       = select_top_k_signs(corpus_seqs, k)
    phonemes    = select_top_m_phonemes(lms, m)
    weights     = compute_oracle_weights(phonemes, lms, k, tau=tau)

    oracle_qc, _ = build_grover_oracle(k, m, weights)
    stats        = compute_stats(oracle_qc)

    return {
        "k_signs":              k,
        "m_phonemes":           m,
        "n_variables":          k * m,
        "tau":                  tau,
        "threshold_int":        weights.threshold_int,
        "top_signs":            signs,
        "top_phonemes":         phonemes,
        "phoneme_w_int":        weights.w_int,
        "num_qubits":           stats.num_qubits,
        "depth":                stats.depth,
        "num_nonlocal_gates":   stats.num_nonlocal_gates,
        "t_gate_count_approx":  stats.t_gate_count_approx,
        "surface_code_d3_phys": stats.surface_code_d3_phys,
    }


if __name__ == "__main__":
    main()
