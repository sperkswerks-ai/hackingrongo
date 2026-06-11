#!/usr/bin/env python3
"""
run_bv_ic_analysis.py — Bernstein-Vazirani algorithm on the rongorongo IC distribution.

Background
----------
BV solves  f(x) = s·x (mod 2)  for a hidden bitstring s in *one* quantum oracle
query, vs O(n) classical queries (by querying each basis vector) or O(2^n)
exhaustive classical search.

This script asks: does the IC-contribution distribution of the rongorongo sign
inventory contain a hidden linear Boolean structure?  We define

    f : {0,1}^7 → {0,1}
    f(x) = 1  if sign at index x has IC contribution above the median, else 0

and test whether f is of the form  f(x) = s·x (mod 2) for some hidden bitstring s.

A null result (no such s) is itself scientifically publishable: it rules out an
entire class of cipher structure and constrains what linear algebraic models can
explain the frequency distribution.  The best linear approximation (BLA) is
still computed and interpreted even in the null case.

Query-complexity comparison
---------------------------
    Quantum (BV)                : 1 oracle query → exact s
    Classical (BV-like)         : n = 7 queries (evaluate on basis vectors)
    Classical (exhaustive)      : 2^n = 128 queries (try every candidate s)

Encodings
---------
Two sign→index encodings are available (``--encoding``):

``ic_rank`` (default)
    Signs indexed by descending IC rank.  With the median threshold this
    makes f affine BY CONSTRUCTION (f(x) = 1 iff rank < n_top — the slope
    is the top bit of the rank).  The hardware run then demonstrates BV
    oracle execution, not a corpus discovery.  Kept as default for
    reproducibility of earlier runs and the CI golden files.

``barthel_bits``
    Signs indexed by the low bits of their Barthel catalogue number, which
    is independent of the IC value being thresholded.  Linearity is then a
    falsifiable property of how IC mass distributes over Barthel codes —
    either outcome (structure found, or null result) is a genuine corpus
    measurement.

Usage
-----
    python scripts/run_bv_ic_analysis.py
    python scripts/run_bv_ic_analysis.py --encoding barthel_bits
    python scripts/run_bv_ic_analysis.py --n-bits 6 --n-top 48
    python scripts/run_bv_ic_analysis.py --draw
    python scripts/run_bv_ic_analysis.py --backend ibmq --ibmq-token <TOKEN>
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

_CORPUS_DIR = PROJECT_ROOT / "data" / "corpus"
_OUTPUT_PATH = PROJECT_ROOT / "outputs" / "quantum" / "bv_ic_result.json"

N_BITS_DEFAULT = 7      # 2^7 = 128 > 120 known rongorongo sign types
N_TOP_DEFAULT  = 64     # top signs covering ~90% of corpus IC


# ── Corpus & IC ───────────────────────────────────────────────────────────────

def load_barthel_sequences(corpus_dir: Path) -> list[list[str]]:
    sequences: list[list[str]] = []
    for path in sorted(corpus_dir.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        seq = [str(g["barthel_code"])
               for g in data.get("glyphs", [])
               if g.get("barthel_code")]
        if len(seq) >= 3:
            sequences.append(seq)
    return sequences


def compute_ic_contributions(corpus_seqs: list[list[str]]) -> dict[str, float]:
    """Return p_i^2 (IC contribution) for each sign i, NOT normalised yet."""
    freq: Counter[str] = Counter()
    for seq in corpus_seqs:
        freq.update(seq)
    total = sum(freq.values())
    if total == 0:
        raise ValueError("Corpus is empty — no glyph tokens found.")
    return {sign: (count / total) ** 2 for sign, count in freq.items()}


def normalise_ic(ic: dict[str, float]) -> dict[str, float]:
    """Normalise IC contributions so max = 1."""
    max_ic = max(ic.values())
    return {s: v / max_ic for s, v in ic.items()}


# ── Sign index ────────────────────────────────────────────────────────────────

def build_sign_index(
    ic_norm: dict[str, float],
    n_top: int,
    n_bits: int,
) -> tuple[dict[int, str], dict[str, int]]:
    """Sort signs by IC (descending), assign integer indices 0..n_top-1.

    Returns (index_to_sign, sign_to_index).
    Domain: {0, …, 2^n_bits − 1}.  Indices ≥ n_top are unoccupied (IC = 0).

    CAUTION — with this encoding plus a median threshold, f(x) = 1 iff
    rank(x) < n_top is affine *by construction* (the recovered slope is
    the top bit of the rank).  Use ``--encoding barthel_bits`` for an
    encoding under which linearity is a falsifiable corpus property.
    """
    ranked = sorted(ic_norm.items(), key=lambda kv: -kv[1])[:n_top]
    idx_to_sign = {i: sign for i, (sign, _) in enumerate(ranked)}
    sign_to_idx = {sign: i for i, sign in idx_to_sign.items()}
    return idx_to_sign, sign_to_idx


_BARTHEL_DIGITS = re.compile(r"\d+")


def build_sign_index_barthel(
    ic_norm: dict[str, float],
    n_top: int,
    n_bits: int,
) -> tuple[dict[int, str], dict[str, int], int]:
    """Index the top-IC signs by the low ``n_bits`` of their Barthel code.

    Unlike the IC-rank encoding, the index here is *independent of the
    quantity being thresholded*: which domain points hold high-IC signs is
    determined by Barthel's catalogue numbering, so any linear/affine
    structure that survives is a genuine property of how IC mass
    distributes over the sign codes — not an artefact of rank ordering.

    Collisions (two signs whose codes share low bits) are resolved by
    keeping the higher-IC sign.  Returns
    (index_to_sign, sign_to_index, n_collisions).
    """
    domain_size = 1 << n_bits
    ranked = sorted(ic_norm.items(), key=lambda kv: -kv[1])[:n_top]
    idx_to_sign: dict[int, str] = {}
    n_collisions = 0
    for sign, _ic in ranked:  # descending IC: first claim wins
        m = _BARTHEL_DIGITS.search(sign)
        if not m:
            continue
        idx = int(m.group(0)) % domain_size
        if idx in idx_to_sign:
            n_collisions += 1
            continue
        idx_to_sign[idx] = sign
    sign_to_idx = {sign: i for i, sign in idx_to_sign.items()}
    return idx_to_sign, sign_to_idx, n_collisions


# ── Truth table ───────────────────────────────────────────────────────────────

def build_truth_table(
    ic_norm: dict[str, float],
    idx_to_sign: dict[int, str],
    n_bits: int,
) -> tuple[dict[int, int], float]:
    """Define f(x) = 1 iff IC(sign at index x) is above the median.

    The domain has 2^n_bits elements.  Unoccupied indices (no sign mapped)
    have IC contribution 0.  Median is computed over all 2^n_bits values.

    Returns (truth_table dict, median_threshold).
    """
    domain_size = 1 << n_bits
    ic_values = [ic_norm.get(idx_to_sign.get(x, ""), 0.0) for x in range(domain_size)]

    sorted_vals = sorted(ic_values)
    mid = domain_size // 2
    median = (sorted_vals[mid - 1] + sorted_vals[mid]) / 2.0

    truth_table = {x: int(ic_values[x] > median) for x in range(domain_size)}
    n_ones = sum(truth_table.values())
    log.info(
        "Truth table: domain=%d, median=%.6f, f=1 for %d/%d inputs (%.1f%%)",
        domain_size, median, n_ones, domain_size, 100 * n_ones / domain_size,
    )
    return truth_table, median


# ── Linearity testing ─────────────────────────────────────────────────────────

def check_affine_structure(
    truth_table: dict[int, int],
    n_bits: int,
) -> tuple[float, float, bool, bool]:
    """Brute-force check for linear AND affine structure.

    Linear:  f(x⊕y) = f(x) ⊕ f(y)             for all x,y
    Affine:  f(x⊕y) = f(x) ⊕ f(y) ⊕ f(0)       for all x,y
    (BV recovers the slope s for both; the constant f(0) only adds a global phase.)

    Returns (linear_frac, affine_frac, is_exactly_linear, is_exactly_affine).
    """
    f0 = truth_table[0]
    domain_size = 1 << n_bits
    n_total = domain_size * domain_size
    n_lin = 0
    n_aff = 0
    for x in range(domain_size):
        fx = truth_table[x]
        for y in range(domain_size):
            fy   = truth_table[y]
            fxoy = truth_table[x ^ y]
            if fxoy == fx ^ fy:
                n_lin += 1
            if fxoy == fx ^ fy ^ f0:
                n_aff += 1
    return (n_lin / n_total, n_aff / n_total, n_lin == n_total, n_aff == n_total)


# ── Walsh-Hadamard transform ──────────────────────────────────────────────────

def walsh_hadamard_transform(truth_table: dict[int, int], n_bits: int) -> list[float]:
    """Return the Walsh-Hadamard coefficients W_f(s) for s in 0..2^n-1.

    W_f(s) = sum_{x} (-1)^{f(x) + s·x}
    For a linear f(x) = s_0·x: W_f(s_0) = 2^n, W_f(s ≠ s_0) = 0.
    """
    n = 1 << n_bits
    W = [float(1 - 2 * truth_table[x]) for x in range(n)]  # {0,1} → {+1,-1}
    step = 1
    while step < n:
        for i in range(0, n, step * 2):
            for j in range(i, i + step):
                a, b = W[j], W[j + step]
                W[j]        = a + b
                W[j + step] = a - b
        step *= 2
    return W


def find_best_linear_approx(
    truth_table: dict[int, int],
    n_bits: int,
    wht: list[float],
) -> tuple[int, float, bool]:
    """Return (s_bla, agreement_fraction, is_affine_variant) for the BLA.

    The BLA is the s maximising |W_f(s)|.
    Agreement fraction = (2^n + |W_f(s_bla)|) / (2^(n+1)).
    If W_f(s_bla) < 0, the best fit is the AFFINE variant f(x) = 1 ⊕ s·x
    (is_affine_variant=True); if > 0, it is the LINEAR variant f(x) = s·x.
    In both cases BV recovers s identically (global phase has no effect on
    measurement outcomes).
    """
    domain_size = 1 << n_bits
    s_bla = max(range(domain_size), key=lambda s: abs(wht[s]))
    agree_frac = (domain_size + abs(wht[s_bla])) / (2 * domain_size)
    is_affine_variant = wht[s_bla] < 0
    return s_bla, agree_frac, is_affine_variant


def linearity_fraction_from_wht(wht: list[float], n_bits: int) -> float:
    """Linearity fraction from WHT: (2^n + (max|W|/2^n)^2 * 2^n) / (2 * 2^n)."""
    n = 1 << n_bits
    max_w = max(abs(w) for w in wht)
    return (n + (max_w / n) ** 2 * n) / (2 * n)


# ── Classical exhaustive search ───────────────────────────────────────────────

def _dot_gf2(a: int, b: int) -> int:
    return bin(a & b).count("1") % 2


def classical_exhaustive_search(
    truth_table: dict[int, int],
    n_bits: int,
) -> tuple[int | None, bool, int, float]:
    """Try all 2^n_bits candidates for the affine slope s.

    Checks both f(x) = s·x and f(x) = 1 ⊕ s·x (BV recovers the same s
    for both).

    Returns (s_exact, is_affine_variant, n_queries, best_agree).
    s_exact is None if no candidate matches exactly.
    n_queries counts total f evaluations performed (domain_size per candidate).
    """
    domain_size = 1 << n_bits
    n_queries = 0
    best_s = 0
    best_agree = 0.0
    exact_s: int | None = None
    exact_is_affine: bool = False

    for s in range(domain_size):
        match_lin = sum(
            1 for x in range(domain_size)
            if truth_table[x] == _dot_gf2(x, s)
        )
        match_aff = domain_size - match_lin  # affine = complementary pattern
        n_queries += domain_size
        best_of_two = max(match_lin, match_aff)
        agree = best_of_two / domain_size
        if agree > best_agree:
            best_agree = agree
            best_s = s
        if match_lin == domain_size:
            exact_s = s
            exact_is_affine = False
        elif match_aff == domain_size:
            exact_s = s
            exact_is_affine = True

    return exact_s, exact_is_affine, n_queries, best_agree


# ── Quantum circuit ───────────────────────────────────────────────────────────

def build_bv_oracle(truth_table: dict[int, int], n_bits: int):
    """Phase-kickback oracle for f using MCX gates.

    Assumes ancilla is in |−⟩ = (|0⟩−|1⟩)/√2 before and after.
    Each x with f(x)=1 contributes a pattern-specific MCX term.
    Gate count: O(|{x: f(x)=1}| × n_bits).
    """
    from qiskit import QuantumCircuit, QuantumRegister
    reg_in  = QuantumRegister(n_bits, "in")
    reg_anc = QuantumRegister(1, "anc")
    qc = QuantumCircuit(reg_in, reg_anc, name="O_f")

    ones = [x for x in range(1 << n_bits) if truth_table[x] == 1]
    for x in ones:
        # Invert qubits where x[i] = 0 so MCX fires exactly on this pattern
        for i in range(n_bits):
            if not ((x >> i) & 1):
                qc.x(reg_in[i])
        qc.mcx(list(reg_in), reg_anc[0])
        # Uncompute inversions
        for i in range(n_bits):
            if not ((x >> i) & 1):
                qc.x(reg_in[i])

    return qc


def build_bv_linear_oracle(s: int, n_bits: int):
    """Efficient CNOT-only oracle for the linear function f(x) = s·x mod 2.

    Only valid when f IS exactly linear with period s.
    Gate count: O(popcount(s)) ≤ n_bits.
    """
    from qiskit import QuantumCircuit, QuantumRegister
    reg_in  = QuantumRegister(n_bits, "in")
    reg_anc = QuantumRegister(1, "anc")
    qc = QuantumCircuit(reg_in, reg_anc, name="O_f_lin")
    for i in range(n_bits):
        if (s >> i) & 1:
            qc.cx(reg_in[i], reg_anc[0])
    return qc


def build_bv_circuit(
    truth_table: dict[int, int],
    n_bits: int,
    s_exact: int | None,
) -> object:
    """Full BV circuit: |0⟩^n|1⟩ → H^⊗(n+1) → O_f → H^⊗n → measure.

    If f is exactly linear (s_exact provided), uses the efficient CNOT oracle.
    Otherwise uses the general MCX oracle (works but BV semantics are approximate).
    """
    from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
    reg_in  = QuantumRegister(n_bits, "in")
    reg_anc = QuantumRegister(1, "anc")
    reg_c   = ClassicalRegister(n_bits, "meas")
    qc = QuantumCircuit(reg_in, reg_anc, reg_c, name="BV")

    # Prepare |−⟩ ancilla
    qc.x(reg_anc[0])
    qc.h(reg_anc[0])

    # Uniform superposition on input
    qc.h(reg_in)

    # Oracle
    if s_exact is not None:
        oracle = build_bv_linear_oracle(s_exact, n_bits)
    else:
        oracle = build_bv_oracle(truth_table, n_bits)
    qc.compose(oracle, qubits=list(reg_in) + list(reg_anc), inplace=True)

    # Hadamard to extract s
    qc.h(reg_in)
    qc.measure(reg_in, reg_c)
    return qc


# ── Samplers ──────────────────────────────────────────────────────────────────

def run_statevector(qc, n_bits: int, shots: int = 1) -> tuple[dict[str, int], float]:
    from qiskit.primitives import StatevectorSampler
    t0 = time.monotonic()
    sampler = StatevectorSampler()
    job = sampler.run([qc], shots=shots)
    result = job.result()[0]
    counts = dict(result.data.meas.get_counts())
    return counts, time.monotonic() - t0


def run_ibmq(
    qc,
    n_bits: int,
    shots: int,
    token: str,
    instance: str | None = None,
    backend_name: str | None = None,
) -> tuple[dict[str, int], float, dict]:
    """Returns (counts, elapsed_seconds, provenance_dict)."""
    import os
    from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
    from qiskit.compiler import transpile
    from hackingrongo.quantum_provenance import collect_provenance

    instance = instance or os.environ.get("IBMQ_INSTANCE")
    t0 = time.monotonic()
    service = QiskitRuntimeService(channel="ibm_quantum_platform", token=token, instance=instance)
    backend = (
        service.backend(backend_name) if backend_name
        else service.least_busy(operational=True, simulator=False, min_num_qubits=n_bits + 1)
    )
    log.info("IBM Quantum backend: %s (%d qubits)", backend.name, backend.num_qubits)

    t_qc = transpile(qc, backend=backend, optimization_level=2)
    sampler = SamplerV2(mode=backend)
    job = sampler.run([t_qc], shots=shots)
    log.info("Job submitted: %s", job.job_id())
    job_result = job.result()
    counts = dict(job_result[0].data.meas.get_counts())
    prov = collect_provenance(job, backend)
    return counts, time.monotonic() - t0, prov


def _decode_top_result(counts: dict[str, int]) -> int:
    """Return the most-frequent measurement as an integer (MSB-first string → int)."""
    top_bitstr = max(counts, key=counts.get)
    return int(top_bitstr, 2)


# ── Interpretation ────────────────────────────────────────────────────────────

def interpret_s(
    s: int,
    n_bits: int,
    idx_to_sign: dict[int, str],
    ic_norm: dict[str, float],
) -> dict[str, Any]:
    """Describe which bit positions are set in s and what signs they correspond to."""
    set_bits = [i for i in range(n_bits) if (s >> i) & 1]
    contributing_signs = []
    for bit in set_bits:
        sign = idx_to_sign.get(bit, None)
        ic   = ic_norm.get(sign, 0.0) if sign else 0.0
        contributing_signs.append({
            "bit_position": bit,
            "ic_rank":      bit,           # index = rank (0 = highest IC)
            "barthel_code": sign,
            "ic_norm":      round(ic, 6),
        })
    return {
        "s_int":    s,
        "s_bits":   f"{s:0{n_bits}b}",
        "set_bits": set_bits,
        "contributing_signs": contributing_signs,
        "interpretation": (
            f"s = {s:0{n_bits}b} ({s}) has {len(set_bits)} bit(s) set at positions "
            f"{set_bits}.  These correspond to IC-ranked signs "
            f"{[idx_to_sign.get(b, 'unoccupied') for b in set_bits]}.  "
            "A nonzero s means the IC distribution's linear structure couples the "
            "parities of these sign ranks."
        ) if set_bits else (
            "s = 0 (all bits 0): f(x) = 0 for all x — the trivial zero function.  "
            "This indicates no linear correlation structure."
        ),
    }


# ── Main analysis ─────────────────────────────────────────────────────────────

def run_analysis(
    corpus_dir: Path,
    n_top: int,
    n_bits: int,
    backend: str,
    ibmq_token: str | None,
    ibmq_instance: str,
    ibmq_backend_name: str | None,
    linearity_threshold: float,
    draw: bool,
    encoding: str = "ic_rank",
) -> dict[str, Any]:

    result: dict[str, Any] = {
        "corpus_dir":   str(corpus_dir),
        "n_top_signs":  n_top,
        "n_bits":       n_bits,
        "domain_size":  1 << n_bits,
        "encoding":     encoding,
    }

    # ── Step 1: Load IC contributions ─────────────────────────────────────────
    log.info("Loading corpus from %s …", corpus_dir)
    corpus_seqs = load_barthel_sequences(corpus_dir)
    if not corpus_seqs:
        raise FileNotFoundError(f"No corpus JSON files found in {corpus_dir}")
    total_tokens = sum(len(s) for s in corpus_seqs)
    log.info("  %d tablet(s), %d glyph tokens", len(corpus_seqs), total_tokens)

    ic_raw  = compute_ic_contributions(corpus_seqs)
    ic_norm = normalise_ic(ic_raw)
    ic_total = sum(ic_raw.values())
    log.info("  IC (raw) total = %.6f  (|vocabulary| = %d)", ic_total, len(ic_raw))

    if encoding == "barthel_bits":
        idx_to_sign, sign_to_idx, n_collisions = build_sign_index_barthel(
            ic_norm, n_top, n_bits
        )
        result["encoding_collisions"] = n_collisions
        log.info(
            "  Barthel-bits encoding: %d signs placed, %d collisions dropped",
            len(idx_to_sign), n_collisions,
        )
    else:
        idx_to_sign, sign_to_idx = build_sign_index(ic_norm, n_top, n_bits)
    top_n_ic_share = sum(ic_raw.get(sign, 0) for sign in idx_to_sign.values()) / ic_total
    log.info(
        "  Top %d signs cover %.1f%% of total IC",
        n_top, 100 * top_n_ic_share,
    )

    result["corpus_stats"] = {
        "n_tablets":    len(corpus_seqs),
        "n_tokens":     total_tokens,
        "ic_total_raw": round(ic_total, 8),
        "vocabulary_size": len(ic_raw),
        "top_n_ic_share": round(top_n_ic_share, 4),
    }
    result["top_signs"] = [
        {
            "index":        i,
            "barthel_code": idx_to_sign[i],
            "ic_norm":      round(ic_norm[idx_to_sign[i]], 6),
            "ic_raw":       round(ic_raw.get(idx_to_sign[i], 0), 8),
        }
        for i in sorted(idx_to_sign)
    ]

    # ── Step 2: Build truth table ──────────────────────────────────────────────
    truth_table, median_threshold = build_truth_table(ic_norm, idx_to_sign, n_bits)
    n_ones = sum(truth_table.values())
    result["truth_table"] = {
        "median_threshold": round(median_threshold, 8),
        "n_ones":           n_ones,
        "n_zeros":          (1 << n_bits) - n_ones,
        "balance":          round(n_ones / (1 << n_bits), 4),
    }

    # ── Step 3: Affine/linearity check ────────────────────────────────────────
    log.info("Computing Walsh-Hadamard transform …")
    wht = walsh_hadamard_transform(truth_table, n_bits)
    domain_size = 1 << n_bits

    log.info("Running brute-force affine/linearity check (%d pairs) …",
             domain_size * domain_size)
    lin_frac, aff_frac, is_exactly_linear, is_exactly_affine = check_affine_structure(
        truth_table, n_bits
    )
    log.info(
        "  Linear fraction: %.4f  Affine fraction: %.4f"
        "  (exactly_linear=%s, exactly_affine=%s)",
        lin_frac, aff_frac, is_exactly_linear, is_exactly_affine,
    )

    s_bla, bla_agree, bla_is_affine = find_best_linear_approx(truth_table, n_bits, wht)
    bla_type = "affine (1⊕s·x)" if bla_is_affine else "linear (s·x)"
    log.info(
        "  BLA: s = %d (%s), type = %s, agreement = %.4f",
        s_bla, f"{s_bla:0{n_bits}b}", bla_type, bla_agree,
    )

    # BV is applicable to both linear AND affine f (global phase doesn't matter)
    bv_applicable = is_exactly_linear or is_exactly_affine
    aff_or_lin_frac = aff_frac if not is_exactly_linear else lin_frac
    passes = bv_applicable or aff_or_lin_frac >= linearity_threshold

    result["linearity"] = {
        "linear_fraction":           round(lin_frac, 6),
        "affine_fraction":           round(aff_frac, 6),
        "is_exactly_linear":         is_exactly_linear,
        "is_exactly_affine":         is_exactly_affine,
        "bv_applicable":             bv_applicable,
        "threshold":                 linearity_threshold,
        "passes_threshold":          passes,
        "wht_max_coeff":             round(max(abs(w) for w in wht), 4),
        "wht_max_s":                 int(max(range(domain_size), key=lambda s: abs(wht[s]))),
        "best_approx_s":             s_bla,
        "best_approx_s_bits":        f"{s_bla:0{n_bits}b}",
        "best_approx_type":          bla_type,
        "best_approx_agree":         round(bla_agree, 6),
    }

    # ── Step 4 (classical baseline): exhaustive search ─────────────────────────
    log.info("Running classical exhaustive search over %d candidate s values …", domain_size)
    t_cls = time.monotonic()
    s_exact_cls, cls_is_affine, n_queries_cls, best_agree_cls = classical_exhaustive_search(
        truth_table, n_bits
    )
    elapsed_cls = time.monotonic() - t_cls
    log.info(
        "  Classical result: s_exact=%s (affine=%s), best_agree=%.4f, "
        "n_queries=%d  (%.3fs)",
        s_exact_cls, cls_is_affine, best_agree_cls, n_queries_cls, elapsed_cls,
    )

    result["classical_search"] = {
        "n_queries":          n_queries_cls,
        "elapsed_seconds":    round(elapsed_cls, 4),
        "s_exact":            s_exact_cls,
        "s_exact_is_affine":  cls_is_affine,
        "best_agree":         round(best_agree_cls, 6),
        "description":        (
            f"Exhaustive search checked all {domain_size} candidate s values "
            f"(both linear and affine variants) at {domain_size} inputs each "
            f"= {n_queries_cls} oracle queries total."
        ),
    }

    # Null-result path: neither linear nor affine, and below threshold
    if not passes:
        null_msg = (
            f"IC structure is non-linear and non-affine — BV null result.  "
            f"Affine fraction = {aff_frac:.4f} < threshold {linearity_threshold}.  "
            f"Best approx: s = {s_bla:0{n_bits}b} ({s_bla}) [{bla_type}] "
            f"agrees on {bla_agree:.1%} of inputs.  "
            f"This rules out a linear Boolean separability structure in the IC distribution."
        )
        log.warning(null_msg)
        result["verdict"] = {
            "is_null_result":       True,
            "message":              null_msg,
            "bla_interpretation":   interpret_s(s_bla, n_bits, idx_to_sign, ic_norm),
            "query_comparison": {
                "quantum_queries":   "N/A (null result — BV circuit not run)",
                "classical_queries": n_queries_cls,
            },
        }
        _print_summary(result)
        return result

    # ── Steps 4–6: BV circuit ─────────────────────────────────────────────────
    log.info("Building BV circuit (n_bits=%d) …", n_bits)
    # Pass the exact s to use the efficient CNOT oracle when available
    bv_qc = build_bv_circuit(truth_table, n_bits, s_exact=s_exact_cls)
    circuit_info = {
        "num_qubits":  bv_qc.num_qubits,
        "depth":       bv_qc.depth(),
        "num_gates":   sum(bv_qc.count_ops().values()),
        "oracle_type": "cnot_linear" if s_exact_cls is not None else "mcx_general",
    }
    log.info("  Circuit: %d qubits, depth %d, %d gates",
             circuit_info["num_qubits"], circuit_info["depth"], circuit_info["num_gates"])
    result["bv_circuit"] = circuit_info

    if draw:
        _draw_circuit(bv_qc, n_bits)

    # Run
    shots = 1
    if backend == "ibmq":
        if not ibmq_token:
            raise ValueError("--ibmq-token required for --backend ibmq.")
        counts, elapsed_q, prov = run_ibmq(
            bv_qc, n_bits, shots,
            token=ibmq_token,
            instance=ibmq_instance,
            backend_name=ibmq_backend_name,
        )
        result["hardware_provenance"] = prov
    else:
        counts, elapsed_q = run_statevector(bv_qc, n_bits, shots)

    log.info("  Quantum measurement: %s  (%.4fs)", counts, elapsed_q)

    # Decode
    s_recovered = _decode_top_result(counts)
    matches_exact = (s_recovered == s_exact_cls) if s_exact_cls is not None else None
    log.info(
        "  Recovered s = %d (%s)  matches_exact=%s",
        s_recovered, f"{s_recovered:0{n_bits}b}", matches_exact,
    )

    # ── Step 6: Interpret ─────────────────────────────────────────────────────
    s_interp = interpret_s(s_recovered, n_bits, idx_to_sign, ic_norm)
    log.info("  %s", s_interp["interpretation"])

    f_type_desc = (
        "affine f(x) = 1 ⊕ s·x" if cls_is_affine else "linear f(x) = s·x"
    ) if s_exact_cls is not None else "approximately affine/linear"

    result["quantum_result"] = {
        "backend":            backend,
        "shots":              shots,
        "counts":             counts,
        "elapsed_seconds":    round(elapsed_q, 6),
        "recovered_s_int":    s_recovered,
        "recovered_s_bits":   f"{s_recovered:0{n_bits}b}",
        "matches_exact_s":    matches_exact,
        "f_type":             f_type_desc,
        "interpretation":     s_interp,
    }
    result["verdict"] = {
        "is_null_result":      False,
        "is_exactly_linear":   is_exactly_linear,
        "is_exactly_affine":   is_exactly_affine,
        "affine_fraction":     round(aff_frac, 6),
        "f_type":              f_type_desc,
        "message": (
            f"IC distribution has {f_type_desc} structure with hidden slope "
            f"s = {s_recovered:0{n_bits}b} ({s_recovered}).  "
            f"BV recovered s in 1 quantum query vs {n_queries_cls} classical "
            f"exhaustive queries ({n_bits} with the basis-vector BV classical method).  "
            + (
                "Note: the affine structure is a tautological artefact of the "
                "rank-based index encoding (occupied vs unoccupied indices)."
                if is_exactly_affine and not is_exactly_linear else ""
            )
        ),
        "query_comparison": {
            "quantum_queries":              1,
            "classical_bv_queries":         n_bits,
            "classical_exhaustive_queries": n_queries_cls,
            "quantum_speedup_vs_exhaustive": f"{n_queries_cls}x",
        },
    }

    _print_summary(result)
    return result


# ── Display ───────────────────────────────────────────────────────────────────

def _draw_circuit(bv_qc, n_bits: int) -> None:
    out_dir = PROJECT_ROOT / "outputs" / "quantum"
    out_dir.mkdir(parents=True, exist_ok=True)
    png_path = out_dir / "bv_ic_circuit.png"
    fig = bv_qc.draw(output="mpl", fold=-1)
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    log.info("Circuit diagram saved to %s", png_path)


def _print_summary(result: dict[str, Any]) -> None:
    n_bits  = result["n_bits"]
    lin     = result["linearity"]
    verdict = result["verdict"]
    print(f"\n{'═' * 70}")
    print("  Bernstein-Vazirani IC Analysis — Rongorongo")
    print(f"{'═' * 70}")
    cs = result["corpus_stats"]
    print(f"\n  Corpus: {cs['n_tablets']} tablets, {cs['n_tokens']} tokens, "
          f"{cs['vocabulary_size']} unique signs")
    print(f"  Top {result['n_top_signs']} signs: {cs['top_n_ic_share']:.1%} of IC")
    tt = result["truth_table"]
    print(f"  Truth table: f=1 for {tt['n_ones']}/{result['domain_size']} "
          f"inputs ({tt['balance']:.1%} density)")
    print(f"\n  Linear fraction : {lin['linear_fraction']:.4f}  "
          f"Affine fraction : {lin['affine_fraction']:.4f}  "
          f"(threshold: {lin['threshold']})")
    print(f"  WHT max coeff   : {lin['wht_max_coeff']:.1f} / {result['domain_size']} "
          f"(= {lin['wht_max_coeff']/result['domain_size']:.4f})")
    print(f"  Best approx     : s = {lin['best_approx_s_bits']} "
          f"({lin['best_approx_s']})  [{lin['best_approx_type']}]  "
          f"agrees {lin['best_approx_agree']:.1%}")

    if verdict["is_null_result"]:
        print(f"\n  ✗  NULL RESULT: {verdict['message']}")
        bla = verdict.get("bla_interpretation", {})
        if bla.get("set_bits"):
            print(f"     BLA s set bits: {bla['set_bits']}")
            for cs_entry in bla.get("contributing_signs", []):
                print(f"       bit {cs_entry['bit_position']}: "
                      f"Barthel {cs_entry['barthel_code']}  "
                      f"(IC_norm={cs_entry['ic_norm']:.4f})")
    else:
        qr = result.get("quantum_result", {})
        cmp = verdict["query_comparison"]
        f_type = verdict.get("f_type", "affine/linear")
        print(f"\n  ✓  STRUCTURE FOUND ({f_type})")
        print(f"     s = {qr['recovered_s_bits']} ({qr['recovered_s_int']})")
        print(f"     1 quantum query vs {cmp['classical_exhaustive_queries']} classical exhaustive")
        interp = qr.get("interpretation", {})
        for cs_entry in interp.get("contributing_signs", []):
            print(f"     bit {cs_entry['bit_position']}: "
                  f"Barthel {cs_entry['barthel_code']}  "
                  f"(IC_norm={cs_entry['ic_norm']:.4f})")
        if "tautological" in verdict.get("message", ""):
            print("     (!) Affine structure is a tautology of the rank-based encoding.")
    print()


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="BV algorithm on rongorongo IC distribution.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--corpus-dir",  type=Path, default=_CORPUS_DIR,  metavar="DIR")
    p.add_argument("--output",      type=Path, default=_OUTPUT_PATH, metavar="JSON")
    p.add_argument("--n-bits",      type=int,  default=N_BITS_DEFAULT, metavar="N",
                   help=f"Bits for sign index (default: {N_BITS_DEFAULT})")
    p.add_argument("--n-top",       type=int,  default=N_TOP_DEFAULT,  metavar="N",
                   help=f"Number of top signs to include (default: {N_TOP_DEFAULT})")
    p.add_argument("--linearity-threshold", type=float, default=0.9, metavar="T",
                   help="Min linearity fraction to run BV (default: 0.9)")
    p.add_argument("--encoding", choices=["ic_rank", "barthel_bits"],
                   default="ic_rank",
                   help="Sign→index encoding.  'ic_rank' (default) indexes by "
                        "IC rank — with a median threshold the resulting f is "
                        "affine BY CONSTRUCTION (kept as default for "
                        "reproducibility of earlier runs and CI golden files). "
                        "'barthel_bits' indexes by the low bits of the Barthel "
                        "catalogue number, making linearity a falsifiable "
                        "property of the corpus.")
    p.add_argument("--backend",     choices=["statevector", "ibmq"], default="statevector")
    p.add_argument("--ibmq-token",  default=None,                    metavar="TOKEN")
    p.add_argument("--ibmq-instance", default=None,                   metavar="INST",
                   help="IBM Quantum instance CRN (default: read from IBMQ_INSTANCE env var)")
    p.add_argument("--ibmq-backend",  default=None,                  metavar="NAME")
    p.add_argument("--draw", action="store_true",
                   help="Save BV circuit PNG to outputs/quantum/")
    return p.parse_args()


def main() -> dict:
    import os
    args   = _parse_args()
    token  = args.ibmq_token or os.environ.get("IBMQ_TOKEN")
    result = run_analysis(
        corpus_dir=args.corpus_dir,
        n_top=args.n_top,
        n_bits=args.n_bits,
        backend=args.backend,
        ibmq_token=token,
        ibmq_instance=args.ibmq_instance,
        ibmq_backend_name=args.ibmq_backend,
        linearity_threshold=args.linearity_threshold,
        draw=args.draw,
        encoding=args.encoding,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    log.info("Results written to %s", args.output)
    if result.get("hardware_provenance"):
        from hackingrongo.quantum_provenance import write_versioned_result
        write_versioned_result(result, "bv", f"n{result.get('n_bits', '?')}_ic_analysis")
    return result


if __name__ == "__main__":
    main()
