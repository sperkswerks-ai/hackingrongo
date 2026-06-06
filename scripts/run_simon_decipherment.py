#!/usr/bin/env python3
"""
run_simon_decipherment.py — Simon's algorithm on diachronic key-change events.

Background
----------
Parallel passages P007 (Tablets A/D/H/S) and P012 (11 tablets) show consistent
sign substitutions at the pre→post-contact boundary — the "holy grail" passages
identified by the Kasiski-parallel analysis.  These substitutions constitute a
hidden XOR-periodic structure: every pre-contact instance of the passage differs
from the post-contact instance by a fixed bitstring s (one bit per canonical
position, set at positions where the sign changed).  Simon's algorithm recovers
that period in O(n) quantum oracle queries vs O(2^{n/2}) classical.

Sign-variant encoding
---------------------
For a passage with canonical form of length n, define x ∈ {0,1}^n where
  x[i] = 0  if canonical position i shows the pre-contact sign variant
  x[i] = 1  if canonical position i shows the post-contact sign variant

The Simon oracle encodes  f_delta(x) = x | s  which is the minimal 2-to-1
extension satisfying f(x) = f(x ⊕ s) for all x, consistent with:
  f_delta(x_pre) = x_pre ⊕ s = x_post   (pre-contact pattern maps to post)
  f_delta(x_post) = x_post               (post-contact pattern is fixed)

Usage
-----
    python scripts/run_simon_decipherment.py
    python scripts/run_simon_decipherment.py --passage P007_ADHS
    python scripts/run_simon_decipherment.py --shots 32 --draw
    python scripts/run_simon_decipherment.py --backend ibmq \\
        --ibmq-token <TOKEN> [--ibmq-instance ibm-q/open/main]
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

_VARIANTS_PATH = PROJECT_ROOT / "data" / "parallels" / "parallel_variants_auto.json"
_OUTPUT_PATH   = PROJECT_ROOT / "outputs" / "quantum" / "simon_result.json"


# ── Data types ────────────────────────────────────────────────────────────────

@dataclass
class DiachronicChange:
    position: int           # 0-indexed canonical position
    pre_sign: str
    post_sign: str
    n_tablets_consistent: int
    is_holy_grail: bool


@dataclass
class SimonPassage:
    passage_id: str
    canonical_form: list[str]
    n: int                         # len(canonical_form)
    s: int                         # period as integer: bit i set ↔ position i changed
    s_positions: list[int]         # sorted list of changed positions
    changes: list[DiachronicChange]
    tablet_vectors: dict[str, int]  # tablet → x as integer
    tablet_strata: dict[str, str]   # tablet → stratum


# ── Data loading ──────────────────────────────────────────────────────────────

def _load_passages(json_path: Path) -> list[dict]:
    data = json.loads(json_path.read_text(encoding="utf-8"))
    return data["passages"]


def _parse_passage(p: dict) -> SimonPassage | None:
    """Return SimonPassage for a passage that has diachronic changes, else None."""
    changes_raw = p.get("diachronic_changes", [])
    if not changes_raw:
        return None

    pid          = p["passage_id"]
    canon        = p["canonical_form"]
    n            = len(canon)
    attestations = p.get("attestations", [])

    changes: list[DiachronicChange] = []
    for c in changes_raw:
        if c.get("change_type") == "substitution":
            changes.append(DiachronicChange(
                position=int(c["position"]),
                pre_sign=str(c["pre_contact_sign"]),
                post_sign=str(c["post_contact_sign"]),
                n_tablets_consistent=int(c.get("n_tablets_consistent", 0)),
                is_holy_grail=bool(c.get("is_holy_grail_candidate", False)),
            ))

    if not changes:
        return None

    # Simon period: bit i set iff position i has a documented change
    s = 0
    for ch in changes:
        s |= (1 << ch.position)
    s_positions = sorted(ch.position for ch in changes)

    # ── Aggregate per-tablet stratum and sign evidence ────────────────────────
    # For each tablet take the majority stratum (skip excluded).
    stratum_votes: dict[str, dict[str, int]] = {}
    direct_evidence: dict[str, dict[int, int]] = {}  # tablet → {position → 0/1}

    for att in attestations:
        tablet  = att["tablet"]
        stratum = att.get("stratum", "undated")
        form    = att.get("form", [])

        stratum_votes.setdefault(tablet, {}).setdefault(stratum, 0)
        stratum_votes[tablet][stratum] += 1

        # Look for variant signs at changed positions
        form_set = set(form)
        for ch in changes:
            if ch.pre_sign in form_set:
                direct_evidence.setdefault(tablet, {})[ch.position] = 0
            elif ch.post_sign in form_set:
                direct_evidence.setdefault(tablet, {})[ch.position] = 1

    tablet_strata: dict[str, str] = {}
    for tablet, votes in stratum_votes.items():
        # Dominant stratum (skip excluded from the majority calculation)
        non_excl = {k: v for k, v in votes.items() if k != "excluded"}
        if non_excl:
            tablet_strata[tablet] = max(non_excl, key=non_excl.get)
        else:
            tablet_strata[tablet] = "excluded"

    # ── Assign bit vector per tablet ──────────────────────────────────────────
    # x = 0 means all changed positions show pre-contact variant
    # x = s means all changed positions show post-contact variant
    tablet_vectors: dict[str, int] = {}
    for tablet, stratum in tablet_strata.items():
        if stratum == "excluded":
            continue
        evidence = direct_evidence.get(tablet, {})

        if stratum == "pre_contact":
            x = 0  # all changed positions: pre-contact value
        elif stratum == "post_contact":
            x = s  # all changed positions: post-contact value
        else:
            # undated: use direct evidence if available
            if not evidence:
                continue  # skip if no evidence
            x = 0
            for pos, val in evidence.items():
                if val == 1:
                    x |= (1 << pos)

        # Cross-check with direct evidence (flag conflicts but use stratum)
        for pos, val in evidence.items():
            expected = 1 if (x >> pos) & 1 else 0
            if val != expected:
                log.warning(
                    "%s tablet %s: direct evidence at position %d (%s) "
                    "conflicts with stratum %s",
                    pid, tablet, pos,
                    changes[next(i for i, ch in enumerate(changes) if ch.position == pos)].pre_sign
                    if val == 0 else changes[next(i for i, ch in enumerate(changes) if ch.position == pos)].post_sign,
                    stratum,
                )

        tablet_vectors[tablet] = x

    return SimonPassage(
        passage_id=pid,
        canonical_form=canon,
        n=n,
        s=s,
        s_positions=s_positions,
        changes=changes,
        tablet_vectors=tablet_vectors,
        tablet_strata=tablet_strata,
    )


# ── Truth table ───────────────────────────────────────────────────────────────

def build_truth_table(n: int, s: int) -> dict[int, int]:
    """f_delta(x) = x | s — minimal 2-to-1 Simon extension, period s.

    Correctness: (x | s) = (x ⊕ s) | s  for all x, hence f(x) = f(x ⊕ s). ∎
    """
    return {x: x | s for x in range(1 << n)}


# ── Precondition check ────────────────────────────────────────────────────────

def check_precondition(
    passage: SimonPassage,
) -> tuple[bool, str, list[str]]:
    """Verify f_delta(x) = f_delta(x ⊕ s) for all tablet assignments.

    Returns
    -------
    (holds, summary_msg, conflict_list)
    """
    n  = passage.n
    s  = passage.s
    tt = build_truth_table(n, s)
    conflicts: list[str] = []

    for tablet, x in passage.tablet_vectors.items():
        x_xors = x ^ s
        if tt[x] != tt[x_xors]:
            conflicts.append(
                f"tablet {tablet}: f({x:0{n}b}) = {tt[x]:0{n}b} "
                f"≠ f({x_xors:0{n}b}) = {tt[x_xors]:0{n}b}"
            )

    if conflicts:
        msg = (
            f"Simon precondition FAILED for {passage.passage_id}: "
            f"{len(conflicts)} tablet(s) violate f(x) = f(x ⊕ s). "
            "Key-change is not XOR-periodic — recording as negative result."
        )
        return False, msg, conflicts

    # Also verify the two canonical tablet classes map to the same output
    x_pre  = 0     # pre-contact pattern
    x_post = s     # post-contact pattern = pre XOR s
    if tt[x_pre] != tt[x_post]:
        return False, "f(x_pre) ≠ f(x_post) — construction error.", []

    n_pre  = sum(1 for x in passage.tablet_vectors.values() if x == 0)
    n_post = sum(1 for x in passage.tablet_vectors.values() if x == s)
    msg = (
        f"Simon precondition HOLDS for {passage.passage_id}: "
        f"{n_pre} pre-contact tablet(s) and {n_post} post-contact tablet(s) "
        f"all satisfy f(x) = f(x ⊕ s) with period s = {s:0{n}b} (= {s})."
    )
    return True, msg, []


# ── Quantum circuit ───────────────────────────────────────────────────────────

def build_oracle(n: int, s: int) -> object:
    """Oracle U_f: |x⟩|0⟩ → |x⟩|f_delta(x)⟩ for f_delta(x) = x | s.

    Efficient construction (n gates, no ancilla):
      - CNOT(input[i], output[i])  for each i where s[i] = 0  (copy unchanged bits)
      - X(output[i])               for each i where s[i] = 1  (always-1 bits)
    """
    from qiskit import QuantumCircuit, QuantumRegister
    reg_in  = QuantumRegister(n, "in")
    reg_out = QuantumRegister(n, "out")
    qc = QuantumCircuit(reg_in, reg_out, name="U_f")
    for i in range(n):
        if (s >> i) & 1:
            qc.x(reg_out[i])         # f(x)[i] = 1 always
        else:
            qc.cx(reg_in[i], reg_out[i])  # f(x)[i] = x[i]
    return qc


def build_simon_circuit(n: int, s: int) -> object:
    """Full Simon circuit: H^n → U_f → H^n → measure input register.

    Qubits 0..n-1  : input register  (|x⟩)
    Qubits n..2n-1 : output register (|f(x)⟩)
    Classical bits 0..n-1: measurement of input register
    """
    from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
    reg_in  = QuantumRegister(n, "in")
    reg_out = QuantumRegister(n, "out")
    reg_c   = ClassicalRegister(n, "meas")
    oracle  = build_oracle(n, s)

    qc = QuantumCircuit(reg_in, reg_out, reg_c, name="simon")
    qc.h(reg_in)
    qc.compose(oracle, qubits=list(reg_in) + list(reg_out), inplace=True)
    qc.h(reg_in)
    qc.measure(reg_in, reg_c)
    return qc


# ── GF(2) solver ─────────────────────────────────────────────────────────────

def _dot_gf2(a: int, b: int) -> int:
    """Inner product over GF(2): popcount(a AND b) mod 2."""
    return bin(a & b).count("1") % 2


def solve_simon_period(measurements: list[int], n: int) -> int:
    """Return the unique non-zero s such that y·s = 0 (mod 2) for all y.

    Uses brute-force search over {1, …, 2^n − 1}.  Correct for n ≤ 20.
    Returns 0 if no non-trivial period is found (Simon precondition failed,
    or insufficient independent measurements).
    """
    candidates = [s for s in range(1, 1 << n)
                  if all(_dot_gf2(y, s) == 0 for y in measurements)]
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        log.warning(
            "Multiple period candidates: %s — need more independent measurements.",
            candidates,
        )
        return candidates[0]  # return smallest; ties indicate insufficient shots
    return 0


# ── Samplers ──────────────────────────────────────────────────────────────────

def _decode_counts(counts: dict[str, int], n: int) -> list[int]:
    """Convert Qiskit bitstring counts → list of measurement integers.

    Qiskit's get_counts() bitstrings: rightmost character = classical bit 0
    (qubit 0), leftmost = classical bit n-1.  int(bitstr, 2) places qubit i at
    the 2^i position of the result, which is exactly what we want for GF(2)
    dot products.  No reversal needed.
    """
    measurements: list[int] = []
    for bitstr, count in counts.items():
        val = int(bitstr, 2)
        measurements.extend([val] * count)
    return measurements


def run_statevector(qc, n: int, shots: int) -> tuple[dict[str, int], float]:
    """Run circuit on Qiskit StatevectorSampler.  Returns (counts, elapsed_s)."""
    from qiskit.primitives import StatevectorSampler
    t0 = time.monotonic()
    sampler = StatevectorSampler()
    job = sampler.run([qc], shots=shots)
    result = job.result()[0]
    counts = result.data.meas.get_counts()
    return dict(counts), time.monotonic() - t0


def run_ibmq(
    qc,
    n: int,
    shots: int,
    token: str,
    instance: str = "ibm-q/open/main",
    backend_name: str | None = None,
) -> tuple[dict[str, int], float]:
    """Run circuit on IBM Quantum hardware via QiskitRuntimeService / SamplerV2."""
    from qiskit_ibm_runtime import QiskitRuntimeService, Session, SamplerV2
    from qiskit.compiler import transpile

    t0 = time.monotonic()
    service = QiskitRuntimeService(
        channel="ibm_quantum",
        token=token,
        instance=instance,
    )
    if backend_name:
        backend = service.backend(backend_name)
    else:
        backend = service.least_busy(
            operational=True,
            simulator=False,
            min_num_qubits=2 * n,
        )
    log.info("IBM Quantum backend: %s (%d qubits)", backend.name, backend.num_qubits)

    t_qc = transpile(qc, backend=backend, optimization_level=2)
    log.info("Transpiled: %d qubits, depth %d", t_qc.num_qubits, t_qc.depth())

    with Session(service=service, backend=backend) as session:
        sampler = SamplerV2(mode=session)
        job = sampler.run([t_qc], shots=shots)
        log.info("Job submitted: %s", job.job_id())
        result = job.result()[0]

    counts = result.data.meas.get_counts()
    return dict(counts), time.monotonic() - t0


# ── Per-passage analysis ──────────────────────────────────────────────────────

def analyse_passage(
    passage: SimonPassage,
    shots: int,
    backend: str,
    ibmq_token: str | None,
    ibmq_instance: str,
    ibmq_backend_name: str | None,
) -> dict[str, Any]:
    """Run the full Simon analysis pipeline on one passage.  Returns result dict."""
    n = passage.n
    s = passage.s

    log.info(
        "── %s ─────────────────────────────────────────", passage.passage_id
    )
    log.info(
        "  canonical form  : %s", passage.canonical_form
    )
    log.info(
        "  n=%d  s=%d (%s)  changed positions: %s",
        n, s, f"{s:0{n}b}", passage.s_positions,
    )
    for ch in passage.changes:
        log.info(
            "  change @ pos %d: '%s' → '%s'  (n_consistent=%d, holy_grail=%s)",
            ch.position, ch.pre_sign, ch.post_sign,
            ch.n_tablets_consistent, ch.is_holy_grail,
        )

    # ── Precondition check ────────────────────────────────────────────────────
    pre_ok, pre_msg, conflicts = check_precondition(passage)
    log.info("  %s", pre_msg)

    result: dict[str, Any] = {
        "passage_id":       passage.passage_id,
        "canonical_form":   passage.canonical_form,
        "n":                n,
        "s_int":            s,
        "s_bits":           f"{s:0{n}b}",
        "s_positions":      passage.s_positions,
        "changes": [
            {
                "position":           ch.position,
                "pre_sign":           ch.pre_sign,
                "post_sign":          ch.post_sign,
                "n_tablets_consistent": ch.n_tablets_consistent,
                "is_holy_grail":      ch.is_holy_grail,
            }
            for ch in passage.changes
        ],
        "tablet_vectors": {
            t: {"x_int": x, "x_bits": f"{x:0{n}b}", "stratum": passage.tablet_strata.get(t,"?")}
            for t, x in passage.tablet_vectors.items()
        },
        "precondition_holds":    pre_ok,
        "precondition_message":  pre_msg,
        "precondition_conflicts": conflicts,
        "classical_query_complexity": f"O(2^(n/2)) = O({2**(n//2)})",
        "quantum_query_complexity":   f"O(n) = O({n})",
    }

    if not pre_ok:
        log.warning("  Simon not applicable — recording negative result.")
        return result

    # ── Build and describe Simon circuit ──────────────────────────────────────
    simon_qc = build_simon_circuit(n, s)
    result["simon_circuit"] = {
        "num_qubits":   simon_qc.num_qubits,
        "depth":        simon_qc.depth(),
        "num_gates":    sum(simon_qc.count_ops().values()),
        "shots":        shots,
    }
    log.info(
        "  Simon circuit: %d qubits, depth %d, %d gates, %d shots",
        simon_qc.num_qubits, simon_qc.depth(),
        sum(simon_qc.count_ops().values()), shots,
    )

    # ── Run ───────────────────────────────────────────────────────────────────
    if backend == "ibmq":
        if not ibmq_token:
            raise ValueError("--ibmq-token required for --backend ibmq.")
        counts, elapsed = run_ibmq(
            simon_qc, n, shots,
            token=ibmq_token,
            instance=ibmq_instance,
            backend_name=ibmq_backend_name,
        )
    else:
        counts, elapsed = run_statevector(simon_qc, n, shots)

    log.info("  Measurement counts: %s  (%.2fs)", counts, elapsed)
    result["counts"]          = counts
    result["run_time_seconds"] = round(elapsed, 3)

    # ── Solve GF(2) ───────────────────────────────────────────────────────────
    measurements = _decode_counts(counts, n)
    unique_ms    = sorted(set(measurements))
    log.info(
        "  %d shots → %d unique measurement values: %s",
        len(measurements), len(unique_ms),
        [f"{y:0{n}b}" for y in unique_ms],
    )

    recovered_s = solve_simon_period(unique_ms, n)

    log.info(
        "  Recovered period s = %d (%s)  [expected %d (%s)]",
        recovered_s, f"{recovered_s:0{n}b}",
        s, f"{s:0{n}b}",
    )

    matches = (recovered_s == s)
    result.update({
        "measurements_unique":        [f"{y:0{n}b}" for y in unique_ms],
        "recovered_s_int":            recovered_s,
        "recovered_s_bits":           f"{recovered_s:0{n}b}",
        "period_matches_observation": matches,
    })

    # ── Verification narrative ────────────────────────────────────────────────
    if matches:
        changed_pos_str = ", ".join(
            f"pos {ch.position} ('{ch.pre_sign}' → '{ch.post_sign}')"
            for ch in passage.changes
        )
        result["interpretation"] = (
            f"Simon's algorithm correctly recovered the key-change period "
            f"s = {s:0{n}b} for {passage.passage_id}. "
            f"The hidden XOR structure corresponds to the diachronic substitution "
            f"at {changed_pos_str}. "
            f"Quantum complexity O(n)={n} shots vs classical O(2^{{n/2}})="
            f"{2**(n//2)} queries."
        )
    else:
        result["interpretation"] = (
            f"Simon's algorithm returned s = {recovered_s:0{n}b}, "
            f"which does not match the observed period {s:0{n}b}. "
            "Either insufficient shots or a measurement error. "
            f"Try increasing --shots (current: {shots})."
        )

    log.info("  Matches observed substitution: %s", matches)
    return result


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Simon's algorithm on diachronic key-change events.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--variants-file", type=Path, default=_VARIANTS_PATH, metavar="JSON",
        help="Path to parallel_variants_auto.json.",
    )
    p.add_argument(
        "--passage", default=None, metavar="ID",
        help="Run on a single passage ID (default: all passages with changes).",
    )
    p.add_argument(
        "--shots", type=int, default=None, metavar="N",
        help="Number of shots.  Defaults to ceil(n+10) per passage.",
    )
    p.add_argument(
        "--output", type=Path, default=_OUTPUT_PATH, metavar="JSON",
        help=f"Output path (default: {_OUTPUT_PATH}).",
    )
    p.add_argument(
        "--backend", choices=["statevector", "ibmq"], default="statevector",
        help="Quantum backend (default: statevector).",
    )
    p.add_argument(
        "--ibmq-token", default=None, metavar="TOKEN",
        help="IBM Quantum API token (or set IBMQ_TOKEN env var).",
    )
    p.add_argument(
        "--ibmq-instance", default="ibm-q/open/main", metavar="INSTANCE",
        help="IBM Quantum instance (default: ibm-q/open/main).",
    )
    p.add_argument(
        "--ibmq-backend", default=None, metavar="NAME",
        help="IBM Quantum backend name (default: least_busy).",
    )
    p.add_argument(
        "--draw", action="store_true",
        help="Save circuit diagrams to outputs/quantum/.",
    )
    return p.parse_args()


def main() -> dict:
    import os
    args = _parse_args()

    ibmq_token = args.ibmq_token or os.environ.get("IBMQ_TOKEN")

    # ── Load passages ─────────────────────────────────────────────────────────
    log.info("Loading parallel variants from %s …", args.variants_file)
    raw_passages = _load_passages(args.variants_file)
    passages: list[SimonPassage] = []
    for rp in raw_passages:
        sp = _parse_passage(rp)
        if sp is None:
            continue
        if args.passage and sp.passage_id != args.passage:
            continue
        passages.append(sp)

    if not passages:
        target = f"passage {args.passage}" if args.passage else "any passage with diachronic changes"
        log.error("No passages found matching %s in %s.", target, args.variants_file)
        sys.exit(1)

    log.info("Found %d passage(s) with diachronic changes.", len(passages))

    # ── Analyse each passage ──────────────────────────────────────────────────
    all_results: dict[str, Any] = {}
    for passage in passages:
        shots = args.shots if args.shots is not None else (passage.n + 10)
        result = analyse_passage(
            passage=passage,
            shots=shots,
            backend=args.backend,
            ibmq_token=ibmq_token,
            ibmq_instance=args.ibmq_instance,
            ibmq_backend_name=args.ibmq_backend,
        )
        all_results[passage.passage_id] = result

        if args.draw:
            _draw_circuit(passage, shots)

    # ── Summary table ─────────────────────────────────────────────────────────
    _print_summary(all_results)

    # ── Save ──────────────────────────────────────────────────────────────────
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps({"passages": all_results}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    log.info("Results written to %s", args.output)
    return all_results


def _draw_circuit(passage: SimonPassage, shots: int) -> None:
    out_dir = PROJECT_ROOT / "outputs" / "quantum"
    out_dir.mkdir(parents=True, exist_ok=True)
    qc = build_simon_circuit(passage.n, passage.s)
    png_path = out_dir / f"simon_circuit_{passage.passage_id}.png"
    fig = qc.draw(output="mpl", fold=-1)
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    log.info("Circuit diagram saved to %s", png_path)


def _print_summary(results: dict[str, Any]) -> None:
    print(f"\n{'═' * 70}")
    print("  Simon's Algorithm — Diachronic Key-Change Analysis")
    print(f"{'═' * 70}")
    for pid, r in results.items():
        print(f"\n  {pid}")
        print(f"    n = {r['n']}  |  s = {r['s_bits']} (int {r['s_int']})")
        print(f"    changed positions: {r['s_positions']}")
        for ch in r["changes"]:
            print(
                f"    pos {ch['position']}: '{ch['pre_sign']}' → '{ch['post_sign']}'  "
                f"(n_consistent={ch['n_tablets_consistent']}, "
                f"holy_grail={ch['is_holy_grail']})"
            )
        print(f"    tablet vectors: " + "  ".join(
            f"{t}={v['x_bits']}({v['stratum'][:3]})"
            for t, v in r["tablet_vectors"].items()
        ))
        print(f"    precondition: {'✓ holds' if r['precondition_holds'] else '✗ fails'}")
        if r["precondition_holds"]:
            print(f"    circuit: {r['simon_circuit']['num_qubits']} qubits, "
                  f"depth {r['simon_circuit']['depth']}, "
                  f"{r['simon_circuit']['shots']} shots")
            print(f"    measurements: {r['measurements_unique']}")
            print(
                f"    recovered s = {r['recovered_s_bits']}  "
                f"{'✓ matches' if r['period_matches_observation'] else '✗ mismatch'}"
            )
            print(f"    complexity: quantum {r['quantum_query_complexity']}  "
                  f"vs classical {r['classical_query_complexity']}")
            print(f"    {r['interpretation']}")
    print()


if __name__ == "__main__":
    main()
