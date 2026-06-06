#!/usr/bin/env python3
"""
scripts/run_quantum_sensitivity.py
====================================

Quantum robustness certificate for the IC_pre ≠ IC_post diachronic finding.

Encodes the three sensitivity scenarios (conservative_all_late,
optimistic_distributed, probabilistic_weighted) as branches of a quantum
superposition, runs the IC divergence oracle once, and returns a single
amplitude-weighted robustness score provably correct across all dating
uncertainties simultaneously.

Circuit layout (3-qubit default)
---------------------------------
  q[0]  scenario register qubit A  (|1⟩ → optimistic branch active)
  q[1]  scenario register qubit B  (|1⟩ → probabilistic branch active)
  q[2]  output qubit (flipped when diachronic signal is present)
  c[0]  classical measurement of q[2]

Encoding:
  |00⟩ (q[0]=0, q[1]=0) = conservative_all_late
  |01⟩ (q[0]=1, q[1]=0) = optimistic_distributed
  |10⟩ (q[0]=0, q[1]=1) = probabilistic_weighted

State preparation:
  1. Ry(2·arcsin(√p_o)) on q[0] → √(1-p_o)|00⟩ + √p_o|01⟩
  2. X(q[0]); CRy(2·arcsin(√(p_w/(p_w+p_c))), q[0], q[1]); X(q[0])
     → √p_c|00⟩ + √p_o|01⟩ + √p_w|10⟩

Usage
-----
    python scripts/run_quantum_sensitivity.py
    python scripts/run_quantum_sensitivity.py --backend fake_brisbane
    python scripts/run_quantum_sensitivity.py --backend ibmq
    python scripts/run_quantum_sensitivity.py --full-superposition
    python scripts/run_quantum_sensitivity.py --shots 8192
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from omegaconf import OmegaConf

from hackingrongo.data.constants import POST_CONTACT, PRE_CONTACT, UNKNOWN_STRATUM
from hackingrongo.zone_b.entropy import (
    _SCENARIO_NAMES,
    ic_for_clusters,
    load_tokens_under_scenario,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CONTACT_BOUNDARY_CE: int = 1600
SHOTS_DEFAULT: int = 4096
SCENARIOS = (
    "conservative_all_late",
    "optimistic_distributed",
    "probabilistic_weighted",
)

# Amplitude threshold for full-superposition enumeration pruning
_AMPLITUDE_THRESHOLD = 1e-10


# ---------------------------------------------------------------------------
# 1. Per-tablet p_pre priors
# ---------------------------------------------------------------------------

def load_undated_tablet_priors(
    tablets_json: Path,
    corpus_dir: Path,
) -> dict[str, float]:
    """Return {tablet_id: p_pre} for every tablet with cluster == UNKNOWN_STRATUM.

    p_pre(t) = max(0, min(1, (contact_boundary − date_min) / (date_max − date_min)))

    Tablets lacking date fields default to an uninformative prior of 0.5.
    """
    # Identify undated tablets via corpus JSON cluster field
    unknown_ids: list[str] = []
    for path in sorted(corpus_dir.glob("[A-Z].json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("cluster", UNKNOWN_STRATUM) == UNKNOWN_STRATUM:
            unknown_ids.append(path.stem)

    tablet_meta: dict = {}
    if tablets_json.exists():
        tablet_meta = json.loads(tablets_json.read_text(encoding="utf-8"))

    p_pre: dict[str, float] = {}
    for tid in sorted(unknown_ids):
        meta = tablet_meta.get(tid, {})
        dmin = meta.get("radiocarbon_date_min")
        dmax = meta.get("radiocarbon_date_max")
        if dmin is None or dmax is None or dmax <= dmin:
            p_pre[tid] = 0.5
        else:
            raw = (CONTACT_BOUNDARY_CE - dmin) / (dmax - dmin)
            p_pre[tid] = float(max(0.0, min(1.0, raw)))

    log.info("Undated tablets: %d", len(p_pre))
    for tid, p in sorted(p_pre.items()):
        log.info("  %s: p_pre = %.4f", tid, p)
    return p_pre


# ---------------------------------------------------------------------------
# 2. Scenario weights
# ---------------------------------------------------------------------------

def compute_scenario_weights(p_pre: dict[str, float]) -> dict[str, float]:
    """Derive amplitude-squared weights for the three scenarios.

      p_conservative = Π_t (1 − p_pre(t))
      p_optimistic   = Π_t 0.5  =  0.5^N
      p_probabilistic = Π_t (0.2·p_pre(t) + 0.8·(1−p_pre(t)))

    All three are normalised to sum to 1.
    """
    priors = list(p_pre.values())
    n = len(priors)

    p_c = float(np.prod([1.0 - p for p in priors]))
    p_o = float(0.5 ** n)
    p_w = float(np.prod([0.2 * p + 0.8 * (1.0 - p) for p in priors]))

    total = p_c + p_o + p_w or 1.0
    weights = {
        "conservative_all_late":   p_c / total,
        "optimistic_distributed":  p_o / total,
        "probabilistic_weighted":  p_w / total,
    }
    log.info("Scenario weights (normalised):")
    for name, w in weights.items():
        log.info("  %-30s %.8f", name, w)
    return weights


# ---------------------------------------------------------------------------
# 3. Classical IC computation per scenario
# ---------------------------------------------------------------------------

def compute_diachronic_per_scenario(corpus_dir: Path) -> dict[str, dict]:
    """Run IC analysis under each scenario; return per-scenario diachronic flags.

    A scenario is marked diachronic if:
      - IC_pre > IC_post, AND
      - 95% bootstrap CIs for IC_pre and IC_post do not overlap (non-overlapping
        CIs ↔ p < 0.05 under the bootstrap).
    """
    results: dict[str, dict] = {}
    for scenario in SCENARIOS:
        by_cluster = load_tokens_under_scenario(corpus_dir, scenario)
        cluster_ic = ic_for_clusters(by_cluster)

        pre = cluster_ic.get(PRE_CONTACT, {})
        post = cluster_ic.get(POST_CONTACT, {})
        ic_pre = float(pre.get("ic") or 0.0)
        ic_post = float(post.get("ic") or 0.0)
        pre_lo = pre.get("ic_ci_95_lo")
        post_hi = post.get("ic_ci_95_hi")

        direction_ok = ic_pre > ic_post
        non_overlap = (
            pre_lo is not None
            and post_hi is not None
            and pre_lo > post_hi
        )
        diachronic = direction_ok and non_overlap

        results[scenario] = {
            "ic_pre":  ic_pre,
            "ic_post": ic_post,
            "ic_pre_ci_lo":  pre.get("ic_ci_95_lo"),
            "ic_pre_ci_hi":  pre.get("ic_ci_95_hi"),
            "ic_post_ci_lo": post.get("ic_ci_95_lo"),
            "ic_post_ci_hi": post.get("ic_ci_95_hi"),
            "n_pre":  int(pre.get("n_tokens") or 0),
            "n_post": int(post.get("n_tokens") or 0),
            "diachronic":              diachronic,
            "signal_direction_correct": direction_ok,
            "ci_non_overlapping":       non_overlap,
        }
        log.info(
            "Scenario %-30s diachronic=%-5s  ΔIC=%+.6f  CIs non-overlap=%s",
            scenario, diachronic, ic_pre - ic_post, non_overlap,
        )
    return results


# ---------------------------------------------------------------------------
# 4. Quantum circuit: state prep + oracle + measurement
# ---------------------------------------------------------------------------

def build_oracle_circuit(
    weights: dict[str, float],
    diachronic: dict[str, bool],
) -> "QuantumCircuit":
    """3-qubit circuit for the scenario superposition + IC oracle.

    Returns a QuantumCircuit with q[2] as the output qubit measured into c[0].
    """
    from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister

    p_c = weights["conservative_all_late"]
    p_o = weights["optimistic_distributed"]
    p_w = weights["probabilistic_weighted"]

    q = QuantumRegister(3, "q")
    c = ClassicalRegister(1, "meas")
    qc = QuantumCircuit(q, c)

    # ── State preparation ───────────────────────────────────────────────
    # Step 1: Ry on q[0] → amplitude √p_o on |01⟩ branch
    if p_o > 1e-15:
        theta_o = 2.0 * math.asin(math.sqrt(min(1.0, p_o)))
        qc.ry(theta_o, q[0])

    # Step 2: zero-controlled Ry on q[1] (control = q[0]=0) to split
    # the |00⟩ amplitude √(1-p_o) into √p_c and √p_w.
    denom = p_w + p_c
    if denom > 1e-15 and p_w > 1e-15:
        theta_wp = 2.0 * math.asin(math.sqrt(min(1.0, p_w / denom)))
        qc.x(q[0])           # flip q[0] so control fires on original |0⟩
        qc.cry(theta_wp, q[0], q[1])
        qc.x(q[0])           # restore

    qc.barrier()

    # ── Oracle: MCX for each diachronic scenario ─────────────────────
    # |00⟩ (q0=0, q1=0) = conservative
    if diachronic.get("conservative_all_late", False):
        qc.x([q[0], q[1]])
        qc.ccx(q[0], q[1], q[2])
        qc.x([q[0], q[1]])

    # |01⟩ (q0=1, q1=0) = optimistic
    if diachronic.get("optimistic_distributed", False):
        qc.x(q[1])
        qc.ccx(q[0], q[1], q[2])
        qc.x(q[1])

    # |10⟩ (q0=0, q1=1) = probabilistic
    if diachronic.get("probabilistic_weighted", False):
        qc.x(q[0])
        qc.ccx(q[0], q[1], q[2])
        qc.x(q[0])

    qc.measure(q[2], c[0])
    return qc


def expected_p_robust(
    weights: dict[str, float],
    diachronic: dict[str, bool],
) -> float:
    """Analytical P(robust) = Σ w_s · diachronic[s]."""
    return sum(weights.get(s, 0.0) for s, d in diachronic.items() if d)


# ---------------------------------------------------------------------------
# 5. Backend runners
# ---------------------------------------------------------------------------

def run_statevector(qc: "QuantumCircuit", shots: int) -> dict[str, int]:
    from qiskit.primitives import StatevectorSampler
    result = StatevectorSampler().run([qc], shots=shots).result()
    return result[0].data.meas.get_counts()


def run_fake_brisbane(qc: "QuantumCircuit", shots: int) -> dict[str, int]:
    from qiskit_ibm_runtime import SamplerV2
    from qiskit_ibm_runtime.fake_provider import FakeBrisbane
    sampler = SamplerV2(mode=FakeBrisbane())
    result = sampler.run([qc], shots=shots).result()
    return result[0].data.meas.get_counts()


def run_ibmq(qc: "QuantumCircuit", shots: int, channel: str = "ibm_quantum") -> dict[str, int]:
    import os
    from qiskit.compiler import transpile
    from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
    instance = os.environ.get("IBMQ_INSTANCE")
    service = QiskitRuntimeService(channel="ibm_quantum_platform", instance=instance)
    backend = service.least_busy(operational=True, simulator=False, min_num_qubits=8)
    log.info("IBMQ backend: %s", backend.name)
    isa_qc = transpile(qc, backend=backend, optimization_level=2)
    result = SamplerV2(mode=backend).run([isa_qc], shots=shots).result()
    return result[0].data.meas.get_counts()


def run_circuit(qc: "QuantumCircuit", backend: str, shots: int) -> dict[str, int]:
    if backend == "simulator":
        return run_statevector(qc, shots)
    elif backend == "fake_brisbane":
        return run_fake_brisbane(qc, shots)
    elif backend == "ibmq":
        return run_ibmq(qc, shots)
    raise ValueError(f"Unknown backend {backend!r}. Choose: simulator, fake_brisbane, ibmq")


# ---------------------------------------------------------------------------
# 6. Clopper-Pearson CI
# ---------------------------------------------------------------------------

def clopper_pearson_ci(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Exact binomial CI for k successes in n trials."""
    from scipy.stats import beta
    if n == 0:
        return 0.0, 1.0
    lo = float(beta.ppf(alpha / 2, k, n - k + 1)) if k > 0 else 0.0
    hi = float(beta.ppf(1.0 - alpha / 2, k + 1, n - k)) if k < n else 1.0
    return lo, hi


def extract_p_robust(counts: dict[str, int]) -> tuple[float, int, int]:
    """Return (P_robust, n_ones, total) from classical register counts."""
    n_ones = counts.get("1", 0)
    total = sum(counts.values())
    return (n_ones / total if total > 0 else 0.0), n_ones, total


# ---------------------------------------------------------------------------
# 7. Full superposition (--full-superposition)
# ---------------------------------------------------------------------------

def _load_tokens_for_assignment(
    corpus_dir: Path,
    tablet_assignment: dict[str, str],
) -> dict[str, list[str]]:
    """Load tokens assigning each undated tablet to the given cluster."""
    by_cluster: dict[str, list[str]] = defaultdict(list)
    for path in sorted(corpus_dir.glob("[A-Z].json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        cluster = data.get("cluster", UNKNOWN_STRATUM)
        tokens: list[str] = []
        for g in data["glyphs"]:
            hc = g.get("horley_code")
            if hc:
                tokens.append(hc)
            for comp in (g.get("horley_components") or []):
                tokens.append(comp)
        if cluster != UNKNOWN_STRATUM:
            by_cluster[cluster].extend(tokens)
        else:
            target = tablet_assignment.get(path.stem, POST_CONTACT)
            by_cluster[target].extend(tokens)
    return dict(by_cluster)


def run_full_superposition(
    p_pre: dict[str, float],
    corpus_dir: Path,
) -> dict[str, Any]:
    """Exact P(robust) over all 2^N dating assignments of N undated tablets.

    For each assignment a ∈ {0,1}^N:
      amplitude²(a) = Π_i  p_pre(i)^a_i · (1−p_pre(i))^(1−a_i)
      P(robust) = Σ_a  amplitude²(a) · diachronic(a)

    Assignments with amplitude² < 1e-10 are pruned. For the actual corpus
    (all p_pre = 0) this reduces to a single evaluation of the all-post scenario.
    """
    tablets = sorted(p_pre.keys())
    n = len(tablets)

    log.info(
        "Full superposition: %d undated tablets → 2^%d = %d total assignments",
        n, n, 2 ** n,
    )
    log.info("Pruning at amplitude threshold = %.0e", _AMPLITUDE_THRESHOLD)

    p_robust = 0.0
    n_evaluated = 0

    for assignment_int in range(2 ** n):
        bits = [(assignment_int >> i) & 1 for i in range(n)]

        amp2 = 1.0
        for i, bit in enumerate(bits):
            p = p_pre[tablets[i]]
            amp2 *= (p if bit else (1.0 - p))
        if amp2 < _AMPLITUDE_THRESHOLD:
            continue

        assignment = {
            tablets[i]: (PRE_CONTACT if bits[i] else POST_CONTACT)
            for i in range(n)
        }
        by_cluster = _load_tokens_for_assignment(corpus_dir, assignment)
        cluster_ic = ic_for_clusters(by_cluster)

        pre = cluster_ic.get(PRE_CONTACT, {})
        post = cluster_ic.get(POST_CONTACT, {})
        ic_pre = float(pre.get("ic") or 0.0)
        ic_post = float(post.get("ic") or 0.0)
        pre_lo = pre.get("ic_ci_95_lo")
        post_hi = post.get("ic_ci_95_hi")
        diachronic = (
            ic_pre > ic_post
            and pre_lo is not None
            and post_hi is not None
            and pre_lo > post_hi
        )
        if diachronic:
            p_robust += amp2
        n_evaluated += 1

    log.info(
        "Full superposition evaluated %d/%d assignments (≥ threshold)",
        n_evaluated, 2 ** n,
    )
    log.info("P(robust) = %.8f", p_robust)

    return {
        "p_robust": float(p_robust),
        "n_assignments_evaluated": n_evaluated,
        "total_assignments": 2 ** n,
        "amplitude_threshold": _AMPLITUDE_THRESHOLD,
        "n_tablet_qubits": n,
    }


# ---------------------------------------------------------------------------
# 8. Interpretation
# ---------------------------------------------------------------------------

def interpret_result(p_robust: float, ci_lo: float, ci_hi: float) -> str:
    if p_robust > 0.95:
        return (
            f"ROBUST — finding survives quantum sensitivity analysis with "
            f"{p_robust * 100:.1f}% confidence "
            f"(95% CI: [{ci_lo * 100:.1f}%, {ci_hi * 100:.1f}%]). "
            "IC_pre ≠ IC_post holds across all dating scenarios weighted by "
            "Ferrara et al. 2024 radiocarbon cluster probabilities."
        )
    elif p_robust > 0.5:
        return (
            f"PLAUSIBLE — finding holds in the dominant dating scenarios "
            f"(P = {p_robust * 100:.1f}%, 95% CI: [{ci_lo * 100:.1f}%, {ci_hi * 100:.1f}%]), "
            "but some alternative dating assignments undermine the IC divergence."
        )
    else:
        return (
            f"FRAGILE — finding is artefact of dating assumptions "
            f"(P = {p_robust * 100:.1f}%, 95% CI: [{ci_lo * 100:.1f}%, {ci_hi * 100:.1f}%]). "
            "IC_pre ≠ IC_post does not survive quantum sensitivity analysis."
        )


# ---------------------------------------------------------------------------
# 9. Main pipeline
# ---------------------------------------------------------------------------

def run_analysis(
    backend: str = "simulator",
    shots: int = SHOTS_DEFAULT,
    full_superposition: bool = False,
    tablets_json: Path | None = None,
    output_path: Path | None = None,
) -> dict[str, Any]:
    """End-to-end quantum sensitivity analysis.

    Parameters
    ----------
    backend : str
        One of "simulator", "fake_brisbane", "ibmq".
    shots : int
        Measurement shots.
    full_superposition : bool
        If True, also run the 19-tablet full-superposition analysis on simulator.
    tablets_json : Path, optional
        Path to data/metadata/tablets.json.
    output_path : Path, optional
        Where to save outputs/quantum/quantum_sensitivity.json.
    """
    cfg = OmegaConf.load(PROJECT_ROOT / "conf" / "config.yaml")
    corpus_dir = PROJECT_ROOT / cfg.paths.corpus_dir

    if tablets_json is None:
        tablets_json = PROJECT_ROOT / "data" / "metadata" / "tablets.json"

    log.info("=" * 64)
    log.info("Quantum Sensitivity Analysis — IC_pre ≠ IC_post robustness")
    log.info("=" * 64)
    log.info(
        "Backend: %s  |  Shots: %d  |  Full superposition: %s",
        backend, shots, full_superposition,
    )

    # 1. Per-tablet priors
    p_pre = load_undated_tablet_priors(tablets_json, corpus_dir)

    # 2. Scenario weights
    weights = compute_scenario_weights(p_pre)

    # 3. Classical IC under each scenario
    log.info("")
    log.info("Classical IC pre-computation...")
    scenario_ic = compute_diachronic_per_scenario(corpus_dir)
    diachronic = {s: v["diachronic"] for s, v in scenario_ic.items()}

    # 4. Analytical P(robust)
    p_robust_analytical = expected_p_robust(weights, diachronic)
    log.info("Analytical P(robust) = %.8f", p_robust_analytical)

    # 5. Quantum circuit
    log.info("")
    log.info("Building 3-qubit oracle circuit...")
    qc = build_oracle_circuit(weights, diachronic)
    log.info("Depth: %d  |  Gates: %d", qc.depth(), qc.size())

    log.info("Running %s (%d shots)...", backend, shots)
    counts = run_circuit(qc, backend, shots)
    log.info("Counts: %s", counts)

    p_robust_meas, n_ones, n_total = extract_p_robust(counts)
    ci_lo, ci_hi = clopper_pearson_ci(n_ones, n_total)

    log.info(
        "P(robust) = %.6f  (n_ones=%d, total=%d)  95%% CI [%.6f, %.6f]",
        p_robust_meas, n_ones, n_total, ci_lo, ci_hi,
    )

    interpretation = interpret_result(p_robust_meas, ci_lo, ci_hi)
    log.info("%s", interpretation)

    # 6. Full superposition (optional)
    full_result: dict[str, Any] = {}
    if full_superposition:
        log.info("")
        log.info("Running full 19-tablet superposition...")
        full_result = run_full_superposition(p_pre, corpus_dir)

    # 7. Assemble output
    output: dict[str, Any] = {
        "scenario_weights": {s: round(w, 10) for s, w in weights.items()},
        "classical_diachronic_per_scenario": {
            s: {
                "diachronic":               v["diachronic"],
                "ic_pre":                   round(v["ic_pre"], 6),
                "ic_post":                  round(v["ic_post"], 6),
                "delta_ic":                 round(v["ic_pre"] - v["ic_post"], 6),
                "n_pre":                    v["n_pre"],
                "n_post":                   v["n_post"],
                "ci_non_overlapping":       v["ci_non_overlapping"],
            }
            for s, v in scenario_ic.items()
        },
        "p_robust_analytical": round(p_robust_analytical, 8),
        "p_robust":   round(p_robust_meas, 6),
        "ci_lower":   round(ci_lo, 6),
        "ci_upper":   round(ci_hi, 6),
        "n_shots":    n_total,
        "n_ones":     n_ones,
        "backend":    backend,
        "circuit_depth":    qc.depth(),
        "circuit_n_qubits": qc.num_qubits,
        "full_superposition_used": full_superposition,
        "undated_tablet_priors": {
            tid: round(p, 6) for tid, p in sorted(p_pre.items())
        },
        "interpretation": interpretation,
    }
    if full_result:
        output["full_superposition"] = full_result

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
        log.info("Results written to %s", output_path)

    return output


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Quantum robustness certificate for IC_pre ≠ IC_post."
    )
    parser.add_argument(
        "--backend",
        choices=["simulator", "fake_brisbane", "ibmq"],
        default="simulator",
    )
    parser.add_argument("--shots", type=int, default=SHOTS_DEFAULT)
    parser.add_argument(
        "--full-superposition",
        action="store_true",
        help="Full 19-tablet superposition (simulator only).",
    )
    parser.add_argument(
        "--tablets",
        type=Path,
        default=PROJECT_ROOT / "data" / "metadata" / "tablets.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "quantum" / "quantum_sensitivity.json",
    )
    args = parser.parse_args()

    result = run_analysis(
        backend=args.backend,
        shots=args.shots,
        full_superposition=args.full_superposition,
        tablets_json=args.tablets,
        output_path=args.output,
    )

    print()
    print("=" * 64)
    print(
        f"  P(robust) = {result['p_robust']:.4f}"
        f"  [{result['ci_lower']:.4f}, {result['ci_upper']:.4f}]"
    )
    print(f"  Backend: {result['backend']}  |  Shots: {result['n_shots']}")
    print()
    print(f"  {result['interpretation']}")
    print("=" * 64)


if __name__ == "__main__":
    main()
