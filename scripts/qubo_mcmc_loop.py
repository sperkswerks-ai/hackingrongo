"""
qubo_mcmc_loop.py — Quantum-MCMC iterative refinement loop.

Alternates between two complementary solvers, using each to warm-start
the other:

    Round 1: QUBO (quantum/hybrid)  → best assignment
    Round 2: MCMC                   → refine neighbourhood of QUBO result
    Round 3: QUBO                   → re-anneal from MCMC best
    Round 4: MCMC                   → refine …
    …

Rationale
---------
Quantum annealing (or hybrid) excels at escaping local optima in the
combinatorial landscape; MCMC excels at fine-grained local refinement
guided by the full LM posterior.  Neither alone is optimal: annealing
finds good regions quickly but cannot exploit the LM gradient precisely;
MCMC exploits the LM precisely but gets stuck if initialised poorly.
The loop lets each solver feed its best result to the other as a
warm-start, concentrating wall-clock time where it is most useful.

Usage
-----
    python scripts/qubo_mcmc_loop.py \\
        --corpus-dir data/corpus \\
        --lm-dir     data/language_models \\
        --output     outputs/decipherment/loop_result.json \\
        --iterations 3 \\
        --solver hybrid \\
        --dwave-token $DWAVE_API_TOKEN

    # Smoke test (CPU only, very fast):
    python scripts/qubo_mcmc_loop.py --smoke-test
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run(cmd: list[str], label: str) -> int:
    """Run a subprocess, streaming output.  Returns exit code."""
    log.info("=== %s ===", label)
    log.info("CMD: %s", " ".join(cmd))
    proc = subprocess.Popen(
        cmd,
        cwd=PROJECT_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        print(line, end="")
    proc.wait()
    return proc.returncode


def _qubo_cmd(
    corpus_dir: Path,
    lm_dir: Path,
    output: Path,
    solver: str,
    num_reads: int,
    dwave_token: str | None,
    hybrid_time_limit: int,
    init_from: Path | None,
    smoke_test: bool,
    crib: str | None,
) -> list[str]:
    cmd = [
        sys.executable, "scripts/run_qubo_decipherment.py",
        "--corpus-dir", str(corpus_dir),
        "--lm-dir",     str(lm_dir),
        "--output",     str(output),
        "--solver",     solver,
        "--num-reads",  str(num_reads),
        "--hybrid-time-limit", str(hybrid_time_limit),
    ]
    if dwave_token:
        cmd += ["--dwave-token", dwave_token]
    if init_from and init_from.exists():
        cmd += ["--init-from", str(init_from)]
    if smoke_test:
        cmd.append("--smoke-test")
    if crib:
        cmd += ["--crib", crib]
    return cmd


def _mcmc_cmd(
    corpus_dir: Path,
    lm_dir: Path,
    output: Path,
    init_from: Path | None,
    num_iterations: int,
    smoke_test: bool,
    crib: str | None,
) -> list[str]:
    """Build the run_decipherment.py command for the MCMC leg."""
    cmd = [
        sys.executable, "scripts/run_decipherment.py",
        "hydra.job.chdir=false",
        f"paths.corpus_dir={corpus_dir}",
        f"paths.outputs_dir={output.parent}",
        f"zone_c.mcmc.num_iterations={num_iterations}",
    ]
    if init_from and init_from.exists():
        cmd += [f"++decipherment.init_from={init_from}"]
    if smoke_test:
        cmd += [
            "zone_c.mcmc.num_iterations=500",
            "zone_c.mcmc.burn_in=50",
        ]
    return cmd


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(
        description="Iterative QUBO↔MCMC refinement loop.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--corpus-dir",    type=Path, default=None)
    p.add_argument("--lm-dir",        type=Path, default=None)
    p.add_argument("--output",        type=Path, default=None)
    p.add_argument("--iterations",    type=int,  default=3, metavar="N",
                   help="Number of QUBO→MCMC round-trips (default: 3).")
    p.add_argument("--solver",        default="auto",
                   choices=["auto", "hybrid", "dwave", "neal", "tabu"],
                   help="QUBO solver for each quantum leg.")
    p.add_argument("--num-reads",     type=int,  default=1000, metavar="N")
    p.add_argument("--hybrid-time-limit", type=int, default=30, metavar="SECS")
    p.add_argument("--mcmc-iterations",   type=int, default=20000, metavar="N",
                   help="MCMC iterations per refinement leg (default: 20 000).")
    p.add_argument("--dwave-token",   default=None, metavar="TOKEN")
    p.add_argument("--crib",          default=None, metavar="SIGN=PHONEME[,…]",
                   help="Known-plaintext crib forwarded to both QUBO and MCMC.")
    p.add_argument("--smoke-test",    action="store_true",
                   help="Tiny problem (10 signs, 500 MCMC iters) for CI / wiring check.")
    args = p.parse_args()

    dwave_token = args.dwave_token or os.environ.get("DWAVE_API_TOKEN")

    # Resolve paths from config if not given.
    corpus_dir = args.corpus_dir
    lm_dir     = args.lm_dir
    output     = args.output
    if corpus_dir is None or lm_dir is None or output is None:
        try:
            from omegaconf import OmegaConf
            cfg = OmegaConf.load(PROJECT_ROOT / "conf" / "config.yaml")
            if corpus_dir is None:
                corpus_dir = PROJECT_ROOT / cfg.paths.corpus_dir
            if lm_dir is None:
                lm_dir = PROJECT_ROOT / "data" / "language_models"
            if output is None:
                output = PROJECT_ROOT / cfg.paths.outputs_dir / "decipherment" / "loop_result.json"
        except Exception:
            pass

    if corpus_dir is None or not corpus_dir.exists():
        log.error("Corpus directory not found.  Pass --corpus-dir.")
        sys.exit(1)
    if lm_dir is None or not lm_dir.exists():
        log.error("LM directory not found.  Pass --lm-dir.")
        sys.exit(1)
    assert output is not None
    output.parent.mkdir(parents=True, exist_ok=True)

    qubo_out  = output.parent / "loop_qubo_latest.json"
    mcmc_out  = output.parent / "loop_mcmc_latest.json"

    best_result_path: Path | None = None
    best_lm_score: float = -float("inf")

    t_start = time.perf_counter()

    for round_idx in range(1, args.iterations + 1):
        log.info("─── Round %d / %d ────────────────────────────", round_idx, args.iterations)

        # ── QUBO leg ─────────────────────────────────────────────────────────
        qubo_rc = _run(
            _qubo_cmd(
                corpus_dir=corpus_dir,
                lm_dir=lm_dir,
                output=qubo_out,
                solver=args.solver,
                num_reads=args.num_reads,
                dwave_token=dwave_token,
                hybrid_time_limit=args.hybrid_time_limit,
                init_from=mcmc_out if round_idx > 1 else None,
                smoke_test=args.smoke_test,
                crib=args.crib,
            ),
            label=f"Round {round_idx} — QUBO ({args.solver})",
        )
        if qubo_rc != 0:
            log.error("QUBO leg failed (exit %d) — aborting loop.", qubo_rc)
            sys.exit(qubo_rc)

        # Read QUBO LM score for progress tracking.
        if qubo_out.exists():
            try:
                qdata = json.loads(qubo_out.read_text())
                q_score = float(qdata.get("lm_score", -float("inf")))
                log.info("Round %d QUBO LM score: %.4f", round_idx, q_score)
                if q_score > best_lm_score:
                    best_lm_score = q_score
                    best_result_path = qubo_out
            except Exception:
                pass

        # ── MCMC leg ─────────────────────────────────────────────────────────
        mcmc_rc = _run(
            _mcmc_cmd(
                corpus_dir=corpus_dir,
                lm_dir=lm_dir,
                output=mcmc_out,
                init_from=qubo_out,
                num_iterations=args.mcmc_iterations,
                smoke_test=args.smoke_test,
                crib=args.crib,
            ),
            label=f"Round {round_idx} — MCMC refinement",
        )
        if mcmc_rc != 0:
            log.warning("MCMC leg exited %d — continuing with QUBO result.", mcmc_rc)
        else:
            if mcmc_out.exists():
                try:
                    mdata = json.loads(mcmc_out.read_text())
                    hyps = mdata.get("hypotheses", [])
                    if hyps:
                        m_score = float(hyps[0].get("overall_lm_score", -float("inf")))
                        log.info("Round %d MCMC best LM score: %.4f", round_idx, m_score)
                        if m_score > best_lm_score:
                            best_lm_score = m_score
                            best_result_path = mcmc_out
                except Exception:
                    pass

    elapsed = time.perf_counter() - t_start
    log.info("Loop complete: %d rounds in %.1f s.  Best LM score: %.4f",
             args.iterations, elapsed, best_lm_score)

    # Copy best result to final output path.
    if best_result_path and best_result_path.exists():
        import shutil
        shutil.copy(best_result_path, output)
        log.info("Best result written to %s", output)
    else:
        log.warning("No valid result found — check logs above.")


if __name__ == "__main__":
    main()
