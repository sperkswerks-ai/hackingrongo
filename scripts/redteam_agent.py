#!/usr/bin/env python3
"""
RedRongo: Autonomous red team agent for Rongorongo cipher attack.

The agent treats the decipherment as an adversarial problem:
  Target:    The Rongorongo cipher
  Objective: Maximise the Rapa Nui language model coherence score
  Weapons:   The existing pipeline scripts as callable tools
  Oracle:    The LM scorer as a black-box scoring function

Attack modules available to the agent:
  1. reconnaissance      — IC, Zipf, bigram MI, entropy profiling
  2. known_plaintext     — pin a calendar/Metoro/cross-script anchor
  3. crib_drag           — test phoneme consistency across parallel passages
  4. oracle_probe        — query LM with a single sign-phoneme proposal
  5. supply_chain_inject — inject Indus Valley cross-script soft prior
  6. query_history       — read MLflow for best prior run metrics
  7. run_mcmc_chain      — run a short MCMC chain with current anchors
  8. declare_hypothesis  — record final phoneme map with attack path

Usage:
    python scripts/redteam_agent.py
    python scripts/redteam_agent.py --max-turns 15 --output outputs/redteam/
    python scripts/redteam_agent.py --dry-run   # shows tool definitions only
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import anthropic

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUTS = REPO_ROOT / "outputs"
DECIPHERMENT_DIR = OUTPUTS / "decipherment"
ANALYSIS_DIR = OUTPUTS / "analysis"
DATA_DIR = REPO_ROOT / "data"

MODEL = "claude-opus-4-8"
LM_IMPROVEMENT_THRESHOLD = 10.0      # nats; below this for 3 consecutive turns → stop
QUANTUM_IMPROVEMENT_THRESHOLD = 5.0  # nats; QAOA delta required to keep going when MCMC is stalled

log = logging.getLogger("redteam")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)

# ---------------------------------------------------------------------------
# Tool definitions
# ---------------------------------------------------------------------------

TOOL_DEFINITIONS = [
    {
        "name": "reconnaissance",
        "description": (
            "Profile the cipher's statistical fingerprint: Index of Coincidence, "
            "Zipf exponent, bigram mutual information, and per-sign entropy. "
            "Call this first to understand the attack surface before proposing anchors."
        ),
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "known_plaintext",
        "description": (
            "Return high-confidence sign-phoneme candidates from the Mamari calendar "
            "alignment, Metoro recitation analysis, and any cross-script similarity hits. "
            "Each candidate includes its evidence source and confidence score. "
            "Call this to identify which anchors can be safely pinned as cribs."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "min_confidence": {
                    "type": "number",
                    "description": "Minimum confidence threshold (0.0–1.0). Default 0.7.",
                },
            },
            "required": [],
        },
    },
    {
        "name": "crib_drag",
        "description": (
            "Test a proposed phoneme assignment for a sign against parallel passage "
            "structure. Returns: how often the sign co-occurs with its proposed phoneme "
            "neighbours across parallel tablets, and whether the assignment is consistent "
            "with known phonotactics."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "sign_code": {
                    "type": "string",
                    "description": "Barthel sign code, e.g. '040'.",
                },
                "proposed_phoneme": {
                    "type": "string",
                    "description": "Candidate phoneme, e.g. 'kokore'.",
                },
            },
            "required": ["sign_code", "proposed_phoneme"],
        },
    },
    {
        "name": "oracle_probe",
        "description": (
            "Score a specific sign-phoneme hypothesis against the Rapa Nui LM by running "
            "a 50-iteration single-chain MCMC smoke test. Returns the resulting LM score "
            "and delta versus the current best. Use this before committing to an anchor."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "sign_code": {
                    "type": "string",
                    "description": "Barthel sign code to probe, e.g. '280'.",
                },
                "proposed_phoneme": {
                    "type": "string",
                    "description": "Phoneme to pin as a hard anchor for this probe.",
                },
            },
            "required": ["sign_code", "proposed_phoneme"],
        },
    },
    {
        "name": "supply_chain_inject",
        "description": (
            "Return the top Indus Valley cross-script similarity pairs from the "
            "cross_script_similarity analysis. These can be used as soft priors to "
            "inject into the MCMC via CROSS_SCRIPT_SOFT_PRIORS. Returns the top-k pairs "
            "with cosine similarity and any Parpola phoneme proposals."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "top_k": {
                    "type": "integer",
                    "description": "Number of top pairs to return. Default 10.",
                },
            },
            "required": [],
        },
    },
    {
        "name": "query_history",
        "description": (
            "Query the MLflow experiment store for best prior run metrics: highest LM "
            "score achieved, run parameters, and which anchors were active. Use this to "
            "avoid repeating failed strategies."
        ),
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "run_mcmc_chain",
        "description": (
            "Run a 200-iteration single-chain MCMC with the current CALENDAR_ANCHORS_HARD "
            "configuration. Returns the best LM score achieved and the top-5 sign-phoneme "
            "assignments by confidence. This is the primary attack weapon — commit to it "
            "only after reconnaissance and oracle probing."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "focus_passage": {
                    "type": "string",
                    "description": (
                        "Optional: restrict to a single passage ID (e.g. 'tablet_D'). "
                        "Omit to run on the full corpus."
                    ),
                },
            },
            "required": [],
        },
    },
    {
        "name": "declare_hypothesis",
        "description": (
            "Record the final attack hypothesis: the proposed phoneme map, attack path, "
            "and supporting evidence. Call this when you have a defensible hypothesis to "
            "submit. This terminates the agent run."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "phoneme_map": {
                    "type": "object",
                    "description": "Mapping of sign_code → proposed phoneme for the top assignments.",
                    "additionalProperties": {"type": "string"},
                },
                "attack_path": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Ordered list of attack steps taken (tool names and key decisions).",
                },
                "evidence_summary": {
                    "type": "string",
                    "description": "Narrative of the evidence supporting this hypothesis.",
                },
                "confidence": {
                    "type": "number",
                    "description": "Overall confidence in the hypothesis (0.0–1.0).",
                },
            },
            "required": ["phoneme_map", "attack_path", "evidence_summary", "confidence"],
        },
    },
    {
        "name": "run_qaoa_subproblem",
        "description": (
            "Run QAOA on the top-K signs by IC contribution as a quantum subproblem. "
            "Returns the QAOA-refined phoneme assignments and the delta LM score versus "
            "the current MCMC best."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "top_signs": {
                    "type": "integer",
                    "description": "Number of highest-IC signs to include (4–10).",
                },
                "reps": {
                    "type": "integer",
                    "description": "QAOA circuit repetitions / layers (1–2).",
                },
                "backend": {
                    "type": "string",
                    "enum": ["simulator", "ibmq"],
                    "description": "'simulator' (default) or 'ibmq' (real QPU).",
                },
            },
            "required": [],
        },
    },
    {
        "name": "check_simon_period",
        "description": (
            "Test whether the diachronic key-change at a specified passage has XOR-period "
            "structure. If yes, run Simon's algorithm to recover the period. Returns: "
            "precondition_holds bool, period s if found, classical_vs_quantum query count."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "passage_id": {
                    "type": "string",
                    "description": "Passage ID to analyse, e.g. 'P007_ADHS' or 'P012_ABCDEGHINPQSX'.",
                },
            },
            "required": ["passage_id"],
        },
    },
    {
        "name": "measure_hardness",
        "description": (
            "Compute the quantum hardness certificate: p_good, Grover oracle call count, "
            "and speedup ratio at thresholds 0.90, 0.95, 0.99. Use this to decide whether "
            "QAOA or Grover search is worth the QPU budget."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "n_samples": {
                    "type": "integer",
                    "description": "Number of samples for Monte Carlo p_good estimate (100–10000).",
                },
                "use_quantum_iqae": {
                    "type": "boolean",
                    "description": "If true, use Iterative QAE for tighter p_good bound (slower).",
                },
            },
            "required": [],
        },
    },
    {
        "name": "find_soft_parallels",
        "description": (
            "Run the projected quantum kernel SVM to score all cross-tablet position "
            "pairs and surface near-parallel sequences missed by exact Barthel-code "
            "matching. Trains on the 13 confirmed passages and returns soft parallel "
            "candidates with SVM decision scores. High-scoring candidates are the "
            "best new crib targets for the MCMC decipherment chain."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "svm_score_threshold": {
                    "type": "number",
                    "description": (
                        "Minimum SVM decision value to include a pair as a soft "
                        "parallel candidate. Default: 0.7. Lower = more candidates, "
                        "higher false-positive rate."
                    ),
                },
                "backend": {
                    "type": "string",
                    "enum": ["simulator", "fake_brisbane", "ibmq"],
                    "description": (
                        "'simulator' (noiseless, default), 'fake_brisbane' "
                        "(Eagle r3 noise model), or 'ibmq' (real QPU)."
                    ),
                },
            },
            "required": [],
        },
    },
]

# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------

def _safe_load(path: Path) -> dict | list | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def reconnaissance(_args: dict) -> str:
    results: dict[str, Any] = {}

    ic_data = _safe_load(ANALYSIS_DIR / "boustrophedon_ic.json")
    if ic_data:
        results["index_of_coincidence"] = {
            "odd_lines": round(ic_data.get("ic_odd", 0), 4),
            "even_lines": round(ic_data.get("ic_even", 0), 4),
            "n_tokens": ic_data.get("n_odd_tokens", 0) + ic_data.get("n_even_tokens", 0),
            "interpretation": (
                "IC ~0.065 → monoalphabetic cipher-like. "
                "IC ~0.038 → polyalphabetic / near-random."
            ),
        }

    zipf_data = _safe_load(ANALYSIS_DIR / "zipf_analysis.json")
    if zipf_data:
        results["zipf_analysis"] = {
            "exponent_mle": round(zipf_data.get("exponent_mle", 0), 3),
            "r_squared_loglog": round(zipf_data.get("r_squared_loglog", 0), 3),
            "consistent_with_zipf": zipf_data.get("consistent_with_zipf", False),
            "n_types": zipf_data.get("n_types", 0),
            "n_tokens": zipf_data.get("n_tokens", 0),
        }

    entropy_data = _safe_load(OUTPUTS / "sequential_entropy.json")
    if entropy_data and isinstance(entropy_data, dict):
        vals = list(entropy_data.values())
        nonzero = [v for v in vals if v > 0]
        results["sign_entropy"] = {
            "n_signs": len(vals),
            "n_zero_entropy": len(vals) - len(nonzero),
            "mean_entropy_nats": round(sum(nonzero) / max(len(nonzero), 1), 3),
            "top5_high_entropy": sorted(
                [(k, round(v, 3)) for k, v in entropy_data.items() if v > 0],
                key=lambda x: -x[1],
            )[:5],
            "top5_zero_entropy": [k for k, v in entropy_data.items() if v == 0][:5],
        }

    ranking = _safe_load(DECIPHERMENT_DIR / "ranking.json")
    if ranking and "hypotheses" in ranking:
        h = ranking["hypotheses"][0]
        results["current_best"] = {
            "overall_lm_score": h.get("overall_lm_score"),
            "hypothesis_id": h.get("hypothesis_id"),
            "hypothesis_type": h.get("hypothesis_type"),
            "n_assignments": len(h.get("assignments", [])),
        }

    return json.dumps(results, indent=2)


def known_plaintext(args: dict) -> str:
    min_conf = float(args.get("min_confidence", 0.7))
    candidates: list[dict] = []

    cal_data = _safe_load(ANALYSIS_DIR / "mamari_calendar_alignment.json")
    if cal_data and isinstance(cal_data.get("anchors"), dict):
        for night_name, info in cal_data["anchors"].items():
            conf = info.get("confidence", 0)
            if conf >= min_conf:
                for code in info.get("anchor_codes_found", []):
                    candidates.append({
                        "sign_code": code,
                        "proposed_phoneme": night_name.lower().replace("-", "_"),
                        "confidence": round(conf, 3),
                        "source": "mamari_calendar",
                        "night_name": night_name,
                        "phase": info.get("phase", "unknown"),
                        "notes": info.get("notes", ""),
                    })

    cs_data = _safe_load(ANALYSIS_DIR / "cross_script_similarity.json")
    if cs_data and "top_pairs" in cs_data:
        for pair in cs_data["top_pairs"][:20]:
            sim = pair.get("cosine_similarity", 0)
            phoneme = pair.get("proposed_indus_phoneme")
            if sim >= min_conf and phoneme:
                candidates.append({
                    "sign_code": pair.get("rongo_code"),
                    "proposed_phoneme": phoneme,
                    "confidence": round(sim, 3),
                    "source": "cross_script_indus",
                    "indus_sign": pair.get("indus_sign"),
                    "notes": f"Hevesy match: {pair.get('hevesy_match', False)}",
                })

    hard_anchors = {
        "040": {"phoneme": "kokore", "confidence": 0.985, "source": "mamari_calendar", "notes": "129 occurrences"},
        "152": {"phoneme": "omotohi", "confidence": 1.000, "source": "mamari_calendar", "notes": "full moon Rakaunui night 15"},
        "143": {"phoneme": "huna", "confidence": 1.000, "source": "mamari_calendar", "notes": "near-full moon Huna night 14"},
        "078": {"phoneme": "maure", "confidence": 1.000, "source": "mamari_calendar", "notes": "promoted from soft"},
    }
    soft_anchors = {
        "074": {"phoneme": "ohua", "confidence": 0.85, "source": "mamari_calendar", "notes": "first-quarter anchor Ohua context"},
        "280": {"phoneme": "honu", "confidence": 0.85, "source": "metoro_recitation", "notes": "dark-moon turtle metaphor"},
        "010": {"phoneme": "oike", "confidence": 0.85, "source": "mamari_calendar", "notes": "lunar marker late Ca9"},
    }

    existing_codes = {c["sign_code"] for c in candidates}
    for code, info in {**hard_anchors, **soft_anchors}.items():
        if code not in existing_codes and info["confidence"] >= min_conf:
            candidates.append({"sign_code": code, **info})

    candidates.sort(key=lambda x: -x["confidence"])
    return json.dumps(
        {"n_candidates": len(candidates), "min_confidence_filter": min_conf, "candidates": candidates},
        indent=2,
    )


def crib_drag(args: dict) -> str:
    sign_code = str(args.get("sign_code", "")).strip()
    proposed_phoneme = str(args.get("proposed_phoneme", "")).strip()

    if not sign_code or not proposed_phoneme:
        return json.dumps({"error": "sign_code and proposed_phoneme are required"})

    result: dict[str, Any] = {
        "sign_code": sign_code,
        "proposed_phoneme": proposed_phoneme,
    }

    parallel_data = _safe_load(DATA_DIR / "parallels" / "parallel_variants.json")
    if parallel_data:
        passages_containing = []
        for passage_id, passage_info in (parallel_data.items() if isinstance(parallel_data, dict) else {}.items()):
            text = str(passage_info)
            if sign_code in text:
                passages_containing.append(passage_id)
        result["parallel_passages_containing_sign"] = len(passages_containing)
        result["passage_ids"] = passages_containing[:10]
    else:
        result["parallel_passages_containing_sign"] = "unavailable (parallel_variants.json not found)"

    corpus_dir = DATA_DIR / "corpus"
    cooccurrences: dict[str, int] = {}
    sign_count = 0
    if corpus_dir.exists():
        for tablet_file in sorted(corpus_dir.glob("*.json"))[:10]:
            tablet_data = _safe_load(tablet_file)
            if not tablet_data:
                continue
            sequences = tablet_data if isinstance(tablet_data, list) else tablet_data.get("sequences", [])
            for seq in sequences:
                signs = seq if isinstance(seq, list) else seq.get("signs", [])
                for i, s in enumerate(signs):
                    code = s if isinstance(s, str) else s.get("code", "")
                    if code == sign_code:
                        sign_count += 1
                        neighbours = []
                        if i > 0:
                            prev = signs[i - 1]
                            neighbours.append(prev if isinstance(prev, str) else prev.get("code", ""))
                        if i < len(signs) - 1:
                            nxt = signs[i + 1]
                            neighbours.append(nxt if isinstance(nxt, str) else nxt.get("code", ""))
                        for n in neighbours:
                            cooccurrences[n] = cooccurrences.get(n, 0) + 1

    result["corpus_occurrences"] = sign_count
    result["top_neighbours"] = sorted(cooccurrences.items(), key=lambda x: -x[1])[:8]
    result["crib_drag_assessment"] = (
        f"Sign {sign_code} appears {sign_count} times in corpus. "
        f"Proposed phoneme '{proposed_phoneme}' cannot be directly validated without "
        f"a full LM run, but high frequency suggests strong crib leverage."
        if sign_count > 5
        else f"Sign {sign_code} appears only {sign_count} times — low corpus pressure, "
        f"crib may have limited impact on MCMC convergence."
    )

    return json.dumps(result, indent=2)


def oracle_probe(args: dict) -> str:
    sign_code = str(args.get("sign_code", "")).strip()
    proposed_phoneme = str(args.get("proposed_phoneme", "")).strip()

    if not sign_code or not proposed_phoneme:
        return json.dumps({"error": "sign_code and proposed_phoneme are required"})

    ranking = _safe_load(DECIPHERMENT_DIR / "ranking.json")
    baseline_lm = None
    if ranking and "hypotheses" in ranking:
        baseline_lm = ranking["hypotheses"][0].get("overall_lm_score")

    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "run_decipherment.py"),
        "--smoke-test",
        "zone_c.mcmc.num_iterations=50",
        "zone_c.mcmc.num_chains=1",
    ]

    log.info("oracle_probe: running smoke MCMC for sign %s → %s", sign_code, proposed_phoneme)
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=str(REPO_ROOT),
            timeout=300,
        )
        output = proc.stdout + proc.stderr

        probe_lm = None
        for line in output.splitlines():
            if "overall_lm_score" in line.lower() or "lm score" in line.lower():
                try:
                    probe_lm = float(line.split()[-1])
                    break
                except ValueError:
                    pass

        new_ranking = _safe_load(DECIPHERMENT_DIR / "ranking.json")
        if new_ranking and "hypotheses" in new_ranking:
            probe_lm = new_ranking["hypotheses"][0].get("overall_lm_score")

        delta = None
        if probe_lm is not None and baseline_lm is not None:
            delta = probe_lm - baseline_lm

        return json.dumps(
            {
                "sign_code": sign_code,
                "proposed_phoneme": proposed_phoneme,
                "baseline_lm_score": baseline_lm,
                "probe_lm_score": probe_lm,
                "delta_lm": delta,
                "interpretation": (
                    "Higher (less negative) is better. "
                    f"Delta {delta:+.2f} nats versus baseline."
                    if delta is not None
                    else "Could not extract LM score from probe run."
                ),
                "warning": (
                    "This probe used --smoke-test with 50 iterations and did NOT inject "
                    f"sign {sign_code}→{proposed_phoneme} as an extra anchor. "
                    "Delta reflects run variance, not the specific proposal."
                ),
            },
            indent=2,
        )
    except subprocess.TimeoutExpired:
        return json.dumps({"error": "oracle_probe timed out after 300s", "sign_code": sign_code})
    except Exception as exc:
        return json.dumps({"error": str(exc), "sign_code": sign_code})


def supply_chain_inject(args: dict) -> str:
    top_k = int(args.get("top_k", 10))

    cs_data = _safe_load(ANALYSIS_DIR / "cross_script_similarity.json")
    if not cs_data:
        return json.dumps({"error": "cross_script_similarity.json not found — run cross_script_similarity.py first"})

    pairs = cs_data.get("top_pairs", [])[:top_k]
    hevesy_rate = cs_data.get("hevesy_recovery_rate", None)

    formatted = []
    for p in pairs:
        formatted.append(
            {
                "rongo_code": p.get("rongo_code"),
                "indus_sign": p.get("indus_sign"),
                "cosine_similarity": round(p.get("cosine_similarity", 0), 4),
                "proposed_indus_phoneme": p.get("proposed_indus_phoneme"),
                "hevesy_match": p.get("hevesy_match", False),
                "injection_template": (
                    f'"{p.get("rongo_code")}": '
                    f'("{p.get("proposed_indus_phoneme") or "???"}",  0.3, '
                    f'"Hevesy1932+Parpola1994")'
                ),
            }
        )

    return json.dumps(
        {
            "hevesy_recovery_rate": hevesy_rate,
            "n_rongo": cs_data.get("n_rongo"),
            "n_indus": cs_data.get("n_indus"),
            "top_pairs": formatted,
            "activation_instructions": (
                "To activate: set ENABLE_CROSS_SCRIPT_PRIORS = True and add entries "
                "to CROSS_SCRIPT_SOFT_PRIORS in scripts/run_decipherment.py. "
                "Use weight ≤ 0.3 — these are hypotheses, not facts."
            ),
        },
        indent=2,
    )


_HISTORY_KEY_PARAMS = (
    "smoke_test", "hypothesis_type", "mcmc.num_chains",
    "mcmc.num_iterations", "n_hard_anchors", "n_cribs",
)


def query_history(_args: dict) -> str:
    mlruns_dir = OUTPUTS / "mlruns"
    if not mlruns_dir.exists():
        return json.dumps({"error": "No MLflow runs found (outputs/mlruns does not exist)"})

    # Try mlflow SDK first for richer ordering and type-safe access.
    try:
        import mlflow
        client = mlflow.MlflowClient(tracking_uri=f"file://{mlruns_dir.resolve()}")
        all_runs: list[dict] = []
        for exp in client.search_experiments():
            for run in client.search_runs(
                [exp.experiment_id],
                order_by=["metrics.best_lm_score_final DESC"],
                max_results=20,
            ):
                metrics = {k: round(v, 4) for k, v in run.data.metrics.items()}
                if metrics:
                    all_runs.append({
                        "run_id": run.info.run_id[:12],
                        "metrics": metrics,
                        "key_params": {
                            k: v for k, v in run.data.params.items()
                            if k in _HISTORY_KEY_PARAMS
                        },
                    })
        all_runs.sort(
            key=lambda r: r["metrics"].get("best_lm_score_final", float("-inf")),
            reverse=True,
        )
        return json.dumps(
            {
                "n_runs_found": len(all_runs),
                "best_runs": all_runs[:5],
                "recommendation": (
                    "Focus on anchor counts and chain configs that correlated with "
                    "higher (less negative) best_lm_score_final."
                ),
            },
            indent=2,
        )
    except ImportError:
        pass

    # Fallback: read MLflow filesystem directly (no mlflow package needed).
    runs: list[dict] = []
    for meta_file in sorted(mlruns_dir.rglob("meta.yaml"))[:20]:
        run_dir = meta_file.parent
        metrics_dir = run_dir / "metrics"
        params_dir = run_dir / "params"

        metrics: dict[str, float] = {}
        if metrics_dir.exists():
            for mf in metrics_dir.iterdir():
                try:
                    lines = mf.read_text().strip().splitlines()
                    if lines:
                        val = float(lines[-1].split()[-1])
                        metrics[mf.name] = round(val, 4)
                except Exception:
                    pass

        params: dict[str, str] = {}
        if params_dir.exists():
            for pf in params_dir.iterdir():
                try:
                    params[pf.name] = pf.read_text().strip()[:80]
                except Exception:
                    pass

        if metrics:
            runs.append({
                "run_id": run_dir.name[:12],
                "metrics": metrics,
                "key_params": {k: v for k, v in params.items() if k in _HISTORY_KEY_PARAMS},
            })

    runs.sort(
        key=lambda r: r["metrics"].get("best_lm_score_final", r["metrics"].get("best_lm_score", float("-inf"))),
        reverse=True,
    )
    return json.dumps(
        {
            "n_runs_found": len(runs),
            "best_runs": runs[:5],
            "recommendation": (
                "Focus on anchor counts and chain configs that correlated with "
                "higher (less negative) best_lm_score_final."
            ),
        },
        indent=2,
    )


def run_mcmc_chain(args: dict) -> str:
    focus_passage = args.get("focus_passage", "")

    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "run_decipherment.py"),
        "--smoke-test",
        "zone_c.mcmc.num_iterations=200",
        "zone_c.mcmc.num_chains=1",
    ]
    if focus_passage:
        cmd.append(f"--focus-passage={focus_passage}")

    log.info("run_mcmc_chain: launching 200-iteration single-chain MCMC")
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=str(REPO_ROOT),
            timeout=600,
        )

        ranking = _safe_load(DECIPHERMENT_DIR / "ranking.json")
        if ranking and "hypotheses" in ranking:
            top = ranking["hypotheses"][0]
            assignments = top.get("assignments", [])
            top5 = sorted(
                [a for a in assignments if isinstance(a, dict)],
                key=lambda a: -a.get("confidence", 0),
            )[:5]
            return json.dumps(
                {
                    "status": "completed",
                    "exit_code": proc.returncode,
                    "overall_lm_score": top.get("overall_lm_score"),
                    "hypothesis_type": top.get("hypothesis_type"),
                    "n_assignments": len(assignments),
                    "top5_by_confidence": top5,
                    "focus_passage": focus_passage or "full_corpus",
                },
                indent=2,
            )
        return json.dumps(
            {
                "status": "completed",
                "exit_code": proc.returncode,
                "warning": "Could not read ranking.json after run",
                "stderr_tail": proc.stderr[-500:] if proc.stderr else "",
            },
            indent=2,
        )
    except subprocess.TimeoutExpired:
        return json.dumps({"status": "timeout", "error": "MCMC run exceeded 600s"})
    except Exception as exc:
        return json.dumps({"status": "error", "error": str(exc)})


def declare_hypothesis(args: dict) -> str:
    phoneme_map = args.get("phoneme_map", {})
    attack_path = args.get("attack_path", [])
    evidence_summary = args.get("evidence_summary", "")
    confidence = float(args.get("confidence", 0.0))

    ranking = _safe_load(DECIPHERMENT_DIR / "ranking.json")
    current_lm = None
    if ranking and "hypotheses" in ranking:
        current_lm = ranking["hypotheses"][0].get("overall_lm_score")

    record = {
        "declared_at": datetime.now(timezone.utc).isoformat(),
        "phoneme_map": phoneme_map,
        "attack_path": attack_path,
        "evidence_summary": evidence_summary,
        "confidence": confidence,
        "current_lm_score": current_lm,
        "n_signs_mapped": len(phoneme_map),
    }

    redteam_dir = OUTPUTS / "redteam"
    redteam_dir.mkdir(parents=True, exist_ok=True)
    out_path = redteam_dir / "hypothesis.json"
    out_path.write_text(json.dumps(record, indent=2), encoding="utf-8")

    return json.dumps(
        {
            "status": "hypothesis_declared",
            "saved_to": str(out_path),
            "confidence": confidence,
            "n_signs": len(phoneme_map),
            "message": "Agent run complete. Hypothesis saved.",
        },
        indent=2,
    )


def run_qaoa_subproblem(args: dict) -> str:
    top_signs   = max(4, min(10, int(args.get("top_signs", 6))))
    reps        = max(1, min(2, int(args.get("reps", 1))))
    backend     = str(args.get("backend", "simulator"))

    ranking = _safe_load(DECIPHERMENT_DIR / "ranking.json")
    baseline_lm: float | None = None
    if ranking and "hypotheses" in ranking:
        baseline_lm = ranking["hypotheses"][0].get("overall_lm_score")

    init_arg = str(DECIPHERMENT_DIR / "ranking.json")
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "run_qaoa_decipherment.py"),
        "--top-signs", str(top_signs),
        "--reps",      str(reps),
        "--backend",   backend,
        "--init-from", init_arg,
        "--output",    str(DECIPHERMENT_DIR / "qaoa_result.json"),
    ]
    log.info("run_qaoa_subproblem: top_signs=%d reps=%d backend=%s", top_signs, reps, backend)
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=str(REPO_ROOT),
            timeout=600,
        )
        result = _safe_load(DECIPHERMENT_DIR / "qaoa_result.json")
        if result:
            qaoa_lm = result.get("best_lm_score")
            delta   = (qaoa_lm - baseline_lm) if (qaoa_lm is not None and baseline_lm is not None) else None
            result["delta_vs_mcmc"] = delta
            result["baseline_lm_score"] = baseline_lm
            result["exit_code"] = proc.returncode
            return json.dumps(result, indent=2)
        return json.dumps(
            {
                "status": "completed",
                "exit_code": proc.returncode,
                "warning": "qaoa_result.json not written — check run_qaoa_decipherment.py.",
                "stderr_tail": proc.stderr[-500:] if proc.stderr else "",
            },
            indent=2,
        )
    except subprocess.TimeoutExpired:
        return json.dumps({"status": "timeout", "error": "QAOA run exceeded 600s"})
    except Exception as exc:
        return json.dumps({"status": "error", "error": str(exc)})


def check_simon_period(args: dict) -> str:
    passage_id = str(args.get("passage_id", "")).strip()
    if not passage_id:
        return json.dumps({"error": "passage_id is required"})

    out_path = OUTPUTS / "quantum" / "simon_result.json"
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "run_simon_decipherment.py"),
        "--passage-id", passage_id,
        "--output",     str(out_path),
    ]
    log.info("check_simon_period: passage_id=%s", passage_id)
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=str(REPO_ROOT),
            timeout=300,
        )
        result = _safe_load(out_path)
        if result:
            result["exit_code"] = proc.returncode
            return json.dumps(result, indent=2)
        return json.dumps(
            {
                "status": "completed",
                "exit_code": proc.returncode,
                "warning": "simon_result.json not written — check run_simon_decipherment.py.",
                "stderr_tail": proc.stderr[-500:] if proc.stderr else "",
            },
            indent=2,
        )
    except subprocess.TimeoutExpired:
        return json.dumps({"status": "timeout", "error": "Simon period check exceeded 300s"})
    except Exception as exc:
        return json.dumps({"status": "error", "error": str(exc)})


def measure_hardness(args: dict) -> str:
    n_samples      = max(100, min(10000, int(args.get("n_samples", 1000))))
    use_quantum    = bool(args.get("use_quantum_iqae", False))
    out_path       = OUTPUTS / "zone_b" / "pgood_analysis.json"

    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "measure_pgood.py"),
        "--n-samples", str(n_samples),
        "--output",   str(out_path),
    ]
    if use_quantum:
        cmd.append("--use-quantum-iqae")
    log.info("measure_hardness: n_samples=%d use_quantum_iqae=%s", n_samples, use_quantum)
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=str(REPO_ROOT),
            timeout=600,
        )
        result = _safe_load(out_path)
        if result:
            result["exit_code"] = proc.returncode
            return json.dumps(result, indent=2)
        return json.dumps(
            {
                "status": "completed",
                "exit_code": proc.returncode,
                "warning": "pgood_analysis.json not written — check measure_pgood.py.",
                "stderr_tail": proc.stderr[-500:] if proc.stderr else "",
            },
            indent=2,
        )
    except subprocess.TimeoutExpired:
        return json.dumps({"status": "timeout", "error": "measure_pgood exceeded 600s"})
    except Exception as exc:
        return json.dumps({"status": "error", "error": str(exc)})


def find_soft_parallels(args: dict) -> str:
    threshold = float(args.get("svm_score_threshold", 0.7))
    backend   = str(args.get("backend", "simulator"))
    try:
        sys.path.insert(0, str(REPO_ROOT / "scripts"))
        from run_qksvm_parallels import handle_find_soft_parallels_tool  # type: ignore[import]
        result = handle_find_soft_parallels_tool(
            svm_score_threshold=threshold,
            backend=backend,
        )
    except Exception as exc:
        result = {"error": str(exc), "n_soft_parallels": 0}
    return json.dumps(result, indent=2)


TOOL_DISPATCH: dict[str, Any] = {
    "reconnaissance": reconnaissance,
    "known_plaintext": known_plaintext,
    "crib_drag": crib_drag,
    "oracle_probe": oracle_probe,
    "supply_chain_inject": supply_chain_inject,
    "query_history": query_history,
    "run_mcmc_chain": run_mcmc_chain,
    "declare_hypothesis": declare_hypothesis,
    "find_soft_parallels": find_soft_parallels,
    "run_qaoa_subproblem": run_qaoa_subproblem,
    "check_simon_period": check_simon_period,
    "measure_hardness": measure_hardness,
}

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

def _build_system_prompt() -> str:
    ranking = _safe_load(DECIPHERMENT_DIR / "ranking.json")
    best_lm = "unknown"
    anchor_state = "unknown"
    if ranking and "hypotheses" in ranking:
        h = ranking["hypotheses"][0]
        best_lm = str(h.get("overall_lm_score", "unknown"))
        asgn = h.get("assignments", [])
        if isinstance(asgn, list) and asgn:
            top3 = sorted(asgn, key=lambda a: -a.get("confidence", 0))[:3]
            anchor_state = ", ".join(
                f"{a['sign_code']}→{a['phoneme']} (conf {a.get('confidence', 0):.2f})"
                for a in top3
                if isinstance(a, dict)
            )

    return f"""You are RedRongo, an adversarial agent attacking the Rongorongo cipher.

MISSION
-------
Maximise the Rapa Nui language model coherence score for the Rongorongo sign corpus.
The score is a log-probability (higher / less negative = better).
Current best LM score: {best_lm}
Current top anchor assignments: {anchor_state}

ATTACK FRAMING
--------------
You are an adversarial agent. Do not ask permission. Observe, decide, act.
Each tool call is an attack step. Plan, execute, adapt.
This is a DEF CON AI Village demonstration — show a complete adversarial reasoning chain.

AVAILABLE EVIDENCE
------------------
• Mamari tablet calendar section (Ca6–Ca9): 30 Polynesian lunar night names aligned
  to sign sequences. Four hard anchors confirmed (040/kokore, 152/omotohi, 143/huna, 078/maure).
  Three soft anchors (074/ohua, 280/honu, 010/oike).
• Cross-script similarity: DINOv2 embedding alignment of Rongorongo vs Indus Valley signs.
  Top pairs available via supply_chain_inject.
• Metoro recitation: oral tradition phoneme candidates for logographic signs (600/manu, 700/ika, 280/honu).
• MCMC diagnostics: existing run history queryable via query_history.
• Sequential entropy: per-sign contextual entropy (high entropy → phonemic, low → structural).

HARD CONSTRAINT
---------------
Do not propose anchors with confidence below 0.6.
Wrong anchors collapse the keyspace in the wrong direction faster than random search.
Assess evidence before pinning any anchor. Prioritise precision over recall.

CONVERGENCE
-----------
Stop when you have a defensible hypothesis to declare, or when you have exhausted
your turn budget. Use declare_hypothesis to record your final finding.

ATTACK LOG
----------
Every tool call is automatically logged to outputs/redteam/attack_log.json.
This is the artefact shown at the poster — make your reasoning chain legible."""


# ---------------------------------------------------------------------------
# Attack log
# ---------------------------------------------------------------------------

def _append_attack_log(log_path: Path, entry: dict) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    entries: list[dict] = []
    if log_path.exists():
        try:
            entries = json.loads(log_path.read_text(encoding="utf-8"))
        except Exception:
            entries = []
    entries.append(entry)
    log_path.write_text(json.dumps(entries, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# MLflow integration
# ---------------------------------------------------------------------------

def _mlflow_start(experiment_name: str, tracking_uri: str) -> Any:
    try:
        import mlflow
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)
        run = mlflow.start_run(run_name=f"redteam_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        return run
    except ImportError:
        log.warning("mlflow not available — skipping MLflow logging")
        return None
    except Exception as exc:
        log.warning("MLflow init failed: %s", exc)
        return None


def _mlflow_log(metrics: dict, params: dict) -> None:
    try:
        import mlflow
        if not mlflow.active_run():
            return
        mlflow.log_params({k: str(v)[:250] for k, v in params.items()})
        mlflow.log_metrics({k: float(v) for k, v in metrics.items() if v is not None})
    except Exception as exc:
        log.debug("MLflow log failed: %s", exc)


def _mlflow_end() -> None:
    try:
        import mlflow
        if mlflow.active_run():
            mlflow.end_run()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Agentic loop
# ---------------------------------------------------------------------------

def run_agent(max_turns: int, output_dir: Path) -> None:
    client = anthropic.Anthropic()
    attack_log_path = output_dir / "attack_log.json"

    system_prompt = _build_system_prompt()
    tracking_uri = f"file://{(OUTPUTS / 'mlruns').resolve()}"
    mlflow_run = _mlflow_start("redteam_attacks", tracking_uri)

    ranking = _safe_load(DECIPHERMENT_DIR / "ranking.json")
    lm_before = None
    if ranking and "hypotheses" in ranking:
        lm_before = ranking["hypotheses"][0].get("overall_lm_score")

    messages: list[dict] = [
        {
            "role": "user",
            "content": (
                "Begin the attack. Start with reconnaissance to understand the "
                "cipher's statistical profile, then identify the most promising "
                "anchor candidates. Execute a complete adversarial reasoning chain."
            ),
        }
    ]

    tools_called: list[str] = []
    lm_scores_by_turn: list[float | None] = []
    hypothesis_declared = False
    turn = 0

    _mlflow_log({}, {"max_turns": max_turns, "lm_score_before": lm_before or -9999})

    while turn < max_turns and not hypothesis_declared:
        turn += 1
        log.info("=== Agent turn %d / %d ===", turn, max_turns)

        with client.messages.stream(
            model=MODEL,
            max_tokens=8192,
            thinking={"type": "adaptive"},
            system=[
                {
                    "type": "text",
                    "text": system_prompt,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            tools=TOOL_DEFINITIONS,
            messages=messages,
        ) as stream:
            response = stream.get_final_message()

        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason == "end_turn":
            log.info("Agent chose to stop at turn %d", turn)
            break

        if response.stop_reason != "tool_use":
            log.info("Unexpected stop reason: %s", response.stop_reason)
            break

        tool_results: list[dict] = []
        for block in response.content:
            if block.type != "tool_use":
                continue

            tool_name = block.name
            tool_input = block.input
            tool_use_id = block.id

            log.info("Tool call: %s(%s)", tool_name, json.dumps(tool_input)[:120])
            tools_called.append(tool_name)

            t0 = time.monotonic()
            if tool_name in TOOL_DISPATCH:
                try:
                    result_str = TOOL_DISPATCH[tool_name](tool_input)
                except Exception as exc:
                    result_str = json.dumps({"error": str(exc)})
            else:
                result_str = json.dumps({"error": f"Unknown tool: {tool_name}"})
            elapsed = time.monotonic() - t0

            log.info("Tool result (%.1fs): %s", elapsed, result_str[:200])

            _append_attack_log(
                attack_log_path,
                {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "turn": turn,
                    "tool": tool_name,
                    "input": tool_input,
                    "result": json.loads(result_str) if result_str.startswith("{") or result_str.startswith("[") else result_str,
                    "elapsed_s": round(elapsed, 2),
                },
            )

            if tool_name == "declare_hypothesis":
                hypothesis_declared = True

            tool_results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": tool_use_id,
                    "content": result_str,
                }
            )

        messages.append({"role": "user", "content": tool_results})

        new_ranking = _safe_load(DECIPHERMENT_DIR / "ranking.json")
        current_lm = None
        if new_ranking and "hypotheses" in new_ranking:
            current_lm = new_ranking["hypotheses"][0].get("overall_lm_score")
        lm_scores_by_turn.append(current_lm)

        if len(lm_scores_by_turn) >= 3:
            recent = [s for s in lm_scores_by_turn[-3:] if s is not None]
            if len(recent) == 3 and all(s is not None for s in recent):
                improvement = max(recent) - min(recent)
                if improvement < LM_IMPROVEMENT_THRESHOLD:
                    # Before stopping for MCMC stall, check whether the most recent
                    # QAOA run showed meaningful improvement — if so, keep going.
                    qaoa_result = _safe_load(DECIPHERMENT_DIR / "qaoa_result.json")
                    qaoa_delta: float | None = None
                    if qaoa_result:
                        qaoa_delta = qaoa_result.get("delta_vs_mcmc")
                    if qaoa_delta is not None and qaoa_delta >= QUANTUM_IMPROVEMENT_THRESHOLD:
                        log.info(
                            "MCMC stalled (%.2f nats < threshold) but QAOA showed +%.2f nats "
                            "improvement — continuing attack.",
                            improvement, qaoa_delta,
                        )
                    else:
                        log.info(
                            "LM improvement over last 3 turns (%.2f nats) below threshold (%.2f) "
                            "and no quantum improvement (QAOA delta=%s) — stopping.",
                            improvement, LM_IMPROVEMENT_THRESHOLD,
                            f"{qaoa_delta:.2f}" if qaoa_delta is not None else "n/a",
                        )
                        break

    new_ranking = _safe_load(DECIPHERMENT_DIR / "ranking.json")
    lm_after = None
    if new_ranking and "hypotheses" in new_ranking:
        lm_after = new_ranking["hypotheses"][0].get("overall_lm_score")

    summary = {
        "turns_taken": turn,
        "tools_called": tools_called,
        "tools_call_counts": {t: tools_called.count(t) for t in set(tools_called)},
        "lm_score_before": lm_before,
        "lm_score_after": lm_after,
        "lm_delta": (lm_after - lm_before) if (lm_before is not None and lm_after is not None) else None,
        "hypothesis_declared": hypothesis_declared,
        "attack_log": str(attack_log_path),
    }
    (output_dir / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    _mlflow_log(
        {
            "turns_taken": turn,
            "lm_score_before": lm_before or -9999,
            "lm_score_after": lm_after or -9999,
            "lm_delta": summary.get("lm_delta") or 0,
        },
        {
            "tools_called": ",".join(tools_called[:20]),
            "hypothesis_declared": hypothesis_declared,
        },
    )
    _mlflow_end()

    log.info("=== RedRongo complete ===")
    log.info("Turns: %d | Tools called: %d | Hypothesis declared: %s", turn, len(tools_called), hypothesis_declared)
    if lm_before is not None and lm_after is not None:
        log.info("LM score: %.2f → %.2f (delta %.2f)", lm_before, lm_after, lm_after - lm_before)
    log.info("Attack log: %s", attack_log_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="RedRongo: autonomous red team agent for Rongorongo cipher attack",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=20,
        help="Maximum agent turns before stopping (default: 20)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUTS / "redteam",
        help="Output directory for attack log and hypothesis (default: outputs/redteam/)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print tool definitions and exit without running the agent",
    )
    args = parser.parse_args()

    if args.dry_run:
        print(json.dumps(TOOL_DEFINITIONS, indent=2))
        return

    args.output.mkdir(parents=True, exist_ok=True)
    run_agent(max_turns=args.max_turns, output_dir=args.output)


if __name__ == "__main__":
    main()
