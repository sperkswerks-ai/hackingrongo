"""
scripts/run_self_training.py

Ensemble self-training loop for rongorongo phoneme decipherment.

The idea
--------
A single MCMC run produces N hypotheses that each agree on some assignments
and disagree on others.  Where they agree at high confidence we have a
strong signal — stronger than any single hypothesis.  Self-training
promotes those consensus assignments to anchors and re-runs, letting the
next iteration explore the remaining uncertain positions with the agreed-on
positions locked or damped.  Iterate until convergence.

Algorithm (per iteration k)
---------------------------
  1. MCMC sampling with the current crib set C_k and proposal weights W_k.
  2. Beam-search refinement seeded from the top MCMC samples.
  3. Score every hypothesis (MCMC ∪ beam) against the KN-smoothed LM.
  4. Consensus extraction:
       Soft promotion  — sign appears in ≥ min_consensus of the top-K
                         hypotheses with the same phoneme AND mean MCMC
                         confidence ≥ threshold_k AND evidence ≥ min_evidence.
                         → phoneme gets a global proposal boost in W_{k+1}
                         → sign's IC weight is damped (less re-exploration)
       Hard graduation — sign was SOFT last round AND is now unanimous
                         across all top-K AND confidence hasn't dropped.
                         → added to C_{k+1} as a permanent crib
  5. C_{k+1} = C_k ∪ new hard cribs
     W_{k+1} = W_k ∪ phoneme boosts for new soft anchors

Convergence
-----------
  Stop when no new promotions are made OR the top-hypothesis LM score
  improvement is < delta_tol OR we reach max_iterations.

Error-propagation defences
--------------------------
  * Confidence threshold starts at threshold_start (default 0.90) and
    relaxes linearly to threshold_end (default 0.70) — conservative early.
  * Only signs with evidence_count ≥ min_evidence (default 10) enter the
    candidate pool.
  * Hard graduation requires unanimity across ALL top-K in the CURRENT
    round plus consensus in the PREVIOUS round.
  * Maximum new promotions per round: max_new_soft (8) and max_new_hard (3).
  * Base calendar cribs and soft anchors are never overridden.
  * All promotions are recorded for the audit trail.

Output
------
  outputs/self_training/
    iter_{n:02d}/ranking.json
    iter_{n:02d}/mcmc_diagnostics.json
    self_training_summary.json
    self_training_report.html

Usage
-----
    python scripts/run_self_training.py
    python scripts/run_self_training.py --max-iterations 4 --top-k 5
    python scripts/run_self_training.py --smoke-test
"""

from __future__ import annotations

import argparse
import html as _html
import json
import logging
import math
import os
import statistics
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))  # make scripts/ importable

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Re-use infrastructure from run_decipherment
# ---------------------------------------------------------------------------

from run_decipherment import (   # noqa: E402
    CALENDAR_ANCHORS_HARD,
    CALENDAR_ANCHORS_SOFT,
    _CALENDAR_SOFT_BOOST,
    _DEFAULT_PHONEME_INVENTORY,
    _build_corpus_sequences,
    _build_stratum_scores,
    _make_hypothesis,
    _score_map_all_tablets,
    _validate_anchors,
)

# ---------------------------------------------------------------------------
# Self-training constants
# ---------------------------------------------------------------------------

SOFT_BOOST_SCALE   = 3.0   # confidence × scale → phoneme proposal weight
SOFT_IC_DAMP       = 0.20  # IC weight for soft-anchored signs (vs baseline 1.0)
GRAD_THRESHOLD     = 0.85  # hard-graduation confidence floor
MAX_NEW_SOFT       = 8     # cap per iteration
MAX_NEW_HARD       = 3     # cap per iteration
SCORE_PLATEAU_TOL  = 1e-4  # minimum LM improvement to continue


# ---------------------------------------------------------------------------
# State dataclasses
# ---------------------------------------------------------------------------

@dataclass
class Promotion:
    sign: str
    phoneme: str
    confidence: float
    consensus_count: int      # how many of top-K agreed
    top_k: int
    iteration: int
    kind: str                 # "soft" | "hard"


@dataclass
class IterationResult:
    iteration: int
    top_lm_score: float
    n_hard_cribs: int          # total (calendar + self-training)
    n_soft_anchors: int
    new_soft: list[Promotion]
    new_hard: list[Promotion]
    mcmc_converged: bool
    rhat: float | None
    acceptance_mean: float
    n_hypotheses_scored: int


@dataclass
class SelfTrainingState:
    hard_cribs:      dict[str, str]                     = field(default_factory=dict)
    soft_anchors:    dict[str, tuple[str, float]]       = field(default_factory=dict)
    prev_soft:       dict[str, tuple[str, float]]       = field(default_factory=dict)
    history:         list[IterationResult]              = field(default_factory=list)
    convergence:     str                                = "running"


# ---------------------------------------------------------------------------
# Proposal weight builders
# ---------------------------------------------------------------------------

def _build_priors(
    phoneme_inventory: list[str],
    soft_anchors: dict[str, tuple[str, float]],
) -> list[float]:
    """Merge calendar soft boosts with self-training soft boosts."""
    weights = {ph: _CALENDAR_SOFT_BOOST.get(ph, 1.0) for ph in phoneme_inventory}
    for _sign, (phoneme, conf) in soft_anchors.items():
        if phoneme in weights:
            weights[phoneme] = max(weights[phoneme], conf * SOFT_BOOST_SCALE)
    return [weights[ph] for ph in phoneme_inventory]


def _build_ic_weights(
    sign_ids: list[str],
    soft_anchors: dict[str, tuple[str, float]],
    hard_cribs: dict[str, str],
) -> dict[str, float]:
    """Low IC weight for soft-anchored signs — keep them sticky without pinning."""
    weights: dict[str, float] = {}
    for sign in sign_ids:
        if sign in hard_cribs:
            continue  # cribs are excluded from proposals anyway
        if sign in soft_anchors:
            weights[sign] = SOFT_IC_DAMP
    return weights


# ---------------------------------------------------------------------------
# Consensus extraction
# ---------------------------------------------------------------------------

def _extract_consensus(
    hypotheses: list[Any],
    all_cribs: dict[str, str],
    prev_soft: dict[str, tuple[str, float]],
    threshold: float,
    min_consensus: int,
    min_evidence: int,
    iteration: int,
) -> tuple[list[Promotion], list[Promotion]]:
    """Identify new soft promotions and hard graduations from the top-K hypotheses.

    Returns (new_soft, new_hard) — sorted by confidence descending, capped.
    """
    k = len(hypotheses)

    # Tally votes and confidence across hypotheses
    votes:     dict[str, Counter]          = defaultdict(Counter)
    conf_sums: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    evidence:  dict[str, int]              = {}

    for hyp in hypotheses:
        for asgn in hyp.assignments:
            s, p = asgn.sign_code, asgn.phoneme
            if p == "<UNK>":
                continue
            votes[s][p] += 1
            conf_sums[s][p] += asgn.confidence
            evidence[s] = max(evidence.get(s, 0), asgn.evidence_count)

    new_soft: list[Promotion] = []
    new_hard: list[Promotion] = []

    for sign, vote_counter in votes.items():
        if sign in all_cribs:
            continue
        if evidence.get(sign, 0) < min_evidence:
            continue

        best_phoneme, best_count = vote_counter.most_common(1)[0]
        if best_phoneme == "<UNK>":
            continue

        mean_conf = conf_sums[sign][best_phoneme] / max(best_count, 1)

        # Hard graduation: unanimous now + was soft last round + confidence holds
        if (
            best_count == k
            and sign in prev_soft
            and prev_soft[sign][0] == best_phoneme
            and mean_conf >= GRAD_THRESHOLD
        ):
            new_hard.append(Promotion(
                sign=sign, phoneme=best_phoneme, confidence=mean_conf,
                consensus_count=best_count, top_k=k,
                iteration=iteration, kind="hard",
            ))
            continue

        # Soft promotion
        if best_count >= min_consensus and mean_conf >= threshold:
            new_soft.append(Promotion(
                sign=sign, phoneme=best_phoneme, confidence=mean_conf,
                consensus_count=best_count, top_k=k,
                iteration=iteration, kind="soft",
            ))

    # Sort by confidence descending; apply caps
    new_soft.sort(key=lambda p: -p.confidence)
    new_hard.sort(key=lambda p: -p.confidence)
    return new_soft[:MAX_NEW_SOFT], new_hard[:MAX_NEW_HARD]


# ---------------------------------------------------------------------------
# Single iteration runner
# ---------------------------------------------------------------------------

def _run_iteration(
    cfg:               Any,
    project_root:      Path,
    out_dir:           Path,
    all_tablets:       list,
    corpus_sequences:  list[list[str]],
    sign_ids:          list[str],
    lm_scorer:         Any,
    decoder:           Any,
    state:             SelfTrainingState,
    top_k:             int,
    config_hash:       str,
    smoke_test:        bool,
    iteration:         int,
) -> tuple[list[Any], Any, Any]:
    """Run one MCMC + beam iteration. Returns (ranked_hypotheses, mcmc_result, beam_result)."""
    from hackingrongo.zone_c.mcmc import MCMCSampler

    # Build current anchor set
    all_cribs = {**CALENDAR_ANCHORS_HARD, **state.hard_cribs}

    # Build phoneme inventory
    _anchor_extras = [
        ph for ph in (
            set(CALENDAR_ANCHORS_HARD.values()) |
            {ph for ph, _ in CALENDAR_ANCHORS_SOFT.values()} |
            {ph for ph, _ in state.soft_anchors.values()}
        )
        if ph not in _DEFAULT_PHONEME_INVENTORY
    ]
    phoneme_inventory = list(_DEFAULT_PHONEME_INVENTORY) + _anchor_extras

    _validate_anchors(all_cribs, phoneme_inventory, label=f"iter {iteration} cribs")

    phoneme_priors = _build_priors(phoneme_inventory, state.soft_anchors)
    ic_weights     = _build_ic_weights(sign_ids, state.soft_anchors, all_cribs)

    active_cribs = {k: v for k, v in all_cribs.items() if k in sign_ids}

    log.info(
        "[iter %d] Hard cribs: %d | Soft anchors: %d | Phoneme inventory: %d",
        iteration, len(active_cribs), len(state.soft_anchors), len(phoneme_inventory),
    )

    sampler = MCMCSampler(
        cfg=cfg,
        lm_scorer=lm_scorer,
        corpus_sequences=corpus_sequences,
        sign_ids=sign_ids,
        phoneme_inventory=phoneme_inventory,
        phoneme_priors=phoneme_priors,
        sign_ic_weights=ic_weights if ic_weights else None,
        cribs=active_cribs,
        seed=int(cfg.seed) + iteration * 1000,
    )

    mcmc_result = sampler.run()
    rhat_str = (
        f"{mcmc_result.gelman_rubin_rhat:.4f}"
        if mcmc_result.gelman_rubin_rhat is not None else "N/A"
    )
    acc_mean = (
        statistics.mean(mcmc_result.acceptance_rates)
        if mcmc_result.acceptance_rates else 0.0
    )
    log.info(
        "[iter %d] MCMC done: %d sample(s), R-hat=%s, converged=%s, acceptance=%.3f",
        iteration, len(mcmc_result.top_samples), rhat_str,
        mcmc_result.converged, acc_mean,
    )

    beam_result = decoder.decode(
        sign_ids=sign_ids,
        corpus_sequences=corpus_sequences,
        seed_hypotheses=mcmc_result.top_samples,
    )
    log.info(
        "[iter %d] Beam done: %d hypothesis/es, %d step(s), early_stop=%s",
        iteration, len(beam_result.top_hypotheses),
        beam_result.n_steps, beam_result.early_stopped,
    )

    # Build and score hypothesis pool
    hyp_pool: dict[tuple, Any] = {}

    for sample in mcmc_result.top_samples:
        key = tuple(sorted(sample.phoneme_map.items()))
        hyp_pool[key] = _make_hypothesis(
            run_id=f"st_iter{iteration:02d}",
            phoneme_map=sample.phoneme_map,
            mcmc_log_posterior=sample.log_posterior,
            beam_score=0.0,
            sign_ids=sign_ids,
            corpus_sequences=corpus_sequences,
            all_tablets=all_tablets,
            lm_scorer=lm_scorer,
            mcmc_samples=mcmc_result.top_samples,
            config_hash=config_hash,
        )

    def _seed_lp(pm: dict) -> float:
        best_lp, best_ov = -math.inf, -1
        for s in mcmc_result.top_samples:
            ov = sum(1 for sg, ph in pm.items() if s.phoneme_map.get(sg) == ph)
            if ov > best_ov:
                best_ov, best_lp = ov, s.log_posterior
        return best_lp

    for bhyp in beam_result.top_hypotheses:
        key = tuple(sorted(bhyp.phoneme_map.items()))
        if key in hyp_pool:
            hyp_pool[key].beam_score = round(bhyp.log_score, 6)
        else:
            hyp_pool[key] = _make_hypothesis(
                run_id=f"st_iter{iteration:02d}",
                phoneme_map=bhyp.phoneme_map,
                mcmc_log_posterior=_seed_lp(bhyp.phoneme_map),
                beam_score=bhyp.log_score,
                sign_ids=sign_ids,
                corpus_sequences=corpus_sequences,
                all_tablets=all_tablets,
                lm_scorer=lm_scorer,
                mcmc_samples=mcmc_result.top_samples,
                config_hash=config_hash,
            )

    ranked = sorted(
        hyp_pool.values(),
        key=lambda h: h.overall_lm_score if math.isfinite(h.overall_lm_score) else -math.inf,
        reverse=True,
    )[:top_k]

    for i, hyp in enumerate(ranked, 1):
        hyp.hypothesis_id = f"ST{iteration:02d}H{i:04d}"

    # Save iteration outputs
    iter_dir = out_dir / f"iter_{iteration:02d}"
    iter_dir.mkdir(parents=True, exist_ok=True)

    from hackingrongo.results.schema import HypothesisRanking
    ranking_obj = HypothesisRanking(hypotheses=ranked, ranking_metric="overall_lm_score")
    ranking_obj.save(iter_dir / "ranking.json")

    diag = {
        "iteration":      iteration,
        "n_chains":       mcmc_result.n_chains,
        "n_samples":      len(mcmc_result.top_samples),
        "rhat":           (round(mcmc_result.gelman_rubin_rhat, 4)
                           if mcmc_result.gelman_rubin_rhat is not None else None),
        "converged":      mcmc_result.converged,
        "acceptance_mean": round(acc_mean, 4),
        "n_hard_cribs":   len(active_cribs),
        "n_soft_anchors": len(state.soft_anchors),
        "hard_cribs":     dict(active_cribs),
        "soft_anchors":   {s: {"phoneme": ph, "confidence": round(cf, 4)}
                           for s, (ph, cf) in state.soft_anchors.items()},
    }
    (iter_dir / "mcmc_diagnostics.json").write_text(
        json.dumps(diag, indent=2), encoding="utf-8"
    )

    if ranked:
        log.info(
            "[iter %d] Top hypothesis %s: LM score %.4f",
            iteration, ranked[0].hypothesis_id, ranked[0].overall_lm_score,
        )

    return ranked, mcmc_result, beam_result


# ---------------------------------------------------------------------------
# Terminal output
# ---------------------------------------------------------------------------

_SEP = "─" * 72


def _print_iteration_banner(
    iteration: int,
    result: IterationResult,
    state: SelfTrainingState,
) -> None:
    print(f"\n{'═' * 72}")
    print(f"  Self-Training  ·  Iteration {iteration}")
    print(f"  Top LM score   : {result.top_lm_score:.6f}")
    print(f"  Hard cribs     : {result.n_hard_cribs}  (calendar + self-training)")
    print(f"  Soft anchors   : {result.n_soft_anchors}")
    print(f"  MCMC converged : {result.mcmc_converged}"
          + (f"  R-hat={result.rhat:.4f}" if result.rhat else ""))
    print(f"  Acceptance     : {result.acceptance_mean:.3f}")
    print(_SEP)

    if result.new_hard:
        print("  HARD graduations (will be pinned next round):")
        for p in result.new_hard:
            print(f"    {p.sign:>8}  →  {p.phoneme:<14}  conf={p.confidence:.3f}"
                  f"  ({p.consensus_count}/{p.top_k} unanimous)")
    else:
        print("  No hard graduations.")

    if result.new_soft:
        print("  SOFT promotions (phoneme boost + IC damp):")
        for p in result.new_soft:
            print(f"    {p.sign:>8}  →  {p.phoneme:<14}  conf={p.confidence:.3f}"
                  f"  ({p.consensus_count}/{p.top_k} agree)")
    else:
        print("  No soft promotions.")

    print(f"{'═' * 72}")


def _print_final_summary(state: SelfTrainingState) -> None:
    print(f"\n{'╔' + '═'*70 + '╗'}")
    print(f"  SELF-TRAINING COMPLETE  ·  {len(state.history)} iteration(s)")
    print(f"  Convergence: {state.convergence}")
    print(_SEP)
    print(f"  {'Iter':>4}  {'Top LM':>10}  {'Hard':>5}  {'Soft':>5}  "
          f"{'+Hard':>5}  {'+Soft':>5}")
    print(f"  {'─'*4}  {'─'*10}  {'─'*5}  {'─'*5}  {'─'*5}  {'─'*5}")
    for r in state.history:
        print(f"  {r.iteration:>4}  {r.top_lm_score:>10.6f}  "
              f"{r.n_hard_cribs:>5}  {r.n_soft_anchors:>5}  "
              f"{len(r.new_hard):>5}  {len(r.new_soft):>5}")
    print(_SEP)
    if state.hard_cribs:
        print("  Self-training hard cribs:")
        for sign, ph in sorted(state.hard_cribs.items()):
            print(f"    {sign:>8} → {ph}")
    if state.soft_anchors:
        print("  Residual soft anchors:")
        for sign, (ph, cf) in sorted(state.soft_anchors.items()):
            print(f"    {sign:>8} → {ph}  conf={cf:.3f}")
    print(f"{'╚' + '═'*70 + '╝'}\n")


# ---------------------------------------------------------------------------
# HTML report
# ---------------------------------------------------------------------------

_CSS = """\
:root{--bg:#0d0f12;--surface:#161920;--surface2:#1e2229;--border:#2a2e38;
      --text:#d0d4dc;--muted:#6b7280;--accent:#c4a96d;
      --green:#4ade80;--yellow:#facc15;--red:#f87171;--blue:#93c5fd;}
*{box-sizing:border-box;margin:0;padding:0;}
body{background:var(--bg);color:var(--text);
     font-family:'JetBrains Mono',monospace;font-size:12px;line-height:1.65;}
.wrap{max-width:1060px;margin:0 auto;padding:52px 28px;}
h1{font-size:22px;color:var(--accent);margin-bottom:6px;}
.sub{color:var(--muted);font-size:10px;margin-bottom:36px;}
.mission{background:var(--surface);border:1px solid var(--border);
         border-left:3px solid var(--red);border-radius:0 5px 5px 0;
         padding:18px 22px;margin-bottom:36px;}
.mission-title{color:var(--red);font-size:13px;margin-bottom:10px;}
.meta{color:var(--muted);font-size:10px;line-height:2;}
.meta b{color:#aaa;}
.iter-card{background:var(--surface);border:1px solid var(--border);
           border-radius:5px;margin-bottom:24px;overflow:hidden;}
.iter-header{padding:10px 18px;border-bottom:1px solid var(--border);
             display:flex;gap:18px;align-items:baseline;}
.iter-num{color:var(--accent);font-size:16px;font-weight:600;}
.iter-score{color:var(--green);}
.iter-body{padding:14px 18px;}
.stat-row{display:flex;gap:14px;flex-wrap:wrap;margin-bottom:14px;}
.stat{background:var(--surface2);border:1px solid var(--border);
      border-radius:4px;padding:10px 16px;min-width:120px;}
.stat-label{font-size:9px;color:var(--muted);text-transform:uppercase;}
.stat-value{font-size:22px;font-weight:600;margin-top:2px;color:var(--accent);}
.stat-sub{font-size:9px;color:var(--muted);margin-top:1px;}
table{width:100%;border-collapse:collapse;margin-top:10px;}
th{padding:5px 10px;text-align:left;font-size:9px;color:var(--muted);
   border-bottom:1px solid var(--border);text-transform:uppercase;}
td{padding:4px 10px;border-bottom:1px solid rgba(42,46,56,.3);}
.code{color:var(--accent);}  .ph{color:var(--blue);}
.hard-badge{background:rgba(74,222,128,.15);color:var(--green);
            font-size:9px;padding:1px 6px;border-radius:2px;}
.soft-badge{background:rgba(250,204,21,.1);color:var(--yellow);
            font-size:9px;padding:1px 6px;border-radius:2px;}
.cal-badge{background:rgba(147,197,253,.12);color:var(--blue);
           font-size:9px;padding:1px 6px;border-radius:2px;}
.verdict{border-left:3px solid var(--accent);padding:14px 20px;
         background:var(--surface);border-radius:0 5px 5px 0;margin:24px 0;}
.verdict strong{color:var(--accent);}
"""


def _build_html_report(
    state: SelfTrainingState,
    generated: str,
    args_meta: dict,
) -> str:
    # Mission brief
    conv_colour = {"no_new_promotions": "green", "score_plateau": "yellow",
                   "max_iterations": "red"}.get(state.convergence, "muted")
    mission = (
        f'<div class="mission">'
        f'<div class="mission-title">// SELF-TRAINING RUN</div>'
        f'<div class="meta">'
        f'<b>Generated</b> {_html.escape(generated)}<br>'
        f'<b>Iterations run</b> {len(state.history)}<br>'
        f'<b>Convergence</b> <span style="color:var(--{conv_colour})">'
        f'{_html.escape(state.convergence)}</span><br>'
        f'<b>Base calendar cribs</b> {len(CALENDAR_ANCHORS_HARD)}<br>'
        f'<b>Self-training hard cribs</b> {len(state.hard_cribs)}<br>'
        f'<b>Residual soft anchors</b> {len(state.soft_anchors)}<br>'
        + "".join(
            f"<b>{_html.escape(k)}</b> {_html.escape(str(v))}<br>"
            for k, v in args_meta.items()
        )
        + "</div></div>"
    )

    # Score trajectory
    scores = [r.top_lm_score for r in state.history if math.isfinite(r.top_lm_score)]
    if len(scores) >= 2:
        total_delta = scores[-1] - scores[0]
        delta_str = f"{total_delta:+.6f}"
        traj_html = (
            f'<div style="background:var(--surface2);border:1px solid var(--border);'
            f'border-radius:4px;padding:12px 18px;margin-bottom:24px;">'
            f'<span style="color:var(--muted);font-size:10px">Score trajectory: </span>'
            + " → ".join(f'<span style="color:var(--green)">{s:.4f}</span>' for s in scores)
            + f'<span style="color:var(--muted);font-size:10px"> &nbsp; Δ = '
            f'<span style="color:{"var(--green)" if total_delta >= 0 else "var(--red)"}">'
            f"{delta_str}</span></span></div>"
        )
    else:
        traj_html = ""

    # Per-iteration cards
    cards = ""
    for r in state.history:
        conv_flag = ""
        if r.iteration == len(state.history) - 1:
            conv_flag = (
                f'<span style="color:var(--{conv_colour});margin-left:12px;font-size:10px">'
                f'[{_html.escape(state.convergence)}]</span>'
            )

        promo_rows = ""
        for p in r.new_hard + r.new_soft:
            badge = (
                '<span class="hard-badge">HARD</span>' if p.kind == "hard"
                else '<span class="soft-badge">SOFT</span>'
            )
            promo_rows += (
                f'<tr>'
                f'<td class="code">{_html.escape(p.sign)}</td>'
                f'<td class="ph">{_html.escape(p.phoneme)}</td>'
                f'<td>{badge}</td>'
                f'<td>{p.confidence:.3f}</td>'
                f'<td>{p.consensus_count}/{p.top_k}</td>'
                f'</tr>'
            )
        promo_table = (
            '<table><thead><tr>'
            '<th>Sign</th><th>Phoneme</th><th>Kind</th>'
            '<th>Confidence</th><th>Consensus</th>'
            f'</tr></thead><tbody>{promo_rows}</tbody></table>'
        ) if promo_rows else '<p style="color:var(--muted);margin-top:8px">No new promotions.</p>'

        rhat_str = f"{r.rhat:.4f}" if r.rhat else "n/a"
        cards += (
            f'<div class="iter-card">'
            f'<div class="iter-header">'
            f'<span class="iter-num">Iter {r.iteration}</span>'
            f'<span class="iter-score">{r.top_lm_score:.6f}</span>'
            f'<span style="color:var(--muted);font-size:10px">'
            f'R-hat {rhat_str} · acc {r.acceptance_mean:.3f} · '
            f'converged {r.mcmc_converged}</span>'
            + conv_flag
            + "</div>"
            f'<div class="iter-body">'
            f'<div class="stat-row">'
            f'<div class="stat"><div class="stat-label">Hard cribs</div>'
            f'<div class="stat-value">{r.n_hard_cribs}</div></div>'
            f'<div class="stat"><div class="stat-label">Soft anchors</div>'
            f'<div class="stat-value">{r.n_soft_anchors}</div></div>'
            f'<div class="stat"><div class="stat-label">+Hard</div>'
            f'<div class="stat-value" style="color:var(--green)">{len(r.new_hard)}</div></div>'
            f'<div class="stat"><div class="stat-label">+Soft</div>'
            f'<div class="stat-value" style="color:var(--yellow)">{len(r.new_soft)}</div></div>'
            f'<div class="stat"><div class="stat-label">Hypotheses</div>'
            f'<div class="stat-value">{r.n_hypotheses_scored}</div></div>'
            f'</div>'
            + promo_table
            + "</div></div>"
        )

    # Final anchor set
    cal_rows = "".join(
        f'<tr><td class="code">{_html.escape(s)}</td>'
        f'<td class="ph">{_html.escape(ph)}</td>'
        f'<td><span class="cal-badge">CALENDAR</span></td>'
        f'<td>1.000</td><td>—</td></tr>'
        for s, ph in sorted(CALENDAR_ANCHORS_HARD.items())
    )
    st_hard_rows = "".join(
        f'<tr><td class="code">{_html.escape(s)}</td>'
        f'<td class="ph">{_html.escape(ph)}</td>'
        f'<td><span class="hard-badge">SELF-TRAINING</span></td>'
        f'<td>—</td>'
        f'<td>{_html.escape(_find_promotion_iter(state, s, "hard"))}</td></tr>'
        for s, ph in sorted(state.hard_cribs.items())
    )
    st_soft_rows = "".join(
        f'<tr><td class="code">{_html.escape(s)}</td>'
        f'<td class="ph">{_html.escape(ph)}</td>'
        f'<td><span class="soft-badge">SOFT</span></td>'
        f'<td>{cf:.3f}</td>'
        f'<td>{_html.escape(_find_promotion_iter(state, s, "soft"))}</td></tr>'
        for s, (ph, cf) in sorted(state.soft_anchors.items())
    )
    anchor_table = (
        '<table><thead><tr>'
        '<th>Sign</th><th>Phoneme</th><th>Source</th>'
        '<th>Confidence</th><th>Promoted at</th>'
        f'</tr></thead><tbody>{cal_rows}{st_hard_rows}{st_soft_rows}</tbody></table>'
    )

    verdict_text = {
        "no_new_promotions": (
            "Loop converged cleanly: no new consensus assignments emerged in the final "
            "iteration. The self-training process exhausted the extractable signal — "
            "remaining uncertain positions require additional corpus evidence or a "
            "revised phoneme inventory."
        ),
        "score_plateau": (
            f"Loop stopped on score plateau: top-hypothesis LM improvement fell below "
            f"{SCORE_PLATEAU_TOL}. The decipherment has reached a local optimum under "
            "the current anchor set and corpus."
        ),
        "max_iterations": (
            "Maximum iterations reached. Re-run with --max-iterations to continue, "
            "or inspect the anchor audit trail for remaining uncertain positions."
        ),
    }.get(state.convergence, f"Convergence: {state.convergence}")

    return (
        "<!DOCTYPE html><html lang='en'>"
        "<head><meta charset='utf-8'>"
        "<title>Rongorongo — Self-Training Report</title>"
        f"<style>{_CSS}</style></head>"
        "<body><div class='wrap'>"
        "<h1>Rongorongo Decipherment · Ensemble Self-Training</h1>"
        "<div class='sub'>"
        "MCMC → consensus extraction → anchor promotion → re-run · "
        "iterates until convergence"
        "</div>"
        + mission
        + traj_html
        + cards
        + '<div class="section" style="margin-top:32px">'
        '<div style="font-size:14px;color:var(--text);border-bottom:1px solid var(--border);'
        'padding-bottom:8px;margin-bottom:16px">Final Anchor Set</div>'
        + anchor_table
        + "</div>"
        + f'<div class="verdict" style="margin-top:28px"><strong>Convergence verdict</strong>'
          f'<p style="font-size:12px;margin-top:8px">{_html.escape(verdict_text)}</p></div>'
        + "</div></body></html>"
    )


def _find_promotion_iter(state: SelfTrainingState, sign: str, kind: str) -> str:
    for r in state.history:
        pool = r.new_hard if kind == "hard" else r.new_soft
        if any(p.sign == sign for p in pool):
            return f"iter {r.iteration}"
    return "?"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Ensemble self-training loop for rongorongo phoneme decipherment.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--max-iterations",  type=int,   default=5)
    p.add_argument("--top-k",           type=int,   default=5,
                   help="Number of top hypotheses used for consensus extraction.")
    p.add_argument("--min-consensus",   type=int,   default=2,
                   help="Minimum hypotheses agreeing on a phoneme for soft promotion.")
    p.add_argument("--min-evidence",    type=int,   default=10,
                   help="Minimum corpus evidence count for a sign to be promoted.")
    p.add_argument("--threshold-start", type=float, default=0.90,
                   help="Confidence threshold at iteration 0 (conservative).")
    p.add_argument("--threshold-end",   type=float, default=0.70,
                   help="Confidence threshold at max-iterations (permissive).")
    p.add_argument("--output-dir", type=Path,
                   default=PROJECT_ROOT / "outputs" / "self_training")
    p.add_argument("--smoke-test", action="store_true",
                   help="Tablet D only, 1 chain × 200 iterations, 2 self-training rounds.")
    p.add_argument("--chains", type=int, default=None,
                   help="Override zone_c.mcmc.num_chains (default: from config.yaml).")
    p.add_argument("--iterations", type=int, default=None,
                   help="Override zone_c.mcmc.num_iterations (default: from config.yaml).")
    p.add_argument("--beam-width", type=int, default=None,
                   help="Override zone_c.beam_search.beam_width (default: from config.yaml).")
    p.add_argument("--beam-depth", type=int, default=None,
                   help="Override zone_c.beam_search.max_depth. "
                        "Use 50-100 for full-corpus runs to avoid 30+ min beam search.")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = _parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # ── Config ───────────────────────────────────────────────────────────────
    from omegaconf import OmegaConf
    cfg_path = PROJECT_ROOT / "conf" / "config.yaml"
    cfg = OmegaConf.load(cfg_path)

    if args.smoke_test:
        log.info("Smoke-test mode: 1 chain × 200 iter, 2 self-training rounds.")
        cfg = OmegaConf.merge(cfg, OmegaConf.create({
            "zone_c": {
                "mcmc": {
                    "num_chains": 1,
                    "num_iterations": 200,
                    "burn_in": 50,
                    "thin": 2,
                    "top_k": args.top_k,
                },
                "beam_search": {"beam_width": 3, "max_depth": 15},
                "validation": {"top_n_hypotheses": args.top_k},
            }
        }))
        args.max_iterations = 2
        args.threshold_start = 0.70
        args.threshold_end   = 0.60
        args.min_evidence    = 3

    # Apply explicit chain/iteration/beam overrides (for full-corpus runs at tractable scale)
    mcmc_overrides: dict = {}
    beam_overrides: dict = {}
    if args.chains is not None:
        mcmc_overrides["num_chains"] = args.chains
    if args.iterations is not None:
        mcmc_overrides["num_iterations"] = args.iterations
        mcmc_overrides["burn_in"] = max(50, args.iterations // 10)
        mcmc_overrides["thin"]    = max(1,  args.iterations // 500)
    if args.beam_width is not None:
        beam_overrides["beam_width"] = args.beam_width
    if args.beam_depth is not None:
        beam_overrides["max_depth"] = args.beam_depth
    override_payload: dict = {}
    if mcmc_overrides:
        override_payload["mcmc"] = mcmc_overrides
    if beam_overrides:
        override_payload["beam_search"] = beam_overrides
    if override_payload:
        cfg = OmegaConf.merge(cfg, OmegaConf.create({"zone_c": override_payload}))
        log.info(
            "Config overrides: MCMC %s  beam %s",
            {k: cfg.zone_c.mcmc[k] for k in mcmc_overrides},
            {k: cfg.zone_c.beam_search[k] for k in beam_overrides},
        )

    from hackingrongo.data.corpus import load_corpus, split_by_cluster
    from hackingrongo.results.schema import hash_config_file
    from hackingrongo.zone_c.beam_search import BeamSearchDecoder
    from hackingrongo.zone_c.lm_scoring import LMScorer

    config_hash = hash_config_file(cfg_path)

    # ── MLflow setup ─────────────────────────────────────────────────────────
    _mlflow_active = False
    try:
        import mlflow as _mlflow
        _tracking_uri = os.environ.get(
            "MLFLOW_TRACKING_URI",
            f"file://{(PROJECT_ROOT / 'outputs' / 'mlruns').resolve()}",
        )
        _mlflow.set_tracking_uri(_tracking_uri)
        _mlflow.set_experiment("rongorongo_decipherment")
        _run_label = "self-training-smoke" if args.smoke_test else "self-training-full"
        _ts = datetime.now(tz=timezone.utc).strftime("%Y%m%d-%H%M")
        _mlflow.start_run(run_name=f"{_run_label}-{_ts}")
        _mlflow.log_params({
            "smoke_test":        args.smoke_test,
            "max_iterations":    args.max_iterations,
            "top_k":             args.top_k,
            "min_consensus":     args.min_consensus,
            "min_evidence":      args.min_evidence,
            "threshold_start":   args.threshold_start,
            "threshold_end":     args.threshold_end,
            "mcmc.num_chains":   int(cfg.zone_c.mcmc.num_chains),
            "mcmc.num_iterations": int(cfg.zone_c.mcmc.num_iterations),
            "beam.width":        int(cfg.zone_c.beam_search.beam_width),
            "beam.max_depth":    int(cfg.zone_c.beam_search.max_depth),
            "n_hard_anchors":    len(CALENDAR_ANCHORS_HARD),
            "config_hash":       config_hash[:16],
        })
        _mlflow_active = True
        log.info("MLflow run started → %s", _tracking_uri)
    except ImportError:
        log.warning("mlflow not installed — tracking disabled.")

    # ── Corpus ───────────────────────────────────────────────────────────────
    log.info("Loading corpus …")
    all_tablets = load_corpus(cfg, PROJECT_ROOT)
    corpus_sequences, sign_ids = _build_corpus_sequences(all_tablets, args.smoke_test)

    # ── Models ───────────────────────────────────────────────────────────────
    log.info("Loading language models …")
    lm_scorer = LMScorer(cfg, PROJECT_ROOT)
    decoder   = BeamSearchDecoder(cfg=cfg, lm_scorer=lm_scorer)

    # ── Self-training state ──────────────────────────────────────────────────
    state = SelfTrainingState()

    generated = datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    prev_top_score = -math.inf

    # ── Loop ─────────────────────────────────────────────────────────────────
    for iteration in range(args.max_iterations):
        threshold = (
            args.threshold_start
            - iteration * (args.threshold_start - args.threshold_end)
            / max(args.max_iterations - 1, 1)
        )
        log.info("[iter %d] Confidence threshold: %.3f", iteration, threshold)

        ranked, mcmc_result, _ = _run_iteration(
            cfg=cfg,
            project_root=PROJECT_ROOT,
            out_dir=args.output_dir,
            all_tablets=all_tablets,
            corpus_sequences=corpus_sequences,
            sign_ids=sign_ids,
            lm_scorer=lm_scorer,
            decoder=decoder,
            state=state,
            top_k=args.top_k,
            config_hash=config_hash,
            smoke_test=args.smoke_test,
            iteration=iteration,
        )

        if not ranked:
            log.warning("[iter %d] No hypotheses produced — stopping.", iteration)
            state.convergence = "no_hypotheses"
            break

        top_score = ranked[0].overall_lm_score

        # Extract consensus
        all_cribs = {**CALENDAR_ANCHORS_HARD, **state.hard_cribs}
        acc_mean = (statistics.mean(mcmc_result.acceptance_rates)
                    if mcmc_result.acceptance_rates else 0.0)

        new_soft, new_hard = _extract_consensus(
            ranked[:args.top_k],
            all_cribs=all_cribs,
            prev_soft=state.prev_soft,
            threshold=threshold,
            min_consensus=args.min_consensus,
            min_evidence=args.min_evidence,
            iteration=iteration,
        )

        # Filter: skip signs already pinned; ensure a sign graduating to hard
        # this round is removed from new_soft to prevent double-recording.
        graduating_signs = {p.sign for p in new_hard
                            if p.sign not in state.hard_cribs
                            and p.sign not in CALENDAR_ANCHORS_HARD}
        new_soft = [p for p in new_soft
                    if p.sign not in state.soft_anchors
                    and p.sign not in state.hard_cribs
                    and p.sign not in all_cribs
                    and p.sign not in graduating_signs]
        new_hard = [p for p in new_hard
                    if p.sign not in state.hard_cribs
                    and p.sign not in CALENDAR_ANCHORS_HARD]

        result = IterationResult(
            iteration=iteration,
            top_lm_score=top_score,
            n_hard_cribs=len(all_cribs),
            n_soft_anchors=len(state.soft_anchors),
            new_soft=new_soft,
            new_hard=new_hard,
            mcmc_converged=mcmc_result.converged,
            rhat=mcmc_result.gelman_rubin_rhat,
            acceptance_mean=round(acc_mean, 4),
            n_hypotheses_scored=len(ranked),
        )
        state.history.append(result)
        _print_iteration_banner(iteration, result, state)

        # ── MLflow: per-iteration time-series metrics ─────────────────────────
        if _mlflow_active and math.isfinite(top_score):
            _iter_metrics: dict[str, float] = {
                "lm_score":         top_score,
                "n_hard_cribs":     float(len(all_cribs)),
                "n_soft_anchors":   float(len(state.soft_anchors)),
                "new_hard_count":   float(len(new_hard)),
                "new_soft_count":   float(len(new_soft)),
                "mcmc_converged":   float(int(mcmc_result.converged)),
                "mcmc_acceptance":  round(acc_mean, 4),
            }
            if mcmc_result.gelman_rubin_rhat is not None:
                _iter_metrics["mcmc_rhat"] = mcmc_result.gelman_rubin_rhat
            _mlflow.log_metrics(_iter_metrics, step=iteration)

        # Convergence checks
        if not new_soft and not new_hard:
            state.convergence = "no_new_promotions"
            break

        if (math.isfinite(top_score) and math.isfinite(prev_top_score)
                and abs(top_score - prev_top_score) < SCORE_PLATEAU_TOL):
            state.convergence = "score_plateau"
            break

        # Promote
        state.prev_soft = dict(state.soft_anchors)
        for p in new_soft:
            state.soft_anchors[p.sign] = (p.phoneme, p.confidence)
            log.info("[iter %d] +soft: %s → %s (conf=%.3f)",
                     iteration, p.sign, p.phoneme, p.confidence)
        for p in new_hard:
            state.hard_cribs[p.sign] = p.phoneme
            state.soft_anchors.pop(p.sign, None)
            log.info("[iter %d] +hard: %s → %s (conf=%.3f)",
                     iteration, p.sign, p.phoneme, p.confidence)

        prev_top_score = top_score
    else:
        state.convergence = "max_iterations"

    _print_final_summary(state)

    # ── Outputs ──────────────────────────────────────────────────────────────
    summary = {
        "generated":         generated,
        "convergence":       state.convergence,
        "n_iterations":      len(state.history),
        "self_training_hard_cribs":  state.hard_cribs,
        "residual_soft_anchors": {
            s: {"phoneme": ph, "confidence": round(cf, 4)}
            for s, (ph, cf) in state.soft_anchors.items()
        },
        "score_trajectory": [
            r.top_lm_score for r in state.history
            if math.isfinite(r.top_lm_score)
        ],
        "history": [
            {
                "iteration":        r.iteration,
                "top_lm_score":     r.top_lm_score,
                "n_hard_cribs":     r.n_hard_cribs,
                "n_soft_anchors":   r.n_soft_anchors,
                "mcmc_converged":   r.mcmc_converged,
                "rhat":             r.rhat,
                "acceptance_mean":  r.acceptance_mean,
                "n_hypotheses":     r.n_hypotheses_scored,
                "new_soft": [
                    {"sign": p.sign, "phoneme": p.phoneme,
                     "confidence": round(p.confidence, 4),
                     "consensus": f"{p.consensus_count}/{p.top_k}"}
                    for p in r.new_soft
                ],
                "new_hard": [
                    {"sign": p.sign, "phoneme": p.phoneme,
                     "confidence": round(p.confidence, 4)}
                    for p in r.new_hard
                ],
            }
            for r in state.history
        ],
    }
    summary_path = args.output_dir / "self_training_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log.info("Summary → %s", summary_path)

    # Promote the final best hypothesis to outputs/decipherment/ranking.json
    # so downstream scripts (validate_glosses_calendar, generate_final_report)
    # automatically pick up the self-training result.
    final_iter = len(state.history) - 1
    final_ranking_src = args.output_dir / f"iter_{final_iter:02d}" / "ranking.json"
    if final_ranking_src.exists():
        import os, tempfile
        decipherment_dir = PROJECT_ROOT / "outputs" / "decipherment"
        decipherment_dir.mkdir(parents=True, exist_ok=True)
        dest = decipherment_dir / "ranking.json"
        tmp_fd, tmp_name = tempfile.mkstemp(dir=decipherment_dir, suffix=".tmp")
        try:
            with os.fdopen(tmp_fd, "w", encoding="utf-8") as fh:
                fh.write(final_ranking_src.read_text(encoding="utf-8"))
            os.replace(tmp_name, dest)
            log.info("Promoted final ranking → %s", dest)
        except BaseException:
            os.unlink(tmp_name)
            raise

    args_meta = {
        "max_iterations": args.max_iterations,
        "top_k": args.top_k,
        "min_consensus": args.min_consensus,
        "min_evidence": args.min_evidence,
        "threshold_start": args.threshold_start,
        "threshold_end": args.threshold_end,
        "smoke_test": args.smoke_test,
    }
    html = _build_html_report(state, generated, args_meta)
    report_path = args.output_dir / "self_training_report.html"
    report_path.write_text(html, encoding="utf-8")
    log.info("HTML report → %s", report_path)

    # ── MLflow: summary metrics + artifacts + end run ─────────────────────────
    if _mlflow_active:
        try:
            score_traj = summary.get("score_trajectory", [])
            if score_traj:
                _mlflow.log_metric("final_lm_score", score_traj[-1])
                if len(score_traj) >= 2:
                    _mlflow.log_metric("lm_score_delta", score_traj[-1] - score_traj[0])
            _mlflow.log_metric("n_st_hard_cribs_promoted", float(len(state.hard_cribs)))
            _mlflow.log_metric("n_st_soft_anchors_final",  float(len(state.soft_anchors)))
            _mlflow.log_param("convergence", state.convergence)

            _st_artifacts = [
                summary_path,
                report_path,
                PROJECT_ROOT / "outputs" / "decipherment" / "ranking.json",
                PROJECT_ROOT / "outputs" / "analysis" / "calendar_gloss_validation.json",
            ]
            for p in _st_artifacts:
                if Path(p).exists():
                    folder = "analysis" if "analysis" in str(p) else "self_training"
                    _mlflow.log_artifact(str(p), artifact_path=folder)

            _mlflow.end_run()
            log.info("MLflow run ended. Tracking URI: %s", _mlflow.get_tracking_uri())
        except Exception as _e:
            log.warning("MLflow finalization failed: %s", _e)


if __name__ == "__main__":
    main()
