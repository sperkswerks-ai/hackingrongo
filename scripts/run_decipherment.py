"""
Zone C orchestration: MCMC phoneme-assignment sampling → beam-search
refinement → DecryptionHypothesis output.

Usage (local)
-------------
    conda run -n hackingrongo python scripts/run_decipherment.py
    conda run -n hackingrongo python scripts/run_decipherment.py --smoke-test
    conda run -n hackingrongo python scripts/run_decipherment.py \\
        zone_c.mcmc.num_iterations=100000 \\
        zone_c.mcmc.num_chains=8

Smoke-test mode
---------------
Pass ``--smoke-test`` to restrict to Tablet D, 1 chain × 500 iterations.

Focused passage mode
--------------------
Pass ``--focus-passage=P001`` (or any passage ID from parallel_variants_auto.json)
to run MCMC on just the glyph sequences from that passage across all its
tablet attestations.  This converges faster and gives cleaner results because
cross-tablet alignment directly constrains sign→phoneme mapping.
"""

from __future__ import annotations

import contextlib
import json
import logging
import math
import os
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Path bootstrap
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------------------------------
# --smoke-test: intercept before Hydra consumes argv
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# --focus-passage and --smoke-test: intercept before Hydra consumes argv.
#
# These blocks only run when this script is executed directly.  When the
# module is imported (e.g. by run_self_training.py or diagnose_anchor_conflicts.py)
# they are skipped so the importer's sys.argv is never contaminated with
# Hydra-style override strings that argparse cannot understand.
# ---------------------------------------------------------------------------

_FOCUS_PASSAGE: str | None = None
_SMOKE_TEST: bool = False

if __name__ == "__main__":
    for _arg in list(sys.argv):
        if _arg.startswith("--focus-passage="):
            _FOCUS_PASSAGE = _arg.split("=", 1)[1].strip()
            sys.argv.remove(_arg)
            break

    _SMOKE_TEST = "--smoke-test" in sys.argv
    if _SMOKE_TEST:
        sys.argv.remove("--smoke-test")
        # Minimal overrides for a fast wiring check.
        sys.argv += [
            "zone_c.mcmc.num_chains=1",   # single chain; avoids ProcessPoolExecutor fork hang
            "zone_c.mcmc.num_iterations=500",
            "zone_c.mcmc.burn_in=100",
            "zone_c.mcmc.thin=5",
            "zone_c.mcmc.top_k=5",
            "zone_c.beam_search.beam_width=3",
            "zone_c.beam_search.max_depth=15",
            "zone_c.validation.top_n_hypotheses=5",
        ]

import hydra  # noqa: E402
import numpy as np  # noqa: E402
from omegaconf import DictConfig  # noqa: E402

from hackingrongo.data.corpus import load_corpus, split_by_cluster  # noqa: E402
from hackingrongo.results.schema import (  # noqa: E402
    DecryptionHypothesis,
    HypothesisRanking,
    PhonemeAssignment,
    StratumScore,
    hash_config_file,
)
from hackingrongo.zone_c.beam_search import BeamSearchDecoder, BeamSearchResult  # noqa: E402
from hackingrongo.zone_c.lm_scoring import LMScorer, PhonemeMap  # noqa: E402
from hackingrongo.zone_c.mcmc import (  # noqa: E402
    MCMCResult,
    MCMCSample,
    MCMCSampler,
    _DEFAULT_PHONEME_INVENTORY,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Calendar anchors — known-plaintext evidence from Mamari calendar section
# ---------------------------------------------------------------------------
# Four high-confidence anchors from align_mamari_calendar.py output
# (confidence ≥ 0.87, anchor code found in Ca6–Ca9 alignment).
# Three soft priors from the same alignment at confidence 0.70.
#
# HARD anchors are pinned as cribs in every MCMC chain — the sampler never
# proposes a change away from them.  SOFT anchors boost the phoneme's weight
# in the random-reassignment proposal so the chain converges back quickly if
# another proposal moves the sign off its preferred value.
#
# Boost factors calibrated to alignment confidence scores (baseline = 1.0).
CALENDAR_ANCHORS_HARD: dict[str, str] = {
    "040": "kokore",    # 129 occurrences — hard pin required at full scale
    "152": "omotohi",   # confidence 1.000 — full moon (Rākaunui, night 15)
    "143": "huna",      # confidence 1.000 — near-full moon (Huna, night 14)
    "078": "maure",     # 20 occurrences — promoted from soft; corpus pressure too high
}

CALENDAR_ANCHORS_SOFT: dict[str, tuple[str, float]] = {
    "074": ("ohua", 0.85),   # first-quarter anchor (Ōhua context); weight increased
    "280": ("honu", 0.85),   # dark-moon turtle metaphor; Metoro recitation
    "010": ("oike", 0.85),   # lunar marker; late Ca9 dark-moon period
}

# Backward-compatible alias consumed by the mixed-model path below.
CALENDAR_ANCHORS: dict[str, str] = dict(CALENDAR_ANCHORS_HARD)

# Logographic taxograms (external evidence): pinned constraints in the mixed
# model. These signs are excluded from LM scoring in that model and only
# constrain surrounding phonemic context.
LOGOGRAPHIC_TAXOGRAMS: dict[str, str] = {
    "600": "manu",
    "700": "ika",
    "280": "honu",
    "690": "tangata manu",
}

# Phoneme → proposal weight multiplier (above the uniform baseline of 1.0).
# Hard-anchor phonemes get the strongest boost; soft-anchor phonemes moderate.
_CALENDAR_SOFT_BOOST: dict[str, float] = {
    "omotohi": 4.0,   # confidence 1.0 → strongest boost
    "huna":    4.0,   # confidence 1.0
    "maure":   4.0,   # confidence 1.0
    "kokore":  3.5,   # confidence 0.985
    "ohua":    2.0,   # soft prior, confidence 0.70
    "honu":    2.0,   # soft prior
    "oike":    2.0,   # soft prior
    "hetu":    1.8,   # P007 lunar context (Tablet D) → weak boost
}


def _validate_anchors(
    anchors: dict[str, str],
    phoneme_inventory: list[str],
    label: str = "anchors",
) -> None:
    """Raise ValueError if any anchor phoneme is absent from the inventory.

    Silent failures here cost a full MCMC run (hours).  This check fires
    at startup so misconfigured phoneme strings are caught immediately.
    """
    inv_set = set(phoneme_inventory)
    missing = [(sign, ph) for sign, ph in anchors.items() if ph not in inv_set]
    if missing:
        raise ValueError(
            f"Calendar {label} phonemes not in inventory: {missing}. "
            f"Inventory has {len(phoneme_inventory)} entries. "
            f"Check spelling against zone_c phoneme_inventory or add the "
            f"missing phoneme to _anchor_extras."
        )


def _build_anchored_initial_map(
    sign_ids: list[str],
    phoneme_inventory: list[str],
    rng: Any,
) -> dict[str, str]:
    """Random initial map with hard calendar anchors pinned to known phonemes."""
    m = {sign: rng.choices(phoneme_inventory)[0] for sign in sign_ids}
    for sign, phoneme in CALENDAR_ANCHORS_HARD.items():
        if sign in m and phoneme in phoneme_inventory:
            m[sign] = phoneme
    return m


def _build_calendar_phoneme_priors(phoneme_inventory: list[str]) -> list[float]:
    """Proposal weight vector with calendar phonemes boosted above baseline."""
    return [_CALENDAR_SOFT_BOOST.get(ph, 1.0) for ph in phoneme_inventory]


def _strip_non_scoring_signs(
    sequences: list[list[str]],
    non_scoring_signs: set[str],
) -> list[list[str]]:
    """Remove taxogram signs from LM-scored sequences.

    Used by the mixed model where logographic taxograms are fixed constraints
    rather than syllabic LM evidence.
    """
    if not non_scoring_signs:
        return sequences
    out: list[list[str]] = []
    for seq in sequences:
        pruned = [s for s in seq if s not in non_scoring_signs]
        if pruned:
            out.append(pruned)
    return out


# ---------------------------------------------------------------------------
# Corpus preparation
# ---------------------------------------------------------------------------


def _build_corpus_sequences(
    tablets: list,
    smoke_test: bool,
) -> tuple[list[list[str]], list[str]]:
    """Extract barthel-code sequences and distinct sign IDs for MCMC/beam.

    In smoke-test mode only the pre_contact tablets (Tablet D) are used
    as the sampler's search space; stratum scoring still runs over the
    full corpus so the output schema is always complete.
    """
    if smoke_test:
        by_cluster = split_by_cluster(tablets)
        target = by_cluster.get("pre_contact", []) or tablets
        log.info(
            "Smoke test: MCMC/beam restricted to %d tablet(s): %s",
            len(target), [t.tablet_id for t in target],
        )
    else:
        target = tablets

    sequences: list[list[str]] = [
        [tok.barthel_code for tok in t.tokens] for t in target
    ]
    sign_ids = sorted({code for seq in sequences for code in seq})
    log.info(
        "Corpus for sampler: %d sequence(s), %d distinct sign(s).",
        len(sequences), len(sign_ids),
    )
    return sequences, sign_ids


# ---------------------------------------------------------------------------
# Parallel variant loading
# ---------------------------------------------------------------------------


def _load_focus_passage_sequences(
    variants_path: Path,
    passage_id: str,
) -> tuple[list[list[str]], list[str]] | None:
    """Extract corpus sequences and sign_ids for a single named passage.

    Returns (sequences, sign_ids) drawn from the canonical form and all
    variant attestations of the passage, or None if the passage is not found.
    """
    if not variants_path.exists():
        log.warning("--focus-passage: %s not found.", variants_path)
        return None
    raw = json.loads(variants_path.read_text(encoding="utf-8"))
    passages = raw.get("passages", raw.get("variants", raw if isinstance(raw, list) else []))
    for entry in passages:
        if not isinstance(entry, dict):
            continue
        pid = entry.get("passage_id", entry.get("id", ""))
        if str(pid) != str(passage_id):
            continue
        seqs: list[list[str]] = []
        canonical = entry.get("canonical_form", [])
        if canonical:
            seqs.append([str(g) for g in canonical])
        for variant in entry.get("variants", []):
            form = variant.get("form", variant.get("glyphs", []))
            if form:
                seqs.append([str(g) for g in form])
        if not seqs:
            log.warning("--focus-passage: passage %s has no sequences.", passage_id)
            return None
        sign_ids = sorted({code for seq in seqs for code in seq})
        log.info(
            "Focus passage %s: %d sequence(s), %d distinct sign(s).",
            passage_id, len(seqs), len(sign_ids),
        )
        return seqs, sign_ids
    log.warning("--focus-passage: passage ID '%s' not found in %s.", passage_id, variants_path.name)
    return None


def _load_parallel_variants(path: Path) -> list[list[str]]:
    """Extract glyph-code sequences from parallel_variants_auto.json.

    Returns a flat list of passage sequences (one per variant occurrence),
    or [] when the file is absent or in an unrecognised format.  The file
    is optional — MCMC runs without it; its presence improves cross-stratum
    validation detail.
    """
    if not path.exists():
        log.info(
            "parallel_variants_auto.json not found at %s; "
            "cross-passage validation will be skipped.",
            path,
        )
        return []
    try:
        raw: Any = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        log.warning("Failed to parse %s: %s", path.name, exc)
        return []

    # Top-level may be a list of passages, or {"passages": [...]} (the format
    # written by cross_reference_parallels.py).  Older exports used "variants".
    if isinstance(raw, list):
        entries = raw
    else:
        entries = raw.get("passages", raw.get("variants", []))
    seqs: list[list[str]] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        # Collect the canonical form and every variant form.
        canonical = entry.get("canonical_form", [])
        if canonical:
            seqs.append([str(g) for g in canonical])
        for variant in entry.get("variants", []):
            form = variant.get("form", variant.get("glyphs", []))
            if form:
                seqs.append([str(g) for g in form])

    log.info(
        "Loaded %d parallel passage sequence(s) from %s.", len(seqs), path.name
    )
    return seqs


# ---------------------------------------------------------------------------
# Scoring (single pass over all tablets per hypothesis)
# ---------------------------------------------------------------------------


def _score_map_all_tablets(
    phoneme_map: PhonemeMap,
    all_tablets: list,
    lm_scorer: LMScorer,
    non_scoring_signs: set[str] | None = None,
) -> tuple[float, dict[str, list[tuple[float, float]]]]:
    """Score a phoneme map against all tablets in one pass.

    Returns
    -------
    overall_lm_score : float
        Mean ensemble log₂-probability across all tablets with finite scores.
    by_stratum : dict[str, list[tuple[float, float]]]
        ``{stratum: [(ensemble_log_prob, coverage), ...]}`` — one entry per
        tablet in that stratum (some log_probs may be ``-inf``).
    """
    overall_lps: list[float] = []
    by_stratum: dict[str, list[tuple[float, float]]] = {}

    for tablet in all_tablets:
        seq = [
            phoneme_map.get(tok.barthel_code, "<UNK>")
            for tok in tablet.tokens
            if non_scoring_signs is None or tok.barthel_code not in non_scoring_signs
        ]
        if not seq:
            continue
        result = lm_scorer.score(seq)
        lp, cov = result.ensemble_log_prob, result.coverage
        if math.isfinite(lp):
            overall_lps.append(lp)
        by_stratum.setdefault(tablet.stratum, []).append((lp, cov))

    overall = statistics.mean(overall_lps) if overall_lps else -math.inf
    return overall, by_stratum


def _build_stratum_scores(
    by_stratum: dict[str, list[tuple[float, float]]],
    lm_scorer: LMScorer,
) -> list[StratumScore]:
    """Convert per-tablet (lp, coverage) pairs into StratumScore objects."""
    scores: list[StratumScore] = []
    for stratum in sorted(by_stratum):
        if stratum == "excluded":
            continue
        pairs = by_stratum[stratum]
        lps = [lp for lp, _ in pairs if math.isfinite(lp)]
        coverages = [cov for _, cov in pairs]

        mean_lp = statistics.mean(lps) if lps else -math.inf
        std_lp = statistics.stdev(lps) if len(lps) > 1 else 0.0
        # Consistency = mean n-gram coverage across tablets in this stratum.
        consistency = statistics.mean(coverages) if coverages else 0.0
        langs_above = lm_scorer.languages_available if math.isfinite(mean_lp) else []

        scores.append(
            StratumScore(
                stratum=stratum,
                consistency_score=round(consistency, 6),
                lm_score_mean=round(mean_lp, 6) if math.isfinite(mean_lp) else -math.inf,
                lm_score_std=round(std_lp, 6),
                n_passages=len(lps),
                languages_above_baseline=langs_above,
            )
        )
    return scores


def _build_assignments(
    phoneme_map: PhonemeMap,
    sign_ids: list[str],
    corpus_sequences: list[list[str]],
    mcmc_samples: list[MCMCSample],
) -> list[PhonemeAssignment]:
    """Build PhonemeAssignment objects with posterior confidence.

    Confidence = fraction of top MCMC samples that agree with this
    sign→phoneme assignment.  For beam-refined assignments that differ
    from every MCMC sample, confidence is legitimately 0: the beam found
    a locally better assignment than the sampler explored.
    """
    n_samples = len(mcmc_samples)
    freq: dict[str, dict[str, int]] = {s: {} for s in sign_ids}
    for sample in mcmc_samples:
        for sign, ph in sample.phoneme_map.items():
            if sign in freq:
                freq[sign][ph] = freq[sign].get(ph, 0) + 1

    evidence: dict[str, int] = {s: 0 for s in sign_ids}
    for seq in corpus_sequences:
        for code in seq:
            if code in evidence:
                evidence[code] += 1

    return [
        PhonemeAssignment(
            sign_code=sign,
            phoneme=phoneme_map.get(sign, "<UNK>"),
            confidence=round(
                freq[sign].get(phoneme_map.get(sign, "<UNK>"), 0) / max(n_samples, 1),
                6,
            ),
            evidence_count=evidence.get(sign, 0),
        )
        for sign in sorted(sign_ids)
    ]


# ---------------------------------------------------------------------------
# Hypothesis construction
# ---------------------------------------------------------------------------


def _make_hypothesis(
    run_id: str,
    phoneme_map: PhonemeMap,
    mcmc_log_posterior: float,
    beam_score: float,
    sign_ids: list[str],
    corpus_sequences: list[list[str]],
    all_tablets: list,
    lm_scorer: LMScorer,
    mcmc_samples: list[MCMCSample],
    config_hash: str,
    non_scoring_signs: set[str] | None = None,
    hypothesis_type: str = "syllabic",
) -> DecryptionHypothesis:
    overall_lp, by_stratum = _score_map_all_tablets(
        phoneme_map, all_tablets, lm_scorer, non_scoring_signs=non_scoring_signs
    )
    return DecryptionHypothesis(
        hypothesis_id="",  # assigned after ranking
        run_id=run_id,
        hypothesis_type=hypothesis_type,
        assignments=_build_assignments(
            phoneme_map, sign_ids, corpus_sequences, mcmc_samples
        ),
        stratum_scores=_build_stratum_scores(by_stratum, lm_scorer),
        overall_lm_score=round(overall_lp, 6) if math.isfinite(overall_lp) else -math.inf,
        mcmc_log_posterior=round(mcmc_log_posterior, 6),
        beam_score=round(beam_score, 6),
        config_hash=config_hash,
    )


# ---------------------------------------------------------------------------
# MLflow tracking helpers
# ---------------------------------------------------------------------------

def _mlflow_tracking_uri(project_root: Path) -> str:
    """Resolve tracking URI: env var > default local path."""
    return os.environ.get(
        "MLFLOW_TRACKING_URI",
        f"file://{(project_root / 'outputs' / 'mlruns').resolve()}",
    )


@contextlib.contextmanager
def _mlflow_run(cfg: "DictConfig", project_root: Path, config_hash: str):
    """Context manager that wraps a decipherment run in an MLflow run.

    Logs all zone_c config parameters on entry, then yields.  After the
    body completes, caller is expected to call _mlflow_log_results().
    Gracefully no-ops if mlflow is unavailable.
    """
    try:
        import mlflow
    except ImportError:
        yield None
        return

    mlflow.set_tracking_uri(_mlflow_tracking_uri(project_root))
    mlflow.set_experiment("rongorongo_decipherment")

    mc = cfg.zone_c.mcmc
    bs = cfg.zone_c.beam_search
    run_label = "smoke" if _SMOKE_TEST else "full"
    if _FOCUS_PASSAGE:
        run_label = f"passage-{_FOCUS_PASSAGE}"
    ts = datetime.now(tz=timezone.utc).strftime("%Y%m%d-%H%M")
    run_name = f"decipherment-{run_label}-{ts}"

    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params({
            "smoke_test":              _SMOKE_TEST,
            "focus_passage":           _FOCUS_PASSAGE or "",
            "seed":                    int(cfg.seed),
            "mcmc.num_chains":         int(mc.num_chains),
            "mcmc.num_iterations":     int(mc.num_iterations),
            "mcmc.burn_in":            int(mc.burn_in),
            "mcmc.thin":               int(mc.thin),
            "mcmc.occupancy_weight":   float(mc.occupancy_penalty_weight),
            "mcmc.max_per_phoneme":    int(mc.max_signs_per_phoneme),
            "mcmc.lm_guided_prob":     float(getattr(mc, "lm_guided_prob", 0.0)),
            "beam.width":              int(bs.beam_width),
            "beam.max_depth":          int(bs.max_depth),
            "n_hard_anchors":          len(CALENDAR_ANCHORS_HARD),
            "n_soft_anchors":          len(CALENDAR_ANCHORS_SOFT),
            "config_hash":             config_hash[:16],
        })
        yield run


def _mlflow_log_results(
    out_dir: Path,
    project_root: Path,
    ranked: list,
    mcmc_result: Any,
    sign_ids: list,
    active_anchors: dict,
) -> None:
    """Log metrics and output artifacts after _run() completes."""
    try:
        import mlflow
    except ImportError:
        return
    if not mlflow.active_run():
        return

    # ── Metrics ──────────────────────────────────────────────────────────────
    metrics: dict[str, float] = {
        "n_hypotheses_ranked": float(len(ranked)),
        "sign_inventory_size": float(len(sign_ids)),
        "n_active_hard_anchors": float(len(active_anchors)),
    }
    if ranked:
        top = ranked[0]
        metrics.update({
            "top_lm_score":           top.overall_lm_score,
            "top_mcmc_log_posterior": top.mcmc_log_posterior,
            "top_beam_score":         top.beam_score,
        })
    if mcmc_result is not None:
        if mcmc_result.gelman_rubin_rhat is not None:
            metrics["mcmc_rhat"] = mcmc_result.gelman_rubin_rhat
        if mcmc_result.acceptance_rates:
            metrics["mcmc_acceptance_mean"] = float(
                statistics.mean(mcmc_result.acceptance_rates)
            )
        metrics["mcmc_converged"] = float(int(mcmc_result.converged))
    mlflow.log_metrics({k: v for k, v in metrics.items() if math.isfinite(v)})

    # ── Artifacts (JSON outputs + HTML report only — no LMs or image data) ───
    _ARTIFACT_PATHS = [
        out_dir / "ranking.json",
        out_dir / "ranking.csv",
        out_dir / "mcmc_diagnostics.json",
        out_dir / "decipherment_report.html",
        out_dir / "mixed_model" / "model_comparison.json",
        project_root / "outputs" / "analysis" / "mamari_calendar_alignment.json",
        project_root / "outputs" / "analysis" / "calendar_gloss_validation.json",
        project_root / "outputs" / "analysis" / "anchor_conflict_diagnosis.json",
    ]
    for p in _ARTIFACT_PATHS:
        if p.exists():
            folder = "analysis" if "analysis" in str(p) else "decipherment"
            mlflow.log_artifact(str(p), artifact_path=folder)

    log.info(
        "MLflow run %s: metrics and artifacts logged → %s",
        mlflow.active_run().info.run_id[:8],
        _mlflow_tracking_uri(project_root),
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


@hydra.main(config_path="../conf", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    """Zone C decipherment: MCMC + beam search."""
    import hydra.utils as hu

    project_root = Path(hu.get_original_cwd())
    out_dir = project_root / cfg.paths.outputs_dir / "decipherment"
    out_dir.mkdir(parents=True, exist_ok=True)

    config_hash = hash_config_file(project_root / "conf" / "config.yaml")
    top_n: int = int(cfg.zone_c.validation.top_n_hypotheses)

    with _mlflow_run(cfg, project_root, config_hash):
        _run(cfg, project_root, out_dir, config_hash, top_n)


def _run(
    cfg: DictConfig,
    project_root: Path,
    out_dir: Path,
    config_hash: str,
    top_n: int,
) -> None:
    # ── Corpus ───────────────────────────────────────────────────────────────
    log.info("Loading corpus …")
    all_tablets = load_corpus(cfg, project_root)
    if not all_tablets:
        log.error("Corpus is empty — cannot run decipherment.")
        sys.exit(1)

    tablets_by_stratum = split_by_cluster(all_tablets)
    for stratum, tabs in sorted(tablets_by_stratum.items()):
        log.info("  Stratum '%s': %d tablet(s).", stratum, len(tabs))

    corpus_sequences, sign_ids = _build_corpus_sequences(all_tablets, _SMOKE_TEST)

    # ── Parallel passages (optional validation signal) ────────────────────────
    variants_path = (
        project_root / "data" / "parallels" / "parallel_variants_auto.json"
    )

    # ── Focus-passage override ────────────────────────────────────────────────
    if _FOCUS_PASSAGE:
        result = _load_focus_passage_sequences(variants_path, _FOCUS_PASSAGE)
        if result is not None:
            corpus_sequences, sign_ids = result
            log.info(
                "FOCUS MODE: MCMC restricted to passage %s "
                "(%d sequences, %d signs).",
                _FOCUS_PASSAGE, len(corpus_sequences), len(sign_ids),
            )
        else:
            log.warning(
                "Could not load passage %s — falling back to full corpus.",
                _FOCUS_PASSAGE,
            )
    parallel_seqs = _load_parallel_variants(variants_path)

    # ── LM scorer ────────────────────────────────────────────────────────────
    log.info("Loading language models …")
    lm_scorer = LMScorer(cfg, project_root)
    log.info(
        "LMScorer ready. Languages with ≥1 loaded LM: %s",
        lm_scorer.languages_available or ["(none — check Step 1)"],
    )

    # ── MCMC ─────────────────────────────────────────────────────────────────
    mc = cfg.zone_c.mcmc
    log.info(
        "MCMC: %d chain(s) × %d iterations (burn-in %d, thin %d) …",
        mc.num_chains, mc.num_iterations, mc.burn_in, mc.thin,
    )
    # ── Phoneme inventory: default CV syllables + all calendar anchor phonemes ─
    # Multi-syllable calendar logograms (omotohi, kokore, huna, maure, …) must
    # appear in the inventory so that:
    #   (a) _validate_anchors can assert their presence before the run starts,
    #   (b) cribs init in MCMCSampler resolves them on construction,
    #   (c) _build_calendar_phoneme_priors finds them for soft-boost weights.
    _all_anchor_phonemes = set(CALENDAR_ANCHORS_HARD.values()) | {
        ph for ph, _ in CALENDAR_ANCHORS_SOFT.values()
    }
    _anchor_extras = [
        ph for ph in _all_anchor_phonemes if ph not in _DEFAULT_PHONEME_INVENTORY
    ]
    phoneme_inventory = list(_DEFAULT_PHONEME_INVENTORY) + _anchor_extras

    # Loud failure at startup rather than silent no-op deep in the chain.
    _validate_anchors(CALENDAR_ANCHORS_HARD, phoneme_inventory, label="hard anchors")
    _validate_anchors(
        {s: ph for s, (ph, _) in CALENDAR_ANCHORS_SOFT.items()},
        phoneme_inventory,
        label="soft anchors",
    )

    calendar_priors = _build_calendar_phoneme_priors(phoneme_inventory)

    # ── MCMC: pass cribs directly so the sampler excludes them from proposals ─
    # Hard-anchored signs are added to _crib_signs → removed from _free_sign_ids →
    # never touched by _propose() for the entire chain run.
    active_anchors = {k: v for k, v in CALENDAR_ANCHORS_HARD.items() if k in sign_ids}
    sampler = MCMCSampler(
        cfg=cfg,
        lm_scorer=lm_scorer,
        corpus_sequences=corpus_sequences,
        sign_ids=sign_ids,
        phoneme_inventory=phoneme_inventory,
        phoneme_priors=calendar_priors,
        cribs=active_anchors,
        seed=int(cfg.seed),
    )
    active_boosts = {ph: w for ph, w in _CALENDAR_SOFT_BOOST.items() if ph in phoneme_inventory}
    log.info("Calendar hard anchors (cribs): %s", active_anchors)
    log.info("Calendar soft boosts: %s", active_boosts)
    mcmc_result: MCMCResult = sampler.run()

    rhat_str = (
        f"{mcmc_result.gelman_rubin_rhat:.4f}"
        if mcmc_result.gelman_rubin_rhat is not None
        else "N/A"
    )
    log.info(
        "MCMC done: %d sample(s), R-hat=%s, converged=%s, "
        "mean acceptance=%.3f.",
        len(mcmc_result.top_samples),
        rhat_str,
        mcmc_result.converged,
        float(np.mean(mcmc_result.acceptance_rates))
        if mcmc_result.acceptance_rates
        else 0.0,
    )

    # ── Beam search ──────────────────────────────────────────────────────────
    bsc = cfg.zone_c.beam_search
    log.info(
        "Beam search: width=%d, max_depth=%d, seeding from %d MCMC sample(s) …",
        bsc.beam_width, bsc.max_depth, len(mcmc_result.top_samples),
    )
    decoder = BeamSearchDecoder(cfg=cfg, lm_scorer=lm_scorer)
    beam_result: BeamSearchResult = decoder.decode(
        sign_ids=sign_ids,
        corpus_sequences=corpus_sequences,
        seed_hypotheses=mcmc_result.top_samples,
    )
    log.info(
        "Beam search done: %d hypothesis/es, %d step(s), early_stop=%s.",
        len(beam_result.top_hypotheses),
        beam_result.n_steps,
        beam_result.early_stopped,
    )

    # ── Score parallel passages with best hypothesis (diagnostic) ─────────────
    if parallel_seqs and mcmc_result.top_samples:
        best_map = mcmc_result.top_samples[0].phoneme_map
        par_lps: list[float] = []
        for seq in parallel_seqs:
            translated = [best_map.get(g, "<UNK>") for g in seq]
            r = lm_scorer.score(translated)
            if math.isfinite(r.ensemble_log_prob):
                par_lps.append(r.ensemble_log_prob)
        if par_lps:
            par_mean = statistics.mean(par_lps)
            log.info(
                "Top MCMC map: mean LM score over %d parallel passage(s) = %.4f",
                len(par_lps), par_mean,
            )
    # ── Build and rank hypotheses ─────────────────────────────────────────────
    log.info("Scoring %d MCMC + %d beam hypothesis/es …",
             len(mcmc_result.top_samples), len(beam_result.top_hypotheses))

    # canonical_key → DecryptionHypothesis; beam overwrites MCMC on collision.
    hyp_pool: dict[tuple, DecryptionHypothesis] = {}

    for sample in mcmc_result.top_samples:
        key = tuple(sorted(sample.phoneme_map.items()))
        hyp_pool[key] = _make_hypothesis(
            run_id="local",
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

    # Precompute key → log_posterior for seed lookup on beam-only hypotheses.
    # Beam search refines MCMC seeds (changing ≥1 assignment), so beam keys
    # rarely match MCMC keys exactly.  For those cases we carry forward the
    # posterior of the most-similar MCMC seed rather than defaulting to 0.0.
    _mcmc_key_lp: dict[tuple, float] = {
        tuple(sorted(s.phoneme_map.items())): s.log_posterior
        for s in mcmc_result.top_samples
    }

    def _seed_lp(phoneme_map: PhonemeMap) -> float:
        """Log-posterior of the closest MCMC seed (by assignment overlap)."""
        best_lp = -math.inf
        best_overlap = -1
        for sample in mcmc_result.top_samples:
            overlap = sum(
                1 for sign, ph in phoneme_map.items()
                if sample.phoneme_map.get(sign) == ph
            )
            if overlap > best_overlap:
                best_overlap = overlap
                best_lp = sample.log_posterior
        return best_lp

    for bhyp in beam_result.top_hypotheses:
        key = tuple(sorted(bhyp.phoneme_map.items()))
        if key in hyp_pool:
            hyp_pool[key].beam_score = round(bhyp.log_score, 6)
        else:
            hyp_pool[key] = _make_hypothesis(
                run_id="local",
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
        key=lambda h: (
            h.overall_lm_score if math.isfinite(h.overall_lm_score) else -math.inf
        ),
        reverse=True,
    )[:top_n]

    # ── Mixed model: syllabic MCMC + pinned logographic taxograms ───────────
    mixed_ranked: list[DecryptionHypothesis] = []
    mixed_non_scoring_signs = {s for s in LOGOGRAPHIC_TAXOGRAMS if s in sign_ids}
    if mixed_non_scoring_signs:
        log.info(
            "Mixed model enabled: pinned taxograms=%s (excluded from LM scoring)",
            {s: LOGOGRAPHIC_TAXOGRAMS[s] for s in sorted(mixed_non_scoring_signs)},
        )
        mixed_cribs = dict(active_anchors)
        for s in mixed_non_scoring_signs:
            mixed_cribs[s] = LOGOGRAPHIC_TAXOGRAMS[s]

        mixed_extras = [
            ph for ph in mixed_cribs.values() if ph not in _DEFAULT_PHONEME_INVENTORY
        ]
        mixed_phoneme_inventory = list(_DEFAULT_PHONEME_INVENTORY) + mixed_extras
        mixed_priors = _build_calendar_phoneme_priors(mixed_phoneme_inventory)
        mixed_sequences = _strip_non_scoring_signs(corpus_sequences, mixed_non_scoring_signs)

        if not mixed_sequences:
            log.warning("Mixed model skipped: all sequences emptied by taxogram filtering.")
        else:
            mixed_sampler = MCMCSampler(
                cfg=cfg,
                lm_scorer=lm_scorer,
                corpus_sequences=mixed_sequences,
                sign_ids=sign_ids,
                phoneme_inventory=mixed_phoneme_inventory,
                phoneme_priors=mixed_priors,
                cribs=mixed_cribs,
                seed=int(cfg.seed),
            )
            mixed_mcmc_result: MCMCResult = mixed_sampler.run()
            log.info(
                "Mixed MCMC done: %d sample(s), converged=%s.",
                len(mixed_mcmc_result.top_samples), mixed_mcmc_result.converged,
            )

            mixed_beam_result: BeamSearchResult = decoder.decode(
                sign_ids=sign_ids,
                corpus_sequences=mixed_sequences,
                seed_hypotheses=mixed_mcmc_result.top_samples,
            )

            mixed_pool: dict[tuple, DecryptionHypothesis] = {}
            for sample in mixed_mcmc_result.top_samples:
                key = tuple(sorted(sample.phoneme_map.items()))
                mixed_pool[key] = _make_hypothesis(
                    run_id="local",
                    phoneme_map=sample.phoneme_map,
                    mcmc_log_posterior=sample.log_posterior,
                    beam_score=0.0,
                    sign_ids=sign_ids,
                    corpus_sequences=corpus_sequences,
                    all_tablets=all_tablets,
                    lm_scorer=lm_scorer,
                    mcmc_samples=mixed_mcmc_result.top_samples,
                    config_hash=config_hash,
                    non_scoring_signs=mixed_non_scoring_signs,
                    hypothesis_type="mixed_syllabic_logographic",
                )

            def _mixed_seed_lp(phoneme_map: PhonemeMap) -> float:
                best_lp = -math.inf
                best_overlap = -1
                for sample in mixed_mcmc_result.top_samples:
                    overlap = sum(
                        1 for sign, ph in phoneme_map.items()
                        if sample.phoneme_map.get(sign) == ph
                    )
                    if overlap > best_overlap:
                        best_overlap = overlap
                        best_lp = sample.log_posterior
                return best_lp

            for bhyp in mixed_beam_result.top_hypotheses:
                key = tuple(sorted(bhyp.phoneme_map.items()))
                if key in mixed_pool:
                    mixed_pool[key].beam_score = round(bhyp.log_score, 6)
                else:
                    mixed_pool[key] = _make_hypothesis(
                        run_id="local",
                        phoneme_map=bhyp.phoneme_map,
                        mcmc_log_posterior=_mixed_seed_lp(bhyp.phoneme_map),
                        beam_score=bhyp.log_score,
                        sign_ids=sign_ids,
                        corpus_sequences=corpus_sequences,
                        all_tablets=all_tablets,
                        lm_scorer=lm_scorer,
                        mcmc_samples=mixed_mcmc_result.top_samples,
                        config_hash=config_hash,
                        non_scoring_signs=mixed_non_scoring_signs,
                        hypothesis_type="mixed_syllabic_logographic",
                    )

            mixed_ranked = sorted(
                mixed_pool.values(),
                key=lambda h: (
                    h.overall_lm_score if math.isfinite(h.overall_lm_score) else -math.inf
                ),
                reverse=True,
            )[:top_n]
            for i, hyp in enumerate(mixed_ranked, 1):
                hyp.hypothesis_id = f"MX{i:04d}"

            if mixed_ranked:
                log.info(
                    "Top mixed hypothesis: %s  overall_lm=%.4f",
                    mixed_ranked[0].hypothesis_id,
                    mixed_ranked[0].overall_lm_score,
                )

    for i, hyp in enumerate(ranked, 1):
        hyp.hypothesis_id = f"H{i:04d}"

    if ranked:
        top = ranked[0]
        log.info(
            "Top hypothesis: %s  overall_lm=%.4f  mcmc_lp=%.4f  beam=%.4f",
            top.hypothesis_id, top.overall_lm_score,
            top.mcmc_log_posterior, top.beam_score,
        )

    # ── Write outputs ─────────────────────────────────────────────────────────
    for hyp in ranked:
        hyp.save(out_dir / f"hypothesis_{hyp.hypothesis_id}.json")

    ranking = HypothesisRanking(hypotheses=ranked, ranking_metric="overall_lm_score")
    ranking_json = out_dir / "ranking.json"
    ranking_csv  = out_dir / "ranking.csv"
    ranking_md   = out_dir / "ranking.md"
    ranking.save(ranking_json)
    ranking_csv.write_text(ranking.to_csv(), encoding="utf-8")
    ranking_md.write_text(ranking.to_markdown(), encoding="utf-8")

    if mixed_ranked:
        mixed_dir = out_dir / "mixed_model"
        mixed_dir.mkdir(parents=True, exist_ok=True)
        for hyp in mixed_ranked:
            hyp.save(mixed_dir / f"hypothesis_{hyp.hypothesis_id}.json")

        mixed_ranking = HypothesisRanking(
            hypotheses=mixed_ranked,
            ranking_metric="overall_lm_score",
        )
        mixed_json = mixed_dir / "ranking_mixed.json"
        mixed_csv = mixed_dir / "ranking_mixed.csv"
        mixed_md = mixed_dir / "ranking_mixed.md"
        mixed_ranking.save(mixed_json)
        mixed_csv.write_text(mixed_ranking.to_csv(), encoding="utf-8")
        mixed_md.write_text(mixed_ranking.to_markdown(), encoding="utf-8")

        if ranked:
            comparison = {
                "primary_model": {
                    "hypothesis_id": ranked[0].hypothesis_id,
                    "hypothesis_type": ranked[0].hypothesis_type,
                    "overall_lm_score": ranked[0].overall_lm_score,
                },
                "mixed_model": {
                    "hypothesis_id": mixed_ranked[0].hypothesis_id,
                    "hypothesis_type": mixed_ranked[0].hypothesis_type,
                    "overall_lm_score": mixed_ranked[0].overall_lm_score,
                    "taxogram_cribs": {
                        s: LOGOGRAPHIC_TAXOGRAMS[s]
                        for s in sorted(mixed_non_scoring_signs)
                    },
                    "taxogram_signs_excluded_from_lm": sorted(mixed_non_scoring_signs),
                },
                "delta_mixed_minus_primary": (
                    mixed_ranked[0].overall_lm_score - ranked[0].overall_lm_score
                ),
            }
            cmp_path = mixed_dir / "model_comparison.json"
            cmp_path.write_text(json.dumps(comparison, indent=2), encoding="utf-8")
            log.info(
                "Model comparison written to %s (delta=%.4f)",
                cmp_path,
                comparison["delta_mixed_minus_primary"],
            )

    # ── MCMC diagnostics sidecar (consumed by the HTML report) ────────────────
    acceptance_mean = (
        float(np.mean(mcmc_result.acceptance_rates))
        if mcmc_result.acceptance_rates else None
    )
    active_anchors = {k: v for k, v in CALENDAR_ANCHORS_HARD.items() if k in sign_ids}
    mcmc_diag: dict[str, Any] = {
        "n_chains":           mcmc_result.n_chains,
        "n_samples_per_chain": mcmc_result.n_samples_per_chain,
        "gelman_rubin_rhat":  (
            round(mcmc_result.gelman_rubin_rhat, 4)
            if mcmc_result.gelman_rubin_rhat is not None else None
        ),
        "geweke_z":           (
            round(mcmc_result.geweke_z, 4)
            if getattr(mcmc_result, "geweke_z", None) is not None else None
        ),
        "converged":          mcmc_result.converged,
        "acceptance_rates":   [round(r, 4) for r in mcmc_result.acceptance_rates],
        "acceptance_mean":    round(acceptance_mean, 4) if acceptance_mean is not None else None,
        "parallel_tempering": bool(sampler._pt_enabled),
        "calendar_anchors":   active_anchors,
        "sign_inventory_size": len(sign_ids),
    }
    if sampler._pt_enabled:
        mcmc_diag["pt_n_temperatures"] = sampler._pt_n_temperatures
        mcmc_diag["pt_t_max"]          = sampler._pt_t_max
        mcmc_diag["pt_swap_interval"]  = sampler._pt_swap_interval
    diag_path = out_dir / "mcmc_diagnostics.json"
    diag_path.write_text(json.dumps(mcmc_diag, indent=2), encoding="utf-8")
    log.info("MCMC diagnostics written to %s", diag_path)

    # HTML scholar report
    report_path = out_dir / "decipherment_report.html"
    try:
        from hackingrongo.results.decipherment_report import save_decipherment_report
        pgood_path_auto = out_dir.parent / "zone_b" / "pgood_analysis.json"
        qubo_path_auto  = out_dir / "qubo_result.json"
        mixed_cmp_auto  = out_dir / "mixed_model" / "model_comparison.json"
        freq_path_auto  = out_dir.parent / "zone_b" / "freq_match.json"
        morph_path_auto = out_dir.parent / "morpheme_segments.json"
        save_decipherment_report(
            ranking_json, report_path, top_n=20,
            pgood_path  = pgood_path_auto  if pgood_path_auto.exists()  else None,
            qubo_path   = qubo_path_auto   if qubo_path_auto.exists()   else None,
            mixed_compare_path = mixed_cmp_auto if mixed_cmp_auto.exists() else None,
            diag_path   = diag_path        if diag_path.exists()        else None,
            freq_path   = freq_path_auto   if freq_path_auto.exists()   else None,
            morph_path  = morph_path_auto  if morph_path_auto.exists()  else None,
        )
    except ImportError:
        raise  # missing dependency — not a graceful-degradation case
    except Exception as exc:
        log.warning("Could not generate HTML report: %s", exc, exc_info=True)

    log.info(
        "Written %d hypothesis file(s) + ranking.{json,csv,md} + decipherment_report.html → %s",
        len(ranked), out_dir,
    )
    if mixed_ranked and ranked:
        log.info(
            "Primary vs mixed top LM scores: %.4f vs %.4f",
            ranked[0].overall_lm_score,
            mixed_ranked[0].overall_lm_score,
        )

    # ── MLflow: log metrics + artifacts for this run ──────────────────────────
    _mlflow_log_results(
        out_dir=out_dir,
        project_root=project_root,
        ranked=ranked,
        mcmc_result=mcmc_result,
        sign_ids=sign_ids,
        active_anchors={k: v for k, v in CALENDAR_ANCHORS_HARD.items() if k in sign_ids},
    )



if __name__ == "__main__":
    main()
