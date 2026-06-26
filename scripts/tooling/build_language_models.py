# ============================================================================
# DEPRECATED — SYLLABIC SUBSTITUTION-CIPHER TRACK (set down 2026-06, in place).
# Part of the sign→phoneme substitution-cipher hypothesis, which was tested and
# set down as a recorded NEGATIVE RESULT — preserved as an archive, NOT fixed,
# tuned, or deleted. Do not extend this module. The structural/logographic track
# supersedes it. Full rationale + on-disk numbers: DEPRECATED_SYLLABIC.md (root).
# ============================================================================
"""
TOOLING — one-time data preparation; not part of the reproducible analysis pipeline.

Build and serialise all Polynesian language models for Zone C scoring.

Usage
-----
From the project root:

    python scripts/build_language_models.py

Or via Hydra overrides:

    python scripts/build_language_models.py zone_c.lm_scoring.languages=[old_rapa_nui]

The script reads ``conf/config.yaml`` via Hydra, iterates over every
(language, n-gram order) pair configured in ``zone_c.lm_scoring``, builds
one Kneser-Ney smoothed :class:`~hackingrongo.data.rapa_nui_corpus.NGramLM`
per pair, and serialises each to a JSON file under
``data/language_models/``.

N-gram order
------------
Default configuration builds orders 3, 4, and 5 (see
``conf/config.yaml → zone_c.lm_scoring.ngram_orders``).  Order-5 models
for ``pre_contact`` and ``post_contact`` have the Hawaiian ``smoothing``
LM attached as a cross-lingual KN backoff for unseen 4/5-gram contexts
(α = 0.15).  This implements modified Kneser-Ney with backoff to the
Hawaiian smoothing LM as described in the long-range dependency
hypothesis.

Self-training comparison (3-gram vs 5-gram)
-------------------------------------------
After building, run self-training twice to compare score trajectories::

    # 3-gram baseline (order-3 LM already on disk)
    python scripts/run_self_training.py \\
        zone_c.lm_scoring.primary_order=3 \\
        mlflow.run_name=self_training_3gram

    # 5-gram with Hawaiian backoff
    python scripts/run_self_training.py \\
        zone_c.lm_scoring.primary_order=5 \\
        mlflow.run_name=self_training_5gram_hawaiian_backoff

Compare the ``top_lm_score`` trajectories in MLflow.  A larger
self-training delta for the 5-gram run supports the long-range dependency
hypothesis.

For languages without text corpora (Hawaiian, Māori, Tahitian), the script
logs a warning and skips — those LMs can be added once the corpora are
obtained from POLLEX or Max Planck's CLICS.

Logging
-------
A concise one-line summary per language/order is printed to stdout.
Full diagnostics go to the configured log file (``outputs/hackingrongo.log``).
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import hydra
from omegaconf import DictConfig, open_dict

# Ensure the project package is importable when run from the repo root.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

logger = logging.getLogger(__name__)

_SEED: int = 20260606


@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    """Entry point — builds all configured language models."""
    from hackingrongo.repro import set_global_seed
    set_global_seed(_SEED)

    from hackingrongo.data.rapa_nui_corpus import build_all_lms

    project_root = Path(hydra.utils.get_original_cwd())

    # Auto-ingest every Rapa Nui source under data/lm_sources/ into the
    # pre_contact and post_contact LM sources.  Source specs are resolved
    # relative to polynesian_texts_dir by build_all_lms, hence the "../"
    # prefix.  Files are populated by scripts/fetch_lm_sources_extended.py
    # (kohaumotu dictionary, Kieviet examples, ASJP, …); the leading-underscore
    # report file is skipped.  Adding a new source is now drop-a-file, not a
    # code change.
    _lm_sources_dir = project_root / "data" / "lm_sources"
    _src_files = sorted(
        p for p in _lm_sources_dir.glob("*.txt") if not p.name.startswith("_")
    )
    if _src_files:
        with open_dict(cfg):
            for _era in ("pre_contact", "post_contact"):
                if _era not in cfg.zone_c.lm_scoring.lm_sources:
                    continue
                _sources = list(cfg.zone_c.lm_scoring.lm_sources[_era])
                for _f in _src_files:
                    _spec = f"../lm_sources/{_f.name}"
                    if _spec not in _sources:
                        _sources.append(_spec)
                cfg.zone_c.lm_scoring.lm_sources[_era] = _sources
        _total = sum(
            sum(1 for ln in f.read_text(encoding="utf-8").splitlines() if ln.strip())
            for f in _src_files
        )
        logger.info(
            "Auto-ingested %d data/lm_sources/ file(s) (%d non-empty lines) "
            "into pre_contact and post_contact LM sources: %s",
            len(_src_files), _total, ", ".join(f.name for f in _src_files),
        )
    else:
        logger.warning(
            "No data/lm_sources/*.txt files found; run "
            "scripts/fetch_lm_sources_extended.py --all to populate them.",
        )

    orders = list(cfg.zone_c.lm_scoring.ngram_orders)
    eras = list(cfg.zone_c.lm_scoring.lms)
    max_order = max(int(o) for o in orders)

    logger.info(
        "Building language models. Era LMs: %s  Orders: %s",
        eras,
        orders,
    )

    out_dir = project_root / "data" / "language_models"
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    build_all_lms(cfg, project_root)
    elapsed = time.time() - t0

    logger.info(
        "Language model build complete (%.1fs).  Files written to: %s", elapsed, out_dir
    )

    # ----------------------------------------------------------------
    # Log build metadata to MLflow so the 3-gram vs 5-gram comparison
    # can be tracked by run name in the same experiment.
    # ----------------------------------------------------------------
    try:
        import mlflow

        mlflow_dir = project_root / "outputs" / "mlruns"
        mlflow.set_tracking_uri(mlflow_dir.as_uri())
        mlflow.set_experiment("rongorongo_lm_build")

        with mlflow.start_run(run_name=f"lm_build_maxorder{max_order}"):
            mlflow.log_param("eras", ",".join(str(e) for e in eras))
            mlflow.log_param("ngram_orders", ",".join(str(o) for o in orders))
            mlflow.log_param("max_order", max_order)
            mlflow.log_param(
                "hawaiian_backoff",
                "yes" if max_order >= 4 else "no",
            )
            mlflow.log_param("backoff_alpha", 0.15)
            mlflow.log_param("backoff_from_order", 4)
            mlflow.log_metric("build_time_seconds", elapsed)
            logger.info(
                "MLflow run logged to experiment 'rongorongo_lm_build' "
                "(run: lm_build_maxorder%d).",
                max_order,
            )
    except Exception as exc:
        logger.debug("MLflow logging skipped: %s", exc)


if __name__ == "__main__":
    for _arg in list(sys.argv):
        if _arg.startswith("--seed="):
            _SEED = int(_arg.split("=", 1)[1].strip())
            sys.argv.remove(_arg)
            break
    main()
