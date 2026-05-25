"""
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
from pathlib import Path

import hydra
from omegaconf import DictConfig

# Ensure the project package is importable when run from the repo root.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    """Entry point — builds all configured language models."""
    from hackingrongo.data.rapa_nui_corpus import build_all_lms

    project_root = Path(hydra.utils.get_original_cwd())

    logger.info(
        "Building language models. Era LMs: %s  Orders: %s",
        list(cfg.zone_c.lm_scoring.lms),
        list(cfg.zone_c.lm_scoring.ngram_orders),
    )

    out_dir = project_root / "data" / "language_models"
    out_dir.mkdir(parents=True, exist_ok=True)

    build_all_lms(cfg, project_root)

    logger.info("Language model build complete.  Files written to: %s", out_dir)


if __name__ == "__main__":
    main()
