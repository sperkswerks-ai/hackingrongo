"""
Zone B driver — contact analysis, IC sensitivity, and bipartite visualisation.

Writes all outputs to outputs/zone_b/:
  sensitivity_analysis.json      — IC pre/post across 3 dating scenarios
  contact_partition.json         — per-sign G² partition (default scenario)
  contact_partition_bipartite.html — Plotly bipartite graph of the partition

Usage
-----
    conda run -n hackingrongo python scripts/run_zone_b.py
    conda run -n hackingrongo python scripts/run_zone_b.py --min-g2 5.0
    conda run -n hackingrongo python scripts/run_zone_b.py --out-dir path/to/dir
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from omegaconf import OmegaConf  # noqa: E402

from hackingrongo.zone_b.astronomical_analysis import run_all_tests  # noqa: E402
from hackingrongo.zone_b.contact_analysis import (  # noqa: E402
    CHI2_P05,
    analyse,
    contact_sensitivity_analysis,
    write_bipartite_html,
)
from hackingrongo.zone_b.entropy import sensitivity_analysis  # noqa: E402
from hackingrongo.results.entropy_report import save_entropy_report  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run Zone B contact and IC sensitivity analysis."
    )
    p.add_argument(
        "--min-g2",
        type=float,
        default=CHI2_P05,
        metavar="G2",
        help=f"G² significance threshold (default: {CHI2_P05:.3f}, p<0.05).",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory (default: outputs/zone_b/ from config).",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    cfg = OmegaConf.load(PROJECT_ROOT / "conf" / "config.yaml")
    out_dir: Path = args.out_dir or (
        PROJECT_ROOT / cfg.paths.outputs_dir / "zone_b"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    log.info("Zone B outputs → %s", out_dir)

    # IC sensitivity across the three dating scenarios + boustrophedon voice-split
    scenarios = [s["name"] for s in cfg.corpus.temporal_model.sensitivity_scenarios]
    log.info("Running IC sensitivity analysis (%d scenarios)…", len(scenarios))
    sensitivity_json = out_dir / "sensitivity_analysis.json"
    sensitivity_analysis(
        scenarios=scenarios,
        output_path=sensitivity_json,
    )
    log.info("Writing IC / entropy HTML report…")
    try:
        save_entropy_report(
            sensitivity_json=sensitivity_json,
            output_path=out_dir / "entropy_report.html",
        )
    except Exception as exc:
        log.warning("Entropy HTML report failed (non-fatal): %s", exc)

    # Contact partition — default (exclude unknown tablets)
    log.info("Running contact partition analysis…")
    records = analyse(min_g2=args.min_g2)
    partition_path = out_dir / "contact_partition.json"
    partition_path.write_text(json.dumps(records, indent=2), encoding="utf-8")
    log.info(
        "Contact partition: %d signs written → %s",
        len(records),
        partition_path,
    )

    # Bipartite HTML visualisation
    log.info("Writing bipartite HTML…")
    try:
        write_bipartite_html(
            records=records,
            cfg=cfg,
            output_path=out_dir / "contact_partition_bipartite.html",
        )
    except ImportError:
        log.warning(
            "plotly not installed — bipartite HTML skipped. "
            "Install with: pip install plotly"
        )

    # Astronomical sign analysis
    log.info("Running astronomical hypothesis tests…")
    corpus_dir = PROJECT_ROOT / cfg.paths.corpus_dir
    astro_path = out_dir / "astronomical_candidates.json"
    run_all_tests(
        corpus_dir=corpus_dir,
        output_path=astro_path,
    )

    # Astronomical HTML report
    try:
        from hackingrongo.results.astronomical_report import save_astronomical_report

        svg_catalog = PROJECT_ROOT / "data" / "glyphs" / "svg" / "catalog.json"
        save_astronomical_report(
            candidates_path=astro_path,
            svg_catalog_path=svg_catalog,
            output_path=out_dir / "astronomical_report.html",
        )
    except Exception as exc:
        log.warning("Astronomical HTML report failed (non-fatal): %s", exc)

    log.info("Zone B done.")


if __name__ == "__main__":
    main()
