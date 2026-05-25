"""
hackingrongo.pipeline — End-to-end pipeline orchestrator.

Runs the five pipeline steps in sequence with prerequisite gates,
progress timing, and a JSON run manifest.

Usage
-----
    python -m hackingrongo.pipeline
    python -m hackingrongo.pipeline --skip-training
    python -m hackingrongo.pipeline --smoke-test
    python -m hackingrongo.pipeline --steps 1,2,3
    python -m hackingrongo.pipeline --dry-run

Steps
-----
    1  Build Polynesian language models (Zone C scoring dependency)
    1b Segment 3D tablet renders into per-glyph crops (feeds Zone A training)
    2  Train Zone A convolutional autoencoder + extract embeddings
    3  Analyse embeddings: UMAP, HDBSCAN, divergence report (Zone A output)
    4  Zone B analysis battery:
         4a  IC / entropy sensitivity analysis (3 dating scenarios)
         4b  G² contact analysis sensitivity (3 dating scenarios)
         4c  Compound glyph candidate detection
         4d  Compound candidate HTML report
         4e  Parallel passage cross-reference (algorithmic)
         4f  Diachronic passage HTML report
         4g  Astronomical glyph candidate analysis
         4h  Astronomical HTML report
         4i  Quantum hardness (p_good) analysis
         4j  QUBO quantum annealing key search
         4k  Zone C fusion layer training (Zone A + Zone B priors)
         4l  Frequency-language match (Zipf α, Spearman ρ, χ²)
         4m  Zellig Harris morpheme segmentation
    5  Zone C decipherment: MCMC + beam-search
    5b Zone C HTML scholar report (decipherment_report.html)

Flags
-----
    --smoke-test        Override training to 1 epoch / batch 8 for a fast
                        end-to-end check that the wiring works.
    --skip-training     Skip Step 2.  Requires outputs/embeddings_cache.pt
                        to already exist (e.g. restored from Drive).
    --steps N[,N...]    Run only the listed top-level steps (1–5).
                        Sub-steps (4a–4d) are always run together when 4
                        is included.
    --dry-run           Print all commands that would be executed without
                        running them.
    --keep-going        Continue to subsequent steps even when a step fails.
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

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

# pipeline.py lives at hackingrongo/hackingrongo/pipeline.py
# → project root is two levels up
PROJECT_ROOT = Path(__file__).resolve().parent.parent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("pipeline")

# ---------------------------------------------------------------------------
# ANSI colours (gracefully degraded on non-TTY)
# ---------------------------------------------------------------------------

_USE_COLOUR = sys.stdout.isatty()


def _c(text: str, code: str) -> str:
    return f"\033[{code}m{text}\033[0m" if _USE_COLOUR else text


def _bold(t: str) -> str:   return _c(t, "1")
def _green(t: str) -> str:  return _c(t, "32")
def _yellow(t: str) -> str: return _c(t, "33")
def _red(t: str) -> str:    return _c(t, "31")
def _cyan(t: str) -> str:   return _c(t, "36")
def _dim(t: str) -> str:    return _c(t, "2")


# ---------------------------------------------------------------------------
# Step runner
# ---------------------------------------------------------------------------

def _run(
    label: str,
    cmd: list[str],
    *,
    dry_run: bool = False,
    env: dict | None = None,
) -> tuple[int, float]:
    """Run a subprocess, stream its output, return (returncode, elapsed_s)."""
    log.info("%s  %s", _dim("$"), _dim(" ".join(str(c) for c in cmd)))
    if dry_run:
        return 0, 0.0

    t0 = time.monotonic()
    proc = subprocess.Popen(
        cmd,
        cwd=PROJECT_ROOT,
        stdout=sys.stdout,
        stderr=sys.stderr,
        env=env,
        shell=False,
    )
    proc.wait()
    elapsed = time.monotonic() - t0
    return proc.returncode, elapsed


# ---------------------------------------------------------------------------
# Prerequisite checks
# ---------------------------------------------------------------------------

def _check(path: Path, description: str) -> bool:
    """Return True if *path* exists, else log a clear error and return False."""
    if path.exists():
        return True
    log.error(
        "%s required but not found: %s",
        description, path.relative_to(PROJECT_ROOT),
    )
    return False


def _check_any_file(directory: Path, glob: str, description: str) -> bool:
    matches = list(directory.glob(glob)) if directory.exists() else []
    if matches:
        return True
    log.error(
        "%s: no files matching %r in %s",
        description, glob, directory.relative_to(PROJECT_ROOT),
    )
    return False


# ---------------------------------------------------------------------------
# Individual steps
# ---------------------------------------------------------------------------

def step1b_segment_3d_glyphs(dry_run: bool = False) -> tuple[int, float]:
    """Segment rendered 3D tablet views into per-glyph crops.

    Reads PNGs from data/glyphs/synthetic_views/tablet_{B,C,D}/ (produced by
    render_tablet_views.py) and writes individual glyph crops to
    data/glyphs/3d_crops/ using CLAHE + Canny + connected-component analysis.

    Runs for tablets B, C, D × recto (r) and verso (v) sides.
    Skips tablets whose synthetic_views directory does not yet exist.
    """
    renders_root = PROJECT_ROOT / "data" / "glyphs" / "synthetic_views"
    corpus_dir   = PROJECT_ROOT / "data" / "corpus"
    output_root  = PROJECT_ROOT / "data" / "glyphs" / "3d_crops"

    if not renders_root.exists():
        log.error(
            "data/glyphs/synthetic_views/ not found — "
            "run scripts/render_tablet_views.py first."
        )
        return 1, 0.0

    def _detect_sides(tablet: str) -> list[str]:
        """Read side codes actually present in this tablet's corpus JSON."""
        import json as _json
        path = corpus_dir / f"{tablet}.json"
        if not path.exists():
            return ["r", "v"]  # safe default
        data = _json.loads(path.read_text(encoding="utf-8"))
        sides = sorted({g["side"] for g in data.get("glyphs", []) if "side" in g})
        return sides or ["r", "v"]

    tablets = ["B", "C", "D"]
    total_rc = 0
    total_elapsed = 0.0

    for tablet in tablets:
        render_dir = renders_root / f"tablet_{tablet}"
        if not render_dir.exists():
            log.warning("Render dir not found, skipping: %s", render_dir)
            continue
        sides = _detect_sides(tablet)

        # Compute ROI to exclude INSCRIBE page chrome (fractions of image size).
        # Canvas-only screenshots include WebGL-rendered UI chrome:
        #   left-nav icons: ~3%   top-header: ~14%
        #   right info panel: starts ~56%   bottom copyright: ~88%
        _roi_arg: str | None = None
        try:
            import PIL.Image as _pil
            sample_png = next(iter(sorted(render_dir.glob("*.png"))), None)
            if sample_png:
                _W, _H = _pil.open(sample_png).size
                _x0 = int(0.032 * _W)
                _y0 = int(0.140 * _H)
                _x1 = int(0.540 * _W)   # exclude right info panel (starts ~56%)
                _y1 = int(0.880 * _H)   # exclude bottom copyright
                _roi_arg = f"{_x0},{_y0},{_x1},{_y1}"
                log.info("3D crop ROI for tablet %s: %s (from %dx%d render)", tablet, _roi_arg, _W, _H)
        except Exception as _e:
            log.warning("Could not compute ROI for tablet %s: %s", tablet, _e)

        for side in sides:
            corpus_path = corpus_dir / f"{tablet}.json"
            cmd = [
                sys.executable, "scripts/segment_3d_glyphs.py",
                "--tablet",    tablet,
                "--side",      side,
                "--renders",   str(render_dir),
                "--corpus",    str(corpus_path),
                "--output",    str(output_root),
                "--crop-size", "128",
                "--num-views", "6",
            ]
            if _roi_arg:
                cmd += ["--roi", _roi_arg]
            rc, elapsed = _run(
                f"segment_3d_{tablet}_{side}",
                cmd,
                dry_run=dry_run,
            )
            total_rc = max(total_rc, rc)
            total_elapsed += elapsed

    return total_rc, total_elapsed


def step1_build_lms(dry_run: bool = False) -> tuple[int, float]:
    """Build Polynesian n-gram language models for Zone C scoring."""
    # Gate: corpus text files must exist
    poly_dir = PROJECT_ROOT / "data" / "polynesian_texts"
    if not poly_dir.exists():
        log.warning(
            "data/polynesian_texts/ not found — LM build will likely fail. "
            "Run scripts/fetch_abvd_corpus.py, parse_ids.py, and "
            "fetch_hawaiian_corpus.py first."
        )

    return _run(
        "build_language_models",
        [sys.executable, "scripts/build_language_models.py"],
        dry_run=dry_run,
    )


def step2_train_autoencoder(
    smoke_test: bool = False,
    dry_run: bool = False,
) -> tuple[int, float]:
    """Train the Zone A convolutional autoencoder and save embeddings."""
    # Gate: glyph images must exist
    if not dry_run:
        svg_dir = PROJECT_ROOT / "data" / "glyphs" / "svg"
        bc_dir  = PROJECT_ROOT / "data" / "glyphs" / "barthel_corpus"
        if not any([
            list(svg_dir.glob("*.svg"))      if svg_dir.exists() else [],
            list(bc_dir.glob("*.png"))       if bc_dir.exists()  else [],
            list((PROJECT_ROOT / "data" / "glyphs").glob("**/*.png")),
        ]):
            log.error(
                "No glyph images found under data/glyphs/. "
                "Run scripts/scrape_glyphs.py and/or "
                "scripts/extract_barthel_glyphs.py first."
            )
            return 1, 0.0

    cmd = [sys.executable, "scripts/train_autoencoder.py"]
    if smoke_test:
        cmd += [
            "zone_a.autoencoder.num_epochs=1",
            "zone_a.autoencoder.batch_size=8",
            "zone_a.autoencoder.checkpoint_interval_epochs=1",
            "zone_a.autoencoder.log_interval_steps=5",
        ]
    return _run("train_autoencoder", cmd, dry_run=dry_run)


def step3_analyze_embeddings(dry_run: bool = False) -> tuple[int, float]:
    """Project embeddings to UMAP, cluster, generate divergence report."""
    if not dry_run and not _check(
        PROJECT_ROOT / "outputs" / "embeddings_cache.pt",
        "Embeddings cache (run Step 2 first, or restore from Drive)",
    ):
        return 1, 0.0

    return _run(
        "analyze_embeddings",
        [sys.executable, "scripts/analyze_embeddings.py"],
        dry_run=dry_run,
    )


def step4a_entropy(dry_run: bool = False) -> tuple[int, float]:
    """IC / entropy sensitivity analysis under all three dating scenarios."""
    out = PROJECT_ROOT / "outputs" / "sensitivity_analysis.json"
    return _run(
        "entropy_sensitivity",
        [
            sys.executable, "-m", "hackingrongo.zone_b.entropy",
            "--scenario", "conservative_all_late",
            "--scenario", "optimistic_distributed",
            "--scenario", "probabilistic_weighted",
            "--output", str(out),
        ],
        dry_run=dry_run,
    )


def step4a_entropy_report(dry_run: bool = False) -> tuple[int, float]:
    """Render IC / entropy + boustrophedon voice-split HTML report."""
    sensitivity_json = PROJECT_ROOT / "outputs" / "sensitivity_analysis.json"
    out = PROJECT_ROOT / "outputs" / "analysis" / "entropy_report.html"
    if not dry_run and not _check(sensitivity_json, "sensitivity_analysis.json (run Step 4a first)"):
        return 1, 0.0
    return _run(
        "entropy_report",
        [
            sys.executable, "-m", "hackingrongo.results.entropy_report",
            "--input",  str(sensitivity_json),
            "--output", str(out),
        ],
        dry_run=dry_run,
    )


def step4b_contact(dry_run: bool = False) -> tuple[int, float]:
    """G² contact analysis sensitivity under all three dating scenarios."""
    out = PROJECT_ROOT / "outputs" / "contact_sensitivity.json"
    return _run(
        "contact_sensitivity",
        [
            sys.executable, "-m", "hackingrongo.zone_b.contact_analysis",
            "--scenario", "all",
            "--output", str(out),
        ],
        dry_run=dry_run,
    )


def step4c_compound_detector(dry_run: bool = False) -> tuple[int, float]:
    """Detect compound glyph candidates from Zone A embeddings."""
    analysis_dir = PROJECT_ROOT / "outputs" / "analysis"
    if not dry_run and not _check(
        analysis_dir / "cluster_vs_barthel.csv",
        "cluster_vs_barthel.csv (run Step 3 first)",
    ):
        return 1, 0.0

    corpus_dir = PROJECT_ROOT / "data" / "corpus"
    out = analysis_dir / "compound_candidates.json"
    return _run(
        "compound_detector",
        [
            sys.executable, "-m", "hackingrongo.zone_b.compound_detector",
            "--analysis-dir", str(analysis_dir),
            "--corpus-dir",   str(corpus_dir),
            "--output",       str(out),
        ],
        dry_run=dry_run,
    )


def step4d_compound_report(dry_run: bool = False) -> tuple[int, float]:
    """Generate HTML compound candidate report for scholar review."""
    analysis_dir = PROJECT_ROOT / "outputs" / "analysis"
    if not dry_run and not _check(
        analysis_dir / "compound_candidates.json",
        "compound_candidates.json (run Step 4c first)",
    ):
        return 1, 0.0

    return _run(
        "compound_report",
        [
            sys.executable, "-m", "hackingrongo.results.compound_report",
            "--candidates",  str(analysis_dir / "compound_candidates.json"),
            "--svg-catalog", str(PROJECT_ROOT / "data" / "glyphs" / "svg" / "catalog.json"),
            "--corpus-dir",  str(PROJECT_ROOT / "data" / "corpus"),
            "--output",      str(analysis_dir / "compound_report.html"),
        ],
        dry_run=dry_run,
    )

def step4e_parallel_passages(dry_run: bool = False) -> tuple[int, float]:
    """Algorithmic parallel passage cross-reference search."""
    parallels_dir = PROJECT_ROOT / "data" / "parallels"
    corpus_dir    = PROJECT_ROOT / "data" / "corpus"

    if not dry_run and not _check(
        parallels_dir / "horley_parallels.csv",
        "horley_parallels.csv (required for cross-reference)",
    ):
        return 1, 0.0

    if not dry_run and not _check(
        corpus_dir,
        "data/corpus/ (run scripts/build_corpus.py first)",
    ):
        return 1, 0.0

    out = parallels_dir / "parallel_variants_auto.json"
    return _run(
        "cross_reference_parallels",
        [
            sys.executable, "scripts/cross_reference_parallels.py",
            "--input",     str(parallels_dir / "horley_parallels.csv"),
            "--corpus",    str(corpus_dir),
            "--config",    str(PROJECT_ROOT / "conf" / "config.yaml"),
            "--tablets",   str(PROJECT_ROOT / "data" / "metadata" / "tablets.json"),
            "--output",    str(out),
            "--threshold", "1",
        ],
        dry_run=dry_run,
    )


def step4f_passage_report(dry_run: bool = False) -> tuple[int, float]:
    """Generate diachronic parallel passage HTML report for scholar review."""
    variants_path = PROJECT_ROOT / "data" / "parallels" / "parallel_variants_auto.json"

    if not dry_run and not _check(
        variants_path,
        "parallel_variants_auto.json (run Step 4e first)",
    ):
        return 1, 0.0

    out_dir = PROJECT_ROOT / "outputs" / "analysis" / "passage_reports"
    return _run(
        "passage_report",
        [
            sys.executable, "-m", "hackingrongo.results.passage_report",
            "--input",  str(variants_path),
            "--output", str(out_dir),
            "--filter-score", "0.0",
        ],
        dry_run=dry_run,
    )
def step4g_astronomical(dry_run: bool = False) -> tuple[int, float]:
    """Astronomical glyph candidate analysis."""
    return _run(
        "astronomical_analysis",
        [
            sys.executable, "-m",
            "hackingrongo.zone_b.astronomical_analysis",
            "--corpus-dir",  str(PROJECT_ROOT / "data" / "corpus"),
            "--output",      str(PROJECT_ROOT / "outputs" / "zone_b" /
                                "astronomical_candidates.json"),
            "hydra.job.chdir=false",
        ],
        dry_run=dry_run,
    )


def step4h_astronomical_report(dry_run: bool = False) -> tuple[int, float]:
    """Generate astronomical HTML report."""
    candidates_path = (PROJECT_ROOT / "outputs" / "zone_b" /
                       "astronomical_candidates.json")
    if not dry_run and not _check(candidates_path,
                                  "astronomical_candidates.json (run step 4g first)"):
        return 1, 0.0
    return _run(
        "astronomical_report",
        [
            sys.executable, "-m",
            "hackingrongo.results.astronomical_report",
            "--candidates", str(candidates_path),
            "--output",     str(PROJECT_ROOT / "outputs" / "analysis" /
                                "astronomical_report.html"),
        ],
        dry_run=dry_run,
    )


def step4i_pgood_analysis(dry_run: bool = False) -> tuple[int, float]:
    """Quantum hardness (p_good) analysis."""
    return _run(
        "measure_pgood",
        [
            sys.executable, "scripts/measure_pgood.py",
            "--corpus-dir", str(PROJECT_ROOT / "data" / "corpus"),
            "--lm-dir",     str(PROJECT_ROOT / "data" / "language_models"),
            "--n-samples",  "10000",
            "--output",     str(PROJECT_ROOT / "outputs" / "zone_b" /
                                "pgood_analysis.json"),
        ],
        dry_run=dry_run,
    )


def step4j_qubo_decipherment(dry_run: bool = False) -> tuple[int, float]:
    """QUBO quantum annealing key search."""
    ranking = PROJECT_ROOT / "outputs" / "decipherment" / "ranking.json"
    cmd = [
        sys.executable, "scripts/run_qubo_decipherment.py",
        "--corpus-dir", str(PROJECT_ROOT / "data" / "corpus"),
        "--lm-dir",     str(PROJECT_ROOT / "data" / "language_models"),
        "--solver",     "neal",
        "--num-reads",  "1000",
        "--output",     str(PROJECT_ROOT / "outputs" / "decipherment" /
                            "qubo_result.json"),
    ]
    if ranking.exists():
        cmd += ["--init-from", str(ranking)]
    return _run("qubo_decipherment", cmd, dry_run=dry_run)


def step4l_freq_match(dry_run: bool = False) -> tuple[int, float]:
    """Frequency-language match: Zipf α, Spearman ρ, χ² fit vs. each LM."""
    return _run(
        "freq_match",
        [
            sys.executable, "scripts/run_freq_match.py",
            "--corpus-dir", str(PROJECT_ROOT / "data" / "corpus"),
            "--lm-dir",     str(PROJECT_ROOT / "data" / "language_models"),
            "--output",     str(PROJECT_ROOT / "outputs" / "zone_b" / "freq_match.json"),
        ],
        dry_run=dry_run,
    )


def step4m_morpheme_seg(dry_run: bool = False) -> tuple[int, float]:
    """Zellig Harris successor-entropy morpheme boundary segmentation."""
    return _run(
        "morpheme_segmentation",
        [
            sys.executable, "scripts/segment_morphemes.py",
            "--corpus-dir", str(PROJECT_ROOT / "data" / "corpus"),
            "--output",     str(PROJECT_ROOT / "outputs" / "morpheme_segments.json"),
        ],
        dry_run=dry_run,
    )


def step4k_train_fusion(
    smoke_test: bool = False,
    dry_run: bool = False,
) -> tuple[int, float]:
    """Train the Zone C fusion layer (Zone A embeddings + Zone B structural priors).

    Requires:
    * ``outputs/embeddings_cache.pt`` — Zone A autoencoder embeddings
    * ``outputs/analysis/compound_candidates.json`` — Zone B compound scores
    * ``data/corpus/`` — rongorongo corpus (for training targets)

    Produces: ``outputs/zone_c/fusion_checkpoint.pt``.
    """
    embeddings_cache = PROJECT_ROOT / "outputs" / "embeddings_cache.pt"
    compound_json = PROJECT_ROOT / "outputs" / "analysis" / "compound_candidates.json"

    if not dry_run:
        if not _check(embeddings_cache, "Embeddings cache (run Step 2 first)"):
            return 1, 0.0
        if not _check(compound_json, "Compound candidates (run Step 4c first)"):
            return 1, 0.0

    cmd = [
        sys.executable, "scripts/train_fusion.py",
        "--embeddings",  str(embeddings_cache),
        "--compounds",   str(compound_json),
        "--corpus-dir",  str(PROJECT_ROOT / "data" / "corpus"),
        "--output",      str(PROJECT_ROOT / "outputs" / "zone_c" / "fusion_checkpoint.pt"),
    ]
    if smoke_test:
        cmd.append("--smoke-test")
    return _run("train_fusion", cmd, dry_run=dry_run)


def step5b_decipherment_report(dry_run: bool = False) -> tuple[int, float]:
    """Render the scholar-facing HTML report from Zone C ranking output."""
    ranking_path = PROJECT_ROOT / "outputs" / "decipherment" / "ranking.json"
    if not dry_run and not _check(ranking_path, "ranking.json (run Step 5 first)"):
        return 1, 0.0

    out = ranking_path.parent / "decipherment_report.html"
    cmd = [
        sys.executable, "-m", "hackingrongo.results.decipherment_report",
        "--ranking", str(ranking_path),
        "--output",  str(out),
    ]
    # Auto-wire optional enrichment files when they exist.
    optional_args: list[tuple[str, Path]] = [
        ("--pgood",     PROJECT_ROOT / "outputs" / "zone_b" / "pgood_analysis.json"),
        ("--qubo",      ranking_path.parent / "qubo_result.json"),
        ("--diag",      ranking_path.parent / "mcmc_diagnostics.json"),
        ("--freq-match", PROJECT_ROOT / "outputs" / "zone_b" / "freq_match.json"),
        ("--morphemes", PROJECT_ROOT / "outputs" / "morpheme_segments.json"),
    ]
    for flag, path in optional_args:
        if path.exists():
            cmd += [flag, str(path)]
    return _run("decipherment_report", cmd, dry_run=dry_run)


def step5_zone_c(smoke_test: bool = False, dry_run: bool = False) -> tuple[int, float]:
    """Zone C MCMC + beam-search decipherment."""
    if not dry_run:
        lm_dir = PROJECT_ROOT / "data" / "language_models"
        if not _check_any_file(lm_dir, "*.json", "Language models (run Step 1 first)"):
            return 1, 0.0
        if not _check(
            PROJECT_ROOT / "outputs" / "embeddings_cache.pt",
            "Embeddings cache (run Step 2 first)",
        ):
            return 1, 0.0

    cmd = [sys.executable, "scripts/run_decipherment.py"]
    if smoke_test:
        cmd.append("--smoke-test")
    return _run("run_decipherment", cmd, dry_run=dry_run)


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------

def _write_manifest(results: list[dict[str, Any]], dry_run: bool) -> None:
    manifest = {
        "generated": datetime.now(tz=timezone.utc).isoformat(),
        "dry_run": dry_run,
        "steps": results,
    }
    out_path = PROJECT_ROOT / "outputs" / "pipeline_run.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    log.info("Run manifest written: %s", out_path.relative_to(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="hackingrongo end-to-end pipeline orchestrator.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run Step 2 with 1 epoch / batch 8 for a fast wiring check.",
    )
    p.add_argument(
        "--skip-training",
        action="store_true",
        help="Skip Step 2.  Requires outputs/embeddings_cache.pt to exist.",
    )
    p.add_argument(
        "--steps",
        default=None,
        metavar="N[,N...]",
        help=(
            "Comma-separated step numbers to run (1–5).  "
            "Omit to run all.  Example: --steps 1,2,3"
        ),
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing them.",
    )
    p.add_argument(
        "--keep-going",
        action="store_true",
        help="Continue to subsequent steps even when a step fails.",
    )
    return p.parse_args()


def _parse_steps(steps_str: str | None) -> set[str]:
    """Parse --steps value; return set of enabled step IDs."""
    valid = {"1", "1b", "2", "3", "4", "4a", "4ar", "4b", "4c", "4d", "4e", "4f",
             "4g", "4h", "4i", "4j", "4k", "4l", "4m", "5", "5b"}
    if steps_str is None:
        return valid
    result: set[str] = set()
    for part in steps_str.split(","):
        part = part.strip()
        if part == "4":
            result.update({"4a", "4ar", "4b", "4c", "4d", "4e", "4f",
                           "4g", "4h", "4i", "4j", "4k", "4l", "4m"})
        elif part == "5":
            result.update({"5", "5b"})
        elif part in valid:
            result.add(part)
        else:
            raise SystemExit(
                f"Invalid step {part!r}. Valid values: {', '.join(sorted(valid))}."
            )
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = _parse_args()
    enabled = _parse_steps(args.steps)
    dry_run = args.dry_run

    if dry_run:
        log.info("%s", _yellow("DRY RUN — no commands will be executed"))

    wall_start = time.monotonic()
    results: list[dict[str, Any]] = []

    # ── Step registry ────────────────────────────────────────────────────────
    # Each entry: (step_id, label, callable, enabled_condition)
    steps: list[tuple[str, str, Any]] = [
        ("1",   "Build language models",              lambda: step1_build_lms(dry_run)),
        ("1b",  "Segment 3D renders → glyph crops",   lambda: step1b_segment_3d_glyphs(dry_run)),
        ("2",  "Train Zone A autoencoder",         lambda: step2_train_autoencoder(args.smoke_test, dry_run)),
        ("3",  "Analyse embeddings (Zone A)",      lambda: step3_analyze_embeddings(dry_run)),
        ("4a",  "IC / entropy sensitivity",          lambda: step4a_entropy(dry_run)),
        ("4ar", "IC / entropy HTML report",         lambda: step4a_entropy_report(dry_run)),
        ("4b",  "G² contact sensitivity",           lambda: step4b_contact(dry_run)),
        ("4c", "Compound glyph detection",         lambda: step4c_compound_detector(dry_run)),
        ("4d", "Compound HTML report",             lambda: step4d_compound_report(dry_run)),
        ("4e", "Parallel passage cross-reference", lambda: step4e_parallel_passages(dry_run)),
        ("4f", "Diachronic passage report",        lambda: step4f_passage_report(dry_run)),
        ("4g", "Astronomical glyph analysis",     lambda: step4g_astronomical(dry_run)),
        ("4h", "Astronomical HTML report",         lambda: step4h_astronomical_report(dry_run)),
        ("4i", "Quantum hardness (p_good) analysis",  lambda: step4i_pgood_analysis(dry_run)),
        ("4j", "QUBO quantum annealing key search",   lambda: step4j_qubo_decipherment(dry_run)),
        ("4k", "Zone C fusion layer training",        lambda: step4k_train_fusion(args.smoke_test, dry_run)),
        ("4l", "Frequency-language match",             lambda: step4l_freq_match(dry_run)),
        ("4m", "Morpheme segmentation",                lambda: step4m_morpheme_seg(dry_run)),
        ("5",  "Zone C decipherment",               lambda: step5_zone_c(args.smoke_test, dry_run)),
        ("5b", "Zone C HTML report",               lambda: step5b_decipherment_report(dry_run)),
    ]

    def _step_enabled(sid: str) -> bool:
        if sid not in enabled:
            # Also check if top-level number is enabled
            top = sid[0]
            if top not in enabled:
                return False
        if sid == "2" and args.skip_training:
            return False
        return True

    # ── Run ──────────────────────────────────────────────────────────────────
    any_failed = False

    for sid, label, fn in steps:
        if not _step_enabled(sid):
            log.info("%s  %s", _dim(f"[{sid}]"), _dim(f"SKIP  {label}"))
            results.append({"step": sid, "label": label, "status": "skipped"})
            continue

        banner = f"{'─' * 60}\n  Step {sid}  {label}\n{'─' * 60}"
        print()
        print(_bold(_cyan(banner)))
        t0 = time.monotonic()

        rc, elapsed = fn()
        duration = time.monotonic() - t0
        duration_str = f"{duration:.1f}s"

        if rc == 0:
            print(_bold(_green(f"  ✓  Step {sid} complete  ({duration_str})")))
            results.append({
                "step": sid, "label": label,
                "status": "ok", "elapsed_s": round(duration, 2),
            })
        else:
            print(_bold(_red(f"  ✗  Step {sid} FAILED  (exit {rc}, {duration_str})")))
            results.append({
                "step": sid, "label": label,
                "status": "failed", "exit_code": rc, "elapsed_s": round(duration, 2),
            })
            any_failed = True
            if not args.keep_going:
                log.error(
                    "Pipeline stopped at Step %s.  "
                    "Re-run with --keep-going to continue past failures.",
                    sid,
                )
                _write_manifest(results, dry_run)
                sys.exit(rc)

    # ── Summary ──────────────────────────────────────────────────────────────
    total = time.monotonic() - wall_start
    print()
    print(_bold("─" * 60))
    print(_bold("  Pipeline summary"))
    print(_bold("─" * 60))
    for r in results:
        sym  = "✓" if r["status"] == "ok" else ("─" if r["status"] == "skipped" else "✗")
        secs = f"  {r['elapsed_s']:.1f}s" if "elapsed_s" in r else ""
        col  = _green if r["status"] == "ok" else (_dim if r["status"] == "skipped" else _red)
        print(col(f"  [{r['step']}]  {sym}  {r['label']}{secs}"))
    print()
    total_str = f"{total:.1f}s"
    if any_failed:
        print(_bold(_red(f"  Pipeline finished with failures  ({total_str})")))
    else:
        print(_bold(_green(f"  Pipeline complete  ({total_str})")))
    print(_bold("─" * 60))

    _write_manifest(results, dry_run)

    if any_failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
