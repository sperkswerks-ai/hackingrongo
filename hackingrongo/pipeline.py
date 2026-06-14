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
    --ring {1,2,all}    Analysis ring to execute.
                        1   = classical core (no ML or quantum) — default
                        2   = Ring 1 + ML, 3-D processing, and quantum
                        all = everything
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
# Per-step timeout (set once by main(), read by every _run() call)
# ---------------------------------------------------------------------------

_STEP_TIMEOUT: float | None = None  # seconds; None = no limit

# ---------------------------------------------------------------------------
# Ring definitions
# ---------------------------------------------------------------------------

_RING_1_STEPS: frozenset[str] = frozenset({
    # Classical / linguistic core — no ML training, no quantum
    "1",
    "4a", "4ar",
    "4b",
    "4c", "4d",
    "4e", "4f",
    "4l",
    "4m",
    "4n",
    "4o",
    "4s",
    "5", "5b",
})

_RING_2_STEPS: frozenset[str] = _RING_1_STEPS | frozenset({
    # ML, 3-D processing, and quantum extensions
    "1b",
    "2", "3",
    "4g", "4h",
    "4i", "4i_simon", "4i_bv",
    "4j", "4k",
    "4p",
    "4q",
    "4r",
})


# ---------------------------------------------------------------------------
# Stage checkpointing
# ---------------------------------------------------------------------------

_STAGE_CHECKPOINT_DIR = Path(__file__).resolve().parent.parent / "outputs" / "checkpoints" / "pipeline_stages"


def mark_stage_complete(stage_name: str) -> None:
    """Write a .done sentinel for *stage_name* so the next run can skip it."""
    _STAGE_CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    (_STAGE_CHECKPOINT_DIR / f"{stage_name}.done").write_text(
        datetime.now(tz=timezone.utc).isoformat(), encoding="utf-8"
    )


def stage_completed(stage_name: str) -> bool:
    """Return True if *stage_name* has a .done sentinel from a previous run."""
    return (_STAGE_CHECKPOINT_DIR / f"{stage_name}.done").exists()

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
    timeout: float | None = None,
) -> tuple[int, float]:
    """Run a subprocess, stream its output, return (returncode, elapsed_s).

    If *timeout* seconds elapse before the process exits, it is killed and
    returncode -1 is returned so the pipeline can mark the step as failed
    rather than hanging indefinitely.
    """
    log.info("%s  %s", _dim("$"), _dim(" ".join(str(c) for c in cmd)))
    if dry_run:
        return 0, 0.0

    effective_timeout = timeout if timeout is not None else _STEP_TIMEOUT
    t0 = time.monotonic()
    proc = subprocess.Popen(
        cmd,
        cwd=PROJECT_ROOT,
        stdout=sys.stdout,
        stderr=sys.stderr,
        env=env,
        shell=False,
    )
    try:
        proc.wait(timeout=effective_timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
        elapsed = time.monotonic() - t0
        log.error(
            "Step '%s' timed out after %.0fs — process killed.",
            label, elapsed,
        )
        return -1, elapsed
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
                sys.executable, "scripts/exploratory/segment_3d_glyphs.py",
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


def step1_build_lms(dry_run: bool = False, seed: int = 20260606) -> tuple[int, float]:
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
        [sys.executable, "scripts/tooling/build_language_models.py", f"--seed={seed}"],
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
    """Project embeddings to UMAP, cluster, generate divergence + embedding reports."""
    if not dry_run and not _check(
        PROJECT_ROOT / "outputs" / "embeddings_cache.pt",
        "Embeddings cache (run Step 2 first, or restore from Drive)",
    ):
        return 1, 0.0

    return _run(
        "analyze_embeddings",
        # hydra.job.chdir=false keeps cwd at PROJECT_ROOT so the relative
        # cfg.paths.embeddings_cache ("outputs/embeddings_cache.pt") resolves
        # correctly; without it Hydra cd's into its run dir and the cache (and
        # the analysis outputs) land/look in the wrong place.
        [sys.executable, "scripts/analyze_embeddings.py", "hydra.job.chdir=false"],
        dry_run=dry_run,
    )


def step4a_entropy(dry_run: bool = False, seed: int = 20260606) -> tuple[int, float]:
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
            "--seed", str(seed),
        ],
        dry_run=dry_run,
    )


def step4a_entropy_report(dry_run: bool = False, seed: int = 20260606) -> tuple[int, float]:
    """Render IC / entropy + boustrophedon voice-split HTML report."""
    sensitivity_json = PROJECT_ROOT / "outputs" / "sensitivity_analysis.json"
    out = PROJECT_ROOT / "outputs" / "analysis" / "entropy_report.html"
    if not dry_run and not _check(sensitivity_json, "sensitivity_analysis.json (run Step 4a first)"):
        return 1, 0.0
    return _run(
        "entropy_report",
        [
            sys.executable, "-m", "hackingrongo.results.entropy_report",
            "--sensitivity", str(sensitivity_json),
            "--output", str(out),
            "--seed", str(seed),
        ],
        dry_run=dry_run,
    )


def step4b_contact(dry_run: bool = False, seed: int = 20260606) -> tuple[int, float]:
    """G² contact analysis sensitivity under all three dating scenarios."""
    out = PROJECT_ROOT / "outputs" / "contact_sensitivity.json"
    return _run(
        "contact_sensitivity",
        [
            sys.executable, "-m", "hackingrongo.zone_b.contact_analysis",
            "--scenario", "all",
            "--output", str(out),
            "--seed", str(seed),
        ],
        dry_run=dry_run,
    )


def step4c_compound_detector(dry_run: bool = False, seed: int = 20260606) -> tuple[int, float]:
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
            "--seed", str(seed),
        ],
        dry_run=dry_run,
    )


def step4d_compound_report(dry_run: bool = False, seed: int = 20260606) -> tuple[int, float]:
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
            "--seed", str(seed),
        ],
        dry_run=dry_run,
    )

def step4e_parallel_passages(dry_run: bool = False, seed: int = 20260606) -> tuple[int, float]:
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
            "--seed", str(seed),
        ],
        dry_run=dry_run,
    )


def step4f_passage_report(dry_run: bool = False, seed: int = 20260606) -> tuple[int, float]:
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
            "--seed", str(seed),
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
            # NOTE: astronomical_analysis uses argparse, not Hydra — passing
            # hydra.job.chdir=false makes argparse abort ("unrecognized arguments").
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
            "--n-samples",  "2000",
            "--output",     str(PROJECT_ROOT / "outputs" / "zone_b" /
                                "pgood_analysis.json"),
        ],
        dry_run=dry_run,
    )


def step4i_simon_decipherment(dry_run: bool = False) -> tuple[int, float]:
    """Simon's algorithm on diachronic key-change events P007 and P012."""
    variants = PROJECT_ROOT / "data" / "parallels" / "parallel_variants_auto.json"
    if not dry_run and not _check(variants, "parallel_variants_auto.json (run Step 4e first)"):
        return 1, 0.0
    return _run(
        "simon_decipherment",
        [
            sys.executable, "scripts/run_simon_decipherment.py",
            "--variants-file", str(variants),
            "--output", str(PROJECT_ROOT / "outputs" / "quantum" / "simon_all_results.json"),
        ],
        dry_run=dry_run,
    )


def step4i_bv_ic_analysis(dry_run: bool = False) -> tuple[int, float]:
    """Bernstein-Vazirani algorithm on the IC contribution distribution."""
    return _run(
        "bv_ic_analysis",
        [
            sys.executable, "scripts/run_bv_ic_analysis.py",
            "--corpus-dir", str(PROJECT_ROOT / "data" / "corpus"),
            "--output",     str(PROJECT_ROOT / "outputs" / "quantum" / "bv_ic_result.json"),
        ],
        dry_run=dry_run,
    )


def step4p_qksvm_parallels(dry_run: bool = False) -> tuple[int, float]:
    """Projected QK-SVM soft parallel passage detection."""
    variants = PROJECT_ROOT / "data" / "parallels" / "parallel_variants_auto.json"
    if not dry_run and not _check(variants, "parallel_variants_auto.json (run Step 4e first)"):
        return 1, 0.0
    return _run(
        "qksvm_parallels",
        [
            sys.executable, "scripts/run_qksvm_parallels.py",
            "--corpus-dir",   str(PROJECT_ROOT / "data" / "corpus"),
            "--variants-file", str(variants),
            "--output",       str(PROJECT_ROOT / "outputs" / "quantum" / "soft_parallels_qksvm.json"),
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


def step4q_qaoa_decipherment(dry_run: bool = False) -> tuple[int, float]:
    """QAOA hybrid decipherment: quantum-approximate optimization over top signs."""
    ranking = PROJECT_ROOT / "outputs" / "decipherment" / "ranking.json"
    cmd = [
        sys.executable, "scripts/run_qaoa_decipherment.py",
        "--corpus-dir", str(PROJECT_ROOT / "data" / "corpus"),
        "--lm-dir",     str(PROJECT_ROOT / "data" / "language_models"),
        # statevector runs the logical 16-qubit ansatz directly.  fake_brisbane
        # transpiles onto the full 127-qubit device, which its BasicSimulator
        # (Aer is not installed) rejects at >24 qubits.  Real-hardware demos use
        # --backend ibmq via a manual invocation.
        "--backend",    "statevector",
        "--output",     str(PROJECT_ROOT / "outputs" / "decipherment" /
                            "qaoa_result.json"),
    ]
    if ranking.exists():
        cmd += ["--init-from", str(ranking)]
    return _run("qaoa_decipherment", cmd, dry_run=dry_run)


def step4r_network_centrality(dry_run: bool = False) -> tuple[int, float]:
    """Bigram PMI network centrality: betweenness, PageRank, HITS, diachronic shift."""
    cmd = [
        sys.executable, "scripts/run_network_centrality.py",
        "--corpus-dir", str(PROJECT_ROOT / "data" / "corpus"),
        "--output-dir", str(PROJECT_ROOT / "outputs" / "network"),
    ]
    return _run("network_centrality", cmd, dry_run=dry_run)


def step4l_freq_match(dry_run: bool = False, seed: int = 20260606) -> tuple[int, float]:
    """Frequency-language match: Zipf α, Spearman ρ, χ² fit vs. each LM."""
    return _run(
        "freq_match",
        [
            sys.executable, "scripts/run_freq_match.py",
            "--corpus-dir", str(PROJECT_ROOT / "data" / "corpus"),
            "--lm-dir",     str(PROJECT_ROOT / "data" / "language_models"),
            "--output",     str(PROJECT_ROOT / "outputs" / "zone_b" / "freq_match.json"),
            "--seed",       str(seed),
        ],
        dry_run=dry_run,
    )


def step4m_morpheme_seg(dry_run: bool = False, seed: int = 20260606) -> tuple[int, float]:
    """Zellig Harris successor-entropy morpheme boundary segmentation."""
    return _run(
        "morpheme_segmentation",
        [
            sys.executable, "scripts/segment_morphemes.py",
            "--corpus-dir", str(PROJECT_ROOT / "data" / "corpus"),
            "--output",     str(PROJECT_ROOT / "outputs" / "morpheme_segments.json"),
            "--seed",       str(seed),
        ],
        dry_run=dry_run,
    )


def step4o_contact_partition(dry_run: bool = False, seed: int = 20260606) -> tuple[int, float]:
    """Contact partition: G² bipartite sign frequency analysis (pre vs post contact).

    Runs contact_analysis.py with --report to produce:
      outputs/contact_partition.json          (flat record list, backward-compat)
      outputs/analysis/contact_partition_report.html
      outputs/contact_partition_bipartite.html (plotly graph, if plotly installed)
    """
    return _run(
        "contact_partition",
        [
            sys.executable, "-m", "hackingrongo.zone_b.contact_analysis",
            "--output",  str(PROJECT_ROOT / "outputs" / "contact_partition.json"),
            "--report",  str(PROJECT_ROOT / "outputs" / "analysis" / "contact_partition_report.html"),
            "--plot",    str(PROJECT_ROOT / "outputs" / "contact_partition_bipartite.html"),
            "--seed",    str(seed),
        ],
        dry_run=dry_run,
    )


def step4s_diachronic_substitutions(dry_run: bool = False, seed: int = 20260606) -> tuple[int, float]:
    """Mine pre↔post-contact sign substitution pairs from parallel passages.

    Slot-aligns cross-stratum parallel passages and corroborates each
    substitution against the contact-partition G² bias, emitting tie_pairs
    consumed by the MCMC equivalence-tie constraint in run_decipherment.py.

    Depends on step4e_parallel_passages (parallel_variants_auto.json) and
    step4o_contact_partition (contact_partition.json); both are upstream.

    Output
    ------
    outputs/analysis/diachronic_substitutions.json
    """
    parallel_auto = PROJECT_ROOT / "data" / "parallels" / "parallel_variants_auto.json"
    parallel_base = PROJECT_ROOT / "data" / "parallels" / "parallel_variants.json"
    if not dry_run and not (parallel_auto.exists() or parallel_base.exists()):
        log.warning(
            "step4s_diachronic_substitutions: no parallel_variants*.json found — "
            "skipping (run step4e_parallel_passages first)."
        )
        return 1, 0.0
    return _run(
        "diachronic_substitutions",
        [sys.executable, "scripts/mine_diachronic_substitutions.py"],
        dry_run=dry_run,
    )


def step4n_pozdniakov(dry_run: bool = False, seed: int = 20260606) -> tuple[int, float]:
    """Pozdniakov (1996/2011) paradigmatic analysis + HTML report.

    Identifies sign substitution pairs from parallel passage variants,
    groups them into equivalence classes, and compares to Pozdniakov's
    published 15 reference classes.  Also cross-validates against the
    MCMC top hypothesis when ranking.json is available.

    Outputs
    -------
    outputs/analysis/pozdniakov_paradigmatic.json
    outputs/analysis/pozdniakov_report.html
    """
    parallel_auto = PROJECT_ROOT / "data" / "parallels" / "parallel_variants_auto.json"
    parallel_base = PROJECT_ROOT / "data" / "parallels" / "parallel_variants.json"
    if not dry_run and not (parallel_auto.exists() or parallel_base.exists()):
        log.warning(
            "step4n_pozdniakov: no parallel_variants*.json found — skipping "
            "(run step4e_parallel_passages first)."
        )
        return 1, 0.0
    return _run(
        "pozdniakov_paradigmatic",
        [
            sys.executable, "scripts/generate_pozdniakov_report.py",
            "--seed", str(seed),
        ],
        dry_run=dry_run,
    )


def step4k_train_fusion(
    smoke_test: bool = False,
    dry_run: bool = False,
) -> tuple[int, float]:
    """Train the Zone C fusion layer in-process (no subprocess).

    Loads Zone A embeddings and Zone B sign classifications, assembles
    Zone B prior vectors, trains :class:`~hackingrongo.zone_c.fusion.FusionLayer`
    with early-stop on validation-loss plateau, then saves the best checkpoint
    and marks the stage done.

    Requires
    --------
    * ``outputs/embeddings_cache.pt`` — Zone A autoencoder embeddings
    * ``outputs/zone_b_cache.pkl`` — :class:`~hackingrongo.zone_b.sign_classifier.SignInventory`
      (optional: falls back to neutral UNKNOWN classifications when absent)
    * ``data/corpus/`` — corpus for IC / bigram features 9–12

    Produces
    --------
    ``outputs/checkpoints/fusion_layer.pt``
    """
    import json as _json
    import pickle
    import time

    embeddings_cache = PROJECT_ROOT / "outputs" / "embeddings_cache.pt"
    zone_b_cache_path = PROJECT_ROOT / "outputs" / "zone_b_cache.pkl"
    compound_json = PROJECT_ROOT / "outputs" / "analysis" / "compound_candidates.json"
    checkpoint_path = PROJECT_ROOT / "outputs" / "checkpoints" / "fusion_layer.pt"

    if dry_run:
        log.info("DRY RUN: step4k would train fusion → %s", checkpoint_path)
        return 0, 0.0

    if not _check(embeddings_cache, "Embeddings cache (run Step 2 first)"):
        return 1, 0.0

    t0 = time.monotonic()

    import numpy as np
    import torch
    import torch.nn as nn
    from omegaconf import OmegaConf
    from torch.utils.data import DataLoader, TensorDataset

    from hackingrongo.zone_b.priors import (
        ZoneBPriorBuilder,
        build_zone_b_prior,
        compute_corpus_sign_stats,
    )
    from hackingrongo.zone_b.sign_classifier import (
        SignClass,
        SignClassification,
        SignInventory,
    )
    from hackingrongo.zone_c.fusion import (
        FusionLayer,
        build_fusion_optimizer,
        build_fusion_scheduler,
        save_fusion_checkpoint,
        train_fusion_epoch,
    )

    try:
        cfg = OmegaConf.load(PROJECT_ROOT / "conf" / "config.yaml")
    except Exception as exc:
        log.error("Failed to load config.yaml: %s", exc)
        return 1, time.monotonic() - t0

    if smoke_test:
        cfg = OmegaConf.merge(
            cfg,
            OmegaConf.create({"zone_c": {"fusion": {"num_epochs": 2, "batch_size": 8}}}),
        )

    num_epochs: int = int(cfg.zone_c.fusion.num_epochs)
    batch_size: int = int(cfg.zone_c.fusion.batch_size)
    zone_b_dim: int = int(cfg.zone_b.prior_output_dim)
    output_dim: int = int(cfg.zone_c.fusion.output_dim)
    val_fraction: float = 0.15
    patience: int = 5
    min_delta: float = 1e-5

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(
        "4k fusion: device=%s  epochs=%d  batch=%d  output_dim=%d",
        device, num_epochs, batch_size, output_dim,
    )

    # ── Load Zone A embeddings ────────────────────────────────────────────────
    emb_data = torch.load(embeddings_cache, weights_only=True)
    zone_a: torch.Tensor = emb_data["embeddings"].float()
    barthel_codes: list[str] = list(emb_data["barthel_codes"])
    sign_codes_unique: list[str] = sorted(set(barthel_codes))
    log.info(
        "Zone A: %d tokens, dim=%d, %d unique codes",
        len(barthel_codes), zone_a.shape[1], len(sign_codes_unique),
    )

    # ── Load sign classifications from zone_b_cache.pkl (or defaults) ────────
    inventory: SignInventory | None = None
    if zone_b_cache_path.exists():
        try:
            obj = pickle.loads(zone_b_cache_path.read_bytes())
            if isinstance(obj, SignInventory):
                inventory = obj
                log.info("zone_b_cache.pkl loaded: %d sign(s)", len(inventory.classifications))
            else:
                log.warning(
                    "zone_b_cache.pkl contains %s, expected SignInventory — using defaults",
                    type(obj).__name__,
                )
        except Exception as exc:
            log.warning("Could not load zone_b_cache.pkl (%s) — using defaults", exc)

    if inventory is None:
        inventory = SignInventory(classifications={
            code: SignClassification(
                code=code,
                sign_class=SignClass.UNKNOWN,
                confidence=0.0,
                frequency_percentile=0.5,
                omission_rate=0.0,
                positional_entropy=0.0,
            )
            for code in sign_codes_unique
        })
        log.info("Using neutral UNKNOWN classifications for all %d sign codes", len(sign_codes_unique))

    # ── Load compound scores ──────────────────────────────────────────────────
    compound_scores: dict[str, float] = {}
    if compound_json.exists():
        try:
            raw = _json.loads(compound_json.read_text(encoding="utf-8"))
            candidates = raw if isinstance(raw, list) else raw.get("candidates", [])
            for c in candidates:
                code = str(c.get("barthel_code", c.get("code", "")))
                prob = float(c.get("compound_probability", c.get("score", 0.0)))
                if code:
                    compound_scores[code] = prob
        except Exception as exc:
            log.warning("Could not load compound scores: %s", exc)

    # ── Corpus statistics for features 9–12 ──────────────────────────────────
    corpus_dir = PROJECT_ROOT / "data" / "corpus"
    corpus_stats = None
    if corpus_dir.exists():
        try:
            corpus_stats = compute_corpus_sign_stats(corpus_dir)
        except Exception as exc:
            log.warning("compute_corpus_sign_stats failed (%s) — features 9-12 default to 0", exc)

    # ── Zone B prior: one vector per unique sign code, then expand per token ──
    prior_per_code, builder = build_zone_b_prior(
        sign_codes=sign_codes_unique,
        inventory=inventory,
        cfg=cfg,
        compound_scores=compound_scores,
        corpus_stats=corpus_stats,
        device=device,
    )
    code_to_idx = {code: i for i, code in enumerate(sign_codes_unique)}
    token_idx = torch.tensor(
        [code_to_idx.get(c, 0) for c in barthel_codes], dtype=torch.long
    )
    zone_b_expanded: torch.Tensor = prior_per_code[token_idx].detach().cpu()

    # ── Self-supervised targets: truncated SVD of Zone A ─────────────────────
    try:
        from sklearn.decomposition import TruncatedSVD  # type: ignore[import]
        n, d = zone_a.shape
        n_components = min(output_dim, d, n - 1)
        svd = TruncatedSVD(n_components=n_components, random_state=42)
        coords = svd.fit_transform(zone_a.numpy().astype(np.float32))
        if n_components < output_dim:
            coords = np.concatenate(
                [coords, np.zeros((n, output_dim - n_components), dtype=np.float32)], axis=1
            )
        targets = torch.from_numpy(coords[:, :output_dim].astype(np.float32))
        log.info("SVD targets: %d components → shape %s", n_components, list(targets.shape))
    except ImportError:
        log.warning("scikit-learn unavailable — using random-projection targets")
        targets = torch.randn(zone_a.shape[0], output_dim)

    # ── Patch config if actual zone_a_dim differs from config value ───────────
    actual_a_dim = zone_a.shape[1]
    if actual_a_dim != int(cfg.zone_c.fusion.zone_a_dim):
        log.warning(
            "Config zone_a_dim=%d but actual embedding dim=%d — patching config",
            int(cfg.zone_c.fusion.zone_a_dim), actual_a_dim,
        )
        cfg = OmegaConf.merge(
            cfg,
            OmegaConf.create({"zone_c": {"fusion": {"zone_a_dim": actual_a_dim}}}),
        )

    # ── Train / validation split ──────────────────────────────────────────────
    n_total = len(zone_a)
    n_val = max(1, int(n_total * val_fraction))
    n_train = n_total - n_val
    perm = torch.randperm(n_total, generator=torch.Generator().manual_seed(42))
    train_idx, val_idx = perm[:n_train], perm[n_train:]

    train_ds = TensorDataset(
        zone_a[train_idx], zone_b_expanded[train_idx], targets[train_idx]
    )
    val_ds = TensorDataset(
        zone_a[val_idx], zone_b_expanded[val_idx], targets[val_idx]
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=0)
    log.info("4k fusion: train=%d  val=%d  batches/epoch=%d", n_train, n_val, len(train_loader))

    # ── Model, optimizer, scheduler ──────────────────────────────────────────
    fusion = FusionLayer(cfg).to(device)
    optimizer = build_fusion_optimizer(fusion, cfg)
    scheduler = build_fusion_scheduler(optimizer, cfg)
    loss_fn = nn.MSELoss()

    # ── Training loop with early stopping on val loss ─────────────────────────
    best_val_loss = float("inf")
    no_improve_epochs = 0
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(num_epochs):
        train_loss = train_fusion_epoch(fusion, train_loader, optimizer, cfg, device, epoch)

        fusion.eval()
        val_total = 0.0
        val_batches = 0
        with torch.no_grad():
            for za, zb, tgt in val_loader:
                za, zb, tgt = za.to(device), zb.to(device), tgt.to(device)
                val_total += loss_fn(fusion(za, zb), tgt).item()
                val_batches += 1
        val_loss = val_total / max(val_batches, 1)

        log.info(
            "Epoch %d/%d  train=%.6f  val=%.6f",
            epoch + 1, num_epochs, train_loss, val_loss,
        )

        if scheduler is not None:
            scheduler.step()

        if val_loss < best_val_loss - min_delta:
            best_val_loss = val_loss
            no_improve_epochs = 0
            save_fusion_checkpoint(fusion, optimizer, epoch + 1, val_loss, checkpoint_path)
        else:
            no_improve_epochs += 1
            if no_improve_epochs >= patience:
                log.info(
                    "Early stopping at epoch %d — no val improvement for %d epochs",
                    epoch + 1, patience,
                )
                break

    # Write final checkpoint if early stopping fired before any improvement saved one
    if not checkpoint_path.exists():
        save_fusion_checkpoint(fusion, optimizer, num_epochs, best_val_loss, checkpoint_path)

    log.info(
        "4k fusion complete: best_val_loss=%.6f  checkpoint=%s",
        best_val_loss, checkpoint_path.relative_to(PROJECT_ROOT),
    )
    mark_stage_complete("4k_fusion")
    return 0, time.monotonic() - t0


def step5b_decipherment_report(dry_run: bool = False, seed: int = 20260606) -> tuple[int, float]:
    """Render the scholar-facing HTML report from Zone C ranking output."""
    ranking_path = PROJECT_ROOT / "outputs" / "decipherment" / "ranking.json"
    if not dry_run and not _check(ranking_path, "ranking.json (run Step 5 first)"):
        return 1, 0.0

    out = ranking_path.parent / "decipherment_report.html"
    cmd = [
        sys.executable, "-m", "hackingrongo.results.decipherment_report",
        "--ranking", str(ranking_path),
        "--output",  str(out),
        "--seed",    str(seed),
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


def step5_zone_c(
    smoke_test: bool = False,
    dry_run: bool = False,
    skip_fusion: bool = False,
    seed: int = 20260606,
) -> tuple[int, float]:
    """Zone C MCMC + beam-search decipherment.

    When ``outputs/checkpoints/fusion_layer.pt`` exists and *skip_fusion* is
    False, passes ``--fusion-checkpoint`` to the decipherment script so the
    MCMC uses fused (Zone A + Zone B) embeddings for sign proposal weights
    instead of raw sequential-entropy weights.  Pass ``--skip-fusion`` at the
    CLI to force the pre-fusion fallback behaviour.
    """
    if not dry_run:
        lm_dir = PROJECT_ROOT / "data" / "language_models"
        if not _check_any_file(lm_dir, "*.json", "Language models (run Step 1 first)"):
            return 1, 0.0
        if not _check(
            PROJECT_ROOT / "outputs" / "embeddings_cache.pt",
            "Embeddings cache (run Step 2 first)",
        ):
            return 1, 0.0

    fusion_ckpt = PROJECT_ROOT / "outputs" / "checkpoints" / "fusion_layer.pt"
    cmd = [sys.executable, "scripts/run_decipherment.py", f"--seed={seed}"]
    if smoke_test:
        cmd.append("--smoke-test")
    if not skip_fusion and fusion_ckpt.exists():
        cmd.append(f"--fusion-checkpoint={fusion_ckpt}")
        log.info("Fusion checkpoint found — passing to decipherment: %s", fusion_ckpt)
    elif skip_fusion:
        log.info("--skip-fusion: using raw Zone A embeddings for MCMC proposal weights")
    else:
        log.info("No fusion checkpoint — using sequential-entropy proposal weights")
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
    p.add_argument(
        "--step-timeout",
        type=float,
        default=3600.0,
        metavar="SECONDS",
        help=(
            "Hard wall-time limit per step in seconds (default: 3600 = 1 hour).  "
            "The step is killed and marked failed if it exceeds this limit.  "
            "Set to 0 to disable."
        ),
    )
    p.add_argument(
        "--no-cache",
        action="store_true",
        help="Ignore stage checkpoints and re-run every step.",
    )
    p.add_argument(
        "--skip-fusion",
        action="store_true",
        help=(
            "Step 5: ignore the fusion checkpoint even if present and fall back "
            "to sequential-entropy MCMC proposal weights."
        ),
    )
    p.add_argument(
        "--seed",
        type=int,
        default=20260606,
        metavar="INT",
        help="Global RNG seed threaded into every Ring 1 subprocess (default: 20260606).",
    )
    p.add_argument(
        "--ring",
        choices=["1", "2", "all"],
        default="1",
        metavar="{1,2,all}",
        help=(
            "Analysis ring to run: "
            "1 = classical core (no ML/quantum); "
            "2 = Ring 1 + ML, 3-D, and quantum; "
            "all = every pipeline step. "
            "Default: 1. "
            "Ignored when --steps is provided."
        ),
    )
    return p.parse_args()


def _parse_steps(steps_str: str | None) -> set[str]:
    """Parse --steps value; return set of enabled step IDs."""
    valid = {"1", "1b", "2", "3", "4", "4a", "4ar", "4b", "4c", "4d", "4e", "4f",
             "4g", "4h", "4i", "4i_simon", "4i_bv", "4j", "4k", "4l", "4m",
             "4p", "5", "5b"}
    if steps_str is None:
        return valid
    result: set[str] = set()
    for part in steps_str.split(","):
        part = part.strip()
        if part == "4":
            result.update({"4a", "4ar", "4b", "4c", "4d", "4e", "4f",
                           "4g", "4h", "4i", "4i_simon", "4i_bv", "4j", "4k", "4l", "4m",
                           "4p"})
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
    global _STEP_TIMEOUT  # noqa: PLW0603
    args = _parse_args()
    dry_run = args.dry_run
    seed = args.seed

    from hackingrongo.repro import set_global_seed
    set_global_seed(seed)

    # ── MLflow experiment tracking (optional — pipeline works without mlflow) ──
    _mlflow: Any = None
    try:
        import os as _os
        import mlflow as _mlflow
        _os.environ.setdefault("MLFLOW_TRACKING_URI", "./mlruns")
        _mlflow.set_experiment("hackingrongo_pipeline")
    except ImportError:
        _mlflow = None

    # Ring filtering: when --steps is absent let --ring control what runs.
    if args.steps is not None:
        enabled = _parse_steps(args.steps)
        _prefix_fallback = False  # _parse_steps already expanded bare step numbers
    elif args.ring == "1":
        enabled = _RING_1_STEPS
        _prefix_fallback = False
    elif args.ring == "2":
        enabled = _RING_2_STEPS
        _prefix_fallback = False
    else:
        enabled = _parse_steps(None)  # all — "4" in enabled activates 4n/4o/etc.
        _prefix_fallback = True

    _STEP_TIMEOUT = args.step_timeout if args.step_timeout > 0 else None
    if _STEP_TIMEOUT:
        log.info("Step timeout: %.0fs per step", _STEP_TIMEOUT)

    ring_label = {"1": "Ring 1 (classical core)", "2": "Ring 2 (core + ML/quantum)", "all": "all steps"}
    log.info("Ring: %s  (%d step(s) enabled)", ring_label.get(args.ring, args.ring), len(enabled))

    if dry_run:
        log.info("%s", _yellow("DRY RUN — no commands will be executed"))

    wall_start = time.monotonic()
    results: list[dict[str, Any]] = []

    # ── Step registry ────────────────────────────────────────────────────────
    # Each entry: (step_id, label, callable, enabled_condition)
    steps: list[tuple[str, str, Any]] = [
        ("1",   "Build language models",              lambda: step1_build_lms(dry_run, seed)),
        ("1b",  "Segment 3D renders → glyph crops",   lambda: step1b_segment_3d_glyphs(dry_run)),
        ("2",  "Train Zone A autoencoder",         lambda: step2_train_autoencoder(args.smoke_test, dry_run)),
        ("3",  "Analyse embeddings (Zone A)",      lambda: step3_analyze_embeddings(dry_run)),
        ("4a",  "IC / entropy sensitivity",          lambda: step4a_entropy(dry_run, seed)),
        ("4ar", "IC / entropy HTML report",         lambda: step4a_entropy_report(dry_run, seed)),
        ("4b",  "G² contact sensitivity",           lambda: step4b_contact(dry_run, seed)),
        ("4c", "Compound glyph detection",         lambda: step4c_compound_detector(dry_run, seed)),
        ("4d", "Compound HTML report",             lambda: step4d_compound_report(dry_run, seed)),
        ("4e", "Parallel passage cross-reference", lambda: step4e_parallel_passages(dry_run, seed)),
        ("4f", "Diachronic passage report",        lambda: step4f_passage_report(dry_run, seed)),
        ("4g", "Astronomical glyph analysis",     lambda: step4g_astronomical(dry_run)),
        ("4h", "Astronomical HTML report",         lambda: step4h_astronomical_report(dry_run)),
        ("4i", "Quantum hardness (p_good) analysis",  lambda: step4i_pgood_analysis(dry_run)),
        ("4i_simon", "Simon's algo on diachronic key-changes", lambda: step4i_simon_decipherment(dry_run)),
        ("4i_bv",   "BV algorithm on IC distribution",         lambda: step4i_bv_ic_analysis(dry_run)),
        ("4p", "QK-SVM soft parallel detection",            lambda: step4p_qksvm_parallels(dry_run)),
        ("4j", "QUBO quantum annealing key search",   lambda: step4j_qubo_decipherment(dry_run)),
        ("4q", "QAOA hybrid decipherment",            lambda: step4q_qaoa_decipherment(dry_run)),
        ("4r", "Network centrality (PMI bigram graph)", lambda: step4r_network_centrality(dry_run)),
        ("4k", "Zone C fusion layer training",        lambda: step4k_train_fusion(args.smoke_test, dry_run)),
        ("4l", "Frequency-language match",             lambda: step4l_freq_match(dry_run, seed)),
        ("4m", "Morpheme segmentation",                lambda: step4m_morpheme_seg(dry_run, seed)),
        ("4o", "Contact partition (bipartite)",          lambda: step4o_contact_partition(dry_run, seed)),
        ("4n", "Pozdniakov paradigmatic analysis",      lambda: step4n_pozdniakov(dry_run, seed)),
        ("4s", "Diachronic substitution mining",         lambda: step4s_diachronic_substitutions(dry_run, seed)),
        ("5",  "Zone C decipherment",               lambda: step5_zone_c(args.smoke_test, dry_run, args.skip_fusion, seed)),
        ("5b", "Zone C HTML report",               lambda: step5b_decipherment_report(dry_run, seed)),
    ]

    def _step_enabled(sid: str) -> bool:
        if sid not in enabled:
            # Prefix fallback: "4" in enabled → "4n", "4o", … are also enabled.
            # Only active in --ring all mode; ring sets list steps explicitly.
            if not _prefix_fallback or sid[0] not in enabled:
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

        # Skip steps already completed in a previous run (unless --no-cache).
        if not args.no_cache and stage_completed(sid):
            log.info(
                "%s  %s", _dim(f"[{sid}]"),
                _dim(f"ALREADY DONE (checkpoint)  {label}"),
            )
            results.append({"step": sid, "label": label, "status": "cached"})
            continue

        banner = f"{'─' * 60}\n  Step {sid}  {label}\n{'─' * 60}"
        print()
        print(_bold(_cyan(banner)))
        t0 = time.monotonic()

        if _mlflow is not None:
            with _mlflow.start_run(run_name=sid):
                _mlflow.log_param("step", sid)
                _mlflow.log_param("ring", args.ring)
                _mlflow.log_param("steps", args.steps or "")
                rc, elapsed = fn()
                duration = time.monotonic() - t0
                _mlflow.log_metric("return_code", float(rc))
                _mlflow.log_metric("elapsed_s", round(duration, 2))
        else:
            rc, elapsed = fn()
            duration = time.monotonic() - t0
        duration_str = f"{duration:.1f}s"

        if rc == 0:
            mark_stage_complete(sid)
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
        status = r["status"]
        if status == "ok":
            sym, col, note = "✓", _green, ""
        elif status == "cached":
            sym, col, note = "✓", _dim, "  (cached)"
        elif status == "skipped":
            sym, col, note = "─", _dim, ""
        else:
            sym, col, note = "✗", _red, ""
        secs = f"  {r['elapsed_s']:.1f}s" if "elapsed_s" in r else ""
        print(col(f"  [{r['step']}]  {sym}  {r['label']}{secs}{note}"))
    print()
    total_str = f"{total:.1f}s"
    if any_failed:
        print(_bold(_red(f"  Pipeline finished with failures  ({total_str})")))
    else:
        print(_bold(_green(f"  Pipeline complete  ({total_str})")))
    print(_bold("─" * 60))

    _write_manifest(results, dry_run)

    if _mlflow is not None:
        with _mlflow.start_run(run_name="__pipeline_total__"):
            _mlflow.log_param("ring", args.ring)
            _mlflow.log_param("steps", args.steps or "")
            _mlflow.log_metric("pipeline_elapsed_s", round(total, 2))

    if any_failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
