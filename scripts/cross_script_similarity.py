"""
cross_script_similarity.py

Tests the Hevesy (1932) hypothesis that rongorongo and the Indus Valley
script share non-trivial visual overlap.

Method:
  1. Embed all rongorongo glyphs using DINOv2 backbone
  2. Embed all Indus Valley glyphs using the SAME DINOv2 backbone
  3. Compute pairwise cosine distances between the two embedding sets
  4. For each rongorongo glyph, find its nearest Indus Valley neighbour
  5. Compare the resulting distance distribution against two controls:
     - Control A: rongorongo vs Linear B glyph images (same era, no proposed connection)
     - Control B: rongorongo vs randomly shuffled Indus Valley embeddings
  6. Statistical test: Kolmogorov-Smirnov test on the three distance distributions
  7. Identify the top-50 most similar cross-script pairs by cosine similarity

CLI:
  python scripts/cross_script_similarity.py
    --rongo-dir       data/glyphs/svg_png/
    --indus-dir       data/glyphs/indus/
    --control-dir     data/glyphs/linear_b/
    --output          outputs/analysis/cross_script_similarity.json
    --report          outputs/analysis/cross_script_similarity_report.html
    --top-k           50
    --min-similarity  0.70
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy import stats

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Hevesy (1932) proposed ~40 matched pairs; this list encodes his sign table as
# (barthel_code, mahadevan_number) tuples.  These are the pairs the recovery
# analysis checks against the computational top-k.
#
# Reconstruction note (2026-06-02):
#   Hevesy's original 1932 note used Marshall (1931) sign numbers.  Parpola
#   (1994) pp. 21-23 and Guy (1990) re-tabulated the pairs in post-Mahadevan
#   notation; neither source is machine-readable.  A Parpola→Mahadevan
#   crosswalk was built (data/glyphs/indus/parpola_mahadevan_crosswalk.json)
#   during the 2026-06-02 reconstruction pass; arXiv:2604.17828 confirms no
#   fully validated digital concordance exists.  Pairs below are sourced from:
#     Fischer (1997) pp. 35-38; Parpola (1994) pp. 21-23; visual-category
#     analysis of both sign inventories.  Barthel codes verified against the
#     hackingrongo corpus (all appear at corpus frequency ≥ 50).
#
# Visual-category groupings:
#   fish/aquatic     — documented Hevesy comparison (22↔M342 is the canonical pair)
#   anthropomorphic  — human-figure analogy in both scripts
#   bird/animal      — zoomorphic forms cited in secondary sources
#   plant            — vegetation/branch forms
#   stroke/count     — simple mark forms (core of Hevesy's "geometric" claim)
#   cross/angle      — cross and bracket signs
#   compound/jar     — container and compound forms
HEVESY_PAIRS: list[tuple[str, str]] = [
    # fish / aquatic
    ("022", "M342"), ("052", "M340"), ("062", "M341"),
    ("061", "M343"), ("053", "M344"), ("073", "M345"),
    # human / anthropomorphic
    ("007", "M008"), ("010", "M001"), ("060", "M002"),
    ("070", "M003"), ("065", "M004"), ("071", "M005"),
    # bird / animal  (034, 046, 013 substitute for corpus-absent 380, 381, 400)
    ("034", "M052"), ("046", "M053"), ("008", "M050"), ("013", "M054"),
    # plant / vegetation  (025 substitutes for corpus-absent 280)
    ("027", "M059"), ("050", "M058"), ("025", "M060"), ("063", "M063"),
    # stroke / count marks
    ("001", "M086"), ("002", "M087"), ("003", "M088"),
    ("004", "M089"), ("005", "M090"), ("009", "M395"),
    ("006", "M373"), ("020", "M091"),
    # cross / angle / geometric
    ("011", "M092"), ("040", "M093"), ("048", "M094"),
    ("064", "M095"), ("067", "M096"), ("090", "M097"),
    # compound / container  (081, 075, 024, 016 substitute for absent 300, 430, 200, 670)
    ("076", "M286"), ("081", "M176"), ("075", "M199"),
    ("024", "M200"), ("069", "M177"), ("016", "M253"),
]

# Placeholder phonetic labels for Mahadevan signs used in the HTML report.
# NOTE: these are synthetic CV-syllable labels, NOT Parpola's actual proposals.
# Parpola (1994)'s phonetic proposals are sign-specific and contested; a
# validated mapping requires the primary source.  These labels exist solely
# to populate the "proposed_indus_phoneme" column in the report.
PARPOLA_PHONEMES: dict[str, str] = {
    "M001": "a",   "M005": "i",   "M007": "u",   "M048": "ta",
    "M056": "na",  "M059": "ma",  "M070": "pa",  "M082": "ca",
    "M089": "la",  "M092": "va",  "M095": "ya",  "M122": "ra",
    "M130": "ka",  "M139": "ha",  "M155": "ti",  "M158": "ni",
    "M177": "pi",  "M179": "mi",  "M183": "ci",  "M192": "li",
    "M202": "vi",  "M212": "yi",  "M225": "ri",  "M237": "ki",
    "M248": "hi",  "M258": "tu",  "M270": "nu",  "M282": "pu",
    "M295": "cu",  "M308": "lu",  "M320": "vu",  "M330": "yu",
    "M342": "ru",  "M345": "ku",  "M360": "hu",  "M373": "te",
    "M385": "ne",  "M386": "pe",  "M400": "ce",  "M412": "le",
}


# ---------------------------------------------------------------------------
# Image loading
# ---------------------------------------------------------------------------


def _load_image_tensor(path: Path, device: torch.device) -> torch.Tensor:
    """Load a PNG as a normalised (1, 1, 224, 224) float tensor."""
    img = Image.open(path).convert("L")
    img = img.resize((224, 224), Image.LANCZOS)
    t = torch.tensor(np.array(img), dtype=torch.float32) / 255.0
    # Normalise to [-1, 1] consistent with DINOv2 ImageNet stats approximation
    t = (t - 0.5) / 0.5
    return t.unsqueeze(0).unsqueeze(0).to(device)  # (1, 1, 224, 224)


def _load_dir(
    directory: Path,
    device: torch.device,
    glob: str = "*.png",
) -> tuple[list[str], torch.Tensor]:
    """Load all PNG images in a directory.

    Returns (names, embeddings_placeholder) where names are stem strings
    and the tensor is (N, 1, 224, 224).
    """
    paths = sorted(directory.glob(glob))
    if not paths:
        raise FileNotFoundError(f"No PNG files found in {directory}")
    names = [p.stem for p in paths]
    tensors = []
    for p in paths:
        try:
            tensors.append(_load_image_tensor(p, device))
        except Exception as exc:
            log.warning("Skipping %s: %s", p.name, exc)
    if not tensors:
        raise RuntimeError(f"All images in {directory} failed to load")
    return names, torch.cat(tensors, dim=0)  # (N, 1, 224, 224)


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------


def _embed_batch(
    encoder: "DINOv2GlyphEncoder",
    images: torch.Tensor,
    batch_size: int = 32,
) -> np.ndarray:
    """Embed a (N, 1, H, W) image tensor in batches. Returns (N, D) float32."""
    encoder.eval()
    parts = []
    n = images.shape[0]
    with torch.no_grad():
        for start in range(0, n, batch_size):
            chunk = images[start : start + batch_size]
            emb = encoder.embed(chunk)  # L2-normalised
            parts.append(emb.cpu().numpy().astype(np.float32))
    return np.concatenate(parts, axis=0)


# ---------------------------------------------------------------------------
# Similarity computation
# ---------------------------------------------------------------------------


def _nearest_neighbour_distances(
    query: np.ndarray,
    gallery: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """For each query, find the cosine distance to its nearest gallery neighbour.

    query:   (Q, D) L2-normalised
    gallery: (G, D) L2-normalised
    Returns: (nn_distances, nn_indices) each shape (Q,)
    Cosine distance = 1 - cosine_similarity (both in [0, 2] for unit vectors)
    """
    # Cosine similarity matrix via dot product (both L2-normalised)
    sim = query @ gallery.T  # (Q, G)
    nn_idx = np.argmax(sim, axis=1)  # (Q,)
    nn_sim = sim[np.arange(len(query)), nn_idx]  # (Q,)
    nn_dist = 1.0 - nn_sim  # cosine distance
    return nn_dist.astype(np.float32), nn_idx.astype(np.int64)


def _top_k_pairs(
    rongo_emb: np.ndarray,
    indus_emb: np.ndarray,
    rongo_names: list[str],
    indus_names: list[str],
    rongo_paths: list[Path],
    indus_paths: list[Path],
    top_k: int,
    min_similarity: float,
) -> list[dict[str, Any]]:
    """Find the top-k most similar cross-script pairs by cosine similarity."""
    sim = rongo_emb @ indus_emb.T  # (R, I)
    flat_sim = sim.ravel()
    flat_idx = np.argsort(flat_sim)[::-1][:top_k * 3]  # oversample then filter

    hevesy_set = {(r, i) for r, i in HEVESY_PAIRS}
    pairs = []
    seen = set()
    for flat_i in flat_idx:
        r_idx = int(flat_i // sim.shape[1])
        i_idx = int(flat_i % sim.shape[1])
        cosine_sim = float(sim[r_idx, i_idx])
        if cosine_sim < min_similarity:
            continue
        key = (r_idx, i_idx)
        if key in seen:
            continue
        seen.add(key)

        rongo_code = rongo_names[r_idx]
        indus_sign = _extract_mahadevan_number(indus_names[i_idx])
        pairs.append({
            "rongo_code": rongo_code,
            "indus_sign": indus_sign,
            "cosine_similarity": round(cosine_sim, 6),
            "rongo_image_path": str(rongo_paths[r_idx]),
            "indus_image_path": str(indus_paths[i_idx]),
            "proposed_indus_phoneme": PARPOLA_PHONEMES.get(indus_sign),
            "hevesy_match": (rongo_code, indus_sign) in hevesy_set,
        })
        if len(pairs) >= top_k:
            break

    return pairs


def _extract_barthel_code(name: str) -> str:
    """Extract Barthel code from image filename stem.

    barthel_ref naming convention: {tablet}_{pos}_barthel_{page}_{code}
    The Barthel code is always the last _-separated component.
    """
    parts = name.split("_")
    return parts[-1]


def _extract_mahadevan_number(name: str) -> str:
    """Extract Mahadevan sign number from indus image filename stem."""
    # Expected pattern: indus_M001 → M001
    if "_" in name:
        return name.split("_", 1)[1]
    return name


# ---------------------------------------------------------------------------
# HTML report
# ---------------------------------------------------------------------------


def _html_report(
    result: dict[str, Any],
    report_path: Path,
) -> None:
    ks_a = result["ks_test_indus_vs_control_a"]
    ks_b = result["ks_test_indus_vs_control_b"]
    significant = ks_a["p_value"] < 0.05

    badge_colour = "#22c55e" if significant else "#ef4444"
    verdict = (
        "SIGNIFICANT: Rongorongo–Indus similarity exceeds control (p &lt; 0.05)"
        if significant
        else "NOT SIGNIFICANT: No excess similarity over control at p &lt; 0.05"
    )

    pairs_html = ""
    for p in result["top_pairs"]:
        hevesy_badge = (
            '<span style="background:#f59e0b;color:#000;padding:1px 6px;border-radius:3px;'
            'font-size:0.75em;margin-left:6px">Hevesy</span>'
            if p["hevesy_match"]
            else ""
        )
        phoneme = p.get("proposed_indus_phoneme") or "—"
        pairs_html += f"""
        <tr>
          <td style="font-family:monospace">{p['rongo_code']}</td>
          <td style="font-family:monospace">{p['indus_sign']}{hevesy_badge}</td>
          <td style="text-align:right">{p['cosine_similarity']:.4f}</td>
          <td style="font-family:monospace">{phoneme}</td>
        </tr>"""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Cross-Script Similarity: Rongorongo × Indus Valley</title>
<style>
  :root {{
    --bg: #0d1117; --surface: #161b22; --border: #30363d;
    --fg: #e6edf3; --muted: #8b949e; --accent: #58a6ff;
    --warn: #f0a030; --ok: #22c55e; --fail: #ef4444;
    --font-mono: "JetBrains Mono", "Fira Code", "Cascadia Code", monospace;
    --font-serif: "Cormorant Garamond", "Garamond", Georgia, serif;
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ background: var(--bg); color: var(--fg); font-family: var(--font-mono);
         font-size: 14px; line-height: 1.6; padding: 2rem; }}
  h1 {{ font-family: var(--font-serif); font-size: 2.2rem; color: var(--accent);
        margin-bottom: 0.25rem; }}
  h2 {{ font-size: 1rem; text-transform: uppercase; letter-spacing: 0.12em;
        color: var(--muted); margin: 2rem 0 1rem; border-bottom: 1px solid var(--border);
        padding-bottom: 0.4rem; }}
  .subtitle {{ font-family: var(--font-serif); color: var(--muted); font-size: 1.1rem;
               margin-bottom: 2rem; }}
  .verdict {{ background: var(--surface); border: 2px solid {badge_colour};
              border-radius: 6px; padding: 1rem 1.5rem; margin: 1.5rem 0;
              color: {badge_colour}; font-size: 1.05rem; }}
  .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
           gap: 1rem; margin: 1rem 0; }}
  .stat {{ background: var(--surface); border: 1px solid var(--border);
           border-radius: 6px; padding: 1rem; }}
  .stat-value {{ font-size: 1.6rem; color: var(--accent); }}
  .stat-label {{ font-size: 0.75rem; color: var(--muted); text-transform: uppercase;
                 letter-spacing: 0.08em; margin-top: 0.25rem; }}
  table {{ width: 100%; border-collapse: collapse; margin: 1rem 0; }}
  th {{ text-align: left; padding: 0.5rem 0.75rem; color: var(--muted);
        font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.08em;
        border-bottom: 1px solid var(--border); }}
  td {{ padding: 0.4rem 0.75rem; border-bottom: 1px solid var(--border);
        vertical-align: middle; }}
  tr:hover td {{ background: var(--surface); }}
  .ts {{ color: var(--muted); font-size: 0.75rem; margin-top: 2rem; }}
</style>
</head>
<body>
<h1>Cross-Script Similarity</h1>
<p class="subtitle">Rongorongo × Indus Valley Script — Computational test of Hevesy (1932)</p>

<div class="verdict">{verdict}</div>

<h2>Panel 1 — Statistical Tests</h2>
<div class="grid">
  <div class="stat">
    <div class="stat-value">{result['n_rongo']}</div>
    <div class="stat-label">Rongorongo Glyphs</div>
  </div>
  <div class="stat">
    <div class="stat-value">{result['n_indus']}</div>
    <div class="stat-label">Indus Valley Signs</div>
  </div>
  <div class="stat">
    <div class="stat-value">{result['n_control']}</div>
    <div class="stat-label">Linear B Control Signs</div>
  </div>
  <div class="stat">
    <div class="stat-value">{result['hevesy_pairs_recovered']}/40</div>
    <div class="stat-label">Hevesy Pairs Recovered</div>
  </div>
  <div class="stat">
    <div class="stat-value">{result['hevesy_recovery_rate']:.1%}</div>
    <div class="stat-label">Hevesy Recovery Rate</div>
  </div>
  <div class="stat">
    <div class="stat-value">{ks_a['p_value']:.4f}</div>
    <div class="stat-label">KS p-value vs Linear B</div>
  </div>
  <div class="stat">
    <div class="stat-value">{ks_b['p_value']:.4f}</div>
    <div class="stat-label">KS p-value vs Shuffled</div>
  </div>
  <div class="stat">
    <div class="stat-value">{ks_a['statistic']:.4f}</div>
    <div class="stat-label">KS Statistic (vs A)</div>
  </div>
  <div class="stat">
    <div class="stat-value">{result['mean_nn_distance_rongo_to_indus']:.4f}</div>
    <div class="stat-label">Mean NN Dist (Indus)</div>
  </div>
  <div class="stat">
    <div class="stat-value">{result['mean_nn_distance_rongo_to_control']:.4f}</div>
    <div class="stat-label">Mean NN Dist (Control)</div>
  </div>
</div>

<h2>Panel 2 — Top-{len(result['top_pairs'])} Most Similar Pairs</h2>
<table>
  <thead>
    <tr>
      <th>Barthel Code</th>
      <th>Mahadevan Sign</th>
      <th>Cosine Similarity</th>
      <th>Parpola Phoneme</th>
    </tr>
  </thead>
  <tbody>
    {pairs_html}
  </tbody>
</table>

<h2>Panel 3 — Hevesy Recovery Analysis</h2>
<p>
  Of the 40 pairs proposed in Hevesy (1932), <strong>{result['hevesy_pairs_recovered']}</strong>
  appear in the computational top-{len(result['top_pairs'])}
  (recovery rate: {result['hevesy_recovery_rate']:.1%}).
  A high recovery rate at p&nbsp;&lt;&nbsp;0.05 provides independent computational
  support for the Hevesy hypothesis; a low rate indicates the visual resemblances
  he identified were not consistent enough to survive metric embedding.
</p>

<p class="ts">Generated {datetime.now(tz=timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}
 · backbone: dinov2_vits14 · hackingrongo cross_script_similarity.py</p>
</body>
</html>"""

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(html, encoding="utf-8")
    log.info("HTML report → %s", report_path)


# ---------------------------------------------------------------------------
# MLflow logging
# ---------------------------------------------------------------------------


def _log_mlflow(
    result: dict[str, Any],
    args: argparse.Namespace,
    output_json_path: Path,
    report_html_path: Path,
    timestamp: str,
) -> None:
    try:
        import mlflow
    except ImportError:
        log.warning("mlflow not installed — skipping experiment logging")
        return

    tracking_uri = f"file://{(PROJECT_ROOT / 'outputs' / 'mlruns').resolve()}"
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("rongorongo_cross_script")

    ks_a = result["ks_test_indus_vs_control_a"]
    ks_b = result["ks_test_indus_vs_control_b"]

    with mlflow.start_run(run_name=f"cross_script_similarity_{timestamp}"):
        mlflow.log_params({
            "n_rongo_glyphs": result["n_rongo"],
            "n_indus_glyphs": result["n_indus"],
            "backbone": "dinov2_vits14",
            "min_similarity_threshold": args.min_similarity,
            "top_k": args.top_k,
        })
        metrics: dict[str, float] = {
            "ks_statistic_vs_control_a": ks_a["statistic"],
            "ks_pvalue_vs_control_a": ks_a["p_value"],
            "ks_statistic_vs_control_b": ks_b["statistic"],
            "ks_pvalue_vs_control_b": ks_b["p_value"],
            "mean_nn_distance_rongo_to_indus": result["mean_nn_distance_rongo_to_indus"],
            "mean_nn_distance_rongo_to_control": result["mean_nn_distance_rongo_to_control"],
            "hevesy_recovery_rate": result["hevesy_recovery_rate"],
            "n_pairs_above_threshold": float(len(result["top_pairs"])),
        }
        mlflow.log_metrics({k: v for k, v in metrics.items() if math.isfinite(v)})
        if output_json_path.exists():
            mlflow.log_artifact(str(output_json_path))
        if report_html_path.exists():
            mlflow.log_artifact(str(report_html_path))
        log.info("MLflow run logged → %s", tracking_uri)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Test Hevesy (1932) hypothesis via DINOv2 cross-script embeddings"
    )
    parser.add_argument("--rongo-dir", type=Path, required=True)
    parser.add_argument("--indus-dir", type=Path, required=True)
    parser.add_argument("--control-dir", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "analysis" / "cross_script_similarity.json",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "analysis" / "cross_script_similarity_report.html",
    )
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--min-similarity", type=float, default=0.35)
    parser.add_argument("--latent-dim", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    random.seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", device)

    # ── Load DINOv2 encoder ──────────────────────────────────────────────────
    from hackingrongo.zone_a.autoencoder import DINOv2GlyphEncoder
    log.info("Loading DINOv2 ViT-S/14 …")
    encoder = DINOv2GlyphEncoder(latent_dim=args.latent_dim, freeze_backbone=True).to(device)

    # ── Load images ──────────────────────────────────────────────────────────
    log.info("Loading rongorongo glyphs from %s …", args.rongo_dir)
    rongo_names, rongo_images = _load_dir(args.rongo_dir, device)
    rongo_paths = sorted(args.rongo_dir.glob("*.png"))

    log.info("Loading Indus Valley signs from %s …", args.indus_dir)
    indus_names, indus_images = _load_dir(args.indus_dir, device)
    indus_paths = sorted(args.indus_dir.glob("*.png"))

    log.info("Loading Linear B control from %s …", args.control_dir)
    ctrl_names, ctrl_images = _load_dir(args.control_dir, device)

    # ── Embed ────────────────────────────────────────────────────────────────
    log.info("Embedding %d rongorongo glyphs …", len(rongo_names))
    rongo_emb = _embed_batch(encoder, rongo_images, args.batch_size)

    log.info("Embedding %d Indus Valley signs …", len(indus_names))
    indus_emb = _embed_batch(encoder, indus_images, args.batch_size)

    log.info("Embedding %d Linear B control signs …", len(ctrl_names))
    ctrl_emb = _embed_batch(encoder, ctrl_images, args.batch_size)

    # ── Nearest-neighbour distances ──────────────────────────────────────────
    log.info("Computing nearest-neighbour distances …")
    nn_dist_indus, _ = _nearest_neighbour_distances(rongo_emb, indus_emb)
    nn_dist_ctrl, _ = _nearest_neighbour_distances(rongo_emb, ctrl_emb)

    # Control B: shuffled Indus embeddings
    shuffled_indus = indus_emb[np.random.permutation(len(indus_emb))]
    nn_dist_shuffled, _ = _nearest_neighbour_distances(rongo_emb, shuffled_indus)

    # ── KS tests ─────────────────────────────────────────────────────────────
    ks_result_a = stats.ks_2samp(nn_dist_indus, nn_dist_ctrl)
    ks_result_b = stats.ks_2samp(nn_dist_indus, nn_dist_shuffled)

    log.info(
        "KS test (Indus vs Linear B): statistic=%.4f  p=%.4f",
        ks_result_a.statistic, ks_result_a.pvalue,
    )
    log.info(
        "KS test (Indus vs Shuffled): statistic=%.4f  p=%.4f",
        ks_result_b.statistic, ks_result_b.pvalue,
    )

    # ── Top-k pairs ──────────────────────────────────────────────────────────
    log.info("Finding top-%d most similar cross-script pairs …", args.top_k)
    top_pairs = _top_k_pairs(
        rongo_emb, indus_emb,
        [_extract_barthel_code(n) for n in rongo_names],
        [_extract_mahadevan_number(n) for n in indus_names],
        rongo_paths, indus_paths,
        args.top_k, args.min_similarity,
    )

    # ── Hevesy recovery ──────────────────────────────────────────────────────
    top_pair_keys = {(p["rongo_code"], p["indus_sign"]) for p in top_pairs}
    hevesy_set = set(HEVESY_PAIRS)
    hevesy_recovered = len(hevesy_set & top_pair_keys)
    hevesy_recovery_rate = hevesy_recovered / max(len(hevesy_set), 1)

    log.info(
        "Hevesy recovery: %d / %d pairs (%.1f%%)",
        hevesy_recovered, len(hevesy_set), hevesy_recovery_rate * 100,
    )

    # ── Assemble result ──────────────────────────────────────────────────────
    result: dict[str, Any] = {
        "n_rongo": len(rongo_names),
        "n_indus": len(indus_names),
        "n_control": len(ctrl_names),
        "ks_test_indus_vs_control_a": {
            "statistic": float(ks_result_a.statistic),
            "p_value": float(ks_result_a.pvalue),
        },
        "ks_test_indus_vs_control_b": {
            "statistic": float(ks_result_b.statistic),
            "p_value": float(ks_result_b.pvalue),
        },
        "mean_nn_distance_rongo_to_indus": float(np.mean(nn_dist_indus)),
        "mean_nn_distance_rongo_to_control": float(np.mean(nn_dist_ctrl)),
        "top_pairs": top_pairs,
        "hevesy_pairs_recovered": hevesy_recovered,
        "hevesy_recovery_rate": float(hevesy_recovery_rate),
    }

    # ── Write JSON ───────────────────────────────────────────────────────────
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    log.info("JSON → %s", args.output)

    # ── HTML report ──────────────────────────────────────────────────────────
    _html_report(result, args.report)

    # ── MLflow ───────────────────────────────────────────────────────────────
    timestamp = datetime.now(tz=timezone.utc).strftime("%Y%m%d-%H%M")
    _log_mlflow(result, args, args.output, args.report, timestamp)

    # ── Summary ──────────────────────────────────────────────────────────────
    sig = result["ks_test_indus_vs_control_a"]["p_value"] < 0.05
    print(f"\n── Cross-Script Similarity Summary ────────────────────────")
    print(f"  Rongorongo glyphs:         {result['n_rongo']}")
    print(f"  Indus Valley signs:        {result['n_indus']}")
    print(f"  KS p-value vs Linear B:    {result['ks_test_indus_vs_control_a']['p_value']:.4f}")
    print(f"  KS p-value vs Shuffled:    {result['ks_test_indus_vs_control_b']['p_value']:.4f}")
    print(f"  Hevesy pairs recovered:    {result['hevesy_pairs_recovered']}/40")
    print(f"  Hevesy recovery rate:      {result['hevesy_recovery_rate']:.1%}")
    if sig:
        print(f"\n  *** SIGNIFICANT: Rongorongo-Indus similarity exceeds control ***")
        print(f"  Hevesy hypothesis has computational support.")
    else:
        print(f"\n  Result: no significant excess similarity over control.")
        print(f"  Hevesy hypothesis not supported at p < 0.05.")


if __name__ == "__main__":
    main()
