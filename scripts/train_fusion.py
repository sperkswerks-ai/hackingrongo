"""train_fusion.py — Zone C Fusion Layer Pre-training Script.

Trains the :class:`~hackingrongo.zone_c.fusion.FusionLayer` using a
self-supervised objective: the fusion layer (Zone A embedding + Zone B
structural prior) learns to reproduce a low-rank projection of the Zone A
embedding, with the Zone B conditioning providing structural supervision.

Training objective
------------------
1. Compute a truncated-SVD projection of the Zone A embeddings to
   ``cfg.zone_c.fusion.output_dim`` dimensions.  This projection captures
   the maximum-variance structure in the embedding space.
2. For each glyph, build the Zone B prior vector from compound-candidate
   scores and default structural features.
3. Train the fusion layer to minimise MSE between its output and the
   truncated-SVD projection target.

Zone B conditioning allows the fusion layer to incorporate structural
knowledge (compound status, likely functional category) into the
embedding refinement, even before the full sign-classifier has been run.

Usage
-----
    python scripts/train_fusion.py \\
        --embeddings outputs/embeddings_cache.pt \\
        --compounds  outputs/analysis/compound_candidates.json \\
        --corpus-dir data/corpus \\
        --output     outputs/zone_c/fusion_checkpoint.pt

    # Smoke test (2 epochs, 8-sample batches)
    python scripts/train_fusion.py --smoke-test
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class FusionDataset(Dataset):
    """Dataset of (zone_a_emb, zone_b_prior, target) triples.

    Parameters
    ----------
    zone_a : torch.Tensor  Shape ``(N, zone_a_dim)``.
    zone_b : torch.Tensor  Shape ``(N, zone_b_dim)``.
    targets : torch.Tensor Shape ``(N, output_dim)``.
    """

    def __init__(
        self,
        zone_a: torch.Tensor,
        zone_b: torch.Tensor,
        targets: torch.Tensor,
    ) -> None:
        assert zone_a.shape[0] == zone_b.shape[0] == targets.shape[0]
        self.zone_a = zone_a
        self.zone_b = zone_b
        self.targets = targets

    def __len__(self) -> int:
        return self.zone_a.shape[0]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.zone_a[idx], self.zone_b[idx], self.targets[idx]


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def _load_embeddings(path: Path) -> tuple[torch.Tensor, list[str]]:
    """Load Zone A embeddings cache.

    Returns
    -------
    tuple[Tensor, list[str]]
        ``(embeddings, barthel_codes)`` where embeddings has shape ``(N, D)``.
    """
    data = torch.load(path, weights_only=True)
    embs: torch.Tensor = data["embeddings"].float()
    codes: list[str] = list(data["barthel_codes"])
    log.info("Embeddings: %d glyphs, dim=%d", len(codes), embs.shape[1])
    return embs, codes


def _load_compound_scores(path: Path) -> dict[str, float]:
    """Load compound-candidate scores from JSON.

    Returns
    -------
    dict[str, float]
        Maps Barthel code → compound probability score ∈ [0, 1].
    """
    if not path.exists():
        log.warning("Compound candidates not found at %s — using zeros.", path)
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    scores: dict[str, float] = {}
    candidates = data if isinstance(data, list) else data.get("candidates", [])
    for c in candidates:
        code = str(c.get("barthel_code", c.get("code", "")))
        prob = float(c.get("compound_probability", c.get("score", 0.0)))
        if code:
            scores[code] = prob
    log.info("Compound scores loaded: %d candidates.", len(scores))
    return scores


def _build_zone_b_priors(
    codes: list[str],
    compound_scores: dict[str, float],
    output_dim: int,
) -> torch.Tensor:
    """Build simple Zone B prior vectors.

    Uses :class:`~hackingrongo.zone_b.priors.ZoneBPriorBuilder` with
    default (neutral) classifications to produce a ``(N, output_dim)``
    prior matrix.  The compound score is incorporated where available.

    Parameters
    ----------
    codes : list[str]
    compound_scores : dict[str, float]
    output_dim : int  Desired prior dimension.

    Returns
    -------
    torch.Tensor  Shape ``(N, output_dim)``.
    """
    from hackingrongo.zone_b.priors import ZoneBPriorBuilder
    from hackingrongo.zone_b.sign_classifier import (
        SignClass,
        SignClassification,
        SignInventory,
    )

    # Build a minimal inventory with default UNKNOWN classifications.
    classifications = {}
    for code in set(codes):
        classifications[code] = SignClassification(
            code=code,
            sign_class=SignClass.UNKNOWN,
            confidence=0.0,
            frequency_percentile=0.5,
            omission_rate=0.0,
            positional_entropy=0.0,
        )
    inventory = SignInventory(classifications=classifications)

    builder = ZoneBPriorBuilder(output_dim=output_dim)
    raw = builder.build_feature_tensor(codes, inventory, compound_scores)
    with torch.no_grad():
        prior = builder(raw)
    log.info("Zone B priors built: shape %s.", list(prior.shape))
    return prior.detach()


def _build_targets(zone_a: torch.Tensor, output_dim: int) -> torch.Tensor:
    """Build self-supervised targets via truncated SVD of Zone A embeddings.

    Computes the top-``output_dim`` principal components of the Zone A
    embedding matrix.  These SVD coordinates are the targets the fusion
    layer is trained to reproduce; they capture the maximum-variance
    structure in the embedding space.

    Parameters
    ----------
    zone_a : torch.Tensor  Shape ``(N, D)``.
    output_dim : int

    Returns
    -------
    torch.Tensor  Shape ``(N, output_dim)``, float32.
    """
    from sklearn.decomposition import TruncatedSVD  # type: ignore

    n, d = zone_a.shape
    n_components = min(output_dim, d, n - 1)
    log.info(
        "Computing truncated SVD (%d → %d components) for fusion targets …",
        d, n_components,
    )
    arr = zone_a.numpy().astype(np.float32)
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    coords = svd.fit_transform(arr)  # (N, n_components)

    # Pad to output_dim if needed.
    if n_components < output_dim:
        pad = np.zeros((n, output_dim - n_components), dtype=np.float32)
        coords = np.concatenate([coords, pad], axis=1)

    targets = torch.from_numpy(coords[:, :output_dim].astype(np.float32))
    log.info("Targets: shape %s.", list(targets.shape))
    return targets


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def _train(
    fusion: "FusionLayer",
    zone_b_builder: "ZoneBPriorBuilder",
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: "torch.optim.lr_scheduler.LRScheduler | None",
    cfg: "DictConfig",
    device: torch.device,
    num_epochs: int,
    checkpoint_interval: int,
    output_path: Path,
) -> None:
    from hackingrongo.zone_c.fusion import save_fusion_checkpoint, train_fusion_epoch

    best_loss = float("inf")
    for epoch in range(num_epochs):
        loss = train_fusion_epoch(fusion, loader, optimizer, cfg, device, epoch)
        if scheduler is not None:
            scheduler.step()
        if loss < best_loss:
            best_loss = loss
        if (epoch + 1) % checkpoint_interval == 0 or (epoch + 1) == num_epochs:
            save_fusion_checkpoint(fusion, optimizer, epoch + 1, loss, output_path)

    log.info("Training complete. Best loss = %.6f.", best_loss)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train Zone C fusion layer (Zone A + Zone B priors).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--embeddings",  type=Path, default=None, metavar="PT")
    p.add_argument("--compounds",   type=Path, default=None, metavar="JSON")
    p.add_argument("--corpus-dir",  type=Path, default=None, metavar="DIR")
    p.add_argument("--output",      type=Path, default=None, metavar="PT")
    p.add_argument(
        "--smoke-test", action="store_true",
        help="Run 2 epochs with batch_size=8 for a fast wiring check.",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    # ── Resolve paths from config if not provided ─────────────────────────────
    try:
        from omegaconf import OmegaConf
        cfg = OmegaConf.load(PROJECT_ROOT / "conf" / "config.yaml")
    except Exception as exc:
        log.error("Failed to load config.yaml: %s", exc)
        sys.exit(1)

    embeddings_path = args.embeddings or (
        PROJECT_ROOT / cfg.paths.get("embeddings_cache", "outputs/embeddings_cache.pt")
    )
    compound_path = args.compounds or (
        PROJECT_ROOT / "outputs" / "analysis" / "compound_candidates.json"
    )
    output_path = args.output or (
        PROJECT_ROOT / "outputs" / "zone_c" / "fusion_checkpoint.pt"
    )

    if not embeddings_path.exists():
        log.error("Embeddings cache not found: %s", embeddings_path)
        sys.exit(1)

    # ── Smoke-test overrides ──────────────────────────────────────────────────
    if args.smoke_test:
        log.info("Smoke-test mode: 2 epochs, batch_size=8.")
        from omegaconf import OmegaConf  # noqa: F401
        cfg = OmegaConf.merge(
            cfg,
            OmegaConf.create({"zone_c": {"fusion": {"num_epochs": 2, "batch_size": 8}}}),
        )

    num_epochs: int = int(cfg.zone_c.fusion.num_epochs)
    batch_size: int = int(cfg.zone_c.fusion.batch_size)
    ckpt_interval: int = int(cfg.zone_c.fusion.get("checkpoint_interval_epochs", 5))
    zone_b_dim: int = int(cfg.zone_b.prior_output_dim)
    output_dim: int = int(cfg.zone_c.fusion.output_dim)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", device)

    # ── Load data ─────────────────────────────────────────────────────────────
    zone_a, codes = _load_embeddings(embeddings_path)
    compound_scores = _load_compound_scores(compound_path)

    # ── Build Zone B priors ───────────────────────────────────────────────────
    zone_b = _build_zone_b_priors(codes, compound_scores, zone_b_dim)

    # ── Build self-supervised targets ─────────────────────────────────────────
    targets = _build_targets(zone_a, output_dim)

    # ── Align zone_a_dim in config with actual embedding dim ─────────────────
    actual_zone_a_dim = zone_a.shape[1]
    if actual_zone_a_dim != int(cfg.zone_c.fusion.zone_a_dim):
        log.warning(
            "Config zone_a_dim=%d but actual embedding dim=%d — overriding.",
            int(cfg.zone_c.fusion.zone_a_dim), actual_zone_a_dim,
        )
        from omegaconf import OmegaConf
        cfg = OmegaConf.merge(
            cfg,
            OmegaConf.create({"zone_c": {"fusion": {"zone_a_dim": actual_zone_a_dim}}}),
        )

    # ── Build dataset and loader ──────────────────────────────────────────────
    dataset = FusionDataset(zone_a, zone_b, targets)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=0
    )
    log.info(
        "Dataset: %d samples  batch_size=%d  steps/epoch=%d",
        len(dataset), batch_size, len(loader),
    )

    # ── Build model ───────────────────────────────────────────────────────────
    from hackingrongo.zone_b.priors import ZoneBPriorBuilder
    from hackingrongo.zone_c.fusion import (
        FusionLayer,
        build_fusion_optimizer,
        build_fusion_scheduler,
    )

    fusion = FusionLayer(cfg).to(device)
    optimizer = build_fusion_optimizer(fusion, cfg)
    scheduler = build_fusion_scheduler(optimizer, cfg)

    # ── Train ─────────────────────────────────────────────────────────────────
    log.info("Training fusion layer for %d epochs …", num_epochs)
    _train(
        fusion=fusion,
        zone_b_builder=ZoneBPriorBuilder(output_dim=zone_b_dim),
        loader=loader,
        optimizer=optimizer,
        scheduler=scheduler,
        cfg=cfg,
        device=device,
        num_epochs=num_epochs,
        checkpoint_interval=ckpt_interval,
        output_path=output_path,
    )
    log.info("Done. Checkpoint: %s", output_path)


if __name__ == "__main__":
    main()
