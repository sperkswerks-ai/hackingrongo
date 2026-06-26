# ============================================================================
# DEPRECATED — SYLLABIC SUBSTITUTION-CIPHER TRACK (set down 2026-06, in place).
# Part of the sign→phoneme substitution-cipher hypothesis, which was tested and
# set down as a recorded NEGATIVE RESULT — preserved as an archive, NOT fixed,
# tuned, or deleted. Do not extend this module. The structural/logographic track
# supersedes it. Full rationale + on-disk numbers: DEPRECATED_SYLLABIC.md (root).
# ============================================================================
"""
hackingrongo.zone_c.fusion
============================

NDDR-style fusion layer combining Zone A glyph embeddings with Zone B
structural prior vectors into a joint representation for decipherment search.

Zone A produces a fused embedding of dimension ``cfg.zone_c.fusion.zone_a_dim``
(384 by default: concatenation of the 128-dim autoencoder, Siamese, and
sequence-model embeddings).  Zone B produces a structural prior vector of
dimension ``cfg.zone_c.fusion.zone_b_dim`` (64 by default).

The fusion layer learns to blend these into a
``cfg.zone_c.fusion.output_dim``-dimensional (256 by default) joint
representation using two linear projections with optional BatchNorm,
activation, and dropout — the 1D analogue of the NDDR-CNN 1×1
channel-wise convolution.

Public API
----------
``FusionLayer``
    ``nn.Module``.  Call ``forward(zone_a_emb, zone_b_prior)`` → fused tensor.
``build_fusion_optimizer`` / ``build_fusion_scheduler``
    Factory helpers reading directly from ``cfg``.
``train_fusion_epoch``
    Single training epoch for the fusion layer.
``save_fusion_checkpoint`` / ``load_fusion_checkpoint``
    Checkpoint helpers.

Status: PENDING PIPELINE INTEGRATION
--------------------------------------
This module is complete and tested in isolation, but is **not yet called
by** :mod:`~hackingrongo.pipeline`.  Two pieces must be implemented before
the fusion layer is live:

1. **Zone B prior vector construction** — :class:`FusionLayer` expects a
   ``zone_b_prior`` tensor of shape ``(B, zone_b_dim)`` (default 64-dim).
   This vector should be assembled from the outputs of
   :mod:`~hackingrongo.zone_b.sign_classifier` (per-sign class probabilities,
   omission rate, frequency percentile) and
   :mod:`~hackingrongo.zone_b.compound_detector` (compound probability score).
   A ``build_zone_b_prior(sign_classifications, compound_scores, cfg)``
   helper function needs to be written (likely in a new ``zone_b/priors.py``
   module).

2. **Pipeline step** — A ``step_train_fusion`` function in
   :mod:`~hackingrongo.pipeline` (between steps 4 and 5) that:
   (a) loads Zone A embeddings from ``outputs/embeddings_cache.pt``,
   (b) constructs the Zone B prior vectors per glyph,
   (c) runs :func:`train_fusion_epoch` to convergence,
   (d) saves the checkpoint for Zone C (MCMC / beam search) to consume.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from omegaconf import DictConfig
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)

_ACTIVATION_REGISTRY: dict[str, type[nn.Module]] = {
    "relu": nn.ReLU,
    "leaky_relu": nn.LeakyReLU,
    "gelu": nn.GELU,
}


def _build_activation(name: str) -> nn.Module:
    key = name.lower()
    if key not in _ACTIVATION_REGISTRY:
        raise ValueError(
            f"Unsupported activation '{name}'. "
            f"Choose from: {sorted(_ACTIVATION_REGISTRY)}."
        )
    return _ACTIVATION_REGISTRY[key]()


# ---------------------------------------------------------------------------
# FusionLayer
# ---------------------------------------------------------------------------


class FusionLayer(nn.Module):
    """Two-layer NDDR-style fusion of Zone A and Zone B representations.

    Architecture::

        input  : concat(zone_a_emb, zone_b_prior)  — shape (B, zone_a_dim + zone_b_dim)
        layer1 : Linear(in → hidden) → [BatchNorm1d] → Activation → [Dropout]
        layer2 : Linear(hidden → output_dim) → [BatchNorm1d] → Activation → [Dropout]
        output : shape (B, output_dim)

    The hidden dimension is the midpoint between the input and output
    dimensions: ``(zone_a_dim + zone_b_dim + output_dim) // 2``.

    Parameters
    ----------
    cfg : DictConfig
        Root Hydra config.  Reads ``cfg.zone_c.fusion``.
    """

    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()
        fc = cfg.zone_c.fusion
        in_dim: int = int(fc.zone_a_dim) + int(fc.zone_b_dim)
        hidden_dim: int = (in_dim + int(fc.output_dim)) // 2
        out_dim: int = int(fc.output_dim)
        use_bn: bool = bool(fc.use_batch_norm)
        dropout: float = float(fc.dropout_rate)
        act_name: str = str(fc.activation)

        self.layer1 = self._make_block(in_dim, hidden_dim, use_bn, dropout, act_name)
        self.layer2 = self._make_block(hidden_dim, out_dim, use_bn, dropout, act_name)

        logger.debug(
            "FusionLayer: (%d+%d) → %d → %d  bn=%s  dropout=%.2f",
            int(fc.zone_a_dim), int(fc.zone_b_dim), hidden_dim, out_dim,
            use_bn, dropout,
        )

    @staticmethod
    def _make_block(
        in_dim: int,
        out_dim: int,
        use_bn: bool,
        dropout: float,
        act_name: str,
    ) -> nn.Sequential:
        layers: list[nn.Module] = [
            nn.Linear(in_dim, out_dim, bias=not use_bn),
        ]
        if use_bn:
            layers.append(nn.BatchNorm1d(out_dim))
        layers.append(_build_activation(act_name))
        if dropout > 0.0:
            layers.append(nn.Dropout(p=dropout))
        return nn.Sequential(*layers)

    def forward(
        self,
        zone_a_emb: torch.Tensor,
        zone_b_prior: torch.Tensor,
    ) -> torch.Tensor:
        """Fuse Zone A embedding and Zone B prior into a joint representation.

        Parameters
        ----------
        zone_a_emb : torch.Tensor
            Zone A fused embedding, shape ``(B, zone_a_dim)``.
        zone_b_prior : torch.Tensor
            Zone B structural prior, shape ``(B, zone_b_dim)``.

        Returns
        -------
        torch.Tensor
            Fused representation, shape ``(B, output_dim)``.
        """
        x = torch.cat([zone_a_emb, zone_b_prior], dim=-1)
        x = self.layer1(x)
        return self.layer2(x)


# ---------------------------------------------------------------------------
# Optimizer and scheduler factories
# ---------------------------------------------------------------------------


def build_fusion_optimizer(
    fusion: FusionLayer,
    cfg: DictConfig,
) -> torch.optim.Optimizer:
    """Build the optimizer for the fusion layer from config.

    Parameters
    ----------
    fusion : FusionLayer
    cfg : DictConfig
        Reads ``cfg.zone_c.fusion.optimizer``, ``lr``, ``weight_decay``.

    Returns
    -------
    torch.optim.Optimizer
    """
    fc = cfg.zone_c.fusion
    name = str(fc.optimizer).lower()
    lr = float(fc.lr)
    wd = float(fc.weight_decay)
    if name == "adam":
        return torch.optim.Adam(fusion.parameters(), lr=lr, weight_decay=wd)
    if name == "adamw":
        return torch.optim.AdamW(fusion.parameters(), lr=lr, weight_decay=wd)
    if name == "sgd":
        return torch.optim.SGD(fusion.parameters(), lr=lr, weight_decay=wd, momentum=0.9)
    raise ValueError(f"Unknown optimizer '{fc.optimizer}'. Choose adam | adamw | sgd.")


def build_fusion_scheduler(
    optimizer: torch.optim.Optimizer,
    cfg: DictConfig,
) -> "torch.optim.lr_scheduler.LRScheduler | None":
    """Build a learning-rate scheduler for fusion layer training.

    Parameters
    ----------
    optimizer : torch.optim.Optimizer
    cfg : DictConfig
        Reads ``cfg.zone_c.fusion.scheduler`` and ``scheduler_T_max``.

    Returns
    -------
    LRScheduler or None
    """
    fc = cfg.zone_c.fusion
    name = str(fc.scheduler).lower()
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=int(fc.scheduler_T_max)
        )
    if name == "none":
        return None
    raise ValueError(f"Unknown scheduler '{fc.scheduler}'. Choose cosine | none.")


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


def train_fusion_epoch(
    fusion: FusionLayer,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    cfg: DictConfig,
    device: torch.device,
    epoch: int = 0,
) -> float:
    """Run one training epoch for the fusion layer.

    The fusion layer is supervised via (zone_a_emb, zone_b_prior, target)
    triples.  In the full pipeline the supervision signal comes from the
    MCMC / beam-search posterior loss; this function handles the simpler
    regression case where explicit target vectors are available.

    Parameters
    ----------
    fusion : FusionLayer
    loader : DataLoader
        Yields ``(zone_a_emb, zone_b_prior, target)`` batches where all
        three are ``torch.Tensor``.
    optimizer : torch.optim.Optimizer
    cfg : DictConfig
    device : torch.device
    epoch : int
        Current epoch index (0-based), for logging only.

    Returns
    -------
    float
        Mean MSE loss over the epoch.
    """
    fusion.train()
    fc = cfg.zone_c.fusion
    grad_clip = float(fc.grad_clip_norm)
    loss_fn = nn.MSELoss()
    total_loss = 0.0
    n_batches = 0

    for batch in loader:
        zone_a_emb, zone_b_prior, target = [b.to(device) for b in batch]
        optimizer.zero_grad()
        fused = fusion(zone_a_emb, zone_b_prior)
        loss = loss_fn(fused, target)
        loss.backward()
        if grad_clip > 0.0:
            nn.utils.clip_grad_norm_(fusion.parameters(), grad_clip)
        optimizer.step()
        total_loss += loss.item()
        n_batches += 1

    mean_loss = total_loss / max(n_batches, 1)
    logger.info("Epoch %d  fusion MSE loss = %.6f", epoch + 1, mean_loss)
    return mean_loss


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------


def save_fusion_checkpoint(
    fusion: FusionLayer,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss: float,
    path: Path,
) -> None:
    """Save fusion layer weights and optimizer state.

    Parameters
    ----------
    fusion : FusionLayer
    optimizer : torch.optim.Optimizer
    epoch : int
        Current epoch (1-based).
    loss : float
        Loss value for this checkpoint.
    path : Path
        Destination ``.pt`` file.  Parent directories are created.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "loss": loss,
            "model_state_dict": fusion.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        path,
    )
    logger.info("Fusion checkpoint saved: %s  (epoch %d  loss=%.6f)", path, epoch, loss)


def load_fusion_checkpoint(
    fusion: FusionLayer,
    path: Path,
    optimizer: torch.optim.Optimizer | None = None,
    device: torch.device | None = None,
) -> int:
    """Load fusion layer weights from a checkpoint.

    Parameters
    ----------
    fusion : FusionLayer
    path : Path
        Checkpoint ``.pt`` file written by :func:`save_fusion_checkpoint`.
    optimizer : torch.optim.Optimizer, optional
        If supplied, optimizer state is also restored.
    device : torch.device, optional
        Target device for ``torch.load``.

    Returns
    -------
    int
        Saved epoch number.
    """
    map_location = device if device is not None else "cpu"
    ckpt = torch.load(path, map_location=map_location, weights_only=True)
    fusion.load_state_dict(ckpt["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    epoch: int = int(ckpt.get("epoch", 0))
    logger.info("Fusion checkpoint loaded: %s  (epoch %d)", path, epoch)
    return epoch
