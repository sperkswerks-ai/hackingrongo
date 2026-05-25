"""hackingrongo.zone_b.priors
============================

Build the Zone B structural prior vector for each rongorongo sign.

The prior vector bundles per-sign evidence from the sign classifier
(functional category, corpus statistics) into a fixed-dimensional
embedding suitable for the Zone C fusion layer.

Architecture
------------
Raw feature extraction (9 scalar features per sign) → learned linear
projection to ``zone_b.prior_output_dim`` (default 64).

The projection weights live inside :class:`ZoneBPriorBuilder`, which is
an ``nn.Module`` trained jointly with :class:`~hackingrongo.zone_c.fusion.FusionLayer`
in the ``step4k_train_fusion`` pipeline step.

Public API
----------
:class:`ZoneBPriorBuilder`
    Stateful builder: raw feature extraction + learned linear projection.
    Call :meth:`~ZoneBPriorBuilder.build_feature_tensor` for raw features
    and :meth:`~ZoneBPriorBuilder.forward` to project to ``output_dim``.

:func:`build_zone_b_prior`
    Convenience function: builds priors for a list of sign codes in one call.

Feature Layout (``RAW_FEATURE_DIM = 9``)
-----------------------------------------
Index 0: ``is_phonetic``         — 1 if sign class is PHONETIC
Index 1: ``is_taxogram``         — 1 if sign class is TAXOGRAM
Index 2: ``is_logogram``         — 1 if sign class is LOGOGRAM
Index 3: ``is_unknown``          — 1 if sign class is UNKNOWN
Index 4: ``confidence``          — classifier confidence ∈ [0, 1]
Index 5: ``frequency_percentile``— corpus frequency rank ∈ [0, 1]
Index 6: ``omission_rate``       — parallel-passage omission rate ∈ [0, 1]
Index 7: ``positional_entropy``  — normalised positional entropy ∈ [0, 1]
Index 8: ``compound_score``      — compound detector probability ∈ [0, 1]
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn as nn
from omegaconf import DictConfig

from hackingrongo.zone_b.sign_classifier import (
    SignClass,
    SignClassification,
    SignInventory,
)

if TYPE_CHECKING:
    pass  # no circular imports

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Feature index constants
# ---------------------------------------------------------------------------

FEAT_PHONETIC: int = 0
FEAT_TAXOGRAM: int = 1
FEAT_LOGOGRAM: int = 2
FEAT_UNKNOWN: int = 3
FEAT_CONFIDENCE: int = 4
FEAT_FREQ_PCT: int = 5
FEAT_OMISSION: int = 6
FEAT_POSITIONAL_ENTROPY: int = 7
FEAT_COMPOUND: int = 8

RAW_FEATURE_DIM: int = 9


# ---------------------------------------------------------------------------
# Raw feature extraction
# ---------------------------------------------------------------------------

def _extract_raw_features(
    classification: SignClassification,
    compound_score: float = 0.0,
    max_entropy: float = 10.0,
) -> np.ndarray:
    """Build a ``RAW_FEATURE_DIM``-dimensional feature vector for one sign.

    Parameters
    ----------
    classification : SignClassification
        Output of :func:`~hackingrongo.zone_b.sign_classifier.classify_inventory`
        for this sign.
    compound_score : float
        Compound probability from the compound detector (0.0 if not detected).
    max_entropy : float
        Upper bound for normalising positional entropy to ``[0, 1]``.

    Returns
    -------
    numpy.ndarray
        Shape ``(RAW_FEATURE_DIM,)``, dtype ``float32``.
    """
    feat = np.zeros(RAW_FEATURE_DIM, dtype=np.float32)
    sc = classification.sign_class

    feat[FEAT_PHONETIC] = float(sc is SignClass.PHONETIC)
    feat[FEAT_TAXOGRAM] = float(sc is SignClass.TAXOGRAM)
    feat[FEAT_LOGOGRAM] = float(sc is SignClass.LOGOGRAM)
    feat[FEAT_UNKNOWN]  = float(sc is SignClass.UNKNOWN)

    feat[FEAT_CONFIDENCE] = float(np.clip(classification.confidence, 0.0, 1.0))
    feat[FEAT_FREQ_PCT]   = float(np.clip(classification.frequency_percentile, 0.0, 1.0))
    feat[FEAT_OMISSION]   = float(np.clip(classification.omission_rate, 0.0, 1.0))
    feat[FEAT_POSITIONAL_ENTROPY] = float(
        np.clip(
            classification.positional_entropy / max(float(max_entropy), 1e-8),
            0.0,
            1.0,
        )
    )
    feat[FEAT_COMPOUND] = float(np.clip(compound_score, 0.0, 1.0))
    return feat


def _make_dummy_classification(code: str) -> SignClassification:
    """Return a neutral :class:`SignClassification` for unknown sign codes."""
    return SignClassification(
        code=code,
        sign_class=SignClass.UNKNOWN,
        confidence=0.0,
        frequency_percentile=0.5,
        omission_rate=0.0,
        positional_entropy=0.0,
    )


# ---------------------------------------------------------------------------
# ZoneBPriorBuilder
# ---------------------------------------------------------------------------

class ZoneBPriorBuilder(nn.Module):
    """Feature extractor + learned linear projection for Zone B priors.

    This module is saved and loaded together with the Zone C
    :class:`~hackingrongo.zone_c.fusion.FusionLayer` checkpoint so that
    the projection weights are always consistent with the downstream fusion.

    Parameters
    ----------
    output_dim : int
        Target embedding dimension.  Must equal ``cfg.zone_b.prior_output_dim``
        (default 64).
    max_entropy : float
        Upper bound used to normalise positional entropy to ``[0, 1]``.
    seed : int
        RNG seed for Xavier weight initialisation.
    """

    RAW_DIM: int = RAW_FEATURE_DIM

    def __init__(
        self,
        output_dim: int = 64,
        max_entropy: float = 10.0,
        seed: int = 0,
    ) -> None:
        super().__init__()
        self._output_dim = output_dim
        self._max_entropy = max_entropy

        gen = torch.Generator()
        gen.manual_seed(seed)
        # No bias, no activation — downstream FusionLayer provides non-linearity.
        self.proj = nn.Linear(self.RAW_DIM, output_dim, bias=False)
        nn.init.xavier_uniform_(self.proj.weight, generator=gen)

    @property
    def output_dim(self) -> int:
        """Projection output dimension."""
        return self._output_dim

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Project raw feature vectors to ``output_dim``.

        Parameters
        ----------
        features : torch.Tensor
            Shape ``(B, RAW_DIM)``, values in ``[0, 1]``.

        Returns
        -------
        torch.Tensor
            Shape ``(B, output_dim)``.
        """
        return self.proj(features)

    def build_feature_tensor(
        self,
        sign_codes: list[str],
        inventory: SignInventory,
        compound_scores: dict[str, float] | None = None,
    ) -> torch.Tensor:
        """Build a raw feature tensor for a list of sign codes.

        Parameters
        ----------
        sign_codes : list[str]
            Ordered list of Barthel codes to encode.
        inventory : SignInventory
            Output of :func:`~hackingrongo.zone_b.sign_classifier.classify_inventory`.
        compound_scores : dict[str, float], optional
            Maps sign code → compound probability (default: all zeros).

        Returns
        -------
        torch.Tensor
            Shape ``(len(sign_codes), RAW_DIM)``, dtype ``float32``.
        """
        if compound_scores is None:
            compound_scores = {}

        rows: list[np.ndarray] = []
        for code in sign_codes:
            sc_rec = inventory.classifications.get(code)
            if sc_rec is None:
                sc_rec = _make_dummy_classification(code)
            rows.append(
                _extract_raw_features(
                    sc_rec,
                    compound_scores.get(code, 0.0),
                    self._max_entropy,
                )
            )
        return torch.from_numpy(np.stack(rows, axis=0))


# ---------------------------------------------------------------------------
# Convenience wrapper
# ---------------------------------------------------------------------------

def build_zone_b_prior(
    sign_codes: list[str],
    inventory: SignInventory,
    cfg: DictConfig,
    compound_scores: dict[str, float] | None = None,
    builder: ZoneBPriorBuilder | None = None,
    device: torch.device | None = None,
) -> tuple[torch.Tensor, ZoneBPriorBuilder]:
    """Build projected Zone B prior vectors for a list of sign codes.

    Parameters
    ----------
    sign_codes : list[str]
        Ordered Barthel codes to encode.
    inventory : SignInventory
        Output of :func:`~hackingrongo.zone_b.sign_classifier.classify_inventory`.
    cfg : DictConfig
        Root Hydra config.  Reads ``cfg.zone_b.prior_output_dim``.
    compound_scores : dict[str, float], optional
        Per-sign compound probability.
    builder : ZoneBPriorBuilder, optional
        Pre-instantiated builder to reuse (e.g. when resuming from a
        fusion checkpoint).  If ``None``, a fresh one is created from
        config.
    device : torch.device, optional
        Target device for the returned tensors.

    Returns
    -------
    tuple[torch.Tensor, ZoneBPriorBuilder]
        ``prior_tensor``: shape ``(N, output_dim)``.
        ``builder``: the builder used (pass back in on subsequent calls
        to avoid re-constructing the projection weights).
    """
    output_dim: int = int(cfg.zone_b.prior_output_dim)
    if builder is None:
        builder = ZoneBPriorBuilder(output_dim=output_dim)

    raw = builder.build_feature_tensor(sign_codes, inventory, compound_scores)
    if device is not None:
        raw = raw.to(device)
        builder = builder.to(device)

    builder.eval()
    with torch.no_grad():
        prior = builder(raw)

    logger.debug(
        "build_zone_b_prior: %d signs → prior shape %s.",
        len(sign_codes),
        list(prior.shape),
    )
    return prior, builder
