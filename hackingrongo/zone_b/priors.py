"""hackingrongo.zone_b.priors
============================

Build the Zone B structural prior vector for each rongorongo sign.

The prior vector bundles per-sign evidence from the sign classifier
(functional category, corpus statistics) into a fixed-dimensional
embedding suitable for the Zone C fusion layer.

Architecture
------------
Raw feature extraction (13 scalar features per sign) → learned linear
projection to ``zone_b.prior_output_dim`` (default 64).

The projection weights live inside :class:`ZoneBPriorBuilder`, which is
an ``nn.Module`` trained jointly with :class:`~hackingrongo.zone_c.fusion.FusionLayer`
in the ``step4k_train_fusion`` pipeline step.

Public API
----------
:class:`CorpusSignStats`
    Dataclass of corpus-derived per-sign statistics.  Computed once by
    :func:`compute_corpus_sign_stats` and passed to
    :meth:`ZoneBPriorBuilder.build_feature_tensor`.

:func:`compute_corpus_sign_stats`
    Scan the corpus and return a :class:`CorpusSignStats` for all signs.

:class:`ZoneBPriorBuilder`
    Stateful builder: raw feature extraction + learned linear projection.
    Call :meth:`~ZoneBPriorBuilder.build_feature_tensor` for raw features
    and :meth:`~ZoneBPriorBuilder.forward` to project to ``output_dim``.

:func:`build_zone_b_prior`
    Convenience function: builds priors for a list of sign codes in one call.

Feature Layout (``RAW_FEATURE_DIM = 13``)
------------------------------------------
Index 0: ``is_phonetic``              — 1 if sign class is PHONETIC
Index 1: ``is_taxogram``              — 1 if sign class is TAXOGRAM
Index 2: ``is_logogram``              — 1 if sign class is LOGOGRAM
Index 3: ``is_unknown``               — 1 if sign class is UNKNOWN
Index 4: ``confidence``               — classifier confidence ∈ [0, 1]
Index 5: ``frequency_percentile``     — corpus frequency rank ∈ [0, 1]
Index 6: ``omission_rate``            — parallel-passage omission rate ∈ [0, 1]
Index 7: ``positional_entropy``       — normalised positional entropy ∈ [0, 1]
Index 8: ``compound_score``           — compound detector probability ∈ [0, 1]
Index 9: ``ic_contribution``          — normalised IC weight: (p_i²/ΣpJ²)/max ∈ [0,1]
Index 10: ``bigram_mi_as_predictor``  — how much this sign predicts its successor,
                                        normalised by H(unigram) ∈ [0, 1]
Index 11: ``cross_tablet_consistency``— 1 − min(1, CV) where CV = std/mean of
                                        per-tablet frequencies ∈ [0, 1]
Index 12: ``is_compound_component``   — 1 if sign appears in any compound's
                                        ``horley_components`` field

Design rationale for new features (indices 9–12)
-------------------------------------------------
The old ``frequency_percentile`` (index 5) is a *rank* — it tells Zone C
that sign 001 is at the 99th percentile and sign 050 at the 70th.  It does
not convey that sign 001 accounts for 20.6 % of corpus IC while sign 050
accounts for 0.2 %.  That 100× difference in IC weight is absent from the
prior, leaving the fusion layer with no information about which sign
assignments matter most for LM scoring.

``ic_contribution`` (index 9) fills this gap: it is the sign's actual
share of the corpus Index of Coincidence, normalised to [0, 1] by dividing
by the maximum such share across the inventory.

``bigram_mi_as_predictor`` (index 10) measures how much knowing this sign
reduces uncertainty about the *next* sign — high values flag signs that
anchor sequential structure (potential taxograms or strongly collocated
signs).  ``cross_tablet_consistency`` (index 11) penalises signs whose
frequency varies wildly across tablets (noisy signals).
``is_compound_component`` (index 12) flags signs that the compound detector
found primarily as components of compound glyphs — these carry semantic
rather than phonetic information and should be weighted differently.
"""

from __future__ import annotations

import json
import logging
import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
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
FEAT_IC_CONTRIBUTION: int = 9
FEAT_BIGRAM_MI: int = 10
FEAT_CROSS_TABLET_CONSISTENCY: int = 11
FEAT_COMPOUND_COMPONENT: int = 12

RAW_FEATURE_DIM: int = 13


# ---------------------------------------------------------------------------
# Corpus-derived sign statistics
# ---------------------------------------------------------------------------


@dataclass
class CorpusSignStats:
    """Per-sign corpus statistics for features 9–12 of the Zone B prior.

    All dicts map Barthel code → scalar in [0, 1].  Missing keys default
    to 0.0 (no information available).

    Attributes
    ----------
    ic_contributions : dict[str, float]
        Normalised IC weight: ``(p_i² / Σpᵢ²) / max_ic_contribution``.
        1.0 for the most IC-dominant sign; 0.0 for absent signs.
    bigram_mi_scores : dict[str, float]
        Normalised per-sign bigram MI-as-predictor:
        ``I_s / H_unigram`` where
        ``I_s = Σ_t P(s,t) log₂ [P(t|s) / P(t)]``.
        Measures how much knowing sign *s* reduces uncertainty about
        the next sign.
    cross_tablet_consistency : dict[str, float]
        ``1 − min(1, CV)`` where CV = std/mean of per-tablet frequencies.
        1.0 = perfectly uniform across tablets; 0.0 = wildly variable.
    compound_component_codes : set[str]
        Barthel codes that appear as components in at least one compound
        glyph's ``horley_components`` field.
    """

    ic_contributions: dict[str, float] = field(default_factory=dict)
    bigram_mi_scores: dict[str, float] = field(default_factory=dict)
    cross_tablet_consistency: dict[str, float] = field(default_factory=dict)
    compound_component_codes: set[str] = field(default_factory=set)

    def ic_contribution(self, code: str) -> float:
        return self.ic_contributions.get(code, 0.0)

    def bigram_mi(self, code: str) -> float:
        return self.bigram_mi_scores.get(code, 0.0)

    def consistency(self, code: str) -> float:
        return self.cross_tablet_consistency.get(code, 0.0)

    def is_compound_component(self, code: str) -> float:
        return 1.0 if code in self.compound_component_codes else 0.0


def compute_corpus_sign_stats(corpus_dir: Path) -> CorpusSignStats:
    """Scan the corpus and return per-sign statistics for prior features 9–12.

    Parameters
    ----------
    corpus_dir : Path
        ``data/corpus/`` directory containing per-tablet JSON files.

    Returns
    -------
    CorpusSignStats
        All dicts are populated for every sign that appears in the corpus.

    Notes
    -----
    **IC contribution** (index 9)
        IC = Σ pᵢ².  Each sign's share is pᵢ² / Σpᵢ².  We normalise
        by the maximum share so the most IC-dominant sign scores 1.0.

    **Bigram MI as predictor** (index 10)
        I_s = Σ_t P(s,t) log₂ [P(t|s) / P(t)]
        This is sign *s*'s row-wise contribution to the corpus bigram
        mutual information — it measures how much knowing that *s* just
        appeared reduces uncertainty about the next sign.  Values are
        normalised by the corpus unigram entropy H(S_n).

    **Cross-tablet consistency** (index 11)
        CV = σ(freq across tablets) / μ(freq across tablets).
        Consistency = 1 − min(1, CV).  Signs that appear on only one
        tablet receive consistency = 0.0.

    **Compound component** (index 12)
        Binary: 1.0 if the sign appears in any ``horley_components``
        list in the corpus, 0.0 otherwise.
    """
    # ── Pass 1: token counts ────────────────────────────────────────────────
    total_counts: Counter = Counter()
    tablet_counts: dict[str, Counter] = {}
    bigrams: Counter = Counter()
    compound_components: set[str] = set()

    for jf in sorted(corpus_dir.glob("[A-Z].json")):
        if "ferrara" in jf.stem:
            continue
        try:
            data = json.loads(jf.read_text(encoding="utf-8"))
        except Exception:
            continue
        tablet_id = jf.stem
        tab_counter: Counter = Counter()
        glyphs = data.get("glyphs", [])
        prev_code: str | None = None

        for g in glyphs:
            code = str(g.get("barthel_code") or "").strip()
            if not code or code == "?" or "!" in code:
                prev_code = None
                continue

            total_counts[code] += 1
            tab_counter[code] += 1

            if prev_code is not None:
                bigrams[(prev_code, code)] += 1

            prev_code = code

            # Compound component membership
            for comp in g.get("horley_components") or []:
                comp = str(comp).strip()
                if comp and comp != "?":
                    compound_components.add(comp)

        if tab_counter:
            tablet_counts[tablet_id] = tab_counter

    if not total_counts:
        return CorpusSignStats()

    # ── IC contributions ────────────────────────────────────────────────────
    n_total = sum(total_counts.values())
    ic_raw: dict[str, float] = {
        code: (cnt / n_total) ** 2 for code, cnt in total_counts.items()
    }
    sum_ic = sum(ic_raw.values())
    max_ic_contrib = max(ic_raw.values()) / sum_ic if sum_ic > 0 else 1.0
    ic_contributions = {
        code: (v / sum_ic) / max_ic_contrib for code, v in ic_raw.items()
    }

    # ── Bigram MI as predictor ───────────────────────────────────────────────
    n_bigrams = sum(bigrams.values())
    # P(s) and P(t) in consistent bigram-normalised probability space.
    unigram_s = {code: cnt / n_total for code, cnt in total_counts.items()}
    unigram_t = {code: cnt / n_total for code, cnt in total_counts.items()}

    bigram_mi_raw: dict[str, float] = defaultdict(float)
    if n_bigrams > 0:
        for (s, t), cnt in bigrams.items():
            p_st = cnt / n_bigrams
            p_s  = unigram_s.get(s, 0.0)
            p_t  = unigram_t.get(t, 0.0)
            # MI contribution: p(s,t) * log2[p(s,t) / (p(s)*p(t))]
            if p_s > 0 and p_t > 0:
                bigram_mi_raw[s] += p_st * math.log2(p_st / (p_s * p_t))

    # Normalise by max MI across signs; clip to [0, 1]
    max_mi = max(bigram_mi_raw.values(), default=1.0)
    max_mi = max(max_mi, 1e-9)
    bigram_mi_scores = {
        code: float(np.clip(v / max_mi, 0.0, 1.0))
        for code, v in bigram_mi_raw.items()
    }

    # ── Cross-tablet consistency ─────────────────────────────────────────────
    cross_tablet: dict[str, float] = {}
    all_tablets = list(tablet_counts.keys())
    for code in total_counts:
        freqs = np.array([tablet_counts[t].get(code, 0) for t in all_tablets],
                         dtype=np.float64)
        n_nonzero = int((freqs > 0).sum())
        if n_nonzero < 2:
            cross_tablet[code] = 0.0  # appears on ≤1 tablet → unreliable
        else:
            mean_f = freqs[freqs > 0].mean()
            std_f  = freqs[freqs > 0].std(ddof=1)
            cv = std_f / mean_f if mean_f > 0 else 0.0
            cross_tablet[code] = float(np.clip(1.0 - cv, 0.0, 1.0))

    logger.info(
        "compute_corpus_sign_stats: %d sign types, max IC contrib=%.4f, "
        "%d compound components",
        len(total_counts), max_ic_contrib, len(compound_components),
    )
    return CorpusSignStats(
        ic_contributions=ic_contributions,
        bigram_mi_scores=bigram_mi_scores,
        cross_tablet_consistency=cross_tablet,
        compound_component_codes=compound_components,
    )


# ---------------------------------------------------------------------------
# Raw feature extraction
# ---------------------------------------------------------------------------

def _extract_raw_features(
    classification: SignClassification,
    compound_score: float = 0.0,
    max_entropy: float = 10.0,
    ic_contribution: float = 0.0,
    bigram_mi: float = 0.0,
    cross_tablet_consistency: float = 0.0,
    is_compound_component: float = 0.0,
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
    ic_contribution : float
        Normalised IC weight from :class:`CorpusSignStats` ∈ [0, 1].
    bigram_mi : float
        Normalised bigram MI-as-predictor from :class:`CorpusSignStats` ∈ [0, 1].
    cross_tablet_consistency : float
        Tablet-frequency consistency score from :class:`CorpusSignStats` ∈ [0, 1].
    is_compound_component : float
        1.0 if the sign appears in any compound's ``horley_components`` list.

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
    feat[FEAT_COMPOUND]              = float(np.clip(compound_score, 0.0, 1.0))
    feat[FEAT_IC_CONTRIBUTION]       = float(np.clip(ic_contribution, 0.0, 1.0))
    feat[FEAT_BIGRAM_MI]             = float(np.clip(bigram_mi, 0.0, 1.0))
    feat[FEAT_CROSS_TABLET_CONSISTENCY] = float(np.clip(cross_tablet_consistency, 0.0, 1.0))
    feat[FEAT_COMPOUND_COMPONENT]    = float(np.clip(is_compound_component, 0.0, 1.0))
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
        corpus_stats: "CorpusSignStats | None" = None,
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
        corpus_stats : CorpusSignStats, optional
            Corpus-derived statistics for features 9–12.  If ``None``,
            those features are set to 0.0 (backward-compatible default).

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
                    ic_contribution=(
                        corpus_stats.ic_contribution(code)
                        if corpus_stats is not None else 0.0
                    ),
                    bigram_mi=(
                        corpus_stats.bigram_mi(code)
                        if corpus_stats is not None else 0.0
                    ),
                    cross_tablet_consistency=(
                        corpus_stats.consistency(code)
                        if corpus_stats is not None else 0.0
                    ),
                    is_compound_component=(
                        corpus_stats.is_compound_component(code)
                        if corpus_stats is not None else 0.0
                    ),
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
    corpus_stats: "CorpusSignStats | None" = None,
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
    corpus_stats : CorpusSignStats, optional
        Corpus-derived statistics for features 9–12.  Pass the output of
        :func:`compute_corpus_sign_stats` here.  If ``None``, the new
        features default to 0.0 (backward-compatible).
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

    raw = builder.build_feature_tensor(
        sign_codes, inventory, compound_scores, corpus_stats
    )
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
