"""
hackingrongo.zone_b.sign_classifier
===============================

Automatic classification of rongorongo signs into functional categories
using corpus statistics and parallel passage evidence.

The three categories match the structural hypotheses tested in Zone C:

* **Taxogram** — a non-phonetic boundary/class marker.  The primary
  diagnostic is a combination of high corpus frequency (top
  ``taxogram_frequency_threshold`` percentile) *and* high omission rate
  in parallel passage variants (≥
  ``taxogram_omission_rate_threshold``).  Glyph ``"200"`` is the
  canonical example with established scholarly consensus.

* **Logogram** — a sign with low positional entropy (it always appears
  in similar syntactic positions), suggesting a fixed word-level
  meaning.  Threshold: ``logogram_positional_entropy_threshold``.

* **Phonetic** — the residual category for signs that do not meet the
  taxogram or logogram criteria.  These are the primary candidates for
  syllabic/phonemic assignment in Zone C.

* **Unknown** — signs with insufficient corpus evidence (frequency
  below ``cfg.zone_b.contact_analysis.min_glyph_frequency``).

Public API
----------
``SignClass``
    Enum of the four functional categories.

``SignClassification``
    All classification evidence for a single sign.

``SignInventory``
    Container mapping Barthel codes → :class:`SignClassification`.

``classify_inventory``
    Factory function.  Accepts the pre-computed corpus statistics from
    Zone B and returns a :class:`SignInventory`.
"""

from __future__ import annotations

import enum
import logging
import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from omegaconf import DictConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# SignClass enum
# ---------------------------------------------------------------------------


class SignClass(enum.Enum):
    """Functional category of a rongorongo sign.

    Values
    ------
    PHONETIC
        Candidate for phonemic / syllabic assignment.
    TAXOGRAM
        Non-phonetic boundary or class marker; primarily glyph ``"200"``.
    LOGOGRAM
        Ideographic sign with stable positional distribution.
    UNKNOWN
        Insufficient corpus evidence for classification.
    """

    PHONETIC = "phonetic"
    TAXOGRAM = "taxogram"
    LOGOGRAM = "logogram"
    UNKNOWN = "unknown"


# ---------------------------------------------------------------------------
# SignClassification
# ---------------------------------------------------------------------------


@dataclass
class SignClassification:
    """All classification evidence for a single rongorongo sign.

    Attributes
    ----------
    code : str
        Barthel code of the sign.
    sign_class : SignClass
        Assigned functional category.
    confidence : float
        Confidence score ``∈ [0, 1]``.  Computed as the minimum margin
        by which the sign's scores exceed the relevant threshold(s).
    frequency_percentile : float
        Position of this sign's relative corpus frequency in the
        empirical frequency distribution (``0.0`` = rarest,
        ``1.0`` = most frequent).
    omission_rate : float
        Fraction of parallel passage variants in which this sign is
        absent at its canonical-form position.  ``0.0`` for signs that
        appear in no canonical form.
    positional_entropy : float
        Shannon entropy (bits) of this sign's positional distribution
        across its tablet occurrences.  Low entropy → restricted
        positional range → logogram candidate.
    sequential_entropy : float
        Shannon entropy (nats) of the distribution over right-context
        neighbours of this sign: H(X_{i+1} | X_i = S).  High entropy
        → sign appears in many different contexts → phonemic candidate.
        Zero for signs that never appear before another sign in the corpus.
        Defaults to 0.0 when not supplied (backward compatible).
    """

    code: str
    sign_class: SignClass
    confidence: float
    frequency_percentile: float
    omission_rate: float
    positional_entropy: float
    sequential_entropy: float = 0.0


# ---------------------------------------------------------------------------
# SignInventory
# ---------------------------------------------------------------------------


@dataclass
class SignInventory:
    """Mapping of Barthel codes to their :class:`SignClassification` records.

    Parameters
    ----------
    classifications : dict[str, SignClassification]
        Maps each Barthel code to its classification.

    Notes
    -----
    Built by :func:`classify_inventory`; not constructed directly.
    """

    classifications: dict[str, SignClassification] = field(
        default_factory=dict
    )

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_class(self, code: str) -> SignClass:
        """Return the :class:`SignClass` for a Barthel code.

        Parameters
        ----------
        code : str
            Barthel code to look up.

        Returns
        -------
        SignClass
            :attr:`~SignClass.UNKNOWN` if the code is not in the
            inventory.
        """
        record = self.classifications.get(code)
        return record.sign_class if record is not None else SignClass.UNKNOWN

    def get_taxograms(self) -> list[str]:
        """Return Barthel codes classified as :attr:`~SignClass.TAXOGRAM`.

        Returns
        -------
        list[str]
            Sorted list.
        """
        return sorted(
            c
            for c, r in self.classifications.items()
            if r.sign_class is SignClass.TAXOGRAM
        )

    def get_logograms(self) -> list[str]:
        """Return Barthel codes classified as :attr:`~SignClass.LOGOGRAM`.

        Returns
        -------
        list[str]
            Sorted list.
        """
        return sorted(
            c
            for c, r in self.classifications.items()
            if r.sign_class is SignClass.LOGOGRAM
        )

    def get_phonetics(self) -> list[str]:
        """Return Barthel codes classified as :attr:`~SignClass.PHONETIC`.

        Returns
        -------
        list[str]
            Sorted list.
        """
        return sorted(
            c
            for c, r in self.classifications.items()
            if r.sign_class is SignClass.PHONETIC
        )

    def to_dict(self) -> dict[str, dict[str, Any]]:
        """Serialise the inventory to a JSON-compatible dict.

        Returns
        -------
        dict[str, dict[str, Any]]
            Maps Barthel code to a dict with keys ``sign_class``,
            ``confidence``, ``frequency_percentile``, ``omission_rate``,
            ``positional_entropy``.
        """
        return {
            code: {
                "sign_class": rec.sign_class.value,
                "confidence": rec.confidence,
                "frequency_percentile": rec.frequency_percentile,
                "omission_rate": rec.omission_rate,
                "positional_entropy": rec.positional_entropy,
                "sequential_entropy": rec.sequential_entropy,
            }
            for code, rec in self.classifications.items()
        }

    def summary(self) -> str:
        """Return a brief human-readable summary line.

        Returns
        -------
        str
        """
        counts: dict[SignClass, int] = {}
        for rec in self.classifications.values():
            counts[rec.sign_class] = counts.get(rec.sign_class, 0) + 1
        parts = [
            f"{sc.value}={counts.get(sc, 0)}" for sc in SignClass
        ]
        return "SignInventory(" + ", ".join(parts) + ")"


# ---------------------------------------------------------------------------
# classify_inventory
# ---------------------------------------------------------------------------


def classify_inventory(
    frequency_stats: dict[str, float],
    omission_rates: dict[str, float],
    positional_entropy: dict[str, float],
    cfg: DictConfig,
    sequential_entropy: dict[str, float] | None = None,
) -> SignInventory:
    """Classify all signs in the corpus into functional categories.

    Parameters
    ----------
    frequency_stats : dict[str, float]
        Maps Barthel code → relative frequency (fraction of total corpus
        tokens).  Produced by Zone B contact analysis.
    omission_rates : dict[str, float]
        Maps Barthel code → parallel-passage omission rate ``∈ [0, 1]``.
        Produced by :func:`~hackingrongo.data.parallels.compute_omission_rates`.
    positional_entropy : dict[str, float]
        Maps Barthel code → Shannon entropy (bits) of the sign's
        positional distribution within its tablet occurrences.
        Produced by Zone B entropy analysis.
    cfg : DictConfig
        Root Hydra config.  Reads:

        * ``cfg.zone_b.sign_classifier.taxogram_frequency_threshold``
        * ``cfg.zone_b.sign_classifier.taxogram_omission_rate_threshold``
        * ``cfg.zone_b.sign_classifier.logogram_positional_entropy_threshold``
        * ``cfg.zone_b.contact_analysis.min_glyph_frequency``
        * ``cfg.zone_b.sign_classifier.sequential_entropy_phonetic_threshold``
          (optional; default 0.5 nats)

    sequential_entropy : dict[str, float] or None, optional
        Maps Barthel code → H(X_{i+1} | X_i = S) in nats (output of
        ``scripts/train_sequential_embeddings.py``).  When provided, high
        sequential entropy boosts the PHONETIC confidence score.
        Pass ``None`` (default) to run without this feature.

    Returns
    -------
    SignInventory
        Fully populated inventory.

    Notes
    -----
    Classification is a two-pass process:

    1. Compute the empirical frequency percentile for every sign.
    2. Apply thresholds in order: taxogram → logogram → phonetic →
       unknown (insufficient evidence).

    The taxogram criterion requires **both** high frequency *and* high
    omission rate because frequency alone is not diagnostic (common signs
    are not necessarily taxograms).  The dual criterion reduces false
    positives compared to Souza (2022) which uses only frequency.
    """
    sc_cfg = cfg.zone_b.sign_classifier
    taxogram_freq_thresh: float = float(sc_cfg.taxogram_frequency_threshold)
    taxogram_omission_thresh: float = float(sc_cfg.taxogram_omission_rate_threshold)
    logogram_entropy_thresh: float = float(
        sc_cfg.logogram_positional_entropy_threshold
    )
    min_freq: float = float(cfg.zone_b.contact_analysis.min_glyph_frequency)
    # Sequential entropy threshold: above this value, a PHONETIC sign's
    # confidence is boosted proportionally.  Read from config with fallback.
    try:
        seq_ent_thresh: float = float(sc_cfg.sequential_entropy_phonetic_threshold)
    except Exception:
        seq_ent_thresh = 0.5  # nats; reasonable default

    all_codes = sorted(
        set(frequency_stats) | set(omission_rates) | set(positional_entropy)
    )

    if not all_codes:
        logger.warning("classify_inventory: no sign codes provided.")
        return SignInventory()

    # Build frequency percentile lookup.
    freq_values = np.array(
        [frequency_stats.get(c, 0.0) for c in all_codes], dtype=np.float64
    )
    if freq_values.max() > 0:
        freq_percentiles = _empirical_percentile_rank(freq_values)
    else:
        freq_percentiles = np.zeros(len(all_codes))

    # Compute total corpus tokens for min-frequency threshold.
    total_freq = sum(frequency_stats.values())

    classifications: dict[str, SignClassification] = {}

    for i, code in enumerate(all_codes):
        freq = frequency_stats.get(code, 0.0)
        omission = omission_rates.get(code, 0.0)
        entropy = positional_entropy.get(code, 0.0)
        seq_ent = sequential_entropy.get(code, 0.0) if sequential_entropy else 0.0
        freq_pct = float(freq_percentiles[i])

        # Insufficient evidence: absolute frequency below corpus minimum.
        abs_freq = freq * total_freq if total_freq > 0 else 0.0
        if abs_freq < min_freq:
            sign_class = SignClass.UNKNOWN
            confidence = 0.0
        elif (
            freq_pct >= taxogram_freq_thresh
            and omission >= taxogram_omission_thresh
        ):
            sign_class = SignClass.TAXOGRAM
            # Confidence = geometric mean of the two margins above threshold.
            freq_margin = freq_pct - taxogram_freq_thresh
            omission_margin = omission - taxogram_omission_thresh
            confidence = min(1.0, math.sqrt(freq_margin * omission_margin + 1e-9))
        elif entropy <= logogram_entropy_thresh:
            sign_class = SignClass.LOGOGRAM
            # Confidence = normalised distance below the entropy threshold.
            confidence = min(
                1.0,
                (logogram_entropy_thresh - entropy) / (logogram_entropy_thresh + 1e-9),
            )
        else:
            sign_class = SignClass.PHONETIC
            # Confidence = distance from both taxogram/logogram boundaries.
            dist_from_taxogram = min(
                taxogram_freq_thresh - freq_pct,
                taxogram_omission_thresh - omission,
            )
            dist_from_logogram = entropy - logogram_entropy_thresh
            confidence = min(
                1.0,
                (dist_from_taxogram + dist_from_logogram) / 2.0,
            )
            confidence = max(0.0, confidence)
            # Sequential entropy boost: high H(context) → more certain PHONETIC.
            # Additive blend capped at 1.0; weight = 0.3 so the boost cannot
            # exceed 0.3 and the positional-entropy signal retains priority.
            if sequential_entropy is not None and seq_ent > seq_ent_thresh:
                seq_boost = 0.3 * min(1.0, (seq_ent - seq_ent_thresh) / (seq_ent_thresh + 1e-9))
                confidence = min(1.0, confidence + seq_boost)

        classifications[code] = SignClassification(
            code=code,
            sign_class=sign_class,
            confidence=confidence,
            frequency_percentile=freq_pct,
            omission_rate=omission,
            positional_entropy=entropy,
            sequential_entropy=seq_ent,
        )

    inventory = SignInventory(classifications=classifications)
    logger.info(
        "Sign classification complete: %s", inventory.summary()
    )
    return inventory


def _empirical_percentile_rank(values: np.ndarray) -> np.ndarray:
    """Compute the empirical percentile rank of each element in ``values``.

    Parameters
    ----------
    values : numpy.ndarray
        1-D array of non-negative float values.

    Returns
    -------
    numpy.ndarray
        Array of the same length; each element is in ``[0, 1]``.
    """
    n = len(values)
    if n == 0:
        return np.array([], dtype=np.float64)
    order = np.argsort(np.argsort(values))  # rank without breaking ties
    return order.astype(np.float64) / max(n - 1, 1)
