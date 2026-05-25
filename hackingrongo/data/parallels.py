"""
hackingrongo.data.parallels
======================

Parallel passage loader and variant alignment utilities.

Parallel passages are sequences of rongorongo glyphs that appear across
multiple tablets in recognisably variant (but cognate) forms.  They are
the primary scholarly evidence for internal structure because they provide
the only sequences where independent identity is established without
decipherment.

This module:

* Loads Horley parallel passages from a CSV file (adapted from
  Souza 2022 ``horley_parallels.csv``, MIT licence).
* Retains the taxogram sign (glyph ``"200"``) as a *tagged* element
  in every form rather than silently discarding it.
* Groups variant spellings by passage ID with full tablet provenance.
* Computes per-position omission rates used by
  :mod:`~hackingrongo.zone_b.sign_classifier` to detect taxogram candidates.

Public API
----------
``PassageVariant``
    Single variant of a parallel passage from one tablet.

``ParallelPassage``
    Complete record: canonical form, all variants, taxogram positions,
    stratum breakdown.

``tag_taxogram_positions``
    Mark 0-based indices in a glyph form where the taxogram appears.

``load_parallel_passages``
    Load from the Horley parallels CSV file.

``load_parallel_variants_json``
    Load from the structured JSON variant file.

``compute_omission_rates``
    Compute per-sign parallel-passage omission rates for the full catalog.
"""

from __future__ import annotations

import csv
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from omegaconf import DictConfig

from hackingrongo.data.catalog import SignCatalog

logger = logging.getLogger(__name__)

# Barthel code for the taxogram sign.  Tagged but never silently dropped.
_TAXOGRAM_CODE: str = "200"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PassageVariant:
    """One variant spelling of a parallel passage from a specific tablet.

    Parameters
    ----------
    form : tuple[str, ...]
        Ordered sequence of Barthel codes constituting this variant.
        Glyph ``"200"`` (taxogram) is retained and its positions are
        tracked via :attr:`ParallelPassage.taxogram_positions`.
    tablet_id : str
        Identifier of the source tablet.
    stratum : str
        Temporal stratum label (``"pre"`` | ``"early"`` | ``"late"``).
    side : str
        Tablet side if known (``"a"`` or ``"b"``), otherwise ``""``.
    start_position : int
        1-based starting position on the tablet, or ``-1`` if unknown.
    """

    form: tuple[str, ...]
    tablet_id: str
    stratum: str
    side: str = ""
    start_position: int = -1


@dataclass
class ParallelPassage:
    """Complete parallel passage with all known variant attestations.

    Attributes
    ----------
    passage_id : str
        Unique passage identifier (e.g. ``"P001"``).
    canonical_form : list[str]
        Reference form used for alignment.  Typically the longest or most
        complete variant; established by the CSV / JSON source.
    variants : list[PassageVariant]
        All known variant spellings, one entry per tablet occurrence.
    taxogram_positions : frozenset[int]
        0-based indices in ``canonical_form`` where glyph ``"200"``
        appears.  Used by Zone C validation to separate phonetic content
        from taxogram occurrences.

    Notes
    -----
    Glyph ``"200"`` is intentionally preserved here.  Souza (2022)
    strips it with ``if i != '200'`` — a silent lossy operation.  This
    module keeps it tagged so downstream code can make an explicit,
    auditable decision about whether to include or exclude it.
    """

    passage_id: str
    canonical_form: list[str]
    variants: list[PassageVariant] = field(default_factory=list)
    taxogram_positions: frozenset[int] = field(default_factory=frozenset)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def n_variants(self) -> int:
        """Number of variant attestations for this passage."""
        return len(self.variants)

    @property
    def attested_strata(self) -> set[str]:
        """Set of strata in which this passage is attested."""
        return {v.stratum for v in self.variants}

    # ------------------------------------------------------------------
    # Stratum / omission utilities
    # ------------------------------------------------------------------

    def get_variants_for_stratum(self, stratum: str) -> list[PassageVariant]:
        """Return all variants from a given stratum.

        Parameters
        ----------
        stratum : str
            Stratum label to filter by.

        Returns
        -------
        list[PassageVariant]
        """
        return [v for v in self.variants if v.stratum == stratum]

    def omission_rate_at_position(self, position: int) -> float:
        """Fraction of variants that omit the canonical sign at ``position``.

        A variant "omits" a position if its ``form`` length is ≤ ``position``
        (the variant is shorter than the canonical at this index).
        High omission rates at a given position — especially for
        sign ``"200"`` — are the primary taxogram detection signal used
        by :mod:`~hackingrongo.zone_b.sign_classifier`.

        Parameters
        ----------
        position : int
            0-based index into ``canonical_form``.

        Returns
        -------
        float
            Value in ``[0, 1]``.  Returns ``0.0`` if there are no variants.
        """
        if not self.variants:
            return 0.0
        omitted = sum(1 for v in self.variants if len(v.form) <= position)
        return omitted / len(self.variants)

    def phonetic_canonical_form(self) -> list[str]:
        """Return the canonical form with taxogram positions removed.

        Returns
        -------
        list[str]
            Canonical form codes at non-taxogram positions only.
        """
        return [
            code
            for i, code in enumerate(self.canonical_form)
            if i not in self.taxogram_positions
        ]


# ---------------------------------------------------------------------------
# Taxogram tagging
# ---------------------------------------------------------------------------


def tag_taxogram_positions(
    form: list[str],
    taxogram_code: str = _TAXOGRAM_CODE,
) -> frozenset[int]:
    """Find 0-based indices in ``form`` where the taxogram sign appears.

    Parameters
    ----------
    form : list[str]
        Ordered glyph code sequence.
    taxogram_code : str
        Barthel code of the taxogram sign.  Defaults to ``"200"``,
        the primary rongorongo taxogram per Horley (2021).

    Returns
    -------
    frozenset[int]
        Immutable set of 0-based indices.
    """
    return frozenset(i for i, code in enumerate(form) if code == taxogram_code)


# ---------------------------------------------------------------------------
# CSV loader
# ---------------------------------------------------------------------------

_REQUIRED_CSV_COLUMNS: frozenset[str] = frozenset(
    {"passage_id", "tablet_id", "side", "stratum", "start_position", "glyph_sequence"}
)


def load_parallel_passages(
    csv_path: Path,
    catalog: SignCatalog,
    cfg: DictConfig,
) -> list[ParallelPassage]:
    """Load parallel passages from the Horley parallels CSV.

    Expected CSV schema (header row required)::

        passage_id, tablet_id, side, stratum, start_position, glyph_sequence

    where ``glyph_sequence`` is a space-separated string of Barthel codes
    (e.g. ``"001 002 200 003"``).

    **CRITICAL:** For Zone C parallel passage scoring to work, the CSV must
    contain MULTIPLE rows with the SAME ``passage_id`` but DIFFERENT
    ``tablet_id`` values. This groups variant attestations of the same passage
    across tablets. Without multi-tablet groups, the alignment scoring has
    nothing to align.

    Each unique ``passage_id`` groups all its variants (different tablet
    attestations) into a single :class:`ParallelPassage` object. The first
    variant encountered defines the canonical form. Glyph ``"200"`` is
    retained in all forms and tagged via :func:`tag_taxogram_positions`
    rather than silently stripped.

    Parameters
    ----------
    csv_path : Path
        Absolute path to ``data/parallels/horley_parallels.csv`` or
        ``data/parallels/horley_parallels_transformed.csv``.
    catalog : SignCatalog
        Used to validate Barthel codes in the CSV.  Codes absent from
        the catalog emit a one-time WARNING rather than raising.
    cfg : DictConfig
        Root Hydra config (reserved for future filtering options).

    Returns
    -------
    list[ParallelPassage]
        Passages sorted by ``passage_id``.

    Raises
    ------
    FileNotFoundError
        If the CSV does not exist at ``csv_path``.
    ValueError
        If the CSV header is missing required columns.
    """
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Horley parallels CSV not found: {csv_path}"
        )

    passages: dict[str, ParallelPassage] = {}
    unknown_codes: set[str] = set()

    with csv_path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None:
            raise ValueError(f"CSV file has no header row: {csv_path}")
        missing = _REQUIRED_CSV_COLUMNS - set(reader.fieldnames)
        if missing:
            raise ValueError(
                f"CSV is missing required columns: {sorted(missing)}"
            )

        for row_num, row in enumerate(reader, start=2):
            passage_id: str = row["passage_id"].strip()
            tablet_id: str = row["tablet_id"].strip()
            side: str = row["side"].strip()
            stratum: str = row["stratum"].strip()
            raw_pos: str = row["start_position"].strip()
            start_position: int = (
                int(raw_pos) if raw_pos.lstrip("-").isdigit() else -1
            )
            glyph_sequence: str = row["glyph_sequence"].strip()

            form: list[str] = [
                code.strip()
                for code in glyph_sequence.split()
                if code.strip()
            ]

            # Validate codes; warn once per unknown code.
            for code in form:
                if code not in catalog.signs and code not in unknown_codes:
                    logger.warning(
                        "Row %d: Barthel code '%s' not in sign catalog.",
                        row_num,
                        code,
                    )
                    unknown_codes.add(code)

            variant = PassageVariant(
                form=tuple(form),
                tablet_id=tablet_id,
                stratum=stratum,
                side=side,
                start_position=start_position,
            )

            if passage_id not in passages:
                # First occurrence defines the canonical form.
                passages[passage_id] = ParallelPassage(
                    passage_id=passage_id,
                    canonical_form=form,
                    variants=[variant],
                    taxogram_positions=tag_taxogram_positions(form),
                )
            else:
                passages[passage_id].variants.append(variant)

    result = sorted(passages.values(), key=lambda p: p.passage_id)
    
    # Count passages by variant abundance
    n_multi_tablet = sum(1 for p in result if p.n_variants >= 2)
    n_single = len(result) - n_multi_tablet
    total_variants = sum(p.n_variants for p in result)
    
    logger.info(
        "Loaded %d parallel passages (%d total variants, %d unknown codes warned).",
        len(result),
        total_variants,
        len(unknown_codes),
    )
    
    # CRITICAL: Zone C alignment scoring requires multi-tablet passages
    if n_multi_tablet == 0:
        logger.critical(
            "⚠ CRITICAL: NO MULTI-TABLET PASSAGES LOADED! Zone C parallel passage "
            "scoring requires passages with n_variants >= 2 (same passage on different "
            "tablets). All %d loaded passages have only 1 tablet attestation. "
            "Populate parallel_variants.json from Horley (2021) Appendix with "
            "cross-tablet groupings for alignment scoring to function.",
            len(result),
        )
    elif n_single > 0:
        logger.warning(
            "Zone C alignment data quality: %d multi-tablet passages (good), "
            "but %d single-tablet passages (will not contribute to alignment scoring). "
            "Consider expanding parallel_variants.json.",
            n_multi_tablet,
            n_single,
        )
    
    return result


# ---------------------------------------------------------------------------
# JSON variant file loader
# ---------------------------------------------------------------------------


def load_parallel_variants_json(
    json_path: Path,
    catalog: SignCatalog,
) -> list[ParallelPassage]:
    """Load structured parallel passage groups from the JSON variant file.

    Use this instead of :func:`load_parallel_passages` when the full
    variant metadata has been curated into
    ``data/parallels/parallel_variants.json``.

    Parameters
    ----------
    json_path : Path
        Absolute path to ``data/parallels/parallel_variants.json``.
    catalog : SignCatalog
        Used to validate Barthel codes (warnings only).

    Returns
    -------
    list[ParallelPassage]
        Passages sorted by ``passage_id``.

    Raises
    ------
    FileNotFoundError
        If the JSON file does not exist.
    """
    if not json_path.exists():
        raise FileNotFoundError(
            f"Parallel variants JSON not found: {json_path}"
        )

    with json_path.open("r", encoding="utf-8") as fh:
        data: Any = json.load(fh)

    if not isinstance(data, dict):
        raise ValueError(
            f"Parallel variants JSON must be a dict at the top level, "
            f"got {type(data).__name__}: {json_path}"
        )

    passages_raw: list[dict[str, Any]] = data.get("passages", [])
    result: list[ParallelPassage] = []
    unknown_codes: set[str] = set()

    for entry_num, entry in enumerate(passages_raw, start=1):
        if "passage_id" not in entry:
            raise ValueError(
                f"Entry #{entry_num} in {json_path} is missing required key 'passage_id'."
            )
        if "canonical_form" not in entry:
            raise ValueError(
                f"Entry #{entry_num} ({entry['passage_id']!r}) in {json_path} "
                "is missing required key 'canonical_form'."
            )

        passage_id: str = str(entry["passage_id"])
        canonical_form: list[str] = list(entry["canonical_form"])
        taxogram_pos = tag_taxogram_positions(canonical_form)

        # Validate Barthel codes in canonical form.
        for code in canonical_form:
            if code not in catalog.signs and code not in unknown_codes:
                logger.warning(
                    "Entry %d (%r): Barthel code '%s' not in sign catalog.",
                    entry_num,
                    passage_id,
                    code,
                )
                unknown_codes.add(code)

        variants: list[PassageVariant] = []
        for v_num, v in enumerate(entry.get("variants", []), start=1):
            if "form" not in v:
                raise ValueError(
                    f"Variant #{v_num} of passage {passage_id!r} in {json_path} "
                    "is missing required key 'form'."
                )
            variants.append(
                PassageVariant(
                    form=tuple(v["form"]),
                    tablet_id=str(v.get("tablet_id", "")),
                    stratum=str(v.get("stratum", "")),
                    side=str(v.get("side", "")),
                    start_position=int(v.get("start_position", -1)),
                )
            )

        result.append(
            ParallelPassage(
                passage_id=passage_id,
                canonical_form=canonical_form,
                variants=variants,
                taxogram_positions=taxogram_pos,
            )
        )

    result.sort(key=lambda p: p.passage_id)

    n_multi_tablet = sum(1 for p in result if p.n_variants >= 2)
    n_single = len(result) - n_multi_tablet

    logger.info(
        "Loaded %d parallel passages from JSON (%d total variants, %d unknown codes warned).",
        len(result),
        sum(p.n_variants for p in result),
        len(unknown_codes),
    )

    if n_multi_tablet == 0:
        logger.critical(
            "⚠ CRITICAL: NO MULTI-TABLET PASSAGES LOADED! Zone C parallel passage "
            "scoring requires passages with n_variants >= 2 (same passage on different "
            "tablets). All %d loaded passages have only 1 tablet attestation. "
            "Populate parallel_variants.json from Horley (2021) Appendix with "
            "cross-tablet groupings for alignment scoring to function.",
            len(result),
        )
    elif n_single > 0:
        logger.warning(
            "Zone C alignment data quality: %d multi-tablet passages (good), "
            "but %d single-tablet passages (will not contribute to alignment scoring). "
            "Consider expanding parallel_variants.json.",
            n_multi_tablet,
            n_single,
        )

    return result


# ---------------------------------------------------------------------------
# Omission-rate utilities (consumed by sign_classifier.py)
# ---------------------------------------------------------------------------


def compute_omission_rates(
    passages: list[ParallelPassage],
    catalog: SignCatalog,
) -> dict[str, float]:
    """Compute parallel-passage omission rates for every sign in the catalog.

    A sign has a high omission rate if, across all parallel passages
    where it appears in the canonical form, it is frequently absent from
    variant spellings.  This is the primary signal used by
    :func:`~hackingrongo.zone_b.sign_classifier.classify_inventory` to detect
    taxogram candidates (high omission rate ≥
    ``cfg.zone_b.sign_classifier.taxogram_omission_rate_threshold``).

    Parameters
    ----------
    passages : list[ParallelPassage]
        Loaded parallel passages.
    catalog : SignCatalog
        Enumerates all known sign codes so that signs absent from any
        canonical form receive a ``0.0`` omission rate.

    Returns
    -------
    dict[str, float]
        Maps each Barthel code to its mean omission rate across all
        passages where it appears canonically.  Signs that do not appear
        in any canonical form have omission rate ``0.0``.
    """
    omission_counts: dict[str, int] = {}
    position_counts: dict[str, int] = {}

    for passage in passages:
        n = passage.n_variants
        if n == 0:
            continue
        for pos, code in enumerate(passage.canonical_form):
            position_counts[code] = position_counts.get(code, 0) + n
            n_omitted = round(passage.omission_rate_at_position(pos) * n)
            omission_counts[code] = omission_counts.get(code, 0) + n_omitted

    rates: dict[str, float] = {}
    for code in catalog.signs:
        total = position_counts.get(code, 0)
        rates[code] = omission_counts.get(code, 0) / total if total > 0 else 0.0

    return rates
