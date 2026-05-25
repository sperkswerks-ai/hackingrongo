"""
hackingrongo.data.catalog
====================

Single source of truth for all rongorongo sign identity and encoding
questions.

This module loads and validates the three catalog artifacts:

* ``horley_encoding.json``  — Barthel (1958) to Horley (2021) code mapping.
* ``allographs.json``       — Horley allograph groupings (variant → canonical).
* ``sign_metadata.json``    — per-sign scholarly metadata (readings, taxogram
                              flag, notes).

**Every module** that needs to look up a sign code imports
:class:`SignCatalog` from here.  No other module may maintain its own
local sign-encoding dict.

Public API
----------
``SignRecord``
    Frozen dataclass: all catalog information for a single sign.

``SignCatalog``
    Container loaded from the three JSON files.  Provides
    ``barthel_to_horley()``, ``horley_to_barthels()``,
    ``get_allograph_group()``, ``get_canonical_id()``,
    ``is_taxogram()``, ``get_taxogram_codes()``, and
    ``barthel_to_implicit_group()``.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from omegaconf import DictConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# SignRecord
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SignRecord:
    """All catalog-level information for a single rongorongo sign.

    Parameters
    ----------
    barthel_code : str
        Barthel (1958) numeric string code, e.g. ``"001"`` or ``"380a"``.
    horley_code : str
        Horley (2021) code for the same sign.  Empty string if the sign
        has no Horley equivalent.
    canonical_id : str
        Canonical sign ID as defined by the allograph grouping.  Equals
        ``barthel_code`` for canonical signs; equals a different code
        for allographic variants.
    is_taxogram : bool
        ``True`` if this sign is classified as a taxogram in
        ``sign_metadata.json``.  The primary taxogram is sign ``"200"``.
    scholarly_readings : tuple[str, ...]
        Proposed phoneme/morpheme readings from the literature
        (Barthel 1958, Fischer 1997, Horley 2021).  Empty if unassigned.
    notes : str
        Free-text notes from ``sign_metadata.json``.
    """

    barthel_code: str
    horley_code: str
    canonical_id: str
    is_taxogram: bool
    scholarly_readings: tuple[str, ...] = field(default_factory=tuple)
    notes: str = ""


# ---------------------------------------------------------------------------
# SignCatalog
# ---------------------------------------------------------------------------


class SignCatalog:
    """Central registry for rongorongo sign identities and encodings.

    Constructed from three JSON catalog files.  Use :meth:`load` to build
    from the Hydra config rather than calling the constructor directly.

    Parameters
    ----------
    horley_encoding : dict[str, str]
        Contents of ``horley_encoding.json``.  Maps each Barthel code
        string to its Horley code string.  Keys starting with ``"_"``
        are schema metadata and are ignored.
    allographs : dict[str, str]
        Contents of ``allographs.json``.  Maps each variant Barthel code
        to its canonical sign ID.
    sign_metadata : dict[str, dict[str, Any]]
        Contents of ``sign_metadata.json``.  Maps Barthel code strings
        to metadata dicts.

    Attributes
    ----------
    signs : dict[str, SignRecord]
        All sign records keyed by Barthel code.

    Methods (key)
    -------------
    ``barthel_to_horley()``, ``horley_to_barthels()`` — encoding translation.
    ``get_canonical_id()``, ``get_allograph_group()`` — allograph navigation.
    ``is_taxogram()``, ``get_taxogram_codes()`` — sign-class queries.
    ``barthel_to_implicit_group()`` — Barthel (1958) morphological taxonomy.
    """

    def __init__(
        self,
        horley_encoding: dict[str, str],
        allographs: dict[str, str],
        sign_metadata: dict[str, dict[str, Any]],
    ) -> None:
        # Strip schema-metadata keys (start with "_")
        horley_clean = {k: v for k, v in horley_encoding.items() if not k.startswith("_")}
        allographs_clean = {k: v for k, v in allographs.items() if not k.startswith("_")}
        meta_clean = {k: v for k, v in sign_metadata.items() if not k.startswith("_")}

        self._barthel_to_horley_map: dict[str, str] = dict(horley_clean)

        # Inverted index: Horley code → list of Barthel codes
        self._horley_to_barthels_map: dict[str, list[str]] = {}
        for barthel, horley in horley_clean.items():
            if horley:
                self._horley_to_barthels_map.setdefault(horley, []).append(barthel)

        self._allographs: dict[str, str] = dict(allographs_clean)

        # Canonical → all variants (including the canonical sign itself)
        self._sign_groups: dict[str, list[str]] = {}
        for variant, canonical in allographs_clean.items():
            self._sign_groups.setdefault(canonical, []).append(variant)

        # Union of all known Barthel codes
        all_codes = (
            set(horley_clean.keys())
            | set(allographs_clean.keys())
            | set(meta_clean.keys())
        )

        self.signs: dict[str, SignRecord] = {}
        for code in sorted(all_codes):
            meta = meta_clean.get(code, {})
            self.signs[code] = SignRecord(
                barthel_code=code,
                horley_code=horley_clean.get(code, ""),
                canonical_id=allographs_clean.get(code, code),
                is_taxogram=bool(meta.get("is_taxogram_candidate", False)),
                scholarly_readings=tuple(meta.get("scholarly_readings", [])),
                notes=str(meta.get("notes", "")),
            )

        logger.debug(
            "SignCatalog initialised: %d signs, %d Barthel→Horley mappings, "
            "%d allograph groups.",
            len(self.signs),
            len(self._barthel_to_horley_map),
            len(self._sign_groups),
        )

    # ------------------------------------------------------------------
    # Encoding translation
    # ------------------------------------------------------------------

    # Strips leading zeros from a numeric prefix: '001' -> '1', '003a' -> '3a'.
    # Used to normalise Barthel corpus codes (zero-padded 3-digit) against the
    # horley_encoding.json keys (unpadded).  The zero-stripping is done lazily
    # (only when the direct lookup misses) to avoid rebuilding the map.
    _ZERO_PAD = re.compile(r'^0+(?=\d)')

    def barthel_to_horley(self, barthel_code: str) -> str | None:
        """Return the Horley (2021) code for a Barthel code.

        The Barthel corpus uses zero-padded 3-digit codes (``"001"``, ``"076"``)
        while ``horley_encoding.json`` uses unpadded keys (``"1"``, ``"76"``).
        This method tries the exact code first, then falls back to the
        zero-stripped form so both representations resolve correctly.

        Parameters
        ----------
        barthel_code : str
            Barthel code to translate (padded or unpadded).

        Returns
        -------
        str or None
            Corresponding Horley code, or ``None`` if the mapping is
            absent or empty.
        """
        result = self._barthel_to_horley_map.get(barthel_code, "")
        if result:
            return result
        # Fallback: strip leading zeros and retry
        normalized = self._ZERO_PAD.sub("", barthel_code)
        if normalized != barthel_code:
            result = self._barthel_to_horley_map.get(normalized, "")
            return result if result else None
        return None

    def horley_to_barthels(self, horley_code: str) -> list[str]:
        """Return all Barthel codes that map to the given Horley code.

        Parameters
        ----------
        horley_code : str
            Horley code to look up.

        Returns
        -------
        list[str]
            Barthel codes mapping to this Horley code, in insertion
            order.  Empty list if the code is unknown.
        """
        return list(self._horley_to_barthels_map.get(horley_code, []))

    # ------------------------------------------------------------------
    # Allograph group navigation
    # ------------------------------------------------------------------

    def get_canonical_id(self, code: str) -> str:
        """Return the canonical sign ID for a Barthel code.

        If the code is already canonical, or is not in the allograph
        catalog at all, the code itself is returned unchanged.

        Parameters
        ----------
        code : str
            Barthel code (canonical or variant) to resolve.

        Returns
        -------
        str
            Canonical Barthel code.
        """
        return self._allographs.get(code, code)

    def get_allograph_group(self, code: str) -> list[str]:
        """Return all Barthel codes in the same allograph group as ``code``.

        Parameters
        ----------
        code : str
            Any Barthel code (canonical or variant).

        Returns
        -------
        list[str]
            All variants sharing the same canonical sign, including the
            canonical sign itself.  Returns ``[code]`` if the code is
            not in any group.
        """
        canonical = self.get_canonical_id(code)
        return list(self._sign_groups.get(canonical, [code]))

    # ------------------------------------------------------------------
    # Sign class queries
    # ------------------------------------------------------------------

    def is_taxogram(self, code: str) -> bool:
        """Return ``True`` if the sign is classified as a taxogram.

        Parameters
        ----------
        code : str
            Barthel code.

        Returns
        -------
        bool
        """
        record = self.signs.get(code)
        return record.is_taxogram if record is not None else False

    def get_taxogram_codes(self) -> list[str]:
        """Return all Barthel codes classified as taxograms.

        Returns
        -------
        list[str]
            Sorted list of taxogram Barthel codes.
        """
        return sorted(c for c, r in self.signs.items() if r.is_taxogram)

    @staticmethod
    def barthel_to_implicit_group(code: str) -> str:
        """Map a Barthel code to Barthel's (1958) implicit sign taxonomy.

        Barthel organised his catalogue by morphological class, encoding the
        taxonomy directly in his numbering scheme.  This method reconstructs
        that classification programmatically without requiring a hand-built
        lookup table.

        Scheme
        ------
        1–199     : ``'objects_plants_phenomena'``
        200–299   : ``'anthropomorphic_head<N>'`` where ``N`` is the tens
                    digit of the code (0–9), reflecting Barthel's head-type
                    sub-series within the 200s.
        300–399   : ``'anthropomorphic_300series'``
        400–499   : ``'miscellaneous_400s'``
        500–599   : ``'miscellaneous_500s'``
        600–699   : ``'bird_headed'``
        700–799   : ``'zoomorphic'``
        otherwise : ``'compound_or_other'``

        Parameters
        ----------
        code : str
            Barthel code string, with or without zero-padding and with or
            without an alphabetic allograph suffix (e.g. ``"076"``,
            ``"380a"``, ``"700"``).

        Returns
        -------
        str
            Group label string.

        References
        ----------
        Barthel, T.S. (1958). *Grundlagen zur Entzifferung der
        Osterinselschrift*. Hamburg: Cram, de Gruyter.
        """
        try:
            n = int(re.sub(r"[^0-9]", "", code))
        except ValueError:
            return "unknown"

        if 1 <= n <= 199:
            return "objects_plants_phenomena"
        elif 200 <= n <= 299:
            head = (n // 10) % 10  # tens digit encodes head type
            return f"anthropomorphic_head{head}"
        elif 300 <= n <= 399:
            return "anthropomorphic_300series"
        elif 400 <= n <= 499:
            return "miscellaneous_400s"
        elif 500 <= n <= 599:
            return "miscellaneous_500s"
        elif 600 <= n <= 699:
            return "bird_headed"
        elif 700 <= n <= 799:
            return "zoomorphic"
        else:
            return "compound_or_other"

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def load(cls, cfg: DictConfig, project_root: Path) -> "SignCatalog":
        """Load all three catalog files and construct a :class:`SignCatalog`.

        Parameters
        ----------
        cfg : DictConfig
            Root Hydra config.  Reads ``cfg.paths.horley_encoding_json``,
            ``cfg.paths.allographs_json``, and
            ``cfg.paths.sign_metadata_json``.
        project_root : Path
            Absolute path to the repository root, obtained via
            ``hydra.utils.get_original_cwd()`` in ``pipeline.py``.

        Returns
        -------
        SignCatalog

        Raises
        ------
        FileNotFoundError
            If any of the three catalog files are missing.
        json.JSONDecodeError
            If any catalog file is not valid JSON.
        """
        paths = cfg.paths
        horley_path = project_root / paths.horley_encoding_json
        allographs_path = project_root / paths.allographs_json
        metadata_path = project_root / paths.sign_metadata_json

        horley_encoding = cls._load_json(horley_path, "Horley encoding")
        allographs = cls._load_json(allographs_path, "allograph catalog")
        sign_metadata = cls._load_json(metadata_path, "sign metadata")

        return cls(horley_encoding, allographs, sign_metadata)

    @staticmethod
    def _load_json(path: Path, label: str) -> dict[str, Any]:
        if not path.exists():
            raise FileNotFoundError(f"{label} file not found: {path}")
        with path.open("r", encoding="utf-8") as fh:
            data: dict[str, Any] = json.load(fh)
        logger.debug(
            "Loaded %s from %s (%d keys).", label, path, len(data)
        )
        return data
