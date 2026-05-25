"""
hackingrongo.data.corpus_xml
=======================

Converter: kohaumotu.org XML tablet files → per-tablet corpus JSON.

This module parses the XML files downloaded from::

    http://kohaumotu.org/Rongorongo/xml/<TABLET>.xml

and writes them to the flat per-tablet JSON format expected by
``hackingrongo.data.corpus.load_corpus()``.

Source
------
The XML files are Philip Spaelti's encoding of Thomas Barthel's
numerical transliteration of the rongorongo corpus, hosted on
kohaumotu.org (a copy of the defunct CEIPP/rongorongo.org site).

Attribution
-----------
Barthel, T.S. (1958). *Grundlagen zur Entzifferung der
Osterinselschrift*. Hamburg: Cram, de Gruyter.
Kohaumotu.org maintainer: Philip Spaelti (atua@kohaumotu.org).

XML Schema (relevant elements)
-------------------------------

.. code-block:: xml

    <corpus>
      <tablet>
        <tablet-code>G</tablet-code>
        <tablet-name>Small Santiago</tablet-name>
        <side>
          <side-code>r</side-code>   <!-- r = recto, v = verso -->
          <side-name>Recto</side-name>
          <line>
            <line-code>01</line-code>
            <line-num>1</line-num>
            <glyph>
              <loc>
                <seg-num>01</seg-num>   <!-- segment within line -->
                <glyph-num>01</glyph-num>
              </loc>
              <code>
                <ceipp>001</ceipp>   <!-- Barthel code, diacritics preserved -->
              </code>
              <link>-</link>   <!-- - juxtaposed | . linked | : stacked -->
            </glyph>
            ...
          </line>
        </side>
      </tablet>
    </corpus>

Diacritics preserved in ``<ceipp>`` values
-------------------------------------------
* ``!``  — glyph is inverted (boustrophedon reversal)
* ``a``–``z`` suffix — variant forms (e.g. ``034c``)
* ``_``  — uncertain / illegible (mapped to ``barthel_code="?"`` in output)

Glyphs with empty ``<ceipp>`` are separators and are skipped entirely.

Output JSON Format
------------------

``data/corpus/<TABLET_ID>.json``::

    {
        "tablet_id": "G",
        "source": "kohaumotu.org",
        "glyphs": [
            {
                "position":     1,
                "barthel_code": "001",
                "side":         "r",
                "line":         "01",
                "segment":      "01",
                "glyph_num":    "01",
                "link":         "-"
            },
            ...
        ]
    }

``position`` is a 1-based global counter across the whole tablet in
reading order (recto lines ascending, then verso lines ascending).

``data/metadata/tablets.json``::

    {
        "A": {
            "name": "Tahua",
            "radiocarbon_date_min": 1650,
            "radiocarbon_date_max": 1900,
            ...
        },
        ...
    }

Public API
----------
``parse_tablet_xml(xml_path)``
    Parse one XML file; return ``(tablet_id, tablet_name, glyphs)``.

``convert_xml_corpus(xml_dir, corpus_out_dir, metadata_out_path)``
    Convert all XML files in ``xml_dir`` and write JSON output.

``build_tablets_json(metadata_out_path, overwrite)``
    Write ``tablets.json`` from the embedded literature values.
"""

from __future__ import annotations

import json
import logging
import xml.etree.ElementTree as ET
from pathlib import Path

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tablet metadata from literature
# ---------------------------------------------------------------------------
# Radiocarbon dates are necessarily approximate; sources are:
# Orliac, C. (2005). The woody plants of the rongorongo tablets.
#   Rapa Nui Journal 19(1):61-66.
# Bahn, P. & Flenley, J. (1992). Easter Island, Earth Island. Thames & Hudson.
# Fischer, S.R. (1997). RongoRongo. Oxford: Clarendon Press.
# Where direct radiocarbon dates are unavailable, conservative collection-
# based bounds (1600-1900) are used.

_TABLET_METADATA: dict[str, dict] = {
    "A": {
        "name": "Tahua",
        "material": "Fraxinus excelsior",
        "institution": "Musée de l'Homme, Paris",
        "radiocarbon_date_min": 1650,
        "radiocarbon_date_max": 1870,
        "condition": "Fine",
        "lines_recto": 8,
        "lines_verso": 8,
        "sign_count": 1825,
        "notes": "Largest tablet; dates from Orliac (2005).",
    },
    "B": {
        "name": "Aruku-Kurenga",
        "material": "Thespesia populnea",
        "institution": "Musée de l'Homme, Paris",
        "radiocarbon_date_min": 1650,
        "radiocarbon_date_max": 1870,
        "condition": "Fine",
        "lines_recto": 10,
        "lines_verso": 12,
        "sign_count": 1135,
        "notes": "",
    },
    "C": {
        "name": "Mamari",
        "material": "Thespesia populnea",
        "institution": "Congregation of the Sacred Hearts, Rome",
        "radiocarbon_date_min": 1650,
        "radiocarbon_date_max": 1870,
        "condition": "Fine",
        "lines_recto": 14,
        "lines_verso": 14,
        "sign_count": 1000,
        "notes": "Contains the lunar calendar.",
    },
    "D": {
        "name": "Échancrée",
        "material": "Podocarpus sp.",
        "institution": "Musée de l'Homme, Paris",
        "radiocarbon_date_min": 1493,
        "radiocarbon_date_max": 1509,
        "condition": "Good",
        "lines_recto": 7,
        "lines_verso": 6,
        "sign_count": 270,
        "notes": "Pre-contact anchor; HPD 95% 1493–1509 CE (Ferrara et al. 2024).",
    },
    "E": {
        "name": "Keiti",
        "material": "unidentified",
        "institution": "Musée de l'Homme, Paris",
        "radiocarbon_date_min": 1650,
        "radiocarbon_date_max": 1870,
        "condition": "Destroyed",
        "lines_recto": 9,
        "lines_verso": 8,
        "sign_count": 822,
        "notes": "Destroyed in 1960s; corpus from Barthel tracings.",
    },
    "F": {
        "name": "Stephen-Chauvet Fragment",
        "material": "unidentified",
        "institution": "Musée de l'Homme, Paris",
        "radiocarbon_date_min": 1600,
        "radiocarbon_date_max": 1900,
        "condition": "Good",
        "lines_recto": 6,
        "lines_verso": 4,
        "sign_count": 51,
        "notes": "",
    },
    "G": {
        "name": "Small Santiago",
        "material": "Thespesia populnea",
        "institution": "Museo Nacional de Historia Natural, Santiago",
        "radiocarbon_date_min": 1650,
        "radiocarbon_date_max": 1870,
        "condition": "Fine",
        "lines_recto": 8,
        "lines_verso": 8,
        "sign_count": 720,
        "notes": "Parallel passages with K.",
    },
    "H": {
        "name": "Great Santiago",
        "material": "Thespesia populnea",
        "institution": "Museo Nacional de Historia Natural, Santiago",
        "radiocarbon_date_min": 1650,
        "radiocarbon_date_max": 1870,
        "condition": "Good",
        "lines_recto": 12,
        "lines_verso": 12,
        "sign_count": 1580,
        "notes": "Major parallel passages with P and Q.",
    },
    "I": {
        "name": "Santiago Staff",
        "material": "unidentified",
        "institution": "Museo Nacional de Historia Natural, Santiago",
        "radiocarbon_date_min": 1650,
        "radiocarbon_date_max": 1870,
        "condition": "Good",
        "lines_recto": 14,
        "lines_verso": 0,
        "sign_count": 2920,
        "notes": "Largest sign count; Fischer's procreation chant hypothesis.",
    },
    "J": {
        "name": "Reimiro 1",
        "material": "unidentified",
        "institution": "Musée de l'Homme, Paris",
        "radiocarbon_date_min": 1600,
        "radiocarbon_date_max": 1900,
        "condition": "Good",
        "lines_recto": 1,
        "lines_verso": 0,
        "sign_count": 2,
        "notes": "Pectoral ornament; minimal inscription.",
    },
    "K": {
        "name": "Small London",
        "material": "Thespesia populnea",
        "institution": "British Museum, London",
        "radiocarbon_date_min": 1650,
        "radiocarbon_date_max": 1870,
        "condition": "Good",
        "lines_recto": 5,
        "lines_verso": 5,
        "sign_count": 163,
        "notes": "Parallel passages with G.",
    },
    "L": {
        "name": "Reimiro 2",
        "material": "unidentified",
        "institution": "Museo Nacional de Historia Natural, Santiago",
        "radiocarbon_date_min": 1600,
        "radiocarbon_date_max": 1900,
        "condition": "Fine",
        "lines_recto": 1,
        "lines_verso": 0,
        "sign_count": 44,
        "notes": "",
    },
    "M": {
        "name": "Great Vienna",
        "material": "Thespesia populnea",
        "institution": "Museum für Völkerkunde, Vienna",
        "radiocarbon_date_min": 1600,
        "radiocarbon_date_max": 1900,
        "condition": "Poor",
        "lines_recto": 9,
        "lines_verso": 0,
        "sign_count": 54,
        "notes": "",
    },
    "N": {
        "name": "Small Vienna",
        "material": "Podocarpus sp.",
        "institution": "Museum für Völkerkunde, Vienna",
        "radiocarbon_date_min": 1600,
        "radiocarbon_date_max": 1900,
        "condition": "Good",
        "lines_recto": 5,
        "lines_verso": 5,
        "sign_count": 172,
        "notes": "",
    },
    "O": {
        "name": "Boomerang",
        "material": "unidentified",
        "institution": "Museum für Völkerkunde, Vienna",
        "radiocarbon_date_min": 1600,
        "radiocarbon_date_max": 1900,
        "condition": "Poor",
        "lines_recto": 7,
        "lines_verso": 0,
        "sign_count": 90,
        "notes": "",
    },
    "P": {
        "name": "Great St. Petersburg",
        "material": "Podocarpus sp.",
        "institution": "Museum of Anthropology and Ethnography, St. Petersburg",
        "radiocarbon_date_min": 1650,
        "radiocarbon_date_max": 1870,
        "condition": "Fine",
        "lines_recto": 11,
        "lines_verso": 11,
        "sign_count": 1163,
        "notes": "Major parallel passages with H and Q.",
    },
    "Q": {
        "name": "Small St. Petersburg",
        "material": "Thespesia populnea",
        "institution": "Museum of Anthropology and Ethnography, St. Petersburg",
        "radiocarbon_date_min": 1650,
        "radiocarbon_date_max": 1870,
        "condition": "Good",
        "lines_recto": 9,
        "lines_verso": 9,
        "sign_count": 718,
        "notes": "Parallel passages with H and P.",
    },
    "R": {
        "name": "Atua-Mata-Riri",
        "material": "unidentified",
        "institution": "Musée de l'Homme, Paris",
        "radiocarbon_date_min": 1600,
        "radiocarbon_date_max": 1900,
        "condition": "Good",
        "lines_recto": 8,
        "lines_verso": 8,
        "sign_count": 357,
        "notes": "",
    },
    "S": {
        "name": "Great Washington",
        "material": "Podocarpus sp.",
        "institution": "Smithsonian Institution, Washington D.C.",
        "radiocarbon_date_min": 1650,
        "radiocarbon_date_max": 1870,
        "condition": "Good",
        "lines_recto": 8,
        "lines_verso": 8,
        "sign_count": 600,
        "notes": "",
    },
    "T": {
        "name": "Honolulu 1",
        "material": "unidentified",
        "institution": "Bishop Museum, Honolulu [#3629]",
        "radiocarbon_date_min": 1600,
        "radiocarbon_date_max": 1900,
        "condition": "Poor",
        "lines_recto": 11,
        "lines_verso": 0,
        "sign_count": 120,
        "notes": "",
    },
    "U": {
        "name": "Honolulu 2",
        "material": "unidentified",
        "institution": "Bishop Museum, Honolulu [#3623]",
        "radiocarbon_date_min": 1600,
        "radiocarbon_date_max": 1900,
        "condition": "Poor",
        "lines_recto": 4,
        "lines_verso": 0,
        "sign_count": 27,
        "notes": "",
    },
    "V": {
        "name": "Honolulu 3",
        "material": "unidentified",
        "institution": "Bishop Museum, Honolulu [#3622]",
        "radiocarbon_date_min": 1600,
        "radiocarbon_date_max": 1900,
        "condition": "Poor",
        "lines_recto": 2,
        "lines_verso": 0,
        "sign_count": 22,
        "notes": "",
    },
    "W": {
        "name": "Honolulu 4",
        "material": "unidentified",
        "institution": "Bishop Museum, Honolulu [#445]",
        "radiocarbon_date_min": 1600,
        "radiocarbon_date_max": 1900,
        "condition": "Splinter",
        "lines_recto": 3,
        "lines_verso": 0,
        "sign_count": 8,
        "notes": "",
    },
    "X": {
        "name": "Tangata Manu",
        "material": "Toromiro",
        "institution": "Musée de l'Homme, Paris",
        "radiocarbon_date_min": 1600,
        "radiocarbon_date_max": 1900,
        "condition": "Fine",
        "lines_recto": 10,
        "lines_verso": 0,
        "sign_count": 37,
        "notes": "Bird-man motif tablet.",
    },
    "Y": {
        "name": "Snuff Box",
        "material": "unidentified",
        "institution": "Musée du Quai Branly, Paris",
        "radiocarbon_date_min": 1600,
        "radiocarbon_date_max": 1900,
        "condition": "Fine",
        "lines_recto": 3,
        "lines_verso": 2,
        "sign_count": 85,
        "notes": "Three-sided incised snuff box.",
    },
}


# ---------------------------------------------------------------------------
# XML parsing
# ---------------------------------------------------------------------------


def parse_tablet_xml(
    xml_path: Path,
) -> tuple[str, str, list[dict]]:
    """Parse one kohaumotu.org XML tablet file.

    Parameters
    ----------
    xml_path : Path
        Path to a ``<TABLET>.xml`` file as downloaded from
        ``http://kohaumotu.org/Rongorongo/xml/``.

    Returns
    -------
    tuple[str, str, list[dict]]
        ``(tablet_id, tablet_name, glyphs)`` where each element of
        ``glyphs`` is a dict with keys
        ``position, barthel_code, side, line, segment, glyph_num, link``.

    Raises
    ------
    FileNotFoundError
        If the XML file does not exist.
    ET.ParseError
        If the XML is malformed.
    """
    if not xml_path.exists():
        raise FileNotFoundError(f"XML corpus file not found: {xml_path}")

    tree = ET.parse(xml_path)
    root = tree.getroot()

    tablet_el = root.find("tablet")
    if tablet_el is None:
        raise ValueError(f"No <tablet> element found in {xml_path.name}")

    tablet_id = (tablet_el.findtext("tablet-code") or xml_path.stem).strip()
    tablet_name = (tablet_el.findtext("tablet-name") or "").strip()

    glyphs: list[dict] = []
    position = 0  # global 1-based counter

    for side_el in tablet_el.findall("side"):
        side_code = (side_el.findtext("side-code") or "").strip()
        for line_el in side_el.findall("line"):
            line_code = (line_el.findtext("line-code") or "").strip()
            for glyph_el in line_el.findall("glyph"):
                ceipp = (glyph_el.findtext("code/ceipp") or "").strip()
                # Skip separators (empty ceipp) entirely
                if not ceipp:
                    continue

                link = (glyph_el.findtext("link") or "").strip()
                seg_num = (glyph_el.findtext("loc/seg-num") or "").strip()
                glyph_num = (glyph_el.findtext("loc/glyph-num") or "").strip()

                # Normalize illegible marker to "?"
                barthel_code = ceipp if ceipp != "_" else "?"

                position += 1
                glyphs.append(
                    {
                        "position": position,
                        "barthel_code": barthel_code,
                        "side": side_code,
                        "line": line_code,
                        "segment": seg_num,
                        "glyph_num": glyph_num,
                        "link": link,
                    }
                )

    logger.debug(
        "Parsed tablet %s (%s): %d glyphs.", tablet_id, tablet_name, len(glyphs)
    )
    return tablet_id, tablet_name, glyphs


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def convert_xml_corpus(
    xml_dir: Path,
    corpus_out_dir: Path,
    overwrite: bool = False,
) -> dict[str, int]:
    """Convert all kohaumotu XML files in ``xml_dir`` to per-tablet JSON.

    Parameters
    ----------
    xml_dir : Path
        Directory containing ``A.xml``, ``B.xml``, … downloaded from
        kohaumotu.org.
    corpus_out_dir : Path
        Output directory for per-tablet JSON files
        (``data/corpus/`` in a standard layout).
        Created automatically if it does not exist.
    overwrite : bool
        If ``False`` (default), skip tablets whose JSON file already
        exists.

    Returns
    -------
    dict[str, int]
        Mapping of ``tablet_id → glyph_count`` for every tablet
        successfully converted.

    Raises
    ------
    FileNotFoundError
        If ``xml_dir`` does not exist.
    """
    if not xml_dir.is_dir():
        raise FileNotFoundError(f"XML directory not found: {xml_dir}")

    corpus_out_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, int] = {}

    for xml_path in sorted(xml_dir.glob("*.xml")):
        tablet_id = xml_path.stem.upper()
        out_path = corpus_out_dir / f"{tablet_id}.json"

        if out_path.exists() and not overwrite:
            logger.info("Skipping %s — output already exists.", tablet_id)
            continue

        try:
            tid, tname, glyphs = parse_tablet_xml(xml_path)
        except Exception as exc:
            logger.warning("Failed to parse %s: %s", xml_path.name, exc)
            continue

        payload = {
            "tablet_id": tid,
            "tablet_name": tname,
            "source": "kohaumotu.org (Philip Spaelti, encoding of Barthel 1958)",
            "source_url": f"http://kohaumotu.org/Rongorongo/xml/{tid}.xml",
            "glyphs": glyphs,
        }
        out_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        results[tid] = len(glyphs)
        logger.info(
            "Converted %s (%s): %d glyphs → %s",
            tid,
            tname,
            len(glyphs),
            out_path.name,
        )

    return results


def build_tablets_json(
    metadata_out_path: Path,
    overwrite: bool = False,
) -> None:
    """Write ``tablets.json`` from embedded literature values.

    Parameters
    ----------
    metadata_out_path : Path
        Destination path, e.g. ``data/metadata/tablets.json``.
        Parent directories are created if needed.
    overwrite : bool
        If ``False`` (default), do not overwrite an existing file.

    Notes
    -----
    The ``radiocarbon_date_min`` / ``radiocarbon_date_max`` values are
    conservative estimates from Orliac (2005) and Fischer (1997).
    Replace with direct measurements when available.
    """
    if metadata_out_path.exists() and not overwrite:
        logger.info(
            "tablets.json already exists at %s; skipping (pass overwrite=True to replace).",
            metadata_out_path,
        )
        return

    metadata_out_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_out_path.write_text(
        json.dumps(_TABLET_METADATA, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    logger.info("Wrote tablets.json (%d entries).", len(_TABLET_METADATA))
