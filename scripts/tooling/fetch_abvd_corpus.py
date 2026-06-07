"""
TOOLING — one-time data preparation; not part of the reproducible analysis pipeline.

Fetch vocabulary forms from the Austronesian Basic Vocabulary Database (ABVD)
and write them into the directory structure expected by
``hackingrongo.data.rapa_nui_corpus.load_text_corpus``.

Source
------
ABVD  (CC-BY 4.0) — Greenhill, Blust & Gray (2008)
https://abvd.eva.mpg.de/austronesian/

The ABVD exposes a per-language TSV download endpoint::

    https://abvd.eva.mpg.de/austronesian/language.php?id=<ID>&action=download&type=tab

Columns: ``word_id``, ``word``, ``item``, ``annotation``, ``cognacy``, ``loan``.
The ``word`` column contains the form in the target language; ``item`` is the
English gloss.  Missing entries are ``—`` or empty and are skipped.

No authentication is required.  One HTTPS request is made per language
(or served from cache if ``--cache-dir`` is supplied).

Output
------
For each target language a text file is written to::

    data/polynesian_texts/<language>/abvd.txt

containing one lexical form per line (lower-cased, diacritics normalised to
plain ASCII vowels).  A ``metadata.json`` is created (or updated) in the
same directory so that ``load_text_corpus`` can attach source attribution.

Usage
-----
Run from the project root (no Hydra, no extra dependencies beyond stdlib):

    python scripts/fetch_abvd_corpus.py

Optional flags::

    --data-dir PATH       output root (default: data/polynesian_texts)
    --cache-dir PATH      cache downloaded TSVs here to avoid re-fetching
    --languages L [L …]   override target languages
    --dry-run             print stats without writing files
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import logging
import sys
import unicodedata
import urllib.request
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_ABVD_ENDPOINT = (
    "https://abvd.eva.mpg.de/austronesian/language.php"
    "?id={id}&action=download&type=tab"
)

# ABVD language IDs verified against abvd.eva.mpg.de (May 2026).
# Tier comments match fetch_lm_sources.sh rationale.
_LANGUAGE_CONFIG: dict[str, dict] = {
    # pre-contact East Polynesian targets
    "rapanui":    {"id": 264, "tier": "pre_contact"},
    "mangarevan": {"id": 253, "tier": "pre_contact"},
    "marquesan":  {"id": 254, "tier": "pre_contact"},
    "tuamotuan":  {"id": 246, "tier": "pre_contact"},
    "tahitian":   {"id": 261, "tier": "pre_contact"},
    # post-contact / high-volume smoothing
    "hawaiian":   {"id": 109, "tier": "post_contact"},
    "maori":      {"id": 256, "tier": "post_contact"},
    # outgroup baselines
    "samoan":     {"id": 259, "tier": "baseline"},
    "tongan":     {"id": 263, "tier": "baseline"},
}

# Zone C primary scoring languages (matches conf/config.yaml).
_DEFAULT_LANGUAGES: list[str] = ["hawaiian", "maori", "tahitian", "rapanui"]

# Diacritic → plain vowel map (Polynesian macron/breve vowels → ASCII).
_NORMALISE_MAP: dict[str, str] = {
    "ā": "a", "ē": "e", "ī": "i", "ō": "o", "ū": "u",
    "ă": "a", "ĕ": "e", "ĭ": "i", "ŏ": "o", "ŭ": "u",
    "á": "a", "é": "e", "í": "i", "ó": "o", "ú": "u",
    "à": "a", "è": "e", "ì": "i", "ò": "o", "ù": "u",
    "â": "a", "ê": "e", "î": "i", "ô": "o", "û": "u",
    # Okina / glottal stop variants — strip
    "\u02bc": "",  # ʼ MODIFIER LETTER APOSTROPHE
    "\u02bb": "",  # ʻ MODIFIER LETTER TURNED COMMA  (Hawaiian okina)
    "\u0027": "",  # plain apostrophe used as okina in some entries
    "\u2018": "",  # LEFT SINGLE QUOTATION MARK
    "\u2019": "",  # RIGHT SINGLE QUOTATION MARK
}

_VOWELS: frozenset[str] = frozenset("aeiou")

# ---------------------------------------------------------------------------
# Download + parsing
# ---------------------------------------------------------------------------


def _fetch_language_tsv(language_id: int, cache_path: Path | None) -> str:
    """Download the ABVD per-language TSV for a given numeric ID."""
    if cache_path is not None and cache_path.exists():
        logger.debug("Cache hit: %s", cache_path)
        return cache_path.read_text(encoding="utf-8")

    url = _ABVD_ENDPOINT.format(id=language_id)
    logger.info("Downloading ABVD id=%d from %s …", language_id, url)
    req = urllib.request.Request(url, headers={"User-Agent": "hackingrongo/1.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:  # noqa: S310
        content = resp.read().decode("utf-8", errors="replace")

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(content, encoding="utf-8")
        logger.debug("Cached to %s", cache_path)

    return content


def _normalise_form(form: str) -> str:
    """Lower-case and normalise diacritics to plain ASCII vowels."""
    form = unicodedata.normalize("NFC", form.lower().strip())
    return "".join(_NORMALISE_MAP.get(ch, ch) for ch in form).strip()


def _is_valid_polynesian_form(form: str) -> bool:
    """Return True if the form contains at least one vowel and no digits."""
    if len(form) < 2:
        return False
    if any(ch.isdigit() for ch in form):
        return False
    return any(ch in _VOWELS for ch in form)


# Missing-form sentinels used in ABVD TSV.
_MISSING: frozenset[str] = frozenset({"-", "—", "?", "x", "UNKNOWN", ""})


def _parse_abvd_tsv(tsv_text: str, lang_name: str) -> list[str]:
    """Parse an ABVD per-language TSV and return normalised word forms.

    The TSV columns are: ``word_id``, ``word``, ``item``, ``annotation``,
    ``cognacy``, ``loan``.  We extract ``word`` only.

    Some entries contain multiple slash-separated alternants (e.g.
    ``kai/kana``); each is treated as a separate form.
    """
    forms: list[str] = []
    skipped = 0
    try:
        reader = csv.DictReader(io.StringIO(tsv_text), delimiter="\t")
        for row in reader:
            raw = row.get("word", "").strip()
            if not raw or raw in _MISSING:
                skipped += 1
                continue
            # Split slash-separated alternants.
            for part in raw.replace(",", "/").split("/"):
                norm = _normalise_form(part)
                if _is_valid_polynesian_form(norm):
                    forms.append(norm)
                else:
                    skipped += 1
    except Exception as exc:
        logger.warning("TSV parse error for '%s': %s", lang_name, exc)

    logger.info(
        "  %s: %d forms extracted (%d skipped).", lang_name, len(forms), skipped
    )
    return forms


def _parse_abvd_tsv_with_cognacy(
    tsv_text: str, lang_name: str
) -> list[tuple[str, str]]:
    """Parse an ABVD TSV and return ``(normalised_form, cognacy_code)`` pairs.

    Rows with missing forms are skipped.  Rows with empty cognacy codes
    are returned with ``cognacy_code=''``.
    """
    result: list[tuple[str, str]] = []
    skipped = 0
    try:
        reader = csv.DictReader(io.StringIO(tsv_text), delimiter="\t")
        for row in reader:
            raw = row.get("word", "").strip()
            if not raw or raw in _MISSING:
                skipped += 1
                continue
            cognacy = row.get("cognacy", "").strip()
            for part in raw.replace(",", "/").split("/"):
                norm = _normalise_form(part)
                if _is_valid_polynesian_form(norm):
                    result.append((norm, cognacy))
                else:
                    skipped += 1
    except Exception as exc:
        logger.warning("TSV parse error (with cognacy) for '%s': %s", lang_name, exc)

    logger.info(
        "  %s: %d (form, cognacy) pairs (%d skipped).",
        lang_name, len(result), skipped,
    )
    return result


# East Polynesian languages from which cognate neighbours are drawn.
_EP_NEIGHBOUR_LANGS: list[str] = ["tahitian", "mangarevan", "marquesan", "tuamotuan"]


def _build_cognate_neighbours(
    rapanui_tsv: str,
    cache_dir: Path | None,
) -> list[str]:
    """Build a list of East Polynesian cognate forms for Rapa Nui entries.

    For each Rapa Nui ABVD entry with a non-empty cognacy code, fetches
    forms from Tahitian, Mangarevan, Marquesan, and Tuamotuan that share
    the same cognacy code.  The resulting list is intended for
    ``rapanui/abvd_cognate_neighbours.txt`` and is consumed by
    ``build_all_lms`` at half weight (``cognate_neighbour_weight: 0.5``
    in config).

    Returns
    -------
    list[str]
        Deduplicated normalised cognate forms from the four neighbours.
    """
    rn_pairs = _parse_abvd_tsv_with_cognacy(rapanui_tsv, "rapanui")
    rn_cognacy_codes: set[str] = {code for _, code in rn_pairs if code}

    if not rn_cognacy_codes:
        logger.warning(
            "No cognacy codes found in Rapa Nui TSV; skipping cognate neighbours."
        )
        return []

    logger.info(
        "Fetching EP cognate neighbours for %d Rapa Nui cognacy codes …",
        len(rn_cognacy_codes),
    )

    cognacy_to_foreign: dict[str, list[str]] = {}
    for neighbour in _EP_NEIGHBOUR_LANGS:
        n_cfg = _LANGUAGE_CONFIG[neighbour]
        n_id: int = n_cfg["id"]
        cache_path = (
            cache_dir / f"abvd_{n_id}_{neighbour}.tsv" if cache_dir else None
        )
        try:
            tsv = _fetch_language_tsv(n_id, cache_path)
        except Exception as exc:
            logger.warning(
                "Failed to fetch '%s' for cognate build: %s", neighbour, exc
            )
            continue

        for form, code in _parse_abvd_tsv_with_cognacy(tsv, neighbour):
            if code and code in rn_cognacy_codes:
                cognacy_to_foreign.setdefault(code, []).append(form)

    seen: set[str] = set()
    neighbours: list[str] = []
    for code in sorted(cognacy_to_foreign):
        for form in cognacy_to_foreign[code]:
            if form not in seen:
                seen.add(form)
                neighbours.append(form)

    logger.info(
        "Cognate neighbours: %d unique forms from %d cognacy codes.",
        len(neighbours), len(cognacy_to_foreign),
    )
    return neighbours


def _write_cognate_neighbours(
    forms: list[str],
    rapanui_dir: Path,
    dry_run: bool,
) -> None:
    """Write ``abvd_cognate_neighbours.txt`` and update ``metadata.json``."""
    if not forms:
        logger.warning("No cognate neighbour forms to write.")
        return

    out_txt = rapanui_dir / "abvd_cognate_neighbours.txt"
    out_meta = rapanui_dir / "metadata.json"

    if dry_run:
        logger.info(
            "[DRY RUN] Would write %d cognate forms → %s", len(forms), out_txt
        )
        return

    rapanui_dir.mkdir(parents=True, exist_ok=True)
    out_txt.write_text("\n".join(forms) + "\n", encoding="utf-8")
    logger.info("Wrote %d cognate neighbour forms → %s", len(forms), out_txt)

    meta: dict = {}
    if out_meta.exists():
        try:
            meta = json.loads(out_meta.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            meta = {}
    sources = [
        s for s in meta.get("sources", [])
        if s.get("file") != "abvd_cognate_neighbours.txt"
    ]
    sources.append({
        "file": "abvd_cognate_neighbours.txt",
        "citation": (
            "Greenhill, S.J., Blust, R., & Gray, R.D. (2008). "
            "ABVD cognate forms from Tahitian (id=261), Mangarevan (id=253), "
            "Marquesan (id=254), Tuamotuan (id=246); filtered to Rapa Nui "
            "cognacy codes. CC-BY 4.0."
        ),
        "genre": "wordlist",
        "weight": 0.5,
        "note": (
            "Applied at half-weight (cognate_neighbour_weight=0.5) "
            "during LM training via Bernoulli sampling in build_all_lms."
        ),
    })
    meta["sources"] = sources
    out_meta.write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    logger.info("Updated metadata → %s", out_meta)


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------


def _write_language(
    lang_name: str,
    lang_id: int,
    tier: str,
    forms: list[str],
    lang_dir: Path,
    dry_run: bool,
) -> None:
    """Write ``abvd.txt`` and update ``metadata.json`` for one language."""
    if not forms:
        logger.warning("No forms collected for '%s'; skipping.", lang_name)
        return

    out_txt = lang_dir / "abvd.txt"
    out_meta = lang_dir / "metadata.json"

    if dry_run:
        logger.info(
            "[DRY RUN] Would write %d forms to %s", len(forms), out_txt
        )
        return

    lang_dir.mkdir(parents=True, exist_ok=True)

    # Deduplicate while preserving order.
    seen: set[str] = set()
    unique_forms: list[str] = []
    for f in forms:
        if f not in seen:
            seen.add(f)
            unique_forms.append(f)

    out_txt.write_text("\n".join(unique_forms) + "\n", encoding="utf-8")
    logger.info("Wrote %d unique forms → %s", len(unique_forms), out_txt)

    # Update (or create) metadata.json.
    meta: dict = {}
    if out_meta.exists():
        try:
            meta = json.loads(out_meta.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            meta = {}

    sources: list[dict] = meta.get("sources", [])
    abvd_entry = {
        "file": "abvd.txt",
        "citation": (
            "Greenhill, S.J., Blust, R., & Gray, R.D. (2008). "
            "The Austronesian Basic Vocabulary Database: From Bioinformatics to Lexomics. "
            "Evolutionary Bioinformatics, 4:271–283. "
            "https://abvd.eva.mpg.de/austronesian/ (CC-BY 4.0)"
        ),
        "genre": "wordlist",
        "abvd_id": lang_id,
        "tier": tier,
    }
    sources = [s for s in sources if s.get("file") != "abvd.txt"]
    sources.append(abvd_entry)
    meta["sources"] = sources
    out_meta.write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    logger.info("Updated metadata → %s", out_meta)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Fetch ABVD vocabulary data for Polynesian language model training."
    )
    p.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/polynesian_texts"),
        help="Root output directory (default: data/polynesian_texts)",
    )
    p.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Cache downloaded TSV files here to avoid re-fetching.",
    )
    p.add_argument(
        "--languages",
        nargs="+",
        default=_DEFAULT_LANGUAGES,
        help=(
            f"Language names to fetch (default: {_DEFAULT_LANGUAGES}). "
            f"All available: {sorted(_LANGUAGE_CONFIG)}."
        ),
    )
    p.add_argument(
        "--with-cognates",
        action="store_true",
        help=(
            "Also build data/polynesian_texts/rapanui/abvd_cognate_neighbours.txt: "
            "East Polynesian forms (Tahitian, Mangarevan, Marquesan, Tuamotuan) "
            "that share ABVD cognacy codes with Rapa Nui entries. "
            "Neighbour TSVs are fetched/cached alongside the regular download."
        ),
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print statistics without writing any files.",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    for lang in args.languages:
        if lang not in _LANGUAGE_CONFIG:
            logger.error(
                "Unknown language '%s'. Available: %s",
                lang,
                sorted(_LANGUAGE_CONFIG),
            )
            sys.exit(1)

    logger.info("Fetching ABVD data for: %s", args.languages)

    for lang in args.languages:
        cfg = _LANGUAGE_CONFIG[lang]
        lang_id: int = cfg["id"]
        tier: str = cfg["tier"]

        cache_path = (
            args.cache_dir / f"abvd_{lang_id}_{lang}.tsv"
            if args.cache_dir is not None
            else None
        )

        try:
            tsv_text = _fetch_language_tsv(lang_id, cache_path)
        except Exception as exc:
            logger.error("Failed to fetch '%s' (id=%d): %s", lang, lang_id, exc)
            continue

        forms = _parse_abvd_tsv(tsv_text, lang)
        _write_language(
            lang, lang_id, tier, forms,
            args.data_dir / lang,
            args.dry_run,
        )

    if args.with_cognates:
        rn_cfg = _LANGUAGE_CONFIG["rapanui"]
        rn_id = rn_cfg["id"]
        rn_cache = (
            args.cache_dir / f"abvd_{rn_id}_rapanui.tsv"
            if args.cache_dir is not None
            else None
        )
        try:
            rn_tsv = _fetch_language_tsv(rn_id, rn_cache)
            cognate_forms = _build_cognate_neighbours(rn_tsv, args.cache_dir)
            _write_cognate_neighbours(
                cognate_forms, args.data_dir / "rapanui", args.dry_run
            )
        except Exception as exc:
            logger.error("Cognate neighbour build failed: %s", exc)

    logger.info(
        "Done.  Run `python scripts/build_language_models.py` to build LMs from the new corpora."
    )


if __name__ == "__main__":
    main()
