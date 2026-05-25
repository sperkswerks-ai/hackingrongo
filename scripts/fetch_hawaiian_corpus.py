"""
Download the Hawaiian Corpus Project unigram frequency list and write a
plain word-per-line vocabulary file for the smoothing language model.

Source
------
dohliam (2016-). *Hawaiian Corpus Project: Data from a corpus of written
Hawaiian*. GitHub. https://github.com/dohliam/hawaiian-corpus  (CC0)

Derived from Ulukau, the Hawaiian Electronic Library (ulukau.org).
The corpus contains 10.7 million words of Hawaiian newspaper text
(modern, non-scriptural, sourced from Ulukau).

Primary output
--------------
    data/polynesian_texts/nupepa_hawaiian/haw_unigrams.txt

One normalised Hawaiian word per line (deduplicated, ~56 000 types).
Used by ``build_all_lms`` as the source for the smoothing LM — the LM
is NOT scored directly in the ensemble; it provides Kneser-Ney back-off
for n-grams absent from both the pre- and post-contact LMs.

Optional output (``--with-ngrams``)
------------------------------------
    data/polynesian_texts/nupepa_hawaiian/2grams_haw.txt
    data/polynesian_texts/nupepa_hawaiian/3grams_haw.txt
    data/polynesian_texts/nupepa_hawaiian/4grams_haw.txt

Pre-built word n-gram frequency tables (``<count> TAB <w1>,<w2>,...``
format).  When present, ``build_all_lms`` detects and ingests them via
``NGramLM.ingest_ngram_table()`` rather than the raw-text pipeline,
giving the smoothing LM genuine bigram/trigram coverage on top of the
unigram vocabulary.

Fancy upgrade path
------------------
For full running-text smoothing, Papakilo Database (papakilodatabase.com)
mirrors ~12 723 issues / 72 146 pages of the Nūpepa Hawaiian newspaper
archive.  Add scraped text to ``nupepa_hawaiian/`` as additional ``.txt``
files and re-run ``build_language_models.py`` — ``build_all_lms`` will
train on all ``.txt`` files in the directory automatically.

Usage
-----
Run from the project root:

    python scripts/fetch_hawaiian_corpus.py

Optional flags::

    --data-dir PATH       output root (default: data/polynesian_texts)
    --cache-dir PATH      cache downloaded files here to avoid re-fetching
    --with-ngrams         also download 2-gram through 4-gram freq tables
    --dry-run             print what would be written without writing

After running, rebuild the LMs::

    python scripts/build_language_models.py
"""

from __future__ import annotations

import argparse
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

_GITHUB_RAW = "https://raw.githubusercontent.com/dohliam/hawaiian-corpus/master/data"

_FREQLIST_URL = f"{_GITHUB_RAW}/freqlist_haw.txt"

# N-gram files downloaded only with --with-ngrams.
_NGRAM_FILES: list[tuple[str, str]] = [
    ("ngrams/2grams_haw.txt", "2grams_haw.txt"),
    ("ngrams/3grams_haw.txt", "3grams_haw.txt"),
    ("ngrams/4grams_haw.txt", "4grams_haw.txt"),
]

_CITATION = (
    "dohliam. Hawaiian Corpus Project. "
    "https://dohliam.github.io/corpus/haw. "
    "Derived from Ulukau Hawaiian Electronic Library. 10.7M words. CC0."
)

# Diacritic normalisation: Hawaiian macrons, okina variants → plain ASCII.
_NORMALISE_MAP: dict[str, str] = {
    "\u0101": "a", "\u0113": "e", "\u012b": "i", "\u014d": "o", "\u016b": "u",
    "\u0103": "a", "\u0115": "e", "\u012d": "i", "\u014f": "o", "\u016d": "u",
    "\u00e1": "a", "\u00e9": "e", "\u00ed": "i", "\u00f3": "o", "\u00fa": "u",
    "\u00e0": "a", "\u00e8": "e", "\u00ec": "i", "\u00f2": "o", "\u00f9": "u",
    "\u00e2": "a", "\u00ea": "e", "\u00ee": "i", "\u00f4": "o", "\u00fb": "u",
    "\u02bc": "",  # ʼ MODIFIER LETTER APOSTROPHE
    "\u02bb": "",  # ʻ MODIFIER LETTER TURNED COMMA (Hawaiian okina)
    "\u0027": "",  # plain apostrophe used as okina
    "\u2018": "",  # LEFT SINGLE QUOTATION MARK
    "\u2019": "",  # RIGHT SINGLE QUOTATION MARK
}

_VOWELS: frozenset[str] = frozenset("aeiou")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalise(word: str) -> str:
    word = unicodedata.normalize("NFC", word.lower().strip())
    return "".join(_NORMALISE_MAP.get(ch, ch) for ch in word).strip()


def _is_valid(word: str) -> bool:
    return (
        len(word) >= 2
        and any(ch in _VOWELS for ch in word)
        and not any(ch.isdigit() for ch in word)
    )


def _parse_freqlist(raw: bytes) -> list[str]:
    """Parse freqlist_haw.txt (``count TAB word``) → deduplicated normalised word list."""
    text = raw.decode("utf-8", errors="replace")
    seen: set[str] = set()
    words: list[str] = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        tab = line.find("\t")
        if tab < 0:
            continue
        norm = _normalise(line[tab + 1:])
        if norm and norm not in seen and _is_valid(norm):
            seen.add(norm)
            words.append(norm)
    logger.info("Parsed %d unique normalised word types.", len(words))
    return words


def _fetch(url: str, cache_path: Path | None) -> bytes:
    if cache_path is not None and cache_path.exists():
        logger.info("Cache hit: %s", cache_path)
        return cache_path.read_bytes()

    logger.info("Downloading %s …", url)
    req = urllib.request.Request(url, headers={"User-Agent": "hackingrongo/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:  # noqa: S310
            content: bytes = resp.read()
    except Exception as exc:
        raise RuntimeError(f"Failed to download {url}: {exc}") from exc

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_bytes(content)
        logger.info("Cached to %s", cache_path)

    return content


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Download Hawaiian Corpus Project data for the smoothing LM.",
    )
    p.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/polynesian_texts"),
        help="Output root directory (default: data/polynesian_texts).",
    )
    p.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Cache downloaded files here to avoid re-fetching.",
    )
    p.add_argument(
        "--with-ngrams",
        action="store_true",
        help=(
            "Also download 2-gram/3-gram/4-gram frequency tables. "
            "build_all_lms ingests them via NGramLM.ingest_ngram_table() "
            "for bigram/trigram smoothing coverage beyond vocabulary back-off."
        ),
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be downloaded/written without doing it.",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    out_dir: Path = args.data_dir / "nupepa_hawaiian"

    if args.dry_run:
        logger.info("[DRY RUN] Would write %s/haw_unigrams.txt", out_dir)
        if args.with_ngrams:
            for _, local_name in _NGRAM_FILES:
                logger.info("[DRY RUN] Would write %s/%s", out_dir, local_name)
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Unigram word list (primary smoothing LM source) ---
    cache_freqlist = args.cache_dir / "freqlist_haw.txt" if args.cache_dir else None
    try:
        raw = _fetch(_FREQLIST_URL, cache_freqlist)
    except RuntimeError as exc:
        logger.error("%s", exc)
        sys.exit(1)

    words = _parse_freqlist(raw)
    unigrams_path = out_dir / "haw_unigrams.txt"
    unigrams_path.write_text("\n".join(words) + "\n", encoding="utf-8")
    logger.info("Wrote %d words → %s", len(words), unigrams_path)

    # --- Optional n-gram frequency tables ---
    if args.with_ngrams:
        for github_path, local_name in _NGRAM_FILES:
            url = f"{_GITHUB_RAW}/{github_path}"
            cache_path = args.cache_dir / local_name if args.cache_dir else None
            try:
                content = _fetch(url, cache_path)
            except RuntimeError as exc:
                logger.warning("Skipping %s: %s", local_name, exc)
                continue
            out_path = out_dir / local_name
            out_path.write_bytes(content)
            logger.info("Wrote %s (%d lines)", out_path, content.count(b"\n"))

    # --- metadata.json ---
    meta_path = out_dir / "metadata.json"
    meta: dict = {}
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            meta = {}

    sources = [
        s for s in meta.get("sources", [])
        if "dohliam" not in s.get("citation", "") and "hawaiian-corpus" not in s.get("citation", "")
    ]
    sources.append({
        "file": "haw_unigrams.txt",
        "citation": _CITATION,
        "genre": "newspaper",
        "notes": (
            "Unigram frequency list used for smoothing LM backoff only. "
            "Not scored directly in ensemble."
        ),
    })
    if args.with_ngrams:
        sources.append({
            "files": [local_name for _, local_name in _NGRAM_FILES],
            "citation": _CITATION,
            "genre": "frequency_table",
            "notes": (
                "Pre-built n-gram frequency tables; ingested via "
                "NGramLM.ingest_ngram_table() for bigram/trigram smoothing coverage."
            ),
        })
    meta["sources"] = sources
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Wrote metadata → %s", meta_path)

    logger.info(
        "Done.  Run `python scripts/build_language_models.py` to rebuild the smoothing LM."
    )


if __name__ == "__main__":
    main()
