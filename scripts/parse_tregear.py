"""
Parse Tregear's *Maori-Polynesian Comparative Dictionary* (1891) and extract
per-language word forms for Polynesian language model training.

Source
------
Tregear, Edward (1891). *The Maori-Polynesian Comparative Dictionary*.
Wellington: Lyon and Blair.  Public domain.

Internet Archive scans (DjVu OCR text, freely downloadable, no auth required):
  Primary:   https://archive.org/download/maoripolynesian00treggoog/maoripolynesian00treggoog_djvu.txt
  Fallback:  https://archive.org/download/cu31924026916480/cu31924026916480_djvu.txt

Dictionary format
-----------------
Each entry follows this structure::

    HEADWORD, part-of-speech.  Maori definition text.
    Haw., hawaiian-form, optional-English-gloss.
    Tah., tahitian-form, optional-English-gloss.
    Sam., samoan-form.  Ton., tongan-form.  Ra., rarotongan-form.

*Headword* is the Māori lemma, written in ALL-CAPITALS.  Comparative forms
for other Polynesian languages are introduced by a two-or-three-letter
abbreviation followed by a full-stop (sometimes garbled by OCR).

Language abbreviations used by Tregear
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Haw.   Hawaiian          Tah.   Tahitian
    Sam.   Samoan            Ton.   Tongan
    Ra.    Rarotongan        Fut.   Futunan
    Mang.  Mangarevan        Mar.   Marquesan
    Fiji   Fijian            E.I.   Easter Island (Rapa Nui)

Output
------
For each target language a text file is written to::

    data/polynesian_texts/<language>/tregear.txt

containing one word form per line.  A ``metadata.json`` is created (or
updated) in the same directory so ``load_text_corpus`` can attach the
Tregear citation.

Usage
-----
    python scripts/parse_tregear.py

Optional flags::

    --data-dir PATH      output root (default: data/polynesian_texts)
    --cache-dir PATH     cache downloaded text here to avoid re-fetching
    --input PATH         use a locally downloaded DjVu text file instead
    --languages L [L …]  override target languages
    --dry-run            print statistics without writing any files

OCR robustness
--------------
The DjVu OCR text from Internet Archive contains common artefacts:
  • Ligatures rendered as ``ff``, ``fi``, ``fl``
  • Accented characters dropped or replaced with plain ASCII
  • Language abbreviations missing their trailing period (``Haw`` vs ``Haw.``)
  • Headwords split across lines (``A KA`` instead of ``AKA``)
  • Form and gloss on separate lines due to line-wrapping

The parser normalises all of these conservatively: it joins lines within an
entry, uses a flexible regex for language abbreviations, and strips English
gloss tokens that contain non-Polynesian characters.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
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
# Download URLs (tried in order)
# ---------------------------------------------------------------------------

_DJVU_URLS: list[str] = [
    # Google scan — usually cleaner OCR
    (
        "https://archive.org/download/maoripolynesian00treggoog"
        "/maoripolynesian00treggoog_djvu.txt"
    ),
    # Cornell ABBYY scan — fallback
    (
        "https://archive.org/download/cu31924026916480"
        "/cu31924026916480_djvu.txt"
    ),
]

# ---------------------------------------------------------------------------
# Language configuration
# ---------------------------------------------------------------------------

# Maps our config language names to the abbreviation patterns Tregear uses.
# Multiple patterns are tried (OCR may omit or garble punctuation).
_LANG_ABBREVS: dict[str, list[str]] = {
    "maori":    [],           # headword itself is the Māori form
    "hawaiian": ["Haw"],
    "tahitian": ["Tah"],
    "samoan":   ["Sam"],
    "rarotongan": ["Ra"],
    "futunan":  ["Fut"],
    "mangareva": ["Mang"],
    "marquesan": ["Mar"],
    "rapanui":  ["E.I", "E. I", "Easter Island"],
}

# Subset we actually write output for by default (matches Zone C config).
_DEFAULT_TARGET_LANGUAGES: list[str] = ["maori", "hawaiian", "tahitian", "rapanui"]

# ---------------------------------------------------------------------------
# Normalisation helpers
# ---------------------------------------------------------------------------

# Diacritic → ASCII vowel (same mapping as in fetch_abvd_corpus.py).
_NORMALISE_MAP: dict[str, str] = {
    "ā": "a", "ē": "e", "ī": "i", "ō": "o", "ū": "u",
    "ă": "a", "ĕ": "e", "ĭ": "i", "ŏ": "o", "ŭ": "u",
    "á": "a", "é": "e", "í": "i", "ó": "o", "ú": "u",
    "à": "a", "è": "e", "ì": "i", "ò": "o", "ù": "u",
    "â": "a", "ê": "e", "î": "i", "ô": "o", "û": "u",
    # Okina / glottal stop variants — strip
    "\u02bc": "", "\u02bb": "", "\u0027": "",
    "\u2018": "", "\u2019": "",
}

_VOWELS: frozenset[str] = frozenset("aeiou")

# Per-language valid character inventories (after normalisation to plain ASCII).
# Only characters that appear in the language's phonological system are listed.
# The character 'g' is included wherever 'ng' is a valid digraph; standalone
# 'g' is separately gated by _LANGS_NG_ONLY below.
_LANG_VALID_CHARS: dict[str, frozenset[str]] = {
    # Rapa Nui: h k m n(g) p r t + vowels.  Excludes b c d f j l q s v w x y z.
    "rapanui":    frozenset("aehikmngoprtu"),
    # Māori: Rapa Nui inventory plus w (and f as wh- transcription variant).
    "maori":      frozenset("aefhikmngoprtuw"),
    # Hawaiian: h k l m n p w + vowels.  No r, no standalone g.
    "hawaiian":   frozenset("aehiklmnopuw"),
    # Tahitian: f h m n p r t v + vowels.  No k, no g.
    "tahitian":   frozenset("aefhimnoprtuv"),
    # Samoan: f g l m n p s t v + vowels.
    "samoan":     frozenset("aefgilmnostupv"),
    # Rarotongan: k m n(g) p r t + vowels.
    "rarotongan": frozenset("aeikmngoprtu"),
    # Futunan: f k l m n(g) p s t v + vowels.
    "futunan":    frozenset("aefiklmngostuv"),
    # Mangarevan: g k m n(g) p r t + vowels.
    "mangareva":  frozenset("aegikmngoprtU".lower()),
    # Marquesan: f h k m n(g) p t + vowels.
    "marquesan":  frozenset("aefhikmngoptu"),
}

# Languages where 'g' only appears as part of the 'ng' digraph.
_LANGS_NG_ONLY: frozenset[str] = frozenset(
    {"rapanui", "maori", "rarotongan", "mangareva", "marquesan", "futunan"}
)


def _normalise(word: str) -> str:
    """Lower-case, NFC-normalise, replace diacritics with ASCII equivalents."""
    word = unicodedata.normalize("NFC", word.lower().strip())
    return "".join(_NORMALISE_MAP.get(ch, ch) for ch in word)


def _has_invalid_consonant_cluster(word: str) -> bool:
    """Return True if *word* contains adjacent consonants other than the digraph 'ng'."""
    i = 0
    while i < len(word) - 1:
        c1, c2 = word[i], word[i + 1]
        if c1 not in _VOWELS and c2 not in _VOWELS:
            if c1 == "n" and c2 == "g":
                i += 2  # ng is a single phoneme; skip both chars
            else:
                return True
        else:
            i += 1
    return False


def _is_valid_polynesian_form(word: str, language: str) -> bool:
    """Return True iff *word* passes strict phonological filters for *language*.

    Enforces:
    1. Minimum length 2.
    2. Must begin with a letter (rejects parenthetical OCR fragments).
    3. Must end with a vowel (all target Polynesian languages are V-final).
    4. Every character must be in the language-specific phoneme inventory.
    5. In languages where 'g' only occurs in 'ng', standalone 'g' is rejected.
    6. No consonant clusters other than the digraph 'ng'.
    """
    if len(word) < 2:
        return False
    if not word[0].isalpha():
        return False
    if word[-1] not in _VOWELS:
        return False
    valid = _LANG_VALID_CHARS.get(language, frozenset())
    if not all(ch in valid for ch in word):
        return False
    if language in _LANGS_NG_ONLY:
        for i, ch in enumerate(word):
            if ch == "g" and (i == 0 or word[i - 1] != "n"):
                return False
    if _has_invalid_consonant_cluster(word):
        return False
    return True


# ---------------------------------------------------------------------------
# Entry-level parsing
# ---------------------------------------------------------------------------

# A headword line: word(s) in ALL-CAPS (2+ chars), possibly comma or period.
# Allows for OCR spacing artefacts within the headword (e.g. "A KA" → joined).
_HEADWORD_RE = re.compile(
    r"^([A-Z][A-Z\s\-]{1,30}?)\s*,\s*(?:s\.|v\.|adj\.|adv\.|n\.|p\.|part\.|conj\.|prep\.)",
    re.MULTILINE,
)

# Language abbreviation followed by a word form.
# Handles: "Haw., form," | "Haw. form," | "Haw,form" | "Haw form"
def _build_lang_pattern(abbrevs: list[str]) -> re.Pattern[str] | None:
    if not abbrevs:
        return None
    alts = "|".join(re.escape(a) for a in abbrevs)
    return re.compile(
        # Abbreviation (with optional dot), optional comma/space
        rf"(?:{alts})"
        r"\.?,?\s+"
        # Capture the form: letters (including Polynesian chars), hyphen, apostrophe
        r"([a-zA-ZāēīōūÀ-ÿʻʼ'\u02bc\u02bb\-]{2,})",
        re.IGNORECASE,
    )


_COMPILED_LANG_PATTERNS: dict[str, re.Pattern[str] | None] = {
    lang: _build_lang_pattern(abbrevs)
    for lang, abbrevs in _LANG_ABBREVS.items()
}

# Pattern to detect a headword (used for entry splitting).
_CAPS_WORD_LINE = re.compile(r"^\s*[A-Z][A-Z\-\s]{1,}(?:,|\.|$)")


def _split_entries(text: str) -> list[str]:
    """Split the full DjVu text into per-entry blocks.

    Entries are separated by one of:
    1. A blank line followed by an ALL-CAPS headword.
    2. A form-feed character (\\f) used as a page separator.

    Returns a list of raw entry strings (not yet cleaned).
    """
    # Normalise line endings and form feeds.
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\f", "\n\n")

    entries: list[str] = []
    current_lines: list[str] = []
    prev_blank = False

    for line in text.splitlines():
        stripped = line.strip()

        is_new_headword = (
            prev_blank
            and bool(_CAPS_WORD_LINE.match(stripped))
            and len(stripped) >= 3
        )

        if is_new_headword and current_lines:
            entries.append("\n".join(current_lines))
            current_lines = []

        current_lines.append(line)
        prev_blank = stripped == ""

    if current_lines:
        entries.append("\n".join(current_lines))

    return entries


def _extract_headword(entry: str) -> str | None:
    """Extract the Māori headword from a raw entry block."""
    m = _HEADWORD_RE.search(entry)
    if m:
        # Collapse any OCR-induced internal spaces (e.g. "A KA" → "AKA").
        raw = m.group(1).strip()
        return re.sub(r"\s+", "", raw).lower()
    # Fallback: look for the first ALL-CAPS word on the first non-blank line.
    for line in entry.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        m2 = re.match(r"^([A-Z][A-Z\-]{1,})\s*[,.]", stripped)
        if m2:
            return m2.group(1).lower()
        break
    return None


def _extract_lang_forms(
    entry: str, pattern: re.Pattern[str], language: str
) -> tuple[list[str], int]:
    """Extract all word forms for a language from a single entry.

    Returns ``(accepted_forms, n_rejected)`` so callers can accumulate
    per-language rejection counts for final logging.
    """
    collapsed = " ".join(entry.split())
    raw_forms: list[str] = pattern.findall(collapsed)
    results: list[str] = []
    n_rejected = 0
    for raw in raw_forms:
        norm = _normalise(raw)
        if _is_valid_polynesian_form(norm, language):
            results.append(norm)
        else:
            n_rejected += 1
    return results, n_rejected


# ---------------------------------------------------------------------------
# Main parsing pipeline
# ---------------------------------------------------------------------------


def parse_tregear(text: str, target_languages: list[str]) -> dict[str, list[str]]:
    """Parse the full DjVu OCR text and return per-language word form lists.

    Parameters
    ----------
    text : str
        Raw DjVu OCR text from Internet Archive.
    target_languages : list[str]
        Language names to extract (must be keys in ``_LANG_ABBREVS``).

    Returns
    -------
    dict[str, list[str]]
        ``{language: [form, …]}`` — forms may include duplicates; call
        :func:`_deduplicate` before writing.
    """
    forms: dict[str, list[str]] = {lang: [] for lang in target_languages}
    rejections: dict[str, int] = {lang: 0 for lang in target_languages}

    logger.info("Splitting text into entries …")
    entries = _split_entries(text)
    logger.info("Found %d candidate entries.", len(entries))

    n_parsed = 0
    for entry in entries:
        if len(entry.strip()) < 5:
            continue

        for lang in target_languages:
            if lang == "maori":
                hw = _extract_headword(entry)
                if hw:
                    if _is_valid_polynesian_form(hw, "maori"):
                        forms["maori"].append(hw)
                    else:
                        rejections["maori"] += 1
            else:
                pattern = _COMPILED_LANG_PATTERNS.get(lang)
                if pattern is None:
                    continue
                lang_forms, n_rej = _extract_lang_forms(entry, pattern, lang)
                forms[lang].extend(lang_forms)
                rejections[lang] += n_rej

        n_parsed += 1

    logger.info("Parsed %d non-trivial entries.", n_parsed)
    for lang in target_languages:
        n_accepted = len(forms[lang])
        n_rejected = rejections[lang]
        total = n_accepted + n_rejected
        pct = 100.0 * n_accepted / total if total > 0 else 0.0
        logger.info(
            "  %s: %d accepted, %d rejected (%.0f%% pass rate).",
            lang, n_accepted, n_rejected, pct,
        )

    return forms


def _deduplicate(forms: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for f in forms:
        if f not in seen:
            seen.add(f)
            result.append(f)
    return result


# ---------------------------------------------------------------------------
# Download helper
# ---------------------------------------------------------------------------


def _download_text(cache_path: Path | None) -> str:
    if cache_path is not None and cache_path.exists():
        logger.info("Loading cached text from %s", cache_path)
        return cache_path.read_text(encoding="utf-8", errors="replace")

    last_error: Exception | None = None
    for url in _DJVU_URLS:
        logger.info("Downloading %s …", url)
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "hackingrongo/1.0"})
            with urllib.request.urlopen(req, timeout=60) as resp:  # noqa: S310
                content = resp.read().decode("utf-8", errors="replace")
            if len(content) < 10_000:
                logger.warning("Downloaded file looks too small (%d bytes); skipping.", len(content))
                continue
            if cache_path is not None:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                cache_path.write_text(content, encoding="utf-8")
                logger.info("Cached to %s", cache_path)
            return content
        except Exception as exc:
            logger.warning("Failed to fetch %s: %s", url, exc)
            last_error = exc

    msg = "All download URLs failed."
    if last_error is not None:
        msg += f"  Last error: {last_error}"
    msg += (
        "\nDownload the DjVu text manually from Internet Archive and supply "
        "it with --input <path>."
    )
    raise RuntimeError(msg)


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------

_TREGEAR_CITATION = (
    "Tregear, Edward (1891). The Maori-Polynesian Comparative Dictionary. "
    "Wellington: Lyon and Blair. Public domain. "
    "https://archive.org/details/maoripolynesian00treggoog"
)


def _write_language(
    lang: str,
    forms: list[str],
    lang_dir: Path,
    dry_run: bool,
) -> None:
    unique = _deduplicate(forms)
    if not unique:
        logger.warning("No forms for '%s'; skipping.", lang)
        return

    out_txt = lang_dir / "tregear.txt"
    out_meta = lang_dir / "metadata.json"

    if dry_run:
        logger.info("[DRY RUN] Would write %d forms → %s", len(unique), out_txt)
        return

    lang_dir.mkdir(parents=True, exist_ok=True)
    out_txt.write_text("\n".join(unique) + "\n", encoding="utf-8")
    logger.info("Wrote %d unique forms → %s", len(unique), out_txt)

    meta: dict = {}
    if out_meta.exists():
        try:
            meta = json.loads(out_meta.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            meta = {}

    sources = [s for s in meta.get("sources", []) if s.get("file") != "tregear.txt"]
    sources.append(
        {
            "file": "tregear.txt",
            "citation": _TREGEAR_CITATION,
            "genre": "wordlist",
            "notes": (
                "Extracted from DjVu OCR text; Māori headwords + comparative forms. "
                "OCR noise may be present."
            ),
        }
    )
    meta["sources"] = sources
    out_meta.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Updated metadata → %s", out_meta)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Parse Tregear (1891) Maori-Polynesian Comparative Dictionary "
            "and extract per-language word forms for LM training."
        )
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
        help="Cache downloaded DjVu text here to avoid re-fetching.",
    )
    p.add_argument(
        "--input",
        type=Path,
        default=None,
        dest="input_path",
        help="Path to a locally downloaded DjVu text file (skips download).",
    )
    p.add_argument(
        "--languages",
        nargs="+",
        default=_DEFAULT_TARGET_LANGUAGES,
        help=(
            f"Languages to extract (default: {_DEFAULT_TARGET_LANGUAGES}). "
            f"Available: {sorted(_LANG_ABBREVS)}."
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
        if lang not in _LANG_ABBREVS:
            logger.error(
                "Unknown language '%s'. Available: %s", lang, sorted(_LANG_ABBREVS)
            )
            sys.exit(1)

    # Acquire text.
    if args.input_path is not None:
        if not args.input_path.exists():
            logger.error("Input file not found: %s", args.input_path)
            sys.exit(1)
        logger.info("Reading local file: %s", args.input_path)
        text = args.input_path.read_text(encoding="utf-8", errors="replace")
    else:
        cache = (
            args.cache_dir / "tregear_djvu.txt"
            if args.cache_dir is not None
            else None
        )
        text = _download_text(cache)

    logger.info("Text length: %d chars.", len(text))

    # Parse.
    forms_by_lang = parse_tregear(text, args.languages)

    # Write outputs.
    for lang, forms in forms_by_lang.items():
        _write_language(lang, forms, args.data_dir / lang, args.dry_run)

    logger.info(
        "Done.  Run `python scripts/build_language_models.py` to rebuild LMs."
    )


if __name__ == "__main__":
    main()
