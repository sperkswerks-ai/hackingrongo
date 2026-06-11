#!/usr/bin/env python3
"""
fetch_kohaumotu_dictionary.py — Scrape the Kohaumotu Rapanui-English dictionary.

Fetches kohaumotu.org/Rongorongo/Dictionary/dictionary_complete.html,
extracts every Rapa Nui headword from the <dt> elements, normalises
diacritics and glottal-stop markers to plain ASCII, and writes one
headword per line to data/lm_sources/rapanui_kohaumotu_dictionary.txt.

Multi-form entries (e.g. "á, á-á") are split on commas so each form
gets its own line.

SSL note
--------
kohaumotu.org uses a self-signed certificate.  Verification is
intentionally disabled — see the same pattern in scrape_glyphs.py.
Never reuse the SSL context created here for other hosts.

Usage
-----
    python scripts/fetch_kohaumotu_dictionary.py
    python scripts/fetch_kohaumotu_dictionary.py --output data/lm_sources/rapanui_kohaumotu_dictionary.txt
    python scripts/fetch_kohaumotu_dictionary.py --dry-run
"""
from __future__ import annotations

import argparse
import logging
import re
import ssl
import sys
import time
import unicodedata
import urllib.request
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DICT_URL = "https://kohaumotu.org/Rongorongo/Dictionary/dictionary_complete.html"

_DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "lm_sources" / "rapanui_kohaumotu_dictionary.txt"

# Characters valid in a Rapa Nui word after ASCII normalisation.
# Includes 'v', which is phonemic in Rapa Nui and accepted by the
# structural syllable validation in hackingrongo.data.phoneme_inventory.
_RAPANUI_CHARS: frozenset[str] = frozenset("aehikmngoprtuv ")

# Characters that represent glottal stops / okina in various encodings.
_OKINA_CHARS = "ʼʻʾʿ'‘’`"

# Pattern that matches one <dt> element (headwords are plain text, no inner tags).
_DT_RE = re.compile(r"<dt>(.*?)</dt>", re.DOTALL | re.IGNORECASE)

# ---------------------------------------------------------------------------
# SSL context — intentionally bypasses certificate verification for this host
# ---------------------------------------------------------------------------

def _make_ssl_ctx() -> ssl.SSLContext:
    log.warning(
        "SSL certificate verification disabled for kohaumotu.org. "
        "Self-signed certificate; intentional. "
        "Do not reuse this context for other hosts."
    )
    ctx = ssl.create_default_context()
    ctx.check_hostname = False  # nosec B501
    ctx.verify_mode = ssl.CERT_NONE  # nosec B501
    return ctx


_SSL_CTX = _make_ssl_ctx()


def _fetch(url: str) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": "hackingrongo-scraper/1.0"})
    with urllib.request.urlopen(req, context=_SSL_CTX, timeout=30) as resp:
        return resp.read().decode("utf-8", errors="replace")


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------

def _strip_diacritics(text: str) -> str:
    """Decompose Unicode then drop combining marks, yielding ASCII base letters."""
    nfd = unicodedata.normalize("NFD", text)
    return "".join(ch for ch in nfd if unicodedata.category(ch) != "Mn")


def normalise_headword(raw: str) -> str:
    """Clean a single raw headword string from a <dt> element.

    Steps
    -----
    1. HTML-decode ``&amp;``, ``&lt;``, ``&gt;``, ``&nbsp;``.
    2. Strip diacritics (á → a, ê → e, etc.) via NFD decomposition.
    3. Replace glottal-stop / okina characters with nothing.
    4. Lower-case.
    5. Collapse runs of whitespace and strip leading/trailing space.
    """
    text = raw
    text = text.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">").replace("&nbsp;", " ")
    text = _strip_diacritics(text)
    text = text.translate(str.maketrans("", "", _OKINA_CHARS))
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


def extract_headwords(html: str) -> list[str]:
    """Return a de-duplicated, sorted list of normalised Rapa Nui headwords.

    Each <dt> element may contain multiple comma-separated forms; these are
    split into individual entries.  Entries that contain no characters from
    the Rapa Nui phonological inventory after normalisation are dropped.
    """
    seen: set[str] = set()
    words: list[str] = []

    for m in _DT_RE.finditer(html):
        raw_dt = m.group(1).strip()
        normalised = normalise_headword(raw_dt)

        # Split comma-separated multi-form entries (e.g. "a, a-a")
        for part in normalised.split(","):
            word = part.strip(" -")  # also trim stray hyphens at boundaries
            if not word:
                continue
            # Keep only entries whose characters are all in the Rapa Nui set
            # (or spaces/hyphens for multi-word / compound entries).
            allowed = _RAPANUI_CHARS | frozenset("- ")
            if not all(c in allowed for c in word):
                log.debug("Skipping entry with non-Rapa Nui chars: %r", word)
                continue
            if word not in seen:
                seen.add(word)
                words.append(word)

    words.sort()
    return words


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--output", type=Path, default=_DEFAULT_OUTPUT, metavar="TXT",
        help="Destination file (default: data/lm_sources/rapanui_kohaumotu_dictionary.txt).",
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Fetch and parse but do not write the output file.",
    )
    p.add_argument(
        "--delay", type=float, default=0.0, metavar="SECS",
        help="Sleep this many seconds after fetching (polite rate-limiting, default 0).",
    )
    args = p.parse_args()

    log.info("Fetching %s …", DICT_URL)
    try:
        html = _fetch(DICT_URL)
    except Exception as exc:
        log.error("Failed to fetch dictionary page: %s", exc)
        sys.exit(1)

    if args.delay > 0:
        time.sleep(args.delay)

    headwords = extract_headwords(html)
    log.info("Extracted %d unique headwords from %d <dt> entries.", len(headwords), len(_DT_RE.findall(html)))

    if args.dry_run:
        log.info("DRY RUN — no file written. Sample (first 20):")
        for w in headwords[:20]:
            log.info("  %r", w)
        return

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(headwords) + "\n", encoding="utf-8")
    log.info("Written %d headwords → %s", len(headwords), args.output)


if __name__ == "__main__":
    main()
