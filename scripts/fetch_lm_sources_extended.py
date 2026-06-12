#!/usr/bin/env python3
"""
fetch_lm_sources_extended.py — download & process additional Rapa Nui LM sources.

Unifies the auxiliary language-model source fetchers under one entry point and
writes every output into ``data/lm_sources/`` (which
``build_language_models.py`` ingests automatically — see that script).

Sources
-------
``kohaumotu``
    Kohau Motu Rongorongo dictionary headwords, scraped from
    kohaumotu.org/Rongorongo/Dictionary/dictionary_complete.html.  Reuses the
    parser in :mod:`scripts.fetch_kohaumotu_dictionary`.
    → ``data/lm_sources/rapanui_kohaumotu.txt``

``asjp``
    ASJP Rapa Nui 40-item wordlist from asjp.clld.org.  NOTE: ASJP records
    forms in ASJPcode (a reduced phonetic alphabet), not Rapa Nui orthography,
    so most forms are dropped by the phonotactic filter.  Best-effort, low yield.
    → ``data/lm_sources/rapanui_asjp.txt``

``kieviet``
    Rapa Nui example sentences from Kieviet (2017), *A Grammar of Rapa Nui*
    (Language Science Press), via the OAPEN pre-rendered plain-text rendition.
    Lines are classified as Rapa Nui by orthography + phonotactics, giving
    running text (the object-language lines of the interlinear examples), not
    just wordlists.
    → ``data/lm_sources/rapanui_kieviet_examples.txt``

After fetching, deduplicates across every ``data/lm_sources/*.txt`` plus the
existing ABVD wordlist (if found), and logs per-source and merged vocabulary
sizes.  A machine-readable summary is written to
``data/lm_sources/_vocab_report.json`` (the leading underscore keeps it out of
the LM-source glob).

Usage
-----
    python scripts/fetch_lm_sources_extended.py --all
    python scripts/fetch_lm_sources_extended.py --source kieviet
    python scripts/fetch_lm_sources_extended.py --dedup-only
    python scripts/fetch_lm_sources_extended.py --all --offline   # skip network
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import unicodedata
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger("fetch_lm_sources_extended")

_LM_SOURCES_DIR = PROJECT_ROOT / "data" / "lm_sources"

# OAPEN handle 20.500.12657/30840 — Kieviet (2017), A Grammar of Rapa Nui.
# The .pdf.txt bitstream is a pre-rendered plain-text version of the book.
_KIEVIET_TXT_URL = (
    "https://library.oapen.org/rest/bitstreams/"
    "083062ef-bcc9-4ad3-be0e-baf93327368a/retrieve"
)
_ASJP_URL = "https://asjp.clld.org/languages/RAPA_NUI"

_HTTP_TIMEOUT = 30
_USER_AGENT = "hackingrongo-lm-fetch/1.0 (research; +https://github.com/sperkswerks-ai/hackingrongo)"

# ---------------------------------------------------------------------------
# Rapa Nui orthographic / phonotactic word recognition
# ---------------------------------------------------------------------------

# Letters that may appear in a normalised Rapa Nui word (g = velar nasal /ŋ/).
_RN_ALPHABET = frozenset("aeiouhkmngprtv")
_RN_VOWELS = frozenset("aeiou")
# A Rapa Nui word is one or more (C)V syllables; "ng" is the only digraph onset.
_RN_WORD_RE = re.compile(r"(?:(?:ng|[hkmnprtv])?[aeiou])+$")
# Diacritic / okina characters folded away before recognition.
_OKINA = "ʼʻʾʿ'‘’`"


def _normalise_word(raw: str) -> str:
    """Lowercase, strip diacritics and okina, keep only letters."""
    decomposed = unicodedata.normalize("NFD", raw.lower())
    stripped = "".join(c for c in decomposed if not unicodedata.combining(c))
    out = []
    for ch in stripped:
        if ch in _OKINA:
            continue
        if ch == "ŋ":
            out.append("g")
        elif ch.isalpha():
            out.append(ch)
    return "".join(out)


def is_rapa_nui_word(word: str) -> bool:
    """True iff *word* (already normalised) is a plausible Rapa Nui word.

    Requires every character to be in the Rapa Nui alphabet, the word to
    contain a vowel, and the whole word to decompose into (C)V syllables.
    Single bare vowels are accepted (a, e, i, o, u are real RN words).
    """
    if not word or any(c not in _RN_ALPHABET for c in word):
        return False
    if not any(c in _RN_VOWELS for c in word):
        return False
    return _RN_WORD_RE.match(word) is not None


def rapa_nui_words_in_line(line: str) -> tuple[list[str], float]:
    """Return (rn_words, fraction) for a text *line*.

    ``rn_words`` is the in-order list of Rapa Nui-plausible word forms;
    ``fraction`` is rn_words / total alphabetic tokens (0.0 if none).
    """
    raw_tokens = re.findall(r"[^\W\d_]+", line, flags=re.UNICODE)
    if not raw_tokens:
        return [], 0.0
    rn = [w for w in (_normalise_word(t) for t in raw_tokens) if is_rapa_nui_word(w)]
    return rn, len(rn) / len(raw_tokens)


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------

def _http_get(url: str, accept: str | None = None) -> bytes:
    import requests

    headers = {"User-Agent": _USER_AGENT}
    if accept:
        headers["Accept"] = accept
    resp = requests.get(url, headers=headers, timeout=_HTTP_TIMEOUT)
    resp.raise_for_status()
    return resp.content


# ---------------------------------------------------------------------------
# Source fetchers — each returns the number of forms/lines written, or -1 on fail
# ---------------------------------------------------------------------------

def fetch_kohaumotu(out_path: Path) -> int:
    """Scrape Kohau Motu dictionary headwords (reuses the existing parser)."""
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "fkd", PROJECT_ROOT / "scripts" / "fetch_kohaumotu_dictionary.py"
        )
        fkd = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(fkd)
        html = fkd._fetch(fkd.DICT_URL)
        words = fkd.extract_headwords(html)
    except Exception as exc:
        log.warning("kohaumotu fetch failed (%s): %s", type(exc).__name__, exc)
        return -1
    forms = sorted({_normalise_word(w) for w in words if is_rapa_nui_word(_normalise_word(w))})
    out_path.write_text("\n".join(forms) + "\n", encoding="utf-8")
    log.info("kohaumotu: wrote %d headwords → %s", len(forms), out_path.name)
    return len(forms)


def fetch_asjp(out_path: Path) -> int:
    """Best-effort ASJP Rapa Nui wordlist (ASJPcode, mostly non-orthographic)."""
    rows: list[str] = []
    for accept in ("text/csv", "application/csv", None):
        try:
            blob = _http_get(_ASJP_URL + (".csv" if accept else ""), accept=accept)
            text = blob.decode("utf-8", errors="replace")
            if text.lstrip().lower().startswith("<!doctype") or "<html" in text[:200].lower():
                continue
            rows = text.splitlines()
            break
        except Exception as exc:
            log.debug("asjp attempt (accept=%s) failed: %s", accept, exc)
    if not rows:
        log.warning("asjp: could not retrieve a parseable wordlist — skipped.")
        return -1
    # Pull a plausible "form/value" column from each CSV row.
    import csv as _csv
    forms: set[str] = set()
    reader = _csv.DictReader(rows)
    value_keys = [k for k in (reader.fieldnames or [])
                  if k and k.lower() in ("value", "form", "counterpart", "word")]
    for row in reader:
        for k in value_keys:
            w = _normalise_word(row.get(k, ""))
            if is_rapa_nui_word(w):
                forms.add(w)
    out_path.write_text("\n".join(sorted(forms)) + "\n", encoding="utf-8")
    log.info("asjp: wrote %d orthographic forms → %s (ASJPcode forms dropped)",
             len(forms), out_path.name)
    return len(forms)


def fetch_kieviet(out_path: Path, min_words: int = 2, min_fraction: float = 0.6) -> int:
    """Extract Rapa Nui example lines from the Kieviet (2017) grammar text.

    Downloads the OAPEN pre-rendered plain text and keeps lines whose tokens
    are overwhelmingly Rapa Nui by orthography + phonotactics — i.e. the
    object-language lines of the interlinear examples — dropping English
    translations and the small-caps gloss lines.
    """
    try:
        blob = _http_get(_KIEVIET_TXT_URL)
    except Exception as exc:
        log.warning("kieviet text download failed (%s): %s", type(exc).__name__, exc)
        return -1
    text = blob.decode("utf-8", errors="replace")

    kept: list[str] = []
    seen: set[str] = set()
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if len(line) < 4:
            continue
        rn_words, frac = rapa_nui_words_in_line(line)
        # Require several Rapa Nui words and a high RN fraction so English
        # translation / gloss lines and headers are excluded.
        if len(rn_words) >= min_words and frac >= min_fraction:
            sentence = " ".join(rn_words)
            if sentence not in seen:
                seen.add(sentence)
                kept.append(sentence)
    if not kept:
        log.warning("kieviet: no Rapa Nui lines recognised — skipped.")
        return -1
    out_path.write_text("\n".join(kept) + "\n", encoding="utf-8")
    log.info("kieviet: wrote %d Rapa Nui example lines → %s", len(kept), out_path.name)
    return len(kept)


# ---------------------------------------------------------------------------
# Dedup + reporting
# ---------------------------------------------------------------------------

def _find_abvd() -> Path | None:
    """Locate the existing ABVD Rapa Nui wordlist, wherever it lives."""
    candidates = list((PROJECT_ROOT / "data").rglob("*abvd*"))
    files = [p for p in candidates if p.is_file() and p.suffix in (".txt", ".tsv", ".csv")]
    return files[0] if files else None


def _vocab_of(path: Path) -> set[str]:
    """Unique normalised Rapa Nui word forms in a source file (token-level)."""
    vocab: set[str] = set()
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return vocab
    for line in text.splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        for tok in re.findall(r"[^\W\d_]+", line, flags=re.UNICODE):
            w = _normalise_word(tok)
            if is_rapa_nui_word(w):
                vocab.add(w)
    return vocab


def deduplicate_and_report(lm_dir: Path) -> dict:
    """Log per-source and merged vocabulary sizes across lm_sources + ABVD."""
    sources = sorted(p for p in lm_dir.glob("*.txt") if not p.name.startswith("_"))
    abvd = _find_abvd()
    if abvd:
        sources.append(abvd)

    per_source: dict[str, int] = {}
    merged: set[str] = set()
    log.info("─" * 60)
    log.info("Vocabulary report (unique Rapa Nui word forms)")
    log.info("─" * 60)
    for src in sources:
        vocab = _vocab_of(src)
        per_source[src.name] = len(vocab)
        new = len(vocab - merged)
        merged |= vocab
        log.info("  %-40s %6d forms  (+%d new)", src.name, len(vocab), new)
    log.info("─" * 60)
    log.info("  MERGED (deduplicated total)              %6d forms", len(merged))
    log.info("─" * 60)

    report = {
        "per_source": per_source,
        "abvd_file": str(abvd.relative_to(PROJECT_ROOT)) if abvd else None,
        "merged_total": len(merged),
        "n_sources": len(sources),
    }
    (lm_dir / "_vocab_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

_SOURCES = {
    "kohaumotu": ("rapanui_kohaumotu.txt", fetch_kohaumotu),
    "asjp":      ("rapanui_asjp.txt",      fetch_asjp),
    "kieviet":   ("rapanui_kieviet_examples.txt", fetch_kieviet),
}


def main() -> int:
    p = argparse.ArgumentParser(
        description="Download & process additional Rapa Nui LM sources.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--all", action="store_true", help="Fetch every source.")
    p.add_argument("--source", choices=sorted(_SOURCES), action="append",
                   default=[], help="Fetch a specific source (repeatable).")
    p.add_argument("--dedup-only", action="store_true",
                   help="Skip fetching; just dedup + report over existing files.")
    p.add_argument("--offline", action="store_true",
                   help="Skip all network fetches (implies --dedup-only behaviour).")
    p.add_argument("--lm-dir", type=Path, default=_LM_SOURCES_DIR)
    args = p.parse_args()

    args.lm_dir.mkdir(parents=True, exist_ok=True)

    targets: list[str] = []
    if not (args.dedup_only or args.offline):
        targets = sorted(_SOURCES) if args.all or not args.source else args.source

    results: dict[str, int] = {}
    for name in targets:
        filename, fn = _SOURCES[name]
        log.info("Fetching source: %s", name)
        results[name] = fn(args.lm_dir / filename)

    report = deduplicate_and_report(args.lm_dir)

    failed = [n for n, r in results.items() if r is not None and r < 0]
    if failed:
        log.warning("Sources that failed or yielded nothing: %s", ", ".join(failed))
    log.info("Done. Merged vocabulary: %d forms across %d source file(s).",
             report["merged_total"], report["n_sources"])
    # Non-fatal: a blocked network source shouldn't fail the whole run.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
