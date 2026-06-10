#!/usr/bin/env python3
"""
build_allographs.py — Generate data/catalog/allographs.json from corpus scan.

Scans every unique barthel_code in data/corpus/*.json and emits a
variant → canonical mapping.  The canonical form is the 3-digit zero-padded
numeric base: allograph letters (a–z), the uppercase V variant marker, and
the !/?  diacritics are all stripped.

Codes excluded from the mapping (cannot be reduced to a single base):
  - Compound tokens containing connection characters  . - : ;
    (e.g. "076:042", "000!-016t?")
  - Range estimates of the form (N-M) or (N-M)!
    (e.g. "(0-3)!", "(10-20)")
  - Illegible placeholders (?, empty string)

Usage
-----
    python scripts/tooling/build_allographs.py
    python scripts/tooling/build_allographs.py --corpus-dir data/corpus \\
        --output data/catalog/allographs.json
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

_DEFAULT_CORPUS = PROJECT_ROOT / "data" / "corpus"
_DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "catalog" / "allographs.json"

# Connection characters that mark fused/ligature compound tokens.
_COMPOUND_CHARS: frozenset[str] = frozenset(".:;-")

# Simple sign: up to 3 digits + one optional lowercase allograph letter +
# one optional diacritic (! = inverted, ? = uncertain).
# Matches: 001, 001a, 001!, 001a!, 001a?
# Does NOT match: 001V (uppercase), 001af (two letters), 076:042 (compound)
_SIMPLE = re.compile(r"^(\d{1,3}[a-z]?)([!?])?$")


def _canonical_base(code: str) -> str | None:
    """Return the canonical 3-digit numeric base for a raw corpus Barthel code.

    Returns ``None`` for codes that cannot be reduced to a single base
    (compounds, range estimates, illegible placeholders).

    Examples
    --------
    >>> _canonical_base("001a!")
    '001'
    >>> _canonical_base("076V")
    '076'
    >>> _canonical_base("001af")
    '001'
    >>> _canonical_base("076:042")   # compound
    None
    >>> _canonical_base("(0-3)!")    # range estimate
    None
    """
    if not code or code in ("?", "_"):
        return None
    if code.startswith("("):
        return None
    if _COMPOUND_CHARS & set(code):
        return None

    # Standard simple allograph: strip diacritic, then allograph letter.
    m = _SIMPLE.fullmatch(code)
    if m:
        base = m.group(1)          # e.g. "001a" from "001a!"
        digits = re.sub(r"[^0-9]", "", base)
        return digits.zfill(3) if digits else None

    # Fallback for uppercase-V variants (001V), multi-letter suffixes
    # (001af, 001Va, 001bf), and other non-standard suffixes (003*).
    m2 = re.match(r"^(\d{1,3})", code)
    if m2:
        return m2.group(1).zfill(3)

    return None


def build(corpus_dir: Path, output_path: Path) -> None:
    # ── 1. Collect every unique barthel_code across all corpus files ───────────
    raw_codes: set[str] = set()
    corpus_files = sorted(corpus_dir.glob("*.json"))
    for f in corpus_files:
        data = json.loads(f.read_text(encoding="utf-8"))
        for g in data.get("glyphs", []):
            code = str(g.get("barthel_code", "")).strip()
            if code:
                raw_codes.add(code)

    log.info(
        "Scanned %d corpus files — %d unique raw codes.",
        len(corpus_files),
        len(raw_codes),
    )

    # ── 2. Derive variant → canonical mapping (omit identity entries) ─────────
    mapping: dict[str, str] = {}
    n_compound = n_range = n_unresolvable = 0

    for code in sorted(raw_codes):
        if code.startswith("("):
            n_range += 1
            continue
        if _COMPOUND_CHARS & set(code):
            n_compound += 1
            continue

        canonical = _canonical_base(code)
        if canonical is None:
            n_unresolvable += 1
            continue

        if canonical != code:
            mapping[code] = canonical

    # ── 3. Compute and log the reduction ──────────────────────────────────────
    normalized_types = {mapping.get(c, c) for c in raw_codes}
    n_raw = len(raw_codes)
    n_canonical = len(normalized_types)
    pct = 100.0 * (1.0 - n_canonical / n_raw) if n_raw else 0.0

    log.info(
        "Sign normalization: %d raw types → %d canonical types "
        "(%.1f%% reduction, %d variant entries).",
        n_raw,
        n_canonical,
        pct,
        len(mapping),
    )
    log.info(
        "  Excluded from mapping: %d compound codes, %d range estimates, "
        "%d unresolvable.",
        n_compound,
        n_range,
        n_unresolvable,
    )

    # ── 4. Write allographs.json ───────────────────────────────────────────────
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload: dict[str, object] = {
        "_schema": "allograph-catalog-v1",
        "_description": (
            "Maps every attested variant Barthel sign code to its canonical base code "
            "(3-digit zero-padded numeric, allograph suffix and !? diacritics stripped). "
            "Auto-generated by scripts/tooling/build_allographs.py from the corpus scan."
        ),
        "_raw_types": n_raw,
        "_canonical_types": n_canonical,
    }
    payload.update(mapping)

    output_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    log.info("Written %d variant entries → %s", len(mapping), output_path)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--corpus-dir",
        type=Path,
        default=_DEFAULT_CORPUS,
        metavar="DIR",
        help="Directory of corpus JSON files (default: data/corpus).",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=_DEFAULT_OUTPUT,
        metavar="JSON",
        help="Output path (default: data/catalog/allographs.json).",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    if not args.corpus_dir.exists():
        log.error("Corpus directory not found: %s", args.corpus_dir)
        sys.exit(1)
    build(args.corpus_dir, args.output)
