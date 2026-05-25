"""
Parse the Intercontinental Dictionary Series (IDS) Rapa Nui contribution
and write era-stratified word forms for Polynesian language model training.

Source
------
Key, M.R. & Comrie, B. (eds.) 2015. The Intercontinental Dictionary Series.
Leipzig: Max Planck Institute for Evolutionary Anthropology.
https://ids.clld.org  (CC-BY 4.0)

IDS Rapa Nui contribution 238 aggregates five historical lexical sources,
spanning from de Agüera (1770) to Englert (1978):

    de Agüera 1770       — first European contact wordlist (~200 entries)
    Thomson 1891         — Smithsonian survey; pre-missionary era
    Roussel 1908         — Roussel's vocabulary; early missionary period
    Fuentes 1960         — mid-20th century; modern Rapa Nui
    Englert 1978         — comprehensive modern dictionary

The 'source' column allows era stratification to match the diachronic model
in hackingrongo (pre-contact anchor / post-contact cluster).

Download URL
------------
https://ids.clld.org/contributions/238.tab

TSV columns (tab-separated, UTF-8):
    ID, Language_ID, Parameter_ID, Value, Source, Comment, ...

The ``Value`` column contains the Rapa Nui word form; ``Source`` contains
the bibliographic source key used above.

Output
------
Writes to ``data/polynesian_texts/rapanui/ids.txt`` (all sources combined)
and optionally to ``data/polynesian_texts/rapanui/ids_pre_contact.txt`` and
``data/polynesian_texts/rapanui/ids_post_contact.txt`` when --stratify is
supplied.

Usage
-----
    python scripts/parse_ids.py

Optional flags::

    --data-dir PATH       output root (default: data/polynesian_texts)
    --cache-dir PATH      cache downloaded TSV here to avoid re-fetching
    --stratify            also write era-stratified output files
    --dry-run             print statistics without writing any files
"""

from __future__ import annotations

import argparse
import csv
import io
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
# Constants
# ---------------------------------------------------------------------------

_IDS_URL = "https://ids.clld.org/contributions/238.tab"

# NB: The live IDS contribution 238 TSV (as of 2025) has no 'Source' column.
# Comment abbreviations found in the data are: JF, Rouss, Englert, M, Thom.
# De Agüera 1770 forms appear in the canonical Rapa Nui_Phonemic column as
# modern phonemic reconstructions — correctly classified as post_contact by
# the absence of a source tag.  The de Agüera entries in _SOURCE_TO_TIER
# below are therefore unreachable dead code; retained for if/when IDS adds
# source columns or the format reverts to row-per-source.
#
# Practical consequence: the pre-contact LM is Thomson (~400 comment segments)
# + Roussel (~945) ONLY — not de Agüera.  This is an upstream data limitation.
# Because Roussel (~70 % of pre-contact forms) was a post-contact missionary
# vocabulary, consider a sensitivity run with Thomson only (see config.yaml
# mcmc.sensitivity_runs) to test whether findings hold on the chronologically
# cleaner ~400-form corpus.  See: https://ids.clld.org/contributions/238

# Source keys as they appear in the IDS 'Source' column → era tier.
# Sources not in this map are placed in 'unclassified'.
_SOURCE_TO_TIER: dict[str, str] = {
    "de Agüera 1770":  "pre_contact",
    "deaguera1770":    "pre_contact",   # normalised key variant
    "Thomson 1891":    "pre_contact",
    "thomson1891":     "pre_contact",
    "Roussel 1908":    "pre_contact",   # pre-missionary vocabulary
    "roussel1908":     "pre_contact",
    "Fuentes 1960":    "post_contact",
    "fuentes1960":     "post_contact",
    "Englert 1978":    "post_contact",
    "englert1978":     "post_contact",
}

_VOWELS: frozenset[str] = frozenset("aeiou")

# Abbreviations used in the IDS comment field to tag source-specific forms.
# The current IDS TSV (238.tab) no longer carries a 'Source' column; instead,
# per-source variants appear inline: "Thom oone, kaina; Rouss heenua; Englert henua".
_COMMENT_ABBREV_TO_TIER: dict[str, str] = {
    "Thom":    "pre_contact",   # Thomson 1891
    "Rouss":   "pre_contact",   # Roussel 1908
    "Englert": "post_contact",  # Englert 1978
    "JF":      "post_contact",  # Fuentes 1960 (Jaussen-Fuentes)
    "M":       "post_contact",  # Métraux 1940
}
# Compiled once; matches any known abbreviation at the start of a ; segment.
_COMMENT_ABBREV_RE = re.compile(
    r'^(' + '|'.join(re.escape(k) for k in _COMMENT_ABBREV_TO_TIER) + r')\s+(.+)$'
)

_NORMALISE_MAP: dict[str, str] = {
    "\u0101": "a", "\u0113": "e", "\u012b": "i", "\u014d": "o", "\u016b": "u",
    "\u0103": "a", "\u0115": "e", "\u012d": "i", "\u014f": "o", "\u016d": "u",
    "\u00e1": "a", "\u00e9": "e", "\u00ed": "i", "\u00f3": "o", "\u00fa": "u",
    "\u00e0": "a", "\u00e8": "e", "\u00ec": "i", "\u00f2": "o", "\u00f9": "u",
    "\u00e2": "a", "\u00ea": "e", "\u00ee": "i", "\u00f4": "o", "\u00fb": "u",
    "\u02bc": "", "\u02bb": "", "\u0027": "", "\u2018": "", "\u2019": "",
}

_IDS_CITATION = (
    "Key, M.R. & Comrie, B. (eds.) 2015. The Intercontinental Dictionary Series. "
    "Leipzig: Max Planck Institute for Evolutionary Anthropology. "
    "https://ids.clld.org (CC-BY 4.0). "
    "Rapa Nui contribution 238: de Agüera (1770), Thomson (1891), Roussel (1908), "
    "Fuentes (1960), Englert (1978)."
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fetch_tsv(cache_path: Path | None) -> str:
    if cache_path is not None and cache_path.exists():
        logger.info("Loading cached IDS TSV from %s", cache_path)
        return cache_path.read_text(encoding="utf-8")

    logger.info("Downloading IDS Rapa Nui contribution 238 …")
    req = urllib.request.Request(_IDS_URL, headers={"User-Agent": "hackingrongo/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:  # noqa: S310
            content = resp.read().decode("utf-8", errors="replace")
    except Exception as exc:
        raise RuntimeError(
            f"Failed to download IDS data from {_IDS_URL}: {exc}\n"
            "Try supplying a locally downloaded file with --cache-dir."
        ) from exc

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(content, encoding="utf-8")
        logger.info("Cached to %s", cache_path)

    return content


def _normalise(form: str) -> str:
    form = unicodedata.normalize("NFC", form.lower().strip())
    return "".join(_NORMALISE_MAP.get(ch, ch) for ch in form).strip()


def _is_valid(form: str) -> bool:
    if len(form) < 2 or any(ch.isdigit() for ch in form):
        return False
    return any(ch in _VOWELS for ch in form)


def _resolve_tier(source_raw: str) -> str:
    """Map a raw 'Source' field value to pre_contact / post_contact / unclassified."""
    # Try exact match first, then normalised lower-case without spaces.
    if source_raw in _SOURCE_TO_TIER:
        return _SOURCE_TO_TIER[source_raw]
    key = source_raw.lower().replace(" ", "").replace(".", "")
    for k, tier in _SOURCE_TO_TIER.items():
        if k.lower().replace(" ", "").replace(".", "") == key:
            return tier
    # Partial substring match as final fallback.
    low = source_raw.lower()
    for k, tier in _SOURCE_TO_TIER.items():
        if k.lower()[:8] in low:
            return tier
    return "unclassified"


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def parse_ids_tsv(tsv_text: str) -> dict[str, list[str]]:
    """Parse IDS contribution TSV, returning forms keyed by tier.

    Returns
    -------
    dict[str, list[str]]
        Keys: ``"all"``, ``"pre_contact"``, ``"post_contact"``, ``"unclassified"``.
    """
    forms: dict[str, list[str]] = {
        "all": [],
        "pre_contact": [],
        "post_contact": [],
        "unclassified": [],
    }
    skipped = 0

    def _add(norm: str, tier: str) -> None:
        forms["all"].append(norm)
        forms[tier].append(norm)

    def _ingest(raw: str, tier: str) -> int:
        """Normalise raw (comma/slash-separated) and add valid forms; return skip count."""
        n_skipped = 0
        for part in re.split(r'[,/]', raw):
            norm = _normalise(part.strip())
            if _is_valid(norm):
                _add(norm, tier)
            else:
                n_skipped += 1
        return n_skipped

    try:
        reader = csv.DictReader(io.StringIO(tsv_text), delimiter="\t")
        for row in reader:
            # Current IDS 238.tab format: canonical form in 'Rapa Nui_Phonemic';
            # source-specific variants embedded in 'comment' as "ABBREV form1, form2".
            # Fall back to legacy 'Value'/'Form' columns for forward compat.
            canonical = (
                row.get("Rapa Nui_Phonemic")
                or row.get("Value")
                or row.get("Form")
                or ""
            ).strip()
            if not canonical or canonical in ("-", "—", "?"):
                skipped += 1
                continue

            # Canonical form → post_contact (modern synthesised form).
            skipped += _ingest(canonical, "post_contact")

            # Variant spellings column (if present).
            variants = (row.get("Rapa Nui_Phonemic (vars)") or "").strip()
            if variants:
                skipped += _ingest(variants, "post_contact")

            # Comment field: parse "ABBREV form1, form2; ABBREV2 form3" segments.
            comment = (row.get("comment") or "").strip()
            if comment:
                for segment in comment.split(";"):
                    segment = segment.strip()
                    m = _COMMENT_ABBREV_RE.match(segment)
                    if m:
                        abbrev, forms_str = m.group(1), m.group(2)
                        tier = _COMMENT_ABBREV_TO_TIER[abbrev]
                        skipped += _ingest(forms_str, tier)

    except Exception as exc:
        logger.warning("TSV parse error: %s", exc)

    for key, lst in forms.items():
        logger.info("  IDS tier '%s': %d forms.", key, len(lst))
    logger.info("  Skipped %d invalid/missing entries.", skipped)

    return forms


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------


def _deduplicate(lst: list[str]) -> list[str]:
    seen: set[str] = set()
    return [x for x in lst if not (x in seen or seen.add(x))]  # type: ignore[func-returns-value]


def _write_output(
    lang_dir: Path,
    filename: str,
    forms: list[str],
    dry_run: bool,
) -> None:
    if not forms:
        logger.warning("No forms for '%s'; skipping.", filename)
        return
    unique = _deduplicate(forms)
    out = lang_dir / filename
    if dry_run:
        logger.info("[DRY RUN] Would write %d forms → %s", len(unique), out)
        return
    lang_dir.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(unique) + "\n", encoding="utf-8")
    logger.info("Wrote %d unique forms → %s", len(unique), out)


def _update_metadata(lang_dir: Path, dry_run: bool, stratified: bool) -> None:
    out_meta = lang_dir / "metadata.json"
    if dry_run:
        return
    lang_dir.mkdir(parents=True, exist_ok=True)
    meta: dict = {}
    if out_meta.exists():
        try:
            meta = json.loads(out_meta.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            meta = {}

    sources = [s for s in meta.get("sources", []) if not s.get("file", "").startswith("ids")]
    sources.append(
        {
            "file": "ids.txt",
            "citation": _IDS_CITATION,
            "genre": "wordlist",
            "notes": (
                "All sources combined. "
                + ("See ids_pre_contact.txt and ids_post_contact.txt for stratified versions." if stratified else "")
            ),
        }
    )
    if stratified:
        for tier in ("pre_contact", "post_contact"):
            sources.append(
                {
                    "file": f"ids_{tier}.txt",
                    "citation": _IDS_CITATION,
                    "genre": "wordlist",
                    "tier": tier,
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
        description="Parse IDS Rapa Nui contribution 238 for LM training."
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
        help="Cache downloaded TSV here to avoid re-fetching.",
    )
    p.add_argument(
        "--stratify",
        action="store_true",
        help="Also write ids_pre_contact.txt and ids_post_contact.txt.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print statistics without writing any files.",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    cache_path = args.cache_dir / "ids_rapanui_238.tsv" if args.cache_dir else None

    try:
        tsv_text = _fetch_tsv(cache_path)
    except RuntimeError as exc:
        logger.error("%s", exc)
        sys.exit(1)

    logger.info("IDS TSV: %d chars.", len(tsv_text))
    forms = parse_ids_tsv(tsv_text)

    lang_dir = args.data_dir / "rapanui"

    _write_output(lang_dir, "ids.txt", forms["all"], args.dry_run)

    if args.stratify:
        _write_output(lang_dir, "ids_pre_contact.txt", forms["pre_contact"], args.dry_run)
        _write_output(lang_dir, "ids_post_contact.txt", forms["post_contact"], args.dry_run)
        if forms["unclassified"]:
            logger.info(
                "%d unclassified forms not written (add source keys to _SOURCE_TO_TIER to include them).",
                len(forms["unclassified"]),
            )

    _update_metadata(lang_dir, args.dry_run, args.stratify)
    logger.info(
        "Done.  Run `python scripts/build_language_models.py` to rebuild LMs."
    )


if __name__ == "__main__":
    main()
