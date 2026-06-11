"""
TOOLING — one-time data preparation; not part of the reproducible analysis pipeline.

Build script: enrich corpus JSON files with Horley codes and cluster labels.

Reads the per-tablet JSON files produced by ``hackingrongo.data.corpus_xml``
and writes them back with three new fields per glyph:

* ``barthel_base``  — Barthel code stripped of diacritics (``!``, ``?``),
                      used as the catalog lookup key.
* ``horley_code``   — Horley (2021) code from the sign catalog, or ``null``
                      if the sign has no Horley equivalent or is a compound /
                      range token.
* ``inverted``      — ``true`` when the ``!`` diacritic is present (glyph
                      appears inverted in the boustrophedon reading direction).
* ``uncertain``     — ``true`` when the ``?`` diacritic is present
                      (identification uncertain).

A ``cluster`` field is also added at the tablet level (one of
``"pre_contact"``, ``"post_contact"``, ``"excluded"``, ``"unknown"``).

Usage
-----
From the project root (``hackingrongo/``)::

    conda run python scripts/build_corpus.py
    conda run python scripts/build_corpus.py --dry-run   # preview stats only

The script is idempotent: running it again after the fields are already
present simply overwrites them with the same values.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from omegaconf import OmegaConf

from hackingrongo.data.catalog import SignCatalog
from hackingrongo.data.corpus import assign_cluster

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s  %(message)s",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# CEIPP code parsing
# ---------------------------------------------------------------------------

# A *simple* sign code after diacritic stripping: 1–3 digits followed by
# an optional letter tail.  The first letter, when lowercase, is the
# Barthel variant letter and is kept in the base; any further letters
# (CEIPP positional/ligature markers like "fy", "x", or uppercase "V")
# are modifiers and are stripped for catalog lookup but preserved in the
# glyph record.  Examples:
#   "022bfy" → base "022b", modifiers "fy"
#   "001V"   → base "001",  modifiers "V"
#   "044ax"  → base "044a", modifiers "x"
_SIMPLE = re.compile(r"^(\d{1,3})([a-zA-Z]*)$")

# Characters that signal a compound token (multiple signs in one <ceipp>).
_COMPOUND_CHARS = frozenset("-.;:")

# Splitter for compound CEIPP tokens: splits on . - : ; and the apostrophe-like
# fused-sign connector (ascii 39 or unicode variants seen in some encodings).
_COMPOUND_SPLIT = re.compile(r"[.\-:;'\u2019]")

# Per-component modifier stripper: strip trailing ! or ? before lookup
_COMPONENT_MODIFIER = re.compile(r"[!?]+$")

# Per-component: leading zeros from numeric prefix (mirrors catalog normalization)
_COMPONENT_ZERO_PAD = re.compile(r"^0+(?=\d)")


def parse_ceipp(code: str) -> tuple[str | None, bool, bool, str]:
    """Return ``(barthel_base, inverted, uncertain, modifiers)`` for a CEIPP token.

    ``barthel_base`` is the stripped code suitable for catalog lookup, or
    ``None`` for compound tokens, range estimates, and the bare ``"?"``
    illegible placeholder.  ``modifiers`` holds any CEIPP letter markers
    stripped from the tail (e.g. ``"fy"``, ``"V"``) so no transcription
    information is silently discarded.
    """
    # Bare illegible placeholder
    if code == "?":
        return None, False, False, ""

    # Range estimate: "(N-M)" or "(N-M)!"
    if code.startswith("("):
        return None, "!" in code, "?" in code, ""

    # Compound token — connection characters embedded in the code string
    if _COMPOUND_CHARS & set(code):
        return None, "!" in code, "?" in code, ""

    inverted = "!" in code
    uncertain = "?" in code
    stripped = code.replace("!", "").replace("?", "")

    # Simple single-sign token: digits + variant letter + modifier tail
    m = _SIMPLE.fullmatch(stripped)
    if m:
        digits, letters = m.group(1), m.group(2)
        variant = letters[:1] if letters[:1].islower() else ""
        modifiers = letters[len(variant):]
        return digits + variant, inverted, uncertain, modifiers

    # Fallback — unrecognised format
    return None, inverted, uncertain, ""


def decompose_compound(code: str) -> list[str]:
    """Break a compound CEIPP token into resolved base Barthel codes.

    Splits on connection characters (``.``, ``-``, ``:``, ``;``) and the
    fused-sign connector (``'``).  Each part has trailing diacritics (``!``,
    ``?``) and leading zeros stripped, returning only the normalised base
    codes suitable for catalog lookup.

    Range-estimate tokens like ``(10-20)`` are excluded (return empty list).
    """
    if code.startswith("("):
        return []
    parts = _COMPOUND_SPLIT.split(code)
    bases: list[str] = []
    for part in parts:
        part = _COMPONENT_MODIFIER.sub("", part).strip()
        part = _COMPONENT_ZERO_PAD.sub("", part)
        # Accept only non-empty, digit-leading tokens (reject lone modifiers)
        if part and part[0].isdigit():
            bases.append(part)
    return bases


# ---------------------------------------------------------------------------
# Build helpers
# ---------------------------------------------------------------------------


def load_catalog(cfg, project_root: Path) -> SignCatalog:
    return SignCatalog.load(cfg, project_root)


_LEADING_DIGITS = re.compile(r"^\d+")


def _arbitrate_variant(
    base: str, modifiers: str, catalog: SignCatalog
) -> tuple[str, str]:
    """Decide whether a parsed variant letter is genuine or a CEIPP marker.

    The first lowercase letter after the digits is *usually* a Barthel
    variant ("003a"), but some CEIPP markers are also lowercase ("522f").
    The catalog arbitrates: when ``base`` with its variant letter does not
    resolve but the bare digits do, the letter is demoted to a modifier.
    """
    if catalog.barthel_to_horley(base) is not None:
        return base, modifiers
    m = _LEADING_DIGITS.match(base)
    if m and m.group(0) != base and catalog.barthel_to_horley(m.group(0)) is not None:
        return m.group(0), base[len(m.group(0)):] + modifiers
    return base, modifiers


def enrich_tablet(data: dict, cluster: str, catalog: SignCatalog) -> dict:
    """Return *data* with cluster and per-glyph enrichment fields added.

    For simple single-sign tokens, ``horley_code`` is the resolved Horley code
    (or ``None``).  For compound tokens, ``horley_code`` is always ``None`` but
    ``horley_components`` contains the list of resolved Horley codes for each
    decomposed component.  Components that do not resolve are omitted from the
    list (i.e. only successfully resolved components are included).
    """
    data = dict(data)
    data["cluster"] = cluster

    enriched = []
    for glyph in data.get("glyphs", []):
        g = dict(glyph)
        code = g.get("barthel_code", "")
        base, inverted, uncertain, modifiers = parse_ceipp(code)
        if base is not None:
            base, modifiers = _arbitrate_variant(base, modifiers, catalog)
        g["barthel_base"] = base
        g["inverted"] = inverted
        g["uncertain"] = uncertain
        g["code_modifiers"] = modifiers or None

        if base is not None:
            # Simple token — direct Horley lookup (catalog normalises zero-padding)
            g["horley_code"] = catalog.barthel_to_horley(base)
            g["horley_components"] = None
            g["barthel_components"] = None
        elif not code.startswith("(") and code != "?":
            # Compound token — decompose, arbitrate each component's variant
            # letter against the catalog, and look up Horley equivalents.
            components = [
                _arbitrate_variant(c, "", catalog)[0]
                for c in decompose_compound(code)
            ]
            resolved = [catalog.barthel_to_horley(c) for c in components]
            resolved = [h for h in resolved if h is not None]
            g["horley_code"] = None  # compound: no single canonical code
            g["horley_components"] = resolved if resolved else None
            g["barthel_components"] = components if components else None
        else:
            g["horley_code"] = None
            g["horley_components"] = None
            g["barthel_components"] = None

        enriched.append(g)

    data["glyphs"] = enriched
    return data


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def build(dry_run: bool = False) -> None:
    cfg = OmegaConf.load(PROJECT_ROOT / "conf" / "config.yaml")
    catalog = load_catalog(cfg, PROJECT_ROOT)
    corpus_dir = PROJECT_ROOT / cfg.paths.corpus_dir

    tablet_files = sorted(corpus_dir.glob("[A-Z].json"))
    if not tablet_files:
        log.error("No tablet JSON files found in %s", corpus_dir)
        sys.exit(1)

    total_glyphs = 0
    total_resolved = 0
    total_compound = 0
    total_uncertain = 0
    total_inverted = 0

    # Per-cluster accumulators: cluster -> {total, resolved, tablets}
    cluster_stats: dict[str, dict] = {}

    for path in tablet_files:
        data = json.loads(path.read_text(encoding="utf-8"))
        tablet_id = data["tablet_id"]
        cluster = assign_cluster(tablet_id, cfg)
        enriched = enrich_tablet(data, cluster, catalog)

        glyphs = enriched["glyphs"]
        n = len(glyphs)
        resolved = sum(1 for g in glyphs if g["horley_code"] is not None)
        compound = sum(
            1 for g in glyphs
            if g["barthel_base"] is None and g["barthel_code"] != "?"
        )
        compound_with_components = sum(
            1 for g in glyphs if g.get("horley_components")
        )
        n_uncertain = sum(1 for g in glyphs if g["uncertain"])
        n_inverted = sum(1 for g in glyphs if g["inverted"])

        total_glyphs += n
        total_resolved += resolved
        total_compound += compound
        total_uncertain += n_uncertain
        total_inverted += n_inverted

        cs = cluster_stats.setdefault(cluster, {"total": 0, "resolved": 0, "tablets": []})
        cs["total"] += n
        cs["resolved"] += resolved
        cs["tablets"].append(tablet_id)

        pct = 100.0 * resolved / n if n else 0.0
        log.info(
            "  %-3s  cluster=%-14s  %4d glyphs  "
            "%4d Horley (%.0f%%)  %3d compound (%d decomposed)  "
            "%3d inverted  %3d uncertain",
            tablet_id,
            cluster,
            n,
            resolved,
            pct,
            compound,
            compound_with_components,
            n_inverted,
            n_uncertain,
        )

        if not dry_run:
            path.write_text(
                json.dumps(enriched, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

    log.info("")
    log.info(
        "Corpus totals: %d glyphs  |  %d Horley resolved (%.1f%%)  "
        "|  %d compound  |  %d inverted  |  %d uncertain",
        total_glyphs,
        total_resolved,
        100.0 * total_resolved / total_glyphs if total_glyphs else 0.0,
        total_compound,
        total_inverted,
        total_uncertain,
    )

    # --- By-cluster breakdown ---
    log.info("")
    log.info("By-cluster breakdown:")
    for cluster in ("pre_contact", "post_contact", "excluded", "unknown"):
        cs = cluster_stats.get(cluster)
        if not cs:
            continue
        n, r = cs["total"], cs["resolved"]
        pct = 100.0 * r / n if n else 0.0
        tablets_str = ", ".join(cs["tablets"])
        if cluster == "excluded":
            log.info(
                "  %-14s  tablets=[%s]  %5d glyphs  "
                "%4d Horley (%.0f%%)  [excluded from temporal analysis]",
                cluster, tablets_str, n, r, pct,
            )
        elif cluster == "unknown":
            log.info(
                "  %-14s  %d tablets      %5d glyphs  "
                "%4d Horley (%.0f%%)  [undated — await dating]",
                cluster, len(cs["tablets"]), n, r, pct,
            )
        else:
            log.info(
                "  %-14s  tablets=[%s]  %5d glyphs  "
                "%4d Horley (%.0f%%)",
                cluster, tablets_str, n, r, pct,
            )

    # Emit machine-readable stats dict for downstream use
    stats = {
        "total_glyphs": total_glyphs,
        "horley_resolved": total_resolved,
        "resolution_rate": round(total_resolved / total_glyphs, 4) if total_glyphs else 0.0,
        "compound_unresolvable": total_compound,
        "inverted": total_inverted,
        "uncertain": total_uncertain,
        "by_cluster": {},
    }
    for cluster, cs in cluster_stats.items():
        n, r = cs["total"], cs["resolved"]
        entry: dict = {
            "tablets": cs["tablets"],
            "total_glyphs": n,
            "horley_resolved": r,
            "resolution_rate": round(r / n, 4) if n else 0.0,
        }
        if cluster == "excluded":
            entry["note"] = "European wood provenance — excluded from temporal analysis"
        stats["by_cluster"][cluster] = entry

    log.info("")
    log.info("stats_json=%s", json.dumps(stats, separators=(",", ":")))

    if dry_run:
        log.info("Dry-run mode — no files written.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Enrich corpus JSON files with Horley codes and cluster labels.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print statistics without writing any files.",
    )
    args = parser.parse_args()
    build(dry_run=args.dry_run)


if __name__ == "__main__":
    main()
