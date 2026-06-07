"""
TOOLING — one-time data preparation; not part of the reproducible analysis pipeline.

link_svg_to_corpus.py — back-fill Barthel/Horley codes into catalog.json.

The kohaumotu SVG path IDs encode side as ``a`` (recto) or ``b`` (verso).
The corpus JSON files encode side as ``r`` (recto) or ``v`` (verso).
The original scraper built corpus lookup keys with ``r``/``v`` but queried
them with ``a``/``b``, so every lookup returned nothing.

This script:
  1. Rebuilds a corpus position index from ``data/corpus/`` (side: r/v).
  2. For every record in ``data/glyphs/svg/catalog.json``, maps the SVG
     side (a→r, b→v) and looks up the matching corpus entry.
  3. Writes an updated catalog.json and prints linkage statistics.

Usage::

    python scripts/link_svg_to_corpus.py [--catalog PATH] [--corpus-dir PATH]
                                         [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

log = logging.getLogger(__name__)

# Normalize corpus side codes to kohaumotu convention (a=recto, b=verso).
# Some corpus tablets use r/v (scholarly), others already use a/b (kohaumotu).
_SIDE_TO_AB: dict[str, str] = {"r": "a", "v": "b", "a": "a", "b": "b"}


# ---------------------------------------------------------------------------
# Build corpus index
# ---------------------------------------------------------------------------

def build_corpus_index(corpus_dir: Path) -> dict[str, dict]:
    """Return ``{pos_key: corpus_fields}`` keyed on sequential within-line position.

    Position key format: ``{TABLET}{SIDE_ab}{LINE:02d}-{LINE_SEQ:03d}``
    e.g. ``Ba01-017``

    Both side codes are normalised to kohaumotu ``a``/``b`` convention so the
    key space matches SVG filenames directly (no mapping needed at lookup time).
    Glyph numbering is sequential within each (tablet, side, line) group sorted
    by the corpus ``position`` field — this handles corpora that reset
    ``glyph_num`` at every segment boundary.
    """
    from collections import defaultdict

    index: dict[str, dict] = {}
    for path in sorted(corpus_dir.glob("[A-Z].json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        tid = data["tablet_id"]
        cluster = data.get("cluster", "unknown")

        # Group glyphs by (side_ab, line), preserving global position order.
        groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
        for g in sorted(data["glyphs"], key=lambda g: g.get("position", 0)):
            side_ab = _SIDE_TO_AB.get(g.get("side", "a"), "a")
            line = g.get("line", "01")
            groups[(side_ab, line)].append(g)

        for (side_ab, line), glyphs in groups.items():
            for line_seq, g in enumerate(glyphs, 1):
                key = f"{tid}{side_ab}{line}-{line_seq:03d}"
                index[key] = {
                    "barthel_code": g.get("barthel_code"),
                    "horley_code": g.get("horley_code"),
                    "horley_components": g.get("horley_components"),
                    "inverted": g.get("inverted", False),
                    "uncertain": g.get("uncertain", False),
                    "cluster": cluster,
                }
    return index


# ---------------------------------------------------------------------------
# Linkage
# ---------------------------------------------------------------------------

def link(catalog_path: Path, corpus_dir: Path, *, dry_run: bool = False) -> None:
    log.info("Loading catalog: %s", catalog_path)
    raw = json.loads(catalog_path.read_text(encoding="utf-8"))
    records: list[dict] = raw.get("records", raw) if isinstance(raw, dict) else raw

    log.info("Building corpus index from: %s", corpus_dir)
    index = build_corpus_index(corpus_dir)
    log.info("Corpus index: %d position keys", len(index))

    linked = 0
    missing = 0
    already_had = 0

    for rec in records:
        # SVG side is already in a/b — use directly as the lookup key.
        svg_side = rec.get("side", "a")
        tablet = rec.get("tablet", "?")
        line = rec.get("line", "01")
        glyph_num = rec.get("glyph_num", "001")
        pos_key = f"{tablet}{svg_side}{line}-{int(glyph_num):03d}"

        entry = index.get(pos_key)
        if entry is None:
            missing += 1
            continue

        if rec.get("barthel_code") is not None:
            already_had += 1
            # Still update in case corpus was corrected
            for field in ("barthel_code", "horley_code", "horley_components",
                          "inverted", "uncertain", "cluster"):
                rec[field] = entry[field]
        else:
            for field in ("barthel_code", "horley_code", "horley_components",
                          "inverted", "uncertain", "cluster"):
                rec[field] = entry[field]
            if entry.get("barthel_code"):
                linked += 1

    total = len(records)
    now_coded = sum(1 for r in records if r.get("barthel_code"))
    print(f"Total catalog records : {total}")
    print(f"Newly linked          : {linked}")
    print(f"Already had code      : {already_had}")
    print(f"No corpus match       : {missing}")
    print(f"Now with Barthel code : {now_coded} ({100*now_coded/total:.1f}%)")

    if dry_run:
        print("(dry-run — catalog not written)")
        return

    # Write back preserving original structure
    if isinstance(raw, dict):
        raw["records"] = records
        out = raw
    else:
        out = records

    catalog_path.write_text(json.dumps(out, ensure_ascii=False, indent=2),
                            encoding="utf-8")
    log.info("Wrote updated catalog: %s", catalog_path)
    print(f"Catalog written: {catalog_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--catalog",
                   default="data/glyphs/svg/catalog.json",
                   help="Path to catalog.json (default: data/glyphs/svg/catalog.json)")
    p.add_argument("--corpus-dir",
                   default="data/corpus",
                   help="Directory containing per-tablet corpus JSON files "
                        "(default: data/corpus)")
    p.add_argument("--dry-run", action="store_true",
                   help="Report what would change without writing the catalog")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(message)s",
    )
    link(
        catalog_path=Path(args.catalog),
        corpus_dir=Path(args.corpus_dir),
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
