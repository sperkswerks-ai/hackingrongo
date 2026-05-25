"""
Scrape individual glyph SVGs from kohaumotu.org/Rongorongo_new/views/.

Each tablet page (lines.php?item={T}&type=b) contains one inline <svg>
per line, each holding one <path id="glyph{T}{side}{line}-{num}-b"> per
glyph.  This script:

1. Fetches every tablet page (A–Y, Barthel encoding, type=b).
2. Parses each <path> element with its glyph ID.
3. Computes an approximate viewBox by scanning the path `d` coordinates.
4. Writes a standalone SVG file per glyph:
       data/glyphs/svg/{TABLET}/{SIDE}{LINE}-{NUM}.svg
5. Builds a catalog JSON:
       data/glyphs/svg/catalog.json
   mapping glyph position IDs to corpus-JSON Barthel codes and Horley codes.

Rate-limiting: one tablet per 1 s with a per-line 0.2 s pause to be polite
to the server.

Usage
-----
    conda run -n base python scripts/scrape_glyphs.py
    conda run -n base python scripts/scrape_glyphs.py --tablets D O
    conda run -n base python scripts/scrape_glyphs.py --dry-run  # parse only, no write
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import ssl
import sys
import time
import urllib.request
from pathlib import Path
from typing import Iterator
from xml.etree import ElementTree as ET

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

BASE_URL = "https://kohaumotu.org/Rongorongo_new/views/lines.php?item={tablet}&type=b"

ALL_TABLETS = list("ABCDEFGHIJKLMNOPQRSTUVWXY")

# Regex to extract a glyph path element.  The page uses plain HTML so
# we use regex rather than a full XML parser (the page is not well-formed XML).
_PATH_RE = re.compile(
    r'<path\s+id="(glyph[^"]+)"\s+d="([^"]+)"',
    re.DOTALL,
)

# SVG path number extractor: all numeric tokens (integers and floats,
# including sign).  We use these to approximate the bounding box.
_NUM_RE = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")


# ---------------------------------------------------------------------------
# SSL context (kohaumotu.org has a self-signed / untrusted chain)
# ---------------------------------------------------------------------------

def _make_ssl_ctx() -> ssl.SSLContext:
    # kohaumotu.org uses a self-signed certificate that is not trusted by
    # system CAs.  Verification is intentionally disabled for this specific
    # scraping use case.  Never reuse this context for other hosts.
    log.warning(
        "SSL certificate verification disabled for kohaumotu.org. "
        "This is intentional: the site uses a self-signed certificate. "
        "Do not reuse this SSL context for other hosts."
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
# Path bounding-box estimation
# ---------------------------------------------------------------------------

def _approx_bbox(d: str) -> tuple[float, float, float, float]:
    """Return (x_min, y_min, x_max, y_max) by scanning all numbers in path `d`.

    SVG Bezier paths store coordinate values as space/comma-separated numbers
    after command letters.  For absolute-coordinate paths (as used by
    kohaumotu), alternating numbers approximate the x and y extents.
    We simply take min/max of all numbers grouped by parity to get (x, y).
    """
    nums = [float(n) for n in _NUM_RE.findall(d)]
    if not nums:
        return 0.0, 0.0, 64.0, 64.0

    # Separate x and y values by position parity (0-indexed in pairs)
    xs = nums[0::2]
    ys = nums[1::2]

    # Fallback if one list is empty
    if not xs:
        xs = nums
    if not ys:
        ys = nums

    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)

    # Add a small margin so strokes don't clip at the SVG edge
    margin = 4.0
    return x_min - margin, y_min - margin, x_max + margin, y_max + margin


# ---------------------------------------------------------------------------
# ID parsing
# ---------------------------------------------------------------------------

_GLYPH_ID_RE = re.compile(
    r"^glyph([A-Z])([ab])(\d{2})-(\d{3})-b$",
    re.ASCII,
)


def parse_glyph_id(glyph_id: str) -> dict | None:
    """Parse a kohaumotu path ID into constituent fields.

    ID format: ``glyph{TABLET}{SIDE}{LINE:02d}-{GLYPH_NUM:03d}-b``

    Returns a dict with keys ``tablet``, ``side``, ``line``, ``glyph_num``
    or ``None`` if the ID doesn't match.
    """
    m = _GLYPH_ID_RE.match(glyph_id)
    if not m:
        return None
    return {
        "tablet": m.group(1),
        "side": m.group(2),
        "line": m.group(3),
        "glyph_num": m.group(4),
    }


# ---------------------------------------------------------------------------
# Corpus index (Barthel + Horley codes per glyph position)
# ---------------------------------------------------------------------------

def build_corpus_index(corpus_dir: Path) -> dict[str, dict]:
    """Return {position_key: {barthel_code, horley_code, horley_components}}.

    Position key: ``{TABLET}{SIDE}{LINE:02d}-{GLYPH_NUM:03d}``
    """
    index: dict[str, dict] = {}
    for path in sorted(corpus_dir.glob("[A-Z].json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        tid = data["tablet_id"]
        for g in data["glyphs"]:
            side = g.get("side", "?")
            line = g.get("line", "00")
            glyph_num = g.get("glyph_num", "000")
            key = f"{tid}{side}{line}-{int(glyph_num):03d}"
            index[key] = {
                "barthel_code": g.get("barthel_code"),
                "horley_code": g.get("horley_code"),
                "horley_components": g.get("horley_components"),
                "inverted": g.get("inverted", False),
                "uncertain": g.get("uncertain", False),
                "position": g.get("position"),
                "cluster": data.get("cluster", "unknown"),
            }
    return index


# ---------------------------------------------------------------------------
# SVG builder
# ---------------------------------------------------------------------------

_SVG_TMPL = """\
<svg xmlns="http://www.w3.org/2000/svg"
     width="{width:.1f}" height="{height:.1f}"
     viewBox="{x_min:.4f} {y_min:.4f} {vb_w:.4f} {vb_h:.4f}">
  <path d="{d}"/>
</svg>
"""


def build_svg(d_attr: str) -> str:
    """Wrap a single path `d` attribute in a standalone SVG with tight viewBox."""
    x_min, y_min, x_max, y_max = _approx_bbox(d_attr)
    vb_w = max(x_max - x_min, 1.0)
    vb_h = max(y_max - y_min, 1.0)
    # Canonical output size: 64 × 64 logical units (rescaled via viewBox)
    scale = 64.0 / max(vb_w, vb_h)
    return _SVG_TMPL.format(
        width=vb_w * scale,
        height=vb_h * scale,
        x_min=x_min,
        y_min=y_min,
        vb_w=vb_w,
        vb_h=vb_h,
        d=d_attr,
    )


# ---------------------------------------------------------------------------
# Scraper
# ---------------------------------------------------------------------------

def iter_tablet_paths(html: str) -> Iterator[tuple[str, str]]:
    """Yield (glyph_id, d_attribute) for every <path id="glyph..."> in *html*."""
    for m in _PATH_RE.finditer(html):
        yield m.group(1), m.group(2)


def scrape_tablet(
    tablet: str,
    corpus_index: dict[str, dict],
    out_dir: Path,
    dry_run: bool = False,
) -> list[dict]:
    """Scrape one tablet page; return list of catalog records."""
    url = BASE_URL.format(tablet=tablet)
    log.info("Fetching %s …", url)
    try:
        html = _fetch(url)
    except Exception as exc:
        log.error("  Failed to fetch %s: %s", tablet, exc)
        return []

    tablet_dir = out_dir / tablet
    if not dry_run:
        tablet_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict] = []
    n_written = 0

    for glyph_id, d_attr in iter_tablet_paths(html):
        parsed = parse_glyph_id(glyph_id)
        if parsed is None:
            log.debug("  Skipping unrecognised ID %s", glyph_id)
            continue

        pos_key = f"{parsed['tablet']}{parsed['side']}{parsed['line']}-{parsed['glyph_num']}"
        corpus_info = corpus_index.get(pos_key, {})

        svg_filename = f"{parsed['side']}{parsed['line']}-{parsed['glyph_num']}.svg"
        svg_path = tablet_dir / svg_filename

        record: dict = {
            "glyph_id": glyph_id,
            "tablet": parsed["tablet"],
            "side": parsed["side"],
            "line": parsed["line"],
            "glyph_num": parsed["glyph_num"],
            "svg_path": str(svg_path.relative_to(out_dir.parent)),
            "barthel_code": corpus_info.get("barthel_code"),
            "horley_code": corpus_info.get("horley_code"),
            "horley_components": corpus_info.get("horley_components"),
            "inverted": corpus_info.get("inverted", False),
            "cluster": corpus_info.get("cluster"),
        }
        records.append(record)

        if not dry_run:
            svg_content = build_svg(d_attr)
            svg_path.write_text(svg_content, encoding="utf-8")
            n_written += 1

    log.info(
        "  Tablet %s: %d glyphs found, %d SVGs written",
        tablet, len(records), n_written,
    )
    return records


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Scrape rongorongo glyph SVGs from kohaumotu.org.",
    )
    parser.add_argument(
        "--tablets",
        nargs="+",
        metavar="T",
        default=None,
        help=(
            "Tablet letters to scrape (default: all A–Y). "
            "Example: --tablets D O Q"
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse HTML and build catalog without writing SVG files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to write SVG files (default: data/glyphs/svg).",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=1.0,
        metavar="SECS",
        help="Seconds to sleep between tablet requests (default: 1.0).",
    )
    args = parser.parse_args()

    from omegaconf import OmegaConf
    cfg = OmegaConf.load(PROJECT_ROOT / "conf" / "config.yaml")
    corpus_dir = PROJECT_ROOT / cfg.paths.corpus_dir

    out_dir = args.output_dir or (PROJECT_ROOT / cfg.paths.glyphs_dir / "svg")
    if not args.dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)

    corpus_index = build_corpus_index(corpus_dir)
    log.info("Corpus index built: %d position records", len(corpus_index))

    tablets = args.tablets or ALL_TABLETS
    all_records: list[dict] = []

    for i, tablet in enumerate(tablets):
        if tablet not in ALL_TABLETS:
            log.warning("Unknown tablet %r — skipping", tablet)
            continue
        records = scrape_tablet(tablet, corpus_index, out_dir, dry_run=args.dry_run)
        all_records.extend(records)
        if i < len(tablets) - 1:
            time.sleep(args.delay)

    # Write catalog
    catalog_path = out_dir / "catalog.json"
    catalog = {
        "total_glyphs": len(all_records),
        "tablets_scraped": tablets,
        "records": all_records,
    }
    if not args.dry_run:
        catalog_path.write_text(json.dumps(catalog, indent=2), encoding="utf-8")
        log.info("Catalog written to %s (%d records)", catalog_path, len(all_records))
    else:
        log.info("DRY RUN complete — %d records (no files written)", len(all_records))
        # Print a sample
        for r in all_records[:5]:
            log.info("  Sample: %s → barthel=%s horley=%s", r["glyph_id"], r["barthel_code"], r["horley_code"])


if __name__ == "__main__":
    main()
