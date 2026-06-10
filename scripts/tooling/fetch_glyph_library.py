#!/usr/bin/env python3
"""
fetch_glyph_library.py — Download canonical Barthel reference GIFs from kohaumotu.org.

Fetches the 602 scholarly canonical sign drawings from the Glyph Library
(kohaumotu.org/Rongorongo/Glyph_Library/{001-099,...}/{NNN}.GIF) and
saves them as grayscale PNGs to data/glyphs/barthel_glyph_lib/.

These ~39×40 px images are black-ink-on-white scholarly drawings — one
per Barthel code number — covering all 602 attested canonical forms.
GlyphImageDataset uses this directory as a last-resort fallback after
barthel_ref/, barthel_corpus/, and 3d_crops/ all fail to resolve an
image for a corpus token.

SSL note: kohaumotu.org uses a self-signed certificate.  Verification is
intentionally disabled — same pattern as scrape_glyphs.py.

Usage
-----
    python scripts/tooling/fetch_glyph_library.py
    python scripts/tooling/fetch_glyph_library.py --dry-run
    python scripts/tooling/fetch_glyph_library.py --delay 0.3 --output data/glyphs/barthel_glyph_lib
"""
from __future__ import annotations

import argparse
import io
import logging
import re
import ssl
import sys
import time
import urllib.request
from pathlib import Path

from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

_DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "glyphs" / "barthel_glyph_lib"

_BASE_URL = "http://kohaumotu.org/Rongorongo/Glyph_Library"
_RANGES = [
    "001-099", "100-199", "200-299", "300-399",
    "400-499", "500-599", "600-699", "700-799",
]
_GIF_RE = re.compile(r'<img src="(\d{3}\.GIF)"', re.IGNORECASE)


# ---------------------------------------------------------------------------
# SSL context — intentionally bypasses verification for this host
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


def _fetch_bytes(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "hackingrongo-scraper/1.0"})
    with urllib.request.urlopen(req, context=_SSL_CTX, timeout=30) as r:
        return r.read()


def _fetch_text(url: str) -> str:
    return _fetch_bytes(url).decode("utf-8", errors="replace")


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

def discover_gif_urls() -> list[tuple[str, str]]:
    """Return list of (url, stem) for every GIF found on the range index pages.

    ``stem`` is the zero-padded 3-digit code, e.g. ``"001"``, ``"076"``.
    """
    results: list[tuple[str, str]] = []
    for rng in _RANGES:
        index_url = f"{_BASE_URL}/{rng}/index.html"
        log.info("Scanning %s …", index_url)
        try:
            html = _fetch_text(index_url)
        except Exception as exc:
            log.error("  Failed to fetch index for range %s: %s", rng, exc)
            continue
        gifs = _GIF_RE.findall(html)
        for gif_name in gifs:
            stem = gif_name.split(".")[0]
            url = f"{_BASE_URL}/{rng}/{gif_name}"
            results.append((url, stem))
        log.info("  %d GIFs found in %s", len(gifs), rng)
    return results


# ---------------------------------------------------------------------------
# Download and convert
# ---------------------------------------------------------------------------

def download_gifs(
    urls: list[tuple[str, str]],
    out_dir: Path,
    delay: float,
    skip_existing: bool = True,
) -> tuple[int, int, int]:
    """Fetch each GIF, convert to grayscale PNG, save to ``out_dir``.

    Returns (n_saved, n_skipped, n_errors).
    """
    n_saved = n_skipped = n_errors = 0
    for url, stem in urls:
        out_path = out_dir / f"{stem}.png"
        if skip_existing and out_path.exists():
            n_skipped += 1
            continue
        try:
            raw = _fetch_bytes(url)
        except Exception as exc:
            log.warning("  Failed to fetch %s: %s", url, exc)
            n_errors += 1
            continue
        try:
            img = Image.open(io.BytesIO(raw)).convert("L")
            img.save(out_path, format="PNG", optimize=True)
            n_saved += 1
        except Exception as exc:
            log.warning("  Failed to decode GIF at %s: %s", url, exc)
            n_errors += 1
            continue
        if delay > 0:
            time.sleep(delay)

    return n_saved, n_skipped, n_errors


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--output", type=Path, default=_DEFAULT_OUTPUT, metavar="DIR",
        help="Destination directory (default: data/glyphs/barthel_glyph_lib).",
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Discover URLs but do not download any files.",
    )
    p.add_argument(
        "--delay", type=float, default=0.2, metavar="SECS",
        help="Pause between HTTP requests (default: 0.2 s).",
    )
    p.add_argument(
        "--force", action="store_true",
        help="Re-download even if the PNG already exists.",
    )
    args = p.parse_args()

    gif_urls = discover_gif_urls()
    log.info("Total GIF URLs discovered: %d", len(gif_urls))

    if args.dry_run:
        log.info("DRY RUN — no files written.  Sample URLs:")
        for url, stem in gif_urls[:5]:
            log.info("  %s → %s.png", url, stem)
        return

    args.output.mkdir(parents=True, exist_ok=True)
    n_saved, n_skipped, n_errors = download_gifs(
        gif_urls, args.output, delay=args.delay, skip_existing=not args.force
    )
    log.info(
        "Done: %d saved, %d already existed, %d errors → %s",
        n_saved, n_skipped, n_errors, args.output,
    )
    if n_errors:
        sys.exit(1)


if __name__ == "__main__":
    main()
