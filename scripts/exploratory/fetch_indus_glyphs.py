"""
EXPLORATORY — speculative / tangential analysis; not part of the reproducible analysis pipeline.

fetch_indus_glyphs.py

Downloads the Mahadevan (1977) Indus Valley sign list images from the archive
specified in data/glyphs/indus/sources.yaml.  Skips signs already downloaded.

Usage:
    python scripts/fetch_indus_glyphs.py [--sources data/glyphs/indus/sources.yaml]
                                          [--out-dir data/glyphs/indus]
                                          [--size 64]
"""

from __future__ import annotations

import argparse
import io
import logging
import sys
import time
import urllib.request
from pathlib import Path

import yaml
from PIL import Image

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _fetch_one(url: str, out_path: Path, size: int, retries: int = 3) -> bool:
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "hackingrongo/1.0"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = resp.read()
            img = Image.open(io.BytesIO(data)).convert("L")
            img = img.resize((size, size), Image.LANCZOS)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            img.save(out_path)
            return True
        except Exception as exc:
            if attempt < retries - 1:
                log.warning("Attempt %d failed for %s: %s — retrying", attempt + 1, url, exc)
                time.sleep(2 ** attempt)
            else:
                log.error("Failed to fetch %s after %d attempts: %s", url, retries, exc)
    return False


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Download Indus Valley glyph images")
    parser.add_argument(
        "--sources",
        type=Path,
        default=PROJECT_ROOT / "data" / "glyphs" / "indus" / "sources.yaml",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "glyphs" / "indus",
    )
    parser.add_argument("--size", type=int, default=64)
    args = parser.parse_args(argv)

    if not args.sources.exists():
        log.error("Sources file not found: %s", args.sources)
        sys.exit(1)

    cfg = yaml.safe_load(args.sources.read_text(encoding="utf-8"))
    base_url: str = cfg.get("archive_base_url", "")
    sign_list: list[dict] = cfg.get("sign_list", [])
    size: int = args.size or int(cfg.get("image_size", 64))

    log.info("Fetching %d Indus Valley signs → %s", len(sign_list), args.out_dir)

    n_skipped = n_ok = n_fail = 0
    for entry in sign_list:
        number: str = entry["number"]
        filename: str = entry["filename"]
        out_path = args.out_dir / filename

        if out_path.exists():
            n_skipped += 1
            continue

        url: str = entry.get("url") or f"{base_url}/{number.lower()}.png"
        ok = _fetch_one(url, out_path, size)
        if ok:
            n_ok += 1
            log.info("  %s → %s", number, filename)
        else:
            n_fail += 1

    log.info(
        "Done. Downloaded: %d  Skipped (already present): %d  Failed: %d",
        n_ok, n_skipped, n_fail,
    )
    if n_fail > 0:
        log.warning(
            "%d sign(s) could not be fetched. Check sources.yaml URLs and network access.",
            n_fail,
        )


if __name__ == "__main__":
    main()
