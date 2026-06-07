"""
hackingrongo.provenance
=======================

Stamp every output JSON with a ``_provenance`` block so any result can be
traced back to exact code + data + hardware state.

Usage (non-quantum)
-------------------
    from hackingrongo.provenance import stamp
    result = {"key": "value", ...}
    stamp(result, seed=args.seed)
    output_path.write_text(json.dumps(result, indent=2))

Usage (quantum run)
-------------------
    from hackingrongo.provenance import stamp
    from hackingrongo.quantum_provenance import collect_provenance
    hw = collect_provenance(job, backend)
    stamp(result, seed=args.seed, quantum=hw)

Usage (post-hoc file stamping)
------------------------------
    from hackingrongo.provenance import stamp_file
    stamp_file(Path("outputs/ranking.json"), seed=_SEED)

Schema
------
::

    "_provenance": {
        "git_sha":       "<40-char hex or 'unknown'>",
        "timestamp_utc": "<ISO-8601 UTC>",
        "seed":          20260606,
        "python_version": "3.11.9",
        "packages": {
            "qiskit": "1.2.4",
            "qiskit-ibm-runtime": "0.23.0",
            "numpy": "1.26.4",
            "torch": "2.2.2",
            "networkx": "3.3",
            "scikit-learn": "1.5.0"
        },
        "corpus_sha256": "<sha256 of sorted corpus JSONs>",
        "quantum": {           // present only for quantum runs
            "job_id": "...",
            "backend_name": "ibm_marrakesh",
            "calibration_timestamp": "2025-11-14T08:30:00+00:00"
        }
    }
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import logging
import subprocess
import sys
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_CORPUS_DIR = _PROJECT_ROOT / "data" / "corpus"

_TRACKED_PACKAGES = (
    "qiskit",
    "qiskit-ibm-runtime",
    "numpy",
    "torch",
    "networkx",
    "scikit-learn",
)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def stamp(
    result: dict[str, Any],
    seed: int | None = None,
    quantum: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Add a ``_provenance`` block to *result* in-place and return it."""
    prov: dict[str, Any] = {
        "git_sha":        _git_sha(),
        "timestamp_utc":  datetime.now(tz=timezone.utc).isoformat(),
        "seed":           seed,
        "python_version": sys.version.split()[0],
        "packages":       _package_versions(),
        "corpus_sha256":  _corpus_checksum(),
    }
    if quantum is not None:
        prov["quantum"] = quantum
    result["_provenance"] = prov
    return result


def stamp_file(
    path: Path,
    seed: int | None = None,
    quantum: dict[str, Any] | None = None,
) -> None:
    """Load a JSON file, add ``_provenance``, and write it back atomically.

    Safe to call on files produced by serialisers that don't expose a
    pre-write hook (e.g. ``HypothesisRanking.save()``).
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        log.warning(
            "stamp_file: %s contains a %s, not a dict — wrapping in {'records': ...}",
            path, type(data).__name__,
        )
        data = {"records": data}
    stamp(data, seed=seed, quantum=quantum)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=_PROJECT_ROOT,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return "unknown"


def _package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for pkg in _TRACKED_PACKAGES:
        try:
            versions[pkg] = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            versions[pkg] = None
    return versions


@lru_cache(maxsize=1)
def _corpus_checksum() -> str:
    """SHA-256 of the byte-concatenation of sorted corpus JSON files."""
    if not _CORPUS_DIR.is_dir():
        return "corpus_dir_not_found"
    files = sorted(_CORPUS_DIR.glob("*.json"))
    if not files:
        return "no_corpus_json_files"
    h = hashlib.sha256()
    for p in files:
        h.update(p.read_bytes())
    return h.hexdigest()
