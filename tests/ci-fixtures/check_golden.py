#!/usr/bin/env python3
"""
Golden-file comparison for CI quantum script smoke tests.

Usage:
    python tests/ci-fixtures/check_golden.py \\
        --simon-out <path/to/simon_output.json> \\
        --bv-out    <path/to/bv_output.json>

Exits 0 on pass, 1 on any mismatch (with a diff printed to stderr).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_GOLDEN_DIR = Path(__file__).parent / "golden"
_SIMON_GOLDEN = _GOLDEN_DIR / "simon_P007_ADHS.json"
_BV_GOLDEN    = _GOLDEN_DIR / "bv_ic.json"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _check_simon(actual_path: Path) -> list[str]:
    raw = _load(actual_path)
    # Support both cumulative wrapper {"passages": {...}} and direct dict
    if "passages" in raw and isinstance(raw["passages"], dict):
        actual = raw["passages"].get("P007_ADHS", {})
    elif "passages" in raw and isinstance(raw["passages"], list):
        matches = [p for p in raw["passages"] if p.get("passage_id") == "P007_ADHS"]
        actual = matches[0] if matches else {}
    else:
        actual = raw
    golden = _load(_SIMON_GOLDEN)
    errors = []
    for key, expected in golden.items():
        got = actual.get(key)
        if got != expected:
            errors.append(f"  simon.{key}: expected {expected!r}, got {got!r}")
    return errors


def _check_bv(actual_path: Path) -> list[str]:
    raw = _load(actual_path)
    golden = _load(_BV_GOLDEN)
    errors = []

    # Flatten the BV result for comparison
    qr   = raw.get("quantum_result", {})
    lin  = raw.get("linearity", {})
    verd = raw.get("verdict", {})
    actual = {
        "recovered_s_int":   qr.get("recovered_s_int"),
        "recovered_s_bits":  qr.get("recovered_s_bits"),
        "matches_exact_s":   qr.get("matches_exact_s"),
        "affine_fraction":   lin.get("affine_fraction"),
        "is_null_result":    verd.get("is_null_result"),
        "f_type":            qr.get("f_type"),
    }
    for key, expected in golden.items():
        got = actual.get(key)
        if got != expected:
            errors.append(f"  bv.{key}: expected {expected!r}, got {got!r}")
    return errors


def main() -> int:
    p = argparse.ArgumentParser(description="CI golden-file comparator for quantum scripts.")
    p.add_argument("--simon-out", type=Path, required=True, metavar="JSON")
    p.add_argument("--bv-out",    type=Path, required=True, metavar="JSON")
    args = p.parse_args()

    all_errors: list[str] = []

    print("Checking Simon golden …")
    errs = _check_simon(args.simon_out)
    if errs:
        all_errors += ["SIMON MISMATCH:"] + errs
        print("  FAIL")
    else:
        print("  PASS (period_matches_observation=True, recovered_s_int=8)")

    print("Checking BV golden …")
    errs = _check_bv(args.bv_out)
    if errs:
        all_errors += ["BV MISMATCH:"] + errs
        print("  FAIL")
    else:
        print("  PASS (affine_fraction=1.0, recovered_s_int=16)")

    if all_errors:
        print("\nGOLDEN COMPARISON FAILED:", file=sys.stderr)
        for line in all_errors:
            print(line, file=sys.stderr)
        return 1

    print("\nAll golden checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
