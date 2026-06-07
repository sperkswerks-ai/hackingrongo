"""
tests.test_determinism
======================

Verify that running the same Ring 1 analysis step twice with an identical seed
produces bitwise-identical results.

Calls ``sensitivity_analysis()`` directly (no subprocess) so no corpus files
are required beyond what the entropy module already handles; the test is
skipped automatically when the corpus is absent.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_CORPUS_DIR = _PROJECT_ROOT / "data" / "corpus"


def _has_corpus() -> bool:
    return _CORPUS_DIR.is_dir() and any(_CORPUS_DIR.glob("*.json"))


@pytest.mark.skipif(not _has_corpus(), reason="corpus data not present")
def test_entropy_step_is_deterministic() -> None:
    """sensitivity_analysis() with the same seed produces bitwise-identical IC values."""
    from hackingrongo.repro import set_global_seed
    from hackingrongo.zone_b.entropy import sensitivity_analysis

    seed = 20260606
    scenarios = ["conservative_all_late"]

    set_global_seed(seed)
    result1 = sensitivity_analysis(scenarios=scenarios)

    set_global_seed(seed)
    result2 = sensitivity_analysis(scenarios=scenarios)

    # Serialise to JSON to get a canonical, bitwise-comparable representation.
    j1 = json.dumps(result1, sort_keys=True)
    j2 = json.dumps(result2, sort_keys=True)

    assert j1 == j2, (
        "sensitivity_analysis() returned different results on two calls with "
        f"seed={seed}. IC values are not deterministic."
    )
