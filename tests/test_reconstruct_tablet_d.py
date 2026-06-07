"""
Smoke tests for scripts/exploratory/reconstruct_tablet_d.py.

These tests exercise the importable logic only — no corpus data required.
"""

import sys
from pathlib import Path

# Ensure project root is on sys.path so `scripts.exploratory.*` imports work.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


def test_convergence_score_logic():
    """Top-1 hit + high MCMC confidence → convergent (>= 0.7)."""
    from scripts.exploratory.reconstruct_tablet_d import compute_convergence_score

    score = compute_convergence_score(
        seq_top_k=[{"sign": "536", "rank": 1}],
        mcmc_phoneme="me",
        mcmc_confidence=0.9,
    )
    assert score >= 0.7, f"Expected score >= 0.7, got {score}"


def test_convergence_score_no_match():
    """Empty evidence → score 0.0."""
    from scripts.exploratory.reconstruct_tablet_d import compute_convergence_score

    score = compute_convergence_score(
        seq_top_k=[],
        mcmc_phoneme=None,
        mcmc_confidence=None,
    )
    assert score == 0.0, f"Expected 0.0, got {score}"


def test_target_type_classification():
    """classify_target_type returns correct labels for each marker pattern."""
    from scripts.exploratory.reconstruct_tablet_d import classify_target_type

    assert classify_target_type("536?") == "uncertain"
    assert classify_target_type("(10-20)!") == "range"
    assert classify_target_type("050V") == "variant"
    assert classify_target_type("007") is None
