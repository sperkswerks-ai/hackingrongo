"""
tests.test_self_training
========================

Unit tests for:
  (a) write_self_training_report() — produces valid HTML from synthetic
      iter_NN/ranking.json files.
  (b) _compute_stability() — correctly classifies stable, unstable, and
      partial signs across 2 synthetic iterations.

No corpus data or heavy ML dependencies required.
"""

from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Import helpers from run_self_training without executing main()
# ---------------------------------------------------------------------------

# Add scripts/ to the path so we can import run_self_training directly
_SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from run_self_training import (  # noqa: E402
    IterationResult,
    Promotion,
    SelfTrainingState,
    _compute_stability,
    write_self_training_report,
)


# ---------------------------------------------------------------------------
# Synthetic data builders
# ---------------------------------------------------------------------------

def _make_ranking(
    hypotheses: list[dict],
) -> dict:
    """Build a minimal ranking.json structure."""
    return {"hypotheses": hypotheses}


def _make_hypothesis(
    lm_score: float,
    assignments: list[dict],
) -> dict:
    return {
        "hypothesis_id": "H0001",
        "overall_lm_score": lm_score,
        "assignments": assignments,
    }


def _make_assignment(sign_code: str, phoneme: str, confidence: float = 0.90) -> dict:
    return {"sign_code": sign_code, "phoneme": phoneme, "confidence": confidence}


def _write_ranking_files(tmp_path: Path, iter_data: list[dict]) -> None:
    """Write iter_NN/ranking.json files into tmp_path."""
    for i, data in enumerate(iter_data):
        d = tmp_path / f"iter_{i:02d}"
        d.mkdir(parents=True, exist_ok=True)
        (d / "ranking.json").write_text(json.dumps(data), encoding="utf-8")


def _make_state_with_promotions() -> SelfTrainingState:
    """Build a minimal SelfTrainingState with two iterations and some promotions."""
    state = SelfTrainingState()
    soft_p = Promotion(
        sign="001", phoneme="ma", confidence=0.85,
        consensus_count=3, top_k=5, iteration=0, kind="soft",
    )
    hard_p = Promotion(
        sign="002", phoneme="ku", confidence=0.92,
        consensus_count=5, top_k=5, iteration=1, kind="hard",
    )
    state.history = [
        IterationResult(
            iteration=0,
            top_lm_score=-1.2345,
            n_hard_cribs=4,
            n_soft_anchors=0,
            new_soft=[soft_p],
            new_hard=[],
            mcmc_converged=False,
            rhat=1.15,
            acceptance_mean=0.234,
            n_hypotheses_scored=10,
        ),
        IterationResult(
            iteration=1,
            top_lm_score=-1.1200,
            n_hard_cribs=4,
            n_soft_anchors=1,
            new_soft=[],
            new_hard=[hard_p],
            mcmc_converged=True,
            rhat=1.05,
            acceptance_mean=0.240,
            n_hypotheses_scored=10,
        ),
    ]
    state.hard_cribs = {"002": "ku"}
    state.soft_anchors = {"001": ("ma", 0.85)}
    state.convergence = "no_new_promotions"
    return state


# ---------------------------------------------------------------------------
# Test (a): write_self_training_report() produces valid HTML
# ---------------------------------------------------------------------------

class TestWriteSelfTrainingReport:
    def test_produces_html_file(self, tmp_path: Path) -> None:
        """write_self_training_report() creates self_training_report.html."""
        # Write synthetic ranking files so _compute_stability() has data
        iter0_hyp = _make_hypothesis(-1.2, [
            _make_assignment("001", "ma"),
            _make_assignment("002", "ku"),
            _make_assignment("003", "ri"),
        ])
        iter1_hyp = _make_hypothesis(-1.1, [
            _make_assignment("001", "ma"),
            _make_assignment("002", "ku"),
            _make_assignment("003", "ta"),   # unstable: changes from iter_00
        ])
        _write_ranking_files(tmp_path, [
            _make_ranking([iter0_hyp]),
            _make_ranking([iter1_hyp]),
        ])

        state = _make_state_with_promotions()
        generated = "2026-06-05T00:00:00Z"
        args_meta = {
            "max_iterations": 4,
            "top_k": 5,
            "min_consensus": 2,
            "min_evidence": 10,
            "threshold_start": 0.90,
            "threshold_end": 0.70,
            "smoke_test": False,
        }

        report_path = write_self_training_report(state, generated, args_meta, tmp_path)

        assert report_path.exists(), "Report file was not created"
        html = report_path.read_text(encoding="utf-8")
        assert html.strip().startswith("<!DOCTYPE html"), "Output is not HTML"

    def test_html_contains_required_sections(self, tmp_path: Path) -> None:
        """Report HTML includes LM chart SVG, confidence table, stability stats."""
        iter0_hyp = _make_hypothesis(-1.5, [
            _make_assignment("010", "a"),
            _make_assignment("020", "e"),
        ])
        iter1_hyp = _make_hypothesis(-1.3, [
            _make_assignment("010", "a"),   # stable
            _make_assignment("020", "i"),   # unstable
        ])
        _write_ranking_files(tmp_path, [
            _make_ranking([iter0_hyp]),
            _make_ranking([iter1_hyp]),
        ])

        state = _make_state_with_promotions()
        report_path = write_self_training_report(
            state, "2026-06-05T00:00:00Z",
            {"max_iterations": 2, "top_k": 5, "min_consensus": 2,
             "min_evidence": 5, "threshold_start": 0.9, "threshold_end": 0.7,
             "smoke_test": True},
            tmp_path,
        )
        html = report_path.read_text(encoding="utf-8")

        assert "<svg" in html, "LM chart SVG is missing"
        assert "<polyline" in html, "SVG chart polyline is missing"
        assert "Promotion Confidence Trajectories" in html, "Confidence section heading missing"
        assert "Assignment Stability" in html, "Stability section heading missing"
        # stability numbers present
        assert "Stable" in html
        assert "Unstable" in html

    def test_stability_json_written(self, tmp_path: Path) -> None:
        """write_self_training_report() also writes stability_analysis.json."""
        iter0_hyp = _make_hypothesis(-2.0, [_make_assignment("099", "na")])
        iter1_hyp = _make_hypothesis(-1.9, [_make_assignment("099", "na")])
        _write_ranking_files(tmp_path, [
            _make_ranking([iter0_hyp]),
            _make_ranking([iter1_hyp]),
        ])
        state = SelfTrainingState()
        state.convergence = "max_iterations"

        write_self_training_report(state, "2026-06-05T00:00:00Z", {}, tmp_path)

        stab_path = tmp_path / "stability_analysis.json"
        assert stab_path.exists(), "stability_analysis.json was not written"
        data = json.loads(stab_path.read_text())
        assert "stable" in data and "unstable" in data and "partial" in data

    def test_no_iterations_still_produces_html(self, tmp_path: Path) -> None:
        """Empty state (0 iterations) should still produce a valid HTML file."""
        state = SelfTrainingState()
        state.convergence = "no_hypotheses"
        report_path = write_self_training_report(
            state, "2026-06-05T00:00:00Z", {}, tmp_path
        )
        html = report_path.read_text(encoding="utf-8")
        assert "<!DOCTYPE html" in html


# ---------------------------------------------------------------------------
# Test (b): _compute_stability() classifies signs correctly
# ---------------------------------------------------------------------------

class TestComputeStability:
    def test_stable_signs(self, tmp_path: Path) -> None:
        """Signs with identical phoneme in both iterations are classified stable."""
        iter0_hyp = _make_hypothesis(-1.0, [
            _make_assignment("A", "ma", 0.95),
            _make_assignment("B", "ku", 0.88),
        ])
        iter1_hyp = _make_hypothesis(-0.9, [
            _make_assignment("A", "ma", 0.96),
            _make_assignment("B", "ku", 0.90),
        ])
        _write_ranking_files(tmp_path, [
            _make_ranking([iter0_hyp]),
            _make_ranking([iter1_hyp]),
        ])

        result = _compute_stability(tmp_path)

        assert result["n_runs"] == 2
        stable_signs = {e["sign_code"] for e in result["stable"]}
        assert "A" in stable_signs
        assert "B" in stable_signs
        assert result["unstable"] == []
        assert result["partial"] == []

    def test_unstable_signs(self, tmp_path: Path) -> None:
        """Signs that change phoneme between iterations are classified unstable."""
        iter0_hyp = _make_hypothesis(-1.0, [
            _make_assignment("X", "ta", 0.80),
        ])
        iter1_hyp = _make_hypothesis(-0.9, [
            _make_assignment("X", "ko", 0.75),   # different phoneme
        ])
        _write_ranking_files(tmp_path, [
            _make_ranking([iter0_hyp]),
            _make_ranking([iter1_hyp]),
        ])

        result = _compute_stability(tmp_path)

        unstable_signs = {e["sign_code"] for e in result["unstable"]}
        assert "X" in unstable_signs
        assert result["stable"] == []

    def test_partial_signs(self, tmp_path: Path) -> None:
        """Signs absent from one iteration are classified partial."""
        iter0_hyp = _make_hypothesis(-1.0, [
            _make_assignment("P", "re", 0.70),
            _make_assignment("Q", "nu", 0.65),
        ])
        iter1_hyp = _make_hypothesis(-0.8, [
            _make_assignment("Q", "nu", 0.68),
            # P is absent from iter_01
        ])
        _write_ranking_files(tmp_path, [
            _make_ranking([iter0_hyp]),
            _make_ranking([iter1_hyp]),
        ])

        result = _compute_stability(tmp_path)

        partial_signs = {e["sign_code"] for e in result["partial"]}
        assert "P" in partial_signs
        stable_signs = {e["sign_code"] for e in result["stable"]}
        assert "Q" in stable_signs

    def test_mixed_classification(self, tmp_path: Path) -> None:
        """Correctly handles stable, unstable, and partial signs together."""
        iter0_hyp = _make_hypothesis(-1.2, [
            _make_assignment("STABLE", "ma"),
            _make_assignment("UNSTABLE", "ti"),
            _make_assignment("ONLY_0", "wa"),
        ])
        iter1_hyp = _make_hypothesis(-1.0, [
            _make_assignment("STABLE", "ma"),
            _make_assignment("UNSTABLE", "ri"),  # changes
            # ONLY_0 absent
        ])
        _write_ranking_files(tmp_path, [
            _make_ranking([iter0_hyp]),
            _make_ranking([iter1_hyp]),
        ])

        result = _compute_stability(tmp_path)

        stable_codes   = {e["sign_code"] for e in result["stable"]}
        unstable_codes = {e["sign_code"] for e in result["unstable"]}
        partial_codes  = {e["sign_code"] for e in result["partial"]}

        assert "STABLE"   in stable_codes
        assert "UNSTABLE" in unstable_codes
        assert "ONLY_0"   in partial_codes
        assert len(stable_codes) + len(unstable_codes) + len(partial_codes) == 3

    def test_no_ranking_files_returns_empty(self, tmp_path: Path) -> None:
        """Returns empty result dict when no iter_NN/ranking.json files exist."""
        result = _compute_stability(tmp_path)
        assert result["n_runs"] == 0
        assert result["stable"] == []
        assert result["unstable"] == []
        assert result["lm_scores"] == {}

    def test_lm_scores_recorded(self, tmp_path: Path) -> None:
        """LM scores for each iteration are recorded in result dict."""
        iter0_hyp = _make_hypothesis(-2.5, [_make_assignment("S", "a")])
        iter1_hyp = _make_hypothesis(-2.1, [_make_assignment("S", "a")])
        _write_ranking_files(tmp_path, [
            _make_ranking([iter0_hyp]),
            _make_ranking([iter1_hyp]),
        ])

        result = _compute_stability(tmp_path)

        assert "iter_00" in result["lm_scores"]
        assert "iter_01" in result["lm_scores"]
        assert math.isclose(result["lm_scores"]["iter_00"], -2.5)
        assert math.isclose(result["lm_scores"]["iter_01"], -2.1)
