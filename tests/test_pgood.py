"""
tests.test_pgood
================

Tests for Layer 5Q (measure_pgood.py) and Layer 4Q (run_qubo_decipherment.py).

Tests 1–2 invoke the scripts as subprocesses against the real corpus and
language models; they are skipped when those data files are absent.

Tests 3–4 use self-contained synthetic data and are always runnable:
  - test 3 tests the QUBO matrix construction purely in-memory
  - test 4 requires neal/dimod and is skipped if they are not installed
"""

from __future__ import annotations

import importlib.util
import json
import math
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
_CORPUS_DIR  = PROJECT_ROOT / "data" / "corpus"
_LM_DIR      = PROJECT_ROOT / "data" / "language_models"
_SCRIPT_PGOOD = PROJECT_ROOT / "scripts" / "measure_pgood.py"
_SCRIPT_QUBO  = PROJECT_ROOT / "scripts" / "run_qubo_decipherment.py"


# ---------------------------------------------------------------------------
# Skip helpers
# ---------------------------------------------------------------------------

def _has_corpus() -> bool:
    return _CORPUS_DIR.exists() and any(_CORPUS_DIR.glob("*.json"))


def _has_lms() -> bool:
    return (
        _LM_DIR.exists()
        and (_LM_DIR / "pre_contact_lm.json").exists()
        and (_LM_DIR / "post_contact_lm.json").exists()
    )


def _has_solver() -> bool:
    return (
        importlib.util.find_spec("neal") is not None
        or importlib.util.find_spec("dimod") is not None
    )


# ---------------------------------------------------------------------------
# Synthetic LM fixture (used by tests 3 and 4)
# ---------------------------------------------------------------------------

_SYNTHETIC_SIGNS    = [f"s{i:02d}" for i in range(10)]
_SYNTHETIC_PHONEMES = ["ku", "ma", "ri", "ta", "ko"]

_SYNTHETIC_SEQUENCES = [
    ["ku", "ma", "ri", "ta", "ko"],
    ["ku", "ri", "ma", "ko", "ta"],
    ["ma", "ta", "ku", "ri", "ko"],
    ["ta", "ko", "ma", "ri", "ku"],
]


def _build_synthetic_lm(path: Path) -> None:
    """Write a tiny NGramLM trained on _SYNTHETIC_SEQUENCES to path."""
    from hackingrongo.data.rapa_nui_corpus import NGramLM
    lm = NGramLM(order=2, language="rapa_nui")
    for seq in _SYNTHETIC_SEQUENCES:
        lm.update(seq)
    lm.finalise()
    lm.save(path)


def _build_synthetic_corpus(corpus_dir: Path) -> None:
    """Write a tiny tablet JSON using _SYNTHETIC_SIGNS as Barthel codes."""
    tablet = {
        "tablet": "synthetic",
        "glyphs": [
            {"barthel_code": sign, "side": "a", "line": 1, "glyph_num": i + 1}
            for i, sign in enumerate(_SYNTHETIC_SIGNS * 2)
        ],
    }
    (corpus_dir / "synthetic.json").write_text(
        json.dumps(tablet), encoding="utf-8"
    )


# ---------------------------------------------------------------------------
# Test 1: smoke test of measure_pgood.py against real data
# ---------------------------------------------------------------------------

class TestPgoodSmoke:
    def test_pgood_smoke(self, tmp_path):
        """measure_pgood.py --smoke-test produces valid JSON with correct schema."""
        if not _has_corpus():
            pytest.skip("Corpus data not available.")
        if not _has_lms():
            pytest.skip("Language models not available — run build_language_models.py first.")

        output = tmp_path / "pgood_smoke.json"
        proc = subprocess.run(
            [
                sys.executable, str(_SCRIPT_PGOOD),
                "--smoke-test",
                "--corpus-dir", str(_CORPUS_DIR),
                "--lm-dir",     str(_LM_DIR),
                "--output",     str(output),
            ],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert proc.returncode == 0, (
            f"measure_pgood.py --smoke-test failed (exit {proc.returncode}):\n"
            f"{proc.stderr[-2000:]}"
        )
        assert output.exists(), "Output JSON was not written."

        result = json.loads(output.read_text(encoding="utf-8"))

        # Top-level schema
        assert result["n_samples"] == 100
        assert result["n_finite"] > 0
        assert result["n_signs"] > 0
        assert result["n_phonemes"] > 0
        assert "score_distribution" in result
        assert "thresholds" in result
        assert "interpretation" in result
        assert "sampling_time_seconds" in result

        dist = result["score_distribution"]
        for key in ("mean", "std", "min", "max", "percentiles"):
            assert key in dist, f"score_distribution missing key: {key}"
        assert math.isfinite(dist["mean"])
        assert dist["std"] >= 0.0
        assert dist["min"] <= dist["max"]

        # Per-threshold schema
        taus_seen = []
        for t in result["thresholds"]:
            for key in ("tau", "p_good", "grover_oracle_calls",
                        "classical_random_calls", "quantum_speedup_ratio",
                        "mcmc_iterations", "mcmc_vs_grover_ratio"):
                assert key in t, f"threshold entry missing key: {key}"

            p_good = t["p_good"]
            assert 0.0 <= p_good <= 1.0, f"p_good out of range: {p_good}"

            gc = t["grover_oracle_calls"]
            cc = t["classical_random_calls"]
            if p_good > 0 and gc > 0 and cc > 0:
                assert gc < cc, (
                    f"Grover calls ({gc}) should be < classical calls ({cc}) "
                    f"for tau={t['tau']}"
                )

            taus_seen.append(t["tau"])

        # All requested thresholds present
        assert set(taus_seen) >= {0.90, 0.95, 0.99}


# ---------------------------------------------------------------------------
# Test 2: p_good is monotonically decreasing with increasing threshold
# ---------------------------------------------------------------------------

class TestPgoodMonotone:
    def test_pgood_monotone(self, tmp_path):
        """Higher tau → lower p_good across all three default thresholds."""
        if not _has_corpus():
            pytest.skip("Corpus data not available.")
        if not _has_lms():
            pytest.skip("Language models not available.")

        output = tmp_path / "pgood_monotone.json"
        proc = subprocess.run(
            [
                sys.executable, str(_SCRIPT_PGOOD),
                "--smoke-test",          # 100 samples — fast enough
                "--corpus-dir", str(_CORPUS_DIR),
                "--lm-dir",     str(_LM_DIR),
                "--thresholds", "0.90,0.95,0.99",
                "--output",     str(output),
            ],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert proc.returncode == 0, proc.stderr[-2000:]
        result = json.loads(output.read_text(encoding="utf-8"))

        by_tau = {t["tau"]: t["p_good"] for t in result["thresholds"]}
        assert set(by_tau) >= {0.90, 0.95, 0.99}

        # Monotone: tau=0.90 ≥ tau=0.95 ≥ tau=0.99
        # (equal is fine when p_good=0 for all three, i.e. no good samples at all)
        assert by_tau[0.90] >= by_tau[0.95], (
            f"p_good[0.90]={by_tau[0.90]:.4e} < p_good[0.95]={by_tau[0.95]:.4e} "
            "— violates monotonicity"
        )
        assert by_tau[0.95] >= by_tau[0.99], (
            f"p_good[0.95]={by_tau[0.95]:.4e} < p_good[0.99]={by_tau[0.99]:.4e} "
            "— violates monotonicity"
        )


# ---------------------------------------------------------------------------
# Test 3: QUBO matrix structural properties (no solver, no data files)
# ---------------------------------------------------------------------------

class TestQuboBuildMatrix:
    def test_qubo_builds(self, tmp_path):
        """build_qubo produces a matrix with correct structural properties."""
        spec = importlib.util.spec_from_file_location("rqd", str(_SCRIPT_QUBO))
        rqd  = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(rqd)

        lm_path = tmp_path / "lm.json"
        _build_synthetic_lm(lm_path)

        from hackingrongo.data.rapa_nui_corpus import NGramLM
        lm = NGramLM.load(lm_path)

        signs    = _SYNTHETIC_SIGNS
        phonemes = _SYNTHETIC_PHONEMES
        n_s = len(signs)
        n_p = len(phonemes)

        Q = rqd.build_qubo(
            signs, phonemes, [lm], _SYNTHETIC_SEQUENCES,
            lambda1=10.0, lambda2=5.0, max_per_phoneme=5,
        )

        # ── Symmetry: upper-triangular keys only (i ≤ j) ─────────────────────
        for (i, j) in Q:
            assert i <= j, f"Non-upper-triangular key ({i}, {j}) in QUBO."

        # ── All variables represented ──────────────────────────────────────────
        all_vars = {rqd._var(s, p, n_p) for s in range(n_s) for p in range(n_p)}
        diag_vars = {i for (i, j) in Q if i == j}
        assert diag_vars == all_vars, (
            f"Missing diagonal entries: {all_vars - diag_vars}"
        )

        # ── Diagonal terms are negative ───────────────────────────────────────
        # Objective: minimise H → maximise LM score → diagonal = -unigram - λ1
        # Both contributions are negative so the combined diagonal < 0.
        for (i, j), val in Q.items():
            if i == j:
                assert val < 0, (
                    f"Diagonal Q[{i},{i}] = {val:.4f} should be negative "
                    "(objective + one-hot penalty both contribute negative diagonal)."
                )

        # ── Off-diagonal one-hot terms are positive ───────────────────────────
        # Pairs (var(s,p), var(s,q)) for same sign s, different phonemes p≠q
        # get +2*λ1 = +20.0.  Check a sample of such pairs.
        for s in range(n_s):
            for p in range(n_p):
                for q in range(p + 1, n_p):
                    key = (rqd._var(s, p, n_p), rqd._var(s, q, n_p))
                    assert key in Q, f"Missing one-hot penalty key {key}."
                    assert Q[key] > 0, (
                        f"One-hot penalty Q{key} = {Q[key]:.4f} should be positive."
                    )

        # ── Off-diagonal capacity terms are positive ──────────────────────────
        # Pairs (var(s,p), var(t,p)) for same phoneme p, different signs s≠t
        # get +2*λ2 = +10.0.  Check a sample of such pairs.
        for p in range(n_p):
            for s in range(n_s):
                for t in range(s + 1, n_s):
                    key = (rqd._var(s, p, n_p), rqd._var(t, p, n_p))
                    assert key in Q, f"Missing capacity penalty key {key}."
                    assert Q[key] > 0, (
                        f"Capacity penalty Q{key} = {Q[key]:.4f} should be positive."
                    )

        # ── Total variable count ───────────────────────────────────────────────
        expected_vars = n_s * n_p
        assert len(diag_vars) == expected_vars


# ---------------------------------------------------------------------------
# Test 4: end-to-end QUBO solve with neal on synthetic data
# ---------------------------------------------------------------------------

class TestQuboNealSolves:
    def test_qubo_neal_solves(self, tmp_path):
        """run_qubo_decipherment.py --solver neal produces a valid result JSON."""
        if not _has_solver():
            pytest.skip("Neither neal nor dimod is installed.")

        lm_dir     = tmp_path / "lm_dir"
        corpus_dir = tmp_path / "corpus"
        lm_dir.mkdir()
        corpus_dir.mkdir()

        _build_synthetic_lm(lm_dir / "pre_contact_lm.json")
        _build_synthetic_lm(lm_dir / "post_contact_lm.json")
        _build_synthetic_corpus(corpus_dir)

        output = tmp_path / "qubo_result.json"
        proc = subprocess.run(
            [
                sys.executable, str(_SCRIPT_QUBO),
                "--corpus-dir",  str(corpus_dir),
                "--lm-dir",      str(lm_dir),
                "--solver",      "neal" if importlib.util.find_spec("neal") else "tabu",
                "--num-reads",   "10",
                "--output",      str(output),
            ],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert proc.returncode == 0, (
            f"run_qubo_decipherment.py failed (exit {proc.returncode}):\n"
            f"{proc.stderr[-2000:]}"
        )
        assert output.exists(), "qubo_result.json was not written."

        result = json.loads(output.read_text(encoding="utf-8"))

        # Top-level schema
        for key in ("solver", "n_reads", "best_energy", "best_lm_score",
                    "phoneme_assignments", "annealing_time_seconds", "qubo_size"):
            assert key in result, f"qubo_result.json missing key: {key}"

        assert result["n_reads"] == 10
        assert result["solver"] in ("neal", "tabu", "dwave")
        assert result["annealing_time_seconds"] >= 0.0

        best_lm = result["best_lm_score"]
        assert best_lm is not None and math.isfinite(best_lm), (
            f"best_lm_score should be a finite float, got {best_lm!r}"
        )

        qs = result["qubo_size"]
        assert qs["variables"] == len(_SYNTHETIC_SIGNS) * len(_SYNTHETIC_PHONEMES)
        assert qs["couplings"] > 0

        # Phoneme assignments schema
        assignments = result["phoneme_assignments"]
        assert len(assignments) == len(_SYNTHETIC_SIGNS), (
            f"Expected {len(_SYNTHETIC_SIGNS)} assignments, got {len(assignments)}"
        )
        for a in assignments:
            assert "sign_code" in a
            assert "phoneme" in a
            assert "confidence" in a
            assert a["sign_code"] in _SYNTHETIC_SIGNS
            assert 0.0 <= a["confidence"] <= 1.0, (
                f"Confidence out of range for {a['sign_code']}: {a['confidence']}"
            )
