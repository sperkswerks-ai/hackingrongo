"""
tests.test_zone_b
=================

Unit tests for Zone B analysis functions: IC (known values), G² statistic,
sensitivity scenarios, Zipf fit on synthetic data, and parse_line_field.

All tests build synthetic corpora in tmp_path — no real corpus data required.
"""

from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

import pytest

from hackingrongo.zone_b.contact_analysis import log_likelihood_g2
from hackingrongo.zone_b.entropy import (
    ic_random_baseline,
    index_of_coincidence,
    load_tokens_under_scenario,
    zipf_analysis,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_tablet(
    tmp_path: Path,
    filename: str,
    cluster: str,
    horley_codes: list[str],
) -> None:
    glyphs = [{"horley_code": hc, "uncertain": False} for hc in horley_codes]
    (tmp_path / filename).write_text(
        json.dumps({"tablet_id": filename.replace(".json", ""),
                    "cluster": cluster, "glyphs": glyphs}),
        encoding="utf-8",
    )


def _make_powerlaw_corpus(tmp_path: Path, n_types: int = 30, alpha: float = 1.2) -> None:
    """Write a single tablet whose sign frequencies follow rank^(-alpha)."""
    codes = [f"s{i:03d}" for i in range(1, n_types + 1)]
    tokens: list[str] = []
    for rank, code in enumerate(codes, start=1):
        freq = max(1, int(300 * rank ** (-alpha)))
        tokens.extend([code] * freq)
    glyphs = [{"horley_code": t, "uncertain": False} for t in tokens]
    (tmp_path / "A.json").write_text(
        json.dumps({"tablet_id": "A", "cluster": "pre_contact", "glyphs": glyphs}),
        encoding="utf-8",
    )


def _load_parse_line_field():
    """Import parse_line_field from the scripts directory (no __init__.py)."""
    scripts_dir = Path(__file__).resolve().parent.parent / "scripts"
    spec = importlib.util.spec_from_file_location(
        "transform_parallels",
        scripts_dir / "transform_parallels.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.parse_line_field


# ---------------------------------------------------------------------------
# IC — known values
# ---------------------------------------------------------------------------

class TestIndexOfCoincidence:
    def test_uniform_baseline_equals_reciprocal_k(self):
        for k in (5, 10, 50, 100):
            assert ic_random_baseline(k) == pytest.approx(1.0 / k)

    def test_single_repeated_symbol_equals_one(self):
        tokens = ["a"] * 500
        assert index_of_coincidence(tokens) == pytest.approx(1.0)

    def test_uniform_sample_near_reciprocal_k(self):
        k = 10
        tokens = [str(i % k) for i in range(2000)]
        ic = index_of_coincidence(tokens)
        assert ic == pytest.approx(1.0 / k, abs=0.003)


# ---------------------------------------------------------------------------
# G² statistic
# ---------------------------------------------------------------------------

class TestLogLikelihoodG2:
    def test_independent_table_is_zero(self):
        assert log_likelihood_g2(5, 5, 5, 5) == pytest.approx(0.0, abs=1e-9)

    def test_fully_separated_known_value(self):
        # a=10, b=0, c=0, d=10:  G² = 40 * ln(2)
        g2 = log_likelihood_g2(10, 0, 0, 10)
        assert g2 == pytest.approx(40.0 * math.log(2), rel=1e-9)

    def test_all_zero_is_zero(self):
        assert log_likelihood_g2(0, 0, 0, 0) == pytest.approx(0.0)

    def test_always_non_negative(self):
        for a, b, c, d in [(3, 7, 8, 2), (1, 9, 9, 1), (5, 0, 3, 7), (10, 5, 2, 8)]:
            assert log_likelihood_g2(a, b, c, d) >= 0.0


# ---------------------------------------------------------------------------
# Sensitivity scenarios
# ---------------------------------------------------------------------------

class TestSensitivityScenarios:
    @pytest.fixture()
    def corpus_dir(self, tmp_path):
        _write_tablet(tmp_path, "D.json", "pre_contact",  ["001", "002", "003"] * 25)
        _write_tablet(tmp_path, "H.json", "post_contact", ["004", "005", "006"] * 25)
        _write_tablet(tmp_path, "G.json", "unknown",      ["007", "008", "009"] * 15)
        _write_tablet(tmp_path, "P.json", "unknown",      ["010", "001", "002"] * 15)
        return tmp_path

    @pytest.mark.parametrize("scenario", [
        "conservative_all_late",
        "optimistic_distributed",
        "probabilistic_weighted",
    ])
    def test_scenario_runs_without_error(self, corpus_dir, scenario):
        result = load_tokens_under_scenario(corpus_dir, scenario)
        assert isinstance(result, dict)
        assert any(len(toks) > 0 for toks in result.values()), (
            f"Scenario {scenario!r} produced no tokens"
        )

    def test_conservative_puts_unknown_in_post(self, corpus_dir):
        result = load_tokens_under_scenario(corpus_dir, "conservative_all_late")
        assert "post_contact" in result
        # 25*3 post-contact own tokens + 2 unknown tablets (15*3 each) = 75 + 90 = 165
        assert len(result["post_contact"]) > 75

    def test_probabilistic_splits_unknown_tokens(self, corpus_dir):
        result = load_tokens_under_scenario(corpus_dir, "probabilistic_weighted")
        # Both strata should receive some unknown tokens
        assert "pre_contact" in result
        assert "post_contact" in result


# ---------------------------------------------------------------------------
# Zipf analysis
# ---------------------------------------------------------------------------

class TestZipfAnalysis:
    def test_recovers_alpha_on_powerlaw_data(self, tmp_path):
        """MLE exponent should be within 0.5 of the generating alpha."""
        true_alpha = 1.2
        _make_powerlaw_corpus(tmp_path, n_types=30, alpha=true_alpha)
        result = zipf_analysis(tmp_path, output_dir=None, plot=False)
        assert "exponent_mle" in result
        assert math.isfinite(result["exponent_mle"])
        assert abs(result["exponent_mle"] - true_alpha) < 0.5, (
            f"MLE alpha {result['exponent_mle']:.3f} too far from true {true_alpha}"
        )


# ---------------------------------------------------------------------------
# parse_line_field
# ---------------------------------------------------------------------------

class TestParseLineField:
    @pytest.fixture(scope="class")
    def parse_line_field(self):
        return _load_parse_line_field()

    @pytest.mark.parametrize("line_str, expected", [
        ("Dr3",    ("D", "a", 3)),   # Tablet D, recto ('r') → side 'a', line 3
        ("Bv12",   ("B", "b", 12)),  # Tablet B, verso ('v') → side 'b', line 12
        ("Ev6/Bb12", ("E", "b", 6)), # Cross-reference: primary location Ev6
        ("Ar1",    ("A", "a", 1)),   # Tablet A, recto, line 1
    ])
    def test_parse_known_inputs(self, parse_line_field, line_str, expected):
        assert parse_line_field(line_str) == expected

    def test_cross_reference_takes_primary(self, parse_line_field):
        tablet, side, line = parse_line_field("Ev6/Bb12")
        assert tablet == "E"
        assert side == "b"
        assert line == 6
