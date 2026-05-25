"""
tests.test_schema
=================

Tests for hackingrongo.results.schema serialisation and deserialisation.

Key regression tests:
* ``from_dict`` must NOT mutate the caller's dict (the dict-pop bug fixed
  in the bug-fix pass).
* Calling ``from_dict`` twice on the same dict must produce two identical
  objects (no data is lost from the first call).
* Round-trip ``from_dict(json.loads(h.to_json()))`` must reproduce all fields.
"""

from __future__ import annotations

import json

import pytest

from hackingrongo.results.schema import (
    DecryptionHypothesis,
    HypothesisRanking,
    PhonemeAssignment,
    StratumScore,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _phoneme_assignment(**overrides) -> dict:
    base = {
        "sign_code": "001",
        "phoneme": "ku",
        "confidence": 0.9,
        "evidence_count": 10,
        "stratum_breakdown": {"early": -4.5},
    }
    base.update(overrides)
    return base


def _stratum_score(**overrides) -> dict:
    base = {
        "stratum": "early",
        "consistency_score": 0.75,
        "lm_score_mean": -5.0,
        "lm_score_std": 0.5,
        "n_passages": 3,
        "languages_above_baseline": ["rapa_nui", "hawaiian"],
    }
    base.update(overrides)
    return base


def _hypothesis(**overrides) -> dict:
    base = {
        "hypothesis_id": "H0001",
        "run_id": "run_abc123",
        "hypothesis_type": "syllabic",
        "assignments": [_phoneme_assignment()],
        "stratum_scores": [_stratum_score()],
        "overall_lm_score": -10.0,
        "mcmc_log_posterior": -15.0,
        "beam_score": 0.0,
        "created_at": "2026-01-01T00:00:00+00:00",
        "config_hash": "deadbeef" * 8,
    }
    base.update(overrides)
    return base


def _ranking(**overrides) -> dict:
    base = {
        "hypotheses": [_hypothesis(), _hypothesis(hypothesis_id="H0002", overall_lm_score=-11.0)],
        "ranking_metric": "overall_lm_score",
        "generated_at": "2026-01-01T00:00:00+00:00",
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# PhonemeAssignment
# ---------------------------------------------------------------------------


class TestPhonemeAssignment:
    def test_construction(self):
        a = PhonemeAssignment(**_phoneme_assignment())
        assert a.sign_code == "001"
        assert a.phoneme == "ku"
        assert a.confidence == pytest.approx(0.9)
        assert a.evidence_count == 10

    def test_default_stratum_breakdown(self):
        a = PhonemeAssignment(sign_code="001", phoneme="ku", confidence=0.5, evidence_count=1)
        assert a.stratum_breakdown == {}


# ---------------------------------------------------------------------------
# StratumScore
# ---------------------------------------------------------------------------


class TestStratumScore:
    def test_construction(self):
        s = StratumScore(**_stratum_score())
        assert s.stratum == "early"
        assert s.n_passages == 3
        assert "rapa_nui" in s.languages_above_baseline

    def test_default_languages(self):
        s = StratumScore(
            stratum="late", consistency_score=0.5,
            lm_score_mean=-6.0, lm_score_std=1.0, n_passages=2,
        )
        assert s.languages_above_baseline == []


# ---------------------------------------------------------------------------
# DecryptionHypothesis.from_dict
# ---------------------------------------------------------------------------


class TestDecryptionHypothesisFromDict:
    def test_round_trip(self):
        data = _hypothesis()
        h = DecryptionHypothesis.from_dict(data)
        h2 = DecryptionHypothesis.from_dict(json.loads(h.to_json()))
        assert h2.hypothesis_id == h.hypothesis_id
        assert h2.run_id == h.run_id
        assert h2.overall_lm_score == pytest.approx(h.overall_lm_score)
        assert len(h2.assignments) == len(h.assignments)
        assert h2.assignments[0].sign_code == h.assignments[0].sign_code
        assert len(h2.stratum_scores) == len(h.stratum_scores)
        assert h2.stratum_scores[0].stratum == h.stratum_scores[0].stratum

    def test_from_dict_does_not_mutate_caller(self):
        """from_dict must not pop keys from the caller's dict."""
        data = _hypothesis()
        original_keys = set(data.keys())
        original_assignments = list(data["assignments"])
        original_stratum_scores = list(data["stratum_scores"])

        DecryptionHypothesis.from_dict(data)

        assert set(data.keys()) == original_keys
        assert data["assignments"] == original_assignments
        assert data["stratum_scores"] == original_stratum_scores

    def test_from_dict_twice_same_dict(self):
        """Calling from_dict twice on the same dict must produce equal objects."""
        data = _hypothesis()
        h1 = DecryptionHypothesis.from_dict(data)
        h2 = DecryptionHypothesis.from_dict(data)
        assert h1.hypothesis_id == h2.hypothesis_id
        assert len(h1.assignments) == len(h2.assignments)
        assert len(h1.stratum_scores) == len(h2.stratum_scores)

    def test_empty_assignments_and_stratum_scores(self):
        data = _hypothesis(assignments=[], stratum_scores=[])
        h = DecryptionHypothesis.from_dict(data)
        assert h.assignments == []
        assert h.stratum_scores == []

    def test_nested_assignment_fields(self):
        data = _hypothesis(assignments=[_phoneme_assignment(sign_code="076", phoneme="ma")])
        h = DecryptionHypothesis.from_dict(data)
        assert h.assignments[0].sign_code == "076"
        assert h.assignments[0].phoneme == "ma"

    def test_config_hash_preserved(self):
        data = _hypothesis(config_hash="abcd1234")
        h = DecryptionHypothesis.from_dict(data)
        assert h.config_hash == "abcd1234"


# ---------------------------------------------------------------------------
# HypothesisRanking.from_dict
# ---------------------------------------------------------------------------


class TestHypothesisRankingFromDict:
    def test_round_trip(self):
        data = _ranking()
        r = HypothesisRanking.from_dict(data)
        r2 = HypothesisRanking.from_dict(json.loads(r.to_json()))
        assert r2.ranking_metric == r.ranking_metric
        assert len(r2.hypotheses) == len(r.hypotheses)
        assert r2.hypotheses[0].hypothesis_id == r.hypotheses[0].hypothesis_id

    def test_from_dict_does_not_mutate_caller(self):
        data = _ranking()
        original_keys = set(data.keys())
        original_hypotheses = list(data["hypotheses"])

        HypothesisRanking.from_dict(data)

        assert set(data.keys()) == original_keys
        assert data["hypotheses"] == original_hypotheses

    def test_from_dict_twice_same_dict(self):
        data = _ranking()
        r1 = HypothesisRanking.from_dict(data)
        r2 = HypothesisRanking.from_dict(data)
        assert len(r1.hypotheses) == len(r2.hypotheses)
        assert r1.ranking_metric == r2.ranking_metric

    def test_empty_hypotheses(self):
        data = _ranking(hypotheses=[])
        r = HypothesisRanking.from_dict(data)
        assert r.hypotheses == []

    def test_top_n(self):
        data = _ranking()
        r = HypothesisRanking.from_dict(data)
        assert len(r.top_n(1)) == 1
        assert r.top_n(1)[0].hypothesis_id == r.hypotheses[0].hypothesis_id

    def test_hypotheses_nested_assignments(self):
        data = _ranking()
        r = HypothesisRanking.from_dict(data)
        assert isinstance(r.hypotheses[0].assignments[0], PhonemeAssignment)
