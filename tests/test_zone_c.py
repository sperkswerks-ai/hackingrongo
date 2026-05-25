"""
tests.test_zone_c
=================

Smoke and unit tests for Zone C: LMScorer, MCMCSampler, BeamSearchDecoder,
and HypothesisRanking file-based serialisation.

No corpus data files required — tests use tiny synthetic NGramLMs and sign
sequences built in-memory.
"""

from __future__ import annotations

import math
import subprocess
import sys
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from hackingrongo.data.rapa_nui_corpus import NGramLM
from hackingrongo.results.schema import (
    DecryptionHypothesis,
    HypothesisRanking,
    PhonemeAssignment,
    StratumScore,
    load_ranking,
)
from hackingrongo.zone_c.lm_scoring import LMScorer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SIGN_IDS = [str(i) for i in range(1, 11)]

_PHONEMES = ["ku", "ma", "ri", "ta", "ko", "pa", "re", "nu", "ti", "wa"]

_CORPUS_SEQUENCES = [
    _PHONEMES[:],
    ["ku", "ma", "ko", "pa", "ri"],
    ["ta", "re", "nu", "ti", "wa"],
    ["ku", "ri", "ta", "nu", "wa"],
]


def _train_lm(path: Path, sequences: list[list[str]], order: int = 2) -> None:
    lm = NGramLM(order=order, language="rapa_nui")
    for seq in sequences:
        lm.update(seq)
    lm.finalise()
    lm.save(path)


def _minimal_cfg(lm_filename: str = "lm.json") -> OmegaConf:
    return OmegaConf.create({
        "zone_c": {
            "lm_scoring": {
                "ngram_orders": [2],
                "oov_log_prob_per_order": {"2": -20.0},
                "ensemble_weights": {"rapa_nui": 1.0},
                "lms": ["rapa_nui"],
                "lm_files": {"rapa_nui": lm_filename},
            },
            "mcmc": {
                "num_chains": 2,
                "num_iterations": 200,
                "burn_in": 50,
                "thin": 2,
                "top_k": 5,
                "gelman_rubin_threshold": 1.2,
                "target_acceptance_rate": 0.234,
                "adaptation_interval": 50,
                "reassign_prob": 0.3,
                "full_rescore_interval": 50,
            },
            "beam_search": {
                "beam_width": 5,
                "max_depth": 20,
                "length_penalty_alpha": 0.6,
                "prune_threshold": 0.01,
                "early_stopping_patience": 3,
                "min_improvement": 0.001,
                "top_k": 5,
            },
        },
        "data": {"lm_tokenization_level": "word"},
    })


# ---------------------------------------------------------------------------
# LMScorer
# ---------------------------------------------------------------------------

class TestLMScorer:
    def test_raises_on_missing_ensemble_weight(self, tmp_path):
        """LMScorer raises ValueError when a listed language has no ensemble weight."""
        cfg = OmegaConf.create({
            "zone_c": {
                "lm_scoring": {
                    "ngram_orders": [2],
                    "oov_log_prob_per_order": {"2": -20.0},
                    "ensemble_weights": {"rapa_nui": 1.0},
                    "lms": ["rapa_nui", "hawaiian"],
                    "lm_files": {"rapa_nui": "lm.json", "hawaiian": "lm2.json"},
                }
            },
            "data": {"lm_tokenization_level": "word"},
        })
        with pytest.raises(ValueError, match="hawaiian"):
            LMScorer(cfg, tmp_path)

    def test_score_returns_finite_float(self, tmp_path):
        """score() on a trained LM returns a finite negative log-probability."""
        lm_path = tmp_path / "lm.json"
        _train_lm(lm_path, _CORPUS_SEQUENCES)
        scorer = LMScorer(_minimal_cfg(), tmp_path)
        result = scorer.score(["ku", "ma", "ri"])
        assert math.isfinite(result.ensemble_log_prob)
        assert result.ensemble_log_prob < 0.0


# ---------------------------------------------------------------------------
# MCMCSampler
# ---------------------------------------------------------------------------

class TestMCMCSampler:
    @pytest.fixture()
    def scorer(self, tmp_path):
        lm_path = tmp_path / "lm.json"
        _train_lm(lm_path, _CORPUS_SEQUENCES)
        return LMScorer(_minimal_cfg(), tmp_path)

    @pytest.fixture()
    def mcmc_result(self, scorer):
        from hackingrongo.zone_c.mcmc import MCMCSampler
        corpus_seqs = [_SIGN_IDS[:5], _SIGN_IDS[5:]]
        sampler = MCMCSampler(
            cfg=_minimal_cfg(),
            lm_scorer=scorer,
            corpus_sequences=corpus_seqs,
            sign_ids=_SIGN_IDS,
            seed=42,
        )
        return sampler.run()

    def test_smoke_acceptance_rate_in_range(self, mcmc_result):
        """2 chains × 200 iterations with 10 signs: acceptance rate ∈ (0.05, 0.95)."""
        for rate in mcmc_result.acceptance_rates:
            assert 0.05 < rate < 0.95, f"Acceptance rate {rate} out of (0.05, 0.95)"

    def test_top_samples_sorted_descending(self, mcmc_result):
        lps = [s.log_posterior for s in mcmc_result.top_samples]
        assert lps == sorted(lps, reverse=True), "top_samples not sorted by log_posterior desc"

    def test_delta_matches_full_rescore(self, scorer):
        """score_delta must equal score(new) − score(old) to floating-point precision."""
        seq = ["ku", "ma", "ri", "ta", "ko"]
        pos, old_ph, new_ph = 2, "ri", "pa"

        old_lp = scorer.score(seq).ensemble_log_prob
        new_seq = list(seq)
        new_seq[pos] = new_ph
        new_lp = scorer.score(new_seq).ensemble_log_prob

        delta_full = new_lp - old_lp
        delta_incremental = scorer.score_delta(seq, [pos], [old_ph], [new_ph])

        assert abs(delta_full - delta_incremental) < 1e-9, (
            f"Cache drift: full={delta_full:.12f}, incremental={delta_incremental:.12f}"
        )


# ---------------------------------------------------------------------------
# BeamSearchDecoder
# ---------------------------------------------------------------------------

class TestBeamSearchDecoder:
    def test_produces_non_empty_sorted_results(self, tmp_path):
        from hackingrongo.zone_c.beam_search import BeamSearchDecoder
        lm_path = tmp_path / "lm.json"
        _train_lm(lm_path, _CORPUS_SEQUENCES)
        cfg = _minimal_cfg()
        scorer = LMScorer(cfg, tmp_path)

        decoder = BeamSearchDecoder(cfg=cfg, lm_scorer=scorer)
        result = decoder.decode(
            sign_ids=_SIGN_IDS,
            corpus_sequences=[_SIGN_IDS[:5], _SIGN_IDS[5:]],
            seed_hypotheses=None,
        )

        assert len(result.top_hypotheses) > 0, "BeamSearchDecoder returned no hypotheses"

        alpha = float(cfg.zone_c.beam_search.length_penalty_alpha)
        scores = [h.normalised_score(alpha) for h in result.top_hypotheses]
        assert scores == sorted(scores, reverse=True), (
            "top_hypotheses not sorted by normalised_score descending"
        )


# ---------------------------------------------------------------------------
# HypothesisRanking file-based serialisation
# ---------------------------------------------------------------------------

class TestHypothesisRankingSerialisation:
    def test_save_load_roundtrip_no_data_loss(self, tmp_path):
        """HypothesisRanking.save() → load_ranking() recovers all fields exactly."""
        hyp = DecryptionHypothesis(
            hypothesis_id="H0001",
            run_id="test-run",
            hypothesis_type="syllabic",
            assignments=[
                PhonemeAssignment(sign_code="076", phoneme="ku",
                                  confidence=0.85, evidence_count=7),
            ],
            stratum_scores=[
                StratumScore(stratum="pre_contact", consistency_score=0.75,
                             lm_score_mean=-5.5, lm_score_std=0.4, n_passages=3,
                             languages_above_baseline=["rapa_nui"]),
            ],
            overall_lm_score=-5.5,
            mcmc_log_posterior=-12.3,
            beam_score=0.0,
            config_hash="deadbeef" * 8,
        )
        ranking = HypothesisRanking(
            hypotheses=[hyp],
            ranking_metric="overall_lm_score",
        )
        out = tmp_path / "ranking.json"
        ranking.save(out)
        assert out.exists()

        loaded = load_ranking(out)
        assert loaded.ranking_metric == "overall_lm_score"
        assert len(loaded.hypotheses) == 1

        h = loaded.hypotheses[0]
        assert h.hypothesis_id == "H0001"
        assert h.overall_lm_score == pytest.approx(-5.5)
        assert h.mcmc_log_posterior == pytest.approx(-12.3)
        assert h.config_hash == "deadbeef" * 8
        assert len(h.assignments) == 1
        assert h.assignments[0].sign_code == "076"
        assert h.assignments[0].phoneme == "ku"
        assert h.stratum_scores[0].stratum == "pre_contact"
        assert "rapa_nui" in h.stratum_scores[0].languages_above_baseline


# ---------------------------------------------------------------------------
# run_decipherment.py --smoke-test
# ---------------------------------------------------------------------------

class TestRunDeciphermentSmoke:
    def test_smoke_test_exits_zero(self):
        """run_decipherment.py --smoke-test should exit 0 when corpus is available."""
        try:
            import hydra  # noqa: F401
        except ImportError:
            pytest.skip("hydra-core not installed in this environment")

        project_root = Path(__file__).resolve().parent.parent
        corpus_dir = project_root / "data" / "corpus"
        if not corpus_dir.exists() or not any(corpus_dir.glob("[A-Z].json")):
            pytest.skip("Corpus data not available in this environment")

        script = project_root / "scripts" / "run_decipherment.py"
        proc = subprocess.run(
            [sys.executable, str(script), "--smoke-test"],
            cwd=str(project_root),
            capture_output=True,
            text=True,
            timeout=300,
        )
        assert proc.returncode == 0, (
            f"--smoke-test failed (exit {proc.returncode}):\n"
            f"{proc.stderr[-3000:]}"
        )
