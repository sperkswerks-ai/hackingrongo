"""
tests.test_equivalence_ties
===========================

Unit tests for the equivalence-tie loader (run_decipherment._build_equivalence_ties)
and the diachronic substitution miner's contact-partition corroboration logic
(mine_diachronic_substitutions).

These cover the safety gates that protect the anchor set from noisy
auto-discovered substitution classes:
  * oversized union-find runaways are dropped,
  * classes chaining two differently anchored signs are dropped,
  * members absent from the corpus are pruned (singleton classes vanish),
  * disabling via config returns None.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from omegaconf import OmegaConf

_SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from run_decipherment import _build_equivalence_ties  # noqa: E402
from mine_diachronic_substitutions import (  # noqa: E402
    _normalize_code,
    corroborate,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _cfg(enabled=True, max_class_size=6, drop_anchor_conflicts=True):
    return OmegaConf.create({
        "zone_c": {"mcmc": {"equivalence_ties": {
            "enabled": enabled,
            "max_class_size": max_class_size,
            "drop_anchor_conflicts": drop_anchor_conflicts,
        }}}
    })


def _write_pozd(tmp_path: Path, classes: list[list[str]]) -> Path:
    analysis = tmp_path / "outputs" / "analysis"
    analysis.mkdir(parents=True, exist_ok=True)
    (analysis / "pozdniakov_paradigmatic.json").write_text(
        json.dumps({"equivalence_classes": classes}), encoding="utf-8"
    )
    return tmp_path


def _write_diac(tmp_path: Path, tie_pairs: list[list[str]]) -> None:
    analysis = tmp_path / "outputs" / "analysis"
    analysis.mkdir(parents=True, exist_ok=True)
    (analysis / "diachronic_substitutions.json").write_text(
        json.dumps({"tie_pairs": tie_pairs}), encoding="utf-8"
    )


# ---------------------------------------------------------------------------
# _build_equivalence_ties
# ---------------------------------------------------------------------------

class TestBuildEquivalenceTies:
    def test_disabled_returns_none(self, tmp_path):
        _write_pozd(tmp_path, [["1", "2"]])
        out = _build_equivalence_ties(
            _cfg(enabled=False), tmp_path, ["1", "2"], anchored_phonemes={},
        )
        assert out is None

    def test_missing_config_key_returns_none(self, tmp_path):
        _write_pozd(tmp_path, [["1", "2"]])
        out = _build_equivalence_ties(
            OmegaConf.create({"zone_c": {"mcmc": {}}}),
            tmp_path, ["1", "2"], anchored_phonemes={},
        )
        assert out is None

    def test_basic_class_loaded(self, tmp_path):
        _write_pozd(tmp_path, [["1", "2", "3"]])
        out = _build_equivalence_ties(
            _cfg(), tmp_path, ["1", "2", "3", "4"], anchored_phonemes={},
        )
        assert out == [["1", "2", "3"]]

    def test_unknown_members_pruned_and_singletons_dropped(self, tmp_path):
        _write_pozd(tmp_path, [["1", "999"], ["2", "3"]])
        out = _build_equivalence_ties(
            _cfg(), tmp_path, ["1", "2", "3"], anchored_phonemes={},
        )
        # ["1","999"] → ["1"] (singleton, dropped); ["2","3"] kept.
        assert out == [["2", "3"]]

    def test_oversized_class_dropped(self, tmp_path):
        big = [str(i) for i in range(10)]
        _write_pozd(tmp_path, [big])
        out = _build_equivalence_ties(
            _cfg(max_class_size=6), tmp_path, big, anchored_phonemes={},
        )
        assert out is None

    def test_anchor_conflict_class_dropped(self, tmp_path):
        # 040=kokore and 010=oike must never be tied together.
        _write_pozd(tmp_path, [["010", "040", "300", "680"]])
        out = _build_equivalence_ties(
            _cfg(drop_anchor_conflicts=True),
            tmp_path,
            ["010", "040", "300", "680"],
            anchored_phonemes={"040": "kokore", "010": "oike"},
        )
        assert out is None

    def test_single_anchor_class_propagates(self, tmp_path):
        # One anchored sign + free signs is allowed (intended multiplication).
        _write_pozd(tmp_path, [["040", "300", "680"]])
        out = _build_equivalence_ties(
            _cfg(drop_anchor_conflicts=True),
            tmp_path,
            ["040", "300", "680"],
            anchored_phonemes={"040": "kokore"},
        )
        assert out == [["040", "300", "680"]]

    def test_anchor_conflict_kept_when_gate_disabled(self, tmp_path):
        _write_pozd(tmp_path, [["010", "040"]])
        out = _build_equivalence_ties(
            _cfg(drop_anchor_conflicts=False),
            tmp_path,
            ["010", "040"],
            anchored_phonemes={"040": "kokore", "010": "oike"},
        )
        assert out == [["010", "040"]]

    def test_diachronic_tie_pairs_merged(self, tmp_path):
        _write_pozd(tmp_path, [["1", "2"]])
        _write_diac(tmp_path, [["3", "4"]])
        out = _build_equivalence_ties(
            _cfg(), tmp_path, ["1", "2", "3", "4"], anchored_phonemes={},
        )
        assert sorted(out) == [["1", "2"], ["3", "4"]]


# ---------------------------------------------------------------------------
# Diachronic miner corroboration
# ---------------------------------------------------------------------------

class TestContactCorroboration:
    def test_normalize_code_strips_padding_and_compounds(self):
        assert _normalize_code("052") == "52"
        assert _normalize_code("303s") == "303s"
        assert _normalize_code("200.9") == "200.9"
        assert _normalize_code("040") == _normalize_code("40")

    def test_supports_when_pre_sign_pre_biased(self):
        bias = {"52": {"bias": "pre_biased"}, "100": {"bias": "post_biased"}}
        corr, pre_b, post_b = corroborate("052", "100", bias)
        assert corr == "supports"
        assert pre_b == "pre_biased"
        assert post_b == "post_biased"

    def test_contradicts_when_pre_sign_post_biased(self):
        bias = {"52": {"bias": "post_biased"}}
        corr, _, _ = corroborate("052", "100", bias)
        assert corr == "contradicts"

    def test_neutral_when_no_bias_record(self):
        corr, pre_b, post_b = corroborate("052", "100", {})
        assert corr == "neutral"
        assert pre_b is None and post_b is None
