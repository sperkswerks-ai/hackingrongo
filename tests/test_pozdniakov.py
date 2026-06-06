"""
tests.test_pozdniakov
=====================

Unit tests for the Pozdniakov paradigmatic analysis implemented in
scripts/generate_pozdniakov_report.py:

  (a) find_paradigmatic_pairs() on a synthetic parallel variants JSON
      with known single-position substitutions.
  (b) Equivalence class formation via union-find.
  (c) Comparison to reference classes (recall/precision/F1).
  (d) MCMC cross-validation (phoneme similarity fractions).

No real corpus files or language models are required.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Load functions directly from the script (avoids import side-effects)
# ---------------------------------------------------------------------------

_SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "generate_pozdniakov_report",
        _SCRIPTS_DIR / "generate_pozdniakov_report.py",
    )
    mod = importlib.util.module_from_spec(spec)
    # Prevent __main__ block and heavy imports from running at load time
    try:
        spec.loader.exec_module(mod)
    except Exception:
        pass
    return mod


# Only import the pure functions we need — no side effects
_mod = _load_module()

find_paradigmatic_pairs     = _mod.find_paradigmatic_pairs
_compare_to_reference       = _mod._compare_to_reference
cross_validate_with_mcmc    = _mod.cross_validate_with_mcmc
POZDNIAKOV_REFERENCE_CLASSES = _mod.POZDNIAKOV_REFERENCE_CLASSES


# ---------------------------------------------------------------------------
# Helpers to build synthetic passage data
# ---------------------------------------------------------------------------

def _passage(passage_id: str, attestations: list[dict]) -> dict:
    return {"passage_id": passage_id, "attestations": attestations}


def _att(form: list[str], tablet: str, stratum: str = "post_contact") -> dict:
    return {"form": form, "tablet": tablet, "stratum": stratum}


# ---------------------------------------------------------------------------
# (a) find_paradigmatic_pairs — basic extraction
# ---------------------------------------------------------------------------

class TestFindParadPairs:
    def test_single_substitution_detected(self):
        """A pair differing at exactly one position is found as a paradigmatic pair."""
        passages = [
            _passage("P001", [
                _att(["A", "X", "C"], "tablet1"),
                _att(["A", "Y", "C"], "tablet2"),
                _att(["A", "X", "C"], "tablet3"),
                _att(["A", "Y", "C"], "tablet4"),
            ])
        ]
        result = find_paradigmatic_pairs(passages, min_attestations=1, min_tablets=2)
        pair_keys = {(p["s1"], p["s2"]) for p in result["pairs"]}
        # Canonical key is alphabetical, so X < Y → ("X", "Y")
        assert ("X", "Y") in pair_keys

    def test_multi_position_difference_excluded(self):
        """Pairs that differ at more than one position are NOT paradigmatic pairs."""
        passages = [
            _passage("P002", [
                _att(["A", "X", "C"], "tablet1"),
                _att(["B", "Y", "C"], "tablet2"),
                _att(["A", "X", "C"], "tablet3"),
                _att(["B", "Y", "C"], "tablet4"),
            ])
        ]
        result = find_paradigmatic_pairs(passages, min_attestations=1, min_tablets=2)
        pair_keys = {(p["s1"], p["s2"]) for p in result["pairs"]}
        # Two differences: (A,B) at pos 0 and (X,Y) at pos 1
        # Neither should appear as a paradigmatic pair from these attestations alone
        assert ("X", "Y") not in pair_keys
        assert ("A", "B") not in pair_keys

    def test_min_attestation_filter(self):
        """Pairs with fewer than min_attestations observations are excluded."""
        passages = [
            _passage("P003", [
                _att(["Q", "X", "Z"], "tablet1"),
                _att(["Q", "Y", "Z"], "tablet2"),
            ])
        ]
        # 1 occurrence, requires ≥ 3 → should be excluded
        result = find_paradigmatic_pairs(passages, min_attestations=3, min_tablets=1)
        assert result["n_pairs_found"] == 0

    def test_min_tablets_filter(self):
        """Pairs attested on only one tablet are excluded when min_tablets=2."""
        passages = [
            _passage("P004", [
                _att(["M", "X", "N"], "tablet1"),
                _att(["M", "Y", "N"], "tablet1"),
                _att(["M", "X", "N"], "tablet1"),
                _att(["M", "Y", "N"], "tablet1"),
            ])
        ]
        result = find_paradigmatic_pairs(passages, min_attestations=2, min_tablets=2)
        assert result["n_pairs_found"] == 0

    def test_pair_counting_accumulates_across_passages(self):
        """Observations across multiple passages accumulate for the same pair."""
        passages = [
            _passage("P005", [
                _att(["A", "X", "B"], "tablet1"),
                _att(["A", "Y", "B"], "tablet2"),
            ]),
            _passage("P006", [
                _att(["C", "X", "D"], "tablet1"),
                _att(["C", "Y", "D"], "tablet3"),
            ]),
        ]
        # Each passage contributes 1 observation of (X,Y); total = 2
        result = find_paradigmatic_pairs(passages, min_attestations=2, min_tablets=2)
        pair_keys = {(p["s1"], p["s2"]) for p in result["pairs"]}
        assert ("X", "Y") in pair_keys

    def test_attestation_count_is_correct(self):
        """n_attestations reflects the actual number of variant pairs observed."""
        passages = [
            _passage("P007", [
                _att(["001", "X", "003"], "tabletA"),
                _att(["001", "Y", "003"], "tabletB"),
                _att(["001", "X", "003"], "tabletA"),  # same forms again
                _att(["001", "Y", "003"], "tabletC"),
            ])
        ]
        result = find_paradigmatic_pairs(passages, min_attestations=1, min_tablets=1)
        found = [p for p in result["pairs"] if p["s1"] == "X" and p["s2"] == "Y"]
        assert len(found) == 1
        # There are 2 forms of each variant; 2×2 = 4 pairs, but counting pairs of attestations
        # where each pair (f1, f2) with f1 ≠ f2 contributes to (X,Y)
        assert found[0]["n_attestations"] >= 2

    def test_empty_passages(self):
        """Empty passage list returns zero pairs and zero classes."""
        result = find_paradigmatic_pairs([], min_attestations=1, min_tablets=1)
        assert result["n_pairs_found"] == 0
        assert result["n_classes_found"] == 0


# ---------------------------------------------------------------------------
# (b) Equivalence class formation
# ---------------------------------------------------------------------------

class TestEquivalenceClasses:
    def test_transitive_grouping(self):
        """Signs A↔B and B↔C should be grouped into one class {A, B, C}."""
        passages = [
            _passage("P_AB", [
                _att(["X", "A", "Z"], "t1"),
                _att(["X", "B", "Z"], "t2"),
                _att(["X", "A", "Z"], "t3"),
                _att(["X", "B", "Z"], "t4"),
            ]),
            _passage("P_BC", [
                _att(["Y", "B", "W"], "t1"),
                _att(["Y", "C", "W"], "t2"),
                _att(["Y", "B", "W"], "t3"),
                _att(["Y", "C", "W"], "t4"),
            ]),
        ]
        result = find_paradigmatic_pairs(passages, min_attestations=2, min_tablets=2)
        classes = [frozenset(c) for c in result["equivalence_classes"]]
        abc = frozenset({"A", "B", "C"})
        assert any(abc.issubset(c) for c in classes), (
            f"Expected {{A,B,C}} to be in the same class; got {classes}"
        )

    def test_disjoint_pairs_form_separate_classes(self):
        """Signs from unrelated substitution contexts form separate classes."""
        passages = [
            _passage("P_PQ", [
                _att(["1", "P", "2"], "t1"),
                _att(["1", "Q", "2"], "t2"),
                _att(["1", "P", "2"], "t3"),
                _att(["1", "Q", "2"], "t4"),
            ]),
            _passage("P_MN", [
                _att(["7", "M", "8"], "t1"),
                _att(["7", "N", "8"], "t2"),
                _att(["7", "M", "8"], "t3"),
                _att(["7", "N", "8"], "t4"),
            ]),
        ]
        result = find_paradigmatic_pairs(passages, min_attestations=2, min_tablets=2)
        classes = [frozenset(c) for c in result["equivalence_classes"]]
        # {P,Q} and {M,N} should be distinct classes
        pq = frozenset({"P", "Q"})
        mn = frozenset({"M", "N"})
        assert any(pq.issubset(c) for c in classes), f"Missing {{P,Q}}: {classes}"
        assert any(mn.issubset(c) for c in classes), f"Missing {{M,N}}: {classes}"
        # And neither class mixes P/Q with M/N
        assert not any(len(pq & c) > 0 and len(mn & c) > 0 for c in classes), (
            "P/Q and M/N should be in separate classes"
        )

    def test_single_pair_forms_one_class(self):
        """A single valid pair (s1, s2) forms exactly one equivalence class of size 2."""
        passages = [
            _passage("P_only", [
                _att(["START", "S1", "END"], "t1"),
                _att(["START", "S2", "END"], "t2"),
                _att(["START", "S1", "END"], "t3"),
                _att(["START", "S2", "END"], "t4"),
            ])
        ]
        result = find_paradigmatic_pairs(passages, min_attestations=2, min_tablets=2)
        assert result["n_classes_found"] == 1
        cls = frozenset(result["equivalence_classes"][0])
        assert cls == frozenset({"S1", "S2"})


# ---------------------------------------------------------------------------
# (c) Comparison to reference classes
# ---------------------------------------------------------------------------

class TestCompareToReference:
    def _mk_ref(self) -> list[frozenset[str]]:
        return [
            frozenset({"A", "B"}),
            frozenset({"C", "D", "E"}),
            frozenset({"F", "G"}),
        ]

    def test_perfect_recall_and_precision(self):
        """Identical recovered and reference → recall=1, precision=1, F1=1."""
        ref = self._mk_ref()
        result = _compare_to_reference(ref, ref)
        assert result["recall"] == pytest.approx(1.0)
        assert result["precision"] == pytest.approx(1.0)
        assert result["f1"] == pytest.approx(1.0)

    def test_zero_recall_when_no_overlap(self):
        """Completely disjoint recovered classes → recall=0, precision=0."""
        ref = [frozenset({"A", "B"}), frozenset({"C", "D"})]
        rec = [frozenset({"X", "Y"}), frozenset({"P", "Q"})]
        result = _compare_to_reference(rec, ref)
        assert result["recall"] == pytest.approx(0.0)
        assert result["precision"] == pytest.approx(0.0)
        assert result["f1"] == pytest.approx(0.0)

    def test_partial_match(self):
        """Recovering 1 of 3 reference classes gives recall ~ 0.33."""
        ref = self._mk_ref()
        rec = [frozenset({"A", "B"})]   # matches first ref class only
        result = _compare_to_reference(rec, ref)
        assert result["recall"] == pytest.approx(1 / 3, rel=0.01)
        assert result["precision"] == pytest.approx(1.0)

    def test_superset_still_matches(self):
        """A recovered class that is a superset of a reference class still matches
        (Jaccard > 0.5 when the reference is large enough)."""
        ref = [frozenset({"A", "B", "C", "D"})]
        rec = [frozenset({"A", "B", "C", "D", "E"})]
        # Jaccard = 4/5 = 0.8 > 0.5
        result = _compare_to_reference(rec, ref)
        assert result["recall"] == pytest.approx(1.0)

    def test_f1_combines_precision_recall(self):
        """F1 is 2*P*R/(P+R)."""
        ref = [frozenset({"A", "B"}), frozenset({"C", "D"})]
        rec = [frozenset({"A", "B"}), frozenset({"X", "Y"})]  # 1 match, 1 miss
        result = _compare_to_reference(rec, ref)
        assert result["recall"] == pytest.approx(0.5)
        assert result["precision"] == pytest.approx(0.5)
        assert result["f1"] == pytest.approx(0.5)

    def test_reference_classes_constant_has_15_entries(self):
        """Sanity check: the hardcoded reference list has exactly 15 classes."""
        assert len(POZDNIAKOV_REFERENCE_CLASSES) == 15

    def test_all_reference_classes_have_at_least_2_members(self):
        """Every Pozdniakov reference class must have ≥ 2 members."""
        for cls in POZDNIAKOV_REFERENCE_CLASSES:
            assert len(cls) >= 2, f"Class {cls} has fewer than 2 members"


# ---------------------------------------------------------------------------
# (d) MCMC cross-validation
# ---------------------------------------------------------------------------

class TestMCMCCrossValidation:
    def test_identical_phonemes_similarity_one(self):
        """Pairs assigned the same phoneme have similarity 1.0 → all above threshold."""
        pairs = [
            {"s1": "001", "s2": "002"},
            {"s1": "003", "s2": "004"},
        ]
        phoneme_map = {"001": "ma", "002": "ma", "003": "ku", "004": "ku"}
        result = cross_validate_with_mcmc(pairs, phoneme_map, similarity_threshold=0.5)
        assert result["fraction_above_threshold"] == pytest.approx(1.0)

    def test_completely_different_phonemes(self):
        """Pairs with maximally different phonemes → similarity < threshold."""
        pairs = [{"s1": "A", "s2": "B"}]
        phoneme_map = {"A": "ma", "B": "ku"}
        result = cross_validate_with_mcmc(pairs, phoneme_map, similarity_threshold=0.9)
        assert result["n_above_threshold"] == 0

    def test_missing_phoneme_not_counted(self):
        """If a sign has no phoneme assignment, the pair is not scored."""
        pairs = [{"s1": "X", "s2": "Y"}]
        phoneme_map = {"X": "ma"}  # Y is missing
        result = cross_validate_with_mcmc(pairs, phoneme_map)
        assert result["n_pairs_scored"] == 0
        assert result["fraction_above_threshold"] is None

    def test_partial_coverage(self):
        """Mix of scored and unscored pairs: fraction computed over scored only."""
        pairs = [
            {"s1": "S1", "s2": "S2"},   # both present, similar
            {"s1": "S3", "s2": "S4"},   # S4 missing
        ]
        phoneme_map = {"S1": "ma", "S2": "mo", "S3": "ku"}
        # S1/S2 similarity: "ma" vs "mo" → edit dist 1, max_len 2 → sim 0.5
        result = cross_validate_with_mcmc(pairs, phoneme_map, similarity_threshold=0.4)
        assert result["n_pairs_evaluated"] == 2
        assert result["n_pairs_scored"] == 1
        assert result["fraction_above_threshold"] is not None

    def test_output_keys_present(self):
        """Result dict always contains the expected keys."""
        result = cross_validate_with_mcmc([], {})
        required_keys = {
            "similarity_threshold", "n_pairs_evaluated", "n_pairs_scored",
            "n_above_threshold", "fraction_above_threshold", "pair_details",
            "interpretation",
        }
        assert required_keys.issubset(result.keys())


# ---------------------------------------------------------------------------
# Integration: find_paradigmatic_pairs + compare_to_reference roundtrip
# ---------------------------------------------------------------------------

class TestIntegration:
    def test_synthetic_corpus_recovers_known_pairs(self):
        """A synthetic corpus with 2 hard-wired substitution pairs recovers both."""
        # Two passages, each contributing one known paradigmatic pair
        passages = [
            _passage("SYN_P1", [
                _att(["START", "ALPHA", "END"], "tabA"),
                _att(["START", "BETA",  "END"], "tabB"),
                _att(["START", "ALPHA", "END"], "tabA"),
                _att(["START", "BETA",  "END"], "tabC"),
            ]),
            _passage("SYN_P2", [
                _att(["X", "GAMMA", "Y"], "tabA"),
                _att(["X", "DELTA", "Y"], "tabB"),
                _att(["X", "GAMMA", "Y"], "tabC"),
                _att(["X", "DELTA", "Y"], "tabD"),
            ]),
        ]
        result = find_paradigmatic_pairs(passages, min_attestations=2, min_tablets=2)
        pair_keys = {(p["s1"], p["s2"]) for p in result["pairs"]}
        assert ("ALPHA", "BETA") in pair_keys
        assert ("DELTA", "GAMMA") in pair_keys or ("GAMMA", "DELTA") in pair_keys

    def test_comparison_fields_are_within_range(self):
        """All metric values returned by _compare_to_reference are in [0, 1]."""
        ref = [frozenset({"A", "B"}), frozenset({"C", "D"})]
        rec = [frozenset({"A", "B", "C"}), frozenset({"X", "Y"})]
        result = _compare_to_reference(rec, ref)
        for key in ("recall", "precision", "f1"):
            val = result[key]
            assert 0.0 <= val <= 1.0, f"{key} = {val} out of [0,1]"
