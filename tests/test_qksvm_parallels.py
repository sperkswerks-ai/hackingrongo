"""
tests/test_qksvm_parallels.py — Unit tests for QK-SVM parallel passage detector.

Tests use purely synthetic data so no real corpus or pipeline outputs are needed.
All tests should run without GPU, IBMQ credentials, or large downloads.
"""

from __future__ import annotations

import json
import random
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

# Ensure project root is on the path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.run_qksvm_parallels import (
    CorpusStats,
    PositionPair,
    SoftParallelCandidate,
    build_corpus_stats,
    build_feature_map,
    build_positive_pairs,
    compute_feature_matrix,
    compute_feature_vector,
    group_into_soft_passages,
    kernel_alignment,
    pqk_matrix,
    sample_negatives,
    write_soft_parallels,
)

# ─────────────────────────────────────────────────────────────────────────────
# Synthetic fixtures
# ─────────────────────────────────────────────────────────────────────────────

SIGNS = ["001", "002", "003", "004", "005", "006", "007", "008", "009", "010"]

SYNTHETIC_TABLET_SEQS: dict[str, list[str]] = {
    "T1": ["001", "002", "003", "001", "005", "006", "007", "002", "009", "010",
           "001", "003", "005", "007", "009", "001", "002", "004", "006", "008"],
    "T2": ["002", "003", "001", "004", "006", "005", "008", "007", "010", "009",
           "002", "004", "006", "008", "010", "003", "001", "005", "007", "009"],
    "T3": ["003", "001", "002", "005", "004", "007", "006", "009", "008", "010",
           "003", "001", "004", "002", "006", "005", "008", "007", "010", "009"],
    "T4": ["001", "004", "007", "002", "005", "008", "003", "006", "009", "010",
           "001", "002", "003", "004", "005", "006", "007", "008", "009", "010"],
    "T5": ["010", "009", "008", "007", "006", "005", "004", "003", "002", "001",
           "001", "003", "005", "007", "009", "002", "004", "006", "008", "010"],
}

SYNTHETIC_PASSAGES = [
    {
        "passage_id": "SYNTH_P001",
        "canonical_form": ["001", "002", "003"],
        "n_tablets": 2,
        "attestations": [
            {"tablet": "T1", "form": ["001", "002", "003"], "stratum": "post_contact",
             "start_position": 0},
            {"tablet": "T2", "form": ["001", "002", "003"], "stratum": "post_contact",
             "start_position": 2},
        ],
        "diachronic_changes": [],
        "interest_score": 0.8,
    },
    {
        "passage_id": "SYNTH_P002",
        "canonical_form": ["005", "006"],
        "n_tablets": 2,
        "attestations": [
            {"tablet": "T1", "form": ["005", "006"], "stratum": "pre_contact",
             "start_position": 4},
            {"tablet": "T3", "form": ["005", "006"], "stratum": "post_contact",
             "start_position": 3},
        ],
        "diachronic_changes": [],
        "interest_score": 0.6,
    },
    {
        "passage_id": "SYNTH_P003",
        "canonical_form": ["007", "009"],
        "n_tablets": 3,
        "attestations": [
            {"tablet": "T1", "form": ["007", "009"], "stratum": "post_contact",
             "start_position": 6},
            {"tablet": "T2", "form": ["007", "009"], "stratum": "post_contact",
             "start_position": 6},
            {"tablet": "T4", "form": ["007", "009"], "stratum": "pre_contact",
             "start_position": 0},
        ],
        "diachronic_changes": [],
        "interest_score": 0.7,
    },
]


@pytest.fixture(scope="module")
def corpus_stats():
    return build_corpus_stats(SYNTHETIC_TABLET_SEQS, SYNTHETIC_PASSAGES)


@pytest.fixture(scope="module")
def positives():
    return build_positive_pairs(SYNTHETIC_PASSAGES, SYNTHETIC_TABLET_SEQS)


@pytest.fixture(scope="module")
def training_pairs(positives):
    confirmed = {(p.tablet_a, p.pos_a, p.tablet_b, p.pos_b) for p in positives}
    rng = random.Random(42)
    negatives = sample_negatives(SYNTHETIC_TABLET_SEQS, confirmed, 20, rng)
    return positives + negatives


@pytest.fixture(scope="module")
def feature_map():
    return build_feature_map(8, reps=1)   # reps=1 for speed


# ─────────────────────────────────────────────────────────────────────────────
# (a) Feature vector: shape (n_pairs, 8), all values in [0, 1] or finite
# ─────────────────────────────────────────────────────────────────────────────

class TestFeatureVectors:
    def test_shape(self, training_pairs, corpus_stats):
        X = compute_feature_matrix(
            training_pairs, SYNTHETIC_TABLET_SEQS, corpus_stats, None
        )
        assert X.shape == (len(training_pairs), 8), (
            f"Expected ({len(training_pairs)}, 8) got {X.shape}"
        )

    def test_all_values_in_unit_interval(self, training_pairs, corpus_stats):
        X = compute_feature_matrix(
            training_pairs, SYNTHETIC_TABLET_SEQS, corpus_stats, None
        )
        assert np.all(X >= 0.0), f"Values below 0: {X[X < 0]}"
        assert np.all(X <= 1.0), f"Values above 1: {X[X > 1]}"

    def test_all_finite(self, training_pairs, corpus_stats):
        X = compute_feature_matrix(
            training_pairs, SYNTHETIC_TABLET_SEQS, corpus_stats, None
        )
        assert np.isfinite(X).all(), "Feature matrix contains inf/nan"

    def test_positive_pairs_have_high_freq_ratio(self, positives, corpus_stats):
        """Confirmed same-sign pairs should have freq_ratio = 1.0."""
        for p in positives:
            if p.sign_a == p.sign_b:
                vec = compute_feature_vector(
                    p, SYNTHETIC_TABLET_SEQS, corpus_stats, None
                )
                # Feature 5 is cross_tablet_freq_ratio: same sign → 1.0
                assert abs(vec[5] - 1.0) < 1e-9, (
                    f"Same-sign pair {p.sign_a}/{p.sign_b} has freq_ratio={vec[5]:.4f}"
                )

    def test_delta_ic_zero_for_same_sign(self, corpus_stats):
        pair = PositionPair("T1", 0, "001", "T2", 0, "001", label=1)
        vec = compute_feature_vector(pair, SYNTHETIC_TABLET_SEQS, corpus_stats, None)
        assert vec[0] == pytest.approx(0.0), f"delta_IC != 0 for same sign: {vec[0]}"

    def test_positional_distance_boundary(self, corpus_stats):
        """Pair at (pos=0, len=20) vs (pos=0, len=20): distance = 0."""
        pair = PositionPair("T1", 0, "001", "T2", 0, "002", label=0)
        vec = compute_feature_vector(pair, SYNTHETIC_TABLET_SEQS, corpus_stats, None)
        assert vec[7] == pytest.approx(0.0, abs=1e-9)

        pair2 = PositionPair("T1", 0, "001", "T2", 19, "002", label=0)
        vec2 = compute_feature_vector(pair2, SYNTHETIC_TABLET_SEQS, corpus_stats, None)
        assert vec2[7] == pytest.approx(1.0, abs=1e-6)


# ─────────────────────────────────────────────────────────────────────────────
# (b) Projected kernel matrix: symmetric and positive semi-definite
# ─────────────────────────────────────────────────────────────────────────────

class TestProjectedKernel:
    @pytest.fixture(scope="class")
    def small_X(self, corpus_stats):
        pairs = build_positive_pairs(SYNTHETIC_PASSAGES, SYNTHETIC_TABLET_SEQS)[:6]
        rng = random.Random(99)
        confirmed = {(p.tablet_a, p.pos_a, p.tablet_b, p.pos_b) for p in pairs}
        negs = sample_negatives(SYNTHETIC_TABLET_SEQS, confirmed, 6, rng)
        all_p = pairs + negs
        return compute_feature_matrix(all_p, SYNTHETIC_TABLET_SEQS, corpus_stats, None)

    def test_pqk_symmetry(self, small_X, feature_map):
        K, _ = pqk_matrix(small_X, small_X, feature_map, backend="simulator")
        assert K.shape == (len(small_X), len(small_X))
        np.testing.assert_allclose(K, K.T, atol=1e-10, err_msg="Kernel matrix not symmetric")

    def test_pqk_psd(self, small_X, feature_map):
        K, _ = pqk_matrix(small_X, small_X, feature_map, backend="simulator")
        # Add small diagonal jitter for numerical stability
        K_jitter = K + 1e-9 * np.eye(len(small_X))
        eigenvalues = np.linalg.eigvalsh(K_jitter)
        assert np.all(eigenvalues >= -1e-8), (
            f"Kernel matrix has negative eigenvalue: min={eigenvalues.min():.2e}"
        )

    def test_pqk_diagonal_positive(self, small_X, feature_map):
        K, _ = pqk_matrix(small_X, small_X, feature_map, backend="simulator")
        assert np.all(np.diag(K) > 0), "Diagonal entries must be positive"

    def test_pqk_values_in_unit_interval(self, small_X, feature_map):
        K, _ = pqk_matrix(small_X, small_X, feature_map, backend="simulator")
        assert np.all(K >= -1e-9), f"Negative kernel value: min={K.min():.4f}"
        assert np.all(K <= 1.0 + 1e-9), f"Kernel value > 1: max={K.max():.4f}"

    def test_n_circuits_equals_twice_data_size(self, small_X, feature_map):
        """Projected kernel requires len(X_a) + len(X_b) circuits."""
        K, n_c = pqk_matrix(small_X, small_X, feature_map, backend="simulator")
        assert n_c == 2 * len(small_X), (
            f"Expected {2 * len(small_X)} circuits, got {n_c}"
        )

    def test_kernel_alignment_perfect_labels(self, small_X, feature_map):
        """Kernel alignment is in [-1, 1]."""
        K, _ = pqk_matrix(small_X, small_X, feature_map, backend="simulator")
        y = np.array([1] * (len(small_X) // 2) + [0] * (len(small_X) - len(small_X) // 2))
        ka = kernel_alignment(K, y)
        assert -1.0 <= ka <= 1.0, f"Kernel alignment out of range: {ka}"


# ─────────────────────────────────────────────────────────────────────────────
# (c) SVC trains without error on synthetic precomputed kernel
# ─────────────────────────────────────────────────────────────────────────────

class TestSVCTraining:
    def test_svc_precomputed_trains(self, small_K_and_labels):
        K, y = small_K_and_labels
        from sklearn.svm import SVC
        svm = SVC(kernel="precomputed", probability=True, C=1.0, random_state=42)
        svm.fit(K, y)
        preds = svm.predict(K)
        assert len(preds) == len(y)
        assert set(preds).issubset({0, 1})

    def test_svc_decision_function_shape(self, small_K_and_labels):
        K, y = small_K_and_labels
        from sklearn.svm import SVC
        svm = SVC(kernel="precomputed", probability=True, C=1.0, random_state=42)
        svm.fit(K, y)
        dec = svm.decision_function(K[:3])
        assert dec.shape == (3,), f"Unexpected decision function shape: {dec.shape}"

    def test_svc_probability_sums_to_one(self, small_K_and_labels):
        K, y = small_K_and_labels
        from sklearn.svm import SVC
        svm = SVC(kernel="precomputed", probability=True, C=1.0, random_state=42)
        svm.fit(K, y)
        proba = svm.predict_proba(K[:4])
        np.testing.assert_allclose(
            proba.sum(axis=1), np.ones(4), atol=1e-6, err_msg="Probabilities don't sum to 1"
        )

    @pytest.fixture(scope="class")
    def small_K_and_labels(self, corpus_stats, feature_map):
        pairs = build_positive_pairs(SYNTHETIC_PASSAGES, SYNTHETIC_TABLET_SEQS)[:5]
        rng = random.Random(77)
        confirmed = {(p.tablet_a, p.pos_a, p.tablet_b, p.pos_b) for p in pairs}
        negs = sample_negatives(SYNTHETIC_TABLET_SEQS, confirmed, 10, rng)
        all_p = pairs + negs
        X = compute_feature_matrix(all_p, SYNTHETIC_TABLET_SEQS, corpus_stats, None)
        y = np.array([p.label for p in all_p])
        fm = build_feature_map(8, reps=1)
        K, _ = pqk_matrix(X, X, fm, backend="simulator")
        return K, y


# ─────────────────────────────────────────────────────────────────────────────
# (d) Soft parallel output JSON has the required fields
# ─────────────────────────────────────────────────────────────────────────────

class TestOutputJSON:
    REQUIRED_FIELDS = {
        "tablet_pair", "position_range", "signs_tablet_a", "signs_tablet_b",
        "svm_score", "feature_vector", "nearest_confirmed_passage",
    }
    REQUIRED_TOP_FIELDS = {
        "generated_by", "n_candidates", "score_threshold",
        "cv_quantum", "cv_classical", "candidates",
    }

    def test_required_top_level_fields(self, output_json):
        missing = self.REQUIRED_TOP_FIELDS - set(output_json.keys())
        assert not missing, f"Missing top-level fields: {missing}"

    def test_required_candidate_fields(self, output_json):
        for cand in output_json["candidates"]:
            missing = self.REQUIRED_FIELDS - set(cand.keys())
            assert not missing, f"Candidate missing fields: {missing}"

    def test_svm_score_is_float(self, output_json):
        for cand in output_json["candidates"]:
            assert isinstance(cand["svm_score"], (int, float))

    def test_signs_are_lists_of_strings(self, output_json):
        for cand in output_json["candidates"]:
            assert isinstance(cand["signs_tablet_a"], list)
            assert all(isinstance(s, str) for s in cand["signs_tablet_a"])

    def test_tablet_pair_is_two_element_list(self, output_json):
        for cand in output_json["candidates"]:
            assert isinstance(cand["tablet_pair"], list)
            assert len(cand["tablet_pair"]) == 2

    def test_feature_vector_length_8(self, output_json):
        for cand in output_json["candidates"]:
            fv = cand["feature_vector"]
            if fv:  # may be empty list if no positions
                assert len(fv) == 8, f"Feature vector wrong length: {len(fv)}"

    def test_n_candidates_matches_list(self, output_json):
        assert output_json["n_candidates"] == len(output_json["candidates"])

    @pytest.fixture(scope="class")
    def output_json(self, tmp_path_factory):
        """Write a synthetic output file and load it back."""
        out = tmp_path_factory.mktemp("out") / "soft_parallels.json"
        candidates = [
            SoftParallelCandidate(
                tablet_pair=("T1", "T2"),
                position_range_a=(0, 2),
                position_range_b=(0, 2),
                signs_tablet_a=["001", "002", "003"],
                signs_tablet_b=["001", "002", "003"],
                svm_scores=[0.8, 0.85, 0.9],
                feature_vectors=[
                    [0.0, 0.1, 0.2, 0.3, 0.4, 1.0, 0.7, 0.05],
                    [0.0, 0.1, 0.2, 0.3, 0.4, 1.0, 0.7, 0.05],
                    [0.0, 0.1, 0.2, 0.3, 0.4, 1.0, 0.7, 0.05],
                ],
                nearest_confirmed_passage="SYNTH_P001",
                mean_svm_score=0.85,
            )
        ]
        training_summary = {
            "mean_accuracy": 0.9,
            "std_accuracy": 0.05,
            "mean_f1": 0.88,
            "std_f1": 0.06,
            "mean_auc_roc": 0.92,
            "std_auc_roc": 0.04,
            "kernel_alignment_full": 0.45,
            "total_circuits": 246,
            "n_samples": 30,
            "n_positives": 6,
            "n_negatives": 24,
        }
        rbf_summary = {
            "mean_accuracy": 0.85,
            "std_accuracy": 0.06,
            "mean_f1": 0.83,
            "std_f1": 0.07,
            "mean_auc_roc": 0.89,
            "std_auc_roc": 0.05,
        }
        write_soft_parallels(candidates, SYNTHETIC_PASSAGES, out, training_summary, rbf_summary)
        return json.loads(out.read_text())


# ─────────────────────────────────────────────────────────────────────────────
# Additional: grouping into passages
# ─────────────────────────────────────────────────────────────────────────────

class TestGrouping:
    def test_adjacent_pairs_form_chain(self):
        """Three consecutive position pairs should form one candidate."""
        pairs_and_scores = []
        for i in range(3):
            p = PositionPair(
                tablet_a="T1", pos_a=i, sign_a=SIGNS[i],
                tablet_b="T2", pos_b=i, sign_b=SIGNS[i],
            )
            pairs_and_scores.append((p, 0.9))

        candidates = group_into_soft_passages(pairs_and_scores, threshold=0.5)
        assert len(candidates) == 1
        assert len(candidates[0].signs_tablet_a) == 3

    def test_non_adjacent_pairs_form_separate_chains(self):
        """Pairs with gap > 3 should NOT be chained."""
        pairs_and_scores = []
        for i in [0, 1, 10, 11]:
            p = PositionPair(
                tablet_a="T1", pos_a=i, sign_a=SIGNS[i % 10],
                tablet_b="T2", pos_b=i, sign_b=SIGNS[i % 10],
            )
            pairs_and_scores.append((p, 0.9))

        candidates = group_into_soft_passages(pairs_and_scores, threshold=0.5)
        assert len(candidates) == 2

    def test_threshold_filters_low_scores(self):
        pairs_and_scores = [
            (PositionPair("T1", 0, "001", "T2", 0, "001"), 0.3),
            (PositionPair("T1", 1, "002", "T2", 1, "002"), 0.3),
            (PositionPair("T1", 2, "003", "T2", 2, "003"), 0.3),
        ]
        candidates = group_into_soft_passages(pairs_and_scores, threshold=0.7)
        assert len(candidates) == 0


# ─────────────────────────────────────────────────────────────────────────────
# Additional: training set construction
# ─────────────────────────────────────────────────────────────────────────────

class TestTrainingSet:
    def test_positive_pairs_are_cross_tablet(self, positives):
        for p in positives:
            assert p.tablet_a != p.tablet_b, "Positive pair from same tablet"

    def test_positive_pairs_have_label_1(self, positives):
        assert all(p.label == 1 for p in positives)

    def test_positive_pairs_have_passage_id(self, positives):
        assert all(p.passage_id is not None for p in positives)

    def test_negatives_not_in_confirmed(self, positives):
        confirmed = {(p.tablet_a, p.pos_a, p.tablet_b, p.pos_b) for p in positives}
        confirmed |= {(p.tablet_b, p.pos_b, p.tablet_a, p.pos_a) for p in positives}
        rng = random.Random(0)
        negatives = sample_negatives(SYNTHETIC_TABLET_SEQS, confirmed, 20, rng)
        for n in negatives:
            key1 = (n.tablet_a, n.pos_a, n.tablet_b, n.pos_b)
            key2 = (n.tablet_b, n.pos_b, n.tablet_a, n.pos_a)
            assert key1 not in confirmed and key2 not in confirmed

    def test_negatives_have_label_0(self, positives):
        rng = random.Random(0)
        confirmed = {(p.tablet_a, p.pos_a, p.tablet_b, p.pos_b) for p in positives}
        negatives = sample_negatives(SYNTHETIC_TABLET_SEQS, confirmed, 10, rng)
        assert all(n.label == 0 for n in negatives)
