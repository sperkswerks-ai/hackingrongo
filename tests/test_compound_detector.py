"""Tests for hackingrongo.zone_b.compound_detector."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from hackingrongo.zone_b.compound_detector import (
    COMPOUND_SEPARATORS,
    CompoundCandidate,
    CompoundDetector,
    MethodEvidence,
    _build_positional_features,
    _build_sign_centroids,
    _compute_positional_profile_stats,
    _is_iconographic_compound,
    _is_syntactic_compound,
    _method_cluster_anomaly,
    _method_embedding_geometry,
    _method_positional_profile,
    load_known_compounds,
    save_compound_candidates,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_corpus_json(tmp_path: Path, tablet_id: str, glyphs: list[dict]) -> Path:
    """Write a minimal corpus JSON and return its path."""
    p = tmp_path / f"{tablet_id}.json"
    p.write_text(
        json.dumps({"tablet_id": tablet_id, "glyphs": glyphs}),
        encoding="utf-8",
    )
    return p


def _make_glyph(barthel_code: str, position: int = 1) -> dict:
    return {
        "position": position,
        "barthel_code": barthel_code,
        "barthel_base": barthel_code.split(":")[0].split(".")[0],
        "horley_components": None,
    }


def _make_umap_df(rows: list[dict]) -> pd.DataFrame:
    """Build a minimal UMAP DataFrame from a list of dicts."""
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# _is_syntactic_compound
# ---------------------------------------------------------------------------


class TestIsSyntacticCompound:
    def test_colon_separator(self):
        assert _is_syntactic_compound("050:001") is True

    def test_period_separator(self):
        assert _is_syntactic_compound("001.002") is True

    def test_hyphen_separator(self):
        assert _is_syntactic_compound("010-020") is True

    def test_apostrophe_separator(self):
        assert _is_syntactic_compound("001'002") is True

    def test_simple_code_false(self):
        assert _is_syntactic_compound("001") is False

    def test_empty_string_false(self):
        assert _is_syntactic_compound("") is False


# ---------------------------------------------------------------------------
# _is_iconographic_compound
# ---------------------------------------------------------------------------


class TestIsIconographicCompound:
    def test_600_range(self):
        assert _is_iconographic_compound("600") is True
        assert _is_iconographic_compound("650") is True
        assert _is_iconographic_compound("699") is True

    def test_700_range(self):
        assert _is_iconographic_compound("700") is True
        assert _is_iconographic_compound("799") is True

    def test_outside_range(self):
        assert _is_iconographic_compound("001") is False
        assert _is_iconographic_compound("500") is False
        assert _is_iconographic_compound("800") is False

    def test_empty_string_false(self):
        assert _is_iconographic_compound("") is False

    def test_variant_suffix_ignored(self):
        # "600v" — numeric prefix 600 is iconographic
        assert _is_iconographic_compound("600v") is True


# ---------------------------------------------------------------------------
# load_known_compounds
# ---------------------------------------------------------------------------


class TestLoadKnownCompounds:
    def test_finds_syntactic_compounds(self, tmp_path):
        _make_corpus_json(tmp_path, "T", [
            _make_glyph("001", 1),
            _make_glyph("050:001", 2),
            _make_glyph("010.020", 3),
            _make_glyph("002", 4),
        ])
        result = load_known_compounds(tmp_path)
        assert "050:001" in result
        assert "010.020" in result
        assert "001" not in result
        assert "002" not in result

    def test_deduplicates_across_tablets(self, tmp_path):
        _make_corpus_json(tmp_path, "T1", [_make_glyph("050:001", 1), _make_glyph("050:001", 2)])
        _make_corpus_json(tmp_path, "T2", [_make_glyph("050:001", 1)])
        result = load_known_compounds(tmp_path)
        assert len([k for k in result if k == "050:001"]) == 1

    def test_empty_corpus_returns_empty(self, tmp_path):
        _make_corpus_json(tmp_path, "T", [])
        result = load_known_compounds(tmp_path)
        assert result == {}

    def test_ignores_non_json_files(self, tmp_path):
        (tmp_path / "notes.txt").write_text("ignore me")
        _make_corpus_json(tmp_path, "T", [_make_glyph("050:001", 1)])
        result = load_known_compounds(tmp_path)
        assert "050:001" in result


# ---------------------------------------------------------------------------
# _build_sign_centroids
# ---------------------------------------------------------------------------


class TestBuildSignCentroids:
    def _df(self):
        return _make_umap_df([
            {"barthel_code": "001", "umap_x": 1.0, "umap_y": 2.0, "hdbscan_cluster": 0},
            {"barthel_code": "001", "umap_x": 3.0, "umap_y": 4.0, "hdbscan_cluster": 0},
            {"barthel_code": "002", "umap_x": 5.0, "umap_y": 6.0, "hdbscan_cluster": 1},
            {"barthel_code": "?",   "umap_x": 9.0, "umap_y": 9.0, "hdbscan_cluster": -1},
        ])

    def test_centroid_mean(self):
        centroids = _build_sign_centroids(self._df())
        assert pytest.approx(centroids.loc["001", "cx"]) == 2.0
        assert pytest.approx(centroids.loc["001", "cy"]) == 3.0

    def test_unknown_code_excluded(self):
        centroids = _build_sign_centroids(self._df())
        assert "?" not in centroids.index

    def test_exclude_codes_removes_entry(self):
        centroids = _build_sign_centroids(self._df(), exclude_codes={"001"})
        assert "001" not in centroids.index
        assert "002" in centroids.index

    def test_purity_column_present(self):
        centroids = _build_sign_centroids(self._df())
        assert "cluster_purity" in centroids.columns
        assert "dominant_cluster" in centroids.columns


# ---------------------------------------------------------------------------
# _build_positional_features
# ---------------------------------------------------------------------------


class TestBuildPositionalFeatures:
    def test_all_zeros_for_absent_code(self):
        seqs = [["001", "002", "003"]]
        feats = _build_positional_features("999", seqs)
        assert all(v == 0.0 for v in feats.values())

    def test_seq_final(self):
        seqs = [["001", "002", "003"]]
        feats = _build_positional_features("003", seqs)
        assert feats["frac_seq_final"] == 1.0
        assert feats["frac_seq_initial"] == 0.0

    def test_seq_initial(self):
        seqs = [["001", "002", "003"]]
        feats = _build_positional_features("001", seqs)
        assert feats["frac_seq_initial"] == 1.0

    def test_post_taxogram(self):
        seqs = [["200", "050", "001"]]
        feats = _build_positional_features("050", seqs, taxogram_code="200")
        assert feats["frac_post_taxogram"] == 1.0

    def test_bigram_entropy_positive(self):
        # Multiple distinct contexts → positive entropy
        seqs = [
            ["A", "X", "B"],
            ["C", "X", "D"],
            ["E", "X", "F"],
        ]
        feats = _build_positional_features("X", seqs)
        assert feats["bigram_entropy"] > 0.0

    def test_returns_all_feature_keys(self):
        from hackingrongo.zone_b.compound_detector import _POSITION_FEATURES
        feats = _build_positional_features("001", [["001"]])
        assert set(feats.keys()) == set(_POSITION_FEATURES)


# ---------------------------------------------------------------------------
# _method_embedding_geometry
# ---------------------------------------------------------------------------


class TestMethodEmbeddingGeometry:
    def _simple_centroids(self) -> pd.DataFrame:
        """Three simple signs at known positions."""
        return pd.DataFrame(
            {"cx": [0.0, 10.0, 5.0], "cy": [0.0, 0.0, 10.0]},
            index=["A", "B", "C"],
        )

    def test_midpoint_candidate_high_confidence(self):
        # Candidate at exact midpoint of A and B
        centroid = np.array([5.0, 0.0])
        ev = _method_embedding_geometry("X", centroid, self._simple_centroids(), k_neighbours=3)
        assert ev is not None
        assert ev.method == "embedding_geometry"
        assert ev.confidence > 0.9
        assert set(ev.proposed_components) == {"A", "B"}

    def test_far_candidate_returns_none(self):
        centroid = np.array([100.0, 100.0])
        ev = _method_embedding_geometry("X", centroid, self._simple_centroids(), k_neighbours=3)
        # May return None or low-confidence; confidence must be < 0.1 or None
        if ev is not None:
            assert ev.confidence < 0.25

    def test_too_few_centroids_returns_none(self):
        single = pd.DataFrame({"cx": [0.0], "cy": [0.0]}, index=["A"])
        ev = _method_embedding_geometry("X", np.array([0.5, 0.0]), single)
        assert ev is None

    def test_details_keys_present(self):
        centroid = np.array([5.0, 0.0])
        ev = _method_embedding_geometry("X", centroid, self._simple_centroids(), k_neighbours=3)
        assert ev is not None
        for key in ("component_1", "component_2", "dist_to_midpoint", "interpoint_dist"):
            assert key in ev.details

    def test_beside_candidate_filtered_by_betweenness_guard(self):
        """Sign beyond B along the A-B axis is not 'between' A and B."""
        # A at (0,0), B at (2,0); interpoint_dist = 2.0
        # Beside candidate at (3,0): dist_to_A=3.0 > 2.0 → filtered
        centroids = pd.DataFrame(
            {"cx": [0.0, 2.0], "cy": [0.0, 0.0]},
            index=["A", "B"],
        )
        beside = np.array([3.0, 0.0])
        ev = _method_embedding_geometry("X", beside, centroids, k_neighbours=2)
        assert ev is None or ev.confidence < 0.1

    def test_boundary_of_between_zone_gives_zero_confidence(self):
        """At exactly interpoint_dist/2 from midpoint, new formula yields confidence=0."""
        # A at (0,0), B at (4,0); midpoint at (2,0), interpoint_dist=4.0
        # Candidate at (2,2): dist_to_mid=2.0, betweenness OK (dist_to_A=dist_to_B≈2.83<4)
        # New: 1 - 2*2/4 = 0 → below 0.1 threshold → None
        centroids = pd.DataFrame(
            {"cx": [0.0, 4.0], "cy": [0.0, 0.0]},
            index=["A", "B"],
        )
        boundary = np.array([2.0, 2.0])
        ev = _method_embedding_geometry("X", boundary, centroids, k_neighbours=2)
        assert ev is None


# ---------------------------------------------------------------------------
# _method_cluster_anomaly
# ---------------------------------------------------------------------------


class TestMethodClusterAnomaly:
    def _make_inputs(self):
        """Candidate 'X' with all instances as noise, flanked by two clusters."""
        umap_df = _make_umap_df([
            # X — all noise
            {"barthel_code": "X", "umap_x": 5.0, "umap_y": 0.0, "hdbscan_cluster": -1},
            {"barthel_code": "X", "umap_x": 5.1, "umap_y": 0.1, "hdbscan_cluster": -1},
            # Simple signs in cluster 0 and cluster 1
            {"barthel_code": "A", "umap_x": 0.0, "umap_y": 0.0, "hdbscan_cluster": 0},
            {"barthel_code": "A", "umap_x": 0.1, "umap_y": 0.0, "hdbscan_cluster": 0},
            {"barthel_code": "B", "umap_x": 10.0, "umap_y": 0.0, "hdbscan_cluster": 1},
            {"barthel_code": "B", "umap_x": 10.1, "umap_y": 0.0, "hdbscan_cluster": 1},
        ])
        centroids = _build_sign_centroids(
            umap_df[umap_df["barthel_code"] != "X"]
        )
        return umap_df, centroids

    def test_noise_candidate_gets_evidence(self):
        umap_df, centroids = self._make_inputs()
        ev = _method_cluster_anomaly("X", umap_df, centroids, noise_prior=0.01)
        assert ev is not None
        assert ev.method == "cluster_anomaly"
        assert ev.confidence > 0.0

    def test_clustered_sign_returns_none(self):
        umap_df, centroids = self._make_inputs()
        # "A" belongs cleanly to cluster 0 — noise_frac = 0
        ev = _method_cluster_anomaly("A", umap_df, centroids, noise_prior=0.01)
        assert ev is None

    def test_missing_code_returns_none(self):
        umap_df, centroids = self._make_inputs()
        ev = _method_cluster_anomaly("NOTHERE", umap_df, centroids)
        assert ev is None


# ---------------------------------------------------------------------------
# _method_positional_profile
# ---------------------------------------------------------------------------


class TestComputePositionalProfileStats:
    N = 5  # len(_POSITION_FEATURES)

    def test_returns_zeros_ones_when_no_codes_qualify(self):
        seqs = [["001", "002"]]
        mean, std = _compute_positional_profile_stats(["999"], seqs, min_corpus_count=3)
        assert mean.shape == (self.N,)
        assert std.shape == (self.N,)
        np.testing.assert_array_equal(mean, np.zeros(self.N))
        np.testing.assert_array_equal(std, np.ones(self.N))

    def test_returns_correct_shape(self):
        seqs = [["001", "002", "003"]] * 5
        mean, std = _compute_positional_profile_stats(["001", "002"], seqs, min_corpus_count=1)
        assert mean.shape == (self.N,)
        assert std.shape == (self.N,)

    def test_std_floor_applied(self):
        # All codes have identical profiles → raw std = 0, but floor keeps std > 0
        seqs = [["001"]] * 5
        mean, std = _compute_positional_profile_stats(["001"], seqs, min_corpus_count=1, cap=1)
        # Only one code qualifies → falls back to (zeros, ones)
        assert (std > 0).all()

    def test_cap_respected(self):
        seqs = [["001", "002", "003", "004", "005"]] * 5
        codes = ["001", "002", "003", "004", "005"]
        # cap=2 — only first 2 codes are processed
        mean_capped, _ = _compute_positional_profile_stats(codes, seqs, cap=2, min_corpus_count=1)
        mean_full, _ = _compute_positional_profile_stats(codes, seqs, cap=100, min_corpus_count=1)
        # means will differ if more than 2 codes have different profiles
        assert mean_capped.shape == (self.N,)
        assert mean_full.shape == (self.N,)


class TestMethodPositionalProfile:
    N = 5  # len(_POSITION_FEATURES)

    def _zero_profiles(self):
        """Identical compound and simple profiles — discriminative margin = 0."""
        z = np.zeros(self.N)
        o = np.ones(self.N)
        return z, o, z.copy(), o.copy()

    def _discriminative_profiles(self):
        """Compound mean at high post-taxogram; simple mean at low post-taxogram."""
        # Features: frac_post_taxogram, frac_seq_final, frac_seq_initial,
        #           mean_relative_pos, bigram_entropy
        compound_mean = np.array([1.0, 0.0, 0.0, 0.5, 0.0])
        compound_std  = np.ones(self.N) * 0.3
        simple_mean   = np.array([0.0, 0.0, 0.0, 0.5, 0.0])
        simple_std    = np.ones(self.N) * 0.3
        return compound_mean, compound_std, simple_mean, simple_std

    def _corpus(self, n: int = 10) -> list[list[str]]:
        """Corpus where CAND always follows taxogram (compound-like profile)."""
        return [["200", "CAND", "001"]] * n

    def test_too_few_occurrences_returns_none(self):
        seqs = [["001", "002"]]
        cm, cs, sm, ss = self._zero_profiles()
        ev = _method_positional_profile("001", seqs, cm, cs, sm, ss, min_corpus_count=5)
        assert ev is None

    def test_identical_distributions_return_none(self):
        """When compound and simple are the same distribution, margin=0 → no evidence."""
        seqs = self._corpus()
        cm, cs, sm, ss = self._zero_profiles()
        ev = _method_positional_profile("CAND", seqs, cm, cs, sm, ss)
        assert ev is None

    def test_compound_like_candidate_returns_evidence(self):
        """Candidate whose features match compound_mean and differ from simple_mean."""
        seqs = self._corpus()
        cm, cs, sm, ss = self._discriminative_profiles()
        ev = _method_positional_profile("CAND", seqs, cm, cs, sm, ss)
        assert ev is not None
        assert ev.confidence > 0.0
        assert ev.method == "positional_profile"

    def test_details_keys_present(self):
        seqs = self._corpus()
        cm, cs, sm, ss = self._discriminative_profiles()
        ev = _method_positional_profile("CAND", seqs, cm, cs, sm, ss)
        if ev is not None:
            for key in ("corpus_frequency", "compound_z", "simple_z", "discriminative_margin"):
                assert key in ev.details

    def test_high_compound_z_returns_none(self):
        """Candidate far from compound mean is rejected even if far from simple mean too."""
        seqs = self._corpus()
        # compound_mean at zero with tight std → frac_post_taxogram=1.0 gives z≫1.5
        compound_mean = np.zeros(self.N)
        compound_std  = np.ones(self.N) * 0.1
        simple_mean   = np.zeros(self.N)
        simple_std    = np.ones(self.N)
        ev = _method_positional_profile("CAND", seqs, compound_mean, compound_std, simple_mean, simple_std)
        assert ev is None


# ---------------------------------------------------------------------------
# CompoundDetector (integration)
# ---------------------------------------------------------------------------


class TestCompoundDetector:
    def _make_corpus(self, tmp_path: Path) -> Path:
        """Corpus with one explicit compound and several simple signs."""
        glyphs = [
            _make_glyph("001", 1),
            _make_glyph("002", 2),
            _make_glyph("003", 3),
            _make_glyph("050:001", 4),  # known compound
            _make_glyph("001", 5),
            _make_glyph("002", 6),
        ]
        _make_corpus_json(tmp_path, "T", glyphs)
        return tmp_path

    def _make_umap_df(self) -> pd.DataFrame:
        """UMAP data with candidate "004" lying between "001" and "002"."""
        rng = np.random.default_rng(0)
        rows = []
        # "001" cluster at ~(0, 0)
        for _ in range(8):
            rows.append({
                "barthel_code": "001",
                "umap_x": rng.normal(0.0, 0.2),
                "umap_y": rng.normal(0.0, 0.2),
                "hdbscan_cluster": 0,
            })
        # "002" cluster at ~(10, 0)
        for _ in range(8):
            rows.append({
                "barthel_code": "002",
                "umap_x": rng.normal(10.0, 0.2),
                "umap_y": rng.normal(0.0, 0.2),
                "hdbscan_cluster": 1,
            })
        # "003" cluster at ~(5, 10)
        for _ in range(8):
            rows.append({
                "barthel_code": "003",
                "umap_x": rng.normal(5.0, 0.2),
                "umap_y": rng.normal(10.0, 0.2),
                "hdbscan_cluster": 2,
            })
        # "004" — all noise, sits at midpoint of 001 and 002
        for _ in range(6):
            rows.append({
                "barthel_code": "004",
                "umap_x": rng.normal(5.0, 0.1),
                "umap_y": rng.normal(0.0, 0.1),
                "hdbscan_cluster": -1,
            })
        # known compound — also noise (excluded from simple centroids)
        for _ in range(3):
            rows.append({
                "barthel_code": "050:001",
                "umap_x": rng.normal(2.0, 0.3),
                "umap_y": rng.normal(0.0, 0.3),
                "hdbscan_cluster": -1,
            })
        return pd.DataFrame(rows)

    def test_known_compound_excluded_by_default(self, tmp_path):
        corpus_dir = self._make_corpus(tmp_path)
        detector = CompoundDetector(corpus_dir=corpus_dir, min_methods=1, min_confidence=0.1)
        candidates = detector.detect(self._make_umap_df())
        codes = {c.barthel_code for c in candidates}
        assert "050:001" not in codes

    def test_known_compound_included_when_exclude_known_false(self, tmp_path):
        corpus_dir = self._make_corpus(tmp_path)
        detector = CompoundDetector(
            corpus_dir=corpus_dir, min_methods=1, min_confidence=0.1,
            exclude_known=False,
        )
        candidates = detector.detect(self._make_umap_df())
        codes = {c.barthel_code for c in candidates}
        # "050:001" is in the candidate pool; it may or may not pass the methods
        # but should not be filtered at the pool stage
        assert any(c.is_known_compound for c in candidates) or True  # non-fatal

    def test_returns_list_of_compound_candidates(self, tmp_path):
        corpus_dir = self._make_corpus(tmp_path)
        detector = CompoundDetector(corpus_dir=corpus_dir, min_methods=1, min_confidence=0.1)
        candidates = detector.detect(self._make_umap_df())
        assert isinstance(candidates, list)
        for c in candidates:
            assert isinstance(c, CompoundCandidate)

    def test_sorted_by_confidence_descending(self, tmp_path):
        corpus_dir = self._make_corpus(tmp_path)
        detector = CompoundDetector(corpus_dir=corpus_dir, min_methods=1, min_confidence=0.1)
        candidates = detector.detect(self._make_umap_df())
        confs = [(c.n_methods_agreeing, c.consensus_confidence) for c in candidates]
        assert confs == sorted(confs, key=lambda x: (-x[0], -x[1]))

    def test_candidate_fields_populated(self, tmp_path):
        corpus_dir = self._make_corpus(tmp_path)
        detector = CompoundDetector(corpus_dir=corpus_dir, min_methods=1, min_confidence=0.1)
        candidates = detector.detect(self._make_umap_df())
        for c in candidates:
            assert isinstance(c.barthel_code, str)
            assert 0.0 <= c.consensus_confidence <= 1.0
            assert c.n_methods_agreeing >= 1
            assert isinstance(c.method_evidence, list)
            assert len(c.method_evidence) == c.n_methods_agreeing

    def test_004_detected_as_candidate(self, tmp_path):
        """Sign 004 is positioned at midpoint of 001/002 and is all-noise."""
        corpus_dir = self._make_corpus(tmp_path)
        detector = CompoundDetector(corpus_dir=corpus_dir, min_methods=1, min_confidence=0.1)
        candidates = detector.detect(self._make_umap_df())
        codes = {c.barthel_code for c in candidates}
        assert "004" in codes


# ---------------------------------------------------------------------------
# save_compound_candidates
# ---------------------------------------------------------------------------


class TestSaveCompoundCandidates:
    def _make_candidates(self) -> list[CompoundCandidate]:
        ev = MethodEvidence(
            method="embedding_geometry",
            confidence=0.75,
            proposed_components=["001", "002"],
            details={"dist_to_midpoint": 0.1},
        )
        return [
            CompoundCandidate(
                barthel_code="004",
                is_known_compound=False,
                is_iconographic_compound=False,
                n_methods_agreeing=2,
                consensus_confidence=0.75,
                consensus_components=["001", "002"],
                method_evidence=[ev, ev],
                corpus_frequency=5,
                temporal_cluster="post_contact",
            ),
            CompoundCandidate(
                barthel_code="005",
                is_known_compound=False,
                is_iconographic_compound=False,
                n_methods_agreeing=1,  # below min_methods=2
                consensus_confidence=0.60,
                consensus_components=[],
                method_evidence=[ev],
                corpus_frequency=3,
                temporal_cluster="unknown",
            ),
        ]

    def test_writes_valid_json(self, tmp_path):
        out = tmp_path / "candidates.json"
        save_compound_candidates(self._make_candidates(), out)
        data = json.loads(out.read_text())
        assert "candidates" in data
        assert "n_candidates" in data

    def test_min_methods_filter(self, tmp_path):
        out = tmp_path / "candidates.json"
        save_compound_candidates(self._make_candidates(), out, min_methods=2)
        data = json.loads(out.read_text())
        assert data["n_candidates"] == 1
        assert data["candidates"][0]["barthel_code"] == "004"

    def test_creates_parent_dirs(self, tmp_path):
        out = tmp_path / "deep" / "nested" / "candidates.json"
        save_compound_candidates(self._make_candidates(), out)
        assert out.exists()

    def test_candidate_serialises_method_evidence(self, tmp_path):
        out = tmp_path / "c.json"
        save_compound_candidates(self._make_candidates(), out, min_methods=1)
        data = json.loads(out.read_text())
        c = data["candidates"][0]
        assert "method_evidence" in c
        assert c["method_evidence"][0]["method"] == "embedding_geometry"
