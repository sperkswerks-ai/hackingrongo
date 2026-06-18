"""
tests.test_sign_fingerprint
===========================

Covers the distributional sign-role classifier:
  (a) feature-vector shape + finiteness on synthetic corpus records,
  (b) threshold logic producing the expected role on hand-built fingerprints,
  (c) diachronic role-stability computation on a known-stable case.

No corpus files required — synthetic records and fingerprints are built inline.
"""

from __future__ import annotations

import math

from hackingrongo.zone_b.sign_fingerprint import (
    SignFingerprint,
    assign_roles,
    compute_features,
    diachronic_stability,
)

_FEATURE_KEYS = {
    "betweenness", "pagerank", "positional_entropy", "neighbor_diversity",
    "own_frequency", "slot_predictability", "passage_anchor_score",
    "direction_skew",
}


def _synthetic_records():
    """A 2-tablet corpus where sign 'A' is frequent (>=5) and well-connected."""
    recs = []
    seq = ["A", "B", "A", "C", "A", "D", "A", "B", "A", "E"]
    for t in ("T1", "T2"):
        for pos, code in enumerate(seq, start=1):
            recs.append({"tablet": t, "side": "a", "line": "01",
                         "position": pos, "code": code, "stratum": "post_contact"})
    return recs


# ---------------------------------------------------------------------------
# (a) feature vector shape + finiteness
# ---------------------------------------------------------------------------

class TestFeatureVector:
    def test_shape_and_finite(self):
        feats, freq = compute_features(_synthetic_records(), boundaries=set(), min_freq=5)
        assert "A" in feats, "frequent sign A should be in the freq>=5 core"
        for code, f in feats.items():
            assert set(f) == _FEATURE_KEYS, f"{code} missing/extra feature keys"
            for k, v in f.items():
                assert isinstance(v, float) and math.isfinite(v), f"{code}.{k} not finite: {v}"
            assert freq[code] >= 5

    def test_below_threshold_excluded(self):
        feats, freq = compute_features(_synthetic_records(), boundaries=set(), min_freq=5)
        # B appears 4×, C/D/E 2× → below 5 → excluded from the core.
        assert "A" in feats
        assert all(c not in feats for c in ("C", "D", "E"))

    def test_passage_anchor_score(self):
        recs = _synthetic_records()
        # Mark every 'A' at position 1 as a boundary on both tablets.
        boundaries = {("T1", 1), ("T2", 1)}
        feats, _ = compute_features(recs, boundaries=boundaries, min_freq=5)
        # A occurs 10× total; 2 of those (pos 1 on each tablet) are boundaries.
        assert abs(feats["A"]["passage_anchor_score"] - 0.2) < 1e-9


# ---------------------------------------------------------------------------
# (b) threshold logic on hand-built fingerprints
# ---------------------------------------------------------------------------

def _feat(betw, ndiv, freq_rel, pent, slot, anchor=0.0, prank=0.0, dskew=0.0):
    return {"betweenness": betw, "pagerank": prank, "positional_entropy": pent,
            "neighbor_diversity": ndiv, "own_frequency": freq_rel,
            "slot_predictability": slot, "passage_anchor_score": anchor,
            "direction_skew": dskew}


class TestThresholdLogic:
    def test_determinative_maps_to_taxogram(self):
        # A strongly successor-diverse sign (proclitic) vs symmetric signs.
        features = {
            "DET": _feat(betw=0.01, ndiv=0.5, freq_rel=0.01, pent=0.9, slot=0.3, dskew=0.7),
            "f1":  _feat(0.01, 0.5, 0.20, 0.9, 0.3, dskew=0.0),
            "f2":  _feat(0.01, 0.5, 0.20, 0.9, 0.3, dskew=0.0),
            "f3":  _feat(0.01, 0.5, 0.20, 0.9, 0.3, dskew=0.05),
            "f4":  _feat(0.01, 0.5, 0.20, 0.9, 0.3, dskew=-0.05),
        }
        freq = {k: 50 for k in features}  # all above the reliability floor
        roles, _ = assign_roles(features, freq)
        assert roles["DET"].role == "taxogram"
        assert roles["DET"].subtype == "determinative"
        assert roles["DET"].rule == "determinative:proclitic"  # +skew ⇒ precedes the class

    def test_postclitic_determinative_side(self):
        # Strong predecessor diversity ⇒ negative skew ⇒ postclitic.
        features = {
            "DET": _feat(0.01, 0.5, 0.01, 0.9, 0.3, dskew=-0.7),
            "f1":  _feat(0.01, 0.5, 0.20, 0.9, 0.3, dskew=0.0),
            "f2":  _feat(0.01, 0.5, 0.20, 0.9, 0.3, dskew=0.0),
        }
        roles, _ = assign_roles(features, {k: 50 for k in features})
        assert roles["DET"].role == "taxogram"
        assert roles["DET"].rule == "determinative:postclitic"

    def test_low_frequency_skew_not_determinative(self):
        # Same strong skew but too few attestations to be reliable → not a taxogram.
        features = {
            "RARE": _feat(0.01, 0.5, 0.01, 0.9, 0.3, dskew=0.9),
            "f1":   _feat(0.01, 0.5, 0.20, 0.9, 0.3, dskew=0.0),
        }
        roles, _ = assign_roles(features, {"RARE": 6, "f1": 50})
        assert roles["RARE"].role != "taxogram"

    def test_particle_maps_to_taxogram_particle(self):
        # High frequency, high slot predictability, low positional entropy.
        features = {
            "PART": _feat(betw=0.01, ndiv=0.3, freq_rel=0.30, pent=0.1, slot=0.95),
            "f1":   _feat(0.01, 0.5, 0.02, 0.9, 0.3),
            "f2":   _feat(0.01, 0.5, 0.02, 0.9, 0.3),
            "f3":   _feat(0.01, 0.5, 0.02, 0.9, 0.3),
        }
        freq = {k: 50 for k in features}
        roles, _ = assign_roles(features, freq)
        assert roles["PART"].role == "taxogram"
        assert roles["PART"].subtype == "particle"

    def test_logogram(self):
        # High positional entropy, low slot predictability, not a bridge.
        features = {
            "LOG": _feat(betw=0.01, ndiv=0.5, freq_rel=0.10, pent=0.95, slot=0.05),
            "f1":  _feat(0.01, 0.5, 0.10, 0.40, 0.80),
            "f2":  _feat(0.01, 0.5, 0.10, 0.40, 0.80),
        }
        freq = {k: 50 for k in features}
        roles, _ = assign_roles(features, freq)
        assert roles["LOG"].role == "logogram"

    def test_anchor_subtype_overrides(self):
        features = {
            "ANC": _feat(betw=0.01, ndiv=0.5, freq_rel=0.10, pent=0.95, slot=0.05, anchor=0.8),
            "f1":  _feat(0.01, 0.5, 0.10, 0.40, 0.80),
        }
        roles, _ = assign_roles(features, {k: 50 for k in features}, anchor_thresh=0.5)
        assert roles["ANC"].subtype == "anchor"

    def test_records_feature_values_for_audit(self):
        features = {"X": _feat(0.5, 1.0, 0.1, 0.5, 0.5)}
        roles, _ = assign_roles(features, {"X": 10})
        # Every assignment carries the feature values that produced it.
        assert roles["X"].features == features["X"]
        assert roles["X"].rule  # non-empty rule string


# ---------------------------------------------------------------------------
# (c) diachronic stability
# ---------------------------------------------------------------------------

class TestDiachronicStability:
    def _fp(self, code, role):
        return SignFingerprint(code=code, frequency=10, features=_feat(0, 0, 0, 0, 0),
                               role=role, subtype=None, rule="x")

    def test_stable_and_changed(self):
        pre = {"S": self._fp("S", "logogram"),   # stable
               "C": self._fp("C", "phonetic"),   # changes
               "P": self._fp("P", "taxogram")}   # pre-only (ignored)
        post = {"S": self._fp("S", "logogram"),
                "C": self._fp("C", "logogram"),
                "Q": self._fp("Q", "phonetic")}  # post-only (ignored)
        out = diachronic_stability(pre, post)
        assert out["n_signs_in_both_strata"] == 2     # S, C
        assert out["role_stability"] == 0.5            # S stable, C changed
        assert out["stable_signs"] == ["S"]
        assert out["role_changes"] == [
            {"code": "C", "pre_role": "phonetic", "post_role": "logogram"}
        ]

    def test_empty_overlap(self):
        out = diachronic_stability({"A": self._fp("A", "logogram")},
                                   {"B": self._fp("B", "phonetic")})
        assert out["n_signs_in_both_strata"] == 0
        assert out["role_stability"] is None
