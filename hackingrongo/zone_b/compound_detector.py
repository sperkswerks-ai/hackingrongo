"""
hackingrongo.zone_b.compound_detector
======================================

Detects new compound glyph candidates — signs that Barthel did not
explicitly mark as compounds but whose embedding geometry or corpus
behaviour is consistent with compound structure.

Background
----------
Barthel (1958) encoded four syntactic compound types using punctuation
in the sign code:

    X:Y   stacked     — one sign above another
    X.Y   linked      — physically connected
    X-Y   juxtaposed  — side by side
    X'Y   fused       — merged into one form

These 187 explicitly-marked compounds represent *known* cases.  The
corpus almost certainly contains additional compounds that Barthel
missed, misclassified as simple signs, or could not resolve because
the underlying components were unclear from the original drawings.

Detection approach
------------------
Three independent methods, each producing a confidence score in [0, 1].
A sign is reported as a compound candidate only when at least two methods
agree (confidence ≥ ``min_confidence``).  This cross-validation design
is consistent with the robustness philosophy used elsewhere in Zone B.

**Method 1 — Embedding neighbourhood geometry (UMAP)**
  A compound sign's embedding should lie between its two constituent
  signs in UMAP space.  For each unknown sign U, we find its k nearest
  neighbours in the UMAP projection.  If two or more neighbours are
  known simple signs S₁, S₂ such that the midpoint of S₁ and S₂ is
  close to U, then U is a candidate compound of (S₁, S₂).
  Confidence = 1 − (dist(U, midpoint(S₁,S₂)) / dist(S₁,S₂)).

**Method 2 — Cluster membership anomaly**
  Signs that fall in the HDBSCAN noise bucket (cluster = -1) but have
  at least two nearest neighbours in different, high-purity clusters
  are anomalous with respect to the sign inventory.  This is the
  expected behaviour of a compound: it doesn't belong to either of its
  component's clusters.  Confidence = purity of the two flanking
  clusters × (1 − noise_prior).

**Method 3 — Corpus position statistics**
  Known compounds have characteristic positional distributions: they
  appear more often at specific syntactic positions (post-taxogram,
  sequence-final) than simple signs of comparable frequency.  A
  candidate sign that matches the positional profile of known compounds
  despite not being marked as one scores positively on this method.
  Confidence = cosine similarity between the sign's positional feature
  vector and the mean positional vector of known compounds.

Public API
----------
``CompoundCandidate``
    Dataclass: one candidate with multi-method evidence.

``CompoundDetector``
    Main class.  ``detect(umap_df)`` → ``list[CompoundCandidate]``.

``load_known_compounds``
    Load the ground-truth set from corpus JSON files.

``save_compound_candidates``
    Write candidates to JSON for downstream use and scholar review.
"""

from __future__ import annotations

import json
import logging
import math
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Barthel punctuation characters that mark explicit syntactic compounds.
COMPOUND_SEPARATORS: tuple[str, ...] = (":", ".", "-", "'")

# Barthel numeric ranges classified as iconographic compounds.
# Bird-headed (600-699) and other zoomorphic (700-799) — Barthel's
# iconographic compound class, distinct from syntactic compounds.
_ICONOGRAPHIC_COMPOUND_RANGES: list[tuple[int, int]] = [
    (600, 699),
    (700, 799),
]

# Positional features used for Method 3.
_POSITION_FEATURES = (
    "frac_post_taxogram",   # fraction of occurrences that follow glyph 200
    "frac_seq_final",       # fraction at end of line
    "frac_seq_initial",     # fraction at start of line
    "mean_relative_pos",    # mean position / line length
    "bigram_entropy",       # Shannon entropy of bigram context
)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class MethodEvidence:
    """Evidence from one detection method for a compound candidate.

    Attributes
    ----------
    method : str
        One of ``"embedding_geometry"``, ``"cluster_anomaly"``,
        ``"positional_profile"``.
    confidence : float
        Score in [0, 1]; higher = more likely compound.
    proposed_components : list[str]
        The two (or more) constituent sign codes proposed by this method.
        Empty list if the method cannot resolve components.
    details : dict[str, Any]
        Method-specific supporting statistics for scholar review.
    """

    method: str
    confidence: float
    proposed_components: list[str] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class CompoundCandidate:
    """A sign that two or more detection methods flag as a compound.

    Attributes
    ----------
    barthel_code : str
        The candidate sign's Barthel code.
    is_known_compound : bool
        True if Barthel already marked this as a compound (syntactic
        punctuation in code).  Used for precision validation.
    is_iconographic_compound : bool
        True if the code falls in Barthel's iconographic compound range
        (600-799).  These are structurally different from syntactic
        compounds and are flagged for scholar disambiguation.
    n_methods_agreeing : int
        Number of detection methods that flag this sign (1–3).
    consensus_confidence : float
        Mean confidence across all agreeing methods.
    consensus_components : list[str]
        Component codes agreed on by the most methods.  May be empty
        if methods disagree on decomposition.
    method_evidence : list[MethodEvidence]
        Full per-method evidence for transparent scholar review.
    corpus_frequency : int
        Total occurrences in the corpus.
    temporal_cluster : str
        ``"pre_contact"``, ``"post_contact"``, or ``"mixed"`` —
        whether this sign appears predominantly in one stratum.
    """

    barthel_code: str
    is_known_compound: bool
    is_iconographic_compound: bool
    n_methods_agreeing: int
    consensus_confidence: float
    consensus_components: list[str]
    method_evidence: list[MethodEvidence]
    corpus_frequency: int = 0
    temporal_cluster: str = "unknown"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Ground-truth loading
# ---------------------------------------------------------------------------


def load_known_compounds(corpus_dir: Path) -> dict[str, list[str]]:
    """Load explicitly marked compound codes from the corpus.

    Parameters
    ----------
    corpus_dir : Path
        Directory containing per-tablet JSON files.

    Returns
    -------
    dict[str, list[str]]
        Maps compound Barthel code → list of constituent codes
        (from ``horley_components`` field; may be partial or empty).
    """
    known: dict[str, list[str]] = {}
    for jf in sorted(corpus_dir.glob("*.json")):
        tablet = json.loads(jf.read_text(encoding="utf-8"))
        for g in tablet.get("glyphs", []):
            bc = str(g.get("barthel_code", ""))
            if any(sep in bc for sep in COMPOUND_SEPARATORS):
                components = g.get("horley_components") or []
                if bc not in known:
                    known[bc] = list(components)
    logger.info("Known compounds loaded: %d unique codes.", len(known))
    return known


def _is_syntactic_compound(code: str) -> bool:
    return any(sep in str(code) for sep in COMPOUND_SEPARATORS)


def _is_iconographic_compound(code: str) -> bool:
    digits = re.sub(r"[^0-9]", "", str(code))[:3]
    if not digits:
        return False
    n = int(digits)
    return any(lo <= n <= hi for lo, hi in _ICONOGRAPHIC_COMPOUND_RANGES)


# ---------------------------------------------------------------------------
# UMAP feature extraction
# ---------------------------------------------------------------------------


def _build_sign_centroids(
    umap_df: pd.DataFrame,
    exclude_codes: set[str] | None = None,
) -> pd.DataFrame:
    """Compute per-sign mean UMAP position from all instances.

    Parameters
    ----------
    umap_df : pd.DataFrame
        cluster_vs_barthel.csv with columns
        ``barthel_code``, ``umap_x``, ``umap_y``, ``hdbscan_cluster``.
    exclude_codes : set[str], optional
        Codes to exclude from the centroid table (e.g. known compounds).

    Returns
    -------
    pd.DataFrame
        Index = barthel_code; columns = ``cx``, ``cy``, ``n``,
        ``cluster_purity``, ``dominant_cluster``.
    """
    valid = umap_df[
        ~umap_df["barthel_code"].isin({"?", "", "nan"})
        & umap_df["barthel_code"].notna()
    ].copy()
    if exclude_codes:
        valid = valid[~valid["barthel_code"].isin(exclude_codes)]

    centroids = valid.groupby("barthel_code").agg(
        cx=("umap_x", "mean"),
        cy=("umap_y", "mean"),
        n=("umap_x", "count"),
    )

    # Cluster purity: fraction of instances in the dominant cluster
    def _purity(grp: pd.DataFrame) -> tuple[float, int]:
        non_noise = grp[grp["hdbscan_cluster"] != -1]
        if len(non_noise) == 0:
            return 0.0, -1
        dominant = non_noise["hdbscan_cluster"].mode().iloc[0]
        purity = (non_noise["hdbscan_cluster"] == dominant).sum() / len(grp)
        return float(purity), int(dominant)

    purity_data = (
        valid.groupby("barthel_code")
        .apply(_purity, include_groups=False)
        .reset_index()
    )
    purity_data.columns = ["barthel_code", "_purity_tuple"]
    purity_data["cluster_purity"] = purity_data["_purity_tuple"].apply(lambda x: x[0])
    purity_data["dominant_cluster"] = purity_data["_purity_tuple"].apply(lambda x: x[1])
    purity_data = purity_data.drop(columns=["_purity_tuple"]).set_index("barthel_code")

    return centroids.join(purity_data, how="left")


# ---------------------------------------------------------------------------
# Method 1: Embedding neighbourhood geometry
# ---------------------------------------------------------------------------


def _method_embedding_geometry(
    candidate_code: str,
    candidate_centroid: np.ndarray,
    simple_centroids: pd.DataFrame,
    k_neighbours: int = 20,
    min_interpoint_dist: float = 0.5,
) -> MethodEvidence | None:
    """Test whether the candidate lies at the midpoint of two simple signs.

    Parameters
    ----------
    candidate_code : str
    candidate_centroid : np.ndarray
        Shape (2,) — UMAP (x, y).
    simple_centroids : pd.DataFrame
        Centroids of all simple (non-compound) signs.
    k_neighbours : int
        Number of nearest simple-sign neighbours to consider.
    min_interpoint_dist : float
        Minimum UMAP distance between the two proposed components;
        pairs closer than this are likely allographs, not compounds.

    Returns
    -------
    MethodEvidence or None
        None if no plausible compound decomposition was found.
    """
    if len(simple_centroids) < 2:
        return None

    pts = simple_centroids[["cx", "cy"]].values
    codes = simple_centroids.index.tolist()

    # Find k nearest simple-sign neighbours
    dists = np.sqrt(((pts - candidate_centroid) ** 2).sum(axis=1))
    nn_idx = np.argsort(dists)[:k_neighbours]

    best_confidence = 0.0
    best_pair: tuple[str, str] | None = None
    best_details: dict[str, Any] = {}

    for i in range(len(nn_idx)):
        for j in range(i + 1, len(nn_idx)):
            ii, jj = nn_idx[i], nn_idx[j]
            c1 = pts[ii]
            c2 = pts[jj]
            interpoint_dist = float(np.linalg.norm(c1 - c2))
            if interpoint_dist < min_interpoint_dist:
                continue  # too close — likely allographs

            # Geometric betweenness guard: the candidate must be closer to
            # both components than the components are to each other.  Without
            # this, signs that are beside (not between) a pair still score.
            dist_to_c1 = float(dists[ii])
            dist_to_c2 = float(dists[jj])
            if dist_to_c1 > interpoint_dist or dist_to_c2 > interpoint_dist:
                continue

            midpoint = (c1 + c2) / 2.0
            dist_to_mid = float(np.linalg.norm(candidate_centroid - midpoint))

            # Confidence: 1 at midpoint, 0 when dist_to_mid = interpoint_dist/2.
            # The "between" zone is dist_to_mid < interpoint_dist/2 — the old
            # formula (dividing by interpoint_dist) was twice as permissive.
            confidence = max(0.0, 1.0 - 2.0 * dist_to_mid / (interpoint_dist + 1e-9))

            if confidence > best_confidence:
                best_confidence = confidence
                best_pair = (codes[ii], codes[jj])
                best_details = {
                    "component_1": codes[ii],
                    "component_2": codes[jj],
                    "dist_to_midpoint": round(dist_to_mid, 4),
                    "interpoint_dist": round(interpoint_dist, 4),
                    "dist_to_c1": round(dist_to_c1, 4),
                    "dist_to_c2": round(dist_to_c2, 4),
                }

    if best_confidence < 0.1 or best_pair is None:
        return None

    return MethodEvidence(
        method="embedding_geometry",
        confidence=round(best_confidence, 4),
        proposed_components=list(best_pair),
        details=best_details,
    )


# ---------------------------------------------------------------------------
# Method 2: Cluster membership anomaly
# ---------------------------------------------------------------------------


def _method_cluster_anomaly(
    candidate_code: str,
    umap_df: pd.DataFrame,
    simple_centroids: pd.DataFrame,
    noise_prior: float = 0.13,
    k_neighbours: int = 10,
) -> MethodEvidence | None:
    """Score a sign based on whether it is a cluster-boundary outlier.

    A compound sign is expected to:
    - Fall in or near the noise bucket (no cluster of its own), AND
    - Have nearest neighbours in at least two different clusters.

    Parameters
    ----------
    candidate_code : str
    umap_df : pd.DataFrame
    simple_centroids : pd.DataFrame
    noise_prior : float
        Baseline noise rate in the corpus (from HDBSCAN stats).
    k_neighbours : int
    """
    # Get all instances of this sign
    sign_rows = umap_df[umap_df["barthel_code"] == candidate_code]
    if len(sign_rows) == 0:
        return None

    # Fraction of instances that are noise
    noise_frac = float((sign_rows["hdbscan_cluster"] == -1).mean())

    # Must be substantially noisier than baseline
    if noise_frac < noise_prior * 1.5:
        return None

    # Find nearest simple-sign neighbours for the centroid
    candidate_centroid = sign_rows[["umap_x", "umap_y"]].mean().values
    pts = simple_centroids[["cx", "cy"]].values
    codes = simple_centroids.index.tolist()

    dists = np.sqrt(((pts - candidate_centroid) ** 2).sum(axis=1))
    nn_idx = np.argsort(dists)[:k_neighbours]

    # Get the dominant clusters of nearest neighbours
    neighbour_clusters: list[int] = []
    neighbour_purities: list[float] = []
    for idx in nn_idx:
        code = codes[idx]
        if "dominant_cluster" in simple_centroids.columns:
            dc = int(simple_centroids.loc[code, "dominant_cluster"])
            pu = float(simple_centroids.loc[code, "cluster_purity"])
            if dc != -1:
                neighbour_clusters.append(dc)
                neighbour_purities.append(pu)

    unique_clusters = set(neighbour_clusters)
    if len(unique_clusters) < 2:
        return None

    # Confidence: noise excess × mean purity of flanking clusters
    noise_excess = (noise_frac - noise_prior) / (1.0 - noise_prior + 1e-9)
    mean_purity = float(np.mean(neighbour_purities)) if neighbour_purities else 0.0
    confidence = min(1.0, noise_excess * mean_purity)

    if confidence < 0.1:
        return None

    # Propose constituents as the two nearest neighbours from different clusters
    cluster_to_nearest: dict[int, tuple[str, float]] = {}
    for idx in nn_idx:
        code = codes[idx]
        cl = (
            int(simple_centroids.loc[code, "dominant_cluster"])
            if "dominant_cluster" in simple_centroids.columns
            else -1
        )
        d = float(dists[idx])
        if cl != -1 and (cl not in cluster_to_nearest or d < cluster_to_nearest[cl][1]):
            cluster_to_nearest[cl] = (code, d)

    proposed = [v[0] for v in sorted(cluster_to_nearest.values(), key=lambda x: x[1])[:2]]

    return MethodEvidence(
        method="cluster_anomaly",
        confidence=round(confidence, 4),
        proposed_components=proposed,
        details={
            "noise_fraction": round(noise_frac, 4),
            "noise_prior": round(noise_prior, 4),
            "n_instances": int(len(sign_rows)),
            "n_unique_neighbour_clusters": len(unique_clusters),
            "mean_neighbour_purity": round(mean_purity, 4),
        },
    )


# ---------------------------------------------------------------------------
# Method 3: Positional profile
# ---------------------------------------------------------------------------


def _build_positional_features(
    code: str,
    corpus_sequences: list[list[str]],
    taxogram_code: str = "200",
) -> dict[str, float]:
    """Compute positional feature vector for a sign code."""
    post_taxogram = 0
    seq_final = 0
    seq_initial = 0
    total_positions: list[float] = []
    bigram_counts: Counter = Counter()
    total = 0

    for seq in corpus_sequences:
        n = len(seq)
        for pos, token in enumerate(seq):
            if token != code:
                continue
            total += 1
            if pos > 0 and seq[pos - 1] == taxogram_code:
                post_taxogram += 1
            if pos == n - 1:
                seq_final += 1
            if pos == 0:
                seq_initial += 1
            total_positions.append(pos / max(n - 1, 1))
            prev_tok = seq[pos - 1] if pos > 0 else "<BOS>"
            next_tok = seq[pos + 1] if pos < n - 1 else "<EOS>"
            bigram_counts[(prev_tok, next_tok)] += 1

    if total == 0:
        return {f: 0.0 for f in _POSITION_FEATURES}

    bg_total = sum(bigram_counts.values())
    bigram_entropy = 0.0
    for cnt in bigram_counts.values():
        p = cnt / bg_total
        if p > 0:
            bigram_entropy -= p * math.log2(p)

    return {
        "frac_post_taxogram": post_taxogram / total,
        "frac_seq_final": seq_final / total,
        "frac_seq_initial": seq_initial / total,
        "mean_relative_pos": float(np.mean(total_positions)),
        "bigram_entropy": bigram_entropy,
    }


def _compute_positional_profile_stats(
    codes: list[str],
    corpus_sequences: list[list[str]],
    min_corpus_count: int = 3,
    cap: int = 200,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute mean and std of positional feature vectors for *codes*.

    Called once per :meth:`CompoundDetector.detect` run, not per candidate.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``(mean, std)`` each of shape ``(len(_POSITION_FEATURES),)``.
        Returns ``(zeros, ones)`` when fewer than 2 codes qualify.
    """
    feat_vecs: list[list[float]] = []
    for code in codes[:cap]:
        if sum(seq.count(code) for seq in corpus_sequences) < min_corpus_count:
            continue
        feats = _build_positional_features(code, corpus_sequences)
        feat_vecs.append([feats[f] for f in _POSITION_FEATURES])
    if len(feat_vecs) < 2:
        n = len(_POSITION_FEATURES)
        return np.zeros(n), np.ones(n)
    arr = np.array(feat_vecs)
    return arr.mean(axis=0), arr.std(axis=0) + 1e-9


def _method_positional_profile(
    candidate_code: str,
    corpus_sequences: list[list[str]],
    compound_mean: np.ndarray,
    compound_std: np.ndarray,
    simple_mean: np.ndarray,
    simple_std: np.ndarray,
    min_corpus_count: int = 3,
) -> MethodEvidence | None:
    """Score a candidate by comparing its positional profile to both the
    compound and simple-sign distributions.

    A sign is flagged only when it is distinctively closer to the compound
    distribution than to the simple distribution *and* within 1.5 standard
    deviations of the compound mean.  The old implementation compared only
    to the compound mean with a 3 SD threshold, which fired on the vast
    majority of signs regardless of their true nature.

    Parameters
    ----------
    candidate_code : str
    corpus_sequences : list[list[str]]
        Glyph token sequences from the corpus.
    compound_mean, compound_std : np.ndarray
        Precomputed from :func:`_compute_positional_profile_stats` on known
        compound codes.  Must be computed once per detect() call, not here.
    simple_mean, simple_std : np.ndarray
        Same for a representative sample of simple (non-compound) signs.
    min_corpus_count : int
        Skip candidates with fewer than this many occurrences.
    """
    total = sum(seq.count(candidate_code) for seq in corpus_sequences)
    if total < min_corpus_count:
        return None

    candidate_feats = _build_positional_features(candidate_code, corpus_sequences)
    candidate_vec = np.array([candidate_feats[f] for f in _POSITION_FEATURES])

    # Mean absolute z-score distance to each reference distribution.
    compound_z = float(np.mean(np.abs((candidate_vec - compound_mean) / compound_std)))
    simple_z = float(np.mean(np.abs((candidate_vec - simple_mean) / simple_std)))

    # Gate 1: candidate must be within 1.5 SD of the compound distribution.
    # (The old threshold of 3.0 SD covers ~95% of any normal distribution.)
    if compound_z >= 1.5:
        return None

    # Gate 2: discriminative margin must be positive — candidate must be
    # measurably more compound-like than simple-like.
    margin = simple_z - compound_z
    confidence = max(0.0, min(1.0, margin / 2.0))

    if confidence < 0.1:
        return None

    return MethodEvidence(
        method="positional_profile",
        confidence=round(confidence, 4),
        proposed_components=[],  # positional method cannot resolve components
        details={
            "corpus_frequency": total,
            "compound_z": round(compound_z, 4),
            "simple_z": round(simple_z, 4),
            "discriminative_margin": round(margin, 4),
            **{f"feat_{k}": round(v, 4) for k, v in candidate_feats.items()},
        },
    )


# ---------------------------------------------------------------------------
# Main detector
# ---------------------------------------------------------------------------


class CompoundDetector:
    """Detect new compound glyph candidates from Zone A embedding outputs.

    Parameters
    ----------
    corpus_dir : Path
        Directory containing per-tablet corpus JSON files.
    min_confidence : float
        Minimum per-method confidence to count a method as agreeing.
        Default 0.25.
    min_methods : int
        Minimum number of agreeing methods for a sign to be reported.
        Default 2 (cross-validation requirement).
    k_neighbours : int
        Nearest-neighbour count for embedding geometry and cluster
        anomaly methods.  Default 20.
    exclude_known : bool
        If True, skip signs already explicitly marked as compounds by
        Barthel.  Set False to run validation on known compounds.
        Default True.
    """

    def __init__(
        self,
        corpus_dir: Path,
        min_confidence: float = 0.25,
        min_methods: int = 2,
        k_neighbours: int = 20,
        exclude_known: bool = True,
    ) -> None:
        self._corpus_dir = corpus_dir
        self._min_confidence = min_confidence
        self._min_methods = min_methods
        self._k_neighbours = k_neighbours
        self._exclude_known = exclude_known

        self._known_compounds = load_known_compounds(corpus_dir)
        self._known_compound_codes = set(self._known_compounds.keys())

    # ------------------------------------------------------------------
    # Corpus sequence loading
    # ------------------------------------------------------------------

    def _load_corpus_sequences(self) -> list[list[str]]:
        """Load all glyph sequences from the corpus in tablet order."""
        sequences: list[list[str]] = []
        for jf in sorted(self._corpus_dir.glob("*.json")):
            tablet = json.loads(jf.read_text(encoding="utf-8"))
            tokens = [
                str(g.get("barthel_code", ""))
                for g in tablet.get("glyphs", [])
                if g.get("barthel_code")
            ]
            if tokens:
                sequences.append(tokens)
        return sequences

    # ------------------------------------------------------------------
    # Per-sign temporal cluster assignment
    # ------------------------------------------------------------------

    @staticmethod
    def _temporal_cluster(code: str, umap_df: pd.DataFrame) -> str:
        rows = umap_df[umap_df["barthel_code"] == code]
        if len(rows) == 0:
            return "unknown"
        if "temporal_cluster" in rows.columns:
            clusters = rows["temporal_cluster"].dropna().unique().tolist()
            if len(clusters) == 1:
                return str(clusters[0])
            return "mixed"
        return "unknown"

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def detect(
        self,
        umap_df: pd.DataFrame,
    ) -> list[CompoundCandidate]:
        """Run all three detection methods and return compound candidates.

        Parameters
        ----------
        umap_df : pd.DataFrame
            Output of ``analyze_embeddings.py``: columns
            ``barthel_code``, ``barthel_family``, ``umap_x``,
            ``umap_y``, ``hdbscan_cluster``.

        Returns
        -------
        list[CompoundCandidate]
            Sorted by ``(n_methods_agreeing DESC, consensus_confidence DESC)``.
            Only signs with ``n_methods_agreeing >= min_methods`` are included.
        """
        logger.info("CompoundDetector: loading corpus sequences…")
        corpus_sequences = self._load_corpus_sequences()

        logger.info("CompoundDetector: building UMAP centroids…")
        non_compound_codes = {
            c for c in umap_df["barthel_code"].unique()
            if c and c != "?"
            and not _is_syntactic_compound(c)
            and not _is_iconographic_compound(c)
            and c not in self._known_compound_codes
        }
        simple_centroids = _build_sign_centroids(
            umap_df,
            exclude_codes=self._known_compound_codes,
        )
        simple_centroids = simple_centroids[
            simple_centroids.index.isin(non_compound_codes)
        ]
        logger.info("  Simple sign centroids: %d", len(simple_centroids))

        # Precompute positional profiles once — the old design recomputed the
        # compound distribution inside _method_positional_profile on every
        # candidate call (O(n_candidates × n_compounds × corpus)).
        logger.info("CompoundDetector: precomputing positional profiles…")
        all_corpus_codes = sorted({
            tok for seq in corpus_sequences for tok in seq
            if tok and tok != "?"
        })
        simple_codes_for_profile = [
            c for c in all_corpus_codes
            if not _is_syntactic_compound(c)
            and not _is_iconographic_compound(c)
            and c not in self._known_compound_codes
        ]
        compound_pos_mean, compound_pos_std = _compute_positional_profile_stats(
            list(self._known_compound_codes),
            corpus_sequences,
            min_corpus_count=3,
            cap=200,
        )
        simple_pos_mean, simple_pos_std = _compute_positional_profile_stats(
            simple_codes_for_profile,
            corpus_sequences,
            min_corpus_count=3,
            cap=300,
        )
        logger.info(
            "  Positional profile: %d compound codes, %d simple codes.",
            min(len(self._known_compound_codes), 200),
            min(len(simple_codes_for_profile), 300),
        )

        noise_prior = float(
            (umap_df["hdbscan_cluster"] == -1).sum() / max(len(umap_df), 1)
        )
        logger.info("  Corpus noise prior: %.3f", noise_prior)

        if self._exclude_known:
            candidate_pool = [
                c for c in umap_df["barthel_code"].unique()
                if c and c != "?"
                and not _is_syntactic_compound(c)
            ]
        else:
            candidate_pool = [
                c for c in umap_df["barthel_code"].unique()
                if c and c != "?"
            ]

        logger.info(
            "CompoundDetector: running detection on %d candidates…",
            len(candidate_pool),
        )

        candidates: list[CompoundCandidate] = []

        for code in candidate_pool:
            code_rows = umap_df[umap_df["barthel_code"] == code]
            if len(code_rows) == 0:
                continue

            centroid = code_rows[["umap_x", "umap_y"]].mean().values

            evidence: list[MethodEvidence] = []

            ev1 = _method_embedding_geometry(
                code, centroid, simple_centroids,
                k_neighbours=self._k_neighbours,
            )
            if ev1 is not None and ev1.confidence >= self._min_confidence:
                evidence.append(ev1)

            ev2 = _method_cluster_anomaly(
                code, umap_df, simple_centroids,
                noise_prior=noise_prior,
                k_neighbours=self._k_neighbours,
            )
            if ev2 is not None and ev2.confidence >= self._min_confidence:
                evidence.append(ev2)

            ev3 = _method_positional_profile(
                code, corpus_sequences,
                compound_pos_mean, compound_pos_std,
                simple_pos_mean, simple_pos_std,
            )
            if ev3 is not None and ev3.confidence >= self._min_confidence:
                evidence.append(ev3)

            if len(evidence) < self._min_methods:
                continue

            # Consensus components: codes proposed by the most methods
            all_proposed: list[str] = []
            for ev in evidence:
                all_proposed.extend(ev.proposed_components)
            component_votes = Counter(all_proposed)
            consensus_components = [
                c for c, _ in component_votes.most_common(2)
                if component_votes[c] >= 1
            ]

            consensus_confidence = float(np.mean([ev.confidence for ev in evidence]))
            freq = int(len(code_rows))

            candidates.append(
                CompoundCandidate(
                    barthel_code=code,
                    is_known_compound=code in self._known_compound_codes,
                    is_iconographic_compound=_is_iconographic_compound(code),
                    n_methods_agreeing=len(evidence),
                    consensus_confidence=round(consensus_confidence, 4),
                    consensus_components=consensus_components,
                    method_evidence=evidence,
                    corpus_frequency=freq,
                    temporal_cluster=self._temporal_cluster(code, umap_df),
                )
            )

        candidates.sort(key=lambda c: (-c.n_methods_agreeing, -c.consensus_confidence))
        logger.info(
            "CompoundDetector: %d candidates found (%d with all 3 methods agreeing).",
            len(candidates),
            sum(1 for c in candidates if c.n_methods_agreeing == 3),
        )
        return candidates


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def save_compound_candidates(
    candidates: list[CompoundCandidate],
    output_path: Path,
    min_methods: int = 2,
) -> None:
    """Write compound candidates to a JSON file for downstream use.

    Parameters
    ----------
    candidates : list[CompoundCandidate]
    output_path : Path
        Destination JSON path.  Parent directories are created.
    min_methods : int
        Only write candidates with at least this many agreeing methods.
    """
    filtered = [c for c in candidates if c.n_methods_agreeing >= min_methods]
    payload = {
        "n_candidates": len(filtered),
        "n_all_methods": sum(1 for c in filtered if c.n_methods_agreeing == 3),
        "n_two_methods": sum(1 for c in filtered if c.n_methods_agreeing == 2),
        "candidates": [c.to_dict() for c in filtered],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    logger.info(
        "Compound candidates written: %d entries → %s", len(filtered), output_path
    )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def _parse_args():
    import argparse

    p = argparse.ArgumentParser(
        description="Detect new compound glyph candidates from Zone A embeddings."
    )
    p.add_argument(
        "--analysis-dir",
        type=Path,
        default=Path("outputs/analysis"),
        help="Directory containing cluster_vs_barthel.csv.",
    )
    p.add_argument(
        "--corpus-dir",
        type=Path,
        default=Path("data/corpus"),
        help="Directory containing per-tablet corpus JSON files.",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSON path (default: <analysis-dir>/compound_candidates.json).",
    )
    p.add_argument(
        "--min-confidence",
        type=float,
        default=0.25,
        help="Minimum per-method confidence threshold (default: 0.25).",
    )
    p.add_argument(
        "--min-methods",
        type=int,
        default=2,
        help="Minimum agreeing methods to report a candidate (default: 2).",
    )
    p.add_argument(
        "--include-known",
        action="store_true",
        help="Include known Barthel-marked compounds (for precision validation).",
    )
    return p.parse_args()


def main() -> None:
    import logging as _logging

    _logging.basicConfig(level=_logging.INFO, format="%(levelname)s  %(message)s")
    args = _parse_args()

    csv_path = args.analysis_dir / "cluster_vs_barthel.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"cluster_vs_barthel.csv not found at {csv_path}. "
            "Run scripts/analyze_embeddings.py first."
        )

    umap_df = pd.read_csv(csv_path)

    detector = CompoundDetector(
        corpus_dir=args.corpus_dir,
        min_confidence=args.min_confidence,
        min_methods=args.min_methods,
        exclude_known=not args.include_known,
    )

    candidates = detector.detect(umap_df)

    output_path = args.output or (args.analysis_dir / "compound_candidates.json")
    save_compound_candidates(candidates, output_path, min_methods=args.min_methods)

    print(f"\n── Compound Detection Results ──────────────────────────")
    print(f"  Total candidates:             {len(candidates)}")
    print(f"  All 3 methods agreeing:       {sum(1 for c in candidates if c.n_methods_agreeing == 3)}")
    print(f"  2 methods agreeing:           {sum(1 for c in candidates if c.n_methods_agreeing == 2)}")
    print(f"  Including known (validation): {sum(1 for c in candidates if c.is_known_compound)}")
    print(f"\nTop 10 candidates:")
    print(f"  {'Code':<20} {'Conf':>6}  {'Methods':>7}  {'Components'}")
    print(f"  {'-'*20} {'-'*6}  {'-'*7}  {'-'*20}")
    for c in candidates[:10]:
        comp_str = " + ".join(c.consensus_components) if c.consensus_components else "—"
        known_str = " [KNOWN]" if c.is_known_compound else ""
        print(f"  {c.barthel_code:<20} {c.consensus_confidence:>6.3f}  {c.n_methods_agreeing:>7}  {comp_str}{known_str}")
    print(f"\nFull results written to: {output_path}")


if __name__ == "__main__":
    main()
