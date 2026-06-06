"""analyze_embeddings.py — Post-training visual sign space analysis.

Loads autoencoder embeddings from outputs/embeddings_cache.pt, projects them
to 2D with UMAP, clusters with HDBSCAN, then compares the data-driven clusters
against Barthel sign families.

Usage
-----
    python scripts/analyze_embeddings.py

The script reads all parameters from conf/config.yaml (analysis.* keys).

Outputs (written to outputs/analysis/)
---------------------------------------
umap_embeddings.png        — scatter plot coloured by Barthel code
cluster_vs_barthel.json    — per-cluster breakdown: which Barthel codes appear
cluster_vs_barthel.csv     — flat table: glyph_id, barthel_code, umap_x,
                             umap_y, hdbscan_cluster
hierarchy_vs_barthel.json  — cophenetic correlation between the embedding
                             dendrogram and Barthel's implicit 3-level tree
                             (century → tens group → units) for signs 200–399
hierarchy_vs_barthel.png   — side-by-side dendrogram comparison
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import DictConfig

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_embeddings(cache_path: Path) -> tuple[np.ndarray, list[str | None]]:
    """Load embeddings cache produced by Zone A training.

    Expected format: dict with keys
        "embeddings"    — float tensor (N, D)
        "barthel_codes" — list[str | None] of length N  (may be omitted)

    Written by :func:`save_embeddings_cache` or by the training pipeline.
    Returns (embeddings np.ndarray, barthel_codes list).
    """
    data = torch.load(cache_path, map_location="cpu", weights_only=True)
    if isinstance(data, dict) and "embeddings" in data:
        # Current format: {"embeddings": Tensor(N,D), "barthel_codes": list}
        embeddings = data["embeddings"].float().numpy()
        codes = data.get("barthel_codes", [None] * len(embeddings))
    else:
        # Legacy format saved by older train_autoencoder.py:
        # dict[(tablet_id, position) -> np.ndarray]  — no barthel codes
        keys = sorted(data.keys())
        embeddings = np.stack([np.asarray(data[k], dtype=np.float32) for k in keys])
        codes = [None] * len(embeddings)
    return embeddings, list(codes)


def save_embeddings_cache(
    cache_path: Path,
    embeddings: np.ndarray,
    barthel_codes: list[str | None],
) -> None:
    """Write an embeddings cache file in the format expected by this script.

    Call this from the Zone A training pipeline after :func:`extract_embeddings`
    returns::

        emb_dict = extract_embeddings(model, dataset, cfg, device)
        codes = [dataset.tokens[i].barthel_code for i in range(len(dataset))]
        vecs  = np.stack([emb_dict[(t.tablet_id, t.position)]
                          for t in dataset.tokens])
        save_embeddings_cache(Path(cfg.paths.embeddings_cache), vecs, codes)

    Parameters
    ----------
    cache_path : Path
        Destination ``.pt`` file.  Parent directories are created.
    embeddings : numpy.ndarray
        Float32 array of shape ``(N, D)``.
    barthel_codes : list[str | None]
        Parallel list of Barthel codes (``None`` for uncoded glyphs).
    """
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "embeddings": torch.from_numpy(embeddings.astype(np.float32)),
            "barthel_codes": list(barthel_codes),
        },
        cache_path,
    )
    log.info("Embeddings cache written: %s (%d vectors)", cache_path, len(embeddings))


def _barthel_family(code: str | None) -> str:
    """Return the iconographic family for a Barthel code.

    Delegates to ``barthel_families.json`` via :mod:`hackingrongo.data.passage_alignment`
    so the lookup is consistent across the entire codebase.  Families are from
    Barthel (1958) as documented in Fischer (1997): anthropomorphic, zoomorphic,
    botanical, celestial, geometric, composite, positional, or unknown.

    Using the JSON lookup avoids arithmetic derivation (``code // 100``) which
    conflates Barthel's *ordering* with his *iconographic taxonomy*.
    """
    if not code:
        return "unlabeled"
    try:
        from hackingrongo.data.passage_alignment import _get_family
        result = _get_family(code)
        return result if result else "unlabeled"
    except ImportError:
        pass
    # Fallback: load directly if import unavailable
    _fam_map: dict[str, str] = {}
    try:
        import json as _json
        _p = Path(__file__).resolve().parents[1] / "data" / "catalog" / "barthel_families.json"
        if _p.exists():
            _raw = _json.loads(_p.read_text(encoding="utf-8"))
            _fam_map = {k: v for k, v in _raw.items() if not k.startswith("_")}
    except Exception:
        pass
    if not code:
        return "unlabeled"
    fam = _fam_map.get(code)
    if fam:
        return fam
    digits = "".join(c for c in code if c.isdigit())
    if digits:
        padded = digits.zfill(3)
        fam = _fam_map.get(padded) or _fam_map.get(digits)
        if fam:
            return fam
    return "unlabeled"


def _adjusted_rand_index(labels_true: list, labels_pred: list) -> float:
    """Compute Adjusted Rand Index between two label arrays."""
    from sklearn.metrics import adjusted_rand_score
    return float(adjusted_rand_score(labels_true, labels_pred))


# ---------------------------------------------------------------------------
# Hierarchy analysis helpers
# ---------------------------------------------------------------------------

def _barthel_tree_distance(na: int, nb: int) -> int:
    """Tree distance between two integer Barthel codes in the 200–399 range.

    Barthel's numbering encodes a 3-level ultrametric tree:

        Root (all anthropomorphic)
          ├─ 2XX (century group)
          │    ├─ 20X  …  29X  (tens group = head type)
          │    │    └─ 200 … 299  (leaf = individual sign)
          └─ 3XX
               ├─ 30X  …  39X
               └─ 300 … 399

    Distances are 0 (same sign) / 1 (same tens) / 2 (same century,
    different tens) / 3 (different centuries).
    """
    if na == nb:
        return 0
    if na // 10 == nb // 10:
        return 1
    if na // 100 == nb // 100:  # within same century block (200s or 300s)
        return 2
    return 3


def _build_barthel_condensed(codes_int: list[int]) -> np.ndarray:
    """Return a condensed distance vector for Barthel's tree over *codes_int*.

    The vector ordering matches :func:`scipy.spatial.distance.pdist` —
    upper-triangle row-major.
    """
    n = len(codes_int)
    dists: list[int] = []
    for i in range(n):
        for j in range(i + 1, n):
            dists.append(_barthel_tree_distance(codes_int[i], codes_int[j]))
    return np.array(dists, dtype=np.float64)


def run_hierarchy_analysis(
    embeddings: np.ndarray,
    codes: list[str | None],
    cfg: "DictConfig",
    out_dir: Path,
) -> dict:
    """Compare the embedding dendrogram to Barthel's implicit sign tree.

    Restricts analysis to signs 200–399 (Barthel's anthropomorphic blocks),
    which encode a 3-level tree (century → tens group → units digit).

    Aggregates all corpus instances for each sign code to a mean embedding,
    then runs agglomerative clustering and computes the cophenetic correlation
    with Barthel's pairwise tree distances.

    Parameters
    ----------
    embeddings : ndarray, shape (N, D)
        All autoencoder embeddings.
    codes : list[str | None]
        Parallel Barthel code for each embedding.
    cfg : DictConfig
        Root Hydra config (reads ``analysis.hierarchy_analysis``).
    out_dir : Path
        Directory where JSON and PNG outputs are written.

    Returns
    -------
    dict
        Result record suitable for JSON serialisation.
    """
    try:
        from scipy.cluster.hierarchy import linkage, cophenet, dendrogram
        from scipy.spatial.distance import pdist
    except ImportError:
        raise ImportError("scipy is required: pip install scipy")

    hcfg = cfg.analysis.hierarchy_analysis
    min_inst = int(hcfg.min_instances)
    linkage_method: str = str(hcfg.linkage_method)

    # ------------------------------------------------------------------
    # 1. Aggregate to mean embedding per sign code; restrict to 200–399
    # ------------------------------------------------------------------
    from collections import defaultdict
    bucket: dict[int, list[np.ndarray]] = defaultdict(list)
    for vec, code in zip(embeddings, codes):
        if not code:
            continue
        digits = "".join(c for c in code if c.isdigit())
        if not digits:
            continue
        n = int(digits)
        if 200 <= n <= 399:
            bucket[n].append(vec)

    eligible = {n: vecs for n, vecs in bucket.items() if len(vecs) >= min_inst}
    if len(eligible) < 3:
        log.warning(
            "Hierarchy analysis: only %d eligible sign codes in 200–399 "
            "(need ≥ 3, min_instances=%d). Skipping.",
            len(eligible), min_inst,
        )
        return {"status": "skipped", "reason": "insufficient_codes",
                "n_eligible": len(eligible)}

    codes_int = sorted(eligible)
    mean_vecs = np.stack([np.mean(eligible[n], axis=0) for n in codes_int])
    log.info(
        "Hierarchy analysis: %d sign codes in 200–399 "
        "(median %.0f instances/code, dim=%d).",
        len(codes_int),
        float(np.median([len(eligible[n]) for n in codes_int])),
        mean_vecs.shape[1],
    )

    # ------------------------------------------------------------------
    # 2. Build Barthel condensed distance and linkage
    # ------------------------------------------------------------------
    d_barthel = _build_barthel_condensed(codes_int)
    # 'average' linkage recovers an ultrametric exactly; use it for Barthel.
    Z_barthel = linkage(d_barthel, method="average")
    c_barthel_self, _ = cophenet(Z_barthel, d_barthel)
    log.debug("Barthel linkage self-cophenetic: %.4f (expect ≈1.0)", c_barthel_self)

    # ------------------------------------------------------------------
    # 3. Build embedding condensed distance and linkage
    # ------------------------------------------------------------------
    # Ward requires Euclidean; use cosine + average for non-Euclidean space,
    # or fall back to ward on L2-normalised vectors.
    if linkage_method == "ward":
        norms = np.linalg.norm(mean_vecs, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        normed = mean_vecs / norms
        d_embed = pdist(normed, metric="euclidean")
        Z_embed = linkage(d_embed, method="ward")
        embed_metric_label = "cosine (L2-normalised → euclidean for ward)"
    else:
        d_embed = pdist(mean_vecs, metric="cosine")
        Z_embed = linkage(d_embed, method=linkage_method)
        embed_metric_label = f"cosine + {linkage_method}"

    c_embed_self, _ = cophenet(Z_embed, d_embed)
    log.info("Embedding linkage self-cophenetic: %.4f", c_embed_self)

    # ------------------------------------------------------------------
    # 4. Cophenetic correlation of embedding hierarchy against Barthel tree
    # ------------------------------------------------------------------
    c_cross, _ = cophenet(Z_embed, d_barthel)
    log.info(
        "Cophenetic correlation (embedding hierarchy vs Barthel tree): %.4f",
        c_cross,
    )

    # Direct Pearson between raw pairwise distances (no hierarchy):
    pearson_direct = float(np.corrcoef(d_embed, d_barthel)[0, 1])
    log.info("Direct Pearson(d_embed, d_barthel): %.4f", pearson_direct)

    # ------------------------------------------------------------------
    # 5. Group-level breakdown — how many codes per group
    # ------------------------------------------------------------------
    from collections import Counter
    group_counts: dict[str, int] = Counter(
        f"{'2' if n < 300 else '3'}XX / {n // 10}X" for n in codes_int
    )

    result = {
        "status": "ok",
        "n_codes": len(codes_int),
        "linkage_method": linkage_method,
        "embedding_metric": embed_metric_label,
        "barthel_linkage_self_cophenetic": round(float(c_barthel_self), 6),
        "embedding_linkage_self_cophenetic": round(float(c_embed_self), 6),
        "cophenetic_cross_embed_vs_barthel": round(float(c_cross), 6),
        "direct_pearson_distances": round(float(pearson_direct), 6),
        "codes_analyzed": codes_int,
        "group_counts": dict(sorted(group_counts.items())),
    }

    json_path = out_dir / Path(hcfg.result_json).name
    json_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    log.info("Hierarchy analysis written: %s", json_path)

    _write_dendrogram_plot(
        Z_barthel, Z_embed, codes_int,
        out_dir / Path(hcfg.dendrogram_plot).name,
        c_cross,
    )

    return result


def _write_dendrogram_plot(
    Z_barthel: np.ndarray,
    Z_embed: np.ndarray,
    codes_int: list[int],
    out_path: Path,
    cophenetic_corr: float,
) -> None:
    """Write a side-by-side dendrogram comparison PNG."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from scipy.cluster.hierarchy import dendrogram
    except ImportError:
        log.warning("matplotlib not available — skipping dendrogram plot.")
        return

    labels = [str(n) for n in codes_int]
    fig, axes = plt.subplots(1, 2, figsize=(max(16, len(codes_int) // 2), 7))

    for ax, Z, title in (
        (axes[0], Z_barthel, "Barthel implicit tree\n(century → tens → units)"),
        (axes[1], Z_embed,
         f"Embedding dendrogram\n(cophenetic r = {cophenetic_corr:.3f})"),
    ):
        dendrogram(
            Z, labels=labels, ax=ax,
            leaf_rotation=90, leaf_font_size=6,
            color_threshold=0.7 * max(Z[:, 2]),
        )
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Barthel code")
        ax.set_ylabel("Distance")

    plt.tight_layout()
    plt.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Dendrogram plot written: %s", out_path)


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def run_analysis(cfg: DictConfig) -> None:
    try:
        import umap as umap_module
    except ImportError:
        raise ImportError(
            "umap-learn is required: pip install umap-learn"
        )
    try:
        from sklearn.cluster import HDBSCAN
    except ImportError:
        raise ImportError(
            "scikit-learn >=1.3 is required for HDBSCAN: pip install scikit-learn"
        )

    cache_path = Path(cfg.paths.embeddings_cache)
    if not cache_path.exists():
        raise FileNotFoundError(
            f"Embeddings cache not found at {cache_path}. "
            "Run Zone A autoencoder training first."
        )

    log.info("Loading embeddings from %s", cache_path)
    embeddings, codes = _load_embeddings(cache_path)
    log.info("  %d embeddings, dim=%d", len(embeddings), embeddings.shape[1])

    # --- UMAP projection ------------------------------------------------
    ucfg = cfg.analysis.umap
    log.info("Projecting to %dD with UMAP (metric=%s, n_neighbors=%d) …",
             ucfg.n_components, ucfg.metric, ucfg.n_neighbors)
    reducer = umap_module.UMAP(
        n_components=ucfg.n_components,
        metric=ucfg.metric,
        n_neighbors=ucfg.n_neighbors,
        min_dist=ucfg.min_dist,
        random_state=ucfg.random_state,
        init=getattr(ucfg, "init", "pca"),
    )
    embedding_2d = reducer.fit_transform(embeddings)
    log.info("  UMAP done. Shape: %s", embedding_2d.shape)

    # --- HDBSCAN clustering ---------------------------------------------
    _backbone = str(cfg.zone_a.get("backbone", "custom")).lower()
    hcfg = cfg.analysis.hdbscan_dinov2 if _backbone == "dinov2" else cfg.analysis.hdbscan
    log.info("Clustering with HDBSCAN (min_cluster_size=%d, min_samples=%d) …",
             hcfg.min_cluster_size, hcfg.min_samples)
    clusterer = HDBSCAN(
        min_cluster_size=hcfg.min_cluster_size,
        min_samples=hcfg.min_samples,
        metric=hcfg.metric,
        cluster_selection_method=hcfg.cluster_selection_method,
    )
    cluster_labels = clusterer.fit_predict(embedding_2d)
    n_clusters = len(set(cluster_labels) - {-1})
    n_noise = int((cluster_labels == -1).sum())
    log.info("  %d clusters found, %d points marked as noise", n_clusters, n_noise)

    # --- Outputs --------------------------------------------------------
    out_dir = Path(cfg.paths.outputs_dir) / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    tcfg = cfg.analysis.taxonomy_comparison
    min_inst = tcfg.min_instances_for_evaluation

    from collections import Counter
    from sklearn.metrics import (
        adjusted_rand_score,
        normalized_mutual_info_score,
        homogeneity_completeness_v_measure,
    )

    # --- Coverage stats -------------------------------------------------
    code_counts = Counter(c for c in codes if c)
    n_labeled = sum(1 for c in codes if c)
    eligible_mask = np.array([
        bool(c) and code_counts[c] >= min_inst
        for c in codes
    ])
    n_eligible = int(eligible_mask.sum())
    label_coverage_rate = n_labeled / max(len(codes), 1)
    eligible_coverage_rate = n_eligible / max(len(codes), 1)
    log.info(
        "  Coverage: %d/%d labeled (%.1f%%), %d eligible for metrics (%.1f%%)",
        n_labeled, len(codes), label_coverage_rate * 100,
        n_eligible, eligible_coverage_rate * 100,
    )

    # --- Cluster-quality metrics helper ---------------------------------
    def _compute_cluster_metrics(
        true_labels: list, pred_labels: np.ndarray
    ) -> dict[str, float | None]:
        """Return ARI, NMI, homogeneity, completeness, V-measure."""
        if len(true_labels) < 2:
            return {k: None for k in (
                "ari", "nmi", "homogeneity", "completeness", "v_measure"
            )}
        hom, com, vme = homogeneity_completeness_v_measure(true_labels, pred_labels)
        return {
            "ari": float(adjusted_rand_score(true_labels, pred_labels)),
            "nmi": float(normalized_mutual_info_score(
                true_labels, pred_labels, average_method="arithmetic"
            )),
            "homogeneity": float(hom),
            "completeness": float(com),
            "v_measure": float(vme),
        }

    # --- Compute metrics at family level and code level -----------------
    agreement: str
    family_metrics: dict[str, float | None]
    code_metrics: dict[str, float | None]
    if n_eligible >= 2:
        eligible_codes = [c for c, m in zip(codes, eligible_mask) if m]
        families_true = [_barthel_family(c) for c in eligible_codes]
        clusters_subset = cluster_labels[eligible_mask]
        family_metrics = _compute_cluster_metrics(families_true, clusters_subset)
        code_metrics = _compute_cluster_metrics(eligible_codes, clusters_subset)
        ari = family_metrics["ari"]
        log.info(
            "  Family-level — ARI: %.4f  NMI: %.4f  V-measure: %.4f",
            ari, family_metrics["nmi"], family_metrics["v_measure"],
        )
        log.info(
            "  Code-level  — ARI: %.4f  NMI: %.4f  V-measure: %.4f",
            code_metrics["ari"], code_metrics["nmi"], code_metrics["v_measure"],
        )
        agreement = "consistent" if ari >= tcfg.ari_agreement_threshold else "divergent"
        log.info(
            "  Interpretation: %s with Barthel taxonomy (threshold=%.2f)",
            agreement, tcfg.ari_agreement_threshold,
        )
    else:
        ari = None
        _null_metrics: dict[str, float | None] = {
            k: None for k in ("ari", "nmi", "homogeneity", "completeness", "v_measure")
        }
        family_metrics = dict(_null_metrics)
        code_metrics = dict(_null_metrics)
        agreement = "insufficient_data"
        log.warning(
            "  Too few eligible labeled instances for metric computation (%d).",
            n_eligible,
        )

    # --- Per-cluster Barthel breakdown ----------------------------------
    cluster_breakdown: dict[str, dict] = {}
    for cluster_id in sorted(set(cluster_labels)):
        mask = cluster_labels == cluster_id
        cluster_codes = [codes[i] for i in range(len(codes)) if mask[i]]
        cluster_families = [_barthel_family(c) for c in cluster_codes]
        breakdown_codes = Counter(c for c in cluster_codes if c)
        breakdown_families = Counter(cluster_families)
        n_coded = sum(breakdown_codes.values())
        # Purity: fraction of coded glyphs dominated by the top code/family
        top_code_item = breakdown_codes.most_common(1)
        top_family_item = breakdown_families.most_common(1)
        top_code = top_code_item[0][0] if top_code_item else None
        top_code_count = top_code_item[0][1] if top_code_item else 0
        top_family = top_family_item[0][0] if top_family_item else None
        top_family_count = top_family_item[0][1] if top_family_item else 0
        cluster_size = int(mask.sum())
        label = "noise" if cluster_id == -1 else str(cluster_id)
        cluster_breakdown[label] = {
            "size": cluster_size,
            "n_coded": n_coded,
            "dominant_code": top_code,
            "dominant_code_fraction": (
                round(top_code_count / n_coded, 4) if n_coded else None
            ),
            "dominant_family": top_family,
            "dominant_family_fraction": (
                round(top_family_count / cluster_size, 4) if cluster_size else None
            ),
            "barthel_codes": dict(breakdown_codes.most_common(20)),
            "barthel_families": dict(breakdown_families),
        }

    # Sanity check: flag if fewer than 40% of non-noise clusters are single-family pure.
    # A drop below this threshold indicates corrupted embeddings or a mis-configured
    # min_cluster_size that is merging unrelated sign families.
    real_clusters = {
        lbl: info for lbl, info in cluster_breakdown.items() if lbl != "noise"
    }
    if real_clusters:
        pure_count = sum(
            1 for info in real_clusters.values()
            if len(info["barthel_families"]) == 1
        )
        purity_rate = pure_count / len(real_clusters)
        if purity_rate < 0.40:
            log.warning(
                "Zone A sanity check FAILED: only %.1f%% of clusters are single-family pure "
                "(%d/%d). Check DINOv2 preprocessing pipeline — embeddings may be corrupted. "
                "Also verify hdbscan.min_cluster_size in config.yaml matches previous runs.",
                purity_rate * 100, pure_count, len(real_clusters),
            )
        else:
            log.info(
                "Zone A sanity check OK: %.1f%% single-family pure clusters (%d/%d).",
                purity_rate * 100, pure_count, len(real_clusters),
            )
        # Negative ARI combined with a low purity rate is a strong signal that
        # the embedding space has collapsed or been corrupted.
        if ari is not None and ari < 0 and purity_rate < 0.40:
            log.warning(
                "Zone A double-failure: ARI=%.4f (negative) AND purity=%.1f%% (<40%%). "
                "Do NOT proceed to Zone B/C — results will be meaningless.",
                ari, purity_rate * 100,
            )

    comparison_result = {
        "n_embeddings": len(embeddings),
        "n_clusters": n_clusters,
        "n_noise_points": n_noise,
        # Coverage — how many glyphs actually contributed to metric computation
        "n_labeled": n_labeled,
        "n_eligible_for_metrics": n_eligible,
        "label_coverage_rate": round(label_coverage_rate, 4),
        "eligible_coverage_rate": round(eligible_coverage_rate, 4),
        # Family-level metrics (Barthel century-block groupings, 8 classes)
        "adjusted_rand_index": ari,
        "ari_threshold": float(tcfg.ari_agreement_threshold),
        "interpretation": agreement,
        "family_metrics": family_metrics,
        # Code-level metrics (individual Barthel sign codes, finer-grained)
        "code_metrics": code_metrics,
        "clusters": cluster_breakdown,
    }
    json_path = out_dir / Path(tcfg.cluster_comparison_json).name
    json_path.write_text(
        json.dumps(comparison_result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    log.info("Cluster comparison written: %s", json_path)

    # Build temporal_cluster mapping from corpus + config
    corpus_dir = Path(cfg.paths.corpus_dir)
    code_to_tablet: dict[str, str] = {}
    for corpus_file in sorted(corpus_dir.glob("[A-Z].json")):
        tablet_id = corpus_file.stem
        try:
            glyphs = json.loads(corpus_file.read_text())
            for g in (glyphs if isinstance(glyphs, list) else glyphs.get("glyphs", [])):
                bc = g.get("barthel_code")
                if bc and bc not in code_to_tablet:
                    code_to_tablet[str(bc)] = tablet_id
        except Exception:
            pass

    # Map tablet_id to temporal cluster based on config
    temporal_config = cfg.corpus.temporal_model
    clusters_cfg = temporal_config.get("clusters", {})
    pre_contact_tablets = set(clusters_cfg.get("pre_contact", {}).get("tablets", []))
    post_contact_tablets = set(clusters_cfg.get("post_contact", {}).get("tablets", []))
    excluded_tablets = set(clusters_cfg.get("excluded_from_temporal_analysis", {}).get("tablets", []))

    def _get_temporal_cluster(code: str | None) -> str:
        if not code:
            return "unknown"
        tablet = code_to_tablet.get(str(code))
        if not tablet:
            return "unknown"
        if tablet in pre_contact_tablets:
            return "pre_contact"
        if tablet in post_contact_tablets:
            return "post_contact"
        if tablet in excluded_tablets:
            return "excluded"
        return "unknown"

    # CSV
    import csv
    csv_path = out_dir / Path(tcfg.cluster_comparison_csv).name
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["index", "barthel_code", "barthel_family",
                         "umap_x", "umap_y", "hdbscan_cluster", "temporal_cluster"])
        for i, (code, xy, cl) in enumerate(zip(codes, embedding_2d, cluster_labels)):
            writer.writerow([
                i,
                code or "",
                _barthel_family(code),
                f"{xy[0]:.6f}",
                f"{xy[1]:.6f}",
                int(cl),
                _get_temporal_cluster(code),
            ])
    log.info("CSV written: %s", csv_path)

    # Generate divergence report (HTML artifact for scholarly outreach)
    try:
        from hackingrongo.results.divergence_report import (
            DivergenceReportConfig,
            save_divergence_report,
        )
        svg_catalog = Path(hydra.utils.get_original_cwd()) / "data/glyphs/svg/catalog.json"
        try:
            run_id = cfg.mlflow.get("run_id", "—")
            experiment_name = cfg.mlflow.experiment_name
        except Exception:
            run_id = "—"
            experiment_name = "—"
        save_divergence_report(
            analysis_dir=out_dir,
            svg_catalog_path=svg_catalog,
            output_path=out_dir / "divergence_report.html",
            run_metadata={
                "run_id": run_id,
                "experiment": experiment_name,
                "corpus": f"{len(codes)} glyphs · 25 tablets",
            },
        )
    except Exception as exc:
        log.warning("Divergence report generation failed (non-fatal): %s", exc)

    # Compound detection
    try:
        import pandas as pd
        from hackingrongo.zone_b.compound_detector import (
            CompoundDetector,
            save_compound_candidates,
        )
        from hackingrongo.results.compound_report import save_compound_report

        corpus_dir = Path(cfg.paths.corpus_dir)
        svg_catalog = Path(hydra.utils.get_original_cwd()) / "data/glyphs/svg/catalog.json"
        umap_df = pd.DataFrame({
            "barthel_code":   [c or "" for c in codes],
            "barthel_family": [_barthel_family(c) for c in codes],
            "umap_x":         embedding_2d[:, 0],
            "umap_y":         embedding_2d[:, 1],
            "hdbscan_cluster": cluster_labels.tolist(),
        })

        candidates = CompoundDetector(corpus_dir).detect(umap_df)
        candidates_path = out_dir / "compound_candidates.json"
        save_compound_candidates(candidates, candidates_path)

        validation = CompoundDetector(corpus_dir, exclude_known=False).detect(umap_df)
        save_compound_candidates(
            validation,
            out_dir / "compound_validation.json",
            min_methods=1,
        )

        save_compound_report(
            candidates_path=candidates_path,
            svg_catalog_path=svg_catalog,
            output_path=out_dir / "compound_report.html",
            corpus_dir=corpus_dir,
        )
        log.info("Compound detection: %d candidates.", len(candidates))
    except Exception as exc:
        log.warning("Compound detection failed (non-fatal): %s", exc)

    # Passage reports
    try:
        from hackingrongo.results.passage_report import PassageReportGenerator
        import json as _json

        passages_json = (
            Path(hydra.utils.get_original_cwd()) / cfg.paths.parallel_variants_json
        )
        # Fall back to auto-generated file when config file is missing or empty
        _auto = passages_json.parent / "parallel_variants_auto.json"
        if passages_json.exists():
            _raw = _json.loads(passages_json.read_text(encoding="utf-8"))
            if not _raw.get("passages") and _auto.exists():
                log.info(
                    "parallel_variants.json has no passages — using %s instead.", _auto.name
                )
                passages_json = _auto
        elif _auto.exists():
            log.info("parallel_variants.json not found — using %s.", _auto.name)
            passages_json = _auto

        if passages_json.exists():
            passage_dir = out_dir / "passage_reports"
            PassageReportGenerator().generate_report(passages_json, passage_dir)
            log.info("Passage reports written → %s", passage_dir)
        else:
            log.info("Passage report skipped — no parallel_variants JSON found.")
    except Exception as exc:
        log.warning("Passage report generation failed (non-fatal): %s", exc)

    # UMAP scatter plot
    _write_umap_plot(
        embedding_2d, codes, cluster_labels,
        out_dir / Path(tcfg.umap_plot).name,
    )

    # --- Hierarchy analysis: embedding dendrogram vs Barthel tree (200–399) ---
    run_hierarchy_analysis(embeddings, codes, cfg, out_dir)

    # --- Embedding space report (scatter plots + explanations) ---
    try:
        from hackingrongo.results.embedding_report import save_embedding_report
        save_embedding_report(
            analysis_dir=out_dir,
            output_path=out_dir / "embedding_report.html",
            run_metadata={
                "corpus": f"{len(codes)} glyphs · 25 tablets",
                "clusters": str(n_clusters),
            },
        )
    except Exception as exc:
        log.warning("Embedding report generation failed (non-fatal): %s", exc)


def _write_umap_plot(
    embedding_2d: np.ndarray,
    codes: list[str | None],
    cluster_labels: np.ndarray,
    out_path: Path,
) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        log.warning("matplotlib not available — skipping UMAP plot.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    # Left panel: coloured by Barthel family
    families = [_barthel_family(c) for c in codes]
    unique_families = sorted(set(families))
    palette = plt.colormaps.get_cmap("tab10").resampled(max(len(unique_families), 1))
    family_color = {f: palette(i) for i, f in enumerate(unique_families)}
    colors = [family_color[f] for f in families]
    axes[0].scatter(embedding_2d[:, 0], embedding_2d[:, 1],
                    c=colors, s=6, alpha=0.6, linewidths=0)
    axes[0].set_title("UMAP — coloured by Barthel sign family")
    axes[0].set_xlabel("UMAP 1")
    axes[0].set_ylabel("UMAP 2")
    handles = [plt.Line2D([0], [0], marker="o", color="w",
                           markerfacecolor=family_color[f], markersize=6, label=f)
               for f in unique_families]
    axes[0].legend(handles=handles, fontsize=7, loc="best")

    # Right panel: coloured by HDBSCAN cluster (-1 = noise in grey)
    n_clusters = len(set(cluster_labels) - {-1})
    cmap_clusters = plt.colormaps.get_cmap("hsv").resampled(max(n_clusters, 1))
    cl_colors = [
        (0.6, 0.6, 0.6, 0.3) if cl == -1 else cmap_clusters(cl)
        for cl in cluster_labels
    ]
    axes[1].scatter(embedding_2d[:, 0], embedding_2d[:, 1],
                    c=cl_colors, s=6, alpha=0.7, linewidths=0)
    axes[1].set_title(f"UMAP — HDBSCAN clusters (k={n_clusters})")
    axes[1].set_xlabel("UMAP 1")
    axes[1].set_ylabel("UMAP 2")

    plt.tight_layout()
    plt.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("UMAP plot written: %s", out_path)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

@hydra.main(version_base="1.3", config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)s %(name)s: %(message)s")
    run_analysis(cfg)


if __name__ == "__main__":
    main()
