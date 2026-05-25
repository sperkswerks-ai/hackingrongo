"""
Dating priority ranking — assign_cluster_probability on undated tablets.

Runs the empirical-prior classifier on all 19 undated tablets and ranks them
by dating entropy H(pre, post) = -Σ p_i log2(p_i).

Tablets closest to H = 1.0 bit (maximum uncertainty) are the highest priority
for physical radiocarbon dating — each additional AMS date on a high-entropy
tablet yields the most information about the scribal tradition's chronology.

With ``--features``, a Gaussian Naïve Bayes classifier is trained on the five
dated tablets (D=pre, B/C/O/Q=post) using the specified per-tablet features.
Posteriors replace the flat empirical prior, re-ranking undated tablets by
feature-informed uncertainty.

The output table is suitable for inclusion in the CFP supporting materials.

Usage
-----
    conda run python scripts/dating_priority.py
    conda run python scripts/dating_priority.py --json
    conda run python scripts/dating_priority.py \
        --features resolved_token_count sign_diversity \
        --output outputs/dating_priority.json
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from omegaconf import OmegaConf  # noqa: E402

from hackingrongo.data.corpus import assign_cluster, assign_cluster_probability  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)


def dating_entropy(probs: dict[str, float]) -> float:
    """H(pre, post) in bits — ignores the 'unknown' bucket."""
    h = 0.0
    for key in ("pre_contact", "post_contact"):
        p = probs.get(key, 0.0)
        if p > 0:
            h -= p * math.log2(p)
    return h


def load_tablet_glyph_counts(corpus_dir: Path) -> dict[str, dict]:
    """Return {tablet_id: {total, resolved, resolved_token_count, sign_diversity, cluster}} from enriched JSON."""
    counts: dict[str, dict] = {}
    for path in sorted(corpus_dir.glob("[A-Z].json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        tid = data["tablet_id"]
        glyphs = data["glyphs"]
        total = len(glyphs)
        resolved_list = [g["horley_code"] for g in glyphs if g.get("horley_code")]
        resolved = len(resolved_list)
        diversity = len(set(resolved_list))
        counts[tid] = {
            "total": total,
            "resolved": resolved,
            "resolved_token_count": resolved,
            "sign_diversity": diversity,
            "cluster": data.get("cluster", "unknown"),
        }
    return counts


# ---------------------------------------------------------------------------
# Gaussian Naïve Bayes (2-class)
# ---------------------------------------------------------------------------

_VALID_FEATURES = ("resolved_token_count", "sign_diversity")


def _gnb_train(
    feature_matrix: list[list[float]],
    labels: list[int],  # 0 = pre_contact, 1 = post_contact
    var_floor: float = 1.0,
) -> dict:
    """Train a Gaussian NB classifier; return model dict.

    Parameters
    ----------
    feature_matrix : list of [f1, f2, ...] rows (one per training tablet)
    labels         : 0 = pre_contact, 1 = post_contact
    var_floor      : minimum per-feature variance (regularisation)

    Notes
    -----
    With only 1 pre_contact training example the within-class variance is
    undefined.  We substitute the pooled variance of the full training set,
    floored at *var_floor*, to avoid a degenerate Gaussian.
    """
    n_feat = len(feature_matrix[0])
    classes = (0, 1)
    model: dict = {"classes": classes, "n_features": n_feat, "mean": {}, "var": {}, "log_prior": {}}

    for cls in classes:
        idx = [i for i, y in enumerate(labels) if y == cls]
        n_cls = len(idx)
        prior = n_cls / len(labels)
        model["log_prior"][cls] = math.log(prior) if prior > 0 else -1e9

        means = [sum(feature_matrix[i][f] for i in idx) / n_cls for f in range(n_feat)]
        model["mean"][cls] = means

        if n_cls == 1:
            # Pooled variance from all training examples
            all_mean = [sum(feature_matrix[i][f] for i in range(len(labels))) / len(labels)
                        for f in range(n_feat)]
            pooled_var = [
                max(var_floor,
                    sum((feature_matrix[i][f] - all_mean[f]) ** 2 for i in range(len(labels)))
                    / len(labels))
                for f in range(n_feat)
            ]
            model["var"][cls] = pooled_var
        else:
            model["var"][cls] = [
                max(var_floor,
                    sum((feature_matrix[i][f] - means[f]) ** 2 for i in idx) / n_cls)
                for f in range(n_feat)
            ]

    return model


def _gnb_predict_proba(model: dict, x: list[float]) -> dict[int, float]:
    """Return {class: posterior_probability} for feature vector *x*."""
    log_posts: dict[int, float] = {}
    for cls in model["classes"]:
        lp = model["log_prior"][cls]
        for f, xi in enumerate(x):
            mu = model["mean"][cls][f]
            sigma2 = model["var"][cls][f]
            lp += -0.5 * math.log(2 * math.pi * sigma2) - (xi - mu) ** 2 / (2 * sigma2)
        log_posts[cls] = lp

    # Normalise in log-space (subtract max for numerical stability)
    max_lp = max(log_posts.values())
    exp_lp = {cls: math.exp(lp - max_lp) for cls, lp in log_posts.items()}
    total = sum(exp_lp.values())
    return {cls: v / total for cls, v in exp_lp.items()}


# ---------------------------------------------------------------------------
# Ranking
# ---------------------------------------------------------------------------


def rank(
    emit_json: bool = False,
    features: list[str] | None = None,
    output_path: Path | None = None,
) -> None:
    cfg = OmegaConf.load(PROJECT_ROOT / "conf" / "config.yaml")
    corpus_dir = PROJECT_ROOT / cfg.paths.corpus_dir

    glyph_counts = load_tablet_glyph_counts(corpus_dir)

    # Validate requested features
    if features:
        for f in features:
            if f not in _VALID_FEATURES:
                raise ValueError(f"Unknown feature {f!r}. Valid: {_VALID_FEATURES}")
        log.info("GNB features: %s", features)
    else:
        features = []

    # All 25 tablets
    all_tablets = sorted(glyph_counts.keys())

    # ------------------------------------------------------------------
    # Optional: train GNB on dated tablets
    # ------------------------------------------------------------------
    gnb_model = None
    class_label = {"pre_contact": 0, "post_contact": 1}

    if features:
        train_X: list[list[float]] = []
        train_y: list[int] = []
        for tid in all_tablets:
            cluster = assign_cluster(tid, cfg)
            if cluster not in class_label:
                continue
            row = [float(glyph_counts[tid][f]) for f in features]
            train_X.append(row)
            train_y.append(class_label[cluster])

        gnb_model = _gnb_train(train_X, train_y)
        class_counts = {cls: train_y.count(cls) for cls in (0, 1)}
        log.info(
            "GNB trained: %d pre_contact + %d post_contact tablets  features=%s",
            class_counts[0], class_counts[1], features,
        )
        for cls_name, cls_idx in class_label.items():
            means_str = ", ".join(f"{f}={gnb_model['mean'][cls_idx][i]:.1f}" for i, f in enumerate(features))
            log.info("  %s mean: %s", cls_name, means_str)

    records: list[dict] = []
    for tid in all_tablets:
        cluster = assign_cluster(tid, cfg)
        gc = glyph_counts[tid]

        if gnb_model is not None and cluster == "unknown":
            x = [float(gc[f]) for f in features]
            post = _gnb_predict_proba(gnb_model, x)
            probs = {
                "pre_contact": round(post[0], 4),
                "post_contact": round(post[1], 4),
                "unknown": 0.0,
            }
        else:
            probs = assign_cluster_probability(tid, cfg)

        h = dating_entropy(probs)
        rec = {
            "tablet": tid,
            "cluster": cluster,
            "p_pre_contact": round(probs.get("pre_contact", 0.0), 4),
            "p_post_contact": round(probs.get("post_contact", 0.0), 4),
            "entropy_bits": round(h, 4),
            "total_glyphs": gc["total"],
            "horley_resolved": gc["resolved"],
            "resolution_rate": round(gc["resolved"] / gc["total"], 4) if gc["total"] else 0.0,
        }
        if gnb_model is not None:
            rec["resolved_token_count"] = gc["resolved_token_count"]
            rec["sign_diversity"] = gc["sign_diversity"]
        records.append(rec)

    # Sort: undated first by entropy desc, then dated/excluded
    undated = sorted(
        [r for r in records if r["cluster"] == "unknown"],
        key=lambda r: -r["entropy_bits"],
    )
    dated = [r for r in records if r["cluster"] != "unknown"]

    log.info("")
    log.info("=" * 72)
    log.info("Dating Priority Ranking — undated tablets by entropy H(pre, post)")
    log.info("=" * 72)
    log.info("  Prior:  1 pre_contact anchor (D), 4 post_contact anchors (B,C,O,Q)")
    log.info("  Empirical prior: p_pre=0.200  p_post=0.800")
    log.info("  NOTE: All undated tablets share the same prior (no feature vectors).")
    log.info("        Entropy is uniform at %.4f bits until Zone A embeddings", dating_entropy({"pre_contact": 0.2, "post_contact": 0.8}))
    log.info("        are available. Ranking here is by glyph count (proxy for")
    log.info("        information yield per AMS date).")
    log.info("")
    log.info("  Rank  Tablet  cluster   p_pre  p_post  H(bits)  glyphs  resolved  res%%")
    log.info("  " + "-" * 68)
    for i, r in enumerate(undated, 1):
        log.info(
            "  %4d    %-3s   %-9s  %.3f   %.3f  %.4f   %5d     %4d   %4.0f%%",
            i, r["tablet"], r["cluster"],
            r["p_pre_contact"], r["p_post_contact"],
            r["entropy_bits"],
            r["total_glyphs"], r["horley_resolved"],
            100.0 * r["resolution_rate"],
        )

    log.info("")
    log.info("Dated / excluded tablets (for reference):")
    log.info("  %-3s  %-14s  p_pre  p_post  glyphs  resolved", "Tab", "cluster")
    for r in sorted(dated, key=lambda r: r["tablet"]):
        log.info("   %-3s  %-14s  %.3f   %.3f   %5d     %4d",
                 r["tablet"], r["cluster"],
                 r["p_pre_contact"], r["p_post_contact"],
                 r["total_glyphs"], r["horley_resolved"])

    log.info("")
    log.info("Highest-priority undated tablets for AMS dating (by glyph count):")
    by_glyphs = sorted(undated, key=lambda r: -r["total_glyphs"])
    for r in by_glyphs[:5]:
        log.info(
            "  %-3s  %5d glyphs  %4d resolved (%.0f%%)",
            r["tablet"], r["total_glyphs"], r["horley_resolved"],
            100.0 * r["resolution_rate"],
        )

    if emit_json:
        print(json.dumps({"undated": undated, "dated": dated}, indent=2))

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_data = {
            "undated": undated,
            "dated": dated,
            "features_used": features if gnb_model else [],
            "classifier": "gnb" if gnb_model else "empirical_prior",
        }
        if gnb_model:
            output_data["gnb_model"] = {
                "features": features,
                "classes": ["pre_contact", "post_contact"],
                "mean_pre": gnb_model["mean"][0],
                "mean_post": gnb_model["mean"][1],
                "var_pre": gnb_model["var"][0],
                "var_post": gnb_model["var"][1],
            }
        output_path.write_text(json.dumps(output_data, indent=2), encoding="utf-8")
        log.info("Dating priority written to %s", output_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Dating priority ranking for undated rongorongo tablets.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON output.",
    )
    parser.add_argument(
        "--features",
        nargs="+",
        metavar="FEAT",
        choices=list(_VALID_FEATURES),
        default=None,
        help=(
            "Per-tablet features to use for Gaussian Naïve Bayes ranking. "
            f"Valid: {', '.join(_VALID_FEATURES)}"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Write JSON results to this path.",
    )
    args = parser.parse_args()
    rank(emit_json=args.json, features=args.features, output_path=args.output)


if __name__ == "__main__":
    main()
