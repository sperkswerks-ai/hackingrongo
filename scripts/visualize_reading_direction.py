"""
scripts/visualize_reading_direction.py

Two visualisations of the per-tablet reading-direction preference from
Test 6 (leave-one-tablet-out perplexity, reading_order_v2.py).

Plot 1 — Direction preference chart
    One horizontal bar per tablet, sorted by margin descending.
    Green bars = a→b preferred, orange bars = b→a preferred.
    Bar length encodes margin magnitude.
    Includes strata annotations (pre-contact D is highlighted).

Plot 2 — Token count vs direction preference
    Scatter: x = tablet token count, y = margin (direction-signed).
    Point color and shape encode the preferred direction.
    Useful for diagnosing whether tablet size confounds the finding.

Output
------
outputs/analysis/reading_direction_bars.png
outputs/analysis/reading_direction_scatter.png
outputs/analysis/reading_direction_combined.png  (both panels)

Usage
-----
    python scripts/visualize_reading_direction.py
    python scripts/visualize_reading_direction.py --v2-json outputs/analysis/reading_order_v2.json
    python scripts/visualize_reading_direction.py --smoke-test
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

# Known strata for annotation
_PRECONTACT   = frozenset({"D"})
_POSTCONTACT  = frozenset({"B", "C", "O", "Q"})
_EXCLUDED     = frozenset({"A"})


def _stratum_label(tablet: str) -> str:
    if tablet in _PRECONTACT:
        return "pre-contact"
    if tablet in _POSTCONTACT:
        return "post-contact"
    if tablet in _EXCLUDED:
        return "excluded"
    return "undated"


def _token_counts(corpus_dir: Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    for path in corpus_dir.glob("*.json"):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            counts[path.stem] = len(data.get("glyphs", []))
        except Exception:
            pass
    return counts


def make_plots(
    per_tablet: list[dict],
    token_counts: dict[str, int],
    output_dir: Path,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import numpy as np

    # ── Colours ──────────────────────────────────────────────────────────
    BG      = "#0d0f12"
    SURFACE = "#1e2229"
    TEXT    = "#d0d4dc"
    MUTED   = "#6b7280"
    ACCENT  = "#c4a96d"
    AB_COL  = "#4ade80"   # green for a→b
    BA_COL  = "#f97316"   # orange for b→a
    PRE_EDGE = "#fbbf24"  # gold border for pre-contact

    plt.rcParams.update({
        "figure.facecolor": BG, "axes.facecolor": BG, "axes.edgecolor": SURFACE,
        "text.color": TEXT, "axes.labelcolor": TEXT, "xtick.color": MUTED,
        "ytick.color": TEXT, "grid.color": SURFACE, "font.family": "monospace",
        "font.size": 9,
    })

    # Sort by margin descending (largest first) for bar chart
    sorted_data = sorted(per_tablet, key=lambda r: r["margin"], reverse=True)
    tablets = [r["tablet"] for r in sorted_data]
    margins = [r["margin"] for r in sorted_data]
    colors  = [AB_COL if r["winner"] == "ab" else BA_COL for r in sorted_data]
    strata  = [_stratum_label(r["tablet"]) for r in sorted_data]

    # ── Plot 1: Horizontal bars ───────────────────────────────────────────
    fig1, ax1 = plt.subplots(figsize=(9, max(5, len(tablets) * 0.35)))
    fig1.patch.set_facecolor(BG)

    y_pos = range(len(tablets))
    bars = ax1.barh(list(y_pos), margins, color=colors, height=0.65, zorder=2)

    # Highlight pre-contact and post-contact
    for i, (tablet, stratum) in enumerate(zip(tablets, strata)):
        if stratum == "pre-contact":
            ax1.barh(i, margins[i], color=colors[i], height=0.65,
                     edgecolor=PRE_EDGE, linewidth=1.5, zorder=3)
            ax1.text(margins[i] + 0.01, i, "★ Tablet D (pre-contact)",
                     va="center", fontsize=7.5, color=PRE_EDGE)
        elif stratum == "post-contact":
            ax1.text(-0.005, i, "●", ha="right", va="center",
                     fontsize=8, color=ACCENT)

    ax1.set_yticks(list(y_pos))
    ax1.set_yticklabels(tablets, fontsize=9)
    ax1.set_xlabel("Perplexity margin |PPL(a→b) − PPL(b→a)|", color=MUTED)
    ax1.set_title("Per-tablet reading direction preference (Test 6 LTOO)",
                  color=ACCENT, pad=12, fontsize=11)
    ax1.axvline(0, color=MUTED, linewidth=0.5, zorder=1)
    ax1.grid(axis="x", alpha=0.3, zorder=0)
    ax1.invert_yaxis()

    leg_patches = [
        mpatches.Patch(color=AB_COL, label="Prefers a→b (recto first)"),
        mpatches.Patch(color=BA_COL, label="Prefers b→a (verso first)"),
    ]
    ax1.legend(handles=leg_patches, loc="lower right", facecolor=SURFACE,
               edgecolor=MUTED, labelcolor=TEXT, fontsize=8)

    n_ab = sum(1 for r in per_tablet if r["winner"] == "ab")
    n_ba = len(per_tablet) - n_ab
    ax1.text(
        0.98, 0.01,
        f"a→b: {n_ab} tablets   b→a: {n_ba} tablets",
        transform=ax1.transAxes, ha="right", va="bottom",
        fontsize=8, color=MUTED,
    )

    plt.tight_layout(pad=1.4)
    out1 = output_dir / "reading_direction_bars.png"
    fig1.savefig(out1, dpi=150, bbox_inches="tight")
    plt.close(fig1)
    log.info("Bars chart → %s", out1)

    # ── Plot 2: Token count vs direction ──────────────────────────────────
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    fig2.patch.set_facecolor(BG)

    for r in per_tablet:
        tcount = token_counts.get(r["tablet"], 0)
        signed_margin = r["margin"] if r["winner"] == "ab" else -r["margin"]
        col = AB_COL if r["winner"] == "ab" else BA_COL
        stratum = _stratum_label(r["tablet"])
        marker = "*" if stratum == "pre-contact" else ("s" if stratum == "post-contact" else "o")
        ms = 120 if stratum == "pre-contact" else 60
        ax2.scatter(tcount, signed_margin, color=col, marker=marker, s=ms,
                    zorder=3, edgecolors=PRE_EDGE if stratum == "pre-contact" else col,
                    linewidths=1.5 if stratum == "pre-contact" else 0.5)
        ax2.annotate(r["tablet"], (tcount, signed_margin),
                     textcoords="offset points", xytext=(5, 3),
                     fontsize=7, color=MUTED)

    ax2.axhline(0, color=MUTED, linewidth=0.8, zorder=1, linestyle="--")
    ax2.fill_between(ax2.get_xlim(), 0, ax2.get_ylim()[1] if ax2.get_ylim()[1] > 0 else 1,
                     alpha=0.04, color=AB_COL, zorder=0)
    ax2.fill_between(ax2.get_xlim(), ax2.get_ylim()[0] if ax2.get_ylim()[0] < 0 else -1, 0,
                     alpha=0.04, color=BA_COL, zorder=0)

    ax2.set_xlabel("Tablet token count", color=MUTED)
    ax2.set_ylabel("Signed margin (+ = a→b, − = b→a)", color=MUTED)
    ax2.set_title("Token count vs reading direction preference",
                  color=ACCENT, pad=12, fontsize=11)
    ax2.grid(alpha=0.2, zorder=0)
    ax2.text(0.02, 0.97, "↑ prefers a→b", transform=ax2.transAxes,
             va="top", color=AB_COL, fontsize=8)
    ax2.text(0.02, 0.03, "↓ prefers b→a", transform=ax2.transAxes,
             va="bottom", color=BA_COL, fontsize=8)

    # Fit line to check size correlation
    try:
        xs = np.array([token_counts.get(r["tablet"], 0) for r in per_tablet], dtype=float)
        ys = np.array([r["margin"] if r["winner"] == "ab" else -r["margin"]
                       for r in per_tablet], dtype=float)
        if xs.std() > 0:
            m_fit = np.polyfit(xs, ys, 1)
            x_line = np.linspace(xs.min(), xs.max(), 100)
            ax2.plot(x_line, np.polyval(m_fit, x_line), color=ACCENT,
                     linewidth=1, linestyle=":", alpha=0.6, zorder=2)
    except Exception:
        pass

    plt.tight_layout(pad=1.4)
    out2 = output_dir / "reading_direction_scatter.png"
    fig2.savefig(out2, dpi=150, bbox_inches="tight")
    plt.close(fig2)
    log.info("Scatter chart → %s", out2)

    # ── Combined figure ───────────────────────────────────────────────────
    fig3, (a3l, a3r) = plt.subplots(1, 2, figsize=(16, max(5, len(tablets) * 0.32)))
    fig3.patch.set_facecolor(BG)
    fig3.suptitle("Rongorongo reading direction — per-tablet analysis (Test 6 LTOO)",
                  color=ACCENT, fontsize=12, y=1.01)

    # Left: bar chart
    a3l.barh(list(y_pos), margins, color=colors, height=0.65)
    a3l.set_yticks(list(y_pos)); a3l.set_yticklabels(tablets, fontsize=8)
    a3l.invert_yaxis()
    a3l.set_xlabel("|PPL margin|", color=MUTED, fontsize=8)
    a3l.set_title(f"Direction preference  (a→b: {n_ab}, b→a: {n_ba})",
                  color=TEXT, fontsize=9)
    for i, tablet in enumerate(tablets):
        if _stratum_label(tablet) == "pre-contact":
            a3l.get_yticklabels()[i].set_color(PRE_EDGE)

    # Right: scatter
    for r in per_tablet:
        tc = token_counts.get(r["tablet"], 0)
        sm = r["margin"] if r["winner"] == "ab" else -r["margin"]
        col = AB_COL if r["winner"] == "ab" else BA_COL
        a3r.scatter(tc, sm, color=col, s=50, zorder=3)
        a3r.annotate(r["tablet"], (tc, sm), xytext=(4, 2),
                     textcoords="offset points", fontsize=6.5, color=MUTED)
    a3r.axhline(0, color=MUTED, linewidth=0.6, linestyle="--")
    a3r.set_xlabel("Token count", color=MUTED, fontsize=8)
    a3r.set_ylabel("Signed margin", color=MUTED, fontsize=8)
    a3r.set_title("Size vs preference", color=TEXT, fontsize=9)

    plt.tight_layout(pad=1.2)
    out3 = output_dir / "reading_direction_combined.png"
    fig3.savefig(out3, dpi=150, bbox_inches="tight")
    plt.close(fig3)
    log.info("Combined chart → %s", out3)


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

def _smoke_test() -> None:
    fake_data = [
        {"tablet": "D", "ppl_ab": 150.0, "ppl_ba": 155.0, "winner": "ab", "margin": 5.0},
        {"tablet": "B", "ppl_ab": 200.0, "ppl_ba": 198.0, "winner": "ba", "margin": 2.0},
        {"tablet": "C", "ppl_ab": 170.0, "ppl_ba": 168.0, "winner": "ba", "margin": 2.0},
    ]
    token_counts = {"D": 1500, "B": 2000, "C": 1000}
    out = Path("/tmp/rongo_viz_smoke")
    out.mkdir(exist_ok=True)
    try:
        make_plots(fake_data, token_counts, out)
        log.info("Smoke test passed — charts written to %s", out)
    except ImportError as e:
        log.warning("matplotlib not available: %s (smoke test skipped)", e)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Visualise per-tablet reading direction preference."
    )
    p.add_argument(
        "--v2-json", type=Path,
        default=PROJECT_ROOT / "outputs" / "analysis" / "reading_order_v2.json",
    )
    p.add_argument(
        "--corpus-dir", type=Path,
        default=PROJECT_ROOT / "data" / "corpus",
    )
    p.add_argument(
        "--output-dir", type=Path,
        default=PROJECT_ROOT / "outputs" / "analysis",
    )
    p.add_argument("--smoke-test", action="store_true")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    if args.smoke_test:
        _smoke_test()
        return

    if not args.v2_json.exists():
        log.error("reading_order_v2.json not found: %s", args.v2_json)
        log.error("Run:  python scripts/reading_order_v2.py  first.")
        sys.exit(1)

    v2 = json.loads(args.v2_json.read_text(encoding="utf-8"))
    per_tablet: list[dict] = v2.get("test6", {}).get("per_tablet", [])
    if not per_tablet:
        log.error("test6 per_tablet data missing from %s", args.v2_json)
        sys.exit(1)

    log.info("Loaded %d tablet records from test6.", len(per_tablet))
    token_counts = _token_counts(args.corpus_dir)
    log.info("Token counts loaded for %d tablets.", len(token_counts))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    make_plots(per_tablet, token_counts, args.output_dir)


if __name__ == "__main__":
    main()
