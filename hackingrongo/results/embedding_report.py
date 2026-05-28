"""
hackingrongo.results.embedding_report
======================================

Generates an HTML report presenting the Zone A UMAP + HDBSCAN scatter
plots in annotated, accessible form for scholars and ML practitioners.

The two raw PNGs produced by ``scripts/analyze_embeddings.py`` are difficult
to interpret without context: they show thousands of colored dots with no
explanation of what proximity means, what the colors represent, or why the
cluster structure matters for rongorongo research.  This report wraps those
plots in the explanatory scaffolding needed to make them meaningful.

Pipeline position
-----------------
Called automatically at the end of ``scripts/analyze_embeddings.py`` after
UMAP projection and HDBSCAN clustering are complete.  Can also be run
standalone::

    python -m hackingrongo.results.embedding_report \\
        --analysis-dir outputs/analysis \\
        --output outputs/analysis/embedding_report.html

Outputs
-------
    outputs/analysis/embedding_report.html   — self-contained HTML

Public API
----------
``build_embedding_report``   → HTML string
``save_embedding_report``    → writes HTML file
"""

from __future__ import annotations

import argparse
import base64
import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Barthel family metadata
# ---------------------------------------------------------------------------

_FAMILY_COLOURS: dict[str, str] = {
    "objects_plants_phenomena":      "#54b87a",
    "anthropomorphic_type2":         "#e07b54",
    "anthropomorphic_type3":         "#d4754a",
    "miscellaneous_300series":       "#c478d4",
    "miscellaneous_anthropomorphic": "#d4a817",
    "bird_headed":                   "#5b8dd9",
    "zoomorphic":                    "#7bc4a0",
    "additional":                    "#aaaaaa",
    "unlabeled":                     "#888888",
}

_FAMILY_LABELS: dict[str, str] = {
    "objects_plants_phenomena":      "Objects · Plants · Phenomena (1–199)",
    "anthropomorphic_type2":         "Anthropomorphic type-2 head (200–299)",
    "anthropomorphic_type3":         "Anthropomorphic type-3 head (300–399)",
    "miscellaneous_300series":       "Miscellaneous 300-series head (400–499)",
    "miscellaneous_anthropomorphic": "Miscellaneous anthropomorphic (500–599)",
    "bird_headed":                   "Bird-headed figures (600–699)",
    "zoomorphic":                    "Zoomorphic figures (700–799)",
    "additional":                    "Additional / poorly attested (800+)",
    "unlabeled":                     "Unlabeled / unknown",
}

_FAMILY_DESCRIPTIONS: dict[str, str] = {
    "objects_plants_phenomena":
        "The largest family by sign count. Includes tools, plants, celestial "
        "objects, and abstract shapes. Visually the most heterogeneous group — "
        "expect this color to scatter across the map rather than forming one island.",
    "anthropomorphic_type2":
        "Human-like figures with a distinctive type-2 head form. Includes the "
        "critical sign 200 (taxogram). Should cluster near the type-3 family "
        "if the model captures humanoid posture over head detail.",
    "anthropomorphic_type3":
        "Similar posture to type-2 but with a different head morphology. "
        "Barthel distinguished these on iconographic grounds; the model may "
        "merge them if head detail is less salient than body silhouette.",
    "miscellaneous_300series":
        "Figures combining 300-series head types with varied body forms. "
        "A catch-all category — expect mixing with adjacent families.",
    "miscellaneous_anthropomorphic":
        "Additional anthropomorphic forms that don't fit neatly into Barthel's "
        "main categories. Likely to appear as a transitional zone between islands.",
    "bird_headed":
        "Figures with avian heads on humanoid bodies. Visually distinctive — "
        "should form a coherent cluster. Overlap with zoomorphic signs is possible "
        "where body posture is more salient than head type.",
    "zoomorphic":
        "Animal-body signs, primarily fish and reptile forms. Should cluster "
        "together unless the autoencoder conflates them with similarly shaped "
        "objects from the 1–199 range.",
    "additional":
        "Rarely attested signs and late additions to the catalog. Too sparse "
        "for reliable clustering — expect these to fall in the gray noise region.",
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _load_png_as_b64(path: Path) -> str | None:
    """Return a base64 data URI for a PNG, or None if the file doesn't exist."""
    if not path.exists():
        return None
    try:
        data = base64.b64encode(path.read_bytes()).decode()
        return f"data:image/png;base64,{data}"
    except Exception as exc:
        logger.warning("Could not load %s: %s", path, exc)
        return None


def _load_cluster_stats(analysis_dir: Path) -> dict[str, Any] | None:
    """Load cluster_vs_barthel.json if available."""
    p = analysis_dir / "cluster_vs_barthel.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("Could not load cluster stats: %s", exc)
        return None


# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------

_CSS = """\
:root {
  --bg: #ffffff; --surface: #f8f8fa; --surface2: #f0f0f5;
  --border: #d0d0dd; --text: #1a1a1a; --muted: #666666;
  --accent: #c4a96d; --accent2: #7b9ee0; --accent3: #54b87a;
}
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
  background: var(--bg); color: var(--text);
  font-family: 'Cormorant Garamond', 'Palatino Linotype', Georgia, serif;
  font-size: 16px; line-height: 1.65;
}
.wrap { max-width: 1100px; margin: 0 auto; padding: 52px 28px; }

/* ── Header ── */
.report-header { border-bottom: 2px solid var(--border);
                 padding-bottom: 38px; margin-bottom: 48px; }
.report-title { font-size: 34px; font-weight: 600; color: #000; letter-spacing: -0.3px; }
.report-subtitle { font-size: 17px; color: var(--accent); font-style: italic; margin-top: 6px; }
.report-meta { margin-top: 18px; font-family: 'JetBrains Mono', monospace;
               font-size: 11px; color: var(--muted); line-height: 2.2; }
.report-meta b { color: #333; }

/* ── Intro prose ── */
.intro { max-width: 800px; margin-bottom: 44px; }
.intro p { font-size: 15px; color: #333; line-height: 1.9; margin-bottom: 14px; }
.intro p:last-child { margin-bottom: 0; }
.intro b { color: #000; }
.intro code { font-family: 'JetBrains Mono', monospace; font-size: 12px;
              background: var(--surface2); border: 1px solid var(--border);
              border-radius: 2px; padding: 1px 5px; }

/* ── Section headers ── */
.section-head { font-size: 24px; font-weight: 600; color: #000;
                margin: 48px 0 6px; border-top: 1px solid var(--border);
                padding-top: 30px; }
.section-sub { font-size: 14px; color: var(--muted); font-style: italic;
               margin-bottom: 24px; }

/* ── Callout boxes ── */
.callout { border-left: 3px solid var(--accent); background: var(--surface);
           border-radius: 0 6px 6px 0; padding: 16px 20px;
           margin: 20px 0; max-width: 800px; }
.callout.technical { border-color: var(--accent2); }
.callout.finding   { border-color: var(--accent3); }
.callout-label { font-family: 'JetBrains Mono', monospace; font-size: 9.5px;
                 letter-spacing: 0.1em; text-transform: uppercase;
                 color: var(--muted); margin-bottom: 6px; }
.callout p { font-size: 13.5px; color: #333; line-height: 1.85; }
.callout p + p { margin-top: 8px; }
.callout b { color: #000; }
.callout code { font-family: 'JetBrains Mono', monospace; font-size: 11px;
                background: var(--surface2); border: 1px solid var(--border);
                border-radius: 2px; padding: 1px 5px; }

/* ── Plot embed ── */
.plot-wrap { background: var(--surface); border: 1px solid var(--border);
             border-radius: 8px; padding: 20px; margin: 24px 0;
             overflow: hidden; text-align: center; }
.plot-wrap img { max-width: 100%; height: auto; border-radius: 4px; }
.plot-caption { font-size: 12px; color: var(--muted); margin-top: 12px;
                font-style: italic; line-height: 1.6; }
.plot-pending { background: var(--surface2); border: 1px dashed var(--border);
                border-radius: 8px; padding: 36px 24px; text-align: center;
                margin: 24px 0; }
.plot-pending-title { font-family: 'JetBrains Mono', monospace; font-size: 13px;
                      color: var(--muted); margin-bottom: 10px; }
.plot-pending-body { font-size: 13px; color: var(--muted); line-height: 1.7; }
.plot-pending-body code { font-family: 'JetBrains Mono', monospace; font-size: 11px;
                          background: #e8e8e8; border-radius: 2px; padding: 1px 5px; }

/* ── Panel annotations (left / right descriptions) ── */
.panels-row { display: grid; grid-template-columns: 1fr 1fr; gap: 24px;
              margin: 20px 0 0; }
.panel-annotation { background: var(--surface2); border: 1px solid var(--border);
                    border-radius: 6px; padding: 16px 18px; }
.panel-title { font-size: 14px; font-weight: 600; color: #111;
               margin-bottom: 8px; border-bottom: 1px solid var(--border);
               padding-bottom: 6px; }
.panel-body { font-size: 13px; color: #444; line-height: 1.75; }
.panel-body b { color: #000; }
.panel-body ul { margin: 8px 0 0 18px; }
.panel-body li { margin-bottom: 4px; }

/* ── Family legend ── */
.legend-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(260px, 1fr));
               gap: 10px; margin: 20px 0 30px; }
.legend-card { display: flex; gap: 10px; align-items: flex-start;
               background: var(--surface); border: 1px solid var(--border);
               border-radius: 6px; padding: 10px 14px; }
.legend-swatch { flex-shrink: 0; width: 14px; height: 14px; border-radius: 3px;
                 margin-top: 3px; }
.legend-text { flex: 1; }
.legend-label { font-size: 12.5px; font-weight: 600; color: #111; }
.legend-desc { font-size: 11.5px; color: var(--muted); line-height: 1.55;
               margin-top: 2px; }

/* ── Metrics table ── */
.metrics-row { display: flex; flex-wrap: wrap; gap: 14px; margin: 20px 0 32px; }
.metric-card { background: var(--surface); border: 1px solid var(--border);
               border-radius: 7px; padding: 14px 18px; min-width: 120px; text-align: center; }
.metric-n { font-family: 'JetBrains Mono', monospace; font-size: 24px;
            font-weight: 500; color: var(--accent); display: block; }
.metric-label { font-size: 10.5px; color: var(--muted); line-height: 1.4;
                margin-top: 2px; display: block; }
.metric-interp { font-size: 11px; color: var(--accent3); margin-top: 4px; display: block; }

/* ── Interpretation table ── */
.interp-table { width: 100%; border-collapse: collapse; font-size: 13.5px;
                margin: 16px 0 32px; }
.interp-table th { text-align: left; padding: 8px 14px;
                   font-family: 'JetBrains Mono', monospace; font-size: 10px;
                   letter-spacing: 0.08em; text-transform: uppercase;
                   color: var(--muted); border-bottom: 1px solid var(--border);
                   background: var(--surface); }
.interp-table td { padding: 10px 14px; border-bottom: 1px solid var(--border);
                   color: #333; line-height: 1.65; vertical-align: top; }
.interp-table tr:last-child td { border-bottom: none; }
.interp-table td:first-child { font-family: 'JetBrains Mono', monospace;
                                font-size: 11.5px; color: var(--accent); width: 120px; }

/* ── Footer ── */
.report-footer { border-top: 1px solid var(--border); margin-top: 52px;
                 padding-top: 26px; font-size: 12px; color: var(--muted);
                 line-height: 2.0; }
.report-footer a { color: var(--accent); text-decoration: none; }
.report-footer code { background: var(--surface2); border: 1px solid var(--border);
                      border-radius: 2px; padding: 1px 5px;
                      font-family: 'JetBrains Mono', monospace; }

@media (max-width: 700px) {
  .panels-row { grid-template-columns: 1fr; }
  .legend-grid { grid-template-columns: 1fr; }
  .metrics-row { flex-direction: column; }
}
"""


# ---------------------------------------------------------------------------
# HTML rendering helpers
# ---------------------------------------------------------------------------


def _legend_html() -> str:
    cards = ""
    for fam, colour in _FAMILY_COLOURS.items():
        label = _FAMILY_LABELS.get(fam, fam)
        desc  = _FAMILY_DESCRIPTIONS.get(fam, "")
        cards += (
            f'<div class="legend-card">'
            f'<div class="legend-swatch" style="background:{colour}"></div>'
            f'<div class="legend-text">'
            f'<div class="legend-label">{label}</div>'
            f'<div class="legend-desc">{desc}</div>'
            f'</div></div>'
        )
    return f'<div class="legend-grid">{cards}</div>'


def _plot_block(b64: str | None, caption: str, pending_msg: str) -> str:
    if b64:
        return (
            f'<div class="plot-wrap">'
            f'<img src="{b64}" alt="{caption}">'
            f'<div class="plot-caption">{caption}</div>'
            f'</div>'
        )
    return (
        f'<div class="plot-pending">'
        f'<div class="plot-pending-title">Plot not yet generated</div>'
        f'<div class="plot-pending-body">{pending_msg}</div>'
        f'</div>'
    )


def _metrics_html(stats: dict[str, Any] | None) -> str:
    if not stats:
        return (
            '<div class="callout">'
            '<div class="callout-label">Metrics</div>'
            '<p>No cluster statistics available yet — run '
            '<code>scripts/analyze_embeddings.py</code> to generate.</p>'
            '</div>'
        )

    n_embeddings = stats.get("n_embeddings", "—")
    n_clusters   = stats.get("n_clusters", "—")
    n_noise      = stats.get("n_noise_points", "—")
    n_labeled    = stats.get("n_labeled", "—")
    ari          = stats.get("adjusted_rand_index")
    interpretation = stats.get("interpretation", "—")
    fm           = stats.get("family_metrics") or {}
    nmi          = fm.get("nmi")
    v_measure    = fm.get("v_measure")

    def _fmt(v: float | None, decimals: int = 3) -> str:
        return f"{v:.{decimals}f}" if v is not None else "—"

    noise_pct = (
        f"{n_noise / n_embeddings * 100:.1f}%"
        if isinstance(n_noise, int) and isinstance(n_embeddings, int) and n_embeddings
        else "—"
    )

    ari_interp = ""
    if ari is not None:
        if ari >= 0.5:
            ari_interp = "strong alignment"
        elif ari >= 0.3:
            ari_interp = "moderate alignment"
        elif ari >= 0.1:
            ari_interp = "weak alignment"
        else:
            ari_interp = "near-random"

    cards = [
        (str(n_embeddings), "glyph instances\nin corpus", ""),
        (str(n_clusters),   "HDBSCAN clusters\ndiscovered", ""),
        (noise_pct,         "of glyphs in\nnoise bucket", ""),
        (_fmt(ari),         "Adjusted Rand Index\n(family alignment)", ari_interp),
        (_fmt(nmi),         "Normalised Mutual\nInformation", ""),
        (_fmt(v_measure),   "V-measure\n(harmonic mean)", ""),
    ]

    def _card(n: str, lbl: str, interp: str) -> str:
        interp_span = f'<span class="metric-interp">{interp}</span>' if interp else ""
        return (
            f'<div class="metric-card">'
            f'<span class="metric-n">{n}</span>'
            f'<span class="metric-label">{lbl}</span>'
            f'{interp_span}'
            f'</div>'
        )
    cards_html = "".join(_card(n, lbl, interp) for n, lbl, interp in cards)
    return f'<div class="metrics-row">{cards_html}</div>'


def _interpretation_table_html(stats: dict[str, Any] | None) -> str:
    rows = [
        ("tight island,\nsame color",
         "Barthel's category aligns with visual form. The model independently "
         "confirms that these signs share enough visual structure to group "
         "together — supporting the iconographic logic behind Barthel's taxonomy."),
        ("tight island,\nmixed colors",
         "Visual similarity cuts across Barthel's categories. The model sees "
         "a coherent shape family where Barthel distinguished semantic sub-types. "
         "These are the most informative divergences — see the Divergence Report "
         "for glyph-level detail."),
        ("sparse scatter,\nsame color",
         "Barthel's category is visually diverse. The signs share semantic "
         "or functional identity but not a unified visual form — the model "
         "correctly distributes them across different shape neighborhoods."),
        ("gray dots\n(noise)",
         "Signs that don't cluster with any family. Likely: rare signs seen "
         "too few times for the model to generalise; compound glyphs whose "
         "visual form is a hybrid of two clusters; or highly variant allographs. "
         "Gray density correlates with scribal variation and sign frequency."),
        ("bridge between\ntwo islands",
         "Intermediate forms — signs that share visual features with two "
         "distinct families. Candidates for allograph relationships or "
         "compound glyphs. The Compound Glyph Report analyzes these cases."),
    ]
    rows_html = "".join(
        f'<tr><td>{pat}</td><td>{meaning}</td></tr>'
        for pat, meaning in rows
    )
    return (
        '<table class="interp-table">'
        '<thead><tr><th>What you see</th><th>What it means</th></tr></thead>'
        f'<tbody>{rows_html}</tbody>'
        '</table>'
    )


# ---------------------------------------------------------------------------
# Full HTML document
# ---------------------------------------------------------------------------


def _render_html(
    umap_b64:  str | None,
    hier_b64:  str | None,
    stats:     dict[str, Any] | None,
    generated: str,
    run_meta:  dict[str, str] | None = None,
) -> str:
    meta_items = ""
    if run_meta:
        meta_items = " &nbsp;·&nbsp; ".join(
            f"<b>{k}:</b> {v}" for k, v in run_meta.items()
        )
    meta_items = (meta_items + " &nbsp;·&nbsp; " if meta_items else "") + f"<b>Generated:</b> {generated}"

    n_clusters_str = str(stats.get("n_clusters", "?")) if stats else "?"
    ari_str = f"{stats['adjusted_rand_index']:.3f}" if stats and stats.get("adjusted_rand_index") is not None else "—"
    interp_str = stats.get("interpretation", "pending").replace("_", " ") if stats else "pending"

    umap_block = _plot_block(
        umap_b64,
        "Figure 1 — UMAP projection of Zone A autoencoder embeddings. "
        "Left: each dot colored by Barthel sign family. "
        "Right: each dot colored by the HDBSCAN cluster the model discovered.",
        "Run <code>scripts/analyze_embeddings.py</code> after Zone A autoencoder "
        "training to generate this plot.",
    )

    hier_block = _plot_block(
        hier_b64,
        "Figure 2 — Dendrogram comparison for signs 200–399. "
        "Left: Barthel's implicit 3-level tree (century → tens group → units). "
        "Right: hierarchy derived from autoencoder embedding distances. "
        "Cophenetic correlation measures how closely the two trees agree.",
        "Run <code>scripts/analyze_embeddings.py</code> to generate this plot.",
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>hackingrongo — Sign Space Visualization</title>
<link href="https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,400;0,600;1,400&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>{_CSS}</style>
</head>
<body>
<div class="wrap">

<!-- ── HEADER ── -->
<div class="report-header">
  <div class="report-title">hackingrongo<br>Sign Space Visualization</div>
  <div class="report-subtitle">
    What the autoencoder learned about rongorongo glyph geometry
  </div>
  <div class="report-meta">{meta_items}</div>
</div>

<!-- ── INTRODUCTION ── -->
<div class="intro">
  <p>
    This report presents the output of Zone A — the visual embedding pipeline
    — in a form designed to be read by both rongorongo scholars and machine
    learning practitioners.  The key question Zone A asks is:
    <b>if you taught a neural network to recognize rongorongo glyphs purely
    from their visual shapes, which signs would it group together — and does
    that grouping match what Barthel (1958) decided by hand?</b>
  </p>
  <p>
    The answer is shown as a scatter plot: every glyph instance in the corpus
    becomes a dot in a 2D map where <b>proximity = visual similarity</b>.
    Signs that look alike end up close together, regardless of what Barthel
    called them or what tablet they came from.  The colored pattern of that
    map — and the places where it diverges from Barthel's categories — is
    the central finding of Zone A.
  </p>
</div>

<!-- ── HOW IT WORKS ── -->
<div class="section-head">How It Works</div>
<div class="section-sub">A plain-English account of the three-step pipeline</div>

<div class="callout technical">
  <div class="callout-label">For ML practitioners</div>
  <p>
    A convolutional autoencoder is trained on cropped glyph images (25 tablets,
    ~14 000 instances). The bottleneck layer produces a fixed-length embedding
    vector per glyph. UMAP projects the embedding cloud to 2D (metric: cosine,
    n_neighbors from config). HDBSCAN clusters the 2D projection without a
    pre-specified k. Adjusted Rand Index against Barthel's 8-family taxonomy
    is the primary alignment metric. Current run: <b>{n_clusters_str} clusters</b>,
    ARI = <b>{ari_str}</b> ({interp_str} with Barthel taxonomy).
  </p>
</div>

<div class="callout">
  <div class="callout-label">For rongorongo scholars</div>
  <p>
    <b>Step 1 — The autoencoder.</b>
    We showed a neural network hundreds of thousands of individual glyph images
    cropped from all 25 tablets.  The network's job was to memorise each image
    using only a short list of numbers — and then reconstruct it from those numbers
    alone.  To do that well, the numbers had to capture what matters visually:
    the overall silhouette, the orientation of limbs, the presence of head
    features, and so on.  Signs that look alike end up with similar numbers.
  </p>
  <p>
    <b>Step 2 — UMAP.</b>
    Each glyph is now a point in a high-dimensional space defined by those
    numbers.  UMAP flattens that space down to two dimensions — like making a
    flat map of a globe — while trying to keep nearby points nearby.  The
    result is the scatter plot: every dot is one glyph occurrence, and dots
    that are close together are visually similar to the network.
  </p>
  <p>
    <b>Step 3 — HDBSCAN clustering.</b>
    HDBSCAN looks at the map and finds dense islands of dots, labelling each
    island as a cluster.  Crucially, it does this <i>without knowing anything
    about Barthel's sign families</i>.  It is discovering natural visual groups
    from scratch.  Dots in sparse areas that don't belong to any island are
    coloured gray (noise).
  </p>
</div>

<!-- ── THE UMAP SCATTER PLOT ── -->
<div class="section-head">The UMAP Scatter Plot</div>
<div class="section-sub">The geometry of the rongorongo sign inventory</div>

{umap_block}

<div class="panels-row">
  <div class="panel-annotation">
    <div class="panel-title">Left panel — Colored by Barthel sign family</div>
    <div class="panel-body">
      <p>Each dot is one glyph instance, colored by Barthel's classification
      (see legend below). <b>Tight single-color islands</b> mean the model's
      visual grouping matches Barthel's category — confirmation that his
      distinctions reflect real visual structure. <b>Mixed-color zones</b>
      reveal where the model groups signs that Barthel placed in different
      families — the most interesting places for scholars to look.</p>
      <ul>
        <li>Do the anthropomorphic families (orange, rust) cluster together or separate?</li>
        <li>Do bird-headed signs (blue) stay isolated from zoomorphic (teal)?</li>
        <li>Where does the large 1–199 family (green) scatter to?</li>
      </ul>
    </div>
  </div>
  <div class="panel-annotation">
    <div class="panel-title">Right panel — Colored by HDBSCAN cluster</div>
    <div class="panel-body">
      <p>The model's own groupings, discovered without any labels. Each color
      is a cluster; <b>gray dots are noise</b> — signs that don't fit
      comfortably into any cluster.</p>
      <ul>
        <li>How many natural visual families does the model find?</li>
        <li>Do the cluster boundaries line up with Barthel's family boundaries?</li>
        <li>Which signs fall in the gray noise cloud — rare signs, compounds,
        or scribal variants?</li>
      </ul>
      <p>Compare the two panels: wherever a cluster in the right panel is
      color-pure in the left panel, the two classification systems agree.
      Wherever they disagree is where new scholarship could emerge.</p>
    </div>
  </div>
</div>

<!-- ── HOW TO READ THE MAP ── -->
<div class="section-head">How to Read the Map</div>
<div class="section-sub">A guide to interpreting spatial patterns</div>

{_interpretation_table_html(stats)}

<div class="callout finding">
  <div class="callout-label">Key interpretive principle</div>
  <p>
    The axes (UMAP 1, UMAP 2) have <b>no inherent meaning</b> — they are
    mathematical coordinates, not measurements of anything like "complexity"
    or "frequency."  What matters is only <b>relative distance</b>: two dots
    close together are visually similar; two dots far apart are visually
    different.  The overall shape of the cloud can rotate or reflect between
    runs without affecting the findings.
  </p>
</div>

<!-- ── SIGN FAMILY LEGEND ── -->
<div class="section-head">Barthel Sign Family Legend</div>
<div class="section-sub">
  What each color represents and what to expect from each family's distribution
</div>

{_legend_html()}

<!-- ── THE DENDROGRAM COMPARISON ── -->
<div class="section-head">Hierarchical Structure: Barthel vs. Embeddings</div>
<div class="section-sub">
  Does the model recover Barthel's 3-level tree for signs 200–399?
</div>

<div class="callout">
  <div class="callout-label">What this shows</div>
  <p>
    Barthel's numbering of the anthropomorphic signs (200–399) encodes an
    implicit 3-level tree: <b>century block</b> (200s vs 300s) →
    <b>tens group</b> (head type within each century) →
    <b>units digit</b> (individual sign variant).  If the autoencoder learned
    the same visual hierarchy, the tree it produces from embedding distances
    should resemble Barthel's tree.
  </p>
  <p>
    The <b>cophenetic correlation</b> in the right panel title measures how
    closely the two trees agree: 1.0 = perfect agreement, 0.0 = random.
    A moderate or high score here is evidence that Barthel's fine-grained
    distinctions within the anthropomorphic range reflect real visual
    sub-structure that the model has independently recovered.
  </p>
</div>

{hier_block}

<!-- ── METRICS ── -->
<div class="section-head">Cluster Quality Metrics</div>
<div class="section-sub">
  How well do the data-driven clusters align with Barthel's sign families?
</div>

{_metrics_html(stats)}

<div class="callout technical">
  <div class="callout-label">Metric glossary</div>
  <p>
    <b>Adjusted Rand Index (ARI)</b> — measures overlap between two clusterings
    (HDBSCAN labels vs Barthel families), adjusted for chance.
    Range: −1 to 1; 0 = random; 1 = perfect.
    ARI ≥ 0.3 is typically considered meaningful agreement for unsupervised clustering.
  </p>
  <p>
    <b>Normalised Mutual Information (NMI)</b> — how much information knowing one
    clustering gives you about the other, normalised to [0, 1].
  </p>
  <p>
    <b>V-measure</b> — harmonic mean of homogeneity (clusters contain one family)
    and completeness (each family is in one cluster).  Balanced between the two errors.
  </p>
</div>

<!-- ── WHAT TO DO NEXT ── -->
<div class="section-head">What to Do With This</div>
<div class="section-sub">How this report connects to the rest of the pipeline</div>

<div class="callout finding">
  <div class="callout-label">Next steps</div>
  <p>
    <b>Cluster divergences in detail:</b> The
    <a href="divergence_report.html">Divergence Report</a>
    shows every mixed cluster with the actual glyph drawings side by side,
    and provides a data-driven explanation of why each divergence occurred.
    If you see an interesting mixed zone in the scatter plot, the divergence
    report will tell you exactly which signs are involved.
  </p>
  <p>
    <b>Compound glyph candidates:</b> Signs in the gray noise cloud between
    two clusters are prime compound glyph candidates. The
    <a href="compound_report.html">Compound Glyph Report</a>
    lists these computationally, ranked by how much evidence the pipeline
    found for each.  The scholarly comparison section of that report also
    documents how our findings relate to prior compound glyph scholarship.
  </p>
</div>

<!-- ── FOOTER ── -->
<div class="report-footer">
  <p><b>hackingrongo</b> · Sign Space Visualization · MIT License ·
  <a href="https://github.com/violasarah2000/hackingrongo" target="_blank">GitHub</a></p>
  <p>Embedding pipeline: convolutional autoencoder (Zone A) →
  UMAP projection → HDBSCAN clustering.</p>
  <p>Barthel sign family taxonomy: Barthel (1958)
  <em>Grundlagen zur Entzifferung der Osterinselschrift</em>.</p>
  <p>This is a computational hypothesis report, not a decipherment claim.
  All findings require expert epigraphic review.</p>
  <p><b>SperksWerks LLC</b> ·
  <a href="https://sperkswerks.ai" target="_blank">sperkswerks.ai</a> ·
  <a href="mailto:studio@sperkswerks.ai">studio@sperkswerks.ai</a></p>
</div>

</div>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_embedding_report(
    analysis_dir: Path,
    run_metadata: dict[str, str] | None = None,
) -> str:
    """Build the embedding space report HTML.

    Parameters
    ----------
    analysis_dir : Path
        Directory containing ``umap_embeddings.png``,
        ``hierarchy_vs_barthel.png``, and ``cluster_vs_barthel.json``.
        All three are optional; if absent the corresponding sections show
        a "pending" placeholder.
    run_metadata : dict[str, str], optional
        Extra key→value pairs shown in the report header (e.g. run_id,
        experiment name, corpus size).

    Returns
    -------
    str
        Complete self-contained HTML document.
    """
    from datetime import datetime, timezone

    umap_b64 = _load_png_as_b64(analysis_dir / "umap_embeddings.png")
    hier_b64 = _load_png_as_b64(analysis_dir / "hierarchy_vs_barthel.png")
    stats    = _load_cluster_stats(analysis_dir)

    if umap_b64:
        logger.info("Loaded umap_embeddings.png")
    else:
        logger.info("umap_embeddings.png not found — placeholder will be shown.")

    generated = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    return _render_html(umap_b64, hier_b64, stats, generated, run_metadata)


def save_embedding_report(
    analysis_dir: Path,
    output_path: Path,
    run_metadata: dict[str, str] | None = None,
) -> None:
    """Build and write the embedding space report.

    Parameters
    ----------
    analysis_dir : Path
        Directory containing the plot PNGs and JSON from
        ``scripts/analyze_embeddings.py``.
    output_path : Path
        Destination ``.html`` file.  Parent directories are created.
    run_metadata : dict[str, str], optional
        Extra key→value pairs for the report header.
    """
    html = build_embedding_report(analysis_dir, run_metadata)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    logger.info("Embedding report written: %s (%d bytes).", output_path, len(html))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate the Zone A embedding space visualization report."
    )
    p.add_argument(
        "--analysis-dir", type=Path,
        default=Path("outputs/analysis"),
        help="Directory with umap_embeddings.png, hierarchy_vs_barthel.png, "
             "cluster_vs_barthel.json (default: outputs/analysis).",
    )
    p.add_argument(
        "--output", type=Path,
        default=None,
        help="Output HTML path (default: <analysis-dir>/embedding_report.html).",
    )
    return p.parse_args()


def main() -> None:
    import logging as _logging
    _logging.basicConfig(level=_logging.INFO, format="%(levelname)s  %(message)s")
    args = _parse_args()
    output = args.output or (args.analysis_dir / "embedding_report.html")
    save_embedding_report(
        analysis_dir=args.analysis_dir,
        output_path=output,
    )
    print(f"Report written to: {output}")


if __name__ == "__main__":
    main()
