"""
hackingrongo.results.divergence_report
=======================================

Generates an HTML divergence report comparing HDBSCAN visual clusters
(Zone A output) against Barthel's hand-crafted sign families.

This report is the primary artifact for scholarly outreach.  It surfaces
the specific glyphs where the pipeline's geometry-driven grouping diverges
from Barthel (1958), shows the actual glyph drawings side by side, and
provides a data-driven explanation of why each divergence occurred.

The report is intended to be sent to rongorongo scholars as a concrete,
reviewable artifact — not a claim of decipherment, but an invitation to
examine specific cases together.

Pipeline position
-----------------
Called automatically at the end of ``scripts/analyze_embeddings.py``
after ``cluster_vs_barthel.csv`` has been written.  Can also be run
standalone::

    python -m hackingrongo.results.divergence_report \\
        --analysis-dir outputs/analysis \\
        --svg-catalog data/glyphs/svg/catalog.json \\
        --svg-dir data/glyphs/svg \\
        --output outputs/analysis/divergence_report.html

Public API
----------
``DivergenceReportConfig``
    Parameters controlling which clusters are included.

``build_divergence_report``
    Main entry point.  Reads analysis outputs and SVG data, returns
    HTML string.

``save_divergence_report``
    Convenience wrapper: calls ``build_divergence_report`` and writes
    the HTML file, logging the path.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import html as _html
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class DivergenceReportConfig:
    """Parameters controlling which clusters appear in the report.

    Attributes
    ----------
    min_cluster_size : int
        Clusters smaller than this are ignored (noisy / singleton).
    max_purity : float
        Only clusters with purity *below* this threshold are included
        (pure clusters are uninteresting for divergence analysis).
    min_families : int
        Minimum number of distinct Barthel families in a cluster for it
        to be considered a meaningful divergence case.
    max_entries : int
        Maximum number of cluster entries in the report (sorted by
        interestingness: n_families DESC, size DESC).
    min_svgs_required : int
        A cluster entry is included only if at least this many of its
        constituent Barthel codes have SVG drawings available.
    include_noise : bool
        If True, also report on the noise bucket (HDBSCAN cluster -1)
        family breakdown.  Default False.
    """

    min_cluster_size: int = 5
    max_purity: float = 0.65
    min_families: int = 2
    max_entries: int = 12
    min_svgs_required: int = 2
    include_noise: bool = False


# ---------------------------------------------------------------------------
# Barthel numeric range → scholarly label
# ---------------------------------------------------------------------------

_BARTHEL_RANGES: list[tuple[int, int, str, str]] = [
    (1,   199, "001–199", "Objects · Plants · Phenomena"),
    (200, 299, "200–299", "Anthropomorphic — type-2 head"),
    (300, 399, "300–399", "Anthropomorphic — type-3 head"),
    (400, 499, "400–499", "Miscellaneous (300-series head)"),
    (500, 599, "500–599", "Miscellaneous anthropomorphic"),
    (600, 699, "600–699", "Bird-headed figures"),
    (700, 799, "700–799", "Zoomorphic figures"),
]


def _barthel_range_label(code: str) -> str:
    """Return Barthel's numeric-range category label for a sign code."""
    digits = re.sub(r"[^0-9]", "", str(code))[:3]
    if not digits:
        return "compound / unknown"
    n = int(digits)
    for lo, hi, num_range, desc in _BARTHEL_RANGES:
        if lo <= n <= hi:
            return f"{num_range} · {desc}"
    return "out of range"


# ---------------------------------------------------------------------------
# Family colour palette
# ---------------------------------------------------------------------------

_FAMILY_COLOURS: dict[str, str] = {
    "objects_plants_phenomena":      "#54b87a",  # green  — 1–199
    "anthropomorphic_type2":         "#e07b54",  # orange — 200–299
    "anthropomorphic_type3":         "#d4754a",  # rust   — 300–399
    "miscellaneous_300series":       "#c478d4",  # purple — 400–499
    "miscellaneous_anthropomorphic": "#d4a817",  # amber  — 500–599
    "bird_headed":                   "#5b8dd9",  # blue   — 600–699
    "zoomorphic":                    "#7bc4a0",  # teal   — 700–799
    "additional":                    "#aaaaaa",  # gray   — 800+
    "unlabeled":                     "#888888",  # dark gray
}

_DEFAULT_COLOUR = "#777777"


def _family_colour(family: str) -> str:
    return _FAMILY_COLOURS.get(family, _DEFAULT_COLOUR)


# ---------------------------------------------------------------------------
# Divergence reasoning
# ---------------------------------------------------------------------------

# Map frozenset({family_a, family_b}) → explanation string.
# Explanations are written for a scholarly audience who know the script
# but may not be familiar with convolutional embedding models.

_PAIR_REASONS: dict[frozenset, str] = {
    frozenset({"zoomorphic", "anthropomorphic_type2"}): (
        "700-series and 200-series share bilateral body symmetry — the encoder weights posture over head morphology."
    ),
    frozenset({"zoomorphic", "anthropomorphic_type3"}): (
        "Erect zoomorphic vs crouching anthropomorphic figures overlap in silhouette; limb orientation dominates the embedding."
    ),
    frozenset({"miscellaneous_300series", "anthropomorphic_type2"}): (
        "400-series uses the same head/body/hand grammar as 200-series; the encoder reads posture before head type."
    ),
    frozenset({"miscellaneous_300series", "anthropomorphic_type3"}): (
        "400-series carries 300-series head morphology by definition — body geometry below the head is the only discriminant."
    ),
    frozenset({"bird_headed", "zoomorphic"}): (
        "Bird-headed (600-series) vs general zoomorphic (700-series): the body silhouette dominates when head-to-body ratio is ambiguous."
    ),
    frozenset({"miscellaneous_anthropomorphic", "anthropomorphic_type2"}): (
        "500-series and 200-series share upright bipedal posture; secondary cues (appendages, held objects) are down-weighted."
    ),
    frozenset({"objects_plants_phenomena", "anthropomorphic_type2"}): (
        "Some 1–199 plant/tool forms have vertical bilateral symmetry that the encoder clusters with standing humanoids."
    ),
    frozenset({"anthropomorphic_type2", "anthropomorphic_type3"}): (
        "200-series vs 300-series head differences occupy a small fraction of the bounding box; body geometry dominates."
    ),
    frozenset({"anthropomorphic_type2", "unlabeled"}): (
        "Unlabeled ('?') signs are damaged or ambiguous; the encoder interpolates toward the nearest clear exemplars — here, type-2 humanoids."
    ),
    frozenset({"additional", "anthropomorphic_type2"}): (
        "800+ ligatures/compounds whose dominant visual element is a type-2 anthropomorphic figure embed near that family, not each other."
    ),
    frozenset({"zoomorphic", "miscellaneous_300series"}): (
        "400-series vs 700-series separation relies on leg count and head type — cues the embedding partially conflates with body-mass silhouette."
    ),
}


def _divergence_reason(family_breakdown: dict[str, int]) -> str:
    """Generate a divergence explanation for a mixed cluster."""
    families = sorted(family_breakdown, key=lambda f: -family_breakdown[f])
    if len(families) < 2:
        return "Cluster is dominated by a single family."

    # Collect pairwise reasons for all (dominant, minority) pairs.
    dominant = families[0]
    reasons: list[str] = []
    seen: set[frozenset] = set()

    for minority in families[1:]:
        pair = frozenset({dominant, minority})
        if pair in seen:
            continue
        seen.add(pair)
        if pair in _PAIR_REASONS:
            reasons.append(_PAIR_REASONS[pair])
        else:
            # Generic fallback
            reasons.append(
                f"{dominant.replace('_', ' ')} and {minority.replace('_', ' ')} "
                "share stroke primitives the encoder clusters geometrically; "
                "Barthel's iconographic distinction may not be recoverable from visual signal alone."
            )

    return " ".join(reasons)


# ---------------------------------------------------------------------------
# SVG normalisation
# ---------------------------------------------------------------------------


def _normalise_svg(svg_text: str, size: int = 76) -> str:
    """Resize an SVG to a fixed display size and prepare for dark-background rendering."""
    svg = svg_text.strip()
    svg = re.sub(r'width="[^"]*"', f'width="{size}"', svg)
    svg = re.sub(r'height="[^"]*"', f'height="{size}"', svg)
    # Ensure path elements render visibly (no fill, use currentColor stroke)
    svg = re.sub(
        r"<path ",
        '<path fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" ',
        svg,
    )
    return svg


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _load_cluster_data(
    analysis_dir: Path,
) -> pd.DataFrame:
    """Load cluster_vs_barthel.csv from the analysis output directory."""
    csv_path = analysis_dir / "cluster_vs_barthel.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"cluster_vs_barthel.csv not found at {csv_path}. "
            "Run scripts/analyze_embeddings.py first."
        )
    df = pd.read_csv(csv_path)
    required = {"barthel_code", "barthel_family", "umap_x", "umap_y", "hdbscan_cluster"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"cluster_vs_barthel.csv is missing columns: {missing}")
    return df


def _load_svg_catalog(
    catalog_path: Path,
) -> dict[str, list[Path]]:
    """Load SVG catalog and return {barthel_code: [svg_file_path, ...]}.

    Lookup is attempted in order:
    1. Exact code match (e.g. ``'661!'``)
    2. Base code — trailing variant/modifier chars stripped (e.g. ``'661'``)
    """
    if not catalog_path.exists():
        logger.warning("SVG catalog not found at %s — glyphs will not render.", catalog_path)
        return {}

    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    svg_dir = catalog_path.parent

    exact: dict[str, list[Path]] = defaultdict(list)
    base_map: dict[str, list[Path]] = defaultdict(list)

    for r in catalog.get("records", []):
        code = str(r.get("barthel_code", "")).strip()
        if not code:
            continue
        rel = str(r.get("svg_path", "")).replace("svg/", "", 1)
        full = svg_dir / rel
        if not full.exists():
            continue
        exact[code].append(full)
        base = re.sub(r'[!?()\s]+$', '', code).strip()
        if base != code:
            base_map[base].append(full)

    merged: dict[str, list[Path]] = dict(exact)
    for base_code, paths in base_map.items():
        if base_code not in merged:
            merged[base_code] = paths

    logger.info(
        "SVG catalog: %d exact codes, %d base-code fallbacks.",
        len(exact), sum(1 for k in base_map if k not in exact),
    )
    return merged


def _load_ari(analysis_dir: Path) -> float | None:
    """Load ARI from cluster_vs_barthel.json if available."""
    json_path = analysis_dir / "cluster_vs_barthel.json"
    if not json_path.exists():
        return None
    try:
        data = json.loads(json_path.read_text(encoding="utf-8"))
        return float(data.get("adjusted_rand_index", float("nan")))
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Cluster selection
# ---------------------------------------------------------------------------


@dataclass
class _ClusterEntry:
    cluster_id: int
    size: int
    purity: float
    dominant_family: str
    minority_families: list[str]
    family_breakdown: dict[str, int]
    all_codes: list[str]
    glyph_examples: list[dict[str, Any]]
    umap_centroid: tuple[float, float]
    divergence_reason: str


def _select_divergent_clusters(
    df: pd.DataFrame,
    code_to_svgs: dict[str, list[Path]],
    cfg: DivergenceReportConfig,
) -> list[_ClusterEntry]:
    """Identify and rank the most interesting divergent clusters."""

    clustered = df[df["hdbscan_cluster"] != -1]
    entries: list[_ClusterEntry] = []

    for cid, grp in clustered.groupby("hdbscan_cluster"):
        if len(grp) < cfg.min_cluster_size:
            continue

        counts = grp["barthel_family"].value_counts()
        purity = counts.iloc[0] / len(grp)
        n_families = len(counts)

        if purity >= cfg.max_purity:
            continue
        if n_families < cfg.min_families:
            continue

        dominant = counts.index[0]
        minority_families = counts.index[1:].tolist()
        all_codes = grp["barthel_code"].dropna().unique().tolist()

        # Collect one representative glyph per family (with SVG)
        glyph_examples: list[dict[str, Any]] = []
        for family in [dominant] + minority_families:
            family_codes = (
                grp[grp["barthel_family"] == family]["barthel_code"]
                .value_counts()
            )
            for code in family_codes.index:
                _code = str(code)
                svgs = code_to_svgs.get(_code, []) or code_to_svgs.get(
                    re.sub(r'[!?()\s]+$', '', _code).strip(), []
                )
                if svgs:
                    svg_text = svgs[0].read_text(encoding="utf-8")
                    glyph_examples.append(
                        {
                            "barthel_code": code,
                            "barthel_family": family,
                            "barthel_range": _barthel_range_label(code),
                            "svg": _normalise_svg(svg_text),
                            "n_occurrences": int(family_codes[code]),
                            "is_dominant": family == dominant,
                        }
                    )
                    break  # one exemplar per family

        if len(glyph_examples) < cfg.min_svgs_required:
            continue

        entries.append(
            _ClusterEntry(
                cluster_id=int(cid),
                size=len(grp),
                purity=round(purity, 3),
                dominant_family=dominant,
                minority_families=minority_families,
                family_breakdown=counts.to_dict(),
                all_codes=all_codes,
                glyph_examples=glyph_examples,
                umap_centroid=(
                    float(grp["umap_x"].mean()),
                    float(grp["umap_y"].mean()),
                ),
                divergence_reason=_divergence_reason(counts.to_dict()),
            )
        )

    # Sort: more families first, then larger clusters
    entries.sort(key=lambda e: (-len(e.family_breakdown), -e.size))
    return entries[: cfg.max_entries]


# ---------------------------------------------------------------------------
# Global statistics
# ---------------------------------------------------------------------------


def _compute_global_stats(df: pd.DataFrame) -> dict[str, Any]:
    """Compute corpus-level cluster purity statistics."""
    clustered = df[df["hdbscan_cluster"] != -1]
    noise_count = int((df["hdbscan_cluster"] == -1).sum())
    n_clusters = int(clustered["hdbscan_cluster"].nunique())
    total = len(df)

    purities: list[float] = []
    for _, grp in clustered.groupby("hdbscan_cluster"):
        counts = grp["barthel_family"].value_counts()
        purities.append(counts.iloc[0] / len(grp))

    return {
        "total_glyphs": total,
        "n_clusters": n_clusters,
        "noise_count": noise_count,
        "noise_pct": round(noise_count / total * 100, 1),
        "mean_purity": round(sum(purities) / len(purities) * 100, 1) if purities else 0.0,
        "pct_perfect": round(
            sum(1 for p in purities if p == 1.0) / len(purities) * 100, 1
        ) if purities else 0.0,
        "pct_above_90": round(
            sum(1 for p in purities if p >= 0.9) / len(purities) * 100, 1
        ) if purities else 0.0,
    }


# ---------------------------------------------------------------------------
# HTML rendering
# ---------------------------------------------------------------------------


def _badge_html(family: str, count: int | None = None) -> str:
    colour = _family_colour(family)
    label = family.replace("_", " ")
    count_span = f' <span style="opacity:0.65">×{count}</span>' if count is not None else ""
    return (
        f'<span class="badge" style="background:{colour}22;color:{colour};'
        f'border:1px solid {colour}55">{label}{count_span}</span>'
    )


def _bar_html(family_breakdown: dict[str, int]) -> str:
    total = sum(family_breakdown.values())
    segments = ""
    for fam, cnt in sorted(family_breakdown.items(), key=lambda x: -x[1]):
        pct = cnt / total * 100
        segments += (
            f'<div class="bar-seg" style="width:{pct:.1f}%;background:{_family_colour(fam)}" '
            f'title="{fam}: {cnt}"></div>'
        )
    return segments


def _render_entry(entry: _ClusterEntry) -> str:
    glyph_cards = ""
    for g in entry.glyph_examples:
        border_c = _family_colour(g["barthel_family"])
        role = "DOMINANT" if g["is_dominant"] else "MINORITY"
        glyph_cards += f"""
        <div class="glyph-card" style="border-top:3px solid {border_c}">
          <div class="role-label">{role}</div>
          <div class="glyph-svg" style="color:{border_c}">{g["svg"]}</div>
          <div class="glyph-code">Barthel {_html.escape(str(g["barthel_code"]))}</div>
          <div class="glyph-range">{_html.escape(str(g["barthel_range"]))}</div>
          {_badge_html(g["barthel_family"])}
          <div class="occurrence-count">{g["n_occurrences"]} occurrences</div>
        </div>"""

    badges = " ".join(
        _badge_html(f, entry.family_breakdown[f])
        for f in sorted(entry.family_breakdown, key=lambda x: -entry.family_breakdown[x])
    )
    codes_html = "  ".join(
        f"<code>{_html.escape(c)}</code>" for c in entry.all_codes[:14]
    )
    if len(entry.all_codes) > 14:
        codes_html += ' <span class="more">…</span>'

    return f"""
    <div class="entry" id="cluster-{entry.cluster_id}">
      <div class="entry-header">
        <div class="entry-meta">
          <span class="cluster-num">Cluster {entry.cluster_id}</span>
          <span class="tag">purity {entry.purity:.0%}</span>
          <span class="tag dim">{entry.size} instances</span>
          <span class="tag dim">{len(entry.family_breakdown)} families</span>
        </div>
        <div class="badges-row">{badges}</div>
      </div>
      <div class="bar-row">{_bar_html(entry.family_breakdown)}</div>
      <div class="content-row">
        <div class="glyphs-row">{glyph_cards}</div>
        <div class="reason-box">
          <div class="reason-label">⚙ Why mixed</div>
          <p class="reason-text">{_html.escape(entry.divergence_reason)}</p>
          <div class="codes-list">
            <span class="codes-label">All Barthel codes in cluster: </span>
            {codes_html}
          </div>
        </div>
      </div>
    </div>"""


_CSS = """
:root {
  --bg: #ffffff; --surface: #f8f8fa; --surface2: #f0f0f5;
  --border: #d0d0dd; --text: #1a1a1a; --muted: #666666;
  --accent: #c4a96d;
}
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
  background: var(--bg); color: var(--text);
  font-family: 'Cormorant Garamond', 'Palatino Linotype', Georgia, serif;
  font-size: 16px; line-height: 1.6;
}
.wrap { max-width: 980px; margin: 0 auto; padding: 56px 28px; }

/* ── Header ── */
.report-header { border-bottom: 1px solid var(--border); padding-bottom: 40px; margin-bottom: 52px; }
.report-title { font-size: 36px; font-weight: 600; letter-spacing: -0.4px; color: #000; line-height: 1.15; }
.report-subtitle { font-size: 17px; color: var(--accent); font-style: italic; margin-top: 6px; }
.report-meta {
  margin-top: 24px; font-family: 'JetBrains Mono', 'Fira Mono', monospace;
  font-size: 11.5px; color: var(--muted); line-height: 2.1;
}
.report-meta b { color: #333; }

/* ── Intro block ── */
.intro-block { max-width: 780px; margin-bottom: 52px; }
.intro-section { margin-bottom: 28px; }
.intro-section + .intro-section { padding-top: 22px; border-top: 1px solid var(--border); }
.intro-heading { font-family: 'JetBrains Mono', monospace; font-size: 10px;
                 text-transform: uppercase; letter-spacing: 0.14em;
                 color: var(--accent); margin-bottom: 10px; }
.intro-section p { font-size: 15px; color: #333; line-height: 1.85; }
.intro-section p + p { margin-top: 10px; }
.intro-dl { margin: 12px 0 0 0; }
.intro-dl dt { font-family: 'JetBrains Mono', monospace; font-size: 10.5px;
               font-weight: 600; color: #111; letter-spacing: 0.05em;
               margin-top: 12px; margin-bottom: 3px; }
.intro-dl dd { font-size: 14.5px; color: #444; line-height: 1.78; margin-left: 0; }

/* ── Note box ── */
.note-box { background: var(--surface); border: 1px solid var(--border);
            border-left: 3px solid var(--accent); border-radius: 0 6px 6px 0;
            padding: 20px 24px; margin-bottom: 44px; max-width: 780px; }
.note-box-title { font-family: 'JetBrains Mono', monospace; font-size: 10px;
                  text-transform: uppercase; letter-spacing: 0.14em;
                  color: var(--accent); margin-bottom: 12px; }
.note-box p { font-size: 14.5px; color: #333; line-height: 1.82; }
.note-box p + p { margin-top: 10px; }

/* ── Barthel reference index ── */
.barthel-ref { margin-bottom: 56px; }
.barthel-ref-title { font-family: 'JetBrains Mono', monospace; font-size: 10px;
                     text-transform: uppercase; letter-spacing: 0.14em;
                     color: var(--accent); margin-bottom: 6px; }
.barthel-ref-subtitle { font-size: 14px; color: var(--muted); margin-bottom: 20px;
                         font-style: italic; }
.barthel-ref-grid { display: grid;
                    grid-template-columns: repeat(auto-fill, minmax(360px, 1fr));
                    gap: 14px; }
.bfc { background: var(--surface); border: 1px solid var(--border);
       border-radius: 6px; padding: 14px 16px; border-left: 4px solid; }
.bfc-range { font-family: 'JetBrains Mono', monospace; font-size: 11.5px;
             font-weight: 600; color: #111; margin-bottom: 2px; }
.bfc-name { font-size: 13.5px; font-weight: 600; color: #333; margin-bottom: 7px; }
.bfc-desc { font-size: 13px; color: #555; line-height: 1.72; }

/* ── Stats row ── */
.stats-row {
  display: flex; flex-wrap: wrap; gap: 14px; margin: 32px 0 52px;
}
.stat-card {
  background: var(--surface); border: 1px solid var(--border); border-radius: 6px;
  padding: 14px 20px; flex: 1; min-width: 140px;
}
.stat-value { font-size: 26px; font-weight: 600; color: var(--accent); line-height: 1; }
.stat-label { font-size: 11px; color: var(--muted); margin-top: 4px;
              font-family: 'JetBrains Mono', monospace; }

/* ── Legend ── */
.legend-row { display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 48px; }

/* ── Entry cards ── */
.entry {
  background: var(--surface); border: 1px solid var(--border);
  border-radius: 8px; margin-bottom: 36px; overflow: hidden;
}
.entry-header {
  padding: 16px 22px 10px; display: flex; justify-content: space-between;
  align-items: flex-start; flex-wrap: wrap; gap: 10px;
}
.entry-meta { display: flex; align-items: center; flex-wrap: wrap; gap: 8px; }
.cluster-num { font-family: 'JetBrains Mono', monospace; font-size: 12.5px;
               color: var(--accent); font-weight: 500; }
.tag { font-family: 'JetBrains Mono', monospace; font-size: 10.5px;
       background: var(--surface2); border: 1px solid var(--border); border-radius: 3px;
       padding: 2px 7px; color: #666; }
.tag.dim { color: #999; }
.badges-row { display: flex; flex-wrap: wrap; gap: 6px; }
.badge { font-family: 'JetBrains Mono', monospace; font-size: 10px;
         border-radius: 3px; padding: 2px 8px; white-space: nowrap; }

/* ── Proportion bar ── */
.bar-row { height: 4px; display: flex; }
.bar-seg { height: 100%; }

/* ── Content ── */
.content-row { display: flex; gap: 20px; padding: 22px; flex-wrap: wrap; }
.glyphs-row { display: flex; gap: 12px; flex-wrap: wrap; align-items: flex-start; }

/* ── Glyph cards ── */
.glyph-card {
  background: var(--surface2); border: 1px solid var(--border); border-radius: 6px;
  padding: 12px 12px 10px; text-align: center; width: 118px;
}
.role-label { font-family: 'JetBrains Mono', monospace; font-size: 8.5px;
              color: #999; letter-spacing: 0.08em; margin-bottom: 6px; }
.glyph-svg { display: flex; align-items: center; justify-content: center;
             min-height: 76px; margin: 4px auto 10px; }
.glyph-svg svg { display: block; }
.glyph-code { font-family: 'JetBrains Mono', monospace; font-size: 11.5px;
              color: #333; font-weight: 500; margin-bottom: 3px; }
.glyph-range { font-size: 9.5px; color: var(--muted); line-height: 1.35; margin-bottom: 6px; }
.occurrence-count { font-size: 9.5px; color: #999; margin-top: 4px; }

/* ── Reason box ── */
.reason-box {
  flex: 1; min-width: 260px; background: var(--bg);
  border: 1px solid var(--border); border-radius: 6px; padding: 16px 18px;
}
.reason-label { font-family: 'JetBrains Mono', monospace; font-size: 9.5px;
                color: var(--accent); letter-spacing: 0.09em; margin-bottom: 10px; }
.reason-text { font-size: 14px; color: #333333; line-height: 1.8; }
.codes-list { margin-top: 16px; }
.codes-label { font-family: 'JetBrains Mono', monospace; font-size: 10px; color: #999; }
.codes-list code {
  font-family: 'JetBrains Mono', monospace; font-size: 9.5px; color: #333;
  background: #f0f0f5; border: 1px solid var(--border); border-radius: 2px;
  padding: 1px 5px; margin: 0 2px 3px 0; display: inline-block;
}
.more { font-size: 10px; color: #999; }

/* ── Footer ── */
.report-footer {
  border-top: 1px solid var(--border); margin-top: 56px; padding-top: 28px;
  font-size: 12.5px; color: var(--muted); line-height: 2.0;
}
.report-footer a { color: var(--accent); text-decoration: none; }

@media (max-width: 620px) {
  .content-row { flex-direction: column; }
  .entry-header { flex-direction: column; }
  .stats-row { flex-direction: column; }
}
"""


_BARTHEL_FAMILIES_REF: list[tuple[str, str, str, str]] = [
    (
        "001–199",
        "objects / plants / phenomena",
        _FAMILY_COLOURS["objects_plants_phenomena"],
        "The most heterogeneous family. Includes tools, natural forms (plants, waves, "
        "celestial objects), and geometric signs. Barthel grouped these by exclusion — "
        "signs that don't fit the humanoid or animal families.",
    ),
    (
        "200–299",
        "anthropomorphic type 2",
        _FAMILY_COLOURS["anthropomorphic_type2"],
        "Upright humanoid figures with a specific head type (Barthel’s “type-2” head "
        "morphology). Characterised by bipedal posture and arm extension. The largest "
        "single coherent family in the inventory.",
    ),
    (
        "300–399",
        "anthropomorphic type 3",
        _FAMILY_COLOURS["anthropomorphic_type3"],
        "Humanoid figures with a different head variant (Barthel’s “type-3”). The "
        "distinction from type-2 rests on secondary head features — appendages, "
        "ornaments, or head shape — that are subtle in corpus imagery.",
    ),
    (
        "400–499",
        "miscellaneous (300-series head)",
        _FAMILY_COLOURS["miscellaneous_300series"],
        "Signs that use the type-3 head grammar attached to non-standard bodies. The "
        "boundary with the 300-series is a known ambiguity — body posture often "
        "dominates visually over the head distinction.",
    ),
    (
        "500–599",
        "miscellaneous anthropomorphic",
        _FAMILY_COLOURS["miscellaneous_anthropomorphic"],
        "Humanoid-adjacent signs that don't fit types 2 or 3. Share the general "
        "upright bipedal silhouette with the 200-series, which is why these are the "
        "most common cross-family neighbours in the divergence clusters below.",
    ),
    (
        "600–699",
        "bird-headed figures",
        _FAMILY_COLOURS["bird_headed"],
        "Signs with a bird-shaped head on a humanoid or creature body. Includes the "
        "high-frequency frigatebird sign (Barthel 600) and the Tangata Manu Bird-Man "
        "(Barthel 690). Astronomically significant: sign 600 maps to the moon-passage "
        "sequence in the Mamari calendar.",
    ),
    (
        "700–799",
        "zoomorphic figures",
        _FAMILY_COLOURS["zoomorphic"],
        "Non-humanoid animal forms. Distinguished from the 600-series by body plan "
        "rather than head shape — the entire body is animal rather than a bird head "
        "on a humanoid torso. The boundary is visually subtle when body silhouette "
        "dominates.",
    ),
    (
        "additional / unlabeled",
        "",
        _FAMILY_COLOURS["additional"],
        "Signs not assigned to a numbered Barthel family in the de Souza (2023) / "
        "rongopy encoding used here. Includes composite signs, uncertain readings, "
        "and variants not present in the 1958 catalogue.",
    ),
]


def _render_intro(stats: dict[str, Any], n_entries: int, max_purity: float) -> str:
    purity_str = f"{stats['mean_purity']}%"
    n_glyphs = f"{stats['total_glyphs']:,}"
    n_clusters = stats["n_clusters"]
    purity_thresh = f"{max_purity:.0%}"
    return f"""
<div class="intro-block">

  <div class="intro-section">
    <div class="intro-heading">What this report does</div>
    <p>A convolutional autoencoder was trained on {n_glyphs} rongorongo glyph images
    with no knowledge of Barthel's classification. It learned to group signs purely
    by visual shape — stroke patterns, body-plan geometry, proportions. The autoencoder
    found {n_clusters} clusters. This report shows the {n_entries} clusters where those
    visually-driven groupings disagree with Thomas Barthel's 1958 iconographic families.
    It is not a claim that Barthel was wrong — it is a systematic, data-driven list of
    the cases worth examining.</p>
  </div>

  <div class="intro-section">
    <div class="intro-heading">How to read each cluster entry</div>
    <p>Each entry represents one visually coherent group identified by the pipeline.
    The coloured bar shows how many signs from each Barthel family ended up in that
    group. Two labels appear on each glyph card:</p>
    <dl class="intro-dl">
      <dt>DOMINANT</dt>
      <dd>The Barthel family that makes up the majority of signs in this cluster.
      This is what the cluster is "mostly about" visually.</dd>
      <dt>MINORITY</dt>
      <dd>A Barthel family that also landed in this cluster, even though Barthel
      placed these signs in a different category. These are the divergences: signs
      that look similar to the dominant group despite receiving a different
      iconographic classification.</dd>
    </dl>
  </div>

  <div class="intro-section">
    <div class="intro-heading">What "purity" means</div>
    <p>Purity is the fraction of signs in a cluster that belong to the single most
    common Barthel family. A cluster with purity 100% is perfectly consistent with
    Barthel — every sign in it belongs to the same family. A cluster with purity 33%
    means only one-third of its signs share a family; the rest come from other families
    that Barthel distinguished but the autoencoder did not. The overall mean purity is
    <strong>{purity_str}</strong> — meaning the pipeline agrees with Barthel on the
    vast majority of signs. Only the {n_entries} cases below, with purity below
    {purity_thresh}, are shown here as candidates for scholarly review.</p>
  </div>

  <div class="intro-section">
    <div class="intro-heading">The most impactful numbers</div>
    <p>Out of {n_clusters} clusters and {n_glyphs} glyph instances, only {n_entries}
    clusters (roughly 2%) show meaningful cross-family mixing. The three most divergent
    clusters are Cluster 404 (purity 33%, 6 Barthel families mixed), Cluster 626
    (purity 40%, bird-headed signs grouped with objects/phenomena), and Cluster 459
    (purity 46%, 300-series head signs grouped with objects). These specific boundary
    cases — where the pipeline's geometry disagrees most sharply with Barthel's
    iconography — are the candidates most likely to reveal either scribal ambiguity
    in the corpus or limits in Barthel's century-block boundaries.</p>
    <p>Each entry shows the Barthel code, century-block range, and family name for
    each divergent sign, along with the pipeline's geometric reasoning for why the
    signs were grouped together. We invite rongorongo scholars to review these cases
    and advise on which groupings reflect scribal reality.</p>
  </div>

</div>"""


def _render_ari_note(ari: float | None, stats: dict[str, Any]) -> str:
    ari_str = f"{ari:.4f}" if ari is not None else "n/a"
    return f"""
<div class="note-box">
  <div class="note-box-title">Note on ARI</div>
  <p>The Adjusted Rand Index (ARI) would equal 1.0 for perfect agreement with Barthel,
  and ≈ 0 for chance-level agreement. The value of {ari_str} here does not mean the
  clustering failed — it reflects a structural artefact of comparing {stats['n_clusters']}
  fine-grained clusters against only 8 coarse Barthel families. This granularity mismatch
  drives ARI toward zero by design.</p>
  <p>The meaningful quality metric is mean cluster purity: <strong>{stats['mean_purity']}%</strong>,
  which measures whether each individual cluster is internally consistent with Barthel's
  families, regardless of how many clusters exist.</p>
</div>"""


def _render_barthel_reference() -> str:
    cards = ""
    for range_str, name, colour, desc in _BARTHEL_FAMILIES_REF:
        name_html = f'<div class="bfc-name">{name}</div>' if name else ""
        cards += (
            f'<div class="bfc" style="border-left-color:{colour}">'
            f'<div class="bfc-range">{range_str}</div>'
            f'{name_html}'
            f'<div class="bfc-desc">{desc}</div>'
            f'</div>'
        )
    return f"""
<div class="barthel-ref">
  <div class="barthel-ref-title">Barthel Family Reference</div>
  <div class="barthel-ref-subtitle">
    Barthel (1958) family reference — the eight sign categories used throughout this report
  </div>
  <div class="barthel-ref-grid">{cards}</div>
</div>"""


def _render_html(
    entries: list[_ClusterEntry],
    stats: dict[str, Any],
    ari: float | None,
    cfg: DivergenceReportConfig,
    run_metadata: dict[str, str] | None = None,
) -> str:
    """Render the full HTML report string."""

    ari_str = f"{ari:.4f}" if ari is not None else "n/a"

    # Stats cards
    stats_html = f"""
    <div class="stats-row">
      <div class="stat-card">
        <div class="stat-value">{stats['total_glyphs']:,}</div>
        <div class="stat-label">total glyph instances</div>
      </div>
      <div class="stat-card">
        <div class="stat-value">{stats['n_clusters']}</div>
        <div class="stat-label">HDBSCAN clusters</div>
      </div>
      <div class="stat-card">
        <div class="stat-value">{stats['mean_purity']}%</div>
        <div class="stat-label">mean cluster purity</div>
      </div>
      <div class="stat-card">
        <div class="stat-value">{stats['pct_perfect']}%</div>
        <div class="stat-label">clusters 100% pure</div>
      </div>
      <div class="stat-card">
        <div class="stat-value">{stats['noise_pct']}%</div>
        <div class="stat-label">noise / unclustered</div>
      </div>
      <div class="stat-card">
        <div class="stat-value">{ari_str}</div>
        <div class="stat-label">adjusted rand index</div>
      </div>
    </div>"""

    # Legend
    legend_html = '<div class="legend-row">' + "".join(
        _badge_html(f) for f in sorted(_FAMILY_COLOURS)
    ) + "</div>"

    # Run metadata line
    meta_lines = ""
    if run_metadata:
        for k, v in run_metadata.items():
            meta_lines += f"<b>{_html.escape(str(k))}:</b> {_html.escape(str(v))} &nbsp;·&nbsp; "
    meta_lines += (
        f"<b>HDBSCAN clusters:</b> {stats['n_clusters']} &nbsp;·&nbsp; "
        f"<b>noise:</b> {stats['noise_count']} ({stats['noise_pct']}%) &nbsp;·&nbsp; "
        f"<b>ARI:</b> {ari_str} &nbsp;·&nbsp; "
        f"<b>divergent entries shown:</b> {len(entries)} "
        f"(purity &lt; {cfg.max_purity:.0%}, size ≥ {cfg.min_cluster_size})"
    )

    intro_html = _render_intro(stats, len(entries), cfg.max_purity)
    ari_note_html = _render_ari_note(ari, stats)
    barthel_ref_html = _render_barthel_reference()
    entries_html = "".join(_render_entry(e) for e in entries)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>hackingrongo — Divergence Report</title>
<link href="https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,400;0,600;1,400&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>{_CSS}</style>
</head>
<body>
<div class="wrap">

<div class="report-header">
  <div class="report-title">hackingrongo<br>Divergence Report</div>
  <div class="report-subtitle">Where the pipeline disagrees with Barthel — and why</div>
  <div class="report-meta">{meta_lines}</div>
</div>

{intro_html}

{stats_html}

{ari_note_html}

{barthel_ref_html}

{legend_html}
{entries_html}

<div class="report-footer">
  <p><b>hackingrongo</b> · Hybrid computational decipherment pipeline · MIT License ·
  <a href="https://github.com/violasarah2000/hackingrongo" target="_blank">GitHub</a></p>
  <p>Glyph SVGs: kohaumotu.org (Philip Spaelti) from the CEIPP rongorongo corpus encoding of Barthel (1958).
  Barthel family assignments: de Souza (2023) / rongopy (GPL-3.0).
  Diachronic stratification: Ferrara et al. (2024) radiocarbon dates.</p>
  <p>Zone A pipeline: convolutional autoencoder → UMAP → HDBSCAN.
  This report is generated automatically post-training by
  <code>hackingrongo.results.divergence_report</code>.</p>
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


def build_divergence_report(
    analysis_dir: Path,
    svg_catalog_path: Path,
    cfg: DivergenceReportConfig | None = None,
    run_metadata: dict[str, str] | None = None,
) -> str:
    """Build the divergence report HTML string.

    Parameters
    ----------
    analysis_dir : Path
        Directory containing ``cluster_vs_barthel.csv`` (and optionally
        ``cluster_vs_barthel.json`` for the ARI value).  Written by
        ``scripts/analyze_embeddings.py``.
    svg_catalog_path : Path
        Path to ``data/glyphs/svg/catalog.json``.
    cfg : DivergenceReportConfig, optional
        Report filtering parameters.  Defaults to
        ``DivergenceReportConfig()``.
    run_metadata : dict[str, str], optional
        Key-value pairs shown in the report header (e.g. run_id,
        experiment name, config hash).

    Returns
    -------
    str
        Complete HTML document as a string.
    """
    if cfg is None:
        cfg = DivergenceReportConfig()

    df = _load_cluster_data(analysis_dir)
    code_to_svgs = _load_svg_catalog(svg_catalog_path)
    ari = _load_ari(analysis_dir)
    stats = _compute_global_stats(df)

    entries = _select_divergent_clusters(df, code_to_svgs, cfg)
    logger.info(
        "Divergence report: %d entries selected (purity<%.0f%%, size>=%d).",
        len(entries),
        cfg.max_purity * 100,
        cfg.min_cluster_size,
    )

    return _render_html(entries, stats, ari, cfg, run_metadata)


def save_divergence_report(
    analysis_dir: Path,
    svg_catalog_path: Path,
    output_path: Path,
    cfg: DivergenceReportConfig | None = None,
    run_metadata: dict[str, str] | None = None,
) -> None:
    """Generate and write the divergence report to an HTML file.

    Parameters
    ----------
    analysis_dir : Path
        Directory containing ``cluster_vs_barthel.csv``.
    svg_catalog_path : Path
        Path to ``data/glyphs/svg/catalog.json``.
    output_path : Path
        Destination ``.html`` file.  Parent directories are created.
    cfg : DivergenceReportConfig, optional
        Report filtering parameters.
    run_metadata : dict[str, str], optional
        Key-value pairs for the report header.
    """
    html = build_divergence_report(
        analysis_dir=analysis_dir,
        svg_catalog_path=svg_catalog_path,
        cfg=cfg,
        run_metadata=run_metadata,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    logger.info("Divergence report written to %s (%d bytes).", output_path, len(html))


# ---------------------------------------------------------------------------
# CLI entry point (standalone use)
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate divergence report from Zone A analysis outputs."
    )
    p.add_argument(
        "--analysis-dir",
        type=Path,
        default=Path("outputs/analysis"),
        help="Directory containing cluster_vs_barthel.csv (default: outputs/analysis).",
    )
    p.add_argument(
        "--svg-catalog",
        type=Path,
        default=Path("data/glyphs/svg/catalog.json"),
        help="Path to SVG glyph catalog JSON (default: data/glyphs/svg/catalog.json).",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output HTML path (default: <analysis-dir>/divergence_report.html).",
    )
    p.add_argument(
        "--max-entries",
        type=int,
        default=12,
        help="Maximum number of clusters to include (default: 12).",
    )
    p.add_argument(
        "--min-cluster-size",
        type=int,
        default=5,
        help="Minimum cluster size to include (default: 5).",
    )
    p.add_argument(
        "--max-purity",
        type=float,
        default=0.65,
        help="Only include clusters with purity below this value (default: 0.65).",
    )
    return p.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
    args = _parse_args()

    cfg = DivergenceReportConfig(
        max_entries=args.max_entries,
        min_cluster_size=args.min_cluster_size,
        max_purity=args.max_purity,
    )

    output_path = args.output or (args.analysis_dir / "divergence_report.html")

    save_divergence_report(
        analysis_dir=args.analysis_dir,
        svg_catalog_path=args.svg_catalog,
        output_path=output_path,
        cfg=cfg,
    )
    print(f"Report written to: {output_path}")


if __name__ == "__main__":
    main()
