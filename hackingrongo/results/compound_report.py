"""
hackingrongo.results.compound_report
======================================

Generates an HTML report of compound glyph candidates for scholar review.

Each entry shows:
  - The candidate glyph drawing (SVG, from the actual corpus instance)
  - The proposed constituent component drawings where available
  - Confidence score and which methods agreed
  - Tablet provenance (name, side, line, position, temporal stratum)
  - Human-readable reasoning explaining why the pipeline flagged this sign

Inputs
------
  outputs/analysis/compound_candidates.json   — from compound_detector.py
  data/glyphs/svg/catalog.json                — SVG instance catalog
  data/glyphs/svg/                            — SVG files

Output
------
  outputs/analysis/compound_report.html

Public API
----------
``build_compound_report``   → HTML string
``save_compound_report``    → writes HTML file

CLI
---
    python -m hackingrongo.results.compound_report \\
        --candidates outputs/analysis/compound_candidates.json \\
        --svg-catalog data/glyphs/svg/catalog.json \\
        --output outputs/analysis/compound_report.html
"""

from __future__ import annotations

import argparse
import base64
import json
import logging
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tablet / stratum metadata
# ---------------------------------------------------------------------------

TABLET_NAMES: dict[str, str] = {
    "A": "Tahua", "B": "Aruku-Kurenga", "C": "Mamari",
    "D": "Échancrée", "E": "Keiti", "F": "Stephen-Chauvet Fragment",
    "G": "Small Santiago", "H": "Great Santiago", "I": "Santiago Staff",
    "J": "Reimiro 1", "K": "Small London", "L": "Reimiro 2",
    "M": "Great Vienna", "N": "Small Vienna", "O": "Boomerang",
    "P": "Great St. Petersburg", "Q": "Small St. Petersburg",
    "R": "Atua-Mata-Riri", "S": "Great Washington",
    "T": "Honolulu 1", "U": "Honolulu 2", "V": "Honolulu 3",
    "W": "Honolulu 4", "X": "Tangata Manu", "Y": "Snuff Box",
}

STRATUM_LABELS: dict[str, str] = {
    "pre_contact":  "pre-contact  (~1493–1509 CE)",
    "post_contact": "post-contact (~1800–1870 CE)",
    "mixed":        "mixed strata",
    "unknown":      "stratum unknown",
}

# ---------------------------------------------------------------------------
# SVG loading — same pattern as divergence_report.py
# ---------------------------------------------------------------------------


def _load_svg_catalog(catalog_path: Path) -> dict[str, list[Path]]:
    """Return {barthel_code: [image_path, ...]} for all available glyph images.

    Lookup order per code:
    1. Exact SVG match from svg/catalog.json
    2. Base SVG match (trailing ``!?()`` stripped) from svg/catalog.json
    3. PNG fallback from barthel_catalog.json (corpus scan preferred over
       Barthel reference scan) — covers the ~920 codes present in tablets
       that were not SVG-scraped (A, C, E, G, H, J, K, L, N, P, Q, R, S, T, U, Y).
    """
    if not catalog_path.exists():
        logger.warning("SVG catalog not found: %s", catalog_path)
        return {}

    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    svg_dir = catalog_path.parent

    exact: dict[str, list[Path]] = defaultdict(list)
    base_map: dict[str, list[Path]] = defaultdict(list)

    for r in catalog.get("records", []):
        code = str(r.get("barthel_code", "")).strip()
        if not code:
            continue
        rel  = str(r.get("svg_path", "")).replace("svg/", "", 1)
        full = svg_dir / rel
        if not full.exists():
            continue
        exact[code].append(full)
        base = re.sub(r'[!?()\s]+$', '', code).strip()
        if base != code:
            base_map[base].append(full)
        numeric_base = re.sub(r'[a-zA-Z!?()\s].*$', '', code).strip()
        if numeric_base and numeric_base != code and numeric_base != base:
            base_map[numeric_base].append(full)

    merged: dict[str, list[Path]] = dict(exact)
    for base_code, paths in base_map.items():
        if base_code not in merged:
            merged[base_code] = paths

    # PNG fallback: load barthel_catalog.json, which covers all 25 tablets.
    # Prefer corpus-scan sources (barthel_tafeln) over reference scans
    # (barthel_formentafeln) by processing reference entries first so corpus
    # entries overwrite them in the staging dict.
    bc_path = catalog_path.parent.parent / "barthel_catalog.json"
    if bc_path.exists():
        glyph_dir = catalog_path.parent.parent
        bc_records = json.loads(bc_path.read_text(encoding="utf-8")).get("records", [])
        # Stage PNGs: code → best path.  Corpus scan beats reference scan.
        png_stage: dict[str, Path] = {}
        for source_pref in ("barthel_formentafeln", "barthel_tafeln"):
            for r in bc_records:
                if r.get("source") != source_pref:
                    continue
                code = str(r.get("barthel_code") or "").strip()
                png_rel = r.get("path", "")
                if not code or not png_rel or not png_rel.endswith(".png"):
                    continue
                png_full = glyph_dir / png_rel
                if png_full.exists():
                    png_stage[code] = png_full
        # Only fill codes (and their numeric bases) that have no SVG
        for code, png_path in png_stage.items():
            if code not in merged:
                merged[code] = [png_path]
            numeric_base = re.sub(r'[a-zA-Z!?()\s].*$', '', code).strip()
            if numeric_base and numeric_base != code and numeric_base not in merged:
                merged[numeric_base] = [png_path]
        logger.info(
            "barthel_catalog fallback: %d PNG codes added (%d total SVG codes).",
            sum(1 for c in png_stage if c not in exact),
            len(exact),
        )
    else:
        logger.warning(
            "barthel_catalog.json not found at %s — PNG fallback disabled.", bc_path
        )

    return merged


def _normalise_svg(svg_text: str, size: int = 88) -> str:
    """Resize SVG and prepare for dark-background rendering."""
    svg = svg_text.strip()
    svg = re.sub(r'width="[^"]*"',  f'width="{size}"',  svg)
    svg = re.sub(r'height="[^"]*"', f'height="{size}"', svg)
    svg = re.sub(
        r"<path ",
        '<path fill="none" stroke="currentColor" stroke-width="1.5" '
        'stroke-linecap="round" stroke-linejoin="round" ',
        svg,
    )
    return svg


def _best_instance_svg(
    code: str,
    catalog: dict[str, list[Path]],
    size: int = 88,
) -> str | None:
    """Return an HTML fragment (inline SVG or base64 img) for the first available instance.

    Tries exact code first, then base code (trailing ``!?()`` stripped).
    SVG files are returned as normalised inline SVG strings.
    PNG files (barthel_catalog fallbacks) are returned as self-contained
    ``<img>`` tags with base64-encoded data URIs so the report stays portable.
    """
    instances = catalog.get(code, [])
    if not instances:
        base = re.sub(r'[!?()\s]+$', '', code).strip()
        instances = catalog.get(base, [])
    if not instances:
        numeric_base = re.sub(r'[a-zA-Z!?()\s].*$', '', code).strip()
        if numeric_base and numeric_base != code:
            instances = catalog.get(numeric_base, [])
    if not instances:
        return None
    path = instances[0]
    try:
        if path.suffix.lower() == ".png":
            b64 = base64.b64encode(path.read_bytes()).decode()
            # PNG glyphs are dark-on-light scans; display at requested size,
            # centred, with a subtle invert so they match the SVG rendering style.
            return (
                f'<img src="data:image/png;base64,{b64}" '
                f'style="max-width:{size}px;max-height:{size}px;'
                f'display:block;margin:auto;filter:invert(0)" '
                f'alt="Barthel {code}">'
            )
        return _normalise_svg(path.read_text(encoding="utf-8"), size=size)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Corpus provenance index
# ---------------------------------------------------------------------------


def _build_provenance_index(corpus_dir: Path) -> dict[str, list[dict[str, Any]]]:
    """Return {barthel_code: [{tablet, side, line, glyph_num}, ...]}."""
    index: dict[str, list[dict[str, Any]]] = defaultdict(list)
    if not corpus_dir.exists():
        return {}
    for jf in sorted(corpus_dir.glob("*.json")):
        tablet_id = jf.stem
        try:
            data = json.loads(jf.read_text(encoding="utf-8"))
        except Exception:
            continue
        for g in data.get("glyphs", []):
            code = str(g.get("barthel_code", "")).strip()
            if not code:
                continue
            index[code].append({
                "tablet": tablet_id,
                "side":   g.get("side", "?"),
                "line":   g.get("line", "?"),
                "glyph_num": g.get("glyph_num", "?"),
            })
    return dict(index)


def _provenance_html(
    code: str,
    provenance: dict[str, list[dict[str, Any]]],
) -> str:
    """Build a compact HTML provenance string for a sign."""
    instances = provenance.get(code, [])
    if not instances:
        return "<span class='muted'>Not found in corpus.</span>"

    by_tablet: dict[str, list[str]] = defaultdict(list)
    for inst in instances:
        tab  = inst["tablet"]
        side = inst.get("side", "?")
        line = inst.get("line", "?")
        gn   = inst.get("glyph_num", "?")
        by_tablet[tab].append(f"{side}{line}·{gn}")

    parts = []
    for tab in sorted(by_tablet.keys()):
        name = TABLET_NAMES.get(tab, tab)
        locs = ", ".join(by_tablet[tab][:4])
        more = f" +{len(by_tablet[tab]) - 4} more" if len(by_tablet[tab]) > 4 else ""
        parts.append(f"<b>{tab}</b> {name} — {locs}{more}")
    return "<br>".join(parts)


# ---------------------------------------------------------------------------
# Confidence tier
# ---------------------------------------------------------------------------


def _confidence_tier(score: float, n_methods: int) -> tuple[str, str]:
    """(label, hex_colour) for the confidence tier chip."""
    if n_methods == 3:
        return "HIGH — all 3 methods", "#4caf7d"
    if score >= 0.6:
        return "MEDIUM-HIGH", "#8bc34a"
    if score >= 0.4:
        return "MEDIUM", "#d4a817"
    return "LOW", "#e07b54"


# ---------------------------------------------------------------------------
# Method reasoning
# ---------------------------------------------------------------------------


def _reasoning_html(candidate: dict[str, Any]) -> str:
    """Build human-readable per-method reasoning paragraphs."""
    evidence = candidate.get("method_evidence", [])
    paras: list[str] = []

    for ev in evidence:
        method = ev.get("method", "")
        conf   = float(ev.get("confidence", 0.0))
        comps  = ev.get("proposed_components", [])
        det    = ev.get("details", {})
        c1, c2 = (comps + ["?", "?"])[:2]

        if method == "embedding_geometry":
            d_mid   = det.get("dist_to_midpoint", "?")
            d_inter = det.get("interpoint_dist", "?")
            paras.append(
                f"<b>Embedding geometry</b> (confidence {conf:.2f}): "
                f"In the UMAP projection of Zone A autoencoder embeddings, "
                f"this sign's centroid sits near the midpoint between signs "
                f"<code>{c1}</code> and <code>{c2}</code> "
                f"(distance to midpoint: {float(d_mid):.3f} UMAP units; "
                f"component separation: {float(d_inter):.3f} units). "
                f"A compound of two signs is geometrically expected to embed "
                f"between its constituents, since the autoencoder encodes visual "
                f"structure rather than iconographic category."
            )

        elif method == "cluster_anomaly":
            noise_frac  = det.get("noise_fraction", 0.0)
            noise_prior = det.get("noise_prior", 0.0)
            n_clust     = det.get("n_unique_neighbour_clusters", "?")
            n_inst      = det.get("n_instances", "?")
            paras.append(
                f"<b>Cluster membership anomaly</b> (confidence {conf:.2f}): "
                f"{round(float(noise_frac) * 100)}% of this sign's "
                f"{n_inst} corpus instances fall in the HDBSCAN noise bucket "
                f"(corpus baseline: {round(float(noise_prior) * 100)}%). "
                f"Its nearest neighbours in UMAP space belong to "
                f"{n_clust} distinct high-purity clusters, including signs "
                f"<code>{c1}</code> and <code>{c2}</code>. "
                f"Compound signs are expected to be noise-classified because "
                f"they don't belong to either constituent's cluster."
            )

        elif method == "positional_profile":
            z          = float(det.get("mean_abs_z_score", 0.0))
            freq       = det.get("corpus_frequency", "?")
            frac_post  = det.get("feat_frac_post_taxogram")
            frac_final = det.get("feat_frac_seq_final")
            detail_str = ""
            if frac_post is not None and frac_final is not None:
                detail_str = (
                    f" It follows the taxogram (sign 200) in "
                    f"{round(float(frac_post) * 100)}% of occurrences and appears "
                    f"sequence-final in {round(float(frac_final) * 100)}% of lines — "
                    f"both characteristic of known compound positions."
                )
            paras.append(
                f"<b>Positional profile</b> (confidence {conf:.2f}): "
                f"Across its {freq} corpus occurrences, this sign's "
                f"sequence-position distribution resembles that of known "
                f"compound signs (mean absolute z-score vs compound profile: "
                f"{z:.2f} — lower = more similar).{detail_str} "
                f"Note: this method cannot resolve constituent components."
            )

    if not paras:
        return "<p class='reason-text muted'>No method evidence available.</p>"
    return "".join(f"<p class='reason-text'>{p}</p>" for p in paras)


# ---------------------------------------------------------------------------
# Component card strip
# ---------------------------------------------------------------------------


def _component_cards_html(
    components: list[str],
    catalog: dict[str, list[Path]],
) -> str:
    if not components:
        return (
            "<div class='no-components'>"
            "Component decomposition not resolved by these methods."
            "</div>"
        )
    cards = ""
    for comp in components[:3]:
        svg = _best_instance_svg(comp, catalog, size=56)
        svg_block = (
            f'<div class="comp-svg">{svg}</div>'
            if svg
            else '<div class="comp-svg comp-missing">no SVG</div>'
        )
        cards += (
            f'<div class="comp-card">'
            f'{svg_block}'
            f'<div class="comp-code">Barthel {comp}</div>'
            f'</div>'
        )
    return f'<div class="comp-row">{cards}</div>'


# ---------------------------------------------------------------------------
# Single entry card
# ---------------------------------------------------------------------------


def _render_entry(
    rank: int,
    candidate: dict[str, Any],
    catalog: dict[str, list[Path]],
    provenance: dict[str, list[dict[str, Any]]],
) -> str:
    code       = candidate["barthel_code"]
    conf       = float(candidate["consensus_confidence"])
    n_methods  = int(candidate["n_methods_agreeing"])
    components = candidate.get("consensus_components", [])
    freq       = candidate.get("corpus_frequency", "?")
    stratum    = candidate.get("temporal_cluster", "unknown")
    is_icono   = candidate.get("is_iconographic_compound", False)

    tier_label, tier_colour = _confidence_tier(conf, n_methods)
    stratum_label = STRATUM_LABELS.get(stratum, stratum)

    main_svg = _best_instance_svg(code, catalog, size=96)
    main_svg_block = (
        f'<div class="main-svg">{main_svg}</div>'
        if main_svg
        else '<div class="main-svg main-missing">no SVG</div>'
    )

    filled   = round(conf * 20)
    conf_bar = "█" * filled + "░" * (20 - filled)

    icono_note = (
        "<div class='icono-note'>⚠ This sign falls in Barthel's iconographic "
        "compound range (600–799: bird-headed / zoomorphic). Its detection as a "
        "syntactic compound warrants expert disambiguation.</div>"
        if is_icono else ""
    )

    comp_header = (
        "Proposed components: "
        + "".join(f"<code>{c}</code> + " for c in components).rstrip(" + ")
        if components else "Components: not resolved"
    )

    return f"""
    <div class="entry" id="rank-{rank}">

      <div class="entry-header">
        <div class="rank-badge">#{rank}</div>
        <div class="entry-title">
          <span class="code-label">Barthel {code}</span>
          <span class="tier-tag"
                style="background:{tier_colour}22;color:{tier_colour};
                       border:1px solid {tier_colour}55">{tier_label}</span>
        </div>
        <div class="conf-block">
          <span class="conf-value">{conf:.3f}</span>
          <span class="conf-bar">{conf_bar}</span>
          <span class="methods-tag">{n_methods}/3 methods</span>
        </div>
      </div>

      <div class="bar-accent" style="background:linear-gradient(90deg,
           {tier_colour} {conf*100:.0f}%, var(--border) {conf*100:.0f}%)"></div>

      <div class="entry-body">

        <!-- Left: glyph drawings -->
        <div class="glyph-col">
          <div class="section-label">Candidate glyph</div>
          {main_svg_block}
          <div class="glyph-meta">
            <div class="glyph-code">Barthel {code}</div>
            <div class="glyph-freq">{freq} corpus occurrences</div>
            <div class="glyph-stratum">{stratum_label}</div>
          </div>

          <div class="section-label" style="margin-top:18px">{comp_header}</div>
          {_component_cards_html(components, catalog)}
          {icono_note}
        </div>

        <!-- Middle: provenance -->
        <div class="provenance-col">
          <div class="section-label">Tablet provenance</div>
          <div class="provenance-text">{_provenance_html(code, provenance)}</div>
        </div>

        <!-- Right: reasoning -->
        <div class="reasoning-col">
          <div class="section-label">Detection reasoning</div>
          {_reasoning_html(candidate)}
        </div>

      </div>
    </div>"""


# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------

_CSS = """\
:root {
  --bg: #ffffff; --surface: #f8f8fa; --surface2: #f0f0f5;
  --border: #d0d0dd; --text: #1a1a1a; --muted: #666666;
  --accent: #c4a96d; --accent2: #7b9ee0;
}
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
  background: var(--bg); color: var(--text);
  font-family: 'Cormorant Garamond', 'Palatino Linotype', Georgia, serif;
  font-size: 16px; line-height: 1.65;
}
.wrap { max-width: 1100px; margin: 0 auto; padding: 52px 28px; }

/* ── Header ── */
.report-header { border-bottom: 1px solid var(--border);
                 padding-bottom: 38px; margin-bottom: 48px; }
.report-title { font-size: 34px; font-weight: 600; color: #000; letter-spacing: -0.3px; }
.report-subtitle { font-size: 17px; color: var(--accent); font-style: italic; margin-top: 6px; }
.report-meta { margin-top: 22px; font-family: 'JetBrains Mono', monospace;
               font-size: 11px; color: var(--muted); line-height: 2.2; }
.report-meta b { color: #333; }
.abstract { margin-top: 22px; font-size: 14.5px; color: #333333;
            max-width: 760px; line-height: 1.85; }
.abstract p + p { margin-top: 10px; }

/* ── Confidence legend ── */
.legend { display: flex; flex-wrap: wrap; gap: 10px;
          margin: 28px 0 44px; align-items: center; }
.legend-label { font-size: 12px; color: var(--muted); margin-right: 4px; }
.legend-chip { font-family: 'JetBrains Mono', monospace; font-size: 10px;
               border-radius: 3px; padding: 3px 9px; }

/* ── Entry ── */
.entry { background: var(--surface); border: 1px solid var(--border);
         border-radius: 8px; margin-bottom: 32px; overflow: hidden; }
.entry-header { padding: 16px 22px 10px; display: flex; align-items: center;
                gap: 14px; flex-wrap: wrap; }
.rank-badge { font-family: 'JetBrains Mono', monospace; font-size: 11px;
              color: var(--muted); min-width: 28px; }
.entry-title { display: flex; align-items: center; gap: 10px; flex: 1; }
.code-label { font-family: 'JetBrains Mono', monospace; font-size: 14px;
              color: var(--accent); font-weight: 500; }
.tier-tag { font-family: 'JetBrains Mono', monospace; font-size: 9.5px;
            border-radius: 3px; padding: 2px 8px; white-space: nowrap; }
.conf-block { display: flex; align-items: center; gap: 10px; }
.conf-value { font-family: 'JetBrains Mono', monospace; font-size: 18px;
              color: #000; font-weight: 500; min-width: 48px; }
.conf-bar { font-family: 'JetBrains Mono', monospace; font-size: 10px;
            color: var(--accent); letter-spacing: -1px; }
.methods-tag { font-family: 'JetBrains Mono', monospace; font-size: 9.5px;
               color: var(--muted); }
.bar-accent { height: 3px; }

/* ── Entry body ── */
.entry-body { display: grid; grid-template-columns: 200px 220px 1fr; }
.glyph-col, .provenance-col, .reasoning-col {
  padding: 20px 22px; border-right: 1px solid var(--border);
}
.reasoning-col { border-right: none; }
.section-label { font-family: 'JetBrains Mono', monospace; font-size: 9px;
                 color: var(--muted); letter-spacing: 0.1em;
                 text-transform: uppercase; margin-bottom: 10px; }
.main-svg { display: flex; align-items: center; justify-content: center;
            background: var(--surface2); border: 1px solid var(--border);
            border-radius: 5px; padding: 10px; min-height: 108px;
            color: var(--accent); }
.main-missing { color: var(--muted); font-size: 11px; }
.glyph-meta { margin-top: 10px; }
.glyph-code { font-family: 'JetBrains Mono', monospace; font-size: 11px; color: #333; }
.glyph-freq { font-size: 11px; color: var(--muted); margin-top: 2px; }
.glyph-stratum { font-size: 10.5px; color: var(--accent2); margin-top: 2px; }

/* ── Components ── */
.comp-row { display: flex; gap: 8px; flex-wrap: wrap; margin-top: 6px; }
.comp-card { background: var(--surface2); border: 1px solid var(--border);
             border-radius: 4px; padding: 8px; text-align: center; min-width: 72px; }
.comp-svg { display: flex; align-items: center; justify-content: center;
            min-height: 56px; color: #7bc4a0; }
.comp-missing { font-size: 9px; color: var(--muted); }
.comp-code { font-family: 'JetBrains Mono', monospace; font-size: 9px;
             color: var(--muted); margin-top: 4px; }
.no-components { font-size: 11px; color: var(--muted); font-style: italic; margin-top: 6px; }
.icono-note { margin-top: 10px; font-size: 10.5px; color: #b8860b;
              background: #d4a81720; border: 1px solid #d4a81740;
              border-radius: 4px; padding: 7px 10px; line-height: 1.5; }

/* ── Provenance ── */
.provenance-text { font-size: 12px; color: #333333; line-height: 2.0; }
.provenance-text b { color: var(--accent); }

/* ── Reasoning ── */
.reason-text { font-size: 13.5px; color: #333333; line-height: 1.85; margin-bottom: 12px; }
.reason-text b { color: #000; }
.reason-text code { font-family: 'JetBrains Mono', monospace; font-size: 11px;
                    background: var(--surface2); border: 1px solid var(--border);
                    border-radius: 2px; padding: 1px 5px; }
.reason-text.muted { color: var(--muted); font-style: italic; }
.muted { color: var(--muted); }

/* ── Footer ── */
.report-footer { border-top: 1px solid var(--border); margin-top: 52px;
                 padding-top: 26px; font-size: 12px; color: var(--muted); line-height: 2.0; }
.report-footer a { color: var(--accent); text-decoration: none; }
.report-footer code { background: var(--surface2); border: 1px solid var(--border); border-radius: 2px; padding: 1px 5px; }

@media (max-width: 820px) {
  .entry-body { grid-template-columns: 1fr; }
  .glyph-col, .provenance-col {
    border-right: none; border-bottom: 1px solid var(--border);
  }
}
"""

# ---------------------------------------------------------------------------
# Full HTML document
# ---------------------------------------------------------------------------


def _render_html(
    candidates: list[dict[str, Any]],
    catalog: dict[str, list[Path]],
    provenance: dict[str, list[dict[str, Any]]],
    report_meta: dict[str, Any],
) -> str:
    n_total   = report_meta.get("n_candidates", len(candidates))
    n_all3    = report_meta.get("n_all_methods", 0)
    n_two     = report_meta.get("n_two_methods", 0)
    generated = report_meta.get("generated", "—")

    entries_html = "".join(
        _render_entry(rank, cand, catalog, provenance)
        for rank, cand in enumerate(candidates, start=1)
    )

    legend_chips = "".join(
        f'<span class="legend-chip" '
        f'style="background:{colour}22;color:{colour};border:1px solid {colour}55">'
        f'{label}</span>'
        for label, colour in [
            ("HIGH — all 3 methods", "#4caf7d"),
            ("MEDIUM-HIGH ≥ 0.6", "#8bc34a"),
            ("MEDIUM ≥ 0.4", "#d4a817"),
            ("LOW < 0.4", "#e07b54"),
        ]
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>hackingrongo — Compound Glyph Candidates</title>
<link href="https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,400;0,600;1,400&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>{_CSS}</style>
</head>
<body>
<div class="wrap">

<div class="report-header">
  <div class="report-title">hackingrongo<br>Compound Glyph Candidates</div>
  <div class="report-subtitle">Signs flagged as potential compound glyphs — for scholar review</div>
  <div class="report-meta">
    <b>Total candidates:</b> {n_total} &nbsp;·&nbsp;
    <b>All 3 methods:</b> {n_all3} &nbsp;·&nbsp;
    <b>2 methods:</b> {n_two} &nbsp;·&nbsp;
    <b>Generated:</b> {generated}
  </div>
  <div class="abstract">
    <p>Barthel (1958) explicitly marked compound glyphs using punctuation in the sign code
    (stacked <code>X:Y</code>, linked <code>X.Y</code>, juxtaposed <code>X-Y</code>,
    fused <code>X'Y</code>). This report lists additional signs that the hackingrongo
    Zone A pipeline flags as probable compounds — signs Barthel did not explicitly mark,
    but whose embedding geometry, cluster membership, and/or corpus positional behaviour
    are consistent with compound structure.</p>
    <p>Each entry shows the glyph drawing from the actual corpus, the proposed constituent
    components where resolvable, all tablet locations, and a plain-language explanation of
    the detection evidence. Candidates are ranked by consensus confidence across the three
    independent detection methods. <b>We invite rongorongo scholars to review these
    candidates and advise on which represent genuine scribal compounds.</b></p>
    <p><b>Precision disclosure:</b> P@k against Barthel's 47 explicitly marked compounds
    has not been independently evaluated; the consensus confidence score reflects
    internal model agreement across three detection methods, not a validated
    precision estimate, and every candidate requires expert epigraphic verification.</p>
  </div>
</div>

<div class="legend">
  <span class="legend-label">Confidence tier:</span>
  {legend_chips}
</div>

{entries_html}

<div class="report-footer">
  <p><b>hackingrongo</b> · Compound detection pipeline · MIT License ·
  <a href="https://github.com/violasarah2000/hackingrongo" target="_blank">GitHub</a></p>
  <p>Detection methods: (1) UMAP embedding midpoint geometry · (2) HDBSCAN cluster-boundary
  anomaly · (3) corpus positional profile similarity to known compounds.</p>
  <p>Glyph SVGs from Barthel (1958) corpus encoding via kohaumotu.org (Philip Spaelti) / CEIPP.
  Known compound ground truth: Barthel (1958) syntactic punctuation conventions.</p>
  <p>This is a computational hypothesis report, not a decipherment claim.
  All candidates require expert review.</p>
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


def build_compound_report(
    candidates_path: Path,
    svg_catalog_path: Path,
    corpus_dir: Path | None = None,
    max_candidates: int = 50,
) -> str:
    """Build the compound candidate report HTML.

    Parameters
    ----------
    candidates_path : Path
        ``compound_candidates.json`` written by ``compound_detector.py``.
    svg_catalog_path : Path
        ``data/glyphs/svg/catalog.json``.
    corpus_dir : Path, optional
        ``data/corpus/`` directory.  Used to build tablet provenance.
        If omitted, falls back to sibling ``data/corpus/`` relative to
        ``candidates_path.parent.parent.parent``.
    max_candidates : int
        Maximum number of candidates to include (sorted by confidence).

    Returns
    -------
    str
        Complete HTML document.
    """
    if not candidates_path.exists():
        raise FileNotFoundError(
            f"Compound candidates file not found: {candidates_path}\n"
            "Run compound_detector.py first."
        )

    data = json.loads(candidates_path.read_text(encoding="utf-8"))
    candidates = sorted(
        data.get("candidates", []),
        key=lambda c: (-c.get("n_methods_agreeing", 0), -c.get("consensus_confidence", 0)),
    )[:max_candidates]

    catalog = _load_svg_catalog(svg_catalog_path)

    # Provenance: try supplied dir, then guess from candidates path
    if corpus_dir is None:
        corpus_dir = candidates_path.parent.parent.parent / "data" / "corpus"
    provenance = _build_provenance_index(corpus_dir)
    if not provenance:
        logger.warning(
            "No provenance loaded (corpus_dir=%s) — provenance column will be empty.",
            corpus_dir,
        )

    logger.info(
        "Building compound report: %d candidates, %d SVG codes, %d provenance codes.",
        len(candidates), len(catalog), len(provenance),
    )

    from datetime import datetime, timezone
    report_meta = {
        **data,
        "generated": datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
    }

    return _render_html(candidates, catalog, provenance, report_meta)


def save_compound_report(
    candidates_path: Path,
    svg_catalog_path: Path,
    output_path: Path,
    corpus_dir: Path | None = None,
    max_candidates: int = 50,
) -> None:
    """Generate and write the compound report to an HTML file.

    Parameters
    ----------
    candidates_path : Path
    svg_catalog_path : Path
    output_path : Path
        Destination ``.html`` file.  Parent directories are created.
    corpus_dir : Path, optional
    max_candidates : int
    """
    html = build_compound_report(
        candidates_path=candidates_path,
        svg_catalog_path=svg_catalog_path,
        corpus_dir=corpus_dir,
        max_candidates=max_candidates,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    logger.info(
        "Compound report written: %s (%d bytes).", output_path, len(html)
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate compound glyph candidate report for scholar review."
    )
    p.add_argument(
        "--candidates", type=Path,
        default=Path("outputs/analysis/compound_candidates.json"),
        help="compound_candidates.json from compound_detector.py.",
    )
    p.add_argument(
        "--svg-catalog", type=Path,
        default=Path("data/glyphs/svg/catalog.json"),
        help="SVG glyph catalog JSON (default: data/glyphs/svg/catalog.json).",
    )
    p.add_argument(
        "--corpus-dir", type=Path,
        default=None,
        help="Corpus directory for tablet provenance (default: auto-detected).",
    )
    p.add_argument(
        "--output", type=Path,
        default=None,
        help="Output HTML path (default: <candidates dir>/compound_report.html).",
    )
    p.add_argument(
        "--max-candidates", type=int, default=50,
        help="Maximum candidates to include (default: 50).",
    )
    return p.parse_args()


def main() -> None:
    import logging as _logging
    _logging.basicConfig(level=_logging.INFO, format="%(levelname)s  %(message)s")
    args = _parse_args()
    output = args.output or (args.candidates.parent / "compound_report.html")
    save_compound_report(
        candidates_path=args.candidates,
        svg_catalog_path=args.svg_catalog,
        output_path=output,
        corpus_dir=args.corpus_dir,
        max_candidates=args.max_candidates,
    )
    print(f"Report written to: {output}")


if __name__ == "__main__":
    main()
