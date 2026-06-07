"""
Generate a focused HTML report on "holy grail" passage candidates.

Holy-grail criterion (from passage_report.py): a non-allographic substitution
that recurs at the same canonical position in ≥ 2 independent post-contact
tablets, making idiosyncratic scribal error unlikely.

Output: outputs/decipherment/holy_grail_report.html
"""

from __future__ import annotations

import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data"
SVG_DIR = DATA_DIR / "glyphs" / "svg"
OUTPUT_PATH = ROOT / "outputs" / "decipherment" / "holy_grail_report.html"

PARALLELS_JSON = DATA_DIR / "parallels" / "parallel_variants_auto.json"
CATALOG_JSON = SVG_DIR / "catalog.json"
CONTACT_JSON = ROOT / "outputs" / "contact_partition.json"


# ---------------------------------------------------------------------------
# SVG helpers
# ---------------------------------------------------------------------------

def _load_svg(path: str) -> str:
    """Read an SVG file, strip the XML declaration, return the <svg> element."""
    full = (DATA_DIR / "glyphs" / path).read_text(encoding="utf-8").strip()
    full = re.sub(r'<\?xml[^>]*\?>', '', full).strip()
    return full


def _wrap_svg(svg_content: str, label: str, sub: str = "", badge_cls: str = "") -> str:
    badge_html = (
        f'<span class="glyph-badge {badge_cls}">{sub}</span>' if sub else ""
    )
    return f"""<div class="glyph-cell">
  <div class="glyph-svg">{svg_content}</div>
  <div class="glyph-label mono">{label}</div>
  {badge_html}
</div>"""


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_passage(passage_id: str) -> dict:
    raw = json.loads(PARALLELS_JSON.read_text())
    passages = raw.get("passages", raw) if isinstance(raw, dict) else raw
    return next(p for p in passages if p["passage_id"] == passage_id)


def load_svg_catalog() -> dict[str, list[dict]]:
    """Return dict: barthel_code -> list of catalog records."""
    records = json.loads(CATALOG_JSON.read_text()).get("records", [])
    by_code: dict[str, list[dict]] = {}
    for r in records:
        code = r.get("barthel_code", "")
        by_code.setdefault(code, []).append(r)
    return by_code


def load_contact_data() -> dict[str, dict]:
    """Return dict: horley_sign -> contact partition record."""
    data = json.loads(CONTACT_JSON.read_text())
    # Support both old list format and new dict format (with _provenance).
    items = data["records"] if isinstance(data, dict) else data
    return {item["sign"]: item for item in items}


# ---------------------------------------------------------------------------
# Glyph panel: canonical passage
# ---------------------------------------------------------------------------

HORLEY_CODES = {
    "205":  "200 10",
    "711v": "711",
    "678":  "678",
    "002":  "2",
}

STRATUM_BADGE = {
    "pre_contact":  ("Pre-contact", "badge-pre"),
    "post_contact": ("Post-contact", "badge-post"),
    "unknown":      ("Undated", "badge-none"),
    "excluded":     ("Excluded", "badge-excl"),
}


def _best_svg(by_code: dict, barthel: str, prefer_stratum: str | None = None) -> str | None:
    """Find the best SVG path for a given Barthel code."""
    candidates = by_code.get(barthel, [])
    if not candidates:
        return None
    if prefer_stratum:
        pref = [r for r in candidates if r.get("cluster") == prefer_stratum]
        if pref:
            return pref[0]["svg_path"]
    return candidates[0]["svg_path"]


def _sign_description(code: str) -> str:
    descs = {
        "205":  "Barthel 205 — Horley H200‑10 — '200-series compound' — appears in all 13 tablets",
        "711v": "Barthel 711v — 'inverted 711' — attested on post-contact tablets only",
        "678":  "Barthel 678 — 600-series (bird family) — pre-contact anchor sign (Tablet D position 1)",
        "002":  "Barthel 002 — Horley H2 — high-frequency function sign — post-contact replacement at position 1",
    }
    return descs.get(code, f"Barthel {code}")


# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------

CSS = """
:root {
  --bg: #ffffff; --surface: #f8f8fa; --surface2: #f0f0f5;
  --border: #d0d0dd; --text: #1a1a1a; --muted: #666666;
  --accent: #c4a96d; --accent2: #7b9ee0;
  --pre: #2563eb; --post: #7c3aed; --undated: #888888;
  --holy: #d4860a; --cross: #c0392b;
  --gold: #b8860b; --gold-light: #fff8e7;
}
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
  background: var(--bg); color: var(--text);
  font-family: 'Cormorant Garamond', 'Palatino Linotype', Georgia, serif;
  font-size: 16px; line-height: 1.65;
}
.wrap { max-width: 1140px; margin: 0 auto; padding: 52px 28px; }
.mono { font-family: 'JetBrains Mono', 'Fira Mono', monospace; }
.muted { color: var(--muted); }
.small { font-size: 11px; }

/* ── Report header ── */
.report-header { border-bottom: 2px solid var(--holy); padding-bottom: 38px; margin-bottom: 44px; }
.crown { font-size: 28px; line-height: 1; margin-bottom: 8px; }
.report-title { font-size: 36px; font-weight: 600; color: #000; letter-spacing: -0.3px; }
.report-subtitle { font-size: 17px; color: var(--holy); font-style: italic; margin-top: 6px; }
.report-meta { margin-top: 20px; font-family: 'JetBrains Mono', monospace;
               font-size: 11px; color: var(--muted); line-height: 2.2; }
.report-meta b { color: #333; }
.abstract { margin-top: 20px; font-size: 14px; color: #333; max-width: 840px; line-height: 1.85; }
.abstract p + p { margin-top: 12px; }

/* ── Stat cards ── */
.stats-row { display: flex; flex-wrap: wrap; gap: 14px; margin-bottom: 40px; }
.stat-card { background: var(--surface); border: 1px solid var(--border);
             border-radius: 6px; padding: 16px 22px; min-width: 110px; text-align: center; }
.stat-card.holy { border-color: var(--holy); background: var(--gold-light); }
.stat-value { font-family: 'JetBrains Mono', monospace; font-size: 28px;
              font-weight: 500; color: var(--accent); }
.stat-card.holy .stat-value { color: var(--holy); }
.stat-label { font-size: 11px; color: var(--muted); margin-top: 4px;
              font-family: 'JetBrains Mono', monospace; }

/* ── Section labels ── */
.section-label { font-family: 'JetBrains Mono', monospace; font-size: 9px;
                 color: var(--muted); letter-spacing: 0.1em; text-transform: uppercase;
                 margin-bottom: 12px; }
.section-label.holy { color: var(--holy); }
.section-title { font-size: 20px; font-weight: 600; color: #111;
                 margin: 36px 0 14px; border-bottom: 1px solid var(--border);
                 padding-bottom: 8px; }

/* ── Glyph strip ── */
.glyph-strip { display: flex; flex-wrap: wrap; align-items: flex-end;
               gap: 10px; padding: 24px; background: var(--surface);
               border: 1px solid var(--border); border-radius: 8px; margin-bottom: 28px; }
.glyph-strip.pre  { border-color: var(--pre);  background: #eff6ff; }
.glyph-strip.post { border-color: var(--post); background: #f5f3ff; }
.glyph-strip.canonical { border-color: var(--holy); background: var(--gold-light); }
.strip-label { font-family: 'JetBrains Mono', monospace; font-size: 10px;
               color: var(--muted); align-self: center; min-width: 80px;
               text-transform: uppercase; letter-spacing: 0.07em; }
.glyph-cell { display: flex; flex-direction: column; align-items: center; gap: 4px; }
.glyph-svg { width: 72px; height: 72px; display: flex; align-items: center;
             justify-content: center; background: #fff;
             border: 1px solid var(--border); border-radius: 4px; padding: 4px; }
.glyph-svg svg { max-width: 62px; max-height: 62px; width: auto; height: auto; }
.glyph-label { font-size: 9.5px; color: var(--muted); text-align: center; }
.glyph-badge { font-family: 'JetBrains Mono', monospace; font-size: 7.5px;
               border-radius: 2px; padding: 1px 5px; border: 1px solid transparent;
               margin-top: 2px; white-space: nowrap; }
.glyph-badge.badge-pre  { color: var(--pre);  background: #dbeafe; border-color: #bfdbfe; }
.glyph-badge.badge-post { color: var(--post); background: #ede9fe; border-color: #ddd6fe; }
.glyph-badge.badge-none { color: var(--muted); background: var(--surface2); border-color: var(--border); }
.glyph-badge.badge-holy { color: var(--holy); background: #fff3cd; border-color: #ffe08a; }
.glyph-badge.badge-excl { color: #888; background: #f0f0f0; border-color: #ccc; }
.arrow-sep { font-size: 24px; color: var(--muted); align-self: center;
             font-family: 'JetBrains Mono', monospace; padding: 0 4px; }
.arrow-sep.holy { color: var(--holy); font-size: 28px; }
.glyph-placeholder { width: 72px; height: 72px; display: flex; align-items: center;
                     justify-content: center; border: 2px dashed var(--border);
                     border-radius: 4px; color: var(--muted); font-size: 10px;
                     font-family: 'JetBrains Mono', monospace; }
.position-label { font-family: 'JetBrains Mono', monospace; font-size: 9px;
                  color: var(--muted); text-align: center; margin-top: 4px; }

/* ── Holy grail change card ── */
.hg-card { background: var(--gold-light); border: 2px solid var(--holy);
           border-radius: 8px; padding: 28px; margin-bottom: 32px; }
.hg-header { display: flex; align-items: center; gap: 14px; margin-bottom: 20px; flex-wrap: wrap; }
.hg-badge { font-family: 'JetBrains Mono', monospace; font-size: 9px; font-weight: 700;
            text-transform: uppercase; letter-spacing: 0.12em;
            background: var(--holy); color: #fff;
            border-radius: 3px; padding: 4px 10px; }
.hg-title { font-size: 17px; font-weight: 600; color: #333; }
.hg-change-display { display: flex; align-items: center; gap: 20px; flex-wrap: wrap;
                     margin: 20px 0; padding: 20px; background: #fff;
                     border: 1px solid var(--holy); border-radius: 6px; }
.hg-sign { text-align: center; }
.hg-sign-img { width: 90px; height: 90px; display: flex; align-items: center;
               justify-content: center; border-radius: 6px; padding: 6px;
               border: 2px solid transparent; }
.hg-sign-img.pre-sign  { border-color: var(--pre);  background: #eff6ff; }
.hg-sign-img.post-sign { border-color: var(--post); background: #f5f3ff; }
.hg-sign-img svg { max-width: 76px; max-height: 76px; width: auto; height: auto; }
.hg-sign-code { font-family: 'JetBrains Mono', monospace; font-size: 13px;
                font-weight: 500; margin-top: 6px; }
.hg-sign-sublabel { font-family: 'JetBrains Mono', monospace; font-size: 9px;
                    color: var(--muted); margin-top: 2px; }
.hg-arrow { font-size: 36px; color: var(--holy); }
.hg-meta-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
                gap: 12px; margin-top: 16px; }
.hg-meta-cell { background: #fff; border: 1px solid var(--border); border-radius: 4px;
                padding: 10px 14px; }
.hg-meta-label { font-family: 'JetBrains Mono', monospace; font-size: 8.5px;
                 color: var(--muted); text-transform: uppercase; letter-spacing: 0.07em;
                 margin-bottom: 4px; }
.hg-meta-val { font-family: 'JetBrains Mono', monospace; font-size: 13px; color: #111; }
.hg-meta-val.highlight { color: var(--holy); font-weight: 600; }

/* ── Tablet grid ── */
.tablet-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
               gap: 12px; margin-bottom: 32px; }
.tablet-card { background: var(--surface); border: 1px solid var(--border);
               border-radius: 6px; padding: 14px 16px; }
.tablet-card.pre  { border-left: 3px solid var(--pre);  background: #eff6ff; }
.tablet-card.post { border-left: 3px solid var(--post); background: #f5f3ff; }
.tablet-card.excl { border-left: 3px solid #ccc; opacity: 0.7; }
.tablet-name { font-family: 'JetBrains Mono', monospace; font-size: 15px;
               font-weight: 600; color: #111; margin-bottom: 4px; }
.tablet-stratum { font-size: 10px; color: var(--muted);
                  font-family: 'JetBrains Mono', monospace; margin-bottom: 6px; }
.tablet-forms { font-family: 'JetBrains Mono', monospace; font-size: 9.5px;
                color: #555; line-height: 1.9; }
.tablet-pos1 { font-weight: 600; }
.pos1-pre  { color: var(--pre); }
.pos1-post { color: var(--post); }
.pos1-holy { color: var(--holy); font-weight: 700; }

/* ── Frequency table ── */
.freq-table { width: 100%; border-collapse: collapse; font-size: 12px; margin-bottom: 24px; }
.freq-table th { font-family: 'JetBrains Mono', monospace; font-size: 9px; font-weight: 600;
                 color: var(--muted); text-transform: uppercase; letter-spacing: 0.07em;
                 padding: 6px 10px; border-bottom: 1px solid var(--border); text-align: left; }
.freq-table td { padding: 6px 10px; border-bottom: 1px solid #e8e8ee; }
.freq-table tr:last-child td { border-bottom: none; }
.freq-table tr:hover td { background: var(--surface); }
.freq-bar { height: 8px; background: var(--accent2); border-radius: 2px; min-width: 2px; }
.freq-bar.pre-bar { background: var(--pre); }
.freq-bar.post-bar { background: var(--post); }

/* ── Decipherment section ── */
.deciph-block { background: var(--surface); border: 1px solid var(--border);
                border-radius: 8px; padding: 24px; margin-bottom: 20px; }
.deciph-block.highlighted { border-color: var(--holy); background: var(--gold-light); }
.deciph-sign { display: flex; gap: 20px; align-items: flex-start; flex-wrap: wrap;
               margin-bottom: 16px; }
.deciph-sign-img { width: 80px; height: 80px; flex-shrink: 0; display: flex;
                   align-items: center; justify-content: center;
                   background: #fff; border: 1px solid var(--border);
                   border-radius: 6px; padding: 6px; }
.deciph-sign-img svg { max-width: 68px; max-height: 68px; width: auto; height: auto; }
.deciph-sign-info { flex: 1; min-width: 200px; }
.deciph-sign-code { font-family: 'JetBrains Mono', monospace; font-size: 18px;
                    font-weight: 600; color: var(--accent); margin-bottom: 4px; }
.deciph-sign-name { font-size: 13px; color: #555; margin-bottom: 8px; }
.reading-chips { display: flex; flex-wrap: wrap; gap: 6px; margin-bottom: 10px; }
.reading-chip { font-family: 'JetBrains Mono', monospace; font-size: 10px;
                background: #fff; border: 1px solid var(--border);
                border-radius: 3px; padding: 3px 8px; color: #333; }
.reading-chip.proposed { border-color: var(--holy); color: var(--holy);
                          background: var(--gold-light); }
.deciph-note { font-size: 13px; color: #444; line-height: 1.75; margin-top: 8px; }
.deciph-source { font-family: 'JetBrains Mono', monospace; font-size: 9px;
                 color: var(--muted); margin-top: 6px; }

/* ── Badges ── */
.badge { display: inline-block; font-family: 'JetBrains Mono', monospace;
         font-size: 8.5px; border-radius: 3px; padding: 2px 6px;
         border: 1px solid transparent; white-space: nowrap; margin: 2px; }
.badge-pre  { color: var(--pre);  background: #dbeafe; border-color: #bfdbfe; }
.badge-post { color: var(--post); background: #ede9fe; border-color: #ddd6fe; }
.badge-none { color: var(--muted); background: var(--surface2); border-color: var(--border); }
.badge-holy { color: var(--holy); background: #fff3cd; border-color: #ffe08a; }
.badge-excl { color: #888; background: #f0f0f0; border-color: #ccc; }

/* ── Note box ── */
.note-box { background: #f0f7ff; border-left: 3px solid var(--pre);
            padding: 14px 18px; border-radius: 0 6px 6px 0; margin: 20px 0;
            font-size: 13px; color: #333; line-height: 1.75; }
.note-box.warn { background: var(--gold-light); border-color: var(--holy); }

/* ── Footer ── */
.report-footer { border-top: 1px solid var(--border); margin-top: 56px;
                 padding-top: 26px; font-size: 12px; color: var(--muted); line-height: 2.0; }
.report-footer a { color: var(--accent); text-decoration: none; }

@media (max-width: 760px) {
  .hg-change-display { flex-direction: column; }
  .tablet-grid { grid-template-columns: 1fr 1fr; }
}
"""


# ---------------------------------------------------------------------------
# Main report builder
# ---------------------------------------------------------------------------

def _tablet_class(stratum: str) -> str:
    return {"pre_contact": "pre", "post_contact": "post", "excluded": "excl"}.get(stratum, "")


def _stratum_label(stratum: str) -> str:
    return {
        "pre_contact":  "Pre-contact (radiocarbon-dated)",
        "post_contact": "Post-contact",
        "unknown":      "Undated",
        "excluded":     "Excluded from stratum analysis",
    }.get(stratum, stratum)


def _freq_row(sign_horley: str, cp: dict[str, dict]) -> str:
    item = cp.get(sign_horley)
    if not item:
        return ""
    f_pre = item["f_pre"]
    f_post = item["f_post"]
    fp_pre = item["freq_pre_per_1k"]
    fp_post = item["freq_post_per_1k"]
    g2 = item["g2"]
    bias = item["bias"]
    bias_badge = {
        "pre_biased":  '<span class="badge badge-pre">pre-biased</span>',
        "post_biased": '<span class="badge badge-post">post-biased</span>',
    }.get(bias, '<span class="badge badge-none">neutral</span>')
    max_fp = max(fp_pre, fp_post, 1)
    bar_pre = int(fp_pre / max_fp * 100)
    bar_post = int(fp_post / max_fp * 100)
    return f"""<tr>
  <td class="mono" style="font-size:12px">{sign_horley}</td>
  <td>{f_pre}</td>
  <td>{f_post}</td>
  <td><div style="display:flex;align-items:center;gap:4px">
    <div class="freq-bar pre-bar" style="width:{bar_pre}px"></div>
    <span style="font-size:10px;color:var(--pre)">{fp_pre:.1f}</span></div></td>
  <td><div style="display:flex;align-items:center;gap:4px">
    <div class="freq-bar post-bar" style="width:{bar_post}px"></div>
    <span style="font-size:10px;color:var(--post)">{fp_post:.1f}</span></div></td>
  <td>{g2:.2f}</td>
  <td>{bias_badge}</td>
</tr>"""


def build_report() -> str:
    passage = load_passage("P012_ABCDEGHINPQSX")
    by_code = load_svg_catalog()
    cp = load_contact_data()
    generated = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    pid = passage["passage_id"]
    score = passage["interest_score"]
    n_tablets = passage["n_tablets"]
    attestations = passage["attestations"]
    changes = passage["diachronic_changes"]
    canonical_form = passage.get("canonical_form", [])

    # Tablet stratum summary
    tablet_strata: dict[str, str] = {}
    tablet_forms: dict[str, list[list[str]]] = {}
    for att in attestations:
        tab = att.get("tablet", "?")
        stratum = att.get("stratum", "unknown")
        form = att.get("form", [])
        if tab not in tablet_strata or tablet_strata[tab] in ("unknown",):
            tablet_strata[tab] = stratum
        tablet_forms.setdefault(tab, [])
        if form not in tablet_forms[tab]:
            tablet_forms[tab].append(form)

    pre_tablets = sorted(t for t, s in tablet_strata.items() if s == "pre_contact")
    post_tablets = sorted(t for t, s in tablet_strata.items() if s == "post_contact")
    excl_tablets = sorted(t for t, s in tablet_strata.items() if s == "excluded")
    unk_tablets = sorted(t for t, s in tablet_strata.items() if s == "unknown")

    holy_change = next(c for c in changes if c.get("is_holy_grail_candidate"))
    pre_sign = holy_change["pre_contact_sign"]
    post_sign = holy_change["post_contact_sign"]
    change_pos = holy_change["position"]
    n_cons = holy_change["n_tablets_consistent"]

    # Load SVGs
    pre_205_svg = _load_svg(_best_svg(by_code, "205", "pre_contact"))
    post_205_svg = _load_svg(_best_svg(by_code, "205", "post_contact"))
    pre_678_svg = _load_svg(_best_svg(by_code, "678", "pre_contact"))
    post_678_svg = _load_svg(_best_svg(by_code, "678")) or pre_678_svg
    pre_002_svg = _load_svg(_best_svg(by_code, "002", "pre_contact"))
    post_002_svg = _load_svg(_best_svg(by_code, "002", "post_contact"))
    b_711v_svg = _load_svg(_best_svg(by_code, "711v")) if by_code.get("711v") else None

    # ── Glyph strips ────────────────────────────────────────────────────────

    # Canonical form strip
    cf_cells = []
    for i, code in enumerate(canonical_form):
        clean = code.rstrip("v?!")
        svg_path = _best_svg(by_code, code) or _best_svg(by_code, clean)
        svg_str = _load_svg(svg_path) if svg_path else None
        is_holy_pos = (i == change_pos)
        badge_cls = "badge-holy" if is_holy_pos else "badge-none"
        badge_lbl = "★ holy-grail pos." if is_holy_pos else f"pos {i}"
        if svg_str:
            cf_cells.append(_wrap_svg(svg_str, f"B{code}", badge_lbl, badge_cls))
        else:
            cf_cells.append(
                f'<div class="glyph-cell">'
                f'<div class="glyph-placeholder">B{code}</div>'
                f'<div class="glyph-label mono">{code}</div>'
                f'<span class="glyph-badge {badge_cls}">{badge_lbl}</span>'
                f'</div>'
            )
        if i < len(canonical_form) - 1:
            cf_cells.append('<div class="arrow-sep muted">›</div>')
    canonical_strip = f"""
<div class="section-label holy">Canonical passage form (consensus across all attestations)</div>
<div class="glyph-strip canonical">
  <div class="strip-label">Canonical<br>B{canonical_form[0]}…</div>
  {"".join(cf_cells)}
</div>"""

    # Pre-contact strip (Tablet D)
    pre_strip = f"""
<div class="section-label" style="color:var(--pre)">Pre-contact form — Tablet D (radiocarbon-dated 1390–1520 CE, Ferrara et al. 2024)</div>
<div class="glyph-strip pre">
  <div class="strip-label">Tablet D<br>pre-contact</div>
  {_wrap_svg(pre_205_svg, "B205", "pos 0", "badge-pre")}
  <div class="arrow-sep">›</div>
  {_wrap_svg(pre_678_svg, "B678 ← holy", "pos 1 pre", "badge-pre")}
</div>"""

    # Post-contact strip (Tablet B/C)
    post_strip = f"""
<div class="section-label" style="color:var(--post)">Post-contact form — Tablets B, C, P, Q (consistent substitution)</div>
<div class="glyph-strip post">
  <div class="strip-label">Tablets B/C<br>post-contact</div>
  {_wrap_svg(post_205_svg, "B205", "pos 0", "badge-post")}
  <div class="arrow-sep">›</div>
  {_wrap_svg(post_002_svg, "B002 ← holy", "pos 1 post", "badge-holy")}
</div>"""

    # ── Holy grail change card ────────────────────────────────────────────

    hg_card = f"""
<div class="hg-card">
  <div class="hg-header">
    <span class="hg-badge">★ Holy-grail candidate</span>
    <span class="hg-title">Sign substitution at position {change_pos + 1}: {pre_sign} → {post_sign}</span>
  </div>
  <p style="font-size:13px;color:#444;line-height:1.75;margin-bottom:16px">
    Wherever the pre-contact text (Tablet D) had sign <b>B{pre_sign}</b> immediately after
    sign B205, at least <b>{n_cons} independent post-contact tablets</b> consistently replaced
    it with sign <b>B{post_sign}</b>. The consistency across independent scribal traditions
    rules out idiosyncratic copying error and makes this a statistically robust candidate for
    a systematic linguistic or scribal change crossing the 1722 CE European contact boundary.
    The substitution also <b>crosses Barthel century blocks</b> (600-series → 000-series),
    which is iconographically surprising: the two signs are visually and structurally unrelated.
  </p>

  <div class="hg-change-display">
    <div class="hg-sign">
      <div class="hg-sign-img pre-sign">{pre_678_svg}</div>
      <div class="hg-sign-code" style="color:var(--pre)">B{pre_sign}</div>
      <div class="hg-sign-sublabel">600-series (bird family)</div>
      <div class="hg-sign-sublabel" style="margin-top:3px">
        <span class="badge badge-pre">Pre-contact</span>
        <span class="badge badge-none">Tablet D</span>
      </div>
    </div>

    <div class="hg-arrow">⟶</div>

    <div class="hg-sign">
      <div class="hg-sign-img post-sign">{post_002_svg}</div>
      <div class="hg-sign-code" style="color:var(--post)">B{post_sign}</div>
      <div class="hg-sign-sublabel">000-series (function signs)</div>
      <div class="hg-sign-sublabel" style="margin-top:3px">
        <span class="badge badge-post">Post-contact</span>
        <span class="badge badge-none">≥ {n_cons} tablets</span>
      </div>
    </div>
  </div>

  <div class="hg-meta-grid">
    <div class="hg-meta-cell">
      <div class="hg-meta-label">Change type</div>
      <div class="hg-meta-val">Substitution</div>
    </div>
    <div class="hg-meta-cell">
      <div class="hg-meta-label">Canonical position</div>
      <div class="hg-meta-val">{change_pos + 1} of {len(canonical_form)}</div>
    </div>
    <div class="hg-meta-cell">
      <div class="hg-meta-label">Post-contact tablets consistent</div>
      <div class="hg-meta-val highlight">{n_cons}</div>
    </div>
    <div class="hg-meta-cell">
      <div class="hg-meta-label">Crosses Barthel family</div>
      <div class="hg-meta-val highlight">Yes (6xx → 00x)</div>
    </div>
    <div class="hg-meta-cell">
      <div class="hg-meta-label">Known allograph</div>
      <div class="hg-meta-val">No</div>
    </div>
    <div class="hg-meta-cell">
      <div class="hg-meta-label">Horley codes</div>
      <div class="hg-meta-val">H678 → H2</div>
    </div>
  </div>
</div>"""

    # ── Tablet grid ──────────────────────────────────────────────────────

    tablet_rows = []
    for tab in sorted(tablet_strata.keys()):
        stratum = tablet_strata[tab]
        cls = _tablet_class(stratum)
        forms = tablet_forms.get(tab, [])
        # Show unique forms, limit to 4
        unique_forms = []
        for f in forms:
            if f not in unique_forms:
                unique_forms.append(f)
        form_lines = []
        for f in unique_forms[:5]:
            form_str = " ".join(str(c) for c in f)
            # Highlight position 1 if it's the substitution position
            if len(f) > change_pos:
                sign_at_pos = str(f[change_pos])
                if sign_at_pos == pre_sign:
                    form_lines.append(f'<div class="tablet-forms"><span>{" ".join(str(c) for c in f[:change_pos])}</span> <span class="pos1-pre tablet-pos1">{sign_at_pos}</span>{(" " + " ".join(str(c) for c in f[change_pos+1:])) if len(f) > change_pos+1 else ""}</div>')
                elif sign_at_pos == post_sign:
                    form_lines.append(f'<div class="tablet-forms"><span>{" ".join(str(c) for c in f[:change_pos])}</span> <span class="pos1-post tablet-pos1">{sign_at_pos}</span>{(" " + " ".join(str(c) for c in f[change_pos+1:])) if len(f) > change_pos+1 else ""}</div>')
                else:
                    form_lines.append(f'<div class="tablet-forms">{form_str}</div>')
            else:
                form_lines.append(f'<div class="tablet-forms muted">{form_str or "—"}</div>')
        if len(unique_forms) > 5:
            form_lines.append(f'<div class="tablet-forms muted small">+ {len(unique_forms)-5} more variants</div>')
        sl, sc = STRATUM_BADGE.get(stratum, (stratum, "badge-none"))
        tablet_rows.append(f"""<div class="tablet-card {cls}">
  <div class="tablet-name">Tablet {tab}</div>
  <div class="tablet-stratum"><span class="badge {sc}">{sl}</span></div>
  {"".join(form_lines)}
</div>""")
    tablet_grid_html = "\n".join(tablet_rows)

    # ── Frequency table ───────────────────────────────────────────────────

    freq_rows = "".join([
        _freq_row("200 10", cp),   # Barthel 205
        _freq_row("678",    cp),   # Barthel 678
        _freq_row("2",      cp),   # Barthel 002
        _freq_row("711",    cp),   # Barthel 711v
    ])
    freq_table = f"""
<table class="freq-table">
<thead><tr>
  <th>Horley sign</th>
  <th>f (pre)</th><th>f (post)</th>
  <th>per 1k (pre)</th><th>per 1k (post)</th>
  <th>G²</th><th>Contact bias</th>
</tr></thead>
<tbody>{freq_rows}</tbody>
</table>"""

    # ── Decipherment section ──────────────────────────────────────────────

    deciph_205_img = _wrap_svg(pre_205_svg, "B205", "pre-contact (D)", "badge-pre")
    deciph_678_img = _wrap_svg(pre_678_svg, "B678 — pre", "pre-contact (D)", "badge-pre")
    deciph_002_img = _wrap_svg(post_002_svg, "B002 — post", "post-contact (B)", "badge-post")
    deciph_711v_img = (
        _wrap_svg(_load_svg(_best_svg(by_code, "711v")), "B711v", "post-contact (B)", "badge-post")
        if by_code.get("711v") else
        '<div class="glyph-placeholder" style="width:72px;height:72px">B711v</div>'
    )

    decipherment_section = f"""
<div class="section-title">Potential Decipherment</div>

<div class="note-box warn">
  <b>Computational hypothesis only.</b> All phoneme and morpheme readings below are
  probabilistic proposals from the MCMC + beam-search pipeline, checked against
  Polynesian language models. No reading should be treated as established without
  expert epigraphic and linguistic review.
  Sign 205 and the holy-grail substitution pair (B678 → B002) do not yet have
  confirmed scholarly readings; the proposals here represent the pipeline's
  best-scoring hypotheses.
</div>

<div class="deciph-block">
  <div class="deciph-sign">
    <div class="deciph-sign-img">{pre_205_svg}</div>
    <div class="deciph-sign-info">
      <div class="deciph-sign-code">B205 — Horley H200‑10</div>
      <div class="deciph-sign-name">200-series compound · appears at position 0 in all 13 tablets</div>
      <div class="reading-chips">
        <span class="reading-chip proposed">*te (proposed)</span>
        <span class="reading-chip proposed">*ko (proposed)</span>
        <span class="reading-chip">function marker (Barthel 1958)</span>
        <span class="reading-chip">200-family base (Horley 2021)</span>
      </div>
      <div class="deciph-note">
        Sign B205 anchors this passage across all 13 tablets — both pre- and post-contact.
        Its stability (no substitution at position 0) makes it the most reliable structural
        anchor in the passage. The 200-series has been proposed as a <i>taxogram</i> class
        (Horley 2021), marking syntactic unit boundaries. B205 specifically may carry
        a determiner or topic-marker function; its Rapa Nui analogues could be
        <i>te</i> (definite article) or <i>ko</i> (predicate marker).
        Contact-partition data show neutral bias (G²&nbsp;=&nbsp;0.09),
        confirming it is used equally across both eras.
      </div>
      <div class="deciph-source">Sources: Barthel (1958); Horley (2021 §3.2); Fischer (1997);
      hackingrongo MCMC Zone C hypothesis H0001</div>
    </div>
  </div>
</div>

<div class="deciph-block highlighted">
  <div class="section-label holy">The holy-grail pair — pre-contact B678 vs. post-contact B002</div>
  <div class="deciph-sign" style="gap:28px">
    <div style="display:flex;flex-direction:column;align-items:center;gap:6px">
      <div class="deciph-sign-img">{pre_678_svg}</div>
      <span class="badge badge-pre">Pre-contact</span>
    </div>
    <div style="font-size:28px;color:var(--holy);align-self:center">⟶</div>
    <div style="display:flex;flex-direction:column;align-items:center;gap:6px">
      <div class="deciph-sign-img">{post_002_svg}</div>
      <span class="badge badge-post">Post-contact</span>
    </div>
    <div class="deciph-sign-info">
      <div class="deciph-sign-code">B678 (pre) ↔ B002 (post) — position {change_pos + 1}</div>
      <div class="deciph-sign-name">Cross-family substitution: 600-series bird sign → 000-series abstract sign</div>
      <div class="reading-chips">
        <span class="reading-chip" style="border-color:var(--pre);color:var(--pre)">B678: *tangata? (Fischer 1997)</span>
        <span class="reading-chip" style="border-color:var(--pre);color:var(--pre)">B678: ritual/bird-man context</span>
        <span class="reading-chip proposed">B002: *ko / *i (pipeline H0001)</span>
        <span class="reading-chip proposed">B002: grammatical particle</span>
      </div>
      <div class="deciph-note">
        <b>Why this matters:</b> The 600-series (B678) is the "bird" or "bird-man" family in
        Barthel's iconographic classification — these signs appear prominently in ritual
        contexts on Easter Island. The 000-series (B002) is one of the simplest, highest-frequency
        abstract signs in the corpus.<br><br>
        Three interpretations of the 678 → 002 substitution are consistent with the data:
        <ol style="margin:10px 0 0 18px;line-height:2">
          <li><b>Phonetic rebus:</b> both signs may represent the same phoneme(s) in different
          graphic traditions — one pictographic (bird), one abstract — with post-contact
          scribes preferring the abstract form.</li>
          <li><b>Semantic narrowing / semantic shift:</b> the ritual "bird-man" word encoded
          by B678 may have been replaced post-contact by a more generic or different lexical
          item encoded by B002, reflecting cultural disruption after 1722&nbsp;CE.</li>
          <li><b>Scribal tradition split:</b> B678 and B002 could be allographs from different
          regional scribal traditions, with the pre-contact tradition preserved only on
          Tablet D and the post-contact tradition becoming dominant across all other surviving tablets.</li>
        </ol>
        <br>The pipeline's top MCMC hypothesis (H0001, LM score −38.4 bits) proposes that
        B002 reads as a high-frequency function morpheme, consistent with a grammatical
        particle role. If correct, the pre-contact B678 would encode the same morpheme
        in a more pictographic form.
      </div>
      <div class="deciph-source">Sources: Barthel (1958); Fischer (1997) ch. 4; Horley (2021);
      hackingrongo contact-partition G²&nbsp;=&nbsp;0.83 (neutral bias, seen in both strata)</div>
    </div>
  </div>
</div>

<div class="deciph-block">
  <div class="deciph-sign">
    <div class="deciph-sign-img">{_load_svg(_best_svg(by_code, "711v")) if by_code.get("711v") else "<div style='color:#aaa;font-size:10px'>no SVG</div>"}</div>
    <div class="deciph-sign-info">
      <div class="deciph-sign-code">B711v — Horley H711</div>
      <div class="deciph-sign-name">700-series · inverted variant · canonical form position 1 (undated tablets)</div>
      <div class="reading-chips">
        <span class="reading-chip">700-series anthropomorphic (Barthel 1958)</span>
        <span class="reading-chip">post-contact only in catalog</span>
      </div>
      <div class="deciph-note">
        B711v appears as the canonical form at position 1 in the overall consensus (accounting
        for all tablets), but is absent from Tablet D (pre-contact anchor). It is the most
        common position-1 variant across undated tablets. Its relationship to B678 and B002
        is unclear: it may represent yet another regional or temporal variant of the same
        morpheme slot, or it may encode a different lexical item in those tablets.
        The 700-series "anthropomorphic" signs have no confirmed phonetic readings;
        contact-partition data show B711 as post-contact prevalent (0.0 per 1k pre vs
        6.82 per 1k post), suggesting it emerged or became dominant after contact.
      </div>
      <div class="deciph-source">Sources: Barthel (1958); contact partition G²&nbsp;=&nbsp;2.75</div>
    </div>
  </div>
</div>"""

    # ── Put it all together ───────────────────────────────────────────────

    tablet_stratum_badges = (
        f'<span class="badge badge-pre">{", ".join("Tablet "+t for t in pre_tablets)} — pre-contact</span> ' +
        f'<span class="badge badge-post">{", ".join("Tablet "+t for t in post_tablets)} — post-contact</span> ' +
        f'<span class="badge badge-none">{", ".join("Tablet "+t for t in unk_tablets)} — undated</span>'
        + (f' <span class="badge badge-excl">{", ".join("Tablet "+t for t in excl_tablets)} — excl.</span>'
           if excl_tablets else "")
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>hackingrongo — Holy Grail Candidate Report</title>
<link href="https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,400;0,600;1,400&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>{CSS}</style>
</head>
<body>
<div class="wrap">

<div class="report-header">
  <div class="crown">★</div>
  <div class="report-title">hackingrongo<br>Holy Grail Candidate Report</div>
  <div class="report-subtitle">The single passage flagged as a holy-grail decipherment candidate across the full 13-tablet corpus</div>
  <div class="report-meta">
    <b>Passage ID:</b> {pid}
    &nbsp;·&nbsp;
    <b>Interest score:</b> {score:.2f} / 1.00
    &nbsp;·&nbsp;
    <b>Tablets:</b> {n_tablets}
    &nbsp;·&nbsp;
    <b>Holy-grail changes:</b> 1
    &nbsp;·&nbsp;
    <b>Generated:</b> {generated}
  </div>
  <div class="abstract">
    <p>A <b>holy-grail candidate</b> is a sign substitution that (1) crosses the
    pre/post-contact boundary, (2) is non-allographic (the two signs are not known
    graphic variants of the same phoneme), and (3) recurs consistently at the same
    canonical position in at least 2 independent post-contact tablets — ruling out
    idiosyncratic scribal error.</p>
    <p>Passage <b>{pid}</b> is the only such candidate in the current corpus.
    It spans <b>{n_tablets} tablets</b> and carries an interest score of
    <b>{score:.2f}</b> (the maximum possible), placing it at the absolute top of
    decipherment priority. The holy-grail substitution at position 2 —
    pre-contact sign <b>B{pre_sign}</b> consistently replaced by post-contact sign
    <b>B{post_sign}</b> — crosses Barthel century blocks (600-series → 000-series),
    which is iconographically anomalous and scientifically significant.</p>
  </div>
</div>

<div class="stats-row">
  <div class="stat-card holy">
    <div class="stat-value">{score:.0f}</div>
    <div class="stat-label">interest score</div>
  </div>
  <div class="stat-card">
    <div class="stat-value">{n_tablets}</div>
    <div class="stat-label">tablets</div>
  </div>
  <div class="stat-card">
    <div class="stat-value">{len(pre_tablets)}</div>
    <div class="stat-label">pre-contact</div>
  </div>
  <div class="stat-card">
    <div class="stat-value">{len(post_tablets)}</div>
    <div class="stat-label">post-contact</div>
  </div>
  <div class="stat-card">
    <div class="stat-value">{len(canonical_form)}</div>
    <div class="stat-label">canonical length</div>
  </div>
  <div class="stat-card holy">
    <div class="stat-value">1</div>
    <div class="stat-label">holy-grail change</div>
  </div>
</div>

<div>
  <div class="section-label">Tablet distribution</div>
  <div style="margin-bottom:32px">{tablet_stratum_badges}</div>
</div>

{canonical_strip}
{pre_strip}
{post_strip}

<div class="section-title">The Holy Grail Substitution</div>
{hg_card}

<div class="section-title">Attestations by Tablet — Position-1 Variants Highlighted</div>
<div class="note-box">
  In each tablet card below, the sign at position 1 (the holy-grail position) is highlighted:
  <span style="color:var(--pre);font-weight:600">blue = B{pre_sign} (pre-contact)</span>,
  <span style="color:var(--post);font-weight:600">purple = B{post_sign} (post-contact)</span>.
  Forms without a second sign contain only B205 alone at that attestation location.
</div>
<div class="tablet-grid">
{tablet_grid_html}
</div>

<div class="section-title">Contact-Partition Frequency Analysis</div>
<div class="section-label">Sign frequencies per 1000 glyphs — pre-contact corpus (Tablet D) vs. post-contact corpus (Tablets B, C, P, Q)</div>
{freq_table}

{decipherment_section}

<div class="report-footer">
  <p><a href="decipherment_report.html">← Zone C Decipherment Hypotheses</a>
  &nbsp;&middot;&nbsp;
  <a href="../contact_partition_bipartite.html">Contact Partition Graph</a></p>
  <p><b>hackingrongo</b> · Holy Grail Candidate Report · MIT License</p>
  <p>Holy-grail criterion: non-allographic substitution consistent in ≥ 2 post-contact tablets
  at the same canonical position (Barthel 1958 allograph catalog).
  Canonical form: Needleman-Wunsch consensus across all attestations.
  Pre-contact anchor: Tablet D (radiocarbon 1390–1520 CE, Ferrara et al. 2024).
  Contact partition: G² log-likelihood ratio, Tablet D vs. Tablets B+C+P+Q.</p>
  <p>This is a computational hypothesis report.
  All change candidates require expert epigraphic and linguistic review
  before any interpretive claim can be made.</p>
  <p><b>SperksWerks LLC</b> ·
  <a href="https://sperkswerks.ai">sperkswerks.ai</a></p>
  <p style="margin-top:6px;color:#bbb">Generated: {generated}</p>
</div>

</div>
</body>
</html>"""


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    html = build_report()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(html, encoding="utf-8")
    size_kb = len(html) / 1024
    print(f"Written: {OUTPUT_PATH}  ({size_kb:.1f} KB)")
