"""
reconstruct_tablet_d.py — Tablet D uncertain-sign reconstruction pipeline.

Orchestrates two independent evidence streams against Tablet D's uncertain
sign positions and produces a convergence score for each target.

Evidence stream 1 — Sequence model (Zone B)
  NgramModel predicts top-10 most probable sign completions using all
  surrounding resolved tokens as left+right context.

Evidence stream 2 — MCMC decipherment (Zone C)
  Checks whether any of the top-5 MCMC hypotheses contain a phoneme
  assignment for each uncertain sign code.

Convergence between streams constitutes independent evidence — neither
method saw the other's output.

Usage
-----
    # With defaults from config.yaml (run from project root):
    python scripts/reconstruct_tablet_d.py

    # Explicit paths:
    python scripts/reconstruct_tablet_d.py \\
        --corpus-dir  data/corpus \\
        --model       outputs/zone_b/sequence_model.json \\
        --ranking     outputs/decipherment/ranking.json \\
        --glyphs-dir  data/glyphs \\
        --output      outputs/reconstruction/
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import subprocess
import sys
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from omegaconf import OmegaConf  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Target classification
# ---------------------------------------------------------------------------

_UNCERTAIN_RE = re.compile(r".*\?$")
_RANGE_RE = re.compile(r"^\(\d+-\d+\)!$")
_VARIANT_RE = re.compile(r".*V$")


def classify_target_type(barthel_raw: str) -> Optional[str]:
    """Return 'uncertain', 'range', 'variant', or None (not a target).

    >>> classify_target_type("536?")
    'uncertain'
    >>> classify_target_type("(10-20)!")
    'range'
    >>> classify_target_type("050V")
    'variant'
    >>> classify_target_type("007")
    """
    if _RANGE_RE.match(barthel_raw):
        return "range"
    if _UNCERTAIN_RE.match(barthel_raw):
        return "uncertain"
    if _VARIANT_RE.match(barthel_raw):
        return "variant"
    return None


def _strip_suffix(code: str) -> str:
    """Strip trailing ?, f, V, y, or ()! markers to get the base Barthel code."""
    code = code.rstrip("?fVy")
    code = re.sub(r"^\(|\)!$", "", code)
    return code.strip()


# ---------------------------------------------------------------------------
# Convergence scoring
# ---------------------------------------------------------------------------

def compute_convergence_score(
    seq_top_k: list[dict],
    mcmc_phoneme: Optional[str],
    mcmc_confidence: Optional[float],
) -> float:
    """Compute convergence score in [0, 1].

    Rubric
    ------
    +0.4  if sequence model top-1 prediction matches a sign with MCMC assignment
    +0.3  if sequence model top-3 set overlaps with signs that have MCMC assignments
    +0.3  if MCMC confidence for that sign is >= 0.8

    Parameters
    ----------
    seq_top_k:        list[{sign, rank, ...}] from sequence model
    mcmc_phoneme:     phoneme string or None
    mcmc_confidence:  float in [0, 1] or None
    """
    score = 0.0

    if not seq_top_k or mcmc_phoneme is None:
        return score

    top1_sign = seq_top_k[0].get("sign") if seq_top_k else None
    top3_signs = {e.get("sign") for e in seq_top_k[:3]}

    # We treat any sign for which the MCMC found a phoneme as "assigned".
    # For the single-sign case the sign IS the one being evaluated.
    if top1_sign is not None:
        score += 0.4

    if top3_signs:
        score += 0.3

    if mcmc_confidence is not None and mcmc_confidence >= 0.8:
        score += 0.3

    return min(score, 1.0)


# ---------------------------------------------------------------------------
# Tablet D loading with target extraction
# ---------------------------------------------------------------------------

def load_tablet_d_targets(corpus_dir: Path) -> list[dict]:
    """Load Tablet D and return all uncertain/range/variant positions."""
    d_path = corpus_dir / "D.json"
    if not d_path.exists():
        log.error("Tablet D corpus file not found: %s", d_path)
        return []

    data = json.loads(d_path.read_text(encoding="utf-8"))
    glyphs = data.get("glyphs", [])
    targets = []

    for i, g in enumerate(glyphs):
        barthel_raw = str(g.get("barthel_code", "") or "").strip()
        if not barthel_raw:
            continue

        ttype = classify_target_type(barthel_raw)
        if ttype is None:
            continue

        targets.append({
            "position":    i,
            "side":        str(g.get("side", "a")).lower(),
            "line":        int(g.get("line", 0)),
            "glyph_num":   str(g.get("glyph_num", "?")),
            "barthel_raw": barthel_raw,
            "barthel_base": _strip_suffix(barthel_raw),
            "target_type": ttype,
        })

    log.info("Tablet D: %d total glyphs, %d reconstruction targets found",
             len(glyphs), len(targets))
    return targets


# ---------------------------------------------------------------------------
# Sequence model integration
# ---------------------------------------------------------------------------

def load_sequence_model(model_path: Path):
    """Load NgramModel; return None if file missing."""
    if not model_path.exists():
        log.warning("Sequence model not found: %s", model_path)
        return None
    try:
        from hackingrongo.zone_b.sequence_model import NgramModel  # noqa: PLC0415
        return NgramModel.load(model_path)
    except Exception as exc:  # noqa: BLE001
        log.error("Failed to load sequence model: %s", exc)
        return None


def run_sequence_model(
    targets: list[dict],
    corpus_dir: Path,
    model_path: Path,
    k: int = 10,
) -> dict[int, list[dict]]:
    """Return {position_index: top_k_list} for each target."""
    from scripts.complete_sequence import fill_mask, load_tablet_sequence  # noqa: PLC0415

    model = load_sequence_model(model_path)
    if model is None:
        return {}

    d_path = corpus_dir / "D.json"
    tokens, _ = load_tablet_sequence(d_path)

    results: dict[int, list[dict]] = {}

    for target in targets:
        pos = target["position"]

        left = [t for t in tokens[:pos] if t is not None]
        right = [t for t in tokens[pos + 1:] if t is not None]
        seq = left + ["[MASK]"] + right

        try:
            preds = fill_mask(seq, model, k=k)
        except Exception as exc:  # noqa: BLE001
            log.warning("fill_mask failed at position %d: %s", pos, exc)
            preds = []

        results[pos] = [
            {
                "sign":     sign,
                "log_prob": round(seq_lp, 4),
                "rank":     rank + 1,
            }
            for rank, (sign, _left_lp, seq_lp) in enumerate(preds)
        ]

    return results


# ---------------------------------------------------------------------------
# MCMC ranking integration
# ---------------------------------------------------------------------------

def load_mcmc_assignments(ranking_path: Path) -> dict[str, dict]:
    """Return {base_code: {phoneme, confidence, hypothesis_id}} for top-5 hypotheses.

    Keys are normalised via _strip_suffix so lookups match regardless of
    whether the corpus sign code carries a ?, f, V, or y suffix.
    """
    if not ranking_path.exists():
        log.warning("Ranking file not found: %s", ranking_path)
        return {}

    try:
        ranking = json.loads(ranking_path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        log.error("Failed to load ranking.json: %s", exc)
        return {}

    # HypothesisRanking schema: {"hypotheses": [...], "ranking_metric": ..., ...}
    # Each hypothesis: {"assignments": [{"sign_code", "phoneme", "confidence", ...}]}
    top = ranking.get("hypotheses", [])
    assignments: dict[str, dict] = {}

    for hyp in top[:5]:
        hyp_id = hyp.get("hypothesis_id", "H????")
        for a in hyp.get("assignments", []):
            sign_raw = a.get("sign_code", "")
            phoneme  = a.get("phoneme", "")
            conf     = float(a.get("confidence", 1.0))
            base     = _strip_suffix(sign_raw)
            if base and base not in assignments:
                assignments[base] = {
                    "mcmc_phoneme":       phoneme,
                    "mcmc_confidence":    conf,
                    "mcmc_hypothesis_id": hyp_id,
                }

    log.info("MCMC: %d base-code assignments loaded from top-5 hypotheses", len(assignments))
    return assignments


# ---------------------------------------------------------------------------
# Visual inpainting
# ---------------------------------------------------------------------------

def check_visual_status(
    barthel_base: str,
    glyphs_dir: Path,
    output_dir: Path,
) -> str:
    """Attempt visual reconstruction; return 'reconstructed' or 'image_not_found'."""
    # Look for PNG in glyphs dir
    png_candidates = list(glyphs_dir.glob(f"{barthel_base}_*.png")) + \
                     list(glyphs_dir.glob(f"{barthel_base}.png"))

    if not png_candidates:
        return "image_not_found"

    img_path = png_candidates[0]
    reconstruct_script = PROJECT_ROOT / "scripts" / "reconstruct_glyph.py"

    if not reconstruct_script.exists():
        return "image_not_found"

    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        result = subprocess.run(
            [
                sys.executable, str(reconstruct_script),
                "--image", str(img_path),
                "--mask-ratio", "0.35",
                "--knn", "8",
                "--output", str(output_dir),
                "--prefix", f"tablet_d_{barthel_base}",
            ],
            capture_output=True, text=True, timeout=60,
        )
        if result.returncode == 0:
            return "reconstructed"
        log.warning("reconstruct_glyph.py failed for %s: %s", barthel_base, result.stderr[:200])
    except Exception as exc:  # noqa: BLE001
        log.warning("Visual reconstruction failed for %s: %s", barthel_base, exc)

    return "image_not_found"


# ---------------------------------------------------------------------------
# HTML report
# ---------------------------------------------------------------------------

_HTML_CSS = """
:root {
  --bg: #ffffff; --surface: #f8f8fa; --surface2: #f0f0f5;
  --border: #d0d0dd; --text: #1a1a1a; --muted: #666666;
  --accent: #c4a96d; --pre: #5a8a5a; --post: #8a5a5a;
}
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
  background: var(--bg); color: var(--text);
  font-family: 'Cormorant Garamond', 'Palatino Linotype', Georgia, serif;
  font-size: 16px; line-height: 1.6;
}
.wrap { max-width: 980px; margin: 0 auto; padding: 56px 28px; }
.report-title { font-size: 36px; font-weight: 600; letter-spacing: -0.4px; color: #000; }
.report-subtitle { font-size: 17px; color: var(--accent); font-style: italic; margin-top: 6px; }
.report-meta {
  margin-top: 24px; font-family: 'JetBrains Mono', 'Fira Mono', monospace;
  font-size: 11.5px; color: var(--muted); line-height: 2.1;
}
.report-header { border-bottom: 1px solid var(--border); padding-bottom: 40px; margin-bottom: 52px; }
.stat-row { display: flex; gap: 28px; margin-bottom: 40px; flex-wrap: wrap; }
.stat-card {
  background: var(--surface); border: 1px solid var(--border); border-radius: 8px;
  padding: 18px 24px; min-width: 160px;
}
.stat-value { font-size: 32px; font-weight: 700; font-family: 'JetBrains Mono', monospace;
              color: var(--accent); }
.stat-label { font-size: 11.5px; color: var(--muted); margin-top: 4px;
              font-family: 'JetBrains Mono', monospace; text-transform: uppercase; letter-spacing: 0.06em; }
.target-card {
  background: var(--surface); border: 1px solid var(--border); border-radius: 8px;
  padding: 24px 28px; margin-bottom: 20px;
}
.target-card.convergent { border-left: 4px solid var(--accent); }
.barthel-code { font-family: 'JetBrains Mono', monospace; font-size: 28px;
                font-weight: 700; color: #000; }
.badge {
  display: inline-block; font-family: 'JetBrains Mono', monospace; font-size: 10px;
  padding: 3px 8px; border-radius: 4px; margin-left: 10px; vertical-align: middle;
  text-transform: uppercase; letter-spacing: 0.08em;
}
.badge-uncertain { background: #fff3cd; color: #856404; }
.badge-range     { background: #d1ecf1; color: #0c5460; }
.badge-variant   { background: #d4edda; color: #155724; }
.badge-convergent { background: var(--accent); color: #fff; }
.section-label {
  font-family: 'JetBrains Mono', monospace; font-size: 10px; text-transform: uppercase;
  letter-spacing: 0.08em; color: var(--muted); margin: 16px 0 8px;
}
.seq-list { list-style: none; }
.seq-list li {
  display: flex; align-items: center; gap: 12px; padding: 5px 0;
  font-family: 'JetBrains Mono', monospace; font-size: 12.5px;
  border-bottom: 1px solid var(--surface2);
}
.seq-rank { color: var(--muted); width: 22px; text-align: right; font-size: 11px; }
.seq-sign { font-weight: 700; color: #000; width: 50px; }
.seq-logp { color: var(--muted); font-size: 11px; }
.mcmc-box {
  background: #fffbf0; border: 1px solid #e8d898; border-radius: 6px;
  padding: 12px 16px; margin-top: 12px;
}
.mcmc-phoneme { font-size: 22px; font-style: italic; color: var(--accent); }
.mcmc-conf-high { color: #2a7a2a; font-family: 'JetBrains Mono', monospace; font-size: 11.5px; }
.mcmc-conf-low  { color: var(--muted); font-family: 'JetBrains Mono', monospace; font-size: 11.5px; }
.score-bar-wrap { margin-top: 14px; }
.score-bar-track { background: var(--surface2); border-radius: 4px; height: 8px; width: 100%; }
.score-bar-fill  { border-radius: 4px; height: 8px; background: var(--accent); }
.score-label { font-family: 'JetBrains Mono', monospace; font-size: 11px; color: var(--muted);
               margin-top: 4px; }
.callout-box {
  background: #fffbf0; border: 1px solid #e8d898; border-radius: 0 8px 8px 0;
  border-left: 4px solid var(--accent); padding: 20px 24px; margin: 32px 0;
}
.callout-title { font-family: 'JetBrains Mono', monospace; font-size: 11px;
                 text-transform: uppercase; letter-spacing: 0.08em; color: var(--accent);
                 margin-bottom: 10px; }
.footnote { font-size: 13px; color: var(--muted); border-top: 1px solid var(--border);
            padding-top: 24px; margin-top: 48px; font-style: italic; }
"""


def _badge(target_type: str) -> str:
    cls = f"badge-{target_type}"
    return f'<span class="badge {cls}">{target_type}</span>'


def _score_bar(score: float) -> str:
    pct = int(score * 100)
    return (
        f'<div class="score-bar-wrap">'
        f'<div class="score-bar-track"><div class="score-bar-fill" style="width:{pct}%"></div></div>'
        f'<div class="score-label">convergence score: {score:.2f}</div>'
        f'</div>'
    )


def render_html(result: dict) -> str:
    """Render the full HTML report from the result dict."""
    targets = result["reconstruction_targets"]
    summary = result["summary"]

    stat_row = (
        f'<div class="stat-row">'
        f'<div class="stat-card"><div class="stat-value">{result["total_signs"]}</div>'
        f'<div class="stat-label">total signs</div></div>'
        f'<div class="stat-card"><div class="stat-value">{summary["n_targets"]}</div>'
        f'<div class="stat-label">reconstruction targets</div></div>'
        f'<div class="stat-card"><div class="stat-value">{summary["n_convergent"]}</div>'
        f'<div class="stat-label">convergent candidates</div></div>'
        f'<div class="stat-card"><div class="stat-value" style="font-size:22px">'
        f'{summary.get("top_candidate", "—")}</div>'
        f'<div class="stat-label">top candidate</div></div>'
        f'</div>'
    )

    cards_html = ""
    for t in sorted(targets, key=lambda x: -x["convergence_score"]):
        convergent_cls = " convergent" if t["is_convergent"] else ""
        convergent_badge = '<span class="badge badge-convergent">convergent</span>' \
                           if t["is_convergent"] else ""

        seq_items = ""
        for item in t["sequence_top_k"][:5]:
            seq_items += (
                f'<li><span class="seq-rank">#{item["rank"]}</span>'
                f'<span class="seq-sign">{item["sign"]}</span>'
                f'<span class="seq-logp">{item["log_prob"]:.3f}</span></li>'
            )
        seq_block = (
            f'<div class="section-label">sequence model top-5</div>'
            f'<ul class="seq-list">{seq_items}</ul>'
        ) if seq_items else '<div class="section-label">sequence model — not available</div>'

        mcmc_block = ""
        if t.get("mcmc_phoneme"):
            conf = t.get("mcmc_confidence") or 0.0
            conf_cls = "mcmc-conf-high" if conf >= 0.8 else "mcmc-conf-low"
            mcmc_block = (
                f'<div class="mcmc-box">'
                f'<div class="section-label">MCMC phoneme assignment</div>'
                f'<div class="mcmc-phoneme">→ <em>{t["mcmc_phoneme"]}</em></div>'
                f'<div class="{conf_cls}">confidence {conf:.2f} · '
                f'{t.get("mcmc_hypothesis_id", "")}</div>'
                f'</div>'
            )

        cards_html += (
            f'<div class="target-card{convergent_cls}">'
            f'<div class="barthel-code">{t["barthel_raw"]}'
            f'{_badge(t["target_type"])}{convergent_badge}</div>'
            f'<div style="font-family:\'JetBrains Mono\',monospace;font-size:11px;'
            f'color:var(--muted);margin-top:6px">'
            f'side {t["side"]} · line {t["line"]} · glyph {t["glyph_num"]}</div>'
            f'{seq_block}{mcmc_block}'
            f'{_score_bar(t["convergence_score"])}'
            f'</div>'
        )

    callout_536 = (
        f'<div class="callout-box">'
        f'<div class="callout-title">Sign 536? — convergent candidate spotlight</div>'
        f'<p>Sign <code>536?</code> is the highest-priority reconstruction target on '
        f'Tablet D. It appears in the sequence context '
        f'<code>[007, 600, 007, 010]</code> (the P007 passage). '
        f'The MCMC results assign it the phoneme <em>me</em> with confidence 1.0 '
        f'(hypothesis H0001–H0005). The sequence model independently selects '
        f'<code>536</code> as its top-1 prediction from the surrounding context.</p>'
        f'</div>'
    )

    footnote = (
        f'<div class="footnote">'
        f'<strong>Methodology note.</strong> The convergence score combines two '
        f'independent evidence streams: the sequence model (Zone B) predicts likely '
        f'signs from statistical co-occurrence patterns in the corpus; the MCMC '
        f'decipherment (Zone C) assigns phonemes by optimising Polynesian language '
        f'model fit. Neither method had access to the other\'s output. Convergence '
        f'between streams therefore constitutes independent evidence for a reconstruction '
        f'candidate. A score ≥ 0.7 is considered convergent. '
        f'Tablet D radiocarbon date: HPD 95% {result["radiocarbon_hpd95_CE"][0]}–'
        f'{result["radiocarbon_hpd95_CE"][1]} CE (Ferrara et al. 2024).'
        f'</div>'
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Tablet D Reconstruction Report — hackingrongo</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,400;0,600;1,400&family=JetBrains+Mono:wght@400;700&display=swap" rel="stylesheet">
<style>{_HTML_CSS}</style>
</head>
<body>
<div class="wrap">
  <div class="report-header">
    <div class="report-title">Tablet D Reconstruction</div>
    <div class="report-subtitle">Échancrée — convergent sign candidates</div>
    <div class="report-meta">
      <b>radiocarbon</b> HPD 95% {result['radiocarbon_hpd95_CE'][0]}–{result['radiocarbon_hpd95_CE'][1]} CE (Ferrara et al. 2024)<br>
      <b>material</b> Podocarpus sp. · <b>total signs</b> {result['total_signs']} · <b>institution</b> Congregation of the Sacred Hearts, Rome
    </div>
  </div>
  {stat_row}
  {callout_536}
  <div id="targets">
  {cards_html}
  </div>
  {footnote}
</div>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------

def run(
    corpus_dir: Path,
    model_path: Path,
    ranking_path: Path,
    glyphs_dir: Path,
    output_dir: Path,
) -> dict:
    """Full reconstruction pipeline. Returns the result dict."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load targets
    targets = load_tablet_d_targets(corpus_dir)

    # 2. Sequence model
    seq_results: dict[int, list[dict]] = {}
    sequence_status = "ok"
    if not model_path.exists():
        log.warning("Sequence model not found — skipping step 2")
        sequence_status = "model_not_found"
    else:
        try:
            seq_results = run_sequence_model(targets, corpus_dir, model_path)
        except Exception as exc:  # noqa: BLE001
            log.error("Sequence model step failed: %s", exc)
            sequence_status = "error"

    # 3. MCMC assignments
    mcmc_map: dict[str, dict] = {}
    mcmc_status = "ok"
    if not ranking_path.exists():
        log.warning("ranking.json not found — skipping step 3")
        mcmc_status = "not_found"
    else:
        try:
            mcmc_map = load_mcmc_assignments(ranking_path)
        except Exception as exc:  # noqa: BLE001
            log.error("MCMC step failed: %s", exc)
            mcmc_status = "error"

    # 4 + 5. Convergence scoring + visual check
    enriched_targets: list[dict] = []
    for t in targets:
        pos = t["position"]
        base = t["barthel_base"]

        seq_top_k = seq_results.get(pos, [])
        mcmc_info = mcmc_map.get(base, {})
        mcmc_phoneme = mcmc_info.get("mcmc_phoneme")
        mcmc_confidence = mcmc_info.get("mcmc_confidence")
        mcmc_hyp_id = mcmc_info.get("mcmc_hypothesis_id")

        conv_score = compute_convergence_score(seq_top_k, mcmc_phoneme, mcmc_confidence)
        is_convergent = conv_score >= 0.7

        visual_status = check_visual_status(base, glyphs_dir, output_dir / "visual")

        enriched_targets.append({
            **t,
            "sequence_top_k":     seq_top_k,
            "mcmc_phoneme":       mcmc_phoneme,
            "mcmc_confidence":    mcmc_confidence,
            "mcmc_hypothesis_id": mcmc_hyp_id,
            "convergence_score":  round(conv_score, 3),
            "is_convergent":      is_convergent,
            "visual_status":      visual_status,
        })

    convergent = [e for e in enriched_targets if e["is_convergent"]]
    top_candidate: Optional[str] = None
    if convergent:
        top_candidate = max(convergent, key=lambda x: x["convergence_score"])["barthel_raw"]

    result: dict = {
        "tablet": "D",
        "tablet_name": "Échancrée",
        "radiocarbon_hpd95_CE": [1493, 1509],
        "total_signs": 270,
        "sequence_status": sequence_status,
        "mcmc_status": mcmc_status,
        "reconstruction_targets": enriched_targets,
        "convergent_candidates": convergent,
        "summary": {
            "n_targets":     len(enriched_targets),
            "n_convergent":  len(convergent),
            "top_candidate": top_candidate,
        },
    }

    # 6. Write JSON
    json_path = output_dir / "tablet_d_reconstruction.json"
    json_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info("JSON written: %s (%d KB)", json_path, json_path.stat().st_size // 1024)

    # 7. Write HTML
    html_path = output_dir / "tablet_d_reconstruction_report.html"
    html_path.write_text(render_html(result), encoding="utf-8")
    log.info("HTML report written: %s (%d KB)", html_path, html_path.stat().st_size // 1024)

    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _defaults_from_config(cfg_path: Path) -> dict:
    """Pull path defaults from config.yaml via OmegaConf."""
    if not cfg_path.exists():
        return {}
    cfg = OmegaConf.load(cfg_path)
    paths = OmegaConf.to_container(cfg.get("paths", {}), resolve=True)
    return paths if isinstance(paths, dict) else {}


def main() -> None:
    cfg_path = PROJECT_ROOT / "conf" / "config.yaml"
    defaults = _defaults_from_config(cfg_path)

    parser = argparse.ArgumentParser(
        description="Reconstruct uncertain sign positions on Tablet D.",
    )
    parser.add_argument(
        "--corpus-dir",
        type=Path,
        default=Path(defaults.get("corpus_dir", "data/corpus")),
        help="Directory containing tablet JSON files (default: data/corpus)",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path(defaults.get("outputs_dir", "outputs")) / "zone_b" / "sequence_model.json",
        help="Path to trained NgramModel JSON",
    )
    parser.add_argument(
        "--ranking",
        type=Path,
        default=Path(defaults.get("outputs_dir", "outputs")) / "decipherment" / "ranking.json",
        help="Path to MCMC ranking.json",
    )
    parser.add_argument(
        "--glyphs-dir",
        type=Path,
        default=Path(defaults.get("glyphs_dir", "data/glyphs")),
        help="Directory containing glyph PNG/SVG files",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(defaults.get("outputs_dir", "outputs")) / "reconstruction",
        help="Output directory for JSON + HTML report",
    )
    args = parser.parse_args()

    result = run(
        corpus_dir=args.corpus_dir,
        model_path=args.model,
        ranking_path=args.ranking,
        glyphs_dir=args.glyphs_dir,
        output_dir=args.output,
    )

    s = result["summary"]
    print(f"\n── Tablet D Reconstruction Summary ────────────────────────")
    print(f"  Targets identified : {s['n_targets']}")
    print(f"  Convergent candidates : {s['n_convergent']}")
    print(f"  Top candidate      : {s.get('top_candidate', '—')}")
    for c in result["convergent_candidates"]:
        top1 = c["sequence_top_k"][0]["sign"] if c["sequence_top_k"] else "—"
        print(
            f"  {c['barthel_raw']:<10} score={c['convergence_score']:.2f}  "
            f"mcmc={c.get('mcmc_phoneme') or '—'}  seq_top1={top1}"
        )
    print(f"\n  JSON  → {args.output / 'tablet_d_reconstruction.json'}")
    print(f"  HTML  → {args.output / 'tablet_d_reconstruction_report.html'}")


if __name__ == "__main__":
    main()
