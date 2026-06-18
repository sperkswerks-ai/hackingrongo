"""
scripts/run_sign_fingerprint.py — distributional sign-role "service discovery".

Classifies each frequency-core sign's functional role from its distributional
fingerprint alone (betweenness, pagerank, positional entropy, neighbour
diversity, frequency, slot predictability, passage-anchor score), maps it to a
``SignClass`` with an interpretable rule, and validates by recomputing roles
independently on the pre- and post-contact strata.

Headline metric: **role_stability** — the fraction of signs whose role survives
the contact boundary.  A role that is stable across the boundary is far more
trustworthy than one that flips.

Outputs
-------
outputs/network/sign_fingerprint.json
outputs/network/sign_fingerprint_report.html

Usage
-----
    python scripts/run_sign_fingerprint.py
    python scripts/run_sign_fingerprint.py --min-freq 5 --anchor-thresh 0.5
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter
from datetime import datetime, timezone
from html import escape as _esc
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from hackingrongo.zone_b.sign_fingerprint import (   # noqa: E402
    assign_roles,
    compute_features,
    diachronic_stability,
    load_glyph_records,
    load_passage_boundaries,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

CAVEAT = (
    "These roles are DISTRIBUTIONAL HYPOTHESES, not confirmed linguistic "
    "functions. A sign that behaves like a determinative (high betweenness, "
    "broad neighbour diversity) is not proven to be one — the fingerprint "
    "describes how the sign is used, not what it means."
)


def _load_canon():
    try:
        from omegaconf import OmegaConf
        from hackingrongo.data.catalog import SignCatalog
        cat_dir = PROJECT_ROOT / "data" / "catalog"
        cfg = OmegaConf.create({"paths": {
            "horley_encoding_json": str(cat_dir / "horley_encoding.json"),
            "allographs_json":      str(cat_dir / "allographs.json"),
            "sign_metadata_json":   str(cat_dir / "sign_metadata.json"),
        }})
        return SignCatalog.load(cfg, PROJECT_ROOT).get_canonical_id
    except Exception as exc:
        log.warning("SignCatalog unavailable (%s) — using raw codes.", exc)
        return lambda c: c


def run(corpus_dir: Path, variants_path: Path, min_freq: int, anchor_thresh: float) -> dict[str, Any]:
    canon = _load_canon()
    records = load_glyph_records(corpus_dir, canon)
    boundaries = load_passage_boundaries(variants_path)
    log.info("Loaded %d glyph records, %d passage-boundary positions.",
             len(records), len(boundaries))

    feats, freq = compute_features(records, boundaries, min_freq=min_freq)
    roles, stats = assign_roles(feats, freq, anchor_thresh=anchor_thresh)
    log.info("Classified %d core signs (frequency >= %d).", len(roles), min_freq)

    # Diachronic validation on the cleanly-dated strata only.
    pre  = [r for r in records if r["stratum"] == "pre_contact"]
    post = [r for r in records if r["stratum"] == "post_contact"]
    pf, pfr = compute_features(pre,  boundaries, min_freq=min_freq)
    qf, qfr = compute_features(post, boundaries, min_freq=min_freq)
    pre_roles,  _ = assign_roles(pf, pfr, anchor_thresh=anchor_thresh)
    post_roles, _ = assign_roles(qf, qfr, anchor_thresh=anchor_thresh)
    stability = diachronic_stability(pre_roles, post_roles)
    log.info("Diachronic role_stability = %s over %d signs in both strata.",
             f"{stability['role_stability']:.3f}" if stability["role_stability"] is not None else "N/A",
             stability["n_signs_in_both_strata"])

    role_counts = Counter(fp.role for fp in roles.values())
    subtype_counts = Counter(fp.subtype for fp in roles.values() if fp.subtype)
    signs_sorted = sorted(roles.values(), key=lambda fp: -fp.features["betweenness"])

    # ---- Validation guardrails: a taxogram is a *finding* only if corroborated
    #      by an independent signal, not by the distributional rule alone. ----
    stable_set = set(stability.get("stable_signs", []))
    taxo_validation = []
    for code, fp in sorted(roles.items()):
        if fp.role != "taxogram":
            continue
        # rule is e.g. "determinative:proclitic" or "...+anchor"; take the side only.
        side = fp.rule.split(":", 1)[1].split("+", 1)[0] if ":" in fp.rule else None
        in_compound = (":" in code) or ("." in code) or ("-" in code)
        diachronically_stable = code in stable_set
        corroborated = bool(in_compound or diachronically_stable)
        taxo_validation.append({
            "code": code,
            "side": side,
            "frequency": fp.frequency,
            "direction_skew": round(fp.features["direction_skew"], 4),
            "in_compound": in_compound,
            "diachronically_stable": diachronically_stable,
            "corroborated": corroborated,
        })
    n_taxo = len(taxo_validation)
    n_corrob = sum(t["corroborated"] for t in taxo_validation)
    log.info("Taxogram candidates: %d (%d corroborated by an independent signal).",
             n_taxo, n_corrob)

    return {
        "_schema_version": "1.0",
        "generated": datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "caveat": CAVEAT,
        "parameters": {"min_freq": min_freq, "anchor_thresh": anchor_thresh},
        "thresholds": {k: round(float(v), 6) for k, v in stats.items()},
        "n_signs_classified": len(roles),
        "role_counts": dict(role_counts),
        "subtype_counts": dict(subtype_counts),
        "diachronic": stability,
        "taxogram_validation": {
            "n_taxograms": n_taxo,
            "n_corroborated": n_corrob,
            "note": ("A taxogram is corroborated when it is also a compound/ligature "
                     "member OR keeps its role across the contact boundary. "
                     "Un-corroborated candidates are distributional noise until "
                     "another signal supports them."),
            "candidates": taxo_validation,
        },
        # Consumed by the Zone C MCMC loader to down-weight/exclude taxograms:
        "taxogram_signs": sorted(s for s, fp in roles.items() if fp.role == "taxogram"),
        "signs": [fp.as_dict() for fp in signs_sorted],
    }


# ---------------------------------------------------------------------------
# HTML report (dark theme)
# ---------------------------------------------------------------------------

_CSS = """
body{background:#14110f;color:#e8e2d8;font-family:'JetBrains Mono',monospace;margin:0;padding:32px;line-height:1.5}
h1{font-size:20px;color:#e8b04b;border-bottom:2px solid #5a4a36;padding-bottom:8px}
h2{font-size:15px;color:#cda76a;margin-top:26px}
.sub{color:#9a8e7c;font-size:12px;margin:4px 0 16px}
.headline{font-size:34px;color:#e8b04b;font-weight:700}
.card{display:inline-block;background:#1c1814;border:1px solid #5a4a36;border-radius:6px;padding:14px 20px;margin:6px 10px 6px 0;vertical-align:top}
.caveat{background:#3a2418;border-left:3px solid #c8702f;padding:12px 16px;font-size:12px;color:#e8c7a8;margin:14px 0}
table{border-collapse:collapse;width:100%;font-size:11.5px;margin-top:8px}
th,td{text-align:left;padding:4px 8px;border-bottom:1px solid #2e2820}
th{color:#cda76a;border-bottom:1px solid #5a4a36}
.mono{font-variant-ligatures:none}
.b{display:inline-block;padding:1px 6px;border-radius:3px;font-size:10.5px}
.b-tax{background:#2a2438;color:#b79be0}.b-log{background:#1f3d2a;color:#7fdf9f}
.b-pho{background:#23303d;color:#8fc0e0}.b-anc{background:#3a2a1a;color:#d8a060}
"""


def _badge(role: str) -> str:
    cls = {"taxogram": "b-tax", "logogram": "b-log", "phonetic": "b-pho"}.get(role, "")
    return f'<span class="b {cls}">{role}</span>'


def render_html(r: dict[str, Any]) -> str:
    d = r["diachronic"]
    stab = d["role_stability"]
    stab_str = f"{stab*100:.0f}%" if stab is not None else "N/A"
    rc = r["role_counts"]
    rows = []
    for fp in r["signs"][:80]:
        f = fp["features"]
        sub = f' <span class="b b-anc">{fp["subtype"]}</span>' if fp["subtype"] else ""
        rows.append(
            f'<tr><td class="mono">{_esc(fp["code"])}</td><td>{fp["frequency"]}</td>'
            f'<td>{_badge(fp["role"])}{sub}</td>'
            f'<td>{f["betweenness"]:.4f}</td><td>{f["neighbor_diversity"]:.2f}</td>'
            f'<td>{f.get("direction_skew", 0.0):+.2f}</td>'
            f'<td>{f["positional_entropy"]:.2f}</td><td>{f["slot_predictability"]:.2f}</td>'
            f'<td>{f["passage_anchor_score"]:.2f}</td><td class="mono">{_esc(fp["rule"])}</td></tr>'
        )

    tv = r.get("taxogram_validation", {})
    if tv.get("candidates"):
        _corrob_cell = '<b style="color:#7fdf9f">corroborated</b>'
        _uncon_cell = '<span style="color:#c8702f">unconfirmed</span>'
        tax_rows = "".join(
            f'<tr><td class="mono">{_esc(c["code"])}</td>'
            f'<td>{_esc(c["side"] or "—")}</td><td>{c["frequency"]}</td>'
            f'<td>{c["direction_skew"]:+.2f}</td>'
            f'<td>{"✓" if c["in_compound"] else "·"}</td>'
            f'<td>{"✓" if c["diachronically_stable"] else "·"}</td>'
            f'<td>{_corrob_cell if c["corroborated"] else _uncon_cell}</td></tr>'
            for c in tv["candidates"]
        )
        taxo_block = (
            f'<h2>Taxogram candidates &amp; corroboration</h2>'
            f'<div class="sub">{_esc(tv["note"])} '
            f'{tv["n_corroborated"]}/{tv["n_taxograms"]} candidates are corroborated by an independent signal.</div>'
            f'<table><thead><tr><th>sign</th><th>side</th><th>freq</th><th>dir-skew</th>'
            f'<th>compound?</th><th>diachronically stable?</th><th>verdict</th></tr></thead>'
            f'<tbody>{tax_rows}</tbody></table>'
        )
    else:
        taxo_block = (
            '<h2>Taxogram candidates</h2>'
            '<div class="sub">No sign meets the determinative criterion '
            '(strong directional binding on a reliably-attested sign). '
            'A null here is itself a result: no distributionally-detectable classifiers.</div>'
        )
    changes = "".join(
        f'<tr><td class="mono">{_esc(c["code"])}</td><td>{_badge(c["pre_role"])}</td>'
        f'<td>{_badge(c["post_role"])}</td></tr>' for c in d["role_changes"]
    ) or '<tr><td colspan="3" style="color:#9a8e7c">none</td></tr>'

    return f"""<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">
<title>Rongorongo — Sign Functional Fingerprint</title><style>{_CSS}</style></head>
<body>
<h1>Sign Functional Fingerprint — Distributional Service Discovery</h1>
<div class="sub">Functional role inferred from each sign's behavioural fingerprint ·
Generated {_esc(r["generated"])}</div>

<div class="card"><div class="sub">Diachronic role stability (headline)</div>
<div class="headline">{stab_str}</div>
<div class="sub">{d["n_stable"]}/{d["n_signs_in_both_strata"]} signs keep their role across the contact boundary</div></div>
<div class="card"><div class="sub">Signs classified (freq ≥ {r["parameters"]["min_freq"]})</div>
<div class="headline">{r["n_signs_classified"]}</div>
<div class="sub">taxogram {rc.get("taxogram",0)} · logogram {rc.get("logogram",0)} · phonetic {rc.get("phonetic",0)}</div></div>

<div class="caveat"><b>Caveat.</b> {_esc(r["caveat"])}</div>

<h2>Roles that change across the contact boundary</h2>
<div class="sub">Flagged as especially interesting — possible functional correlates of the
P007/P012 diachronic substitutions. A role that flips is a weaker hypothesis than one that holds.</div>
<table><thead><tr><th>sign</th><th>pre-contact role</th><th>post-contact role</th></tr></thead>
<tbody>{changes}</tbody></table>

{taxo_block}

<h2>Per-sign fingerprints (top 80 by betweenness)</h2>
<div class="sub">Every assignment shows the feature values that produced it — auditable, not asserted.
btwn = betweenness · ndiv = neighbour diversity · dir = direction skew (+proclitic / −postclitic) ·
pent = positional entropy · slot = slot predictability · anc = passage-anchor.</div>
<table><thead><tr><th>sign</th><th>freq</th><th>role</th><th>btwn</th><th>ndiv</th><th>dir</th><th>pent</th><th>slot</th><th>anc</th><th>rule</th></tr></thead>
<tbody>{"".join(rows)}</tbody></table>
</body></html>"""


def main() -> None:
    p = argparse.ArgumentParser(description="Distributional sign-role classification with diachronic validation.")
    p.add_argument("--corpus-dir", type=Path, default=PROJECT_ROOT / "data" / "corpus")
    p.add_argument("--variants-file", type=Path,
                   default=PROJECT_ROOT / "data" / "parallels" / "parallel_variants_auto.json")
    p.add_argument("--min-freq", type=int, default=5,
                   help="Frequency-core threshold; signs below this are not classified (default 5).")
    p.add_argument("--anchor-thresh", type=float, default=0.5,
                   help="passage_anchor_score >= this → subtype 'anchor' (default 0.5).")
    p.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs" / "network")
    args = p.parse_args()

    result = run(args.corpus_dir, args.variants_file, args.min_freq, args.anchor_thresh)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "sign_fingerprint.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    (args.output_dir / "sign_fingerprint_report.html").write_text(
        render_html(result), encoding="utf-8")

    rc = result["role_counts"]
    d = result["diachronic"]
    print(f"Classified {result['n_signs_classified']} signs: "
          f"taxogram={rc.get('taxogram',0)} logogram={rc.get('logogram',0)} phonetic={rc.get('phonetic',0)}")
    print(f"Diachronic role_stability: "
          f"{d['role_stability']:.3f} ({d['n_stable']}/{d['n_signs_in_both_strata']})"
          if d['role_stability'] is not None else "Diachronic role_stability: N/A (no shared signs)")
    if d["role_changes"]:
        print("Role changes across boundary:", [c["code"] for c in d["role_changes"]])
    print(f"→ {args.output_dir / 'sign_fingerprint.json'}")
    print(f"→ {args.output_dir / 'sign_fingerprint_report.html'}")


if __name__ == "__main__":
    main()
