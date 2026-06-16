"""
scripts/partition_compounds.py

Partition the corpus's compound/ligature sign codes by cross-tablet recurrence,
WITHOUT decomposing or destroying any original code.

Rationale
---------
After allograph normalization (SignCatalog.get_canonical_id), the canonical
inventory still contains ~190 *compound* codes — ligatures ('006:700'),
multi-sign tokens ('003a;042t'), and Barthel range/uncertainty notations
('(1-2)!').  These are not single base signs, so they neither belong in the
frequency-core inventory by default nor should they be silently decomposed.

Partition (recurrence-based)
----------------------------
* STRUCTURAL compound — attested >= MIN_OCC times across >= MIN_TABLETS tablets.
  Treated as an atomic token and KEPT in the frequency-core inventory.
  Genuine sign-units among these (ligatures / multi-sign) are flagged as
  crib/anchor candidates; range-notation forms are NOT (they are transcription
  artifacts, not signs).
* SINGLETON / rare compound — everything else.  Stays in the data but is
  EXCLUDED from the frequency-core inventory.  NOT decomposed, NOT deleted.

Non-destructive: this is a read-only analysis.  Original compound codes are
preserved verbatim in the output; nothing in data/ is modified.

Output
------
outputs/analysis/compound_partition.json
outputs/analysis/compound_partition.html   (new report section)

Usage
-----
    python scripts/partition_compounds.py
    python scripts/partition_compounds.py --min-occ 3 --min-tablets 2
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from html import escape as _esc
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# A canonical code is a "compound" if it contains any ligature / range /
# multi-sign separator (i.e. it is not a single base Barthel sign).
_COMPOUND_RE = re.compile(r"[:.\-();]")
# Sub-classification of compound *kind*.
_LIGATURE_RE = re.compile(r"[:.]")          # sign fusion: 006:700, 600.390
_MULTISIGN_RE = re.compile(r";")            # multiple signs: 003a;042t
_RANGE_RE = re.compile(r"\(.*[-].*\)|^\d+-\d+")  # Barthel range/uncertainty: (1-2)!


def _classify_kind(code: str) -> str:
    """ligature | multi_sign | range_notation | compound (fallback)."""
    if _LIGATURE_RE.search(code):
        return "ligature"
    if _MULTISIGN_RE.search(code):
        return "multi_sign"
    if _RANGE_RE.search(code):
        return "range_notation"
    return "compound"


def _load_catalog():
    from omegaconf import OmegaConf
    from hackingrongo.data.catalog import SignCatalog
    cat_dir = PROJECT_ROOT / "data" / "catalog"
    cfg = OmegaConf.create({"paths": {
        "horley_encoding_json": str(cat_dir / "horley_encoding.json"),
        "allographs_json":      str(cat_dir / "allographs.json"),
        "sign_metadata_json":   str(cat_dir / "sign_metadata.json"),
    }})
    return SignCatalog.load(cfg, PROJECT_ROOT)


def partition_compounds(
    corpus_dir: Path,
    min_occ: int = 3,
    min_tablets: int = 2,
) -> dict[str, Any]:
    catalog = _load_catalog()
    canon = catalog.get_canonical_id

    occ: Counter[str] = Counter()
    tablets: dict[str, set[str]] = defaultdict(set)

    for path in sorted(corpus_dir.glob("[A-Z].json")):
        tablet_id = path.stem
        try:
            glyphs = json.loads(path.read_text(encoding="utf-8")).get("glyphs", [])
        except Exception:
            continue
        for g in glyphs:
            raw = g.get("barthel_code")
            if not raw:
                continue
            code = canon(str(raw))            # canonical, compounds preserved atomic
            if _COMPOUND_RE.search(code):
                occ[code] += 1
                tablets[code].add(tablet_id)

    def _record(code: str) -> dict[str, Any]:
        kind = _classify_kind(code)
        is_structural = occ[code] >= min_occ and len(tablets[code]) >= min_tablets
        return {
            "code":          code,                       # original, never altered
            "occurrences":   occ[code],
            "n_tablets":     len(tablets[code]),
            "tablets":       sorted(tablets[code]),
            "kind":          kind,
            # Crib/anchor only for genuine sign-units, never range notation.
            "crib_anchor_candidate": bool(
                is_structural and kind in ("ligature", "multi_sign")
            ),
        }

    all_codes = sorted(occ, key=lambda c: (-occ[c], c))
    structural = [_record(c) for c in all_codes
                  if occ[c] >= min_occ and len(tablets[c]) >= min_tablets]
    singleton  = [_record(c) for c in all_codes
                  if not (occ[c] >= min_occ and len(tablets[c]) >= min_tablets)]

    crib_candidates = [r["code"] for r in structural if r["crib_anchor_candidate"]]

    return {
        "_schema_version": "1.0",
        "_note": (
            "Non-destructive recurrence partition of canonical compound codes. "
            "Original codes preserved verbatim; nothing decomposed or deleted. "
            "Structural compounds are kept as atomic tokens in the frequency-core "
            "inventory; singletons stay in the data but are excluded from the core."
        ),
        "generated": datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "criteria": {
            "structural_min_occurrences": min_occ,
            "structural_min_tablets": min_tablets,
        },
        "summary": {
            "n_compound_types":        len(all_codes),
            "n_structural":            len(structural),
            "n_singleton":             len(singleton),
            "n_crib_anchor_candidates": len(crib_candidates),
        },
        "crib_anchor_candidates": crib_candidates,
        # Lists the frequency-core builder (option A) should consume:
        "frequency_core_keep_atomic":  [r["code"] for r in structural],
        "frequency_core_exclude":      [r["code"] for r in singleton],
        "structural": structural,
        "singleton":  singleton,
    }


# ---------------------------------------------------------------------------
# HTML report (new section)
# ---------------------------------------------------------------------------

_CSS = """
body{background:#14110f;color:#e8e2d8;font-family:'JetBrains Mono',monospace;margin:0;padding:32px;line-height:1.5}
h1{font-size:20px;color:#e8b04b;border-bottom:2px solid #5a4a36;padding-bottom:8px}
h2{font-size:15px;color:#cda76a;margin-top:28px}
.sub{color:#9a8e7c;font-size:12px;margin:4px 0 18px}
table{border-collapse:collapse;width:100%;font-size:12px;margin-top:8px}
th,td{text-align:left;padding:5px 10px;border-bottom:1px solid #2e2820}
th{color:#cda76a;border-bottom:1px solid #5a4a36}
.mono{font-variant-ligatures:none}
.badge{display:inline-block;padding:1px 7px;border-radius:3px;font-size:11px}
.b-crib{background:#1f3d1f;color:#7fdf7f}
.b-lig{background:#2a2438;color:#b79be0}
.b-range{background:#3a2a1a;color:#d8a060}
.note{background:#1c1814;border-left:3px solid #5a4a36;padding:10px 14px;font-size:12px;color:#c9bda8;margin:14px 0}
"""


def _rows(records: list[dict[str, Any]]) -> str:
    out = []
    for r in records:
        kc = {"ligature": "b-lig", "multi_sign": "b-lig",
              "range_notation": "b-range"}.get(r["kind"], "")
        crib = '<span class="badge b-crib">crib/anchor candidate</span>' if r["crib_anchor_candidate"] else ""
        out.append(
            f'<tr><td class="mono">{_esc(r["code"])}</td>'
            f'<td>{r["occurrences"]}</td><td>{r["n_tablets"]}</td>'
            f'<td class="mono">{_esc(",".join(r["tablets"]))}</td>'
            f'<td><span class="badge {kc}">{r["kind"]}</span> {crib}</td></tr>'
        )
    return "".join(out)


def render_html(result: dict[str, Any]) -> str:
    s = result["summary"]
    c = result["criteria"]
    return f"""<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">
<title>Rongorongo — Compound Code Partition</title><style>{_CSS}</style></head>
<body>
<h1>Compound Code Partition</h1>
<div class="sub">Canonical compound/ligature codes partitioned by cross-tablet recurrence ·
Generated {_esc(result["generated"])}</div>
<div class="note">{_esc(result["_note"])}<br>
Structural threshold: occurrences &ge; {c["structural_min_occurrences"]} across &ge; {c["structural_min_tablets"]} tablets.
{s["n_compound_types"]} compound types → <b>{s["n_structural"]} structural</b>,
{s["n_singleton"]} singleton · {s["n_crib_anchor_candidates"]} crib/anchor candidates.</div>

<h2>Structural compounds — kept as atomic tokens ({s["n_structural"]})</h2>
<div class="sub">Recurrent across tablets. Genuine sign-units (ligature / multi-sign) flagged as
crib/anchor candidates; range-notation forms are transcription artifacts, not signs.</div>
<table><thead><tr><th>code</th><th>occ</th><th>tablets</th><th>which tablets</th><th>kind</th></tr></thead>
<tbody>{_rows(result["structural"])}</tbody></table>

<h2>Singleton / rare compounds — excluded from frequency core ({s["n_singleton"]})</h2>
<div class="sub">Retained in the data, NOT decomposed, but excluded from the frequency-core
inventory. Showing first 60 of {s["n_singleton"]}.</div>
<table><thead><tr><th>code</th><th>occ</th><th>tablets</th><th>which tablets</th><th>kind</th></tr></thead>
<tbody>{_rows(result["singleton"][:60])}</tbody></table>
</body></html>"""


def main() -> None:
    p = argparse.ArgumentParser(description="Partition canonical compound codes by cross-tablet recurrence.")
    p.add_argument("--corpus-dir", type=Path, default=PROJECT_ROOT / "data" / "corpus")
    p.add_argument("--min-occ", type=int, default=3, help="Min occurrences for 'structural' (default 3).")
    p.add_argument("--min-tablets", type=int, default=2, help="Min distinct tablets for 'structural' (default 2).")
    p.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs" / "analysis")
    args = p.parse_args()

    result = partition_compounds(args.corpus_dir, args.min_occ, args.min_tablets)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "compound_partition.json"
    html_path = args.output_dir / "compound_partition.html"
    json_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    html_path.write_text(render_html(result), encoding="utf-8")

    s = result["summary"]
    print(f"Compound partition: {s['n_compound_types']} types → "
          f"{s['n_structural']} structural, {s['n_singleton']} singleton "
          f"({s['n_crib_anchor_candidates']} crib/anchor candidates).")
    print("Structural compounds:")
    for r in result["structural"]:
        flag = "  ★ crib/anchor" if r["crib_anchor_candidate"] else ""
        print(f"  {r['code']:<18} occ={r['occurrences']:>3} tablets={r['n_tablets']} "
              f"[{r['kind']}]{flag}")
    print(f"→ {json_path}")
    print(f"→ {html_path}")


if __name__ == "__main__":
    main()
