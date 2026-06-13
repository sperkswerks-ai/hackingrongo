"""
scripts/mine_diachronic_substitutions.py

Mine diachronic (pre↔post-contact) sign substitution pairs from parallel
passage slot alignments, corroborated against the contact-partition G² bias.

Method
------
1. Load parallel passages (parallel_variants_auto.json), excluding degenerate
   corpus-wide mega-passages (same caps as find_paradigmatic_pairs).
2. Run passage_alignment.analyze_all_passages with require_cross_stratum=True
   — only passages attested in BOTH the pre-contact (Tablet D) and
   post-contact strata can produce diachronic signal (currently P007, P012).
3. Keep substitution-type DiachronicChanges that are not known allographs:
   the same canonical slot written with different signs across the contact
   boundary is same-phoneme evidence (scribal repertoire change), exactly
   the relation the MCMC equivalence-tie constraint encodes.
4. Corroborate each (pre_sign, post_sign) pair against contact_partition.json:
   * supports    — pre_sign is pre-biased and/or post_sign is post-biased,
                   with neither sign biased the wrong way
   * contradicts — pre_sign is post-biased or post_sign is pre-biased
   * neutral     — neither sign has a significant G² bias record
5. Emit tie_pairs (consumable by run_decipherment's equivalence-tie loader)
   for pairs that are holy-grail candidates (consistent on ≥ 2 post-contact
   tablets) and not contradicted by the contact partition.

Output
------
outputs/analysis/diachronic_substitutions.json

Usage
-----
    python scripts/mine_diachronic_substitutions.py
    python scripts/mine_diachronic_substitutions.py --include-non-holy-grail
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

# Degenerate-passage caps — keep in sync with
# generate_pozdniakov_report.find_paradigmatic_pairs defaults.
MAX_PASSAGE_ATTESTATIONS = 100
MAX_PASSAGE_TABLETS = 8


# ---------------------------------------------------------------------------
# Contact-partition bias lookup
# ---------------------------------------------------------------------------


def _normalize_code(code: str) -> str:
    """Normalise a sign code for cross-source comparison.

    contact_partition.json stores codes without zero padding and with
    compounds space-separated ("52", "200 9"); Barthel-style sources use
    zero-padded codes with modifier suffixes ("052", "303s").  Comparison
    key: leading zeros stripped from each numeric run, lowercase.
    """
    out: list[str] = []
    num = ""
    for ch in code.strip().lower():
        if ch.isdigit():
            num += ch
        else:
            if num:
                out.append(str(int(num)))
                num = ""
            out.append(ch)
    if num:
        out.append(str(int(num)))
    return "".join(out)


def load_contact_bias(path: Path) -> dict[str, dict[str, Any]]:
    """Load contact_partition.json into {normalized_code: record}."""
    if not path.exists():
        log.warning("contact_partition.json not found at %s — corroboration disabled.", path)
        return {}
    records = json.loads(path.read_text(encoding="utf-8"))
    bias: dict[str, dict[str, Any]] = {}
    for rec in records:
        key = _normalize_code(str(rec.get("sign", "")).replace(" ", "."))
        if key:
            bias[key] = rec
    log.info("Contact partition: %d sign bias records loaded.", len(bias))
    return bias


def corroborate(
    pre_sign: str,
    post_sign: str,
    bias: dict[str, dict[str, Any]],
) -> tuple[str, str | None, str | None]:
    """Classify a (pre_sign, post_sign) pair against the G² contact bias.

    Returns (corroboration, pre_sign_bias, post_sign_bias).
    """
    pre_rec = bias.get(_normalize_code(pre_sign))
    post_rec = bias.get(_normalize_code(post_sign))
    pre_bias = pre_rec.get("bias") if pre_rec else None
    post_bias = post_rec.get("bias") if post_rec else None

    # A diachronic substitution predicts: the sign written pre-contact skews
    # pre-biased in corpus-wide frequency; its replacement skews post-biased.
    wrong_way = (pre_bias == "post_biased") or (post_bias == "pre_biased")
    right_way = (pre_bias == "pre_biased") or (post_bias == "post_biased")
    if wrong_way:
        return "contradicts", pre_bias, post_bias
    if right_way:
        return "supports", pre_bias, post_bias
    return "neutral", pre_bias, post_bias


# ---------------------------------------------------------------------------
# Mining
# ---------------------------------------------------------------------------


def mine_pairs(
    parallels_path: Path,
    tablets_path: Path,
    catalog_dir: Path,
    contact_path: Path,
    include_non_holy_grail: bool = False,
) -> dict[str, Any]:
    from omegaconf import OmegaConf

    from hackingrongo.data.catalog import SignCatalog
    from hackingrongo.data.parallels import load_parallel_variants_json
    from hackingrongo.data.passage_alignment import analyze_all_passages

    cfg = OmegaConf.create({
        "paths": {
            "horley_encoding_json": str(catalog_dir / "horley_encoding.json"),
            "allographs_json": str(catalog_dir / "allographs.json"),
            "sign_metadata_json": str(catalog_dir / "sign_metadata.json"),
        }
    })
    catalog = SignCatalog.load(cfg, Path("."))

    tablet_meta: dict[str, dict[str, Any]] = {}
    if tablets_path.exists():
        tablet_meta = json.loads(tablets_path.read_text(encoding="utf-8"))

    passages = load_parallel_variants_json(parallels_path, catalog)

    kept, excluded = [], []
    for p in passages:
        n_att = len(p.variants)
        n_tab = len({v.tablet_id for v in p.variants})
        if n_att > MAX_PASSAGE_ATTESTATIONS or n_tab > MAX_PASSAGE_TABLETS:
            excluded.append({
                "passage_id": p.passage_id,
                "n_attestations": n_att,
                "n_tablets": n_tab,
            })
        else:
            kept.append(p)
    if excluded:
        log.info(
            "Excluded %d degenerate passage(s): %s",
            len(excluded), [e["passage_id"] for e in excluded],
        )

    alignments = analyze_all_passages(
        kept, catalog, tablet_meta, require_cross_stratum=True
    )
    bias = load_contact_bias(contact_path)

    pairs: list[dict[str, Any]] = []
    for alignment in alignments:
        for change in alignment.diachronic_changes:
            if change.change_type != "substitution":
                continue
            if change.is_known_allograph:
                continue  # already merged by get_canonical_id normalisation
            corr, pre_b, post_b = corroborate(
                change.pre_contact_sign, change.post_contact_sign, bias
            )
            pairs.append({
                "pre_sign": change.pre_contact_sign,
                "post_sign": change.post_contact_sign,
                "passage_id": alignment.passage_id,
                "position": change.position,
                "n_tablets_consistent": change.n_tablets_consistent,
                "is_holy_grail": change.is_holy_grail_candidate,
                "crosses_barthel_family": change.crosses_barthel_family,
                "corroboration": corr,
                "pre_sign_bias": pre_b,
                "post_sign_bias": post_b,
            })

    # Tie pairs: same-phoneme constraints safe to feed the MCMC sampler.
    tie_pairs = sorted({
        tuple(sorted((p["pre_sign"], p["post_sign"])))
        for p in pairs
        if (p["is_holy_grail"] or include_non_holy_grail)
        and p["corroboration"] != "contradicts"
    })

    n_supported = sum(1 for p in pairs if p["corroboration"] == "supports")
    n_contradicted = sum(1 for p in pairs if p["corroboration"] == "contradicts")
    return {
        "_schema_version": "1.0",
        "_provenance": (
            "Diachronic substitution pairs mined from cross-stratum parallel "
            "passage slot alignments (passage_alignment.analyze_all_passages, "
            "require_cross_stratum=True), corroborated against "
            "contact_partition.json G² sign bias. tie_pairs feed the MCMC "
            "equivalence-tie constraint in run_decipherment.py."
        ),
        "generated": datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "parameters": {
            "max_passage_attestations": MAX_PASSAGE_ATTESTATIONS,
            "max_passage_tablets": MAX_PASSAGE_TABLETS,
            "include_non_holy_grail": include_non_holy_grail,
            "tie_pair_criteria": "is_holy_grail AND corroboration != contradicts"
            if not include_non_holy_grail
            else "corroboration != contradicts",
        },
        "n_cross_stratum_passages": len(alignments),
        "cross_stratum_passage_ids": [a.passage_id for a in alignments],
        "excluded_passages": excluded,
        "n_pairs": len(pairs),
        "n_supported": n_supported,
        "n_contradicted": n_contradicted,
        "n_tie_pairs": len(tie_pairs),
        "pairs": pairs,
        "tie_pairs": [list(t) for t in tie_pairs],
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Mine pre↔post-contact sign substitution pairs from parallel passages."
    )
    p.add_argument(
        "--parallels", type=Path,
        default=PROJECT_ROOT / "data" / "parallels" / "parallel_variants_auto.json",
    )
    p.add_argument(
        "--tablets", type=Path,
        default=PROJECT_ROOT / "data" / "metadata" / "tablets.json",
    )
    p.add_argument(
        "--catalog-dir", type=Path,
        default=PROJECT_ROOT / "data" / "catalog",
    )
    p.add_argument(
        "--contact", type=Path,
        default=PROJECT_ROOT / "outputs" / "contact_partition.json",
    )
    p.add_argument(
        "--output", type=Path,
        default=PROJECT_ROOT / "outputs" / "analysis" / "diachronic_substitutions.json",
    )
    p.add_argument(
        "--include-non-holy-grail", action="store_true",
        help="Also emit tie pairs for single-tablet substitutions (default: holy-grail only).",
    )
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    result = mine_pairs(
        parallels_path=args.parallels,
        tablets_path=args.tablets,
        catalog_dir=args.catalog_dir,
        contact_path=args.contact,
        include_non_holy_grail=args.include_non_holy_grail,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    log.info(
        "Mined %d substitution pair(s) from %d cross-stratum passage(s): "
        "%d supported, %d contradicted, %d tie pair(s) → %s",
        result["n_pairs"], result["n_cross_stratum_passages"],
        result["n_supported"], result["n_contradicted"],
        result["n_tie_pairs"], args.output,
    )
    for tp in result["tie_pairs"]:
        log.info("  TIE %s ↔ %s", tp[0], tp[1])


if __name__ == "__main__":
    main()
