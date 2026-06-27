"""
scripts/run_sign_typology.py — independent, falsifiable test of whether the sign
inventory splits into phonogram-like and logogram-like populations.

Run BEFORE any mixed decoding. Emits outputs/analysis/sign_typology.json and, ONLY
if the split is statistically real, a frozen data/catalog/sign_type_map.json that
the mixed decoder may consume but never alter.
"""
from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
import sys
sys.path.insert(0, str(PROJECT_ROOT))

from hackingrongo.zone_b.sign_typology import run  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)


def _load_canon():
    try:
        from omegaconf import OmegaConf
        from hackingrongo.data.catalog import SignCatalog
        cd = PROJECT_ROOT / "data" / "catalog"
        cfg = OmegaConf.create({"paths": {
            "horley_encoding_json": str(cd / "horley_encoding.json"),
            "allographs_json":      str(cd / "allographs.json"),
            "sign_metadata_json":   str(cd / "sign_metadata.json")}})
        return SignCatalog.load(cfg, PROJECT_ROOT).get_canonical_id
    except Exception as exc:
        log.warning("SignCatalog unavailable (%s) — raw codes.", exc)
        return lambda c: c


def main() -> None:
    p = argparse.ArgumentParser(description="Falsifiable phonogram/logogram bimodality test.")
    p.add_argument("--corpus-dir", type=Path, default=PROJECT_ROOT / "data" / "corpus")
    p.add_argument("--min-freq", type=int, default=5)
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--prob-threshold", type=float, default=0.8)
    p.add_argument("--out", type=Path, default=PROJECT_ROOT / "outputs" / "analysis" / "sign_typology.json")
    p.add_argument("--type-map-out", type=Path, default=PROJECT_ROOT / "data" / "catalog" / "sign_type_map.json")
    args = p.parse_args()

    res = run(args.corpus_dir, _load_canon(), min_freq=args.min_freq,
              alpha=args.alpha, prob_threshold=args.prob_threshold)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(asdict(res), indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\nDip test p={res.dip_p} | BC={res.bimodality_coefficient} | is_bimodal={res.is_bimodal}")
    print(res.verdict)

    if res.is_bimodal:
        frozen = {"_provenance": "hackingrongo.zone_b.sign_typology — FROZEN; consumed by the "
                                 "mixed decoder, never altered downstream.",
                  "_verdict": res.verdict, "sign_type_map": res.sign_type_map}
        args.type_map_out.write_text(json.dumps(frozen, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"→ frozen sign_type_map written: {args.type_map_out}")
    else:
        print("→ NO sign_type_map written (no defensible split). This is a recorded null result.")
    print(f"→ {args.out}")


if __name__ == "__main__":
    main()
