"""
run_freq_match.py — Frequency-language match analysis.

Computes Zipf exponent comparison, Spearman rank correlation, and
Chi-squared goodness-of-fit between the rongorongo sign frequency
distribution and each Polynesian language-model phoneme distribution.

Optionally, if a QUBO or MCMC ranking result is provided via
``--phoneme-map``, per-assignment correlation statistics are also
computed.

Usage
-----
    python scripts/run_freq_match.py \\
        --corpus-dir data/corpus \\
        --lm-dir     data/language_models \\
        --output     outputs/zone_b/freq_match.json

    # With phoneme map from top MCMC hypothesis:
    python scripts/run_freq_match.py \\
        --phoneme-map outputs/decipherment/ranking.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from hackingrongo.zone_b.entropy import frequency_language_match  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)


def _load_phoneme_map(path: Path) -> dict[str, str] | None:
    """Load a sign→phoneme map from a ranking.json, hypothesis JSON, or qubo_result.json."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        log.warning("Could not load phoneme map from %s: %s", path, exc)
        return None

    # ranking.json → top hypothesis
    if "hypotheses" in data:
        hyps = data["hypotheses"]
        if hyps and "assignments" in hyps[0]:
            return {
                a["sign_code"]: a["phoneme"]
                for a in hyps[0]["assignments"]
            }

    # qubo_result.json
    if "phoneme_assignments" in data:
        return {
            a["sign"]: a["phoneme"]
            for a in data["phoneme_assignments"]
        }

    # individual hypothesis JSON
    if "assignments" in data:
        return {
            a["sign_code"]: a["phoneme"]
            for a in data["assignments"]
        }

    log.warning("Could not parse phoneme map from %s (unexpected schema).", path)
    return None


def main() -> None:
    p = argparse.ArgumentParser(
        description="Frequency-language match analysis for rongorongo.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--corpus-dir",   type=Path, default=None)
    p.add_argument("--lm-dir",       type=Path, default=None)
    p.add_argument("--phoneme-map",  type=Path, default=None, metavar="JSON",
                   help="ranking.json, hypothesis JSON, or qubo_result.json with assignments.")
    p.add_argument("--output",       type=Path, default=None)
    args = p.parse_args()

    # Resolve defaults from config
    corpus_dir = args.corpus_dir
    lm_dir     = args.lm_dir
    output     = args.output
    if corpus_dir is None or lm_dir is None or output is None:
        try:
            from omegaconf import OmegaConf
            cfg = OmegaConf.load(PROJECT_ROOT / "conf" / "config.yaml")
            if corpus_dir is None:
                corpus_dir = PROJECT_ROOT / cfg.paths.corpus_dir
            if lm_dir is None:
                lm_dir = PROJECT_ROOT / "data" / "language_models"
            if output is None:
                output = PROJECT_ROOT / cfg.paths.outputs_dir / "zone_b" / "freq_match.json"
        except Exception:
            pass

    if corpus_dir is None or not corpus_dir.exists():
        log.error("Corpus directory not found. Pass --corpus-dir.")
        sys.exit(1)
    if lm_dir is None or not lm_dir.exists():
        log.error("LM directory not found. Pass --lm-dir.")
        sys.exit(1)
    assert output is not None

    phoneme_map: dict[str, str] | None = None
    if args.phoneme_map:
        phoneme_map = _load_phoneme_map(args.phoneme_map)
        if phoneme_map:
            log.info("Phoneme map loaded: %d assignments from %s", len(phoneme_map), args.phoneme_map)
    else:
        # Auto-discover: try ranking.json in decipherment outputs
        auto_paths = [
            PROJECT_ROOT / "outputs" / "decipherment" / "ranking.json",
            PROJECT_ROOT / "outputs" / "decipherment" / "qubo_result.json",
        ]
        for ap in auto_paths:
            if ap.exists():
                phoneme_map = _load_phoneme_map(ap)
                if phoneme_map:
                    log.info("Auto-discovered phoneme map from %s (%d assignments)", ap, len(phoneme_map))
                    break

    result = frequency_language_match(
        corpus_dir=corpus_dir,
        lm_dir=lm_dir,
        phoneme_map=phoneme_map,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info("Frequency-language match written to %s", output)

    # Print summary
    zipf = result.get("zipf_alpha_signs")
    if zipf:
        log.info("Sign Zipf α = %.3f", zipf)
    best_s = result.get("best_lm_by_spearman")
    if best_s:
        rho = (result.get("spearman_rho_per_lm") or {}).get(best_s)
        log.info("Best Spearman match: %s (ρ=%.3f)", best_s, rho or 0)
    best_c = result.get("best_lm_by_chi2_p")
    if best_c:
        pval = (result.get("chi2_p_value_per_lm") or {}).get(best_c)
        log.info("Best χ² match: %s (p=%.4f)", best_c, pval or 0)


if __name__ == "__main__":
    main()
