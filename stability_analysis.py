#!/usr/bin/env python3
"""
Stability analysis: phoneme-assignment variance across self-training iterations.
Compares the top-ranked hypothesis across all iter_NN runs found under --base-dir.

Usage
-----
    python stability_analysis.py
    python stability_analysis.py --base-dir /path/to/project
    python stability_analysis.py --output outputs/self_training/stability_analysis.json
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stability analysis for self-training iterations.")
    p.add_argument(
        "--base-dir", type=Path, default=PROJECT_ROOT,
        help="Project root directory (default: directory containing this script).",
    )
    p.add_argument(
        "--output", type=Path, default=None,
        help="Path to write stability_analysis.json. "
             "Defaults to <base-dir>/outputs/self_training/stability_analysis.json.",
    )
    return p.parse_args()


def run(base_dir: Path, output: Path | None = None) -> dict:
    """Compute stability and write JSON; returns the result dict."""
    self_training_dir = base_dir / "outputs" / "self_training"
    ranking_paths = sorted(self_training_dir.glob("iter_*/ranking.json"))

    if not ranking_paths:
        print(f"No iter_*/ranking.json files found under {self_training_dir}", file=sys.stderr)
        return {}

    run_assignments: dict[str, dict] = {}
    run_scores: dict[str, float] = {}
    for path in ranking_paths:
        iter_name = path.parent.name
        try:
            d = json.loads(path.read_text(encoding="utf-8"))
            hyp = d["hypotheses"][0]
            run_assignments[iter_name] = {
                asn["sign_code"]: asn for asn in hyp["assignments"]
            }
            run_scores[iter_name] = hyp.get("overall_lm_score", float("nan"))
        except (KeyError, IndexError, json.JSONDecodeError, OSError) as exc:
            print(f"Warning: skipping {path}: {exc}", file=sys.stderr)
            continue

    run_names = sorted(run_assignments)
    n_runs = len(run_names)

    sign_phonemes: dict[str, dict[str, str]] = defaultdict(dict)
    for run_name, asn_map in run_assignments.items():
        for sc, asn in asn_map.items():
            sign_phonemes[sc][run_name] = asn["phoneme"]

    all_signs = sorted(sign_phonemes)
    stable, unstable, partial = [], [], []

    for sc in all_signs:
        present = {rn: sign_phonemes[sc][rn] for rn in run_names if rn in sign_phonemes[sc]}
        if len(present) < n_runs:
            partial.append({"sign_code": sc, "present_in": present})
            continue
        phonemes = [present[rn] for rn in run_names]
        conf = run_assignments[run_names[0]].get(sc, {}).get("confidence")
        if len(set(phonemes)) == 1:
            stable.append({"sign_code": sc, "phoneme": phonemes[0], "confidence": conf})
        else:
            unstable.append({"sign_code": sc, "phonemes_by_run": {rn: present[rn] for rn in run_names}})

    print("LM scores by iteration:")
    for rn, score in sorted(run_scores.items()):
        print(f"  {rn}  lm_score={score:.4f}")
    print()
    print(f"Total signs across all runs : {len(all_signs)}")
    print(f"Present in all {n_runs} runs        : {len(stable) + len(unstable)}")
    print(f"Stable (zero variance)      : {len(stable)}")
    print(f"Unstable (diverges)         : {len(unstable)}")
    print(f"Partial (not in all {n_runs})  : {len(partial)}")

    if stable:
        print()
        print(f"=== STABLE SIGNS (same phoneme across all {n_runs} runs) ===")
        for entry in stable:
            conf_str = f"{entry['confidence']:.3f}" if entry["confidence"] is not None else "?"
            print(f"  {entry['sign_code']:<20s}  ->  {entry['phoneme']:<6s}  conf={conf_str}")

    if unstable:
        print()
        print("=== UNSTABLE SIGNS ===")
        for entry in unstable:
            print(f"  {entry['sign_code']:<20s}  {entry['phonemes_by_run']}")

    if partial:
        print()
        print("=== PARTIAL (absent from ≥1 run) ===")
        for entry in partial:
            print(f"  {entry['sign_code']:<20s}  {entry['present_in']}")

    out = {
        "lm_scores": run_scores,
        "n_runs": n_runs,
        "stable":   stable,
        "unstable": unstable,
        "partial":  partial,
    }

    out_path = output or (base_dir / "outputs" / "self_training" / "stability_analysis.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nResults saved to {out_path}")
    return out


def main() -> None:
    args = _parse_args()
    result = run(args.base_dir, args.output)
    if not result:
        sys.exit(1)


if __name__ == "__main__":
    main()

