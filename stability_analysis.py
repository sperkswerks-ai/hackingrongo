#!/usr/bin/env python3
"""
Stability analysis: phoneme-assignment variance across self-training iterations.
Compares the top-ranked hypothesis across iter_00 through iter_03.
"""
import json
from collections import defaultdict

BASE = "/Users/sarahperkins/Projects/Prototyping/rongorongo/hackingrongo"
RUN_PATHS = [
    ("iter_00", f"{BASE}/outputs/self_training/iter_00/ranking.json"),
    ("iter_01", f"{BASE}/outputs/self_training/iter_01/ranking.json"),
    ("iter_02", f"{BASE}/outputs/self_training/iter_02/ranking.json"),
    ("iter_03", f"{BASE}/outputs/self_training/iter_03/ranking.json"),
]

run_assignments = {}
run_scores = {}
for run_name, path in RUN_PATHS:
    d = json.load(open(path))
    hyp = d["hypotheses"][0]           # top-ranked hypothesis
    run_assignments[run_name] = {
        asn["sign_code"]: asn for asn in hyp["assignments"]
    }
    run_scores[run_name] = hyp.get("overall_lm_score", float("nan"))

print("LM scores by iteration:")
for rn, score in run_scores.items():
    print(f"  {rn}  lm_score={score:.4f}")
print()

run_names = [r for r, _ in RUN_PATHS]

sign_phonemes = defaultdict(dict)
for run_name, asn_map in run_assignments.items():
    for sc, asn in asn_map.items():
        sign_phonemes[sc][run_name] = asn["phoneme"]

all_signs = sorted(sign_phonemes.keys())

stable   = []
unstable = []
partial  = []

for sc in all_signs:
    present = {rn: sign_phonemes[sc][rn] for rn in run_names if rn in sign_phonemes[sc]}
    if len(present) < 4:
        partial.append((sc, present))
        continue
    phonemes = [present[rn] for rn in run_names]
    conf = run_assignments["iter_00"].get(sc, {}).get("confidence", "?")
    if len(set(phonemes)) == 1:
        stable.append((sc, phonemes[0], conf))
    else:
        unstable.append((sc, {rn: present[rn] for rn in run_names}))

print(f"Total signs across all runs : {len(all_signs)}")
print(f"Present in all 4 runs       : {len(stable) + len(unstable)}")
print(f"Stable (zero variance)      : {len(stable)}")
print(f"Unstable (diverges)         : {len(unstable)}")
print(f"Partial (not in all 4)      : {len(partial)}")

print()
print("=== STABLE SIGNS (same phoneme across all 4 runs) ===")
for sc, ph, conf in stable:
    print(f"  {sc:<20s}  ->  {ph:<6s}  conf={conf}")

print()
print("=== UNSTABLE SIGNS ===")
for sc, ph_map in unstable:
    print(f"  {sc:<20s}  {ph_map}")

print()
print("=== PARTIAL (absent from ≥1 run) ===")
for sc, pm in partial:
    print(f"  {sc:<20s}  {pm}")

# Save results
out = {
    "lm_scores": run_scores,
    "stable":   [{"sign_code": sc, "phoneme": ph, "confidence": conf} for sc, ph, conf in stable],
    "unstable": [{"sign_code": sc, "phonemes_by_run": ph_map} for sc, ph_map in unstable],
    "partial":  [{"sign_code": sc, "present_in": pm} for sc, pm in partial],
}
import os
os.makedirs(f"{BASE}/outputs/analysis", exist_ok=True)
with open(f"{BASE}/outputs/analysis/stability_analysis.json", "w") as f:
    json.dump(out, f, indent=2)
print("\nResults saved to outputs/analysis/stability_analysis.json")
