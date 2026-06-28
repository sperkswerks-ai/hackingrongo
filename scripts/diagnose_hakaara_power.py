#!/usr/bin/env python3
"""INSTRUMENT CHECK for the hakaara detector. Not part of the frozen test; touches
no locks/results. Answers: can the detector flag a KNOWN enumeration? If a clean
planted period-13 list is not significant, the corpus null is an artifact.
"""
import os, sys, json
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "structural"))
import hakaara as H

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
periods = np.arange(11.5, 14.5 + 1e-9, 0.05)
k, delta, B = 6, 0.5, 1000


def evaluate(tokens, label, exclude_nonglyph=False):
    tokens = np.array(tokens, dtype=object)
    cand = H.candidates(tokens, 7, 0.03, exclude_nonglyph)
    T, det = H.section_T(tokens, cand, periods, k, delta, want_detail=True)
    rng = np.random.default_rng(1)
    pA, zA, mA = H.permutation_test(tokens, cand, periods, k, delta, T,
                                    lambda t, r: H.uniform_shuffle(t, r), B, rng)
    pB, zB, mB = H.permutation_test(tokens, cand, periods, k, delta, T,
                                    lambda t, r: H.block_shuffle(t, 5, r), B, rng)
    conn = det["connective"] if det else "-"
    print(f"{label:38} conn={conn:>5} T={T:.3f}  "
          f"NullA z={zA:5.2f} p={pA:.4f}  NullB z={zB:5.2f} p={pB:.4f}")
    return pA, pB


def planted(n_entries=10, period=13, jitter=0, seed=0):
    """Random text with a connective 'CON' planted every `period` (+/- jitter)
    glyphs, slots filled with distinct random vocab so the gate passes."""
    rng = np.random.default_rng(seed)
    vocab = [f"V{i:03d}" for i in range(60)]
    seq, pos = [], 0
    for e in range(n_entries):
        gap = period + (rng.integers(-jitter, jitter + 1) if jitter else 0)
        body = list(rng.choice(vocab, size=gap - 1, replace=True))
        seq.append("CON"); seq += body
    seq.append("CON")
    return seq


print("=" * 96)
print("A. SYNTHETIC POSITIVE CONTROLS  (detector MUST flag these)")
print("=" * 96)
evaluate(planted(10, 13, 0, 0), "clean period-13 list (10 entries)")
evaluate(planted(10, 13, 1, 0), "period-13 +/-1 jitter (10 entries)")
evaluate(planted(7, 13, 1, 1),  "short period-13 list (7 entries=k+1)")
# planted list EMBEDDED in surrounding random noise (like the calendar in a side)
rng = np.random.default_rng(2)
noise1 = list(rng.choice([f"V{i:03d}" for i in range(60)], size=180))
noise2 = list(rng.choice([f"V{i:03d}" for i in range(60)], size=180))
evaluate(noise1 + planted(8, 13, 1, 3) + noise2, "period-13 list embedded in 360 noise glyphs")

print()
print("=" * 96)
print("B. NEGATIVE CONTROL  (detector must NOT flag pure noise)")
print("=" * 96)
rng = np.random.default_rng(7)
evaluate(list(rng.choice([f"V{i:03d}" for i in range(60)], size=200)), "pure random text (200 glyphs)")

print()
print("=" * 96)
print("C. THE REAL MAMARI CALENDAR, ISOLATED (Ca lines 06-09, 154 glyphs)")
print("=" * 96)
d = json.load(open(os.path.join(ROOT, "data", "corpus", "C.json")))
cal = [g["barthel_code"] for g in sorted(
        [g for g in d["glyphs"] if g["side"] == "a" and g["line"] in ("06", "07", "08", "09")],
        key=lambda g: g["position"])]
evaluate(cal, "calendar Ca06-09 (re-derived connective)")
# force the documented anchor 008 specifically
cal_arr = np.array(cal, dtype=object)
T008, det = H.section_T(cal_arr, ["008"], periods, k, delta, want_detail=True)
rng = np.random.default_rng(1)
pA, zA, _ = H.permutation_test(cal_arr, ["008"], periods, k, delta, T008,
                               lambda t, r: H.uniform_shuffle(t, r), B, rng)
pB, zB, _ = H.permutation_test(cal_arr, ["008"], periods, k, delta, T008,
                               lambda t, r: H.block_shuffle(t, 5, r), B, rng)
print(f"{'calendar, FORCED connective 008':38} conn=  008 T={T008:.3f}  "
      f"NullA z={zA:5.2f} p={pA:.4f}  NullB z={zB:5.2f} p={pB:.4f}")
print(f"  (008 positions in region: {np.flatnonzero(cal_arr=='008').tolist()})")
