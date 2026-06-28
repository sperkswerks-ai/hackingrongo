#!/usr/bin/env python3
"""SYNTHETIC VALIDATION CONTROLS for the chiasmus detector. No tablet contact.
Tokens are sampled from the REAL corpus unigram frequencies so collision
probability matches reality. Must pass before any corpus run is trusted.

(a) clean planted chiasm     -> MUST detect
(b) random text              -> MUST reject
(c) chiasm buried in noise   -> MUST detect (power; the b=3 condition)
(d) degenerate run / ABAB    -> MUST NOT be called a chiasm (distinctness gate)
"""
import os, sys, json, glob, re
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "structural"))
import chiasmus as C
import hakaara as H

K_CHI, DELTA, B = 3, 0.5, 1000
BLOCKED = frozenset({0, 999})
MATCH = C._match_fn(BLOCKED, True)


def corpus_unigram():
    from collections import Counter
    c = Counter()
    for f in glob.glob(os.path.join(ROOT, "data", "corpus", "?.json")):
        for g in json.load(open(f))["glyphs"]:
            t = g["barthel_code"]
            if H.is_glyph(t, BLOCKED):
                c[t] += 1
    toks = np.array(list(c.keys()), dtype=object)
    p = np.array(list(c.values()), float); p /= p.sum()
    return toks, p


VOCAB, PROB = corpus_unigram()


def noise(n, rng):
    return list(rng.choice(VOCAB, size=n, p=PROB))


def chiasm(depth, rng):
    """Perfect mirror of `depth` distinct glyphs around a single pivot."""
    arms = list(rng.choice(VOCAB, size=depth, replace=False, p=PROB))
    pivot = rng.choice(VOCAB, p=PROB)
    return arms + [pivot] + arms[::-1]


def evaluate(seq, label, b=3):
    tokens = np.array(seq, dtype=object)
    T, det = C.mirror_T(tokens, K_CHI, DELTA, MATCH, want_detail=True)
    rng = np.random.default_rng(7)
    pA, zA, _ = C.permutation_test(tokens, K_CHI, DELTA, MATCH, T,
                                   lambda t, r: H.uniform_shuffle(t, r), B, rng)
    pB, zB, _ = C.permutation_test(tokens, K_CHI, DELTA, MATCH, T,
                                   lambda t, r: H.block_shuffle(t, b, r), B, rng)
    ctr = det["center"] if det else "-"
    print(f"{label:42} L={len(tokens):4}  T={T:2d} ctr={str(ctr):>5}  "
          f"NullA z={zA:5.2f} p={pA:.4f}   NullB(b={b}) z={zB:5.2f} p={pB:.4f}")
    return T, pB


print("collision check: 1/p_c =", round(1 / np.sum(PROB**2), 1), " vocab:", len(VOCAB))
print("=" * 104)
print("(a) CLEAN PLANTED CHIASM  -> must detect")
print("=" * 104)
for d in (3, 4, 6):
    r = np.random.default_rng(100 + d)
    evaluate(noise(8, r) + chiasm(d, r) + noise(8, r), f"clean chiasm depth {d} (+small pad)")

print("\n" + "=" * 104)
print("(b) RANDOM TEXT  -> must reject")
print("=" * 104)
for s in (1, 2, 3):
    evaluate(noise(200, np.random.default_rng(500 + s)), f"random text 200 (seed {s})")

print("\n" + "=" * 104)
print("(c) CHIASM BURIED IN NOISE  -> must detect  (THE b=3 CONDITION)")
print("=" * 104)
for d in (3, 4, 6):
    r = np.random.default_rng(200 + d)
    evaluate(noise(150, r) + chiasm(d, r) + noise(150, r), f"depth-{d} chiasm buried in 300 noise")

print("\n" + "=" * 104)
print("(d) DEGENERATE PATTERNS  -> must NOT be called a chiasm (gate)")
print("=" * 104)
g = str(VOCAB[int(np.argmax(PROB))])
evaluate([g] * 31, f"single-glyph run {g} x31")
two = [str(x) for x in VOCAB[np.argsort(PROB)[-2:]]]
evaluate(([two[0], two[1]] * 16)[:31], f"2-glyph alternation {two[0]}/{two[1]} x31")
r = np.random.default_rng(9)
evaluate(noise(20, r) + [g] * 11 + noise(20, r), f"single-glyph run {g} x11 buried in noise")
