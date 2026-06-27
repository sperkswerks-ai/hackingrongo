#!/usr/bin/env python3
"""Build the FROZEN, HASH-LOCKED inputs for the hakaara recursive-enumeration test.

This script writes four files and must be run exactly once, BEFORE the detector
ever touches a tablet:

    conf/hakaara/freeze.json            - signature parameters (p, eps, k, ...)
    conf/hakaara/freeze.lock            - sha256 of freeze.json
    conf/hakaara/segmentation.frozen.json - section boundaries (physical, content-blind)
    conf/hakaara/segmentation.lock      - sha256 of segmentation.frozen.json

Parameter provenance (ratified 2026-06-27):
  p, eps  = 13 +/- 1.5 glyphs, from Mamari calendar (Tablet C, side a, lines 06-09)
            inter-anchor gaps {13,13,13,12,14}; preamble 22-gap excluded as the
            documented new-moon/intercalary opening, not an inter-entry period.
  k       = 6 slots, from the clean entry-template repetition count, convergent
            with the conventional 6-night kokore group.
  l_max   = 2 glyphs (connective length cap); Kieviet clause-linkers are 1-2 morae.
            The HEADLINE run restricts to 1-gram connectives (conservative subset
            of the cap; the calendar's minimal anchor 008 is one glyph).
  Section = one physical tablet side. Boundaries are physical, never content-chosen.

INTEGRITY: the specific calendar anchor (glyph 008) is NOT carried onto any other
tablet. On every section the connective is re-derived by structural profile only
(short token, count >= k+1), and the null re-derives it identically.
"""
import json, hashlib, glob, os, sys
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONF = os.path.join(ROOT, "conf", "hakaara")
CORPUS = os.path.join(ROOT, "data", "corpus")

# Single-letter tablet corpus files only (skip xml/, composites like D_ferrara2022)
TABLETS = "ABCDEFGHIJKLMNOPQRSTUVWXY"

FREEZE = {
    "frozen_on": "2026-06-27",
    "description": "hakaara recursive-enumeration structural signature; frozen before tablet contact",
    "signature": {
        "primary_statistic": "phase_coherence",          # mean resultant length R(p')
        "gap_band": "cross_check_only",                  # never headline
        "p": 13.0,                                       # entry period, glyphs
        "eps": 1.5,                                       # period tolerance, glyphs (primary)
        "period_grid_step": 0.05,                        # p' scan resolution within [p-eps, p+eps]
        "k": 6,                                           # minimum slots (n = m-1 >= k)
        "l_max": 2,                                       # connective length cap (glyphs)
        "headline_connective_ngram": 1,                  # conservative subset of l_max
        "f_min_count": 7,                                 # hard gate: connective count >= k+1
        "f_min_relfreq": 0.03,                            # soft secondary frequency floor
        "delta": 0.5,                                     # slot distinctness GATE (no positive credit)
        "J": 2,                                           # harmonic depth for gap-band cross-check
    },
    "null": {
        "primary_headline": "block_shuffle",             # Null B, conservative, decides PASS
        "floor": "uniform_shuffle",                      # Null A, reported as floor
        "block_length_b": 5,                             # b < p
        "n_permutations": 1000,
    },
    "features": {
        "robust": "exact barthel_code; phase-coherence + slot-count + slot-length-CV + exact-match distinctness gate",
        "augmented": "barthel_base (allograph-normalized) + slot-head entropy; reported, never rescues a robust null",
        "pass_decided_on": "robust",
    },
    "multiplicity": {"correction": "benjamini_hochberg", "q": 0.05},
    "sweep": {
        "p": [12, 13, 14],
        "eps": [1.0, 1.5, 2.0],     # 1.5 is primary; 2.0 must NEVER be promoted to headline
        "k": [6, 8, 10],            # 6 primary
        "b": [3, 5],                # 5 primary
    },
    "report_caveat": (
        "Consistent with recursive enumeration - which a genealogy would produce, "
        "and so would a king-list, a property record, or a ritual sequence. "
        "The structure is detectable; the content is not."
    ),
    "provenance": {
        "p_eps_k": "Mamari calendar Tablet C side a lines 06-09 (Barthel 1958 enc., kohaumotu.org)",
        "connective_profile": "Kieviet, Rapa Nui clause-linker morphology (short, frequent)",
        "forbidden_sources": ["Manuscript E", "Fischer", "Metraux", "Englert", "any content/genre decipherment"],
    },
}


def load_side(tablet):
    f = os.path.join(CORPUS, f"{tablet}.json")
    if not os.path.exists(f):
        return None
    d = json.load(open(f))
    return d.get("glyphs", [])


def build_segmentation():
    sections = []
    for t in TABLETS:
        glyphs = load_side(t)
        if not glyphs:
            continue
        sides = sorted(set(g["side"] for g in glyphs))
        for side in sides:
            sg = [g for g in glyphs if g["side"] == side]
            sg = sorted(sg, key=lambda g: g["position"])
            codes_exact = [g["barthel_code"] for g in sg]
            codes_base = [g.get("barthel_base") or g["barthel_code"] for g in sg]
            # pin the underlying glyph content so the corpus itself cannot drift
            content_sig = hashlib.sha256(
                ("|".join(codes_exact)).encode("utf-8")
            ).hexdigest()
            sections.append({
                "section_id": f"{t}{side}",
                "tablet": t,
                "side": side,
                "n_glyphs": len(sg),
                "position_start": sg[0]["position"],
                "position_end": sg[-1]["position"],
                "content_sha256": content_sig,
            })
    return sections


def canon(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")


def main():
    if os.path.exists(os.path.join(CONF, "freeze.lock")) and "--force" not in sys.argv:
        print("REFUSING: freeze already exists. Re-running would break the pre-registration.")
        print("Pass --force only if you have NOT yet seen any tablet results.")
        sys.exit(1)
    os.makedirs(CONF, exist_ok=True)

    # freeze.json
    fp = os.path.join(CONF, "freeze.json")
    blob = canon(FREEZE)
    open(fp, "wb").write(blob)
    fdig = hashlib.sha256(blob).hexdigest()
    open(os.path.join(CONF, "freeze.lock"), "w").write(fdig + "\n")

    # segmentation.frozen.json
    seg = {"frozen_on": "2026-06-27",
           "unit": "physical_tablet_side",
           "note": "Boundaries are physical sides; never content-selected. Immutable post-lock.",
           "sections": build_segmentation()}
    sp = os.path.join(CONF, "segmentation.frozen.json")
    sblob = canon(seg)
    open(sp, "wb").write(sblob)
    sdig = hashlib.sha256(sblob).hexdigest()
    open(os.path.join(CONF, "segmentation.lock"), "w").write(sdig + "\n")

    print(f"freeze.json              sha256 {fdig}")
    print(f"segmentation.frozen.json sha256 {sdig}")
    print(f"sections: {len(seg['sections'])}")


if __name__ == "__main__":
    main()
