#!/usr/bin/env python3
"""Build the FROZEN, HASH-LOCKED pre-registration for the chiasmus (mirror) test.

Independent pre-registration (same discipline as hakaara). Section = whole tablet
side (chiasmus is a whole-composition property; windowing would truncate mirrors).

Parameter provenance (ratified 2026-06-28):
  k_chi = 3  : minimal non-trivial literary chiasm (A-B-C-C-B-A = 3 mirrored
               pairs); corpus chance floor = expected longest perfect mirror in
               random text is 1.1-1.8 pairs (p_c exact-code = 0.0091), so depth-3
               is the first depth clearly above chance in every section.
  delta_chi = 0.5 : arm-distinctness GATE (>=50% of arm positions distinct types);
               blocks single-glyph runs. Earns no positive credit.
  b = 3 (Null B block) : < minimal mirror full-width (2*k_chi+1 = 7) so the block
               shuffle reliably breaks minimal mirrors, while >=3 preserves local
               run/trigram texture. PROVISIONAL: confirmed only if the buried-
               chiasm synthetic control detects at b=3.
"""
import json, hashlib, os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONF = os.path.join(ROOT, "conf", "chiasmus")
CORPUS = os.path.join(ROOT, "data", "corpus")
TABLETS = "ABCDEFGHIJKLMNOPQRSTUVWXY"

FREEZE = {
    "frozen_on": "2026-06-28",
    "description": "chiasmus (mirror/ring) structural test; frozen before tablet contact",
    "signature": {
        "match": "exact_barthel_code_contiguous_perfect_mirror",
        "statistic": "max over (center,span) of perfect contiguous mirror depth",
        "no_mismatch_budget": True,
        "no_gapped_matching": True,
        "k_chi": 3,
        "delta_chi": 0.5,
        "center_parities": ["pivot", "gap"],
        "exclude_nonglyph": True,
        "nonglyph_cores": [0, 999],
        "min_section_len": 7,
    },
    "null": {
        "primary_headline": "block_shuffle",
        "floor": "uniform_shuffle",
        "block_length_b": 3,
        "n_permutations": 1000,
    },
    "features": {
        "robust": "exact Barthel code (headline; decides pass)",
        "augmented": "allograph-normalized base code (reported; never rescues a robust null)",
        "pass_decided_on": "robust",
    },
    "multiplicity": {"correction": "benjamini_hochberg", "q": 0.05},
    "sweep": {"k_chi": [3, 4, 5], "delta_chi": [0.5, 0.67], "b": [3, 5]},
    "report_caveat": (
        "Consistent with chiastic (mirror) structure in the arrangement of signs. "
        "This is a structural property; it does not establish that the section is "
        "poetry, ritual, or any genre, and asserts nothing about content, sound, "
        "or meaning."
    ),
    "provenance": {
        "k_chi": "minimal literary chiasm + corpus collision-prob chance floor (p_c=0.0091)",
        "forbidden_sources": ["Manuscript E", "Fischer", "Metraux", "Englert", "content/genre decipherment"],
    },
    "stopping_rule": "final pre-registration; result is recorded; defect justifies re-run only if proven on synthetic data to flip a known-correct answer",
}


def canon(o):
    return json.dumps(o, sort_keys=True, separators=(",", ":")).encode()


def build_segmentation():
    secs = []
    for t in TABLETS:
        f = os.path.join(CORPUS, f"{t}.json")
        if not os.path.exists(f):
            continue
        glyphs = json.load(open(f)).get("glyphs", [])
        for side in sorted(set(g["side"] for g in glyphs)):
            sg = sorted([g for g in glyphs if g["side"] == side], key=lambda g: g["position"])
            codes = [g["barthel_code"] for g in sg]
            if not codes:
                continue
            secs.append({"section_id": f"{t}{side}", "tablet": t, "side": side,
                         "n_glyphs": len(codes), "position_start": sg[0]["position"],
                         "position_end": sg[-1]["position"],
                         "content_sha256": hashlib.sha256("|".join(codes).encode()).hexdigest()})
    return secs


def main():
    if os.path.exists(os.path.join(CONF, "freeze.lock")) and "--force" not in sys.argv:
        print("REFUSING: chiasmus freeze already exists."); sys.exit(1)
    os.makedirs(CONF, exist_ok=True)

    fblob = canon(FREEZE)
    open(os.path.join(CONF, "freeze.json"), "wb").write(fblob)
    fdig = hashlib.sha256(fblob).hexdigest()
    open(os.path.join(CONF, "freeze.lock"), "w").write(fdig + "\n")

    seg = {"frozen_on": "2026-06-28", "unit": "physical_tablet_side",
           "note": "Whole sides; content-blind; immutable post-lock.",
           "sections": build_segmentation()}
    sblob = canon(seg)
    open(os.path.join(CONF, "segmentation.frozen.json"), "wb").write(sblob)
    sdig = hashlib.sha256(sblob).hexdigest()
    open(os.path.join(CONF, "segmentation.lock"), "w").write(sdig + "\n")

    print(f"freeze.json              sha256 {fdig}")
    print(f"segmentation.frozen.json sha256 {sdig}")
    print(f"sections: {len(seg['sections'])}")


if __name__ == "__main__":
    main()
