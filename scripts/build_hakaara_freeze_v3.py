#!/usr/bin/env python3
"""Build the FROZEN, HASH-LOCKED v3 pre-registration -- the FINAL one.

v3 corrects the relfreq instrument defect proven on synthetic data, plus the two
content-blind hygiene items, all ratified BEFORE this run. Three changes vs v1,
each justified independently of whether the calendar passes:

  1. f_min_relfreq = 0.0  (DROP the relative-frequency floor).
     Proven defect: the 0.03 floor scales with section length and filters out
     localized connectives -- it discarded a *known* planted period-13 connective
     in synthetic noise, and discarded the calendar's true anchor 008 at side
     level. Hard gate count >= k+1 = 7 is retained.

  2. exclude_nonglyph = True, nonglyph_cores = [0, 999].
     Blocklist of encoding non-signs: 000 (unidentified), 999 (placeholder absent
     from Barthel's plates), and non-numeric markers ('?', '(N-M)!' gaps). Real
     Barthel signs are codes 1..791 and are all retained. This is a blocklist of
     3 marker classes, NOT a positive whitelist (barthel_families.json is an
     incomplete 131-code subset that would wrongly drop 583 real signs).

  3. Segmentation = sliding window W=(k+2)*p=104, step=52 (as v2), so sections
     are list-sized and a localized enumeration is not diluted.

All other signature params (p, eps, k, l_max, delta, nulls, FDR q) copied verbatim
from the locked v1 freeze. PER RATIFICATION: v3 is final. Its result -- detection
or null -- is the recorded result.
"""
import json, hashlib, os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
V1 = os.path.join(ROOT, "conf", "hakaara")
CONF = os.path.join(ROOT, "conf", "hakaara_v3")
CORPUS = os.path.join(ROOT, "data", "corpus")
TABLETS = "ABCDEFGHIJKLMNOPQRSTUVWXY"
W, STEP = 104, 52


def canon(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")


def load_v1_freeze():
    blob = open(os.path.join(V1, "freeze.json"), "rb").read()
    if hashlib.sha256(blob).hexdigest() != open(os.path.join(V1, "freeze.lock")).read().strip():
        sys.exit("v1 freeze lock mismatch; refusing to derive v3 from an altered v1.")
    return json.loads(blob)


def build_windows():
    sections = []
    for t in TABLETS:
        f = os.path.join(CORPUS, f"{t}.json")
        if not os.path.exists(f):
            continue
        glyphs = json.load(open(f)).get("glyphs", [])
        for side in sorted(set(g["side"] for g in glyphs)):
            sg = sorted([g for g in glyphs if g["side"] == side], key=lambda g: g["position"])
            codes = [g["barthel_code"] for g in sg]
            n = len(codes)
            if n == 0:
                continue
            starts = list(range(0, max(1, n - W + 1), STEP))
            if not starts or starts[-1] + W < n:
                starts.append(max(0, n - W))
            for wi, a in enumerate(sorted(set(starts))):
                b = min(a + W, n)
                sections.append({
                    "section_id": f"{t}{side}:w{wi:02d}", "tablet": t, "side": side,
                    "win_start": a, "win_end": b, "n_glyphs": b - a,
                    "content_sha256": hashlib.sha256("|".join(codes[a:b]).encode()).hexdigest(),
                })
    return sections


def main():
    if os.path.exists(os.path.join(CONF, "freeze.lock")) and "--force" not in sys.argv:
        print("REFUSING: v3 freeze already exists.")
        sys.exit(1)
    os.makedirs(CONF, exist_ok=True)

    fr = load_v1_freeze()
    fr["frozen_on"] = "2026-06-27"
    fr["description"] = "hakaara v3 (FINAL): relfreq defect fixed + non-sign blocklist + windowed seg"
    fr["signature"]["f_min_relfreq"] = 0.0
    fr["signature"]["exclude_nonglyph"] = True
    fr["signature"]["nonglyph_cores"] = [0, 999]
    fr["segmentation"] = {"unit": "sliding_window", "W": W, "step": STEP,
                          "derivation": "W=(k+2)*p=104, step=W/2=52"}
    fr["final_preregistration"] = True
    fr["changes_from_v1"] = ["f_min_relfreq 0.03 -> 0.0 (proven instrument defect)",
                             "exclude_nonglyph=True, nonglyph_cores=[0,999]",
                             "windowed segmentation W=104 step=52"]

    fblob = canon(fr)
    open(os.path.join(CONF, "freeze.json"), "wb").write(fblob)
    fdig = hashlib.sha256(fblob).hexdigest()
    open(os.path.join(CONF, "freeze.lock"), "w").write(fdig + "\n")

    seg = {"frozen_on": "2026-06-27", "unit": "sliding_window", "W": W, "step": STEP,
           "note": "Mechanical windows; content-blind; immutable post-lock.",
           "sections": build_windows()}
    sblob = canon(seg)
    open(os.path.join(CONF, "segmentation.frozen.json"), "wb").write(sblob)
    sdig = hashlib.sha256(sblob).hexdigest()
    open(os.path.join(CONF, "segmentation.lock"), "w").write(sdig + "\n")

    print(f"freeze.json              sha256 {fdig}")
    print(f"segmentation.frozen.json sha256 {sdig}")
    print(f"windows: {len(seg['sections'])}")


if __name__ == "__main__":
    main()
