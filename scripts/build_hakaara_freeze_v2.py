#!/usr/bin/env python3
"""Build the FROZEN, HASH-LOCKED v2 pre-registration: finer (windowed) segmentation.

This is a SEPARATE pre-registration from v1 (conf/hakaara), not a tweak to it.
The v1 record stays untouched. v2 changes exactly two things from v1, both
content-blind and both declared here BEFORE the v2 detector runs:

  1. Segmentation unit = sliding window instead of whole side.
     W = (k+2)*p = 8*13 = 104 glyphs ; step = W/2 = 52 (50% overlap so a
     length-k enumeration cannot be split across a window boundary).
     W is template-derived (k, p from the locked v1 freeze), NOT chosen to
     make any section pass. Applied uniformly to every side.

  2. exclude_nonglyph = True. Drops uncertainty/lacuna markers ('?', '-') and
     the Barthel 000 family (unidentified/destroyed sign) from the connective
     candidate set. A periodic recurrence of destroyed-sign markers is damage,
     not enumeration. This was flagged as a hygiene defect in the v1 run.

ALL signature parameters (p, eps, k, l_max, delta, nulls, FDR q) are copied
verbatim from the locked v1 freeze so the comparison isolates the two changes.
"""
import json, hashlib, glob, os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
V1 = os.path.join(ROOT, "conf", "hakaara")
CONF = os.path.join(ROOT, "conf", "hakaara_v2")
CORPUS = os.path.join(ROOT, "data", "corpus")
TABLETS = "ABCDEFGHIJKLMNOPQRSTUVWXY"

W = 104        # (k+2)*p
STEP = 52      # W/2


def canon(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")


def load_v1_freeze():
    blob = open(os.path.join(V1, "freeze.json"), "rb").read()
    expect = open(os.path.join(V1, "freeze.lock")).read().strip()
    if hashlib.sha256(blob).hexdigest() != expect:
        sys.exit("v1 freeze lock mismatch; refusing to derive v2 from an altered v1.")
    return json.loads(blob)


def build_windows():
    sections = []
    for t in TABLETS:
        f = os.path.join(CORPUS, f"{t}.json")
        if not os.path.exists(f):
            continue
        glyphs = json.load(open(f)).get("glyphs", [])
        for side in sorted(set(g["side"] for g in glyphs)):
            sg = sorted([g for g in glyphs if g["side"] == side],
                        key=lambda g: g["position"])
            codes = [g["barthel_code"] for g in sg]
            n = len(codes)
            if n == 0:
                continue
            # window starts; ensure the final glyphs are covered by a last window
            starts = list(range(0, max(1, n - W + 1), STEP))
            if not starts or starts[-1] + W < n:
                starts.append(max(0, n - W))
            starts = sorted(set(starts))
            for wi, a in enumerate(starts):
                b = min(a + W, n)
                slc = codes[a:b]
                sig = hashlib.sha256("|".join(slc).encode()).hexdigest()
                sections.append({
                    "section_id": f"{t}{side}:w{wi:02d}",
                    "tablet": t, "side": side,
                    "win_start": a, "win_end": b,
                    "n_glyphs": b - a,
                    "content_sha256": sig,
                })
    return sections


def main():
    if os.path.exists(os.path.join(CONF, "freeze.lock")) and "--force" not in sys.argv:
        print("REFUSING: v2 freeze already exists.")
        sys.exit(1)
    os.makedirs(CONF, exist_ok=True)

    freeze = load_v1_freeze()
    freeze["frozen_on"] = "2026-06-27"
    freeze["description"] = ("hakaara v2: finer windowed segmentation + non-glyph "
                             "candidate filter; all signature params copied from v1")
    freeze["segmentation"] = {"unit": "sliding_window", "W": W, "step": STEP,
                              "derivation": "W=(k+2)*p=104, step=W/2=52, template-derived"}
    freeze["signature"]["exclude_nonglyph"] = True
    freeze["changes_from_v1"] = ["windowed segmentation (W=104, step=52)",
                                 "exclude_nonglyph=True ('?','-',000-family dropped)"]

    fp = os.path.join(CONF, "freeze.json")
    fblob = canon(freeze)
    open(fp, "wb").write(fblob)
    fdig = hashlib.sha256(fblob).hexdigest()
    open(os.path.join(CONF, "freeze.lock"), "w").write(fdig + "\n")

    seg = {"frozen_on": "2026-06-27", "unit": "sliding_window", "W": W, "step": STEP,
           "note": "Mechanical windows; content-blind; immutable post-lock.",
           "sections": build_windows()}
    sp = os.path.join(CONF, "segmentation.frozen.json")
    sblob = canon(seg)
    open(sp, "wb").write(sblob)
    sdig = hashlib.sha256(sblob).hexdigest()
    open(os.path.join(CONF, "segmentation.lock"), "w").write(sdig + "\n")

    print(f"freeze.json              sha256 {fdig}")
    print(f"segmentation.frozen.json sha256 {sdig}")
    print(f"windows: {len(seg['sections'])}")


if __name__ == "__main__":
    main()
