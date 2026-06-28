#!/usr/bin/env python3
"""Content-blind CHIASMUS (mirror / ring-composition) detector.

Same methodological family + machinery as the hakaara recursion test: frozen
template, conservative repetition-preserving null, hash-locked pre-registration,
STRUCTURE ONLY -- never content, never a gloss.

Statistic: deepest PERFECT CONTIGUOUS mirror around any center (both parities),
maximized over (center, span). No mismatch budget, no gapped matching -- those
are the degrees of freedom that make chiasmus over-detectable. The null searches
for its best mirror by the same exhaustive max -> a fair comparison.

Pass decided on robust exact-Barthel-code matching against Null B alone.
"""
import json, hashlib, os, sys, csv, argparse
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "structural"))
import hakaara as H          # reuse: shuffles, FDR, is_glyph, lock-verify, section_tokens
SEED = 20260628


# ----------------------------------------------------------------------------
# Mirror statistic
# ----------------------------------------------------------------------------
def _match_fn(blocked, exclude_nonglyph):
    if exclude_nonglyph:
        return lambda a, b: a == b and H.is_glyph(a, blocked)   # damage never mirrors
    return lambda a, b: a == b


def _arm(tokens, li, ri, match):
    """Outward left-arm token list of the perfect contiguous mirror with innermost
    pair (li, ri). Arm length == perfect mirror depth (matched-pair count)."""
    n = len(tokens); arm = []
    while li >= 0 and ri < n and match(tokens[li], tokens[ri]):
        arm.append(tokens[li]); li -= 1; ri += 1
    return arm


def _best_span(arm, k_chi, delta):
    """Deepest span s in [k_chi, len(arm)] whose arm[:s] is >= delta distinct
    (degenerate-run gate; earns no credit, only blocks false positives). 0 if none."""
    best, distinct, seen = 0, 0, set()
    for s in range(1, len(arm) + 1):
        if arm[s - 1] not in seen:
            seen.add(arm[s - 1]); distinct += 1
        if s >= k_chi and distinct / s >= delta:
            best = s
    return best


def mirror_T(tokens, k_chi, delta, match, want_detail=False):
    """Max qualifying perfect-mirror depth over all centers (pivot + gap)."""
    L = len(tokens); best_T, detail = 0, None
    for c in range(1, L - 1):                                   # pivot centers
        s = _best_span(_arm(tokens, c - 1, c + 1, match), k_chi, delta)
        if s > best_T:
            best_T = s
            if want_detail:
                detail = {"center": c, "parity": "pivot", "span": s,
                          "arm": [str(x) for x in tokens[c - s:c]]}
    for p in range(0, L - 1):                                   # gap centers (p|p+1)
        s = _best_span(_arm(tokens, p, p + 1, match), k_chi, delta)
        if s > best_T:
            best_T = s
            if want_detail:
                detail = {"center": p + 0.5, "parity": "gap", "span": s,
                          "arm": [str(x) for x in tokens[p - s + 1:p + 1]]}
    return best_T, detail


# ----------------------------------------------------------------------------
# Permutation test (same shape as hakaara; statistic re-maxed inside the null)
# ----------------------------------------------------------------------------
def permutation_test(tokens, k_chi, delta, match, T_obs, null_fn, B, rng):
    null = np.empty(B)
    for i in range(B):
        sh = null_fn(tokens, rng)
        null[i], _ = mirror_T(sh, k_chi, delta, match)
    ge = int(np.sum(null >= T_obs))
    p = (1 + ge) / (1 + B)
    sd = null.std()
    z = (T_obs - null.mean()) / sd if sd > 0 else float("nan")
    return p, z, float(null.mean())


# ----------------------------------------------------------------------------
# Driver (tablet run; NOT executed until the controls are approved)
# ----------------------------------------------------------------------------
def run(conf):
    freeze = H._verify(conf, "freeze.json", "freeze.lock")
    seg = H._verify(conf, "segmentation.frozen.json", "segmentation.lock")
    sig, nul = freeze["signature"], freeze["null"]
    k_chi, delta = sig["k_chi"], sig["delta_chi"]
    blocked = frozenset(sig.get("nonglyph_cores", [0]))
    xnon = sig.get("exclude_nonglyph", True)
    minlen = sig["min_section_len"]
    b, B = nul["block_length_b"], nul["n_permutations"]
    match_rob = _match_fn(blocked, xnon)
    rng = np.random.default_rng(SEED)

    rows = []
    for sec in seg["sections"]:
        exact, base = H.section_tokens(sec)
        row = {"section": sec["section_id"], "n_glyphs": len(exact)}
        if len(exact) < minlen:
            row.update({"status": "not_evaluable", "T": "-", "center": "-", "span": "-",
                        "z_B": "-", "p_B": "-", "T_aug": "-", "result": "NOT EVALUABLE (too short)"})
            row["_p_B"] = float("nan"); rows.append(row); continue

        T, det = mirror_T(exact, k_chi, delta, match_rob, want_detail=True)
        if T == 0:
            # A scanned section with no qualifying mirror is a TESTED hypothesis
            # with p = P(T_null >= 0) = 1.0 exactly. It belongs in the BH family
            # ("FDR across all evaluable sections"), NOT dropped as nan.
            row.update({"status": "evaluable", "T": 0, "center": "-", "span": "-",
                        "z_B": "-", "p_B": 1.0, "T_aug": "-", "result": "null (no mirror >= k_chi)"})
            row["_p_B"] = 1.0; rows.append(row); continue

        pA, zA, _ = permutation_test(exact, k_chi, delta, match_rob, T,
                                     lambda t, r: H.uniform_shuffle(t, r), B, rng)
        pB, zB, _ = permutation_test(exact, k_chi, delta, match_rob, T,
                                     lambda t, r: H.block_shuffle(t, b, r), B, rng)
        match_aug = _match_fn(blocked, xnon)
        T_aug, _ = mirror_T(base, k_chi, delta, match_aug)
        pBa, _, _ = permutation_test(base, k_chi, delta, match_aug, T_aug,
                                     lambda t, r: H.block_shuffle(t, b, r), B, rng)
        row.update({"status": "evaluable", "T": T, "center": det["center"], "span": det["span"],
                    "z_A": round(zA, 2), "p_A": round(pA, 4),
                    "z_B": round(zB, 2), "p_B": round(pB, 4),
                    "T_aug": T_aug, "p_B_aug": round(pBa, 4), "result": ""})
        row["_p_B"] = pB; rows.append(row)

    eval_idx = [i for i, r in enumerate(rows) if r["status"] == "evaluable" and r["_p_B"] == r["_p_B"]]
    reject = H.bh_reject([rows[i]["_p_B"] for i in eval_idx], freeze["multiplicity"]["q"])
    rej = {eval_idx[j] for j in reject}
    for i, r in enumerate(rows):
        if r["status"] == "evaluable" and r["result"] == "":
            r["result"] = "MIRROR STRUCTURE DETECTED" if i in rej else "null"
    return freeze, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conf", default=os.path.join(ROOT, "conf", "chiasmus"))
    ap.add_argument("--out", default=os.path.join(ROOT, "reports", "chiasmus_results.csv"))
    args = ap.parse_args()
    freeze, rows = run(args.conf)
    cols = ["section", "n_glyphs", "status", "T", "center", "span",
            "z_A", "p_A", "z_B", "p_B", "T_aug", "p_B_aug", "result"]
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        [w.writerow(r) for r in rows]
    det = sum(1 for r in rows if r["result"] == "MIRROR STRUCTURE DETECTED")
    ev = sum(1 for r in rows if r["status"] == "evaluable")
    print(f"Detections (FDR q={freeze['multiplicity']['q']}, Null B, robust): {det} of {ev} evaluable")
    print("CAVEAT:", freeze["report_caveat"])


if __name__ == "__main__":
    main()
