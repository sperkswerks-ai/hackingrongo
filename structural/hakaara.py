#!/usr/bin/env python3
"""hakaara recursive-enumeration STRUCTURAL test (detector).

Runs ONLY against the hash-locked freeze + segmentation. It refuses to run if
either lock does not match -- so neither the parameters nor the section
boundaries can be edited after seeing results without leaving a visible trace.

Output is structure, never content:
    "section X shows / does not show the recursive-enumeration signature,
     effect size z, permutation p" -- never "genealogy", never a gloss.

Integrity rules enforced in code:
  * connective is re-derived on every section by structural profile only
    (1-gram, count >= k+1); the calendar's glyph 008 is never carried over.
  * the same re-derivation runs inside every null replicate (self-correcting
    multiplicity).
  * PASS is decided on robust (exact-code) features against Null B alone.
  * distinctness is a GATE only -- it earns no positive credit.
"""
import json, hashlib, os, sys, csv, re, argparse
import numpy as np
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CORPUS = os.path.join(ROOT, "data", "corpus")
SEED = 20260627


# ----------------------------------------------------------------------------
# Lock verification
# ----------------------------------------------------------------------------
def _verify(conf, json_name, lock_name):
    path = os.path.join(conf, json_name)
    blob = open(path, "rb").read()
    digest = hashlib.sha256(blob).hexdigest()
    expected = open(os.path.join(conf, lock_name)).read().strip()
    if digest != expected:
        sys.exit(f"LOCK MISMATCH on {json_name}: {digest} != {expected}\n"
                 f"Refusing to run. Boundaries/parameters were altered after freeze.")
    return json.loads(blob)


def load_frozen(conf):
    freeze = _verify(conf, "freeze.json", "freeze.lock")
    seg = _verify(conf, "segmentation.frozen.json", "segmentation.lock")
    return freeze, seg


# ----------------------------------------------------------------------------
# Section tokens (with corpus content pinned to the frozen hash)
# ----------------------------------------------------------------------------
def section_tokens(sec):
    d = json.load(open(os.path.join(CORPUS, f"{sec['tablet']}.json")))
    sg = [g for g in d["glyphs"] if g["side"] == sec["side"]]
    sg = sorted(sg, key=lambda g: g["position"])
    exact = [g["barthel_code"] for g in sg]
    base = [g.get("barthel_base") or g["barthel_code"] for g in sg]
    # windowed segmentation slices the side; content hash pins exactly that slice
    if "win_start" in sec:
        a, b = sec["win_start"], sec["win_end"]
        exact, base = exact[a:b], base[a:b]
    content_sig = hashlib.sha256("|".join(exact).encode()).hexdigest()
    if content_sig != sec["content_sha256"]:
        sys.exit(f"CORPUS DRIFT on {sec['section_id']}: glyph content changed since freeze.")
    return np.array(exact, dtype=object), np.array(base, dtype=object)


def is_glyph(tok):
    """Content-blind validity: reject uncertainty/lacuna markers ('?', '-') and
    the Barthel 000 family (unidentified/destroyed sign). A periodic recurrence
    of destroyed-sign markers is damage, not enumeration."""
    m = re.match(r"0*([0-9]+)", str(tok))
    if not m:
        return False
    return m.group(1) != "0"


# ----------------------------------------------------------------------------
# Signature statistic
# ----------------------------------------------------------------------------
def phase_coherence(positions, periods):
    """Max mean-resultant-length over the period grid (omission-robust)."""
    x = positions.astype(float)[:, None]            # (m,1)
    ang = 2.0 * np.pi * x / periods[None, :]        # (m,G)
    S = np.exp(1j * ang).sum(axis=0)                # (G,)
    R = np.abs(S) / len(positions)
    j = int(np.argmax(R))
    return float(R[j]), float(periods[j])


def slot_distinctness(tokens, pos):
    """Distinct-type ratio of inter-connective slots (GATE only)."""
    slots = [tuple(tokens[pos[i] + 1:pos[i + 1]]) for i in range(len(pos) - 1)]
    if not slots:
        return 0.0
    return len(set(slots)) / len(slots)


def slot_length_cv(pos):
    gaps = np.diff(pos).astype(float)
    if len(gaps) == 0 or gaps.mean() == 0:
        return float("nan")
    return float(gaps.std() / gaps.mean())


def slot_head_entropy(tokens, pos):
    heads = [tokens[p + 1] for p in pos[:-1] if p + 1 < len(tokens)]
    if not heads:
        return float("nan")
    c = Counter(heads); n = sum(c.values())
    return float(-sum((v / n) * np.log2(v / n) for v in c.values()))


def candidates(tokens, fmin_count, fmin_relfreq, exclude_nonglyph=False):
    c = Counter(tokens.tolist())
    n = len(tokens)
    return [t for t, k in c.items()
            if k >= fmin_count and (k / n) >= fmin_relfreq
            and (not exclude_nonglyph or is_glyph(t))]


def section_T(tokens, cand_tokens, periods, k, delta, want_detail=False):
    """Max phase coherence over qualifying connective candidates.

    A candidate qualifies iff count >= k+1 (n>=k slots) AND passes the
    distinctness gate. Identical rule is applied to observed and null data.
    """
    best_R, detail = 0.0, None
    # precompute positions per candidate token once
    for t in cand_tokens:
        pos = np.flatnonzero(tokens == t)
        if len(pos) - 1 < k:                         # n = m-1 slots
            continue
        if slot_distinctness(tokens, pos) < delta:   # GATE, no credit
            continue
        R, p_at = phase_coherence(pos, periods)
        if R > best_R:
            best_R = R
            if want_detail:
                detail = {"connective": str(t), "n_slots": int(len(pos) - 1),
                          "period_at_max": round(p_at, 2),
                          "slot_length_cv": round(slot_length_cv(pos), 4),
                          "slot_head_entropy": round(slot_head_entropy(tokens, pos), 3)}
    return best_R, detail


# ----------------------------------------------------------------------------
# Null models
# ----------------------------------------------------------------------------
def uniform_shuffle(tokens, rng):
    return tokens[rng.permutation(len(tokens))]


def block_shuffle(tokens, b, rng):
    """Circular block shuffle: preserves local n-grams up to length b, destroys
    long-range regular spacing at scale p (Null B, headline)."""
    n = len(tokens)
    rot = rng.integers(0, n)
    rolled = np.roll(tokens, rot)
    nb = int(np.ceil(n / b))
    blocks = [rolled[i * b:(i + 1) * b] for i in range(nb)]
    order = rng.permutation(len(blocks))
    return np.concatenate([blocks[i] for i in order])


def permutation_test(tokens, cand_tokens, periods, k, delta, T_obs,
                     null_fn, B, rng):
    if not cand_tokens:
        return float("nan"), float("nan"), float("nan")
    null = np.empty(B)
    for i in range(B):
        sh = null_fn(tokens, rng)
        null[i], _ = section_T(sh, cand_tokens, periods, k, delta)
    ge = int(np.sum(null >= T_obs))
    p = (1 + ge) / (1 + B)
    sd = null.std()
    z = (T_obs - null.mean()) / sd if sd > 0 else float("nan")
    return p, z, float(null.mean())


# ----------------------------------------------------------------------------
# Benjamini-Hochberg
# ----------------------------------------------------------------------------
def bh_reject(pvals, q):
    idx = [i for i, p in enumerate(pvals) if p == p]  # drop nan
    ps = sorted((pvals[i], i) for i in idx)
    m = len(ps)
    reject = set()
    thresh = 0.0
    for rank, (p, i) in enumerate(ps, start=1):
        if p <= (rank / m) * q:
            thresh = rank
    for rank, (p, i) in enumerate(ps, start=1):
        if rank <= thresh:
            reject.add(i)
    return reject


# ----------------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------------
def run(conf):
    freeze, seg = load_frozen(conf)
    sig, nul = freeze["signature"], freeze["null"]
    p, eps, step = sig["p"], sig["eps"], sig["period_grid_step"]
    periods = np.arange(p - eps, p + eps + 1e-9, step)
    k, delta = sig["k"], sig["delta"]
    fmin_c, fmin_r = sig["f_min_count"], sig["f_min_relfreq"]
    xnon = sig.get("exclude_nonglyph", False)   # off in v1, on in v2; pinned by lock
    b, B = nul["block_length_b"], nul["n_permutations"]
    rng = np.random.default_rng(SEED)

    rows = []
    for sec in seg["sections"]:
        exact, base = section_tokens(sec)
        row = {"section": sec["section_id"], "n_glyphs": len(exact)}

        cand_robust = candidates(exact, fmin_c, fmin_r, xnon)
        T_rob, detail = section_T(exact, cand_robust, periods, k, delta, want_detail=True)

        if detail is None:
            # no qualifying connective -> not a recursive-enumeration candidate
            evaluable = len(cand_robust) > 0
            row.update({"status": "evaluable" if evaluable else "not_evaluable",
                        "connective": "-", "n_slots": "-", "period": "-",
                        "Pi_robust": round(T_rob, 4) if evaluable else "-",
                        "slot_cv": "-", "p_A": "-", "z_A": "-",
                        "p_B": "-", "z_B": "-",
                        "Pi_aug": "-", "p_B_aug": "-",
                        "result": "NULL (no enumeration signature)" if evaluable
                                   else "NOT EVALUABLE (no candidate connective)"})
            row["_p_B"] = float("nan")
            rows.append(row)
            continue

        # robust nulls
        pA, zA, _ = permutation_test(exact, cand_robust, periods, k, delta, T_rob,
                                     lambda t, r: uniform_shuffle(t, r), B, rng)
        pB, zB, _ = permutation_test(exact, cand_robust, periods, k, delta, T_rob,
                                     lambda t, r: block_shuffle(t, b, r), B, rng)

        # augmented (allograph-normalized) -- reported, never decides PASS
        cand_aug = candidates(base, fmin_c, fmin_r, xnon)
        T_aug, _ = section_T(base, cand_aug, periods, k, delta, want_detail=True)
        pB_aug, _, _ = permutation_test(base, cand_aug, periods, k, delta, T_aug,
                                        lambda t, r: block_shuffle(t, b, r), B, rng)

        row.update({"status": "evaluable",
                    "connective": detail["connective"], "n_slots": detail["n_slots"],
                    "period": detail["period_at_max"], "Pi_robust": round(T_rob, 4),
                    "slot_cv": detail["slot_length_cv"],
                    "p_A": round(pA, 4), "z_A": round(zA, 2),
                    "p_B": round(pB, 4), "z_B": round(zB, 2),
                    "Pi_aug": round(T_aug, 4), "p_B_aug": round(pB_aug, 4),
                    "result": ""})
        row["_p_B"] = pB
        rows.append(row)

    # FDR across evaluable sections on the HEADLINE (Null B, robust)
    eval_idx = [i for i, r in enumerate(rows) if r["status"] == "evaluable"]
    pvals = {i: rows[i]["_p_B"] for i in eval_idx}
    reject = bh_reject([rows[i]["_p_B"] for i in eval_idx], freeze["multiplicity"]["q"])
    reject_idx = {eval_idx[j] for j in reject}
    for i in eval_idx:
        if rows[i]["result"] == "":
            rows[i]["result"] = ("SIGNATURE DETECTED" if i in reject_idx
                                 else "null")

    return freeze, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conf", default=os.path.join(ROOT, "conf", "hakaara"),
                    help="frozen+locked config dir")
    ap.add_argument("--out", default=os.path.join(ROOT, "reports", "hakaara_results.csv"))
    args = ap.parse_args()

    freeze, rows = run(args.conf)
    cols = ["section", "n_glyphs", "status", "connective", "n_slots", "period",
            "Pi_robust", "slot_cv", "z_A", "p_A", "z_B", "p_B",
            "Pi_aug", "p_B_aug", "result"]
    out = args.out
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)

    # console table
    print("\n=== hakaara recursive-enumeration test : ALL sections "
          "(nulls and non-detections included) ===\n")
    hdr = f"{'sec':4} {'N':>5} {'status':12} {'conn':>6} {'slots':>5} {'per':>5} " \
          f"{'Pi_rob':>6} {'sCV':>5} {'z_B':>6} {'p_B':>7} {'Pi_aug':>6} {'result'}"
    print(hdr); print("-" * len(hdr))
    for r in rows:
        print(f"{r['section']:4} {r['n_glyphs']:>5} {r['status']:12} "
              f"{str(r['connective']):>6} {str(r['n_slots']):>5} {str(r['period']):>5} "
              f"{str(r['Pi_robust']):>6} {str(r['slot_cv']):>5} {str(r['z_B']):>6} "
              f"{str(r['p_B']):>7} {str(r['Pi_aug']):>6} {r['result']}")

    det = [r for r in rows if r["result"] == "SIGNATURE DETECTED"]
    print("\n" + "=" * 78)
    print(f"Detections (FDR q={freeze['multiplicity']['q']}, Null B, robust features): "
          f"{len(det)} of {sum(1 for r in rows if r['status']=='evaluable')} evaluable sections")
    print("=" * 78)
    print("\nMANDATORY CAVEAT (applies to every detection above):")
    print(freeze["report_caveat"])
    print(f"\nfull table -> {out}")


if __name__ == "__main__":
    main()
