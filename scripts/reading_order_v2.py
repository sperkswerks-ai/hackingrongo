"""
scripts/reading_order_v2.py

Seven independent statistical tests for the rongorongo reading direction.
Tests 1–4 replicate reading_order_tests.py exactly.
Tests 5–7 are new, architecturally stronger tests.

Tests
-----
1  Conditional-entropy asymmetry   (forward vs reverse bigrams)
2  N-gram model perplexity          (trained model on forward vs reversed sequences)
3  Line-boundary entropy            (within-line vs cross-line bigrams)
4  Recto/verso LOO perplexity       (a→b vs b→a side ordering, bigram + trigram)
5  Cross-side-only LOO perplexity   (restrict to bigrams that cross the a/b boundary;
                                     amplifies the side-ordering signal by removing
                                     the within-side noise that dilutes Test 4)
6  Leave-one-tablet-out (LTOO)      (train on N−1 tablets, predict the held-out tablet
                                     under both ordering directions; more robust than
                                     token-level LOO)
7  Recto/verso mutual information   (MI between side-a and side-b sign distributions;
                                     structural test orthogonal to sequential direction)

Output
------
outputs/analysis/reading_order_v2_report.html
outputs/analysis/reading_order_v2.json  (machine-readable)

Usage
-----
    python scripts/reading_order_v2.py
    python scripts/reading_order_v2.py --corpus data/corpus --tests 5 6 7
    python scripts/reading_order_v2.py --smoke-test
"""

from __future__ import annotations

import argparse
import html as _html
import json
import logging
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

try:
    from hackingrongo.zone_b.sequence_model import NgramModel
except ImportError:
    NgramModel = None  # type: ignore[assignment,misc]

# ---------------------------------------------------------------------------
# Types and helpers shared with reading_order_tests.py
# ---------------------------------------------------------------------------

TabletRow = tuple[str, int, str]   # (side, line_num, token)

_SIDE_AB: dict[str, int] = {"a": 0, "b": 1, "c": 2}
_SIDE_BA: dict[str, int] = {"b": 0, "a": 1, "c": 2}


def _safe_int(v: object) -> int:
    try:
        return int(v)  # type: ignore[arg-type]
    except (ValueError, TypeError):
        return 0


def _glyph_tokens(g: dict) -> list[str]:
    hc = g.get("horley_code")
    if hc:
        return [str(hc)]
    comps = g.get("horley_components") or []
    return [str(c) for c in comps] if comps else []


def _load_tablet(path: Path, side_order: dict[str, int]) -> list[TabletRow]:
    data = json.loads(path.read_text(encoding="utf-8"))
    glyphs = data.get("glyphs", [])
    by_line: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for g in glyphs:
        side = str(g.get("side", "a")).lower()
        by_line[(side, _safe_int(g.get("line", 0)))].append(g)
    rows: list[TabletRow] = []
    for (side, line_num), line_glyphs in sorted(
        by_line.items(), key=lambda kv: (side_order.get(kv[0][0], 9), kv[0][1])
    ):
        line_glyphs = sorted(line_glyphs, key=lambda g: _safe_int(g.get("glyph_num", 0)))
        n_inv = sum(1 for g in line_glyphs if g.get("inverted", False))
        if line_glyphs and n_inv > len(line_glyphs) / 2:
            line_glyphs = list(reversed(line_glyphs))
        for g in line_glyphs:
            for tok in _glyph_tokens(g):
                rows.append((side, line_num, tok))
    return rows


def load_corpus(corpus_dir: Path, side_order: dict[str, int]) -> dict[str, list[TabletRow]]:
    result: dict[str, list[TabletRow]] = {}
    for path in sorted(corpus_dir.glob("*.json")):
        if path.stem.upper() in {"A"}:
            continue  # excluded (European wood)
        rows = _load_tablet(path, side_order)
        if rows:
            result[path.stem] = rows
    return result


def _seqs(corpus: dict[str, list[TabletRow]]) -> list[list[str]]:
    return [[tok for _, _, tok in rows] for rows in corpus.values()]


# ---------------------------------------------------------------------------
# Entropy / perplexity primitives
# ---------------------------------------------------------------------------

def _bigram_counts(sequences: list[list[str]]) -> tuple[Counter, Counter]:
    bigrams: Counter = Counter()
    unigrams: Counter = Counter()
    for seq in sequences:
        for i in range(len(seq) - 1):
            bigrams[(seq[i], seq[i + 1])] += 1
            unigrams[seq[i]] += 1
    return bigrams, unigrams


def _conditional_entropy(bigrams: Counter, unigrams: Counter) -> float:
    total = sum(bigrams.values())
    if total == 0:
        return 0.0
    h = 0.0
    for (s, t), cnt in bigrams.items():
        p_st = cnt / total
        p_t_given_s = cnt / unigrams[s]
        h -= p_st * math.log2(p_t_given_s)
    return h


def _loo_perplexity(sequences: list[list[str]], order: int = 2, alpha: float = 0.5) -> float:
    vocab: set[str] = set()
    for seq in sequences:
        vocab.update(seq)
    V = len(vocab) + 2

    def _count(seqs: list[list[str]]) -> tuple[Counter, Counter]:
        counts: Counter = Counter()
        ctx: Counter = Counter()
        for seq in seqs:
            padded = ["<BOS>"] * (order - 1) + seq + ["<EOS>"]
            for i in range(order - 1, len(padded)):
                ng = tuple(padded[i - order + 1 : i + 1])
                counts[ng] += 1
                ctx[ng[:-1]] += 1
        return counts, ctx

    def _score(seq: list[str], counts: Counter, ctx: Counter) -> tuple[float, int]:
        padded = ["<BOS>"] * (order - 1) + seq + ["<EOS>"]
        lp = 0.0
        for i in range(order - 1, len(padded)):
            ng = tuple(padded[i - order + 1 : i + 1])
            lp += math.log2((counts[ng] + alpha) / (ctx[ng[:-1]] + alpha * V))
        return lp, len(padded) - (order - 1)

    total_lp = 0.0
    total_n = 0
    for i in range(len(sequences)):
        train = [s for j, s in enumerate(sequences) if j != i]
        counts, ctx = _count(train)
        lp, n = _score(sequences[i], counts, ctx)
        total_lp += lp
        total_n += n
    return 2.0 ** (-total_lp / total_n) if total_n else float("inf")


def _ngram_perplexity(model: Any, sequences: list[list[str]]) -> float:
    total_lp = 0.0
    total_n = 0
    for seq in sequences:
        if len(seq) < model.order:
            continue
        total_lp += model.score(seq)
        total_n += len(seq)
    return 2.0 ** (-total_lp / total_n) if total_n else float("inf")


# ---------------------------------------------------------------------------
# Tests 1–4 (replicated from reading_order_tests.py)
# ---------------------------------------------------------------------------

def test1_conditional_entropy(sequences: list[list[str]]) -> dict:
    rev = [list(reversed(s)) for s in sequences]
    bg_f, uni_f = _bigram_counts(sequences)
    bg_r, uni_r = _bigram_counts(rev)
    h_f = _conditional_entropy(bg_f, uni_f)
    h_r = _conditional_entropy(bg_r, uni_r)
    delta = h_r - h_f
    if delta > 0.05:
        direction, verdict = "forward", f"forward entropy lower by {delta:.4f} bits"
    elif delta < -0.05:
        direction, verdict = "reverse", f"reverse entropy lower by {-delta:.4f} bits"
    else:
        direction, verdict = "neutral", f"no directional preference (Δ = {delta:+.4f} bits)"
    return {"h_forward": h_f, "h_reverse": h_r, "delta": delta,
            "direction": direction, "verdict": verdict}


def test2_perplexity(model: Any, sequences: list[list[str]]) -> dict:
    if model is None:
        return {"verdict": "skipped — no model", "direction": "unknown"}
    rev = [list(reversed(s)) for s in sequences]
    ppl_f = _ngram_perplexity(model, sequences)
    ppl_r = _ngram_perplexity(model, rev)
    ratio = ppl_r / ppl_f if ppl_f > 0 else float("inf")
    direction = "forward" if ppl_f < ppl_r else "reverse"
    verdict = (
        f"model prefers {'forward' if ppl_f < ppl_r else 'reverse'} "
        f"({ratio:.3f}× PPL ratio)"
    )
    return {"ppl_forward": ppl_f, "ppl_reverse": ppl_r, "ratio": ratio,
            "direction": direction, "verdict": verdict}


def test3_line_boundary(corpus: dict[str, list[TabletRow]]) -> dict:
    w_bg: Counter = Counter()
    w_uni: Counter = Counter()
    c_bg: Counter = Counter()
    c_uni: Counter = Counter()
    for rows in corpus.values():
        for i in range(len(rows) - 1):
            s0, l0, t0 = rows[i]
            s1, l1, t1 = rows[i + 1]
            if s0 == s1 and l0 == l1:
                w_bg[(t0, t1)] += 1; w_uni[t0] += 1
            else:
                c_bg[(t0, t1)] += 1; c_uni[t0] += 1
    h_w = _conditional_entropy(w_bg, w_uni)
    h_c = _conditional_entropy(c_bg, c_uni)
    delta = h_c - h_w
    direction = "confirmed" if delta > 0.1 else ("unexpected" if delta < -0.1 else "neutral")
    verdict = f"cross-line H higher by {delta:.4f} bits" if delta > 0 else f"Δ={delta:.4f}"
    return {"h_within": h_w, "h_cross": h_c, "delta": delta,
            "direction": direction, "verdict": verdict,
            "n_within": sum(w_bg.values()), "n_cross": sum(c_bg.values())}


def test4_recto_verso(corpus_dir: Path) -> dict:
    corp_ab = load_corpus(corpus_dir, _SIDE_AB)
    corp_ba = load_corpus(corpus_dir, _SIDE_BA)
    ppl_ab2 = _loo_perplexity(_seqs(corp_ab), order=2)
    ppl_ab3 = _loo_perplexity(_seqs(corp_ab), order=3)
    ppl_ba2 = _loo_perplexity(_seqs(corp_ba), order=2)
    ppl_ba3 = _loo_perplexity(_seqs(corp_ba), order=3)
    votes_ab = int(ppl_ab2 < ppl_ba2) + int(ppl_ab3 < ppl_ba3)
    preferred = "ab" if votes_ab == 2 else ("ba" if votes_ab == 0 else "mixed")
    margin2 = abs(ppl_ab2 - ppl_ba2)
    margin3 = abs(ppl_ab3 - ppl_ba3)
    verdict = (
        f"a→b preferred (margin bigram={margin2:.3f}, trigram={margin3:.3f})"
        if preferred == "ab" else
        f"b→a preferred (margin bigram={margin2:.3f}, trigram={margin3:.3f})"
        if preferred == "ba" else
        f"mixed signal — bigram and trigram disagree (margin={margin2:.3f}/{margin3:.3f})"
    )
    return {"ppl_ab2": ppl_ab2, "ppl_ab3": ppl_ab3,
            "ppl_ba2": ppl_ba2, "ppl_ba3": ppl_ba3,
            "votes_ab": votes_ab, "preferred_order": preferred,
            "margin_bigram": margin2, "margin_trigram": margin3,
            "verdict": verdict}


# ---------------------------------------------------------------------------
# Test 5: Cross-side-only LOO perplexity
# ---------------------------------------------------------------------------

def _extract_cross_side_bigrams(
    corpus: dict[str, list[TabletRow]],
) -> list[list[str]]:
    """Extract only the (last-sign-on-side-a, first-sign-on-side-b) transitions."""
    cross_seqs: list[list[str]] = []
    for rows in corpus.values():
        for i in range(len(rows) - 1):
            s0, _, t0 = rows[i]
            s1, _, t1 = rows[i + 1]
            if s0 != s1:   # side boundary
                cross_seqs.append([t0, t1])
    return cross_seqs


def test5_cross_side_perplexity(corpus_dir: Path) -> dict:
    """LOO perplexity restricted to cross-side boundary bigrams only.

    Removing within-side bigrams isolates the signal that drives the side
    ordering.  A strong result here amplifies a weak Test 4; if Test 5 is
    also weak the side-ordering signal is genuinely marginal.
    """
    corp_ab = load_corpus(corpus_dir, _SIDE_AB)
    corp_ba = load_corpus(corpus_dir, _SIDE_BA)

    cross_ab = _extract_cross_side_bigrams(corp_ab)
    cross_ba = _extract_cross_side_bigrams(corp_ba)

    n_cross = len(cross_ab)
    if n_cross < 10:
        return {
            "verdict": f"insufficient cross-side bigrams ({n_cross}); skipped",
            "direction": "unknown",
            "n_cross_bigrams": n_cross,
        }

    ppl_ab = _loo_perplexity(cross_ab, order=2)
    ppl_ba = _loo_perplexity(cross_ba, order=2)

    margin = abs(ppl_ab - ppl_ba)
    preferred = "ab" if ppl_ab < ppl_ba else "ba"
    strength = "strong" if margin > 1.0 else ("moderate" if margin > 0.1 else "weak")
    verdict = (
        f"{preferred} preferred; margin={margin:.3f} ({strength} signal) "
        f"over {n_cross} cross-side transitions"
    )
    return {
        "n_cross_bigrams": n_cross,
        "ppl_ab": ppl_ab,
        "ppl_ba": ppl_ba,
        "margin": margin,
        "preferred_order": preferred,
        "signal_strength": strength,
        "verdict": verdict,
    }


# ---------------------------------------------------------------------------
# Test 6: Leave-one-tablet-out (LTOO) prediction
# ---------------------------------------------------------------------------

def test6_ltoo(corpus_dir: Path, order: int = 2, alpha: float = 0.5) -> dict:
    """Train on N−1 tablets, evaluate on the held-out tablet under both orderings.

    Repeats for every tablet.  The ordering that wins more leave-out rounds
    is the preferred direction.  Reports wins, median margin, and per-tablet
    detail.
    """
    corp_ab = load_corpus(corpus_dir, _SIDE_AB)
    corp_ba = load_corpus(corpus_dir, _SIDE_BA)

    tablets_ab = {tid: [tok for _, _, tok in rows] for tid, rows in corp_ab.items()}
    tablets_ba = {tid: [tok for _, _, tok in rows] for tid, rows in corp_ba.items()}
    tablet_ids = sorted(tablets_ab.keys())

    vocab: set[str] = set()
    for seq in tablets_ab.values():
        vocab.update(seq)
    V = len(vocab) + 2

    def _count_seqs(seqs: list[list[str]]) -> tuple[Counter, Counter]:
        counts: Counter = Counter()
        ctx: Counter = Counter()
        for seq in seqs:
            padded = ["<BOS>"] * (order - 1) + seq + ["<EOS>"]
            for i in range(order - 1, len(padded)):
                ng = tuple(padded[i - order + 1 : i + 1])
                counts[ng] += 1
                ctx[ng[:-1]] += 1
        return counts, ctx

    def _score_seq(seq: list[str], counts: Counter, ctx: Counter) -> float:
        padded = ["<BOS>"] * (order - 1) + seq + ["<EOS>"]
        lp = 0.0
        for i in range(order - 1, len(padded)):
            ng = tuple(padded[i - order + 1 : i + 1])
            lp += math.log2((counts[ng] + alpha) / (ctx[ng[:-1]] + alpha * V))
        n = len(padded) - (order - 1)
        return 2.0 ** (-lp / n) if n else float("inf")

    wins_ab = 0
    wins_ba = 0
    margins: list[float] = []
    details: list[dict] = []

    for held_id in tablet_ids:
        train_ab = [seq for tid, seq in tablets_ab.items() if tid != held_id]
        train_ba = [seq for tid, seq in tablets_ba.items() if tid != held_id]
        c_ab, x_ab = _count_seqs(train_ab)
        c_ba, x_ba = _count_seqs(train_ba)
        ppl_ab = _score_seq(tablets_ab[held_id], c_ab, x_ab)
        ppl_ba = _score_seq(tablets_ba[held_id], c_ba, x_ba)
        winner = "ab" if ppl_ab < ppl_ba else "ba"
        margin = abs(ppl_ab - ppl_ba)
        if winner == "ab":
            wins_ab += 1
        else:
            wins_ba += 1
        margins.append(margin)
        details.append({"tablet": held_id, "ppl_ab": round(ppl_ab, 3),
                        "ppl_ba": round(ppl_ba, 3), "winner": winner,
                        "margin": round(margin, 3)})

    n_total = len(tablet_ids)
    preferred = "ab" if wins_ab > wins_ba else ("ba" if wins_ba > wins_ab else "tied")
    median_margin = sorted(margins)[len(margins) // 2] if margins else 0.0
    verdict = (
        f"{preferred} preferred: {max(wins_ab, wins_ba)}/{n_total} tablets; "
        f"median margin={median_margin:.3f}"
    )
    return {
        "n_tablets": n_total,
        "wins_ab": wins_ab,
        "wins_ba": wins_ba,
        "preferred_order": preferred,
        "median_margin": round(median_margin, 3),
        "ngram_order": order,
        "verdict": verdict,
        "per_tablet": details,
    }


# ---------------------------------------------------------------------------
# Test 7: Recto/verso mutual information
# ---------------------------------------------------------------------------

def test7_mutual_information(corpus: dict[str, list[TabletRow]]) -> dict:
    """MI between side-a and side-b sign frequency distributions.

    High MI → the two sides share structural vocabulary (same scribe /
    related content).  Low MI → independent inscriptions.  This is a
    structural test orthogonal to sequential direction.
    """
    side_a_counts: Counter = Counter()
    side_b_counts: Counter = Counter()
    co_counts: Counter = Counter()   # (a_sign, b_sign) pairs per tablet

    n_tablets_both_sides = 0
    for rows in corpus.values():
        a_signs = [tok for side, _, tok in rows if side == "a"]
        b_signs = [tok for side, _, tok in rows if side == "b"]
        if not a_signs or not b_signs:
            continue
        n_tablets_both_sides += 1
        a_cnt: Counter = Counter(a_signs)
        b_cnt: Counter = Counter(b_signs)
        for a_sign, na in a_cnt.items():
            for b_sign, nb in b_cnt.items():
                co_counts[(a_sign, b_sign)] += na * nb
            side_a_counts[a_sign] += na
        for b_sign, nb in b_cnt.items():
            side_b_counts[b_sign] += nb

    if not co_counts:
        return {"verdict": "no tablets with both sides", "direction": "unknown"}

    n_co = sum(co_counts.values())
    n_a  = sum(side_a_counts.values())
    n_b  = sum(side_b_counts.values())

    mi = 0.0
    for (a_sign, b_sign), cnt in co_counts.items():
        p_ab = cnt / n_co
        p_a  = side_a_counts[a_sign] / n_a
        p_b  = side_b_counts[b_sign] / n_b
        if p_a > 0 and p_b > 0 and p_ab > 0:
            mi += p_ab * math.log2(p_ab / (p_a * p_b))

    h_a = -sum((c / n_a) * math.log2(c / n_a) for c in side_a_counts.values() if c > 0)
    h_b = -sum((c / n_b) * math.log2(c / n_b) for c in side_b_counts.values() if c > 0)
    nmi = mi / ((h_a + h_b) / 2) if (h_a + h_b) > 0 else 0.0

    interpretation = (
        "high shared vocabulary (related content / same scribe)" if nmi > 0.3
        else ("moderate overlap" if nmi > 0.1
              else "low overlap (independent or divergent content)")
    )
    verdict = f"NMI={nmi:.4f} ({interpretation}); {n_tablets_both_sides} bilateral tablets"
    return {
        "mutual_information_bits": round(mi, 4),
        "h_side_a_bits": round(h_a, 4),
        "h_side_b_bits": round(h_b, 4),
        "nmi": round(nmi, 4),
        "n_tablets_both_sides": n_tablets_both_sides,
        "n_unique_side_a": len(side_a_counts),
        "n_unique_side_b": len(side_b_counts),
        "interpretation": interpretation,
        "verdict": verdict,
    }


# ---------------------------------------------------------------------------
# HTML report
# ---------------------------------------------------------------------------

_CSS = """\
:root{--bg:#0d0f12;--surface:#161920;--surface2:#1e2229;--border:#2a2e38;
      --text:#d0d4dc;--muted:#6b7280;--accent:#c4a96d;
      --pass:#4ade80;--warn:#facc15;--fail:#f87171;--neutral:#94a3b8;}
*{box-sizing:border-box;margin:0;padding:0;}
body{background:var(--bg);color:var(--text);
     font-family:'Cormorant Garamond','Palatino Linotype',Georgia,serif;
     font-size:16px;line-height:1.65;}
.wrap{max-width:980px;margin:0 auto;padding:52px 28px;}
h1{font-size:28px;font-weight:600;color:var(--accent);margin-bottom:4px;}
.meta{color:var(--muted);font-family:'JetBrains Mono',monospace;font-size:11px;
      margin-bottom:44px;}
.test-block{background:var(--surface);border:1px solid var(--border);
            border-radius:6px;margin-bottom:28px;overflow:hidden;}
.test-header{padding:14px 20px;border-bottom:1px solid var(--border);
             display:flex;gap:14px;align-items:baseline;}
.test-num{font-family:'JetBrains Mono',monospace;font-size:11px;
          color:var(--muted);min-width:54px;}
.test-title{font-weight:600;font-size:18px;flex:1;}
.verdict-chip{font-family:'JetBrains Mono',monospace;font-size:10px;
              padding:3px 10px;border-radius:3px;white-space:nowrap;}
.verdict-pass{background:rgba(74,222,128,.15);color:var(--pass);}
.verdict-warn{background:rgba(250,204,21,.15);color:var(--warn);}
.verdict-fail{background:rgba(248,113,113,.15);color:var(--fail);}
.verdict-neutral{background:rgba(148,163,184,.1);color:var(--neutral);}
.test-body{padding:20px 24px;}
.kv-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(200px,1fr));gap:12px 24px;}
.kv{font-family:'JetBrains Mono',monospace;font-size:12px;}
.kv-label{color:var(--muted);font-size:10px;text-transform:uppercase;letter-spacing:.06em;}
.kv-value{color:var(--text);font-size:14px;margin-top:2px;}
.verdict-text{margin-top:16px;font-size:14px;color:var(--text);line-height:1.7;}
.caveat{font-size:12px;color:var(--muted);margin-top:8px;font-style:italic;}
table.detail{width:100%;border-collapse:collapse;margin-top:16px;
             font-family:'JetBrains Mono',monospace;font-size:11px;}
table.detail th{text-align:left;padding:5px 10px;color:var(--muted);
                border-bottom:1px solid var(--border);}
table.detail td{padding:4px 10px;border-bottom:1px solid rgba(42,46,56,.5);}
table.detail tr:last-child td{border-bottom:none;}
.summary-bar{margin-top:36px;padding:20px 24px;background:var(--surface2);
             border-radius:6px;font-family:'JetBrains Mono',monospace;font-size:12px;}
.summary-title{font-size:13px;color:var(--accent);margin-bottom:10px;font-weight:600;}
"""


def _chip(result: dict) -> str:
    direction = result.get("direction", result.get("preferred_order", ""))
    verdict   = result.get("verdict", "")
    if direction in ("forward", "ab") or "strong" in verdict or "confirmed" in verdict:
        cls = "verdict-pass"
    elif direction in ("neutral", "mixed", "tied", "unknown") or "weak" in verdict:
        cls = "verdict-warn"
    elif direction in ("reverse", "ba") or "unexpected" in verdict:
        cls = "verdict-fail"
    else:
        cls = "verdict-neutral"
    return f'<span class="verdict-chip {cls}">{_html.escape(direction or "?")}</span>'


def _kv(label: str, value: Any) -> str:
    if isinstance(value, float):
        value = f"{value:.4f}"
    return (
        f'<div class="kv">'
        f'<div class="kv-label">{_html.escape(str(label))}</div>'
        f'<div class="kv-value">{_html.escape(str(value))}</div>'
        f'</div>'
    )


def _detail_table(rows: list[dict], cols: list[str]) -> str:
    ths = "".join(f"<th>{_html.escape(c)}</th>" for c in cols)
    trs = ""
    for row in rows:
        tds = "".join(f"<td>{_html.escape(str(row.get(c, '')))}</td>" for c in cols)
        trs += f"<tr>{tds}</tr>"
    return f'<table class="detail"><thead><tr>{ths}</tr></thead><tbody>{trs}</tbody></table>'


TEST_META = [
    (1, "Conditional Entropy Asymmetry",
     "H(Sₙ | Sₙ₋₁) forward vs. reverse. Lower forward entropy → left-to-right confirmed.",
     "original"),
    (2, "N-gram Model Perplexity",
     "Trained model perplexity on forward vs. reversed sequences.",
     "original"),
    (3, "Line-Boundary Entropy",
     "Cross-line bigrams should have higher entropy than within-line bigrams if boundaries are real.",
     "original"),
    (4, "Recto/Verso LOO Perplexity",
     "Token-level LOO perplexity under a→b vs b→a side ordering (bigram + trigram). "
     "Note: within-side bigrams dominate and dilute this signal — see Test 5.",
     "original"),
    (5, "Cross-Side-Only LOO Perplexity",
     "Restrict LOO to bigrams that cross the a/b boundary. Removes within-side noise. "
     "A strong result here validates a weak Test 4; a continued weak result means the "
     "side-ordering signal is genuinely marginal.",
     "new"),
    (6, "Leave-One-Tablet-Out (LTOO)",
     "Train on N−1 tablets, predict the held-out tablet under both orderings. "
     "Robust to tablet-level idiosyncrasy; more conservative than token LOO.",
     "new"),
    (7, "Recto/Verso Mutual Information",
     "MI between side-a and side-b sign distributions across tablets. "
     "High MI → shared vocabulary (same scribe / related content). "
     "Structural test orthogonal to sequential direction.",
     "new"),
]


def build_html_report(results: dict) -> str:
    blocks: list[str] = []

    for (test_num, title, description, kind) in TEST_META:
        key = f"test{test_num}"
        if key not in results:
            continue
        r = results[key]
        badge = (" <span style='color:var(--accent);font-size:11px;font-family:monospace'>"
                 "[NEW]</span>" if kind == "new" else "")

        kv_items = {
            1: [("H forward", r.get("h_forward")), ("H reverse", r.get("h_reverse")),
                ("delta", r.get("delta"))],
            2: [("PPL forward", r.get("ppl_forward")), ("PPL reverse", r.get("ppl_reverse")),
                ("ratio", r.get("ratio"))],
            3: [("H within-line", r.get("h_within")), ("H cross-line", r.get("h_cross")),
                ("delta", r.get("delta")), ("N within", r.get("n_within")),
                ("N cross", r.get("n_cross"))],
            4: [("PPL a→b bigram", r.get("ppl_ab2")), ("PPL b→a bigram", r.get("ppl_ba2")),
                ("PPL a→b trigram", r.get("ppl_ab3")), ("PPL b→a trigram", r.get("ppl_ba3")),
                ("margin bigram", r.get("margin_bigram")), ("margin trigram", r.get("margin_trigram"))],
            5: [("PPL a→b", r.get("ppl_ab")), ("PPL b→a", r.get("ppl_ba")),
                ("margin", r.get("margin")), ("N cross-side", r.get("n_cross_bigrams")),
                ("signal strength", r.get("signal_strength"))],
            6: [("N tablets", r.get("n_tablets")), ("wins a→b", r.get("wins_ab")),
                ("wins b→a", r.get("wins_ba")), ("median margin", r.get("median_margin")),
                ("n-gram order", r.get("ngram_order"))],
            7: [("MI (bits)", r.get("mutual_information_bits")), ("NMI", r.get("nmi")),
                ("H side-a", r.get("h_side_a_bits")), ("H side-b", r.get("h_side_b_bits")),
                ("tablets (bilateral)", r.get("n_tablets_both_sides"))],
        }.get(test_num, [])

        kv_html = '<div class="kv-grid">'
        for label, val in kv_items:
            if val is not None:
                kv_html += _kv(label, val)
        kv_html += "</div>"

        detail_html = ""
        if test_num == 6 and "per_tablet" in r:
            detail_html = _detail_table(
                r["per_tablet"][:20],
                ["tablet", "ppl_ab", "ppl_ba", "winner", "margin"],
            )

        caveat_html = ""
        if test_num == 4:
            caveat_html = (
                '<p class="caveat">⚠ Test 4 margin is historically 0.03–0.04 PPL. '
                "See Test 5 for the isolated cross-side signal.</p>"
            )

        blocks.append(
            f'<div class="test-block">'
            f'<div class="test-header">'
            f'<span class="test-num">TEST {test_num}</span>'
            f'<span class="test-title">{_html.escape(title)}{badge}</span>'
            f'{_chip(r)}'
            f'</div>'
            f'<div class="test-body">'
            f'<p style="color:var(--muted);font-size:13px;margin-bottom:14px">'
            f'{_html.escape(description)}</p>'
            f'{kv_html}'
            f'<div class="verdict-text">{_html.escape(r.get("verdict", ""))}</div>'
            f'{caveat_html}'
            f'{detail_html}'
            f'</div></div>'
        )

    # Summary convergence bar
    converging = [
        t for t in [1, 2, 3, 4, 5, 6]
        if results.get(f"test{t}", {}).get("direction", "") in ("forward", "ab", "confirmed")
        or results.get(f"test{t}", {}).get("preferred_order", "") in ("ab",)
    ]
    summary_html = (
        f'<div class="summary-bar">'
        f'<div class="summary-title">Convergence summary</div>'
        f'<div>{len(converging)}/6 directional tests favour a→b (forward / recto-first) ordering. '
        f"A finding is robust only when all tests that measure the same signal are consistent.</div>"
        f"</div>"
    )

    return (
        "<!DOCTYPE html><html lang='en'>"
        "<head><meta charset='utf-8'>"
        "<title>Rongorongo Reading Order v2</title>"
        f"<style>{_CSS}</style></head>"
        "<body><div class='wrap'>"
        "<h1>Rongorongo Reading-Direction Tests (v2)</h1>"
        f"<div class='meta'>corpus: {results.get('corpus_tablets',0)} tablets · "
        f"{results.get('corpus_tokens',0):,} tokens · "
        f"tests run: {results.get('tests_run',[])}</div>"
        + "".join(blocks)
        + summary_html
        + "</div></body></html>"
    )


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

def _smoke_test() -> None:
    seqs = [
        ["a", "b", "c", "a", "b"],
        ["b", "a", "c", "b", "a"],
        ["a", "a", "b", "c"],
    ]
    r1 = test1_conditional_entropy(seqs)
    assert "h_forward" in r1, "test1 failed"
    corpus_mock: dict[str, list[TabletRow]] = {
        "T1": [("a", 1, "x"), ("a", 1, "y"), ("b", 2, "z"), ("b", 2, "w")],
        "T2": [("a", 1, "y"), ("a", 1, "z"), ("b", 2, "x")],
    }
    r3 = test3_line_boundary(corpus_mock)
    assert "h_within" in r3, "test3 failed"
    r7 = test7_mutual_information(corpus_mock)
    assert "nmi" in r7, "test7 failed"
    log.info("Smoke test passed.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Seven reading-direction tests for rongorongo (v2).",
    )
    p.add_argument("--corpus", type=Path, default=None)
    p.add_argument("--model", type=Path, default=None)
    p.add_argument(
        "--tests", nargs="+", type=int, default=[1, 3, 4, 5, 6, 7],
        metavar="N", help="Which tests to run (default: all except 2).",
    )
    p.add_argument(
        "--output", type=Path,
        default=PROJECT_ROOT / "outputs" / "analysis" / "reading_order_v2_report.html",
    )
    p.add_argument(
        "--output-json", type=Path,
        default=PROJECT_ROOT / "outputs" / "analysis" / "reading_order_v2.json",
    )
    p.add_argument("--smoke-test", action="store_true")
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    if args.smoke_test:
        _smoke_test()
        return

    corpus_dir = args.corpus
    if corpus_dir is None:
        try:
            from omegaconf import OmegaConf
            cfg = OmegaConf.load(PROJECT_ROOT / "conf" / "config.yaml")
            corpus_dir = PROJECT_ROOT / cfg.paths.corpus_dir
        except Exception as exc:
            log.debug("Could not resolve corpus_dir from config: %s", exc)
    if corpus_dir is None or not corpus_dir.exists():
        log.error("Corpus not found. Pass --corpus <path>.")
        sys.exit(1)

    model = None
    if 2 in args.tests and NgramModel is not None:
        model_path = args.model
        if model_path is None:
            try:
                from omegaconf import OmegaConf
                cfg = OmegaConf.load(PROJECT_ROOT / "conf" / "config.yaml")
                model_path = PROJECT_ROOT / cfg.paths.outputs_dir / "zone_b" / "sequence_model.json"
            except Exception:
                pass
        if model_path and model_path.exists():
            model = NgramModel.load(model_path)
            log.info("Model loaded: order=%d vocab=%d", model.order, len(model.vocab))
        else:
            log.warning("Model not found — Test 2 skipped.")

    log.info("Loading corpus from %s …", corpus_dir)
    corpus_ab = load_corpus(corpus_dir, _SIDE_AB)
    seqs = _seqs(corpus_ab)
    n_tokens = sum(len(s) for s in seqs)
    log.info("%d tablets, %d tokens.", len(corpus_ab), n_tokens)

    tests = set(args.tests)
    results: dict[str, Any] = {
        "corpus_tablets": len(corpus_ab),
        "corpus_tokens": n_tokens,
        "tests_run": sorted(tests),
    }

    if 1 in tests:
        log.info("Test 1: conditional entropy …")
        results["test1"] = test1_conditional_entropy(seqs)
    if 2 in tests and model is not None:
        log.info("Test 2: n-gram perplexity …")
        results["test2"] = test2_perplexity(model, seqs)
    if 3 in tests:
        log.info("Test 3: line-boundary entropy …")
        results["test3"] = test3_line_boundary(corpus_ab)
    if 4 in tests:
        log.info("Test 4: recto/verso LOO …")
        results["test4"] = test4_recto_verso(corpus_dir)
    if 5 in tests:
        log.info("Test 5: cross-side-only LOO …")
        results["test5"] = test5_cross_side_perplexity(corpus_dir)
    if 6 in tests:
        log.info("Test 6: LTOO …")
        results["test6"] = test6_ltoo(corpus_dir)
    if 7 in tests:
        log.info("Test 7: mutual information …")
        results["test7"] = test7_mutual_information(corpus_ab)

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(results, indent=2), encoding="utf-8")
    log.info("JSON → %s", args.output_json)

    html_str = build_html_report(results)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(html_str, encoding="utf-8")
    log.info("HTML → %s", args.output)

    print(f"\nConclusion summary:")
    for key in [f"test{i}" for i in sorted(tests)]:
        if key in results:
            r = results[key]
            print(f"  {key}: {r.get('verdict', r.get('interpretation', '?'))}")


if __name__ == "__main__":
    main()
