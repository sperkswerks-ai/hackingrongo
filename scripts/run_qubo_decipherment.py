# ============================================================================
# DEPRECATED — SYLLABIC SUBSTITUTION-CIPHER TRACK (set down 2026-06, in place).
# Part of the sign→phoneme substitution-cipher hypothesis, which was tested and
# set down as a recorded NEGATIVE RESULT — preserved as an archive, NOT fixed,
# tuned, or deleted. Do not extend this module. The structural/logographic track
# supersedes it. Full rationale + on-disk numbers: DEPRECATED_SYLLABIC.md (root).
# ============================================================================
"""
run_qubo_decipherment.py — Layer 4Q: QUBO formulation of the sign→phoneme
assignment problem.

Reformulates finding the best rongorongo sign→phoneme map as a Quadratic
Unconstrained Binary Optimisation (QUBO) and solves it via:

  (a) D-Wave Leap cloud QPU  — if dwave-ocean-sdk is installed and
      DWAVE_API_TOKEN is set (or --dwave-token is given).
  (b) neal simulated annealing — CPU fallback, no account needed.
  (c) dimod TabuSampler      — second CPU fallback.

QUBO objective
--------------
Minimise:

  H = -Σ_{s,p} unigram_score(s→p) * x_{s,p}          (LM objective)
    + λ1 * Σ_s  (Σ_p x_{s,p} - 1)²                   (one sign → one phoneme)
    + λ2 * Σ_p  (max(0, Σ_s x_{s,p} - k))²            (≤k signs per phoneme)

where x_{s,p} ∈ {0,1}.  "Minimise H" ≡ "maximise LM score subject to
assignment constraints."

The quadratic penalty expansion of (Σ_p x_{s,p} - 1)² produces:
  • diagonal terms: -1 * λ1 for each x_{s,p}  (encourages selection)
  • off-diagonal: +2 * λ1 for pairs (x_{s,p}, x_{s,q}), p≠q (same sign)

The capacity penalty Σ_p (Σ_s x_{s,p} - k)² is expanded and added for
pairs (x_{s,p}, x_{t,p}), s≠t (same phoneme):
  +2 * λ2 * (bigram_cooccurrence(s,t,p) if bigrams, else 1)

Usage
-----
    python scripts/run_qubo_decipherment.py \\
        --corpus-dir data/corpus \\
        --lm-dir     data/language_models \\
        --num-reads  1000 \\
        --output     outputs/decipherment/qubo_result.json

    # MCMC warm-start (uses top hypothesis as initial state)
    python scripts/run_qubo_decipherment.py \\
        --init-from outputs/decipherment/ranking.json

    # Force D-Wave QPU
    python scripts/run_qubo_decipherment.py \\
        --solver dwave --dwave-token <TOKEN>
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from hackingrongo.data.rapa_nui_corpus import NGramLM  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)


# External-evidence logographic taxograms used in the mixed model.
# These are pinned as hard variables in QUBO and excluded from LM terms.
LOGOGRAPHIC_TAXOGRAMS: dict[str, str] = {
    "600": "manu",
    "700": "ika",
    "280": "honu",
    "690": "tangata manu",
}


# ---------------------------------------------------------------------------
# Data loading  (shared with measure_pgood.py)
# ---------------------------------------------------------------------------

def _load_corpus(corpus_dir: Path) -> list[list[str]]:
    sequences: list[list[str]] = []
    for path in sorted(corpus_dir.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        glyphs = data.get("glyphs", [])
        seq = [str(g["barthel_code"]) for g in glyphs if g.get("barthel_code")]
        if len(seq) >= 3:
            sequences.append(seq)
    return sequences


def _load_lms(lm_dir: Path) -> list[NGramLM]:
    lms: list[NGramLM] = []
    for name in ["pre_contact_lm", "post_contact_lm"]:
        path = lm_dir / f"{name}.json"
        if path.exists():
            log.info("Loading %s …", path.name)
            lms.append(NGramLM.load(path))
        else:
            log.warning("Not found, skipping: %s", path)
    if not lms:
        log.error("No LMs found in %s — run build_language_models.py first.", lm_dir)
        sys.exit(1)
    return lms


def _phoneme_inventory(lms: list[NGramLM]) -> list[str]:
    """Canonical Rapa Nui syllable inventory (shared with MCMC and p_good).

    The QUBO search space must match the inventory Zone C MCMC explores
    and measure_pgood.py characterises, or the classical-vs-quantum
    complexity comparison is over different problems.  The *lms*
    argument is kept for signature compatibility but is no longer
    consulted: deriving the inventory from LM vocabularies let tokenizer
    artifacts (phonotactically impossible syllables) inflate the space.
    """
    from hackingrongo.data.phoneme_inventory import RAPA_NUI_SYLLABLES
    return list(RAPA_NUI_SYLLABLES)


# ---------------------------------------------------------------------------
# LM scoring helpers
# ---------------------------------------------------------------------------

def _unigram_score(phoneme: str, lms: list[NGramLM]) -> float:
    """Mean log-prob of phoneme as a one-token sentence, across all LMs.

    Uses score_sequence([phoneme]) rather than log_prob((phoneme,)) so that
    the call works for any LM order: score_sequence pads with boundary tokens
    and evaluates all covering n-grams, returning a valid likelihood proxy.
    """
    total = 0.0
    n = 0
    for lm in lms:
        lp = lm.score_sequence([phoneme])
        if math.isfinite(lp):
            total += lp
            n += 1
    return total / n if n > 0 else -20.0


def _bigram_score(prev_phoneme: str, phoneme: str, lms: list[NGramLM]) -> float:
    """Mean log₂ score of the bigram (prev_phoneme, phoneme) across all LMs.

    Uses ``score_sequence([prev, phoneme])`` rather than
    ``log_prob((prev, phoneme))`` so the call works for any LM order — the
    rebuilt LMs are order 3–5, and ``log_prob`` requires the tuple length to
    equal the model order (a 2-tuple against an order-5 model raises
    ``ValueError: ngram length 2 != model order 5``).  This mirrors the
    order-agnostic convention already used by :func:`_unigram_score`:
    ``score_sequence`` pads with boundary tokens and sums all covering
    n-grams, giving a valid transition-likelihood proxy at any order.
    """
    total = 0.0
    n = 0
    for lm in lms:
        lp = lm.score_sequence([prev_phoneme, phoneme])
        if math.isfinite(lp):
            total += lp
            n += 1
    return total / n if n > 0 else -20.0


def _build_bigram_matrix(
    phonemes: list[str],
    lms: list[NGramLM],
    cache_path: Path | None = None,
    lm_dir: Path | None = None,
) -> np.ndarray:
    """Build or reload a (n_phonemes × n_phonemes) bigram log-prob matrix.

    ``matrix[p1_idx, p2_idx]`` = mean log P(p2 | p1) across all
    bigram-capable LMs, falling back to -20.0 for non-finite values.

    The result is cached to *cache_path* as a ``.npy`` file.  The cache
    is considered stale and recomputed when any ``*.json`` file in
    *lm_dir* has a modification time newer than the cache file.
    """
    n = len(phonemes)

    if cache_path is not None and cache_path.exists():
        cache_mtime = cache_path.stat().st_mtime
        stale = False
        if lm_dir is not None and lm_dir.is_dir():
            for lm_file in lm_dir.glob("*.json"):
                if lm_file.stat().st_mtime > cache_mtime:
                    log.info(
                        "LM file %s is newer than bigram cache — invalidating.",
                        lm_file.name,
                    )
                    stale = True
                    break
        if not stale:
            try:
                mat = np.load(cache_path)
                if mat.shape == (n, n):
                    log.info(
                        "Bigram score matrix loaded from cache (%s, %d×%d).",
                        cache_path, n, n,
                    )
                    return mat
                else:
                    log.warning(
                        "Cached bigram matrix shape %s != (%d, %d) — recomputing.",
                        mat.shape, n, n,
                    )
            except Exception as exc:
                log.warning("Bigram matrix cache load failed (%s) — recomputing.", exc)

    log.info(
        "Precomputing bigram score matrix (%d × %d = %d pairs) …",
        n, n, n * n,
    )
    t0 = time.perf_counter()
    mat = np.empty((n, n), dtype=np.float64)
    for p1_idx, p1 in enumerate(phonemes):
        for p2_idx, p2 in enumerate(phonemes):
            mat[p1_idx, p2_idx] = _bigram_score(p1, p2, lms)
    # Replace non-finite values with the sentinel used throughout.
    mat = np.where(np.isfinite(mat), mat, -20.0)
    elapsed = time.perf_counter() - t0
    log.info(
        "Bigram score matrix computed in %.1f s (%.0f pairs/s).",
        elapsed, n * n / max(elapsed, 1e-9),
    )

    if cache_path is not None:
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(cache_path, mat)
            log.info("Bigram score matrix cached → %s", cache_path)
        except Exception as exc:
            log.warning("Could not write bigram matrix cache: %s", exc)

    return mat


def _score_assignment(
    phone_map: dict[str, str],
    corpus_seqs: list[list[str]],
    lms: list[NGramLM],
    non_scoring_signs: set[str] | None = None,
) -> float:
    """Mean per-token log-prob of corpus under the assignment (for comparison)."""
    total_lp = 0.0
    total_n = 0
    for seq in corpus_seqs:
        translated = [
            phone_map.get(s, "<UNK>")
            for s in seq
            if non_scoring_signs is None or s not in non_scoring_signs
        ]
        if not translated:
            continue
        for lm in lms:
            if len(translated) >= lm.order:
                total_lp += lm.score_sequence(translated)
                total_n += len(translated)
    return total_lp / total_n if total_n > 0 else -math.inf


# ---------------------------------------------------------------------------
# QUBO construction
# ---------------------------------------------------------------------------

def _var(s_idx: int, p_idx: int, n_phonemes: int) -> int:
    """Flatten (sign_index, phoneme_index) → single QUBO variable index."""
    return s_idx * n_phonemes + p_idx


def build_qubo(
    signs: list[str],
    phonemes: list[str],
    lms: list[NGramLM],
    corpus_seqs: list[list[str]],
    lambda1: float = 10.0,
    lambda2: float = 5.0,
    max_per_phoneme: int = 5,
    lambda3: float = 1.0,
    bigram_weight: float = 1.0,
    max_bigram_pairs: int = 500,
    cribs: dict[str, str] | None = None,
    crib_penalty: float = 200.0,
    taxogram_signs: set[str] | None = None,
    bigram_cache_path: Path | None = None,
    lm_dir: Path | None = None,
) -> tuple[dict[tuple[int, int], float], dict]:
    """Build the QUBO matrix Q.

    Returns a tuple ``(Q, bigram_meta)``:
    - ``Q``: dict {(i, j): coefficient} in upper-triangular form (i ≤ j),
      compatible with dimod.BinaryQuadraticModel.
    - ``bigram_meta``: dict with keys ``matrix``, ``top_pairs``, ``max_adj``,
      ``lambda3_eff``, ``n_couplings``, ``score_min``, ``score_max`` (or empty
      dict when bigram couplings are disabled).

    Variable layout: var(s, p) = s * n_phonemes + p.

    QUBO construction
    -----------------
    Objective diagonal:
        Q[var(s,p), var(s,p)] += -unigram_score(p)  (maximise → minimise negative)

    One-hot penalty (one phoneme per sign):
        Q[var(s,p), var(s,p)] += -λ1               (linear, from expansion of (Σx-1)²)
        Q[var(s,p), var(s,q)] += 2*λ1  for p≠q     (quadratic, same sign s)

    Capacity penalty (≤ k signs per phoneme):
        Q[var(s,p), var(t,p)] += 2*λ2  for s≠t     (quadratic, same phoneme p)

    Bigram context couplings (corpus-adjacency weighted, optional):
        For the top-``max_bigram_pairs`` most frequent (s1, s2) adjacencies in
        the corpus, adds a cross-sign phoneme-pair coupling:
            Q[var(s1,p1), var(s2,p2)] += -λ3 * bigram_score(p1, p2) * adj(s1,s2)
        This encourages consecutive signs to be assigned phonemes that form
        high-probability bigrams in the Polynesian LMs, closing the gap between
        the unigram-only objective and the full sequence LM scored by MCMC.
        Disabled when λ3=0 or max_bigram_pairs=0.

    The constant terms in the penalty expansions do not affect the argmin
    so they are omitted.
    """
    n_signs    = len(signs)
    n_phonemes = len(phonemes)
    Q: dict[tuple[int, int], float] = {}
    sign_index = {s: i for i, s in enumerate(signs)}
    taxogram_idx = {
        sign_index[s] for s in (taxogram_signs or set()) if s in sign_index
    }

    def _add(i: int, j: int, val: float) -> None:
        if i > j:
            i, j = j, i
        key = (i, j)
        Q[key] = Q.get(key, 0.0) + val

    log.info("Building QUBO: %d signs × %d phonemes = %d variables …",
             n_signs, n_phonemes, n_signs * n_phonemes)

    # ── Objective: maximise LM unigram score ─────────────────────────────────
    for s_idx in range(n_signs):
        if s_idx in taxogram_idx:
            continue
        for p_idx, phoneme in enumerate(phonemes):
            score = _unigram_score(phoneme, lms)
            v = _var(s_idx, p_idx, n_phonemes)
            _add(v, v, -score)

    # ── One-hot penalty: each sign maps to exactly one phoneme ────────────────
    for s_idx in range(n_signs):
        for p_idx in range(n_phonemes):
            v = _var(s_idx, p_idx, n_phonemes)
            _add(v, v, -lambda1)                     # from (Σx - 1)² diagonal
        for p_idx in range(n_phonemes):
            for q_idx in range(p_idx + 1, n_phonemes):
                vi = _var(s_idx, p_idx, n_phonemes)
                vj = _var(s_idx, q_idx, n_phonemes)
                _add(vi, vj, 2.0 * lambda1)

    # ── Capacity penalty: at most k signs per phoneme ─────────────────────────
    for p_idx in range(n_phonemes):
        for s_idx in range(n_signs):
            if s_idx in taxogram_idx:
                continue
            for t_idx in range(s_idx + 1, n_signs):
                if t_idx in taxogram_idx:
                    continue
                vi = _var(s_idx, p_idx, n_phonemes)
                vj = _var(t_idx, p_idx, n_phonemes)
                _add(vi, vj, 2.0 * lambda2)

    # ── Bigram context couplings: corpus-adjacency weighted ───────────────────
    lambda3_eff = lambda3 * bigram_weight
    bigram_meta: dict = {}

    if lambda3_eff > 0.0 and max_bigram_pairs > 0:
        # Count how often each (s1, s2) ordered pair appears consecutively.
        adj_counts: dict[tuple[int, int], int] = {}
        for seq in corpus_seqs:
            for a, b in zip(seq, seq[1:]):
                si = sign_index.get(a, -1)
                sj = sign_index.get(b, -1)
                if (
                    si >= 0 and sj >= 0 and si != sj
                    and si not in taxogram_idx and sj not in taxogram_idx
                ):
                    key = (si, sj)
                    adj_counts[key] = adj_counts.get(key, 0) + 1

        # Keep only the most frequent pairs to bound coupling count.
        top_pairs = sorted(adj_counts.items(), key=lambda kv: -kv[1])[:max_bigram_pairs]
        if top_pairs:
            max_adj = float(top_pairs[0][1])
            # Precompute full bigram score matrix once as a numpy array,
            # then cache to disk.  Avoids n_pairs × n_phonemes² LM calls.
            bigram_matrix = _build_bigram_matrix(
                phonemes, lms,
                cache_path=bigram_cache_path,
                lm_dir=lm_dir,
            )

            coupling_scores: list[float] = []
            n_bigram_couplings = 0
            for (s1_idx, s2_idx), count in top_pairs:
                weight = count / max_adj
                for p1_idx in range(n_phonemes):
                    for p2_idx in range(n_phonemes):
                        bscore = float(bigram_matrix[p1_idx, p2_idx])
                        if math.isfinite(bscore) and bscore > -20.0:
                            vi = _var(s1_idx, p1_idx, n_phonemes)
                            vj = _var(s2_idx, p2_idx, n_phonemes)
                            _add(vi, vj, -lambda3_eff * bscore * weight)
                            coupling_scores.append(bscore)
                            n_bigram_couplings += 1

            score_min = float(np.min(coupling_scores)) if coupling_scores else 0.0
            score_max = float(np.max(coupling_scores)) if coupling_scores else 0.0
            log.info(
                "Bigram couplings: %d total (λ3_eff=%.3f, "
                "score range [%.3f, %.3f], top %d adj pairs).",
                n_bigram_couplings, lambda3_eff,
                score_min, score_max, len(top_pairs),
            )
            bigram_meta = {
                "matrix":       bigram_matrix,
                "top_pairs":    top_pairs,
                "max_adj":      max_adj,
                "lambda3_eff":  lambda3_eff,
                "n_couplings":  n_bigram_couplings,
                "score_min":    score_min,
                "score_max":    score_max,
            }

    # ── Crib constraints: known-plaintext fragments ──────────────────────────
    # For each (sign, phoneme) pair in cribs, force x_{s,p}=1 by adding a
    # large negative reward for the correct assignment and a large positive
    # penalty for every other phoneme for that sign.  The penalty dominates
    # the one-hot and capacity terms so the crib is always satisfied.
    if cribs:
        phoneme_index = {p: i for i, p in enumerate(phonemes)}
        for sign, phoneme in cribs.items():
            if sign not in sign_index:
                log.warning("Crib sign %r not in corpus inventory — skipped.", sign)
                continue
            if phoneme not in phoneme_index:
                log.warning("Crib phoneme %r not in inventory — skipped.", phoneme)
                continue
            s_idx = sign_index[sign]
            p_idx = phoneme_index[phoneme]
            # Reward the crib assignment strongly.
            v_correct = _var(s_idx, p_idx, n_phonemes)
            _add(v_correct, v_correct, -crib_penalty)
            # Penalise all other phoneme assignments for this sign.
            for q_idx in range(n_phonemes):
                if q_idx != p_idx:
                    v_wrong = _var(s_idx, q_idx, n_phonemes)
                    _add(v_wrong, v_wrong, +crib_penalty)
            log.info("Crib pinned: sign %r → phoneme %r (penalty=%.0f)", sign, phoneme, crib_penalty)

    n_couplings = sum(1 for (i, j) in Q if i != j)
    log.info("QUBO built: %d variables, %d couplings.", n_signs * n_phonemes, n_couplings)
    return Q, bigram_meta


def _qubo_to_bqm(Q: dict[tuple[int, int], float]) -> Any:
    """Convert upper-triangular Q dict to a dimod BinaryQuadraticModel."""
    import dimod  # deferred — only needed at solve time
    bqm = dimod.BinaryQuadraticModel(vartype="BINARY")
    for (i, j), val in Q.items():
        if i == j:
            bqm.add_variable(i, val)
        else:
            bqm.add_interaction(i, j, val)
    return bqm


# ---------------------------------------------------------------------------
# Warm-start from MCMC ranking
# ---------------------------------------------------------------------------

def _init_from_ranking(
    ranking_path: Path,
    signs: list[str],
    phonemes: list[str],
) -> dict[int, int] | None:
    """Return an initial_state dict {var_index: 0|1} from top MCMC hypothesis."""
    try:
        ranking = json.loads(ranking_path.read_text(encoding="utf-8"))
        hyps = ranking.get("hypotheses", [])
        if not hyps:
            log.warning("ranking.json has no hypotheses — ignoring --init-from.")
            return None
        top = hyps[0]
        assignments = top.get("assignments", top.get("phoneme_assignments", []))
        sign_to_phoneme: dict[str, str] = {
            a["sign_code"]: a["phoneme"] for a in assignments
        }
        phoneme_set = set(phonemes)
        n_phonemes  = len(phonemes)
        p_index     = {p: i for i, p in enumerate(phonemes)}
        state: dict[int, int] = {}
        for s_idx, sign in enumerate(signs):
            ph = sign_to_phoneme.get(sign)
            for p_idx in range(n_phonemes):
                v = _var(s_idx, p_idx, n_phonemes)
                state[v] = 1 if (ph is not None and phonemes[p_idx] == ph and ph in phoneme_set) else 0
        log.info("Warm-start from %s (top hypothesis: %s).",
                 ranking_path.name, top.get("hypothesis_id", "?"))
        return state
    except Exception as exc:
        log.warning("Could not load warm-start from %s: %s", ranking_path, exc)
        return None


# ---------------------------------------------------------------------------
# Solver dispatch
# ---------------------------------------------------------------------------

def _solve_dwave(
    bqm: Any,
    num_reads: int,
    token: str,
    initial_state: dict[int, int] | None,
) -> tuple[Any, float]:
    """Submit to D-Wave Leap QPU via cloud API (direct embedding).

    Works reliably for problems with ≤ ~500 logical variables on Advantage.
    For larger problems (full corpus) use _solve_hybrid instead.
    """
    from dwave.system import DWaveSampler, EmbeddingComposite  # type: ignore
    log.info("Connecting to D-Wave Leap QPU …")
    sampler = EmbeddingComposite(DWaveSampler(token=token))
    kwargs: dict[str, Any] = {"num_reads": num_reads, "label": "hackingrongo-qubo"}
    if initial_state is not None:
        kwargs["initial_state"] = initial_state
    t0 = time.perf_counter()
    response = sampler.sample(bqm, **kwargs)
    elapsed = time.perf_counter() - t0
    log.info("D-Wave QPU sampling complete in %.1f s.", elapsed)
    return response, elapsed


def _solve_hybrid(
    bqm: Any,
    token: str,
    time_limit: int = 30,
) -> tuple[Any, float]:
    """Submit to D-Wave LeapHybridSampler (quantum+classical hybrid).

    Handles arbitrarily large BQMs without minor-embedding constraints.
    Recommended for the full rongorongo corpus (~4 k+ variables).
    Minimum time_limit enforced by the SDK is 3 s.

    Parameters
    ----------
    bqm :
        dimod BinaryQuadraticModel to minimise.
    token :
        D-Wave Leap API token.
    time_limit :
        Wall-clock budget in seconds allocated on the hybrid solver.
        Larger values explore more of the landscape (default: 30 s).
    """
    from dwave.system import LeapHybridSampler  # type: ignore
    log.info("Connecting to D-Wave LeapHybridSampler (time_limit=%d s) …", time_limit)
    sampler = LeapHybridSampler(token=token)
    t0 = time.perf_counter()
    response = sampler.sample(bqm, time_limit=time_limit, label="hackingrongo-hybrid")
    elapsed = time.perf_counter() - t0
    log.info("LeapHybridSampler complete in %.1f s.", elapsed)
    return response, elapsed


def _numpy_sa(bqm: Any, num_reads: int, initial_state: dict[int, int] | None) -> Any:
    """Pure numpy simulated annealing — no D-Wave packages required.

    Metropolis-Hastings over binary variables. Returns a dimod.SampleSet
    so the rest of the pipeline is unaffected.
    """
    import dimod
    import numpy as np

    variables = list(bqm.variables)
    n = len(variables)
    idx = {v: i for i, v in enumerate(variables)}

    linear = np.array([bqm.linear.get(v, 0.0) for v in variables], dtype=np.float64)
    # Symmetric Q with Q[i,i]=0; each quadratic bias split evenly.
    Q = np.zeros((n, n), dtype=np.float64)
    for (u, v), bias in bqm.quadratic.items():
        i, j = idx[u], idx[v]
        Q[i, j] += bias / 2.0
        Q[j, i] += bias / 2.0

    n_steps = max(2000, n * 4)
    rng = np.random.default_rng(42)
    samples: list[dict] = []
    energies: list[float] = []

    seed_x: np.ndarray | None = None
    if initial_state is not None:
        seed_x = np.array([float(initial_state.get(v, 0)) for v in variables])

    for _ in range(num_reads):
        x = seed_x.copy() if seed_x is not None else rng.integers(0, 2, size=n).astype(np.float64)
        T = 2.0
        decay = (1e-4 / T) ** (1.0 / n_steps)
        for _ in range(n_steps):
            T *= decay
            k = int(rng.integers(n))
            flip = 1.0 - 2.0 * x[k]
            # delta E for flipping x[k]: (1-2x_k)*(linear[k] + 2*(Q[k]@x))
            delta = flip * (linear[k] + 2.0 * float(Q[k] @ x))
            if delta <= 0.0 or rng.random() < np.exp(-delta / max(T, 1e-10)):
                x[k] += flip
        energy = float(linear @ x + x @ Q @ x + bqm.offset)
        samples.append({v: int(x[idx[v]]) for v in variables})
        energies.append(energy)

    return dimod.SampleSet.from_samples(samples, vartype=dimod.BINARY, energy=energies)


def _solve_neal(
    bqm: Any,
    num_reads: int,
    initial_state: dict[int, int] | None,
) -> tuple[Any, float]:
    """Simulated annealing — tries D-Wave packages, falls back to numpy SA."""
    t0 = time.perf_counter()
    try:
        try:
            from dwave.samplers import SimulatedAnnealingSampler
        except ImportError:
            import neal  # type: ignore
            SimulatedAnnealingSampler = neal.SimulatedAnnealingSampler
        log.info("Running D-Wave SimulatedAnnealingSampler (%d reads) …", num_reads)
        sampler = SimulatedAnnealingSampler()
        kwargs: dict[str, Any] = {"num_reads": num_reads}
        if initial_state is not None:
            kwargs["initial_states"] = {k: v for k, v in initial_state.items()}
        response = sampler.sample(bqm, **kwargs)
    except ImportError:
        log.info("neal/dwave-samplers not available — using numpy SA (%d reads) …", num_reads)
        response = _numpy_sa(bqm, num_reads, initial_state)
    elapsed = time.perf_counter() - t0
    log.info("Simulated annealing complete in %.1f s.", elapsed)
    return response, elapsed


def _solve_tabu(
    bqm: Any,
    num_reads: int,
    initial_state: dict[int, int] | None,
) -> tuple[Any, float]:
    """Tabu search via dimod/tabu package, with safe SA fallback."""
    sampler: Any | None = None
    sampler_name = ""

    try:
        import dimod  # type: ignore
        if hasattr(dimod, "TabuSampler"):
            sampler = dimod.TabuSampler()
            sampler_name = "dimod.TabuSampler"
    except ImportError:
        sampler = None

    if sampler is None:
        try:
            import tabu  # type: ignore
            sampler = tabu.TabuSampler()
            sampler_name = "tabu.TabuSampler"
        except ImportError:
            log.warning(
                "TabuSampler unavailable (dimod has no TabuSampler and tabu package is missing) — "
                "falling back to simulated annealing."
            )
            return _solve_neal(bqm, num_reads, initial_state)

    log.info("Running %s (%d reads) …", sampler_name, num_reads)
    kwargs: dict[str, Any] = {"num_reads": num_reads}
    if initial_state is not None:
        kwargs["initial_states"] = [initial_state]
    t0 = time.perf_counter()
    response = sampler.sample(bqm, **kwargs)
    elapsed = time.perf_counter() - t0
    log.info("TabuSampler complete in %.1f s.", elapsed)
    return response, elapsed


def _pick_solver(
    requested: str,
    dwave_token: str | None,
    n_variables: int = 0,
) -> str:
    """Resolve 'auto' to an available solver name.

    Auto-selection priority when a Leap token is present:
      - ``hybrid``  — for n_variables > 500 (LeapHybridSampler, no embedding limits)
      - ``dwave``   — for n_variables ≤ 500 (direct QPU, fast)
    Without a token: ``neal`` → ``tabu`` → error.
    """
    if requested != "auto":
        return requested

    if dwave_token:
        try:
            import dwave.system  # noqa: F401
            if n_variables > 500:
                log.info(
                    "Problem has %d variables — selecting LeapHybridSampler "
                    "(direct QPU embedding not reliable at this scale).",
                    n_variables,
                )
                return "hybrid"
            return "dwave"
        except ImportError:
            log.warning("dwave-ocean-sdk not installed — falling back to neal/tabu.")

    try:
        from dwave.samplers import SimulatedAnnealingSampler  # noqa: F401
        return "neal"
    except ImportError:
        pass

    try:
        import neal  # noqa: F401
        return "neal"
    except ImportError:
        pass

    try:
        import dimod  # noqa: F401
        return "tabu"
    except ImportError:
        pass

    log.error(
        "No QUBO solver available. Install one of:\n"
        "  pip install dwave-samplers dimod\n"
        "  pip install dwave-ocean-sdk  # for QPU / hybrid"
    )
    sys.exit(1)


def solve(
    bqm: Any,
    solver: str,
    num_reads: int,
    dwave_token: str | None,
    initial_state: dict[int, int] | None,
    hybrid_time_limit: int = 30,
) -> tuple[Any, float, str]:
    """Dispatch to the chosen solver. Returns (sampleset, elapsed_s, solver_used)."""
    if solver == "hybrid":
        if not dwave_token:
            log.error("hybrid solver requires --dwave-token or DWAVE_API_TOKEN env var.")
            sys.exit(1)
        response, elapsed = _solve_hybrid(bqm, dwave_token, time_limit=hybrid_time_limit)
        return response, elapsed, "hybrid"
    elif solver == "dwave":
        if not dwave_token:
            log.error("D-Wave solver requires --dwave-token or DWAVE_API_TOKEN env var.")
            sys.exit(1)
        response, elapsed = _solve_dwave(bqm, num_reads, dwave_token, initial_state)
        return response, elapsed, "dwave"
    elif solver == "neal":
        response, elapsed = _solve_neal(bqm, num_reads, initial_state)
        return response, elapsed, "neal"
    elif solver == "tabu":
        response, elapsed = _solve_tabu(bqm, num_reads, initial_state)
        return response, elapsed, "tabu"
    else:
        log.error("Unknown solver: %s", solver)
        sys.exit(1)


# ---------------------------------------------------------------------------
# Extract assignment from sample
# ---------------------------------------------------------------------------

def _extract_assignment(
    sample: dict[int, int],
    signs: list[str],
    phonemes: list[str],
) -> dict[str, str]:
    """Convert a binary sample {var_index: 0|1} to a sign→phoneme map.

    For each sign, picks the phoneme with x_{s,p}=1.  If a sign has no
    active variable (all zeros), assigns the highest-probability phoneme
    (index 0).
    """
    n_phonemes = len(phonemes)
    phone_map: dict[str, str] = {}
    for s_idx, sign in enumerate(signs):
        selected: str | None = None
        for p_idx, phoneme in enumerate(phonemes):
            v = _var(s_idx, p_idx, n_phonemes)
            if sample.get(v, 0) == 1:
                selected = phoneme
                break
        phone_map[sign] = selected if selected is not None else phonemes[0]
    return phone_map


def _assignment_confidence(
    sample: dict[int, int],
    signs: list[str],
    phonemes: list[str],
    sampleset: Any,
) -> dict[str, float]:
    """Estimate per-sign confidence as fraction of reads agreeing with best sample."""
    n_phonemes = len(phonemes)
    best_phone_map = _extract_assignment(sample, signs, phonemes)
    agree_counts: dict[str, int] = {s: 0 for s in signs}
    total_reads = 0

    try:
        records = sampleset.record
        for record in records:
            total_reads += int(record.num_occurrences)
            samp = dict(zip(range(len(record.sample)), record.sample))
            for s_idx, sign in enumerate(signs):
                ph = best_phone_map[sign]
                p_idx = phonemes.index(ph)
                v = _var(s_idx, p_idx, n_phonemes)
                if samp.get(v, 0) == 1:
                    agree_counts[sign] += int(record.num_occurrences)
    except Exception:
        return {s: 1.0 for s in signs}

    if total_reads == 0:
        return {s: 1.0 for s in signs}
    return {s: agree_counts[s] / total_reads for s in signs}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="QUBO formulation of rongorongo sign→phoneme assignment.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--corpus-dir",   type=Path,  default=None, metavar="DIR")
    p.add_argument("--lm-dir",       type=Path,  default=None, metavar="DIR")
    p.add_argument("--init-from",    type=Path,  default=None, metavar="JSON",
                   help="Path to ranking.json for MCMC warm-start.")
    p.add_argument("--solver",       default="auto",
                   choices=["auto", "hybrid", "dwave", "neal", "tabu"],
                   help="QUBO solver: hybrid=LeapHybridSampler (recommended for "
                        "full corpus), dwave=QPU direct (≤500 vars), "
                        "neal=CPU SA, tabu=CPU tabu (default: auto-detect).")
    p.add_argument("--hybrid-time-limit", type=int, default=30, metavar="SECS",
                   help="Wall-clock seconds allocated to LeapHybridSampler "
                        "(default: 30; min 3 enforced by SDK).")
    p.add_argument("--crib", default=None, metavar="SIGN=PHONEME[,…]",
                   help="Known-plaintext crib: comma-separated SIGN=PHONEME pairs "
                        "(e.g. '200=tangata,076=ko').  These assignments are "
                        "pinned with a hard QUBO penalty so they are always "
                        "satisfied.  Sign IDs are Barthel codes.")
    p.add_argument(
        "--disable-taxogram-cribs",
        action="store_true",
        help=(
            "Disable default logographic taxogram constraints "
            "(600=manu, 700=ika, 280=honu, 690=tangata manu)."
        ),
    )
    p.add_argument("--crib-penalty", type=float, default=200.0, metavar="F",
                   help="Penalty weight for crib constraints (default: 200.0; "
                        "should dominate lambda1 and lambda2).")
    p.add_argument("--num-reads",    type=int,   default=1000, metavar="N",
                   help="Number of annealing reads/shots (default: 1000).")
    p.add_argument("--output",       type=Path,  default=None, metavar="JSON")
    p.add_argument("--dwave-token",  default=None, metavar="TOKEN",
                   help="D-Wave Leap API token (default: DWAVE_API_TOKEN env var).")
    p.add_argument("--lambda1",      type=float, default=10.0, metavar="F",
                   help="One-hot penalty weight (default: 10.0).")
    p.add_argument("--lambda2",      type=float, default=5.0, metavar="F",
                   help="Capacity penalty weight (default: 5.0).")
    p.add_argument("--bigram-weight", type=float, default=1.0, metavar="F",
                   help="Scale factor for lambda3 bigram couplings relative to "
                        "lambda1/lambda2 (default: 1.0; set 0 to disable bigrams).")
    p.add_argument("--max-per-phoneme", type=int, default=5, metavar="K",
                   help="Max signs per phoneme (default: 5).")
    p.add_argument("--max-signs", type=int, default=60, metavar="N",
                   help="Cap the QUBO to the top-N most frequent signs (default: 60). "
                        "The full corpus (~2000 signs × 50 phonemes ≈ 100k binary "
                        "variables) is intractable for neal or any QUBO solver; this "
                        "restricts to the decipherment-relevant core. Crib/taxogram "
                        "signs are always retained. Set 0 to disable the cap.")
    p.add_argument("--smoke-test",   action="store_true",
                   help="Use 10 signs × 10 phonemes, 50 reads (fast wiring check).")
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    dwave_token = args.dwave_token or os.environ.get("DWAVE_API_TOKEN")

    corpus_dir = args.corpus_dir
    lm_dir     = args.lm_dir
    output     = args.output

    if corpus_dir is None or lm_dir is None or output is None:
        try:
            from omegaconf import OmegaConf
            cfg = OmegaConf.load(PROJECT_ROOT / "conf" / "config.yaml")
            if corpus_dir is None:
                corpus_dir = PROJECT_ROOT / cfg.paths.corpus_dir
            if lm_dir is None:
                lm_dir = PROJECT_ROOT / "data" / "language_models"
            if output is None:
                output = (
                    PROJECT_ROOT / cfg.paths.outputs_dir
                    / "decipherment" / "qubo_result.json"
                )
        except Exception:
            pass

    if corpus_dir is None or not corpus_dir.exists():
        log.error("Corpus directory not found. Pass --corpus-dir.")
        sys.exit(1)
    if lm_dir is None or not lm_dir.exists():
        log.error("LM directory not found. Pass --lm-dir.")
        sys.exit(1)

    # ── Load ──────────────────────────────────────────────────────────────────
    log.info("Loading corpus from %s …", corpus_dir)
    corpus_seqs = _load_corpus(corpus_dir)
    if not corpus_seqs:
        log.error("No corpus sequences found.")
        sys.exit(1)

    lms = _load_lms(lm_dir)
    all_signs    = sorted({code for seq in corpus_seqs for code in seq})
    all_phonemes = _phoneme_inventory(lms)

    if not all_phonemes:
        log.error("Phoneme inventory is empty — LM vocab absent or malformed.")
        sys.exit(1)

    # ── Parse cribs ───────────────────────────────────────────────────────────
    cribs: dict[str, str] = {}
    if not args.disable_taxogram_cribs:
        cribs.update(LOGOGRAPHIC_TAXOGRAMS)
    if args.crib:
        for pair in args.crib.split(","):
            pair = pair.strip()
            if "=" not in pair:
                log.warning("Crib entry %r has no '=' — skipped.", pair)
                continue
            sign, phoneme = pair.split("=", 1)
            cribs[sign.strip()] = phoneme.strip()
        if cribs:
            log.info("Cribs loaded: %s", cribs)

    if args.smoke_test:
        log.info("Smoke-test mode: truncating to 10 signs × 10 phonemes, 50 reads.")
        all_signs    = all_signs[:10]
        all_phonemes = all_phonemes[:10]
        args.num_reads = 50
    elif args.max_signs and len(all_signs) > args.max_signs:
        # Bound the QUBO to a tractable subproblem: the top-N most frequent signs
        # (the decipherment-relevant core), always retaining crib/taxogram signs.
        # The full ~2000 signs × 50 phonemes ≈ 100k binary variables is far beyond
        # what neal or any QUBO solver handles.  Mirrors the QAOA subproblem.
        from collections import Counter
        _freq = Counter(code for seq in corpus_seqs for code in seq)
        _ranked = sorted(all_signs, key=lambda s: _freq.get(s, 0), reverse=True)
        _keep = list(_ranked[: args.max_signs])
        _kept = set(_keep)
        _extra_cribs = [c for c in cribs if c in _freq and c not in _kept]
        _keep.extend(_extra_cribs)
        log.info(
            "Restricting QUBO to top %d signs by frequency%s — from %d total.",
            args.max_signs,
            f" (+{len(_extra_cribs)} crib)" if _extra_cribs else "",
            len(all_signs),
        )
        all_signs = sorted(_keep)

    n_signs    = len(all_signs)
    n_phonemes = len(all_phonemes)
    log.info("Sign inventory  : %d signs",    n_signs)
    log.info("Phoneme inventory: %d phonemes", n_phonemes)
    log.info("QUBO variables  : %d",           n_signs * n_phonemes)

    # ── Warm-start ────────────────────────────────────────────────────────────
    initial_state: dict[int, int] | None = None
    mcmc_best_lm_score: float | None = None
    mcmc_phone_map: dict[str, str] = {}

    if args.init_from and args.init_from.exists():
        initial_state = _init_from_ranking(args.init_from, all_signs, all_phonemes)
        try:
            ranking = json.loads(args.init_from.read_text(encoding="utf-8"))
            hyps = ranking.get("hypotheses", [])
            if hyps:
                mcmc_best_lm_score = hyps[0].get("overall_lm_score")
                mcmc_phone_map = {a["sign_code"]: a["phoneme"]
                                  for a in hyps[0].get("assignments", [])}
        except Exception:
            pass
    elif args.init_from:
        log.warning("--init-from path not found: %s", args.init_from)

    # ── Build QUBO ────────────────────────────────────────────────────────────
    bigram_cache_path = PROJECT_ROOT / "outputs" / "decipherment" / "bigram_score_matrix.npy"
    Q, bigram_meta = build_qubo(
        all_signs, all_phonemes, lms, corpus_seqs,
        lambda1=args.lambda1,
        lambda2=args.lambda2,
        bigram_weight=args.bigram_weight,
        max_per_phoneme=args.max_per_phoneme,
        cribs=cribs or None,
        crib_penalty=args.crib_penalty,
        taxogram_signs=(set(LOGOGRAPHIC_TAXOGRAMS) if not args.disable_taxogram_cribs else set()),
        bigram_cache_path=bigram_cache_path,
        lm_dir=lm_dir,
    )
    bqm = _qubo_to_bqm(Q)
    n_couplings = sum(1 for (i, j) in Q if i != j)

    # ── Solve ─────────────────────────────────────────────────────────────────
    solver_name = _pick_solver(args.solver, dwave_token, n_variables=n_signs * n_phonemes)
    log.info("Using solver: %s", solver_name)

    sampleset, elapsed, solver_used = solve(
        bqm, solver_name, args.num_reads, dwave_token, initial_state,
        hybrid_time_limit=args.hybrid_time_limit,
    )

    # ── Extract best sample ───────────────────────────────────────────────────
    best_sample = sampleset.first.sample
    best_energy = float(sampleset.first.energy)

    phone_map  = _extract_assignment(best_sample, all_signs, all_phonemes)
    confidence = _assignment_confidence(best_sample, all_signs, all_phonemes, sampleset)

    # ── Score with LMs ────────────────────────────────────────────────────────
    taxogram_signs = set(LOGOGRAPHIC_TAXOGRAMS) if not args.disable_taxogram_cribs else set()
    best_lm_score = _score_assignment(
        phone_map,
        corpus_seqs,
        lms,
        non_scoring_signs=taxogram_signs,
    )
    # Apples-to-apples baseline: score the MCMC map with the SAME per-token
    # scorer.  mcmc_best_lm_score (ranking.json overall_lm_score) is a
    # corpus-level TOTAL on a different scale, so best_lm_score − mcmc_best
    # produced a meaningless Δ (e.g. +3116).  Only this baseline is comparable.
    mcmc_baseline_lm_score: float | None = None
    if mcmc_phone_map:
        mcmc_baseline_lm_score = _score_assignment(
            mcmc_phone_map, corpus_seqs, lms, non_scoring_signs=taxogram_signs,
        )

    # ── Benchmark: bigram coupling contribution ───────────────────────────────
    bigram_contribution = 0.0
    if bigram_meta:
        bm_mat     = bigram_meta.get("matrix")
        bm_pairs   = bigram_meta.get("top_pairs", [])
        bm_max_adj = bigram_meta.get("max_adj", 1.0)
        bm_l3      = bigram_meta.get("lambda3_eff", 0.0)
        if bm_mat is not None and bm_pairs and bm_l3 > 0:
            ph_index = {p: i for i, p in enumerate(all_phonemes)}
            s_index  = {s: i for i, s in enumerate(all_signs)}
            for (s1_idx, s2_idx), count in bm_pairs:
                weight = count / bm_max_adj
                p1 = phone_map.get(all_signs[s1_idx])
                p2 = phone_map.get(all_signs[s2_idx])
                if p1 and p2:
                    i1 = ph_index.get(p1, -1)
                    i2 = ph_index.get(p2, -1)
                    if i1 >= 0 and i2 >= 0:
                        bs = float(bm_mat[i1, i2])
                        if math.isfinite(bs) and bs > -20.0:
                            bigram_contribution += bm_l3 * bs * weight
    log.info(
        "Bigram coupling added %.4f nats to objective vs unigram-only baseline.",
        bigram_contribution,
    )

    # ── Print results ─────────────────────────────────────────────────────────
    print(f"\n{'═' * 64}")
    print(f"  QUBO Decipherment Result — Rongorongo Layer 4Q")
    print(f"  Solver : {solver_used}   Reads : {args.num_reads}")
    print(f"  Signs  : {n_signs}   Phonemes : {n_phonemes}   "
          f"Variables : {n_signs * n_phonemes}")
    print(f"{'═' * 64}")
    print(f"\n  Best energy     : {best_energy:.4f}")
    print(f"  LM score (QUBO) : {best_lm_score:.4f} (mean per-token log₂p)")
    if mcmc_baseline_lm_score is not None:
        improvement = best_lm_score - mcmc_baseline_lm_score
        symbol = "▲" if improvement > 0 else "▼"
        print(f"  MCMC baseline (same scorer) : {mcmc_baseline_lm_score:.4f} (mean per-token log₂p)")
        print(f"  Δ vs MCMC baseline          : {improvement:+.4f} {symbol}")
    if mcmc_best_lm_score is not None:
        print(f"  MCMC ranking total (diff. scale) : {mcmc_best_lm_score:.4f}  (corpus total — NOT comparable to per-token scores)")
    print(f"  Annealing time  : {elapsed:.1f} s")
    print()
    print(f"  Top 10 sign→phoneme assignments (by confidence):")
    sorted_signs = sorted(phone_map, key=lambda s: -confidence.get(s, 0))
    for sign in sorted_signs[:10]:
        ph  = phone_map[sign]
        con = confidence.get(sign, 0.0)
        bar = "█" * round(con * 10) + "░" * (10 - round(con * 10))
        print(f"    {sign:>8} → {ph:<6}  conf={con:.3f}  {bar}")
    print()

    # ── Build output dict ─────────────────────────────────────────────────────
    phoneme_assignments = [
        {
            "sign_code":  sign,
            "phoneme":    phone_map[sign],
            "confidence": round(confidence.get(sign, 0.0), 6),
        }
        for sign in sorted(all_signs)
    ]

    result: dict[str, Any] = {
        "solver":                solver_used,
        "n_reads":               args.num_reads,
        "best_energy":           round(best_energy, 6),
        "best_lm_score":         round(best_lm_score, 6) if math.isfinite(best_lm_score) else None,
        # improvement_over_mcmc compares best_lm_score vs mcmc_baseline_lm_score —
        # BOTH per-token means from _score_assignment, so the Δ is meaningful.
        "mcmc_baseline_lm_score": round(mcmc_baseline_lm_score, 6) if mcmc_baseline_lm_score is not None else None,
        "improvement_over_mcmc": (
            round(best_lm_score - mcmc_baseline_lm_score, 6)
            if mcmc_baseline_lm_score is not None and math.isfinite(best_lm_score)
            else None
        ),
        # Corpus-level TOTAL from ranking.json — different scale, NOT comparable;
        # kept for provenance only, not used in improvement_over_mcmc.
        "mcmc_ranking_total_lm_score": round(mcmc_best_lm_score, 6) if mcmc_best_lm_score is not None else None,
        "cribs":                 cribs if cribs else {},
        "taxogram_signs_excluded_from_lm": sorted(taxogram_signs),
        "crib_penalty":          args.crib_penalty if cribs else None,
        "phoneme_assignments":   phoneme_assignments,
        "annealing_time_seconds": round(elapsed, 2),
        "qubo_size": {
            "variables": n_signs * n_phonemes,
            "couplings": n_couplings,
        },
        "qubo_parameters": {
            "lambda1":          args.lambda1,
            "lambda2":          args.lambda2,
            "bigram_weight":    args.bigram_weight,
            "lambda3_eff":      bigram_meta.get("lambda3_eff", 0.0) if bigram_meta else 0.0,
            "max_per_phoneme":  args.max_per_phoneme,
            "bigram_couplings": bigram_meta.get("n_couplings", 0) if bigram_meta else 0,
            "bigram_score_range": (
                [bigram_meta["score_min"], bigram_meta["score_max"]]
                if bigram_meta else None
            ),
            "bigram_contribution_nats": round(bigram_contribution, 6),
        },
    }

    # ── Save ──────────────────────────────────────────────────────────────────
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        log.info("Result written to %s", output)


if __name__ == "__main__":
    main()
