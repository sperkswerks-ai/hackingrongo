"""
hackingrongo.zone_c.mcmc
=========================

Metropolis-Hastings sampler for rongorongo phoneme assignment.

State
-----
A **phoneme map** π : {sign_id → phoneme} is a complete bijective-ish
assignment of every distinct sign in the corpus to a phoneme/syllable
token.  Because the sign inventory (~120 signs) is larger than the
Polynesian phoneme inventory (~45 syllables), the map is many-to-one
(multiple signs may map to the same phoneme).  This deliberately mimics
the syllabic redundancy observed in Linear B.

The sampler explores this combinatorial space by proposing random swaps
or reassignments, accepting or rejecting using a Metropolis-Hastings
criterion with the ensemble LM log-probability as the (unnormalised)
log-posterior.

Multi-chain convergence
-----------------------
``cfg.zone_c.mcmc.num_chains`` independent chains run with different random
seeds.  Convergence is assessed with the Gelman-Rubin R-hat statistic
(requires ``num_chains ≥ 2``).  After burn-in is discarded and thinning
applied, all chains are merged and the top-K assignments by log-posterior
are returned.

Adaptive proposal
-----------------
The proposal width (probability of a random-reassignment move vs a swap
move) is adapted every ``adaptation_interval`` steps to target an
acceptance rate near ``target_acceptance_rate`` (default 0.234, the
asymptotically optimal rate for high-dimensional MH).

Public API
----------
``MCMCSampler``
    ``run() -> MCMCResult``  —  main entry point.
``MCMCResult``
    Dataclass holding top-K samples, chain diagnostics, and convergence
    metadata.
"""

from __future__ import annotations

import logging
import math
import os
import random
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Sequence

import numpy as np
from omegaconf import DictConfig

from hackingrongo.zone_c.lm_scoring import LMScorer, LMScoringResult, PhonemeMap

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from hackingrongo.zone_c.mcmc import MCMCSampler  # for type hint only


# ---------------------------------------------------------------------------
# Module-level worker for parallel chain execution
# ---------------------------------------------------------------------------


def _run_chain_worker(
    sampler: "MCMCSampler",
    chain_id: int,
    seed: "int | None",
) -> "tuple[list, float]":
    """Top-level wrapper so ProcessPoolExecutor can pickle the target callable.

    Must live at module scope (not inside a class or closure) to satisfy the
    pickle protocol used by the spawn/fork start methods.  Logging in worker
    processes may be suppressed depending on the multiprocessing start method;
    the parent process logs aggregated results after ``f.result()`` returns.
    """
    return sampler._run_chain(chain_id, seed)


# ---------------------------------------------------------------------------
# Polynesian syllable inventory (Rapa Nui phonotactics)
# ---------------------------------------------------------------------------

_RAPA_NUI_CONSONANTS: tuple[str, ...] = ("h", "k", "m", "n", "ng", "p", "r", "t")
_VOWELS: tuple[str, ...] = ("a", "e", "i", "o", "u")

_DEFAULT_PHONEME_INVENTORY: list[str] = [v for v in _VOWELS] + [
    c + v for c in _RAPA_NUI_CONSONANTS for v in _VOWELS
]  # 5 bare vowels + 40 CV syllables = 45 tokens


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class MCMCSample:
    """A single accepted sample from the MCMC chain.

    Attributes
    ----------
    phoneme_map : PhonemeMap
        Complete sign → phoneme assignment at this state.
    log_posterior : float
        Ensemble LM log₂-probability for this assignment.
    iteration : int
        Iteration index within the chain (after thinning).
    chain_id : int
        Index of the chain that produced this sample.
    """

    phoneme_map: PhonemeMap
    log_posterior: float
    iteration: int
    chain_id: int = 0


@dataclass
class MCMCResult:
    """Aggregated results from multi-chain MCMC.

    Attributes
    ----------
    top_samples : list[MCMCSample]
        Top-K unique samples ranked by ``log_posterior`` (descending).
    acceptance_rates : list[float]
        Per-chain acceptance rate after burn-in.
    gelman_rubin_rhat : float | None
        R-hat statistic computed across chains.  ``None`` if fewer than
        2 chains were run.
    converged : bool
        ``True`` iff the chain has converged by the appropriate diagnostic:
        R-hat < ``cfg.zone_c.mcmc.gelman_rubin_threshold`` for multi-chain
        runs; ``abs(geweke_z) < 2.0`` for single-chain runs.
    geweke_z : float | None
        Geweke Z-score for single-chain stationarity.  Compares the mean of
        the first 10 % of post-burn-in samples to the last 50 %.  ``None``
        for multi-chain runs (R-hat used instead).  ``|z| < 2.0`` ≈ p < 0.05.
    n_chains : int
        Number of chains.
    n_samples_per_chain : int
        Effective sample count per chain (after burn-in and thinning).
    """

    top_samples: list[MCMCSample]
    acceptance_rates: list[float] = field(default_factory=list)
    gelman_rubin_rhat: float | None = None
    converged: bool = False
    geweke_z: float | None = None
    n_chains: int = 1
    n_samples_per_chain: int = 0


# ---------------------------------------------------------------------------
# MCMCSampler
# ---------------------------------------------------------------------------


class MCMCSampler:
    """Metropolis-Hastings sampler over phoneme assignment maps.

    Parameters
    ----------
    cfg : DictConfig
        Root Hydra config.  Reads ``cfg.zone_c.mcmc``.
    lm_scorer : LMScorer
        Fully loaded :class:`~hackingrongo.zone_c.lm_scoring.LMScorer`.
    corpus_sequences : list[list[str]]
        List of glyph-token sequences (one per stratum or tablet).  The
        sampler scores the assignment against all sequences jointly.
    sign_ids : list[str]
        Exhaustive list of distinct sign IDs that appear in
        ``corpus_sequences``.  Every sign must appear in the initial
        map produced by :meth:`_random_initial_map`.
    phoneme_inventory : list[str] | None
        Candidate phonemes to assign.  If ``None``, the default 45-token
        Rapa Nui inventory is used.
    phoneme_priors : list[float] | None
        Sampling weights over ``phoneme_inventory`` used in the
        random-reassignment proposal and initial map construction.
        Must have the same length as the resolved inventory.  If ``None``,
        uniform weights are used (equivalent to ``rng.choice``).
    seed : int | None
        Global random seed for reproducibility.  Individual chain seeds
        are derived as ``seed + chain_id``.
    """

    def __init__(
        self,
        cfg: DictConfig,
        lm_scorer: LMScorer,
        corpus_sequences: list[list[str]],
        sign_ids: list[str],
        phoneme_inventory: list[str] | None = None,
        phoneme_priors: list[float] | None = None,
        seed: int | None = None,
        cribs: dict[str, str] | None = None,
        sign_ic_weights: dict[str, float] | None = None,
    ) -> None:
        self._scorer = lm_scorer
        self._corpus_sequences = corpus_sequences
        self._sign_ids = list(sign_ids)
        self._phoneme_inventory = (
            list(phoneme_inventory)
            if phoneme_inventory is not None
            else list(_DEFAULT_PHONEME_INVENTORY)
        )
        if phoneme_priors is not None:
            if len(phoneme_priors) != len(self._phoneme_inventory):
                raise ValueError(
                    f"phoneme_priors length ({len(phoneme_priors)}) must match "
                    f"phoneme_inventory length ({len(self._phoneme_inventory)})"
                )
            if any(p < 0.0 for p in phoneme_priors):
                raise ValueError(
                    "phoneme_priors must be non-negative (random.choices requirement)"
                )
            self._phoneme_priors: list[float] = list(phoneme_priors)
        else:
            self._phoneme_priors = [1.0] * len(self._phoneme_inventory)
        self._seed = seed

        # Crib (known-plaintext) assignments: pinned sign → phoneme pairs.
        # Crib signs are never modified by proposals and always initialised
        # to their crib phoneme.
        self._cribs: dict[str, str] = dict(cribs) if cribs else {}
        self._crib_signs: frozenset[str] = frozenset(self._cribs)
        # Free signs are those not pinned by a crib — only these are proposed.
        self._free_sign_ids: list[str] = [
            s for s in self._sign_ids if s not in self._crib_signs
        ]
        if self._cribs:
            logger.info(
                "MCMCSampler: %d crib assignments pinned: %s",
                len(self._cribs),
                self._cribs,
            )

        # IC-weighted proposal: signs with higher IC contribution receive
        # proportionally more proposal attempts, so their assignments are
        # refined more aggressively.  Weights are normalised to sum=1.
        # Falls back to uniform if not supplied.
        self._sign_proposal_weights: dict[str, float] = {}
        if sign_ic_weights:
            free = self._free_sign_ids if self._free_sign_ids else self._sign_ids
            raw_w = [max(sign_ic_weights.get(s, 0.0), 1e-6) for s in free]
            total_w = sum(raw_w)
            self._sign_proposal_weights = {
                s: w / total_w for s, w in zip(free, raw_w)
            }
            logger.info(
                "MCMCSampler: IC-weighted proposals enabled for %d free signs "
                "(top-3 by weight: %s).",
                len(free),
                sorted(self._sign_proposal_weights, key=self._sign_proposal_weights.get, reverse=True)[:3],
            )

        mc = cfg.zone_c.mcmc
        self._num_chains: int = int(mc.num_chains)
        self._num_iterations: int = int(mc.num_iterations)
        self._burn_in: int = int(mc.burn_in)
        self._thin: int = int(mc.thin)
        self._top_k: int = int(mc.top_k)
        self._rhat_threshold: float = float(mc.gelman_rubin_threshold)
        self._target_acceptance: float = float(mc.target_acceptance_rate)
        self._adaptation_interval: int = int(mc.adaptation_interval)
        self._reassign_prob_init: float = float(mc.reassign_prob)
        self._full_rescore_interval: int = int(getattr(mc, "full_rescore_interval", 1000))

        # Structural anti-collapse prior:
        #   prior = -weight * Σ_p max(0, n_p - cap)^2
        # where n_p is the number of signs mapped to phoneme p.
        self._occupancy_cap: int = int(getattr(mc, "max_signs_per_phoneme", 4))
        self._occupancy_weight: float = float(getattr(mc, "occupancy_penalty_weight", 0.0))
        if self._occupancy_cap < 1:
            raise ValueError("max_signs_per_phoneme must be >= 1")
        if self._occupancy_weight < 0.0:
            raise ValueError("occupancy_penalty_weight must be >= 0.0")
        if self._occupancy_weight > 0.0:
            logger.info(
                "MCMCSampler occupancy prior enabled: cap=%d, weight=%.3f",
                self._occupancy_cap,
                self._occupancy_weight,
            )

        # LM-guided (Gibbs-style) proposal parameters.
        self._lm_guided_prob: float = float(getattr(mc, "lm_guided_prob", 0.0))
        self._lm_guided_n_candidates: int = int(getattr(mc, "lm_guided_n_candidates", 3))
        self._lm_guided_top_k: int = int(getattr(mc, "lm_guided_top_k", 5))
        self._max_wall_time: float = float(getattr(mc, "max_wall_time_seconds", 0))

        # Parallel tempering config (replica exchange MCMC).
        from omegaconf import OmegaConf  # local import to avoid circular at module level
        self._pt_enabled: bool = bool(
            OmegaConf.select(cfg, "zone_c.mcmc.parallel_tempering.enabled", default=False)
        )
        self._pt_n_temperatures: int = int(
            OmegaConf.select(cfg, "zone_c.mcmc.parallel_tempering.n_temperatures", default=4)
        )
        self._pt_t_max: float = float(
            OmegaConf.select(cfg, "zone_c.mcmc.parallel_tempering.t_max", default=5.0)
        )
        self._pt_swap_interval: int = int(
            OmegaConf.select(cfg, "zone_c.mcmc.parallel_tempering.swap_interval", default=100)
        )

        # Precompute sign → per-sequence position index for incremental scoring.
        self._sign_positions: dict[str, list[list[int]]] = self._build_position_index(corpus_sequences)

    # ------------------------------------------------------------------
    # Position index
    # ------------------------------------------------------------------

    def _build_position_index(
        self, corpus_sequences: list[list[str]]
    ) -> dict[str, list[list[int]]]:
        """Build sign → per-sequence position lists, computed once at construction.

        Returns
        -------
        dict[str, list[list[int]]]
            ``index[sign][seq_idx]`` is a sorted list of 0-based token positions
            where ``sign`` appears in sequence ``seq_idx``.
        """
        index: dict[str, list[list[int]]] = {s: [] for s in self._sign_ids}
        for seq_idx, seq in enumerate(corpus_sequences):
            pos_in_seq: dict[str, list[int]] = {}
            for pos, sign in enumerate(seq):
                pos_in_seq.setdefault(sign, []).append(pos)
            for s in self._sign_ids:
                index[s].append(pos_in_seq.get(s, []))
        return index

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(self) -> MCMCResult:
        """Run all chains and return aggregated results.

        Dispatches to :meth:`_run_parallel_tempering` when
        ``cfg.zone_c.mcmc.parallel_tempering.enabled`` is true, otherwise
        runs independent chains with Gelman-Rubin diagnostics.

        Returns
        -------
        MCMCResult
        """
        if self._pt_enabled:
            logger.info(
                "Parallel tempering enabled: %d rungs, T_max=%.1f, swap every %d iters.",
                self._pt_n_temperatures, self._pt_t_max, self._pt_swap_interval,
            )
            return self._run_parallel_tempering()
        all_samples: list[list[MCMCSample]] = []
        all_rates: list[float] = []

        chain_configs: list[tuple[int, int | None]] = [
            (chain_id, (self._seed + chain_id) if self._seed is not None else None)
            for chain_id in range(self._num_chains)
        ]

        # Colab's Jupyter kernel is not fork-safe: the default 'fork' start method
        # lets worker processes inherit complex kernel state and they hang silently.
        # On Colab we use the 'spawn' start method instead, which starts clean
        # worker processes without kernel state.  HACKINGRONGO_SEQUENTIAL=1
        # forces sequential execution regardless of environment.
        _force_sequential = os.environ.get("HACKINGRONGO_SEQUENTIAL", "") == "1"
        _in_colab = (
            "COLAB_RELEASE_TAG" in os.environ
            or "COLAB_GPU" in os.environ
        )

        # Resolve which multiprocessing context (if any) to use.
        _mp_context = None
        if self._num_chains > 1 and not _force_sequential:
            if _in_colab:
                import multiprocessing as _mp
                try:
                    _mp_context = _mp.get_context("spawn")
                    logger.info(
                        "Colab environment: using 'spawn' mp_context for "
                        "ProcessPoolExecutor to avoid fork/kernel hang."
                    )
                except Exception as _ctx_err:
                    logger.warning(
                        "Could not obtain 'spawn' context (%s); "
                        "will run chains sequentially.",
                        _ctx_err,
                    )
            else:
                _mp_context = None  # default fork/forkserver on non-Colab

        if self._num_chains > 1 and not _force_sequential and (_mp_context is not None or not _in_colab):
            n_workers = min(self._num_chains, os.cpu_count() or 4)
            # Per-chain timeout: 20 min per 1 000 iterations (generous).
            _timeout = max(1200, self._num_iterations * 1.2)
            _executor_kwargs: dict = {"max_workers": n_workers}
            if _mp_context is not None:
                _executor_kwargs["mp_context"] = _mp_context
            try:
                with ProcessPoolExecutor(**_executor_kwargs) as pool:
                    futures = [
                        pool.submit(_run_chain_worker, self, chain_id, seed)
                        for chain_id, seed in chain_configs
                    ]
                    chain_results = [f.result(timeout=_timeout) for f in futures]
                all_samples = [r[0] for r in chain_results]
                all_rates = [r[1] for r in chain_results]
            except Exception as exc:
                logger.warning(
                    "Parallel chain execution failed (%s); falling back to sequential. "
                    "Ensure MCMCSampler state (including LMScorer) is picklable.",
                    exc,
                )
                all_samples, all_rates = [], []
                for chain_id, seed in chain_configs:
                    samples, rate = self._run_chain(chain_id, seed)
                    all_samples.append(samples)
                    all_rates.append(rate)
        else:
            if (_in_colab or _force_sequential) and self._num_chains > 1:
                reason = "HACKINGRONGO_SEQUENTIAL=1" if _force_sequential else "spawn context unavailable"
                logger.info(
                    "Running %d chains sequentially (%s).",
                    self._num_chains, reason,
                )
            for chain_id, seed in chain_configs:
                samples, rate = self._run_chain(chain_id, seed)
                all_samples.append(samples)
                all_rates.append(rate)

        for chain_id, (samples, rate) in enumerate(zip(all_samples, all_rates)):
            logger.info(
                "Chain %d: %d post-burn samples, acceptance=%.3f",
                chain_id, len(samples), rate,
            )

        rhat, converged = self._compute_rhat(all_samples, self._rhat_threshold)

        geweke_z: float | None = None
        if rhat is not None:
            logger.info("Gelman-Rubin R-hat = %.4f  (converged=%s)", rhat, converged)
        elif len(all_samples) == 1 and all_samples[0]:
            geweke_z = self._geweke_z(all_samples[0])
            converged = abs(geweke_z) < 2.0
            logger.info("Geweke Z-score = %.4f  (converged=%s)", geweke_z, converged)

        flat: list[MCMCSample] = [s for chain in all_samples for s in chain]
        flat.sort(key=lambda s: s.log_posterior, reverse=True)

        # Deduplicate: keep first occurrence of each unique assignment by
        # comparing canonical sorted-tuple representations.
        seen: set[tuple[tuple[str, str], ...]] = set()
        top_k: list[MCMCSample] = []
        for s in flat:
            key = tuple(sorted(s.phoneme_map.items()))
            if key not in seen:
                seen.add(key)
                top_k.append(s)
            if len(top_k) >= self._top_k:
                break

        n_per_chain = len(all_samples[0]) if all_samples else 0
        return MCMCResult(
            top_samples=top_k,
            acceptance_rates=all_rates,
            gelman_rubin_rhat=rhat,
            converged=converged,
            geweke_z=geweke_z,
            n_chains=self._num_chains,
            n_samples_per_chain=n_per_chain,
        )

    # ------------------------------------------------------------------
    # Parallel tempering (replica exchange MCMC)
    # ------------------------------------------------------------------

    def _run_parallel_tempering(self) -> MCMCResult:
        """Replica-exchange MCMC over a geometric temperature ladder.

        Runs :attr:`_pt_n_temperatures` chains at temperatures
        ``[1.0, …, t_max]`` (geometric spacing).  Every
        :attr:`_pt_swap_interval` iterations, adjacent-temperature pairs
        propose a Metropolis-Hastings swap:

            log α_swap = (1/T_i − 1/T_{i+1}) × (lp_{i+1} − lp_i)

        Only the cold chain (T=1.0) collects post-burn samples and
        contributes to the returned :class:`MCMCResult`.  Hot chains
        explore the landscape freely and pass good states downward.
        """
        n_T = self._pt_n_temperatures
        t_max = self._pt_t_max
        swap_interval = self._pt_swap_interval

        # Geometric ladder: temps[0]=1.0 (cold), temps[-1]=t_max (hot).
        if n_T == 1:
            temps: list[float] = [1.0]
        else:
            temps = [t_max ** (i / (n_T - 1)) for i in range(n_T)]

        rng = random.Random(self._seed)
        chain_maps = [
            self._random_initial_map(
                random.Random((self._seed + i) if self._seed is not None else None)
            )
            for i in range(n_T)
        ]
        chain_seqs = [self._translate_seqs(m) for m in chain_maps]
        chain_lps = [self._log_posterior_full(s, m) for s, m in zip(chain_seqs, chain_maps)]
        reassign_probs = [self._reassign_prob_init] * n_T
        recent_accepted = [0] * n_T
        accepted_post_burn = 0
        post_burn_steps = 0
        n_swap_proposed = 0
        n_swap_accepted = 0
        samples: list[MCMCSample] = []

        logger.info(
            "PT temperature ladder: %s",
            [f"{t:.2f}" for t in temps],
        )

        for it in range(self._num_iterations):
            # ── Update each rung independently ──────────────────────────────
            for t_idx in range(n_T):
                T = temps[t_idx]
                proposal, changes = self._propose(chain_maps[t_idx], rng, reassign_probs[t_idx])
                delta = self._compute_delta(chain_seqs[t_idx], changes)
                proposal_lp = (
                    chain_lps[t_idx]
                    + delta
                    + self._occupancy_prior_delta(chain_maps[t_idx], changes)
                )
                # Tempered acceptance: scale log-ratio by 1/T.
                log_alpha = (proposal_lp - chain_lps[t_idx]) / T
                if math.log(max(rng.random(), 1e-300)) < log_alpha:
                    chain_maps[t_idx] = proposal
                    chain_lps[t_idx] = proposal_lp
                    for sign, (_, new_ph) in changes.items():
                        for seq_idx, positions in enumerate(
                            self._sign_positions[sign]
                        ):
                            for pos in positions:
                                chain_seqs[t_idx][seq_idx][pos] = new_ph
                    recent_accepted[t_idx] += 1
                    if t_idx == 0 and it >= self._burn_in:
                        accepted_post_burn += 1

            # ── Periodic full rescore for cold chain (fp drift guard) ────────
            if (it + 1) % self._full_rescore_interval == 0:
                chain_lps[0] = self._log_posterior_full(chain_seqs[0], chain_maps[0])

            # ── Replica exchange swaps ───────────────────────────────────────
            if (it + 1) % swap_interval == 0:
                for t_idx in range(n_T - 1):
                    T_i, T_j = temps[t_idx], temps[t_idx + 1]
                    lp_i, lp_j = chain_lps[t_idx], chain_lps[t_idx + 1]
                    log_swap = (1.0 / T_i - 1.0 / T_j) * (lp_j - lp_i)
                    n_swap_proposed += 1
                    if math.log(max(rng.random(), 1e-300)) < log_swap:
                        chain_maps[t_idx], chain_maps[t_idx + 1] = (
                            chain_maps[t_idx + 1], chain_maps[t_idx]
                        )
                        chain_seqs[t_idx], chain_seqs[t_idx + 1] = (
                            chain_seqs[t_idx + 1], chain_seqs[t_idx]
                        )
                        chain_lps[t_idx], chain_lps[t_idx + 1] = (
                            chain_lps[t_idx + 1], chain_lps[t_idx]
                        )
                        n_swap_accepted += 1

            # ── Adapt proposal widths ────────────────────────────────────────
            if it < self._burn_in and (it + 1) % self._adaptation_interval == 0:
                for t_idx in range(n_T):
                    local_rate = recent_accepted[t_idx] / self._adaptation_interval
                    if local_rate > self._target_acceptance:
                        reassign_probs[t_idx] = min(reassign_probs[t_idx] * 1.1, 0.9)
                    else:
                        reassign_probs[t_idx] = max(reassign_probs[t_idx] * 0.9, 0.05)
                recent_accepted = [0] * n_T

            # ── Collect cold-chain samples ───────────────────────────────────
            if it >= self._burn_in:
                post_burn_steps += 1
                if (it - self._burn_in) % self._thin == 0:
                    samples.append(
                        MCMCSample(
                            phoneme_map=dict(chain_maps[0]),
                            log_posterior=chain_lps[0],
                            iteration=it,
                            chain_id=0,
                        )
                    )

        swap_rate = n_swap_accepted / max(n_swap_proposed, 1)
        cold_rate = accepted_post_burn / max(post_burn_steps, 1)
        logger.info(
            "PT complete: cold-chain acceptance=%.3f, swap rate=%.3f (%d/%d)",
            cold_rate, swap_rate, n_swap_accepted, n_swap_proposed,
        )

        geweke_z = self._geweke_z(samples) if len(samples) >= 10 else None
        converged = abs(geweke_z) < 2.0 if geweke_z is not None else False

        samples.sort(key=lambda s: s.log_posterior, reverse=True)
        seen: set[tuple[tuple[str, str], ...]] = set()
        top_k: list[MCMCSample] = []
        for s in samples:
            key = tuple(sorted(s.phoneme_map.items()))
            if key not in seen:
                seen.add(key)
                top_k.append(s)
            if len(top_k) >= self._top_k:
                break

        return MCMCResult(
            top_samples=top_k,
            acceptance_rates=[cold_rate],
            gelman_rubin_rhat=None,
            converged=converged,
            geweke_z=geweke_z,
            n_chains=n_T,
            n_samples_per_chain=len(samples),
        )

    # ------------------------------------------------------------------
    # Chain runner
    # ------------------------------------------------------------------

    def _run_chain(
        self,
        chain_id: int,
        seed: int | None,
    ) -> tuple[list[MCMCSample], float]:
        """Run a single Markov chain.

        Returns
        -------
        tuple[list[MCMCSample], float]
            (post-burn-in thinned samples, acceptance_rate_after_burnin)
        """
        rng = random.Random(seed)
        current_map = self._random_initial_map(rng)
        # Translate corpus once; update in-place on each accepted step.
        current_phoneme_seqs: list[list[str]] = self._translate_seqs(current_map)
        current_lp = self._log_posterior_full(current_phoneme_seqs, current_map)

        accepted_post_burn = 0
        post_burn_steps = 0

        samples: list[MCMCSample] = []
        reassign_prob = self._reassign_prob_init
        recent_accepted = 0
        _chain_t0 = time.monotonic()
        _wall_limit = self._max_wall_time if self._max_wall_time > 0 else float("inf")

        for it in range(self._num_iterations):
            if it > 0 and it % 500 == 0:
                if time.monotonic() - _chain_t0 > _wall_limit:
                    logger.warning(
                        "Chain %d: wall-time limit %.0fs reached at iteration %d — stopping early.",
                        chain_id, _wall_limit, it,
                    )
                    break
            if self._lm_guided_prob > 0.0 and rng.random() < self._lm_guided_prob:
                proposal, changes = self._greedy_proposal(
                    current_map, current_phoneme_seqs, rng
                )
            else:
                proposal, changes = self._propose(current_map, rng, reassign_prob)
            # Incremental delta: only recompute n-grams touching changed positions.
            proposal_lp = (
                current_lp
                + self._compute_delta(current_phoneme_seqs, changes)
                + self._occupancy_prior_delta(current_map, changes)
            )

            log_alpha = proposal_lp - current_lp
            if math.log(max(rng.random(), 1e-300)) < log_alpha:
                current_map = proposal
                current_lp = proposal_lp
                # Update translated sequences in-place.
                for sign, (_, new_ph) in changes.items():
                    for seq_idx, positions in enumerate(self._sign_positions[sign]):
                        for pos in positions:
                            current_phoneme_seqs[seq_idx][pos] = new_ph
                recent_accepted += 1
                if it >= self._burn_in:
                    accepted_post_burn += 1

            # Periodic full rescore: reset current_lp from scratch to prevent
            # floating-point drift accumulating across thousands of delta updates.
            if (it + 1) % self._full_rescore_interval == 0:
                verified_lp = self._log_posterior_full(current_phoneme_seqs, current_map)
                drift = abs(verified_lp - current_lp)
                if drift > 1e-6:
                    logger.debug(
                        "Chain %d it %d: fp drift %.2e — resetting lp %.6f → %.6f",
                        chain_id, it, drift, current_lp, verified_lp,
                    )
                current_lp = verified_lp

            if it >= self._burn_in:
                post_burn_steps += 1
                if (it - self._burn_in) % self._thin == 0:
                    samples.append(
                        MCMCSample(
                            phoneme_map=dict(current_map),
                            log_posterior=current_lp,
                            iteration=it,
                            chain_id=chain_id,
                        )
                    )

            # Adapt proposal every adaptation_interval steps during burn-in.
            if it < self._burn_in and (it + 1) % self._adaptation_interval == 0:
                local_rate = recent_accepted / self._adaptation_interval
                if local_rate > self._target_acceptance:
                    reassign_prob = min(reassign_prob * 1.1, 0.9)
                else:
                    reassign_prob = max(reassign_prob * 0.9, 0.05)
                recent_accepted = 0

            # Periodic progress report every 10 000 iterations.
            # Also write to a sidecar file so spawn-context workers are visible
            # in Kaggle/Colab where child-process stdout is not streamed.
            _progress_every = 10_000
            if it > 0 and it % _progress_every == 0:
                logger.info(
                    "Chain %d  %d/%d iters  (%.0f%%)  lp=%.3f",
                    chain_id, it, self._num_iterations,
                    100.0 * it / self._num_iterations,
                    current_lp,
                )
                try:
                    import pathlib, datetime
                    _prog_path = pathlib.Path("outputs") / "_mcmc_progress.log"
                    _prog_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(_prog_path, "a") as _pf:
                        _pf.write(
                            f"{datetime.datetime.now().strftime('%H:%M:%S')}  "
                            f"chain {chain_id}  {it}/{self._num_iterations}  "
                            f"({100.0*it/self._num_iterations:.0f}%)  lp={current_lp:.3f}\n"
                        )
                except Exception:
                    pass

        post_rate = accepted_post_burn / max(post_burn_steps, 1)
        return samples, post_rate

    def _occupancy_log_prior(self, phoneme_map: PhonemeMap) -> float:
        """Structural log-prior discouraging many-to-one phoneme collapse.

        Penalty is normalised by n_free (number of non-crib signs) so the
        scale is independent of corpus or sign-inventory size.  Without this,
        the quadratic sum grows with inventory size and overwhelms the LM
        signal at full-corpus scale.
        """
        if self._occupancy_weight <= 0.0:
            return 0.0

        counts = Counter(phoneme_map.values())
        penalty = 0.0
        for n in counts.values():
            excess = max(0, n - self._occupancy_cap)
            penalty += float(excess * excess)
        n_free = max(len(self._free_sign_ids), 1)
        return -self._occupancy_weight * penalty / n_free

    def _occupancy_prior_delta(
        self,
        current_map: PhonemeMap,
        changes: dict[str, tuple[str, str]],
    ) -> float:
        """Delta of the occupancy prior under proposed sign reassignment(s)."""
        if self._occupancy_weight <= 0.0 or not changes:
            return 0.0

        proposal = dict(current_map)
        for sign, (_, new_ph) in changes.items():
            proposal[sign] = new_ph
        return self._occupancy_log_prior(proposal) - self._occupancy_log_prior(current_map)

    # ------------------------------------------------------------------
    # Posterior scoring
    # ------------------------------------------------------------------

    def _log_posterior_full(
        self,
        phoneme_seqs: list[list[str]],
        phoneme_map: PhonemeMap,
    ) -> float:
        """Full log-posterior from pre-translated phoneme sequences.

        Called once per chain initialisation only.  All subsequent steps
        use :meth:`_compute_delta` for O(affected_positions) updates.
        """
        total = 0.0
        for phoneme_seq in phoneme_seqs:
            result = self._scorer.score(phoneme_seq)
            if math.isfinite(result.ensemble_log_prob):
                total += result.ensemble_log_prob
            else:
                total -= 1000.0
        total += self._occupancy_log_prior(phoneme_map)
        return total

    def _translate_seqs(self, phoneme_map: PhonemeMap) -> list[list[str]]:
        """Translate all corpus sequences to phoneme sequences."""
        return [
            [phoneme_map.get(sign, "<UNK>") for sign in seq]
            for seq in self._corpus_sequences
        ]

    def _compute_delta(
        self,
        phoneme_seqs: list[list[str]],
        changes: dict[str, tuple[str, str]],
    ) -> float:
        """Incremental log-posterior delta from changing one or two signs.

        Parameters
        ----------
        phoneme_seqs : list[list[str]]
            Current translated sequences (not modified here).
        changes : dict[str, tuple[str, str]]
            ``{sign_id: (old_phoneme, new_phoneme)}`` for each modified sign.

        Returns
        -------
        float
            Delta in total ensemble log-posterior (new − old).
        """
        if not changes:
            return 0.0

        total_delta = 0.0
        for seq_idx, phoneme_seq in enumerate(phoneme_seqs):
            changed_positions: list[int] = []
            old_phonemes: list[str] = []
            new_phonemes: list[str] = []
            for sign, (old_ph, new_ph) in changes.items():
                positions = self._sign_positions[sign][seq_idx]
                changed_positions.extend(positions)
                old_phonemes.extend([old_ph] * len(positions))
                new_phonemes.extend([new_ph] * len(positions))
            if changed_positions:
                total_delta += self._scorer.score_delta(
                    phoneme_seq, changed_positions, old_phonemes, new_phonemes
                )
        return total_delta

    # ------------------------------------------------------------------
    # Proposal
    # ------------------------------------------------------------------

    def _greedy_proposal(
        self,
        current: PhonemeMap,
        phoneme_seqs: list[list[str]],
        rng: random.Random,
    ) -> tuple[PhonemeMap, dict[str, tuple[str, str]]]:
        """LM-guided proposal: find the ``(sign, phoneme)`` swap with highest delta.

        Samples :attr:`_lm_guided_n_candidates` signs and tries the top-k
        phonemes (weighted by prior) for each, proposing the change that
        gives the largest improvement in log-posterior.  Falls back to a
        no-op (zero changes) if all deltas are negative (the MH step will
        then likely reject, but the chain remains valid).

        Cost: O(n_candidates × top_k) delta evaluations per step vs O(1)
        for a standard random-walk proposal.  The fraction of steps using
        this move is controlled by :attr:`_lm_guided_prob`.
        """
        free = self._free_sign_ids if self._free_sign_ids else self._sign_ids
        n_cand = min(self._lm_guided_n_candidates, len(free))
        signs_to_try = self._sample_sign(free, rng, n=n_cand)

        # Draw top_k phonemes weighted by prior (fast unigram guidance).
        top_k = min(self._lm_guided_top_k, len(self._phoneme_inventory))
        phoneme_pool = rng.choices(
            self._phoneme_inventory,
            weights=self._phoneme_priors,
            k=top_k * 2,  # oversample to get top_k unique
        )
        # Deduplicate while preserving order.
        seen: set[str] = set()
        phonemes_dedup: list[str] = []
        for p in phoneme_pool:
            if p not in seen:
                seen.add(p)
                phonemes_dedup.append(p)
                if len(phonemes_dedup) == top_k:
                    break

        best_delta = -float("inf")
        best_sign: str = signs_to_try[0]
        best_old_ph: str = current[best_sign]
        best_new_ph: str = best_old_ph  # default: no change

        for sign in signs_to_try:
            old_ph = current[sign]
            for new_ph in phonemes_dedup:
                if new_ph == old_ph:
                    continue
                delta = self._compute_delta(phoneme_seqs, {sign: (old_ph, new_ph)})
                if delta > best_delta:
                    best_delta = delta
                    best_sign = sign
                    best_old_ph = old_ph
                    best_new_ph = new_ph

        proposal = dict(current)
        changes: dict[str, tuple[str, str]] = {}
        if best_new_ph != best_old_ph:
            proposal[best_sign] = best_new_ph
            changes[best_sign] = (best_old_ph, best_new_ph)
        return proposal, changes

    def _sample_sign(
        self,
        sign_ids: list[str],
        rng: random.Random,
        n: int = 1,
    ) -> list[str]:
        """Sample ``n`` sign(s) from ``sign_ids`` using IC proposal weights.

        Falls back to uniform sampling when no weights are set.
        """
        if self._sign_proposal_weights and n == 1:
            weights = [self._sign_proposal_weights.get(s, 1e-6) for s in sign_ids]
            return rng.choices(sign_ids, weights=weights, k=1)
        if self._sign_proposal_weights and n == 2 and len(sign_ids) >= 2:
            # Weighted sampling without replacement: draw first, then second from remainder.
            weights = [self._sign_proposal_weights.get(s, 1e-6) for s in sign_ids]
            s1 = rng.choices(sign_ids, weights=weights, k=1)[0]
            rest = [s for s in sign_ids if s != s1]
            rest_w = [self._sign_proposal_weights.get(s, 1e-6) for s in rest]
            s2 = rng.choices(rest, weights=rest_w, k=1)[0]
            return [s1, s2]
        return rng.sample(sign_ids, n)

    def _propose(
        self,
        current: PhonemeMap,
        rng: random.Random,
        reassign_prob: float,
    ) -> tuple[PhonemeMap, dict[str, tuple[str, str]]]:
        """Produce a proposal and return ``(proposal, changes)``.

        Sign selection is weighted by IC contribution when ``sign_ic_weights``
        was supplied at construction — high-IC signs are proposed more often,
        focusing search budget where it matters most for LM scoring.
        Crib signs are never included in proposals.
        """
        proposal = dict(current)
        changes: dict[str, tuple[str, str]] = {}
        sign_ids = self._free_sign_ids if self._free_sign_ids else self._sign_ids

        if len(sign_ids) < 2 or rng.random() < reassign_prob:
            # Random reassignment: pick one free sign (IC-weighted), assign a random phoneme.
            sign = self._sample_sign(sign_ids, rng, n=1)[0]
            old_ph = proposal[sign]
            new_ph = rng.choices(self._phoneme_inventory, weights=self._phoneme_priors)[0]
            proposal[sign] = new_ph
            if old_ph != new_ph:
                changes[sign] = (old_ph, new_ph)
        else:
            # Swap: pick two free signs (IC-weighted), exchange their phoneme assignments.
            s1, s2 = self._sample_sign(sign_ids, rng, n=2)
            old_ph1, old_ph2 = proposal[s1], proposal[s2]
            proposal[s1], proposal[s2] = old_ph2, old_ph1
            if old_ph1 != old_ph2:
                changes[s1] = (old_ph1, old_ph2)
                changes[s2] = (old_ph2, old_ph1)

        return proposal, changes

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _random_initial_map(self, rng: random.Random) -> PhonemeMap:
        """Build a random initial phoneme map; crib signs are pre-set."""
        m = {
            sign: rng.choices(self._phoneme_inventory, weights=self._phoneme_priors)[0]
            for sign in self._sign_ids
        }
        for sign, phoneme in self._cribs.items():
            if sign in m:
                m[sign] = phoneme
        return m

    # ------------------------------------------------------------------
    # Convergence diagnostics
    # ------------------------------------------------------------------

    @staticmethod
    def _geweke_z(
        samples: list[MCMCSample],
        first: float = 0.1,
        last: float = 0.5,
    ) -> float:
        """Geweke Z-score for single-chain stationarity.

        Compares the mean of the first ``first`` fraction of the
        post-burn-in log-posterior trace to the last ``last`` fraction.
        Returns 0.0 if either segment has fewer than 2 elements or
        zero variance (chain did not move).

        References
        ----------
        Geweke, J. (1992). Evaluating the Accuracy of Sampling-Based
        Approaches to the Calculation of Posterior Moments.
        *Bayesian Statistics 4*.
        """
        trace = np.array([s.log_posterior for s in samples])
        n = len(trace)
        if n < 10:
            return 0.0
        a = trace[: int(n * first)]
        b = trace[int(n * (1 - last)) :]
        var = a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b)
        return float((a.mean() - b.mean()) / np.sqrt(var)) if var > 0 else 0.0

    @staticmethod
    def _compute_rhat(
        all_samples: list[list[MCMCSample]],
        rhat_threshold: float = 1.1,
    ) -> tuple[float | None, bool]:
        """Compute the scalar Gelman-Rubin R-hat across chains.

        Uses the log-posterior trace as the univariate diagnostic
        quantity.  Returns ``(None, False)`` if fewer than 2 chains are
        available or chains are empty.  Convergence is declared when
        ``rhat < rhat_threshold``.

        References
        ----------
        Gelman, A. & Rubin, D. (1992). Inference from Iterative
        Simulation Using Multiple Sequences.  *Statistical Science*.
        """
        if len(all_samples) < 2:
            return None, False

        chain_traces: list[np.ndarray] = []
        for chain in all_samples:
            if len(chain) == 0:
                return None, False
            chain_traces.append(np.array([s.log_posterior for s in chain]))

        # Truncate to the shortest chain.
        min_len = min(len(t) for t in chain_traces)
        if min_len < 2:
            return None, False
        traces = np.stack([t[:min_len] for t in chain_traces])  # (M, N)

        M, N = traces.shape
        chain_means = traces.mean(axis=1)       # (M,)
        grand_mean = chain_means.mean()

        # Between-chain variance B.
        B = N / (M - 1) * np.sum((chain_means - grand_mean) ** 2)

        # Within-chain variance W.
        within = np.var(traces, axis=1, ddof=1)
        W = within.mean()

        if W < 1e-10:
            # All chains collapsed; report as not converged.
            return None, False

        var_hat = (N - 1) / N * W + B / N
        rhat = float(np.sqrt(var_hat / W))
        return rhat, rhat < rhat_threshold
