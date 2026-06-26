# Deprecated: the syllabic substitution-cipher track (a recorded negative result)

**Status:** set down 2026-06, in place. Not deleted. The code remains in the
repository and in git history as a preserved negative result. The modules carry
`DEPRECATED — SYLLABIC SUBSTITUTION-CIPHER TRACK` header banners.

This document reports the syllabic null as carefully as any positive result.
Every number below is pulled from a named on-disk file; anything not confirmable
from a file is marked `[NOT FOUND]`.

---

## The hypothesis we tested

That rongorongo is a **phonetic / syllabic substitution cipher**: each sign maps
to a Rapa Nui syllable, and the correct sign→phoneme key can be recovered by
searching for the assignment under which the tablet sequences score best against
**era-stratified Rapa Nui language models** (pre-contact: Thomson 1891 + Roussel
1908; post-contact: Fuentes 1960 + Englert 1978 + IDS), with a canonical
50-syllable inventory (`hackingrongo/data/phoneme_inventory.py`).

The search is Bayesian: Metropolis–Hastings MCMC over sign→phoneme maps
(`hackingrongo/zone_c/mcmc.py`), scored by KN-smoothed n-gram LMs
(`hackingrongo/zone_c/lm_scoring.py`), with beam-search refinement
(`beam_search.py`) and optional fusion-layer proposal weights (`fusion.py`).
Quantum variants of the *same* key search — QUBO annealing
(`run_qubo_decipherment.py`), QAOA (`run_qaoa_decipherment.py`) — and the
p_good / Grover hardness framing (`measure_pgood.py`, `build_grover_oracle.py`)
all assume this same phoneme substitution key, and are deprecated with it.

## How it was tested

The full pipeline (`scripts/run_decipherment.py`): build era-stratified phoneme
LMs (`scripts/tooling/build_language_models.py`), run multi-chain MCMC over the
sign→phoneme map against those LMs, check mixing with convergence diagnostics
(Gelman–Rubin R-hat across chains; Geweke z within a chain), and rank the
resulting hypotheses by language-model score.

## The result (file-grounded)

**The full decipherment run converged as a sampler — but produced no defensible
decipherment.** Two `mcmc_diagnostics.json` runs exist and must not be conflated:

| Run | file | inventory | chains × samples | R-hat | Geweke z | converged | acc. |
|---|---|---|---|---|---|---|---|
| **Full** (AzureML 2026-06-15) | `outputs/decipherment/mcmc_diagnostics.json` (Azure) | **824 signs** | **4 × 450** | **0.9994** | (not computed) | **true** | 0.3953 |
| Reduced / single-chain | `outputs/decipherment/mcmc_diagnostics.json` (local checkout) | 98 signs | 1 × 80 | none (1 chain) | **−12.6812** | **false** | 0.3275 |

The **full** run (824-sign inventory, 4 chains) reached Gelman–Rubin
**R-hat = 0.9994** — the chains mixed and agreed. The reduced single-chain run
(98 signs, 80 samples) did not converge (Geweke z = −12.68) and, being a single
chain, has no R-hat; it is a smoke/diagnostic run, not the decipherment result.

**Convergence of the sampler is not a decipherment.** R-hat ≈ 1 means the chains
agree on a posterior — it does **not** mean the recovered key is linguistically
real. On scrutiny the converged run yielded **no sign→phoneme assignment that was
ever defensible as a reading**, and none was reported as a finding. Two
file-grounded facts make the point concrete:

- **The data prefer logograms over a pure syllabary.** A mixed logo-syllabic
  model scored **+123.2 LM units better** than the pure syllabic model
  (`outputs/decipherment/mixed_model/model_comparison.json`: syllabic
  `overall_lm_score` −3500.88 vs mixed −3377.71). The best-scoring account treats
  signs such as 280/600/690/700 as logograms (honu/manu/tangata-manu/ika), not as
  syllables.
- **The "hardness certificate" rests on the same unsupported key.** The p_good
  analysis (`outputs/zone_b/pgood_analysis.json`) finds the fraction of
  "good" assignments is **0.01 at τ=0.99 — i.e. 1 of 100 sampled assignments
  (`n_good = 1`)**, so the certificate is a single-sample estimate built around a
  phoneme substitution key that the scoring evidence does not support.

## The interpretation

We are **setting the syllabic substitution-cipher hypothesis down**, for three
reasons taken together: (1) the full MCMC converges yet yields no linguistically
defensible sign→phoneme assignment; (2) on its own scoring, the corpus prefers a
**mixed logo-syllabic** model (+123) over a pure syllabary; and (3) the field
leans toward a logographic or logo-syllabic script. This motivates a pivot to a
**structural / logographic** approach (template-matching against known oral
genres), which carries forward all of the structural machinery (corpus,
normalization, allograph/compound handling, parallel passages, contact partition,
network centrality, the distributional sign-role classifier, stratification,
provenance/reporting).

**This is a hypothesis we set down, not a proof.** Convergence of the sampler is
not disproof of syllabicity, and the absence of a defensible decipherment is not
proof that the script is non-syllabic. We are reallocating effort to a more
promising track, and recording why.

## Preservation note

Nothing here is deleted. The deprecated modules remain in the repository and in
git history as a recorded negative result, each marked with a header banner
pointing to this file. They are preserved, not fixed, tuned, or removed.

**Deprecated modules:**
`hackingrongo/zone_c/mcmc.py`, `beam_search.py`, `lm_scoring.py`, `fusion.py`;
`scripts/run_decipherment.py`, `run_qubo_decipherment.py`,
`run_qaoa_decipherment.py`, `measure_pgood.py`, `build_grover_oracle.py`,
`scripts/tooling/build_language_models.py`; and the phoneme LMs in
`data/language_models/`.

**Held for review (not yet deprecated):** `hackingrongo/zone_c/mixed_decoder.py`
— the logo-syllabic hybrid that produced the +123 result. It is part-syllabic
but is also the most direct bridge toward the logographic track, and may be a
*seed* for it rather than a dead end.

<!-- Numbers verified against: outputs/decipherment/mcmc_diagnostics.json (local: 1 chain/80/R-hat null/Geweke -12.6812/converged false/acc 0.3275/inv 98); the AzureML 2026-06-15 megabundle copy of the same file (4 chains/450/R-hat 0.9994/converged true/acc 0.3953/inv 824); outputs/decipherment/mixed_model/model_comparison.json (syllabic -3500.878333, mixed -3377.713401, delta +123.165); outputs/zone_b/pgood_analysis.json (tau 0.99, p_good 0.01, n_good 1, 100 samples). The full-run mcmc_diagnostics lives on the Azure instance, not this checkout. -->
