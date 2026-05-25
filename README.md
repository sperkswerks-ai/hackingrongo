# Hacking Rongorongo Project by SperksWerks

The Rongorongo script is the only known indigenous "writing" system of Oceania. It has never been deciphered. No bilingual text exists. No key. No known plaintext. And there are only 15,273 glyphs across 26 wooden objects.

This project treats decipherment as a cryptanalysis problem and applies a six-layer adversarial attack pipeline with visual embeddings, statistical analysis, oracle inversion, differentiable projection learning, adversarial validation, and quantum hardness analysis.

> The first computational pipeline to apply diachronic stratification, differential oracle attacks, and quantum complexity analysis to the Rongorongo corpus.

---
## Key Findings So Far

| Finding | Value | Method |
|:--------|:------|:-------|
| Visual clusters recovered | 695 at 94.6% mean purity | Zone A autoencoder |
| IC_pre ≠ IC_post | Statistically significant | Zone B, 3 sensitivity scenarios |
| Multi-tablet parallel passages | 13 found (3 pre-contact) | Zone B Kasiski cross-reference |
| H125 attestations | 18 tablets, 356 matches | Zone B |
| Sign 152 (full moon) | Calendar-exclusive, score 1.0 | Zone B astronomical analysis |
| Sign 040 (night-count) | Calendar-dominant, score 0.62 | Zone B astronomical analysis |
| Phoneme assignments produced | 120 signs mapped | Zone C MCMC + beam search |
| Sign 600 (bird) → *ha* | Breath/life in Polynesian | Zone C top hypothesis |
| Quantum speedup estimate | Varies by p_good threshold | `scripts/measure_pgood.py` |

The IC divergence finding holds under all three sensitivity scenarios (`conservative_all_late`, `optimistic_distributed`, `probabilistic_weighted`) and has not been reported in prior computational work on Rongorongo.

---

## For Hackers

This project demonstrates classical and quantum approaches to ciphertext-only attacks on an unknown cipher with no key, no known plaintext, and a 15,273 token corpus.

| Rongorongo concept | Cryptanalytic equivalent |
|---|---|
| Sign inventory (~120 signs) | Ciphertext alphabet |
| Phoneme map π | Substitution key |
| MCMC over phoneme assignments | Key search / simulated annealing |
| QUBO annealing | Quantum key search |
| Parallel passages | Identical plaintext attack material |
| Taxogram (sign 200) | Crib (known-function symbol) |
| Radiocarbon date boundary | Known key-change point |
| Mamari calendar section | Known-plaintext fragment |
| p_good measurement | Quantum hardness certificate |

The MCMC sampler uses Metropolis-Hastings with an adaptive proposal targeting 23.4% acceptance (Roberts et al.) — the same class of technique that cracked the Zodiac's Z340 cipher in 2020 after 51 years. The corpus is harder: unknown cipher type, unknown language, no ground truth.

---

## For Linguists

The pipeline makes no assumption that rongorongo is phonetic. Zone C tests the phonetic hypothesis by asking: is there any sign→phoneme assignment that makes the rongorongo sequences score significantly better under a Rapa Nui language model than a random permutation?

The diachronic stratification framework uses the Ferrara et al. 2024 radiocarbon dates as an analytical lens no prior computational study has applied. IC_pre ≠ IC_post — the script's statistical properties differ across the contact boundary. Three pre-contact parallel passages allow direct comparison of pre- and post-contact forms of the same sequence.

The language models are stratified by era:
- **Pre-contact LM** — Thomson (1891) + Roussel (1908) wordlists, ~1,345 forms, cognate-weighted with ABVD East Polynesian neighbours
- **Post-contact LM** — Fuentes (1960) + Englert (1978) + IDS canonical forms, ~2,754 forms
- **Smoothing LM** — Hawaiian Corpus Project unigram table, ~56,000 word types

This stratification means the decipherment search scores pre-contact tablet sequences against a language model appropriate to their era, not against modern Rapa Nui.


---

## For Data Scientists

**Zone A** trains a convolutional autoencoder on 13,967 SVG-rasterised glyph images with zero labels. UMAP + HDBSCAN produces 695 clusters at 94.6% mean purity against Barthel's 1958 taxonomy — which the model never saw.

**Zone B** applies three sensitivity scenarios (conservative_all_late, optimistic_distributed, probabilistic_weighted) to every finding. A result is only reported as robust if it holds within 10% across all three.

**Zone C** runs Metropolis-Hastings with Gelman-Rubin R-hat and Geweke Z convergence diagnostics. Incremental delta scoring reduces per-iteration cost from O(N) to O(k).

**Quantum analysis** (scripts/measure_pgood.py, scripts/run_qubo_decipherment.py) follows the methodology of Di Santo & Lanziani (2025) in computing p_good (the fraction of random assignments scoring above threshold) to derive Grover oracle call estimates and compare classical vs quantum search complexity. The QUBO formulation follows Zhang & Feng (2022) for cryptanalytic applications of D-Wave annealing.

---

## For Quantum Researchers

**p_good measurement** (scripts/measure_pgood.py) samples 10,000 random sign→phoneme assignments and measures what fraction score above threshold under the Rapa Nui language model. This produces a Grover oracle call estimate as a theoretical quantum speedup and characterises the hardness of the problem in information-theoretic terms. Runs in minutes on CPU.

**QUBO key search** (scripts/run_qubo_decipherment.py) reformulates the sign→phoneme assignment problem as a Quadratic Unconstrained Binary Optimisation problem, following Zhang & Feng (2022). The --solver neal flag runs classical simulated annealing via the D-Wave neal library (no account required). The --solver dwave flag submits to real D-Wave Advantage QPU hardware via D-Wave Leap (free research tier: cloud.dwavesys.com).

The quantum analysis does not promise to solve Rongorongo. It provides the first empirical quantum complexity characterisation of the decipherment problem — measuring how hard it is in quantum computational terms and what that hardness implies about the script's mathematical structure.

---

## Differences from Souza's Rongopy Project (2023)

| Aspect | Souza (2023) | This project |
|:-------|:-------------|:-------------|
| **Temporal model** | Flat corpus | Diachronic: Tablet D pre-contact anchor (Ferrara 2024); B/C/O/Q post-contact; A excluded |
| **Robustness testing** | None | Three sensitivity scenarios; finding reported only if holds within 10% across all three |
| **Visual embeddings** | None | Convolutional autoencoder; 695 clusters at 94.6% purity |
| **Compound detection** | None | 3-method cross-validated detector |
| **Parallel passages** | In-memory only | Algorithmic cross-reference; permutation test; diachronic variant analysis |
| **Astronomical analysis** | None | 5-test candidate detector; Mamari calendar anchor; Dietrich correspondence |
| **Sequence completion** | None | N-gram completion; damaged glyph reconstruction |
| **Quantum analysis** | None | p_good measurement; QUBO key search; D-Wave integration |
| **Results format** | Untyped CSV | Typed dataclasses; CSV; JSON; HTML |

---

## Limitations and Honest Caveats

The Rongorongo glyphs included for computational analysis are only from wooden objects and reimiro ornaments, however, cave petroglyphs have not been included. The IC finding is evidence of structure, not proof of language. Zone C has not yet produced a validated decipherment hypothesis. The Barthel family labels are an arithmetic function of the sign code, not a faithful representation of Barthel's iconographic taxonomy. 16.5% of corpus tokens have no image file (415 unique codes). These fall back to zero tensors during training. Running `scripts/extract_barthel_glyphs.py --source both` with the Barthel (1958) PDFs in `data/barthel_pdfs/` will close most of this gap.

---

## For more information
See the DATA_SOURCES.md for references and QUICKSTART.md for usage.