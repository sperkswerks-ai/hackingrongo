# Hacking Rongorongo Project by SperksWerks

The Rongorongo script is the only known indigenous writing system of Oceania. It has never been deciphered. No bilingual text exists. No key. No known plaintext. And there are only 15,273 glyphs across 25 inscribed wooden objects (Barthel's tablets A–Y; the Poike palimpsest Z is excluded as effaced and of disputed legibility).

This project treats decipherment as a cryptanalysis problem and applies a six-layer pipeline — visual embeddings, statistical analysis, oracle inversion, differentiable projection learning, parallel-passage analysis, and quantum complexity analysis — and now works directly from the **3D tablet geometry**, not just 2D facsimiles.

> A reproducible pipeline applying diachronic stratification, differential oracle attacks, and quantum complexity analysis to the Rongorongo corpus — an analytical lens no prior computational study has applied. It reports its null and artifactual results with the same prominence as its positive ones.

**I carefully tie every quantitative claim to its source file and make best effort to grade confidence of results.**
---
## Status (June 2026)

Two big things happened in June 2026:

1. **Data overhaul + pipeline rerun.** Language models are now built with structural (C)V phonotactic validation (removing tokenizer artifacts that were 40–92% of LM vocabularies); the sign→phoneme search space is a single canonical **50-syllable** Rapa Nui inventory shared by MCMC/QUBO/QAOA/p_good (`hackingrongo/data/phoneme_inventory.py`); and the CEIPP parser now resolves variant/modifier codes — recovering 563 tokens and collapsing 1,330 → **639 base signs**. The pipeline was rerun on the corrected corpus (on AzureML). **Several earlier headline numbers were walked back in the process — see the table.**
2. **3D geometry pipeline (new).** We decode the INSCRIBE 3D tablet meshes and analyse the actual carved geometry — to our knowledge the first computational study to do so. See [3D Glyph Geometry](#3d-glyph-geometry-new).

---
## Key Findings So Far

Graded **robust** (survives sensitivity analysis) · **suggestive** (directional, not robust) · **null/exploratory**. Values are from the corrected (AzureML, 2026-06-15) run.

| Finding | Tier | Value | Source |
|:--------|:-----|:------|:-------|
| IC pre- vs post-contact | suggestive → **confounded** | Raw IC_pre > IC_post in all 3 scenarios, but strata differ in inventory size up to ~5×; inventory-normalised IC *reverses*; `robust: false` (rel. variation ~55%) | `outputs/sensitivity_analysis.json` |
| Multi-tablet parallel passages | robust | 13 recovered (97 of Horley's 146 groups token-matched; 13 retained as multi-tablet unique sequences) | `data/parallels/parallel_variants_auto.json` |
| Contact-partition frequency shift | suggestive | 20 signs at uncorrected p<.05, but only **1** ("200 9") survives Bonferroni/Benjamini–Hochberg | `outputs/contact_partition.json` |
| Pozdniakov paradigmatic replication | **null** | 0 of 15 reference classes recovered (F1 = 0.000); structurally suggestive near-misses | `outputs/analysis/pozdniakov_paradigmatic.json` |
| Mixed logo-syllabic vs pure syllabic | suggestive | Mixed model scores **+123 LM units** better, treating 280/600/690/700 as logographic cribs (honu/manu/tangata-manu/ika) | `outputs/decipherment/mixed_model/model_comparison.json` |
| Sign 76 / 532 functional convergence | suggestive | High-skew/high-centrality hubs, consistent with Davletshin's *ko* (76) and *ʻariki* (532) readings — candidate-level | preprint §3.7 |
| Visual clusters (Zone A) | null/exploratory | 772 clusters, interpretation **"divergent"** (ARI 0.014 vs Barthel families / 0.060 vs codes); the old "94.6% purity" was a misleading metric and is retired | `outputs/analysis/cluster_vs_barthel.json` |
| BV linear structure in IC | **null** | Affine structure is a tautological artifact of the rank encoding — no genuine linear structure | `quantum_results/bv/` |
| Simon separation on key-changes | null/exploratory | Periods recovered, but separation is **absent** at the available n=2,4 | `quantum_results/simon/` |
| QAOA vs MCMC (corrected scorer) | null/exploratory | +0.543 log-prob on a 4-sign toy subproblem — negligible (the +7687 in the old file was a scale artifact) | `quantum_results/qaoa/` |
| Quantum hardness (p_good) | suggestive | p_good ≈ 0.0005 at τ=0.99 → Grover ratio, but n_good = 1 → high uncertainty | `outputs/zone_b/pgood_analysis.json` |
| MCMC decipherment convergence | — | Converged: R-hat = **0.9994** over 4 chains. Convergence ≠ a validated decipherment; **no phoneme assignment is presented as a finding** | `outputs/decipherment/mcmc_diagnostics.json` |

The BV and Pozdniakov nulls are themselves informative, and we report them as prominently as the positive results.

---

## 3D Glyph Geometry (new)

The INSCRIBE Project openly publishes high-resolution 3D scans of Tablets B (Aruku Kurenga), C (Mamari), and D (Échancrée) in CNR-ISTI **Nexus** (`.nxz`) format. We work from that geometry directly:

1. **Decode** — build the Nexus C++ toolchain (`scripts/tooling/setup_nexus_tools.sh`) and extract full-resolution PLY (`nxz_to_ply.py`). Decoded sizes: **D 10.1M, B 5.75M, C 4.33M vertices** (~20M triangles total).
2. **Render** — `render_mesh_highres.py` produces 4K **raking-light + relief** images from the real mesh (slab-aware, camera face-on, light swept across the carvings — the digital analogue of Reflectance Transformation Imaging). Incisions are shallow (~2 grey-levels), so the relief pass high-passes the depth buffer to make carvings pop, independent of lighting.
3. **Segment** — `segment_relief_glyphs.py` cuts the relief into per-glyph crops **labelled by corpus position** (the corpus glyph-count per line drives the split, fixing the old ~33% under-segmentation), at native resolution. Orientation verified against Barthel's Tafeln plates.

This yields a clean, high-fidelity, corpus-labelled glyph set (`data/glyphs/glyph_crops/`, tagged `source_quality: "3d_relief_highdef"`) — kept **separate** from the 2D-facsimile SVG corpus (see [Data provenance](#data-provenance--governance)). Big shout-out to the **INSCRIBE Project** for the open 3D models (cited in `DATA_SOURCES.md`).

---

## For Hackers

Classical and quantum approaches to ciphertext-only attacks on an unknown cipher with no key, no known plaintext, and a 15,273-token corpus.

| Rongorongo concept | Cryptanalytic equivalent |
|---|---|
| Sign inventory (639 base signs; top 120 ≈ 83% of identified tokens) | Ciphertext alphabet |
| Phoneme map π (canonical 50-syllable Rapa Nui inventory) | Substitution key |
| MCMC over phoneme assignments | Key search / simulated annealing |
| QUBO annealing | Quantum key search |
| Parallel passages | Identical plaintext attack material |
| Contact-shifted sign ("200 9", the lone G²-survivor) | Crib candidate (frequency tell across the key-change) |
| Radiocarbon date boundary | Known key-change point |
| Mamari calendar section | Known-plaintext fragment |
| p_good measurement | Quantum hardness certificate |

The MCMC sampler uses Metropolis-Hastings with an adaptive proposal targeting 23.4% acceptance (Roberts, Gelman & Gilks 1997) — the same class of technique that cracked the Zodiac Z340 cipher in 2020 after 51 years. The corpus is harder: unknown cipher type, unknown language, no ground truth.

---

## For Linguists

The pipeline makes no assumption that rongorongo is phonetic. Zone C tests the phonetic hypothesis by asking: is there any sign→phoneme assignment that makes the sequences score significantly better under a Rapa Nui language model than a random permutation?

The diachronic framework uses the Ferrara et al. 2024 radiocarbon dates as a lens no prior computational study has applied. The raw IC is higher pre-contact than post-contact, but we now treat this as **confounded by sign-inventory size** rather than a clean diachronic signal: the strata differ in inventory up to ~5×, and inventory-normalised IC reverses direction depending on how undated tablets are assigned (preprint §3.1). With only Tablet D securely dated pre-contact and 19 of 25 tablets undated, every diachronic claim is underpowered.

A genuine convergence worth noting: Davletshin (2012) reads sign **76** as the prominence marker *ko* (before personal names) and **532** as *ʻariki* "chief"; de Souza (2025) gives statistical support. Our blind distributional analysis independently flags both as high-centrality / directionally-bound signs — consistent with those readings, though we confirm only "behaves like a load-bearing particle," not the phonetic value.

Language models are stratified by era:
- **Pre-contact LM** — Thomson (1891) + Roussel (1908), ~1,345 forms, cognate-weighted with ABVD East Polynesian neighbours
- **Post-contact LM** — Fuentes (1960) + Englert (1978) + IDS forms, ~2,754 forms
- **Smoothing LM** — Hawaiian Corpus Project unigrams, ~56,000 word types

All LM tokens pass structural (C)V phonotactic validation (Rapa Nui admits no codas/clusters); *nga*→*ga*, *v* is phonemic, the glottal stop is unrepresented (the source wordlists don't mark it consistently). Conventions in `hackingrongo/data/phoneme_inventory.py`.

---

## For Data Scientists

**Zone A** trains a convolutional autoencoder on rasterised glyph images with zero labels, then clusters embeddings (UMAP + HDBSCAN) and scores against Barthel's 1958 taxonomy — which the model never saw. On the corrected run this scored **"divergent"** (772 clusters, ARI 0.014–0.060), i.e. the clusters do *not* recover Barthel taxonomy well. The 2D SVG facsimiles are limited inputs; the new 3D-relief crops are a real shot at better signal.

**Zone B** applies three sensitivity scenarios to every finding; a result is reported as robust only if it holds within 10% across all three. The contact-partition G² test is reported **after multiple-comparison correction** (Bonferroni/BH), which collapses 20 uncorrected hits to 1.

**Zone C** runs Metropolis-Hastings with Geweke Z and, for multi-chain runs, Gelman-Rubin R-hat. The corrected run **converged** (R-hat 0.9994 over 4 chains) — but convergence means the chains mixed, not that the decipherment is valid; the layer is a hypothesis-generation engine, not a source of endorsed assignments. Incremental delta scoring reduces per-iteration cost from O(N) to O(k).

**Quantum analysis** (`scripts/measure_pgood.py`, `scripts/run_qubo_decipherment.py`) computes p_good (fraction of random assignments scoring above threshold) to derive Grover oracle-call estimates and compare classical vs quantum search complexity, and reformulates sign→phoneme assignment as a QUBO solved by simulated annealing (D-Wave's `neal` library, on CPU — not a QPU). All search layers draw from the same 50-syllable inventory, so complexities are measured over the same space; p_good sampling is verified bit-identical between its vectorised and reference scorers.

---

## For Quantum Researchers

**p_good measurement** samples random sign→phoneme assignments and measures what fraction score above threshold under the Rapa Nui LM, yielding a Grover oracle-call estimate. Runs in minutes on CPU. **Caveat:** at τ=0.99 only ~1 sample in 2,000 is "good" (n_good = 1), so the estimate carries large uncertainty, and the Grover √-advantage is over *random* search — MCMC is not random search.

**QUBO key search** reformulates sign→phoneme assignment as a QUBO. **The actual runs used `--solver neal` — D-Wave's *simulated-annealing library*, which runs on CPU (no quantum hardware, no account).** `--solver dwave` / `--solver hybrid` would submit to a real D-Wave Advantage QPU, but those paths have **not** been exercised — there is no D-Wave QPU run on record. So, to be clear: **all of this project's real quantum-hardware runs are on IBM** (`ibm_marrakesh`); the QUBO layer is a classical annealer. (Note: the last QUBO run also collapsed to a degenerate all-vowel assignment — a known bug to fix before relying on it.)

**Hardware runs** (Bernstein–Vazirani and Simon on `ibm_marrakesh`, 156 qubits; job IDs in `RESULTS.md`) are hardware-executed **demonstrations, not discoveries** — the oracles encode structure first extracted classically, so no quantum advantage is claimed. BV recovers an affine structure that is a *tautological artifact* of the rank encoding; Simon recovers diachronic key-change periods but shows **no separation** at the available n=2,4; QAOA, on a common per-token scale, improves on MCMC by a negligible +0.543. The attack map, evidence tiers, and noise/oracle caveats are in `QUANTUM_ATTACKS.md`.

To our knowledge this is the first empirical quantum-complexity characterisation of the rongorongo decipherment problem — measuring how hard it is and what that implies about the script's structure — not a claim to solve it.

---

## Differences from de Souza's Rongopy Project (2023)

Jonas Gregorio de Souza's `rongopy` (2023, GPL-3.0) is the closest prior computational baseline; his concurrent statistical reassessment of texts I, Gv, T (de Souza 2025, *DSH* 40(4)) is engaged in preprint §3.7.

| Aspect | de Souza (2023) | This project |
|:-------|:----------------|:-------------|
| **Temporal model** | Flat corpus | Diachronic: Tablet D pre-contact anchor (Ferrara 2024); B/C/O/Q post-contact; A excluded |
| **Robustness testing** | None | Three sensitivity scenarios; multiple-comparison correction on G² |
| **Visual embeddings** | None | Convolutional autoencoder; unsupervised UMAP + HDBSCAN (scored against Barthel; currently divergent) |
| **3D geometry** | None | Decode INSCRIBE Nexus meshes → relief render → corpus-labelled glyph crops |
| **Compound detection** | None | Multi-method cross-validated detector (231 candidates) |
| **Parallel passages** | In-memory only | Algorithmic cross-reference; permutation test; diachronic variant analysis |
| **Astronomical analysis** | None | 5-test candidate detector; Mamari calendar anchor; Dietrich correspondence |
| **Sequence completion** | None | N-gram completion; damaged-glyph reconstruction (**experimental**, quarantined from IC) |
| **Quantum analysis** | None | p_good; QUBO (D-Wave `neal` simulated annealing, on CPU); BV + Simon + QAOA on **IBM** hardware (`ibm_marrakesh`) |
| **Results format** | Untyped CSV | Typed dataclasses; CSV; JSON; HTML |

---

## Data provenance & governance

Two invariants (full detail in `DATA_SOURCES.md`):

1. **3D-derived imagery is kept separate from 2D facsimiles.** The 3D-relief crops (`data/glyphs/glyph_crops/`, `source_quality: "3d_relief_highdef"`) are a distinct, clean, corpus-labelled set — never silently merged into the SVG training corpus. The dataset loader tracks per-image provenance (`is_3d_crop`) so the 3D set enters training as an explicit, separately-weightable source.
2. **Restoration/reconstruction is experimental and quarantined.** Predicted reconstructions of damaged/illegible (`?`) glyphs are written only to `outputs/`, labelled hypotheses, and **never** fed into the IC / entropy / contact-partition analyses or written back to `data/corpus/`. Those analyses read only the raw transcription.

---

## Limitations and Honest Caveats

Glyphs analysed are from wooden objects and reimiro ornaments; cave petroglyphs are not included. The IC signal is evidence of structure, not proof of language — and is confounded by inventory size (above). Zone C has converged as a sampler but has **not** produced a validated decipherment hypothesis; no phoneme assignment is endorsed. The Barthel family labels are an arithmetic function of the sign code, not Barthel's iconographic taxonomy. Auto-extracted Barthel reference imagery (`extract_barthel_glyphs.py`) is noisy (OCR mislabels, page-title contamination); the clean reference set is being built from the 3D-relief crops instead.

Of the 15,273 corpus tokens, 677 have no single resolvable Barthel base sign: 401 are illegible (`?`) in the source transcription, 276 are compound/ligature tokens (components preserved in `barthel_components`). The phoneme search space is the canonical 50-syllable inventory; the glottal stop is unrepresented and /ŋ/ is written `g`. The corpus follows kohaumotu.org's encoding of Barthel (1958); cross-validation against newer critical transcriptions (the INSCRIBE project's) is an open task beyond Tablet D.

The authoritative results live on the AzureML instance, not this local checkout; the local `outputs/` may be stale.

---

## For more information
- **[`docs/preprint.md`](docs/preprint.md)** — the full reproducible report (per-claim source files, evidence tiers, substantial limitations).
- **`DATA_SOURCES.md`** — provenance, references, and the data-governance policy.
- **`QUICKSTART.md`** — installation and usage · **`docs/CLI.md`** — command reference.
- **`QUANTUM_ATTACKS.md`** — quantum attack map, evidence tiers, noise/oracle caveats.
