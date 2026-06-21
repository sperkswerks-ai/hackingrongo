# Distributional, Diachronic, and Quantum Reconnaissance of the Rongorongo Script: A Reproducible Report on Methods and Their Current Limits

**Preprint — version of 2026-06-18.** Not peer reviewed.

---

## Abstract

Rongorongo, the undeciphered script of Rapa Nui, is here treated as a
ciphertext-only object: we ask what can be inferred about its structure from the
distribution of its signs alone, without assuming any phonetic values. We report
a reproducible six-layer pipeline over a corpus of 15,273 glyphs (25 tablets,
Barthel 1958 catalog) and characterise its results at three explicit evidential
tiers — robust, suggestive, and null/exploratory. The central statistical
observation is not a difference in index of coincidence (IC) between pre- and
post-contact strata but its confound: the strata differ in sign-inventory
size by up to five-fold, and inventory-normalised IC reverses direction
depending only on how undated tablets are assigned. A contact-partition test of
227 signs yields, after multiple-comparison correction, exactly one sign with a
defensible frequency shift. An attempted replication of Pozdniakov's
paradigmatic sign-classes recovers none of 15 reference classes (F1 = 0.000),
with structurally suggestive near-misses. We additionally report quantum-hardware
runs (Bernstein–Vazirani, Simon, QAOA on `ibm_marrakesh`) as algorithmic
demonstrations, not as evidence of quantum advantage; in each case we state the
artifact or absent separation that prevents a stronger claim. The decipherment
layer has not converged, and we present no phoneme assignment as a result; we
position it instead as a hypothesis-generation engine as it proposes ranked,
falsifiable sign→sound candidates for epigraphic scrutiny, and its value lies in
narrowing and structuring the search space, not in any assignment it currently
outputs. We claim a method, a reproducible search-and-falsification framework,
and an honest accounting of what it does and does not show; we do not claim a
decipherment, a translation, or quantum advantage.

<!-- Sources read for Abstract: DATA_SOURCES.md; outputs/sensitivity_analysis.json; outputs/contact_partition.json; outputs/analysis/pozdniakov_paradigmatic.json; outputs/decipherment/mcmc_diagnostics.json; RESULTS.md -->

---

## 1. Introduction

Rongorongo is an undeciphered sign system attested on roughly two dozen wooden
objects from Rapa Nui (Easter Island). Despite a century and a half of study, no
proposed decipherment commands consensus, and even basic parameters whether
the script is logographic, syllabic, or mixed; how many distinct signs it
contains; the direction and unit of reading remain contested. This situation
is structurally similar to a ciphertext-only cryptanalytic problem: we possess
the encoded surface but neither the plaintext language in usable quantity nor a
crib of known correspondences.

We adopt that framing deliberately. Rather than proposing readings, we treat the
corpus as an object of statistical reconnaissance and ask which structural
properties are stable under perturbation of our analytic choices, and which are
artifacts of those choices. The contribution of this report is therefore not a
new claim about what rongorongo says, but a reproducible pipeline and a
disciplined separation of its outputs into what survives scrutiny, what is
merely suggestive, and what fails to replicate.

Prior computational and epigraphic work frames everything that follows. Barthel
(1958) established the sign catalog and numbering still in use. Boris
Kudryavtsev, with collaborators, originated the method of parallel-passage
analysis, identifying repeated sign sequences across tablets which remains
the single most productive structural tool in the field. Systematic
parallel-passage and palaeographic work was extended by Fischer (1997) and,
comprehensively, by Horley (2021). Pozdniakov (e.g. 1996, 2011) developed the
paradigmatic analysis of sign substitutions that we attempt to replicate in
§3.5. Ferrara et al. (2024) provide the radiocarbon dating of the Échancrée
tablet (Tablet D) that underpins our pre-/post-contact stratification.

What is new here is methodological: an end-to-end, seed-controlled pipeline that
couples classical statistics, network analysis, a sequence model, and quantum
algorithms, and that reports its null and artifactual results with the same
prominence as its positive ones.

<!-- Sources read for §1: DATA_SOURCES.md (references); no quantitative claims in this section -->

---

## 2. Corpus and methods

### 2.1 Corpus provenance

The primary corpus (`data/corpus/`) comprises **15,273 glyphs across 25 tablets
(A–Y)**, derived from Barthel's (1958) catalog and numbering. Token counts vary
slightly with tokenisation and tablet inclusion: the quantum IC analysis, which
excludes one tablet, counts 15,271 tokens across 24 tablets. We report the
15,273/25 figures as canonical and flag tokenisation-dependent counts where they
arise. An alternative transcription of Tablet D from Ferrara et al. (2022) is
retained for sensitivity checks.

### 2.2 Sign normalisation and inventory size

The inventory size is itself a contested quantity, and we are explicit about it
because it confounds later results. The raw CEIPP/Barthel transcription contains
2,097 distinct variant codes; letter-suffixed allographs (`022bfy`),
uppercase orientation markers (`001V`), compound connectors (`009:005`), and
uncertainty marks (`?`, `!`). Parsing and arbitrating these against the sign
catalog recovered 563 previously unidentified tokens and **collapsed 1,330
spurious sign-type fragments to 639 base signs**. Allograph canonicalisation
used in the network layer yields 824 canonical signs; the
frequency-restricted core used for distributional classification (frequency ≥ 5)
contains 297 signs.

We therefore emphasise: the script has on the order of 639 base signs
(Barthel's full catalog), not the ~120 sometimes cited for a hypothetical core
syllabary.[^signcount] Analyses that operate on a frequency core (297 signs) do
so as an explicit restriction, not because the script is small.

[^signcount]: An internal project note rounded this figure to ~632; the
on-disk normalisation record (`DATA_SOURCES.md`) gives 639 base signs (from a
1,330→639 collapse), and `data/catalog/horley_encoding.json` contains 641
Barthel→Horley entries. We use 639 throughout and flag the discrepancy here.

### 2.3 Stratification

Diachronic analyses split the corpus into a **pre-contact** stratum (Tablet D,
anchored by the Ferrara et al. 2024 radiocarbon date) and a **post-contact**
stratum (Tablets B, C, O, Q). Tablet A is excluded. Critically, **19 of the 25
tablets are undated** ("unknown" cluster); these are assigned to strata under
three competing scenarios (§3.1), and our undated-tablet classifier
(`dating_priority.json`) reports per-tablet pre/post probabilities that are, for
most tablets, close to 0.5 (e.g. Tablet K: p_pre = 0.446). This thinness of
dated material is the single largest limitation on every diachronic claim below.

### 2.4 The six-layer pipeline

The pipeline (orchestrated by `hackingrongo.pipeline`) comprises: (A) corpus
normalisation and embedding; (B) classical statistics — IC, entropy, Zipf,
network centrality, distributional sign roles; (C) parallel-passage and
paradigmatic analysis; (D) a sequence/decipherment layer (MCMC over
sign→phoneme maps with language-model scoring); (E) quantum algorithms (BV,
Simon, QAOA, Grover-hardness); and (F) reporting. All steps are seed-controlled
(default seed 20260606) and emit machine-readable artifacts to `outputs/` and
`quantum_results/`.

<!-- Sources read for §2: DATA_SOURCES.md (15,273 glyphs/25 tablets; 1,330→639 base signs; 563 recovered; Ferrara 2022/2024); outputs/network/centrality_report.json (2097 raw nodes); outputs/network/sign_fingerprint.json (824 canonical, 297 core); quantum_results/bv/n7_ic_analysis_*.json (15,271 tokens/24 tablets/vocab 2097); outputs/dating_priority.json (19 undated; Tablet K p_pre 0.446); data/catalog/horley_encoding.json (641 entries) -->

---

## 3. Classical results

### 3.1 Index of coincidence: the finding is the confound — *Tier 2 (suggestive)*

The index of coincidence measures the probability that two randomly drawn tokens
are identical; for the rongorongo sign distribution the random baseline (expected
IC under uniform sign use over the shared inventory) is 0.002849. Observed IC
is well above this in every stratum, confirming that sign usage is highly
structured rather than uniform; the inventory-normalised quantity IC×k ranges
1.8–8.0 (pre-contact) and 5.2–7.2 (post-contact), against 1.0 for random use and
1.69 for an English reference.

The widely-quoted observation, however, is a directional one — that raw IC is
higher in the pre-contact stratum than the post-contact stratum and here the
honest result is that the difference is confounded by inventory size, and that
confound is the headline, not a footnote. Across our three stratification
scenarios the raw difference is real in sign but unstable in magnitude:

| Scenario | pre: types / tokens | post: types / tokens | IC_pre | IC_post | Δ (raw) | CIs overlap? |
|---|---|---|---|---|---|---|
| Conservative (undated→late) | 63 / 187 | 348 / 9,440 | 0.02858 | 0.02078 | 0.00780 | no |
| Optimistic (distributed) | 297 / 5,311 | 261 / 4,316 | 0.02694 | 0.01982 | 0.00712 | no |
| Probabilistic (weighted) | 177 / 1,563 | 334 / 8,064 | 0.02393 | 0.02034 | 0.00359 | no |

The strata being compared do not have comparable inventories: in the
conservative scenario the pre-contact stratum has 63 sign types against the
post-contact stratum's 348, a 5.5-fold difference, and only 187 tokens. Because
IC is mechanically inflated by a smaller inventory and a smaller sample, a raw
IC_pre > IC_post comparison is not a like-for-like measurement of
"redundancy" — it is in substantial part a measurement of inventory size.

The decisive evidence that the confound dominates: under inventory-normalised IC
(IC×k), the direction reverses in some scenarios — the post-contact stratum
shows *higher* normalised IC than pre-contact when undated tablets are treated
conservatively. The pipeline's own robustness flag records this:
`robust: false`, with a relative variation across scenarios of 53.94%. We
therefore report the IC result as: *both strata are structured well above
chance; the sign, magnitude, and even direction of any pre/post difference
depend more on undated-tablet assignment and inventory size than on any intrinsic
diachronic property.* Any reader who has seen "IC drops after contact" quoted as
a finding should read it as an inventory-size artifact until dated material is
sufficient to control for it.

For completeness, the sign-frequency distribution follows a power law with
α = 1.300 and R² = 0.926. The fit is good, but α exceeds the canonical
natural-language range [0.9, 1.1]: sign usage is more top-heavy than typical
language corpora (though some logographic scripts also show α > 1).

<!-- Sources read for §3.1: outputs/sensitivity_analysis.json (ic_random_shared 0.002849; all three scenarios' ic/n_types/n_tokens/CIs; deltas 0.007795/0.007117/0.003590; robust false; relative_variation_pct 53.94; all_ci_non_overlapping true); outputs/analysis/entropy_report.html (IC×k 1.8–8.0 pre / 5.2–7.2 post; English 1.69; normalized-IC reversal sentence; Zipf α=1.300, R²=0.926) -->

### 3.2 Reading direction (boustrophedon) — *Tier 3 (null / marginal)*

Rongorongo is written in reverse boustrophedon. If odd and even lines differed
systematically in sign statistics, that would bear on the reading unit. We find
no such difference: IC_odd = 0.01933 (95% CI 0.01852–0.02053) versus
IC_even = 0.02130 (95% CI 0.02044–0.02258). The confidence intervals overlap,
with an overlap fraction of 4.2% — a marginal, sub-threshold trend toward higher
even-line IC, consistent with noise. We report this as a null.

<!-- Sources read for §3.2: outputs/analysis/boustrophedon_ic.json (ic_odd 0.019327, ic_even 0.021302, CIs, overlap_fraction 0.0421, marginal_overlap true) -->

### 3.3 Parallel passages and reconciliation with Horley (146) — *Tier 1 (robust, descriptive)*

Parallel-passage analysis, Kudryavtsev's original contribution to the field,
is the most reliable structural tool available because it depends on observed
sign-sequence repetition rather than on any interpretive assumption. We
re-derived parallels algorithmically (index-based search with Levenshtein
distance ≤ 1, length tolerance 2) and recovered 13 multi-tablet parallel
passages (designated P001–P013), spanning from two-tablet pairs (e.g. P003
across B and O) to the corpus-wide P009 attested 356 times across 18 tablets.

**Explicit reconciliation with the published literature.** Our count of 13 is
far smaller than Horley's (2021) catalog of 146 passage groups, and the
difference is definitional, not a failure of recall. Of Horley's 146
passages, our index-level search found token matches for 97
(`_discovery_stats.passages_with_matches = 97`). We then retained only those that
are (i) repeated across two or more tablets and (ii) collapse to a single
unique canonical sequence under our normalisation, yielding 13. Horley's
146 include single-tablet internal repetitions and finer palaeographic variants
that our multi-tablet, unique-sequence filter deliberately excludes. We therefore
do not claim to reproduce Horley's full catalog; we recover the multi-tablet
backbone of it. The 49 Horley passages with no token-level match under our
threshold (146 − 97) are a known recall gap attributable to our Levenshtein-1
strictness and normalisation, and should be treated as a limitation of our
automated recovery rather than as evidence against those passages. A reviewer
comparing counts should compare like filters: 97 passages matched at the token
level, 13 retained as multi-tablet unique sequences, against Horley's 146
under his broader inclusion criteria.

<!-- Sources read for §3.3: data/parallels/parallel_variants_auto.json (13 passages P001–P013; per-passage attestation/tablet counts; _method Levenshtein threshold 1, length_tolerance 2; _discovery_stats: multi_tablet_passages_found 13, total_horley_passages 146, passages_with_matches 97); DATA_SOURCES.md (Horley 2021; 146 passage groups) -->

### 3.4 Contact partition (G²): one sign survives correction — *Tier 2 (suggestive, now constrained)*

We tested whether individual signs shift in relative frequency between strata
using the G² (log-likelihood-ratio) statistic, across 227 signs attested
sufficiently in both strata. Uncorrected, 20 signs reach p < 0.05 and 6 reach
p < 0.01. But 227 simultaneous tests at α = 0.05 are expected to yield ≈ **11.4
false positives** by chance alone, so the uncorrected count of 20 is close to
the noise floor and cannot be reported as 20 findings.

We therefore applied multiple-comparison correction (G² → p via χ² with 1 df).
**Under both Bonferroni (α/m = 0.05/227 = 2.2×10⁻⁴) and Benjamini–Hochberg
(FDR = 0.05), exactly one sign survives:** the compound "200 9"
(G² = 20.97, p = 4.66×10⁻⁶, pre-biased, attested in both strata). Every other
sign — including the next-ranked "52" (G² = 11.82) and "7" (G² = 10.43) — falls
below significance once the family-wise error is controlled, and several
high-G² candidates ("270 61", "66", "5", "290") are attested in only one
stratum, making their frequency "shift" a presence/absence accident on small
counts rather than a measured change.

The defensible statement is thus narrow: one compound sign ("200 9") shows a
frequency difference between strata that survives correction; the broader
"signs change across contact" pattern does not survive multiple-comparison
control and should not be cited as a multi-sign result.

<!-- Sources read for §3.4: outputs/contact_partition.json (227 entries; per-sign g2/bias/seen_in_both; top signs "200 9" g2=20.971, "52" 11.817, "7" 10.434, "66" 10.188). Correction computed in-analysis: χ²(1 df) p-values; Bonferroni 0.05/227 and BH FDR 0.05 each leave 1 survivor ("200 9", p=4.66e-6); uncorrected p<.05 = 20, p<.01 = 6; expected FP at .05 = 11.4 -->

### 3.5 Paradigmatic classes (Pozdniakov replication): a clean null — *Tier 3 (null with near-misses)*

We attempted to replicate Pozdniakov's paradigmatic sign-classes (sets of signs
that substitute for one another in otherwise-identical contexts) by mining
substitutable pairs from the parallel passages (minimum 3 attestations, ≥ 2
tablets; the three corpus-saturating mega-passages P009/P010/P012 were excluded
to avoid degenerate all-to-all matches). The miner found 47 candidate
substitution pairs forming 7 equivalence classes.

Measured against 15 reference classes drawn from Pozdniakov's published
paradigms, the replication is a **null result: recall = 0.000, precision = 0.000,
F1 = 0.000.** None of the 15 reference classes was matched (Jaccard ≥ a class-match
threshold) by any recovered class. We report this as a null, not as "partial
replication."

The result is, however, a near-miss null rather than a random one, which is
worth recording for future work. Several reference classes are partially
approximated: the reference class {070, 071, 072, 073} is best matched by a
recovered class containing {071, 072} (Jaccard 0.5); {022, 023} by {022, 430y}
(Jaccard 0.333); {300, 301} by a class containing 300 (Jaccard 0.167). The
recovered classes capture fragments of the reference paradigms but neither their
full membership nor their boundaries. The most likely explanations are (i) our
multi-tablet substitution evidence is too sparse to reconstruct full classes
from a handful of attestations, and (ii) Pozdniakov's paradigms incorporate
palaeographic and positional judgments that a purely distributional miner does
not encode. Either way, the honest summary is: we do not recover Pozdniakov's
classes; we recover suggestive fragments of a minority of them.

A separate set of paradigmatic hypothesis tests (`pozdniakov_hypothesis_tests.json`,
sample size 88, 1,000 bootstrap/null draws) is directionally consistent but not
significant: the pre-contact rank correlation is ρ = −0.368 (95% CI −0.674 to
−0.163), but the bootstrap comparison against the post-contact distribution gives
p = 0.674 (the strata are not distinguishable on this statistic). A
date-versus-language-model-score correlation of ρ = −0.645 across tablets exists
but rests on the same thin dated sample and should be read as exploratory.

<!-- Sources read for §3.5: outputs/analysis/pozdniakov_paradigmatic.json (n_pairs_found 47; n_classes_found 7; comparison: recall 0.0, precision 0.0, f1 0.0, n_reference_classes 15, n_recovered_classes 7, 0 matched; class_details best_jaccard 0.5 for {070-073}->{071,072}, 0.333 for {022,023}, 0.167 for {300,301}; excluded_passages P009/P010/P012; params min_attestations 3, min_tablets 2); outputs/analysis/pozdniakov_hypothesis_tests.json (sample_size 88; test1 pre_rho -0.36788, pre_ci [-0.674,-0.163], p_post_boot_ge_pre 0.674; test4 spearman_rho_date_score -0.6445) -->

### 3.6 Network centrality and distributional sign roles — *Tier 2/3*

Treating sign bigrams as a weighted graph (pointwise mutual information edges)
yields a corpus graph of 2,097 raw nodes / 1,684 edges (density 3.8×10⁻⁴); on
the canonical inventory the graph has 824 nodes / 2,044 edges. The highest
betweenness-centrality signs are the high-frequency hubs **008, 001, 003, 076**,
which act as connectors between otherwise weakly-linked sign neighbourhoods.

We additionally classified the 297 frequency-core signs by distributional role
(a "service-discovery" analogy: infer a sign's function from its behaviour, not
its form). An initial determinative test keyed on positional entropy and
neighbour diversity returned zero candidates, because those features are
anti-correlated with betweenness under frequency normalisation and positional
entropy saturates near 1.0 for every frequent sign. Replacing it with a
directional adjacency-asymmetry feature (`direction_skew`: a determinative binds
to a class on one side) surfaced 7 taxogram candidates (proclitic
064/243/532/551; postclitic 098/380/604). We stress that 0 of the 7 are
corroborated by an independent signal: none is a compound member, and none
keeps its role across the contact boundary. The diachronic validation is itself
underpowered; `role_stability` = 0.385 over only 13 signs attested in both
strata, with the pre-contact side being Tablet D alone. These are candidates for
further study, not classifier outputs we endorse; we report them so the negative
corroboration is on record.

<!-- Sources read for §3.6: outputs/network/centrality_report.json (corpus_graph n_nodes 2097, n_edges 1684, density 0.000383); outputs/network/sign_fingerprint.json (824-node/2044-edge canonical graph; top-betweenness 008/001/003/076; 297 core; 7 taxogram candidates proclitic 064/243/532/551, postclitic 098/380/604; 0 corroborated; role_stability 0.385 over 13 signs) -->

### 3.7 Relationship to concurrent work: sign 76 as a candidate prominence marker

Sign 76 is among the most-discussed glyphs in the field. Davletshin (2012)
proposed on epigraphic grounds that it functions as a prominence marker,
the particle *ko* preceding personal names, and de Souza (2025), in a
statistical reassessment of texts I, Gv, and T, provides supporting evidence
through collocation analysis. Our results bear on this proposal only
distributionally, and we are careful not to overstate the connection. In our
data, sign 076 is the second-ranked sign by information content (normalised
IC 0.921, behind only sign 001) and one of the four highest-betweenness hubs in
the bigram-PMI network (§3.6). It also appears as the post-contact member of a
recovered diachronic substitution (678 → 076; §4.2).

This is a genuine, if modest, convergence: a frequent particle such as *ko*,
recurring before a varied class of following names, would be expected to present
exactly as a high-frequency, high-centrality bridge sign — which is what our
purely distributional analysis finds, with no knowledge of the *ko* hypothesis.
We are careful about what this does and does not establish. Centrality alone does
not distinguish a grammatical particle from a frequent logogram, and our own role
classifier does not assign 076 a corroborated taxogram role (§3.6); so our
evidence is consistent with the Davletshin/de Souza reading and independently
motivated, but it does not by itself confirm the phonetic value *ko*. We record
agreement at the level of "sign 76 behaves like a high-load grammatical
particle," which the *ko*-prominence-marker hypothesis would predict, and leave
the phonetic identification to the epigraphic and collocational arguments that
can actually adjudicate it.

A second, weaker instance of the same pattern is worth recording with its
caveat attached. Davletshin (2012) also reads sign 532 as *ʻariki* "chief," a
title that, like *ko*, would precede a personal name. In our blind directional
analysis (§3.6), sign 532 is one of the seven proclitic taxogram candidates
(frequency 21, `direction_skew` = +0.44), i.e. it binds to a more diverse class
of *following* than preceding signs; exactly the left-bound shape a prenominal
title would produce. We stress that this is a candidate, not a corroborated
result: sign 532 is among the **zero** of seven taxogram candidates that survive
our own corroboration check (it is not a compound member and its role does not
hold across the contact boundary; §3.6), and at 21 attestations the asymmetry
estimate is noisy. So the 532 / *ʻariki* convergence is suggestive and directly
parallel to the sign-76 case, but it rests on a candidate our internal validation
explicitly declined to confirm; we present it as a hypothesis worth testing
epigraphically, not as independent support.

<!-- Sources read for §3.7: quantum_results/bv/n7_ic_analysis_*.json (sign 076 ic_norm 0.921, rank 1 behind 001); outputs/network/sign_fingerprint.json (076 among top-4 betweenness, not a corroborated taxogram; 532 proclitic taxogram candidate, frequency 21, direction_skew +0.44, corroborated=false [not in compound, not diachronically stable], 0 of 7 candidates corroborated); quantum_results/simon/P012_*.json (678 -> 076 substitution). Davletshin (2012) and de Souza (2025) are literature pointers, not on-disk sources; the convergences are between their published claims and our distributional figures. -->

---

## 4. Quantum methods and results

We ran three quantum algorithms on IBM hardware. We state at the outset that
none of these establishes a quantum advantage for rongorongo analysis; they
are demonstrations that the relevant structures can be encoded and that the
algorithms behave as theory predicts, each accompanied by the specific reason a
stronger claim is unwarranted. All hardware runs used the 156-qubit
`ibm_marrakesh` backend; calibration timestamps were 2026-06-06T21:45:24-05:00
and 2026-06-07T12:09:58-05:00, and job IDs are recorded in `RESULTS.md` and in
each result file's `hardware_provenance` block.

### 4.1 Bernstein–Vazirani: a real recovery of a tautological structure — *Tier 3 (artifact)*

We encoded the top-64-sign IC distribution as a 7-bit Boolean function (domain
size 128) and tested it for linear/affine structure. The function is **exactly
affine** (affine_fraction = 1.0); BV recovered the hidden string s = 64
(bit 6, corresponding to IC-ranked sign 006) in a single quantum query,
against 16,384 queries for classical exhaustive search (or 7 for the classical
basis-vector BV method). The recovery is correct and the hardware returned the
exact bitstring (job `d8ie0vtv8cos73f5fs30`).

The critical caveat, stated wherever BV is mentioned: the affine structure is
a tautological artifact of the rank-based index encoding. The function encodes
"is index i occupied" over IC-rank indices, and that occupancy pattern is affine
by construction, not because the rongorongo IC distribution possesses hidden
linear structure. The result is a valid demonstration that BV runs correctly on
this hardware; it is not evidence of latent algebraic structure in the
script. The "16,384× speedup" is real only against the artifact.

<!-- Sources read for §4.1: quantum_results/bv/n7_ic_analysis_20260607T030750Z_d8ie0vtv8cos73f5fs30.json (n_top_signs 64, n_bits 7, domain_size 128; linearity.affine_fraction 1.0, is_exactly_affine true, wht_max_s 64; quantum_result recovered_s_int 64 bit 6 -> sign 006, shots 1, matches_exact_s true; classical_search n_queries 16384; verdict message: "tautological artefact of the rank-based index encoding"; query_comparison 1 vs 7 vs 16384); RESULTS.md (backend, calibration, job id) -->

### 4.2 Simon's algorithm: periods recovered, but no separation at these n — *Tier 3 (demonstration)*

We framed two holy-grail diachronic substitutions — pre/post sign changes
consistent across many tablets — as hidden-XOR (Simon) problems. For passage
P007 (substitution 010? → 010 at position 3, consistent across 4 tablets) the
problem has n = 4 bits and Simon's algorithm recovered the period s = 1000;
for P012 (678 → 076 at position 1, consistent across 7 tablets) n = 2 and the
recovered period was s = 10. Both recoveries are correct
(`period_matches_observation: true`) and reproduced across two hardware sessions.

Two facts must be stated together. Simon's algorithm offers an exponential
query-complexity separation, O(n) versus the classical O(2^{n/2}),
asymptotically. At the problem sizes actually available here that separation
is not merely small, it is absent: at n = 4, classical O(2^{n/2}) = O(4)
equals quantum O(n) = O(4); at n = 2, O(2) = O(2). The diachronic key-changes in
this corpus are 1–2 bits wide, so the regime where Simon wins does not arise.
The runs demonstrate correct encoding and execution of Simon's algorithm on real
substitution data; they demonstrate no speedup.

<!-- Sources read for §4.2: quantum_results/simon/P007_ADHS_DHS_20260607T030804Z_d8ie13e6983c73ds1je0.json (n 4, s_bits 1000, recovered_s_bits 1000, period_matches true, classical O(4)/quantum O(4); change 010?->010 pos 3, 4 tablets); quantum_results/simon/P012_ABCDEGHINPQSX_BCDEGHIPQS_20260607T030817Z_d8ie16tv8cos73f5fsd0.json (n 2, s_bits 10, recovered, classical O(2)/quantum O(2); change 678->076 pos 1, 7 tablets); plus the two rerun files of 2026-06-07T1731; RESULTS.md -->

### 4.3 QAOA hybrid decipherment: feasibility, and a negligible corrected improvement — *Tier 3 (feasibility)*

We posed a 4-sign × 4-phoneme assignment subproblem as a 16-qubit QUBO and ran
QAOA (reps = 1) on `ibm_marrakesh` across 26 hardware jobs (lead job
`d8iqmh5v8cos73f5uebg`). The circuit transpiled from depth 2 to depth 291 on
hardware — by itself a useful record of how quickly a small QUBO becomes deep on
a real device.

The original hardware result file reported `improvement_over_mcmc = 7687.4`, and
in an earlier draft of this report we flagged that figure as a scale-mismatch
artifact: it subtracted a corpus-total MCMC language-model score (−7698.3) from a
4-sign subproblem score (−10.9), two quantities not on the same scale. We have
since corrected the scorer (the baseline is now computed with the same per-token
scorer as the QAOA assignment, and the non-comparable corpus-total is retained
separately for provenance only) and regenerated the run on the in-process
`statevector` backend with the identical 4×4, reps = 1 configuration
(`statevector_p1_s4x4_corrected.json`, 25 COBYLA evaluations). On a common
per-token scale the corrected comparison is: hybrid LM score = **−11.268** versus
an MCMC baseline of **−11.811**, i.e. **Δ = +0.543** log-probability units.

The corrected figure is therefore a small, positive, and essentially
negligible improvement on a 4-sign toy subproblem — emphatically not the +7687
artifact, and not evidence of quantum advantage. (For transparency: the
separately-stored corpus-total MCMC score in the regenerated file is −6858.4,
which is not comparable to the per-token scores and is retained only to prevent
the original scale-mixing error from recurring.) The regeneration was run on a
simulator, not hardware, because the artifact lived entirely in the classical
scoring step, not in the quantum execution; the hardware-feasibility facts
(16 qubits, depth 291, 26 jobs on `ibm_marrakesh`) come from the original
hardware run and are cited as such. We present no phoneme assignment from either
run as a finding.

<!-- Sources read for §4.3: quantum_results/qaoa/ibmq_p1_s4x4_20260607T173547Z_d8iqmh5v8cos73f5uebg.json (HARDWARE run: n_qubits 16, top_signs 4, top_phonemes 4, reps 1; circuit_stats depth_pre 2 depth_post 291; 26 job_ids on ibm_marrakesh; original artifact improvement_over_mcmc 7687.402959); quantum_results/qaoa/statevector_p1_s4x4_corrected.json (REGENERATED with corrected per-token scorer: backend statevector, 16 qubits, 4×4, reps 1, 25 COBYLA evals; hybrid_lm_score -11.267826, MCMC baseline same scorer -11.811, improvement_over_mcmc 0.54321; mcmc_ranking_total_lm_score -6858.371842 [non-comparable, provenance only]); RESULTS.md -->

### 4.4 Grover-hardness certificate — *Tier 2 (suggestive, heavily caveated)*

The most defensible quantum contribution is classically computed: a
search-hardness estimate for the sign→phoneme keyspace. Sampling 100 random
assignments over 2,097 signs and 114 phonemes, the language-model score
distribution has mean −11.38 (sd 0.26). At an acceptance threshold τ = 0.99 the
fraction of "good" assignments is **p_good = 0.01**, implying a Grover search
cost of 8 oracle calls against 100 for classical random sampling, a
**12.5× ratio**, and 5,000 MCMC iterations correspond to a 625× ratio over
Grover.

The caveats are substantial and we attach them inline. First, **p_good = 0.01 is
estimated from a single "good" sample out of 100** (n_good = 1); the binomial
uncertainty on a 1/100 estimate is enormous, so the 12.5× figure should be read
as order-of-magnitude at best. Second, the speedup is Grover's generic √
advantage over random search, and MCMC is not random search — it is a guided
sampler that already exploits structure, so the 625× MCMC-vs-Grover ratio
compares unlike procedures. Third, the oracle is idealised (a 16-variable
4-sign × 4-phoneme circuit), and the file itself notes that "near-term hardware
constraints may limit practical implementation." The certificate is a useful
framing of how hard the decipherment search is in principle; it is not a
demonstration that a quantum computer would decipher rongorongo faster in
practice.

<!-- Sources read for §4.4: outputs/zone_b/pgood_analysis.json (n_samples 100, n_signs 2097, n_phonemes 114; score mean -11.378 sd 0.261; tau 0.99 -> p_good 0.01, n_good 1, grover_oracle_calls 8, classical_random_calls 100, quantum_speedup_ratio 12.5, mcmc_iterations 5000, mcmc_vs_grover_ratio 625; interpretation note on near-term hardware) -->

---

## 5. Limitations

This section is deliberately the most substantial, because the honesty of the
report depends on it.

The decipherment layer has not converged. The canonical MCMC run
(`outputs/decipherment/mcmc_diagnostics.json`) records `converged: false`, a
Geweke z-statistic of **−12.68** (values near 0 indicate convergence), a single
chain of 80 samples, and an acceptance rate of 0.328 over a 98-sign inventory. A
single chain cannot yield a Gelman–Rubin R-hat, and indeed every
`mcmc_diagnostics.json` on disk reports `gelman_rubin_rhat: null`. An internal
note recalled an R-hat of 0.9994 for a "converged baseline"; that value is
`[NOT FOUND IN OUTPUTS — VERIFY]` no file on disk substantiates it, and we
will not state it as a result. Consequently we present no sign→phoneme
assignment, gloss, or reading as a finding; the `ranking.json` hypotheses
(H0001–H0005) are search outputs over a non-converged sampler, not linguistic
conclusions.

We nonetheless retain the layer deliberately, reframed as a
hypothesis-generation engine rather than a decipherment: its role is to emit
ranked, explicitly falsifiable sign→sound candidates, constrained by calendar
anchors, parallel-passage structure, and language-model plausibility that an
epigrapher can test and, in most cases, reject. Under this framing
non-convergence bounds what may be claimed (no candidate is endorsed) but does
not make the layer worthless: a sampler that systematically narrows a keyspace of
the size estimated in §4.4 and surfaces structured candidates for falsification
is doing useful work even when its posterior has not mixed. We are explicit that
this is a claim about process (a reproducible way to generate and triage
hypotheses), not about product (any particular reading).

**The IC stratification is confounded by inventory size** (§3.1), with the
direction of the effect reversing under normalisation; it is not robust
(relative variation 53.94%).

**The contact-partition signal collapses under correction** (§3.4): one of 227
signs survives Bonferroni/BH.

**The Pozdniakov replication is a null** (§3.5): 0 of 15 classes, F1 = 0.000.

**The quantum results are demonstrations, not advantages**: BV recovers a
tautological encoding artifact (§4.1); Simon shows no separation at n = 2, 4
(§4.2); the QAOA improvement, once the scale-mixing error is corrected, is a
negligible +0.543 log-prob units on a 4-sign toy subproblem (§4.3); the hardness
certificate's p_good rests on one sample (§4.4).

**The diachronic backbone is thin.** With only Tablet D dated to the pre-contact
stratum and 19 of 25 tablets undated, every pre/post comparison IC,
contact-partition, paradigmatic, sign-role stability is underpowered, and
several are sensitive to how undated tablets are assigned.

**Inventory and tokenisation vary across components.** Different layers operate
on 2,097 raw codes, 824 canonical signs, a 297-sign frequency core, or
component-specific inventories (98 in the decipherment MCMC, 114 phonemes in the
hardness analysis). We have tried to state which inventory each result uses;
cross-layer comparisons should account for this.

<!-- Sources read for §5: outputs/decipherment/mcmc_diagnostics.json (converged false, geweke_z -12.6812, n_chains 1, n_samples 80, acceptance_mean 0.3275, sign_inventory_size 98, gelman_rubin_rhat null); grep across outputs confirming all gelman_rubin_rhat null; outputs/decipherment/ranking.json (H0001–H0005 syllabic assignments); outputs/sensitivity_analysis.json (robust false, 53.94%); outputs/contact_partition.json; outputs/analysis/pozdniakov_paradigmatic.json (f1 0.0); outputs/zone_b/pgood_analysis.json -->

---

## 6. Reproducibility

All results derive from a seed-controlled pipeline (default global seed
20260606). The classical and quantum layers are invoked through
`python -m hackingrongo.pipeline` (ring 1 = classical; ring 2 = + ML/quantum),
with per-step selection via `--steps`; the analysis-script index and full CLI are
documented in `scripts/README.md` and `docs/CLI.md`.

Data and results live in fixed locations: the corpus in `data/corpus/`, the sign
catalog in `data/catalog/`, parallel passages in `data/parallels/`, classical
outputs in `outputs/` (notably `sensitivity_analysis.json`,
`contact_partition.json`, `outputs/analysis/`, `outputs/network/`), the
decipherment layer in `outputs/decipherment/`, and quantum-hardware results in
`quantum_results/` with provenance summarised in `RESULTS.md`. Every quantum
result file carries a `hardware_provenance` block (backend, job ID, calibration
timestamp), and quantitative claims in this report are annotated, per section,
with the files they were read from (the HTML comments above). The
multiple-comparison correction in §3.4 is the only statistic computed during
writing rather than read from a file; it derives χ²(1 df) p-values from the
stored G² values and applies Bonferroni and Benjamini–Hochberg at the stated
levels.

Two reproducibility caveats specific to the quantum layer: hardware results are
not bit-for-bit reproducible because backend calibration drifts between
sessions (we record calibration timestamps for this reason), and the original
QAOA hardware result file predates a scoring fix, so its stored
`improvement_over_mcmc` is the scale-mixing artifact discussed in §4.3. The
corrected figure (§4.3) comes from a regenerated `statevector` run
(`statevector_p1_s4x4_corrected.json`) using the fixed per-token scorer; we
retain the original hardware file unmodified for its calibration/job-ID
provenance rather than overwriting it.

<!-- Sources read for §6: scripts/README.md; docs/CLI.md; RESULTS.md; directory structure of outputs/ and quantum_results/ -->

---

## 7. Conclusion and future work

We have described a reproducible reconnaissance pipeline for rongorongo and
reported its current results without inflation. The robust findings are modest
and structural: sign usage is highly non-uniform in every stratum and follows a
steep power law; a multi-tablet parallel-passage backbone of 13 sequences is
recoverable automatically and reconciles cleanly with the multi-tablet subset of
Horley's catalog; and one compound sign shows a frequency shift across the
contact boundary that survives multiple-comparison correction. The widely-cited
"IC falls after contact" pattern is, on our analysis, dominated by an
inventory-size confound. The Pozdniakov paradigm replication is a null. The
quantum runs are clean algorithmic demonstrations with no advantage at these
problem sizes, and the decipherment layer has not converged.

The single most consequential bottleneck is dating. Almost every diachronic
result is underpowered because only one tablet (D) is securely dated to the
pre-contact stratum. Radiocarbon dating of additional tablets would directly
convert several of this report's underpowered or confounded results into testable
ones: an IC comparison between strata of comparable inventory size (resolving
§3.1), a contact-partition test with enough dated tokens to localise more than
one sign (§3.4), a paradigmatic replication with enough cross-stratum
substitutions to reconstruct full classes (§3.5), and a sign-role stability
analysis over more than 13 shared signs (§3.6). Until then, the most honest
description of the project's state is that it has built and validated the
machinery, established what the corpus can and cannot currently support, and
documented its nulls as carefully as its signals.

<!-- Sources read for §7: synthesis of §§3–5; no new quantitative claims -->

---

## References

- Barthel, T. S. (1958). *Grundlagen zur Entzifferung der Osterinselschrift*. Hamburg: Cram, de Gruyter.
- Barthel, T. S. (1960). Rezente Einwirkungen auf das Runenschreiben der Osterinsulaner. *Baessler-Archiv* 8, 255–274.
- Davletshin, A. (2012). Name in the Kohau Rongorongo script (Easter Island). *Journal de la Société des Océanistes* 134(1), 71–85.
- de Souza, J. G. (2023). *rongopy* (computational decipherment toolkit). GitHub: github.com/jgregoriods/rongopy. GPL-3.0.
- de Souza, J. G. (2025). A statistical reassessment of rongorongo texts I, Gv, and T: implications for genre and content. *Digital Scholarship in the Humanities* 40(4), 1126–1142. DOI: 10.1093/llc/fqaf101.
- Ferrara, M., Lastilla, L., Ravanelli, N., & Valério, M. (2022). Modelling the Rongorongo tablets. *Digital Scholarship in the Humanities* 37(2), 497–526. DOI: 10.1093/llc/fqab045.
- Ferrara, S., Tassoni, L., Kromer, B., et al. (2024). The invention of writing on Rapa Nui (Easter Island): new radiocarbon dates on the Rongorongo script. *Scientific Reports* 14, 2794. DOI: 10.1038/s41598-024-53063-7.
- Fischer, S. R. (1997). *Rongorongo: The Easter Island Script*. Oxford: Clarendon Press.
- Horley, P. (2021). *Rongorongo*. Rapa Nui Press.
- Kudryavtsev, B. G. (1949). Письменность острова Пасхи [The writing of Easter Island]. *Sbornik Muzeya Antropologii i Etnografii* (Collection of the Museum of Anthropology and Ethnography) 11, 175–221. (Posthumous; originator of parallel-passage analysis.)
- Orliac, C. (2005). The woody plants of the rongorongo tablets. *Rapa Nui Journal* 19(1), 61–66.
- Pozdniakov, K. (1996). Les bases du déchiffrement de l'écriture de l'île de Pâques. *Journal de la Société des Océanistes* 103, 289–303.
- Pozdniakov, K., & Pozdniakov, I. (2007). Rapanui writing and the Rapanui language. *Forum for Anthropology and Culture* 3, 3–36.

<!-- All references confirmed. Davletshin 2012 (JSO 134(1):71–85); de Souza, J. G. (2023 rongopy; 2025 DSH 40(4):1126–1142, fqaf101); Ferrara et al. 2024 (Sci Rep 14, 2794); Kudryavtsev 1949 (Sbornik MAE 11:175–221). -->
