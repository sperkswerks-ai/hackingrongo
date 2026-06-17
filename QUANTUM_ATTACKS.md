# Quantum Cryptanalysis of Rongorongo — Attack Map

**Venue:** DEF CON 34 Crypto Village  
**Thesis:** Rongorongo's structural properties as an unknown substitution cipher
map onto the exact algebraic weaknesses that quantum algorithms are designed to
exploit.  The attacks below are not analogies in that they use the mathematical
definitions from cryptanalysis (hidden linearity, period-finding, S-box
weakness) applied directly to the corpus statistics.

---

## Canonical Attack Table

Evidence levels:
- **HARDWARE-EXECUTED** — quantum circuit executed on IBM Quantum hardware;
  job IDs in `RESULTS.md`.  In each case the oracle *encodes structure that
  was first extracted classically from the corpus*: the hardware runs verify
  the quantum encoding and the algorithm's query complexity on real qubits.
  They are demonstrations, not discoveries unavailable to classical analysis
  (see "Oracle-construction caveat" below).
- **DEMONSTRATIVE** — correct quantum algorithm, run on simulator or fake
  backend; no quantum advantage claimed over a classical computer

| # | Cipher vulnerability | Rongorongo instance | Classical attack | Quantum algorithm | Quantum complexity | Classical complexity | Hardware result |
|---|---|---|---|---|---|---|---|
| 1 | **Hidden linear structure** (affine IC distribution) | Sign-frequency IC function f: {0,1}⁷ → {0,1} tested for f(x) = s·x ⊕ c. Under the corpus-honest Barthel-bits encoding the result is a **null**: affine fraction ≈ 0.52, best linear approximation agrees on only 64% of inputs — no hidden linear structure. (The earlier IC-rank encoding produced affine f *by construction*; that run is retained as a hardware verification of the BV circuit, not a corpus finding.) | Exhaustive search over all 2⁷ = 128 candidate slopes | Bernstein–Vazirani | **1 oracle query** (when f is affine) | 128 exhaustive / 7 classical BV queries | **HARDWARE-EXECUTED** · ibm\_marrakesh · BV circuit verified in 1 query on the rank encoding · job ID in `RESULTS.md` · `--encoding barthel_bits` yields the null result |
| 2 | **Key reuse / periodic function** (XOR-periodic sign substitution) | Passages P007 (tablets A,D,H,S) and P012 (11 tablets): every pre-contact instance differs from every post-contact instance at the same canonical positions — a fixed bitstring s extracted classically by the Kasiski cross-reference and embedded in the Simon oracle | Reading s off the parallel-passage table: O(n). (The O(2^{n/2}) baseline applies only to black-box oracle access, which is not this situation.) | Simon's algorithm | **O(n) oracle queries** | O(n) with corpus access / O(2^{n/2}) black-box | **HARDWARE-EXECUTED** · ibm\_marrakesh · Simon circuit recovers the embedded period s on P007 and P012, matching the classical extraction · job IDs in `RESULTS.md` |
| 3 | **Weak S-box** (low-complexity sign→phoneme substitution) | ≈120 sign types map to ≈20 phonemes; PMI coupling matrix has low algebraic rank; assignment is NP-hard classically (QUBO) | Simulated annealing / exhaustive QUBO (O(2ᴺ) worst case) | QAOA (Quantum Approximate Optimization) | O(poly(N)) variational | O(2ᴺ) | **DEMONSTRATIVE** · FakeBrisbane simulator · hybrid QAOA+MCMC assignment; no hardware advantage claimed |
| 4 | **Related-key / IC sensitivity** (entropy leaks under key perturbation) | IC varies predictably across pre/post-contact strata and dating scenarios; sign substitutions shift entropy by a measurable Δ | Classical sensitivity analysis (O(N·S) parameter sweeps) | Grover-amplified sensitivity search | O(√(N·S)) | O(N·S) | **DEMONSTRATIVE** · FakeBrisbane / statevector · Grover oracle identifies high-Δ sign pairs; no hardware run |
| 5 | **Structural leakage via high-betweenness signs** (determinative function) | High-betweenness, low-frequency signs in the PMI bigram graph control information flow — consistent with grammatical determinatives invisible to frequency analysis | Classical PageRank + betweenness (O(VE)) | Szegedy quantum walk PageRank | O(1/δ) mixing time speedup | O(VE) | **DEMONSTRATIVE** · numpy/scipy statevector · L₁ divergence between quantum and classical PageRank reported; no hardware run |
| 6 | **Community boundary leakage** (block structure of pre/post-contact strata) | Fiedler value λ₂ of normalized bigram Laplacian encodes the pre/post-contact community boundary; sign of Fiedler vector bisects the corpus | Classical spectral bisection (O(V²) dense eigen) | QPE on normalized Laplacian | O(1/ε) precision in λ₂ | O(V²) | **DEMONSTRATIVE** · Qiskit statevector QPE · 10-qubit circuit (6 pos + 4 QPE); classical λ₂ and QPE estimate compared; no hardware run |
| 7 | **Parallel passage distinguishability** (related plaintext / soft correlation) | Parallel passage pairs share statistical structure detectable as a quantum kernel; QK-SVM separates parallel from non-parallel passage pairs | Classical SVM with RBF kernel | Quantum kernel SVM (QK-SVM) | Potential kernel evaluation speedup | O(n²d) classical kernel | **DEMONSTRATIVE** · statevector · AUC reported on simulated feature map; no hardware run |

---

## Oracle-construction caveat

Both hardware-executed attacks share the standard limitation of quantum
algorithm demonstrations on real data: **the oracle is built by us, from
structure the classical pipeline already extracted**.  BV's query complexity
is measured against an oracle whose truth table we computed classically;
Simon's period s is XOR-composed from the diachronic substitution table
before the circuit is constructed.  A query-complexity separation only
translates into a real-world advantage when the oracle is given (black-box
access), not when constructing the oracle requires reading all the data
classically first.

What the hardware runs *do* establish:

1. The rongorongo structures in question (IC distribution, diachronic
   substitution pattern) map exactly onto the algebraic forms (Boolean
   linearity, XOR-periodicity) that BV and Simon consume — the encodings
   are lossless and the circuits are correct.
2. The circuits execute within coherence on 156-qubit hardware
   (`ibm_marrakesh`) and return the theoretically expected measurement
   statistics, with full calibration provenance.
3. At scale — a hypothetical sign system with thousands of signs and an
   oracle implementable from a compact specification — these are the
   attacks that would apply.

What they do not establish: a quantum speedup over classical analysis of
this corpus.  We state this up front because it is true, and because the
distinction is exactly what the Crypto Village audience should be
calibrating.

---

## Hardware-Executed Results — Detail

### 1 · BV: Testing the IC distribution for hidden linear structure

**Setup.** The index of coincidence (IC) of the top-64 rongorongo signs is
computed over a 7-bit index encoding (2⁷ = 128 domain elements).  The Boolean
function f: {0,1}⁷ → {0,1} maps each sign index to its IC above/below the
corpus median.  Two encodings are available (`--encoding`):

- **ic_rank** — signs indexed by IC rank.  With a median threshold this makes
  f affine *by construction* (f(x) = 1 iff rank(x) < 64; the slope is the top
  rank bit), so a recovered slope is an artefact of the encoding, not a
  corpus property.
- **barthel_bits** — signs indexed by the low 7 bits of their Barthel
  catalogue number, which is independent of the IC value being thresholded.
  Linearity is then a falsifiable corpus property.

**Result under the honest encoding.** `--encoding barthel_bits` yields a
**null result**: affine fraction ≈ 0.52 (chance level), best linear
approximation agrees on only ~64% of inputs.  The IC distribution has **no
hidden linear Boolean structure** over Barthel code bits.  This rules out an
entire class of algebraic models for the sign-frequency distribution — a
genuine, falsifiable corpus measurement.

**Hardware evidence.** The `ibm_marrakesh` run (rank encoding, affine by
construction) verified single-query BV slope recovery on real hardware:
**1 quantum query** vs 128 classical exhaustive vs 7 classical BV queries.
Job ID and calibration timestamp: see `RESULTS.md` and `quantum_results/bv/`.
Shots: 1 (single-shot BV is deterministic for affine f).

**What this means cryptanalytically.** The one-query hardware run is a
circuit verification, not a corpus discovery — the slope it recovers is the
tautology of the rank encoding.  The corpus finding is the null: rongorongo
sign frequencies are not linearly separable over catalogue-code bits, which
constrains what algebraic key structure could explain them.

---

### 2 · Simon: XOR-periodic sign substitution across tablets

**Setup.** Passages P007 (tablets A, D, H, S) and P012 (11 tablets spanning
both strata) show that every pre-contact instance differs from every
post-contact instance at the same canonical positions — a fixed bitstring s.
This satisfies Simon's hidden-period condition f(x) = f(x ⊕ s), and the
oracle is constructed to encode exactly that substitution pattern.

**Classical baseline — stated honestly.** With access to the parallel-passage
table, s is read off directly in O(n): it is the XOR of changed positions
that the Kasiski cross-reference already computed.  The textbook O(2^{n/2})
classical bound applies only to black-box oracle access, which is not the
situation here.  Simon's exponential separation is therefore *illustrated*,
not *exploited*.

**Quantum result.** Simon's algorithm recovers s in O(n) shots.  The circuit
runs the Simon oracle, measures in the Hadamard basis, and solves the
resulting system of linear equations over GF(2).  The recovered period
matches the classically extracted substitution pattern on both passages.

**Hardware evidence.**
- Backend: `ibm_marrakesh`
- Passages: P007 (n = |canonical form|, s = XOR of changed positions),
  P012 (larger passage, more tablets)
- Period s recovered on both passages, matching the classical extraction
- Job IDs and calibration timestamps: see `RESULTS.md` and `quantum_results/simon/`

**What this means cryptanalytically.** The substantive finding is classical
and diachronic: the pre→post-contact sign substitutions are not a full
replacement (new cipher) but a *structured XOR* of the original — the
cryptanalytic signature of a related-key event at the contact boundary.  The
hardware run establishes that this structure is losslessly expressible as a
Simon-class periodic function and that the recovery circuit executes
correctly on 156-qubit hardware.

---

## Demonstrative Results — What Is and Is Not Claimed

The following analyses run correct quantum algorithms on the rongorongo corpus
but **do not claim a quantum advantage** over a classical computer running on
this problem at this scale.  They are included to show that the full quantum
cryptanalysis stack compiles and runs, and that the corpus structure
*would* become a quantum advantage target at scale.

| Analysis | What the simulator shows | What it does not show |
|---|---|---|
| QAOA decipherment | QAOA + MCMC hybrid finds sign→phoneme assignments with lower QUBO energy than random initialization | Quantum speedup over classical annealing at current qubit counts |
| Quantum walk PageRank | Szegedy walk stationary distribution diverges from classical PageRank by L₁ > 0 — latent hub signs differ | Practical speedup; quantum walk simulation is slower than classical PageRank on a laptop |
| Quantum Fiedler / QPE | 10-qubit QPE circuit estimates Fiedler value λ₂ within precision 1/16 of classical numpy eigh | Eigenvalue estimation speedup (not meaningful at 64-node graphs) |
| Grover sensitivity | Oracle marks high-Δ-IC sign pairs; Grover would find one in O(√N) | Hardware run; current qubit overhead exceeds classical cost at corpus scale |
| QK-SVM | Quantum kernel separates parallel/non-parallel passages in simulation | Kernel evaluation advantage (not demonstrated at this feature dimension) |

---

## Quantum noise & reproducibility

Hardware-executed results carry two distinct kinds of error, and only one of
them averages away:

| Source | Behaviour | Mitigation |
|---|---|---|
| **Shot noise** (finite sampling) | Statistical, zero-mean, shrinks as ~1/√N_shots | More shots |
| **Hardware error** (gate ~0.5–1% per 2-qubit gate, readout ~1–3%, decoherence T1/T2 ≈ 100–300 µs) | **Systematic**, compounds with circuit depth; deep circuits decay toward the maximally-mixed distribution. **Does not** average out with more shots | Shallower circuits, dynamical decoupling, readout mitigation (M3), zero-noise extrapolation |

**Calibration drift.** IBM devices are recalibrated roughly daily; between
calibrations the qubit frequencies, gate fidelities, T1/T2, and readout
discriminators drift. The error model is therefore **non-stationary**: the
*same* circuit with the *same* shot count returns different results hours
apart, or before vs. after a recalibration. This is a moving systematic bias,
not zero-mean scatter, so it cannot be eliminated by averaging.

**Consequence for this project.** A real-hardware run **cannot be a
reproducible baseline** — the number is not re-derivable once the device drifts.
Real-IBMQ results are therefore treated as **dated demonstrations**: report the
backend name, calibration timestamp, job ID, and mean ± std over repeats, and
keep them *separate* from the reproducible figures. The reproducible quantum
artifacts come from a **seeded statevector simulator** (shot noise only, no
device error) or a **cached** computed result.

**Why `p_good` is exempt.** The quantum-hardness analysis (`measure_pgood.py`)
is computed **classically** — it samples and LM-scores random assignments on a
CPU, with no qubit, no shots, and no device. It is therefore fully reproducible
and immune to noise and drift. It is a *query-complexity characterization*
(Grover's `π/(4√p_good)` oracle-call count under an idealized oracle), not a
hardware result — see the oracle-construction caveat above and the terminology
note below.

---

## Terminology note

Throughout this project "quantum advantage" means a **proven asymptotic or
empirical improvement over the best known classical algorithm on the same
problem instance, including the cost of constructing the oracle**.
"Hardware-executed" means the circuit ran on real IBM Quantum hardware and
returned the theoretically expected result; because the oracles encode
classically extracted structure (see "Oracle-construction caveat"), the BV
and Simon runs are hardware verifications of the quantum encoding, not
quantum advantages.  "Demonstrative" means the quantum circuit is correct
and produces a valid result on a simulator, but the problem instance is too
small to distinguish quantum from classical performance.  Under these
definitions, **no analysis in this project claims a quantum advantage** —
the contribution is the first empirical quantum-complexity characterisation
of the rongorongo decipherment problem, applied consistently and
conservatively.
