# CLI Reference

Command-line options for the pipeline orchestrator and the analysis scripts.
Each script's `--help` is the **authoritative** source (it is generated from the
code); this page is a curated reference for the options you'll actually reach
for. Flags added in the June 2026 hardening pass are marked **(new)**.

The script *index* (which script maps to which pipeline step) is in
[`../scripts/README.md`](../scripts/README.md).

---

## Pipeline orchestrator

```
python -m hackingrongo.pipeline [options]
```

| Flag | Default | Meaning |
|---|---|---|
| `--ring {1,2,all}` | `1` | Which analysis ring to run. `1` = classical core (no ML/quantum); `2` = core + ML + quantum; `all` = every step. |
| `--steps N[,N...]` | (all in ring) | Run only the listed steps, e.g. `--steps 4j,4n,4r`. Accepts `1, 1b, 2, 3, 4a, 4ar, 4b, 4c, 4d, 4e, 4f, 4g, 4h, 4i, 4i_simon, 4i_bv, 4j, 4k, 4l, 4m, 4n, 4o, 4p, 4q, 4r, 4s, 5, 5b` (bare `4` expands to all 4-series; `5` → `5,5b`). |
| `--skip-training` | off | Skip Step 2 (Zone A autoencoder). Requires `outputs/embeddings_cache.pt` to already exist. |
| `--skip-fusion` | off | Step 5: ignore `fusion_layer.pt` even if present (use sequential-entropy proposal weights). |
| `--no-cache` | off | Ignore `.done` stage checkpoints and re-run every selected step. **Needed when re-running a step that already succeeded** (otherwise it's skipped as cached). |
| `--keep-going` | off | Continue to later steps even when one fails. |
| `--step-timeout SECONDS` | `3600` | Hard per-step wall-clock limit; the step is killed and marked failed if exceeded. `0` disables. **Set ≥ 7200 for a full `--no-cache` run** — Step 5 (decipherment) alone is ~90–120 min. **Never use `0` unattended** (a runaway step would hang forever). |
| `--dry-run` | off | Print the commands that would run, without executing. (Note: dry-run still writes `.done` sentinels.) |
| `--smoke-test` | off | Fast wiring check (Step 2 = 1 epoch / batch 8). |
| `--seed INT` | `20260606` | Global RNG seed threaded into every subprocess. |

Common recipes:

```bash
# Classical core only (deterministic backbone)
python -m hackingrongo.pipeline --ring 1

# Full coherent run, reusing trained embeddings, generous Step-5 headroom
python -m hackingrongo.pipeline --ring 2 --skip-training --no-cache --keep-going --step-timeout 14400

# Re-run just a few steps (must include --no-cache if they already succeeded)
python -m hackingrongo.pipeline --steps 4j,4n,4r --no-cache --keep-going --step-timeout 1800
```

---

## Decipherment (Zone C)

```
python scripts/run_decipherment.py [hydra overrides] [--smoke-test] [--focus-passage=PXXX] [--fusion-checkpoint=PATH]
```

Hydra-configured (reads `conf/config.yaml`). Override config keys directly, e.g.
`seed=123`, `zone_c.mcmc.num_iterations=2000`. Intercepted non-Hydra flags:

| Flag | Meaning |
|---|---|
| `--smoke-test` | Tablet-D-only fast run (anchor signs absent from the reduced corpus are skipped, non-strict). |
| `--focus-passage=PXXX` | Restrict the run to one parallel-passage group from `parallel_variants_auto.json`. |
| `--fusion-checkpoint=PATH` | Use a Zone C fusion checkpoint for MCMC proposal weights. |

Allograph normalization (`get_canonical_id`), the calendar anchors, equivalence
ties (from `pozdniakov_paradigmatic.json` + `diachronic_substitutions.json`), and
in-process bigram-context scoring are all applied automatically.

---

## Quantum search

### QUBO — `run_qubo_decipherment.py` (step 4j)

| Flag | Default | Meaning |
|---|---|---|
| `--solver {auto,hybrid,dwave,neal,tabu}` | `auto` | Annealer backend. `neal` = local simulated annealing (no account). `dwave` = real D-Wave Leap QPU. |
| `--num-reads N` | `1000` | Annealing reads. |
| `--max-signs N` **(new)** | `60` | Cap the QUBO to the top-N most-frequent signs. The full ~2000-sign corpus → ~100k binary variables is intractable; crib signs are always retained. `0` disables the cap. |
| `--max-per-phoneme K` | `5` | Capacity penalty: max signs per phoneme. |
| `--init-from JSON` | — | Warm-start from a `ranking.json`. |
| `--crib SIGN=PHONEME[,…]` | — | Pin sign→phoneme cribs. |
| `--lambda1 / --lambda2 / --bigram-weight` | `10 / 5 / 1` | One-hot / capacity / bigram-coupling penalty weights. |
| `--smoke-test` | off | 10 signs × 10 phonemes, 50 reads. |

### QAOA — `run_qaoa_decipherment.py` (step 4q)

| Flag | Default | Meaning |
|---|---|---|
| `--backend {simulator,statevector,fake_brisbane,ibmq}` | `simulator` | `statevector`/`simulator` → in-process StatevectorSampler (the pipeline uses `statevector`). `fake_brisbane` requires Aer and is unusable here (transpiles to 127 qubits). `ibmq` → real hardware. |
| `--top-signs / --top-phonemes` | `4 / 4` | Subproblem size → `top_signs × top_phonemes` qubits (default 16). Keep small: statevector is 2ⁿ. |
| `--max-iter N` | `200` | COBYLA iterations. |
| `--reps p` | `1` | QAOA depth. |
| `--init-from JSON` | — | MCMC warm-start / hybrid merge. |

### QK-SVM — `run_qksvm_parallels.py` (step 4p)

| Flag | Default | Meaning |
|---|---|---|
| `--backend {simulator,fake_brisbane,ibmq}` | `simulator` | `ibmq` runs the projected quantum kernel on real hardware (`--ibmq-token`, `--ibmq-instance`, `--ibmq-backend`). |
| `--n-negatives N` | `200` | Negative training pairs (balanced against the capped positives). |
| `--inject-as-cribs` | off | Write top-5 soft-parallel candidates as cribs. |

Degenerate corpus-wide passages (P009/P010/P012) are excluded and positives are
capped automatically so the training kernel stays backend-feasible.

---

## Analysis & normalization

### Network centrality — `run_network_centrality.py` (step 4r)

| Flag | Default | Meaning |
|---|---|---|
| `--no-allograph-norm` **(new)** | off (norm ON) | Disable `SignCatalog.get_canonical_id()` normalization and build the PMI graph on the raw, variant-inflated inventory instead of canonical signs. |
| `--min-cofreq N` | `2` | Minimum bigram co-occurrence to retain an edge. |
| `--pmi-floor F` | `0.0` | Minimum PMI to retain an edge. |
| `--quantum-pagerank` / `--quantum-fiedler` | off | Quantum walk PageRank / Fiedler estimation (Qiskit). |

### Diachronic substitution mining — `mine_diachronic_substitutions.py` (step 4s) **(new)**

| Flag | Default | Meaning |
|---|---|---|
| `--include-non-holy-grail` | off | Also emit tie-pairs for single-tablet substitutions (default: holy-grail / multi-tablet only). |
| `--parallels / --contact / --tablets / --catalog-dir` | data paths | Input overrides. |

### Compound partition — `partition_compounds.py` **(new)**

Non-destructive recurrence partition of canonical compound codes.

| Flag | Default | Meaning |
|---|---|---|
| `--min-occ N` | `3` | Minimum occurrences for a compound to be "structural". |
| `--min-tablets N` | `2` | Minimum distinct tablets for "structural". |
| `--corpus-dir / --output-dir` | data/outputs | Path overrides. |

---

## Report bundling

```
python scripts/bundle_reports.py        # collects every outputs/**/*.html + builds reports_bundle.zip
```
