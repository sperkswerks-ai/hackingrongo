# scripts/

This directory is the **analysis layer** — scripts that `pipeline.py` can call,
plus supporting tools used interactively.  Two sub-directories hold scripts that
are out of scope for the reproducible pipeline:

| Sub-directory | Role |
|---|---|
| `scripts/` *(this level)* | Ring 1 and Ring 2 — called or controlled by `pipeline.py` |
| `scripts/tooling/` | **TOOLING** — one-time data prep; run once before the pipeline |
| `scripts/exploratory/` | **EXPLORATORY** — speculative / tangential; not part of the reproducible analysis |

Run the pipeline with `python -m hackingrongo.pipeline --ring 1` (default) for
the classical core, or `--ring 2` to add ML and quantum steps.

**CLI options:** per-flag documentation for the pipeline and every script lives
in [`../docs/CLI.md`](../docs/CLI.md). Each script's `--help` is authoritative.

---

## Ring 1 — classical core

No ML training, no quantum circuits.  Produces the reproducible statistical
cryptanalysis that underpins the DEF CON presentation.

| Script | Called by pipeline.py | What it does |
|---|---|---|
| `run_decipherment.py` | ✓ step 5 | Zone C MCMC + beam-search decipherment |
| `cross_reference_parallels.py` | ✓ step 4e | Algorithmic parallel passage cross-reference |
| `generate_pozdniakov_report.py` | ✓ step 4n | Pozdniakov (1996/2011) paradigmatic analysis + HTML |
| `mine_diachronic_substitutions.py` | ✓ step 4s | Pre↔post-contact substitution mining, corroborated against the contact partition |
| `run_freq_match.py` | ✓ step 4l | Zipf α, Spearman ρ, χ² fit vs. each language model |
| `segment_morphemes.py` | ✓ step 4m | Zellig Harris successor-entropy morpheme segmentation |
| `run_zone_b.py` | — | Zone B classical analysis runner (interactive use) |
| `reading_order_v2.py` | — | Boustrophedon / spiral reading-order heuristics |
| `reading_order_tests.py` | — | Reading-order hypothesis unit tests |
| `align_mamari_calendar.py` | — | Mamari lunar calendar glyph alignment |
| `validate_glosses_calendar.py` | — | Gloss-against-calendar validation |
| `dating_priority.py` | — | Priority-weighted corpus dating scenarios |
| `diagnose_anchor_conflicts.py` | — | Detect conflicting positional anchors in corpus |
| `partition_compounds.py` | — | Recurrence partition of canonical compound codes (structural vs singleton; non-destructive) |
| `find_deity_names.py` | — | Lexical search for deity-name candidates |
| `gloss_hypotheses.py` | — | Generate and rank phonetic gloss hypotheses |
| `refine_assignments.py` | — | Post-MCMC sign-phoneme assignment refinement |
| `transform_parallels.py` | — | Apply transformations to parallel passage variant sets |
| `test_sign_600_taxogram_hypothesis.py` | — | Test the sign-600 taxogram phoneme hypothesis |
| `bundle_reports.py` | — | Bundle individual analysis HTML reports into one page |
| `generate_final_report.py` | — | Consolidated final analysis report |
| `generate_holy_grail_report.py` | — | Holy-grail hypothesis report (translation attempt) |
| `visualize_reading_direction.py` | — | Render reading-direction overlays on tablet images |

---

## Ring 2 — ML, 3-D, and quantum extensions

Builds on Ring 1.  Requires PyTorch, Qiskit, and optionally GPU / IBM Quantum
credentials.  Run with `python -m hackingrongo.pipeline --ring 2`.

| Script | Called by pipeline.py | What it does |
|---|---|---|
| `train_autoencoder.py` | ✓ step 2 | Zone A convolutional autoencoder training |
| `analyze_embeddings.py` | ✓ step 3 | UMAP projection, HDBSCAN clustering, divergence report |
| `run_simon_decipherment.py` | ✓ step 4i_simon | Simon's algorithm on diachronic key-change passages |
| `run_bv_ic_analysis.py` | ✓ step 4i_bv | Bernstein-Vazirani on IC distribution |
| `run_qubo_decipherment.py` | ✓ step 4j | QUBO quantum annealing key search |
| `run_qaoa_decipherment.py` | ✓ step 4q | QAOA hybrid decipherment |
| `run_network_centrality.py` | ✓ step 4r | Bigram PMI network centrality + quantum walk PageRank |
| `run_qksvm_parallels.py` | ✓ step 4p | Projected QK-SVM soft parallel-passage detection |
| `measure_pgood.py` | ✓ step 4i | Quantum hardness (p_good) analysis |
| `build_grover_oracle.py` | — | Grover oracle construction for IC sensitivity search |
| `run_quantum_sensitivity.py` | — | Grover-amplified entropy-sensitivity search |
| `qubo_mcmc_loop.py` | — | Interactive QUBO + MCMC warm-start loop |
| `train_fusion.py` | — | Zone C fusion-layer training (standalone) |
| `train_sequence_model.py` | — | Train sign-sequence transformer model |
| `train_sequential_embeddings.py` | — | Train sequential embedding model |
| `complete_sequence.py` | — | Sequence-model completion / beam search |
| `compound_compositionality.py` | — | Compound glyph compositionality scoring |
| `compare_top_hypotheses.py` | — | Side-by-side comparison of top decipherment hypotheses |
| `run_self_training.py` | — | Semi-supervised self-training loop |
| `redteam_agent.py` | — | Adversarial red-team agent for hypothesis stress-testing |

---

## scripts/tooling/ — TOOLING (one-time data preparation)

Run these **once** before the pipeline to fetch raw data and build the corpus.
They are **not** executed as part of any pipeline ring.

| Script | What it does |
|---|---|
| `fetch_abvd_corpus.py` | Download vocabulary forms from the ABVD (Polynesian comparanda) |
| `fetch_hawaiian_corpus.py` | Download Hawaiian language corpus from public sources |
| `build_corpus.py` | Enrich per-tablet JSON with Horley codes and cluster labels |
| `build_language_models.py` | Build and serialise Polynesian n-gram language models |
| `link_svg_to_corpus.py` | Link SVG glyph files to corpus JSON entries by position |
| `audit_image_mapping.py` | Audit SVG → corpus mapping; report unmatched glyphs |
| `extract_barthel_glyphs.py` | Extract individual glyphs from Barthel atlas scans |
| `render_tablet_views.py` | Render synthetic 3-D tablet views for each side |
| `scrape_glyphs.py` | Scrape glyph images from online rongorongo databases |
| `parse_ids.py` | Parse IDS (Ideographic Description Sequences) font data |
| `parse_tregear.py` | Parse Tregear's (1891) rongorongo transcription |

---

## scripts/exploratory/ — EXPLORATORY (speculative / tangential)

Cross-script and 3-D analyses that are not part of the reproducible pipeline.
Run interactively when needed.

| Script | What it does |
|---|---|
| `cross_script_similarity.py` | Test the Hevesy hypothesis: rongorongo ↔ Indus Valley script overlap |
| `fetch_indus_glyphs.py` | Download Indus Valley script glyph images |
| `render_indus_glyphs.py` | Render and tile Indus Valley glyph images for comparison |
| `render_linearb_glyphs.py` | Render Linear B glyph images for comparative display |
| `hsp_analysis.py` | Hierarchical substitution pattern analysis |
| `segment_3d_glyphs.py` | Segment rendered 3-D tablet views into per-glyph crops |
| `reconstruct_glyph.py` | Reconstruct damaged glyphs from partial outlines |
| `reconstruct_tablet_d.py` | Tablet-D–specific reconstruction from fragmentary data |
| `inpaint_damaged_glyphs.py` | Inpaint damaged glyph regions using OpenCV |
| `nxz_to_ply_converter.py` | Convert NXZ 3-D scan files to PLY format |
