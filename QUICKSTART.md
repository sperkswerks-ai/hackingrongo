
## Quick Start

```bash
git clone https://github.com/violasarah2000/hackingrongo.git
cd hackingrongo
pip install -e .

# Fetch corpora and build language models
python scripts/parse_ids.py --stratify --cache-dir data/cache
python scripts/fetch_abvd_corpus.py --with-cognates --cache-dir data/cache
python scripts/build_language_models.py

# Run full pipeline
python -m hackingrongo.pipeline

# Quantum hardness analysis (no quantum hardware required)
python scripts/measure_pgood.py \
  --corpus-dir data/corpus \
  --lm-dir data/language_models \
  --n-samples 10000 \
  --output outputs/zone_b/pgood_analysis.json

# QUBO key search (classical simulation — no D-Wave account required)
python scripts/run_qubo_decipherment.py \
  --corpus-dir data/corpus \
  --lm-dir data/language_models \
  --solver neal \
  --num-reads 1000 \
  --output outputs/decipherment/qubo_result.json

# QUBO on real D-Wave QPU (requires free Leap account)
export DWAVE_API_TOKEN=your_token_here
python scripts/run_qubo_decipherment.py \
  --solver dwave \
  --num-reads 1000 \
  --output outputs/decipherment/qubo_dwave_result.json
```

For GPU training (recommended), use the Kaggle notebook — colab_setup.ipynb works on both Kaggle (T4) and Colab (T4). Runtime: ~60 min for 50 epochs.

---

## Repository Layout

```
conf/
  config.yaml                    Hydra root config — all hyperparameters, no literals in source

data/
  baseline/                      Souza (2023) baseline outputs for comparison
  catalog/                       Sign encoding authority (Barthel ↔ Horley mapping, metadata)
  corpus/                        Per-tablet glyph sequences (JSON, from kohaumotu.org XML)
  glyphs/                        Glyph images (SVG + PNG crops from Barthel plates)
  language_models/               Pre-built NGramLM JSON files (build_language_models.py)
  metadata/                      Tablet provenance, radiocarbon dates (tablets.json)
  parallels/                     Parallel passage catalogue + auto-generated variant data
  polynesian_texts/              Raw corpora for LM training (Rapa Nui, Hawaiian, Māori, Tahitian)

hackingrongo/                    Python package
  data/                          Corpus, catalog, dataset, and NGramLM data-loading modules
  zone_a/
    autoencoder.py               Conv autoencoder — encode, decode, inpainting masks, SupCon head
    preprocessing.py             Boustrophedon normalisation, stroke extraction, scale normalisation
  zone_b/
    entropy.py                   IC / Shannon entropy sensitivity analysis (3 dating scenarios)
    contact_analysis.py          G² bigram contact analysis
    compound_detector.py         Compound glyph candidate detection (3-method cross-validation)
    astronomical_analysis.py     Calendar / lunar glyph candidate scoring
    sequence_model.py            Transformer next-glyph prediction
    sign_classifier.py           Taxogram / logogram / unknown sign classification
  zone_c/                        MCMC + beam search + LM scoring (decipherment)
  results/
    divergence_report.py         Zone A cluster-vs-Barthel HTML report
    entropy_report.py            IC / entropy HTML report
    compound_report.py           Compound candidate HTML report (scholar review)
    passage_report.py            Diachronic parallel passage HTML report
    astronomical_report.py       Lunar/calendar candidate HTML report
    decipherment_report.py       Zone C hypothesis ranking HTML report (with quantum section)
    reconstruction_report.py     Fill-the-gap inpainting gallery HTML report
  pipeline.py                    End-to-end orchestrator (steps 1–5b, 4g–4j)

scripts/
  build_language_models.py       Build all NGramLM JSON files (step 1)
  train_autoencoder.py           Zone A training with auto-resume (step 2)
  analyze_embeddings.py          UMAP + HDBSCAN + divergence report (step 3)
  cross_reference_parallels.py   Parallel passage cross-reference search (step 4e)
  run_decipherment.py            Zone C MCMC + beam search (step 5)
  measure_pgood.py               Quantum hardness: p_good, Grover estimates (step 4i)
  run_qubo_decipherment.py       QUBO sign→phoneme search — neal / tabu / D-Wave (step 4j)
  reconstruct_glyph.py           Fill-the-gap: mask a glyph, decode, KNN fallback
  parse_ids.py                   IDS Rapa Nui vocabulary → stratified LM sources
  fetch_abvd_corpus.py           ABVD cognate neighbours for pre-contact LM
  fetch_hawaiian_corpus.py       Hawaiian newspaper corpus for smoothing LM
  build_corpus.py                Kohaumotu XML → per-tablet JSON corpus
  extract_barthel_glyphs.py      Extract glyph PNG crops from Barthel (1958) PDFs
  scrape_glyphs.py               Scrape SVG glyph images from kohaumotu.org
  transform_parallels.py         Convert horley_parallels.csv to pipeline schema
  run_zone_b.py                  Run all Zone B analyses standalone

tests/                           pytest suite
outputs/                         Generated at runtime (gitignored)
  checkpoints/                   Autoencoder epoch checkpoints (.pt)
  embeddings_cache.pt            (N × 128) embedding matrix + Barthel codes
  analysis/                      Zone A/B HTML reports and CSVs
  zone_b/                        pgood_analysis.json, astronomical_candidates.json
  decipherment/                  ranking.json, qubo_result.json, decipherment_report.html
  reconstruction/                Per-glyph strip PNGs, metrics JSONs, reconstruction_report.html
```

---

## Key Scripts

**Data preparation**

| Script | Purpose |
|--------|---------|
| `scripts/build_corpus.py` | Kohaumotu XML → per-tablet JSON corpus |
| `scripts/parse_ids.py` | IDS Rapa Nui vocabulary → stratified LM sources |
| `scripts/fetch_abvd_corpus.py` | ABVD cognate neighbours for pre-contact LM |
| `scripts/fetch_hawaiian_corpus.py` | Hawaiian newspaper corpus for smoothing LM |
| `scripts/build_language_models.py` | Build all NGramLM JSON files (step 1) |
| `scripts/scrape_glyphs.py` | Scrape SVG glyph images from kohaumotu.org |
| `scripts/extract_barthel_glyphs.py` | Extract glyph PNG crops from Barthel (1958) PDFs |
| `scripts/transform_parallels.py` | Convert horley_parallels.csv to pipeline schema |

**Training and analysis**

| Script | Purpose |
|--------|---------|
| `scripts/train_autoencoder.py` | Zone A conv autoencoder training with auto-resume (step 2) |
| `scripts/analyze_embeddings.py` | UMAP + HDBSCAN + divergence report (step 3) |
| `scripts/run_zone_b.py` | Run all Zone B analyses standalone |
| `scripts/cross_reference_parallels.py` | Parallel passage Kasiski cross-reference (step 4e) |
| `scripts/run_decipherment.py` | Zone C MCMC + beam search key search (step 5) |

**Quantum**

| Script | Purpose |
|--------|---------|
| `scripts/measure_pgood.py` | p_good hardness measurement + Grover oracle estimates (step 4i) |
| `scripts/run_qubo_decipherment.py` | QUBO sign→phoneme search via neal / tabu / D-Wave QPU (step 4j) |

**Reconstruction**

| Script | Purpose |
|--------|---------|
| `scripts/reconstruct_glyph.py` | Fill-the-gap: mask a damaged glyph, decode via autoencoder + KNN fallback |

---

## Output Artifacts

After a full pipeline run, `outputs/` contains:

**Zone A — visual embeddings**

| File | Contents |
|:-----|:---------|
| `embeddings_cache.pt` | (N × 128) embedding matrix + Barthel codes for every corpus token |
| `analysis/umap_embeddings.png` | UMAP scatter coloured by Barthel family and HDBSCAN clusters |
| `analysis/cluster_vs_barthel.csv` | Per-token embedding data with cluster and family labels |
| `analysis/divergence_report.html` | Glyphs where pipeline clusters diverge from Barthel — scholar review |

**Zone B — structural analysis**

| File | Contents |
|:-----|:---------|
| `analysis/entropy_report.html` | IC / entropy sensitivity analysis across 3 dating scenarios |
| `analysis/compound_report.html` | Compound glyph candidates ranked by confidence — scholar review |
| `analysis/passage_reports/index.html` | Diachronic parallel passage alignment grids — scholar review |
| `analysis/astronomical_report.html` | Lunar / calendar glyph candidate scoring — scholar review |
| `zone_b/pgood_analysis.json` | p_good values, Grover oracle estimates, MCMC vs quantum speedup |
| `zone_b/astronomical_candidates.json` | Scored calendar / lunar sign candidates |

**Zone C — decipherment**

| File | Contents |
|:-----|:---------|
| `decipherment/ranking.json` | 120-sign phoneme assignments ranked by LM score |
| `decipherment/qubo_result.json` | QUBO best assignment + energy, solver provenance |
| `decipherment/decipherment_report.html` | Scholar-facing hypothesis cards with quantum comparison section |

**Reconstruction**

| File | Contents |
|:-----|:---------|
| `reconstruction/{prefix}_reconstruction.png` | Strip: original \| masked \| decoded \| error [\| knn \| knn_err] |
| `reconstruction/{prefix}_metrics.json` | MSE and SSIM for full image and masked region |
| `reconstruction/reconstruction_report.html` | Gallery of all reconstructions ranked by masked-region SSIM |

All `.html` reports are self-contained (no external dependencies) and designed to be sent directly to rongorongo scholars as collaboration artifacts.