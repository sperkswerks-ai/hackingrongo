All data files under `data/` were downloaded from published, freely accessible sources. None are original to this repository. All copyrights remain with their respective authors.

### Primary Corpus

`data/corpus/` — 15,273 glyphs across 25 tablets (A–Y).
Source: kohaumotu.org (Philip Spaelti), freely redistributable for research.
Primary: Barthel, T.S. (1958). *Grundlagen zur Entzifferung der Osterinselschrift*. Hamburg: Cram, de Gruyter.

`data/corpus/D_ferrara2022.json` — Alternative Tablet D transcription.
Source: Ferrara et al. (2022). *Digital Scholarship in the Humanities*, 37(2). DOI: 10.1093/llc/fqab045.

### Parallel Passages

`data/parallels/horley_parallels.csv` — 146 passage groups.
Derived from de Souza (2023) rongopy (GPL-3.0); original catalogue: Horley, P. (2021). *Rongorongo*. Rapa Nui Press.

### Language Model Corpora

| Source | File | Notes |
|--------|------|-------|
| IDS contribution 238 (Key & Comrie 2015, CC-BY 4.0) | `rapanui/ids*.txt` | Thomson 1891 + Roussel 1908 (pre-contact); Fuentes 1960 + Englert 1978 (post-contact) |
| ABVD (Greenhill et al. 2008, CC-BY 4.0) | `rapanui/abvd_cognate_neighbours.txt` | East Polynesian cognate neighbours |
| Hawaiian Corpus Project (dohliam, CC0) | `nupepa_hawaiian/haw_unigrams.txt` | ~56K Hawaiian word types for smoothing |
| Tregear (1891), public domain | `historical/tregear_1891*.txt` | Māori-Polynesian comparative dictionary |
| Andrews (1865), public domain | `historical/andrews_1865*.txt` | Hawaiian dictionary |

### Sign Catalog

`data/catalog/horley_encoding.json` — Barthel→Horley mapping.
Source: de Souza (2023) rongopy (GPL-3.0); encoding revision: Horley (2021).

### Data Curation Notes (June 2026)

- Language-model syllable validation is now structural (single (C)V onset+nucleus per `hackingrongo/data/phoneme_inventory.py`) rather than character-set based; this removes phonotactically impossible tokens ("gra", "nta", "tto") that the CV-greedy tokenizer produced from source-text artifacts. The velar nasal is canonicalised to the IDS spelling `g` (`nga` → `ga`), and `v` — phonemic in Rapa Nui but previously absent from the character filter — is included.
- CEIPP corpus codes with multi-letter tails (`022bfy`), uppercase markers (`001V`), and compound connectors (`009:005`) are now parsed: the variant letter is arbitrated against the sign catalog, stripped modifiers are preserved in `code_modifiers`, and compound components in `barthel_components`. This recovered 563 previously unidentified tokens and collapsed spurious sign-type fragmentation (1,330 → 639 base signs).

### Data Curation Notes (May 2026)

- Barthel reference images were manually reviewed to remove non-glyph artifacts (notably Arabic numeral labels captured during PDF extraction).
- Ambiguous positional reference filenames (for example `100_42_barthel_...`) are no longer used by default in fallback image resolution.
- SVG glyphs are transparent by design; rasterization composites onto white before grayscale conversion to avoid dark/checkerboard background artifacts in model inputs.
- 3D crop and synthetic-view assets are optional. For baseline training runs, excluding them from packaged datasets is supported and often preferred for cleaner image quality.

---

## License

Code: **MIT License**.

Data under `data/` is redistributed under original licenses. GPL-3.0 applies to files deriving from de Souza (2023) — comply with GPL-3.0 terms if redistributing this repository.

---

## References

- Andrews, L. (1865). *A Dictionary of the Hawaiian Language*. Honolulu: Henry M. Whitney.
- Barthel, T.S. (1958). *Grundlagen zur Entzifferung der Osterinselschrift*. Hamburg: Cram, de Gruyter.
- Barthel, T.S. (1960). Rezente Einwirkungen auf das Runenschreiben der Osterinsulaner. *Baessler-Archiv* 8, 255–274.
- Blixen, O. (1979). El lenguaje secreto de la Isla de Pascua. *Moana* 2(1).
- dohliam. *Hawaiian Corpus Project*. dohliam.github.io/corpus/haw. CC0.
- Ferrara, M., Lastilla, L., Ravanelli, N., & Valério, M. (2022). Modelling the Rongorongo tablets. *Digital Scholarship in the Humanities*, 37(2), 497–526.
- Ferrara, S., Tassoni, L., Kromer, B., et al. (2024). The invention of writing on Rapa Nui (Easter Island): new radiocarbon dates on the Rongorongo script. *Scientific Reports* 14, 2794. DOI: 10.1038/s41598-024-53063-7.
- Fischer, S.R. (1994). Preliminary evidence for cosmogonic texts in rongorongo. *Journal of the Polynesian Society* 103(3), 303–321.
- Fischer, S.R. (1997). *RongoRongo, the Easter Island Script*. Oxford: Clarendon Press.
- Greenhill, S.J., Blust, R., & Gray, R.D. (2008). The Austronesian Basic Vocabulary Database. *Evolutionary Bioinformatics*, 4:271–283.
- Horley, P. (2021). *Rongorongo*. Rapa Nui Press.
- Key, M.R. & Comrie, B. (eds.) (2015). *The Intercontinental Dictionary Series*. Leipzig: Max Planck Institute for Evolutionary Anthropology.
- Kieviet, P. (2017). *A Grammar of Rapa Nui*. Berlin: Language Science Press.
- Orliac, C. (2005). The woody plants of the rongorongo tablets. *Rapa Nui Journal* 19(1), 61–66.
- Roberts, G.O., Gelman, A., & Gilks, W.R. (1997). Weak convergence and optimal scaling of random walk Metropolis algorithms. *Annals of Applied Probability*, 7(1), 110–120.
- Spaelti, P. (2012). kohaumotu.org rongorongo corpus. kohaumotu.org/Rongorongo/
- de Souza, J.G. (2023). *rongopy*. GitHub. github.com/jgregoriods/rongopy. GPL-3.0.
- Tregear, E. (1891). *The Maori-Polynesian Comparative Dictionary*. Wellington: Lyon and Blair.
- POLLEX-Online (pollex.eva.mpg.de); Pasefika (pasefika.com/dictionary); Austronesian Comparative Dictionary (trussel2.com/acd)
- Tregear's Maori-Polynesian Comparative Dictionary (1891)