#!/usr/bin/env bash
# fetch_lm_sources.sh
# Downloads freely available Polynesian language data for stratified LM training
# in hackingrongo. Place this script at the repo root and run from there.
#
# Usage:
#   chmod +x fetch_lm_sources.sh
#   ./fetch_lm_sources.sh
#
# Outputs go to data/polynesian_texts/ mirroring the existing layout.
# All sources are CC-BY 4.0 or public domain unless noted.
#
# Citation required for ABVD:
#   Greenhill SJ, Blust R & Gray RD (2008). The Austronesian Basic Vocabulary
#   Database: From Bioinformatics to Lexomics. Evolutionary Bioinformatics 4:271-283.
#
# Citation required for IDS:
#   Key MR & Comrie B (eds.) 2015. The Intercontinental Dictionary Series.
#   Leipzig: Max Planck Institute for Evolutionary Anthropology. ids.clld.org

set -euo pipefail

# ---------------------------------------------------------------------------
# Directories
# ---------------------------------------------------------------------------
ABVD_DIR="data/polynesian_texts/abvd"
IDS_DIR="data/polynesian_texts/ids"
HIST_DIR="data/polynesian_texts/historical"
NUPEPA_DIR="data/polynesian_texts/nupepa_hawaiian"

mkdir -p "$ABVD_DIR" "$IDS_DIR" "$HIST_DIR" "$NUPEPA_DIR"

ABVD_BASE="https://abvd.eva.mpg.de/austronesian/language.php"

# ---------------------------------------------------------------------------
# Helper: download ABVD language as TSV
# Args: $1=language_id  $2=language_name  $3=lm_tier (pre_contact|post_contact|baseline)
# ---------------------------------------------------------------------------
fetch_abvd() {
    local id="$1"
    local name="$2"
    local tier="$3"
    local outfile="${ABVD_DIR}/${tier}__${name}.tsv"

    if [[ -f "$outfile" ]]; then
        echo "[SKIP] $outfile already exists"
        return
    fi

    echo "[FETCH] ABVD id=${id}  ${name}  (${tier})"
    # ABVD exposes per-language TSV via the Save Data link at the bottom of each
    # language page. The URL pattern is:
    #   language.php?id=<ID>&action=download&type=tab
    # This downloads the full word list with columns:
    #   word_id | word | item | annotation | cognacy | loan
    curl -sSL --retry 3 \
        "${ABVD_BASE}?id=${id}&action=download&type=tab" \
        -o "$outfile"

    # Verify we got a TSV (not an HTML error page)
    if ! head -1 "$outfile" | grep -q $'\t'; then
        echo "[WARN] ${outfile} may not be a valid TSV — check manually"
    fi
}

# ---------------------------------------------------------------------------
# ABVD downloads
# Tier assignment rationale:
#   pre_contact  — East Polynesian languages closest to Rapa Nui ~1500 CE
#                  Mangarevan is the probable stepping-stone; Marquesan is
#                  morphologically closest; Tuamotuan sits between.
#   post_contact — Rapa Nui itself + Hawaiian for n-gram smoothing volume
#   baseline     — Samoan (Nuclear Polynesian) and Tongan (Tongic) as
#                  outgroup controls for Zone C scoring
# ---------------------------------------------------------------------------

# Pre-contact cluster targets
fetch_abvd 264 "rapanui"     "pre_contact"
fetch_abvd 253 "mangarevan"  "pre_contact"
fetch_abvd 254 "marquesan"   "pre_contact"
fetch_abvd 246 "tuamotuan"   "pre_contact"
fetch_abvd 261 "tahitian"    "pre_contact"

# Post-contact cluster targets
fetch_abvd 109 "hawaiian"    "post_contact"
fetch_abvd 256 "maori"       "post_contact"

# Outgroup baselines
fetch_abvd 259 "samoan"      "baseline"
fetch_abvd 263 "tongan"      "baseline"

echo ""
echo "[ABVD] Done. Files in ${ABVD_DIR}/"

# ---------------------------------------------------------------------------
# IDS Rapa Nui dictionary (multi-source: Roussel 1908, Englert 1978,
# Fuentes 1960, Thomson 1891, de Agüera 1770)
# Contribution 238 = Rapa Nui. The tab-separated download includes a
# 'source' column so you can filter by era.
# ---------------------------------------------------------------------------
IDS_URL="https://ids.clld.org/contributions/238.tab"
IDS_OUT="${IDS_DIR}/rapanui_ids_238.tsv"

if [[ -f "$IDS_OUT" ]]; then
    echo "[SKIP] $IDS_OUT already exists"
else
    echo "[FETCH] IDS Rapa Nui contribution 238"
    curl -sSL --retry 3 "$IDS_URL" -o "$IDS_OUT"
    echo "[IDS] Saved to ${IDS_OUT}"
    echo "      Filter column 'source' for era stratification:"
    echo "        pre_contact proxy : 'Roussel 1908', 'Thomson 1891', 'de Agüera 1770'"
    echo "        post_contact      : 'Englert 1978', 'Fuentes 1960'"
fi

# ---------------------------------------------------------------------------
# Historical sources (public domain)
# ---------------------------------------------------------------------------

# Tregear 1891 — Maori-Polynesian Comparative Dictionary
# Covers Rapa Nui, Hawaiian, Maori, Tahitian, Samoan, Tongan with cognates.
# Public domain. Internet Archive identifier: maoripolynesian01treggoog
TREGEAR_OUT="${HIST_DIR}/tregear_1891_maori_polynesian_dict.txt"
TREGEAR_URL="https://archive.org/download/maoripolynesian01treggoog/maoripolynesian01treggoog_djvu.txt"

if [[ -f "$TREGEAR_OUT" ]]; then
    echo "[SKIP] $TREGEAR_OUT already exists"
else
    echo "[FETCH] Tregear 1891 (public domain, Internet Archive)"
    curl -sSL --retry 3 "$TREGEAR_URL" -o "$TREGEAR_OUT"
    echo "[HIST] Saved Tregear 1891 plain text to ${TREGEAR_OUT}"
fi

# Andrews 1865 — A Dictionary of the Hawaiian Language (revised edition)
# The closest large written source to pre-contact East Polynesian phonology.
# Public domain. Internet Archive identifier: ofhawadictionary00andrrich
ANDREWS_OUT="${HIST_DIR}/andrews_1865_hawaiian_dict.txt"
ANDREWS_URL="https://archive.org/download/ofhawadictionary00andrrich/ofhawadictionary00andrrich_djvu.txt"

if [[ -f "$ANDREWS_OUT" ]]; then
    echo "[SKIP] $ANDREWS_OUT already exists"
else
    echo "[FETCH] Andrews 1865 Hawaiian dictionary (public domain, Internet Archive)"
    curl -sSL --retry 3 "$ANDREWS_URL" -o "$ANDREWS_OUT"
    echo "[HIST] Saved Andrews 1865 plain text to ${ANDREWS_OUT}"
fi

# ---------------------------------------------------------------------------
# Nupepa Hawaiian newspaper corpus — 19th-century Hawaiian text
# Provides high-volume East Polynesian running text for n-gram smoothing.
# nupepa.org exposes article-level OCR text. The URL below fetches a
# pre-assembled plain-text dump of the full corpus if available, otherwise
# see NOTE below.
#
# NOTE: nupepa.org does not currently expose a single bulk-download endpoint.
# The recommended approach is to use their search API or to contact the
# project (nupepa.org) directly for a corpus dump — they have provided
# bulk text to researchers before.
#
# As a working alternative, the Internet Archive hosts some nupepa content:
#   https://archive.org/search?query=nupepa+hawaiian+language
# The script below fetches one well-known title (Ka Nupepa Kuokoa, 1861-1927)
# as a sample. Extend the list as needed.
# ---------------------------------------------------------------------------

echo ""
echo "[INFO] Nupepa Hawaiian corpus"
echo "       For a full corpus dump, contact the Ho'olaupa'i project:"
echo "       https://www.nupepa.org/"
echo "       The script fetches a sample title as a placeholder."

NUPEPA_SAMPLE_URL="https://archive.org/download/ka-nupepa-kuokoa-sample/ka-nupepa-kuokoa-sample_djvu.txt"
NUPEPA_SAMPLE_OUT="${NUPEPA_DIR}/ka_nupepa_kuokoa_sample.txt"

# Attempt sample download; skip gracefully if not available
echo "[FETCH] Nupepa sample (Ka Nupepa Kuokoa — may 404 if not archived)"
curl -sSL --retry 2 --fail "$NUPEPA_SAMPLE_URL" -o "$NUPEPA_SAMPLE_OUT" \
    && echo "[NUPEPA] Sample saved to ${NUPEPA_SAMPLE_OUT}" \
    || echo "[NUPEPA] Sample not available at that URL — see NOTE above for manual steps"

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo ""
echo "========================================================"
echo " fetch_lm_sources.sh complete"
echo "========================================================"
echo ""
echo " Suggested LM tier mapping for hackingrongo:"
echo ""
echo "  pre_contact_lm:"
echo "    ABVD: ${ABVD_DIR}/pre_contact__*.tsv"
echo "    IDS (Roussel/Thomson/de Agüera rows): ${IDS_OUT}"
echo "    Historical: ${HIST_DIR}/andrews_1865_hawaiian_dict.txt"
echo ""
echo "  post_contact_lm:"
echo "    ABVD: ${ABVD_DIR}/post_contact__rapanui.tsv"
echo "    IDS (Englert/Fuentes rows): ${IDS_OUT}"
echo "    Existing corpus: data/polynesian_texts/old_rapa_nui/"
echo ""
echo "  smoothing (high-volume East Polynesian):"
echo "    ${NUPEPA_DIR}/"
echo "    ${HIST_DIR}/tregear_1891_maori_polynesian_dict.txt"
echo ""
echo "  baseline/outgroup:"
echo "    ABVD: ${ABVD_DIR}/baseline__*.tsv"
echo ""
echo " ABVD citation: Greenhill SJ et al. (2008) Evol. Bioinformatics 4:271-283"
echo " IDS citation:  Key MR & Comrie B (eds.) 2015. ids.clld.org"
echo "========================================================"
