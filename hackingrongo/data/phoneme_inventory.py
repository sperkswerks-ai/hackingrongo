"""
hackingrongo.data.phoneme_inventory — canonical syllable inventory and
phonotactic validation shared by every decipherment layer.

This module is the single source of truth for the sign→phoneme search
space.  Zone C MCMC/beam search, the QUBO formulation, and the quantum
hardness analysis (p_good / Grover estimates) must all draw their
phoneme inventory from here so that classical and quantum search
complexities are measured over the *same* space.

Orthographic conventions
------------------------
* ``g`` denotes the velar nasal /ŋ/, following the IDS and kohaumotu
  orthography for Rapa Nui (``ga`` = /ŋa/).  Sources that write ``ng``
  are canonicalised to ``g`` by :func:`canonicalize_syllable`.
* The glottal stop /ʔ/ is not represented: the historical wordlists
  feeding the language models (Thomson 1891, Roussel 1908, Fuentes
  1960, Englert 1978 via IDS) do not mark it consistently, and the LM
  build pipeline strips apostrophe/okina characters during
  normalisation.  Adding glottal syllables to the inventory would
  create tokens the LMs can never score above the OOV floor.
* Long vowels are folded to plain vowels (macrons stripped during
  normalisation), so the inventory contains short-vowel syllables only.

Rapa Nui phonotactics are strictly (C)V — no codas, no clusters.
"""

from __future__ import annotations

import re

# ---------------------------------------------------------------------------
# Canonical Rapa Nui inventory
# ---------------------------------------------------------------------------

#: Rapa Nui consonant onsets (``g`` = /ŋ/).  10 phonemic consonants minus
#: the glottal stop (unmarked in LM sources — see module docstring).
RAPA_NUI_CONSONANTS: tuple[str, ...] = (
    "g", "h", "k", "m", "n", "p", "r", "t", "v",
)

VOWELS: tuple[str, ...] = ("a", "e", "i", "o", "u")

#: The canonical sign→phoneme search space: 5 bare vowels + 45 CV
#: syllables = 50 tokens.
RAPA_NUI_SYLLABLES: tuple[str, ...] = VOWELS + tuple(
    c + v for c in RAPA_NUI_CONSONANTS for v in VOWELS
)

# ---------------------------------------------------------------------------
# Per-language structural syllable validation
# ---------------------------------------------------------------------------

# Every Polynesian language here is strictly (C)V.  Character-set
# membership is NOT sufficient validation: it admits consonant clusters
# like "gra" or "tto" whose characters are individually legal.  These
# patterns require a single legal onset (or digraph) followed by exactly
# one vowel.
_SYLLABLE_RE: dict[str, re.Pattern[str]] = {
    # g = /ŋ/ (post-canonicalisation); v is phonemic in Rapa Nui.
    "rapanui":      re.compile(r"(?:ng|[ghkmnprtv])?[aeiou]"),
    "old_rapa_nui": re.compile(r"(?:ng|[ghkmnprtv])?[aeiou]"),
    # Māori: p t k m n ng wh r w h (+ f for sources that write wh as f).
    "maori":        re.compile(r"(?:ng|wh|[ptkmnrwhfg])?[aeiou]"),
    # Hawaiian: p k m n l w h (no t, no ng).
    "hawaiian":     re.compile(r"[pkmnlwh]?[aeiou]"),
    # Tahitian: p t m n f r v h (no k, no ng).
    "tahitian":     re.compile(r"[ptmnfrvh]?[aeiou]"),
}

_DEFAULT_SYLLABLE_RE: re.Pattern[str] = re.compile(
    r"(?:ng|wh|[ghkmnprtvflw])?[aeiou]"
)


def is_valid_syllable(token: str, language: str) -> bool:
    """Return True iff *token* is a structurally valid (C)V syllable.

    Validation is structural (onset + nucleus), not character-based, so
    cluster artifacts of the CV-greedy tokenizer ("gra", "nta", "tto")
    are rejected even though their characters are individually legal.
    """
    if not token:
        return False
    pattern = _SYLLABLE_RE.get(language, _DEFAULT_SYLLABLE_RE)
    return pattern.fullmatch(token) is not None


def canonicalize_syllable(token: str, language: str) -> str:
    """Normalise orthographic variants of the same phoneme.

    For Rapa Nui, ``ng`` + vowel is rewritten to ``g`` + vowel so that
    /ŋa/ accumulates counts under a single spelling regardless of which
    convention the source wordlist used.
    """
    if language in ("rapanui", "old_rapa_nui") and token.startswith("ng"):
        return "g" + token[2:]
    return token


def clean_syllables(tokens: list[str], language: str) -> list[str]:
    """Canonicalise then phonotactically filter a tokenized syllable list."""
    cleaned = [canonicalize_syllable(t, language) for t in tokens]
    return [t for t in cleaned if is_valid_syllable(t, language)]
