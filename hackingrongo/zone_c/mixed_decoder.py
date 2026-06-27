"""
hackingrongo.zone_c.mixed_decoder
===================================

Mixed phonogram/logogram decoder for Zone C — **non-circular, fixed-type**
version.

What changed and why
--------------------
An earlier version let the decoder *relabel* signs (UNCERTAIN → LOGOGRAM) when
doing so improved its own language-model score (a ``type_flip`` pre-scan), and
assigned each logogram a random Rapa Nui morpheme. Both were removed because both
are circular and unfalsifiable: any sign that scores badly as a phonogram simply
got relabelled a logogram and dropped from scoring, which is how the mixed model
manufactured its apparent +123-LM-unit improvement over the pure syllabic model.
A type partition optimised to flatter the model's own fit cannot test anything.

In this version **sign types are fixed input, never optimised output.** They come
only from a *frozen* ``sign_type_map`` produced by an independent, falsifiable
distributional test — :mod:`hackingrongo.zone_b.sign_typology` — which is run
*before* decoding and emits a map **only if the inventory genuinely splits** into
phonogram-like and logogram-like populations (dip test + bimodality coefficient,
against a smooth-continuum null). The decoder consumes that map and **may never
alter it**. All type-partition thresholds (and their sensitivity sweep) live in
``sign_typology``, not here.

Logograms are pinned to the ``<LOGOGRAM>`` token and **excluded** from
phonotactic LM scoring. They are **not** assigned Rapa Nui morphemes — assigning a
morpheme to a sign is an unvalidatable reading, so logograms are left as
``<LOGOGRAM>``, full stop.

DORMANT on the current corpus
-----------------------------
``sign_typology`` found **no defensible bimodal split** on the present feature set
(dip-test p ≈ 0.95; the inventory is a smooth Zipfian continuum), so **no frozen
``sign_type_map`` exists**. With no map, every sign defaults to PHONOGRAM and this
decoder has **zero logograms** — it degenerates to the phonogram-only path. It is
preserved as the correct, non-circular artifact, ready if a future test/feature
set *does* find a real split. The phonogram scoring it relies on lives in the
(deprecated-but-preserved) syllabic machinery; see ``DEPRECATED_SYLLABIC.md``.

Public API
----------
``SignType`` · ``MixedMCMCSampler`` · ``MixedResult`` · ``LOGOGRAM_TOKEN``.
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from omegaconf import DictConfig

from hackingrongo.zone_c.lm_scoring import LMScorer, PhonemeMap
from hackingrongo.zone_c.mcmc import MCMCSampler, MCMCResult

logger = logging.getLogger(__name__)

# Token inserted for logogram (and taxogram) signs. The LM scorer treats it as
# OOV, so these positions are excluded from n-gram windows.
LOGOGRAM_TOKEN: str = "<LOGOGRAM>"

# Default location of the FROZEN type map produced by zone_b.sign_typology.
# Absent unless that test found a real bimodal split.
DEFAULT_TYPE_MAP_PATH = Path("data/catalog/sign_type_map.json")


class SignType(Enum):
    """Classification of a rongorongo sign (fixed input, never optimised)."""
    PHONOGRAM = "phonogram"   # maps to a syllable; participates in LM scoring
    LOGOGRAM  = "logogram"    # pinned to <LOGOGRAM>; excluded from LM scoring; NO morpheme
    TAXOGRAM  = "taxogram"    # grammatical marker; pinned / excluded


@dataclass
class MixedResult:
    """MCMC result plus the (fixed) sign-type partition used."""
    mcmc_result: MCMCResult
    type_map: dict[str, SignType] = field(default_factory=dict)
    n_phonograms: int = 0
    n_logograms: int = 0
    n_taxograms: int = 0
    logogram_fraction: float = 0.0

    @property
    def top_phoneme_map(self) -> PhonemeMap:
        if not self.mcmc_result.top_samples:
            return {}
        return self.mcmc_result.top_samples[0].phoneme_map

    @property
    def best_log_posterior(self) -> float:
        if not self.mcmc_result.top_samples:
            return float("-inf")
        return self.mcmc_result.top_samples[0].log_posterior


class MixedMCMCSampler:
    """MCMC phoneme sampler with a FIXED per-sign phonogram/logogram partition.

    The partition is supplied via ``sign_type_map`` and is immutable: there is no
    mechanism by which the decoder can change a sign's type. Signs absent from the
    map default to PHONOGRAM. Logograms/taxograms are pinned to ``<LOGOGRAM>`` and
    excluded from LM scoring; no morpheme is assigned to them.
    """

    def __init__(
        self,
        cfg: DictConfig,
        lm_scorer: LMScorer,
        corpus_sequences: list[list[str]],
        sign_ids: list[str],
        sign_type_map: dict[str, SignType] | None = None,
        phoneme_inventory: list[str] | None = None,
        phoneme_priors: list[float] | None = None,
        sign_ic_weights: dict[str, float] | None = None,
        seed: int | None = None,
        cribs: dict[str, str] | None = None,
    ) -> None:
        self._sign_ids = list(sign_ids)

        # Resolve types — FIXED. Absent → PHONOGRAM (never invented as logogram).
        type_map: dict[str, SignType] = {}
        for sign in sign_ids:
            t = (sign_type_map or {}).get(sign, SignType.PHONOGRAM)
            type_map[sign] = t if isinstance(t, SignType) else SignType.PHONOGRAM
        # Store immutably so nothing downstream can re-type a sign.
        self._type_map: dict[str, SignType] = dict(type_map)

        self._logogram_signs = frozenset(s for s, t in type_map.items() if t is SignType.LOGOGRAM)
        self._taxogram_signs = frozenset(s for s, t in type_map.items() if t is SignType.TAXOGRAM)
        n_phon = sum(1 for t in type_map.values() if t is SignType.PHONOGRAM)
        logger.info(
            "MixedMCMCSampler: %d signs — %d phonogram, %d logogram, %d taxogram%s.",
            len(sign_ids), n_phon, len(self._logogram_signs), len(self._taxogram_signs),
            "  [DORMANT: no logograms — no frozen split]" if not self._logogram_signs else "",
        )

        # Pin logogram/taxogram signs to <LOGOGRAM> (excluded from LM scoring).
        # No morpheme is assigned — that would be an unvalidatable reading.
        combined_cribs = dict(cribs) if cribs else {}
        for sign in self._logogram_signs | self._taxogram_signs:
            combined_cribs.setdefault(sign, LOGOGRAM_TOKEN)

        self._mcmc = MCMCSampler(
            cfg=cfg, lm_scorer=lm_scorer, corpus_sequences=corpus_sequences,
            sign_ids=sign_ids, phoneme_inventory=phoneme_inventory,
            phoneme_priors=phoneme_priors, seed=seed, cribs=combined_cribs,
            sign_ic_weights=sign_ic_weights,
        )

    # ------------------------------------------------------------------
    # Factory: load the FROZEN map from sign_typology (consume, never alter)
    # ------------------------------------------------------------------

    @classmethod
    def from_frozen_map(
        cls,
        cfg: DictConfig,
        lm_scorer: LMScorer,
        corpus_sequences: list[list[str]],
        sign_ids: list[str],
        type_map_path: Path | None = None,
        **kwargs: Any,
    ) -> "MixedMCMCSampler":
        """Construct from the frozen ``sign_type_map.json`` emitted by
        :mod:`hackingrongo.zone_b.sign_typology`. If the file is absent (no
        defensible split was found), every sign defaults to PHONOGRAM and the
        decoder is dormant — it does not invent a logogram class."""
        path = type_map_path or DEFAULT_TYPE_MAP_PATH
        sign_type_map: dict[str, SignType] = {}
        if Path(path).exists():
            raw = json.loads(Path(path).read_text(encoding="utf-8"))
            label_map = raw.get("sign_type_map", raw)
            for sign, label in label_map.items():
                lab = str(label).upper()
                if lab == "LOGOGRAM":
                    sign_type_map[sign] = SignType.LOGOGRAM
                elif lab == "TAXOGRAM":
                    sign_type_map[sign] = SignType.TAXOGRAM
                else:                                   # PHONOGRAM / UNRESOLVED → phonogram
                    sign_type_map[sign] = SignType.PHONOGRAM
            logger.info("Loaded frozen sign_type_map from %s (%d typed signs).",
                        path, len(sign_type_map))
        else:
            logger.warning(
                "No frozen sign_type_map at %s — sign_typology found no defensible "
                "phonogram/logogram split. Decoder is DORMANT (all PHONOGRAM, zero logograms).",
                path,
            )
        return cls(cfg, lm_scorer, corpus_sequences, sign_ids,
                   sign_type_map=sign_type_map, **kwargs)

    # ------------------------------------------------------------------
    # Run — types are fixed; just decode the phonogram positions.
    # ------------------------------------------------------------------

    def run(self) -> MixedResult:
        mcmc_result = self._mcmc.run()
        n_ph  = sum(1 for t in self._type_map.values() if t is SignType.PHONOGRAM)
        n_log = sum(1 for t in self._type_map.values() if t is SignType.LOGOGRAM)
        n_tax = sum(1 for t in self._type_map.values() if t is SignType.TAXOGRAM)
        return MixedResult(
            mcmc_result=mcmc_result, type_map=dict(self._type_map),
            n_phonograms=n_ph, n_logograms=n_log, n_taxograms=n_tax,
            logogram_fraction=n_log / max(len(self._type_map), 1),
        )

    # ------------------------------------------------------------------
    # Summary helpers
    # ------------------------------------------------------------------

    @property
    def type_summary(self) -> dict[str, int]:
        return Counter(t.value for t in self._type_map.values())

    def type_table(self) -> list[dict[str, str]]:
        ic_w = getattr(self._mcmc, "_sign_proposal_weights", {}) or {}
        return [
            {"sign": s, "type": self._type_map[s].value}
            for s in sorted(self._sign_ids, key=lambda s: ic_w.get(s, 0.0), reverse=True)
        ]
