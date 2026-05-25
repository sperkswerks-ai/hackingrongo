"""
hackingrongo.data.passage_alignment
====================================

Parallel passage alignment and diachronic variant analysis.

Data models and analysis functions for detecting systematic sign changes
between pre-contact and post-contact attestations of the same passage.
The core scientific finding: substitutions that cross Barthel century-block
boundaries consistently across multiple post-contact tablets are the
strongest evidence for a contact-boundary writing event rather than
scribal idiosyncrasy.

Public API
----------
Data models
    ``AlignmentCell``, ``PassageAttestation``, ``DiachronicChange``,
    ``PassageAlignment``

Analysis
    ``analyze_passage(passage, catalog, tablet_meta)``
        Align all variants, detect diachronic changes, score.

    ``analyze_all_passages(passages, catalog, tablet_meta)``
        Batch: process all passages, filter to those with cross-stratum
        signal, return sorted by interest score.

I/O
    ``save_analysis(alignments, output_path)``
        Write JSON compatible with ``passage_report.py``.

CLI::

    python -m hackingrongo.data.passage_alignment \\
        --parallels data/parallels/parallel_variants_auto.json \\
        --tablets  data/metadata/tablets.json \\
        --output   outputs/analysis/diachronic_analysis.json
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Optional

from hackingrongo.data.constants import POST_CONTACT, PRE_CONTACT, UNDATED_STRATUM

if TYPE_CHECKING:
    from hackingrongo.data.catalog import SignCatalog
    from hackingrongo.data.parallels import ParallelPassage

logger = logging.getLogger(__name__)


@dataclass
class AlignmentCell:
    """Single position in sequence alignment."""
    position: int
    query_code: str
    corpus_code: str
    match_type: Literal["match", "substitution", "insertion", "deletion"]
    is_significant: bool = False


@dataclass
class PassageAttestation:
    """Single attestation of a passage on one tablet."""
    tablet: str
    tablet_name: str
    stratum: str  # PRE_CONTACT | POST_CONTACT | UNDATED_STRATUM
    date_range: str  # e.g., "~1493-1509 CE"
    sequence: list[str]  # Barthel codes
    edit_distance: int
    alignment: list[AlignmentCell] = field(default_factory=list)


@dataclass
class DiachronicChange:
    """Linguistic change detected between pre and post-contact attestations."""
    position: int  # position in canonical sequence
    pre_contact_sign: str
    post_contact_sign: str
    change_type: str  # "substitution", "insertion", "deletion"
    is_known_allograph: bool
    crosses_barthel_family: bool
    n_tablets_consistent: int  # how many post-contact tablets show same change
    is_holy_grail_candidate: bool  # also appears in same n-gram context


@dataclass
class PassageAlignment:
    """Complete alignment result for a parallel passage group."""
    passage_id: str
    canonical_sequence: list[str]
    canonical_tablet: str
    canonical_stratum: str
    attestations: list[PassageAttestation] = field(default_factory=list)
    diachronic_changes: list[DiachronicChange] = field(default_factory=list)
    interest_score: float = 0.0

    @property
    def n_tablets(self) -> int:
        """Count unique tablets."""
        return len(set(a.tablet for a in self.attestations))

    @property
    def n_attestations(self) -> int:
        """Total attestation count."""
        return len(self.attestations)

    @property
    def pre_contact_count(self) -> int:
        """Count pre-contact attestations."""
        return sum(1 for a in self.attestations if a.stratum == PRE_CONTACT)

    @property
    def post_contact_count(self) -> int:
        """Count post-contact attestations."""
        return sum(1 for a in self.attestations if a.stratum == POST_CONTACT)

    @property
    def has_diachronic_signal(self) -> bool:
        """Whether passage spans pre/post-contact strata."""
        return self.pre_contact_count > 0 and self.post_contact_count > 0

    @property
    def holy_grail_candidates(self) -> list[DiachronicChange]:
        """Filter for holy grail candidates (strong temporal signal)."""
        return [c for c in self.diachronic_changes if c.is_holy_grail_candidate]

    @property
    def family_crossing_changes(self) -> list[DiachronicChange]:
        """Changes that cross Barthel family boundaries."""
        return [c for c in self.diachronic_changes if c.crosses_barthel_family]


# ---------------------------------------------------------------------------
# Stratum normalization
# ---------------------------------------------------------------------------

_PRE_LABELS: frozenset[str] = frozenset({"pre_contact", "pre"})
# "early" is ambiguous (early-contact period); treat conservatively as undated.
_POST_LABELS: frozenset[str] = frozenset({"post_contact", "late"})


def _normalize_stratum(stratum: str) -> str:
    """Map variant stratum labels to PRE_CONTACT / POST_CONTACT / UNDATED_STRATUM.

    Both Horley (2021) passage labels ("pre", "late") and the tablet-level
    radiocarbon cluster labels ("pre_contact", "post_contact") are accepted.
    "early" is treated as undated: it overlaps both sides of contact and
    including it in either group would contaminate the comparison.
    """
    s = stratum.lower().strip()
    if s in _PRE_LABELS:
        return PRE_CONTACT
    if s in _POST_LABELS:
        return POST_CONTACT
    return UNDATED_STRATUM


# ---------------------------------------------------------------------------
# Barthel century-block classification
# ---------------------------------------------------------------------------

def _barthel_century_block(code: str) -> int:
    """Return the Barthel century block for a code (1–99 → 0, 100–199 → 1, …).

    Barthel organised his catalogue in morphological blocks aligned with the
    hundreds digit of the numeric code.  Two substituting signs in the same
    century block are structurally related (same morphological class); those
    in different blocks cross a genuine sign-family boundary.

    Returns -1 for codes that contain no numeric part (compound, unknown).
    """
    try:
        n = int(re.sub(r"[^0-9]", "", code))
        return n // 100
    except ValueError:
        return -1


# ---------------------------------------------------------------------------
# Needleman-Wunsch global alignment
# ---------------------------------------------------------------------------

def _needleman_wunsch(
    canonical: list[str],
    variant: list[str],
) -> list[AlignmentCell]:
    """Global pairwise alignment of a variant form against the canonical form.

    Uses classic Needleman-Wunsch (match +1, mismatch −1, gap −1) with
    diagonal-first tie-breaking so that substitutions are preferred over
    gap pairs when scores are equal.

    In the returned cells:
    - ``query_code``  = canonical sign (the reference)
    - ``corpus_code`` = variant sign (what was actually inscribed)
    - ``position``    = 0-based index in *canonical*; -1 for insertions
      (the variant has an extra sign with no canonical counterpart)
    """
    n, m = len(canonical), len(variant)
    MATCH, MISMATCH, GAP = 1, -1, -1

    # DP table with int scores (no floats needed for integer costs)
    dp: list[list[int]] = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i * GAP
    for j in range(m + 1):
        dp[0][j] = j * GAP

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            diag = dp[i - 1][j - 1] + (MATCH if canonical[i - 1] == variant[j - 1] else MISMATCH)
            dp[i][j] = max(diag, dp[i - 1][j] + GAP, dp[i][j - 1] + GAP)

    # Traceback — diagonal-first tie-breaking
    cells: list[AlignmentCell] = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0:
            score = MATCH if canonical[i - 1] == variant[j - 1] else MISMATCH
            if dp[i][j] == dp[i - 1][j - 1] + score:
                mt: Literal["match", "substitution", "insertion", "deletion"] = (
                    "match" if canonical[i - 1] == variant[j - 1] else "substitution"
                )
                cells.append(AlignmentCell(
                    position=i - 1,
                    query_code=canonical[i - 1],
                    corpus_code=variant[j - 1],
                    match_type=mt,
                    is_significant=(mt == "substitution"),
                ))
                i -= 1
                j -= 1
                continue
        if i > 0 and (j == 0 or dp[i][j] == dp[i - 1][j] + GAP):
            cells.append(AlignmentCell(
                position=i - 1,
                query_code=canonical[i - 1],
                corpus_code="-",
                match_type="deletion",
                is_significant=True,
            ))
            i -= 1
        else:
            cells.append(AlignmentCell(
                position=-1,
                query_code="-",
                corpus_code=variant[j - 1],
                match_type="insertion",
            ))
            j -= 1

    cells.reverse()
    return cells


# ---------------------------------------------------------------------------
# Interest score
# ---------------------------------------------------------------------------

def _compute_interest_score(
    changes: list[DiachronicChange],
    passage_len: int,
) -> float:
    """Score a passage's diachronic signal on a 0–10 scale.

    Weights (per change):
    - Holy-grail candidate (systematic cross-block substitution): +4
    - Cross-block substitution (not holy-grail): +2
    - Substitution within same block: +1
    - Insertion / deletion: +0.5

    The raw sum is divided by passage length (so longer passages are not
    penalised for having more opportunities to change) and scaled to 0–10.
    """
    if not changes:
        return 0.0
    raw = 0.0
    for c in changes:
        if c.is_holy_grail_candidate:
            raw += 4.0
        elif c.crosses_barthel_family:
            raw += 2.0
        elif c.change_type == "substitution":
            raw += 1.0
        else:
            raw += 0.5
    return round(min((raw / max(passage_len, 1)) * 5.0, 10.0), 3)


# ---------------------------------------------------------------------------
# Core analysis
# ---------------------------------------------------------------------------

def analyze_passage(
    passage: "ParallelPassage",
    catalog: "SignCatalog",
    tablet_meta: dict[str, dict[str, Any]],
) -> PassageAlignment:
    """Align all variants of a passage and detect diachronic sign changes.

    Parameters
    ----------
    passage : ParallelPassage
        Loaded parallel passage with all variant attestations.
    catalog : SignCatalog
        Used to resolve allograph groups for the known-allograph flag.
    tablet_meta : dict
        ``{tablet_id: {name, radiocarbon_date_min, radiocarbon_date_max, …}}``
        as loaded from ``tablets.json``.  Keys missing from this dict produce
        empty string fallbacks rather than errors.

    Returns
    -------
    PassageAlignment
        Populated with attestations, per-position alignments, diachronic
        changes, and an interest score.
    """
    canonical = passage.canonical_form
    attestations: list[PassageAttestation] = []
    # (stratum_norm, alignment) pairs for diachronic position analysis
    stratum_alignments: list[tuple[str, str, list[AlignmentCell]]] = []  # (tablet_id, stratum, cells)

    for variant in passage.variants:
        stratum_norm = _normalize_stratum(variant.stratum)
        meta = tablet_meta.get(variant.tablet_id, {})

        d_min = meta.get("radiocarbon_date_min")
        d_max = meta.get("radiocarbon_date_max")
        if d_min and d_max:
            date_range = f"~{d_min}–{d_max} CE"
        elif d_min:
            date_range = f"after {d_min} CE"
        else:
            date_range = "undated"

        alignment = _needleman_wunsch(canonical, list(variant.form))
        edit_dist = sum(1 for c in alignment if c.match_type != "match")

        attestations.append(PassageAttestation(
            tablet=variant.tablet_id,
            tablet_name=meta.get("name", variant.tablet_id),
            stratum=stratum_norm,
            date_range=date_range,
            sequence=list(variant.form),
            edit_distance=edit_dist,
            alignment=alignment,
        ))
        stratum_alignments.append((variant.tablet_id, stratum_norm, alignment))

    # Pick the canonical tablet: prefer the earliest pre-contact attestation.
    canonical_tablet = ""
    canonical_stratum = UNDATED_STRATUM
    for att in attestations:
        if att.stratum == PRE_CONTACT:
            canonical_tablet = att.tablet
            canonical_stratum = PRE_CONTACT
            break
    if not canonical_tablet and attestations:
        canonical_tablet = attestations[0].tablet
        canonical_stratum = attestations[0].stratum

    # ------------------------------------------------------------------ #
    # Diachronic change detection                                         #
    # ------------------------------------------------------------------ #
    # For each canonical position, accumulate the sign each variant has
    # at that position (corpus_code in the alignment cell; "-" for deletions).
    # Insertions have position=-1 and are excluded from positional analysis.

    # pos → stratum → [(tablet_id, sign), ...]
    pos_signs: dict[int, dict[str, list[tuple[str, str]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for tablet_id, stratum_norm, cells in stratum_alignments:
        for cell in cells:
            if cell.position >= 0:
                pos_signs[cell.position][stratum_norm].append(
                    (tablet_id, cell.corpus_code)
                )

    diachronic_changes: list[DiachronicChange] = []

    for pos in range(len(canonical)):
        pre_entries = pos_signs.get(pos, {}).get(PRE_CONTACT, [])
        post_entries = pos_signs.get(pos, {}).get(POST_CONTACT, [])

        if not pre_entries or not post_entries:
            continue  # need both strata represented at this position

        pre_counter = Counter(sign for _, sign in pre_entries)
        post_counter = Counter(sign for _, sign in post_entries)
        pre_consensus = pre_counter.most_common(1)[0][0]
        post_consensus = post_counter.most_common(1)[0][0]

        if pre_consensus == post_consensus:
            continue  # no change at this position

        # Classify change type
        if pre_consensus == "-":
            change_type: Literal["substitution", "insertion", "deletion"] = "insertion"
        elif post_consensus == "-":
            change_type = "deletion"
        else:
            change_type = "substitution"

        # Barthel century-block check
        pre_block = _barthel_century_block(pre_consensus)
        post_block = _barthel_century_block(post_consensus)
        crosses_family = (
            change_type == "substitution"
            and pre_block >= 0
            and post_block >= 0
            and pre_block != post_block
        )

        # Known allograph: both signs resolve to the same canonical sign
        is_allograph = (
            change_type == "substitution"
            and catalog.get_canonical_id(pre_consensus) == catalog.get_canonical_id(post_consensus)
            and pre_consensus != post_consensus
        )

        # Consistency: number of distinct post-contact tablets showing the
        # post-contact consensus sign at this position.
        n_consistent = len({
            tablet_id
            for tablet_id, sign in post_entries
            if sign == post_consensus
        })

        # Holy-grail candidate: systematic non-allographic substitution that
        # appears consistently on multiple post-contact tablets.
        is_holy_grail = (
            change_type == "substitution"
            and not is_allograph
            and n_consistent >= 2
        )

        diachronic_changes.append(DiachronicChange(
            position=pos,
            pre_contact_sign=pre_consensus,
            post_contact_sign=post_consensus,
            change_type=change_type,
            is_known_allograph=is_allograph,
            crosses_barthel_family=crosses_family,
            n_tablets_consistent=n_consistent,
            is_holy_grail_candidate=is_holy_grail,
        ))

    interest = _compute_interest_score(diachronic_changes, len(canonical))

    return PassageAlignment(
        passage_id=passage.passage_id,
        canonical_sequence=canonical,
        canonical_tablet=canonical_tablet,
        canonical_stratum=canonical_stratum,
        attestations=attestations,
        diachronic_changes=diachronic_changes,
        interest_score=interest,
    )


def analyze_all_passages(
    passages: list["ParallelPassage"],
    catalog: "SignCatalog",
    tablet_meta: dict[str, dict[str, Any]],
    require_cross_stratum: bool = True,
) -> list[PassageAlignment]:
    """Analyze diachronic signal across all parallel passages.

    Parameters
    ----------
    passages : list[ParallelPassage]
        All loaded parallel passages.
    catalog : SignCatalog
        Sign catalog for allograph and century-block lookups.
    tablet_meta : dict
        Tablet metadata dict as loaded from ``tablets.json``.
    require_cross_stratum : bool
        If True (default), drop passages that have no pre-contact attestation
        OR no post-contact attestation — they cannot produce diachronic signal.

    Returns
    -------
    list[PassageAlignment]
        Sorted by ``interest_score`` descending.  If ``require_cross_stratum``
        is True, only passages spanning the contact boundary are returned.
    """
    results: list[PassageAlignment] = []
    n_total = len(passages)
    n_skipped = 0

    for passage in passages:
        alignment = analyze_passage(passage, catalog, tablet_meta)
        if require_cross_stratum and not alignment.has_diachronic_signal:
            n_skipped += 1
            continue
        results.append(alignment)

    results.sort(key=lambda a: a.interest_score, reverse=True)

    n_with_changes = sum(1 for a in results if a.diachronic_changes)
    n_holy_grail = sum(
        1 for a in results if any(c.is_holy_grail_candidate for c in a.diachronic_changes)
    )
    n_cross_family = sum(
        1 for a in results if any(c.crosses_barthel_family for c in a.diachronic_changes)
    )

    logger.info(
        "Diachronic analysis: %d passages in, %d with cross-stratum signal "
        "(%d skipped single-stratum). Changes detected in %d; "
        "%d holy-grail candidates; %d cross-family substitutions.",
        n_total, len(results), n_skipped,
        n_with_changes, n_holy_grail, n_cross_family,
    )
    return results


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------

def _alignment_to_dict(alignment: PassageAlignment) -> dict[str, Any]:
    """Serialize a PassageAlignment to a JSON-compatible dict."""
    return asdict(alignment)


def save_analysis(
    alignments: list[PassageAlignment],
    output_path: Path,
    extra_stats: dict[str, Any] | None = None,
) -> None:
    """Write diachronic analysis results to JSON.

    The output format is compatible with ``passage_report.py``'s
    ``generate_report`` method: a top-level dict with a ``"passages"`` list
    of serialized ``PassageAlignment`` objects.

    Parameters
    ----------
    alignments : list[PassageAlignment]
        Results from ``analyze_all_passages``.
    output_path : Path
        Destination file (parent directory is created if needed).
    extra_stats : dict, optional
        Additional metadata to store in ``"_analysis_stats"``.
    """
    passages_dicts = [_alignment_to_dict(a) for a in alignments]

    n_holy = sum(
        1 for a in alignments if any(c.is_holy_grail_candidate for c in a.diachronic_changes)
    )
    n_cross = sum(
        1 for a in alignments if any(c.crosses_barthel_family for c in a.diachronic_changes)
    )
    total_changes = sum(len(a.diachronic_changes) for a in alignments)

    stats: dict[str, Any] = {
        "passages_analyzed": len(alignments),
        "passages_with_changes": sum(1 for a in alignments if a.diachronic_changes),
        "total_diachronic_changes": total_changes,
        "passages_with_holy_grail_candidates": n_holy,
        "passages_with_cross_family_substitutions": n_cross,
    }
    if extra_stats:
        stats.update(extra_stats)

    output = {
        "_schema_version": "1.0",
        "_description": (
            "Diachronic variant analysis: systematic sign changes between "
            "pre-contact and post-contact parallel passage attestations."
        ),
        "_analysis_stats": stats,
        "passages": passages_dicts,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Diachronic analysis written → %s (%d passages)", output_path, len(alignments))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Diachronic variant analysis for rongorongo parallel passages. "
            "Aligns all variants against the canonical form and classifies "
            "sign changes across the pre-/post-contact boundary."
        )
    )
    p.add_argument(
        "--parallels",
        type=Path,
        default=Path("data/parallels/parallel_variants_auto.json"),
        help="Parallel variants JSON (output of cross_reference_parallels.py)",
    )
    p.add_argument(
        "--tablets",
        type=Path,
        default=Path("data/metadata/tablets.json"),
        help="Tablet metadata JSON with radiocarbon dates and stratum info",
    )
    p.add_argument(
        "--catalog-dir",
        type=Path,
        default=Path("data/catalog"),
        help="Directory containing horley_encoding.json, allographs.json, "
             "sign_metadata.json",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/analysis/diachronic_analysis.json"),
        help="Output JSON path (compatible with passage_report.py)",
    )
    p.add_argument(
        "--all-passages",
        action="store_true",
        help="Include single-stratum passages (default: only cross-stratum)",
    )
    p.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="DEBUG-level logging",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(name)s: %(levelname)s: %(message)s",
    )

    # Load tablet metadata
    tablet_meta: dict[str, dict[str, Any]] = {}
    if args.tablets.exists():
        try:
            tablet_meta = json.loads(args.tablets.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning("Could not load tablets.json (%s): %s", args.tablets, exc)
    else:
        logger.warning("tablets.json not found at %s — date ranges will be empty.", args.tablets)

    # Load sign catalog (graceful fallback if catalog files are absent)
    try:
        from omegaconf import OmegaConf
        from hackingrongo.data.catalog import SignCatalog

        # Build a minimal config pointing at the catalog files
        cfg = OmegaConf.create({
            "paths": {
                "horley_encoding_json": str(args.catalog_dir / "horley_encoding.json"),
                "allographs_json": str(args.catalog_dir / "allographs.json"),
                "sign_metadata_json": str(args.catalog_dir / "sign_metadata.json"),
            }
        })
        catalog = SignCatalog.load(cfg, Path("."))
    except Exception as exc:
        logger.warning(
            "Could not load sign catalog (%s). "
            "Allograph detection will be disabled.", exc,
        )
        # Stub catalog that never recognises allographs
        class _NullCatalog:
            signs: dict = {}
            def get_canonical_id(self, code: str) -> str:
                return code
        catalog = _NullCatalog()  # type: ignore[assignment]

    # Load parallel passages
    from hackingrongo.data.parallels import load_parallel_variants_json
    passages = load_parallel_variants_json(args.parallels, catalog)

    if not passages:
        logger.error("No passages loaded from %s — nothing to analyze.", args.parallels)
        return

    # Run diachronic analysis
    alignments = analyze_all_passages(
        passages,
        catalog,
        tablet_meta,
        require_cross_stratum=not args.all_passages,
    )

    if not alignments:
        logger.warning(
            "No cross-stratum passages found. "
            "Re-run with --all-passages to include single-stratum passages, "
            "or check that parallel_variants_auto.json contains tablets "
            "with both pre_contact and post_contact stratum assignments."
        )

    # Summary to stdout
    n_holy = sum(1 for a in alignments if a.holy_grail_candidates)
    n_cross = sum(1 for a in alignments if a.family_crossing_changes)
    print(f"\nDiachronic Analysis Summary")
    print(f"  Passages with cross-stratum signal : {len(alignments)}")
    print(f"  Passages with detected changes     : "
          f"{sum(1 for a in alignments if a.diachronic_changes)}")
    print(f"  Cross-block substitutions          : {n_cross}")
    print(f"  Holy-grail candidates              : {n_holy}")
    if alignments:
        print(f"\n  Top passages by interest score:")
        for a in alignments[:5]:
            n_changes = len(a.diachronic_changes)
            hg = len(a.holy_grail_candidates)
            print(f"    {a.passage_id:20s}  score={a.interest_score:.2f}  "
                  f"changes={n_changes}  holy-grail={hg}")

    save_analysis(alignments, args.output)
    print(f"\n  Output → {args.output}")


if __name__ == "__main__":
    main()
