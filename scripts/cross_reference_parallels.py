#!/usr/bin/env python3
"""
Zone B: Optimized algorithmic parallel passage cross-reference

Uses efficient indexing and early filtering to make the search practical.
Searches for exact and near-exact matches (distance 0-1) to identify true parallels.
"""

import argparse
import csv
import json
import logging
import math
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import hashlib

from omegaconf import OmegaConf

logger = logging.getLogger(__name__)


@dataclass
class SequenceMatch:
    """Result of matching a Horley passage against corpus sequences."""
    passage_id: str
    passage_sequence: list[str]
    tablet_id: str
    tablet_sequence: list[str]
    start_position: int
    distance: int
    similarity: float


def levenshtein_distance(s1: list[str], s2: list[str]) -> int:
    """Compute Levenshtein distance between code sequences."""
    if len(s1) == 0:
        return len(s2)
    if len(s2) == 0:
        return len(s1)
    
    prev_row = list(range(len(s2) + 1))
    for i, code1 in enumerate(s1):
        curr_row = [i + 1]
        for j, code2 in enumerate(s2):
            cost = 0 if code1 == code2 else 1
            curr_row.append(min(
                curr_row[-1] + 1,
                prev_row[j + 1] + 1,
                prev_row[j] + cost,
            ))
        prev_row = curr_row
    
    return prev_row[-1]


def code_fingerprint(codes: list[str]) -> str:
    """
    Create a fingerprint of a code sequence for quick matching.
    Uses hash of sorted unique codes and length.
    """
    unique_codes = sorted(set(codes))
    fp = f"{len(codes)}:" + ",".join(unique_codes)
    return hashlib.md5(fp.encode()).hexdigest()[:8]


def parse_horley_sequence(seq_str: str) -> list[str]:
    """Parse Horley glyph sequence string to list of codes."""
    seq_str = seq_str.replace("!", "")
    import re
    codes = re.split(r'[.\-]', seq_str)
    # Strip inline tab-comments (e.g. "711v\t# corrected Barthel" → "711v")
    codes = [c.split('\t')[0].split('#')[0].strip() for c in codes]
    codes = [c for c in codes if c]
    return codes


def load_horley_passages(csv_path: Path) -> dict[str, list[str]]:
    """Load Horley passages from CSV."""
    passages = {}
    with csv_path.open('r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            passage_id = f"H{row['ID'].zfill(3)}"
            sequence = parse_horley_sequence(row['Sequence'])
            if sequence:  # Only keep non-empty sequences
                passages[passage_id] = sequence
    
    logger.info(f"Loaded {len(passages)} Horley passages")
    return passages


def load_corpus_codes(corpus_dir: Path) -> dict[str, list[str]]:
    """Load Barthel codes per tablet from corpus JSON files.

    Returns {tablet_id: [code, ...]} for all tablets that load successfully.
    Kept separate from indexing so the permutation test can shuffle codes
    without re-reading disk on every iteration.
    """
    codes_per_tablet: dict[str, list[str]] = {}
    for corpus_file in sorted(corpus_dir.glob("[A-Z].json")):
        tablet_id = corpus_file.stem
        try:
            data = json.loads(corpus_file.read_text())
            glyphs = data.get("glyphs", []) if isinstance(data, dict) else data
            codes = []
            for glyph in glyphs:
                code = glyph.get("barthel_code") or glyph.get("horley_code")
                if code:
                    codes.append(str(code))
            if codes:
                codes_per_tablet[tablet_id] = codes
        except Exception as e:
            logger.warning(f"Failed to load {corpus_file}: {e}")
    return codes_per_tablet


def build_corpus_index_from_codes(
    codes_per_tablet: dict[str, list[str]],
    max_seq_length: int = 8,
) -> dict:
    """Build the fingerprint/length index from pre-loaded per-tablet code lists.

    Accepts the dict returned by :func:`load_corpus_codes` (or a shuffled copy),
    so callers can avoid re-reading disk on every permutation iteration.
    """
    sequences = []
    by_length: dict[int, list] = defaultdict(list)
    by_fingerprint: dict[str, list] = defaultdict(list)

    for tablet_id, codes in codes_per_tablet.items():
        for seq_len in range(1, min(max_seq_length + 1, len(codes) + 1)):
            for start_pos in range(len(codes) - seq_len + 1):
                sequence = codes[start_pos:start_pos + seq_len]
                idx_data = (tablet_id, sequence, start_pos + 1)
                sequences.append(idx_data)
                by_length[seq_len].append(idx_data)
                by_fingerprint[code_fingerprint(sequence)].append(idx_data)

    return {
        "sequences": sequences,
        "by_length": dict(by_length),
        "by_fingerprint": by_fingerprint,
    }


def build_corpus_index(corpus_dir: Path, max_seq_length: int = 8) -> dict:
    """Build an efficient index of corpus sequences.

    Returns a dictionary with:
    - "sequences": list of (tablet, sequence, start_pos)
    - "by_length": dict mapping length -> sequences of that length
    - "by_fingerprint": dict mapping fingerprint -> sequences
    """
    codes_per_tablet = load_corpus_codes(corpus_dir)
    index = build_corpus_index_from_codes(codes_per_tablet, max_seq_length)
    total_seqs = len(index["sequences"])
    logger.info(
        f"Indexed corpus: {len(index['by_length'])} length bins, "
        f"{len(index['by_fingerprint'])} fingerprints, {total_seqs} total sequences"
    )
    return index


def run_permutation_test(
    horley_passages: dict[str, list[str]],
    corpus_codes: dict[str, list[str]],
    observed_count: int,
    n_permutations: int,
    threshold: int = 1,
    length_tolerance: int = 2,
    max_seq_length: int = 8,
    seed: int | None = None,
) -> dict:
    """Shuffle corpus codes N times and count multi-tablet passages each time.

    The null model shuffles each tablet's code sequence independently, preserving
    per-tablet vocabulary and length.  This tests whether the observed parallel-
    passage count exceeds what random ordering produces.

    Parameters
    ----------
    horley_passages:
        Dict of {passage_id: [codes]} as returned by load_horley_passages.
    corpus_codes:
        Dict of {tablet_id: [codes]} as returned by load_corpus_codes.
    observed_count:
        Multi-tablet passage count from the real (unshuffled) run.
    n_permutations:
        Number of shuffle iterations.
    threshold, length_tolerance, max_seq_length:
        Same as the main cross-reference call.
    seed:
        Optional RNG seed for reproducibility.

    Returns
    -------
    dict with keys: observed, n_permutations, permuted_counts, permuted_mean,
    permuted_std, permuted_max, z_score, p_value, n_exceeding_observed.
    """
    rng = random.Random(seed)
    permuted_counts: list[int] = []

    log_interval = max(1, n_permutations // 10)

    for i in range(1, n_permutations + 1):
        # Shuffle each tablet's codes independently
        shuffled: dict[str, list[str]] = {
            tid: rng.sample(codes, len(codes))
            for tid, codes in corpus_codes.items()
        }

        perm_index = build_corpus_index_from_codes(shuffled, max_seq_length)

        all_matches: list[SequenceMatch] = []
        for pid, seq in horley_passages.items():
            if seq:
                all_matches.extend(
                    find_matches_fast(pid, seq, perm_index, threshold, length_tolerance)
                )

        multi = group_multi_tablet_passages(all_matches, min_variants=2)
        permuted_counts.append(len(multi))

        if i % log_interval == 0 or i == n_permutations:
            running_mean = sum(permuted_counts) / len(permuted_counts)
            logger.info(
                "  permutation %d/%d — count=%d  running mean=%.2f",
                i, n_permutations, permuted_counts[-1], running_mean,
            )

    mean = sum(permuted_counts) / len(permuted_counts)
    variance = sum((x - mean) ** 2 for x in permuted_counts) / len(permuted_counts)
    std = math.sqrt(variance)
    n_exceeding = sum(1 for c in permuted_counts if c >= observed_count)
    p_value = n_exceeding / n_permutations
    z_score = (observed_count - mean) / std if std > 0 else float("inf")

    logger.info("")
    logger.info("=" * 60)
    logger.info("Permutation Test Results")
    logger.info("=" * 60)
    logger.info("  Observed multi-tablet passages : %d", observed_count)
    logger.info("  Permuted mean ± std            : %.2f ± %.2f", mean, std)
    logger.info("  Permuted max                   : %d", max(permuted_counts))
    logger.info("  z-score                        : %.2f", z_score)
    logger.info(
        "  Empirical p-value (>= observed): %d/%d = %.4f",
        n_exceeding, n_permutations, p_value,
    )
    if p_value == 0.0:
        logger.info("  p = 0 — observed exceeds all %d permutations", n_permutations)
    logger.info("=" * 60)

    return {
        "observed": observed_count,
        "n_permutations": n_permutations,
        "permuted_counts": permuted_counts,
        "permuted_mean": round(mean, 4),
        "permuted_std": round(std, 4),
        "permuted_max": max(permuted_counts),
        "z_score": round(z_score, 4) if math.isfinite(z_score) else None,
        "p_value": p_value,
        "n_exceeding_observed": n_exceeding,
    }


def find_matches_fast(
    passage_id: str,
    passage_sequence: list[str],
    corpus_index: dict,
    threshold: int = 1,
    length_tolerance: int = 2,
) -> list[SequenceMatch]:
    """
    Efficiently find corpus matches for a passage.
    
    Uses fingerprint indexing to avoid comparing all sequences.
    """
    matches = []
    target_len = len(passage_sequence)
    target_fp = code_fingerprint(passage_sequence)
    
    # Strategy 1: Check sequences with same fingerprint (likely high similarity)
    candidates_by_fp = corpus_index["by_fingerprint"].get(target_fp, [])
    
    # Strategy 2: Check sequences of similar length
    for seq_len in range(max(1, target_len - length_tolerance), target_len + length_tolerance + 1):
        candidates_by_fp.extend(corpus_index["by_length"].get(seq_len, []))
    
    # Deduplicate candidates
    seen = set()
    unique_candidates = []
    for tablet_id, corpus_seq, start_pos in candidates_by_fp:
        key = (tablet_id, start_pos, len(corpus_seq))
        if key not in seen:
            seen.add(key)
            unique_candidates.append((tablet_id, corpus_seq, start_pos))
    
    # Compute distances only for viable candidates
    for tablet_id, corpus_seq, start_pos in unique_candidates:
        distance = levenshtein_distance(passage_sequence, corpus_seq)
        
        if distance <= threshold:
            similarity = 1.0 - (distance / max(len(passage_sequence), len(corpus_seq)))
            matches.append(SequenceMatch(
                passage_id=passage_id,
                passage_sequence=passage_sequence,
                tablet_id=tablet_id,
                tablet_sequence=corpus_seq,
                start_position=start_pos,
                distance=distance,
                similarity=similarity,
            ))
    
    return matches


def group_multi_tablet_passages(
    all_matches: list[SequenceMatch],
    min_variants: int = 2,
) -> dict[tuple, list[SequenceMatch]]:
    """Group matches by canonical sequence."""
    groups = defaultdict(list)
    
    for match in all_matches:
        canonical = tuple(match.passage_sequence)
        groups[canonical].append(match)
    
    # Filter to multi-tablet passages
    multi_tablet = {
        k: v for k, v in groups.items()
        if len(set(m.tablet_id for m in v)) >= min_variants
    }
    
    logger.info(
        f"Found {len(groups)} unique sequences, "
        f"{len(multi_tablet)} with >= {min_variants} tablet attestations"
    )
    
    return multi_tablet


def infer_stratum_from_tablets(tablet_id: str, tablets: dict) -> str:
    """Infer temporal stratum from tablets.json metadata."""
    info = tablets.get(tablet_id, {})
    # tablets.json uses 'temporal_cluster'; treat '?' and 'unknown' as absent
    # so we fall through to the radiocarbon date heuristic.
    cluster = info.get("temporal_cluster", "")
    if cluster and cluster not in ("?", "unknown"):
        return cluster
    # Derive from radiocarbon date ranges (authoritative where cluster is absent).
    rc_min = info.get("radiocarbon_date_min", 0)
    rc_max = info.get("radiocarbon_date_max", 9999)
    if rc_max <= 1600:
        return "pre_contact"
    if rc_min >= 1650:
        return "post_contact"
    return "undated"


def _barthel_family_bucket(code: str) -> str:
    """Return the iconographic family for a sign code.

    Delegates to ``barthel_families.json`` (Barthel 1958, Fischer 1997)
    rather than arithmetic century-block derivation.
    Returns ``'unknown'`` for codes absent from the lookup.
    """
    try:
        from hackingrongo.data.passage_alignment import _get_family
        return _get_family(code)
    except Exception:
        pass
    # Fallback: direct JSON load
    try:
        import json as _json
        from pathlib import Path as _Path
        _p = _Path(__file__).resolve().parents[1] / "data" / "catalog" / "barthel_families.json"
        if _p.exists():
            _raw = _json.loads(_p.read_text(encoding="utf-8"))
            _fam_map = {k: v for k, v in _raw.items() if not k.startswith("_")}
            digits = "".join(c for c in code if c.isdigit())
            return (
                _fam_map.get(code)
                or _fam_map.get(digits.zfill(3))
                or _fam_map.get(digits)
                or "unknown"
            )
    except Exception:
        pass
    return "unknown"


def _compute_diachronic_changes(attestations: list[dict]) -> list[dict]:
    """Compute sign-level changes between pre- and post-contact attestation forms."""
    from collections import Counter

    pre_forms = [a["form"] for a in attestations if a.get("stratum") == "pre_contact"]
    post_forms = [a["form"] for a in attestations if a.get("stratum") == "post_contact"]
    if not pre_forms or not post_forms:
        return []

    def _consensus(forms: list[list[str]]) -> list[str]:
        max_len = max(len(f) for f in forms)
        result = []
        for i in range(max_len):
            at_pos = [f[i] for f in forms if i < len(f)]
            if at_pos:
                result.append(Counter(at_pos).most_common(1)[0][0])
        return result

    pre_cons = _consensus(pre_forms)
    post_cons = _consensus(post_forms)

    changes = []
    for i in range(min(len(pre_cons), len(post_cons))):
        if pre_cons[i] == post_cons[i]:
            continue
        n_consistent = sum(
            1 for f in post_forms
            if i < len(f) and f[i] == post_cons[i]
        )
        changes.append({
            "position": i,
            "pre_contact_sign": pre_cons[i],
            "post_contact_sign": post_cons[i],
            "change_type": "substitution",
            "n_tablets_consistent": n_consistent,
            "is_holy_grail_candidate": n_consistent >= 2,
            "crosses_barthel_family": (
                _barthel_family_bucket(pre_cons[i]) != "unknown"
                and _barthel_family_bucket(post_cons[i]) != "unknown"
                and _barthel_family_bucket(pre_cons[i])
                != _barthel_family_bucket(post_cons[i])
            ),
        })
    return changes


def generate_json(multi_tablet_passages: dict, tablets: dict) -> dict:
    """Generate parallel_variants JSON in the schema expected by passage_report.py."""
    passages = []

    for idx, (canonical_form, matches) in enumerate(multi_tablet_passages.items(), start=1):
        tablet_ids = sorted(set(m.tablet_id for m in matches))
        tablet_str = "".join(tablet_ids)
        passage_id = f"P{idx:03d}_{tablet_str}"

        # attestations — keyed as passage_report.py expects
        attestations = sorted(
            [
                {
                    "form": match.tablet_sequence,
                    "tablet": match.tablet_id,
                    "stratum": infer_stratum_from_tablets(match.tablet_id, tablets),
                    "side": "a",
                    "start_position": match.start_position,
                }
                for match in matches
            ],
            key=lambda a: a["tablet"],
        )

        diachronic_changes = _compute_diachronic_changes(attestations)

        strata = {a["stratum"] for a in attestations}
        has_diachronic = "pre_contact" in strata and "post_contact" in strata
        n_tablets = len(tablet_ids)
        n_holy = sum(1 for c in diachronic_changes if c.get("is_holy_grail_candidate"))
        interest_score = round(
            min(1.0, n_tablets * 0.15 + (0.4 if has_diachronic else 0.0) + n_holy * 0.3),
            4,
        )

        passages.append({
            "passage_id": passage_id,
            "canonical_form": list(canonical_form),
            "n_tablets": n_tablets,
            "attestations": attestations,
            "diachronic_changes": diachronic_changes,
            "interest_score": interest_score,
        })

    result = {
        "_schema_version": "2.0",
        "_provenance": "Auto-generated by cross_reference_parallels.py",
        "_description": "Multi-tablet parallel passages — schema compatible with passage_report.py",
        "_method": "Index-based search with Levenshtein distance matching (threshold=1)",
        "_discovery_stats": {
            "multi_tablet_passages_found": len(passages),
            "unique_sequences": len(multi_tablet_passages),
        },
        "passages": passages,
    }

    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Zone B: Optimized parallel passage cross-reference (fast)"
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/parallels/horley_parallels.csv"),
        help="Input Horley passages CSV",
    )
    parser.add_argument(
        "--corpus",
        type=Path,
        default=Path("data/corpus"),
        help="Corpus directory",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("conf/config.yaml"),
        help="Hydra config",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/parallels/parallel_variants_auto.json"),
        help="Output parallel_variants JSON",
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=1,
        help="Levenshtein distance threshold (1=exact or 1-char substitution)",
    )
    parser.add_argument(
        "--length-tolerance",
        type=int,
        default=2,
        help="Maximum length difference to compare",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    parser.add_argument(
        "--tablets",
        type=Path,
        default=Path("data/metadata/tablets.json"),
        help="Tablet metadata JSON with temporal stratum assignments "
             "(default: data/metadata/tablets.json)",
    )
    parser.add_argument(
        "--permutation-test",
        type=int,
        default=0,
        metavar="N",
        help=(
            "Run a permutation test with N shuffles of the corpus. "
            "Each shuffle independently randomises the code order within every "
            "tablet, then reruns the same cross-reference. Reports the observed "
            "multi-tablet passage count against the permuted null distribution "
            "(mean, std, z-score, empirical p-value). Results appended to the "
            "output JSON under 'permutation_test'. N=1000 is a reasonable default."
        ),
    )
    parser.add_argument(
        "--permutation-seed",
        type=int,
        default=None,
        metavar="SEED",
        help="RNG seed for reproducible permutation tests (default: unseeded).",
    )
    parser.add_argument(
        "--seed", type=int, default=20260606, metavar="INT",
        help="Global RNG seed for reproducibility (default: 20260606).",
    )

    args = parser.parse_args()
    from hackingrongo.repro import set_global_seed
    set_global_seed(args.seed)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(name)s: %(levelname)s: %(message)s",
    )
    
    # Load config
    cfg = OmegaConf.load(args.config)

    # Build tablet metadata dict: start from tablets.json, then overlay corpus
    # cluster values (the corpus JSON is the authoritative source of stratum).
    tablets_meta: dict = {}
    if args.tablets.exists():
        try:
            tablets_meta = json.loads(args.tablets.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning("Could not load tablets.json (%s): %s", args.tablets, exc)
    for cf in sorted(args.corpus.glob("[A-Z].json")):
        tid = cf.stem
        try:
            corpus_data = json.loads(cf.read_text(encoding="utf-8"))
            cluster = corpus_data.get("cluster", "")
            if cluster:
                if tid not in tablets_meta:
                    tablets_meta[tid] = {}
                tablets_meta[tid]["temporal_cluster"] = cluster
        except Exception as exc:
            logger.warning(
                "Skipping tablet metadata overlay for %s due to parse/load error: %s",
                cf,
                exc,
            )

    # Load passages
    horley_passages = load_horley_passages(args.input)

    # Load corpus codes once (reused by both the main run and permutation test)
    corpus_codes = load_corpus_codes(args.corpus)
    corpus_index = build_corpus_index_from_codes(corpus_codes, max_seq_length=8)
    logger.info(
        "Indexed corpus: %d length bins, %d fingerprints, %d total sequences",
        len(corpus_index["by_length"]),
        len(corpus_index["by_fingerprint"]),
        len(corpus_index["sequences"]),
    )

    # Cross-reference
    logger.info(
        f"Cross-referencing {len(horley_passages)} Horley passages "
        f"against corpus (threshold={args.threshold})..."
    )

    all_matches = []
    matched_count = 0

    for i, (passage_id, sequence) in enumerate(horley_passages.items(), start=1):
        if len(sequence) == 0:
            continue

        matches = find_matches_fast(
            passage_id,
            sequence,
            corpus_index,
            threshold=args.threshold,
            length_tolerance=args.length_tolerance,
        )

        all_matches.extend(matches)

        if len(matches) > 0:
            matched_count += 1
            tablets = set(m.tablet_id for m in matches)
            logger.info(
                f"[{i}/{len(horley_passages)}] {passage_id} → "
                f"found on {len(tablets)} tablets ({len(matches)} total matches)"
            )

        if i % 50 == 0:
            logger.info(f"Progress: {i}/{len(horley_passages)} passages processed")

    logger.info(f"\nFound {len(all_matches)} total matches")

    # Group by canonical sequence
    multi_tablet = group_multi_tablet_passages(all_matches, min_variants=2)
    observed_count = len(multi_tablet)

    # Generate JSON
    result = generate_json(multi_tablet, tablets_meta)
    result["_discovery_stats"]["total_horley_passages"] = len(horley_passages)
    result["_discovery_stats"]["passages_with_matches"] = matched_count
    result["_parameters"] = {
        "levenshtein_threshold": args.threshold,
        "length_tolerance": args.length_tolerance,
    }

    # Permutation test
    if args.permutation_test > 0:
        logger.info(
            "\nRunning permutation test (%d shuffles)…", args.permutation_test
        )
        perm_result = run_permutation_test(
            horley_passages=horley_passages,
            corpus_codes=corpus_codes,
            observed_count=observed_count,
            n_permutations=args.permutation_test,
            threshold=args.threshold,
            length_tolerance=args.length_tolerance,
            seed=args.permutation_seed,
        )
        result["permutation_test"] = perm_result

    # Write output
    from hackingrongo.provenance import stamp
    stamp(result, seed=args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))

    logger.info(f"\n✓ Output written to {args.output}")
    logger.info(
        f"\nFinal Summary:\n"
        f"  Input passages: {len(horley_passages)}\n"
        f"  Passages with matches: {matched_count}\n"
        f"  Total matches found: {len(all_matches)}\n"
        f"  Multi-tablet passages: {result['_discovery_stats']['multi_tablet_passages_found']}\n"
        f"  Unique canonical forms: {result['_discovery_stats']['unique_sequences']}"
    )
    if args.permutation_test > 0:
        pt = result["permutation_test"]
        logger.info(
            "  Permutation test: observed=%d  mean=%.2f  z=%.2f  p=%.4f",
            pt["observed"], pt["permuted_mean"], pt["z_score"] or 0, pt["p_value"],
        )


if __name__ == "__main__":
    main()
