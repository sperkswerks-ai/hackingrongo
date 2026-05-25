"""hsp_analysis.py — Hidden Subgroup Problem (HSP) group structure analysis.

Extracts the substitution group structure from parallel rongorongo passages
to assess the feasibility of quantum HSP algorithms (Shor, Hallgren) for
rongorongo decipherment.

Theory
------
If rongorongo passages are related by scribal substitution patterns, these
substitutions form a group G (or sub-semigroup) acting on the sign inventory.
The quantum HSP algorithm can find the generators of a hidden subgroup H ≤ G
in polynomial time when G is abelian (Shor-style) or certain non-abelian cases.

This script computes:

1. **Substitution table** — for each pair of parallel passage variants at the
   same position, record the substitution pair (sign_a → sign_b).  Consistent
   substitutions (same pair occurring in multiple passages) are strong evidence.

2. **Group closure test** — given observed substitutions A→B and B→C, test
   whether A→C also appears.  Closure under composition is a necessary condition
   for the substitutions to form a group.

3. **Generator set** — find a minimal set of substitution pairs that generate
   all observed patterns via composition.

4. **HSP feasibility** — assess whether the substitution group size, closure
   ratio, and structure are compatible with a polynomial-time quantum HSP
   attack.

Usage
-----
    python scripts/hsp_analysis.py \\
        --parallels data/parallels/parallel_variants.json \\
        --output    outputs/analysis/hsp_analysis.json

    python scripts/hsp_analysis.py --smoke-test
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_parallel_variants(path: Path) -> list[dict]:
    """Load parallel passage variants from JSON.

    Supports both ``parallel_variants.json`` (the primary format produced
    by ``transform_parallels.py``) and ``parallel_variants_auto.json``.

    Returns
    -------
    list[dict]
        Each dict has keys ``passage_id``, ``canonical_form``,
        and ``variants`` (list of variant dicts with ``form`` and
        ``tablet_id`` keys).
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        passages = data
    elif isinstance(data, dict):
        passages = data.get("passages", data.get("parallel_passages", []))
    else:
        passages = []
    log.info("Loaded %d passage records from %s.", len(passages), path.name)
    return passages


# ---------------------------------------------------------------------------
# Substitution extraction
# ---------------------------------------------------------------------------

def _extract_substitutions(
    passages: list[dict],
) -> dict[tuple[str, str], int]:
    """Extract directional substitution pairs from all variant pairs.

    For each passage with n_variants >= 2, compare every pair of variants
    at each aligned position.  Record both (a→b) and (b→a).

    Parameters
    ----------
    passages : list[dict]
        Parallel passage records as returned by :func:`_load_parallel_variants`.

    Returns
    -------
    dict[tuple[str, str], int]
        Maps ``(sign_a, sign_b)`` → co-occurrence count across all aligned
        position pairs.  Only non-identity substitutions are included.
    """
    substitutions: dict[tuple[str, str], int] = defaultdict(int)
    multi_tablet_count = 0

    for passage in passages:
        variants = passage.get("variants", [])
        if len(variants) < 2:
            continue
        multi_tablet_count += 1

        # Align variant forms positionally.
        forms: list[list[str]] = [
            list(v.get("form", v.get("signs", []))) for v in variants
        ]

        for v1_idx, v2_idx in combinations(range(len(forms)), 2):
            form1 = forms[v1_idx]
            form2 = forms[v2_idx]
            min_len = min(len(form1), len(form2))
            for pos in range(min_len):
                a = str(form1[pos])
                b = str(form2[pos])
                if a != b:
                    substitutions[(a, b)] += 1
                    substitutions[(b, a)] += 1  # both directions

    log.info(
        "Multi-tablet passages: %d  Unique substitution pairs: %d",
        multi_tablet_count, len(substitutions),
    )
    return dict(substitutions)


# ---------------------------------------------------------------------------
# Group closure analysis
# ---------------------------------------------------------------------------

def _build_substitution_graph(
    substitutions: dict[tuple[str, str], int],
    min_count: int = 1,
) -> dict[str, set[str]]:
    """Build a directed substitution graph: sign → {signs it can substitute to}.

    Parameters
    ----------
    substitutions : dict
    min_count : int
        Minimum co-occurrence count for an edge to be included.

    Returns
    -------
    dict[str, set[str]]
    """
    graph: dict[str, set[str]] = defaultdict(set)
    for (a, b), count in substitutions.items():
        if count >= min_count:
            graph[a].add(b)
    return dict(graph)


def _test_closure(
    graph: dict[str, set[str]],
) -> dict[str, Any]:
    """Test group closure: for each A→B and B→C, check if A→C exists.

    Parameters
    ----------
    graph : dict[str, set[str]]
        Directed substitution adjacency graph.

    Returns
    -------
    dict with keys:
        ``total_compositions``: number of (A→B, B→C) pairs tested.
        ``closed_compositions``: number where A→C also present.
        ``missing_compositions``: list of (A, B, C) triples where A→C is absent.
        ``closure_ratio``: fraction of compositions that are closed.
    """
    total = 0
    closed = 0
    missing: list[tuple[str, str, str]] = []

    all_signs = set(graph.keys()) | {b for s in graph.values() for b in s}
    for a in sorted(all_signs):
        a_targets = graph.get(a, set())
        for b in sorted(a_targets):
            b_targets = graph.get(b, set())
            for c in sorted(b_targets):
                if c == a:
                    continue  # ignore cycles a→b→a
                total += 1
                if c in a_targets:
                    closed += 1
                else:
                    missing.append((a, b, c))

    ratio = closed / max(total, 1)
    return {
        "total_compositions": total,
        "closed_compositions": closed,
        "missing_compositions": [(a, b, c) for a, b, c in missing[:50]],  # cap list
        "missing_count": len(missing),
        "closure_ratio": round(ratio, 4),
    }


# ---------------------------------------------------------------------------
# Generator set
# ---------------------------------------------------------------------------

def _find_generators(
    graph: dict[str, set[str]],
    max_generators: int = 20,
) -> list[tuple[str, str]]:
    """Find a minimal generating set for the observed substitutions.

    Uses a greedy cover algorithm: repeatedly choose the substitution pair
    that covers the most uncovered pairs via 1-step composition.

    Parameters
    ----------
    graph : dict[str, set[str]]
    max_generators : int
        Maximum number of generator pairs to return.

    Returns
    -------
    list[tuple[str, str]]
        Generator pairs ``(a, b)`` meaning "a can substitute b".
    """
    # Collect all edges.
    all_edges: set[tuple[str, str]] = set()
    for a, targets in graph.items():
        for b in targets:
            all_edges.add((a, b))

    if not all_edges:
        return []

    covered: set[tuple[str, str]] = set()
    generators: list[tuple[str, str]] = []
    remaining = set(all_edges)

    for _ in range(min(max_generators, len(all_edges))):
        if not remaining:
            break
        # Choose edge that covers the most uncovered pairs via composition.
        best_edge: tuple[str, str] | None = None
        best_cover: int = 0
        for (a, b) in remaining:
            # Direct coverage.
            new_covered = {(a, b)} - covered
            # One-step composition: a→b→c for c in graph[b], check if (a,c) covered.
            for c in graph.get(b, set()):
                if (a, c) in remaining and (a, c) not in covered:
                    new_covered.add((a, c))
            if len(new_covered) > best_cover:
                best_cover = len(new_covered)
                best_edge = (a, b)

        if best_edge is None:
            break
        generators.append(best_edge)
        a, b = best_edge
        covered.add((a, b))
        for c in graph.get(b, set()):
            covered.add((a, c))
        remaining -= covered

    return generators


# ---------------------------------------------------------------------------
# HSP feasibility assessment
# ---------------------------------------------------------------------------

def _assess_hsp_feasibility(
    n_signs: int,
    substitutions: dict[tuple[str, str], int],
    closure_result: dict,
    generators: list[tuple[str, str]],
) -> dict[str, Any]:
    """Assess feasibility of a quantum HSP attack on rongorongo decipherment.

    Parameters
    ----------
    n_signs : int  Number of distinct signs in the substitution table.
    substitutions : dict  Substitution counts.
    closure_result : dict  Output of :func:`_test_closure`.
    generators : list  Output of :func:`_find_generators`.

    Returns
    -------
    dict
        ``feasibility``: "high" | "moderate" | "low".
        ``reasoning``: human-readable explanation.
        ``group_size_estimate``: estimated order of the substitution group.
        ``n_generators``: minimal generator set size.
    """
    closure_ratio = closure_result["closure_ratio"]
    n_gens = len(generators)
    n_observed = len(substitutions) // 2  # undirected pairs

    # Group size estimate: if generators are independent, |G| ~ 2^n_gens
    # For symmetric groups S_n, |G| = n!, so log2(|G|) ~ n*log2(n)
    group_size_estimate = min(2 ** n_gens, n_signs * (n_signs - 1))

    # Feasibility heuristics:
    #  - High closure ratio (>= 0.7) → strong group structure
    #  - |G| = 2^k (power of 2) → abelian → Shor's algorithm applicable
    #  - Small generator set → shallow quantum circuit needed
    #  - Many observed substitutions → rich structure to exploit
    is_power_of_2 = group_size_estimate > 0 and (group_size_estimate & (group_size_estimate - 1)) == 0

    if closure_ratio >= 0.7 and n_gens <= 8 and n_observed >= 20:
        feasibility = "high"
        reasoning = (
            f"Strong group structure detected (closure ratio={closure_ratio:.2f}, "
            f"{n_gens} generators, {n_observed} observed pairs). "
            "The substitution group appears well-structured; quantum HSP algorithms "
            "(abelian: Shor-style; non-abelian: Regev/Hallgren) should find "
            "hidden subgroup generators efficiently."
        )
    elif closure_ratio >= 0.4 and n_observed >= 10:
        feasibility = "moderate"
        reasoning = (
            f"Partial group structure (closure ratio={closure_ratio:.2f}, "
            f"{n_gens} generators). "
            "The substitutions do not fully close under composition — possible causes: "
            "incomplete parallel passage coverage, scribe-specific variation, "
            "or the underlying structure is a semigroup rather than a group. "
            "Quantum HSP may provide advantage with additional data."
        )
    else:
        feasibility = "low"
        reasoning = (
            f"Weak group structure (closure ratio={closure_ratio:.2f}, "
            f"{n_observed} observed pairs). "
            "Insufficient evidence for a well-defined substitution group. "
            "Quantum HSP algorithms require a group structure that is not "
            "clearly present in the current parallel passage data. "
            "Expanding the parallel passage corpus (via Horley 2021 Appendix A) "
            "is recommended before re-running this analysis."
        )

    if is_power_of_2 and feasibility == "high":
        reasoning += (
            f"  Group order {group_size_estimate} is a power of 2 → "
            "likely abelian 2-group → Shor's quantum algorithm directly applicable."
        )

    return {
        "feasibility": feasibility,
        "reasoning": reasoning,
        "group_size_estimate": group_size_estimate,
        "n_generators": n_gens,
        "n_observed_pairs": n_observed,
        "closure_ratio": closure_ratio,
        "is_likely_abelian": bool(is_power_of_2),
    }


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def run_hsp_analysis(
    parallels_path: Path,
    min_count: int = 1,
) -> dict[str, Any]:
    """Full HSP group structure analysis pipeline.

    Parameters
    ----------
    parallels_path : Path
    min_count : int
        Minimum co-occurrence count for a substitution to be included.

    Returns
    -------
    dict
        Complete analysis results.
    """
    passages = _load_parallel_variants(parallels_path)
    substitutions = _extract_substitutions(passages)

    if not substitutions:
        log.warning("No substitution pairs found — parallel passages may lack multi-tablet variants.")
        return {
            "error": "no_substitutions_found",
            "n_passages": len(passages),
            "n_multi_tablet_passages": 0,
            "substitution_table": {},
            "closure_result": {},
            "generators": [],
            "hsp_feasibility": _assess_hsp_feasibility(0, {}, {"closure_ratio": 0.0, "total_compositions": 0, "closed_compositions": 0, "missing_compositions": [], "missing_count": 0}, []),
        }

    graph = _build_substitution_graph(substitutions, min_count=min_count)
    closure_result = _test_closure(graph)
    generators = _find_generators(graph)

    n_signs = len(set(k for pair in substitutions for k in pair))
    feasibility = _assess_hsp_feasibility(n_signs, substitutions, closure_result, generators)

    # Top-50 most frequent substitution pairs (for reporting).
    top_pairs = sorted(
        [(list(pair), count) for pair, count in substitutions.items()],
        key=lambda x: -x[1],
    )[:50]

    result = {
        "n_passages": len(passages),
        "n_multi_tablet_passages": sum(
            1 for p in passages if len(p.get("variants", [])) >= 2
        ),
        "n_distinct_signs_in_substitutions": n_signs,
        "n_substitution_pairs_total": len(substitutions) // 2,
        "top_substitution_pairs": [
            {"pair": pair, "count": count} for pair, count in top_pairs
        ],
        "closure_result": closure_result,
        "generators": [list(g) for g in generators],
        "hsp_feasibility": feasibility,
    }
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="HSP group structure analysis for rongorongo parallel passages.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--parallels", type=Path, default=None, metavar="JSON",
        help="Path to parallel_variants.json (default: auto from config).",
    )
    p.add_argument(
        "--output", type=Path, default=None, metavar="JSON",
        help="Output JSON path (default: outputs/analysis/hsp_analysis.json).",
    )
    p.add_argument(
        "--min-count", type=int, default=1, metavar="N",
        help="Minimum substitution co-occurrence count (default: 1).",
    )
    p.add_argument(
        "--smoke-test", action="store_true",
        help="Use the first 20 passages only (fast wiring check).",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    # ── Resolve paths ─────────────────────────────────────────────────────────
    try:
        from omegaconf import OmegaConf
        cfg = OmegaConf.load(PROJECT_ROOT / "conf" / "config.yaml")
    except Exception:
        cfg = None

    parallels_path = args.parallels
    if parallels_path is None:
        for candidate in [
            PROJECT_ROOT / "data" / "parallels" / "parallel_variants.json",
            PROJECT_ROOT / "data" / "parallels" / "parallel_variants_auto.json",
        ]:
            if candidate.exists():
                parallels_path = candidate
                break
    if parallels_path is None or not parallels_path.exists():
        log.error(
            "parallel_variants.json not found. "
            "Pass --parallels or run transform_parallels.py first."
        )
        sys.exit(1)

    output_path = args.output or PROJECT_ROOT / "outputs" / "analysis" / "hsp_analysis.json"

    # ── Smoke-test: truncate passages ─────────────────────────────────────────
    if args.smoke_test:
        log.info("Smoke-test mode: using first 20 passages only.")

    # ── Run ───────────────────────────────────────────────────────────────────
    log.info("Loading parallel passages from %s …", parallels_path)
    if args.smoke_test:
        passages_raw = _load_parallel_variants(parallels_path)[:20]
        import tempfile, json as _json
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        ) as tmp:
            _json.dump(passages_raw, tmp)
            tmp_path = Path(tmp.name)
        results = run_hsp_analysis(tmp_path, min_count=args.min_count)
        tmp_path.unlink(missing_ok=True)
    else:
        results = run_hsp_analysis(parallels_path, min_count=args.min_count)

    # ── Print summary ─────────────────────────────────────────────────────────
    feas = results.get("hsp_feasibility", {})
    cl = results.get("closure_result", {})

    print(f"\n{'═' * 66}")
    print("  HSP Group Structure Analysis — Rongorongo Parallel Passages")
    print(f"{'═' * 66}")
    print(f"  Passages loaded          : {results.get('n_passages', 0)}")
    print(f"  Multi-tablet passages    : {results.get('n_multi_tablet_passages', 0)}")
    print(f"  Signs in substitutions   : {results.get('n_distinct_signs_in_substitutions', 0)}")
    print(f"  Unique substitution pairs: {results.get('n_substitution_pairs_total', 0)}")
    print()
    if cl:
        print(f"  Closure analysis:")
        print(f"    Compositions tested  : {cl.get('total_compositions', 0)}")
        print(f"    Closed compositions  : {cl.get('closed_compositions', 0)}")
        print(f"    Closure ratio        : {cl.get('closure_ratio', 0):.4f}")
    print()
    print(f"  Generators: {results.get('generators', [])[:10]} …")
    print()
    print(f"  HSP Feasibility: {feas.get('feasibility', 'unknown').upper()}")
    print(f"  {feas.get('reasoning', '')}")
    print()

    # ── Save ──────────────────────────────────────────────────────────────────
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    log.info("Results written to %s", output_path)


if __name__ == "__main__":
    main()
