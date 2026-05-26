#!/usr/bin/env python3
"""Audit Barthel-code-to-image resolution before Zone A training.

Usage
-----
python scripts/audit_image_mapping.py

Outputs
-------
outputs/analysis/audit_report.json

For each corpus token this script reports:
- resolution tier: exact / ref_fallback / corpus_fallback / missing
- whether fallback changed the resolved code relative to the original code
- merge_suspect flag (from barthel_catalog tafeln records)
- image quality flags (blank / elongated / solid)
- unknown placeholder code classification (separate from true missing-image bugs)
- for range tokens (e.g. "(10-20)!"), what image/code was substituted
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from omegaconf import OmegaConf
from PIL import Image

SIDE_TO_AB: dict[str, str] = {"r": "a", "v": "b", "a": "a", "b": "b"}
RANGE_RE = re.compile(r"^\(\d+\-\d+\)!$")


@dataclass
class CorpusToken:
    tablet_id: str
    stratum: str
    position: int
    barthel_code: str
    side_ab: str
    line: str
    seq_on_line: int
    corpus_key: str


@dataclass
class ResolveResult:
    tier: str
    image_path: str | None
    source: str | None
    resolved_code: str | None
    lookup_key: str | None
    fallback_code_changed: bool
    reason: str | None


def _clean_code(raw: str) -> str:
    clean = re.sub(r"[!?()\s]", "", raw)
    return clean


def _primary_key(raw: str) -> str:
    clean = _clean_code(raw)
    return clean.lstrip("0") or "0"


def _candidate_lookup_keys(raw: str) -> list[tuple[str, str]]:
    """Return candidate keys in dataset resolver order.

    Returns list[(key, reason)] where reason explains transformation.
    """
    key = _primary_key(raw)
    out: list[tuple[str, str]] = []
    out.append((key, "clean"))

    trimmed = re.sub(r"[A-Za-z]+$", "", key)
    if trimmed and trimmed != key:
        out.append((trimmed, "suffix_stripped"))

    for sep in (".", "-"):
        if sep in key:
            first = key.split(sep)[0].lstrip("0") or "0"
            if first and all(first != k for k, _ in out):
                out.append((first, "range_or_compound_first_component"))

    return out


def _is_unknown_placeholder_code(raw: str) -> bool:
    code = (raw or "").strip()
    if not code:
        return True
    if "?" in code:
        return True
    if code.startswith("000!"):
        return True
    return code in {"0", "000", "000!"}


def _extract_code_from_path(path: Path) -> str | None:
    stem = path.stem

    # Dataset index style: <prefix>_barthel_...
    if "_barthel_" in stem:
        prefix = stem.split("_barthel_")[0]
        primary = prefix.split("_")[0].lstrip("0") or "0"
        return primary

    # 3d crops style: ..._<code>
    parts = stem.rsplit("_", 1)
    if len(parts) == 2:
        return parts[1].lstrip("0") or "0"

    return None


def _build_code_index(paths: list[Path]) -> dict[str, Path]:
    index: dict[str, Path] = {}
    for p in sorted(paths):
        code = _extract_code_from_path(p)
        if not code:
            continue
        if code not in index:
            index[code] = p
        base = re.sub(r"[A-Za-z]+$", "", code)
        if base and base != code and base not in index:
            index[base] = p
    return index


def _is_positional_ref_estimate(path: Path) -> bool:
    stem = path.stem
    prefix = stem.split("_barthel_")[0] if "_barthel_" in stem else stem
    return re.fullmatch(r"\d+_\d+", prefix) is not None


def _load_corpus_tokens(corpus_dir: Path) -> list[CorpusToken]:
    tokens: list[CorpusToken] = []

    for path in sorted(corpus_dir.glob("[A-Z].json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        tablet_id = str(data["tablet_id"])
        stratum = str(data.get("cluster", "unknown"))

        grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        for g in sorted(data.get("glyphs", []), key=lambda x: int(x.get("position", 0))):
            side_ab = SIDE_TO_AB.get(str(g.get("side", "a")), "a")
            line = str(g.get("line", "01"))
            grouped[(side_ab, line)].append(g)

        for (side_ab, line), glyphs in grouped.items():
            for seq_on_line, g in enumerate(glyphs, start=1):
                position = int(g.get("position", 0))
                barthel_code = str(g.get("barthel_code", ""))
                corpus_key = f"{tablet_id}{side_ab}{line}-{seq_on_line:03d}"
                tokens.append(
                    CorpusToken(
                        tablet_id=tablet_id,
                        stratum=stratum,
                        position=position,
                        barthel_code=barthel_code,
                        side_ab=side_ab,
                        line=line,
                        seq_on_line=seq_on_line,
                        corpus_key=corpus_key,
                    )
                )

    return tokens


def _load_merge_suspects(barthel_catalog_path: Path) -> dict[str, dict[str, Any]]:
    if not barthel_catalog_path.exists():
        return {}

    raw = json.loads(barthel_catalog_path.read_text(encoding="utf-8"))
    records = raw.get("records", raw if isinstance(raw, list) else [])

    suspects: dict[str, dict[str, Any]] = {}
    for r in records:
        if not isinstance(r, dict):
            continue
        if not r.get("merge_suspect"):
            continue
        key = r.get("corpus_key")
        if not key:
            continue
        suspects[str(key)] = {
            "path": r.get("path"),
            "page": r.get("page"),
            "bbox": r.get("bbox"),
            "source": r.get("source"),
        }

    return suspects


def _load_exact_catalog_index(barthel_catalog_path: Path, glyphs_dir: Path) -> dict[str, dict[str, Any]]:
    if not barthel_catalog_path.exists():
        return {}

    raw = json.loads(barthel_catalog_path.read_text(encoding="utf-8"))
    records = raw.get("records", raw if isinstance(raw, list) else [])
    index: dict[str, dict[str, Any]] = {}
    for rec in records:
        if not isinstance(rec, dict):
            continue
        corpus_key = rec.get("corpus_key")
        rel_path = rec.get("path")
        if not corpus_key or not rel_path:
            continue
        abs_path = glyphs_dir / rel_path
        if not abs_path.exists():
            continue
        index[str(corpus_key)] = {
            "path": abs_path,
            "merge_suspect": bool(rec.get("merge_suspect", False)),
            "barthel_code": rec.get("barthel_code"),
            "source": rec.get("source"),
        }
    return index


def _build_missing_code_remediation(tokens: list[dict[str, Any]], top_n: int = 25) -> list[dict[str, Any]]:
    """Summarize true-missing codes to guide targeted extraction fixes."""
    missing_rows = [r for r in tokens if r.get("resolved_tier") == "missing"]
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in missing_rows:
        grouped[str(row.get("barthel_code", ""))].append(row)

    ranked = sorted(grouped.items(), key=lambda kv: len(kv[1]), reverse=True)
    out: list[dict[str, Any]] = []
    for code, rows in ranked[:top_n]:
        example_keys = [r.get("corpus_key") for r in rows[:5] if r.get("corpus_key")]
        tablets = sorted({str(r.get("tablet_id")) for r in rows if r.get("tablet_id")})
        out.append(
            {
                "barthel_code": code,
                "count": len(rows),
                "tablet_ids": tablets,
                "example_corpus_keys": example_keys,
            }
        )
    return out


def _write_missing_code_remediation_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Missing Code Remediation (True Missing Only)",
        "",
        "Top Barthel codes with unresolved images after excluding placeholder/unknown codes.",
        "",
        "| Barthel code | Count | Tablet IDs | Example corpus keys |",
        "|---|---:|---|---|",
    ]

    for row in rows:
        tablets = ", ".join(row.get("tablet_ids", []))
        examples = ", ".join(row.get("example_corpus_keys", []))
        lines.append(
            f"| {row.get('barthel_code', '')} | {int(row.get('count', 0))} | {tablets} | {examples} |"
        )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _is_valid_image(path: Path) -> dict[str, Any]:
    """Heuristic image quality checks for audit reporting.

    Returns dict with flags: valid, blank, elongated, solid,
    low_contrast_relief, inverted_candidate.
    """
    with Image.open(path) as im:
        arr = np.array(im.convert("L"), dtype=np.uint8)

    # Ink pixels are non-white. Threshold is intentionally conservative.
    ink = arr < 245
    ink_ratio = float(np.mean(ink)) if ink.size else 0.0
    dark_ratio = float(np.mean(arr < 64)) if arr.size else 0.0
    mean = float(arr.mean()) if arr.size else 255.0
    std = float(arr.std()) if arr.size else 0.0
    p1, p99 = np.percentile(arr, [1, 99]) if arr.size else (255.0, 255.0)
    dynamic_range = float(p99 - p1)

    blank = ink_ratio < 0.003

    # Near-black, globally dark images are inversion candidates.
    inverted_candidate = (mean < 110.0 and dark_ratio > 0.75 and dynamic_range < 60.0)

    # 3D relief crops often have very subtle, valid structure on a nearly uniform
    # light background. They are not "solid fills" even though a naive non-white
    # threshold marks most pixels as foreground.
    low_contrast_relief = (
        not blank
        and not inverted_candidate
        and mean > 150.0
        and std < 8.0
        and dynamic_range < 30.0
    )

    # Reserve "solid" for genuinely filled / near-uniform images rather than
    # low-contrast reliefs.
    solid = (
        not blank
        and not inverted_candidate
        and not low_contrast_relief
        and ((ink_ratio > 0.995 and dynamic_range < 12.0) or (std < 1.5 and dynamic_range < 8.0))
    )

    elongated = False
    aspect_ratio = 0.0
    if ink.any():
        ys, xs = np.where(ink)
        h = int(ys.max() - ys.min() + 1)
        w = int(xs.max() - xs.min() + 1)
        if h > 0 and w > 0:
            aspect_ratio = float(max(h / w, w / h))
            elongated = aspect_ratio > 6.0

    valid = not (blank or solid or elongated)

    if blank:
        quality = "blank"
    elif elongated:
        quality = "elongated"
    elif solid:
        quality = "solid"
    elif inverted_candidate:
        quality = "inverted_candidate"
    elif low_contrast_relief:
        quality = "low_contrast_relief"
    else:
        quality = "ok"

    return {
        "valid": valid,
        "blank": blank,
        "solid": solid,
        "elongated": elongated,
        "low_contrast_relief": low_contrast_relief,
        "inverted_candidate": inverted_candidate,
        "ink_ratio": ink_ratio,
        "dark_ratio": dark_ratio,
        "mean": mean,
        "std": std,
        "dynamic_range": dynamic_range,
        "aspect_ratio": aspect_ratio,
        "quality": quality,
    }


def _resolve_token_image(
    token: CorpusToken,
    glyphs_dir: Path,
    filename_pattern: str,
    exact_catalog_index: dict[str, dict[str, Any]],
    ref_index: dict[str, Path],
    corpus_index: dict[str, Path],
) -> ResolveResult:
    exact_name = filename_pattern.format(
        tablet_id=token.tablet_id,
        position=token.position,
        barthel_code=token.barthel_code,
    )
    exact_path = glyphs_dir / exact_name

    original_key = _primary_key(token.barthel_code)

    if exact_path.exists():
        return ResolveResult(
            tier="exact",
            image_path=str(exact_path),
            source="exact_filename_pattern",
            resolved_code=_extract_code_from_path(exact_path),
            lookup_key=original_key,
            fallback_code_changed=False,
            reason="exact_filename_match",
        )

    exact = exact_catalog_index.get(token.corpus_key)
    if exact and not exact.get("merge_suspect", False):
        resolved = _extract_code_from_path(exact["path"])
        return ResolveResult(
            tier="exact",
            image_path=str(exact["path"]),
            source=str(exact.get("source") or "barthel_catalog_exact"),
            resolved_code=resolved,
            lookup_key=original_key,
            fallback_code_changed=(resolved != original_key),
            reason="exact_corpus_key_catalog",
        )

    for lookup_key, reason in _candidate_lookup_keys(token.barthel_code):
        if lookup_key in ref_index:
            p = ref_index[lookup_key]
            resolved = _extract_code_from_path(p)
            return ResolveResult(
                tier="ref_fallback",
                image_path=str(p),
                source="barthel_ref",
                resolved_code=resolved,
                lookup_key=lookup_key,
                fallback_code_changed=(resolved != original_key),
                reason=reason,
            )

        if lookup_key in corpus_index:
            p = corpus_index[lookup_key]
            resolved = _extract_code_from_path(p)
            src = "barthel_corpus_or_3d"
            return ResolveResult(
                tier="corpus_fallback",
                image_path=str(p),
                source=src,
                resolved_code=resolved,
                lookup_key=lookup_key,
                fallback_code_changed=(resolved != original_key),
                reason=reason,
            )

    return ResolveResult(
        tier="missing",
        image_path=None,
        source=None,
        resolved_code=None,
        lookup_key=original_key,
        fallback_code_changed=False,
        reason="no_match",
    )


def run_audit(
    project_root: Path,
    output_path: Path,
    glyphs_dir: Path,
    corpus_dir: Path,
    barthel_catalog_path: Path,
    filename_pattern: str,
    strict_actionable: bool = False,
    remediation_top_n: int = 25,
    include_positional_ref_estimates: bool = False,
) -> dict[str, Any]:
    tokens = _load_corpus_tokens(corpus_dir)
    merge_suspects = _load_merge_suspects(barthel_catalog_path)
    exact_catalog_index = _load_exact_catalog_index(barthel_catalog_path, glyphs_dir)

    ref_paths = list((glyphs_dir / "barthel_ref").glob("*.png")) if (glyphs_dir / "barthel_ref").exists() else []
    if not include_positional_ref_estimates:
        ref_paths = [p for p in ref_paths if not _is_positional_ref_estimate(p)]

    corpus_paths: list[Path] = []
    corpus_dir_img = glyphs_dir / "barthel_corpus"
    if corpus_dir_img.exists():
        for sub in sorted(corpus_dir_img.iterdir()):
            if sub.is_dir() and sub.name != "?":
                corpus_paths.extend(sorted(sub.glob("*.png")))

    crops_dir = glyphs_dir / "3d_crops"
    if crops_dir.exists():
        corpus_paths.extend(sorted(crops_dir.rglob("*.png")))

    ref_index = _build_code_index(ref_paths)
    corpus_index = _build_code_index(corpus_paths)

    per_token: list[dict[str, Any]] = []

    n_exact = 0
    n_fallback_same_code = 0
    n_fallback_different_code = 0
    n_merge_suspect = 0
    n_blank = 0
    n_elongated = 0
    n_solid = 0
    n_low_contrast_relief = 0
    n_inverted_candidate = 0
    n_missing = 0
    n_unknown_code = 0

    range_substitutions: list[dict[str, Any]] = []

    for token in tokens:
        resolved = _resolve_token_image(
            token,
            glyphs_dir=glyphs_dir,
            filename_pattern=filename_pattern,
            exact_catalog_index=exact_catalog_index,
            ref_index=ref_index,
            corpus_index=corpus_index,
        )

        merge_info = merge_suspects.get(token.corpus_key)
        is_merge_suspect = merge_info is not None
        if is_merge_suspect:
            n_merge_suspect += 1

        quality: dict[str, Any] | None = None
        if resolved.image_path:
            p = Path(resolved.image_path)
            if p.exists():
                quality = _is_valid_image(p)
                if quality["blank"]:
                    n_blank += 1
                if quality["elongated"]:
                    n_elongated += 1
                if quality["solid"]:
                    n_solid += 1
                if quality["low_contrast_relief"]:
                    n_low_contrast_relief += 1
                if quality["inverted_candidate"]:
                    n_inverted_candidate += 1
            else:
                resolved.tier = "missing"
                resolved.image_path = None

        if resolved.tier == "missing" and _is_unknown_placeholder_code(token.barthel_code):
            resolved.tier = "unknown_code"
            resolved.reason = "unknown_placeholder_code"

        if resolved.tier == "exact":
            n_exact += 1
        elif resolved.tier in {"ref_fallback", "corpus_fallback"}:
            if resolved.fallback_code_changed:
                n_fallback_different_code += 1
            else:
                n_fallback_same_code += 1
        elif resolved.tier == "missing":
            n_missing += 1
        elif resolved.tier == "unknown_code":
            n_unknown_code += 1

        raw_code = token.barthel_code
        is_range = RANGE_RE.match(raw_code or "") is not None
        if is_range:
            range_substitutions.append(
                {
                    "tablet_id": token.tablet_id,
                    "position": token.position,
                    "barthel_code": raw_code,
                    "corpus_key": token.corpus_key,
                    "resolved_tier": resolved.tier,
                    "resolved_code": resolved.resolved_code,
                    "resolved_path": resolved.image_path,
                    "resolver_reason": resolved.reason,
                }
            )

        per_token.append(
            {
                "tablet_id": token.tablet_id,
                "stratum": token.stratum,
                "position": token.position,
                "barthel_code": raw_code,
                "side": token.side_ab,
                "line": token.line,
                "seq_on_line": token.seq_on_line,
                "corpus_key": token.corpus_key,
                "resolved_tier": resolved.tier,
                "resolved_path": resolved.image_path,
                "resolved_source": resolved.source,
                "resolved_code": resolved.resolved_code,
                "lookup_key": resolved.lookup_key,
                "fallback_code_changed": resolved.fallback_code_changed,
                "resolver_reason": resolved.reason,
                "is_range_token": is_range,
                "merge_suspect": is_merge_suspect,
                "merge_suspect_info": merge_info,
                "quality": quality.get("quality") if quality else ("missing" if resolved.tier == "missing" else None),
                "image_quality": quality,
                "is_unknown_placeholder_code": _is_unknown_placeholder_code(raw_code),
            }
        )

    summary = {
        "n_tokens": len(tokens),
        "n_exact": n_exact,
        "n_fallback_same_code": n_fallback_same_code,
        "n_fallback_different_code": n_fallback_different_code,
        "n_merge_suspect": n_merge_suspect,
        "n_blank": n_blank,
        "n_elongated": n_elongated,
        "n_solid": n_solid,
        "n_low_contrast_relief": n_low_contrast_relief,
        "n_inverted_candidate": n_inverted_candidate,
        "n_missing": n_missing,
        "n_unknown_code": n_unknown_code,
        "n_ref_fallback": sum(1 for r in per_token if r["resolved_tier"] == "ref_fallback"),
        "n_corpus_fallback": sum(1 for r in per_token if r["resolved_tier"] == "corpus_fallback"),
        "n_range_tokens": len(range_substitutions),
    }

    high_risk_tokens = [
        row for row in per_token
        if row["resolved_tier"] == "missing"
        or row["fallback_code_changed"]
        or row["merge_suspect"]
        or row.get("quality") in {"blank", "elongated", "solid", "inverted_candidate"}
    ]

    actionable_tokens = [
        row for row in per_token
        if row["resolved_tier"] == "missing"
        or row.get("quality") in {"blank", "elongated", "solid", "inverted_candidate"}
    ]

    missing_code_remediation = _build_missing_code_remediation(per_token, top_n=remediation_top_n)

    report = {
        "project_root": str(project_root),
        "paths": {
            "glyphs_dir": str(glyphs_dir),
            "corpus_dir": str(corpus_dir),
            "barthel_catalog": str(barthel_catalog_path),
            "output": str(output_path),
        },
        "summary": summary,
        "range_substitutions": range_substitutions,
        "high_risk_count": len(high_risk_tokens),
        "actionable_count": len(actionable_tokens),
        "missing_code_remediation": missing_code_remediation,
        "tokens": per_token,
    }

    if strict_actionable:
        report["tokens"] = actionable_tokens

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    high_risk_path = output_path.with_name("audit_report_high_risk.json")
    high_risk_path.write_text(
        json.dumps(
            {
                "project_root": str(project_root),
                "source_report": str(output_path),
                "count": len(high_risk_tokens),
                "tokens": high_risk_tokens,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    actionable_path = output_path.with_name("audit_report_actionable.json")
    actionable_path.write_text(
        json.dumps(
            {
                "project_root": str(project_root),
                "source_report": str(output_path),
                "strict_actionable": strict_actionable,
                "count": len(actionable_tokens),
                "tokens": actionable_tokens,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    remediation_json_path = output_path.with_name("audit_missing_code_remediation.json")
    remediation_json_path.write_text(
        json.dumps(
            {
                "project_root": str(project_root),
                "source_report": str(output_path),
                "top_n": remediation_top_n,
                "rows": missing_code_remediation,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    remediation_md_path = output_path.with_name("audit_missing_code_remediation.md")
    _write_missing_code_remediation_markdown(remediation_md_path, missing_code_remediation)

    return report


def _parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    cfg = OmegaConf.load(project_root / "conf" / "config.yaml")
    default_pattern = str(cfg.glyph.filename_pattern)

    parser = argparse.ArgumentParser(description="Audit Barthel-code image mapping quality.")
    parser.add_argument("--project-root", type=Path, default=project_root)
    parser.add_argument("--glyphs-dir", type=Path, default=project_root / "data" / "glyphs")
    parser.add_argument("--corpus-dir", type=Path, default=project_root / "data" / "corpus")
    parser.add_argument(
        "--barthel-catalog",
        type=Path,
        default=project_root / "data" / "glyphs" / "barthel_catalog.json",
    )
    parser.add_argument(
        "--filename-pattern",
        default=default_pattern,
        help="Exact image filename pattern (default from conf/config.yaml).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=project_root / "outputs" / "analysis" / "audit_report.json",
    )
    parser.add_argument(
        "--strict-actionable",
        action="store_true",
        help="If set, output report tokens contain only actionable failures (true missing + severe image quality).",
    )
    parser.add_argument(
        "--remediation-top-n",
        type=int,
        default=25,
        help="How many true-missing Barthel codes to include in remediation table outputs.",
    )
    parser.add_argument(
        "--include-positional-ref-estimates",
        action="store_true",
        help="Include ambiguous barthel_ref positional estimate files (e.g. '100_42_barthel_...').",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()

    report = run_audit(
        project_root=args.project_root,
        output_path=args.output,
        glyphs_dir=args.glyphs_dir,
        corpus_dir=args.corpus_dir,
        barthel_catalog_path=args.barthel_catalog,
        filename_pattern=args.filename_pattern,
        strict_actionable=bool(args.strict_actionable),
        remediation_top_n=int(args.remediation_top_n),
        include_positional_ref_estimates=bool(args.include_positional_ref_estimates),
    )

    s = report["summary"]
    print("Image mapping audit complete")
    print(f"  tokens: {s['n_tokens']}")
    print(f"  exact: {s['n_exact']}")
    print(f"  fallback(same code): {s['n_fallback_same_code']}")
    print(f"  fallback(different code): {s['n_fallback_different_code']}")
    print(f"  merge_suspect: {s['n_merge_suspect']}")
    print(f"  blank: {s['n_blank']}  elongated: {s['n_elongated']}  solid: {s['n_solid']}")
    print(f"  missing(true missing image): {s['n_missing']}")
    print(f"  unknown placeholder code: {s['n_unknown_code']}")
    print(f"  actionable failures: {report['actionable_count']}")
    print(f"  strict mode: {'on' if args.strict_actionable else 'off'}")
    print(f"  output: {args.output}")
    print(f"  actionable output: {args.output.with_name('audit_report_actionable.json')}")
    print(f"  remediation output: {args.output.with_name('audit_missing_code_remediation.md')}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
