#!/usr/bin/env python3
"""
Transform horley_parallels.csv to match the expected schema for load_parallel_passages().

Expected input (from Souza 2022 / de Souza 2023):
  ID, Line, Sequence
  1, "Ar1", "041-430!.040-320..."

Expected output:
  passage_id, tablet_id, side, stratum, start_position, glyph_sequence

The Line field already uses Barthel's single-letter tablet IDs directly:
  'Ar1'  = Tablet A, recto,  line 1
  'Bb3'  = Tablet B, verso,  line 3
  'Da3'  = Tablet D, recto,  line 3  ← pre-contact anchor (Ferrara et al. 2024)
  'Gr5'  = Tablet G, recto,  line 5  ← NOT "Great" tablet, just Tablet G
  'Hr2'  = Tablet H, recto,  line 2
  'Pv4'  = Tablet P, verso,  line 4
  'Qv4'  = Tablet Q, verso,  line 4

Side convention:
  'r' (recto) → side 'a' in Barthel notation
  'v' (verso) → side 'b' in Barthel notation

Cross-reference notation like 'Ev6/Bb12' indicates the same passage
spanning a line break across two locations; we take the first location.

NOTE: The previous version of this file contained a HORLEY_TO_BARTHEL
mapping dict that was incorrect in multiple ways:
  1. The Line field prefixes ARE already Barthel tablet letters — no
     translation is needed or correct.
  2. The dict had a duplicate key 'Gr' mapping first to 'E' then to 'H',
     silently discarding the first entry (Python dict behaviour).
  3. Abbreviations like 'Ar' → 'B' would have remapped all Tablet A
     passages to Tablet B, corrupting the entire diachronic analysis.
That dict has been removed entirely.
"""

import argparse
import csv
import re
from pathlib import Path

from omegaconf import OmegaConf


def parse_line_field(line_str: str) -> tuple[str, str, int]:
    """
    Parse the Line field into (tablet_id, side, start_line).

    The Line field format is:
        {Barthel_tablet_letter}{side_char}{line_number}[/{Barthel_tablet_letter}{side_char}{line_number}]

    where side_char is 'r' (recto → 'a') or 'v' (verso → 'b').

    Cross-reference notation (e.g. 'Ev6/Bb12') is handled by taking
    only the primary (first) location.

    Parameters
    ----------
    line_str : str
        Raw value from the Line column, e.g. 'Ar1', 'Da3', 'Ev6/Bb12'.

    Returns
    -------
    tuple[str, str, int]
        (tablet_id, side, start_line) where:
          tablet_id  — Barthel single letter, e.g. 'A', 'D', 'H'
          side       — 'a' (recto) or 'b' (verso)
          start_line — integer line number, or -1 if unparseable
    """
    line_str = line_str.strip()

    # Take only the primary location when cross-reference notation is used
    primary = line_str.split("/")[0].strip()

    # Match: one uppercase letter + 'r' or 'v' + one or more digits
    # Optional trailing content (e.g. '-3' range suffix) is ignored.
    match = re.match(r"^([A-Z])([rv])(\d+)", primary)
    if match:
        tablet_id = match.group(1)                        # 'A', 'B', ..., 'S'
        side      = "a" if match.group(2) == "r" else "b" # recto→a, verso→b
        start_line = int(match.group(3))
        return (tablet_id, side, start_line)

    # Fallback: if the field is just a bare Barthel letter (shouldn't happen
    # with this CSV, but defensive)
    if re.match(r"^[A-Z]$", primary):
        return (primary, "a", -1)

    # Unparseable — log and return sentinel values
    import logging
    logging.getLogger(__name__).warning(
        "Could not parse Line field %r — setting tablet='?', side='a', line=-1",
        line_str,
    )
    return ("?", "a", -1)


def infer_stratum(tablet_id: str, temporal_model: dict) -> str:
    """
    Infer temporal stratum from tablet ID using the temporal model config.

    Parameters
    ----------
    tablet_id : str
        Barthel tablet ID (single letter A–Y).
    temporal_model : dict
        From config.yaml, structured as::

            clusters:
              pre_contact:
                tablets: [D]
              post_contact:
                tablets: [B, C, O, Q]
              excluded_from_temporal_analysis:
                tablets: [A]

    Returns
    -------
    str
        One of: 'pre_contact', 'post_contact', 'excluded', 'undated'.
    """
    clusters = temporal_model.get("clusters", {})

    if tablet_id in clusters.get("pre_contact", {}).get("tablets", []):
        return "pre_contact"
    if tablet_id in clusters.get("post_contact", {}).get("tablets", []):
        return "post_contact"
    if tablet_id in clusters.get("excluded_from_temporal_analysis", {}).get("tablets", []):
        return "excluded"

    return "undated"


def transform_parallels_csv(
    input_csv: Path,
    output_csv: Path,
    config_path: Path,
) -> None:
    """
    Transform horley_parallels.csv to the schema expected by
    load_parallel_passages().

    Parameters
    ----------
    input_csv : Path
        Source CSV with columns: ID, Line, Sequence
    output_csv : Path
        Destination CSV with columns:
        passage_id, tablet_id, side, stratum, start_position, glyph_sequence
    config_path : Path
        Hydra config.yaml path (used to read temporal_model cluster assignments).
    """
    import logging
    log = logging.getLogger(__name__)

    cfg = OmegaConf.load(config_path)
    temporal_model = OmegaConf.to_container(
        cfg.get("corpus", {}).get("temporal_model", {}),
        resolve=True,
    )

    rows_out = []
    n_unparseable = 0

    with input_csv.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            passage_id     = f"P{row['ID'].zfill(3)}"   # P001, P002, …
            line_field     = row["Line"]
            glyph_sequence = row["Sequence"]

            tablet_id, side, start_pos = parse_line_field(line_field)

            if tablet_id == "?":
                n_unparseable += 1

            stratum = infer_stratum(tablet_id, temporal_model)

            rows_out.append({
                "passage_id":     passage_id,
                "tablet_id":      tablet_id,
                "side":           side,
                "stratum":        stratum,
                "start_position": start_pos,
                "glyph_sequence": glyph_sequence,
            })

    fieldnames = [
        "passage_id",
        "tablet_id",
        "side",
        "stratum",
        "start_position",
        "glyph_sequence",
    ]
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_out)

    # Summary
    strata = {}
    tablets = {}
    for r in rows_out:
        strata[r["stratum"]] = strata.get(r["stratum"], 0) + 1
        tablets[r["tablet_id"]] = tablets.get(r["tablet_id"], 0) + 1

    print(f"✓ Transformed {len(rows_out)} rows ({n_unparseable} unparseable)")
    print(f"  Input:   {input_csv}")
    print(f"  Output:  {output_csv}")
    print(f"  Strata:  {dict(sorted(strata.items()))}")
    print(f"  Tablets: {dict(sorted(tablets.items()))}")

    # Sanity check: Tablet D must be pre_contact
    d_rows = [r for r in rows_out if r["tablet_id"] == "D"]
    if d_rows:
        d_strata = set(r["stratum"] for r in d_rows)
        if d_strata == {"pre_contact"}:
            print(f"  ✓ Tablet D ({len(d_rows)} passages) correctly labelled pre_contact")
        else:
            print(f"  ✗ WARNING: Tablet D stratum = {d_strata} — check temporal_model config")
    else:
        print(f"  ✗ WARNING: No Tablet D passages found — check CSV and parse_line_field()")


def main() -> None:
    import logging
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

    parser = argparse.ArgumentParser(
        description=(
            "Transform horley_parallels.csv to the schema expected by "
            "load_parallel_passages(). "
            "The Line field already uses Barthel tablet letters directly — "
            "no tablet-code translation is performed."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/parallels/horley_parallels.csv"),
        help="Input CSV (default: data/parallels/horley_parallels.csv)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/parallels/horley_parallels_transformed.csv"),
        help="Output CSV (default: data/parallels/horley_parallels_transformed.csv)",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("conf/config.yaml"),
        help="Hydra config path (default: conf/config.yaml)",
    )
    args = parser.parse_args()

    transform_parallels_csv(args.input, args.output, args.config)


if __name__ == "__main__":
    main()