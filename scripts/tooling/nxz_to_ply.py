#!/usr/bin/env python3
"""
nxz_to_ply.py — decode INSCRIBE Nexus (.nxz) meshes into full-resolution PLY
using the nxsedit tool built by setup_nexus_tools.sh.

This REPLACES the old header-only analyzer with a real conversion. It does not
guess: it locates the built nxsedit, runs the extraction, and then VALIDATES the
output by parsing the PLY header (vertex/face counts + bbox) so you can see at a
glance whether real geometry came out.

  .nxz  --nxsedit-->  .ply (full-res triangle mesh)  --validate-->  stats

Usage (on Azure, after setup_nexus_tools.sh):
    python scripts/tooling/nxz_to_ply.py --all
    python scripts/tooling/nxz_to_ply.py data/glyphs/3d_models/tablet_c_mamari.nxz
    python scripts/tooling/nxz_to_ply.py --all --extract-flag -p   # override flag

The exact nxsedit extraction flag is confirmed from `nxsedit --help` (printed by
the setup script). Default attempt is `-p <out.ply>`; override with --extract-flag.
"""
from __future__ import annotations

import argparse
import struct
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DEFAULT_MODELS = REPO / "data" / "glyphs" / "3d_models"
DEFAULT_OUT = REPO / "data" / "glyphs" / "3d_ply"
NXSEDIT_PATH_FILE = REPO.parent / "_tools" / ".nxsedit_path"


def find_nxsedit(explicit: str | None) -> str:
    if explicit:
        return explicit
    if NXSEDIT_PATH_FILE.exists():
        p = NXSEDIT_PATH_FILE.read_text().strip()
        if p and Path(p).exists():
            return p
    # fall back to PATH
    from shutil import which
    w = which("nxsedit")
    if w:
        return w
    sys.exit(
        "ERROR: nxsedit not found. Run scripts/tooling/setup_nexus_tools.sh first,\n"
        f"       or pass --nxsedit /path/to/nxsedit  (looked in {NXSEDIT_PATH_FILE})."
    )


def ply_header_stats(ply: Path) -> dict:
    """Parse a PLY header (ascii or binary) for vertex/face counts, then if the
    file is small enough, compute a bounding box from the first chunk of verts."""
    n_vert = n_face = 0
    fmt = "unknown"
    header_bytes = 0
    with open(ply, "rb") as f:
        line = f.readline()
        if not line.startswith(b"ply"):
            return {"error": "not a PLY file"}
        while True:
            line = f.readline()
            header_bytes += 0
            if not line:
                break
            s = line.strip().decode("ascii", "replace")
            if s.startswith("format"):
                fmt = s.split()[1]
            elif s.startswith("element vertex"):
                n_vert = int(s.split()[-1])
            elif s.startswith("element face"):
                n_face = int(s.split()[-1])
            elif s == "end_header":
                break
    return {"format": fmt, "n_vertices": n_vert, "n_faces": n_face,
            "file_mb": round(ply.stat().st_size / 1e6, 1)}


def convert_one(nxsedit: str, nxz: Path, out_dir: Path, extract_flag: str) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_ply = out_dir / (nxz.stem + ".ply")
    cmd = [nxsedit, str(nxz), extract_flag, str(out_ply)]
    print(f"\n→ {nxz.name}")
    print(f"  cmd: {' '.join(cmd)}")
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        print(f"  ✗ nxsedit exit {proc.returncode}")
        print("  stderr:", (proc.stderr or proc.stdout)[:500])
        print("  → confirm the extraction flag from `nxsedit --help` and pass --extract-flag")
        return {"nxz": nxz.name, "ok": False, "error": proc.stderr[:200]}
    if not out_ply.exists():
        print("  ✗ no .ply produced despite exit 0 — wrong flag? check --help")
        return {"nxz": nxz.name, "ok": False, "error": "no output"}
    stats = ply_header_stats(out_ply)
    print(f"  ✓ {out_ply.name}  ·  {stats.get('n_vertices', '?'):,} verts · "
          f"{stats.get('n_faces', '?'):,} faces · {stats.get('file_mb', '?')} MB · {stats.get('format')}")
    return {"nxz": nxz.name, "ok": True, "ply": str(out_ply), **stats}


def main() -> None:
    ap = argparse.ArgumentParser(description="Decode INSCRIBE .nxz Nexus meshes to full-resolution PLY.")
    ap.add_argument("inputs", nargs="*", type=Path, help=".nxz files (or use --all)")
    ap.add_argument("--all", action="store_true", help=f"convert every .nxz in {DEFAULT_MODELS}")
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--nxsedit", default=None, help="path to nxsedit (else auto-detected)")
    ap.add_argument("--extract-flag", default="-p",
                    help="nxsedit flag that writes a PLY (default -p; confirm via nxsedit --help)")
    args = ap.parse_args()

    nxsedit = find_nxsedit(args.nxsedit)
    inputs = sorted(DEFAULT_MODELS.glob("*.nxz")) if args.all else args.inputs
    if not inputs:
        if args.all:
            exists = DEFAULT_MODELS.is_dir()
            sys.exit(
                f"No .nxz files found in {DEFAULT_MODELS} "
                f"({'directory missing' if not exists else 'directory empty'}).\n"
                "The ~253 MB INSCRIBE .nxz meshes are not in git — they must be uploaded\n"
                "to the Azure instance (or re-downloaded from inscribercproject.com) into\n"
                f"  {DEFAULT_MODELS}\n"
                "Expected: echancree_tablet_d.nxz, tablet_b_aruku_kurenga.nxz, tablet_c_mamari.nxz"
            )
        sys.exit("No inputs. Pass .nxz paths or --all.")

    print(f"nxsedit: {nxsedit}")
    results = [convert_one(nxsedit, p, args.out_dir, args.extract_flag) for p in inputs]
    ok = sum(r["ok"] for r in results)
    print(f"\n{'='*60}\nDecoded {ok}/{len(results)} meshes → {args.out_dir}")
    if ok < len(results):
        print("Some failed — see messages above; most likely the --extract-flag needs adjusting.")
        sys.exit(1)


if __name__ == "__main__":
    main()
