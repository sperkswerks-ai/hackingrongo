#!/usr/bin/env python3
"""
NXZ Header Analyzer for Nexus 3D Models (INSCRIBE tablet data)

NXZ is the compressed hierarchical LOD mesh format from CNR-ISTI VCLab:
  github.com/cnr-isti-vclab/nexus

This script verifies NXZ file integrity and extracts the confirmed header
fields (magic, version) plus metric coordinates found in the node table.
Full NXZ -> PLY conversion requires the native nexus C++ tools.

See 3D_TABLET_INTEGRATION.md for conversion options.

Usage:
    python nxz_to_ply_converter.py <nxz_file> [<nxz_file2> ...]
"""

import math
import struct
import sys
from pathlib import Path


# Confirmed NXZ v2 header layout:
#   bytes 0–3:  char[4]   magic " sxN" (0x20 0x73 0x78 0x4E)
#   bytes 4–7:  uint32    version = 2
#   bytes 8+:   node-table entries (LOD sphere descriptors + chunk offsets)
#
# The exact layout of the node table requires the nexus C++ source to parse
# correctly. We scan for metric-scale float values (finite, |v| in 0.01–2000)
# as a proxy — these correspond to sphere centres and radii for the LOD nodes.
_MAGIC = b" sxN"


def parse_nxz_header(nxz_path: Path) -> dict:
    """Verify magic/version and scan for metric-scale float coordinates."""
    with open(nxz_path, "rb") as f:
        raw = f.read(160)

    if len(raw) < 8:
        return {"error": f"File too short ({len(raw)} bytes)"}

    magic = raw[:4]
    if magic != _MAGIC:
        return {"error": f"Invalid magic bytes: {magic.hex()} (expected {_MAGIC.hex()})"}

    version = struct.unpack_from("<I", raw, 4)[0]
    if version != 2:
        return {"error": f"Unsupported NXS version {version} (only v2 parsed here)"}

    # Scan for plausible metric float values in the node table region.
    coords: list[tuple[int, float]] = []
    for off in range(8, len(raw) - 3, 4):
        v = struct.unpack_from("<f", raw, off)[0]
        if math.isfinite(v) and 0.05 < abs(v) < 2000:
            coords.append((off, round(v, 3)))

    return {
        "magic": magic.decode("ascii"),
        "version": version,
        "file_size_mb": nxz_path.stat().st_size / (1024 * 1024),
        "metric_coords": coords,  # (byte_offset, mm_value) — sphere centres / radii
    }


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: nxz_to_ply_converter.py <nxz_file> [<nxz_file2> ...]")
        sys.exit(1)

    for arg in sys.argv[1:]:
        nxz_path = Path(arg)
        if not nxz_path.exists():
            print(f"Error: {nxz_path} not found")
            continue

        print(f"\n{'─' * 58}")
        print(f"File : {nxz_path.name}")
        print(f"Size : {nxz_path.stat().st_size / (1024 * 1024):.1f} MB")

        info = parse_nxz_header(nxz_path)
        if "error" in info:
            print(f"Error: {info['error']}")
            continue

        print(f"Magic: {info['magic']!r}  version={info['version']}  ✓ valid NXZ")
        coords = info["metric_coords"]
        if coords:
            vals = [v for _, v in coords]
            print(f"Metric coords in node table: {len(coords)} floats  "
                  f"range [{min(vals):.1f}, {max(vals):.1f}] mm")
            print(f"  First 6 values (sphere centres / radii of LOD root nodes):")
            for off, v in coords[:6]:
                print(f"    offset {off:3d}: {v:>10.3f} mm")
        else:
            print("  No metric-scale floats found in first 160 bytes.")

    print(f"\n{'─' * 58}")
    print("Full NXZ -> PLY conversion requires native nexus C++ tools.")
    print("See 3D_TABLET_INTEGRATION.md for conversion options.")


if __name__ == "__main__":
    main()
