#!/usr/bin/env bash
#
# setup_nexus_tools.sh — build the CNR-ISTI Nexus C++ toolchain on the Azure
# (Ubuntu) instance so we can decode INSCRIBE .nxz meshes into real geometry.
#
# .nxz = Nexus multiresolution mesh, Corto-compressed. Decoding needs:
#   * corto   (github.com/cnr-isti-vclab/corto)  — the geometry codec
#   * nexus   (github.com/cnr-isti-vclab/nexus)  — nxsbuild / nxsedit / nxsdump
#
# We build the COMMAND-LINE tools only (nxsedit/nxsdump) — not the GUI viewer —
# so we avoid the heavy Qt/OpenGL GUI dependencies. nxsedit still needs Qt5Core.
#
# SAFETY: builds into  $TOOLS_PREFIX  (default <repo>/../_tools), no sudo, no
# system install. Re-runnable. Verifies each build before continuing.
#
# USAGE (on Azure):
#   bash scripts/tooling/setup_nexus_tools.sh
#   # then note the printed nxsedit path; feed it to nxz_to_ply.py
#
set -euo pipefail

REPO="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
TOOLS_PREFIX="${TOOLS_PREFIX:-$(dirname "$REPO")/_tools}"
JOBS="$(nproc 2>/dev/null || echo 4)"
mkdir -p "$TOOLS_PREFIX"
cd "$TOOLS_PREFIX"

echo "════════════════════════════════════════════════════════════════"
echo "  Building Nexus toolchain into: $TOOLS_PREFIX   (jobs=$JOBS)"
echo "════════════════════════════════════════════════════════════════"

# --- 0. system deps -----------------------------------------------------------
# These are the usual Ubuntu deps. If apt needs sudo and you don't have it on the
# compute instance, install equivalents via conda (cmake, qt) and re-run.
echo; echo "STEP 0 — checking build deps (cmake, g++, qmake/qtbase)…"
need=()
command -v cmake >/dev/null || need+=(cmake)
command -v g++   >/dev/null || need+=(build-essential)
command -v git   >/dev/null || need+=(git)
# Qt5Core is required by nxsedit/nxsbuild:
if ! (pkg-config --exists Qt5Core 2>/dev/null || command -v qmake >/dev/null); then
  need+=(qtbase5-dev)
fi
if [ "${#need[@]}" -gt 0 ]; then
  echo "  Missing: ${need[*]}"
  echo "  Trying:  sudo apt-get install -y ${need[*]}"
  sudo apt-get update -y && sudo apt-get install -y "${need[@]}" || {
    echo "  ⚠ apt failed (no sudo?). Install via conda instead, e.g.:"
    echo "      conda install -y -c conda-forge cmake cxx-compiler qt-main"
    echo "  then re-run this script."; exit 1; }
else
  echo "  ✓ build deps present"
fi

# --- 1. corto (geometry codec) -----------------------------------------------
echo; echo "STEP 1 — corto…"
if [ ! -d corto ]; then git clone --depth 1 https://github.com/cnr-isti-vclab/corto.git; fi
cmake -S corto -B corto/build -DCMAKE_BUILD_TYPE=Release >/dev/null
cmake --build corto/build -j "$JOBS"
CORTO_LIB="$(find "$TOOLS_PREFIX/corto/build" -name 'libcorto*' | head -1 || true)"
[ -n "$CORTO_LIB" ] && echo "  ✓ corto built: $CORTO_LIB" || { echo "  ✗ corto lib not found"; exit 1; }

# --- 2. nexus (nxsbuild / nxsedit / nxsdump) ---------------------------------
echo; echo "STEP 2 — nexus…"
if [ ! -d nexus ]; then git clone --depth 1 https://github.com/cnr-isti-vclab/nexus.git; fi
# Point nexus at the corto we just built.
cmake -S nexus -B nexus/build -DCMAKE_BUILD_TYPE=Release \
      -DCORTO_ROOT="$TOOLS_PREFIX/corto" \
      -DBUILD_NXS_VIEW=OFF 2>/dev/null \
  || cmake -S nexus -B nexus/build -DCMAKE_BUILD_TYPE=Release   # fallback: default opts
cmake --build nexus/build -j "$JOBS"

# --- 3. locate + verify the CLI tools ----------------------------------------
echo; echo "STEP 3 — locating built tools…"
NXSEDIT="$(find "$TOOLS_PREFIX/nexus" -name nxsedit -type f -perm -u+x | head -1 || true)"
NXSDUMP="$(find "$TOOLS_PREFIX/nexus" -name nxsdump -type f -perm -u+x | head -1 || true)"
echo "  nxsedit: ${NXSEDIT:-NOT FOUND}"
echo "  nxsdump: ${NXSDUMP:-NOT FOUND}"
if [ -z "$NXSEDIT" ]; then
  echo "  ✗ nxsedit not built. Inspect $TOOLS_PREFIX/nexus/build for errors."; exit 1
fi

# Persist the path so the python wrapper can find it.
echo "$NXSEDIT" > "$TOOLS_PREFIX/.nxsedit_path"
echo "  ✓ saved nxsedit path → $TOOLS_PREFIX/.nxsedit_path"

# --- 4. surface the REAL interface (so we don't guess flags) -----------------
echo; echo "STEP 4 — nxsedit --help (confirm the .nxz→.ply extraction flag):"
echo "────────────────────────────────────────────────────────────────"
"$NXSEDIT" --help 2>&1 | head -40 || "$NXSEDIT" -h 2>&1 | head -40 || true
echo "────────────────────────────────────────────────────────────────"
echo
echo "✓ Nexus toolchain ready."
echo "  Next:  python scripts/tooling/nxz_to_ply.py --all"
echo "  (it reads $TOOLS_PREFIX/.nxsedit_path automatically)"
