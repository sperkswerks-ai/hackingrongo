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
echo; echo "STEP 0 — checking build deps (cmake, g++, Qt6)…"
need=()
command -v cmake >/dev/null || need+=(cmake)
command -v g++   >/dev/null || need+=(build-essential)
command -v git   >/dev/null || need+=(git)
# Current Nexus CMake requires Qt6 (find_package(Qt6) at CMakeLists.txt:16).
qt6_ok() { pkg-config --exists Qt6Core 2>/dev/null || command -v qmake6 >/dev/null \
           || ls /usr/lib/*/cmake/Qt6/Qt6Config.cmake >/dev/null 2>&1; }
if ! qt6_ok; then
  need+=(qt6-base-dev)        # provides Qt6Config.cmake on Ubuntu 22.04+
fi
if [ "${#need[@]}" -gt 0 ]; then
  echo "  Missing: ${need[*]}"
  echo "  Trying:  sudo apt-get install -y ${need[*]}"
  if ! { sudo apt-get update -y && sudo apt-get install -y "${need[@]}"; }; then
    echo "  ⚠ apt failed (no sudo?). Install Qt6 via conda instead:"
    echo "      conda install -y -c conda-forge cmake cxx-compiler qt6-main"
    echo "    then re-run this script (it will auto-detect the conda Qt6)."
    exit 1
  fi
else
  echo "  ✓ build deps present"
fi

# Locate Qt6 so we can hand CMake an explicit prefix (covers apt AND conda installs).
QT6_PREFIX=""
for cand in \
    "$(command -v qmake6 >/dev/null && dirname "$(dirname "$(command -v qmake6)")")" \
    "${CONDA_PREFIX:-}" \
    "$(ls -d /usr/lib/*/cmake/Qt6 2>/dev/null | head -1 | sed 's#/lib/.*##')" ; do
  if [ -n "$cand" ] && ls "$cand"/lib/cmake/Qt6/Qt6Config.cmake >/dev/null 2>&1; then
    QT6_PREFIX="$cand"; break
  fi
done
[ -n "$QT6_PREFIX" ] && echo "  ✓ Qt6 prefix: $QT6_PREFIX" || echo "  (Qt6 on default CMake path)"

# --- 1. corto (geometry codec) -----------------------------------------------
echo; echo "STEP 1 — corto…"
if [ ! -d corto ]; then git clone --depth 1 https://github.com/cnr-isti-vclab/corto.git; fi
cmake -S corto -B corto/build -DCMAKE_BUILD_TYPE=Release >/dev/null
cmake --build corto/build -j "$JOBS"
CORTO_LIB="$(find "$TOOLS_PREFIX/corto/build" -name 'libcorto*' | head -1 || true)"
[ -n "$CORTO_LIB" ] && echo "  ✓ corto built: $CORTO_LIB" || { echo "  ✗ corto lib not found"; exit 1; }

# --- 1.5 vcglib (mesh library Nexus is built on; header-mostly) --------------
echo; echo "STEP 1.5 — vcglib…"
if [ ! -d vcglib ]; then git clone --depth 1 https://github.com/cnr-isti-vclab/vcglib.git; fi
# Configure to generate vcglibConfig.cmake (needed by Nexus find_package(vcglib)).
cmake -S vcglib -B vcglib/build -DCMAKE_BUILD_TYPE=Release >/dev/null 2>&1 || true
# Find the directory containing vcglibConfig.cmake (build dir, else source tree).
VCGLIB_DIR="$(dirname "$(find "$TOOLS_PREFIX/vcglib" -name 'vcglibConfig.cmake' 2>/dev/null | head -1)" 2>/dev/null || true)"
[ -z "$VCGLIB_DIR" ] || [ "$VCGLIB_DIR" = "." ] && VCGLIB_DIR="$TOOLS_PREFIX/vcglib"   # fallback: source root
echo "  ✓ vcglib at: $VCGLIB_DIR"

# --- 2. nexus (nxsbuild / nxsedit / nxsdump) ---------------------------------
echo; echo "STEP 2 — nexus…"
if [ ! -d nexus ]; then git clone --depth 1 https://github.com/cnr-isti-vclab/nexus.git; fi
# Point nexus at the corto we just built, and at Qt6 if we located a prefix.
NEXUS_CMAKE_ARGS=(-DCMAKE_BUILD_TYPE=Release -DCORTO_ROOT="$TOOLS_PREFIX/corto" -DBUILD_NXS_VIEW=OFF)
NEXUS_CMAKE_ARGS+=(-Dvcglib_DIR="$VCGLIB_DIR" -DVCGDIR="$TOOLS_PREFIX/vcglib")
[ -n "$QT6_PREFIX" ] && NEXUS_CMAKE_ARGS+=(-DCMAKE_PREFIX_PATH="$QT6_PREFIX")
cmake -S nexus -B nexus/build "${NEXUS_CMAKE_ARGS[@]}" \
  || cmake -S nexus -B nexus/build -DCMAKE_BUILD_TYPE=Release ${QT6_PREFIX:+-DCMAKE_PREFIX_PATH="$QT6_PREFIX"}
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
