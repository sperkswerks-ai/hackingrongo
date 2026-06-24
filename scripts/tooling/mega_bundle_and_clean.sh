#!/usr/bin/env bash
#
# mega_bundle_and_clean.sh — archive every scattered result artifact into one
# timestamped tarball, then (only on explicit confirmation) purge the clutter so
# the repo is pristine for a fresh HERO run.
#
# SAFETY MODEL
#   * Default mode bundles + verifies and DELETES NOTHING.
#   * Deletion requires BOTH --clean and --confirm, AND a verified bundle.
#   * The KEEP list (code, data, catalog, quantum_results, .git, docs, *.md) is
#     never bundled-for-deletion and never removed.
#
# USAGE
#   bash scripts/tooling/mega_bundle_and_clean.sh                # bundle only
#   bash scripts/tooling/mega_bundle_and_clean.sh --clean --confirm   # bundle, then purge
#
# ENV OVERRIDES
#   ARCHIVE_DIR   where the tarball is written (default: <repo>/../_archive)
#
set -euo pipefail

# --- locate the repo root (robust regardless of where you invoke from) --------
REPO="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
cd "$REPO"
TS="$(date -u +%Y%m%dT%H%M%SZ)"
ARCHIVE_DIR="${ARCHIVE_DIR:-$(dirname "$REPO")/_archive}"
BUNDLE="$ARCHIVE_DIR/megabundle_${TS}.tgz"

# --- what counts as "scattered clutter" (bundled, then eligible for purge) -----
# Globs that don't match are silently skipped. Add paths here if you spot more.
CLUTTER=(
  outputs
  mlruns
  hydra_runs
  outputs/hydra_runs
  output_run*
  multirun
  "*.log"
  nohup.out
  reports_bundle*.zip
  reports_bundle*.tgz
  hackingrongo_checkpoints*.zip
)

# --- what is NEVER deleted (real data / code / irreplaceable hardware runs) ----
# (quantum_results = real IBM hardware submissions — expensive, irreplaceable.)
# These are listed for documentation; the purge step only ever touches CLUTTER.
KEEP_NOTE="KEEP: data/ catalog/ hackingrongo/ scripts/ conf/ docs/ *.md .git/ quantum_results/"

echo "════════════════════════════════════════════════════════════════"
echo "  Repo     : $REPO"
echo "  Archive  : $BUNDLE"
echo "  $KEEP_NOTE"
echo "════════════════════════════════════════════════════════════════"

# --- STAGE 0: survey ----------------------------------------------------------
echo; echo "STAGE 0 — survey (sizes of clutter that exists):"
existing=()
for p in "${CLUTTER[@]}"; do
  for m in $p; do                      # expand globs
    [ -e "$m" ] && existing+=("$m")
  done
done
if [ "${#existing[@]}" -eq 0 ]; then
  echo "  (nothing matching the clutter list — repo already clean?)"
else
  du -sh "${existing[@]}" 2>/dev/null | sort -h || true
fi

# --- STAGE 1: bundle ----------------------------------------------------------
echo; echo "STAGE 1 — building mega bundle (also archives quantum_results for safety):"
mkdir -p "$ARCHIVE_DIR"
# quantum_results is KEPT in place but ALSO copied into the bundle as insurance.
tar czf "$BUNDLE" --ignore-failed-read \
    "${existing[@]}" quantum_results 2>/dev/null || true
echo "  wrote: $BUNDLE"

# --- STAGE 2: verify ----------------------------------------------------------
echo; echo "STAGE 2 — verify bundle integrity:"
if ! tar tzf "$BUNDLE" >/dev/null 2>&1; then
  echo "  ✗ bundle failed to verify — ABORTING (nothing was deleted)."; exit 1
fi
n_entries="$(tar tzf "$BUNDLE" | wc -l | tr -d ' ')"
echo "  ✓ readable · $n_entries entries · $(du -sh "$BUNDLE" | cut -f1)"

# --- STAGE 3: gate ------------------------------------------------------------
if [[ "${1:-}" != "--clean" || "${2:-}" != "--confirm" ]]; then
  echo
  echo "BUNDLE COMPLETE. Nothing deleted."
  echo "  → Download $BUNDLE off the instance and confirm it opens."
  echo "  → THEN re-run with:  bash $0 --clean --confirm"
  exit 0
fi

# --- STAGE 4: purge (only reached with --clean --confirm) ---------------------
echo; echo "STAGE 4 — purge clutter (bundle preserved at $BUNDLE):"
git status --short | grep -vE '^\?\?' && {
  echo "  ⚠ tracked files have uncommitted changes (above). The purge only"
  echo "    removes the clutter paths, not tracked code — continuing."
} || true
for m in "${existing[@]}"; do
  echo "  rm -rf $m"
  rm -rf "$m"
done
echo
echo "STAGE 5 — pristine check:"
echo "  remaining ignored/untracked cruft (git clean dry-run):"
git clean -Xnd | head -20 || true
echo
echo "✓ Done. Clean slate for the HERO run. Bundle: $BUNDLE"
