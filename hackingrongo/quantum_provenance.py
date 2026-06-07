"""
hackingrongo.quantum_provenance
================================

Shared utilities for persisting IBM Quantum hardware run provenance.

Every IBM Quantum job run by the rongorongo project should call
``collect_provenance(job, backend)`` immediately after ``job.result()``
returns, then embed the returned dict in the result JSON under
``"hardware_provenance"``, and finally call ``write_versioned_result``
to commit a permanent, uniquely-named file to ``quantum_results/`` and
append a row to ``RESULTS.md``.

Persistence contract
--------------------
- One file per run: ``quantum_results/<experiment>/<run_key>_<ts>_<job_id>.json``
- Never overwritten: timestamped filename guarantees uniqueness.
- ``RESULTS.md`` at repo root: one markdown table row per hardware run,
  appended on every call to ``write_versioned_result``.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

_REPO_ROOT          = Path(__file__).resolve().parent.parent
QUANTUM_RESULTS_DIR = _REPO_ROOT / "quantum_results"
RESULTS_MD          = _REPO_ROOT / "RESULTS.md"

_MD_HEADER = (
    "# Quantum Hardware Results\n\n"
    "Every row represents one IBM Quantum hardware submission.  "
    "Job IDs can be looked up in the IBM Quantum Platform dashboard.\n\n"
    "| Experiment | Run key | Backend | Job ID | Qubits | "
    "Calibration timestamp | Run timestamp (UTC) | File |\n"
    "|---|---|---|---|---|---|---|---|\n"
)


# ---------------------------------------------------------------------------
# Provenance collection
# ---------------------------------------------------------------------------

def collect_provenance(job: Any, backend: Any) -> dict[str, Any]:
    """Return a dict capturing all identifiers needed to reproduce or verify a run.

    Parameters
    ----------
    job:
        Completed ``QiskitRuntimeService`` job (SamplerV2 result already
        fetched before this call so it doesn't block).
    backend:
        The ``IBMBackend`` (or FakeBackend) the job was submitted to.

    Returns
    -------
    dict with keys:
      job_id, backend_name, backend_num_qubits,
      calibration_timestamp (from backend.properties().last_update_date),
      run_timestamp (UTC ISO-8601), status.
    """
    prov: dict[str, Any] = {
        "run_timestamp": datetime.now(tz=timezone.utc).isoformat(),
    }

    try:
        prov["job_id"] = job.job_id()
    except Exception:
        prov["job_id"] = "unknown"

    try:
        prov["status"] = str(job.status())
    except Exception:
        pass

    try:
        prov["backend_name"] = backend.name
    except Exception:
        prov["backend_name"] = str(type(backend).__name__)

    try:
        prov["backend_num_qubits"] = int(backend.num_qubits)
    except Exception:
        pass

    try:
        props = backend.properties()
        if props is not None:
            dt = props.last_update_date
            prov["calibration_timestamp"] = (
                dt.isoformat() if hasattr(dt, "isoformat") else str(dt)
            )
    except Exception:
        pass

    return prov


def collect_multi_job_provenance(
    job_ids: list[str],
    backend: Any,
) -> dict[str, Any]:
    """Provenance for multi-job runs (e.g. QAOA: one job per COBYLA step).

    Includes the full list of job IDs alongside the backend metadata.
    """
    prov: dict[str, Any] = {
        "run_timestamp":   datetime.now(tz=timezone.utc).isoformat(),
        "job_ids":         job_ids,
        "n_jobs_submitted": len(job_ids),
    }
    try:
        prov["backend_name"] = backend.name
    except Exception:
        prov["backend_name"] = str(type(backend).__name__)

    try:
        prov["backend_num_qubits"] = int(backend.num_qubits)
    except Exception:
        pass

    try:
        props = backend.properties()
        if props is not None:
            dt = props.last_update_date
            prov["calibration_timestamp"] = (
                dt.isoformat() if hasattr(dt, "isoformat") else str(dt)
            )
    except Exception:
        pass

    return prov


# ---------------------------------------------------------------------------
# Versioned file persistence
# ---------------------------------------------------------------------------

def write_versioned_result(
    result: dict[str, Any],
    experiment: str,
    run_key: str,
) -> Path:
    """Write *result* to a uniquely-named file in ``quantum_results/<experiment>/``.

    The filename embeds the run timestamp and the primary IBM job ID so the
    file can never be accidentally overwritten and can be cross-referenced
    against the IBM Quantum Platform dashboard.

    Parameters
    ----------
    result:
        Complete result dict.  Must contain ``"hardware_provenance"`` with
        at least ``"job_id"`` if the run was on real hardware.
    experiment:
        Short identifier for the experiment type, e.g. ``"simon"``,
        ``"bv"``, ``"qaoa"``.
    run_key:
        Per-run discriminator, e.g. the passage ID for Simon or the QAOA
        backend string.  Must be filename-safe (no slashes).

    Returns
    -------
    Path to the written file.
    """
    prov = result.get("hardware_provenance", {})

    # Primary job ID (single-job runs) or first of list (multi-job)
    job_ids = prov.get("job_ids", [])
    job_id  = (job_ids[0] if job_ids else None) or prov.get("job_id", "nojobid")

    ts = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    out_dir = QUANTUM_RESULTS_DIR / experiment
    out_dir.mkdir(parents=True, exist_ok=True)

    # Sanitise run_key (strip slashes, spaces)
    safe_key = run_key.replace("/", "_").replace(" ", "_")
    filename = f"{safe_key}_{ts}_{job_id}.json"
    out_path = out_dir / filename

    out_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    log.info("Versioned quantum result: %s", out_path)

    _append_results_md(experiment, run_key, job_id, prov, filename, experiment)
    return out_path


def _append_results_md(
    experiment: str,
    run_key: str,
    job_id: str,
    prov: dict[str, Any],
    filename: str,
    subdir: str,
) -> None:
    if not RESULTS_MD.exists():
        RESULTS_MD.write_text(_MD_HEADER, encoding="utf-8")

    backend  = prov.get("backend_name", "—")
    n_qubits = prov.get("backend_num_qubits", "—")
    cal_ts   = prov.get("calibration_timestamp", "—")
    run_ts   = prov.get("run_timestamp", "—")

    # For multi-job runs, show count + first job ID
    job_ids = prov.get("job_ids")
    if job_ids and len(job_ids) > 1:
        job_cell = f"`{job_ids[0]}`…(+{len(job_ids)-1})"
    else:
        job_cell = f"`{job_id}`"

    rel_path = f"quantum_results/{subdir}/{filename}"
    row = (
        f"| {experiment} | {run_key} | {backend} | {job_cell} | "
        f"{n_qubits} | {cal_ts} | {run_ts} | `{rel_path}` |\n"
    )
    with open(RESULTS_MD, "a", encoding="utf-8") as fh:
        fh.write(row)
    log.info("RESULTS.md updated.")
