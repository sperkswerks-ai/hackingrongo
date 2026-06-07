# quantum_results/

Permanent, versioned record of every IBM Quantum hardware submission made
during this project.  See [`RESULTS.md`](../RESULTS.md) at the repo root for
the tabulated summary.

## Directory layout

```
quantum_results/
  simon/   — Simon's algorithm runs on diachronic key-change passages
  bv/      — Bernstein–Vazirani runs on IC distribution linearity
  qaoa/    — QAOA hybrid decipherment runs
```

## File naming

Each file is named:

```
<run_key>_<ISO-timestamp>_<IBM-job-id>.json
```

For multi-job QAOA runs the first COBYLA job ID is used; the full list is
inside the file under `hardware_provenance.job_ids`.

Files are **never overwritten**.  The timestamp and job ID in the name
guarantee uniqueness across re-runs.

## Verifying a result

Every file contains a `hardware_provenance` block:

```json
{
  "hardware_provenance": {
    "job_id":                 "cxy1234abcd",
    "backend_name":           "ibm_marrakesh",
    "backend_num_qubits":     156,
    "calibration_timestamp":  "2026-06-05T14:23:01+00:00",
    "run_timestamp":          "2026-06-05T15:04:22.341782+00:00",
    "status":                 "JobStatus.DONE"
  }
}
```

Use the `job_id` to look up the job on
[IBM Quantum Platform](https://quantum.ibm.com/jobs) and compare the
measurement histogram against the `counts` field in the file.
