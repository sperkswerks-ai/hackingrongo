"""
hackingrongo.results.schema
======================

Typed dataclasses for decipherment hypothesis output.

These classes represent the final output of the hackingrongo pipeline — the
ranked list of phoneme-assignment hypotheses with confidence intervals
and cross-stratum scores.  They carry their own serialisation methods
(``to_json``, ``to_csv``, ``to_markdown``) so that results can be
compared across runs, read by a human reviewer without code, and
attached to an academic paper as supplementary data.

Unlike Souza (2022), which produces an untyped ``results.csv`` with no
schema, version information, or connection to the parameters that
generated it, every :class:`DecryptionHypothesis` carries:

* An MLflow ``run_id`` for parameter provenance.
* A ``config_hash`` (SHA-256 of ``conf/config.yaml``) so any run can be
  exactly reproduced from a locked config.
* A ``created_at`` ISO-8601 timestamp.

Public API
----------
``PhonemeAssignment``
    One sign ↔ phoneme mapping with confidence and evidence counts.

``StratumScore``
    Cross-stratum consistency score for one temporal stratum.

``DecryptionHypothesis``
    Complete single-run hypothesis: all assignments + stratum scores +
    scalar quality metrics.  Serialisable to JSON / CSV / Markdown.

``HypothesisRanking``
    Ordered collection of hypotheses from one experiment.  Serialisable
    to all three formats.

``load_hypothesis``
    Deserialise a :class:`DecryptionHypothesis` from a JSON file.

``load_ranking``
    Deserialise a :class:`HypothesisRanking` from a JSON file.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# PhonemeAssignment
# ---------------------------------------------------------------------------


@dataclass
class PhonemeAssignment:
    """A single sign ↔ phoneme hypothesis with confidence evidence.

    Parameters
    ----------
    sign_code : str
        Barthel code of the assigned sign.
    phoneme : str
        Proposed phoneme or syllable value (e.g. ``"ku"``, ``"ma"``).
    confidence : float
        Posterior probability or normalised beam score ``∈ [0, 1]``.
    evidence_count : int
        Number of distinct corpus positions contributing to this
        assignment's score.
    stratum_breakdown : dict[str, float]
        Per-stratum LM score contribution, keyed by stratum label.
    """

    sign_code: str
    phoneme: str
    confidence: float
    evidence_count: int
    stratum_breakdown: dict[str, float] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# StratumScore
# ---------------------------------------------------------------------------


@dataclass
class StratumScore:
    """Cross-stratum consistency score for one temporal stratum.

    Parameters
    ----------
    stratum : str
        Stratum label (``"pre"`` | ``"early"`` | ``"late"``).
    consistency_score : float
        Intra-stratum agreement score ``∈ [0, 1]`` — fraction of
        parallel passages in this stratum that score above the
        ``cross_stratum_significance_alpha`` threshold.
    lm_score_mean : float
        Mean language-model log₂ probability across passages in this
        stratum under the hypothesis.
    lm_score_std : float
        Standard deviation of the per-passage LM scores.
    n_passages : int
        Number of parallel passages used in this stratum's evaluation.
    languages_above_baseline : list[str]
        Languages for which this stratum's LM score exceeds the random
        baseline (``≥ cross_stratum_min_languages`` required for the
        hypothesis to pass validation).
    """

    stratum: str
    consistency_score: float
    lm_score_mean: float
    lm_score_std: float
    n_passages: int
    languages_above_baseline: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# DecryptionHypothesis
# ---------------------------------------------------------------------------


@dataclass
class DecryptionHypothesis:
    """Complete single-run decipherment hypothesis.

    A hypothesis represents one fully evaluated phoneme-assignment
    mapping produced by the MCMC sampler or beam-search decoder.

    Parameters
    ----------
    hypothesis_id : str
        Unique identifier within an experiment (e.g. ``"H0001"``).
    run_id : str
        MLflow run ID for parameter / artefact provenance.
    hypothesis_type : str
        Structural hypothesis tested:
        ``"syllabic"`` | ``"logographic"`` | ``"semasiographic"``.
    assignments : list[PhonemeAssignment]
        One entry per sign in the active inventory, sorted by
        ``sign_code``.
    stratum_scores : list[StratumScore]
        One entry per temporal stratum.
    overall_lm_score : float
        Ensemble-weighted language-model log₂ probability over the full
        parallel-passage test set.
    mcmc_log_posterior : float
        Log posterior probability of the assignment under the MCMC model
        (mean of post-burn-in samples).
    beam_score : float
        Total beam-search log₂ score (``0.0`` if beam search was not
        used).
    created_at : str
        ISO-8601 UTC timestamp of when this hypothesis was generated.
    config_hash : str
        SHA-256 hex digest of ``conf/config.yaml`` at the time of the
        run, for exact reproducibility.
    """

    hypothesis_id: str
    run_id: str
    hypothesis_type: str
    assignments: list[PhonemeAssignment]
    stratum_scores: list[StratumScore]
    overall_lm_score: float
    mcmc_log_posterior: float
    beam_score: float
    created_at: str = field(
        default_factory=lambda: datetime.now(tz=timezone.utc).isoformat()
    )
    config_hash: str = ""

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_json(self, indent: int = 2) -> str:
        """Serialise to a JSON string.

        Parameters
        ----------
        indent : int
            JSON indentation level.

        Returns
        -------
        str
        """
        return json.dumps(asdict(self), indent=indent, ensure_ascii=False)

    def to_csv(self) -> str:
        """Serialise the phoneme assignments to a CSV string.

        Returns
        -------
        str
            CSV with header ``sign_code,phoneme,confidence,evidence_count``.
            One row per :class:`PhonemeAssignment`, sorted by ``sign_code``.
        """
        buf = io.StringIO()
        writer = csv.writer(buf)
        writer.writerow(["sign_code", "phoneme", "confidence", "evidence_count"])
        for a in sorted(self.assignments, key=lambda x: x.sign_code):
            writer.writerow(
                [a.sign_code, a.phoneme, f"{a.confidence:.6f}", a.evidence_count]
            )
        return buf.getvalue()

    def to_markdown(self) -> str:
        """Serialise the hypothesis to a Markdown table.

        Returns
        -------
        str
            Markdown with header rows and a table of assignments.
        """
        lines: list[str] = [
            f"## Decipherment Hypothesis `{self.hypothesis_id}`",
            "",
            f"| Field | Value |",
            f"|-------|-------|",
            f"| Run ID | `{self.run_id}` |",
            f"| Type | {self.hypothesis_type} |",
            f"| Overall LM score | {self.overall_lm_score:.4f} |",
            f"| MCMC log-posterior | {self.mcmc_log_posterior:.4f} |",
            f"| Beam score | {self.beam_score:.4f} |",
            f"| Created | {self.created_at} |",
            f"| Config hash | `{self.config_hash[:12]}...` |",
            "",
            "### Phoneme Assignments",
            "",
            "| Sign | Phoneme | Confidence | Evidence |",
            "|------|---------|------------|----------|",
        ]
        for a in sorted(self.assignments, key=lambda x: x.sign_code):
            lines.append(
                f"| {a.sign_code} | {a.phoneme} | {a.confidence:.4f} | {a.evidence_count} |"
            )
        lines += [
            "",
            "### Stratum Scores",
            "",
            "| Stratum | Consistency | LM Mean | LM Std | Passages | Languages |",
            "|---------|-------------|---------|--------|----------|-----------|",
        ]
        for s in self.stratum_scores:
            langs = ", ".join(s.languages_above_baseline)
            lines.append(
                f"| {s.stratum} | {s.consistency_score:.4f} | "
                f"{s.lm_score_mean:.4f} | {s.lm_score_std:.4f} | "
                f"{s.n_passages} | {langs} |"
            )
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------

    def save(self, path: Path) -> None:
        """Write the hypothesis to a JSON file.

        Parameters
        ----------
        path : Path
            Destination path.  Parent directories are created if needed.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_json(), encoding="utf-8")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DecryptionHypothesis":
        """Deserialise from a plain dict (e.g., parsed JSON).

        Parameters
        ----------
        data : dict[str, Any]
            Dict as produced by :meth:`to_json` / ``json.loads``.

        Returns
        -------
        DecryptionHypothesis
        """
        data = dict(data)
        assignments = [
            PhonemeAssignment(**a) for a in data.pop("assignments", [])
        ]
        stratum_scores = [
            StratumScore(**s) for s in data.pop("stratum_scores", [])
        ]
        return cls(assignments=assignments, stratum_scores=stratum_scores, **data)


# ---------------------------------------------------------------------------
# HypothesisRanking
# ---------------------------------------------------------------------------


@dataclass
class HypothesisRanking:
    """Ordered collection of :class:`DecryptionHypothesis` objects.

    Hypotheses are stored in descending order of their ranking metric
    score (best first).

    Parameters
    ----------
    hypotheses : list[DecryptionHypothesis]
        All hypotheses from the experiment, sorted best-first.
    ranking_metric : str
        Name of the scalar used to sort hypotheses (e.g.
        ``"overall_lm_score"`` or ``"mcmc_log_posterior"``).
    generated_at : str
        ISO-8601 UTC timestamp.
    """

    hypotheses: list[DecryptionHypothesis]
    ranking_metric: str
    generated_at: str = field(
        default_factory=lambda: datetime.now(tz=timezone.utc).isoformat()
    )

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def top_n(self, n: int) -> list[DecryptionHypothesis]:
        """Return the top ``n`` hypotheses (best first).

        Parameters
        ----------
        n : int
            Number of hypotheses to return.

        Returns
        -------
        list[DecryptionHypothesis]
        """
        return self.hypotheses[:n]

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_json(self, indent: int = 2) -> str:
        """Serialise the full ranking to a JSON string.

        Returns
        -------
        str
        """
        return json.dumps(asdict(self), indent=indent, ensure_ascii=False)

    def to_csv(self) -> str:
        """Serialise to a CSV summary of all hypotheses (one row each).

        Returns
        -------
        str
            CSV with columns for all scalar fields of
            :class:`DecryptionHypothesis`.
        """
        buf = io.StringIO()
        writer = csv.writer(buf)
        writer.writerow(
            [
                "rank",
                "hypothesis_id",
                "run_id",
                "hypothesis_type",
                "overall_lm_score",
                "mcmc_log_posterior",
                "beam_score",
                "n_assignments",
                "created_at",
                "config_hash",
            ]
        )
        for rank, hyp in enumerate(self.hypotheses, start=1):
            writer.writerow(
                [
                    rank,
                    hyp.hypothesis_id,
                    hyp.run_id,
                    hyp.hypothesis_type,
                    f"{hyp.overall_lm_score:.6f}",
                    f"{hyp.mcmc_log_posterior:.6f}",
                    f"{hyp.beam_score:.6f}",
                    len(hyp.assignments),
                    hyp.created_at,
                    hyp.config_hash,
                ]
            )
        return buf.getvalue()

    def to_markdown(self) -> str:
        """Serialise the ranking to a Markdown table.

        Returns
        -------
        str
        """
        lines: list[str] = [
            "# Hypothesis Ranking",
            "",
            f"Ranking metric: `{self.ranking_metric}`  ",
            f"Generated: {self.generated_at}",
            "",
            "| Rank | ID | Type | LM Score | MCMC Log-Post | Beam Score |",
            "|------|----|------|----------|---------------|------------|",
        ]
        for rank, hyp in enumerate(self.hypotheses, start=1):
            lines.append(
                f"| {rank} | `{hyp.hypothesis_id}` | {hyp.hypothesis_type} | "
                f"{hyp.overall_lm_score:.4f} | {hyp.mcmc_log_posterior:.4f} | "
                f"{hyp.beam_score:.4f} |"
            )
        return "\n".join(lines)

    def save(self, path: Path) -> None:
        """Write the ranking to a JSON file.

        Parameters
        ----------
        path : Path
            Destination path.  Parent directories are created if needed.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_json(), encoding="utf-8")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "HypothesisRanking":
        """Deserialise from a plain dict.

        Parameters
        ----------
        data : dict[str, Any]

        Returns
        -------
        HypothesisRanking
        """
        data = dict(data)
        hypotheses = [
            DecryptionHypothesis.from_dict(h) for h in data.pop("hypotheses", [])
        ]
        return cls(hypotheses=hypotheses, **data)


# ---------------------------------------------------------------------------
# File-level I/O helpers
# ---------------------------------------------------------------------------


def load_hypothesis(path: Path) -> DecryptionHypothesis:
    """Deserialise a :class:`DecryptionHypothesis` from a JSON file.

    Parameters
    ----------
    path : Path
        Path to a JSON file written by :meth:`DecryptionHypothesis.save`.

    Returns
    -------
    DecryptionHypothesis

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    """
    if not path.exists():
        raise FileNotFoundError(f"Hypothesis file not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    return DecryptionHypothesis.from_dict(data)


def load_ranking(path: Path) -> HypothesisRanking:
    """Deserialise a :class:`HypothesisRanking` from a JSON file.

    Parameters
    ----------
    path : Path
        Path to a JSON file written by :meth:`HypothesisRanking.save`.

    Returns
    -------
    HypothesisRanking

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    """
    if not path.exists():
        raise FileNotFoundError(f"Ranking file not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    return HypothesisRanking.from_dict(data)


def hash_config_file(config_path: Path) -> str:
    """Compute a SHA-256 hex digest of a config file for reproducibility.

    Parameters
    ----------
    config_path : Path
        Path to ``conf/config.yaml``.

    Returns
    -------
    str
        64-character hex string, or ``""`` if the file does not exist.
    """
    if not config_path.exists():
        return ""
    return hashlib.sha256(config_path.read_bytes()).hexdigest()
