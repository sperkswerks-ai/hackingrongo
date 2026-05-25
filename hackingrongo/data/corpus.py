"""
hackingrongo.data.corpus
===================

Corpus loading, stratum assignment, and train/validation splitting for
the rongorongo tablet dataset.

JSON formats assumed
--------------------

``data/metadata/tablets.json``  — one top-level object, tablet_id → metadata::

    {
        "<tablet_id>": {
            "radiocarbon_date_min": <int>,   # earliest year (CE); negative = BCE
            "radiocarbon_date_max": <int>,   # latest year (CE)
            "wood_species":         "<str>",
            "institution":          "<str>"
        },
        ...
    }

``data/corpus/<tablet_id>.json``  — one file per tablet::

    {
        "tablet_id": "<str>",
        "glyphs": [
            {
                "position":     <int>,     # 1-based ordinal; boustrophedon order
                "barthel_code": "<str>",   # Barthel (1958) code, e.g. "001"
                "tablet_id":    "<str>"    # redundant; kept for self-contained records
            },
            ...
        ]
    }

All numeric constants (thresholds, labels, split ratios) are read from
the Hydra ``DictConfig``; no literals appear in this module.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from omegaconf import DictConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GlyphToken:
    """A single glyph occurrence at a specific position within a tablet.

    Parameters
    ----------
    position : int
        1-based ordinal position of the glyph on the tablet face,
        following the boustrophedon reading order of the source data.
    barthel_code : str
        Barthel (1958) sign identifier, e.g. ``"001"``.  Treated as a
        string throughout the pipeline; no numeric sorting is assumed.
    tablet_id : str
        Identifier of the tablet this glyph belongs to, matching the
        key in ``data/metadata/tablets.json``.
    stratum : str
        Temporal cluster label inherited from the parent tablet:
        ``"pre_contact"`` | ``"post_contact"`` | ``"unknown"`` | ``"excluded"``.
    """

    position: int
    barthel_code: str
    tablet_id: str
    stratum: str  # cluster label: "pre_contact" | "post_contact" | "unknown" | "excluded"
    line_num: int = 0   # 1-based line number within the side; 0 if absent
    side: str = "a"     # tablet side (corpus uses "a"/"b" or "r"/"v" depending on source)


@dataclass
class TabletRecord:
    """All loaded data for a single rongorongo tablet.

    Parameters
    ----------
    tablet_id : str
        Short identifier string (e.g. ``"K"`` for the Kohau motu mo
        Rongorongo) matching the key in ``data/metadata/tablets.json``.
    stratum : str
        Temporal cluster label (``"pre_contact"`` | ``"post_contact"`` |
        ``"unknown"`` | ``"excluded"``) derived from explicit tablet ID lookup
        against the two-cluster config.
    date_midpoint : float
        Midpoint year (CE) of the radiocarbon date range.  Stored for
        downstream diachronic analysis; used only for chronological sorting
        and not for cluster assignment.
    tokens : list[GlyphToken]
        Ordered glyph token sequence for this tablet, sorted ascending
        by ``GlyphToken.position``.
    metadata : dict[str, Any]
        Raw metadata fields from ``tablets.json`` (``wood_species``,
        ``institution``, ``radiocarbon_date_min``,
        ``radiocarbon_date_max``, etc.).
    """

    tablet_id: str
    stratum: str
    date_midpoint: float
    tokens: list[GlyphToken] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def num_tokens(self) -> int:
        """Number of glyph tokens in this tablet's sequence."""
        return len(self.tokens)


# ---------------------------------------------------------------------------
# Cluster assignment
# ---------------------------------------------------------------------------


def assign_cluster(tablet_id: str, cfg: DictConfig) -> str:
    """Assign a temporal cluster label to a tablet by explicit ID lookup.

    The two-cluster model makes assignment deterministic for the six
    radiocarbon-dated tablets and ``"unknown"`` for all others.

    Parameters
    ----------
    tablet_id : str
        Tablet identifier, e.g. ``"D"`` or ``"Q"``.
    cfg : DictConfig
        Root Hydra config.  Reads
        ``cfg.corpus.temporal_model.clusters`` and
        ``cfg.corpus.temporal_model.cluster_labels``.

    Returns
    -------
    str
        One of the four canonical cluster labels:

        * ``"pre_contact"``  — Tablet D; pre-contact anchor
        * ``"post_contact"`` — Tablets B, C, O, Q; 19th century
        * ``"excluded"``     — Tablet A; European wood
        * ``"unknown"``      — All other tablets; undated
    """
    tm = cfg.corpus.temporal_model
    cl = tm.cluster_labels
    clusters = tm.clusters

    if tablet_id in list(clusters.excluded_from_temporal_analysis.tablets):
        return str(cl.excluded)
    if tablet_id in list(clusters.pre_contact.tablets):
        return str(cl.pre_contact)
    if tablet_id in list(clusters.post_contact.tablets):
        return str(cl.post_contact)
    return str(cl.unknown)


def assign_cluster_probability(
    tablet_id: str,
    cfg: DictConfig,
    *,
    feature_vector: "np.ndarray | None" = None,
) -> dict[str, float]:
    """Return a probability distribution over cluster labels for a tablet.

    For tablets in the empirically grounded clusters, returns a deterministic
    distribution (1.0 on the known cluster).  For excluded tablets, returns
    ``{unknown: 1.0}``.  For undated tablets, computes an **empirical class
    prior** from the cluster sizes when no ``feature_vector`` is supplied.
    When a ``feature_vector`` is available, a Gaussian Naive Bayes classifier
    trained on the dated tablets is used; because only five usable anchors
    exist (1 pre_contact, 4 post_contact), confidence intervals will be wide —
    this is correct and should be reported, not masked.

    Parameters
    ----------
    tablet_id : str
        Tablet identifier.
    cfg : DictConfig
        Root Hydra config.  Reads ``cfg.corpus.temporal_model``.
    feature_vector : numpy.ndarray or None
        Optional feature vector, e.g. mean Zone A embedding concatenated with
        Zone B contact statistics.  When ``None``, returns the empirical prior.
        GNB implementation is triggered when a vector is supplied but falls back
        to the prior until the classifier is trained (Zone A embeddings required).

    Returns
    -------
    dict[str, float]
        Probability distribution over ``{"pre_contact", "post_contact",
        "unknown"}``.  Values sum to 1.0.

    Notes
    -----
    The tablet entropy ``H = -sum(p * log2(p))`` over
    ``{pre_contact, post_contact}`` is a direct measure of dating uncertainty.
    Tablets with entropy near 1.0 bit are the highest priority for physical
    radiocarbon dating — they would most reduce the corpus's temporal
    uncertainty.
    """
    cluster = assign_cluster(tablet_id, cfg)
    cl = cfg.corpus.temporal_model.cluster_labels
    pre_label = str(cl.pre_contact)
    post_label = str(cl.post_contact)
    unk_label = str(cl.unknown)

    # Deterministic cases for already-anchored tablets.
    if cluster == str(cl.excluded):
        return {pre_label: 0.0, post_label: 0.0, unk_label: 1.0}
    if cluster == pre_label:
        return {pre_label: 1.0, post_label: 0.0, unk_label: 0.0}
    if cluster == post_label:
        return {pre_label: 0.0, post_label: 1.0, unk_label: 0.0}

    # Undated tablet — compute empirical class prior from cluster sizes.
    clusters = cfg.corpus.temporal_model.clusters
    n_pre = len(list(clusters.pre_contact.tablets))
    n_post = len(list(clusters.post_contact.tablets))
    n_total = n_pre + n_post

    if n_total == 0:
        return {pre_label: 0.0, post_label: 0.0, unk_label: 1.0}

    prior_pre = n_pre / n_total
    prior_post = n_post / n_total

    if feature_vector is not None:
        # GNB classifier not yet implemented; requires Zone A embeddings.
        # Returns empirical prior until classifier is trained.
        logger.warning(
            "Tablet '%s': feature_vector supplied but GNB classifier is not yet "
            "trained. Returning empirical class prior. Train the classifier once "
            "Zone A embeddings are available.",
            tablet_id,
        )

    return {pre_label: prior_pre, post_label: prior_post, unk_label: 0.0}


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _compute_date_midpoint(tablet_meta: dict[str, Any], tablet_id: str) -> float:
    """Return a representative year (CE) for chronological sorting.

    Prefers ``date_distribution.median_CE`` when present.  Falls back to the
    arithmetic midpoint of ``radiocarbon_date_min`` / ``radiocarbon_date_max``
    for tablets that do not yet have a calibrated distribution object.

    Parameters
    ----------
    tablet_meta : dict[str, Any]
        Metadata dict for a single tablet from ``tablets.json``.
    tablet_id : str
        Used only in error messages.

    Returns
    -------
    float
        Representative year (CE).  Used only for chronological ordering, not
        for cluster assignment.

    Raises
    ------
    KeyError
        If neither ``date_distribution.median_CE`` nor the
        ``radiocarbon_date_min`` / ``radiocarbon_date_max`` pair is present.
    ValueError
        If ``radiocarbon_date_min > radiocarbon_date_max``.
    """
    # Prefer calibrated median when available.
    dist = tablet_meta.get("date_distribution")
    if dist is not None and "median_CE" in dist:
        return float(dist["median_CE"])

    # Fall back to midpoint of the legacy range fields.
    try:
        date_min: int = int(tablet_meta["radiocarbon_date_min"])
        date_max: int = int(tablet_meta["radiocarbon_date_max"])
    except KeyError as exc:
        raise KeyError(
            f"Tablet '{tablet_id}': missing date field {exc} in metadata and "
            f"no date_distribution.median_CE present."
        ) from exc

    if date_min > date_max:
        raise ValueError(
            f"Tablet '{tablet_id}': radiocarbon_date_min ({date_min}) > "
            f"radiocarbon_date_max ({date_max})."
        )
    return (date_min + date_max) / 2.0


def _build_glyph_tokens(
    raw_glyphs: list[dict[str, Any]],
    tablet_id: str,
    stratum: str,
) -> list[GlyphToken]:
    """Parse raw glyph dicts from a corpus JSON into ``GlyphToken`` objects.

    Parameters
    ----------
    raw_glyphs : list[dict[str, Any]]
        List of glyph records as loaded directly from JSON.  Each must
        contain ``"position"`` (int) and ``"barthel_code"`` (str).
    tablet_id : str
        Parent tablet identifier; injected into each token.
    stratum : str
        Parent tablet stratum label; injected into each token.

    Returns
    -------
    list[GlyphToken]
        Tokens sorted in ascending order by ``position``.

    Raises
    ------
    KeyError
        If a glyph record is missing a required field.
    """
    tokens: list[GlyphToken] = []
    for idx, g in enumerate(raw_glyphs):
        try:
            pos: int = int(g["position"])
            code: str = str(g["barthel_code"])
        except KeyError as exc:
            raise KeyError(
                f"Tablet '{tablet_id}', glyph index {idx}: missing field {exc}."
            ) from exc
        try:
            line_num: int = int(g.get("line", 0))
        except (ValueError, TypeError):
            line_num = 0
        side: str = str(g.get("side", "a"))
        tokens.append(
            GlyphToken(position=pos, barthel_code=code,
                       tablet_id=tablet_id, stratum=stratum,
                       line_num=line_num, side=side)
        )
    tokens.sort(key=lambda t: t.position)
    return tokens


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_tablet_metadata(tablets_json_path: Path) -> dict[str, dict[str, Any]]:
    """Load tablet-level metadata from the central JSON file.

    Parameters
    ----------
    tablets_json_path : Path
        Absolute path to ``data/metadata/tablets.json``.
        Expected schema: top-level object mapping ``tablet_id`` strings
        to metadata dicts containing at minimum
        ``radiocarbon_date_min`` and ``radiocarbon_date_max`` (int, CE).

    Returns
    -------
    dict[str, dict[str, Any]]
        Mapping of tablet identifiers to their raw metadata dicts.

    Raises
    ------
    FileNotFoundError
        If the JSON file does not exist at the given path.
    json.JSONDecodeError
        If the file is not valid JSON.
    """
    if not tablets_json_path.exists():
        raise FileNotFoundError(
            f"Tablet metadata file not found: {tablets_json_path}"
        )
    with tablets_json_path.open("r", encoding="utf-8") as fh:
        data: dict[str, dict[str, Any]] = json.load(fh)
    logger.debug(
        "Loaded metadata for %d tablets from %s", len(data), tablets_json_path
    )
    return data


def load_corpus(cfg: DictConfig, project_root: Path) -> list[TabletRecord]:
    """Load the full rongorongo tablet corpus from disk.

    Reads all ``<tablet_id>.json`` files from the corpus directory,
    cross-references tablet-level metadata (radiocarbon dates, etc.),
    assigns temporal stratum labels, and filters tablets that fall below
    the minimum token count.

    Parameters
    ----------
    cfg : DictConfig
        Root Hydra config.  Uses ``cfg.paths.corpus_dir``,
        ``cfg.paths.tablets_json``, ``cfg.corpus.min_tablet_tokens``,
        and all ``cfg.corpus.stratum_*`` keys.
    project_root : Path
        Absolute path to the repository root, obtained in
        ``pipeline.py`` via ``hydra.utils.get_original_cwd()``.
        All relative paths in ``cfg.paths`` are resolved against this
        root.  Do **not** pass ``os.getcwd()`` here.

    Returns
    -------
    list[TabletRecord]
        All tablets that passed the minimum-token filter, sorted
        ascending by ``date_midpoint`` for deterministic chronological
        ordering.

    Raises
    ------
    FileNotFoundError
        If ``corpus_dir`` or ``tablets_json`` do not exist at their
        resolved paths.
    KeyError, ValueError
        If a corpus JSON or metadata entry is structurally invalid
        (missing required fields or inconsistent date range).

    Notes
    -----
    * Tablets whose ``tablet_id`` appears in the corpus directory but
      is absent from ``tablets.json`` are **skipped with a WARNING**,
      not an error, to allow incremental data additions.
    * Metadata entries with no corresponding corpus file are logged at
      DEBUG level only.
    * Tablets with fewer than ``cfg.corpus.min_tablet_tokens`` tokens
      are excluded and logged at WARNING level.
    """
    corpus_dir: Path = project_root / cfg.paths.corpus_dir
    tablets_json_path: Path = project_root / cfg.paths.tablets_json
    min_tokens: int = cfg.corpus.min_tablet_tokens

    if not corpus_dir.is_dir():
        raise FileNotFoundError(f"Corpus directory not found: {corpus_dir}")

    tablet_metadata = load_tablet_metadata(tablets_json_path)

    corpus_files = sorted(corpus_dir.glob("*.json"))
    if not corpus_files:
        logger.warning("No JSON files found in corpus directory: %s", corpus_dir)

    records: list[TabletRecord] = []
    skipped_no_meta: int = 0
    skipped_too_short: int = 0

    for corpus_file in corpus_files:
        tablet_id: str = corpus_file.stem  # filename without .json extension

        if tablet_id not in tablet_metadata:
            logger.warning(
                "Corpus file '%s' has no entry in tablets.json; skipping.",
                corpus_file.name,
            )
            skipped_no_meta += 1
            continue

        meta: dict[str, Any] = tablet_metadata[tablet_id]
        date_midpoint: float = _compute_date_midpoint(meta, tablet_id)
        stratum: str = assign_cluster(tablet_id, cfg)

        with corpus_file.open("r", encoding="utf-8") as fh:
            raw: dict[str, Any] = json.load(fh)

        raw_glyphs: list[dict[str, Any]] = raw.get("glyphs", [])
        tokens = _build_glyph_tokens(raw_glyphs, tablet_id, stratum)

        if len(tokens) < min_tokens:
            logger.warning(
                "Tablet '%s' has %d token(s), below min_tablet_tokens=%d; skipping.",
                tablet_id,
                len(tokens),
                min_tokens,
            )
            skipped_too_short += 1
            continue

        records.append(
            TabletRecord(
                tablet_id=tablet_id,
                stratum=stratum,
                date_midpoint=date_midpoint,
                tokens=tokens,
                metadata=meta,
            )
        )

    # Log metadata-only entries (present in tablets.json but no corpus file).
    corpus_ids = {f.stem for f in corpus_files}
    for tid in sorted(set(tablet_metadata.keys()) - corpus_ids):
        logger.debug(
            "Tablet '%s' has a metadata entry but no corpus file; not loaded.", tid
        )

    logger.info(
        "Corpus loaded: %d tablet(s) retained, %d skipped (no metadata), "
        "%d skipped (below min_tablet_tokens=%d).",
        len(records),
        skipped_no_meta,
        skipped_too_short,
        min_tokens,
    )

    # Sort chronologically for deterministic downstream ordering.
    records.sort(key=lambda r: r.date_midpoint)
    return records


def split_by_stratum(
    tablets: list[TabletRecord],
) -> dict[str, list[TabletRecord]]:
    """Partition a tablet list into per-cluster sub-lists.

    The function is label-agnostic: it groups tablets by whatever string is
    stored in ``TabletRecord.stratum``, which under the two-cluster model will
    be one of ``"pre_contact"``, ``"post_contact"``, ``"unknown"``, or
    ``"excluded"``.

    Also exported as ``split_by_cluster`` (preferred name).
    """
    result: dict[str, list[TabletRecord]] = {}
    for tablet in tablets:
        result.setdefault(tablet.stratum, []).append(tablet)
    return result


#: Preferred alias for :func:`split_by_stratum`.
split_by_cluster = split_by_stratum


def make_train_val_split(
    tablets: list[TabletRecord],
    cfg: DictConfig,
    rng: np.random.Generator,
) -> tuple[list[TabletRecord], list[TabletRecord]]:
    """Split tablets into training and validation sets.

    The split is applied **independently within each temporal stratum**
    to prevent stratum leakage — i.e. the train/val ratio is maintained
    per stratum, not merely globally.

    Parameters
    ----------
    tablets : list[TabletRecord]
        Full tablet list, as returned by :func:`load_corpus`.
    cfg : DictConfig
        Root Hydra config.  Uses ``cfg.corpus.train_split_ratio``.
        The validation ratio is derived as
        ``1.0 - cfg.corpus.train_split_ratio`` (one source of truth).
    rng : numpy.random.Generator
        Seeded random number generator for reproducible shuffling.
        Construct with ``numpy.random.default_rng(cfg.seed)`` in
        ``pipeline.py``.

    Returns
    -------
    tuple[list[TabletRecord], list[TabletRecord]]
        ``(train_tablets, val_tablets)``.  Stratum proportions are
        approximately preserved in both lists.

    Notes
    -----
    When a stratum contains fewer than 2 tablets, all of its tablets
    are placed in the training set and a WARNING is emitted.  This
    avoids producing an empty validation split for a rare stratum.
    """
    train_ratio: float = float(cfg.corpus.train_split_ratio)
    by_stratum = split_by_stratum(tablets)

    train_list: list[TabletRecord] = []
    val_list: list[TabletRecord] = []

    for stratum, stratum_tablets in sorted(by_stratum.items()):
        n = len(stratum_tablets)
        indices = np.arange(n)
        rng.shuffle(indices)

        if n < 2:
            logger.warning(
                "Stratum '%s' has only %d tablet(s); placing all in training set.",
                stratum,
                n,
            )
            for i in indices:
                train_list.append(stratum_tablets[i])
            continue

        n_train: int = max(1, math.floor(n * train_ratio))
        for i in indices[:n_train]:
            train_list.append(stratum_tablets[i])
        for i in indices[n_train:]:
            val_list.append(stratum_tablets[i])

        logger.debug(
            "Stratum '%s': %d train, %d val (train_ratio=%.2f).",
            stratum,
            n_train,
            n - n_train,
            train_ratio,
        )

    logger.info(
        "Train/val split: %d train tablet(s), %d val tablet(s).",
        len(train_list),
        len(val_list),
    )
    return train_list, val_list


def get_corpus_token_sequence(tablets: list[TabletRecord]) -> list[GlyphToken]:
    """Flatten a list of tablet records into a single ordered token sequence.

    Useful for Zone B analysis methods that operate over the full corpus
    rather than per-tablet.  Tablet order follows the order of the input
    list (chronological after :func:`load_corpus`).

    Parameters
    ----------
    tablets : list[TabletRecord]
        List of tablet records, each with an ordered ``.tokens`` list.

    Returns
    -------
    list[GlyphToken]
        Concatenated list of all glyph tokens across all tablets, in
        tablet-input order and then by per-tablet position order.
    """
    return [token for tablet in tablets for token in tablet.tokens]
