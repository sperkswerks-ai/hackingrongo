"""
hackingrongo.data.dataset
====================

PyTorch ``Dataset`` classes for the three Zone A networks.

Classes
-------
``GlyphImageDataset``
    Maps each glyph token to its PNG image on disk.  Used by both the
    convolutional autoencoder (single-image reconstruction) and as the
    base loader for the Siamese network.

``GlyphSequenceDataset``
    Presents sliding context-window slices over a tablet's glyph token
    sequence for next-glyph prediction training.

``SiamesePairDataset``
    Wraps ``GlyphImageDataset`` and samples (anchor, pair, label)
    triplets according to allograph catalog groupings, with optional
    semi-hard negative mining support.

All hyperparameters (image size, context window, pair sampling ratios,
etc.) are read exclusively from the Hydra ``DictConfig``; no literals
appear in this module.
"""

from __future__ import annotations

import json
import logging
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import DictConfig
from PIL import Image, ImageOps
from torch.utils.data import Dataset
from torchvision import transforms

from hackingrongo.data.corpus import GlyphToken, TabletRecord

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Allograph catalog helpers
# ---------------------------------------------------------------------------


def load_allograph_catalog(
    catalog_path: Path,
) -> dict[str, str]:
    """Load the Horley 2021 allograph groupings from disk.

    Parameters
    ----------
    catalog_path : Path
        Absolute path to ``data/catalog/allographs.json``.

        Expected schema::

            {
                "<variant_barthel_code>": "<canonical_sign_id>",
                ...
            }

        Every glyph variant maps to a canonical sign identifier.
        Glyphs that *are* canonical signs map to themselves.

    Returns
    -------
    dict[str, str]
        Mapping from variant Barthel code string to canonical sign ID.

    Raises
    ------
    FileNotFoundError
        If the catalog file does not exist.
    json.JSONDecodeError
        If the file is not valid JSON.
    """
    if not catalog_path.exists():
        raise FileNotFoundError(
            f"Allograph catalog not found: {catalog_path}"
        )
    with catalog_path.open("r", encoding="utf-8") as fh:
        catalog: dict[str, str] = json.load(fh)
    logger.debug(
        "Loaded allograph catalog: %d variant → canonical mappings from %s",
        len(catalog),
        catalog_path,
    )
    return catalog


def build_sign_groups(
    catalog: dict[str, str],
) -> dict[str, list[str]]:
    """Invert an allograph catalog into sign-group membership lists.

    Parameters
    ----------
    catalog : dict[str, str]
        Output of :func:`load_allograph_catalog`.  Maps each variant
        Barthel code to its canonical sign ID.

    Returns
    -------
    dict[str, list[str]]
        Maps each canonical sign ID to a list of all Barthel codes
        (including the canonical code itself) that belong to that sign
        group.  Used by :class:`SiamesePairDataset` to identify
        same-sign pairs.
    """
    groups: dict[str, list[str]] = {}
    for variant, canonical in catalog.items():
        groups.setdefault(canonical, []).append(variant)
    return groups


# ---------------------------------------------------------------------------
# Image transform factory
# ---------------------------------------------------------------------------


def _make_transform(cfg: DictConfig, training: bool) -> transforms.Compose:
    """Build the torchvision transform pipeline for glyph images.

    Parameters
    ----------
    cfg : DictConfig
        Root Hydra config.  Uses ``cfg.glyph`` (``image_size``,
        ``image_channels``) and ``cfg.glyph.augmentation``.
    training : bool
        When ``True`` and ``cfg.glyph.augmentation.use_augmentation``
        is enabled, stochastic augmentation transforms are prepended.
        Set to ``False`` for validation and inference.

    Returns
    -------
    torchvision.transforms.Compose
        Composed transform that converts a PIL image to a normalised
        float tensor of shape
        ``(image_channels, image_size, image_size)``.
    """
    size: int = int(cfg.glyph.image_size)
    channels: int = int(cfg.glyph.image_channels)
    aug = cfg.glyph.augmentation

    step_list: list[Any] = []

    if training and bool(aug.use_augmentation):
        translate = tuple(float(v) for v in aug.random_affine_translate)
        scale = tuple(float(v) for v in aug.random_affine_scale)

        step_list += [
            transforms.RandomRotation(degrees=float(aug.random_rotation_degrees)),
            transforms.RandomAffine(
                degrees=0,
                translate=translate,
                scale=scale,
            ),
            transforms.ElasticTransform(
                alpha=float(aug.elastic_transform_alpha),
                sigma=float(aug.elastic_transform_sigma),
            ),
        ]

    # Resize, convert to grayscale or RGB, then to tensor + normalise.
    step_list += [
        transforms.Resize((size, size)),
        transforms.Grayscale(num_output_channels=channels)
        if channels == 1
        else transforms.Lambda(lambda img: img.convert("RGB")),
        transforms.ToTensor(),                       # → float in [0, 1]
        transforms.Normalize(
            mean=[0.5] * channels,
            std=[0.5] * channels,
        ),                                           # → float in [-1, 1]
    ]

    if training and bool(aug.use_augmentation):
        noise_std = float(aug.gaussian_noise_std)
        step_list.append(
            transforms.Lambda(
                lambda t: t + torch.randn_like(t) * noise_std
            )
        )

    return transforms.Compose(step_list)


def _looks_like_low_contrast_relief(
    gray: Image.Image,
    *,
    is_3d_crop: bool,
    std_threshold: float,
    dynamic_range_threshold: float,
    mean_floor: float,
    dark_ratio_ceiling: float,
) -> bool:
    """Return True for pale shallow-relief crops that need contrast expansion."""
    arr = np.asarray(gray, dtype=np.uint8)
    if arr.size == 0:
        return False

    mean = float(arr.mean())
    std = float(arr.std())
    p1, p99 = np.percentile(arr, [1, 99])
    dynamic_range = float(p99 - p1)
    dark_ratio = float(np.mean(arr < 64))

    relief_like = (
        mean > mean_floor
        and std < std_threshold
        and dynamic_range < dynamic_range_threshold
        and dark_ratio < dark_ratio_ceiling
    )

    if is_3d_crop:
        return relief_like or (
            mean > (mean_floor - 10.0)
            and std < (std_threshold + 2.0)
            and dynamic_range < (dynamic_range_threshold + 12.0)
            and dark_ratio < dark_ratio_ceiling
        )
    return relief_like


def _enhance_low_contrast_relief(
    image: Image.Image,
    *,
    autocontrast_cutoff: float,
) -> Image.Image:
    """Expand contrast for low-relief grayscale inputs without inverting them."""
    gray = image.convert("L")
    enhanced = ImageOps.autocontrast(gray, cutoff=autocontrast_cutoff)
    return ImageOps.equalize(enhanced)


def _token_corpus_key(token: GlyphToken, seq_on_line: int) -> str:
    side_ab = {"r": "a", "v": "b", "a": "a", "b": "b"}.get(str(token.side), "a")
    line = f"{int(token.line_num):02d}" if int(token.line_num) > 0 else "00"
    return f"{token.tablet_id}{side_ab}{line}-{seq_on_line:03d}"


# ---------------------------------------------------------------------------
# GlyphImageDataset
# ---------------------------------------------------------------------------


class GlyphImageDataset(Dataset):
    """PyTorch Dataset mapping each glyph token to its PNG image.

    Loads glyph images lazily from ``data/glyphs/`` using the filename
    convention ``{tablet_id}_{position}_{barthel_code}.png`` defined in
    ``cfg.glyph.filename_pattern``.

    Used by:

    * The convolutional autoencoder (input = output = single image tensor)
    * :class:`SiamesePairDataset` (as the underlying image provider)

    Parameters
    ----------
    tokens : list[GlyphToken]
        Flat list of glyph tokens to include in this dataset split.
        Typically produced by
        :func:`hackingrongo.data.corpus.get_corpus_token_sequence`.
    glyphs_dir : Path
        Absolute path to the directory containing PNG glyph images.
    cfg : DictConfig
        Root Hydra config.  Uses ``cfg.glyph``.
    training : bool, optional
        If ``True``, applies stochastic augmentation transforms.
        Defaults to ``True``.

    Attributes
    ----------
    tokens : list[GlyphToken]
        The token list passed at construction time.
    vocab : list[str]
        Sorted deduplicated list of all Barthel codes present in
        ``tokens``.  Consistent index → token-ID mapping used by
        :class:`GlyphSequenceDataset`.
    barthel_to_id : dict[str, int]
        Maps each Barthel code string to its integer token ID.

    Notes
    -----
    At construction time, all tokens are checked for a resolvable image
    (corpus-specific PNG or ``barthel_ref/`` fallback).  Tokens with no
    image are **excluded** from ``self.tokens`` and a single WARNING is
    emitted listing the missing Barthel codes.  Zero-tensor substitution
    would contaminate the embedding space and the SupCon loss; exclusion
    is the correct approach.

    Images may be absent because ``barthel_corpus/`` extraction (OCR /
    PDF-alignment) was incomplete, or because a code variant has no entry
    in ``barthel_ref/``.  Running
    ``scripts/extract_barthel_glyphs.py --source both`` regenerates them.
    """

    def __init__(
        self,
        tokens: list[GlyphToken],
        glyphs_dir: Path,
        cfg: DictConfig,
        training: bool = True,
    ) -> None:
        super().__init__()
        self.tokens: list[GlyphToken] = tokens
        self.glyphs_dir: Path = glyphs_dir
        self._cfg = cfg
        self._training = training
        self._transform = _make_transform(cfg, training)
        self._filename_pattern: str = str(cfg.glyph.filename_pattern)
        self._image_size: int = int(cfg.glyph.image_size)
        self._channels: int = int(cfg.glyph.image_channels)
        pp_cfg = cfg.get("zone_a", {}).get("preprocessing", {}) if cfg.get("zone_a") else {}
        self._enhance_low_contrast: bool = bool(pp_cfg.get("enhance_low_contrast", True))
        self._lc_std_threshold: float = float(pp_cfg.get("low_contrast_std_threshold", 10.0))
        self._lc_dynamic_range_threshold: float = float(pp_cfg.get("low_contrast_dynamic_range_threshold", 35.0))
        self._lc_mean_floor: float = float(pp_cfg.get("low_contrast_mean_floor", 150.0))
        self._lc_dark_ratio_ceiling: float = float(pp_cfg.get("low_contrast_dark_ratio_ceiling", 0.15))
        self._lc_autocontrast_cutoff: float = float(pp_cfg.get("low_contrast_autocontrast_cutoff", 1.0))
        self._exclude_merge_suspect_tokens: bool = bool(pp_cfg.get("exclude_merge_suspect_tokens", False))
        self._include_positional_ref_estimates: bool = bool(pp_cfg.get("include_positional_ref_estimates", False))
        self._corpus_key_by_token: dict[tuple[str, int], str] = self._build_corpus_key_index(tokens)
        self._catalog_exact_index: dict[str, dict[str, Any]] = self._build_catalog_exact_index()

        # Build vocabulary from the provided tokens.
        self.vocab: list[str] = sorted(
            {t.barthel_code for t in tokens}
        )
        self.barthel_to_id: dict[str, int] = {
            code: idx for idx, code in enumerate(self.vocab)
        }

        # Build a lookup index for barthel_ref/ reference images (fast O(1) lookup).
        # Files are named like "100_42_barthel_2_043.png"; we index by the
        # primary code (first token before "_barthel_"), leading zeros stripped.
        self._ref_index: dict[str, Path] = self._build_ref_index()

        # Check image coverage and filter out tokens with no resolvable image.
        # Zero-tensor substitution contaminates the embedding space and the
        # SupCon loss without providing any training signal.  Images may be
        # absent because barthel_corpus/ PDF extraction (OCR / alignment) was
        # incomplete, or because a code variant has no canonical barthel_ref/
        # entry.  Run scripts/extract_barthel_glyphs.py --source both to fix.
        candidate_tokens = tokens
        merge_suspect_excluded = 0
        if self._exclude_merge_suspect_tokens:
            filtered_tokens: list[GlyphToken] = []
            for token in tokens:
                corpus_key = self._corpus_key_by_token.get((token.tablet_id, token.position))
                exact = self._catalog_exact_index.get(corpus_key) if corpus_key else None
                if exact and exact.get("merge_suspect", False):
                    merge_suspect_excluded += 1
                    continue
                filtered_tokens.append(token)
            candidate_tokens = filtered_tokens

        _resolved = [(t, self._resolve_image_path(t)) for t in candidate_tokens]
        _missing = [(t, p) for t, p in _resolved if not p.exists()]
        if _missing:
            _missing_codes = sorted({t.barthel_code for t, _ in _missing})
            logger.warning(
                "GlyphImageDataset: %d / %d token(s) have no image file "
                "(%.1f%% of corpus) — excluded from training.  "
                "Run scripts/extract_barthel_glyphs.py --source both to generate them.  "
                "%d unique missing code(s): %s%s",
                len(_missing), len(tokens),
                100.0 * len(_missing) / len(tokens) if tokens else 0.0,
                len(_missing_codes),
                _missing_codes[:20],
                " …" if len(_missing_codes) > 20 else "",
            )

        if self._exclude_merge_suspect_tokens and merge_suspect_excluded:
            logger.warning(
                "GlyphImageDataset: excluded %d merge_suspect token(s) by config "
                "zone_a.preprocessing.exclude_merge_suspect_tokens=true",
                merge_suspect_excluded,
            )

        self.tokens = [t for t, p in _resolved if p.exists()]
        logger.info(
            "GlyphImageDataset: %d / %d tokens have images (%.0f%% coverage).  "
            "ref_index=%d entries.  training=%s",
            len(self.tokens), len(tokens),
            100.0 * len(self.tokens) / len(tokens) if tokens else 0.0,
            len(self._ref_index), training,
        )

    def __len__(self) -> int:
        return len(self.tokens)

    def __getitem__(self, index: int) -> dict[str, Any]:
        """Return a single sample.

        Parameters
        ----------
        index : int
            Index into ``self.tokens``.

        Returns
        -------
        dict with keys:

        ``"image"`` : torch.Tensor
            Float tensor of shape ``(C, H, W)`` where ``C``,
            ``H``, ``W`` equal ``cfg.glyph.image_channels`` and
            ``cfg.glyph.image_size`` respectively.  Values in
            ``[-1, 1]`` after normalisation.
        ``"token_id"`` : int
            Integer Barthel code ID from ``self.barthel_to_id``.
        ``"barthel_code"`` : str
            Raw Barthel code string.
        ``"tablet_id"`` : str
            Source tablet identifier.
        ``"position"`` : int
            Absolute glyph position within the tablet.
        ``"stratum"`` : str
            Temporal stratum label.
        """
        token: GlyphToken = self.tokens[index]
        image_path = self._resolve_image_path(token)
        image_tensor = self._load_image(image_path)

        return {
            "image": image_tensor,
            "token_id": self.barthel_to_id[token.barthel_code],
            "barthel_code": token.barthel_code,
            "tablet_id": token.tablet_id,
            "position": token.position,
            "stratum": token.stratum,
        }

    def _build_ref_index(self) -> dict[str, Path]:
        """Index reference PNG directories by normalised Barthel code.

        Scans two directories in priority order:

        1. ``barthel_ref/``  — reference sign images from Barthel's Formentafeln.
           Files may be OCR-confirmed (``{code}_barthel_{page}_{cell}.png``) or
           positional estimates (``{start}_{rank}_barthel_{page}_{cell}.png``).
           OCR-confirmed files take priority; positional-estimate filenames are
           recognised by a two-part numeric prefix (e.g. ``"100_0"``) and are
           only inserted when no confirmed entry already exists.

        2. ``barthel_corpus/<tablet>/``  — OCR-confirmed per-instance images from
           Barthel's Tafeln (corpus transcription plates).  Subdirectories named
           ``"?"`` (unknown tablet) are excluded.

        For each file the index stores two keys when they differ:

        * The full primary code with leading zeros stripped (e.g. ``"739b"``).
        * The base numeric code with trailing letter suffixes removed
          (e.g. ``"739"``), so that a corpus lookup for the base code can fall
          back to a variant image when no exact match exists.

        Only the first encountered file wins for each key, so ``barthel_ref/``
        (sorted alphabetically) takes precedence over ``barthel_corpus/`` for
        any key already present.
        """
        index: dict[str, Path] = {}

        def _insert(p: Path) -> None:
            stem = p.stem
            prefix = stem.split("_barthel_")[0] if "_barthel_" in stem else stem
            parts = prefix.split("_")
            primary = parts[0].lstrip("0") or "0"
            if primary not in index:
                index[primary] = p
            # Also register the base numeric code (strip trailing letter suffix)
            # so corpus codes without variant suffixes can find a variant image.
            base = re.sub(r"[A-Za-z]+$", "", primary)
            if base and base != primary and base not in index:
                index[base] = p

        ref_dir = self.glyphs_dir / "barthel_ref"
        skipped_positional_ref = 0
        if ref_dir.is_dir():
            for p in sorted(ref_dir.glob("*.png")):
                stem = p.stem
                prefix = stem.split("_barthel_")[0] if "_barthel_" in stem else stem
                # Filenames like "100_42_barthel_..." are positional estimates
                # and may contain non-glyph artifacts (e.g., plate labels).
                if (
                    not self._include_positional_ref_estimates
                    and re.fullmatch(r"\d+_\d+", prefix)
                ):
                    skipped_positional_ref += 1
                    continue
                _insert(p)

        if skipped_positional_ref:
            logger.info(
                "GlyphImageDataset: skipped %d positional barthel_ref estimate file(s); "
                "set zone_a.preprocessing.include_positional_ref_estimates=true to include.",
                skipped_positional_ref,
            )

        corpus_img_dir = self.glyphs_dir / "barthel_corpus"
        if corpus_img_dir.is_dir():
            for subdir in sorted(corpus_img_dir.iterdir()):
                if not subdir.is_dir() or subdir.name == "?":
                    continue
                for p in sorted(subdir.glob("*.png")):
                    _insert(p)

        # 3d_crops/tablet_X/side_Y/{line}_{pos}_{barthel_code}.png
        # Files are named like "L{line}_G{pos}_{code}.png"; the Barthel code
        # is the last underscore-separated field.  These are added last so
        # barthel_ref and barthel_corpus take priority for any code already seen.
        crops_dir = self.glyphs_dir / "3d_crops"
        if crops_dir.is_dir():
            for p in sorted(crops_dir.rglob("*.png")):
                stem = p.stem
                parts = stem.rsplit("_", 1)
                if len(parts) == 2:
                    code = parts[1].lstrip("0") or "0"
                    if code not in index:
                        index[code] = p
                    base = re.sub(r"[A-Za-z]+$", "", code)
                    if base and base != code and base not in index:
                        index[base] = p

        return index

    def _build_corpus_key_index(self, tokens: list[GlyphToken]) -> dict[tuple[str, int], str]:
        """Build {(tablet_id, position): corpus_key} using line-local sequence order."""
        groups: dict[tuple[str, int], list[GlyphToken]] = defaultdict(list)
        for token in tokens:
            side_ab = {"r": "a", "v": "b", "a": "a", "b": "b"}.get(str(token.side), "a")
            groups[(token.tablet_id, token.line_num, side_ab)].append(token)

        index: dict[tuple[str, int], str] = {}
        for _, grouped_tokens in groups.items():
            for seq_on_line, token in enumerate(sorted(grouped_tokens, key=lambda t: t.position), 1):
                index[(token.tablet_id, token.position)] = _token_corpus_key(token, seq_on_line)
        return index

    def _build_catalog_exact_index(self) -> dict[str, dict[str, Any]]:
        """Load exact corpus_key-to-image mappings from barthel_catalog.json."""
        catalog_path = self.glyphs_dir / "barthel_catalog.json"
        if not catalog_path.exists():
            return {}

        raw = json.loads(catalog_path.read_text(encoding="utf-8"))
        records = raw.get("records", raw) if isinstance(raw, dict) else raw
        index: dict[str, dict[str, Any]] = {}
        for rec in records:
            if not isinstance(rec, dict):
                continue
            corpus_key = rec.get("corpus_key")
            rel_path = rec.get("path")
            if not corpus_key or not rel_path:
                continue
            abs_path = self.glyphs_dir / rel_path
            if not abs_path.exists():
                continue
            index[str(corpus_key)] = {
                "path": abs_path,
                "merge_suspect": bool(rec.get("merge_suspect", False)),
                "barthel_code": rec.get("barthel_code"),
            }
        return index

    def _resolve_image_path(self, token: GlyphToken) -> Path:
        """Expand the filename pattern for a given glyph token.

        Parameters
        ----------
        token : GlyphToken
            Token whose image path should be resolved.

        Returns
        -------
        Path
            Absolute path to the expected PNG file.
        """
        filename = self._filename_pattern.format(
            tablet_id=token.tablet_id,
            position=token.position,
            barthel_code=token.barthel_code,
        )
        path = self.glyphs_dir / filename
        if path.exists():
            return path

        corpus_key = self._corpus_key_by_token.get((token.tablet_id, token.position))
        if corpus_key:
            exact = self._catalog_exact_index.get(corpus_key)
            if exact and not exact.get("merge_suspect", False):
                return exact["path"]

        # Fallback: look up a reference image in barthel_ref/ using the
        # pre-built index.  Corpus codes have leading zeros ("004") and
        # optional uncertainty/variant suffixes ("600V", "073f", "382?");
        # the index keys are leading-zero-stripped primary codes ("4",
        # "600", "73", "382").
        if self._ref_index:
            # Strip uncertainty markers and parentheses
            clean = re.sub(r'[!?()\s]', '', str(token.barthel_code))
            # Try: strip leading zeros
            key = clean.lstrip('0') or '0'
            if key in self._ref_index:
                return self._ref_index[key]
            # Try: also strip trailing letter variant suffix (V, f, s, …)
            trimmed = re.sub(r'[A-Za-z]+$', '', key)
            if trimmed and trimmed in self._ref_index:
                return self._ref_index[trimmed]
            # Try: compound code (e.g. "670.076" or "10-20") — use first component
            for sep in ('.', '-'):
                if sep in key:
                    first = key.split(sep)[0].lstrip('0') or '0'
                    if first in self._ref_index:
                        return self._ref_index[first]

        return path  # return original missing path to trigger warning

    def _load_image(self, path: Path) -> torch.Tensor:
        """Load a glyph PNG and apply the configured transforms.

        Returns a zero tensor (with a warning) if the file is absent,
        so partial image directories don't crash training.

        Parameters
        ----------
        path : Path
            Absolute path to the PNG file.

        Returns
        -------
        torch.Tensor
            Transformed image tensor of shape ``(C, H, W)``.
        """
        if not path.exists():
            logger.debug(
                "Glyph image not found (returning zeros): %s", path
            )
            return torch.zeros(
                self._channels, self._image_size, self._image_size
            )
        img = Image.open(path)
        if self._enhance_low_contrast:
            gray = img.convert("L")
            if _looks_like_low_contrast_relief(
                gray,
                is_3d_crop=("3d_crops" in path.parts),
                std_threshold=self._lc_std_threshold,
                dynamic_range_threshold=self._lc_dynamic_range_threshold,
                mean_floor=self._lc_mean_floor,
                dark_ratio_ceiling=self._lc_dark_ratio_ceiling,
            ):
                img = _enhance_low_contrast_relief(
                    gray,
                    autocontrast_cutoff=self._lc_autocontrast_cutoff,
                )
        return self._transform(img)


# ---------------------------------------------------------------------------
# GlyphSequenceDataset
# ---------------------------------------------------------------------------


class GlyphSequenceDataset(Dataset):
    """Dataset of sliding context-window slices for next-glyph prediction.

    Each sample is a context window of ``context_window`` consecutive
    glyph token IDs from a tablet, paired with the immediately following
    token ID as the prediction target.

    Windows **do not cross tablet boundaries** — the sequence model
    learns tablet-internal sequential patterns only.

    Parameters
    ----------
    tablets : list[TabletRecord]
        List of tablet records (train split only; pass val tablets
        separately for evaluation).
    barthel_to_id : dict[str, int]
        Mapping from Barthel code string to integer token ID, as
        produced by :attr:`GlyphImageDataset.barthel_to_id`.  Must be
        built on the *full* corpus vocabulary before splitting into
        train/val, so that unseen IDs do not appear at inference.
    cfg : DictConfig
        Root Hydra config.  Uses ``cfg.zone_a.sequence_model.context_window``.

    Notes
    -----
    Tablets shorter than ``context_window + 1`` tokens produce no
    samples and are silently skipped.
    """

    def __init__(
        self,
        tablets: list[TabletRecord],
        barthel_to_id: dict[str, int],
        cfg: DictConfig,
    ) -> None:
        super().__init__()
        self._barthel_to_id = barthel_to_id
        self._context_window: int = int(cfg.zone_a.sequence_model.context_window)

        # Build the flat list of (context_ids, target_id) windows.
        self._samples: list[tuple[list[int], int]] = []
        skipped_tablets: int = 0

        for tablet in tablets:
            try:
                token_ids = [
                    barthel_to_id[t.barthel_code] for t in tablet.tokens
                ]
            except KeyError as exc:
                raise ValueError(
                    f"Tablet {tablet.tablet_id!r}: Barthel code {exc} not found in "
                    "barthel_to_id. Build barthel_to_id from the full corpus vocab "
                    "before splitting into train/val."
                ) from exc
            n = len(token_ids)
            min_len = self._context_window + 1
            if n < min_len:
                skipped_tablets += 1
                continue
            for start in range(n - self._context_window):
                context = token_ids[start : start + self._context_window]
                target = token_ids[start + self._context_window]
                self._samples.append((context, target))

        if skipped_tablets:
            logger.debug(
                "GlyphSequenceDataset: %d tablet(s) skipped (shorter than "
                "context_window + 1 = %d).",
                skipped_tablets,
                self._context_window + 1,
            )
        logger.debug(
            "GlyphSequenceDataset: %d samples from %d tablet(s), "
            "context_window=%d.",
            len(self._samples),
            len(tablets) - skipped_tablets,
            self._context_window,
        )

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        """Return a single context-window sample.

        Parameters
        ----------
        index : int
            Sample index.

        Returns
        -------
        dict with keys:

        ``"context"`` : torch.Tensor[long], shape ``(context_window,)``
            Integer token IDs of the context glyphs.
        ``"target"`` : torch.Tensor[long], shape ``()``
            Integer token ID of the glyph immediately following the
            context window (the prediction target).
        """
        context, target = self._samples[index]
        return {
            "context": torch.tensor(context, dtype=torch.long),
            "target": torch.tensor(target, dtype=torch.long),
        }


# ---------------------------------------------------------------------------
# SiamesePairDataset
# ---------------------------------------------------------------------------


class SiamesePairDataset(Dataset):
    """Dataset of (anchor, pair, label) image triples for contrastive learning.

    Samples pairs from :class:`GlyphImageDataset` using the allograph
    catalog to define positive (same canonical sign) and negative
    (different canonical sign) relationships.

    Same-sign pairs (label ``1``) are drawn from within the same sign
    group.  Different-sign pairs (label ``0``) are drawn across groups.

    When ``hard_negative_mining`` is enabled (``cfg.zone_a.siamese.
    hard_negative_mining == True``), negative examples are sampled
    preferentially from sign groups whose embedding centroid is within
    ``margin * hard_negative_margin_factor`` of the anchor — i.e.
    semi-hard negatives.  Because centroids are not available at dataset
    construction time, this class exposes an
    :meth:`update_hard_negatives` method that the training loop calls
    after each epoch with the current embedding matrix.

    Parameters
    ----------
    image_dataset : GlyphImageDataset
        Base image dataset providing ``__getitem__`` access and the
        token list.
    allograph_catalog : dict[str, str]
        Output of :func:`load_allograph_catalog`.  Maps variant Barthel
        code → canonical sign ID.
    cfg : DictConfig
        Root Hydra config.  Uses ``cfg.zone_a.siamese``.
    seed : int
        Random seed for reproducible pair sampling.

    Notes
    -----
    Pairs are sampled *lazily* on each ``__getitem__`` call (not
    pre-generated at construction), so ``pairs_per_epoch`` controls the
    effective epoch length via ``__len__``.

    Glyphs whose Barthel code is absent from the allograph catalog are
    assigned a fallback canonical ID equal to their own Barthel code,
    with a one-time WARNING.
    """

    def __init__(
        self,
        image_dataset: GlyphImageDataset,
        allograph_catalog: dict[str, str],
        cfg: DictConfig,
        seed: int,
    ) -> None:
        super().__init__()
        self._image_dataset = image_dataset
        self._cfg_siamese = cfg.zone_a.siamese
        self._pairs_per_epoch: int = int(self._cfg_siamese.pairs_per_epoch)
        self._same_sign_ratio: float = float(self._cfg_siamese.same_sign_ratio)
        self._hard_negative_mining: bool = bool(
            self._cfg_siamese.hard_negative_mining
        )
        self._margin: float = float(self._cfg_siamese.margin)
        self._hard_neg_factor: float = float(
            self._cfg_siamese.hard_negative_margin_factor
        )
        self._min_instances_for_pairing: int = int(
            getattr(self._cfg_siamese, "min_instances_for_pairing", 2)
        )

        self._rng = random.Random(seed)

        # Resolve every token's Barthel code to a canonical sign ID.
        # Tokens not in the catalog keep their own code as fallback.
        warned_missing: set[str] = set()
        self._token_canonical: list[str] = []
        for token in image_dataset.tokens:
            code = token.barthel_code
            if code not in allograph_catalog:
                if code not in warned_missing:
                    logger.warning(
                        "Barthel code '%s' not in allograph catalog; "
                        "treating as its own canonical sign.",
                        code,
                    )
                    warned_missing.add(code)
                self._token_canonical.append(code)
            else:
                self._token_canonical.append(allograph_catalog[code])

        # Build index: canonical_sign_id → list of dataset indices.
        self._sign_to_indices: dict[str, list[int]] = {}
        for idx, canonical in enumerate(self._token_canonical):
            self._sign_to_indices.setdefault(canonical, []).append(idx)

        # All canonical sign IDs (used for negative sampling).
        self._all_sign_ids: list[str] = sorted(self._sign_to_indices.keys())

        # Sign IDs eligible for positive-pair sampling: must have at least
        # min_instances_for_pairing distinct instances so a genuine pair
        # (two different indices) can be drawn.
        self._sign_ids: list[str] = [
            s for s in self._all_sign_ids
            if len(self._sign_to_indices[s]) >= self._min_instances_for_pairing
        ]
        n_sparse = len(self._all_sign_ids) - len(self._sign_ids)
        if n_sparse:
            logger.debug(
                "SiamesePairDataset: %d sign(s) excluded from positive "
                "sampling (fewer than %d instances).",
                n_sparse,
                self._min_instances_for_pairing,
            )

        # Optional hard-negative embedding distance matrix (set externally).
        # Shape: (n_tokens, embedding_dim) numpy array or None.
        self._embeddings: np.ndarray | None = None

        logger.debug(
            "SiamesePairDataset: %d tokens, %d canonical signs "
            "(%d eligible for positive pairs), "
            "pairs_per_epoch=%d, same_sign_ratio=%.2f.",
            len(image_dataset),
            len(self._all_sign_ids),
            len(self._sign_ids),
            self._pairs_per_epoch,
            self._same_sign_ratio,
        )

    def __len__(self) -> int:
        return self._pairs_per_epoch

    def update_hard_negatives(self, embeddings: np.ndarray) -> None:
        """Provide updated glyph embeddings for semi-hard negative mining.

        Call this at the end of each training epoch with the current
        encoder output for all tokens in ``self._image_dataset``.

        Parameters
        ----------
        embeddings : numpy.ndarray
            Array of shape ``(N, D)`` where ``N`` equals
            ``len(self._image_dataset)`` and ``D`` is the embedding
            dimension.  Row ``i`` corresponds to
            ``self._image_dataset.tokens[i]``.
        """
        if embeddings.shape[0] != len(self._image_dataset):
            raise ValueError(
                f"update_hard_negatives: expected embeddings with "
                f"{len(self._image_dataset)} rows, got {embeddings.shape[0]}."
            )
        self._embeddings = embeddings
        logger.debug(
            "SiamesePairDataset: hard-negative embeddings updated, "
            "shape %s.", embeddings.shape
        )

    def __getitem__(self, index: int) -> dict[str, Any]:
        """Sample a single (anchor, pair, label) triple.

        Parameters
        ----------
        index : int
            Unused beyond determining same/different split; each call
            independently samples a random pair.  Provided to satisfy
            the ``Dataset`` interface.

        Returns
        -------
        dict with keys:

        ``"anchor"`` : torch.Tensor
            Image tensor for the anchor glyph, shape ``(C, H, W)``.
        ``"pair"`` : torch.Tensor
            Image tensor for the paired glyph, shape ``(C, H, W)``.
        ``"label"`` : torch.Tensor[float], shape ``()``
            ``1.0`` for a same-sign pair; ``0.0`` for a different-sign
            pair.  Matches the contrastive loss sign convention in
            ``cfg.zone_a.siamese``.
        ``"anchor_idx"`` : int
            Dataset index of the anchor; used by the training loop when
            calling :meth:`update_hard_negatives`.
        ``"pair_idx"`` : int
            Dataset index of the paired sample.
        """
        is_positive: bool = self._rng.random() < self._same_sign_ratio

        if is_positive:
            anchor_idx, pair_idx = self._sample_positive_pair()
        else:
            anchor_idx, pair_idx = self._sample_negative_pair()

        anchor_sample = self._image_dataset[anchor_idx]
        pair_sample = self._image_dataset[pair_idx]
        label = 1.0 if is_positive else 0.0

        return {
            "anchor": anchor_sample["image"],
            "pair": pair_sample["image"],
            "label": torch.tensor(label, dtype=torch.float32),
            "anchor_idx": anchor_idx,
            "pair_idx": pair_idx,
        }

    # ------------------------------------------------------------------
    # Private pair-sampling helpers
    # ------------------------------------------------------------------

    def _sample_positive_pair(self) -> tuple[int, int]:
        """Sample two distinct tokens from the same canonical sign group.

        Returns
        -------
        tuple[int, int]
            ``(anchor_idx, pair_idx)`` into ``self._image_dataset``.
            If the chosen sign group has only one token, both indices
            are the same token (the network sees an identical pair).
        """
        # Pick a sign group that has at least one token (all do by construction).
        sign_id = self._rng.choice(self._sign_ids)
        indices = self._sign_to_indices[sign_id]
        if len(indices) == 1:
            return indices[0], indices[0]
        anchor_idx, pair_idx = self._rng.sample(indices, k=2)
        return anchor_idx, pair_idx

    def _sample_negative_pair(self) -> tuple[int, int]:
        """Sample two tokens from different canonical sign groups.

        When ``hard_negative_mining`` is enabled and embeddings are
        available, selects the negative sign group whose centroid is
        within ``margin * hard_negative_margin_factor`` of the anchor
        embedding (semi-hard negative).  Falls back to uniform random
        selection if no such group exists or embeddings are not yet set.

        Returns
        -------
        tuple[int, int]
            ``(anchor_idx, pair_idx)`` into ``self._image_dataset``.
        """
        anchor_sign = self._rng.choice(self._all_sign_ids)
        anchor_indices = self._sign_to_indices[anchor_sign]
        anchor_idx = self._rng.choice(anchor_indices)

        # Build candidate negative sign IDs (all signs except the anchor's).
        neg_candidates = [s for s in self._all_sign_ids if s != anchor_sign]

        if (
            self._hard_negative_mining
            and self._embeddings is not None
            and len(neg_candidates) > 0
        ):
            pair_idx = self._sample_semi_hard_negative(
                anchor_idx, anchor_sign, neg_candidates
            )
        else:
            neg_sign = self._rng.choice(neg_candidates)
            pair_idx = self._rng.choice(self._sign_to_indices[neg_sign])

        return anchor_idx, pair_idx

    def _sample_semi_hard_negative(
        self,
        anchor_idx: int,
        anchor_sign: str,
        neg_candidates: list[str],
    ) -> int:
        """Select a semi-hard negative index.

        A semi-hard negative is a sample from a different sign group
        whose L2 distance to the anchor is less than
        ``margin * hard_negative_margin_factor``.

        Parameters
        ----------
        anchor_idx : int
            Index of the anchor token in ``self._image_dataset``.
        anchor_sign : str
            Canonical sign ID of the anchor; excluded from candidates.
        neg_candidates : list[str]
            Sign IDs eligible as negative sources.

        Returns
        -------
        int
            Index of the selected negative token.  Falls back to a
            random negative if no semi-hard candidate exists.
        """
        assert self._embeddings is not None  # caller guarantees this
        threshold = self._margin * self._hard_neg_factor
        anchor_emb: np.ndarray = self._embeddings[anchor_idx]

        neg_indices = [
            idx
            for sign_id in neg_candidates
            for idx in self._sign_to_indices[sign_id]
        ]
        dists = np.linalg.norm(
            self._embeddings[neg_indices] - anchor_emb, axis=1
        )
        semi_hard_indices = [
            neg_indices[i] for i in np.where(dists < threshold)[0]
        ]

        if semi_hard_indices:
            return self._rng.choice(semi_hard_indices)

        # Fallback: uniform random negative.
        neg_sign = self._rng.choice(neg_candidates)
        return self._rng.choice(self._sign_to_indices[neg_sign])
