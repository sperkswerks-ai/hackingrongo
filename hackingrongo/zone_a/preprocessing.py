"""
hackingrongo.zone_a.preprocessing
==================================

Preprocessing pipeline applied to every glyph image before it enters any
Zone A network.  Three stages run in sequence when enabled via config:

1. **Boustrophedon normalisation** — Rongorongo is written in reverse
   boustrophedon: even-indexed lines (0-based) run left-to-right and
   odd-indexed lines run right-to-left with each glyph rotated 180°.
   Without normalisation an autoencoder would learn two separate embeddings
   for the same sign depending on which line it appears in.  This stage
   rotates every glyph on an odd-indexed line by 180° so all tokens enter
   the network in canonical orientation.

2. **Stroke extraction** — Converts the grayscale glyph image to a binary
   stroke map using Canny edge detection followed by morphological
   skeletonisation.  The map encodes where carved strokes lie rather than
   raw pixel intensity, making embeddings robust to lighting variation across
   photographs.  The result is a single-channel float32 image in [0, 1].

3. **Scale normalisation** — Crops the tight bounding box of the stroke
   content and resizes to the canonical ``image_size × image_size`` square
   specified in the config, padding symmetrically to preserve aspect ratio.

Public API
----------
``GlyphPreprocessor``
    Callable transform class compatible with ``torchvision.transforms.Compose``.
    Reads all flags from the Hydra config; individual stages can be toggled
    independently.

``build_preprocessor``
    Factory that constructs a ``GlyphPreprocessor`` from a Hydra ``DictConfig``.
"""

from __future__ import annotations

from pathlib import Path

import logging
from typing import Union

try:
    import cv2 as cv2
    _CV2_AVAILABLE = True
except ImportError:  # pragma: no cover
    cv2 = None  # type: ignore[assignment]
    _CV2_AVAILABLE = False
import numpy as np
import torch
from omegaconf import DictConfig

logger = logging.getLogger(__name__)

# Type alias for the image type accepted and returned by each stage.
# Internally we operate on float32 ndarray (H, W) in [0, 1]; the final
# __call__ returns a (1, H, W) float32 Tensor.
_Array = np.ndarray


# ---------------------------------------------------------------------------
# Individual preprocessing stages
# ---------------------------------------------------------------------------

def normalize_boustrophedon(image: _Array, line_index: int) -> _Array:
    """Rotate ``image`` by 180° when ``line_index`` is odd.

    Rongorongo line indices are 0-based.  Odd-indexed lines are written
    right-to-left with glyphs inverted; rotating 180° brings them into the
    same canonical orientation as even-indexed lines.

    Parameters
    ----------
    image:
        Float32 ndarray of shape (H, W), values in [0, 1].
    line_index:
        0-based index of the line on the tablet from which this glyph was
        taken.  Callers must supply this from the ``GlyphToken.line`` field.

    Returns
    -------
    Float32 ndarray of shape (H, W).  Unchanged when ``line_index`` is even.
    """
    if line_index % 2 == 1:
        return np.rot90(image, k=2).copy()
    return image


def extract_stroke_map(image: _Array, *, canny_low: int = 30, canny_high: int = 100) -> _Array:
    """Convert a grayscale glyph image to a binary stroke skeleton.

    Pipeline:
        1. Convert to uint8 if necessary.
        2. Canny edge detection to locate stroke boundaries.
        3. Morphological closing to bridge small gaps in the edge response.
        4. Skeletonisation to reduce strokes to single-pixel-wide centre
           lines, removing lighting-induced thickness variation.
        5. Normalise back to float32 in [0, 1].

    Parameters
    ----------
    image:
        Float32 ndarray (H, W) in [0, 1].
    canny_low:
        Lower hysteresis threshold for Canny.  Values below this are
        rejected unless connected to a strong edge.
    canny_high:
        Upper hysteresis threshold for Canny.  Values above this are
        accepted as definite edges.

    Returns
    -------
    Float32 ndarray (H, W) in {0.0, 1.0} — stroke pixels are 1.0.
    """
    if not _CV2_AVAILABLE:
        raise RuntimeError(
            "extract_stroke_map requires opencv-python; "
            "install it with: pip install opencv-python-headless"
        )
    uint8 = (image * 255).clip(0, 255).astype(np.uint8)
    edges = cv2.Canny(uint8, canny_low, canny_high)

    # Close small gaps so strokes form connected regions before skeletonising.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    # skimage.skeletonize expects a boolean array.
    try:
        from skimage.morphology import skeletonize
    except Exception as exc:
        raise RuntimeError(
            "extract_stroke_map requires scikit-image compatible with the installed NumPy. "
            "Try: pip install -U scikit-image"
        ) from exc
    skeleton = skeletonize(closed > 0)
    return skeleton.astype(np.float32)


def normalize_scale(image: _Array, target_size: int) -> _Array:
    """Crop to the tight bounding box of stroke content and resize.

    If the image has no non-zero pixels (blank stroke map after extraction),
    the full image is used as the bounding box to avoid a degenerate crop.

    Parameters
    ----------
    image:
        Float32 ndarray (H, W) in [0, 1].
    target_size:
        Edge length in pixels of the square output.

    Returns
    -------
    Float32 ndarray (target_size, target_size) in [0, 1].
    """
    rows = np.any(image > 0, axis=1)
    cols = np.any(image > 0, axis=0)

    if rows.any() and cols.any():
        r_min, r_max = np.where(rows)[0][[0, -1]]
        c_min, c_max = np.where(cols)[0][[0, -1]]
        cropped = image[r_min : r_max + 1, c_min : c_max + 1]
    else:
        cropped = image  # blank — fall back to full image

    h, w = cropped.shape
    if h == 0 or w == 0:
        cropped = image

    # Embed the crop in a square canvas and resize, preserving aspect ratio.
    scale = target_size / max(cropped.shape)
    new_h = max(1, int(round(cropped.shape[0] * scale)))
    new_w = max(1, int(round(cropped.shape[1] * scale)))
    if not _CV2_AVAILABLE:
        raise RuntimeError(
            "normalize_scale requires opencv-python; "
            "install it with: pip install opencv-python-headless"
        )
    resized = cv2.resize(cropped, (new_w, new_h), interpolation=cv2.INTER_AREA)

    canvas = np.zeros((target_size, target_size), dtype=np.float32)
    pad_y = (target_size - new_h) // 2
    pad_x = (target_size - new_w) // 2
    canvas[pad_y : pad_y + new_h, pad_x : pad_x + new_w] = resized
    return canvas


# ---------------------------------------------------------------------------
# Composed preprocessor
# ---------------------------------------------------------------------------

class GlyphPreprocessor:
    """Stateless callable that applies the full preprocessing pipeline.

    Compatible with ``torchvision.transforms.Compose``.  Because
    boustrophedon normalisation requires per-glyph line metadata, the
    callable signature is:

        preprocessor(image, line_index) -> torch.Tensor  # shape (1, H, W)

    where ``line_index`` is the 0-based line number on the tablet.

    Parameters
    ----------
    boustrophedon_normalize:
        Rotate odd-line glyphs 180°.
    stroke_extraction:
        Replace pixel image with binary stroke skeleton.
    normalize_scale:
        Crop to content bounding box and resize to ``target_size``.
    target_size:
        Square output edge length in pixels.  Read from
        ``cfg.glyph.image_size``.
    canny_low / canny_high:
        Hysteresis thresholds forwarded to :func:`extract_stroke_map`.
    """

    def __init__(
        self,
        *,
        boustrophedon_normalize: bool,
        stroke_extraction: bool,
        normalize_scale: bool,
        target_size: int,
        canny_low: int = 30,
        canny_high: int = 100,
    ) -> None:
        self.boustrophedon_normalize = boustrophedon_normalize
        self.stroke_extraction = stroke_extraction
        self.normalize_scale_flag = normalize_scale
        self.target_size = target_size
        self.canny_low = canny_low
        self.canny_high = canny_high

    def __call__(
        self,
        image: Union[_Array, torch.Tensor],
        line_index: int,
    ) -> torch.Tensor:
        """Apply enabled preprocessing stages and return a (1, H, W) Tensor.

        Parameters
        ----------
        image:
            Float32 ndarray (H, W) in [0, 1], **or** a (1, H, W) / (H, W)
            float32 Tensor.  If a Tensor is passed it is converted to ndarray
            for processing then back to Tensor on return.
        line_index:
            0-based line index on the tablet for this glyph.
        """
        if isinstance(image, torch.Tensor):
            arr = image.squeeze(0).numpy()  # remove channel dim only; preserve spatial dims
        else:
            arr = np.asarray(image, dtype=np.float32)

        if arr.ndim == 3:
            # (C, H, W) → (H, W) grayscale; take first channel
            arr = arr[0]

        if self.boustrophedon_normalize:
            arr = normalize_boustrophedon(arr, line_index)

        if self.stroke_extraction:
            arr = extract_stroke_map(arr, canny_low=self.canny_low, canny_high=self.canny_high)

        if self.normalize_scale_flag:
            arr = normalize_scale(arr, self.target_size)
        elif arr.shape != (self.target_size, self.target_size):
            if not _CV2_AVAILABLE:
                raise RuntimeError(
                    "GlyphPreprocessor resize requires opencv-python; "
                    "install it with: pip install opencv-python-headless"
                )
            arr = cv2.resize(arr, (self.target_size, self.target_size), interpolation=cv2.INTER_AREA)

        return torch.from_numpy(np.ascontiguousarray(arr)).unsqueeze(0)  # (1, H, W)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_preprocessor(cfg: DictConfig) -> GlyphPreprocessor:
    """Construct a :class:`GlyphPreprocessor` from a Hydra ``DictConfig``.

    Reads from:
        ``cfg.zone_a.preprocessing.*``
        ``cfg.glyph.image_size``

    Parameters
    ----------
    cfg:
        Root Hydra config (the full ``DictConfig``, not a sub-node).

    Returns
    -------
    :class:`GlyphPreprocessor`
    """
    pp = cfg.zone_a.preprocessing
    return GlyphPreprocessor(
        boustrophedon_normalize=bool(pp.boustrophedon_normalize),
        stroke_extraction=bool(pp.stroke_extraction),
        normalize_scale=bool(pp.normalize_scale),
        target_size=int(cfg.glyph.image_size),
        canny_low=int(pp.get("canny_low", 30)),
        canny_high=int(pp.get("canny_high", 100)),
    )


# ---------------------------------------------------------------------------
# SVG → image pipeline (boustrophedon-aware)
# ---------------------------------------------------------------------------

class GlyphImagePipeline:
    """Load a glyph SVG, apply boustrophedon normalisation, and preprocess.

    This class bridges the kohaumotu.org SVG scraper output (stored under
    ``data/glyphs/svg/``) and the :class:`GlyphPreprocessor`.  The pipeline:

    1. Looks up the SVG file for a glyph record from the scraper catalog.
    2. Rasterises the SVG to a grayscale ``float32`` numpy array.
       Rasterisation requires an optional backend — ``cairosvg`` is preferred;
       falling back to ``None`` (returns ``None``) if unavailable.
    3. Applies :func:`normalize_boustrophedon` when the corpus record has
       ``inverted=True`` (i.e. the glyph appears on a right-to-left line).
    4. Runs the remaining preprocessing stages (stroke extraction, scale
       normalisation) via the composed :class:`GlyphPreprocessor`.

    Parameters
    ----------
    preprocessor:
        A configured :class:`GlyphPreprocessor`.
    glyphs_svg_dir:
        Root directory written by ``scripts/scrape_glyphs.py``
        (contains ``catalog.json`` and per-tablet SVG sub-directories).
    """

    def __init__(
        self,
        preprocessor: GlyphPreprocessor,
        glyphs_svg_dir: Path,
    ) -> None:
        self.preprocessor = preprocessor
        self.glyphs_svg_dir = Path(glyphs_svg_dir)
        self._catalog: dict[str, dict] | None = None

    # ------------------------------------------------------------------
    # Catalog
    # ------------------------------------------------------------------

    def _ensure_catalog(self) -> dict[str, dict]:
        """Lazily load scraper catalog; keyed by glyph_id."""
        if self._catalog is not None:
            return self._catalog
        catalog_path = self.glyphs_svg_dir / "catalog.json"
        if not catalog_path.exists():
            logger.warning(
                "SVG catalog not found at %s. Run scripts/scrape_glyphs.py first.",
                catalog_path,
            )
            self._catalog = {}
            return self._catalog

        import json as _json
        raw = _json.loads(catalog_path.read_text(encoding="utf-8"))
        self._catalog = {r["glyph_id"]: r for r in raw.get("records", [])}
        logger.debug("SVG catalog loaded: %d records", len(self._catalog))
        return self._catalog

    def svg_path_for(self, glyph_record: dict) -> Path | None:
        """Return the SVG file path for a corpus glyph record, or None.

        Looks up by constructing the expected kohaumotu path ID from the
        corpus record fields: ``{tablet}{side}{line:02d}-{glyph_num:03d}-b``.
        """
        tablet = glyph_record.get("tablet_id") or glyph_record.get("tablet", "?")
        side = str(glyph_record.get("side", "a"))
        try:
            line = f"{int(glyph_record.get('line', 0)):02d}"
        except (ValueError, TypeError):
            line = str(glyph_record.get("line", "00"))
        try:
            gnum = f"{int(glyph_record.get('glyph_num', 0)):03d}"
        except (ValueError, TypeError):
            gnum = str(glyph_record.get("glyph_num", "000"))

        svg_file = self.glyphs_svg_dir / tablet / f"{side}{line}-{gnum}.svg"
        return svg_file if svg_file.exists() else None

    # ------------------------------------------------------------------
    # Rasterisation
    # ------------------------------------------------------------------

    @staticmethod
    def rasterize_svg(svg_path: Path, target_size: int = 128) -> "_Array | None":
        """Rasterise an SVG file to a float32 grayscale array.

        Uses ``cairosvg`` if available, otherwise returns ``None`` and logs
        a warning.  Install with ``pip install cairosvg``.

        Parameters
        ----------
        svg_path:
            Path to a standalone SVG file (as written by scrape_glyphs.py).
        target_size:
            Output edge length in pixels before later preprocessing stages.

        Returns
        -------
        Float32 ndarray (H, W) in [0, 1], or ``None`` if rasterisation
        is unavailable.
        """
        try:
            import cairosvg  # optional dependency
        except ImportError:
            logger.warning(
                "cairosvg not installed — cannot rasterise SVG. "
                "Install with: pip install cairosvg"
            )
            return None

        png_bytes = cairosvg.svg2png(
            url=str(svg_path),
            output_width=target_size,
            output_height=target_size,
        )
        # Decode PNG bytes to numpy
        import io
        from PIL import Image as _Image  # type: ignore[import]
        img = _Image.open(io.BytesIO(png_bytes)).convert("L")
        arr = np.asarray(img, dtype=np.float32) / 255.0
        return arr

    # ------------------------------------------------------------------
    # Main pipeline call
    # ------------------------------------------------------------------

    def __call__(
        self,
        glyph_record: dict,
    ) -> "torch.Tensor | None":
        """Return a preprocessed (1, H, W) Tensor for a corpus glyph record.

        The boustrophedon normalisation stage uses the ``inverted`` field from
        the corpus record directly — no line-index arithmetic needed.

        Parameters
        ----------
        glyph_record:
            A glyph dict from an enriched corpus JSON file.  Must have at
            minimum: ``tablet_id``/``tablet``, ``side``, ``line``, ``glyph_num``,
            ``inverted``.

        Returns
        -------
        ``torch.Tensor`` of shape (1, H, W) with float32 values in [0, 1],
        or ``None`` if the SVG file was not found or rasterisation failed.
        """
        svg_path = self.svg_path_for(glyph_record)
        if svg_path is None:
            logger.debug("SVG not found for glyph record: %s", glyph_record)
            return None

        arr = self.rasterize_svg(svg_path, target_size=self.preprocessor.target_size * 2)
        if arr is None:
            return None

        # Boustrophedon normalisation: the corpus 'inverted' field is the
        # canonical signal — line_index oddness was already resolved at
        # corpus-build time.
        inverted = bool(glyph_record.get("inverted", False))
        if inverted:
            arr = normalize_boustrophedon(arr, line_index=1)  # odd → rotate 180°

        # Apply remaining stages (stroke extraction, scale normalisation)
        # by calling the preprocessor with line_index=0 (boustrophedon
        # already applied above — do not double-rotate).
        return self.preprocessor(arr, line_index=0)


def build_image_pipeline(cfg: DictConfig, project_root: Path) -> GlyphImagePipeline:
    """Construct a :class:`GlyphImagePipeline` from a Hydra ``DictConfig``.

    Parameters
    ----------
    cfg:
        Root Hydra config.
    project_root:
        Absolute path to the project root (for resolving relative paths).

    Returns
    -------
    :class:`GlyphImagePipeline`
    """
    preprocessor = build_preprocessor(cfg)
    glyphs_svg_dir = project_root / cfg.paths.glyphs_dir / "svg"
    return GlyphImagePipeline(preprocessor=preprocessor, glyphs_svg_dir=glyphs_svg_dir)

