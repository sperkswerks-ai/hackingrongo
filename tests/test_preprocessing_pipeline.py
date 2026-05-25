"""
tests.test_preprocessing_pipeline
===================================

Tests for hackingrongo.zone_a.preprocessing focusing on:
  - Module importability without OpenCV installed
  - normalize_boustrophedon (no OpenCV dependency)
  - GlyphImagePipeline: svg_path_for, rasterize_svg (mocked), __call__ (mocked)
  - GlyphPreprocessor when normalize_scale and stroke_extraction are disabled
    (so no cv2 calls are triggered)

OpenCV-dependent tests are skipped if cv2 is not importable.
"""

from __future__ import annotations

import io
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

# PyTorch compiled against NumPy 1.x fails at torch.from_numpy in NumPy 2.x envs.
# Skip tensor-producing tests rather than let them error with a misleading message.
def _torch_numpy_works() -> bool:
    try:
        torch.from_numpy(np.zeros((2, 2), dtype=np.float32))
        return True
    except RuntimeError:
        return False


_SKIP_TORCH = pytest.mark.skipif(
    not _torch_numpy_works(),
    reason="torch.from_numpy incompatible with installed NumPy (ABI mismatch); upgrade numpy<2 or torch>=2.3)",
)

# ---------------------------------------------------------------------------
# Importability — cv2 must NOT be required at module import time
# ---------------------------------------------------------------------------

def test_preprocessing_importable_without_cv2():
    """preprocessing.py must be importable even if cv2 is absent."""
    # If the import below raises, the test fails — that's the assertion.
    from hackingrongo.zone_a.preprocessing import (  # noqa: F401
        GlyphImagePipeline,
        GlyphPreprocessor,
        normalize_boustrophedon,
    )


# ---------------------------------------------------------------------------
# normalize_boustrophedon (pure numpy, no cv2)
# ---------------------------------------------------------------------------

from hackingrongo.zone_a.preprocessing import normalize_boustrophedon


class TestNormalizeBoustrophedon:
    def _asymmetric_array(self) -> np.ndarray:
        """4×4 float32 array with a bright pixel only in top-left corner."""
        arr = np.zeros((4, 4), dtype=np.float32)
        arr[0, 0] = 1.0
        return arr

    def test_even_line_unchanged(self):
        arr = self._asymmetric_array()
        result = normalize_boustrophedon(arr, line_index=0)
        np.testing.assert_array_equal(result, arr)

    def test_odd_line_rotated_180(self):
        arr = self._asymmetric_array()
        result = normalize_boustrophedon(arr, line_index=1)
        # After 180° rotation, the bright pixel should be at bottom-right
        assert result[3, 3] == 1.0
        assert result[0, 0] == 0.0

    def test_even_2_unchanged(self):
        arr = self._asymmetric_array()
        result = normalize_boustrophedon(arr, line_index=2)
        np.testing.assert_array_equal(result, arr)

    def test_shape_preserved(self):
        arr = np.random.rand(16, 16).astype(np.float32)
        result = normalize_boustrophedon(arr, line_index=3)
        assert result.shape == arr.shape

    def test_double_rotation_is_identity(self):
        arr = np.random.rand(8, 8).astype(np.float32)
        once = normalize_boustrophedon(arr, line_index=1)
        twice = normalize_boustrophedon(once, line_index=1)
        np.testing.assert_allclose(twice, arr)


# ---------------------------------------------------------------------------
# GlyphPreprocessor (no cv2 required when scale/stroke disabled)
# ---------------------------------------------------------------------------

from hackingrongo.zone_a.preprocessing import GlyphPreprocessor


class TestGlyphPreprocessorNoCv2:
    @_SKIP_TORCH
    def test_passthrough_correct_shape(self):
        """With all cv2-dependent stages off, output should be (1, H, W) tensor."""
        pp = GlyphPreprocessor(
            boustrophedon_normalize=False,
            stroke_extraction=False,
            normalize_scale=False,
            target_size=16,
        )
        arr = np.random.rand(16, 16).astype(np.float32)
        result = pp(arr, line_index=0)
        assert isinstance(result, torch.Tensor)
        assert result.shape == (1, 16, 16)

    @_SKIP_TORCH
    def test_boustrophedon_applied(self):
        """Boustrophedon stage rotates the image on odd lines."""
        pp = GlyphPreprocessor(
            boustrophedon_normalize=True,
            stroke_extraction=False,
            normalize_scale=False,
            target_size=4,
        )
        arr = np.zeros((4, 4), dtype=np.float32)
        arr[0, 0] = 1.0
        # line_index=1 → odd → 180° rotation
        result = pp(arr, line_index=1)
        assert result[0, 3, 3].item() == pytest.approx(1.0)
        assert result[0, 0, 0].item() == pytest.approx(0.0)

    @_SKIP_TORCH
    def test_accepts_3d_tensor_input(self):
        pp = GlyphPreprocessor(
            boustrophedon_normalize=False,
            stroke_extraction=False,
            normalize_scale=False,
            target_size=8,
        )
        arr = np.random.rand(1, 8, 8).astype(np.float32)
        result = pp(arr, line_index=0)
        assert result.shape == (1, 8, 8)

    def test_stroke_extraction_raises_without_cv2(self):
        """When cv2 is absent, stroke_extraction=True should raise RuntimeError."""
        from hackingrongo.zone_a import preprocessing as _pp_mod

        if _pp_mod._CV2_AVAILABLE:
            pytest.skip("cv2 is available; skip the 'missing cv2' path")

        pp = GlyphPreprocessor(
            boustrophedon_normalize=False,
            stroke_extraction=True,
            normalize_scale=False,
            target_size=8,
        )
        arr = np.random.rand(8, 8).astype(np.float32)
        with pytest.raises(RuntimeError, match="opencv"):
            pp(arr, line_index=0)


# ---------------------------------------------------------------------------
# GlyphImagePipeline — unit tests (all I/O mocked)
# ---------------------------------------------------------------------------

from hackingrongo.zone_a.preprocessing import GlyphImagePipeline


class TestGlyphImagePipelineSvgPathFor:
    def _pipeline(self, tmp_path: Path) -> GlyphImagePipeline:
        pp = GlyphPreprocessor(
            boustrophedon_normalize=False,
            stroke_extraction=False,
            normalize_scale=False,
            target_size=16,
        )
        return GlyphImagePipeline(preprocessor=pp, glyphs_svg_dir=tmp_path)

    def test_returns_none_when_file_absent(self, tmp_path):
        pipe = self._pipeline(tmp_path)
        record = {"tablet_id": "D", "side": "a", "line": "01", "glyph_num": "3"}
        assert pipe.svg_path_for(record) is None

    def test_returns_path_when_file_present(self, tmp_path):
        svg_dir = tmp_path / "D"
        svg_dir.mkdir(parents=True)
        (svg_dir / "a01-003.svg").write_text("<svg/>", encoding="utf-8")
        pipe = self._pipeline(tmp_path)
        record = {"tablet_id": "D", "side": "a", "line": "1", "glyph_num": "3"}
        result = pipe.svg_path_for(record)
        assert result is not None
        assert result.exists()

    def test_uses_tablet_field_as_fallback(self, tmp_path):
        svg_dir = tmp_path / "O"
        svg_dir.mkdir(parents=True)
        (svg_dir / "b02-010.svg").write_text("<svg/>", encoding="utf-8")
        pipe = self._pipeline(tmp_path)
        # Use 'tablet' instead of 'tablet_id'
        record = {"tablet": "O", "side": "b", "line": "2", "glyph_num": "10"}
        result = pipe.svg_path_for(record)
        assert result is not None


class TestGlyphImagePipelineRasterize:
    def test_returns_none_when_cairosvg_missing(self, tmp_path):
        """rasterize_svg must return None (not raise) when cairosvg is absent."""
        svg_path = tmp_path / "test.svg"
        svg_path.write_text("<svg/>", encoding="utf-8")

        import importlib
        import sys

        # Patch the cairosvg import to raise ImportError
        with patch.dict(sys.modules, {"cairosvg": None}):
            result = GlyphImagePipeline.rasterize_svg(svg_path, target_size=16)
        assert result is None

    def test_returns_array_when_cairosvg_available(self, tmp_path):
        """When cairosvg is present, rasterize_svg returns a float32 ndarray."""
        svg_path = tmp_path / "test.svg"
        svg_path.write_text("<svg/>", encoding="utf-8")

        # Build a tiny PNG in memory as the mock return value
        from PIL import Image
        img = Image.fromarray(np.zeros((16, 16), dtype=np.uint8), mode="L")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        png_bytes = buf.getvalue()

        mock_cairo = MagicMock()
        mock_cairo.svg2png.return_value = png_bytes

        import sys
        with patch.dict(sys.modules, {"cairosvg": mock_cairo}):
            result = GlyphImagePipeline.rasterize_svg(svg_path, target_size=16)

        assert result is not None
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float32
        assert result.shape == (16, 16)


class TestGlyphImagePipelineCall:
    def _pipeline_with_mocked_rasterize(
        self, tmp_path: Path, raster_return: "np.ndarray | None"
    ) -> GlyphImagePipeline:
        pp = GlyphPreprocessor(
            boustrophedon_normalize=True,
            stroke_extraction=False,
            normalize_scale=False,
            target_size=8,
        )
        pipe = GlyphImagePipeline(preprocessor=pp, glyphs_svg_dir=tmp_path)
        # Patch rasterize to return a known array without needing cairosvg
        pipe.rasterize_svg = MagicMock(return_value=raster_return)
        return pipe

    def _make_svg(self, tmp_path: Path, tablet: str, side: str, line: str, num: str) -> None:
        d = tmp_path / tablet
        d.mkdir(parents=True, exist_ok=True)
        (d / f"{side}{line.zfill(2)}-{num.zfill(3)}.svg").write_text("<svg/>")

    def test_returns_none_when_svg_not_found(self, tmp_path):
        pipe = self._pipeline_with_mocked_rasterize(tmp_path, np.zeros((16, 16), dtype=np.float32))
        record = {"tablet_id": "D", "side": "a", "line": "1", "glyph_num": "1", "inverted": False}
        result = pipe(record)
        assert result is None

    def test_returns_none_when_rasterize_fails(self, tmp_path):
        self._make_svg(tmp_path, "D", "a", "01", "001")
        pipe = self._pipeline_with_mocked_rasterize(tmp_path, None)
        record = {"tablet_id": "D", "side": "a", "line": "1", "glyph_num": "1", "inverted": False}
        result = pipe(record)
        assert result is None

    @_SKIP_TORCH
    def test_returns_tensor_on_success(self, tmp_path):
        self._make_svg(tmp_path, "D", "a", "01", "001")
        arr = np.random.rand(16, 16).astype(np.float32)
        pipe = self._pipeline_with_mocked_rasterize(tmp_path, arr)
        record = {"tablet_id": "D", "side": "a", "line": "1", "glyph_num": "1", "inverted": False}
        result = pipe(record)
        assert isinstance(result, torch.Tensor)
        assert result.shape[0] == 1  # channel dimension

    @_SKIP_TORCH
    def test_inverted_glyph_gets_boustrophedon_applied(self, tmp_path):
        self._make_svg(tmp_path, "D", "a", "01", "001")
        arr = np.zeros((8, 8), dtype=np.float32)
        arr[0, 0] = 1.0  # bright pixel top-left
        pipe = self._pipeline_with_mocked_rasterize(tmp_path, arr)

        record_inverted = {"tablet_id": "D", "side": "a", "line": "1", "glyph_num": "1", "inverted": True}
        record_normal  = {"tablet_id": "D", "side": "a", "line": "1", "glyph_num": "1", "inverted": False}

        result_inv = pipe(record_inverted)
        pipe.rasterize_svg.return_value = arr.copy()  # reset mock
        result_norm = pipe(record_normal)

        assert result_inv is not None
        assert result_norm is not None
        # Inverted version should differ from normal (rotation applied)
        assert not torch.allclose(result_inv, result_norm)
