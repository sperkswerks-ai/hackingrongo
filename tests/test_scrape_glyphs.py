"""
tests.test_scrape_glyphs
========================

Unit tests for scripts/scrape_glyphs.py.

All network calls are mocked — no internet access required.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# scrape_glyphs lives in scripts/tooling/ which is not a package; add it to sys.path
SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts" / "tooling"
sys.path.insert(0, str(SCRIPTS_DIR))

import scrape_glyphs as sg  # noqa: E402


# ---------------------------------------------------------------------------
# _approx_bbox
# ---------------------------------------------------------------------------

class TestApproxBbox:
    def test_simple_rect_path(self):
        # M 10 20 H 90 V 80 Z  — four x values: 10,90,90,10; four y: 20,20,80,80
        d = "M 10 20 H 90 V 80 Z"
        x_min, y_min, x_max, y_max = sg._approx_bbox(d)
        # With 4-unit margin: x_min<=10-4, x_max>=90+4
        assert x_min < 10.0
        assert x_max > 90.0
        assert y_min < 20.0
        assert y_max > 80.0

    def test_empty_path_returns_default(self):
        x_min, y_min, x_max, y_max = sg._approx_bbox("")
        # Default fallback: (0, 0, 64, 64)
        assert (x_min, y_min, x_max, y_max) == (0.0, 0.0, 64.0, 64.0)

    def test_single_point(self):
        # Just a single coordinate pair
        x_min, y_min, x_max, y_max = sg._approx_bbox("M 5 7")
        # Both min and max equal the same value (+/- margin)
        assert x_min <= 5.0 - 4.0 + 0.01
        assert x_max >= 5.0 + 4.0 - 0.01


# ---------------------------------------------------------------------------
# parse_glyph_id
# ---------------------------------------------------------------------------

class TestParseGlyphId:
    def test_valid_id(self):
        result = sg.parse_glyph_id("glyphDa01-003-b")
        assert result == {"tablet": "D", "side": "a", "line": "01", "glyph_num": "003"}

    def test_valid_side_b(self):
        result = sg.parse_glyph_id("glyphOb12-047-b")
        assert result == {"tablet": "O", "side": "b", "line": "12", "glyph_num": "047"}

    def test_invalid_returns_none(self):
        assert sg.parse_glyph_id("not-a-glyph") is None
        assert sg.parse_glyph_id("glyphda01-003-b") is None  # lowercase tablet
        assert sg.parse_glyph_id("glyphDa1-003-b") is None   # 1-digit line

    def test_boundary_glyph_001(self):
        result = sg.parse_glyph_id("glyphAa01-001-b")
        assert result is not None
        assert result["glyph_num"] == "001"


# ---------------------------------------------------------------------------
# build_svg
# ---------------------------------------------------------------------------

class TestBuildSvg:
    def test_output_is_valid_xml(self):
        from xml.etree import ElementTree as ET
        d = "M 10 10 L 50 50 L 10 50 Z"
        svg = sg.build_svg(d)
        # Should parse without error
        root = ET.fromstring(svg)
        assert root.tag.endswith("svg")

    def test_path_preserved(self):
        d = "M 20 30 C 40 50 60 50 80 30"
        svg = sg.build_svg(d)
        assert d in svg

    def test_viewbox_present(self):
        d = "M 10 10 L 90 90"
        svg = sg.build_svg(d)
        assert "viewBox" in svg


# ---------------------------------------------------------------------------
# build_corpus_index
# ---------------------------------------------------------------------------

class TestBuildCorpusIndex:
    def test_basic_index(self, tmp_path):
        corpus = {
            "tablet_id": "D",
            "cluster": "A",
            "glyphs": [
                {
                    "side": "a",
                    "line": "01",
                    "glyph_num": "3",
                    "barthel_code": "734",
                    "horley_code": "739 6",
                    "horley_components": None,
                    "inverted": False,
                    "uncertain": False,
                    "position": 3,
                }
            ],
        }
        (tmp_path / "D.json").write_text(json.dumps(corpus), encoding="utf-8")
        idx = sg.build_corpus_index(tmp_path)
        assert "Da01-003" in idx
        entry = idx["Da01-003"]
        assert entry["barthel_code"] == "734"
        assert entry["horley_code"] == "739 6"
        assert entry["cluster"] == "A"

    def test_empty_corpus_dir(self, tmp_path):
        idx = sg.build_corpus_index(tmp_path)
        assert idx == {}

    def test_multi_tablet(self, tmp_path):
        for tid in ("A", "B"):
            corpus = {
                "tablet_id": tid,
                "cluster": "X",
                "glyphs": [
                    {"side": "a", "line": "01", "glyph_num": "1",
                     "barthel_code": "1", "horley_code": "1",
                     "horley_components": None, "inverted": False,
                     "uncertain": False, "position": 1}
                ],
            }
            (tmp_path / f"{tid}.json").write_text(json.dumps(corpus), encoding="utf-8")
        idx = sg.build_corpus_index(tmp_path)
        assert "Aa01-001" in idx
        assert "Ba01-001" in idx


# ---------------------------------------------------------------------------
# iter_tablet_paths
# ---------------------------------------------------------------------------

class TestIterTabletPaths:
    _SAMPLE_HTML = """
    <path id="glyphDa01-001-b" d="M 10 10 L 50 50"/>
    <path id="glyphDa01-002-b" d="M 20 20 C 30 30 40 40 50 20"/>
    <path id="not-a-glyph-id" d="M 0 0"/>
    """

    def test_finds_two_glyph_paths(self):
        results = list(sg.iter_tablet_paths(self._SAMPLE_HTML))
        assert len(results) == 2

    def test_returns_id_and_d(self):
        results = list(sg.iter_tablet_paths(self._SAMPLE_HTML))
        ids = {r[0] for r in results}
        assert "glyphDa01-001-b" in ids
        assert "glyphDa01-002-b" in ids


# ---------------------------------------------------------------------------
# scrape_tablet — dry-run (no network, no file writes)
# ---------------------------------------------------------------------------

class TestScrapeTabletDryRun:
    _MOCK_HTML = (
        '<path id="glyphDa01-001-b" d="M 10 10 L 50 50"/>'
        '<path id="glyphDa01-002-b" d="M 20 20 L 60 60"/>'
    )

    def _make_index(self):
        return {
            "Da01-001": {"barthel_code": "1", "horley_code": "2", "horley_components": None,
                         "inverted": False, "uncertain": False, "position": 1, "cluster": "A"},
            "Da01-002": {"barthel_code": "3", "horley_code": "4", "horley_components": None,
                         "inverted": False, "uncertain": False, "position": 2, "cluster": "A"},
        }

    def test_dry_run_returns_records_without_writing(self, tmp_path):
        corpus_idx = self._make_index()
        with patch.object(sg, "_fetch", return_value=self._MOCK_HTML):
            records = sg.scrape_tablet("D", corpus_idx, tmp_path, dry_run=True)

        assert len(records) == 2
        # No files should be written in dry-run mode
        assert not list(tmp_path.glob("**/*.svg"))

    def test_records_have_expected_fields(self, tmp_path):
        corpus_idx = self._make_index()
        with patch.object(sg, "_fetch", return_value=self._MOCK_HTML):
            records = sg.scrape_tablet("D", corpus_idx, tmp_path, dry_run=True)

        rec = records[0]
        assert "glyph_id" in rec
        assert "barthel_code" in rec
        assert "svg_path" in rec

    def test_real_run_writes_svg_files(self, tmp_path):
        corpus_idx = self._make_index()
        with patch.object(sg, "_fetch", return_value=self._MOCK_HTML):
            records = sg.scrape_tablet("D", corpus_idx, tmp_path, dry_run=False)

        svg_files = list(tmp_path.glob("**/*.svg"))
        assert len(svg_files) == 2

    def test_unknown_glyph_id_skipped(self, tmp_path):
        # A path whose ID isn't in the corpus index
        html = '<path id="glyphZz99-001-b" d="M 0 0"/>'
        with patch.object(sg, "_fetch", return_value=html):
            records = sg.scrape_tablet("Z", {}, tmp_path, dry_run=True)
        # parse_glyph_id will return None for 'z' (lowercase) — skipped
        assert records == []
