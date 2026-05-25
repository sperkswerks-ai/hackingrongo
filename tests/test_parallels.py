"""
tests.test_parallels
====================

Smoke tests for hackingrongo.data.parallels.

Tests cover:

* ``tag_taxogram_positions`` — index tagging in glyph sequences.
* ``ParallelPassage`` — per-passage omission rate, variant filtering,
  phonetic canonical form.
* ``load_parallel_passages`` — CSV parsing, canonical-form assignment,
  taxogram retention, error handling.
* ``compute_omission_rates`` — aggregated omission rates.
"""

from __future__ import annotations

import csv
import io
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from hackingrongo.data.catalog import SignCatalog
from hackingrongo.data.parallels import (
    ParallelPassage,
    PassageVariant,
    compute_omission_rates,
    load_parallel_passages,
    load_parallel_variants_json,
    tag_taxogram_positions,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def catalog() -> SignCatalog:
    horley = {"001": "H001", "002": "H002", "003": "H003", "200": "H200"}
    allographs = {"001": "001", "002": "002", "003": "003", "200": "200"}
    metadata: dict = {
        "200": {"is_taxogram_candidate": True, "scholarly_readings": [], "notes": ""}
    }
    return SignCatalog(horley, allographs, metadata)


@pytest.fixture()
def cfg() -> OmegaConf:
    return OmegaConf.create({})


def _make_csv(rows: list[dict]) -> str:
    """Serialise a list of dicts to a CSV string with the required header."""
    buf = io.StringIO()
    fieldnames = [
        "passage_id",
        "tablet_id",
        "side",
        "stratum",
        "start_position",
        "glyph_sequence",
    ]
    writer = csv.DictWriter(buf, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# TestTagTaxogramPositions
# ---------------------------------------------------------------------------


class TestTagTaxogramPositions:
    def test_single_taxogram(self):
        pos = tag_taxogram_positions(["001", "200", "002"])
        assert pos == frozenset({1})

    def test_multiple_taxograms(self):
        pos = tag_taxogram_positions(["200", "001", "200"])
        assert pos == frozenset({0, 2})

    def test_no_taxogram(self):
        pos = tag_taxogram_positions(["001", "002", "003"])
        assert pos == frozenset()

    def test_empty_form(self):
        pos = tag_taxogram_positions([])
        assert pos == frozenset()

    def test_custom_taxogram_code(self):
        pos = tag_taxogram_positions(["001", "003", "001"], taxogram_code="003")
        assert pos == frozenset({1})


# ---------------------------------------------------------------------------
# TestParallelPassage
# ---------------------------------------------------------------------------


class TestParallelPassage:
    def _passage_with_variants(
        self, n_omitted: int, form_len: int = 3
    ) -> ParallelPassage:
        """Helper: build a passage where `n_omitted` variants are shorter."""
        canonical = ["001", "200", "002"]
        variants = []
        # Full variants
        for i in range(3 - n_omitted):
            variants.append(
                PassageVariant(
                    form=tuple(canonical),
                    tablet_id=f"T{i}",
                    stratum="early",
                )
            )
        # Shortened variants (omit last element)
        for i in range(n_omitted):
            variants.append(
                PassageVariant(
                    form=tuple(canonical[: form_len - 1]),
                    tablet_id=f"S{i}",
                    stratum="late",
                )
            )
        return ParallelPassage(
            passage_id="P001",
            canonical_form=canonical,
            variants=variants,
            taxogram_positions=tag_taxogram_positions(canonical),
        )

    def test_n_variants(self):
        p = self._passage_with_variants(1)
        assert p.n_variants == 3

    def test_omission_rate_zero_when_all_complete(self):
        p = self._passage_with_variants(0)
        # position 2 (last element) — all variants include it
        assert p.omission_rate_at_position(2) == pytest.approx(0.0)

    def test_omission_rate_position_beyond_short_variants(self):
        # 2 of 3 variants have form of length 2; position 2 is beyond them
        p = self._passage_with_variants(2)
        # 2/3 variants omit position 2
        assert p.omission_rate_at_position(2) == pytest.approx(2 / 3)

    def test_omission_rate_no_variants_returns_zero(self):
        p = ParallelPassage(
            passage_id="P000",
            canonical_form=["001"],
            variants=[],
            taxogram_positions=frozenset(),
        )
        assert p.omission_rate_at_position(0) == pytest.approx(0.0)

    def test_get_variants_for_stratum(self):
        p = self._passage_with_variants(1)
        early = p.get_variants_for_stratum("early")
        late = p.get_variants_for_stratum("late")
        assert len(early) == 2
        assert len(late) == 1

    def test_attested_strata(self):
        p = self._passage_with_variants(1)
        assert p.attested_strata == {"early", "late"}

    def test_phonetic_canonical_form_strips_taxogram(self):
        p = ParallelPassage(
            passage_id="P001",
            canonical_form=["001", "200", "002"],
            variants=[],
            taxogram_positions=frozenset({1}),
        )
        phonetic = p.phonetic_canonical_form()
        assert phonetic == ["001", "002"]
        assert "200" not in phonetic

    def test_taxogram_retained_in_canonical_form(self):
        p = ParallelPassage(
            passage_id="P001",
            canonical_form=["001", "200", "002"],
            variants=[],
            taxogram_positions=frozenset({1}),
        )
        # Glyph 200 must still be in canonical_form (not stripped)
        assert "200" in p.canonical_form
        # And its position is tagged
        assert 1 in p.taxogram_positions


# ---------------------------------------------------------------------------
# TestLoadParallelPassages
# ---------------------------------------------------------------------------


class TestLoadParallelPassages:
    def test_loads_two_passages(self, tmp_path, catalog, cfg):
        csv_content = _make_csv(
            [
                {
                    "passage_id": "P001",
                    "tablet_id": "A",
                    "side": "a",
                    "stratum": "early",
                    "start_position": "1",
                    "glyph_sequence": "001 002 003",
                },
                {
                    "passage_id": "P002",
                    "tablet_id": "B",
                    "side": "b",
                    "stratum": "late",
                    "start_position": "5",
                    "glyph_sequence": "002 003",
                },
            ]
        )
        csv_path = tmp_path / "parallels.csv"
        csv_path.write_text(csv_content, encoding="utf-8")
        passages = load_parallel_passages(csv_path, catalog, cfg)
        assert len(passages) == 2
        ids = [p.passage_id for p in passages]
        assert "P001" in ids
        assert "P002" in ids

    def test_first_variant_becomes_canonical(self, tmp_path, catalog, cfg):
        csv_content = _make_csv(
            [
                {
                    "passage_id": "P001",
                    "tablet_id": "A",
                    "side": "a",
                    "stratum": "early",
                    "start_position": "1",
                    "glyph_sequence": "001 002",
                },
                {
                    "passage_id": "P001",
                    "tablet_id": "B",
                    "side": "a",
                    "stratum": "late",
                    "start_position": "3",
                    "glyph_sequence": "001 003",
                },
            ]
        )
        csv_path = tmp_path / "parallels.csv"
        csv_path.write_text(csv_content, encoding="utf-8")
        passages = load_parallel_passages(csv_path, catalog, cfg)
        assert len(passages) == 1
        # First row defines canonical form
        assert passages[0].canonical_form == ["001", "002"]
        # Both rows are variants
        assert passages[0].n_variants == 2

    def test_taxogram_retained_and_tagged(self, tmp_path, catalog, cfg):
        csv_content = _make_csv(
            [
                {
                    "passage_id": "P001",
                    "tablet_id": "A",
                    "side": "a",
                    "stratum": "early",
                    "start_position": "1",
                    "glyph_sequence": "001 200 002",
                },
            ]
        )
        csv_path = tmp_path / "parallels.csv"
        csv_path.write_text(csv_content, encoding="utf-8")
        passages = load_parallel_passages(csv_path, catalog, cfg)
        p = passages[0]
        # 200 must be in canonical_form
        assert "200" in p.canonical_form
        # And at tagged position
        assert 1 in p.taxogram_positions

    def test_raises_on_missing_file(self, tmp_path, catalog, cfg):
        missing = tmp_path / "nonexistent.csv"
        with pytest.raises(FileNotFoundError, match="not found"):
            load_parallel_passages(missing, catalog, cfg)

    def test_raises_on_missing_columns(self, tmp_path, catalog, cfg):
        # CSV with a missing required column
        bad_csv = "passage_id,tablet_id\nP001,A\n"
        csv_path = tmp_path / "bad.csv"
        csv_path.write_text(bad_csv, encoding="utf-8")
        with pytest.raises(ValueError, match="missing required columns"):
            load_parallel_passages(csv_path, catalog, cfg)

    def test_passages_sorted_by_id(self, tmp_path, catalog, cfg):
        csv_content = _make_csv(
            [
                {
                    "passage_id": "P003",
                    "tablet_id": "A",
                    "side": "",
                    "stratum": "early",
                    "start_position": "-1",
                    "glyph_sequence": "001",
                },
                {
                    "passage_id": "P001",
                    "tablet_id": "B",
                    "side": "",
                    "stratum": "late",
                    "start_position": "-1",
                    "glyph_sequence": "002",
                },
            ]
        )
        csv_path = tmp_path / "parallels.csv"
        csv_path.write_text(csv_content, encoding="utf-8")
        passages = load_parallel_passages(csv_path, catalog, cfg)
        assert passages[0].passage_id == "P001"
        assert passages[1].passage_id == "P003"

    def test_unknown_codes_warn_but_continue(
        self, tmp_path, catalog, cfg, caplog
    ):
        csv_content = _make_csv(
            [
                {
                    "passage_id": "P001",
                    "tablet_id": "A",
                    "side": "",
                    "stratum": "early",
                    "start_position": "1",
                    "glyph_sequence": "001 999",
                },
            ]
        )
        csv_path = tmp_path / "parallels.csv"
        csv_path.write_text(csv_content, encoding="utf-8")
        import logging

        with caplog.at_level(logging.WARNING, logger="hackingrongo.data.parallels"):
            passages = load_parallel_passages(csv_path, catalog, cfg)
        assert len(passages) == 1
        assert any("999" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# TestLoadParallelVariantsJson
# ---------------------------------------------------------------------------


class TestLoadParallelVariantsJson:
    """Tests for load_parallel_variants_json — the JSON variant loader.

    Mirrors the coverage of TestLoadParallelPassages to ensure both loaders
    maintain feature and error-handling parity.
    """

    def _write_json(self, tmp_path, passages: list[dict]) -> "Path":
        p = tmp_path / "variants.json"
        p.write_text(
            __import__("json").dumps({"passages": passages}), encoding="utf-8"
        )
        return p

    def test_happy_path_single_passage(self, tmp_path, catalog):
        path = self._write_json(tmp_path, [
            {
                "passage_id": "P001",
                "canonical_form": ["001", "002"],
                "variants": [
                    {"form": ["001", "002"], "tablet_id": "A", "stratum": "early", "side": "a", "start_position": 1},
                ],
            }
        ])
        passages = load_parallel_variants_json(path, catalog)
        assert len(passages) == 1
        assert passages[0].passage_id == "P001"
        assert passages[0].canonical_form == ["001", "002"]

    def test_multi_tablet_passage(self, tmp_path, catalog):
        path = self._write_json(tmp_path, [
            {
                "passage_id": "P001",
                "canonical_form": ["001", "200", "002"],
                "variants": [
                    {"form": ["001", "200", "002"], "tablet_id": "A", "stratum": "early"},
                    {"form": ["001", "002"],         "tablet_id": "B", "stratum": "late"},
                ],
            }
        ])
        passages = load_parallel_variants_json(path, catalog)
        assert passages[0].n_variants == 2

    def test_taxogram_retained_and_tagged(self, tmp_path, catalog):
        path = self._write_json(tmp_path, [
            {
                "passage_id": "P001",
                "canonical_form": ["001", "200", "002"],
                "variants": [{"form": ["001", "200", "002"], "tablet_id": "A", "stratum": "early"}],
            }
        ])
        passages = load_parallel_variants_json(path, catalog)
        p = passages[0]
        assert "200" in p.canonical_form
        assert 1 in p.taxogram_positions

    def test_raises_on_missing_file(self, tmp_path, catalog):
        with pytest.raises(FileNotFoundError, match="not found"):
            load_parallel_variants_json(tmp_path / "nonexistent.json", catalog)

    def test_missing_passage_id_raises(self, tmp_path, catalog):
        path = self._write_json(tmp_path, [
            {"canonical_form": ["001"], "variants": []}  # no passage_id
        ])
        with pytest.raises(ValueError, match="passage_id"):
            load_parallel_variants_json(path, catalog)

    def test_missing_canonical_form_raises(self, tmp_path, catalog):
        path = self._write_json(tmp_path, [
            {"passage_id": "P001", "variants": []}  # no canonical_form
        ])
        with pytest.raises(ValueError, match="canonical_form"):
            load_parallel_variants_json(path, catalog)

    def test_missing_variant_form_raises(self, tmp_path, catalog):
        path = self._write_json(tmp_path, [
            {
                "passage_id": "P001",
                "canonical_form": ["001"],
                "variants": [{"tablet_id": "A"}],  # no form key
            }
        ])
        with pytest.raises(ValueError, match="form"):
            load_parallel_variants_json(path, catalog)

    def test_passages_sorted_by_id(self, tmp_path, catalog):
        path = self._write_json(tmp_path, [
            {"passage_id": "P003", "canonical_form": ["001"], "variants": []},
            {"passage_id": "P001", "canonical_form": ["002"], "variants": []},
        ])
        passages = load_parallel_variants_json(path, catalog)
        assert passages[0].passage_id == "P001"
        assert passages[1].passage_id == "P003"

    def test_unknown_codes_warn(self, tmp_path, catalog, caplog):
        path = self._write_json(tmp_path, [
            {
                "passage_id": "P001",
                "canonical_form": ["001", "999"],  # 999 not in catalog
                "variants": [],
            }
        ])
        import logging
        with caplog.at_level(logging.WARNING, logger="hackingrongo.data.parallels"):
            load_parallel_variants_json(path, catalog)
        assert any("999" in r.message for r in caplog.records)

    def test_no_multi_tablet_logs_critical(self, tmp_path, catalog, caplog):
        path = self._write_json(tmp_path, [
            {
                "passage_id": "P001",
                "canonical_form": ["001"],
                "variants": [{"form": ["001"], "tablet_id": "A", "stratum": "early"}],
            }
        ])
        import logging
        with caplog.at_level(logging.CRITICAL, logger="hackingrongo.data.parallels"):
            load_parallel_variants_json(path, catalog)
        assert any(r.levelname == "CRITICAL" for r in caplog.records)


# ---------------------------------------------------------------------------
# TestComputeOmissionRates
# ---------------------------------------------------------------------------


class TestComputeOmissionRates:
    def test_taxogram_has_high_omission_rate(self, catalog):
        """Glyph 200 omitted in some variants → non-zero omission rate."""
        canonical = ["001", "200", "002"]
        p = ParallelPassage(
            passage_id="P001",
            canonical_form=canonical,
            variants=[
                # Full variant: includes 200
                PassageVariant(
                    form=("001", "200", "002"), tablet_id="A", stratum="early"
                ),
                # Shortened variant: stops before 200
                PassageVariant(
                    form=("001",), tablet_id="B", stratum="late"
                ),
            ],
            taxogram_positions=tag_taxogram_positions(canonical),
        )
        rates = compute_omission_rates([p], catalog)
        # 200 is omitted in 1/2 variants at position 1
        assert rates["200"] == pytest.approx(0.5)

    def test_sign_not_in_any_passage_has_zero_rate(self, catalog):
        canonical = ["001", "002"]
        p = ParallelPassage(
            passage_id="P001",
            canonical_form=canonical,
            variants=[
                PassageVariant(form=("001", "002"), tablet_id="A", stratum="early"),
            ],
            taxogram_positions=frozenset(),
        )
        rates = compute_omission_rates([p], catalog)
        # 003 never appears canonically → 0.0
        assert rates.get("003", 0.0) == pytest.approx(0.0)

    def test_all_rates_in_unit_interval(self, catalog):
        canonical = ["001", "200", "002"]
        p = ParallelPassage(
            passage_id="P001",
            canonical_form=canonical,
            variants=[
                PassageVariant(
                    form=("001", "200", "002"), tablet_id="A", stratum="early"
                ),
            ],
            taxogram_positions=tag_taxogram_positions(canonical),
        )
        rates = compute_omission_rates([p], catalog)
        for code, rate in rates.items():
            assert 0.0 <= rate <= 1.0, f"Out-of-range rate for {code}: {rate}"
