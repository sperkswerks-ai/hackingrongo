"""
tests.test_catalog
==================

Smoke tests for hackingrongo.data.catalog.SignCatalog.

All tests are self-contained, building minimal in-memory JSON structures
and passing them directly to the SignCatalog constructor.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from hackingrongo.data.catalog import SignCatalog, SignRecord


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def minimal_horley() -> dict:
    return {"001": "H001", "001a": "H001a", "002": "H002", "200": "H200"}


@pytest.fixture()
def minimal_allographs() -> dict:
    return {"001": "001", "001a": "001", "002": "002", "200": "200"}


@pytest.fixture()
def minimal_metadata() -> dict:
    return {
        "001": {
            "scholarly_readings": ["ragi"],
            "is_taxogram_candidate": False,
            "notes": "Frequent sign.",
        },
        "200": {
            "scholarly_readings": [],
            "is_taxogram_candidate": True,
            "notes": "Primary taxogram.",
        },
    }


@pytest.fixture()
def catalog(minimal_horley, minimal_allographs, minimal_metadata) -> SignCatalog:
    return SignCatalog(minimal_horley, minimal_allographs, minimal_metadata)


@pytest.fixture()
def cfg(tmp_path) -> OmegaConf:
    return OmegaConf.create(
        {
            "paths": {
                "horley_encoding_json": "horley.json",
                "allographs_json": "allographs.json",
                "sign_metadata_json": "sign_metadata.json",
            }
        }
    )


def _write_catalog_files(
    root: Path, horley: dict, allographs: dict, metadata: dict
) -> None:
    (root / "horley.json").write_text(json.dumps(horley), encoding="utf-8")
    (root / "allographs.json").write_text(json.dumps(allographs), encoding="utf-8")
    (root / "sign_metadata.json").write_text(json.dumps(metadata), encoding="utf-8")


# ---------------------------------------------------------------------------
# SignCatalog construction
# ---------------------------------------------------------------------------


class TestSignCatalogConstruction:
    def test_signs_dict_populated(self, catalog):
        assert "001" in catalog.signs
        assert "200" in catalog.signs

    def test_sign_record_type(self, catalog):
        assert isinstance(catalog.signs["001"], SignRecord)

    def test_schema_keys_excluded(self):
        """Keys starting with '_' in JSON schema must not appear as sign codes."""
        horley = {"_schema_version": "1.0", "001": "H001"}
        allographs = {"001": "001"}
        cat = SignCatalog(horley, allographs, {})
        assert "_schema_version" not in cat.signs
        assert "001" in cat.signs


# ---------------------------------------------------------------------------
# barthel_to_horley
# ---------------------------------------------------------------------------


class TestBarthelToHorley:
    def test_known_code_returns_horley(self, catalog):
        assert catalog.barthel_to_horley("001") == "H001"

    def test_variant_returns_horley(self, catalog):
        assert catalog.barthel_to_horley("001a") == "H001a"

    def test_unknown_code_returns_none(self, catalog):
        assert catalog.barthel_to_horley("999") is None


# ---------------------------------------------------------------------------
# horley_to_barthels
# ---------------------------------------------------------------------------


class TestHorleyToBarthels:
    def test_returns_list(self, catalog):
        result = catalog.horley_to_barthels("H001")
        assert isinstance(result, list)
        assert "001" in result

    def test_unknown_returns_empty(self, catalog):
        assert catalog.horley_to_barthels("H999") == []


# ---------------------------------------------------------------------------
# get_canonical_id
# ---------------------------------------------------------------------------


class TestGetCanonicalId:
    def test_variant_resolves_to_canonical(self, catalog):
        assert catalog.get_canonical_id("001a") == "001"

    def test_canonical_returns_itself(self, catalog):
        assert catalog.get_canonical_id("001") == "001"

    def test_unknown_code_returns_itself(self, catalog):
        assert catalog.get_canonical_id("999") == "999"


# ---------------------------------------------------------------------------
# get_allograph_group
# ---------------------------------------------------------------------------


class TestGetAllographGroup:
    def test_canonical_includes_all_variants(self, catalog):
        group = catalog.get_allograph_group("001")
        assert "001" in group
        assert "001a" in group

    def test_variant_returns_same_group_as_canonical(self, catalog):
        via_variant = catalog.get_allograph_group("001a")
        via_canonical = catalog.get_allograph_group("001")
        assert sorted(via_variant) == sorted(via_canonical)

    def test_unknown_code_returns_singleton(self, catalog):
        assert catalog.get_allograph_group("999") == ["999"]


# ---------------------------------------------------------------------------
# is_taxogram / get_taxogram_codes
# ---------------------------------------------------------------------------


class TestTaxogramQueries:
    def test_glyph_200_is_taxogram(self, catalog):
        assert catalog.is_taxogram("200") is True

    def test_glyph_001_is_not_taxogram(self, catalog):
        assert catalog.is_taxogram("001") is False

    def test_unknown_code_is_not_taxogram(self, catalog):
        assert catalog.is_taxogram("999") is False

    def test_get_taxogram_codes_returns_200(self, catalog):
        codes = catalog.get_taxogram_codes()
        assert "200" in codes
        assert "001" not in codes

    def test_get_taxogram_codes_sorted(self, catalog):
        codes = catalog.get_taxogram_codes()
        assert codes == sorted(codes)


# ---------------------------------------------------------------------------
# scholarly_readings
# ---------------------------------------------------------------------------


class TestScholarlyReadings:
    def test_readings_loaded(self, catalog):
        rec = catalog.signs["001"]
        assert "ragi" in rec.scholarly_readings

    def test_no_readings_empty_tuple(self, catalog):
        rec = catalog.signs["200"]
        assert rec.scholarly_readings == ()


# ---------------------------------------------------------------------------
# SignCatalog.load (from disk)
# ---------------------------------------------------------------------------


class TestSignCatalogLoad:
    def test_loads_from_disk(
        self, tmp_path, cfg, minimal_horley, minimal_allographs, minimal_metadata
    ):
        _write_catalog_files(
            tmp_path, minimal_horley, minimal_allographs, minimal_metadata
        )
        loaded = SignCatalog.load(cfg, tmp_path)
        assert "001" in loaded.signs
        assert loaded.is_taxogram("200") is True

    def test_raises_on_missing_horley_file(
        self, tmp_path, cfg, minimal_allographs, minimal_metadata
    ):
        # Only write allographs and metadata; omit horley.
        (tmp_path / "allographs.json").write_text(
            json.dumps(minimal_allographs), encoding="utf-8"
        )
        (tmp_path / "sign_metadata.json").write_text(
            json.dumps(minimal_metadata), encoding="utf-8"
        )
        with pytest.raises(FileNotFoundError, match="Horley encoding"):
            SignCatalog.load(cfg, tmp_path)

    def test_raises_on_missing_allographs_file(
        self, tmp_path, cfg, minimal_horley, minimal_metadata
    ):
        (tmp_path / "horley.json").write_text(
            json.dumps(minimal_horley), encoding="utf-8"
        )
        (tmp_path / "sign_metadata.json").write_text(
            json.dumps(minimal_metadata), encoding="utf-8"
        )
        with pytest.raises(FileNotFoundError, match="allograph catalog"):
            SignCatalog.load(cfg, tmp_path)


# ---------------------------------------------------------------------------
# barthel_to_implicit_group
# ---------------------------------------------------------------------------


class TestBarthelToImplicitGroup:
    """Tests for SignCatalog.barthel_to_implicit_group.

    The method is a @staticmethod, so any SignCatalog instance works.
    Tests explicitly cover allograph suffixes beyond a/b/c — the bug that
    was previously masked by the incomplete .replace() chain.
    """

    def test_objects_range_bare(self, catalog):
        assert catalog.barthel_to_implicit_group("001") == "objects_plants_phenomena"

    def test_objects_range_padded(self, catalog):
        assert catalog.barthel_to_implicit_group("076") == "objects_plants_phenomena"

    def test_objects_range_unpadded(self, catalog):
        assert catalog.barthel_to_implicit_group("99") == "objects_plants_phenomena"

    def test_suffix_a(self, catalog):
        assert catalog.barthel_to_implicit_group("076a") == "objects_plants_phenomena"

    def test_suffix_b(self, catalog):
        assert catalog.barthel_to_implicit_group("076b") == "objects_plants_phenomena"

    def test_suffix_c(self, catalog):
        assert catalog.barthel_to_implicit_group("076c") == "objects_plants_phenomena"

    def test_suffix_d(self, catalog):
        # Previously returned "unknown" due to incomplete .replace() chain.
        assert catalog.barthel_to_implicit_group("380d") == "anthropomorphic_300series"

    def test_suffix_e(self, catalog):
        assert catalog.barthel_to_implicit_group("380e") == "anthropomorphic_300series"

    def test_suffix_f(self, catalog):
        assert catalog.barthel_to_implicit_group("076f") == "objects_plants_phenomena"

    def test_taxogram_200_series(self, catalog):
        assert catalog.barthel_to_implicit_group("200") == "anthropomorphic_head0"

    def test_taxogram_210(self, catalog):
        assert catalog.barthel_to_implicit_group("210") == "anthropomorphic_head1"

    def test_300_series(self, catalog):
        assert catalog.barthel_to_implicit_group("300") == "anthropomorphic_300series"

    def test_400_series(self, catalog):
        assert catalog.barthel_to_implicit_group("400") == "miscellaneous_400s"

    def test_500_series(self, catalog):
        assert catalog.barthel_to_implicit_group("500") == "miscellaneous_500s"

    def test_600_series(self, catalog):
        assert catalog.barthel_to_implicit_group("600") == "bird_headed"

    def test_700_series(self, catalog):
        assert catalog.barthel_to_implicit_group("700") == "zoomorphic"

    def test_out_of_range_high(self, catalog):
        assert catalog.barthel_to_implicit_group("800") == "compound_or_other"

    def test_non_numeric_returns_unknown(self, catalog):
        assert catalog.barthel_to_implicit_group("xyz") == "unknown"

    def test_empty_string_returns_unknown(self, catalog):
        assert catalog.barthel_to_implicit_group("") == "unknown"
