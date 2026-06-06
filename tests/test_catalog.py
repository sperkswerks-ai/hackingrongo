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


# ---------------------------------------------------------------------------
# get_barthel_family() — new iconographic family lookup
# ---------------------------------------------------------------------------

_FAMILIES_JSON = Path(__file__).resolve().parents[1] / "data" / "catalog" / "barthel_families.json"

_ACTIVE_SIGNS = [
    "001", "002", "003", "004", "005", "006", "007", "008", "009", "010",
    "011", "013", "015", "017", "019", "020", "022", "023", "024", "025",
    "027", "028", "034", "037", "040", "041", "044", "045", "046", "047",
    "048", "050", "052", "053", "056", "057", "060", "061", "062", "063",
    "064", "065", "066", "067", "068", "069", "070", "071", "072", "073",
    "074", "075", "076", "077", "078", "079", "080", "081", "084", "086",
    "087", "090", "091", "092", "093", "095", "099", "124",
    "200", "202", "203", "204", "205", "206", "207", "208",
    "220", "240", "244", "246", "254", "260", "280", "290", "291",
    "300", "301", "305", "306", "320", "326", "330", "360", "379",
    "380", "381", "384", "385", "386", "390",
    "400", "430", "431", "440", "450", "451", "470", "499",
    "522", "530", "591",
    "600", "604", "605", "606", "607", "630", "631", "660", "670", "678", "680",
    "700", "711", "730", "739", "741", "745", "755", "760",
    "999",
]

_VALID_FAMILIES = {
    "anthropomorphic", "zoomorphic", "botanical", "celestial",
    "geometric", "composite", "positional", "unknown",
}
_ARITHMETIC_LABELS = {str(i) for i in range(10)}


@pytest.fixture(scope="module")
def families_json() -> dict[str, str]:
    raw = json.loads(_FAMILIES_JSON.read_text(encoding="utf-8"))
    return {k: v for k, v in raw.items() if not k.startswith("_")}


@pytest.fixture(scope="module")
def catalog_with_families(families_json):
    minimal_horley = {code: f"H{i:03d}" for i, code in enumerate(_ACTIVE_SIGNS)}
    return SignCatalog(minimal_horley, {}, {}, families_json)


class TestGetBarthelFamily:
    # ── (a) All active signs have a family ───────────────────────────────────

    def test_families_json_covers_all_active_signs(self, families_json):
        missing = [c for c in _ACTIVE_SIGNS if c not in families_json]
        assert not missing, f"Missing from barthel_families.json: {missing}"

    def test_get_barthel_family_returns_non_none(self, catalog_with_families):
        for code in _ACTIVE_SIGNS:
            assert catalog_with_families.get_barthel_family(code) is not None

    def test_all_family_labels_are_valid(self, families_json):
        invalid = {k: v for k, v in families_json.items() if v not in _VALID_FAMILIES}
        assert not invalid, f"Invalid family labels: {invalid}"

    def test_catalog_matches_json(self, catalog_with_families, families_json):
        for code in _ACTIVE_SIGNS:
            expected = families_json.get(code, "unknown")
            assert catalog_with_families.get_barthel_family(code) == expected

    # ── (b) No arithmetic derivation ─────────────────────────────────────────

    def test_no_bare_integer_labels(self, families_json):
        bad = {k: v for k, v in families_json.items() if v in _ARITHMETIC_LABELS}
        assert not bad, f"Arithmetic-looking family labels (should be iconographic): {bad}"

    def test_sign_678_not_arithmetic(self, families_json):
        assert families_json.get("678") not in _ARITHMETIC_LABELS

    def test_sign_076_not_arithmetic(self, families_json):
        assert families_json.get("076") not in _ARITHMETIC_LABELS

    def test_no_numeric_family_value(self, families_json):
        for code, fam in families_json.items():
            assert not fam.isdigit(), f"Sign {code!r} has numeric family {fam!r}"

    # ── (c) P007 and P012 sign pair assignments ───────────────────────────────

    def test_p012_sign_678_is_zoomorphic(self, families_json):
        assert families_json.get("678") == "zoomorphic", (
            f"Sign 678 should be 'zoomorphic', got {families_json.get('678')!r}"
        )

    def test_p012_sign_076_is_botanical(self, families_json):
        assert families_json.get("076") == "botanical", (
            f"Sign 076 should be 'botanical', got {families_json.get('076')!r}"
        )

    def test_p012_678_076_cross_family_boundary(self, families_json):
        """678→076 still crosses a family boundary under the corrected taxonomy."""
        fam_678 = families_json.get("678")
        fam_076 = families_json.get("076")
        assert fam_678 not in ("unknown", None)
        assert fam_076 not in ("unknown", None)
        assert fam_678 != fam_076, (
            "P012 Family-Crossing finding would be invalidated: "
            f"678={fam_678!r} == 076={fam_076!r}"
        )

    def test_sign_600_is_zoomorphic(self, families_json):
        assert families_json.get("600") == "zoomorphic"

    def test_200_series_all_anthropomorphic(self, families_json):
        two_hundreds = [c for c in _ACTIVE_SIGNS if c.startswith("2") and len(c) == 3]
        for code in two_hundreds:
            assert families_json.get(code) == "anthropomorphic", (
                f"200-series sign {code!r} should be 'anthropomorphic', got {families_json.get(code)!r}"
            )

    def test_600_series_all_zoomorphic(self, families_json):
        six_hundreds = [c for c in _ACTIVE_SIGNS if c.startswith("6") and len(c) == 3]
        for code in six_hundreds:
            assert families_json.get(code) == "zoomorphic", (
                f"600-series sign {code!r} should be 'zoomorphic', got {families_json.get(code)!r}"
            )

    def test_700_series_all_zoomorphic(self, families_json):
        seven_hundreds = [c for c in _ACTIVE_SIGNS if c.startswith("7") and len(c) == 3]
        for code in seven_hundreds:
            assert families_json.get(code) == "zoomorphic", (
                f"700-series sign {code!r} should be 'zoomorphic', got {families_json.get(code)!r}"
            )

    # ── Lookup robustness ─────────────────────────────────────────────────────

    def test_padded_unpadded_resolve_same(self, families_json):
        cat = SignCatalog({"076": "H076"}, {}, {}, families_json)
        assert cat.get_barthel_family("076") == cat.get_barthel_family("76") == "botanical"

    def test_allograph_suffix_stripped(self, families_json):
        cat = SignCatalog({"380": "H380", "380a": "H380a"}, {}, {}, families_json)
        assert cat.get_barthel_family("380") == "anthropomorphic"
        assert cat.get_barthel_family("380a") == "anthropomorphic"

    def test_missing_code_returns_unknown(self, families_json):
        cat = SignCatalog({}, {}, {}, families_json)
        assert cat.get_barthel_family("999999") == "unknown"

    def test_none_families_returns_unknown(self):
        cat = SignCatalog({}, {}, {}, None)
        assert cat.get_barthel_family("076") == "unknown"
        assert cat.get_barthel_family("600") == "unknown"
