"""
tests.test_data
===============

Smoke tests for hackingrongo.data.corpus and hackingrongo.data.dataset.

All tests are self-contained: they build minimal in-memory JSON
structures and temporary directories via pytest's ``tmp_path`` fixture,
requiring no actual rongorongo data files or glyph images.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from hackingrongo.data.corpus import (
    GlyphToken,
    TabletRecord,
    assign_cluster,
    assign_cluster_probability,
    get_corpus_token_sequence,
    load_corpus,
    load_tablet_metadata,
    make_train_val_split,
    split_by_stratum,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def cfg():
    """Minimal OmegaConf config matching the schema expected by corpus.py."""
    return OmegaConf.create(
        {
            "seed": 42,
            "corpus": {
                "temporal_model": {
                    "type": "two_cluster",
                    "clusters": {
                        "pre_contact": {
                            "tablets": ["D"],
                            "date_hpd_95_CE": [1493, 1509],
                            "confidence": "high",
                        },
                        "post_contact": {
                            "tablets": ["B", "C", "O", "Q"],
                            "date_hpd_95_CE": [1800, 1870],
                            "confidence": "medium",
                        },
                        "excluded_from_temporal_analysis": {
                            "tablets": ["A"],
                            "reason": "European wood provenance",
                        },
                    },
                    "cluster_labels": {
                        "pre_contact": "pre_contact",
                        "post_contact": "post_contact",
                        "unknown": "unknown",
                        "excluded": "excluded",
                    },
                    "undated_assignment": {
                        "method": "probabilistic_classifier",
                        "classifier": "gaussian_naive_bayes",
                        "min_confidence_to_assign": 0.65,
                    },
                    "robustness_threshold": 0.10,
                },
                "train_split_ratio": 0.80,
                "min_tablet_tokens": 2,
            },
            "paths": {
                # Relative to project_root; in tests we override via tmp_path.
                "corpus_dir": "corpus",
                "tablets_json": "tablets.json",
            },
        }
    )


def _write_tablet_json(directory: Path, tablet_id: str, glyphs: list[dict]) -> None:
    """Write a minimal corpus JSON file for a single tablet."""
    data = {"tablet_id": tablet_id, "glyphs": glyphs}
    (directory / f"{tablet_id}.json").write_text(
        json.dumps(data), encoding="utf-8"
    )


def _write_metadata_json(path: Path, entries: dict) -> None:
    """Write a minimal tablets.json metadata file."""
    path.write_text(json.dumps(entries), encoding="utf-8")


def _make_glyphs(n: int, tablet_id: str = "T") -> list[dict]:
    """Generate ``n`` synthetic glyph records."""
    return [
        {"position": i + 1, "barthel_code": f"{(i % 120) + 1:03d}"}
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# assign_cluster
# ---------------------------------------------------------------------------


class TestAssignCluster:
    def test_pre_contact_tablet(self, cfg):
        assert assign_cluster("D", cfg) == "pre_contact"

    def test_post_contact_tablet(self, cfg):
        assert assign_cluster("B", cfg) == "post_contact"

    def test_post_contact_tablet_q(self, cfg):
        assert assign_cluster("Q", cfg) == "post_contact"

    def test_excluded_tablet(self, cfg):
        assert assign_cluster("A", cfg) == "excluded"

    def test_undated_tablet_is_unknown(self, cfg):
        # Any tablet not listed in any cluster → unknown.
        assert assign_cluster("H", cfg) == "unknown"
        assert assign_cluster("Z", cfg) == "unknown"


class TestAssignClusterProbability:
    def test_known_pre_contact_deterministic(self, cfg):
        probs = assign_cluster_probability("D", cfg)
        assert probs["pre_contact"] == 1.0
        assert probs["post_contact"] == 0.0

    def test_known_post_contact_deterministic(self, cfg):
        probs = assign_cluster_probability("Q", cfg)
        assert probs["post_contact"] == 1.0
        assert probs["pre_contact"] == 0.0

    def test_excluded_maps_to_unknown_probability(self, cfg):
        probs = assign_cluster_probability("A", cfg)
        assert probs["unknown"] == 1.0

    def test_undated_returns_empirical_prior(self, cfg):
        # 1 pre_contact, 4 post_contact → prior 0.2 / 0.8
        probs = assign_cluster_probability("H", cfg)
        assert abs(probs["pre_contact"] - 0.2) < 1e-9
        assert abs(probs["post_contact"] - 0.8) < 1e-9
        assert probs["unknown"] == 0.0

    def test_probabilities_sum_to_one(self, cfg):
        for tid in ["D", "Q", "A", "H"]:
            probs = assign_cluster_probability(tid, cfg)
            assert abs(sum(probs.values()) - 1.0) < 1e-9


# ---------------------------------------------------------------------------
# load_tablet_metadata
# ---------------------------------------------------------------------------


class TestLoadTabletMetadata:
    def test_loads_valid_json(self, tmp_path):
        entries = {
            "K": {"radiocarbon_date_min": 1400, "radiocarbon_date_max": 1600,
                  "wood_species": "toromiro", "institution": "BPBM"},
        }
        meta_file = tmp_path / "tablets.json"
        _write_metadata_json(meta_file, entries)

        result = load_tablet_metadata(meta_file)
        assert "K" in result
        assert result["K"]["wood_species"] == "toromiro"

    def test_raises_on_missing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_tablet_metadata(tmp_path / "nonexistent.json")


# ---------------------------------------------------------------------------
# load_corpus — smoke test
# ---------------------------------------------------------------------------


class TestLoadCorpus:
    def _scaffold(self, tmp_path: Path) -> tuple[dict, Path, Path]:
        """Build a minimal valid corpus directory under tmp_path."""
        corpus_dir = tmp_path / "corpus"
        corpus_dir.mkdir()
        meta_path = tmp_path / "tablets.json"

        # Use known-cluster tablet IDs: D=pre_contact, B=post_contact, Q=post_contact.
        # G is undated → unknown cluster.
        metadata = {
            "D": {"radiocarbon_date_min": 1493, "radiocarbon_date_max": 1509,
                  "wood_species": "Podocarpus sp.", "institution": "MH"},
            "B": {"radiocarbon_date_min": 1800, "radiocarbon_date_max": 1860,
                  "wood_species": "Thespesia populnea", "institution": "MH"},
            "G": {"radiocarbon_date_min": 1650, "radiocarbon_date_max": 1870,
                  "wood_species": "Thespesia populnea", "institution": "MNHN"},
        }
        _write_metadata_json(meta_path, metadata)
        for tid, n_glyphs in [("D", 10), ("B", 8), ("G", 5)]:
            _write_tablet_json(corpus_dir, tid, _make_glyphs(n_glyphs, tid))

        return metadata, corpus_dir, meta_path

    def test_returns_all_valid_tablets(self, tmp_path, cfg):
        self._scaffold(tmp_path)
        records = load_corpus(cfg, tmp_path)
        assert len(records) == 3

    def test_stratum_assignment(self, tmp_path, cfg):
        self._scaffold(tmp_path)
        records = load_corpus(cfg, tmp_path)
        by_id = {r.tablet_id: r for r in records}
        assert by_id["D"].stratum == "pre_contact"
        assert by_id["B"].stratum == "post_contact"
        assert by_id["G"].stratum == "unknown"

    def test_tokens_loaded_and_sorted(self, tmp_path, cfg):
        self._scaffold(tmp_path)
        records = load_corpus(cfg, tmp_path)
        by_id = {r.tablet_id: r for r in records}
        positions = [t.position for t in by_id["D"].tokens]
        assert positions == sorted(positions)
        assert len(positions) == 10

    def test_sorted_chronologically(self, tmp_path, cfg):
        self._scaffold(tmp_path)
        records = load_corpus(cfg, tmp_path)
        midpoints = [r.date_midpoint for r in records]
        assert midpoints == sorted(midpoints)

    def test_short_tablet_filtered(self, tmp_path, cfg):
        """Tablets with fewer tokens than min_tablet_tokens are excluded."""
        corpus_dir = tmp_path / "corpus"
        corpus_dir.mkdir()
        meta_path = tmp_path / "tablets.json"
        # "G" has only 1 token; cfg.corpus.min_tablet_tokens == 2.
        metadata = {
            "G": {"radiocarbon_date_min": 1650, "radiocarbon_date_max": 1870,
                  "wood_species": "toromiro", "institution": "BPBM"},
        }
        _write_metadata_json(meta_path, metadata)
        _write_tablet_json(corpus_dir, "G", _make_glyphs(1, "G"))

        records = load_corpus(cfg, tmp_path)
        assert records == []

    def test_corpus_file_without_metadata_skipped(self, tmp_path, cfg):
        """A corpus file with no metadata entry is skipped gracefully."""
        corpus_dir = tmp_path / "corpus"
        corpus_dir.mkdir()
        meta_path = tmp_path / "tablets.json"
        _write_metadata_json(meta_path, {})  # empty metadata
        _write_tablet_json(corpus_dir, "X", _make_glyphs(10, "X"))

        records = load_corpus(cfg, tmp_path)
        assert records == []

    def test_raises_on_missing_corpus_dir(self, tmp_path, cfg):
        meta_path = tmp_path / "tablets.json"
        _write_metadata_json(meta_path, {})
        with pytest.raises(FileNotFoundError):
            load_corpus(cfg, tmp_path)  # corpus/ subdir not created


# ---------------------------------------------------------------------------
# split_by_stratum
# ---------------------------------------------------------------------------


class TestSplitByStratum:
    def _make_record(self, tablet_id: str, stratum: str) -> TabletRecord:
        return TabletRecord(
            tablet_id=tablet_id,
            stratum=stratum,
            date_midpoint=1500.0,
            tokens=[GlyphToken(1, "001", tablet_id, stratum)],
        )

    def test_correct_partition(self):
        records = [
            self._make_record("A", "pre_contact"),
            self._make_record("B", "post_contact"),
            self._make_record("C", "pre_contact"),
        ]
        by_stratum = split_by_stratum(records)
        assert set(by_stratum.keys()) == {"pre_contact", "post_contact"}
        assert len(by_stratum["pre_contact"]) == 2
        assert len(by_stratum["post_contact"]) == 1

    def test_empty_input(self):
        assert split_by_stratum([]) == {}

    def test_order_preserved_within_stratum(self):
        records = [
            self._make_record("A", "pre_contact"),
            self._make_record("B", "pre_contact"),
            self._make_record("C", "pre_contact"),
        ]
        result = split_by_stratum(records)
        assert [r.tablet_id for r in result["pre_contact"]] == ["A", "B", "C"]


# ---------------------------------------------------------------------------
# make_train_val_split
# ---------------------------------------------------------------------------


class TestMakeTrainValSplit:
    def _make_records(self, n: int, stratum: str) -> list[TabletRecord]:
        return [
            TabletRecord(
                tablet_id=f"{stratum}_{i}",
                stratum=stratum,
                date_midpoint=1500.0,
                tokens=[GlyphToken(1, "001", f"{stratum}_{i}", stratum)],
            )
            for i in range(n)
        ]

    def test_no_data_loss(self, cfg):
        records = self._make_records(10, "pre_contact")
        rng = np.random.default_rng(cfg.seed)
        train, val = make_train_val_split(records, cfg, rng)
        assert len(train) + len(val) == 10

    def test_approximate_ratio(self, cfg):
        records = self._make_records(20, "pre_contact")
        rng = np.random.default_rng(cfg.seed)
        train, val = make_train_val_split(records, cfg, rng)
        # floor(20 * 0.80) = 16 train, 4 val
        assert len(train) == math.floor(20 * float(cfg.corpus.train_split_ratio))

    def test_single_tablet_goes_to_train(self, cfg):
        records = self._make_records(1, "pre_contact")
        rng = np.random.default_rng(cfg.seed)
        train, val = make_train_val_split(records, cfg, rng)
        assert len(train) == 1
        assert len(val) == 0

    def test_reproducible_with_same_seed(self, cfg):
        records = self._make_records(20, "pre_contact")
        rng1 = np.random.default_rng(cfg.seed)
        train1, _ = make_train_val_split(records, cfg, rng1)
        rng2 = np.random.default_rng(cfg.seed)
        train2, _ = make_train_val_split(records, cfg, rng2)
        assert [r.tablet_id for r in train1] == [r.tablet_id for r in train2]

    def test_per_stratum_split(self, cfg):
        """Both clusters must be represented in both train and val."""
        pre = self._make_records(10, "pre_contact")
        post = self._make_records(10, "post_contact")
        rng = np.random.default_rng(cfg.seed)
        train, val = make_train_val_split(pre + post, cfg, rng)
        train_strata = {r.stratum for r in train}
        val_strata = {r.stratum for r in val}
        assert "pre_contact" in train_strata and "post_contact" in train_strata
        assert "pre_contact" in val_strata and "post_contact" in val_strata


# ---------------------------------------------------------------------------
# get_corpus_token_sequence
# ---------------------------------------------------------------------------


class TestGetCorpusTokenSequence:
    def test_flattens_in_order(self):
        t1 = TabletRecord("A", "pre_contact", 1400.0,
                          tokens=[GlyphToken(1, "001", "A", "pre_contact"),
                                  GlyphToken(2, "002", "A", "pre_contact")])
        t2 = TabletRecord("B", "post_contact", 1830.0,
                          tokens=[GlyphToken(1, "003", "B", "post_contact")])
        seq = get_corpus_token_sequence([t1, t2])
        assert len(seq) == 3
        assert seq[0].barthel_code == "001"
        assert seq[2].barthel_code == "003"

    def test_empty_input(self):
        assert get_corpus_token_sequence([]) == []


# ===========================================================================
# dataset.py tests
# ===========================================================================

from hackingrongo.data.dataset import (  # noqa: E402
    GlyphImageDataset,
    GlyphSequenceDataset,
    SiamesePairDataset,
    build_sign_groups,
    load_allograph_catalog,
)


# ---------------------------------------------------------------------------
# Shared dataset cfg fixture (extends corpus cfg)
# ---------------------------------------------------------------------------


@pytest.fixture()
def full_cfg():
    """Full OmegaConf config covering both corpus and dataset fields."""
    return OmegaConf.create(
        {
            "seed": 42,
            "corpus": {
                "temporal_model": {
                    "type": "two_cluster",
                    "clusters": {
                        "pre_contact": {
                            "tablets": ["D"],
                            "date_hpd_95_CE": [1493, 1509],
                            "confidence": "high",
                        },
                        "post_contact": {
                            "tablets": ["B", "C", "O", "Q"],
                            "date_hpd_95_CE": [1800, 1870],
                            "confidence": "medium",
                        },
                        "excluded_from_temporal_analysis": {
                            "tablets": ["A"],
                            "reason": "European wood provenance",
                        },
                    },
                    "cluster_labels": {
                        "pre_contact": "pre_contact",
                        "post_contact": "post_contact",
                        "unknown": "unknown",
                        "excluded": "excluded",
                    },
                    "undated_assignment": {
                        "method": "probabilistic_classifier",
                        "classifier": "gaussian_naive_bayes",
                        "min_confidence_to_assign": 0.65,
                    },
                    "robustness_threshold": 0.10,
                },
                "train_split_ratio": 0.80,
                "min_tablet_tokens": 2,
            },
            "paths": {
                "corpus_dir": "corpus",
                "tablets_json": "tablets.json",
                "allographs_json": "catalog/allographs.json",
            },
            "glyph": {
                "image_size": 16,         # tiny for fast tests
                "image_channels": 1,
                "filename_pattern": "{tablet_id}_{position}_{barthel_code}.png",
                "augmentation": {
                    "use_augmentation": False,
                    "random_rotation_degrees": 10.0,
                    "random_affine_translate": [0.05, 0.05],
                    "random_affine_scale": [0.9, 1.1],
                    "gaussian_noise_std": 0.02,
                    "elastic_transform_alpha": 4.0,
                    "elastic_transform_sigma": 0.08,
                },
            },
            "zone_a": {
                "sequence_model": {
                    "context_window": 3,
                },
                "siamese": {
                    "pairs_per_epoch": 8,
                    "same_sign_ratio": 0.5,
                    "hard_negative_mining": False,
                    "margin": 1.0,
                    "hard_negative_margin_factor": 0.5,
                },
            },
        }
    )


def _make_tokens(n: int, tablet_id: str = "T", stratum: str = "pre_contact") -> list[GlyphToken]:
    return [
        GlyphToken(i + 1, f"{(i % 5) + 1:03d}", tablet_id, stratum)
        for i in range(n)
    ]


def _write_token_pngs(glyphs_dir: Path, tokens: list) -> None:
    """Create a minimal 16×16 white PNG for each token at the expected path."""
    from PIL import Image as _PIL_Image
    for tok in tokens:
        fname = f"{tok.tablet_id}_{tok.position}_{tok.barthel_code}.png"
        img = _PIL_Image.new("L", (16, 16), color=255)
        img.save(glyphs_dir / fname)


# ---------------------------------------------------------------------------
# load_allograph_catalog
# ---------------------------------------------------------------------------


class TestLoadAllographCatalog:
    def test_loads_valid_catalog(self, tmp_path):
        catalog = {"001": "001", "001a": "001", "002": "002"}
        p = tmp_path / "allographs.json"
        p.write_text(json.dumps(catalog), encoding="utf-8")
        result = load_allograph_catalog(p)
        assert result["001a"] == "001"

    def test_raises_on_missing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_allograph_catalog(tmp_path / "missing.json")


# ---------------------------------------------------------------------------
# build_sign_groups
# ---------------------------------------------------------------------------


class TestBuildSignGroups:
    def test_groups_variants_under_canonical(self):
        catalog = {"001": "001", "001a": "001", "002": "002"}
        groups = build_sign_groups(catalog)
        assert sorted(groups["001"]) == ["001", "001a"]
        assert groups["002"] == ["002"]

    def test_no_duplicates_in_group(self):
        catalog = {"001": "001"}
        groups = build_sign_groups(catalog)
        assert groups["001"].count("001") == 1


# ---------------------------------------------------------------------------
# GlyphImageDataset
# ---------------------------------------------------------------------------


class TestGlyphImageDataset:
    @staticmethod
    def _write_token_pngs(glyphs_dir: Path, tokens: list) -> None:
        """Create a minimal 16×16 white PNG for each token at the expected path."""
        from PIL import Image as _PIL_Image
        for tok in tokens:
            fname = f"{tok.tablet_id}_{tok.position}_{tok.barthel_code}.png"
            img = _PIL_Image.new("L", (16, 16), color=255)
            img.save(glyphs_dir / fname)

    def test_len_matches_tokens(self, full_cfg, tmp_path):
        tokens = _make_tokens(6)
        self._write_token_pngs(tmp_path, tokens)
        ds = GlyphImageDataset(tokens, tmp_path, full_cfg, training=False)
        assert len(ds) == 6

    def test_vocab_built_correctly(self, full_cfg, tmp_path):
        tokens = _make_tokens(6)
        self._write_token_pngs(tmp_path, tokens)
        ds = GlyphImageDataset(tokens, tmp_path, full_cfg, training=False)
        # Tokens use codes "001"–"005" (6 tokens, 5 unique codes).
        assert len(ds.vocab) == 5
        assert ds.barthel_to_id[ds.vocab[0]] == 0

    def test_missing_image_tokens_excluded(self, full_cfg, tmp_path):
        """Tokens whose image cannot be resolved are excluded from the dataset,
        not substituted with zeros.  Vocabulary still covers all original codes."""
        tokens = _make_tokens(4)
        # Write images for only the first 2 tokens.
        self._write_token_pngs(tmp_path, tokens[:2])
        ds = GlyphImageDataset(tokens, tmp_path, full_cfg, training=False)
        assert len(ds) == 2
        # Vocab built from all original tokens — stable ID mapping for sequences.
        assert len(ds.vocab) == len({t.barthel_code for t in tokens})

    def test_sample_keys(self, full_cfg, tmp_path):
        tokens = _make_tokens(2)
        self._write_token_pngs(tmp_path, tokens)
        ds = GlyphImageDataset(tokens, tmp_path, full_cfg, training=False)
        sample = ds[0]
        assert {"image", "token_id", "barthel_code", "tablet_id",
                "position", "stratum"} <= sample.keys()

    def test_token_id_in_vocab_range(self, full_cfg, tmp_path):
        tokens = _make_tokens(4)
        self._write_token_pngs(tmp_path, tokens)
        ds = GlyphImageDataset(tokens, tmp_path, full_cfg, training=False)
        for i in range(len(ds)):
            assert 0 <= ds[i]["token_id"] < len(ds.vocab)


# ---------------------------------------------------------------------------
# GlyphSequenceDataset
# ---------------------------------------------------------------------------


class TestGlyphSequenceDataset:
    def _make_tablet(self, n: int, tablet_id: str = "T") -> TabletRecord:
        tokens = _make_tokens(n, tablet_id)
        return TabletRecord(tablet_id, "pre_contact", 1400.0, tokens)

    def test_window_count(self, full_cfg):
        # context_window=3; tablet with 7 tokens → 7-3=4 windows
        tablet = self._make_tablet(7)
        barthel_to_id = {f"{i+1:03d}": i for i in range(5)}
        ds = GlyphSequenceDataset([tablet], barthel_to_id, full_cfg)
        assert len(ds) == 4

    def test_sample_shapes(self, full_cfg):
        tablet = self._make_tablet(7)
        barthel_to_id = {f"{i+1:03d}": i for i in range(5)}
        ds = GlyphSequenceDataset([tablet], barthel_to_id, full_cfg)
        sample = ds[0]
        assert sample["context"].shape == (3,)
        assert sample["target"].ndim == 0

    def test_tablet_too_short_produces_no_samples(self, full_cfg):
        # context_window=3; tablet with 3 tokens needs at least 4 → 0 windows
        tablet = self._make_tablet(3)
        barthel_to_id = {f"{i+1:03d}": i for i in range(5)}
        ds = GlyphSequenceDataset([tablet], barthel_to_id, full_cfg)
        assert len(ds) == 0

    def test_windows_do_not_cross_tablet_boundaries(self, full_cfg):
        t1 = self._make_tablet(5, "A")
        t2 = self._make_tablet(5, "B")
        barthel_to_id = {f"{i+1:03d}": i for i in range(5)}
        ds = GlyphSequenceDataset([t1, t2], barthel_to_id, full_cfg)
        # Each tablet: 5-3=2 windows → 4 total
        assert len(ds) == 4

    def test_token_ids_are_long(self, full_cfg):
        tablet = self._make_tablet(6)
        barthel_to_id = {f"{i+1:03d}": i for i in range(5)}
        ds = GlyphSequenceDataset([tablet], barthel_to_id, full_cfg)
        sample = ds[0]
        assert sample["context"].dtype == torch.long
        assert sample["target"].dtype == torch.long


# ---------------------------------------------------------------------------
# SiamesePairDataset
# ---------------------------------------------------------------------------


class TestSiamesePairDataset:
    def _make_image_ds(self, full_cfg, tmp_path, n: int = 10) -> GlyphImageDataset:
        tokens = _make_tokens(n)
        _write_token_pngs(tmp_path, tokens)
        return GlyphImageDataset(tokens, tmp_path, full_cfg, training=False)

    def _make_catalog(self) -> dict[str, str]:
        # codes 001–003 → group "001"; codes 004–005 → group "004"
        return {
            "001": "001", "002": "001", "003": "001",
            "004": "004", "005": "004",
        }

    def test_len_equals_pairs_per_epoch(self, full_cfg, tmp_path):
        img_ds = self._make_image_ds(full_cfg, tmp_path)
        ds = SiamesePairDataset(img_ds, self._make_catalog(), full_cfg, seed=42)
        assert len(ds) == int(full_cfg.zone_a.siamese.pairs_per_epoch)

    def test_sample_keys(self, full_cfg, tmp_path):
        img_ds = self._make_image_ds(full_cfg, tmp_path)
        ds = SiamesePairDataset(img_ds, self._make_catalog(), full_cfg, seed=42)
        sample = ds[0]
        assert {"anchor", "pair", "label", "anchor_idx",
                "pair_idx"} <= sample.keys()

    def test_label_is_float_tensor(self, full_cfg, tmp_path):
        img_ds = self._make_image_ds(full_cfg, tmp_path)
        ds = SiamesePairDataset(img_ds, self._make_catalog(), full_cfg, seed=42)
        sample = ds[0]
        assert sample["label"].dtype == torch.float32

    def test_label_is_zero_or_one(self, full_cfg, tmp_path):
        img_ds = self._make_image_ds(full_cfg, tmp_path)
        ds = SiamesePairDataset(img_ds, self._make_catalog(), full_cfg, seed=0)
        labels = {float(ds[i]["label"]) for i in range(len(ds))}
        assert labels <= {0.0, 1.0}

    def test_update_hard_negatives_wrong_shape_raises(self, full_cfg, tmp_path):
        img_ds = self._make_image_ds(full_cfg, tmp_path, n=10)
        ds = SiamesePairDataset(img_ds, self._make_catalog(), full_cfg, seed=42)
        bad_embeddings = np.zeros((5, 32))  # wrong N
        with pytest.raises(ValueError):
            ds.update_hard_negatives(bad_embeddings)

    def test_update_hard_negatives_accepted(self, full_cfg, tmp_path):
        img_ds = self._make_image_ds(full_cfg, tmp_path, n=10)
        ds = SiamesePairDataset(img_ds, self._make_catalog(), full_cfg, seed=42)
        good_embeddings = np.zeros((10, 32))
        ds.update_hard_negatives(good_embeddings)  # must not raise
        assert ds._embeddings is not None

    def test_reproducible_with_same_seed(self, full_cfg, tmp_path):
        img_ds = self._make_image_ds(full_cfg, tmp_path)
        catalog = self._make_catalog()
        ds1 = SiamesePairDataset(img_ds, catalog, full_cfg, seed=7)
        ds2 = SiamesePairDataset(img_ds, catalog, full_cfg, seed=7)
        labels1 = [float(ds1[i]["label"]) for i in range(8)]
        labels2 = [float(ds2[i]["label"]) for i in range(8)]
        assert labels1 == labels2
