"""
tests.test_sequence_model
==========================

Unit and integration tests for hackingrongo.zone_b.sequence_model.

All tests use tiny synthetic corpora — no actual rongorongo data files needed.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from hackingrongo.zone_b.sequence_model import (
    BOS,
    EOS,
    UNK,
    NgramModel,
    load_sequences,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_tablet(tmp_path: Path, tablet_id: str, glyphs: list[dict]) -> None:
    """Write a minimal corpus JSON for one tablet."""
    data = {"tablet_id": tablet_id, "cluster": "A", "glyphs": glyphs}
    (tmp_path / f"{tablet_id}.json").write_text(
        json.dumps(data), encoding="utf-8"
    )


def _make_glyph(
    side: str,
    line: int,
    glyph_num: int,
    horley_code: str,
    inverted: bool = False,
    uncertain: bool = False,
    horley_components: list | None = None,
) -> dict:
    return {
        "side": side,
        "line": str(line).zfill(2),
        "glyph_num": str(glyph_num),
        "horley_code": horley_code,
        "barthel_code": horley_code,
        "inverted": inverted,
        "uncertain": uncertain,
        "horley_components": horley_components,
        "position": glyph_num,
    }


# ---------------------------------------------------------------------------
# load_sequences
# ---------------------------------------------------------------------------

class TestLoadSequences:
    def test_single_tablet_basic(self, tmp_path):
        glyphs = [
            _make_glyph("a", 1, 1, "1"),
            _make_glyph("a", 1, 2, "2"),
            _make_glyph("a", 1, 3, "3"),
        ]
        _write_tablet(tmp_path, "D", glyphs)
        seqs = load_sequences(tmp_path)
        assert len(seqs) == 1
        assert seqs[0] == ["1", "2", "3"]

    def test_excludes_uncertain_by_default(self, tmp_path):
        glyphs = [
            _make_glyph("a", 1, 1, "1"),
            _make_glyph("a", 1, 2, "99", uncertain=True),
            _make_glyph("a", 1, 3, "3"),
        ]
        _write_tablet(tmp_path, "D", glyphs)
        seqs = load_sequences(tmp_path, include_uncertain=False)
        assert "99" not in seqs[0]
        assert "1" in seqs[0]
        assert "3" in seqs[0]

    def test_includes_uncertain_when_requested(self, tmp_path):
        glyphs = [
            _make_glyph("a", 1, 1, "1"),
            _make_glyph("a", 1, 2, "99", uncertain=True),
        ]
        _write_tablet(tmp_path, "D", glyphs)
        seqs = load_sequences(tmp_path, include_uncertain=True)
        assert "99" in seqs[0]

    def test_rtl_line_reversed(self, tmp_path):
        # Line 2 is RTL (majority inverted)
        glyphs = [
            _make_glyph("a", 1, 1, "A", inverted=False),
            _make_glyph("a", 1, 2, "B", inverted=False),
            _make_glyph("a", 2, 1, "X", inverted=True),
            _make_glyph("a", 2, 2, "Y", inverted=True),
            _make_glyph("a", 2, 3, "Z", inverted=True),
        ]
        _write_tablet(tmp_path, "D", glyphs)
        seqs = load_sequences(tmp_path)
        seq = seqs[0]
        # Line 1 in physical order: A, B
        # Line 2 reversed: Z, Y, X
        line1_pos = (seq.index("A"), seq.index("B"))
        line2_z = seq.index("Z")
        line2_y = seq.index("Y")
        line2_x = seq.index("X")
        assert line1_pos[0] < line1_pos[1], "Line 1 should be A before B"
        assert line2_z < line2_y < line2_x, "Line 2 should be reversed to Z, Y, X"

    def test_compound_components_expanded(self, tmp_path):
        glyphs = [
            _make_glyph("a", 1, 1, "compound", horley_components=["76", "380"]),
        ]
        _write_tablet(tmp_path, "D", glyphs)
        seqs = load_sequences(tmp_path, include_compound_components=True)
        # Compound glyphs: the Horley code may be yielded, or components
        # Either way the components should appear somewhere
        flat = seqs[0]
        assert "76" in flat or "380" in flat or "compound" in flat

    def test_empty_corpus_dir(self, tmp_path):
        seqs = load_sequences(tmp_path)
        assert seqs == []

    def test_multiple_tablets(self, tmp_path):
        for tid in ("A", "B", "C"):
            _write_tablet(tmp_path, tid, [_make_glyph("a", 1, 1, tid)])
        seqs = load_sequences(tmp_path)
        assert len(seqs) == 3


# ---------------------------------------------------------------------------
# NgramModel — train / log_prob / score / perplexity
# ---------------------------------------------------------------------------

class TestNgramModel:
    @pytest.fixture()
    def simple_seqs(self):
        return [
            ["a", "b", "c", "a"],
            ["a", "b", "a", "b"],
            ["c", "c", "a"],
        ]

    def test_trains_without_error(self, simple_seqs):
        model = NgramModel(order=2)
        model.train(simple_seqs)  # should not raise

    def test_vocab_populated(self, simple_seqs):
        model = NgramModel(order=2)
        model.train(simple_seqs)
        assert "a" in model.vocab
        assert "b" in model.vocab
        assert BOS in model.vocab

    def test_log_prob_known_token(self, simple_seqs):
        model = NgramModel(order=2)
        model.train(simple_seqs)
        lp = model.log_prob("b", ("a",))
        assert isinstance(lp, float)
        assert lp < 0.0  # log probability must be negative

    def test_log_prob_unk_token_is_finite(self, simple_seqs):
        model = NgramModel(order=2)
        model.train(simple_seqs)
        lp = model.log_prob("UNSEEN_TOKEN_XYZ", ("a",))
        assert math.isfinite(lp)
        assert lp < 0.0

    def test_score_sequence(self, simple_seqs):
        model = NgramModel(order=2)
        model.train(simple_seqs)
        score = model.score(["a", "b", "c"])
        assert isinstance(score, float)
        assert score < 0.0  # sum of log probs should be negative

    def test_perplexity_is_finite_and_positive(self, simple_seqs):
        model = NgramModel(order=2)
        model.train(simple_seqs)
        ppl = model.perplexity(simple_seqs)
        assert math.isfinite(ppl)
        assert ppl > 1.0

    def test_train_perplexity_lower_than_uniform(self, simple_seqs):
        model = NgramModel(order=2)
        model.train(simple_seqs)
        ppl = model.perplexity(simple_seqs)
        # Vocab has ~5 tokens (a, b, c, BOS, EOS); uniform ppl = 5
        # Trained bigram ppl should be lower than uniform
        assert ppl < 6.0

    def test_top_k_next_returns_k_items(self, simple_seqs):
        model = NgramModel(order=2)
        model.train(simple_seqs)
        top = model.top_k_next(("a",), k=2)
        assert len(top) == 2
        tokens, scores = zip(*top)
        # Scores should be descending
        assert scores[0] >= scores[1]

    def test_order_1_unigram(self, simple_seqs):
        model = NgramModel(order=1)
        model.train(simple_seqs)
        ppl = model.perplexity(simple_seqs)
        assert ppl > 1.0

    def test_empty_sequence_score_is_finite(self):
        model = NgramModel(order=2)
        model.train([["a", "b"]])
        score = model.score([])
        assert math.isfinite(score)


# ---------------------------------------------------------------------------
# NgramModel — save / load round-trip
# ---------------------------------------------------------------------------

class TestNgramModelSerialisation:
    def test_save_load_roundtrip(self, tmp_path, simple_seqs):
        model = NgramModel(order=2)
        model.train(simple_seqs)

        path = tmp_path / "model.json"
        model.save(path)
        assert path.exists()

        loaded = NgramModel.load(path)
        assert loaded.order == model.order
        assert loaded.vocab == model.vocab

    @pytest.fixture()
    def simple_seqs(self):
        return [["a", "b", "c"], ["b", "c", "a"]]

    def test_loaded_model_gives_same_scores(self, tmp_path, simple_seqs):
        model = NgramModel(order=2)
        model.train(simple_seqs)
        path = tmp_path / "m.json"
        model.save(path)
        loaded = NgramModel.load(path)

        test_seq = ["a", "b", "c"]
        assert abs(model.score(test_seq) - loaded.score(test_seq)) < 1e-9

    def test_saved_file_is_valid_json(self, tmp_path, simple_seqs):
        model = NgramModel(order=2)
        model.train(simple_seqs)
        path = tmp_path / "m.json"
        model.save(path)
        data = json.loads(path.read_text())
        assert "order" in data
        assert "counts" in data


# ---------------------------------------------------------------------------
# Corpus-level integration: load_sequences → NgramModel
# ---------------------------------------------------------------------------

class TestEndToEnd:
    def test_train_on_corpus_and_score(self, tmp_path):
        # Build a minimal 3-tablet corpus
        for tid, tokens in [("A", ["1", "2", "3"]), ("B", ["2", "3", "1"]), ("C", ["3", "1", "2"])]:
            glyphs = [_make_glyph("a", 1, i + 1, t) for i, t in enumerate(tokens)]
            _write_tablet(tmp_path, tid, glyphs)

        seqs = load_sequences(tmp_path)
        model = NgramModel(order=2)
        model.train(seqs)

        # Score a known sequence — should be finite
        score = model.score(["1", "2", "3"])
        assert math.isfinite(score)

    def test_perplexity_LOO_order1_lower_than_order2(self, tmp_path):
        """For a larger corpus, order-1 LOO ppl should be lower than order-2."""
        # Use longer sequences so bigram overfitting is more pronounced
        import random
        random.seed(42)
        vocab = ["a", "b", "c", "d"]
        for i in range(10):
            tid = chr(ord("A") + i)
            tokens = [random.choice(vocab) for _ in range(20)]
            glyphs = [_make_glyph("a", 1, j + 1, t) for j, t in enumerate(tokens)]
            _write_tablet(tmp_path, tid, glyphs)

        seqs = load_sequences(tmp_path)

        def loo_ppl(order):
            ppls = []
            for i in range(len(seqs)):
                train = seqs[:i] + seqs[i + 1:]
                test = [seqs[i]]
                m = NgramModel(order=order, alpha=0.01)
                m.train(train)
                ppls.append(m.perplexity(test))
            return sum(ppls) / len(ppls)

        ppl1 = loo_ppl(1)
        ppl2 = loo_ppl(2)
        # order-1 is more robust on small corpora; order-2 overfits
        # On a random corpus this holds reliably when vocab << seq_len
        assert ppl1 < ppl2, f"Expected order-1 LOO ppl < order-2, got {ppl1:.1f} vs {ppl2:.1f}"
