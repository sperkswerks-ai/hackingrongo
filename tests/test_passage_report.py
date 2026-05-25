"""Test passage alignment report generation."""

import json
from pathlib import Path
from hackingrongo.results.passage_report import PassageReportGenerator
from hackingrongo.data.passage_alignment import (
    PassageAlignment, PassageAttestation, DiachronicChange, AlignmentCell
)


def test_report_generation_simple():
    """Test basic report generation."""
    # Create sample data
    attestations = [
        {
            "tablet": "A",
            "tablet_name": "Tahua",
            "stratum": "undated",
            "date_range": "~1500 CE",
            "sequence": ["098", "007"],
            "edit_distance": 0,
            "alignment": [
                {"position": 0, "query_code": "098", "corpus_code": "098", "match_type": "match", "is_significant": False},
                {"position": 1, "query_code": "007", "corpus_code": "007", "match_type": "match", "is_significant": False},
            ]
        },
        {
            "tablet": "B",
            "tablet_name": "Aruku-Kurenga",
            "stratum": "post_contact",
            "date_range": "~1800-1870 CE",
            "sequence": ["098", "007"],
            "edit_distance": 0,
            "alignment": [
                {"position": 0, "query_code": "098", "corpus_code": "098", "match_type": "match", "is_significant": False},
                {"position": 1, "query_code": "007", "corpus_code": "007", "match_type": "match", "is_significant": False},
            ]
        },
    ]

    passage = {
        "passage_id": "TEST_P001",
        "canonical_sequence": ["098", "007"],
        "canonical_tablet": "A",
        "canonical_stratum": "undated",
        "attestations": attestations,
        "diachronic_changes": [],
        "interest_score": 0.85,
    }

    generator = PassageReportGenerator()
    html = generator.render_passage(passage)

    assert "TEST_P001" in html
    assert "098" in html
    assert "007" in html
    assert "Tahua" in html
    assert "Post-contact" in html
    print("✓ Basic report generation works")


def test_report_with_diachronic_changes():
    """Test report with diachronic changes."""
    attestations = [
        {
            "tablet": "D",
            "tablet_name": "Hanga Te Pau",
            "stratum": "pre_contact",
            "date_range": "~1493-1509 CE",
            "sequence": ["004", "430", "022"],
            "edit_distance": 0,
            "alignment": []
        },
        {
            "tablet": "E",
            "tablet_name": "Ai-One",
            "stratum": "post_contact",
            "date_range": "~1800-1870 CE",
            "sequence": ["004", "430", "022"],
            "edit_distance": 0,
            "alignment": []
        },
        {
            "tablet": "S",
            "tablet_name": "Pakeha-Motoro",
            "stratum": "post_contact",
            "date_range": "~1800-1870 CE",
            "sequence": ["004", "430", "022y"],
            "edit_distance": 1,
            "alignment": []
        },
    ]

    changes = [
        {
            "position": 2,
            "pre_contact_sign": "022",
            "post_contact_sign": "022y",
            "change_type": "substitution",
            "is_known_allograph": True,
            "crosses_barthel_family": False,
            "n_tablets_consistent": 1,
            "is_holy_grail_candidate": False,
        },
        {
            "position": 1,
            "pre_contact_sign": "430",
            "post_contact_sign": "430y",
            "change_type": "substitution",
            "is_known_allograph": False,
            "crosses_barthel_family": True,
            "n_tablets_consistent": 2,
            "is_holy_grail_candidate": True,
        },
    ]

    passage = {
        "passage_id": "TEST_P002_HOLY_GRAIL",
        "canonical_sequence": ["004", "430", "022"],
        "canonical_tablet": "D",
        "canonical_stratum": "pre_contact",
        "attestations": attestations,
        "diachronic_changes": changes,
        "interest_score": 0.95,
    }

    generator = PassageReportGenerator()
    html = generator.render_passage(passage)

    assert "Holy Grail" in html
    assert "Family-Crossing" in html
    assert "022y" in html
    print("✓ Report with diachronic changes works")


def test_generate_full_report(tmp_path):
    """Test full report generation from JSON."""
    # Create sample JSON file
    passages = {
        "passages": [
            {
                "passage_id": "P001",
                "canonical_sequence": ["098", "007"],
                "canonical_tablet": "A",
                "canonical_stratum": "undated",
                "attestations": [
                    {
                        "tablet": "A",
                        "tablet_name": "Tahua",
                        "stratum": "undated",
                        "date_range": "~1500 CE",
                        "sequence": ["098", "007"],
                        "edit_distance": 0,
                        "alignment": []
                    }
                ],
                "diachronic_changes": [],
                "interest_score": 0.85,
            },
            {
                "passage_id": "P002",
                "canonical_sequence": ["430", "076"],
                "canonical_tablet": "B",
                "canonical_stratum": "post_contact",
                "attestations": [
                    {
                        "tablet": "B",
                        "tablet_name": "Aruku-Kurenga",
                        "stratum": "post_contact",
                        "date_range": "~1800-1870 CE",
                        "sequence": ["430", "076"],
                        "edit_distance": 0,
                        "alignment": []
                    }
                ],
                "diachronic_changes": [],
                "interest_score": 0.75,
            },
        ]
    }

    json_file = tmp_path / "passages.json"
    json_file.write_text(json.dumps(passages))

    output_dir = tmp_path / "reports"

    generator = PassageReportGenerator()
    generator.generate_report(
        passages_json=json_file,
        output_dir=output_dir,
        filter_interest_score=0.7,
        individual_files=True,
    )

    # Check files were created
    assert (output_dir / "P001.html").exists()
    assert (output_dir / "P002.html").exists()
    assert (output_dir / "index.html").exists()

    # Check content
    p001_html = (output_dir / "P001.html").read_text()
    assert "P001" in p001_html
    assert "098" in p001_html

    index_html = (output_dir / "index.html").read_text()
    assert "P001" in index_html
    assert "P002" in index_html

    print("✓ Full report generation works")


if __name__ == "__main__":
    import tempfile

    test_report_generation_simple()
    test_report_with_diachronic_changes()

    with tempfile.TemporaryDirectory() as tmpdir:
        test_generate_full_report(Path(tmpdir))

    print("\n✅ All tests passed!")
