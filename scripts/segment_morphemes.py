"""CLI entry-point for Zellig Harris morpheme segmentation.

Delegates entirely to hackingrongo.zone_b.morpheme_segmentation.main().

Usage
-----
    python scripts/segment_morphemes.py \\
        --corpus-dir data/corpus \\
        --output     outputs/morpheme_segments.json

    python scripts/segment_morphemes.py \\
        --corpus-dir data/corpus \\
        --threshold  1.5 \\
        --json
"""
from hackingrongo.zone_b.morpheme_segmentation import main

if __name__ == "__main__":
    main()
