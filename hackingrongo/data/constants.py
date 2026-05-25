"""
hackingrongo.data.constants
===========================

Canonical string constants for temporal stratum labels.

Centralising these prevents silent typo-bugs: using ``PRE_CONTACT``
instead of the bare literal ``"pre_contact"`` means a misspelling raises
``NameError`` at module import time rather than silently producing empty
filtered sets or unmatched dict keys at runtime.

Two stratum systems exist in this codebase and are intentionally kept
separate here:

Tablet stratum system (Ferrara et al. 2024 radiocarbon cluster model)
    Used in :mod:`~hackingrongo.data.corpus`, :mod:`~hackingrongo.zone_b.entropy`,
    and :mod:`~hackingrongo.data.passage_alignment`.

Passage-level stratum system (Horley 2021 attestation dating conventions)
    Used in :mod:`~hackingrongo.data.parallels` for :class:`~hackingrongo.data.parallels.PassageVariant`
    stratum labels, which come directly from the Horley parallels CSV.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Tablet temporal strata
# ---------------------------------------------------------------------------

PRE_CONTACT: str = "pre_contact"
"""Tablets with pre-contact radiocarbon dates (anchor: Tablet D, ~1493–1509 CE)."""

POST_CONTACT: str = "post_contact"
"""Tablets with post-contact radiocarbon dates (anchors: B, C, O, Q, ~1800–1870 CE)."""

UNKNOWN_STRATUM: str = "unknown"
"""Tablets without a radiocarbon date; cluster assignment is uncertain."""

EXCLUDED_STRATUM: str = "excluded"
"""Tablets excluded from the primary analysis (e.g. Tablet A: European wood)."""

UNDATED_STRATUM: str = "undated"
"""Parallel-passage attestation with no datable stratum assignment."""

# ---------------------------------------------------------------------------
# Passage-level strata (Horley 2021)
# ---------------------------------------------------------------------------

PASSAGE_PRE: str = "pre"
"""Horley parallel passage stratum label for pre-contact attestations."""

PASSAGE_EARLY: str = "early"
"""Horley parallel passage stratum label for early-contact attestations."""

PASSAGE_LATE: str = "late"
"""Horley parallel passage stratum label for late attestations."""
