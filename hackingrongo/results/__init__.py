"""hackingrongo.results — Decipherment hypothesis schema and report generation."""

from hackingrongo.results.astronomical_report import (  # noqa: F401
    build_astronomical_report,
    save_astronomical_report,
)
from hackingrongo.results.compound_report import (  # noqa: F401
    build_compound_report,
    save_compound_report,
)
from hackingrongo.results.decipherment_report import (  # noqa: F401
    build_decipherment_report,
    save_decipherment_report,
)
from hackingrongo.results.divergence_report import (  # noqa: F401
    build_divergence_report,
    save_divergence_report,
)
from hackingrongo.results.report import (  # noqa: F401
    generate_report,
    save_report,
)
