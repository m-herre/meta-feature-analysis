from __future__ import annotations

from mfa.stats.correction import apply_fdr_correction
from mfa.types import CorrelationResult, FDRMethod


def test_apply_fdr_correction_bh() -> None:
    results = [
        CorrelationResult("c1", "x1", "y", 0.1, 0.01, 10),
        CorrelationResult("c1", "x2", "y", 0.2, 0.02, 10),
        CorrelationResult("c1", "x3", "y", 0.3, 0.2, 10),
    ]
    corrected = apply_fdr_correction(results, method=FDRMethod.BH, alpha=0.05)
    assert corrected is not None
    assert corrected.rejected == (True, True, False)


def test_apply_fdr_correction_is_scoped_per_comparison() -> None:
    results = [
        CorrelationResult("c1", "x1", "y", 0.1, 0.01, 10),
        CorrelationResult("c1", "x2", "y", 0.2, 0.02, 10),
        CorrelationResult("c2", "x1", "y", 0.1, 0.01, 10),
        CorrelationResult("c2", "x2", "y", 0.2, 0.20, 10),
    ]
    corrected = apply_fdr_correction(results, method=FDRMethod.BH, alpha=0.05)
    assert corrected is not None
    assert corrected.adjusted_p_values == (0.02, 0.02, 0.02, 0.2)
    assert corrected.rejected == (True, True, True, False)
