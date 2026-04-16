from __future__ import annotations

import numpy as np
from statsmodels.stats.multitest import multipletests

from ..types import CorrectionResult, FDRMethod

MULTITEST_MAP = {
    FDRMethod.BH: "fdr_bh",
    FDRMethod.HOLM: "holm",
}


def apply_fdr_correction(
    results,
    *,
    method: FDRMethod | None = FDRMethod.BH,
    alpha: float = 0.05,
) -> CorrectionResult | None:
    """Apply multiple-testing correction to a sequence of correlation results."""
    if method is None:
        return None
    results = tuple(results)
    adjusted = [np.nan] * len(results)
    rejected = [False] * len(results)
    grouped_indices: dict[str, list[int]] = {}
    for idx, result in enumerate(results):
        grouped_indices.setdefault(result.comparison_name, []).append(idx)
    for comparison_indices in grouped_indices.values():
        valid_indices = [idx for idx in comparison_indices if not np.isnan(results[idx].p_value)]
        if not valid_indices:
            continue
        valid_p_values = [results[idx].p_value for idx in valid_indices]
        reject_subset, adjusted_subset, _, _ = multipletests(valid_p_values, alpha=alpha, method=MULTITEST_MAP[method])
        for idx, adjusted_p, reject in zip(valid_indices, adjusted_subset, reject_subset, strict=False):
            adjusted[idx] = float(adjusted_p)
            rejected[idx] = bool(reject)
    return CorrectionResult(
        method=method.value,
        alpha=alpha,
        results=results,
        adjusted_p_values=tuple(adjusted),
        rejected=tuple(rejected),
    )
