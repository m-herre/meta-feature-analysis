from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from ..types import ComparisonSpec, CorrelationMethod, CorrelationResult

EXCLUDED_PREDICTOR_COLUMNS = {
    "dataset",
    "comparison_name",
    "group_a_name",
    "group_b_name",
    "group_a_label",
    "group_b_label",
    "n_splits",
    "delta_raw",
    "delta_raw_std",
    "delta_raw_sem",
    "delta_norm_std",
    "delta_norm_sem",
}


def _compute_correlation(x: pd.Series, y: pd.Series, method: CorrelationMethod) -> tuple[float, float]:
    if len(x) < 3:
        return np.nan, np.nan
    if method == CorrelationMethod.SPEARMAN:
        statistic, p_value = spearmanr(x, y)
    else:
        statistic, p_value = pearsonr(x, y)
    return float(statistic), float(p_value)


def _direction_matches(statistic: float, expected_direction: str | None) -> bool | None:
    if expected_direction is None or pd.isna(statistic):
        return None
    if expected_direction == "positive":
        return statistic > 0
    return statistic < 0


def _one_sided_p_value(p_two_sided: float, statistic: float, expected_direction: str | None) -> float:
    if expected_direction is None or pd.isna(p_two_sided):
        return p_two_sided
    direction_matches = _direction_matches(statistic, expected_direction)
    if direction_matches:
        return float(p_two_sided / 2)
    return float(1 - (p_two_sided / 2))


def bootstrap_correlation_ci(
    x: pd.Series,
    y: pd.Series,
    *,
    method: CorrelationMethod,
    n_bootstrap: int,
    confidence_level: float,
    random_state: int = 0,
) -> tuple[float | None, float | None]:
    """Bootstrap a percentile confidence interval for the correlation coefficient."""
    if len(x) < 3:
        return None, None
    rng = np.random.default_rng(random_state)
    stats: list[float] = []
    x_values = x.to_numpy()
    y_values = y.to_numpy()
    for _ in range(n_bootstrap):
        sample_idx = rng.integers(0, len(x_values), len(x_values))
        statistic, _ = _compute_correlation(pd.Series(x_values[sample_idx]), pd.Series(y_values[sample_idx]), method)
        if np.isfinite(statistic):
            stats.append(statistic)
    if not stats:
        return None, None
    alpha = 1 - confidence_level
    return (
        float(np.quantile(stats, alpha / 2)),
        float(np.quantile(stats, 1 - (alpha / 2))),
    )


def infer_predictors(
    analysis_table: pd.DataFrame,
    *,
    target: str,
    predictors: Sequence[str] | None = None,
) -> list[str]:
    """Infer predictor columns from the analysis table."""
    if predictors is not None:
        return list(predictors)
    inferred = []
    for column in analysis_table.columns:
        if column in EXCLUDED_PREDICTOR_COLUMNS or column == target:
            continue
        if column.startswith("best_"):
            continue
        if pd.api.types.is_numeric_dtype(analysis_table[column]):
            inferred.append(column)
    return inferred


def correlate_all(
    analysis_table: pd.DataFrame,
    *,
    comparisons: Sequence[ComparisonSpec],
    predictors: Sequence[str] | None = None,
    target: str = "delta_norm",
    method: CorrelationMethod = CorrelationMethod.SPEARMAN,
    confidence_interval: bool = True,
    ci_bootstrap_samples: int = 10_000,
    ci_confidence_level: float = 0.95,
    random_state: int = 0,
) -> list[CorrelationResult]:
    """Run univariate correlations for every predictor within each comparison."""
    predictor_names = infer_predictors(analysis_table, target=target, predictors=predictors)
    results: list[CorrelationResult] = []
    for comparison in comparisons:
        subset_by_comparison = analysis_table[analysis_table["comparison_name"] == comparison.name].copy()
        for predictor in predictor_names:
            subset = subset_by_comparison[[predictor, target]].dropna()
            statistic, p_value = _compute_correlation(subset[predictor], subset[target], method)
            p_value = _one_sided_p_value(p_value, statistic, comparison.expected_direction)
            ci_lower, ci_upper = (None, None)
            if confidence_interval and len(subset) >= 3:
                ci_lower, ci_upper = bootstrap_correlation_ci(
                    subset[predictor],
                    subset[target],
                    method=method,
                    n_bootstrap=ci_bootstrap_samples,
                    confidence_level=ci_confidence_level,
                    random_state=random_state,
                )
            results.append(
                CorrelationResult(
                    comparison_name=comparison.name,
                    predictor=predictor,
                    target=target,
                    statistic=statistic,
                    p_value=p_value,
                    n_observations=int(len(subset)),
                    ci_lower=ci_lower,
                    ci_upper=ci_upper,
                    direction_confirmed=_direction_matches(statistic, comparison.expected_direction),
                )
            )
    return results

