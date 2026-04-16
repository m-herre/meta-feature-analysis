from __future__ import annotations

from collections.abc import Sequence
from concurrent.futures import as_completed

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, rankdata, spearmanr

from ..parallel import get_executor, resolve_n_jobs
from ..types import ComparisonSpec, CorrelationMethod, CorrelationResult

_BOOTSTRAP_CHUNK_SIZE = 1000

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
    expected_direction: str | None = None,
    random_state: int = 0,
) -> tuple[float | None, float | None]:
    """Bootstrap a percentile confidence interval for the correlation coefficient.

    When *expected_direction* is set the CI is one-sided to match the
    one-sided p-value:
      - "positive" → [alpha-percentile, None]  (lower bound only)
      - "negative" → [None, (1-alpha)-percentile]  (upper bound only)
    When *expected_direction* is None the CI is the usual two-sided interval.
    """
    if len(x) < 3:
        return None, None
    rng = np.random.default_rng(random_state)
    x_values = x.to_numpy()
    y_values = y.to_numpy()
    n = len(x_values)
    boot_stats = np.empty(n_bootstrap, dtype=np.float64)
    for start in range(0, n_bootstrap, _BOOTSTRAP_CHUNK_SIZE):
        end = min(start + _BOOTSTRAP_CHUNK_SIZE, n_bootstrap)
        size = end - start
        indices = rng.integers(0, n, size=(size, n))
        x_boot = x_values[indices]
        y_boot = y_values[indices]
        if method == CorrelationMethod.SPEARMAN:
            # Average-rank ties match scipy.stats.spearmanr; positional ranks
            # (e.g. argsort-of-argsort) would bias CIs under bootstrap duplicates.
            x_boot = rankdata(x_boot, method="average", axis=1)
            y_boot = rankdata(y_boot, method="average", axis=1)
        else:
            x_boot = x_boot.astype(np.float64, copy=False)
            y_boot = y_boot.astype(np.float64, copy=False)
        x_mean = x_boot.mean(axis=1, keepdims=True)
        y_mean = y_boot.mean(axis=1, keepdims=True)
        x_centered = x_boot - x_mean
        y_centered = y_boot - y_mean
        numerator = (x_centered * y_centered).sum(axis=1)
        denominator = np.sqrt((x_centered**2).sum(axis=1) * (y_centered**2).sum(axis=1))
        with np.errstate(divide="ignore", invalid="ignore"):
            boot_stats[start:end] = numerator / denominator
    finite_stats = boot_stats[np.isfinite(boot_stats)]
    if finite_stats.size == 0:
        return None, None
    alpha = 1 - confidence_level
    if expected_direction == "positive":
        return float(np.quantile(finite_stats, alpha)), None
    if expected_direction == "negative":
        return None, float(np.quantile(finite_stats, 1 - alpha))
    return (
        float(np.quantile(finite_stats, alpha / 2)),
        float(np.quantile(finite_stats, 1 - (alpha / 2))),
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


def _correlate_one(
    x_values: np.ndarray,
    y_values: np.ndarray,
    comparison_name: str,
    predictor: str,
    target: str,
    method_value: str,
    expected_direction: str | None,
    confidence_interval: bool,
    ci_bootstrap_samples: int,
    ci_confidence_level: float,
    random_state: int,
) -> CorrelationResult:
    """Run one univariate correlation test. Top-level function for pickle compatibility."""
    method = CorrelationMethod(method_value)
    x = pd.Series(x_values)
    y = pd.Series(y_values)
    statistic, p_value = _compute_correlation(x, y, method)
    p_value = _one_sided_p_value(p_value, statistic, expected_direction)
    ci_lower, ci_upper = (None, None)
    if confidence_interval and len(x) >= 3:
        ci_lower, ci_upper = bootstrap_correlation_ci(
            x,
            y,
            method=method,
            n_bootstrap=ci_bootstrap_samples,
            confidence_level=ci_confidence_level,
            expected_direction=expected_direction,
            random_state=random_state,
        )
    return CorrelationResult(
        comparison_name=comparison_name,
        predictor=predictor,
        target=target,
        statistic=statistic,
        p_value=p_value,
        n_observations=int(len(x)),
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        direction_confirmed=_direction_matches(statistic, expected_direction),
    )


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
    n_jobs: int = 1,
    backend: str = "process",
) -> list[CorrelationResult]:
    """Run univariate correlations for every predictor within each comparison."""
    predictor_names = infer_predictors(analysis_table, target=target, predictors=predictors)
    resolved_n_jobs = resolve_n_jobs(n_jobs)

    # Pre-slice data for each (comparison, predictor) pair
    tasks: list[tuple[np.ndarray, np.ndarray, str, str, str | None]] = []
    for comparison in comparisons:
        subset_by_comparison = analysis_table[analysis_table["comparison_name"] == comparison.name].copy()
        for predictor in predictor_names:
            subset = subset_by_comparison[[predictor, target]].dropna()
            tasks.append(
                (
                    subset[predictor].to_numpy(),
                    subset[target].to_numpy(),
                    comparison.name,
                    predictor,
                    comparison.expected_direction,
                )
            )

    if resolved_n_jobs <= 1:
        results = []
        for x_values, y_values, comp_name, pred, expected_dir in tasks:
            results.append(
                _correlate_one(
                    x_values,
                    y_values,
                    comp_name,
                    pred,
                    target,
                    method.value,
                    expected_dir,
                    confidence_interval,
                    ci_bootstrap_samples,
                    ci_confidence_level,
                    random_state,
                )
            )
        return results

    executor = get_executor(backend, max_workers=resolved_n_jobs)
    try:
        futures = {}
        for idx, (x_values, y_values, comp_name, pred, expected_dir) in enumerate(tasks):
            future = executor.submit(
                _correlate_one,
                x_values,
                y_values,
                comp_name,
                pred,
                target,
                method.value,
                expected_dir,
                confidence_interval,
                ci_bootstrap_samples,
                ci_confidence_level,
                random_state,
            )
            futures[future] = idx

        # Collect results in original order
        results_by_idx: dict[int, CorrelationResult] = {}
        for future in as_completed(futures):
            results_by_idx[futures[future]] = future.result()
        return [results_by_idx[i] for i in range(len(tasks))]
    finally:
        executor.shutdown(wait=True)
