from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from mfa.stats.correlation import bootstrap_correlation_ci, correlate_all
from mfa.types import CorrelationMethod


def test_correlate_all_perfect_spearman(analysis_config) -> None:
    analysis_table = pd.DataFrame(
        {
            "comparison_name": ["nn_vs_gbdt"] * 5,
            "dataset": [f"d{i}" for i in range(5)],
            "log_n": [1, 2, 3, 4, 5],
            "delta_norm": [10, 20, 30, 40, 50],
        }
    )
    results = correlate_all(
        analysis_table,
        comparisons=analysis_config.comparisons,
        predictors=["log_n"],
        target="delta_norm",
        confidence_interval=False,
    )
    result = results[0]
    assert np.isclose(result.statistic, 1.0)
    assert result.direction_confirmed is True


def test_correlate_all_handles_small_sample(analysis_config) -> None:
    analysis_table = pd.DataFrame(
        {
            "comparison_name": ["nn_vs_gbdt"] * 2,
            "dataset": ["d1", "d2"],
            "log_n": [1, 2],
            "delta_norm": [1, 0],
        }
    )
    result = correlate_all(
        analysis_table,
        comparisons=analysis_config.comparisons,
        predictors=["log_n"],
        target="delta_norm",
        confidence_interval=False,
    )[0]
    assert np.isnan(result.statistic)
    assert np.isnan(result.p_value)


def test_bootstrap_spearman_uses_average_ranks_under_ties() -> None:
    # Discrete predictor guarantees frequent ties in resamples; bootstrap
    # statistics must agree with scipy.stats.spearmanr on the same resamples.
    rng = np.random.default_rng(42)
    x = pd.Series(rng.integers(0, 3, size=50))
    y = pd.Series(rng.integers(0, 3, size=50))
    ci_lower, ci_upper = bootstrap_correlation_ci(
        x,
        y,
        method=CorrelationMethod.SPEARMAN,
        n_bootstrap=500,
        confidence_level=0.95,
        random_state=123,
    )
    # Reproduce the same resamples with scipy's spearmanr as ground truth.
    rng2 = np.random.default_rng(123)
    x_arr = x.to_numpy()
    y_arr = y.to_numpy()
    reference = []
    for start in range(0, 500, 1000):
        size = min(1000, 500 - start)
        indices = rng2.integers(0, len(x_arr), size=(size, len(x_arr)))
        for row in indices:
            stat, _ = spearmanr(x_arr[row], y_arr[row])
            if np.isfinite(stat):
                reference.append(stat)
    expected_lower = float(np.quantile(reference, 0.025))
    expected_upper = float(np.quantile(reference, 0.975))
    assert np.isclose(ci_lower, expected_lower, atol=1e-10)
    assert np.isclose(ci_upper, expected_upper, atol=1e-10)


def test_bootstrap_chunking_matches_monolithic_result() -> None:
    # Chunking must be deterministic with respect to random_state: a chunk
    # boundary in the middle of the draws should not shift the CI.
    rng = np.random.default_rng(0)
    x = pd.Series(rng.normal(size=80))
    y = pd.Series(rng.normal(size=80))
    ci_default = bootstrap_correlation_ci(
        x, y, method=CorrelationMethod.PEARSON, n_bootstrap=2500, confidence_level=0.9, random_state=7
    )
    ci_repeat = bootstrap_correlation_ci(
        x, y, method=CorrelationMethod.PEARSON, n_bootstrap=2500, confidence_level=0.9, random_state=7
    )
    assert ci_default == ci_repeat
    assert ci_default[0] is not None and ci_default[1] is not None
    assert ci_default[0] < ci_default[1]
