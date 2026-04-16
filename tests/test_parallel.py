from __future__ import annotations

import os

import numpy as np
import pandas as pd

from mfa.parallel import resolve_n_jobs
from mfa.stats.correlation import bootstrap_correlation_ci, correlate_all
from mfa.types import CorrelationMethod


def test_resolve_n_jobs_positive() -> None:
    assert resolve_n_jobs(4) == 4
    assert resolve_n_jobs(1) == 1


def test_resolve_n_jobs_zero_returns_sequential() -> None:
    assert resolve_n_jobs(0) == (os.cpu_count() or 1)


def test_resolve_n_jobs_negative_returns_cpu_count() -> None:
    result = resolve_n_jobs(-1)
    assert result == (os.cpu_count() or 1)
    assert result >= 1


def test_correlate_all_parallel_matches_sequential(analysis_config) -> None:
    """Parallel and sequential correlate_all must produce identical results."""
    rng = np.random.default_rng(42)
    n = 30
    analysis_table = pd.DataFrame(
        {
            "comparison_name": ["nn_vs_gbdt"] * n,
            "dataset": [f"d{i}" for i in range(n)],
            "log_n": rng.standard_normal(n),
            "n_over_d": rng.standard_normal(n),
            "cat_fraction": rng.uniform(0, 1, n),
            "delta_norm": rng.standard_normal(n),
        }
    )
    common_kwargs = dict(
        comparisons=analysis_config.comparisons,
        predictors=["log_n", "n_over_d", "cat_fraction"],
        target="delta_norm",
        confidence_interval=True,
        ci_bootstrap_samples=500,
    )
    results_seq = correlate_all(analysis_table, n_jobs=1, **common_kwargs)
    results_par = correlate_all(analysis_table, n_jobs=2, backend="process", **common_kwargs)

    assert len(results_seq) == len(results_par)
    for seq, par in zip(results_seq, results_par, strict=True):
        assert seq.comparison_name == par.comparison_name
        assert seq.predictor == par.predictor
        assert seq.n_observations == par.n_observations
        assert np.isclose(seq.statistic, par.statistic, equal_nan=True)
        assert np.isclose(seq.p_value, par.p_value, equal_nan=True)
        if seq.ci_lower is not None:
            assert np.isclose(seq.ci_lower, par.ci_lower, atol=1e-10)
        if seq.ci_upper is not None:
            assert np.isclose(seq.ci_upper, par.ci_upper, atol=1e-10)


def test_bootstrap_vectorized_spearman_perfect() -> None:
    """Vectorized bootstrap on perfectly correlated data should yield CI near 1.0."""
    x = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    y = pd.Series([10.0, 20.0, 30.0, 40.0, 50.0])
    ci_lower, ci_upper = bootstrap_correlation_ci(
        x,
        y,
        method=CorrelationMethod.SPEARMAN,
        n_bootstrap=1000,
        confidence_level=0.95,
    )
    assert ci_lower is not None
    assert ci_upper is not None
    assert ci_lower > 0.8
    assert ci_upper >= ci_lower


def test_bootstrap_vectorized_pearson_perfect() -> None:
    """Vectorized bootstrap on perfectly correlated data should yield CI near 1.0."""
    x = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    y = pd.Series([10.0, 20.0, 30.0, 40.0, 50.0])
    ci_lower, ci_upper = bootstrap_correlation_ci(
        x,
        y,
        method=CorrelationMethod.PEARSON,
        n_bootstrap=1000,
        confidence_level=0.95,
    )
    assert ci_lower is not None
    assert ci_upper is not None
    assert ci_lower > 0.8
    assert ci_upper >= ci_lower
