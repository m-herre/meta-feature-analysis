from __future__ import annotations

import numpy as np
import pandas as pd

from mfa.stats.correlation import correlate_all


def test_statistical_sanity_positive_signal(analysis_config) -> None:
    rng = np.random.default_rng(0)
    x = np.arange(30)
    y = x + rng.normal(scale=2.0, size=30)
    analysis_table = pd.DataFrame(
        {
            "comparison_name": ["nn_vs_gbdt"] * len(x),
            "dataset": [f"d{i}" for i in range(len(x))],
            "log_n": x,
            "delta_norm": y,
        }
    )
    result = correlate_all(
        analysis_table,
        comparisons=analysis_config.comparisons,
        predictors=["log_n"],
        target="delta_norm",
        confidence_interval=False,
    )[0]
    assert result.statistic > 0.8
    assert result.p_value < 0.05


def test_statistical_sanity_null_signal(analysis_config) -> None:
    rng = np.random.default_rng(7)
    x = rng.normal(size=40)
    y = rng.normal(size=40)
    analysis_table = pd.DataFrame(
        {
            "comparison_name": ["nn_vs_gbdt"] * len(x),
            "dataset": [f"d{i}" for i in range(len(x))],
            "log_n": x,
            "delta_norm": y,
        }
    )
    result = correlate_all(
        analysis_table,
        comparisons=analysis_config.comparisons,
        predictors=["log_n"],
        target="delta_norm",
        confidence_interval=False,
    )[0]
    assert abs(result.statistic) < 0.35
    assert result.p_value > 0.05
