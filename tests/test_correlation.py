from __future__ import annotations

import numpy as np
import pandas as pd

from mfa.stats.correlation import correlate_all


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

