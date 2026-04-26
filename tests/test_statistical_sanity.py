from __future__ import annotations

import numpy as np
import pandas as pd

from mfa.stats.correlation import estimate_feature_associations


def test_statistical_sanity_positive_signal() -> None:
    rng = np.random.default_rng(0)
    x = np.arange(30)
    y = x + rng.normal(scale=2.0, size=30)
    analysis_table = pd.DataFrame(
        {
            "dataset": [f"d{i}" for i in range(len(x))],
            "log_n": x,
            "delta_norm": y,
        }
    )

    result = estimate_feature_associations(
        analysis_table,
        table_name="synthetic",
        feature_columns=["log_n"],
        min_n=30,
        bootstrap_repeats=25,
    ).iloc[0]

    assert result["spearman_r"] > 0.8
    assert result["p_value"] < 0.05


def test_statistical_sanity_null_signal() -> None:
    rng = np.random.default_rng(7)
    x = rng.normal(size=40)
    y = rng.normal(size=40)
    analysis_table = pd.DataFrame(
        {
            "dataset": [f"d{i}" for i in range(len(x))],
            "log_n": x,
            "delta_norm": y,
        }
    )

    result = estimate_feature_associations(
        analysis_table,
        table_name="synthetic",
        feature_columns=["log_n"],
        min_n=30,
        bootstrap_repeats=25,
    ).iloc[0]

    assert abs(result["spearman_r"]) < 0.35
    assert result["p_value"] > 0.05
