from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("sklearn")

from mfa.predictive import XGBRegressor, run_predictive_meta_modeling


def test_run_predictive_meta_modeling_returns_metrics_and_baseline() -> None:
    n = 30
    x = np.arange(n, dtype=float)
    table = pd.DataFrame(
        {
            "dataset": [f"d{i}" for i in range(n)],
            "delta_norm": x / n,
            "feature_a": x,
            "feature_b": np.sin(x),
            "feature_c": np.cos(x),
        }
    )

    metrics, predictions, coefficient_summary, control_report, guard = run_predictive_meta_modeling(
        table,
        table_name="synthetic",
        include_classification_controls=False,
        context_columns=["dataset", "delta_norm"],
    )

    expected_models = {
        "mean_baseline",
        "ridge_meta_features",
        "decision_tree_meta_features",
    }
    if XGBRegressor is not None:
        expected_models.add("xgboost_meta_features")

    assert set(metrics["model"]) == expected_models
    assert "mean_baseline" in set(metrics["model"])
    assert predictions["model"].nunique() == len(expected_models)
    assert not coefficient_summary.empty
    assert not control_report.empty
    assert guard.loc[0, "n_feature_predictors"] == 3
