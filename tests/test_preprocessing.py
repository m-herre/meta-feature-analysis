from __future__ import annotations

import numpy as np
import pandas as pd

from mfa.preprocessing import preprocess_analysis_table, reduce_redundant_features


def test_preprocess_analysis_table_drops_high_missing_and_near_constant_features() -> None:
    table = pd.DataFrame(
        {
            "dataset": [f"d{i}" for i in range(10)],
            "delta_norm": np.arange(10, dtype=float),
            "kept": np.arange(10, dtype=float),
            "high_missing": [np.nan] * 3 + list(range(7)),
            "near_constant": [1.0] * 10,
        }
    )

    processed, report = preprocess_analysis_table(
        table,
        ["kept", "high_missing", "near_constant"],
        table_name="synthetic",
        context_columns=["dataset", "delta_norm"],
    )

    assert processed.columns.tolist() == ["dataset", "delta_norm", "kept"]
    assert set(report["feature"]) == {"high_missing", "near_constant"}


def test_preprocess_analysis_table_can_use_reference_table_for_feature_mask() -> None:
    table = pd.DataFrame(
        {
            "dataset": [f"d{i}" for i in range(4)],
            "delta_norm": np.arange(4, dtype=float),
            "locally_complete_globally_missing": np.arange(4, dtype=float),
            "locally_variable_globally_constant": np.arange(4, dtype=float),
            "kept": np.arange(4, dtype=float),
        }
    )
    reference_table = pd.DataFrame(
        {
            "dataset": [f"d{i}" for i in range(10)],
            "locally_complete_globally_missing": [np.nan] * 3 + list(range(7)),
            "locally_variable_globally_constant": [1.0] * 10,
            "kept": np.arange(10, dtype=float),
        }
    )

    processed, report = preprocess_analysis_table(
        table,
        [
            "locally_complete_globally_missing",
            "locally_variable_globally_constant",
            "kept",
        ],
        table_name="synthetic",
        max_feature_missingness=0.2,
        context_columns=["dataset", "delta_norm"],
        filter_table=reference_table,
    )

    assert processed.columns.tolist() == ["dataset", "delta_norm", "kept"]
    assert set(report["feature"]) == {
        "locally_complete_globally_missing",
        "locally_variable_globally_constant",
    }


def test_reduce_redundant_features_collapses_high_spearman_pair() -> None:
    table = pd.DataFrame(
        {
            "dataset": [f"d{i}" for i in range(10)],
            "delta_norm": np.arange(10, dtype=float),
            "feature_a": np.arange(10, dtype=float),
            "feature_b": np.arange(10, dtype=float) * 2,
            "feature_c": np.sin(np.arange(10, dtype=float)),
        }
    )

    reduced, report = reduce_redundant_features(
        table,
        table_name="synthetic",
        context_columns=["dataset", "delta_norm"],
    )

    assert "feature_a" in reduced.columns
    assert "feature_b" not in reduced.columns
    assert "feature_c" in reduced.columns
    assert report.loc[0, "dropped_feature"] == "feature_b"
    assert report.loc[0, "kept_feature"] == "feature_a"
