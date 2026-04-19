from __future__ import annotations

import numpy as np
import pandas as pd

from mfa.aggregation import build_analysis_table
from mfa.gaps.pairwise import GAP_TABLE_COLUMNS


def test_build_analysis_table_dataset_level(analysis_config, synthetic_metafeatures) -> None:
    gap_table = pd.DataFrame(
        [
            [
                "dataset_a",
                0,
                0,
                "nn_vs_gbdt",
                "nn",
                "gbdt",
                "NN",
                "GBDT",
                "positive",
                "nn1",
                0.2,
                1.0,
                "gbdt1",
                0.1,
                0.0,
                0.1,
                1.0,
            ],
            [
                "dataset_a",
                1,
                0,
                "nn_vs_gbdt",
                "nn",
                "gbdt",
                "NN",
                "GBDT",
                "positive",
                "nn1",
                0.1,
                0.5,
                "gbdt1",
                0.2,
                0.75,
                -0.1,
                -0.25,
            ],
            [
                "dataset_b",
                0,
                0,
                "nn_vs_gbdt",
                "nn",
                "gbdt",
                "NN",
                "GBDT",
                "positive",
                "nn1",
                0.3,
                0.8,
                "gbdt1",
                0.2,
                0.2,
                0.1,
                0.6,
            ],
        ],
        columns=[
            "dataset",
            "repeat",
            "fold",
            "comparison_name",
            "group_a_name",
            "group_b_name",
            "group_a_label",
            "group_b_label",
            "expected_direction",
            "best_a_method",
            "best_a_error",
            "best_a_norm_error",
            "best_b_method",
            "best_b_error",
            "best_b_norm_error",
            "delta_raw",
            "delta_norm",
        ],
    )
    analysis = build_analysis_table(gap_table, synthetic_metafeatures, unit=analysis_config.analysis.unit)
    dataset_a = analysis[analysis["dataset"] == "dataset_a"].iloc[0]
    assert dataset_a["n_splits"] == 2
    assert np.isclose(dataset_a["delta_norm"], 0.375)
    assert np.isclose(dataset_a["log_n"], (2.0 + 2.04) / 2)


def test_build_analysis_table_handles_empty_gap_table(analysis_config, synthetic_metafeatures) -> None:
    gap_table = pd.DataFrame(columns=GAP_TABLE_COLUMNS)
    analysis = build_analysis_table(gap_table, synthetic_metafeatures, unit=analysis_config.analysis.unit)
    assert analysis.empty
    assert {"comparison_name", "delta_norm", "log_n", "expected_direction"}.issubset(analysis.columns)


def test_build_analysis_table_propagates_nan_in_metafeature_splits(analysis_config) -> None:
    """Any NaN in a predictor within a dataset must NaN the dataset-level value.

    Otherwise the dataset gets averaged over only the successful splits while
    delta_norm is averaged over all splits — the effective n per predictor
    then drifts silently and correlations use different subsets per feature.
    """
    metafeatures = pd.DataFrame(
        [
            ["dataset_a", 0, 0, 100, 10, 2.0, 10.0, 0.2, 0.05, 0.5, 0.5],
            ["dataset_a", 1, 0, 110, 10, 2.04, 11.0, 0.2, 0.04, 0.6, np.nan],
            ["dataset_b", 0, 0, 80, 8, 1.9, 10.0, 0.1, 0.02, 0.3, 0.3],
            ["dataset_b", 1, 0, 85, 8, 1.93, 10.6, 0.1, 0.02, 0.35, 0.4],
        ],
        columns=[
            "dataset", "repeat", "fold",
            "n", "d", "log_n", "n_over_d",
            "cat_fraction", "missing_fraction", "irregularity",
            "pymfe__stat_a",
        ],
    )
    gap_rows = []
    for dataset, repeat in [("dataset_a", 0), ("dataset_a", 1), ("dataset_b", 0), ("dataset_b", 1)]:
        gap_rows.append(
            [dataset, repeat, 0, "nn_vs_gbdt", "nn", "gbdt", "NN", "GBDT", "positive",
             "nn1", 0.2, 1.0, "gbdt1", 0.1, 0.0, 0.1, 1.0]
        )
    gap_table = pd.DataFrame(
        gap_rows,
        columns=[
            "dataset", "repeat", "fold",
            "comparison_name", "group_a_name", "group_b_name",
            "group_a_label", "group_b_label", "expected_direction",
            "best_a_method", "best_a_error", "best_a_norm_error",
            "best_b_method", "best_b_error", "best_b_norm_error",
            "delta_raw", "delta_norm",
        ],
    )

    analysis = build_analysis_table(gap_table, metafeatures, unit=analysis_config.analysis.unit)

    dataset_a = analysis[analysis["dataset"] == "dataset_a"].iloc[0]
    dataset_b = analysis[analysis["dataset"] == "dataset_b"].iloc[0]
    # dataset_a has one NaN split for pymfe__stat_a -> dataset-level is NaN.
    assert np.isnan(dataset_a["pymfe__stat_a"])
    # dataset_b has no NaN splits -> dataset-level is the mean.
    assert np.isclose(dataset_b["pymfe__stat_a"], 0.35)
    # Delta aggregation is unaffected: averaged over all splits for both datasets.
    assert np.isclose(dataset_a["delta_norm"], 1.0)
    assert np.isclose(dataset_b["delta_norm"], 1.0)
    # Other (complete) meta-features aggregate normally.
    assert np.isclose(dataset_a["log_n"], (2.0 + 2.04) / 2)
