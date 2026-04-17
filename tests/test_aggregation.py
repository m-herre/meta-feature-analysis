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
