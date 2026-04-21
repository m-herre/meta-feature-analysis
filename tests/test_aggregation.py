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


def test_irregularity_composite_is_z_scored_across_datasets_not_splits(analysis_config) -> None:
    """Regression: the composite must z-score across datasets, not split rows.

    Without this, a dataset with more splits dominates the global mean/std
    through its repeated rows, and the composite value becomes a function of
    fold layout rather than dataset properties (flagged by the Codex
    adversarial review).

    Construction: dataset_a has 9 (repeat, fold) splits with identical
    component values; dataset_b has 1 split. Under split-level z-scoring the
    mean is (9*val_a + val_b)/10 and the two datasets' z-scores are not
    mirror images. Under dataset-level z-scoring (2 rows) the z-scores are
    exactly +1 and -1 (or the paper-weight equivalent), symmetric.
    """
    # Two components, one per dataset, deterministic values.
    split_rows = []
    for repeat in range(3):
        for fold in range(3):
            split_rows.append(("dataset_a", repeat, fold, 1.0, 1.0))
    split_rows.append(("dataset_b", 0, 0, 3.0, 3.0))
    metafeatures = pd.DataFrame(
        split_rows,
        columns=["dataset", "repeat", "fold", "irreg_min_cov_eig", "irreg_std_skew"],
    )

    gap_rows = []
    for dataset, repeat, fold, *_ in split_rows:
        gap_rows.append(
            [dataset, repeat, fold, "nn_vs_gbdt", "nn", "gbdt", "NN", "GBDT", "positive",
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

    analysis = build_analysis_table(
        gap_table,
        metafeatures,
        unit=analysis_config.analysis.unit,
        irregularity_components=("irreg_min_cov_eig", "irreg_std_skew"),
    )
    # Two dataset rows. Under dataset-level z-scoring with ddof=0 on 2 rows,
    # the two z-scores are exactly -1 and +1 for each component. min_cov_eig
    # gets weight -0.33, std_skew gets +0.23. Compose per paper weights, then
    # renormalize by (|w| used / |w| total) — here all selected components are
    # available on every row, so the factor is 1.
    # dataset_a has the smaller value (1.0) → z = -1 on both cols.
    # dataset_b has the larger value (3.0) → z = +1 on both cols.
    dataset_a = analysis[analysis["dataset"] == "dataset_a"].iloc[0]
    dataset_b = analysis[analysis["dataset"] == "dataset_b"].iloc[0]
    expected_a = (-0.33) * (-1.0) + (0.23) * (-1.0)
    expected_b = (-0.33) * (+1.0) + (0.23) * (+1.0)
    assert np.isclose(dataset_a["irregularity"], expected_a, atol=1e-10)
    assert np.isclose(dataset_b["irregularity"], expected_b, atol=1e-10)
    # Symmetry property: dataset-level z-scoring implies irregularity_a == -irregularity_b
    # when there are only two datasets. Split-level z-scoring would not satisfy this.
    assert np.isclose(dataset_a["irregularity"], -dataset_b["irregularity"], atol=1e-10)


def test_fold_level_does_not_duplicate_rows_across_comparisons(analysis_config) -> None:
    """Regression: per-dataset irregularity lookup must be unique by dataset.

    When more than one comparison uses the same dataset, the dataset-level
    grouping emits one row per (dataset, comparison). A naive broadcast on
    ``dataset`` alone would merge each fold row against every comparison
    row and silently explode the fold-level count — inflating
    ``n_observations`` for correlations and bootstraps. Flagged by the
    Codex adversarial review.
    """
    metafeatures = pd.DataFrame(
        [
            ["dataset_a", 0, 0, 1.0, 1.0],
            ["dataset_a", 0, 1, 1.0, 1.0],
            ["dataset_b", 0, 0, 3.0, 3.0],
            ["dataset_b", 0, 1, 3.0, 3.0],
        ],
        columns=["dataset", "repeat", "fold", "irreg_min_cov_eig", "irreg_std_skew"],
    )
    gap_rows = []
    for comparison in ("cmp_one", "cmp_two"):
        for dataset, repeat, fold in [
            ("dataset_a", 0, 0), ("dataset_a", 0, 1),
            ("dataset_b", 0, 0), ("dataset_b", 0, 1),
        ]:
            gap_rows.append(
                [dataset, repeat, fold, comparison, "nn", "gbdt", "NN", "GBDT", "positive",
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

    from mfa.types import AnalysisUnit

    fold_level = build_analysis_table(
        gap_table,
        metafeatures,
        unit=AnalysisUnit.FOLD,
        irregularity_components=("irreg_min_cov_eig", "irreg_std_skew"),
    )
    # Expected: one fold row per (dataset, repeat, fold, comparison). 4 splits
    # x 2 comparisons = 8 rows. A duplicating merge would give 16.
    assert len(fold_level) == 8
    # And the dataset-level symmetry property must still hold under the
    # unique-dataset lookup (not weighted by comparison count).
    dataset_level = build_analysis_table(
        gap_table,
        metafeatures,
        unit=AnalysisUnit.DATASET,
        irregularity_components=("irreg_min_cov_eig", "irreg_std_skew"),
    )
    # Two datasets x two comparisons = 4 dataset-level rows; per dataset the
    # irregularity is identical across comparisons (function of dataset only).
    irregularity_by_dataset = (
        dataset_level.groupby("dataset")["irregularity"].nunique().to_dict()
    )
    assert irregularity_by_dataset == {"dataset_a": 1, "dataset_b": 1}
    a_val = dataset_level.loc[dataset_level["dataset"] == "dataset_a", "irregularity"].iloc[0]
    b_val = dataset_level.loc[dataset_level["dataset"] == "dataset_b", "irregularity"].iloc[0]
    assert np.isclose(a_val, -b_val, atol=1e-10)
