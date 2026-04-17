from __future__ import annotations

import numpy as np
import pandas as pd

from mfa.gaps.pairwise import GAP_TABLE_COLUMNS, compute_pairwise_gaps, pick_best_in_group
from mfa.types import ComparisonSpec, GroupDef


def test_pick_best_in_group_uses_deterministic_tie_break(synthetic_results) -> None:
    tied = synthetic_results[synthetic_results["fold"] == 3].copy()
    tied["repeat"] = 1
    tied["fold_in_repeat"] = 0
    tied["norm_error"] = tied["metric_error"]
    best = pick_best_in_group(tied, config_types={"NN_A", "NN_B"}, prefix="a")
    assert best["best_a_method"].item() == "nn_alpha"


def test_compute_pairwise_gaps_delta_sign(analysis_config, synthetic_results) -> None:
    gaps = compute_pairwise_gaps(
        synthetic_results,
        analysis_config.comparisons,
        error_column=analysis_config.analysis.error_column,
    )
    first_split = gaps[(gaps["dataset"] == "dataset_a") & (gaps["repeat"] == 0) & (gaps["fold"] == 0)].iloc[0]
    assert first_split["best_a_method"] == "nn_beta"
    assert first_split["best_b_method"] == "gbdt_alpha"
    assert np.isclose(first_split["delta_raw"], 0.10)
    assert first_split["delta_norm"] > 0


def test_compute_pairwise_gaps_warns_and_returns_empty_schema_when_group_missing(
    analysis_config,
    monkeypatch,
) -> None:
    warnings: list[str] = []

    def capture_warning(message: str, *args) -> None:
        warnings.append(message % args if args else message)

    monkeypatch.setattr("mfa.gaps.pairwise.logger.warning", capture_warning)
    results = pd.DataFrame(
        [
            ["dataset_a", 0, "nn_alpha", 0.20, 0.21, "NN_A", "tuned"],
            ["dataset_a", 3, "nn_beta", 0.25, 0.26, "NN_B", "tuned"],
        ],
        columns=[
            "dataset",
            "fold",
            "method",
            "metric_error",
            "metric_error_val",
            "config_type",
            "method_subtype",
        ],
    )
    gaps = compute_pairwise_gaps(
        results,
        analysis_config.comparisons,
        error_column=analysis_config.analysis.error_column,
    )
    assert gaps.empty
    assert gaps.columns.tolist() == GAP_TABLE_COLUMNS
    assert any("missing group `gbdt`" in message for message in warnings)


def test_compute_pairwise_gaps_can_select_on_val_and_evaluate_on_test(analysis_config) -> None:
    results = pd.DataFrame(
        [
            ["dataset_a", 0, "nn_alpha", 0.02, 0.30, "NN_A", "tuned"],
            ["dataset_a", 0, "nn_beta", 0.50, 0.10, "NN_B", "tuned"],
            ["dataset_a", 0, "gbdt_alpha", 0.20, 0.05, "GBDT_A", "tuned"],
            ["dataset_a", 0, "gbdt_beta", 0.01, 0.20, "GBDT_B", "tuned"],
        ],
        columns=[
            "dataset",
            "fold",
            "method",
            "metric_error",
            "metric_error_val",
            "config_type",
            "method_subtype",
        ],
    )
    gaps = compute_pairwise_gaps(
        results,
        analysis_config.comparisons,
        error_column="metric_error",
        selection_error_column="metric_error_val",
    )
    split = gaps.iloc[0]
    assert split["best_a_method"] == "nn_beta"
    assert split["best_b_method"] == "gbdt_alpha"
    assert np.isclose(split["best_a_error"], 0.50)
    assert np.isclose(split["best_b_error"], 0.20)
    assert np.isclose(split["delta_raw"], 0.30)


def test_compute_pairwise_gaps_uses_real_family_member_when_imputed_row_is_missing(analysis_config) -> None:
    results = pd.DataFrame(
        [
            ["dataset_a", 0, "nn_alpha", np.nan, np.nan, "NN_A", "tuned", True],
            ["dataset_a", 0, "nn_beta", 0.50, 0.40, "NN_B", "tuned", False],
            ["dataset_a", 0, "gbdt_alpha", 0.20, 0.10, "GBDT_A", "tuned", False],
        ],
        columns=[
            "dataset",
            "fold",
            "method",
            "metric_error",
            "metric_error_val",
            "config_type",
            "method_subtype",
            "imputed",
        ],
    )

    gaps = compute_pairwise_gaps(
        results,
        analysis_config.comparisons,
        error_column="metric_error",
        selection_error_column="metric_error_val",
    )

    split = gaps.iloc[0]
    assert split["best_a_method"] == "nn_beta"
    assert split["best_b_method"] == "gbdt_alpha"
    assert np.isclose(split["best_a_error"], 0.50)


def test_compute_pairwise_gaps_excludes_dataset_when_singleton_family_has_no_real_metric() -> None:
    comparisons = [
        ComparisonSpec(
            name="tabpfn_vs_tree",
            group_a=GroupDef(name="tabpfn", config_types=frozenset({"TABPFNV2_GPU"}), label="TabPFNv2"),
            group_b=GroupDef(name="tree", config_types=frozenset({"GBDT_A"}), label="Tree"),
        )
    ]
    results = pd.DataFrame(
        [
            ["dataset_a", 0, "tabpfn_default", np.nan, np.nan, "TABPFNV2_GPU", "default", True],
            ["dataset_a", 0, "gbdt_alpha", 0.20, 0.21, "GBDT_A", "default", False],
            ["dataset_b", 0, "tabpfn_default", 0.30, 0.31, "TABPFNV2_GPU", "default", False],
            ["dataset_b", 0, "gbdt_alpha", 0.10, 0.11, "GBDT_A", "default", False],
        ],
        columns=[
            "dataset",
            "fold",
            "method",
            "metric_error",
            "metric_error_val",
            "config_type",
            "method_subtype",
            "imputed",
        ],
    )

    gaps = compute_pairwise_gaps(
        results,
        comparisons,
        error_column="metric_error",
        selection_error_column="metric_error_val",
    )

    assert gaps["dataset"].tolist() == ["dataset_b"]
    assert gaps["best_a_method"].tolist() == ["tabpfn_default"]
