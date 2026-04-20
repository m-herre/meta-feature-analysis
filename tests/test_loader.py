from __future__ import annotations

from dataclasses import replace

import pandas as pd
import pytest

from mfa.data.loader import load_tabarena_results


class FakeTabArenaContext:
    def __init__(self, frame: pd.DataFrame):
        self.frame = frame
        self.methods = ["artifact_a"]

    def load_hpo_results(self, method: str, holdout: bool = False) -> pd.DataFrame:
        return self.frame.copy()


def test_load_tabarena_results_preserves_requested_metric_columns(analysis_config) -> None:
    config = replace(
        analysis_config,
        analysis=replace(
            analysis_config.analysis,
            error_column="metric_error_test",
            selection_error_column="metric_error_val_alt",
        ),
    )
    frame = pd.DataFrame(
        {
            "dataset": ["dataset_a"],
            "fold": [0],
            "method": ["nn_alpha"],
            "config_type": ["NN_A"],
            "method_subtype": ["tuned"],
            "metric_error_test": [0.20],
            "metric_error_val_alt": [0.21],
            "metric_error": [0.99],
            "metric_error_val": [0.98],
            "irrelevant": ["drop-me"],
        }
    )

    loaded = load_tabarena_results(config, tabarena_context=FakeTabArenaContext(frame))

    assert loaded.columns.tolist() == [
        "dataset",
        "fold",
        "method",
        "config_type",
        "method_subtype",
        "metric_error_test",
        "metric_error_val_alt",
    ]
    assert loaded.loc[0, "metric_error_test"] == pytest.approx(0.20)
    assert loaded.loc[0, "metric_error_val_alt"] == pytest.approx(0.21)


def test_load_tabarena_results_raises_for_missing_requested_metric_column(analysis_config) -> None:
    config = replace(
        analysis_config,
        analysis=replace(
            analysis_config.analysis,
            error_column="metric_error_test",
            selection_error_column="metric_error_val_alt",
        ),
    )
    frame = pd.DataFrame(
        {
            "dataset": ["dataset_a"],
            "fold": [0],
            "method": ["nn_alpha"],
            "config_type": ["NN_A"],
            "method_subtype": ["tuned"],
            "metric_error_test": [0.20],
        }
    )

    with pytest.raises(ValueError, match="metric_error_val_alt"):
        load_tabarena_results(config, tabarena_context=FakeTabArenaContext(frame))


def test_load_tabarena_results_preserves_imputation_metadata_and_nulls_imputed_metrics(analysis_config) -> None:
    frame = pd.DataFrame(
        {
            "dataset": ["dataset_a", "dataset_a"],
            "fold": [0, 0],
            "method": ["tabpfn_default", "rf_default"],
            "config_type": ["TABPFNV2_GPU", "RF"],
            "method_subtype": ["tuned", "tuned"],
            "metric_error": [0.02, 0.20],
            "metric_error_val": [0.01, 0.21],
            "imputed": [True, False],
            "impute_method": ["RandomForest_c1_BAG_L1", None],
        }
    )

    loaded = load_tabarena_results(analysis_config, tabarena_context=FakeTabArenaContext(frame))

    assert loaded.columns.tolist() == [
        "dataset",
        "fold",
        "method",
        "config_type",
        "method_subtype",
        "metric_error",
        "metric_error_val",
        "imputed",
        "impute_method",
    ]
    assert pd.isna(loaded.loc[0, "metric_error"])
    assert pd.isna(loaded.loc[0, "metric_error_val"])
    assert bool(loaded.loc[0, "imputed"]) is True
    assert loaded.loc[0, "impute_method"] == "RandomForest_c1_BAG_L1"
    assert loaded.loc[1, "metric_error"] == pytest.approx(0.20)
    assert loaded.loc[1, "metric_error_val"] == pytest.approx(0.21)


def test_load_tabarena_results_accepts_string_method_variant_override(analysis_config) -> None:
    config = replace(
        analysis_config,
        analysis=replace(analysis_config.analysis, method_variant="default"),
    )
    frame = pd.DataFrame(
        {
            "dataset": ["dataset_a", "dataset_a"],
            "fold": [0, 0],
            "method": ["nn_default", "nn_tuned"],
            "config_type": ["NN_A", "NN_A"],
            "method_subtype": ["default", "tuned"],
            "metric_error": [0.20, 0.10],
            "metric_error_val": [0.21, 0.11],
        }
    )

    loaded = load_tabarena_results(config, tabarena_context=FakeTabArenaContext(frame))

    assert loaded["method"].tolist() == ["nn_default"]
    assert loaded["method_subtype"].tolist() == ["default"]
