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
