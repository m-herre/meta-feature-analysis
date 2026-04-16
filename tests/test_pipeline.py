from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pandas as pd

from mfa.pipeline import run_analysis
from mfa.types import AnalysisResult, CorrectionResult, CorrelationResult


def test_run_analysis_integration(analysis_config, monkeypatch, tmp_path: Path) -> None:
    config = analysis_config
    config = replace(config, cache=replace(config.cache, enabled=False, directory=tmp_path))

    raw_results = pd.DataFrame(
        {
            "dataset": ["dataset_a"],
            "fold": [0],
            "method": ["nn_alpha"],
            "metric_error": [0.2],
            "metric_error_val": [0.21],
            "config_type": ["NN_A"],
            "method_subtype": ["tuned"],
        }
    )
    metafeatures = pd.DataFrame(
        {
            "dataset": ["dataset_a"],
            "repeat": [0],
            "fold": [0],
            "log_n": [2.0],
            "n_over_d": [10.0],
            "irregularity": [0.5],
        }
    )
    gaps = pd.DataFrame(
        {
            "dataset": ["dataset_a"],
            "repeat": [0],
            "fold": [0],
            "comparison_name": ["nn_vs_gbdt"],
            "group_a_name": ["nn"],
            "group_b_name": ["gbdt"],
            "group_a_label": ["NN"],
            "group_b_label": ["GBDT"],
            "best_a_method": ["nn_alpha"],
            "best_a_error": [0.2],
            "best_a_norm_error": [0.5],
            "best_b_method": ["gbdt_alpha"],
            "best_b_error": [0.1],
            "best_b_norm_error": [0.0],
            "delta_raw": [0.1],
            "delta_norm": [0.5],
        }
    )
    analysis_table = pd.DataFrame(
        {
            "dataset": ["dataset_a"],
            "comparison_name": ["nn_vs_gbdt"],
            "group_a_name": ["nn"],
            "group_b_name": ["gbdt"],
            "group_a_label": ["NN"],
            "group_b_label": ["GBDT"],
            "n_splits": [1],
            "delta_norm": [0.5],
            "log_n": [2.0],
            "n_over_d": [10.0],
            "irregularity": [0.5],
        }
    )
    correlations = [CorrelationResult("nn_vs_gbdt", "log_n", "delta_norm", 1.0, 0.01, 5)]
    correction = CorrectionResult("bh", 0.05, tuple(correlations), (0.01,), (True,))

    monkeypatch.setattr("mfa.pipeline.load_tabarena_results", lambda *args, **kwargs: raw_results)
    monkeypatch.setattr("mfa.pipeline._get_task_metadata", lambda *args, **kwargs: pd.DataFrame({"dataset": ["dataset_a"], "tid": [1]}))
    monkeypatch.setattr("mfa.pipeline.build_metafeature_table", lambda *args, **kwargs: metafeatures)
    monkeypatch.setattr("mfa.pipeline.compute_pairwise_gaps", lambda *args, **kwargs: gaps)
    monkeypatch.setattr("mfa.pipeline.build_analysis_table", lambda *args, **kwargs: analysis_table)
    monkeypatch.setattr("mfa.pipeline.correlate_all", lambda *args, **kwargs: correlations)
    monkeypatch.setattr("mfa.pipeline.apply_fdr_correction", lambda *args, **kwargs: correction)

    result = run_analysis(config, datasets=["dataset_a"])
    assert isinstance(result, AnalysisResult)
    assert result.gap_table.equals(gaps)
    assert result.analysis_table.equals(analysis_table)
    assert result.correction_result == correction
