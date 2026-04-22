from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pandas as pd

from mfa.pipeline import run_analysis
from mfa.stats.correlation import correlate_all as real_correlate_all
from mfa.types import AnalysisResult


class FakeTabArenaContext:
    def __init__(self, frame: pd.DataFrame):
        self.frame = frame
        self.methods = ["artifact_a"]
        self.calls = 0

    def load_hpo_results(self, method: str, holdout: bool = False) -> pd.DataFrame:
        self.calls += 1
        return self.frame.copy()


def test_run_analysis_reuses_upstream_caches_across_stats_changes(
    analysis_config,
    monkeypatch,
    tmp_path: Path,
) -> None:
    config = replace(analysis_config, cache=replace(analysis_config.cache, enabled=True, directory=tmp_path))
    raw_results = pd.DataFrame(
        {
            "dataset": ["dataset_a", "dataset_a", "dataset_b", "dataset_b"],
            "fold": [0, 0, 0, 0],
            "method": ["nn_alpha", "gbdt_alpha", "nn_alpha", "gbdt_alpha"],
            "metric_error": [0.20, 0.10, 0.18, 0.12],
            "metric_error_val": [0.21, 0.11, 0.19, 0.13],
            "config_type": ["NN_A", "GBDT_A", "NN_A", "GBDT_A"],
            "method_subtype": ["tuned", "tuned", "tuned", "tuned"],
            "method_metadata": [{"nested": True}] * 4,
        }
    )
    metafeatures = pd.DataFrame(
        {
            "dataset": ["dataset_a", "dataset_b"],
            "repeat": [0, 0],
            "fold": [0, 0],
            "log_n": [2.0, 2.2],
            "n_over_d": [10.0, 12.0],
            "irregularity": [0.4, 0.6],
        }
    )
    task_metadata = pd.DataFrame({"dataset": ["dataset_a", "dataset_b"], "tid": [1, 2]})
    context = FakeTabArenaContext(raw_results)
    metafeature_calls = {"count": 0}
    correlation_calls = {"count": 0}

    def fake_build_metafeature_table(*args, **kwargs) -> pd.DataFrame:
        metafeature_calls["count"] += 1
        return metafeatures.copy()

    def counting_correlate_all(*args, **kwargs):
        correlation_calls["count"] += 1
        return real_correlate_all(*args, **kwargs)

    monkeypatch.setattr("mfa.pipeline.build_metafeature_table", fake_build_metafeature_table)
    monkeypatch.setattr("mfa.pipeline.correlate_all", counting_correlate_all)

    result_first = run_analysis(
        config,
        datasets=["dataset_a", "dataset_b"],
        task_metadata=task_metadata,
        tabarena_context=context,
    )
    result_second = run_analysis(
        config,
        datasets=["dataset_a", "dataset_b"],
        task_metadata=task_metadata,
        tabarena_context=context,
    )
    config_alpha = replace(config, statistics=replace(config.statistics, alpha=0.10))
    result_third = run_analysis(
        config_alpha,
        datasets=["dataset_a", "dataset_b"],
        task_metadata=task_metadata,
        tabarena_context=context,
    )

    assert isinstance(result_first, AnalysisResult)
    assert not result_first.gap_table.empty
    assert not result_first.analysis_table.empty
    assert context.calls == 1
    assert metafeature_calls["count"] == 1
    assert correlation_calls["count"] == 2
    assert result_second.analysis_table.equals(result_first.analysis_table)
    assert result_third.correction_result is not None
    assert result_third.correction_result.alpha == 0.10


def test_run_analysis_reuses_metafeature_cache_when_trace_enabled(
    analysis_config,
    monkeypatch,
    tmp_path: Path,
) -> None:
    config = replace(
        analysis_config,
        cache=replace(analysis_config.cache, enabled=True, directory=tmp_path),
        metafeatures=replace(analysis_config.metafeatures, trace=True),
    )
    raw_results = pd.DataFrame(
        {
            "dataset": ["dataset_a", "dataset_a", "dataset_b", "dataset_b"],
            "fold": [0, 0, 0, 0],
            "method": ["nn_alpha", "gbdt_alpha", "nn_alpha", "gbdt_alpha"],
            "metric_error": [0.20, 0.10, 0.18, 0.12],
            "metric_error_val": [0.21, 0.11, 0.19, 0.13],
            "config_type": ["NN_A", "GBDT_A", "NN_A", "GBDT_A"],
            "method_subtype": ["tuned", "tuned", "tuned", "tuned"],
        }
    )
    metafeatures = pd.DataFrame(
        {
            "dataset": ["dataset_a", "dataset_b"],
            "repeat": [0, 0],
            "fold": [0, 0],
            "log_n": [2.0, 2.2],
            "n_over_d": [10.0, 12.0],
            "irregularity": [0.4, 0.6],
        }
    )
    task_metadata = pd.DataFrame({"dataset": ["dataset_a", "dataset_b"], "tid": [1, 2]})
    context = FakeTabArenaContext(raw_results)
    metafeature_calls = {"count": 0}

    def fake_build_metafeature_table(*args, **kwargs) -> pd.DataFrame:
        metafeature_calls["count"] += 1
        return metafeatures.copy()

    monkeypatch.setattr("mfa.pipeline.build_metafeature_table", fake_build_metafeature_table)

    result_first = run_analysis(
        config,
        datasets=["dataset_a", "dataset_b"],
        task_metadata=task_metadata,
        tabarena_context=context,
    )
    result_second = run_analysis(
        config,
        datasets=["dataset_a", "dataset_b"],
        task_metadata=task_metadata,
        tabarena_context=context,
    )

    assert isinstance(result_first, AnalysisResult)
    assert result_second.analysis_table.equals(result_first.analysis_table)
    assert context.calls == 1
    assert metafeature_calls["count"] == 1


def test_run_analysis_bypasses_metafeature_table_cache_when_pymfe_enabled(
    analysis_config,
    monkeypatch,
    tmp_path: Path,
) -> None:
    config = replace(
        analysis_config,
        cache=replace(analysis_config.cache, enabled=True, directory=tmp_path),
        metafeatures=replace(analysis_config.metafeatures, feature_sets=("basic", "pymfe")),
    )
    raw_results = pd.DataFrame(
        {
            "dataset": ["dataset_a", "dataset_a", "dataset_b", "dataset_b"],
            "fold": [0, 0, 0, 0],
            "method": ["nn_alpha", "gbdt_alpha", "nn_alpha", "gbdt_alpha"],
            "metric_error": [0.20, 0.10, 0.18, 0.12],
            "metric_error_val": [0.21, 0.11, 0.19, 0.13],
            "config_type": ["NN_A", "GBDT_A", "NN_A", "GBDT_A"],
            "method_subtype": ["tuned", "tuned", "tuned", "tuned"],
        }
    )
    metafeatures = pd.DataFrame(
        {
            "dataset": ["dataset_a", "dataset_b"],
            "repeat": [0, 0],
            "fold": [0, 0],
            "log_n": [2.0, 2.2],
            "n_over_d": [10.0, 12.0],
            "pymfe__nr_attr": [3.0, 4.0],
        }
    )
    task_metadata = pd.DataFrame({"dataset": ["dataset_a", "dataset_b"], "tid": [1, 2]})
    context = FakeTabArenaContext(raw_results)
    metafeature_calls = {"count": 0}

    def fake_build_metafeature_table(*args, **kwargs) -> pd.DataFrame:
        metafeature_calls["count"] += 1
        return metafeatures.copy()

    monkeypatch.setattr("mfa.pipeline.build_metafeature_table", fake_build_metafeature_table)

    run_analysis(
        config,
        datasets=["dataset_a", "dataset_b"],
        task_metadata=task_metadata,
        tabarena_context=context,
    )
    run_analysis(
        config,
        datasets=["dataset_a", "dataset_b"],
        task_metadata=task_metadata,
        tabarena_context=context,
    )

    assert context.calls == 1
    assert metafeature_calls["count"] == 2


def test_run_analysis_accepts_string_method_variant_override(
    analysis_config,
    monkeypatch,
    tmp_path: Path,
) -> None:
    config = replace(
        analysis_config,
        cache=replace(analysis_config.cache, enabled=False, directory=tmp_path),
        analysis=replace(analysis_config.analysis, method_variant="default"),
    )
    raw_results = pd.DataFrame(
        {
            "dataset": ["dataset_a", "dataset_a"],
            "fold": [0, 0],
            "method": ["nn_default", "gbdt_default"],
            "metric_error": [0.20, 0.10],
            "metric_error_val": [0.21, 0.11],
            "config_type": ["NN_A", "GBDT_A"],
            "method_subtype": ["default", "default"],
        }
    )
    metafeatures = pd.DataFrame(
        {
            "dataset": ["dataset_a"],
            "repeat": [0],
            "fold": [0],
            "log_n": [2.0],
            "n_over_d": [10.0],
            "irregularity": [0.4],
        }
    )
    task_metadata = pd.DataFrame({"dataset": ["dataset_a"], "tid": [1]})
    context = FakeTabArenaContext(raw_results)
    monkeypatch.setattr("mfa.pipeline.build_metafeature_table", lambda *args, **kwargs: metafeatures.copy())

    result = run_analysis(
        config,
        datasets=["dataset_a"],
        task_metadata=task_metadata,
        tabarena_context=context,
    )

    assert isinstance(result, AnalysisResult)
    assert context.calls == 1
    assert result.gap_table["best_a_method"].tolist() == ["nn_default"]
    assert result.gap_table["best_b_method"].tolist() == ["gbdt_default"]


def test_run_analysis_handles_empty_comparisons_without_crashing(
    analysis_config,
    monkeypatch,
    tmp_path: Path,
) -> None:
    config = replace(analysis_config, cache=replace(analysis_config.cache, enabled=False, directory=tmp_path))
    raw_results = pd.DataFrame(
        {
            "dataset": ["dataset_a"],
            "fold": [0],
            "method": ["nn_alpha"],
            "metric_error": [0.20],
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
            "irregularity": [0.4],
        }
    )
    task_metadata = pd.DataFrame({"dataset": ["dataset_a"], "tid": [1]})
    context = FakeTabArenaContext(raw_results)
    monkeypatch.setattr("mfa.pipeline.build_metafeature_table", lambda *args, **kwargs: metafeatures.copy())

    result = run_analysis(
        config,
        datasets=["dataset_a"],
        task_metadata=task_metadata,
        tabarena_context=context,
    )

    assert result.gap_table.empty
    assert result.analysis_table.empty
    assert result.correction_result is not None
    assert all(item.n_observations == 0 for item in result.correlation_results)


def test_run_analysis_invalidates_raw_cache_when_metric_columns_change(
    analysis_config,
    monkeypatch,
    tmp_path: Path,
) -> None:
    config = replace(analysis_config, cache=replace(analysis_config.cache, enabled=True, directory=tmp_path))
    config_alt_metrics = replace(
        config,
        analysis=replace(
            config.analysis,
            error_column="metric_error_alt",
            selection_error_column="metric_error_val_alt",
        ),
    )
    raw_results = pd.DataFrame(
        {
            "dataset": ["dataset_a", "dataset_a", "dataset_b", "dataset_b"],
            "fold": [0, 0, 0, 0],
            "method": ["nn_alpha", "gbdt_alpha", "nn_alpha", "gbdt_alpha"],
            "metric_error": [0.20, 0.10, 0.18, 0.12],
            "metric_error_val": [0.21, 0.11, 0.19, 0.13],
            "metric_error_alt": [0.40, 0.30, 0.38, 0.32],
            "metric_error_val_alt": [0.41, 0.31, 0.39, 0.33],
            "config_type": ["NN_A", "GBDT_A", "NN_A", "GBDT_A"],
            "method_subtype": ["tuned", "tuned", "tuned", "tuned"],
        }
    )
    metafeatures = pd.DataFrame(
        {
            "dataset": ["dataset_a", "dataset_b"],
            "repeat": [0, 0],
            "fold": [0, 0],
            "log_n": [2.0, 2.2],
            "n_over_d": [10.0, 12.0],
            "irregularity": [0.4, 0.6],
        }
    )
    task_metadata = pd.DataFrame({"dataset": ["dataset_a", "dataset_b"], "tid": [1, 2]})
    context = FakeTabArenaContext(raw_results)
    monkeypatch.setattr("mfa.pipeline.build_metafeature_table", lambda *args, **kwargs: metafeatures.copy())

    result_first = run_analysis(
        config,
        datasets=["dataset_a", "dataset_b"],
        task_metadata=task_metadata,
        tabarena_context=context,
    )
    result_second = run_analysis(
        config_alt_metrics,
        datasets=["dataset_a", "dataset_b"],
        task_metadata=task_metadata,
        tabarena_context=context,
    )

    assert context.calls == 2
    assert not result_first.gap_table.empty
    assert not result_second.gap_table.empty


def test_run_analysis_logs_stage_progress(
    analysis_config,
    monkeypatch,
    tmp_path: Path,
) -> None:
    config = replace(analysis_config, cache=replace(analysis_config.cache, enabled=False, directory=tmp_path))
    raw_results = pd.DataFrame(
        {
            "dataset": ["dataset_a", "dataset_a", "dataset_b", "dataset_b"],
            "fold": [0, 0, 0, 0],
            "method": ["nn_alpha", "gbdt_alpha", "nn_alpha", "gbdt_alpha"],
            "metric_error": [0.20, 0.10, 0.18, 0.12],
            "metric_error_val": [0.21, 0.11, 0.19, 0.13],
            "config_type": ["NN_A", "GBDT_A", "NN_A", "GBDT_A"],
            "method_subtype": ["tuned", "tuned", "tuned", "tuned"],
        }
    )
    metafeatures = pd.DataFrame(
        {
            "dataset": ["dataset_a", "dataset_b"],
            "repeat": [0, 0],
            "fold": [0, 0],
            "log_n": [2.0, 2.2],
            "n_over_d": [10.0, 12.0],
            "irregularity": [0.4, 0.6],
        }
    )
    task_metadata = pd.DataFrame({"dataset": ["dataset_a", "dataset_b"], "tid": [1, 2]})
    context = FakeTabArenaContext(raw_results)
    messages: list[str] = []

    def capture_info(message: str, *args) -> None:
        messages.append(message % args if args else message)

    monkeypatch.setattr("mfa.pipeline.build_metafeature_table", lambda *args, **kwargs: metafeatures.copy())
    monkeypatch.setattr("mfa.pipeline.logger.info", capture_info)

    run_analysis(
        config,
        datasets=["dataset_a", "dataset_b"],
        task_metadata=task_metadata,
        tabarena_context=context,
    )

    assert any("Starting analysis:" in message for message in messages)
    assert any("Stage 1/5 raw results:" in message for message in messages)
    assert any("Stage 2/5 meta-features:" in message for message in messages)
    assert any("Stage 3/5 pairwise gaps:" in message for message in messages)
    assert any("Stage 4/5 analysis table:" in message for message in messages)
    assert any("Stage 5/5 statistics:" in message for message in messages)
    assert any("Analysis complete:" in message for message in messages)
