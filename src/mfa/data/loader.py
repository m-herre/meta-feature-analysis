from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

from ..config import AnalysisConfig

IDENTITY_RESULT_COLUMNS = [
    "dataset",
    "fold",
    "method",
    "config_type",
    "method_subtype",
]


def _filter_methods(methods: Sequence[str], exclude_patterns: Sequence[str]) -> list[str]:
    return [method for method in methods if all(pattern not in method for pattern in exclude_patterns)]


def _required_result_columns(config: AnalysisConfig) -> list[str]:
    columns = [
        *IDENTITY_RESULT_COLUMNS,
        config.analysis.error_column,
    ]
    if config.analysis.selection_error_column is not None:
        columns.append(config.analysis.selection_error_column)
    return list(dict.fromkeys(columns))


def _sanitize_result_frame(df_results: pd.DataFrame, *, required_columns: Sequence[str]) -> pd.DataFrame:
    """Restrict raw results to the cache-safe schema used by the pipeline."""
    missing_columns = [column for column in required_columns if column not in df_results.columns]
    if missing_columns:
        raise ValueError(f"Loaded results are missing required columns: {missing_columns}")
    return df_results.loc[:, list(required_columns)].copy()


def load_tabarena_results(
    config: AnalysisConfig,
    *,
    datasets: Sequence[str] | None = None,
    methods: Sequence[str] | None = None,
    tabarena_context=None,
    holdout: bool = False,
) -> pd.DataFrame:
    """Load and filter TabArena HPO results for the requested method subtype."""
    if tabarena_context is None:
        from tabarena.nips2025_utils.tabarena_context import TabArenaContext

        tabarena_context = TabArenaContext()

    selected_methods = list(methods) if methods is not None else list(tabarena_context.methods)
    selected_methods = _filter_methods(selected_methods, config.analysis.exclude_methods_containing)

    frames: list[pd.DataFrame] = []
    for method in selected_methods:
        frame = tabarena_context.load_hpo_results(method=method, holdout=holdout).copy()
        frames.append(frame)

    if not frames:
        return pd.DataFrame(columns=_required_result_columns(config))

    df_results = pd.concat(frames, ignore_index=True)
    if "method_subtype" not in df_results.columns:
        raise ValueError("TabArena HPO results must include a `method_subtype` column.")
    if "config_type" not in df_results.columns:
        raise ValueError("TabArena HPO results must include a `config_type` column.")

    df_results = df_results[df_results["method_subtype"] == config.analysis.method_variant].copy()
    if datasets is not None:
        dataset_set = set(datasets)
        df_results = df_results[df_results["dataset"].isin(dataset_set)].copy()

    return _sanitize_result_frame(
        df_results,
        required_columns=_required_result_columns(config),
    ).reset_index(drop=True)
