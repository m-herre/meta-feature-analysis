from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from ..config import AnalysisConfig

IDENTITY_RESULT_COLUMNS = [
    "dataset",
    "fold",
    "method",
    "config_type",
    "method_subtype",
]
OPTIONAL_RESULT_COLUMNS = [
    "imputed",
    "impute_method",
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


def _columns_to_keep(df_results: pd.DataFrame, *, required_columns: Sequence[str]) -> list[str]:
    optional_columns = [column for column in OPTIONAL_RESULT_COLUMNS if column in df_results.columns]
    return list(dict.fromkeys([*required_columns, *optional_columns]))


def _null_imputed_metrics(df_results: pd.DataFrame, *, metric_columns: Sequence[str]) -> pd.DataFrame:
    if "imputed" not in df_results.columns:
        return df_results

    normalized = df_results.copy()
    imputed_mask = normalized["imputed"].astype("boolean").fillna(False)
    if not imputed_mask.any():
        return normalized

    normalized.loc[imputed_mask, list(metric_columns)] = np.nan
    return normalized


def _sanitize_result_frame(df_results: pd.DataFrame, *, required_columns: Sequence[str]) -> pd.DataFrame:
    """Restrict raw results to the cache-safe schema used by the pipeline."""
    missing_columns = [column for column in required_columns if column not in df_results.columns]
    if missing_columns:
        raise ValueError(f"Loaded results are missing required columns: {missing_columns}")
    return df_results.loc[:, _columns_to_keep(df_results, required_columns=required_columns)].copy()


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
    df_results = _null_imputed_metrics(
        df_results,
        metric_columns=_required_result_columns(config)[len(IDENTITY_RESULT_COLUMNS):],
    )

    return _sanitize_result_frame(
        df_results,
        required_columns=_required_result_columns(config),
    ).reset_index(drop=True)
