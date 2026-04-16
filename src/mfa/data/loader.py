from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

from ..config import AnalysisConfig

REQUIRED_RESULT_COLUMNS = [
    "dataset",
    "fold",
    "method",
    "metric_error",
    "metric_error_val",
    "config_type",
    "method_subtype",
]


def _filter_methods(methods: Sequence[str], exclude_patterns: Sequence[str]) -> list[str]:
    return [method for method in methods if all(pattern not in method for pattern in exclude_patterns)]


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
        return pd.DataFrame(columns=REQUIRED_RESULT_COLUMNS)

    df_results = pd.concat(frames, ignore_index=True)
    if "method_subtype" not in df_results.columns:
        raise ValueError("TabArena HPO results must include a `method_subtype` column.")
    if "config_type" not in df_results.columns:
        raise ValueError("TabArena HPO results must include a `config_type` column.")

    df_results = df_results[df_results["method_subtype"] == config.analysis.method_variant].copy()
    if datasets is not None:
        dataset_set = set(datasets)
        df_results = df_results[df_results["dataset"].isin(dataset_set)].copy()

    missing_columns = [column for column in REQUIRED_RESULT_COLUMNS if column not in df_results.columns]
    if missing_columns:
        raise ValueError(f"Loaded results are missing required columns: {missing_columns}")

    return df_results.reset_index(drop=True)

