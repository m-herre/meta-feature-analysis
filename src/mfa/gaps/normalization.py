from __future__ import annotations

import pandas as pd

from tabarena.utils.normalized_scorer import NormalizedScorer


def add_normalized_error(
    df: pd.DataFrame,
    *,
    error_column: str = "metric_error",
    method_column: str = "method",
    dataset_column: str = "dataset",
    fold_column: str = "fold",
    output_column: str = "norm_error",
) -> pd.DataFrame:
    """Attach a normalized error column computed per (dataset, fold) task."""
    normalized = df.copy()
    if normalized.empty:
        normalized[output_column] = pd.Series(dtype=float)
        return normalized

    task_keys = [
        tuple(values)
        for values in normalized[[dataset_column, fold_column]].drop_duplicates().itertuples(index=False, name=None)
    ]
    scorer = NormalizedScorer(
        df_results=normalized,
        tasks=task_keys,
        baseline=None,
        metric_error_col=error_column,
        task_col=[dataset_column, fold_column],
        framework_col=method_column,
    )
    normalized[output_column] = [
        scorer.rank(task=(dataset, split_id), error=error)
        for dataset, split_id, error in zip(
            normalized[dataset_column],
            normalized[fold_column],
            normalized[error_column],
        )
    ]
    return normalized

