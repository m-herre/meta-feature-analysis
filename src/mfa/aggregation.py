from __future__ import annotations

import warnings

import pandas as pd

from .types import AnalysisUnit

JOIN_KEY_COLUMNS = ["dataset", "repeat", "fold"]
GROUP_COLUMNS = [
    "dataset",
    "comparison_name",
    "group_a_name",
    "group_b_name",
    "group_a_label",
    "group_b_label",
    "expected_direction",
]
ID_COLUMNS = {
    *JOIN_KEY_COLUMNS,
    *GROUP_COLUMNS,
}

KNOWN_NUMERIC_GAP_COLUMNS = {
    "best_a_error",
    "best_a_norm_error",
    "best_b_error",
    "best_b_norm_error",
    "delta_raw",
    "delta_norm",
}


def _analysis_columns(metafeature_table: pd.DataFrame, gap_table: pd.DataFrame) -> list[str]:
    return list(dict.fromkeys([*metafeature_table.columns.tolist(), *gap_table.columns.tolist()]))


def _empty_analysis_table(metafeature_table: pd.DataFrame, gap_table: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(columns=_analysis_columns(metafeature_table, gap_table))


def _candidate_numeric_columns(analysis: pd.DataFrame, metafeature_table: pd.DataFrame) -> list[str]:
    metafeature_columns = [
        column
        for column in metafeature_table.columns
        if column not in JOIN_KEY_COLUMNS and not column.startswith("_")
    ]
    numeric_columns = [
        column
        for column in analysis.select_dtypes(include="number").columns
        if column not in {"repeat", "fold"}
    ]
    expected = [
        column
        for column in [*metafeature_columns, *sorted(KNOWN_NUMERIC_GAP_COLUMNS)]
        if column in analysis.columns and column not in {"repeat", "fold"}
    ]
    return list(dict.fromkeys([*numeric_columns, *expected]))


def _build_aggregations(analysis: pd.DataFrame, metafeature_table: pd.DataFrame) -> dict[str, tuple[str, str]]:
    aggregations: dict[str, tuple[str, str]] = {
        "n_splits": ("delta_norm", "size"),
    }
    for column in _candidate_numeric_columns(analysis, metafeature_table):
        if column == "delta_norm":
            aggregations["delta_norm"] = (column, "mean")
            aggregations["delta_norm_std"] = (column, "std")
            aggregations["delta_norm_sem"] = (column, "sem")
        elif column == "delta_raw":
            aggregations["delta_raw"] = (column, "mean")
            aggregations["delta_raw_std"] = (column, "std")
            aggregations["delta_raw_sem"] = (column, "sem")
        else:
            aggregations[column] = (column, "mean")
    return aggregations


def build_analysis_table(
    gap_table: pd.DataFrame,
    metafeature_table: pd.DataFrame,
    *,
    unit: AnalysisUnit = AnalysisUnit.DATASET,
) -> pd.DataFrame:
    """Join gaps with meta-features and optionally aggregate to dataset level."""
    if set(JOIN_KEY_COLUMNS).issubset(metafeature_table.columns) and set(JOIN_KEY_COLUMNS).issubset(gap_table.columns):
        analysis = metafeature_table.merge(
            gap_table,
            on=JOIN_KEY_COLUMNS,
            how="inner",
            validate="one_to_many",
        )
    else:
        analysis = _empty_analysis_table(metafeature_table, gap_table)
    if unit == AnalysisUnit.FOLD:
        warnings.warn(
            "Fold-level analysis treats non-independent folds as separate observations.",
            stacklevel=2,
        )
        return analysis.sort_values(["comparison_name", "dataset", "repeat", "fold"]).reset_index(drop=True)

    aggregations = _build_aggregations(analysis, metafeature_table)
    dataset_level = analysis.groupby(GROUP_COLUMNS, as_index=False).agg(**aggregations)
    return dataset_level.sort_values(["comparison_name", "dataset"]).reset_index(drop=True)
