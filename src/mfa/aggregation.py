from __future__ import annotations

import warnings

import pandas as pd

from .types import AnalysisUnit

ID_COLUMNS = {
    "dataset",
    "repeat",
    "fold",
    "comparison_name",
    "group_a_name",
    "group_b_name",
    "group_a_label",
    "group_b_label",
}


def build_analysis_table(
    gap_table: pd.DataFrame,
    metafeature_table: pd.DataFrame,
    *,
    unit: AnalysisUnit = AnalysisUnit.DATASET,
) -> pd.DataFrame:
    """Join gaps with meta-features and optionally aggregate to dataset level."""
    analysis = metafeature_table.merge(
        gap_table,
        on=["dataset", "repeat", "fold"],
        how="inner",
        validate="one_to_many",
    )
    if unit == AnalysisUnit.FOLD:
        warnings.warn(
            "Fold-level analysis treats non-independent folds as separate observations.",
            stacklevel=2,
        )
        return analysis.sort_values(["comparison_name", "dataset", "repeat", "fold"]).reset_index(drop=True)

    group_cols = [
        "dataset",
        "comparison_name",
        "group_a_name",
        "group_b_name",
        "group_a_label",
        "group_b_label",
    ]
    numeric_columns = [
        column
        for column in analysis.select_dtypes(include="number").columns
        if column not in {"repeat", "fold"}
    ]

    aggregations: dict[str, tuple[str, str]] = {
        "n_splits": ("delta_norm", "size"),
    }
    for column in numeric_columns:
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

    dataset_level = analysis.groupby(group_cols, as_index=False).agg(**aggregations)
    return dataset_level.sort_values(["comparison_name", "dataset"]).reset_index(drop=True)

