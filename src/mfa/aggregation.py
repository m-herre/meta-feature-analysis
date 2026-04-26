from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from .metafeatures.basic import (
    CLASSIFICATION_PROBLEM_TYPES as BASIC_CLASSIFICATION_PROBLEM_TYPES,
)
from .metafeatures.irregularity import (
    DEFAULT_IRREGULARITY_COMPONENTS,
    add_irregularity_proxy,
)
from .metafeatures.pymfe_catalog import PYMFE_CLASSIFICATION_ONLY
from .stats.correlation import EXCLUDED_PREDICTOR_COLUMNS
from .types import AnalysisUnit


def _strict_mean(series: pd.Series) -> float:
    """Mean that propagates NaN: any missing split ⇒ dataset-level NaN.

    Meta-feature predictors can be NaN on a subset of splits when an
    extractor (e.g. pymfe) fails for that split. A plain mean silently
    averages over only the successful splits, so the effective sample
    size per predictor drifts away from delta_norm's (which sees every
    split). Propagating NaN forces the dataset to drop from per-predictor
    correlations visibly, via the existing `n_observations` in
    per-predictor association summaries.
    """
    if series.isna().any():
        return float("nan")
    return float(np.mean(series))


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

BASIC_CLASSIFICATION_ONLY_FEATURES = frozenset(
    {
        "n_classes",
        "class_entropy",
        "majority_class_fraction",
        "minority_class_fraction",
        "class_imbalance_ratio",
    }
)
CLASSIFICATION_PROBLEM_TYPES = set(BASIC_CLASSIFICATION_PROBLEM_TYPES)

ANALYSIS_CONTEXT_COLUMNS = [
    "dataset",
    "task_type",
    "comparison_name",
    "group_a_name",
    "group_b_name",
    "group_a_label",
    "group_b_label",
    "expected_direction",
    "n_splits",
    "best_a_error",
    "best_a_norm_error",
    "best_b_error",
    "best_b_norm_error",
    "delta_raw",
    "delta_raw_std",
    "delta_raw_sem",
    "delta_norm",
    "delta_norm_std",
    "delta_norm_sem",
]

KNOWN_NUMERIC_GAP_COLUMNS = {
    "best_a_error",
    "best_a_norm_error",
    "best_b_error",
    "best_b_norm_error",
    "delta_raw",
    "delta_norm",
}


def _is_pymfe_classification_only(column: str) -> bool:
    if not column.startswith("pymfe__"):
        return False
    raw_feature = column.removeprefix("pymfe__").split(".", maxsplit=1)[0]
    return raw_feature in PYMFE_CLASSIFICATION_ONLY


def is_classification_only_feature(column: str) -> bool:
    return column in BASIC_CLASSIFICATION_ONLY_FEATURES or _is_pymfe_classification_only(column)


def infer_numeric_predictors(table: pd.DataFrame, *, target: str = "delta_norm") -> list[str]:
    predictors = []
    for column in table.columns:
        if column == target or column in EXCLUDED_PREDICTOR_COLUMNS or column.startswith("best_"):
            continue
        numeric_values = pd.to_numeric(table[column], errors="coerce")
        if pd.api.types.is_numeric_dtype(table[column]) or numeric_values.notna().any():
            predictors.append(column)
    return predictors


def _analysis_columns(metafeature_table: pd.DataFrame, gap_table: pd.DataFrame) -> list[str]:
    return list(dict.fromkeys([*metafeature_table.columns.tolist(), *gap_table.columns.tolist()]))


def _empty_analysis_table(metafeature_table: pd.DataFrame, gap_table: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(columns=_analysis_columns(metafeature_table, gap_table))


def _candidate_numeric_columns(analysis: pd.DataFrame, metafeature_table: pd.DataFrame) -> list[str]:
    metafeature_columns = [
        column for column in metafeature_table.columns if column not in JOIN_KEY_COLUMNS and not column.startswith("_")
    ]
    numeric_columns = [
        column for column in analysis.select_dtypes(include="number").columns if column not in {"repeat", "fold"}
    ]
    expected = [
        column
        for column in [*metafeature_columns, *sorted(KNOWN_NUMERIC_GAP_COLUMNS)]
        if column in analysis.columns and column not in {"repeat", "fold"}
    ]
    return list(dict.fromkeys([*numeric_columns, *expected]))


def _build_aggregations(analysis: pd.DataFrame, metafeature_table: pd.DataFrame) -> dict[str, tuple[str, object]]:
    metafeature_column_names = {
        column for column in metafeature_table.columns if column not in JOIN_KEY_COLUMNS and not column.startswith("_")
    }
    aggregations: dict[str, tuple[str, object]] = {
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
        elif column in metafeature_column_names:
            aggregations[column] = (column, _strict_mean)
        else:
            aggregations[column] = (column, "mean")
    return aggregations


def _unique_dataset_irregularity(
    dataset_level_df: pd.DataFrame,
    *,
    components: tuple[str, ...] = DEFAULT_IRREGULARITY_COMPONENTS,
) -> pd.DataFrame:
    """Return a ``(dataset, irregularity)`` table with one row per dataset.

    ``dataset_level_df`` is grouped by ``GROUP_COLUMNS`` (which includes
    ``comparison_name``), so the same dataset appears once per comparison
    with identical component values. Z-scoring across those rows would
    double-count datasets that participate in more comparisons and
    silently re-weight the global mean/std by comparison availability
    rather than by dataset properties. We collapse to unique datasets
    first, compute the composite there, and broadcast it back via merge.
    """
    available = [column for column in components if column in dataset_level_df.columns]
    unique_datasets = dataset_level_df[["dataset"]].drop_duplicates().reset_index(drop=True)
    if not available:
        return unique_datasets.assign(irregularity=np.nan)
    unique_rows = dataset_level_df[["dataset", *available]].drop_duplicates(subset=["dataset"]).reset_index(drop=True)
    with_composite = add_irregularity_proxy(unique_rows, components=tuple(available))
    return with_composite[["dataset", "irregularity"]]


def build_analysis_table(
    gap_table: pd.DataFrame,
    metafeature_table: pd.DataFrame,
    *,
    unit: AnalysisUnit = AnalysisUnit.DATASET,
    irregularity_components: tuple[str, ...] = DEFAULT_IRREGULARITY_COMPONENTS,
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

    aggregations = _build_aggregations(analysis, metafeature_table)
    components_present = any(column in metafeature_table.columns for column in irregularity_components)
    if not analysis.empty:
        dataset_level = analysis.groupby(GROUP_COLUMNS, as_index=False, dropna=False).agg(**aggregations)
        if components_present:
            irregularity_lookup = _unique_dataset_irregularity(dataset_level, components=irregularity_components)
            dataset_level = dataset_level.drop(columns=["irregularity"], errors="ignore").merge(
                irregularity_lookup, on="dataset", how="left", validate="many_to_one"
            )
        else:
            irregularity_lookup = None
    else:
        dataset_level = analysis.copy()
        if "irregularity" not in dataset_level.columns:
            dataset_level["irregularity"] = np.nan
        irregularity_lookup = None

    if unit == AnalysisUnit.FOLD:
        warnings.warn(
            "Fold-level analysis treats non-independent folds as separate observations.",
            stacklevel=2,
        )
        if irregularity_lookup is not None:
            # Broadcast the unique-per-dataset composite back onto per-split
            # rows. Merging on `dataset` against a unique-per-dataset lookup
            # keeps fold row counts unchanged even when a dataset participates
            # in multiple comparisons.
            fold_level = analysis.drop(columns=["irregularity"], errors="ignore").merge(
                irregularity_lookup,
                on="dataset",
                how="left",
                validate="many_to_one",
            )
        else:
            fold_level = analysis
        return fold_level.sort_values(["comparison_name", "dataset", "repeat", "fold"]).reset_index(drop=True)

    return dataset_level.sort_values(["comparison_name", "dataset"]).reset_index(drop=True)
