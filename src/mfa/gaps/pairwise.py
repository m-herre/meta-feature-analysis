from __future__ import annotations

import pandas as pd

from .._logging import get_logger
from ..data.split_decoder import add_split_columns
from ..groups import comparison_to_dict
from ..types import ComparisonSpec
from .normalization import add_normalized_error

logger = get_logger(__name__)


def pick_best_in_group(
    df: pd.DataFrame,
    *,
    config_types: frozenset[str] | set[str],
    prefix: str,
    error_column: str = "metric_error",
    norm_error_column: str = "norm_error",
) -> pd.DataFrame:
    """Pick the best-performing method inside a config-type group per split."""
    group_cols = ["dataset", "repeat", "fold_in_repeat"]
    family = df[df["config_type"].isin(config_types)].copy()
    if family.empty:
        return pd.DataFrame(
            columns=[
                "dataset",
                "repeat",
                "fold",
                f"best_{prefix}_method",
                f"best_{prefix}_error",
                f"best_{prefix}_norm_error",
            ]
        )
    family = family.sort_values(group_cols + [norm_error_column, error_column, "method"])
    family = family.groupby(group_cols, as_index=False).first()
    family = family[group_cols + ["method", error_column, norm_error_column]].rename(
        columns={
            "fold_in_repeat": "fold",
            "method": f"best_{prefix}_method",
            error_column: f"best_{prefix}_error",
            norm_error_column: f"best_{prefix}_norm_error",
        }
    )
    return family.reset_index(drop=True)


def compute_pairwise_gaps(
    df_results: pd.DataFrame,
    comparisons: tuple[ComparisonSpec, ...] | list[ComparisonSpec],
    *,
    error_column: str = "metric_error",
    n_folds_per_repeat: int = 3,
) -> pd.DataFrame:
    """Compute per-split pairwise group gaps for all configured comparisons."""
    if df_results.empty:
        return pd.DataFrame(
            columns=[
                "dataset",
                "repeat",
                "fold",
                "comparison_name",
                "group_a_name",
                "group_b_name",
                "group_a_label",
                "group_b_label",
                "best_a_method",
                "best_a_error",
                "best_a_norm_error",
                "best_b_method",
                "best_b_error",
                "best_b_norm_error",
                "delta_raw",
                "delta_norm",
            ]
        )

    prepared = add_split_columns(df_results, n_folds_per_repeat=n_folds_per_repeat)
    prepared = add_normalized_error(prepared, error_column=error_column, fold_column="split_id")

    all_tables: list[pd.DataFrame] = []
    for comparison in comparisons:
        best_a = pick_best_in_group(
            prepared,
            config_types=comparison.group_a.config_types,
            prefix="a",
            error_column=error_column,
        )
        best_b = pick_best_in_group(
            prepared,
            config_types=comparison.group_b.config_types,
            prefix="b",
            error_column=error_column,
        )
        merged = best_a.merge(best_b, on=["dataset", "repeat", "fold"], how="inner", validate="one_to_one")
        if merged.empty:
            logger.warning("Comparison `%s` produced no overlapping splits after filtering.", comparison.name)
            continue
        merged["delta_raw"] = merged["best_a_error"] - merged["best_b_error"]
        merged["delta_norm"] = merged["best_a_norm_error"] - merged["best_b_norm_error"]
        metadata = comparison_to_dict(comparison)
        for column, value in metadata.items():
            merged[column] = value
        all_tables.append(merged)

    if not all_tables:
        return pd.DataFrame()
    return pd.concat(all_tables, ignore_index=True).sort_values(
        ["comparison_name", "dataset", "repeat", "fold"]
    ).reset_index(drop=True)

