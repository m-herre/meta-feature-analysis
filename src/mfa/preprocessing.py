from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

MAX_FEATURE_MISSINGNESS = 0.20
NEAR_CONSTANT_TOP_SHARE = 0.95
NUMERICAL_CONSTANT_REL_TOL = 1e-12
REDUNDANCY_SPEARMAN_THRESHOLD = 0.95


def _near_constant_report(
    table: pd.DataFrame,
    feature_columns: Sequence[str],
    *,
    top_share_threshold: float = NEAR_CONSTANT_TOP_SHARE,
    numerical_constant_rel_tol: float = NUMERICAL_CONSTANT_REL_TOL,
) -> pd.DataFrame:
    rows = []
    for column in feature_columns:
        values = pd.to_numeric(table[column], errors="coerce").dropna()
        n_non_missing = int(values.shape[0])
        if n_non_missing == 0:
            rows.append(
                {
                    "feature": column,
                    "n_non_missing": n_non_missing,
                    "n_unique": 0,
                    "top_share": np.nan,
                    "reason": "all_missing_after_inf_replacement",
                }
            )
            continue

        n_unique = int(values.nunique(dropna=True))
        rounded_values = values.round(12)
        top_share = float(rounded_values.value_counts(normalize=True, dropna=True).iloc[0])
        scale = max(1.0, float(values.abs().median()))
        numerical_span = float(values.max() - values.min())

        reason = None
        if n_unique <= 1:
            reason = "exact_constant"
        elif numerical_span <= numerical_constant_rel_tol * scale:
            reason = "numerical_constant"
        elif top_share >= top_share_threshold:
            reason = "dominant_value_share_ge_0.95"

        if reason is not None:
            rows.append(
                {
                    "feature": column,
                    "n_non_missing": n_non_missing,
                    "n_unique": n_unique,
                    "top_share": top_share,
                    "reason": reason,
                }
            )
    return pd.DataFrame(rows)


def preprocess_analysis_table(
    table: pd.DataFrame,
    feature_columns: Sequence[str],
    *,
    table_name: str,
    max_feature_missingness: float = MAX_FEATURE_MISSINGNESS,
    context_columns: Sequence[str] = (),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    cleaned = table.copy()
    for column in cleaned.columns:
        numeric_values = pd.to_numeric(cleaned[column], errors="coerce")
        if pd.api.types.is_numeric_dtype(cleaned[column]) or numeric_values.notna().any():
            cleaned[column] = numeric_values.replace([np.inf, -np.inf], np.nan)
    feature_columns = [column for column in feature_columns if column in cleaned.columns]

    missing_fraction = cleaned[list(feature_columns)].isna().mean().sort_values(ascending=False)
    high_missing_features = missing_fraction[missing_fraction > max_feature_missingness].index.tolist()
    after_missingness = [column for column in feature_columns if column not in high_missing_features]

    near_constant = _near_constant_report(cleaned, after_missingness)
    near_constant_features = near_constant["feature"].tolist() if not near_constant.empty else []
    retained_features = [column for column in after_missingness if column not in near_constant_features]

    retained_context_columns = [column for column in context_columns if column in cleaned.columns]
    processed = cleaned.loc[:, retained_context_columns + retained_features].copy()

    report_rows = []
    for feature in high_missing_features:
        report_rows.append(
            {
                "table": table_name,
                "stage": "high_missingness",
                "feature": feature,
                "missing_fraction": float(missing_fraction.loc[feature]),
                "reason": f"> {max_feature_missingness:.0%} missing",
            }
        )
    if not near_constant.empty:
        near_constant = near_constant.assign(table=table_name, stage="near_constant")
        near_constant["missing_fraction"] = near_constant["feature"].map(missing_fraction).astype(float)
        report_rows.extend(
            near_constant[
                [
                    "table",
                    "stage",
                    "feature",
                    "missing_fraction",
                    "n_non_missing",
                    "n_unique",
                    "top_share",
                    "reason",
                ]
            ].to_dict("records")
        )

    return processed, pd.DataFrame(report_rows)


def _redundancy_feature_columns(
    table: pd.DataFrame,
    *,
    context_columns: Sequence[str] = (),
    target: str = "delta_norm",
) -> list[str]:
    return [column for column in table.columns if column not in set(context_columns) and column != target]


def _prefer_feature(
    feature_a: str,
    feature_b: str,
    *,
    missing_fraction: pd.Series,
    n_unique: pd.Series,
) -> tuple[str, str]:
    key_a = (
        float(missing_fraction.loc[feature_a]),
        -int(n_unique.loc[feature_a]),
        feature_a,
    )
    key_b = (
        float(missing_fraction.loc[feature_b]),
        -int(n_unique.loc[feature_b]),
        feature_b,
    )
    if key_a <= key_b:
        return feature_a, feature_b
    return feature_b, feature_a


def reduce_redundant_features(
    table: pd.DataFrame,
    *,
    table_name: str,
    threshold: float = REDUNDANCY_SPEARMAN_THRESHOLD,
    context_columns: Sequence[str] = (),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    feature_columns = _redundancy_feature_columns(table, context_columns=context_columns)
    if len(feature_columns) <= 1:
        return table.copy(), pd.DataFrame()

    numeric_features = table[feature_columns].apply(pd.to_numeric, errors="coerce")
    missing_fraction = numeric_features.isna().mean()
    n_unique = numeric_features.nunique(dropna=True)
    corr = numeric_features.corr(method="spearman", min_periods=3).abs()

    high_corr_pairs = []
    for i, feature_a in enumerate(feature_columns):
        for feature_b in feature_columns[i + 1 :]:
            abs_spearman = corr.loc[feature_a, feature_b]
            if pd.notna(abs_spearman) and abs_spearman >= threshold:
                high_corr_pairs.append((float(abs_spearman), feature_a, feature_b))
    high_corr_pairs.sort(key=lambda row: (-row[0], row[1], row[2]))

    dropped_features: set[str] = set()
    report_rows = []
    for abs_spearman, feature_a, feature_b in high_corr_pairs:
        if feature_a in dropped_features or feature_b in dropped_features:
            continue

        kept_feature, dropped_feature = _prefer_feature(
            feature_a,
            feature_b,
            missing_fraction=missing_fraction,
            n_unique=n_unique,
        )
        dropped_features.add(dropped_feature)
        report_rows.append(
            {
                "table": table_name,
                "stage": "redundancy",
                "dropped_feature": dropped_feature,
                "kept_feature": kept_feature,
                "abs_spearman": abs_spearman,
                "dropped_missing_fraction": float(missing_fraction.loc[dropped_feature]),
                "kept_missing_fraction": float(missing_fraction.loc[kept_feature]),
                "dropped_n_unique": int(n_unique.loc[dropped_feature]),
                "kept_n_unique": int(n_unique.loc[kept_feature]),
                "reason": f"abs_spearman >= {threshold:.2f}",
            }
        )

    retained_features = [feature for feature in feature_columns if feature not in dropped_features]
    retained_context_columns = [column for column in context_columns if column in table]
    reduced = table.loc[:, retained_context_columns + retained_features].copy()
    return reduced, pd.DataFrame(report_rows)
