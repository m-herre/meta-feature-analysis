from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

EXCLUDED_PREDICTOR_COLUMNS = {
    "dataset",
    "comparison_name",
    "group_a_name",
    "group_b_name",
    "group_a_label",
    "group_b_label",
    "n_splits",
    "delta_raw",
    "delta_raw_std",
    "delta_raw_sem",
    "delta_norm_std",
    "delta_norm_sem",
}

ASSOCIATION_TARGET = "delta_norm"
ASSOCIATION_MIN_N = 30
ASSOCIATION_ALPHA = 0.05
ASSOCIATION_BOOTSTRAP_REPEATS = 500
ASSOCIATION_CI_TOP_K = None
ASSOCIATION_RANK_STABILITY_TOP_K = 25
ASSOCIATION_RANDOM_SEED = 20260424
INDEPENDENT_UNIT_COLUMN = "dataset"

ROBUST_ASSOCIATION_Q_THRESHOLD = 0.05
ROBUST_ASSOCIATION_MIN_SIGN_CONSISTENCY = 0.90
ROBUST_ASSOCIATION_TOP_N = 25

ROBUST_ASSOCIATION_OUTPUT_COLUMNS = [
    "feature",
    "n",
    "spearman_r",
    "abs_spearman_r",
    "p_value_bh",
    "ci_low",
    "ci_high",
    "bootstrap_sign_consistency",
    "bootstrap_rank_median",
    "bootstrap_rank_q05",
    "bootstrap_rank_q95",
    "bootstrap_top_k_frequency",
]

ROBUST_ASSOCIATION_COLUMN_NAMES = {
    "feature": "feature",
    "n": "n",
    "spearman_r": "spearman_rho",
    "abs_spearman_r": "abs_spearman_rho",
    "p_value_bh": "bh_q_value",
    "ci_low": "bootstrap_ci_low",
    "ci_high": "bootstrap_ci_high",
    "bootstrap_sign_consistency": "bootstrap_sign_consistency",
    "bootstrap_rank_median": "bootstrap_rank_median",
    "bootstrap_rank_q05": "bootstrap_rank_q05",
    "bootstrap_rank_q95": "bootstrap_rank_q95",
    "bootstrap_top_k_frequency": "bootstrap_top_k_frequency",
}


def assert_dataset_level_table(
    table: pd.DataFrame,
    *,
    table_name: str,
    unit_column: str = INDEPENDENT_UNIT_COLUMN,
) -> dict[str, int | str]:
    if unit_column not in table.columns:
        raise KeyError(f"{table_name} is missing the independent unit column: {unit_column}")

    n_rows = len(table)
    n_units = table[unit_column].nunique(dropna=False)
    duplicate_mask = table[unit_column].duplicated(keep=False)
    if duplicate_mask.any():
        duplicate_units = table.loc[duplicate_mask, unit_column].drop_duplicates().head(10).tolist()
        raise ValueError(
            f"{table_name} is not dataset-level: {n_rows} rows but {n_units} unique "
            f"{unit_column} values. Repeated examples: {duplicate_units}. "
            "Aggregate repeated datasets or switch to a cluster-aware bootstrap before association testing."
        )

    return {
        "table": table_name,
        "independent_unit": unit_column,
        "rows": n_rows,
        "unique_units": n_units,
    }


def _bootstrap_spearman_summary(
    x_values: np.ndarray,
    y_values: np.ndarray,
    *,
    rng: np.random.Generator,
    n_repeats: int,
    alpha: float,
    observed_r: float,
) -> tuple[float, float, float, int]:
    n = len(x_values)
    bootstrap_r = []
    for _ in range(n_repeats):
        sample_idx = rng.integers(0, n, size=n)
        sample_x = x_values[sample_idx]
        sample_y = y_values[sample_idx]
        if np.unique(sample_x).size < 2 or np.unique(sample_y).size < 2:
            continue
        statistic = stats.spearmanr(sample_x, sample_y).statistic
        if np.isfinite(statistic):
            bootstrap_r.append(float(statistic))

    if len(bootstrap_r) < 20:
        return np.nan, np.nan, np.nan, len(bootstrap_r)

    bootstrap_r = np.asarray(bootstrap_r, dtype=float)
    ci_low, ci_high = np.quantile(bootstrap_r, [alpha / 2, 1 - alpha / 2])
    observed_sign = np.sign(observed_r)
    sign_consistency = np.mean(np.sign(bootstrap_r) == observed_sign) if observed_sign != 0 else np.nan
    return float(ci_low), float(ci_high), float(sign_consistency), len(bootstrap_r)


def _bootstrap_rank_stability(
    table: pd.DataFrame,
    associations: pd.DataFrame,
    *,
    target: str,
    min_n: int,
    n_repeats: int,
    top_k: int,
    random_seed: int,
) -> pd.DataFrame:
    tested = associations.loc[associations["spearman_r"].notna(), ["feature", "spearman_r"]]
    if tested.empty:
        return pd.DataFrame(
            columns=[
                "feature",
                "bootstrap_rank_median",
                "bootstrap_rank_q05",
                "bootstrap_rank_q95",
                "bootstrap_top_k_frequency",
            ]
        )

    tracked_features = tested.head(top_k)["feature"].tolist()
    rank_records = {feature: [] for feature in tracked_features}
    top_k_hits = {feature: 0 for feature in tracked_features}
    feature_values = {
        feature: pd.to_numeric(table[feature], errors="coerce").to_numpy(dtype=float) for feature in tested["feature"]
    }
    target_values = pd.to_numeric(table[target], errors="coerce").to_numpy(dtype=float)
    rng = np.random.default_rng(random_seed)

    for _ in range(n_repeats):
        sample_idx = rng.integers(0, len(table), size=len(table))
        replicate_rows = []
        sample_y_all = target_values[sample_idx]
        for feature, values in feature_values.items():
            sample_x_all = values[sample_idx]
            valid = np.isfinite(sample_x_all) & np.isfinite(sample_y_all)
            if valid.sum() < min_n:
                continue
            sample_x = sample_x_all[valid]
            sample_y = sample_y_all[valid]
            if np.unique(sample_x).size < 2 or np.unique(sample_y).size < 2:
                continue
            statistic = stats.spearmanr(sample_x, sample_y).statistic
            if np.isfinite(statistic):
                replicate_rows.append((feature, abs(float(statistic))))

        if not replicate_rows:
            continue

        replicate_ranks = pd.Series({feature: abs_r for feature, abs_r in replicate_rows}).rank(
            method="first", ascending=False
        )
        replicate_top = set(replicate_ranks[replicate_ranks <= top_k].index)
        for feature in tracked_features:
            if feature in replicate_ranks.index:
                rank_records[feature].append(float(replicate_ranks[feature]))
                top_k_hits[feature] += int(feature in replicate_top)

    rows = []
    for feature in tracked_features:
        ranks = np.asarray(rank_records[feature], dtype=float)
        if len(ranks) < 20:
            rank_median = np.nan
            rank_q05 = np.nan
            rank_q95 = np.nan
            top_k_frequency = np.nan
        else:
            rank_q05, rank_median, rank_q95 = np.quantile(ranks, [0.05, 0.5, 0.95])
            top_k_frequency = top_k_hits[feature] / len(ranks)
        rows.append(
            {
                "feature": feature,
                "bootstrap_rank_median": rank_median,
                "bootstrap_rank_q05": rank_q05,
                "bootstrap_rank_q95": rank_q95,
                "bootstrap_top_k_frequency": top_k_frequency,
            }
        )
    return pd.DataFrame(rows)


def _infer_association_feature_columns(table: pd.DataFrame, *, target: str) -> list[str]:
    feature_columns = []
    for column in table.columns:
        if column == target or column in EXCLUDED_PREDICTOR_COLUMNS or column.startswith("best_"):
            continue
        numeric_values = pd.to_numeric(table[column], errors="coerce")
        if pd.api.types.is_numeric_dtype(table[column]) or numeric_values.notna().any():
            feature_columns.append(column)
    return feature_columns


def estimate_feature_associations(
    table: pd.DataFrame,
    *,
    table_name: str,
    target: str = ASSOCIATION_TARGET,
    min_n: int = ASSOCIATION_MIN_N,
    bootstrap_repeats: int = ASSOCIATION_BOOTSTRAP_REPEATS,
    ci_top_k: int | None = ASSOCIATION_CI_TOP_K,
    rank_stability_top_k: int = ASSOCIATION_RANK_STABILITY_TOP_K,
    alpha: float = ASSOCIATION_ALPHA,
    random_seed: int = ASSOCIATION_RANDOM_SEED,
    feature_columns: Sequence[str] | None = None,
) -> pd.DataFrame:
    assert_dataset_level_table(table, table_name=table_name)
    if target not in table.columns:
        raise KeyError(f"{table_name} is missing target column: {target}")
    if feature_columns is None:
        feature_columns = _infer_association_feature_columns(table, target=target)
    else:
        missing_features = sorted(set(feature_columns).difference(table.columns))
        if missing_features:
            raise KeyError(f"{table_name} is missing feature columns: {missing_features}")
        feature_columns = list(feature_columns)

    target_values = pd.to_numeric(table[target], errors="coerce")
    rows = []

    for feature in feature_columns:
        feature_values = pd.to_numeric(table[feature], errors="coerce")
        valid = feature_values.notna() & target_values.notna()
        x = feature_values.loc[valid].to_numpy(dtype=float)
        y = target_values.loc[valid].to_numpy(dtype=float)
        n = len(x)
        feature_n_unique = int(pd.Series(x).nunique(dropna=True)) if n else 0
        target_n_unique = int(pd.Series(y).nunique(dropna=True)) if n else 0

        if n < min_n:
            spearman_r = np.nan
            p_value = np.nan
            reason = f"n < {min_n}"
        elif feature_n_unique < 2:
            spearman_r = np.nan
            p_value = np.nan
            reason = "feature has <2 observed values"
        elif target_n_unique < 2:
            spearman_r = np.nan
            p_value = np.nan
            reason = "target has <2 observed values"
        else:
            result = stats.spearmanr(x, y)
            spearman_r = float(result.statistic)
            p_value = float(result.pvalue)
            reason = "tested"

        rows.append(
            {
                "table": table_name,
                "feature": feature,
                "n": n,
                "feature_n_unique": feature_n_unique,
                "spearman_r": spearman_r,
                "abs_spearman_r": abs(spearman_r) if np.isfinite(spearman_r) else np.nan,
                "p_value": p_value,
                "p_value_bh": np.nan,
                "bh_reject_0_05": False,
                "ci_low": np.nan,
                "ci_high": np.nan,
                "bootstrap_sign_consistency": np.nan,
                "bootstrap_repeats_used": 0,
                "reason": reason,
            }
        )

    associations = pd.DataFrame(rows)
    valid_p = associations["p_value"].notna()
    if valid_p.any():
        reject, adjusted_p, _, _ = multipletests(
            associations.loc[valid_p, "p_value"],
            alpha=alpha,
            method="fdr_bh",
        )
        associations.loc[valid_p, "p_value_bh"] = adjusted_p
        associations.loc[valid_p, "bh_reject_0_05"] = reject

    associations = associations.sort_values(
        ["abs_spearman_r", "feature"],
        ascending=[False, True],
        na_position="last",
    ).reset_index(drop=True)
    associations["effect_rank"] = np.arange(1, len(associations) + 1)

    ci_features = associations.loc[associations["spearman_r"].notna(), "feature"]
    if ci_top_k is not None:
        ci_features = ci_features.head(ci_top_k)
    for rank_idx, feature in enumerate(ci_features):
        feature_values = pd.to_numeric(table[feature], errors="coerce")
        valid = feature_values.notna() & target_values.notna()
        x = feature_values.loc[valid].to_numpy(dtype=float)
        y = target_values.loc[valid].to_numpy(dtype=float)
        rng = np.random.default_rng(random_seed + rank_idx)
        observed_r = float(associations.loc[associations["feature"] == feature, "spearman_r"].iloc[0])
        ci_low, ci_high, sign_consistency, repeats_used = _bootstrap_spearman_summary(
            x,
            y,
            rng=rng,
            n_repeats=bootstrap_repeats,
            alpha=alpha,
            observed_r=observed_r,
        )
        row_idx = associations.index[associations["feature"] == feature][0]
        associations.loc[row_idx, "ci_low"] = ci_low
        associations.loc[row_idx, "ci_high"] = ci_high
        associations.loc[row_idx, "bootstrap_sign_consistency"] = sign_consistency
        associations.loc[row_idx, "bootstrap_repeats_used"] = repeats_used

    rank_stability = _bootstrap_rank_stability(
        table,
        associations,
        target=target,
        min_n=min_n,
        n_repeats=bootstrap_repeats,
        top_k=rank_stability_top_k,
        random_seed=random_seed + 100_000,
    )
    associations = associations.merge(rank_stability, on="feature", how="left")
    return associations


def build_robust_association_table(
    associations: pd.DataFrame,
    *,
    table_name: str,
    q_threshold: float = ROBUST_ASSOCIATION_Q_THRESHOLD,
    min_sign_consistency: float = ROBUST_ASSOCIATION_MIN_SIGN_CONSISTENCY,
    top_n: int = ROBUST_ASSOCIATION_TOP_N,
) -> pd.DataFrame:
    required_columns = {
        "feature",
        "n",
        "spearman_r",
        "abs_spearman_r",
        "p_value",
        "p_value_bh",
        "ci_low",
        "ci_high",
        "bootstrap_sign_consistency",
        "bootstrap_rank_median",
        "bootstrap_rank_q05",
        "bootstrap_rank_q95",
        "bootstrap_top_k_frequency",
    }
    missing_columns = sorted(required_columns.difference(associations.columns))
    if missing_columns:
        raise KeyError(f"{table_name} associations are missing required columns: {missing_columns}")

    tested = associations.loc[associations["p_value"].notna()].copy()
    robust = tested.loc[
        tested["p_value_bh"].lt(q_threshold) & tested["bootstrap_sign_consistency"].ge(min_sign_consistency)
    ].copy()

    robust = robust.sort_values(
        ["abs_spearman_r", "bootstrap_sign_consistency", "feature"],
        ascending=[False, False, True],
    )

    output_columns = [column for column in ROBUST_ASSOCIATION_OUTPUT_COLUMNS if column in robust.columns]
    table = robust.loc[:, output_columns].head(top_n).rename(columns=ROBUST_ASSOCIATION_COLUMN_NAMES)
    table.insert(0, "analysis_table", table_name)
    return table.reset_index(drop=True)
