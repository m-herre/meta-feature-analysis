from __future__ import annotations

import numpy as np
import pandas as pd

SENSITIVITY_TARGET = "delta_norm"
SENSITIVITY_UNIT_COLUMN = "dataset"
SENSITIVITY_MIN_N = 30
SENSITIVITY_ALPHA = 0.05
SENSITIVITY_BOOTSTRAP_REPEATS = 500
SENSITIVITY_RANDOM_SEED = 20260424
SENSITIVITY_MAX_TOTAL_PREDICTORS = 8
SENSITIVITY_MAX_FEATURE_PREDICTORS = 3
SENSITIVITY_MIN_SIGN_CONSISTENCY = 0.90

SENSITIVITY_GENERAL_CONTROL_CANDIDATES = [
    ("n_samples", ("n_samples", "n_instances", "n_rows", "n")),
    ("n_features", ("n_features", "n_columns", "n_attrs", "n_attr", "d")),
    (
        "feature_sample_ratio",
        ("d_over_n", "n_features_over_n_samples", "features_per_sample"),
    ),
]

SENSITIVITY_CLASSIFICATION_CONTROL_CANDIDATES = [
    ("n_classes", ("n_classes", "nr_classes")),
    (
        "class_imbalance_ratio",
        ("class_imbalance_ratio", "imbalance_ratio", "majority_minority_ratio"),
    ),
]


def _sensitivity_feature_family(feature: str) -> str:
    if feature.startswith("pymfe__"):
        return feature.split(".", maxsplit=1)[0]
    return feature


def _select_one_feature_per_family(
    robust_table: pd.DataFrame,
    *,
    max_features: int,
) -> tuple[list[str], pd.DataFrame]:
    if robust_table.empty or max_features <= 0:
        return [], pd.DataFrame(columns=["feature", "feature_family", "selection_status"])

    required_columns = {"feature", "abs_spearman_rho", "bootstrap_sign_consistency"}
    missing_columns = sorted(required_columns.difference(robust_table.columns))
    if missing_columns:
        raise KeyError(f"Robust association table is missing columns: {missing_columns}")

    candidates = robust_table.copy()
    candidates["feature_family"] = candidates["feature"].map(_sensitivity_feature_family)
    candidates = candidates.sort_values(
        ["abs_spearman_rho", "bootstrap_sign_consistency", "feature"],
        ascending=[False, False, True],
    )
    family_representatives = candidates.drop_duplicates("feature_family", keep="first").copy()
    selected = family_representatives.head(max_features).copy()
    selected_features = selected["feature"].tolist()

    report = candidates.loc[:, ["feature", "feature_family"]].copy()
    report["selection_status"] = np.select(
        [
            report["feature"].isin(selected_features),
            report["feature_family"].isin(selected["feature_family"]),
        ],
        ["selected", "same_family_as_selected"],
        default="excluded_by_model_cap",
    )
    return selected_features, report


def _assert_dataset_level_for_sensitivity(table: pd.DataFrame, *, table_name: str) -> None:
    if SENSITIVITY_UNIT_COLUMN not in table.columns:
        raise KeyError(f"{table_name} is missing {SENSITIVITY_UNIT_COLUMN!r}")
    duplicates = table[SENSITIVITY_UNIT_COLUMN].duplicated(keep=False)
    if duplicates.any():
        examples = table.loc[duplicates, SENSITIVITY_UNIT_COLUMN].drop_duplicates().head(10).tolist()
        raise ValueError(f"{table_name} is not dataset-level; repeated {SENSITIVITY_UNIT_COLUMN} examples: {examples}")


def _available_controls(
    control_source_table: pd.DataFrame,
    *,
    include_classification_controls: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    candidate_groups = list(SENSITIVITY_GENERAL_CONTROL_CANDIDATES)
    if include_classification_controls:
        candidate_groups.extend(SENSITIVITY_CLASSIFICATION_CONTROL_CANDIDATES)

    controls = pd.DataFrame({SENSITIVITY_UNIT_COLUMN: control_source_table[SENSITIVITY_UNIT_COLUMN]})
    report_rows = []
    for control_name, candidates in candidate_groups:
        source_column = next((column for column in candidates if column in control_source_table.columns), None)
        if source_column is None:
            report_rows.append(
                {
                    "control": control_name,
                    "source_column": np.nan,
                    "included": False,
                    "reason": "not_available",
                }
            )
            continue

        values = pd.to_numeric(control_source_table[source_column], errors="coerce")
        if values.notna().sum() < SENSITIVITY_MIN_N or values.nunique(dropna=True) < 2:
            report_rows.append(
                {
                    "control": control_name,
                    "source_column": source_column,
                    "included": False,
                    "reason": "insufficient_variation_or_n",
                }
            )
            continue

        control_column = f"control__{control_name}"
        controls[control_column] = values.replace([np.inf, -np.inf], np.nan)
        report_rows.append(
            {
                "control": control_name,
                "source_column": source_column,
                "model_column": control_column,
                "included": True,
                "reason": "fixed_a_priori",
            }
        )

    return controls, pd.DataFrame(report_rows)


def _rank_zscore(series: pd.Series) -> pd.Series:
    ranked = pd.to_numeric(series, errors="coerce").rank(method="average", pct=True)
    scale = ranked.std(ddof=0)
    if not np.isfinite(scale) or scale == 0:
        return pd.Series(np.nan, index=series.index, dtype=float)
    return (ranked - ranked.mean()) / scale


def _fit_rank_ols(
    data: pd.DataFrame,
    *,
    target: str,
    predictors: list[str],
) -> dict[str, object] | None:
    clean = data.loc[:, [target, *predictors]].replace([np.inf, -np.inf], np.nan).dropna()
    if len(clean) < SENSITIVITY_MIN_N or len(clean) <= len(predictors) + 2:
        return None

    y = _rank_zscore(clean[target])
    x_columns = []
    x_arrays = []
    for predictor in predictors:
        transformed = _rank_zscore(clean[predictor])
        if transformed.notna().all():
            x_columns.append(predictor)
            x_arrays.append(transformed.to_numpy(dtype=float))

    if not x_columns or len(clean) <= len(x_columns) + 2:
        return None

    x = np.column_stack([np.ones(len(clean)), *x_arrays])
    coefficients, *_ = np.linalg.lstsq(x, y.to_numpy(dtype=float), rcond=None)
    return {
        "n": int(len(clean)),
        "predictors": x_columns,
        "coefficients": dict(zip(x_columns, coefficients[1:], strict=True)),
        "condition_number": float(np.linalg.cond(x)),
    }


def _bootstrap_rank_ols(
    clean: pd.DataFrame,
    *,
    target: str,
    predictors: list[str],
    observed_coefficients: dict[str, float],
    rng: np.random.Generator,
) -> dict[str, dict[str, float]]:
    bootstrap_coefficients = {predictor: [] for predictor in observed_coefficients}
    for _ in range(SENSITIVITY_BOOTSTRAP_REPEATS):
        sample = clean.iloc[rng.integers(0, len(clean), size=len(clean))].reset_index(drop=True)
        fit = _fit_rank_ols(sample, target=target, predictors=predictors)
        if fit is None:
            continue
        for predictor, coefficient in fit["coefficients"].items():
            if predictor in bootstrap_coefficients and np.isfinite(coefficient):
                bootstrap_coefficients[predictor].append(float(coefficient))

    summaries = {}
    for predictor, values in bootstrap_coefficients.items():
        values = np.asarray(values, dtype=float)
        observed = float(observed_coefficients[predictor])
        observed_sign = np.sign(observed)
        if len(values) < 20 or observed_sign == 0:
            summaries[predictor] = {
                "adjusted_ci_low": np.nan,
                "adjusted_ci_high": np.nan,
                "adjusted_bootstrap_sign_consistency": np.nan,
                "adjusted_bootstrap_repeats": int(len(values)),
            }
            continue
        ci_low, ci_high = np.quantile(values, [SENSITIVITY_ALPHA / 2, 1 - SENSITIVITY_ALPHA / 2])
        summaries[predictor] = {
            "adjusted_ci_low": float(ci_low),
            "adjusted_ci_high": float(ci_high),
            "adjusted_bootstrap_sign_consistency": float(np.mean(np.sign(values) == observed_sign)),
            "adjusted_bootstrap_repeats": int(len(values)),
        }
    return summaries


def run_multivariable_sensitivity(
    feature_table: pd.DataFrame,
    robust_table: pd.DataFrame,
    *,
    table_name: str,
    control_source_table: pd.DataFrame | None = None,
    include_classification_controls: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    _assert_dataset_level_for_sensitivity(feature_table, table_name=table_name)
    control_source_table = feature_table if control_source_table is None else control_source_table
    _assert_dataset_level_for_sensitivity(control_source_table, table_name=f"{table_name} controls")

    controls, control_report = _available_controls(
        control_source_table,
        include_classification_controls=include_classification_controls,
    )
    control_columns = [column for column in controls.columns if column != SENSITIVITY_UNIT_COLUMN]
    max_features = max(
        0,
        min(
            SENSITIVITY_MAX_FEATURE_PREDICTORS,
            SENSITIVITY_MAX_TOTAL_PREDICTORS - len(control_columns),
        ),
    )
    selected_features, feature_report = _select_one_feature_per_family(
        robust_table,
        max_features=max_features,
    )

    feature_report = feature_report.assign(analysis_table=table_name)
    control_report = control_report.assign(analysis_table=table_name)
    if not selected_features:
        return (
            pd.DataFrame(
                columns=[
                    "analysis_table",
                    "feature",
                    "feature_family",
                    "n",
                    "univariate_spearman_rho",
                    "adjusted_rank_coefficient",
                    "adjusted_ci_low",
                    "adjusted_ci_high",
                    "adjusted_bootstrap_sign_consistency",
                    "adjusted_bootstrap_repeats",
                    "adjusted_direction_matches_univariate",
                    "adjusted_direction_stable",
                    "predictors_in_model",
                    "controls_in_model",
                    "condition_number",
                ]
            ),
            feature_report,
            control_report,
        )

    missing_features = [feature for feature in selected_features if feature not in feature_table.columns]
    if missing_features:
        raise KeyError(f"Selected robust features are missing from {table_name}: {missing_features}")

    model_data = feature_table.loc[:, [SENSITIVITY_UNIT_COLUMN, SENSITIVITY_TARGET, *selected_features]].merge(
        controls,
        on=SENSITIVITY_UNIT_COLUMN,
        how="left",
        validate="one_to_one",
    )
    predictors = [*selected_features, *control_columns]
    clean_model_data = model_data.loc[:, [SENSITIVITY_TARGET, *predictors]].replace([np.inf, -np.inf], np.nan).dropna()
    fit = _fit_rank_ols(clean_model_data, target=SENSITIVITY_TARGET, predictors=predictors)
    if fit is None:
        raise ValueError(
            f"{table_name}: sensitivity model has insufficient complete cases after fixed controls are added."
        )

    table_seed_offset = sum((idx + 1) * ord(char) for idx, char in enumerate(table_name))
    rng = np.random.default_rng(SENSITIVITY_RANDOM_SEED + table_seed_offset)
    bootstrap_summary = _bootstrap_rank_ols(
        clean_model_data,
        target=SENSITIVITY_TARGET,
        predictors=fit["predictors"],
        observed_coefficients=fit["coefficients"],
        rng=rng,
    )

    robust_lookup = robust_table.set_index("feature")
    controls_in_fit = [predictor for predictor in fit["predictors"] if predictor in control_columns]
    rows = []
    for feature in selected_features:
        if feature not in fit["coefficients"]:
            continue
        adjusted_coefficient = float(fit["coefficients"][feature])
        univariate_rho = float(robust_lookup.loc[feature, "spearman_rho"])
        direction_matches = bool(np.sign(adjusted_coefficient) == np.sign(univariate_rho))
        summary = bootstrap_summary.get(feature, {})
        adjusted_sign_consistency = summary.get("adjusted_bootstrap_sign_consistency", np.nan)
        rows.append(
            {
                "analysis_table": table_name,
                "feature": feature,
                "feature_family": _sensitivity_feature_family(feature),
                "n": fit["n"],
                "univariate_spearman_rho": univariate_rho,
                "adjusted_rank_coefficient": adjusted_coefficient,
                "adjusted_ci_low": summary.get("adjusted_ci_low", np.nan),
                "adjusted_ci_high": summary.get("adjusted_ci_high", np.nan),
                "adjusted_bootstrap_sign_consistency": adjusted_sign_consistency,
                "adjusted_bootstrap_repeats": summary.get("adjusted_bootstrap_repeats", np.nan),
                "adjusted_direction_matches_univariate": direction_matches,
                "adjusted_direction_stable": bool(
                    direction_matches and adjusted_sign_consistency >= SENSITIVITY_MIN_SIGN_CONSISTENCY
                ),
                "predictors_in_model": len(fit["predictors"]),
                "controls_in_model": ", ".join(controls_in_fit),
                "condition_number": fit["condition_number"],
            }
        )

    return pd.DataFrame(rows), feature_report, control_report
