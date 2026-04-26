from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.dummy import DummyRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

try:
    from xgboost import XGBRegressor
except ImportError:
    XGBRegressor = None

PREDICTIVE_TARGET = "delta_norm"
PREDICTIVE_UNIT_COLUMN = "dataset"
PREDICTIVE_MIN_N = 30
PREDICTIVE_RIDGE_ALPHAS = np.logspace(-3, 3, 13)
PREDICTIVE_TOP_COEFFICIENTS = 20
PREDICTIVE_TREE_MAX_DEPTH = 2
PREDICTIVE_TREE_MIN_SAMPLES_LEAF = 5
PREDICTIVE_RANDOM_STATE = 20260424

PREDICTIVE_GENERAL_CONTROL_CANDIDATES = [
    ("n_samples", ("n_samples", "n_instances", "n_rows", "n")),
    ("n_features", ("n_features", "n_columns", "n_attrs", "n_attr", "d")),
    (
        "feature_sample_ratio",
        ("d_over_n", "n_features_over_n_samples", "features_per_sample"),
    ),
]
PREDICTIVE_CLASSIFICATION_CONTROL_CANDIDATES = [
    ("n_classes", ("n_classes", "nr_classes")),
    (
        "class_imbalance_ratio",
        ("class_imbalance_ratio", "imbalance_ratio", "majority_minority_ratio"),
    ),
]


def _predictive_context_columns(context_columns: Sequence[str] = ()) -> set[str]:
    context = set(context_columns)
    context.update(
        {
            PREDICTIVE_UNIT_COLUMN,
            PREDICTIVE_TARGET,
            "comparison_name",
            "task_type",
            "dataset",
        }
    )
    return context


def _assert_dataset_level_for_prediction(
    table: pd.DataFrame,
    *,
    table_name: str,
) -> dict[str, int | str]:
    if PREDICTIVE_UNIT_COLUMN not in table.columns:
        raise KeyError(f"{table_name} is missing {PREDICTIVE_UNIT_COLUMN!r}.")
    n_rows = len(table)
    n_units = table[PREDICTIVE_UNIT_COLUMN].nunique(dropna=False)
    duplicate_mask = table[PREDICTIVE_UNIT_COLUMN].duplicated(keep=False)
    if duplicate_mask.any():
        duplicate_units = table.loc[duplicate_mask, PREDICTIVE_UNIT_COLUMN].drop_duplicates().head(10).tolist()
        raise ValueError(
            f"{table_name} is not dataset-level: {n_rows} rows but {n_units} unique "
            f"{PREDICTIVE_UNIT_COLUMN} values. Repeated examples: {duplicate_units}."
        )
    return {
        "analysis_table": table_name,
        "independent_unit": PREDICTIVE_UNIT_COLUMN,
        "rows": n_rows,
        "unique_units": n_units,
    }


def _numeric_usable(series: pd.Series, *, min_n: int = 3) -> bool:
    values = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)
    values = values.dropna()
    return len(values) >= min_n and values.nunique() > 1


def _feature_columns_for_prediction(table: pd.DataFrame, *, context_columns: Sequence[str] = ()) -> list[str]:
    context = _predictive_context_columns(context_columns)
    columns = []
    for column in table.columns:
        if column in context:
            continue
        if _numeric_usable(table[column]):
            columns.append(column)
    return columns


def _available_controls(
    table: pd.DataFrame,
    *,
    include_classification_controls: bool,
) -> tuple[list[str], pd.DataFrame]:
    candidates = list(PREDICTIVE_GENERAL_CONTROL_CANDIDATES)
    if include_classification_controls:
        candidates.extend(PREDICTIVE_CLASSIFICATION_CONTROL_CANDIDATES)

    selected = []
    rows = []
    for control_name, aliases in candidates:
        present_alias = next(
            (alias for alias in aliases if alias in table.columns and _numeric_usable(table[alias])),
            None,
        )
        if present_alias is not None:
            selected.append(present_alias)
        rows.append(
            {
                "control": control_name,
                "selected_column": present_alias,
                "available": present_alias is not None,
            }
        )
    return selected, pd.DataFrame(rows)


def _ridge_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scaler", StandardScaler()),
            ("ridge", RidgeCV(alphas=PREDICTIVE_RIDGE_ALPHAS, cv=None)),
        ]
    )


def _tree_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            (
                "tree",
                DecisionTreeRegressor(
                    max_depth=PREDICTIVE_TREE_MAX_DEPTH,
                    min_samples_leaf=PREDICTIVE_TREE_MIN_SAMPLES_LEAF,
                    random_state=PREDICTIVE_RANDOM_STATE,
                ),
            ),
        ]
    )


def _xgboost_pipeline() -> Pipeline | None:
    if XGBRegressor is None:
        return None
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            (
                "xgboost",
                XGBRegressor(
                    objective="reg:squarederror",
                    n_estimators=50,
                    max_depth=1,
                    learning_rate=0.05,
                    subsample=0.80,
                    colsample_bytree=0.80,
                    min_child_weight=5,
                    reg_alpha=1.0,
                    reg_lambda=10.0,
                    random_state=PREDICTIVE_RANDOM_STATE,
                    n_jobs=1,
                    verbosity=0,
                ),
            ),
        ]
    )


def _estimator_for_model(model_family: str) -> Pipeline | DummyRegressor | None:
    if model_family == "mean":
        return DummyRegressor(strategy="mean")
    if model_family == "ridge":
        return _ridge_pipeline()
    if model_family == "tree":
        return _tree_pipeline()
    if model_family == "xgboost":
        return _xgboost_pipeline()
    raise ValueError(f"Unknown model family: {model_family!r}.")


def _model_predictions(
    data: pd.DataFrame,
    predictors: list[str],
    *,
    model_name: str,
    model_family: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    y = data[PREDICTIVE_TARGET].astype(float).to_numpy()
    prediction_rows = []
    coefficient_rows = []
    estimator_template = _estimator_for_model(model_family)

    if estimator_template is None:
        return pd.DataFrame(), pd.DataFrame()

    for test_idx in range(len(data)):
        train_idx = np.array([idx for idx in range(len(data)) if idx != test_idx])
        train = data.iloc[train_idx]
        test = data.iloc[[test_idx]]
        y_train = y[train_idx]

        if predictors:
            estimator = _estimator_for_model(model_family)
            if estimator is None:
                return pd.DataFrame(), pd.DataFrame()
            estimator.fit(train[predictors], y_train)
            y_pred = float(estimator.predict(test[predictors])[0])
            alpha = np.nan
            if model_family == "ridge":
                alpha = float(estimator.named_steps["ridge"].alpha_)
                coefficients = estimator.named_steps["ridge"].coef_
                for predictor, coefficient in zip(predictors, coefficients, strict=True):
                    coefficient_rows.append(
                        {
                            "model": model_name,
                            "held_out_dataset": test[PREDICTIVE_UNIT_COLUMN].iloc[0],
                            "predictor": predictor,
                            "coefficient": float(coefficient),
                            "alpha": alpha,
                        }
                    )
        else:
            estimator = _estimator_for_model("mean")
            estimator.fit(np.zeros((len(train), 1)), y_train)
            y_pred = float(estimator.predict(np.zeros((1, 1)))[0])
            alpha = np.nan

        prediction_rows.append(
            {
                "model": model_name,
                "model_family": model_family,
                "dataset": test[PREDICTIVE_UNIT_COLUMN].iloc[0],
                "y_true": float(y[test_idx]),
                "y_pred": y_pred,
                "alpha": alpha,
            }
        )

    return pd.DataFrame(prediction_rows), pd.DataFrame(coefficient_rows)


def _prediction_metrics(predictions: pd.DataFrame) -> dict[str, float | int]:
    if predictions.empty:
        return {
            "n": 0,
            "mae": np.nan,
            "rmse": np.nan,
            "oos_r2": np.nan,
            "spearman_pred_obs": np.nan,
            "sign_accuracy": np.nan,
            "median_alpha": np.nan,
        }

    y_true = predictions["y_true"].astype(float).to_numpy()
    y_pred = predictions["y_pred"].astype(float).to_numpy()
    sse = float(np.sum((y_true - y_pred) ** 2))
    sst = float(np.sum((y_true - y_true.mean()) ** 2))
    nonzero_mask = y_true != 0

    if np.unique(y_true).size > 1 and np.unique(y_pred).size > 1:
        spearman = float(stats.spearmanr(y_true, y_pred).statistic)
    else:
        spearman = np.nan

    return {
        "n": int(len(predictions)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(mean_squared_error(y_true, y_pred) ** 0.5),
        "oos_r2": float(1 - sse / sst) if sst > 0 else np.nan,
        "spearman_pred_obs": spearman,
        "sign_accuracy": (
            float(np.mean(np.sign(y_true[nonzero_mask]) == np.sign(y_pred[nonzero_mask])))
            if nonzero_mask.any()
            else np.nan
        ),
        "median_alpha": (
            float(predictions["alpha"].dropna().median()) if predictions["alpha"].notna().any() else np.nan
        ),
    }


def _coefficient_summary(
    coefficients: pd.DataFrame,
    *,
    feature_columns: list[str],
    top_n: int = PREDICTIVE_TOP_COEFFICIENTS,
) -> pd.DataFrame:
    if coefficients.empty:
        return pd.DataFrame()
    feature_set = set(feature_columns)
    coef = coefficients[coefficients["predictor"].isin(feature_set)].copy()
    if coef.empty:
        return pd.DataFrame()
    observed_direction = np.sign(coef["coefficient"])
    coef["nonzero_direction"] = observed_direction.where(observed_direction != 0, np.nan)
    summary = (
        coef.groupby(["model", "predictor"], as_index=False)
        .agg(
            median_standardized_coefficient=("coefficient", "median"),
            median_abs_standardized_coefficient=(
                "coefficient",
                lambda values: float(np.median(np.abs(values))),
            ),
            q05_standardized_coefficient=(
                "coefficient",
                lambda values: float(np.quantile(values, 0.05)),
            ),
            q95_standardized_coefficient=(
                "coefficient",
                lambda values: float(np.quantile(values, 0.95)),
            ),
            positive_fold_share=(
                "coefficient",
                lambda values: float(np.mean(np.asarray(values) > 0)),
            ),
            negative_fold_share=(
                "coefficient",
                lambda values: float(np.mean(np.asarray(values) < 0)),
            ),
        )
        .sort_values(
            ["model", "median_abs_standardized_coefficient", "predictor"],
            ascending=[True, False, True],
        )
    )
    summary["coefficient_sign_consistency"] = summary[["positive_fold_share", "negative_fold_share"]].max(axis=1)
    return summary.groupby("model", group_keys=False).head(top_n).reset_index(drop=True)


def run_predictive_meta_modeling(
    table: pd.DataFrame,
    *,
    table_name: str,
    include_classification_controls: bool,
    context_columns: Sequence[str] = (),
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    guard = pd.DataFrame([_assert_dataset_level_for_prediction(table, table_name=table_name)])
    if PREDICTIVE_TARGET not in table.columns:
        raise KeyError(f"{table_name} is missing target column {PREDICTIVE_TARGET!r}.")

    feature_columns = _feature_columns_for_prediction(table, context_columns=context_columns)
    control_columns, control_report = _available_controls(
        table,
        include_classification_controls=include_classification_controls,
    )
    feature_columns = [column for column in feature_columns if column not in set(control_columns)]
    control_report.insert(0, "analysis_table", table_name)

    modeling_columns = [
        PREDICTIVE_UNIT_COLUMN,
        PREDICTIVE_TARGET,
        *control_columns,
        *feature_columns,
    ]
    data = table.loc[:, list(dict.fromkeys(modeling_columns))].copy()
    for column in data.columns:
        if column == PREDICTIVE_UNIT_COLUMN:
            continue
        data[column] = pd.to_numeric(data[column], errors="coerce")
    data = data.replace([np.inf, -np.inf], np.nan)
    data = data[data[PREDICTIVE_TARGET].notna()].reset_index(drop=True)

    model_specs = [("mean_baseline", "mean", [])]
    if control_columns:
        model_specs.append(("ridge_controls", "ridge", control_columns))
    model_specs.append(("ridge_meta_features", "ridge", feature_columns))
    if control_columns:
        model_specs.append(
            (
                "ridge_controls_plus_meta_features",
                "ridge",
                control_columns + feature_columns,
            )
        )
    model_specs.append(("decision_tree_meta_features", "tree", feature_columns))
    if control_columns:
        model_specs.append(
            (
                "decision_tree_controls_plus_meta_features",
                "tree",
                control_columns + feature_columns,
            )
        )
    if XGBRegressor is not None:
        model_specs.append(("xgboost_meta_features", "xgboost", feature_columns))
        if control_columns:
            model_specs.append(
                (
                    "xgboost_controls_plus_meta_features",
                    "xgboost",
                    control_columns + feature_columns,
                )
            )

    prediction_frames = []
    coefficient_frames = []
    metric_rows = []
    for model_name, model_family, predictors in model_specs:
        if len(data) < PREDICTIVE_MIN_N:
            predictions = pd.DataFrame()
            coefficients = pd.DataFrame()
        else:
            predictions, coefficients = _model_predictions(
                data,
                predictors,
                model_name=model_name,
                model_family=model_family,
            )
        predictions.insert(0, "analysis_table", table_name)
        if not coefficients.empty:
            coefficients.insert(0, "analysis_table", table_name)
        prediction_frames.append(predictions)
        coefficient_frames.append(coefficients)
        row = {
            "analysis_table": table_name,
            "model": model_name,
            "model_family": model_family,
            "n_predictors": len(predictors),
        }
        row.update(_prediction_metrics(predictions))
        metric_rows.append(row)

    metrics = pd.DataFrame(metric_rows)
    if "mean_baseline" in set(metrics["model"]):
        baseline = metrics.set_index("model").loc["mean_baseline"]
        metrics["delta_mae_vs_baseline"] = baseline["mae"] - metrics["mae"]
        metrics["delta_rmse_vs_baseline"] = baseline["rmse"] - metrics["rmse"]

    predictions = pd.concat(prediction_frames, ignore_index=True)
    coefficients = (
        pd.concat(
            [frame for frame in coefficient_frames if not frame.empty],
            ignore_index=True,
        )
        if any(not frame.empty for frame in coefficient_frames)
        else pd.DataFrame()
    )
    coefficient_summary = _coefficient_summary(coefficients, feature_columns=feature_columns)
    if not coefficient_summary.empty:
        coefficient_summary.insert(0, "analysis_table", table_name)

    guard["n_modeling_rows"] = len(data)
    guard["n_feature_predictors"] = len(feature_columns)
    guard["n_control_predictors"] = len(control_columns)
    guard["xgboost_available"] = XGBRegressor is not None
    return metrics, predictions, coefficient_summary, control_report, guard
