from __future__ import annotations

import numpy as np
import pandas as pd

BASIC_METAFEATURE_SCHEMA_VERSION = 3
HIGH_CARDINALITY_MIN_UNIQUE = 50
HIGH_CARDINALITY_N_FRACTION = 0.1
NEAR_CONSTANT_FRACTION_THRESHOLD = 0.95
CLASSIFICATION_PROBLEM_TYPES = {"binary", "multiclass"}
BASIC_METAFEATURE_COLUMNS = (
    "n",
    "d",
    "log_n",
    "log_d",
    "n_over_d",
    "d_over_n",
    "n_num_features",
    "n_cat_features",
    "num_fraction",
    "cat_fraction",
    "missing_fraction",
    "n_classes",
    "class_entropy",
    "majority_class_fraction",
    "minority_class_fraction",
    "class_imbalance_ratio",
    "mean_cat_cardinality",
    "max_cat_cardinality",
    "high_cardinality_fraction",
    "cat_cardinality_to_n_ratio",
    "row_missing_fraction",
    "feature_missing_fraction",
    "num_missing_fraction",
    "cat_missing_fraction",
    "max_feature_missing_fraction",
    "mean_abs_skew",
    "max_abs_skew",
    "mean_kurtosis",
    "outlier_fraction_iqr",
    "zero_fraction",
    "constant_feature_fraction",
    "near_constant_feature_fraction",
)


def get_categorical_columns(X: pd.DataFrame) -> list[str]:
    """Return columns treated as categorical for meta-feature computation."""
    cat_mask = X.dtypes.apply(
        lambda dtype: (
            pd.api.types.is_object_dtype(dtype)
            or isinstance(dtype, pd.CategoricalDtype)
            or pd.api.types.is_bool_dtype(dtype)
        )
    )
    return X.columns[cat_mask].tolist()


def _nan_target_features() -> dict[str, float]:
    return {
        "n_classes": np.nan,
        "class_entropy": np.nan,
        "majority_class_fraction": np.nan,
        "minority_class_fraction": np.nan,
        "class_imbalance_ratio": np.nan,
    }


def _compute_target_features(y_train: pd.Series | None, problem_type: str | None) -> dict[str, float]:
    if y_train is None or problem_type not in CLASSIFICATION_PROBLEM_TYPES:
        return _nan_target_features()
    y = pd.Series(y_train).dropna()
    class_fractions = y.value_counts(normalize=True, dropna=True)
    if class_fractions.empty:
        return _nan_target_features()
    entropy = -(class_fractions * np.log2(class_fractions)).sum()
    majority_fraction = float(class_fractions.max())
    minority_fraction = float(class_fractions.min())
    return {
        "n_classes": int(len(class_fractions)),
        "class_entropy": float(entropy),
        "majority_class_fraction": majority_fraction,
        "minority_class_fraction": minority_fraction,
        "class_imbalance_ratio": float(majority_fraction / minority_fraction) if minority_fraction > 0 else np.nan,
    }


def _compute_categorical_cardinality_features(X_train: pd.DataFrame, cat_cols: list[str]) -> dict[str, float]:
    if not cat_cols:
        return {
            "mean_cat_cardinality": np.nan,
            "max_cat_cardinality": np.nan,
            "high_cardinality_fraction": np.nan,
            "cat_cardinality_to_n_ratio": np.nan,
        }

    n = len(X_train)
    cardinalities = X_train.loc[:, cat_cols].nunique(dropna=True).astype(float)
    high_threshold = max(HIGH_CARDINALITY_MIN_UNIQUE, HIGH_CARDINALITY_N_FRACTION * n)
    return {
        "mean_cat_cardinality": float(cardinalities.mean()),
        "max_cat_cardinality": float(cardinalities.max()),
        "high_cardinality_fraction": float((cardinalities >= high_threshold).mean()),
        "cat_cardinality_to_n_ratio": float(cardinalities.mean() / n) if n > 0 else np.nan,
    }


def _compute_missingness_features(X_train: pd.DataFrame, cat_cols: list[str], num_cols: list[str]) -> dict[str, float]:
    n, d = X_train.shape
    if d == 0:
        return {
            "row_missing_fraction": np.nan,
            "feature_missing_fraction": np.nan,
            "num_missing_fraction": np.nan,
            "cat_missing_fraction": np.nan,
            "max_feature_missing_fraction": np.nan,
        }

    missing = X_train.isna()
    return {
        "row_missing_fraction": float(missing.any(axis=1).mean()) if n > 0 else np.nan,
        "feature_missing_fraction": float(missing.any(axis=0).mean()),
        "num_missing_fraction": float(missing.loc[:, num_cols].mean().mean()) if num_cols else np.nan,
        "cat_missing_fraction": float(missing.loc[:, cat_cols].mean().mean()) if cat_cols else np.nan,
        "max_feature_missing_fraction": float(missing.mean(axis=0).max()),
    }


def _numeric_frame(X_train: pd.DataFrame, num_cols: list[str]) -> pd.DataFrame:
    if not num_cols:
        return pd.DataFrame(index=X_train.index)
    return X_train.loc[:, num_cols].apply(pd.to_numeric, errors="coerce")


def _compute_distribution_features(X_num: pd.DataFrame) -> dict[str, float]:
    if X_num.shape[1] == 0:
        return {
            "mean_abs_skew": np.nan,
            "max_abs_skew": np.nan,
            "mean_kurtosis": np.nan,
            "outlier_fraction_iqr": np.nan,
            "zero_fraction": np.nan,
        }

    skewness = X_num.skew(axis=0).replace([np.inf, -np.inf], np.nan).abs()
    kurtosis = X_num.kurt(axis=0).replace([np.inf, -np.inf], np.nan)

    q1 = X_num.quantile(0.25)
    q3 = X_num.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - (1.5 * iqr)
    upper = q3 + (1.5 * iqr)
    outlier_mask = X_num.lt(lower, axis=1) | X_num.gt(upper, axis=1)
    valid_mask = X_num.notna()
    valid_count = int(valid_mask.to_numpy().sum())

    return {
        "mean_abs_skew": float(skewness.mean()) if skewness.notna().any() else np.nan,
        "max_abs_skew": float(skewness.max()) if skewness.notna().any() else np.nan,
        "mean_kurtosis": float(kurtosis.mean()) if kurtosis.notna().any() else np.nan,
        "outlier_fraction_iqr": float((outlier_mask & valid_mask).to_numpy().sum() / valid_count)
        if valid_count > 0
        else np.nan,
        "zero_fraction": float((X_num == 0).to_numpy().sum() / valid_count) if valid_count > 0 else np.nan,
    }


def _compute_low_information_features(X_train: pd.DataFrame) -> dict[str, float]:
    _, d = X_train.shape
    if d == 0:
        return {
            "constant_feature_fraction": np.nan,
            "near_constant_feature_fraction": np.nan,
        }

    constant_count = 0
    near_constant_count = 0
    for column in X_train.columns:
        values = X_train[column].dropna()
        if values.empty or values.nunique(dropna=True) <= 1:
            constant_count += 1
            near_constant_count += 1
            continue
        top_frequency = float(values.value_counts(normalize=True, dropna=True).iloc[0])
        if top_frequency >= NEAR_CONSTANT_FRACTION_THRESHOLD:
            near_constant_count += 1
    return {
        "constant_feature_fraction": float(constant_count / d),
        "near_constant_feature_fraction": float(near_constant_count / d),
    }


def compute_basic_metafeatures(
    X_train: pd.DataFrame,
    y_train: pd.Series | None = None,
    problem_type: str | None = None,
) -> dict[str, float]:
    """Compute cheap, interpretable size, composition, target, and shape meta-features."""
    n, d = X_train.shape
    normalized_problem_type = problem_type.lower() if isinstance(problem_type, str) else None
    cat_cols = get_categorical_columns(X_train)
    num_cols = [column for column in X_train.columns if column not in cat_cols]
    X_num = _numeric_frame(X_train, num_cols)

    features = {
        "n": int(n),
        "d": int(d),
        "log_n": float(np.log10(n)) if n > 0 else np.nan,
        "log_d": float(np.log10(d)) if d > 0 else np.nan,
        "n_over_d": float(n / d) if d > 0 else np.nan,
        "d_over_n": float(d / n) if n > 0 else np.nan,
        "n_num_features": int(len(num_cols)),
        "n_cat_features": int(len(cat_cols)),
        "num_fraction": float(len(num_cols) / d) if d > 0 else np.nan,
        "cat_fraction": float(len(cat_cols) / d) if d > 0 else np.nan,
        "missing_fraction": float(X_train.isna().mean().mean()) if d > 0 else np.nan,
    }
    features.update(_compute_target_features(y_train, normalized_problem_type))
    features.update(_compute_categorical_cardinality_features(X_train, cat_cols))
    features.update(_compute_missingness_features(X_train, cat_cols, num_cols))
    features.update(_compute_distribution_features(X_num))
    features.update(_compute_low_information_features(X_train))
    return features
