from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import skew

DEFAULT_IRREGULARITY_COMPONENTS = (
    "irreg_min_cov_eig",
    "irreg_std_skew",
    "irreg_range_skew",
    "irreg_kurtosis_std",
)


def safe_skew(values) -> float:
    """Return sample skewness or NaN when undefined."""
    values = pd.Series(values).replace([np.inf, -np.inf], np.nan).dropna()
    if len(values) < 3 or values.nunique() <= 1:
        return np.nan
    return float(skew(values, bias=False))


def compute_irregularity_components(X_num: pd.DataFrame) -> dict[str, float]:
    """Compute the irregularity proxy components on numeric features only."""
    X_num = X_num.apply(pd.to_numeric, errors="coerce")
    X_num = X_num.loc[:, X_num.nunique(dropna=True) > 1]

    if X_num.shape[1] == 0:
        return {
            "irreg_min_cov_eig": np.nan,
            "irreg_std_skew": np.nan,
            "irreg_range_skew": np.nan,
            "irreg_kurtosis_std": np.nan,
        }

    feature_stds = X_num.std(axis=0, ddof=1)
    feature_ranges = X_num.max(axis=0) - X_num.min(axis=0)
    feature_kurtosis = X_num.kurt(axis=0)

    cov_eig_min = np.nan
    X_scaled = (X_num - X_num.mean()) / X_num.std(axis=0, ddof=0)
    X_scaled = X_scaled.replace([np.inf, -np.inf], np.nan).dropna(axis=0, how="any")
    if X_scaled.shape[0] >= 2 and X_scaled.shape[1] >= 2:
        cov = np.atleast_2d(np.cov(X_scaled.to_numpy(), rowvar=False))
        cov_eig_min = float(np.linalg.eigvalsh(cov).min())

    return {
        "irreg_min_cov_eig": cov_eig_min,
        "irreg_std_skew": safe_skew(feature_stds),
        "irreg_range_skew": safe_skew(feature_ranges),
        "irreg_kurtosis_std": float(feature_kurtosis.std(ddof=1)),
    }


def zscore(series: pd.Series) -> pd.Series:
    """Return the population z-score of a series."""
    std = series.std(ddof=0)
    if pd.isna(std) or std == 0:
        return pd.Series(np.nan, index=series.index, dtype=float)
    return (series - series.mean()) / std


def add_irregularity_proxy(
    df: pd.DataFrame,
    *,
    components: tuple[str, ...] = DEFAULT_IRREGULARITY_COMPONENTS,
    output_col: str = "irregularity",
) -> pd.DataFrame:
    """Combine irregularity component z-scores into one NaN-aware proxy."""
    result = df.copy()
    available = [column for column in components if column in result.columns]
    z_scores = pd.DataFrame({column: zscore(result[column]) for column in available})
    if "irreg_min_cov_eig" in z_scores.columns:
        z_scores["irreg_min_cov_eig"] = -z_scores["irreg_min_cov_eig"]
    result[output_col] = z_scores.mean(axis=1, skipna=True)
    return result

