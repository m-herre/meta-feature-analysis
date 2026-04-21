from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy.stats import hmean, skew

logger = logging.getLogger(__name__)

IRREGULARITY_PROXY_SCHEMA_VERSION = 3

# Weights and component order follow the source paper (Figure 5 / Section 2.2).
# Keys are the per-split component column names emitted by
# ``compute_irregularity_components``; values are the paper's coefficients on
# the per-dataset z-scored components.
IRREGULARITY_PAPER_WEIGHTS: dict[str, float] = {
    "irreg_min_cov_eig": -0.33,
    "irreg_std_skew": 0.23,
    "irreg_range_skew": 0.22,
    "irreg_iqr_hmean": 0.21,
    "irreg_kurtosis_std": 0.21,
}

DEFAULT_IRREGULARITY_COMPONENTS = tuple(IRREGULARITY_PAPER_WEIGHTS.keys())


def safe_skew(values) -> float:
    """Return sample skewness or NaN when undefined."""
    values = pd.Series(values).replace([np.inf, -np.inf], np.nan).dropna()
    if len(values) < 3 or values.nunique() <= 1:
        return np.nan
    return float(skew(values, bias=False))


def _iqr(values: pd.Series) -> float:
    """Interquartile range (Q3 - Q1) on finite values; NaN if fewer than 2 points.

    IQR with linear interpolation is well-defined at n >= 2. The lower bound
    keeps the harmonic-mean component available on datasets where only a
    handful of feature columns are strictly positive (e.g. after centering
    or when most columns are indicator-encoded).
    """
    finite = values.replace([np.inf, -np.inf], np.nan).dropna()
    if len(finite) < 2:
        return np.nan
    q1, q3 = np.quantile(finite.to_numpy(), [0.25, 0.75])
    return float(q3 - q1)


def _per_feature_hmeans(X_num: pd.DataFrame) -> pd.Series:
    """Harmonic mean per column, defined only for strictly positive columns.

    The paper's `IQR of harmonic means of features` is ill-defined when a
    column contains zero or negative values. We skip such columns (emitting
    NaN) rather than shifting or absolute-valuing the data, so the component
    matches the paper's stated operation on the feature columns it applies to.
    """
    hmeans: dict[str, float] = {}
    for column in X_num.columns:
        values = X_num[column].replace([np.inf, -np.inf], np.nan).dropna()
        if values.empty or (values <= 0).any():
            hmeans[column] = np.nan
            continue
        hmeans[column] = float(hmean(values.to_numpy()))
    return pd.Series(hmeans, dtype=float)


def compute_irregularity_components(X_num: pd.DataFrame) -> dict[str, float]:
    """Compute the 5 irregularity-proxy components on numeric features only.

    Components follow the paper's Section 2.2 / Figure 5 definition:

    - `irreg_min_cov_eig`: minimum eigenvalue of the standardized covariance.
    - `irreg_std_skew`:    skewness across per-feature standard deviations.
    - `irreg_range_skew`:  skewness across per-feature ranges.
    - `irreg_iqr_hmean`:   IQR of per-feature harmonic means (positive cols).
    - `irreg_kurtosis_std`: std across per-feature kurtoses.
    """
    X_num = X_num.apply(pd.to_numeric, errors="coerce")
    X_num = X_num.loc[:, X_num.nunique(dropna=True) > 1]

    if X_num.shape[1] == 0:
        return {column: np.nan for column in IRREGULARITY_PAPER_WEIGHTS}

    feature_stds = X_num.std(axis=0, ddof=1)
    feature_ranges = X_num.max(axis=0) - X_num.min(axis=0)
    feature_kurtosis = X_num.kurt(axis=0)
    feature_hmeans = _per_feature_hmeans(X_num)

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
        "irreg_iqr_hmean": _iqr(feature_hmeans),
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
    """Add the paper's ``irregularity`` composite as a weighted sum of z-scores.

    Each listed component is z-scored across rows of ``df`` (the datasets in
    the current analysis run, matching the paper's "standardized across
    datasets used in the analysis"). The composite is

        irregularity = sum_i w_i * z_i   (over components available for row)

    with weights from ``IRREGULARITY_PAPER_WEIGHTS``. When a component is
    NaN for a given row, that component is **dropped from that row's sum and
    the weights are renormalized** to preserve scale — otherwise a single
    missing component (common on datasets where the harmonic-mean component
    is undefined) would silently delete the whole dataset from downstream
    correlation analysis. A coverage summary is logged so reviewers can
    verify how often each component was available.

    A row with *no* available components yields NaN.
    """
    unknown = [column for column in components if column not in IRREGULARITY_PAPER_WEIGHTS]
    if unknown:
        raise ValueError(
            f"Unknown irregularity components: {unknown}. Expected a subset of "
            f"{tuple(IRREGULARITY_PAPER_WEIGHTS)}."
        )

    result = df.copy()
    available = [column for column in components if column in result.columns]
    if set(available) != set(IRREGULARITY_PAPER_WEIGHTS):
        logger.warning(
            "irregularity: using %d/%d paper components (%s); the resulting composite "
            "is a partial reproduction of the published metric.",
            len(available),
            len(IRREGULARITY_PAPER_WEIGHTS),
            ", ".join(available) if available else "<none>",
        )

    if not available:
        result[output_col] = np.nan
        return result

    z_scores = pd.DataFrame({column: zscore(result[column]) for column in available})
    n_rows = len(z_scores)

    weights = pd.Series(
        {column: IRREGULARITY_PAPER_WEIGHTS[column] for column in available},
        dtype=float,
    )
    abs_weights = weights.abs()
    full_abs_weight = float(abs_weights.sum())

    mask = z_scores.notna()
    # Per-row sum of |weight| actually used; 0 when no component is available.
    per_row_abs_weight = mask.astype(float).multiply(abs_weights, axis=1).sum(axis=1)
    weighted_sum = z_scores.fillna(0.0).multiply(weights, axis=1).sum(axis=1)

    with np.errstate(divide="ignore", invalid="ignore"):
        scaled = weighted_sum * (full_abs_weight / per_row_abs_weight)
    scaled = scaled.where(per_row_abs_weight > 0, other=np.nan)
    result[output_col] = scaled

    # Coverage summary — one line per component, so a reviewer can see at a
    # glance how many datasets drove the composite.
    for column in available:
        finite = int(mask[column].sum())
        logger.info(
            "irregularity: component %s available on %d/%d row(s).",
            column,
            finite,
            n_rows,
        )
    finite_rows = int((per_row_abs_weight > 0).sum())
    logger.info(
        "irregularity: composite finite on %d/%d row(s).",
        finite_rows,
        n_rows,
    )
    return result
