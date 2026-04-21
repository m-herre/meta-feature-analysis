from __future__ import annotations

import numpy as np
import pandas as pd

REDUNDANCY_METAFEATURE_SCHEMA_VERSION = 1
HIGH_CORR_THRESHOLD = 0.9
MAX_REDUNDANCY_NUMERIC_FEATURES = 512


def _nan_redundancy_features() -> dict[str, float]:
    return {
        "mean_abs_corr": np.nan,
        "max_abs_corr": np.nan,
        "high_corr_pair_fraction": np.nan,
        "effective_rank": np.nan,
        "participation_ratio": np.nan,
    }


def compute_redundancy_metafeatures(X_num: pd.DataFrame) -> dict[str, float]:
    """Compute numeric redundancy metrics with a hard width cap.

    These features require a full correlation matrix and eigendecomposition, so
    they are intentionally kept outside the cheap `basic` feature set.
    """
    default = _nan_redundancy_features()
    X_num = X_num.apply(pd.to_numeric, errors="coerce")
    X_num = X_num.loc[:, X_num.nunique(dropna=True) > 1]
    if X_num.shape[1] < 2 or X_num.shape[1] > MAX_REDUNDANCY_NUMERIC_FEATURES:
        return default

    corr = X_num.corr().replace([np.inf, -np.inf], np.nan)
    mask = np.triu(np.ones(corr.shape, dtype=bool), k=1)
    pair_values = corr.abs().where(mask).stack().dropna()
    if pair_values.empty:
        corr_features = {
            "mean_abs_corr": np.nan,
            "max_abs_corr": np.nan,
            "high_corr_pair_fraction": np.nan,
        }
    else:
        corr_features = {
            "mean_abs_corr": float(pair_values.mean()),
            "max_abs_corr": float(pair_values.max()),
            "high_corr_pair_fraction": float((pair_values > HIGH_CORR_THRESHOLD).mean()),
        }

    corr_filled = corr.fillna(0.0)
    np.fill_diagonal(corr_filled.values, 1.0)
    eigenvalues = np.linalg.eigvalsh(corr_filled.to_numpy())
    eigenvalues = np.clip(eigenvalues, 0.0, None)
    total = float(eigenvalues.sum())
    square_sum = float(np.square(eigenvalues).sum())
    if total <= 0:
        rank_features = {"effective_rank": np.nan, "participation_ratio": np.nan}
    else:
        probabilities = eigenvalues[eigenvalues > 0] / total
        rank_features = {
            "effective_rank": float(np.exp(-(probabilities * np.log(probabilities)).sum())),
            "participation_ratio": float((total**2) / square_sum) if square_sum > 0 else np.nan,
        }
    return {**corr_features, **rank_features}
