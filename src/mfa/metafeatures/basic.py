from __future__ import annotations

import numpy as np
import pandas as pd


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


def compute_basic_metafeatures(X_train: pd.DataFrame) -> dict[str, float]:
    """Compute simple size and composition meta-features."""
    n, d = X_train.shape
    cat_cols = get_categorical_columns(X_train)
    return {
        "n": int(n),
        "d": int(d),
        "log_n": float(np.log10(n)) if n > 0 else np.nan,
        "n_over_d": float(n / d) if d > 0 else np.nan,
        "cat_fraction": float(len(cat_cols) / d) if d > 0 else np.nan,
        "missing_fraction": float(X_train.isna().mean().mean()) if d > 0 else np.nan,
    }
