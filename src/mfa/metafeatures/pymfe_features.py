from __future__ import annotations

import numpy as np
import pandas as pd

from .basic import get_categorical_columns


def extract_pymfe_features(
    X_train: pd.DataFrame,
    y_train: pd.Series | None,
    *,
    groups: tuple[str, ...],
    summary: tuple[str, ...],
) -> dict[str, float]:
    """Extract optional pymfe meta-features."""
    try:
        from pymfe.mfe import MFE
    except ImportError as err:
        raise ImportError("`pymfe` is not installed. Install `meta-feature-analysis[pymfe]` to enable it.") from err

    X_encoded = X_train.copy()
    categorical_columns = get_categorical_columns(X_encoded)
    for column in categorical_columns:
        if X_encoded[column].isna().any():
            X_encoded[column] = X_encoded[column].fillna(X_encoded[column].mode().iloc[0])
        X_encoded[column] = X_encoded[column].astype("category").cat.codes
    X_encoded = X_encoded.apply(pd.to_numeric, errors="coerce")
    categorical_indices = [X_encoded.columns.get_loc(column) for column in categorical_columns]
    mfe = MFE(groups=list(groups), summary=list(summary))
    mfe.fit(X_encoded.to_numpy(), None if y_train is None else y_train.to_numpy(), cat_cols=categorical_indices)
    names, values = mfe.extract()
    return {
        f"pymfe__{name}": float(value) if value is not None else np.nan
        for name, value in zip(names, values, strict=False)
    }
