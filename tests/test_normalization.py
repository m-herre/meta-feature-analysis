from __future__ import annotations

import numpy as np
import pandas as pd

from mfa.gaps.normalization import add_normalized_error


def test_add_normalized_error_matches_expected_scale() -> None:
    df = pd.DataFrame(
        [
            ["dataset1", 0, "xgboost1", 1.0],
            ["dataset1", 0, "xgboost2", 2.0],
            ["dataset1", 0, "xgboost3", 3.0],
        ],
        columns=["dataset", "fold", "method", "metric_error"],
    )
    normalized = add_normalized_error(df)
    assert np.isclose(normalized.loc[normalized["method"] == "xgboost1", "norm_error"].item(), 0.0)
    assert np.isclose(normalized.loc[normalized["method"] == "xgboost2", "norm_error"].item(), 1.0)
    assert np.isclose(normalized.loc[normalized["method"] == "xgboost3", "norm_error"].item(), 1.0)

