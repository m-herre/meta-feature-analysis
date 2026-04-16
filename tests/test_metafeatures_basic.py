from __future__ import annotations

import math

import numpy as np
import pandas as pd

from mfa.metafeatures.basic import compute_basic_metafeatures


def test_compute_basic_metafeatures_known_values() -> None:
    X = pd.DataFrame(
        {
            "num": [1.0, 2.0, np.nan, 4.0],
            "cat": pd.Series(["a", "b", "a", None], dtype="category"),
            "flag": [True, False, True, True],
        }
    )
    features = compute_basic_metafeatures(X)
    assert features["n"] == 4
    assert features["d"] == 3
    assert math.isclose(features["log_n"], math.log10(4))
    assert math.isclose(features["n_over_d"], 4 / 3)
    assert math.isclose(features["cat_fraction"], 2 / 3)
    assert math.isclose(features["missing_fraction"], (1 / 4 + 1 / 4 + 0) / 3)


def test_compute_basic_metafeatures_zero_columns() -> None:
    X = pd.DataFrame(index=range(5))
    features = compute_basic_metafeatures(X)
    assert features["n"] == 5
    assert features["d"] == 0
    assert np.isnan(features["n_over_d"])
    assert np.isnan(features["cat_fraction"])
    assert np.isnan(features["missing_fraction"])

