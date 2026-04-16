from __future__ import annotations

import numpy as np
import pandas as pd

from mfa.metafeatures.irregularity import add_irregularity_proxy, compute_irregularity_components


def test_compute_irregularity_components_identity_like_covariance() -> None:
    X = pd.DataFrame(
        {
            "x1": [1.0, -1.0, 0.0, 0.0, 0.0, 0.0],
            "x2": [0.0, 0.0, 1.0, -1.0, 0.0, 0.0],
            "x3": [0.0, 0.0, 0.0, 0.0, 2.0, -2.0],
        }
    )
    features = compute_irregularity_components(X)
    assert features["irreg_min_cov_eig"] > 0.1
    assert np.isfinite(features["irreg_std_skew"])


def test_compute_irregularity_components_degenerate_input() -> None:
    X = pd.DataFrame({"x1": [1.0, 1.0, 1.0], "x2": [2.0, 2.0, 2.0]})
    features = compute_irregularity_components(X)
    assert all(np.isnan(value) for value in features.values())


def test_add_irregularity_proxy_flips_min_eigenvalue() -> None:
    df = pd.DataFrame(
        {
            "irreg_min_cov_eig": [3.0, 2.0, 1.0],
            "irreg_std_skew": [0.0, 1.0, 2.0],
            "irreg_range_skew": [0.0, 1.0, 2.0],
            "irreg_kurtosis_std": [0.0, 1.0, 2.0],
        }
    )
    result = add_irregularity_proxy(df)
    assert result["irregularity"].iloc[0] < result["irregularity"].iloc[-1]
