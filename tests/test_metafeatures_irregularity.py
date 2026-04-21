from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mfa.metafeatures.irregularity import (
    IRREGULARITY_PAPER_WEIGHTS,
    add_irregularity_proxy,
    compute_irregularity_components,
    zscore,
)


def test_compute_irregularity_components_identity_like_covariance() -> None:
    X = pd.DataFrame(
        {
            "x1": [1.0, -1.0, 0.0, 0.0, 0.0, 0.0],
            "x2": [0.0, 0.0, 1.0, -1.0, 0.0, 0.0],
            "x3": [0.0, 0.0, 0.0, 0.0, 2.0, -2.0],
        }
    )
    features = compute_irregularity_components(X)
    assert set(features) == set(IRREGULARITY_PAPER_WEIGHTS)
    assert features["irreg_min_cov_eig"] > 0.1
    assert np.isfinite(features["irreg_std_skew"])


def test_compute_irregularity_components_degenerate_input() -> None:
    X = pd.DataFrame({"x1": [1.0, 1.0, 1.0], "x2": [2.0, 2.0, 2.0]})
    features = compute_irregularity_components(X)
    assert set(features) == set(IRREGULARITY_PAPER_WEIGHTS)
    assert all(np.isnan(value) for value in features.values())


def test_compute_iqr_hmean_skips_nonpositive_columns() -> None:
    # 6 columns: 4 strictly positive (contribute), 2 with non-positives (skipped).
    X = pd.DataFrame(
        {
            "p1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "p2": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
            "p3": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            "p4": [5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            "n1": [-1.0, 2.0, 3.0, 4.0, 5.0, 6.0],  # contains negative
            "z1": [0.0, 2.0, 3.0, 4.0, 5.0, 6.0],   # contains zero
        }
    )
    features = compute_irregularity_components(X)
    assert np.isfinite(features["irreg_iqr_hmean"])
    assert features["irreg_iqr_hmean"] >= 0.0


def test_compute_iqr_hmean_nan_when_fewer_than_two_surviving_columns() -> None:
    # Only 1 positive column survives; IQR needs >= 2 finite hmeans.
    X = pd.DataFrame(
        {
            "p1": [1.0, 2.0, 3.0, 4.0],
            "n1": [-1.0, -2.0, -3.0, -4.0],
            "n2": [-5.0, -6.0, -7.0, -8.0],
        }
    )
    features = compute_irregularity_components(X)
    assert np.isnan(features["irreg_iqr_hmean"])


def test_compute_iqr_hmean_finite_with_three_positive_columns() -> None:
    # Regression: the previous guard required >= 4 positive columns; datasets
    # with only 2-3 positive columns silently lost the hmean component, which
    # then propagated NaN through the composite.
    X = pd.DataFrame(
        {
            "p1": [1.0, 2.0, 3.0, 4.0],
            "p2": [10.0, 20.0, 30.0, 40.0],
            "p3": [0.1, 0.2, 0.3, 0.4],
            "n1": [-1.0, -2.0, -3.0, -4.0],
        }
    )
    features = compute_irregularity_components(X)
    assert np.isfinite(features["irreg_iqr_hmean"])


def test_add_irregularity_proxy_matches_paper_weighted_sum() -> None:
    # Construct a component table with distinct values per row so the z-scores
    # are non-degenerate, then check that the output equals the hand-computed
    # weighted sum of z-scored components.
    df = pd.DataFrame(
        {
            "irreg_min_cov_eig":  [3.0, 2.0, 1.5, 0.5, 0.1, 0.05],
            "irreg_std_skew":     [0.0, 0.5, 1.0, 1.5, 2.0, 2.5],
            "irreg_range_skew":   [0.1, 0.4, 0.9, 1.2, 1.8, 2.3],
            "irreg_iqr_hmean":    [5.0, 4.0, 3.0, 2.0, 1.0, 0.5],
            "irreg_kurtosis_std": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
        }
    )
    result = add_irregularity_proxy(df)
    expected = sum(
        IRREGULARITY_PAPER_WEIGHTS[column] * zscore(df[column]) for column in IRREGULARITY_PAPER_WEIGHTS
    )
    np.testing.assert_allclose(result["irregularity"].to_numpy(), expected.to_numpy(), atol=1e-12)


def test_add_irregularity_proxy_monotone_in_min_cov_eig() -> None:
    # Letting min_cov_eig decrease (increasingly-irregular covariance) while
    # the other four components are monotone-increasing should yield
    # monotone-increasing irregularity — the negative weight on min_cov_eig
    # flips its direction so all five components co-push irregularity up.
    df = pd.DataFrame(
        {
            "irreg_min_cov_eig":  [3.0, 2.0, 1.0, 0.5, 0.2, 0.05],
            "irreg_std_skew":     [0.0, 0.4, 0.8, 1.2, 1.6, 2.0],
            "irreg_range_skew":   [0.0, 0.4, 0.8, 1.2, 1.6, 2.0],
            "irreg_iqr_hmean":    [0.0, 0.4, 0.8, 1.2, 1.6, 2.0],
            "irreg_kurtosis_std": [0.0, 0.4, 0.8, 1.2, 1.6, 2.0],
        }
    )
    result = add_irregularity_proxy(df)
    assert result["irregularity"].iloc[0] < result["irregularity"].iloc[-1]


def test_add_irregularity_proxy_renormalizes_when_component_partially_missing() -> None:
    # A single NaN component on one row must not collapse the row to NaN.
    # Without renormalization, a dataset with an undefined hmean would drop
    # from correlation analysis entirely — a silent failure mode flagged by
    # Codex. With renormalization, that row's composite is the weighted sum
    # over the surviving components, rescaled to the full |weight| sum.
    df = pd.DataFrame(
        {
            "irreg_min_cov_eig":  [3.0, 2.0, 1.0, 0.5, 0.2, 0.05],
            "irreg_std_skew":     [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            "irreg_range_skew":   [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            "irreg_iqr_hmean":    [5.0, np.nan, 3.0, 2.0, 1.0, 0.5],
            "irreg_kurtosis_std": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
        }
    )
    result = add_irregularity_proxy(df)
    assert np.isfinite(result["irregularity"]).all()


def test_add_irregularity_proxy_nan_only_when_all_components_missing() -> None:
    df = pd.DataFrame(
        {
            "irreg_min_cov_eig":  [3.0, np.nan, 1.0],
            "irreg_std_skew":     [0.0, np.nan, 2.0],
            "irreg_range_skew":   [0.0, np.nan, 2.0],
            "irreg_iqr_hmean":    [5.0, np.nan, 3.0],
            "irreg_kurtosis_std": [0.0, np.nan, 2.0],
        }
    )
    result = add_irregularity_proxy(df)
    assert np.isfinite(result["irregularity"].iloc[0])
    assert np.isnan(result["irregularity"].iloc[1])
    assert np.isfinite(result["irregularity"].iloc[2])


def test_add_irregularity_proxy_rejects_unknown_component() -> None:
    df = pd.DataFrame({"irreg_min_cov_eig": [1.0, 2.0, 3.0]})
    with pytest.raises(ValueError, match="Unknown irregularity components"):
        add_irregularity_proxy(df, components=("irreg_min_cov_eig", "bogus_component"))
