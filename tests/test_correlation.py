from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mfa.stats.correlation import (
    build_robust_association_table,
    estimate_feature_associations,
)


def _association_frame(n: int = 40) -> pd.DataFrame:
    x = np.arange(n, dtype=float)
    return pd.DataFrame(
        {
            "dataset": [f"d{i}" for i in range(n)],
            "delta_norm": x,
            "strong": x,
            "inverse": -x,
            "weak": np.sin(x),
        }
    )


def test_estimate_feature_associations_rejects_duplicate_units() -> None:
    table = _association_frame(4)
    table.loc[1, "dataset"] = table.loc[0, "dataset"]

    with pytest.raises(ValueError, match="not dataset-level"):
        estimate_feature_associations(table, table_name="synthetic", feature_columns=["strong"], min_n=3)


def test_estimate_feature_associations_adds_bh_and_bootstrap_columns() -> None:
    associations = estimate_feature_associations(
        _association_frame(),
        table_name="synthetic",
        feature_columns=["strong", "inverse", "weak"],
        min_n=30,
        bootstrap_repeats=50,
        rank_stability_top_k=2,
        random_seed=123,
    )

    assert set(["p_value_bh", "bh_reject_0_05", "ci_low", "ci_high"]).issubset(associations.columns)
    assert set(
        [
            "bootstrap_rank_median",
            "bootstrap_rank_q05",
            "bootstrap_rank_q95",
            "bootstrap_top_k_frequency",
        ]
    ).issubset(associations.columns)
    assert associations.loc[associations["feature"].isin(["strong", "inverse"]), "bh_reject_0_05"].all()
    assert associations.loc[associations["feature"] == "strong", "ci_low"].notna().all()


def test_estimate_feature_associations_marks_small_samples_untested() -> None:
    associations = estimate_feature_associations(
        _association_frame(5),
        table_name="synthetic",
        feature_columns=["strong"],
        min_n=30,
        bootstrap_repeats=20,
    )

    row = associations.iloc[0]
    assert row["reason"] == "n < 30"
    assert np.isnan(row["spearman_r"])
    assert not bool(row["bh_reject_0_05"])


def test_estimate_feature_associations_default_inference_excludes_gap_leakage_columns() -> None:
    n = 30
    x = np.arange(n, dtype=float)
    table = pd.DataFrame(
        {
            "dataset": [f"d{i}" for i in range(n)],
            "comparison_name": ["nn_vs_tree"] * n,
            "delta_norm": x,
            "delta_raw": x * 10,
            "best_a_error": x + 1,
            "best_b_error": x + 2,
            "text_context": ["context"] * n,
            "real_feature": np.sin(x),
        }
    )

    associations = estimate_feature_associations(
        table,
        table_name="synthetic",
        min_n=30,
        bootstrap_repeats=20,
        ci_top_k=0,
        rank_stability_top_k=1,
    )

    assert associations["feature"].tolist() == ["real_feature"]


def test_build_robust_association_table_filters_and_renames() -> None:
    associations = estimate_feature_associations(
        _association_frame(),
        table_name="synthetic",
        feature_columns=["strong", "inverse", "weak"],
        min_n=30,
        bootstrap_repeats=50,
        rank_stability_top_k=2,
        random_seed=123,
    )
    robust = build_robust_association_table(
        associations,
        table_name="synthetic",
        min_sign_consistency=0.90,
        top_n=1,
    )

    assert len(robust) == 1
    assert robust.loc[0, "analysis_table"] == "synthetic"
    assert "spearman_rho" in robust.columns
    assert "bh_q_value" in robust.columns
    assert "bootstrap_top_k_frequency" in robust.columns
    assert "bootstrap_top_25_frequency" not in robust.columns
