from __future__ import annotations

import numpy as np

from mfa.gaps.pairwise import compute_pairwise_gaps, pick_best_in_group


def test_pick_best_in_group_uses_deterministic_tie_break(synthetic_results) -> None:
    tied = synthetic_results[synthetic_results["fold"] == 3].copy()
    tied["repeat"] = 1
    tied["fold_in_repeat"] = 0
    tied["norm_error"] = tied["metric_error"]
    best = pick_best_in_group(tied, config_types={"NN_A", "NN_B"}, prefix="a")
    assert best["best_a_method"].item() == "nn_alpha"


def test_compute_pairwise_gaps_delta_sign(analysis_config, synthetic_results) -> None:
    gaps = compute_pairwise_gaps(
        synthetic_results,
        analysis_config.comparisons,
        error_column=analysis_config.analysis.error_column,
    )
    first_split = gaps[(gaps["dataset"] == "dataset_a") & (gaps["repeat"] == 0) & (gaps["fold"] == 0)].iloc[0]
    assert first_split["best_a_method"] == "nn_beta"
    assert first_split["best_b_method"] == "gbdt_alpha"
    assert np.isclose(first_split["delta_raw"], 0.10)
    assert first_split["delta_norm"] > 0

