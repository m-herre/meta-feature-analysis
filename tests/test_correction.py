from __future__ import annotations

import numpy as np
import pandas as pd

from mfa.stats.correlation import estimate_feature_associations


def test_estimate_feature_associations_applies_bh_correction() -> None:
    n = 40
    x = np.arange(n, dtype=float)
    rng = np.random.default_rng(7)
    table = pd.DataFrame(
        {
            "dataset": [f"d{i}" for i in range(n)],
            "delta_norm": x,
            "strong": x,
            "noise": rng.normal(size=n),
        }
    )

    associations = estimate_feature_associations(
        table,
        table_name="synthetic",
        feature_columns=["strong", "noise"],
        min_n=30,
        bootstrap_repeats=25,
    )

    strong = associations.set_index("feature").loc["strong"]
    assert strong["p_value_bh"] < 0.05
    assert bool(strong["bh_reject_0_05"])
    assert associations["p_value_bh"].notna().all()
