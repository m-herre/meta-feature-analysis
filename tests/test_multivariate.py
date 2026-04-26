from __future__ import annotations

import numpy as np
import pandas as pd

from mfa.stats.multivariate import run_multivariable_sensitivity


def test_run_multivariable_sensitivity_empty_result_has_full_schema() -> None:
    n = 30
    feature_table = pd.DataFrame(
        {
            "dataset": [f"d{i}" for i in range(n)],
            "delta_norm": np.arange(n, dtype=float),
        }
    )
    robust_table = pd.DataFrame()

    sensitivity, feature_report, control_report = run_multivariable_sensitivity(
        feature_table,
        robust_table,
        table_name="synthetic",
    )

    assert sensitivity.empty
    assert "adjusted_bootstrap_repeats" in sensitivity.columns
    assert feature_report.empty
    assert not control_report.empty
