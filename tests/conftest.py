from __future__ import annotations

import pandas as pd
import pytest

from mfa.config import parse_config


@pytest.fixture
def config_dict() -> dict:
    return {
        "version": 1,
        "groups": {
            "nn": {
                "config_types": ["NN_A", "NN_B"],
                "label": "NN",
            },
            "gbdt": {
                "config_types": ["GBDT_A", "GBDT_B"],
                "label": "GBDT",
            },
        },
        "comparisons": [
            {
                "name": "nn_vs_gbdt",
                "group_a": "nn",
                "group_b": "gbdt",
                "expected_direction": "positive",
            }
        ],
        "analysis": {
            "unit": "dataset",
            "error_column": "metric_error",
            "selection_error_column": "metric_error_val",
            "method_variant": "tuned",
            "exclude_methods_containing": [],
        },
        "metafeatures": {
            "feature_sets": ["basic", "irregularity"],
            "pymfe_groups": ["general"],
            "pymfe_summary": ["mean"],
            "irregularity_components": [
                "irreg_min_cov_eig",
                "irreg_std_skew",
                "irreg_range_skew",
                "irreg_iqr_hmean",
                "irreg_kurtosis_std",
            ],
        },
        "statistics": {
            "correlation_method": "spearman",
            "alpha": 0.05,
            "fdr_method": "bh",
            "confidence_interval": True,
            "ci_bootstrap_samples": 100,
            "ci_confidence_level": 0.95,
            "multivariate": False,
            "multivariate_method": "ols",
        },
        "cache": {
            "enabled": True,
            "directory": ".mfa_cache",
            "stages": {
                "raw_results": True,
                "metafeatures": True,
                "gaps": True,
                "statistics": True,
            },
        },
        "parallelism": {
            "n_jobs": 1,
            "backend": "process",
        },
    }


@pytest.fixture
def analysis_config(config_dict):
    return parse_config(config_dict)


@pytest.fixture
def synthetic_results() -> pd.DataFrame:
    return pd.DataFrame(
        [
            ["dataset_a", 0, "nn_alpha", 0.30, 0.31, "NN_A", "tuned"],
            ["dataset_a", 0, "nn_beta", 0.20, 0.21, "NN_B", "tuned"],
            ["dataset_a", 0, "gbdt_alpha", 0.10, 0.11, "GBDT_A", "tuned"],
            ["dataset_a", 0, "gbdt_beta", 0.40, 0.41, "GBDT_B", "tuned"],
            ["dataset_a", 3, "nn_alpha", 0.15, 0.16, "NN_A", "tuned"],
            ["dataset_a", 3, "nn_beta", 0.15, 0.16, "NN_B", "tuned"],
            ["dataset_a", 3, "gbdt_alpha", 0.25, 0.26, "GBDT_A", "tuned"],
            ["dataset_a", 3, "gbdt_beta", 0.35, 0.36, "GBDT_B", "tuned"],
            ["dataset_b", 0, "nn_alpha", 0.25, 0.26, "NN_A", "tuned"],
            ["dataset_b", 0, "gbdt_alpha", 0.20, 0.21, "GBDT_A", "tuned"],
        ],
        columns=[
            "dataset",
            "fold",
            "method",
            "metric_error",
            "metric_error_val",
            "config_type",
            "method_subtype",
        ],
    )


@pytest.fixture
def synthetic_metafeatures() -> pd.DataFrame:
    return pd.DataFrame(
        [
            ["dataset_a", 0, 0, 100, 10, 2.0, 10.0, 0.2, 0.05, 0.5],
            ["dataset_a", 1, 0, 110, 10, 2.04, 11.0, 0.2, 0.04, 0.6],
            ["dataset_b", 0, 0, 80, 8, 1.9, 10.0, 0.1, 0.02, 0.3],
        ],
        columns=[
            "dataset",
            "repeat",
            "fold",
            "n",
            "d",
            "log_n",
            "n_over_d",
            "cat_fraction",
            "missing_fraction",
            "irregularity",
        ],
    )
