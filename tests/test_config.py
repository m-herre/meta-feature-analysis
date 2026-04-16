from __future__ import annotations

from pathlib import Path

import pytest

from mfa.cache import compute_config_hash
from mfa.config import ConfigValidationError, load_config, parse_config
from mfa.types import AnalysisUnit, CorrelationMethod


def test_load_default_config() -> None:
    config = load_config(Path("configs/default.yaml"))
    assert config.analysis.unit == AnalysisUnit.DATASET
    assert config.analysis.selection_error_column == "metric_error_val"
    assert config.statistics.correlation_method == CorrelationMethod.SPEARMAN
    assert config.comparisons[0].group_a.name == "nn"
    assert config.comparisons[0].group_b.name == "gbdt"


def test_parse_config_rejects_unknown_group_reference(config_dict) -> None:
    config_dict["comparisons"][0]["group_b"] = "missing"
    with pytest.raises(ConfigValidationError):
        parse_config(config_dict)


def test_parse_config_rejects_unknown_unit(config_dict) -> None:
    config_dict["analysis"]["unit"] = "bad"
    with pytest.raises(ConfigValidationError):
        parse_config(config_dict)


def test_parse_config_rejects_unknown_method_variant(config_dict) -> None:
    config_dict["analysis"]["method_variant"] = "best"
    with pytest.raises(ConfigValidationError):
        parse_config(config_dict)


def test_parse_config_accepts_optional_selection_error_column(config_dict) -> None:
    config_dict["analysis"]["selection_error_column"] = "metric_error_val"
    config = parse_config(config_dict)
    assert config.analysis.selection_error_column == "metric_error_val"


def test_parse_config_defaults_selection_to_validation_metric(config_dict) -> None:
    del config_dict["analysis"]["selection_error_column"]
    config = parse_config(config_dict)
    assert config.analysis.selection_error_column == "metric_error_val"


def test_parse_config_rejects_invalid_selection_error_column(config_dict) -> None:
    config_dict["analysis"]["selection_error_column"] = 1
    with pytest.raises(ConfigValidationError):
        parse_config(config_dict)


def test_config_hash_is_deterministic(analysis_config) -> None:
    left = compute_config_hash(analysis_config.to_dict())
    right = compute_config_hash(load_config(Path("configs/default.yaml")).to_dict())
    assert left
    assert isinstance(left, str)
    assert len(left) == 16
    assert left == compute_config_hash(analysis_config.to_dict())
    assert left != right
