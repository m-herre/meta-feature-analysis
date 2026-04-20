from __future__ import annotations

from pathlib import Path

import pytest
import yaml

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
    assert config.cache.directory == Path.cwd().resolve() / ".mfa_cache"


def test_load_config_resolves_cache_directory_relative_to_project_root(tmp_path, config_dict) -> None:
    project_dir = tmp_path / "meta-feature-analysis"
    config_dir = project_dir / "configs"
    config_dir.mkdir(parents=True)
    config_path = config_dir / "config.yaml"
    with config_path.open("w", encoding="utf-8") as outfile:
        yaml.safe_dump(config_dict, outfile)

    config = load_config(config_path)

    assert config.cache.directory == project_dir / ".mfa_cache"


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


def test_parse_config_defaults_parallelism_when_absent(config_dict) -> None:
    del config_dict["parallelism"]
    config = parse_config(config_dict)
    assert config.parallelism.n_jobs == 1
    assert config.parallelism.backend == "process"


def test_parse_config_parses_parallelism(config_dict) -> None:
    config_dict["parallelism"] = {"n_jobs": -1, "backend": "thread"}
    config = parse_config(config_dict)
    assert config.parallelism.n_jobs == -1
    assert config.parallelism.backend == "thread"


def test_parse_config_rejects_invalid_backend(config_dict) -> None:
    config_dict["parallelism"] = {"backend": "gpu"}
    with pytest.raises(ConfigValidationError):
        parse_config(config_dict)


def test_parse_config_accepts_exclude_problem_types(config_dict) -> None:
    config_dict["analysis"]["exclude_problem_types"] = ["regression"]
    config = parse_config(config_dict)
    assert config.analysis.exclude_problem_types == ("regression",)


def test_parse_config_rejects_invalid_problem_type(config_dict) -> None:
    config_dict["analysis"]["exclude_problem_types"] = ["ordinal"]
    with pytest.raises(ConfigValidationError):
        parse_config(config_dict)


def test_parse_config_defaults_exclude_problem_types_empty(config_dict) -> None:
    config_dict["analysis"].pop("exclude_problem_types", None)
    config = parse_config(config_dict)
    assert config.analysis.exclude_problem_types == ()


def test_parse_config_parses_metafeature_trace(config_dict) -> None:
    config_dict["metafeatures"]["trace"] = True
    config = parse_config(config_dict)
    assert config.metafeatures.trace is True


def test_parse_config_treats_null_metafeature_sequences_as_defaults(config_dict) -> None:
    config_dict["metafeatures"]["pymfe_groups"] = None
    config_dict["metafeatures"]["pymfe_summary"] = None
    config_dict["metafeatures"]["irregularity_components"] = None

    config = parse_config(config_dict)

    assert config.metafeatures.pymfe_groups == ("general", "statistical", "info-theory")
    assert config.metafeatures.pymfe_summary == ("mean", "sd")
    assert config.metafeatures.irregularity_components == (
        "irreg_min_cov_eig",
        "irreg_std_skew",
        "irreg_range_skew",
        "irreg_kurtosis_std",
    )


def test_config_hash_is_deterministic(analysis_config) -> None:
    left = compute_config_hash(analysis_config.to_dict())
    right = compute_config_hash(load_config(Path("configs/default.yaml")).to_dict())
    assert left
    assert isinstance(left, str)
    assert len(left) == 16
    assert left == compute_config_hash(analysis_config.to_dict())
    assert left != right
