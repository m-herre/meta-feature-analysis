from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

import yaml

from .parallel import VALID_BACKENDS
from .types import AnalysisUnit, ComparisonSpec, CorrelationMethod, FDRMethod, GroupDef, MultivariateMethod

VALID_METHOD_VARIANTS = {"default", "tuned", "tuned_ensemble"}
VALID_EXPECTED_DIRECTIONS = {None, "positive", "negative"}
VALID_PROBLEM_TYPES = {"regression", "binary", "multiclass"}


class ConfigValidationError(ValueError):
    """Raised when the YAML configuration is invalid."""


def normalize_method_variants(
    raw_method_variant: Any,
    *,
    field_name: str = "analysis.method_variant",
) -> tuple[str, ...]:
    if isinstance(raw_method_variant, str):
        method_variants = (raw_method_variant,)
    elif isinstance(raw_method_variant, (list, tuple)):
        if not raw_method_variant or not all(isinstance(value, str) for value in raw_method_variant):
            raise ConfigValidationError(f"`{field_name}` must be a non-empty list of strings or a single string.")
        method_variants = tuple(raw_method_variant)
    else:
        raise ConfigValidationError(f"`{field_name}` must be a string or a non-empty list of strings.")

    invalid_variants = [value for value in method_variants if value not in VALID_METHOD_VARIANTS]
    if invalid_variants:
        valid_variants = ", ".join(sorted(VALID_METHOD_VARIANTS))
        raise ConfigValidationError(
            f"`{field_name}` contains invalid values {invalid_variants}; "
            f"must be a subset of: {valid_variants}."
        )
    return method_variants


@dataclass(frozen=True)
class AnalysisSettings:
    unit: AnalysisUnit = AnalysisUnit.DATASET
    error_column: str = "metric_error"
    selection_error_column: str | None = "metric_error_val"
    method_variant: tuple[str, ...] = ("tuned",)
    exclude_methods_containing: tuple[str, ...] = ()
    exclude_problem_types: tuple[str, ...] = ()


@dataclass(frozen=True)
class MetafeatureSettings:
    feature_sets: tuple[str, ...] = ("basic", "irregularity")
    pymfe_groups: tuple[str, ...] = ("general", "statistical", "info-theory")
    pymfe_summary: tuple[str, ...] = ("mean", "sd")
    pymfe_per_feature_timeout_s: float | None = None
    retry_failed_pymfe: bool = False
    trace: bool = False
    irregularity_components: tuple[str, ...] = (
        "irreg_min_cov_eig",
        "irreg_std_skew",
        "irreg_range_skew",
        "irreg_iqr_hmean",
        "irreg_kurtosis_std",
    )


@dataclass(frozen=True)
class StatisticsSettings:
    correlation_method: CorrelationMethod = CorrelationMethod.SPEARMAN
    alpha: float = 0.05
    fdr_method: FDRMethod | None = FDRMethod.BH
    confidence_interval: bool = True
    ci_bootstrap_samples: int = 10_000
    ci_confidence_level: float = 0.95
    multivariate: bool = False
    multivariate_method: MultivariateMethod = MultivariateMethod.OLS


@dataclass(frozen=True)
class CacheStageConfig:
    raw_results: bool = True
    metafeatures: bool = True
    gaps: bool = True
    statistics: bool = True


@dataclass(frozen=True)
class ParallelismSettings:
    n_jobs: int = 1
    backend: str = "process"


@dataclass(frozen=True)
class CacheConfig:
    enabled: bool = True
    directory: Path = Path(".mfa_cache")
    stages: CacheStageConfig = field(default_factory=CacheStageConfig)


@dataclass(frozen=True)
class AnalysisConfig:
    version: int
    groups: dict[str, GroupDef]
    comparisons: tuple[ComparisonSpec, ...]
    analysis: AnalysisSettings = field(default_factory=AnalysisSettings)
    metafeatures: MetafeatureSettings = field(default_factory=MetafeatureSettings)
    statistics: StatisticsSettings = field(default_factory=StatisticsSettings)
    cache: CacheConfig = field(default_factory=CacheConfig)
    parallelism: ParallelismSettings = field(default_factory=ParallelismSettings)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _require_mapping(raw: Any, field_name: str) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise ConfigValidationError(f"`{field_name}` must be a mapping.")
    return raw


def _require_sequence(raw: Any, field_name: str) -> list[Any]:
    if not isinstance(raw, list):
        raise ConfigValidationError(f"`{field_name}` must be a list.")
    return raw


def _parse_enum(enum_cls, value: Any, field_name: str):
    try:
        return enum_cls(value)
    except ValueError as err:
        valid = ", ".join(member.value for member in enum_cls)
        raise ConfigValidationError(f"`{field_name}` must be one of: {valid}.") from err


def _parse_groups(raw_groups: Any) -> dict[str, GroupDef]:
    groups_mapping = _require_mapping(raw_groups, "groups")
    groups: dict[str, GroupDef] = {}
    for name, raw_group in groups_mapping.items():
        group_mapping = _require_mapping(raw_group, f"groups.{name}")
        config_types = tuple(group_mapping.get("config_types", []))
        if not config_types or not all(isinstance(config_type, str) for config_type in config_types):
            raise ConfigValidationError(f"`groups.{name}.config_types` must be a non-empty list of strings.")
        label = group_mapping.get("label", name)
        if not isinstance(label, str) or not label:
            raise ConfigValidationError(f"`groups.{name}.label` must be a non-empty string.")
        groups[name] = GroupDef(name=name, config_types=frozenset(config_types), label=label)
    if not groups:
        raise ConfigValidationError("At least one group must be defined.")
    return groups


def _parse_comparisons(raw_comparisons: Any, groups: dict[str, GroupDef]) -> tuple[ComparisonSpec, ...]:
    comparisons_raw = _require_sequence(raw_comparisons, "comparisons")
    comparisons: list[ComparisonSpec] = []
    for idx, raw_comparison in enumerate(comparisons_raw):
        prefix = f"comparisons[{idx}]"
        mapping = _require_mapping(raw_comparison, prefix)
        name = mapping.get("name")
        if not isinstance(name, str) or not name:
            raise ConfigValidationError(f"`{prefix}.name` must be a non-empty string.")
        group_a_name = mapping.get("group_a")
        group_b_name = mapping.get("group_b")
        if group_a_name not in groups:
            raise ConfigValidationError(f"`{prefix}.group_a` references unknown group `{group_a_name}`.")
        if group_b_name not in groups:
            raise ConfigValidationError(f"`{prefix}.group_b` references unknown group `{group_b_name}`.")
        expected_direction = mapping.get("expected_direction")
        if expected_direction not in VALID_EXPECTED_DIRECTIONS:
            raise ConfigValidationError(f"`{prefix}.expected_direction` must be one of: positive, negative, or null.")
        comparisons.append(
            ComparisonSpec(
                name=name,
                group_a=groups[group_a_name],
                group_b=groups[group_b_name],
                expected_direction=expected_direction,
            )
        )
    if not comparisons:
        raise ConfigValidationError("At least one comparison must be defined.")
    return tuple(comparisons)


def _parse_analysis(raw_analysis: Any) -> AnalysisSettings:
    mapping = _require_mapping(raw_analysis or {}, "analysis")
    method_variants = normalize_method_variants(mapping.get("method_variant", "tuned"))
    error_column = mapping.get("error_column", "metric_error")
    if not isinstance(error_column, str) or not error_column:
        raise ConfigValidationError("`analysis.error_column` must be a non-empty string.")
    selection_error_column = mapping.get("selection_error_column", "metric_error_val")
    if selection_error_column is not None and (
        not isinstance(selection_error_column, str) or not selection_error_column
    ):
        raise ConfigValidationError("`analysis.selection_error_column` must be null or a non-empty string.")
    exclude_methods = tuple(mapping.get("exclude_methods_containing", []))
    if not all(isinstance(value, str) for value in exclude_methods):
        raise ConfigValidationError("`analysis.exclude_methods_containing` must be a list of strings.")
    exclude_problem_types = tuple(mapping.get("exclude_problem_types", []))
    if not all(isinstance(value, str) for value in exclude_problem_types):
        raise ConfigValidationError("`analysis.exclude_problem_types` must be a list of strings.")
    invalid_problem_types = [value for value in exclude_problem_types if value not in VALID_PROBLEM_TYPES]
    if invalid_problem_types:
        valid = ", ".join(sorted(VALID_PROBLEM_TYPES))
        raise ConfigValidationError(
            f"`analysis.exclude_problem_types` contains invalid values {invalid_problem_types}; "
            f"must be a subset of: {valid}."
        )
    return AnalysisSettings(
        unit=_parse_enum(AnalysisUnit, mapping.get("unit", AnalysisUnit.DATASET.value), "analysis.unit"),
        error_column=error_column,
        selection_error_column=selection_error_column,
        method_variant=method_variants,
        exclude_methods_containing=exclude_methods,
        exclude_problem_types=exclude_problem_types,
    )


def _parse_metafeatures(raw_metafeatures: Any) -> MetafeatureSettings:
    mapping = _require_mapping(raw_metafeatures or {}, "metafeatures")

    def _sequence_value(key: str, default: list[str]) -> tuple[str, ...]:
        value = mapping.get(key, default)
        if value is None:
            value = default
        return tuple(value)

    raw_timeout = mapping.get("pymfe_per_feature_timeout_s")
    if raw_timeout is None:
        pymfe_per_feature_timeout_s: float | None = None
    else:
        if isinstance(raw_timeout, bool):
            raise ConfigValidationError(
                "`metafeatures.pymfe_per_feature_timeout_s` must be null or a positive finite number."
            )
        try:
            pymfe_per_feature_timeout_s = float(raw_timeout)
        except (TypeError, ValueError) as err:
            raise ConfigValidationError(
                "`metafeatures.pymfe_per_feature_timeout_s` must be null or a positive finite number."
            ) from err
        if not math.isfinite(pymfe_per_feature_timeout_s) or pymfe_per_feature_timeout_s <= 0:
            raise ConfigValidationError(
                "`metafeatures.pymfe_per_feature_timeout_s` must be null or a positive finite number."
            )

    return MetafeatureSettings(
        feature_sets=_sequence_value("feature_sets", ["basic", "irregularity"]),
        pymfe_groups=_sequence_value("pymfe_groups", ["general", "statistical", "info-theory"]),
        pymfe_summary=_sequence_value("pymfe_summary", ["mean", "sd"]),
        pymfe_per_feature_timeout_s=pymfe_per_feature_timeout_s,
        retry_failed_pymfe=bool(mapping.get("retry_failed_pymfe", False)),
        trace=bool(mapping.get("trace", False)),
        irregularity_components=_sequence_value(
            "irregularity_components",
            [
                "irreg_min_cov_eig",
                "irreg_std_skew",
                "irreg_range_skew",
                "irreg_iqr_hmean",
                "irreg_kurtosis_std",
            ],
        ),
    )


def _parse_statistics(raw_statistics: Any) -> StatisticsSettings:
    mapping = _require_mapping(raw_statistics or {}, "statistics")
    raw_fdr = mapping.get("fdr_method", FDRMethod.BH.value)
    fdr_method = None if raw_fdr is None else _parse_enum(FDRMethod, raw_fdr, "statistics.fdr_method")
    return StatisticsSettings(
        correlation_method=_parse_enum(
            CorrelationMethod,
            mapping.get("correlation_method", CorrelationMethod.SPEARMAN.value),
            "statistics.correlation_method",
        ),
        alpha=float(mapping.get("alpha", 0.05)),
        fdr_method=fdr_method,
        confidence_interval=bool(mapping.get("confidence_interval", True)),
        ci_bootstrap_samples=int(mapping.get("ci_bootstrap_samples", 10_000)),
        ci_confidence_level=float(mapping.get("ci_confidence_level", 0.95)),
        multivariate=bool(mapping.get("multivariate", False)),
        multivariate_method=_parse_enum(
            MultivariateMethod,
            mapping.get("multivariate_method", MultivariateMethod.OLS.value),
            "statistics.multivariate_method",
        ),
    )


def _parse_cache(raw_cache: Any) -> CacheConfig:
    mapping = _require_mapping(raw_cache or {}, "cache")
    stages_mapping = _require_mapping(mapping.get("stages", {}), "cache.stages")
    return CacheConfig(
        enabled=bool(mapping.get("enabled", True)),
        directory=Path(mapping.get("directory", ".mfa_cache")),
        stages=CacheStageConfig(
            raw_results=bool(stages_mapping.get("raw_results", True)),
            metafeatures=bool(stages_mapping.get("metafeatures", True)),
            gaps=bool(stages_mapping.get("gaps", True)),
            statistics=bool(stages_mapping.get("statistics", True)),
        ),
    )


def _parse_parallelism(raw_parallelism: Any) -> ParallelismSettings:
    mapping = _require_mapping(raw_parallelism or {}, "parallelism")
    n_jobs = int(mapping.get("n_jobs", 1))
    backend = str(mapping.get("backend", "process"))
    if backend not in VALID_BACKENDS:
        raise ConfigValidationError(f"`parallelism.backend` must be one of: {', '.join(sorted(VALID_BACKENDS))}.")
    return ParallelismSettings(n_jobs=n_jobs, backend=backend)


def parse_config(raw_config: dict[str, Any]) -> AnalysisConfig:
    if "version" not in raw_config:
        raise ConfigValidationError("`version` is required.")
    groups = _parse_groups(raw_config.get("groups"))
    return AnalysisConfig(
        version=int(raw_config["version"]),
        groups=groups,
        comparisons=_parse_comparisons(raw_config.get("comparisons"), groups),
        analysis=_parse_analysis(raw_config.get("analysis")),
        metafeatures=_parse_metafeatures(raw_config.get("metafeatures")),
        statistics=_parse_statistics(raw_config.get("statistics")),
        cache=_parse_cache(raw_config.get("cache")),
        parallelism=_parse_parallelism(raw_config.get("parallelism")),
    )


def _resolve_project_root(config_path: Path) -> Path:
    resolved_path = config_path.expanduser().resolve()
    if resolved_path.parent.name == "configs":
        return resolved_path.parent.parent
    return resolved_path.parent


def load_config(path: str | Path) -> AnalysisConfig:
    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as infile:
        raw_config = yaml.safe_load(infile) or {}
    if not isinstance(raw_config, dict):
        raise ConfigValidationError("Top-level YAML document must be a mapping.")
    config = parse_config(raw_config)
    if config.cache.directory.is_absolute():
        return config
    project_root = _resolve_project_root(config_path)
    return replace(
        config,
        cache=replace(
            config.cache,
            directory=(project_root / config.cache.directory).resolve(),
        ),
    )
