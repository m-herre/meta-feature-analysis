from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .types import AnalysisUnit, ComparisonSpec, CorrelationMethod, FDRMethod, GroupDef, MultivariateMethod

VALID_METHOD_VARIANTS = {"default", "tuned", "tuned_ensemble"}
VALID_EXPECTED_DIRECTIONS = {None, "positive", "negative"}


class ConfigValidationError(ValueError):
    """Raised when the YAML configuration is invalid."""


@dataclass(frozen=True)
class AnalysisSettings:
    unit: AnalysisUnit = AnalysisUnit.DATASET
    error_column: str = "metric_error"
    selection_error_column: str | None = "metric_error_val"
    method_variant: str = "tuned"
    exclude_methods_containing: tuple[str, ...] = ()


@dataclass(frozen=True)
class MetafeatureSettings:
    feature_sets: tuple[str, ...] = ("basic", "irregularity")
    pymfe_groups: tuple[str, ...] = ("general", "statistical", "info-theory")
    pymfe_summary: tuple[str, ...] = ("mean", "sd")
    irregularity_components: tuple[str, ...] = (
        "irreg_min_cov_eig",
        "irreg_std_skew",
        "irreg_range_skew",
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
            raise ConfigValidationError(
                f"`{prefix}.expected_direction` must be one of: positive, negative, or null."
            )
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
    method_variant = mapping.get("method_variant", "tuned")
    if method_variant not in VALID_METHOD_VARIANTS:
        valid_variants = ", ".join(sorted(VALID_METHOD_VARIANTS))
        raise ConfigValidationError(f"`analysis.method_variant` must be one of: {valid_variants}.")
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
    return AnalysisSettings(
        unit=_parse_enum(AnalysisUnit, mapping.get("unit", AnalysisUnit.DATASET.value), "analysis.unit"),
        error_column=error_column,
        selection_error_column=selection_error_column,
        method_variant=method_variant,
        exclude_methods_containing=exclude_methods,
    )


def _parse_metafeatures(raw_metafeatures: Any) -> MetafeatureSettings:
    mapping = _require_mapping(raw_metafeatures or {}, "metafeatures")
    return MetafeatureSettings(
        feature_sets=tuple(mapping.get("feature_sets", ["basic", "irregularity"])),
        pymfe_groups=tuple(mapping.get("pymfe_groups", ["general", "statistical", "info-theory"])),
        pymfe_summary=tuple(mapping.get("pymfe_summary", ["mean", "sd"])),
        irregularity_components=tuple(
            mapping.get(
                "irregularity_components",
                [
                    "irreg_min_cov_eig",
                    "irreg_std_skew",
                    "irreg_range_skew",
                    "irreg_kurtosis_std",
                ],
            )
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
    )


def load_config(path: str | Path) -> AnalysisConfig:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as infile:
        raw_config = yaml.safe_load(infile) or {}
    if not isinstance(raw_config, dict):
        raise ConfigValidationError("Top-level YAML document must be a mapping.")
    return parse_config(raw_config)
