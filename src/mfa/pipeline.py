from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

from ._logging import get_logger
from .aggregation import build_analysis_table
from .cache import (
    compute_config_hash,
    compute_stage_hash,
    read_dataframe_cache,
    write_dataframe_cache,
)
from .config import AnalysisConfig, normalize_method_variants
from .data.loader import load_tabarena_results
from .gaps.pairwise import compute_pairwise_gaps
from .groups import validate_groups_against_data
from .metafeatures import build_metafeature_table
from .metafeatures.basic import BASIC_METAFEATURE_SCHEMA_VERSION
from .metafeatures.irregularity import IRREGULARITY_PROXY_SCHEMA_VERSION
from .metafeatures.pymfe_catalog import PYMFE_FILTER_SCHEMA_VERSION
from .metafeatures.redundancy import REDUNDANCY_METAFEATURE_SCHEMA_VERSION
from .parallel import resolve_n_jobs
from .types import AnalysisResult

logger = get_logger(__name__)


def _dataset_scope_label(datasets: Sequence[str] | None) -> str:
    if datasets is None:
        return "all benchmark datasets"
    return f"{len(datasets)} selected dataset(s)"


def _frame_summary(df: pd.DataFrame) -> str:
    parts = [f"{len(df)} rows"]
    if "dataset" in df.columns:
        parts.append(f"{df['dataset'].nunique()} dataset(s)")
    return ", ".join(parts)


def _get_task_metadata(task_metadata=None):
    if task_metadata is not None:
        return task_metadata
    from tabarena.nips2025_utils.fetch_metadata import load_task_metadata

    return load_task_metadata()


def _apply_problem_type_filter(
    dataset_list: list[str] | None,
    task_metadata: pd.DataFrame,
    *,
    excluded_problem_types: Sequence[str],
) -> list[str] | None:
    if not excluded_problem_types:
        return dataset_list
    if "problem_type" not in task_metadata.columns or "dataset" not in task_metadata.columns:
        raise ValueError(
            "task_metadata must include `dataset` and `problem_type` columns to apply `exclude_problem_types`."
        )
    excluded = set(excluded_problem_types)
    allowed_datasets = task_metadata.loc[~task_metadata["problem_type"].isin(excluded), "dataset"].tolist()
    allowed_set = set(allowed_datasets)
    if dataset_list is None:
        return sorted(allowed_set)
    return sorted(set(dataset_list) & allowed_set)


def _metadata_problem_types_payload(task_metadata: pd.DataFrame, dataset_list: list[str] | None) -> list[dict] | None:
    if "problem_type" not in task_metadata.columns:
        return None
    name_columns = [column for column in ("dataset", "name") if column in task_metadata.columns]
    if not name_columns:
        return None
    metadata = task_metadata.copy()
    if dataset_list is not None:
        metadata = metadata[metadata[name_columns].isin(set(dataset_list)).any(axis=1)].copy()
    name_column = "dataset" if "dataset" in metadata.columns else name_columns[0]
    payload = metadata[[name_column, "problem_type"]].rename(columns={name_column: "dataset"})
    payload = payload.drop_duplicates().sort_values(["dataset", "problem_type"], na_position="last")
    payload = payload.where(pd.notna(payload), None)
    return payload.to_dict(orient="records")


def _schema_versions_for_feature_sets(feature_sets: tuple[str, ...]) -> dict[str, int]:
    schema_versions: dict[str, int] = {}
    if "basic" in feature_sets:
        schema_versions["basic"] = BASIC_METAFEATURE_SCHEMA_VERSION
    if "irregularity" in feature_sets:
        schema_versions["irregularity"] = IRREGULARITY_PROXY_SCHEMA_VERSION
    if "redundancy" in feature_sets:
        schema_versions["redundancy"] = REDUNDANCY_METAFEATURE_SCHEMA_VERSION
    if "pymfe" in feature_sets:
        schema_versions["pymfe_filter"] = PYMFE_FILTER_SCHEMA_VERSION
    return schema_versions


def _cache_version_hash(config: AnalysisConfig) -> str:
    return compute_config_hash({"version": config.version})


def _comparison_cache_payload(comparisons) -> list[dict]:
    return [
        {
            "name": comparison.name,
            "group_a": {
                "name": comparison.group_a.name,
                "label": comparison.group_a.label,
                "config_types": sorted(comparison.group_a.config_types),
            },
            "group_b": {
                "name": comparison.group_b.name,
                "label": comparison.group_b.label,
                "config_types": sorted(comparison.group_b.config_types),
            },
            "expected_direction": comparison.expected_direction,
        }
        for comparison in comparisons
    ]


def run_analysis(
    config: AnalysisConfig,
    *,
    datasets: Sequence[str] | None = None,
    task_metadata=None,
    tabarena_context=None,
) -> AnalysisResult:
    """Run the full meta-feature analysis pipeline."""
    method_variants = normalize_method_variants(config.analysis.method_variant)
    config_hash = compute_config_hash(config.to_dict())
    cache_version_hash = _cache_version_hash(config)
    cache_dir = config.cache.directory
    dataset_list = sorted(datasets) if datasets is not None else None
    metadata = _get_task_metadata(task_metadata)
    dataset_list = _apply_problem_type_filter(
        dataset_list,
        metadata,
        excluded_problem_types=config.analysis.exclude_problem_types,
    )
    comparison_names = ", ".join(comparison.name for comparison in config.comparisons)
    n_jobs = resolve_n_jobs(config.parallelism.n_jobs)
    backend = config.parallelism.backend
    logger.info(
        "Starting analysis: comparisons=%s; scope=%s; unit=%s; method_variant=%s; n_jobs=%d",
        comparison_names,
        _dataset_scope_label(dataset_list),
        config.analysis.unit.value,
        ",".join(method_variants),
        n_jobs,
    )

    raw_hash = compute_stage_hash(
        "raw_results",
        cache_version_hash,
        {
            "datasets": dataset_list,
            "method_variant": method_variants,
            "exclude_methods_containing": config.analysis.exclude_methods_containing,
            "error_column": config.analysis.error_column,
            "selection_error_column": config.analysis.selection_error_column,
            "imputed_metric_policy": "na_keep_flag",
        },
    )
    raw_results = None
    if config.cache.enabled and config.cache.stages.raw_results:
        raw_results = read_dataframe_cache(cache_dir, 1, "raw_results", raw_hash)
        if raw_results is not None:
            logger.info("Stage 1/4 raw results: cache hit (%s)", _frame_summary(raw_results))
    if raw_results is None:
        logger.info("Stage 1/4 raw results: loading from TabArena")
        raw_results = load_tabarena_results(
            config,
            datasets=dataset_list,
            tabarena_context=tabarena_context,
        )
        if config.cache.enabled and config.cache.stages.raw_results:
            write_dataframe_cache(raw_results, cache_dir, 1, "raw_results", raw_hash)
        logger.info("Stage 1/4 raw results: ready (%s)", _frame_summary(raw_results))

    validate_groups_against_data(raw_results, config.groups)

    metafeature_cache_hash = compute_stage_hash(
        "metafeatures",
        cache_version_hash,
        {
            "datasets": dataset_list,
            "feature_sets": config.metafeatures.feature_sets,
            "pymfe_groups": config.metafeatures.pymfe_groups,
            "pymfe_summary": config.metafeatures.pymfe_summary,
            "retry_failed_pymfe": config.metafeatures.retry_failed_pymfe,
            "irregularity_components": config.metafeatures.irregularity_components,
            "schema_versions": _schema_versions_for_feature_sets(config.metafeatures.feature_sets),
            "problem_types": _metadata_problem_types_payload(metadata, dataset_list),
        },
    )
    metafeature_cache_enabled = config.cache.enabled and config.cache.stages.metafeatures
    pymfe_enabled = "pymfe" in config.metafeatures.feature_sets
    metafeature_table = None
    if config.metafeatures.trace and config.cache.enabled and config.cache.stages.metafeatures:
        logger.info(
            "Stage 2/4 meta-features: trace enabled; metafeature caches remain active, "
            "so live per-split diagnostics appear only on cache misses"
        )
    if metafeature_cache_enabled and pymfe_enabled:
        if config.metafeatures.retry_failed_pymfe:
            logger.info(
                "Stage 2/4 meta-features: pymfe enabled; rebuilding from split cache to allow partial-cache repair"
            )
        else:
            logger.info(
                "Stage 2/4 meta-features: pymfe enabled; rebuilding from split cache and reusing cached "
                "pymfe failures/incomplete outputs as-is"
            )
    if metafeature_cache_enabled and not pymfe_enabled:
        metafeature_table = read_dataframe_cache(cache_dir, 2, "metafeatures", metafeature_cache_hash)
        if metafeature_table is not None:
            logger.info("Stage 2/4 meta-features: cache hit (%s)", _frame_summary(metafeature_table))
    if metafeature_table is None:
        logger.info("Stage 2/4 meta-features: building for %s", _dataset_scope_label(dataset_list))
        metafeature_table = build_metafeature_table(
            metadata,
            datasets=None if dataset_list is None else list(dataset_list),
            feature_sets=config.metafeatures.feature_sets,
            cache_dir=cache_dir,
            use_cache=metafeature_cache_enabled,
            pymfe_groups=config.metafeatures.pymfe_groups,
            pymfe_summary=config.metafeatures.pymfe_summary,
            pymfe_per_feature_timeout_s=config.metafeatures.pymfe_per_feature_timeout_s,
            retry_failed_pymfe=config.metafeatures.retry_failed_pymfe,
            trace=config.metafeatures.trace,
            irregularity_components=config.metafeatures.irregularity_components,
            cache_version=config.version,
            n_jobs=n_jobs,
            backend=backend,
        )
        if metafeature_cache_enabled:
            write_dataframe_cache(metafeature_table, cache_dir, 2, "metafeatures", metafeature_cache_hash)
        logger.info("Stage 2/4 meta-features: ready (%s)", _frame_summary(metafeature_table))
    gap_hash = compute_stage_hash(
        "gaps",
        raw_hash,
        {
            "comparisons": _comparison_cache_payload(config.comparisons),
            "error_column": config.analysis.error_column,
            "selection_error_column": config.analysis.selection_error_column,
        },
    )
    gap_table = None
    if config.cache.enabled and config.cache.stages.gaps:
        gap_table = read_dataframe_cache(cache_dir, 3, "gaps", gap_hash)
        if gap_table is not None:
            logger.info("Stage 3/4 pairwise gaps: cache hit (%s)", _frame_summary(gap_table))
    if gap_table is None:
        logger.info("Stage 3/4 pairwise gaps: computing %d comparison(s)", len(config.comparisons))
        gap_table = compute_pairwise_gaps(
            raw_results,
            config.comparisons,
            error_column=config.analysis.error_column,
            selection_error_column=config.analysis.selection_error_column,
        )
        if config.cache.enabled and config.cache.stages.gaps:
            write_dataframe_cache(gap_table, cache_dir, 3, "gaps", gap_hash)
        logger.info("Stage 3/4 pairwise gaps: ready (%s)", _frame_summary(gap_table))

    logger.info("Stage 4/4 analysis table: joining and aggregating at %s level", config.analysis.unit.value)
    analysis_table = build_analysis_table(
        gap_table,
        metafeature_table,
        unit=config.analysis.unit,
        irregularity_components=config.metafeatures.irregularity_components,
    )
    logger.info("Stage 4/4 analysis table: ready (%s)", _frame_summary(analysis_table))

    logger.info("Analysis complete: config_hash=%s", config_hash)
    return AnalysisResult(
        config_hash=config_hash,
        comparison_name=config.comparisons[0].name if len(config.comparisons) == 1 else None,
        gap_table=gap_table,
        metafeature_table=metafeature_table,
        analysis_table=analysis_table,
    )
