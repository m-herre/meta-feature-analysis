from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

from ._logging import get_logger
from .aggregation import build_analysis_table
from .cache import (
    compute_config_hash,
    compute_stage_hash,
    read_dataframe_cache,
    read_json_cache,
    write_dataframe_cache,
    write_json_cache,
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
from .stats.correction import apply_fdr_correction
from .stats.correlation import correlate_all
from .stats.multivariate import run_multivariate
from .types import AnalysisResult, CorrectionResult, CorrelationResult, MultivariateResult

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


def _statistics_summary(
    correlation_results: Sequence[CorrelationResult],
    correction_result: CorrectionResult | None,
    multivariate_result: MultivariateResult | None,
) -> str:
    parts = [f"{len(correlation_results)} correlation test(s)"]
    if correction_result is not None:
        parts.append(f"{sum(correction_result.rejected)} rejected after {correction_result.method}")
    if multivariate_result is not None:
        parts.append("multivariate model fitted")
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


def _serialize_correlation_result(result: CorrelationResult) -> dict:
    return {
        "comparison_name": result.comparison_name,
        "predictor": result.predictor,
        "target": result.target,
        "statistic": result.statistic,
        "p_value": result.p_value,
        "n_observations": result.n_observations,
        "ci_lower": result.ci_lower,
        "ci_upper": result.ci_upper,
        "direction_confirmed": result.direction_confirmed,
    }


def _deserialize_correlation_result(payload: dict) -> CorrelationResult:
    return CorrelationResult(**payload)


def _serialize_correction_result(result: CorrectionResult | None) -> dict | None:
    if result is None:
        return None
    return {
        "method": result.method,
        "alpha": result.alpha,
        "results": [_serialize_correlation_result(item) for item in result.results],
        "adjusted_p_values": list(result.adjusted_p_values),
        "rejected": list(result.rejected),
    }


def _deserialize_correction_result(payload: dict | None) -> CorrectionResult | None:
    if payload is None:
        return None
    return CorrectionResult(
        method=payload["method"],
        alpha=payload["alpha"],
        results=tuple(_deserialize_correlation_result(item) for item in payload["results"]),
        adjusted_p_values=tuple(payload["adjusted_p_values"]),
        rejected=tuple(payload["rejected"]),
    )


def _serialize_multivariate_result(result: MultivariateResult | None) -> dict | None:
    if result is None:
        return None
    return {
        "comparison_name": result.comparison_name,
        "predictors": list(result.predictors),
        "coefficients": result.coefficients,
        "p_values": result.p_values,
        "r_squared": result.r_squared,
        "adj_r_squared": result.adj_r_squared,
        "vif": result.vif,
        "n_observations": result.n_observations,
    }


def _deserialize_multivariate_result(payload: dict | None) -> MultivariateResult | None:
    if payload is None:
        return None
    return MultivariateResult(
        comparison_name=payload["comparison_name"],
        predictors=tuple(payload["predictors"]),
        coefficients=payload["coefficients"],
        p_values=payload["p_values"],
        r_squared=payload["r_squared"],
        adj_r_squared=payload["adj_r_squared"],
        vif=payload["vif"],
        n_observations=payload["n_observations"],
    )


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


def _analysis_stage_hash(*, gap_hash: str, metafeature_hash: str, unit: str) -> str:
    return compute_stage_hash(
        "analysis_table",
        compute_config_hash({"gap_hash": gap_hash, "metafeature_hash": metafeature_hash}),
        {"unit": unit},
    )


def _dataframe_content_hash(df: pd.DataFrame) -> str:
    stable = df.copy()
    sort_columns = [column for column in ("dataset", "repeat", "fold") if column in stable.columns]
    if sort_columns:
        stable = stable.sort_values(sort_columns).reset_index(drop=True)
    stable = stable.reindex(sorted(stable.columns), axis=1)
    payload = stable.to_json(orient="split", index=False, date_format="iso", double_precision=15)
    return compute_config_hash({"dataframe": payload})


def _statistics_cache_params(
    config: AnalysisConfig,
    *,
    predictor_columns: Sequence[str],
    target: str,
) -> dict:
    return {
        "correlation_method": config.statistics.correlation_method.value,
        "predictors": list(predictor_columns),
        "target": target,
        "fdr_method": None if config.statistics.fdr_method is None else config.statistics.fdr_method.value,
        "alpha": config.statistics.alpha,
        "confidence_interval": config.statistics.confidence_interval,
        "ci_bootstrap_samples": config.statistics.ci_bootstrap_samples,
        "ci_confidence_level": config.statistics.ci_confidence_level,
        "multivariate": config.statistics.multivariate,
        "multivariate_method": config.statistics.multivariate_method.value,
    }


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
            logger.info("Stage 1/5 raw results: cache hit (%s)", _frame_summary(raw_results))
    if raw_results is None:
        logger.info("Stage 1/5 raw results: loading from TabArena")
        raw_results = load_tabarena_results(
            config,
            datasets=dataset_list,
            tabarena_context=tabarena_context,
        )
        if config.cache.enabled and config.cache.stages.raw_results:
            write_dataframe_cache(raw_results, cache_dir, 1, "raw_results", raw_hash)
        logger.info("Stage 1/5 raw results: ready (%s)", _frame_summary(raw_results))

    validate_groups_against_data(raw_results, config.groups)

    metafeature_cache_hash = compute_stage_hash(
        "metafeatures",
        cache_version_hash,
        {
            "datasets": dataset_list,
            "feature_sets": config.metafeatures.feature_sets,
            "pymfe_groups": config.metafeatures.pymfe_groups,
            "pymfe_summary": config.metafeatures.pymfe_summary,
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
            "Stage 2/5 meta-features: trace enabled; metafeature caches remain active, "
            "so live per-split diagnostics appear only on cache misses"
        )
    if metafeature_cache_enabled and pymfe_enabled:
        logger.info(
            "Stage 2/5 meta-features: pymfe enabled; rebuilding from split cache to allow partial-cache repair"
        )
    if metafeature_cache_enabled and not pymfe_enabled:
        metafeature_table = read_dataframe_cache(cache_dir, 2, "metafeatures", metafeature_cache_hash)
        if metafeature_table is not None:
            logger.info("Stage 2/5 meta-features: cache hit (%s)", _frame_summary(metafeature_table))
    if metafeature_table is None:
        logger.info("Stage 2/5 meta-features: building for %s", _dataset_scope_label(dataset_list))
        metafeature_table = build_metafeature_table(
            metadata,
            datasets=None if dataset_list is None else list(dataset_list),
            feature_sets=config.metafeatures.feature_sets,
            cache_dir=cache_dir,
            use_cache=metafeature_cache_enabled,
            pymfe_groups=config.metafeatures.pymfe_groups,
            pymfe_summary=config.metafeatures.pymfe_summary,
            pymfe_per_feature_timeout_s=config.metafeatures.pymfe_per_feature_timeout_s,
            trace=config.metafeatures.trace,
            irregularity_components=config.metafeatures.irregularity_components,
            cache_version=config.version,
            n_jobs=n_jobs,
            backend=backend,
        )
        if metafeature_cache_enabled:
            write_dataframe_cache(metafeature_table, cache_dir, 2, "metafeatures", metafeature_cache_hash)
        logger.info("Stage 2/5 meta-features: ready (%s)", _frame_summary(metafeature_table))
    metafeature_hash = compute_stage_hash(
        "metafeatures_content",
        metafeature_cache_hash,
        {"content_hash": _dataframe_content_hash(metafeature_table)},
    )

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
            logger.info("Stage 3/5 pairwise gaps: cache hit (%s)", _frame_summary(gap_table))
    if gap_table is None:
        logger.info("Stage 3/5 pairwise gaps: computing %d comparison(s)", len(config.comparisons))
        gap_table = compute_pairwise_gaps(
            raw_results,
            config.comparisons,
            error_column=config.analysis.error_column,
            selection_error_column=config.analysis.selection_error_column,
        )
        if config.cache.enabled and config.cache.stages.gaps:
            write_dataframe_cache(gap_table, cache_dir, 3, "gaps", gap_hash)
        logger.info("Stage 3/5 pairwise gaps: ready (%s)", _frame_summary(gap_table))

    logger.info("Stage 4/5 analysis table: joining and aggregating at %s level", config.analysis.unit.value)
    analysis_table = build_analysis_table(
        gap_table,
        metafeature_table,
        unit=config.analysis.unit,
        irregularity_components=config.metafeatures.irregularity_components,
    )
    logger.info("Stage 4/5 analysis table: ready (%s)", _frame_summary(analysis_table))
    predictor_columns = [
        column
        for column in metafeature_table.columns
        if column not in {"dataset", "repeat", "fold"} and not column.startswith("_")
    ]
    # The composite ``irregularity`` is attached during stage-4 aggregation
    # (z-scoring is over datasets, not splits), so it is absent from the
    # stage-2 metafeature table. Surface it as a predictor when available.
    if "irregularity" in analysis_table.columns and "irregularity" not in predictor_columns:
        predictor_columns.append("irregularity")
    analysis_hash = _analysis_stage_hash(
        gap_hash=gap_hash,
        metafeature_hash=metafeature_hash,
        unit=config.analysis.unit.value,
    )

    stats_hash = compute_stage_hash(
        "statistics",
        analysis_hash,
        _statistics_cache_params(config, predictor_columns=predictor_columns, target="delta_norm"),
    )
    stats_payload = None
    if config.cache.enabled and config.cache.stages.statistics:
        stats_payload = read_json_cache(cache_dir, 5, "statistics", stats_hash)
        if stats_payload is not None:
            cached_correlation_results = [
                _deserialize_correlation_result(item) for item in stats_payload["correlation_results"]
            ]
            cached_correction_result = _deserialize_correction_result(stats_payload["correction_result"])
            cached_multivariate_result = _deserialize_multivariate_result(stats_payload["multivariate_result"])
            logger.info(
                "Stage 5/5 statistics: cache hit (%s)",
                _statistics_summary(
                    cached_correlation_results,
                    cached_correction_result,
                    cached_multivariate_result,
                ),
            )

    if stats_payload is None:
        logger.info("Stage 5/5 statistics: running correlation tests")
        correlation_results = correlate_all(
            analysis_table,
            comparisons=config.comparisons,
            predictors=predictor_columns,
            target="delta_norm",
            method=config.statistics.correlation_method,
            confidence_interval=config.statistics.confidence_interval,
            ci_bootstrap_samples=config.statistics.ci_bootstrap_samples,
            ci_confidence_level=config.statistics.ci_confidence_level,
            n_jobs=n_jobs,
            backend=backend,
        )
        correction_result = apply_fdr_correction(
            correlation_results,
            method=config.statistics.fdr_method,
            alpha=config.statistics.alpha,
        )
        multivariate_result = None
        if config.statistics.multivariate and len(config.comparisons) == 1:
            multivariate_result = run_multivariate(
                analysis_table,
                comparison_name=config.comparisons[0].name,
                predictors=predictor_columns,
                target="delta_norm",
                method=config.statistics.multivariate_method,
            )
        stats_payload = {
            "correlation_results": [_serialize_correlation_result(result) for result in correlation_results],
            "correction_result": _serialize_correction_result(correction_result),
            "multivariate_result": _serialize_multivariate_result(multivariate_result),
        }
        if config.cache.enabled and config.cache.stages.statistics:
            write_json_cache(stats_payload, cache_dir, 5, "statistics", stats_hash)
        logger.info(
            "Stage 5/5 statistics: ready (%s)",
            _statistics_summary(correlation_results, correction_result, multivariate_result),
        )
    else:
        correlation_results = [_deserialize_correlation_result(item) for item in stats_payload["correlation_results"]]
        correction_result = _deserialize_correction_result(stats_payload["correction_result"])
        multivariate_result = _deserialize_multivariate_result(stats_payload["multivariate_result"])

    logger.info("Analysis complete: config_hash=%s", config_hash)
    return AnalysisResult(
        config_hash=config_hash,
        comparison_name=config.comparisons[0].name if len(config.comparisons) == 1 else None,
        correlation_results=correlation_results,
        correction_result=correction_result,
        multivariate_result=multivariate_result,
        gap_table=gap_table,
        metafeature_table=metafeature_table,
        analysis_table=analysis_table,
    )
