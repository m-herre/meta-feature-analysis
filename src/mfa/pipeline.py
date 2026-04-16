from __future__ import annotations

from collections.abc import Sequence

from .aggregation import build_analysis_table
from .cache import (
    compute_config_hash,
    compute_stage_hash,
    read_dataframe_cache,
    read_json_cache,
    write_dataframe_cache,
    write_json_cache,
)
from .config import AnalysisConfig
from .data.loader import load_tabarena_results
from .groups import validate_groups_against_data
from .metafeatures import build_metafeature_table
from .stats.correlation import correlate_all
from .stats.correction import apply_fdr_correction
from .stats.multivariate import run_multivariate
from .types import AnalysisResult, CorrelationResult, CorrectionResult, MultivariateResult
from .gaps.pairwise import compute_pairwise_gaps


def _get_task_metadata(task_metadata=None):
    if task_metadata is not None:
        return task_metadata
    from tabarena.nips2025_utils.fetch_metadata import load_task_metadata

    return load_task_metadata()


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
    config_hash = compute_config_hash(config.to_dict())
    cache_version_hash = _cache_version_hash(config)
    cache_dir = config.cache.directory
    dataset_list = sorted(datasets) if datasets is not None else None

    raw_hash = compute_stage_hash(
        "raw_results",
        cache_version_hash,
        {
            "datasets": dataset_list,
            "method_variant": config.analysis.method_variant,
            "exclude_methods_containing": config.analysis.exclude_methods_containing,
        },
    )
    raw_results = None
    if config.cache.enabled and config.cache.stages.raw_results:
        raw_results = read_dataframe_cache(cache_dir, 1, "raw_results", raw_hash)
    if raw_results is None:
        raw_results = load_tabarena_results(
            config,
            datasets=dataset_list,
            tabarena_context=tabarena_context,
        )
        if config.cache.enabled and config.cache.stages.raw_results:
            write_dataframe_cache(raw_results, cache_dir, 1, "raw_results", raw_hash)

    validate_groups_against_data(raw_results, config.groups)

    metadata = _get_task_metadata(task_metadata)
    metafeature_hash = compute_stage_hash(
        "metafeatures",
        cache_version_hash,
        {
            "datasets": dataset_list,
            "feature_sets": config.metafeatures.feature_sets,
            "pymfe_groups": config.metafeatures.pymfe_groups,
            "pymfe_summary": config.metafeatures.pymfe_summary,
            "irregularity_components": config.metafeatures.irregularity_components,
        },
    )
    metafeature_table = None
    if config.cache.enabled and config.cache.stages.metafeatures:
        metafeature_table = read_dataframe_cache(cache_dir, 2, "metafeatures", metafeature_hash)
    if metafeature_table is None:
        metafeature_table = build_metafeature_table(
            metadata,
            datasets=None if dataset_list is None else list(dataset_list),
            feature_sets=config.metafeatures.feature_sets,
            cache_dir=cache_dir,
            use_cache=config.cache.enabled and config.cache.stages.metafeatures,
            pymfe_groups=config.metafeatures.pymfe_groups,
            pymfe_summary=config.metafeatures.pymfe_summary,
            irregularity_components=config.metafeatures.irregularity_components,
            cache_version=config.version,
        )
        if config.cache.enabled and config.cache.stages.metafeatures:
            write_dataframe_cache(metafeature_table, cache_dir, 2, "metafeatures", metafeature_hash)

    gap_hash = compute_stage_hash(
        "gaps",
        raw_hash,
        {
            "comparisons": _comparison_cache_payload(config.comparisons),
            "error_column": config.analysis.error_column,
        },
    )
    gap_table = None
    if config.cache.enabled and config.cache.stages.gaps:
        gap_table = read_dataframe_cache(cache_dir, 3, "gaps", gap_hash)
    if gap_table is None:
        gap_table = compute_pairwise_gaps(
            raw_results,
            config.comparisons,
            error_column=config.analysis.error_column,
        )
        if config.cache.enabled and config.cache.stages.gaps:
            write_dataframe_cache(gap_table, cache_dir, 3, "gaps", gap_hash)

    analysis_table = build_analysis_table(
        gap_table,
        metafeature_table,
        unit=config.analysis.unit,
    )
    predictor_columns = [
        column
        for column in metafeature_table.columns
        if column not in {"dataset", "repeat", "fold"} and not column.startswith("_")
    ]
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

    if stats_payload is None:
        correlation_results = correlate_all(
            analysis_table,
            comparisons=config.comparisons,
            predictors=predictor_columns,
            target="delta_norm",
            method=config.statistics.correlation_method,
            confidence_interval=config.statistics.confidence_interval,
            ci_bootstrap_samples=config.statistics.ci_bootstrap_samples,
            ci_confidence_level=config.statistics.ci_confidence_level,
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
    else:
        correlation_results = [_deserialize_correlation_result(item) for item in stats_payload["correlation_results"]]
        correction_result = _deserialize_correction_result(stats_payload["correction_result"])
        multivariate_result = _deserialize_multivariate_result(stats_payload["multivariate_result"])

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
