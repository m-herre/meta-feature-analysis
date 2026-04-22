from __future__ import annotations

import hashlib
import json
import time
from concurrent.futures import as_completed
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path
from typing import Any

import pandas as pd

from .._logging import get_logger
from ..cache import metafeature_split_cache_dir
from ..parallel import get_executor, resolve_n_jobs
from .basic import BASIC_METAFEATURE_SCHEMA_VERSION
from .irregularity import (
    DEFAULT_IRREGULARITY_COMPONENTS,
    IRREGULARITY_PROXY_SCHEMA_VERSION,
)
from .pymfe_catalog import PYMFE_FILTER_SCHEMA_VERSION
from .pymfe_features import enumerate_pymfe_raw_features
from .redundancy import REDUNDANCY_METAFEATURE_SCHEMA_VERSION
from .registry import extract_requested_metafeatures

logger = get_logger(__name__)


def _split_trace_label(dataset: str, repeat: int, fold: int) -> str:
    return f"{dataset} r{repeat} f{fold}"


def _format_elapsed(seconds: float) -> str:
    total_seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def _stable_feature_hash(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def _metadata_dataset_name(row) -> str:
    for column in ("dataset", "name"):
        value = getattr(row, column, None)
        if isinstance(value, str):
            return value
    raise ValueError("Task metadata must include either `dataset` or `name`.")


def _metadata_task_id(row) -> int:
    for column in ("tid", "task_id"):
        value = getattr(row, column, None)
        if value is not None:
            return int(value)
    raise ValueError("Task metadata must include either `tid` or `task_id`.")


def _metadata_problem_type(row) -> str | None:
    value = getattr(row, "problem_type", None)
    if isinstance(value, str) and value:
        return value.lower()
    return None


SPLIT_CACHE_HASH_COLUMN = "_cache_identity_hash"
SPLIT_CACHE_FAILED_SETS_COLUMN = "_cache_failed_sets"
_SPLIT_CACHE_INTERNAL_COLUMNS = (
    SPLIT_CACHE_HASH_COLUMN,
    SPLIT_CACHE_FAILED_SETS_COLUMN,
    "_feature_set_hash",
)


def _split_cache_path(cache_dir: Path, dataset: str, repeat: int, fold: int) -> Path:
    return metafeature_split_cache_dir(cache_dir) / f"{dataset}__r{repeat}__f{fold}.parquet"


def _metadata_split_dimensions(row) -> tuple[int | None, int | None]:
    n_repeats = getattr(row, "n_repeats", None)
    n_folds = getattr(row, "n_folds", None)
    if n_repeats is None or n_folds is None:
        return None, None
    return int(n_repeats), int(n_folds)


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


def _split_cache_identity(base_payload: dict[str, Any], problem_type: str | None) -> str:
    return _stable_feature_hash({**base_payload, "problem_type": problem_type})


def _encode_failed_sets(failed_sets: dict[str, str]) -> str:
    return json.dumps(failed_sets, sort_keys=True, separators=(",", ":"))


def _decode_failed_sets(value) -> dict[str, str]:
    if value is None or (isinstance(value, float) and pd.isna(value)) or value == "":
        return {}
    try:
        decoded = json.loads(value)
    except (TypeError, ValueError):
        return {}
    if not isinstance(decoded, dict):
        return {}
    return {str(k): str(v) for k, v in decoded.items()}


def _read_cached_split(split_path: Path, cache_identity: str) -> tuple[dict[str, float], dict[str, str]] | None:
    cached = pd.read_parquet(split_path)
    cached_hash = None
    if SPLIT_CACHE_HASH_COLUMN in cached.columns:
        cached_hash = cached[SPLIT_CACHE_HASH_COLUMN].iat[0]
    elif "_feature_set_hash" in cached.columns:
        cached_hash = cached["_feature_set_hash"].iat[0]
    if cached_hash != cache_identity:
        return None
    cached_failed_sets: dict[str, str] = {}
    if SPLIT_CACHE_FAILED_SETS_COLUMN in cached.columns:
        cached_failed_sets = _decode_failed_sets(cached[SPLIT_CACHE_FAILED_SETS_COLUMN].iat[0])
    drop_columns = [column for column in _SPLIT_CACHE_INTERNAL_COLUMNS if column in cached.columns]
    row = cached.drop(columns=drop_columns).iloc[0].to_dict()
    return row, cached_failed_sets


def _write_cached_split(
    split_path: Path,
    split_row: dict[str, Any],
    cache_identity: str,
    failed_sets: dict[str, str],
) -> None:
    split_path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(
        [
            {
                **split_row,
                SPLIT_CACHE_HASH_COLUMN: cache_identity,
                SPLIT_CACHE_FAILED_SETS_COLUMN: _encode_failed_sets(failed_sets),
            }
        ]
    )
    frame.to_parquet(split_path, index=False)


def _pymfe_raw_feature_from_column(column: str, pymfe_summary: tuple[str, ...]) -> str | None:
    if not column.startswith("pymfe__"):
        return None
    name = column.removeprefix("pymfe__")
    for summary in sorted(pymfe_summary, key=len, reverse=True):
        suffix = f".{summary}"
        if name.endswith(suffix) and len(name) > len(suffix):
            return name[: -len(suffix)]
    return name


def _has_pymfe_summary_suffix(column: str, pymfe_summary: tuple[str, ...]) -> bool:
    if not column.startswith("pymfe__"):
        return False
    name = column.removeprefix("pymfe__")
    return any(name.endswith(f".{summary}") and len(name) > len(summary) + 1 for summary in pymfe_summary)


def _pymfe_repair_targets(
    cached_row: dict[str, Any],
    *,
    raw_features: tuple[str, ...],
    pymfe_summary: tuple[str, ...],
) -> dict[str, tuple[str, ...]]:
    """Map missing Pymfe output columns in a cached row back to raw features."""
    columns_by_raw_feature: dict[str, list[str]] = {}
    for column in cached_row:
        raw_feature = _pymfe_raw_feature_from_column(column, pymfe_summary)
        if raw_feature is not None:
            columns_by_raw_feature.setdefault(raw_feature, []).append(column)

    targets: dict[str, tuple[str, ...]] = {}
    for raw_feature in raw_features:
        columns = columns_by_raw_feature.get(raw_feature, [])
        if not columns:
            targets[raw_feature] = (f"pymfe__{raw_feature}",)
            continue

        summary_columns = [column for column in columns if _has_pymfe_summary_suffix(column, pymfe_summary)]
        if summary_columns:
            expected_columns = tuple(f"pymfe__{raw_feature}.{summary}" for summary in pymfe_summary)
            missing_columns = tuple(column for column in expected_columns if column not in cached_row)
            if missing_columns:
                targets[raw_feature] = missing_columns
            continue

    return targets


def _configured_pymfe_raw_features(
    feature_sets: tuple[str, ...],
    pymfe_groups: tuple[str, ...],
    *,
    problem_type: str | None,
) -> tuple[str, ...]:
    if "pymfe" not in feature_sets:
        return ()
    return enumerate_pymfe_raw_features(pymfe_groups, problem_type=problem_type)


def _repair_cached_pymfe_split(
    task,
    cached_row: dict[str, Any],
    cached_failed_sets: dict[str, str],
    *,
    dataset: str,
    repeat: int,
    fold: int,
    problem_type: str | None,
    pymfe_groups: tuple[str, ...],
    pymfe_summary: tuple[str, ...],
    pymfe_per_feature_timeout_s: float | None,
    missing_raw_features: tuple[str, ...],
    trace: bool,
) -> tuple[dict[str, Any], dict[str, str], int]:
    repaired_row, repair_failed_sets = extract_split_metafeatures(
        task,
        dataset,
        repeat,
        fold,
        problem_type=problem_type,
        feature_sets=("pymfe",),
        pymfe_groups=pymfe_groups,
        pymfe_summary=pymfe_summary,
        pymfe_per_feature_timeout_s=pymfe_per_feature_timeout_s,
        pymfe_raw_features=missing_raw_features,
        trace=trace,
    )

    merged_row = dict(cached_row)
    for column, value in repaired_row.items():
        if column.startswith("pymfe__"):
            merged_row[column] = value

    merged_failed_sets = dict(cached_failed_sets)
    if repair_failed_sets:
        merged_failed_sets.update(repair_failed_sets)
    else:
        remaining_targets = _pymfe_repair_targets(
            merged_row,
            raw_features=missing_raw_features,
            pymfe_summary=pymfe_summary,
        )
        if not remaining_targets:
            merged_failed_sets.pop("pymfe", None)

    remaining_targets = _pymfe_repair_targets(
        merged_row,
        raw_features=missing_raw_features,
        pymfe_summary=pymfe_summary,
    )
    unrepaired_outputs = sum(len(columns) for columns in remaining_targets.values())
    return merged_row, merged_failed_sets, unrepaired_outputs


def extract_split_metafeatures(
    task,
    dataset: str,
    repeat: int,
    fold: int,
    *,
    problem_type: str | None = None,
    feature_sets: tuple[str, ...] = ("basic", "irregularity"),
    pymfe_groups: tuple[str, ...] = ("general", "statistical", "info-theory"),
    pymfe_summary: tuple[str, ...] = ("mean", "sd"),
    pymfe_per_feature_timeout_s: float | None = None,
    pymfe_raw_features: tuple[str, ...] | None = None,
    trace: bool = False,
) -> tuple[dict[str, float], dict[str, str]]:
    """Compute all configured meta-features for one train split.

    Returns (features, failed_sets). Per-feature-set failures are captured in
    `failed_sets` so the caller can decide how to handle them globally.
    """
    X_train, y_train, _, _ = task.get_train_test_split(fold=fold, repeat=repeat)
    features: dict[str, float] = {
        "dataset": dataset,
        "repeat": repeat,
        "fold": fold,
    }
    computed, failed_sets = extract_requested_metafeatures(
        X_train,
        y_train,
        problem_type=problem_type,
        feature_sets=feature_sets,
        pymfe_groups=pymfe_groups,
        pymfe_summary=pymfe_summary,
        pymfe_per_feature_timeout_s=pymfe_per_feature_timeout_s,
        pymfe_raw_features=pymfe_raw_features,
        trace=trace,
        trace_label=_split_trace_label(dataset, repeat, fold),
    )
    features.update(computed)
    return features, failed_sets


# Per-worker-process single-slot task cache. OpenMLTaskWrapper pins the
# full dataset (X, y) in memory, so we must never accumulate wrappers
# across datasets — an unbounded dict OOMed the full run (job 222410,
# MaxRSS=896G). A 1-slot cache bounds each worker's footprint to one
# dataset; consecutive splits of the same dataset still hit the cache.
_CACHED_TASK_ID: int | None = None
_CACHED_TASK: Any = None


def _get_cached_task(task_id: int):
    global _CACHED_TASK_ID, _CACHED_TASK
    if _CACHED_TASK_ID == task_id and _CACHED_TASK is not None:
        return _CACHED_TASK
    # Evict previous wrapper before loading the next one so peak RSS
    # stays bounded by one dataset, not two.
    _CACHED_TASK_ID = None
    _CACHED_TASK = None
    from tabarena.benchmark.task.openml import OpenMLTaskWrapper

    task = OpenMLTaskWrapper.from_task_id(task_id)
    _, _, n_samples = task.get_split_dimensions()
    if n_samples != 1:
        raise ValueError("Expected exactly one sample per (repeat, fold) split.")
    _CACHED_TASK_ID = task_id
    _CACHED_TASK = task
    return task


def _process_one_split(
    dataset: str,
    task_id: int,
    repeat: int,
    fold: int,
    problem_type: str | None,
    feature_sets: tuple[str, ...],
    pymfe_groups: tuple[str, ...],
    pymfe_summary: tuple[str, ...],
    pymfe_per_feature_timeout_s: float | None,
    cache_root: str,
    cache_identity: str,
    use_cache: bool,
    retry_failed_pymfe: bool,
    trace: bool,
) -> tuple[str, int, int, dict[str, Any] | None, bool, bool, bool, int, str | None, dict[str, str]]:
    """Handle one (dataset, repeat, fold).

    Returns (dataset, repeat, fold, row, was_cached, was_computed, was_repaired,
    unrepaired_pymfe_outputs, error, failed_sets).
    `error` is only set for unrecoverable failures (e.g. task load). Per-feature-set
    extraction errors are reported via `failed_sets` and do not invalidate the split.
    """
    try:
        cache_path = Path(cache_root)
        split_path = _split_cache_path(cache_path, dataset, repeat, fold)
        if use_cache and split_path.exists():
            try:
                cached = _read_cached_split(split_path, cache_identity)
            except Exception as exc:
                logger.warning(
                    "Meta-features %s r%d f%d: split cache `%s` is unreadable (%s: %s); recomputing split",
                    dataset,
                    repeat,
                    fold,
                    split_path,
                    type(exc).__name__,
                    exc,
                )
                cached = None
            if cached is not None:
                cached_row, cached_failed_sets = cached
                pymfe_raw_features_to_repair: tuple[str, ...] = ()
                if retry_failed_pymfe:
                    pymfe_raw_features = _configured_pymfe_raw_features(
                        feature_sets,
                        pymfe_groups,
                        problem_type=problem_type,
                    )
                    if "pymfe" in cached_failed_sets:
                        pymfe_raw_features_to_repair = pymfe_raw_features
                    else:
                        pymfe_raw_features_to_repair = tuple(
                            _pymfe_repair_targets(
                                cached_row,
                                raw_features=pymfe_raw_features,
                                pymfe_summary=pymfe_summary,
                            )
                        )
                if pymfe_raw_features_to_repair:
                    task = _get_cached_task(task_id)
                    repaired_row, repaired_failed_sets, unrepaired_outputs = _repair_cached_pymfe_split(
                        task,
                        cached_row,
                        cached_failed_sets,
                        dataset=dataset,
                        repeat=repeat,
                        fold=fold,
                        problem_type=problem_type,
                        pymfe_groups=pymfe_groups,
                        pymfe_summary=pymfe_summary,
                        pymfe_per_feature_timeout_s=pymfe_per_feature_timeout_s,
                        missing_raw_features=pymfe_raw_features_to_repair,
                        trace=trace,
                    )
                    _write_cached_split(split_path, repaired_row, cache_identity, repaired_failed_sets)
                    return (
                        dataset,
                        repeat,
                        fold,
                        repaired_row,
                        False,
                        False,
                        True,
                        unrepaired_outputs,
                        None,
                        repaired_failed_sets,
                    )
                return dataset, repeat, fold, cached_row, True, False, False, 0, None, cached_failed_sets
        task = _get_cached_task(task_id)
        split_row, failed_sets = extract_split_metafeatures(
            task,
            dataset,
            repeat,
            fold,
            problem_type=problem_type,
            feature_sets=feature_sets,
            pymfe_groups=pymfe_groups,
            pymfe_summary=pymfe_summary,
            pymfe_per_feature_timeout_s=pymfe_per_feature_timeout_s,
            trace=trace,
        )
        if use_cache:
            _write_cached_split(split_path, split_row, cache_identity, failed_sets)
        return dataset, repeat, fold, split_row, False, True, False, 0, None, failed_sets
    except Exception as exc:
        return dataset, repeat, fold, None, False, False, False, 0, f"{type(exc).__name__}: {exc}", {}


def build_metafeature_table(
    metadata: pd.DataFrame,
    *,
    datasets: list[str] | None = None,
    feature_sets: tuple[str, ...] = ("basic", "irregularity"),
    cache_dir: str | Path = ".mfa_cache",
    use_cache: bool = True,
    pymfe_groups: tuple[str, ...] = ("general", "statistical", "info-theory"),
    pymfe_summary: tuple[str, ...] = ("mean", "sd"),
    pymfe_per_feature_timeout_s: float | None = None,
    retry_failed_pymfe: bool = False,
    trace: bool = False,
    irregularity_components: tuple[str, ...] = DEFAULT_IRREGULARITY_COMPONENTS,
    cache_version: int | None = None,
    n_jobs: int = 1,
    backend: str = "process",
) -> pd.DataFrame:
    """Compute or load per-split meta-features for every requested dataset."""
    overall_start = time.perf_counter()
    cache_root = Path(cache_dir)
    cache_root.mkdir(parents=True, exist_ok=True)
    effective_use_cache = use_cache
    cache_identity_payload = {
        "feature_sets": feature_sets,
        "pymfe_groups": pymfe_groups,
        "pymfe_summary": pymfe_summary,
        "irregularity_components": irregularity_components,
        "schema_versions": _schema_versions_for_feature_sets(feature_sets),
        "cache_version": cache_version,
    }
    metadata_subset = metadata.copy()
    if datasets is not None:
        metadata_subset = metadata_subset[
            metadata_subset[[column for column in ("dataset", "name") if column in metadata_subset.columns]]
            .isin(set(datasets))
            .any(axis=1)
        ].copy()

    total_datasets = len(metadata_subset)
    resolved_n_jobs = resolve_n_jobs(n_jobs)

    logger.info(
        "Meta-features: preparing %d dataset(s) with feature_sets=%s (n_jobs=%d)",
        total_datasets,
        ",".join(feature_sets),
        resolved_n_jobs,
    )
    if trace and use_cache:
        logger.info(
            "Meta-features: trace enabled; split cache remains active, "
            "so live timing and warning diagnostics appear only on cache misses"
        )
    if trace and resolved_n_jobs > 1:
        logger.info(
            "Meta-features: trace enabled with n_jobs=%d; per-split logs may interleave. "
            "Use n_jobs=1 for ordered traces.",
            resolved_n_jobs,
        )

    # Pre-extract dataset info for both sequential and parallel paths
    dataset_tasks: list[tuple[str, int, int | None, int | None, str | None]] = []
    for row in metadata_subset.itertuples(index=False):
        dataset_tasks.append(
            (
                _metadata_dataset_name(row),
                _metadata_task_id(row),
                *_metadata_split_dimensions(row),
                _metadata_problem_type(row),
            )
        )

    if resolved_n_jobs <= 1:
        (
            rows,
            total_cached_splits,
            total_repaired_splits,
            total_recomputed_splits,
            total_unrepaired_pymfe_outputs,
            failed_feature_sets,
        ) = _build_sequential(
            dataset_tasks=dataset_tasks,
            feature_sets=feature_sets,
            pymfe_groups=pymfe_groups,
            pymfe_summary=pymfe_summary,
            pymfe_per_feature_timeout_s=pymfe_per_feature_timeout_s,
            cache_root=cache_root,
            cache_identity_payload=cache_identity_payload,
            use_cache=effective_use_cache,
            retry_failed_pymfe=retry_failed_pymfe,
            overall_start=overall_start,
            trace=trace,
        )
    else:
        (
            rows,
            total_cached_splits,
            total_repaired_splits,
            total_recomputed_splits,
            total_unrepaired_pymfe_outputs,
            failed_feature_sets,
        ) = _build_parallel(
            dataset_tasks=dataset_tasks,
            feature_sets=feature_sets,
            pymfe_groups=pymfe_groups,
            pymfe_summary=pymfe_summary,
            pymfe_per_feature_timeout_s=pymfe_per_feature_timeout_s,
            cache_root=cache_root,
            cache_identity_payload=cache_identity_payload,
            use_cache=effective_use_cache,
            retry_failed_pymfe=retry_failed_pymfe,
            overall_start=overall_start,
            n_jobs=resolved_n_jobs,
            backend=backend,
            trace=trace,
        )

    metafeature_table = pd.DataFrame(rows)
    if not metafeature_table.empty:
        metafeature_table = metafeature_table.sort_values(["dataset", "repeat", "fold"]).reset_index(drop=True)
    logger.info(
        "Meta-features: complete in %s (%d rows; %d cached split(s), %d repaired split(s), "
        "%d recomputed split(s), %d unrepaired pymfe output(s))",
        _format_elapsed(time.perf_counter() - overall_start),
        len(metafeature_table),
        total_cached_splits,
        total_repaired_splits,
        total_recomputed_splits,
        total_unrepaired_pymfe_outputs,
    )

    _log_failed_feature_sets(failed_feature_sets, total_splits=len(metafeature_table))

    # NOTE: The `irregularity` composite is deliberately NOT added here.
    # The paper defines it as a z-score across *datasets*, not across the
    # per-(dataset,repeat,fold) rows this stage emits. Doing it here would
    # let datasets with more splits dominate the normalization. The
    # composite is attached at the analysis-table level in
    # `mfa.aggregation.build_analysis_table`, where rows are one per dataset
    # (or broadcast back for `unit=fold`).
    return metafeature_table


def _log_failed_feature_sets(
    failed_feature_sets: dict[str, dict[tuple[str, int, int], str]],
    *,
    total_splits: int,
) -> None:
    """Emit one WARNING per feature set that failed on any split.

    No columns are dropped: splits that failed a feature set simply lack
    its columns in their per-split row, so the aggregated table carries
    NaN for those (split, feature) cells while preserving values from the
    splits that did succeed.
    """
    if not failed_feature_sets:
        return
    for set_name, failures in failed_feature_sets.items():
        example_key, example_err = next(iter(failures.items()))
        dataset, repeat, fold = example_key
        affected_datasets = sorted({d for d, _, _ in failures})
        logger.warning(
            "Meta-features: feature set `%s` failed on %d/%d split(s) across %d dataset(s) "
            "(e.g. %s r%d f%d: %s); leaving NaN in affected rows — other splits keep their values.",
            set_name,
            len(failures),
            total_splits,
            len(affected_datasets),
            dataset,
            repeat,
            fold,
            example_err,
        )


def _build_sequential(
    *,
    dataset_tasks: list[tuple[str, int, int | None, int | None, str | None]],
    feature_sets: tuple[str, ...],
    pymfe_groups: tuple[str, ...],
    pymfe_summary: tuple[str, ...],
    pymfe_per_feature_timeout_s: float | None,
    cache_root: Path,
    cache_identity_payload: dict[str, Any],
    use_cache: bool,
    retry_failed_pymfe: bool,
    overall_start: float,
    trace: bool,
) -> tuple[list[dict[str, Any]], int, int, int, int, dict[str, dict[tuple[str, int, int], str]]]:
    """Sequential per-dataset processing (original behavior)."""
    rows: list[dict[str, Any]] = []
    total_cached_splits = 0
    total_repaired_splits = 0
    total_recomputed_splits = 0
    total_unrepaired_pymfe_outputs = 0
    total_datasets = len(dataset_tasks)
    failed_feature_sets: dict[str, dict[tuple[str, int, int], str]] = {}
    for dataset_index, (dataset, task_id, n_repeats, n_folds, problem_type) in enumerate(dataset_tasks, start=1):
        split_cache_identity = _split_cache_identity(cache_identity_payload, problem_type)
        task = None
        if n_repeats is None or n_folds is None:
            from tabarena.benchmark.task.openml import OpenMLTaskWrapper

            task = OpenMLTaskWrapper.from_task_id(task_id)
            n_repeats, n_folds, n_samples = task.get_split_dimensions()
            if n_samples != 1:
                raise ValueError("Expected exactly one sample per (repeat, fold) split.")
        n_splits = n_repeats * n_folds
        dataset_cached_splits = 0
        dataset_repaired_splits = 0
        dataset_recomputed_splits = 0
        dataset_unrepaired_pymfe_outputs = 0
        logger.info(
            "Meta-features [%d/%d] %s: starting (%d split(s); elapsed %s)",
            dataset_index,
            total_datasets,
            dataset,
            n_splits,
            _format_elapsed(time.perf_counter() - overall_start),
        )
        dataset_start = time.perf_counter()

        def _ensure_task_loaded(task_id=task_id):
            nonlocal task
            if task is None:
                from tabarena.benchmark.task.openml import OpenMLTaskWrapper

                task = OpenMLTaskWrapper.from_task_id(task_id)
                _, _, n_samples = task.get_split_dimensions()
                if n_samples != 1:
                    raise ValueError("Expected exactly one sample per (repeat, fold) split.")
            return task

        for repeat in range(n_repeats):
            for fold in range(n_folds):
                split_path = _split_cache_path(cache_root, dataset, repeat, fold)
                if use_cache and split_path.exists():
                    try:
                        cached = _read_cached_split(split_path, split_cache_identity)
                    except Exception as exc:
                        logger.warning(
                            "Meta-features %s r%d f%d: split cache `%s` is unreadable (%s: %s); recomputing split",
                            dataset,
                            repeat,
                            fold,
                            split_path,
                            type(exc).__name__,
                            exc,
                        )
                        cached = None
                    if cached is not None:
                        cached_row, cached_failed_sets = cached
                        pymfe_raw_features_to_repair: tuple[str, ...] = ()
                        if retry_failed_pymfe:
                            pymfe_raw_features = _configured_pymfe_raw_features(
                                feature_sets,
                                pymfe_groups,
                                problem_type=problem_type,
                            )
                            if "pymfe" in cached_failed_sets:
                                pymfe_raw_features_to_repair = pymfe_raw_features
                            else:
                                pymfe_raw_features_to_repair = tuple(
                                    _pymfe_repair_targets(
                                        cached_row,
                                        raw_features=pymfe_raw_features,
                                        pymfe_summary=pymfe_summary,
                                    )
                                )
                        if pymfe_raw_features_to_repair:
                            repaired_row, cached_failed_sets, unrepaired_outputs = _repair_cached_pymfe_split(
                                _ensure_task_loaded(),
                                cached_row,
                                cached_failed_sets,
                                dataset=dataset,
                                repeat=repeat,
                                fold=fold,
                                problem_type=problem_type,
                                pymfe_groups=pymfe_groups,
                                pymfe_summary=pymfe_summary,
                                pymfe_per_feature_timeout_s=pymfe_per_feature_timeout_s,
                                missing_raw_features=pymfe_raw_features_to_repair,
                                trace=trace,
                            )
                            _write_cached_split(split_path, repaired_row, split_cache_identity, cached_failed_sets)
                            rows.append(repaired_row)
                            dataset_repaired_splits += 1
                            dataset_unrepaired_pymfe_outputs += unrepaired_outputs
                        else:
                            rows.append(cached_row)
                            dataset_cached_splits += 1
                        for set_name, err in cached_failed_sets.items():
                            failed_feature_sets.setdefault(set_name, {})[(dataset, repeat, fold)] = err
                        continue

                split_row, split_failed_sets = extract_split_metafeatures(
                    _ensure_task_loaded(),
                    dataset,
                    repeat,
                    fold,
                    problem_type=problem_type,
                    feature_sets=feature_sets,
                    pymfe_groups=pymfe_groups,
                    pymfe_summary=pymfe_summary,
                    pymfe_per_feature_timeout_s=pymfe_per_feature_timeout_s,
                    trace=trace,
                )
                for set_name, err in split_failed_sets.items():
                    failed_feature_sets.setdefault(set_name, {})[(dataset, repeat, fold)] = err
                    logger.warning(
                        "Meta-features %s r%d f%d: feature set `%s` failed — %s",
                        dataset,
                        repeat,
                        fold,
                        set_name,
                        err,
                    )
                rows.append(split_row)
                dataset_recomputed_splits += 1
                if use_cache:
                    _write_cached_split(split_path, split_row, split_cache_identity, split_failed_sets)
        total_cached_splits += dataset_cached_splits
        total_repaired_splits += dataset_repaired_splits
        total_recomputed_splits += dataset_recomputed_splits
        total_unrepaired_pymfe_outputs += dataset_unrepaired_pymfe_outputs
        logger.info(
            "Meta-features [%d/%d] %s: done in %s (%d cached, %d repaired, %d recomputed, "
            "%d unrepaired pymfe output(s); total elapsed %s)",
            dataset_index,
            total_datasets,
            dataset,
            _format_elapsed(time.perf_counter() - dataset_start),
            dataset_cached_splits,
            dataset_repaired_splits,
            dataset_recomputed_splits,
            dataset_unrepaired_pymfe_outputs,
            _format_elapsed(time.perf_counter() - overall_start),
        )
    return (
        rows,
        total_cached_splits,
        total_repaired_splits,
        total_recomputed_splits,
        total_unrepaired_pymfe_outputs,
        failed_feature_sets,
    )


def _build_parallel(
    *,
    dataset_tasks: list[tuple[str, int, int | None, int | None, str | None]],
    feature_sets: tuple[str, ...],
    pymfe_groups: tuple[str, ...],
    pymfe_summary: tuple[str, ...],
    pymfe_per_feature_timeout_s: float | None,
    cache_root: Path,
    cache_identity_payload: dict[str, Any],
    use_cache: bool,
    retry_failed_pymfe: bool,
    overall_start: float,
    n_jobs: int,
    backend: str,
    trace: bool,
) -> tuple[list[dict[str, Any]], int, int, int, int, dict[str, dict[tuple[str, int, int], str]]]:
    """Parallel per-split processing: one future per (dataset, repeat, fold).

    Finer granularity than per-dataset keeps the process pool saturated even
    when dataset runtimes are skewed (e.g. a huge dataset dwarfs a tiny one).
    """
    cache_root_str = str(cache_root)

    # Enumerate every split up front. For any dataset missing dims in metadata,
    # load the task once in the main process to discover them (rare in practice).
    missing_dims = [
        (dataset, task_id)
        for dataset, task_id, n_repeats, n_folds, _problem_type in dataset_tasks
        if n_repeats is None or n_folds is None
    ]
    resolved_dims: dict[str, tuple[int, int]] = {}
    if missing_dims:
        from tabarena.benchmark.task.openml import OpenMLTaskWrapper

        for dataset, task_id in missing_dims:
            task = OpenMLTaskWrapper.from_task_id(task_id)
            n_rep, n_fld, n_samples = task.get_split_dimensions()
            if n_samples != 1:
                raise ValueError("Expected exactly one sample per (repeat, fold) split.")
            resolved_dims[dataset] = (n_rep, n_fld)

    split_units: list[tuple[str, int, int, int, str | None, str]] = []
    for dataset, task_id, n_repeats, n_folds, problem_type in dataset_tasks:
        if n_repeats is None or n_folds is None:
            n_repeats, n_folds = resolved_dims[dataset]
        split_cache_identity = _split_cache_identity(cache_identity_payload, problem_type)
        for repeat in range(n_repeats):
            for fold in range(n_folds):
                split_units.append((dataset, task_id, repeat, fold, problem_type, split_cache_identity))

    total_splits = len(split_units)
    total_cached = 0
    total_repaired = 0
    total_recomputed = 0
    total_unrepaired_pymfe_outputs = 0
    failed: list[tuple[str, int, int, str]] = []
    failed_feature_sets: dict[str, dict[tuple[str, int, int], str]] = {}
    rows_by_dataset: dict[str, list[dict[str, Any]]] = {dataset: [] for dataset, *_ in dataset_tasks}

    try:
        from tqdm import tqdm

        has_tqdm = True
    except ImportError:
        has_tqdm = False

    logger.info(
        "Meta-features: submitting %d split(s) across %d dataset(s) to %d worker(s)",
        total_splits,
        len(dataset_tasks),
        n_jobs,
    )

    completed_keys: set[tuple[str, int, int]] = set()
    progress = tqdm(total=total_splits, desc="Meta-features", unit="split") if has_tqdm else None

    def _record(result_tuple: tuple) -> None:
        nonlocal total_cached, total_repaired, total_recomputed, total_unrepaired_pymfe_outputs
        (
            dataset_r,
            repeat_r,
            fold_r,
            row,
            was_cached,
            was_computed,
            was_repaired,
            unrepaired_outputs,
            error,
            split_failed_sets,
        ) = result_tuple
        completed_keys.add((dataset_r, repeat_r, fold_r))
        if progress is not None:
            progress.update(1)
        if error is not None:
            failed.append((dataset_r, repeat_r, fold_r, error))
            logger.warning("Meta-features %s r%d f%d: FAILED — %s", dataset_r, repeat_r, fold_r, error)
            return
        for set_name, err in split_failed_sets.items():
            failed_feature_sets.setdefault(set_name, {})[(dataset_r, repeat_r, fold_r)] = err
            logger.warning(
                "Meta-features %s r%d f%d: feature set `%s` failed — %s",
                dataset_r,
                repeat_r,
                fold_r,
                set_name,
                err,
            )
        rows_by_dataset.setdefault(dataset_r, []).append(row)
        if was_cached:
            total_cached += 1
        if was_computed:
            total_recomputed += 1
        if was_repaired:
            total_repaired += 1
            total_unrepaired_pymfe_outputs += unrepaired_outputs

    # A BrokenProcessPool (commonly an OOM-killed worker) poisons every
    # pending future in the executor, so a single bad split would
    # otherwise abort the whole run despite per-split parquet caches
    # already persisting completed work. We harvest what landed,
    # rebuild the pool once for the splits that did not return, and
    # fall back to sequential in-process execution if the pool breaks
    # again — the cache means retries are idempotent.
    pending = list(split_units)
    max_pool_break_retries = 1
    pool_break_retries = 0
    try:
        while pending:
            executor = get_executor(backend, max_workers=n_jobs)
            broken = False
            try:
                futures = [
                    executor.submit(
                        _process_one_split,
                        dataset,
                        task_id,
                        repeat,
                        fold,
                        problem_type,
                        feature_sets,
                        pymfe_groups,
                        pymfe_summary,
                        pymfe_per_feature_timeout_s,
                        cache_root_str,
                        cache_identity,
                        use_cache,
                        retry_failed_pymfe,
                        trace,
                    )
                    for dataset, task_id, repeat, fold, problem_type, cache_identity in pending
                ]
                for future in as_completed(futures):
                    try:
                        _record(future.result())
                    except BrokenProcessPool as exc:
                        broken = True
                        logger.warning(
                            "Meta-features: process pool broken (%s); %d/%d split(s) harvested, retrying remainder.",
                            exc,
                            len(completed_keys),
                            total_splits,
                        )
                        break
            finally:
                executor.shutdown(wait=not broken, cancel_futures=broken)

            if not broken:
                pending = []
                break

            pending = [unit for unit in pending if (unit[0], unit[2], unit[3]) not in completed_keys]
            if not pending:
                break

            pool_break_retries += 1
            if pool_break_retries > max_pool_break_retries:
                logger.warning(
                    "Meta-features: pool broke %d time(s); finishing %d remaining split(s) sequentially in-process.",
                    pool_break_retries,
                    len(pending),
                )
                for dataset, task_id, repeat, fold, problem_type, cache_identity in pending:
                    _record(
                        _process_one_split(
                            dataset,
                            task_id,
                            repeat,
                            fold,
                            problem_type,
                            feature_sets,
                            pymfe_groups,
                            pymfe_summary,
                            pymfe_per_feature_timeout_s,
                            cache_root_str,
                            cache_identity,
                            use_cache,
                            retry_failed_pymfe,
                            trace,
                        )
                    )
                pending = []
            else:
                logger.warning(
                    "Meta-features: retrying %d remaining split(s) in a fresh pool (attempt %d/%d).",
                    len(pending),
                    pool_break_retries,
                    max_pool_break_retries,
                )
    finally:
        if progress is not None:
            progress.close()

    if failed:
        summary = "; ".join(f"{d} r{r} f{f}: {err}" for d, r, f, err in failed)
        raise RuntimeError(f"Meta-feature extraction failed for {len(failed)}/{total_splits} split(s): {summary}")

    rows: list[dict[str, Any]] = []
    for dataset_rows in rows_by_dataset.values():
        rows.extend(dataset_rows)
    return (
        rows,
        total_cached,
        total_repaired,
        total_recomputed,
        total_unrepaired_pymfe_outputs,
        failed_feature_sets,
    )


__all__ = [
    "build_metafeature_table",
    "extract_split_metafeatures",
]
