from __future__ import annotations

import hashlib
import json
import time
from concurrent.futures import as_completed
from pathlib import Path
from typing import Any

import pandas as pd

from .._logging import get_logger
from ..cache import metafeature_split_cache_dir
from ..parallel import get_executor, resolve_n_jobs
from .irregularity import DEFAULT_IRREGULARITY_COMPONENTS, add_irregularity_proxy
from .registry import extract_requested_metafeatures

logger = get_logger(__name__)


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


SPLIT_CACHE_HASH_COLUMN = "_cache_identity_hash"


def _split_cache_path(cache_dir: Path, dataset: str, repeat: int, fold: int) -> Path:
    return metafeature_split_cache_dir(cache_dir) / f"{dataset}__r{repeat}__f{fold}.parquet"


def _metadata_split_dimensions(row) -> tuple[int | None, int | None]:
    n_repeats = getattr(row, "n_repeats", None)
    n_folds = getattr(row, "n_folds", None)
    if n_repeats is None or n_folds is None:
        return None, None
    return int(n_repeats), int(n_folds)


def _read_cached_split(split_path: Path, cache_identity: str) -> dict[str, float] | None:
    cached = pd.read_parquet(split_path)
    cached_hash = None
    if SPLIT_CACHE_HASH_COLUMN in cached.columns:
        cached_hash = cached[SPLIT_CACHE_HASH_COLUMN].iat[0]
    elif "_feature_set_hash" in cached.columns:
        cached_hash = cached["_feature_set_hash"].iat[0]
    if cached_hash != cache_identity:
        return None
    drop_columns = [column for column in (SPLIT_CACHE_HASH_COLUMN, "_feature_set_hash") if column in cached.columns]
    return cached.drop(columns=drop_columns).iloc[0].to_dict()


def extract_split_metafeatures(
    task,
    dataset: str,
    repeat: int,
    fold: int,
    *,
    feature_sets: tuple[str, ...] = ("basic", "irregularity"),
    pymfe_groups: tuple[str, ...] = ("general", "statistical", "info-theory"),
    pymfe_summary: tuple[str, ...] = ("mean", "sd"),
) -> dict[str, float]:
    """Compute all configured meta-features for one train split."""
    X_train, y_train, _, _ = task.get_train_test_split(fold=fold, repeat=repeat)
    features = {
        "dataset": dataset,
        "repeat": repeat,
        "fold": fold,
    }
    features.update(
        extract_requested_metafeatures(
            X_train,
            y_train,
            feature_sets=feature_sets,
            pymfe_groups=pymfe_groups,
            pymfe_summary=pymfe_summary,
        )
    )
    return features


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
    feature_sets: tuple[str, ...],
    pymfe_groups: tuple[str, ...],
    pymfe_summary: tuple[str, ...],
    cache_root: str,
    cache_identity: str,
    use_cache: bool,
) -> tuple[str, int, int, dict[str, float] | None, bool, bool, str | None]:
    """Handle one (dataset, repeat, fold).

    Returns (dataset, repeat, fold, row, was_cached, was_computed, error).
    """
    try:
        cache_path = Path(cache_root)
        split_path = _split_cache_path(cache_path, dataset, repeat, fold)
        if use_cache and split_path.exists():
            cached_row = _read_cached_split(split_path, cache_identity)
            if cached_row is not None:
                return dataset, repeat, fold, cached_row, True, False, None
        task = _get_cached_task(task_id)
        split_row = extract_split_metafeatures(
            task,
            dataset,
            repeat,
            fold,
            feature_sets=feature_sets,
            pymfe_groups=pymfe_groups,
            pymfe_summary=pymfe_summary,
        )
        if use_cache:
            split_path.parent.mkdir(parents=True, exist_ok=True)
            frame = pd.DataFrame([{**split_row, SPLIT_CACHE_HASH_COLUMN: cache_identity}])
            frame.to_parquet(split_path, index=False)
        return dataset, repeat, fold, split_row, False, True, None
    except Exception as exc:
        return dataset, repeat, fold, None, False, False, f"{type(exc).__name__}: {exc}"


def build_metafeature_table(
    metadata: pd.DataFrame,
    *,
    datasets: list[str] | None = None,
    feature_sets: tuple[str, ...] = ("basic", "irregularity"),
    cache_dir: str | Path = ".mfa_cache",
    use_cache: bool = True,
    pymfe_groups: tuple[str, ...] = ("general", "statistical", "info-theory"),
    pymfe_summary: tuple[str, ...] = ("mean", "sd"),
    irregularity_components: tuple[str, ...] = DEFAULT_IRREGULARITY_COMPONENTS,
    cache_version: int | None = None,
    n_jobs: int = 1,
    backend: str = "process",
) -> pd.DataFrame:
    """Compute or load per-split meta-features for every requested dataset."""
    overall_start = time.perf_counter()
    cache_root = Path(cache_dir)
    cache_root.mkdir(parents=True, exist_ok=True)
    cache_identity = _stable_feature_hash(
        {
            "feature_sets": feature_sets,
            "pymfe_groups": pymfe_groups,
            "pymfe_summary": pymfe_summary,
            "irregularity_components": irregularity_components,
            "cache_version": cache_version,
        }
    )
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

    # Pre-extract dataset info for both sequential and parallel paths
    dataset_tasks: list[tuple[str, int, int | None, int | None]] = []
    for row in metadata_subset.itertuples(index=False):
        dataset_tasks.append(
            (
                _metadata_dataset_name(row),
                _metadata_task_id(row),
                *_metadata_split_dimensions(row),
            )
        )

    if resolved_n_jobs <= 1:
        rows, total_cached_splits, total_computed_splits = _build_sequential(
            dataset_tasks=dataset_tasks,
            feature_sets=feature_sets,
            pymfe_groups=pymfe_groups,
            pymfe_summary=pymfe_summary,
            cache_root=cache_root,
            cache_identity=cache_identity,
            use_cache=use_cache,
            overall_start=overall_start,
        )
    else:
        rows, total_cached_splits, total_computed_splits = _build_parallel(
            dataset_tasks=dataset_tasks,
            feature_sets=feature_sets,
            pymfe_groups=pymfe_groups,
            pymfe_summary=pymfe_summary,
            cache_root=cache_root,
            cache_identity=cache_identity,
            use_cache=use_cache,
            overall_start=overall_start,
            n_jobs=resolved_n_jobs,
            backend=backend,
        )

    metafeature_table = pd.DataFrame(rows)
    if not metafeature_table.empty:
        metafeature_table = metafeature_table.sort_values(["dataset", "repeat", "fold"]).reset_index(drop=True)
    logger.info(
        "Meta-features: complete in %s (%d rows; %d cached split(s), %d computed split(s))",
        _format_elapsed(time.perf_counter() - overall_start),
        len(metafeature_table),
        total_cached_splits,
        total_computed_splits,
    )
    if "irregularity" not in metafeature_table.columns and "irregularity" not in feature_sets:
        return metafeature_table
    return add_irregularity_proxy(metafeature_table, components=irregularity_components)


def _build_sequential(
    *,
    dataset_tasks: list[tuple[str, int, int | None, int | None]],
    feature_sets: tuple[str, ...],
    pymfe_groups: tuple[str, ...],
    pymfe_summary: tuple[str, ...],
    cache_root: Path,
    cache_identity: str,
    use_cache: bool,
    overall_start: float,
) -> tuple[list[dict[str, float]], int, int]:
    """Sequential per-dataset processing (original behavior)."""
    rows: list[dict[str, float]] = []
    total_cached_splits = 0
    total_computed_splits = 0
    total_datasets = len(dataset_tasks)
    for dataset_index, (dataset, task_id, n_repeats, n_folds) in enumerate(dataset_tasks, start=1):
        task = None
        if n_repeats is None or n_folds is None:
            from tabarena.benchmark.task.openml import OpenMLTaskWrapper

            task = OpenMLTaskWrapper.from_task_id(task_id)
            n_repeats, n_folds, n_samples = task.get_split_dimensions()
            if n_samples != 1:
                raise ValueError("Expected exactly one sample per (repeat, fold) split.")
        n_splits = n_repeats * n_folds
        dataset_cached_splits = 0
        dataset_computed_splits = 0
        logger.info(
            "Meta-features [%d/%d] %s: starting (%d split(s); elapsed %s)",
            dataset_index,
            total_datasets,
            dataset,
            n_splits,
            _format_elapsed(time.perf_counter() - overall_start),
        )
        dataset_start = time.perf_counter()
        for repeat in range(n_repeats):
            for fold in range(n_folds):
                split_path = _split_cache_path(cache_root, dataset, repeat, fold)
                if use_cache and split_path.exists():
                    cached_row = _read_cached_split(split_path, cache_identity)
                    if cached_row is not None:
                        rows.append(cached_row)
                        dataset_cached_splits += 1
                        continue

                if task is None:
                    from tabarena.benchmark.task.openml import OpenMLTaskWrapper

                    task = OpenMLTaskWrapper.from_task_id(task_id)
                    _, _, n_samples = task.get_split_dimensions()
                    if n_samples != 1:
                        raise ValueError("Expected exactly one sample per (repeat, fold) split.")
                split_row = extract_split_metafeatures(
                    task,
                    dataset,
                    repeat,
                    fold,
                    feature_sets=feature_sets,
                    pymfe_groups=pymfe_groups,
                    pymfe_summary=pymfe_summary,
                )
                rows.append(split_row)
                dataset_computed_splits += 1
                if use_cache:
                    split_path.parent.mkdir(parents=True, exist_ok=True)
                    cached_row = pd.DataFrame([{**split_row, SPLIT_CACHE_HASH_COLUMN: cache_identity}])
                    cached_row.to_parquet(split_path, index=False)
        total_cached_splits += dataset_cached_splits
        total_computed_splits += dataset_computed_splits
        logger.info(
            "Meta-features [%d/%d] %s: done in %s (%d cached, %d computed; total elapsed %s)",
            dataset_index,
            total_datasets,
            dataset,
            _format_elapsed(time.perf_counter() - dataset_start),
            dataset_cached_splits,
            dataset_computed_splits,
            _format_elapsed(time.perf_counter() - overall_start),
        )
    return rows, total_cached_splits, total_computed_splits


def _build_parallel(
    *,
    dataset_tasks: list[tuple[str, int, int | None, int | None]],
    feature_sets: tuple[str, ...],
    pymfe_groups: tuple[str, ...],
    pymfe_summary: tuple[str, ...],
    cache_root: Path,
    cache_identity: str,
    use_cache: bool,
    overall_start: float,
    n_jobs: int,
    backend: str,
) -> tuple[list[dict[str, float]], int, int]:
    """Parallel per-split processing: one future per (dataset, repeat, fold).

    Finer granularity than per-dataset keeps the process pool saturated even
    when dataset runtimes are skewed (e.g. a huge dataset dwarfs a tiny one).
    """
    cache_root_str = str(cache_root)

    # Enumerate every split up front. For any dataset missing dims in metadata,
    # load the task once in the main process to discover them (rare in practice).
    missing_dims = [
        (dataset, task_id)
        for dataset, task_id, n_repeats, n_folds in dataset_tasks
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

    split_units: list[tuple[str, int, int, int]] = []
    for dataset, task_id, n_repeats, n_folds in dataset_tasks:
        if n_repeats is None or n_folds is None:
            n_repeats, n_folds = resolved_dims[dataset]
        for repeat in range(n_repeats):
            for fold in range(n_folds):
                split_units.append((dataset, task_id, repeat, fold))

    total_splits = len(split_units)
    total_cached = 0
    total_computed = 0
    failed: list[tuple[str, int, int, str]] = []
    rows_by_dataset: dict[str, list[dict[str, float]]] = {dataset: [] for dataset, *_ in dataset_tasks}

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

    executor = get_executor(backend, max_workers=n_jobs)
    try:
        futures = [
            executor.submit(
                _process_one_split,
                dataset,
                task_id,
                repeat,
                fold,
                feature_sets,
                pymfe_groups,
                pymfe_summary,
                cache_root_str,
                cache_identity,
                use_cache,
            )
            for dataset, task_id, repeat, fold in split_units
        ]
        completed_iter = as_completed(futures)
        if has_tqdm:
            completed_iter = tqdm(completed_iter, total=total_splits, desc="Meta-features", unit="split")

        for future in completed_iter:
            dataset, repeat, fold, row, was_cached, was_computed, error = future.result()
            if error is not None:
                failed.append((dataset, repeat, fold, error))
                logger.warning("Meta-features %s r%d f%d: FAILED — %s", dataset, repeat, fold, error)
                continue
            rows_by_dataset.setdefault(dataset, []).append(row)
            if was_cached:
                total_cached += 1
            if was_computed:
                total_computed += 1
    finally:
        executor.shutdown(wait=True)

    if failed:
        summary = "; ".join(f"{d} r{r} f{f}: {err}" for d, r, f, err in failed)
        raise RuntimeError(f"Meta-feature extraction failed for {len(failed)}/{total_splits} split(s): {summary}")

    rows: list[dict[str, float]] = []
    for dataset_rows in rows_by_dataset.values():
        rows.extend(dataset_rows)
    return rows, total_cached, total_computed


__all__ = [
    "build_metafeature_table",
    "extract_split_metafeatures",
]
