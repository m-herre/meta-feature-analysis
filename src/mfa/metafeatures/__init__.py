from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

import pandas as pd

from .._logging import get_logger
from ..cache import metafeature_split_cache_dir
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

    logger.info(
        "Meta-features: preparing %d dataset(s) with feature_sets=%s",
        len(metadata_subset),
        ",".join(feature_sets),
    )
    rows: list[dict[str, float]] = []
    total_cached_splits = 0
    total_computed_splits = 0
    total_datasets = len(metadata_subset)
    for dataset_index, row in enumerate(metadata_subset.itertuples(index=False), start=1):
        dataset = _metadata_dataset_name(row)
        n_repeats, n_folds = _metadata_split_dimensions(row)
        task = None
        if n_repeats is None or n_folds is None:
            from tabarena.benchmark.task.openml import OpenMLTaskWrapper

            task = OpenMLTaskWrapper.from_task_id(_metadata_task_id(row))
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

                    task = OpenMLTaskWrapper.from_task_id(_metadata_task_id(row))
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


__all__ = [
    "build_metafeature_table",
    "extract_split_metafeatures",
]
