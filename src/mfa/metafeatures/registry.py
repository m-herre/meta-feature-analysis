from __future__ import annotations

import time

from .._logging import get_logger
from .basic import BASIC_METAFEATURE_COLUMNS, compute_basic_metafeatures, get_categorical_columns
from .irregularity import compute_irregularity_components
from .pymfe_features import extract_pymfe_features
from .redundancy import REDUNDANCY_METAFEATURE_COLUMNS, compute_redundancy_metafeatures

VALID_FEATURE_SETS = {"basic", "irregularity", "pymfe", "redundancy"}
logger = get_logger(__name__)


def _trace_prefix(trace_label: str | None) -> str:
    if trace_label:
        return f"Meta-features {trace_label}"
    return "Meta-features"


def validate_feature_sets(feature_sets: tuple[str, ...]) -> None:
    unknown = sorted(set(feature_sets) - VALID_FEATURE_SETS)
    if unknown:
        raise ValueError(f"Unknown feature sets: {unknown}")


def _nan_placeholder_features(columns: tuple[str, ...]) -> dict[str, float]:
    return {column: float("nan") for column in columns}


def extract_requested_metafeatures(
    X_train,
    y_train,
    *,
    problem_type: str | None = None,
    feature_sets: tuple[str, ...],
    pymfe_groups: tuple[str, ...],
    pymfe_summary: tuple[str, ...],
    pymfe_per_feature_timeout_s: float | None = None,
    pymfe_raw_features: tuple[str, ...] | None = None,
    trace: bool = False,
    trace_label: str | None = None,
) -> tuple[dict[str, float], dict[str, str]]:
    """Extract all configured feature sets for one training split.

    `basic`, `redundancy`, and `pymfe` are treated as best-effort so split
    caches can preserve healthy feature sets while marking failed ones for
    later repair. `irregularity` remains strict because it feeds the composite
    analysis feature and should surface internal failures immediately.
    """
    validate_feature_sets(feature_sets)
    features: dict[str, float] = {}
    failed_sets: dict[str, str] = {}
    trace_prefix = _trace_prefix(trace_label)

    if "basic" in feature_sets:
        basic_start = time.perf_counter() if trace else None
        try:
            basic_features = compute_basic_metafeatures(X_train, y_train, problem_type=problem_type)
            features.update(basic_features)
            if trace:
                logger.info(
                    "%s: feature set `basic`: calculated %d feature(s) in %.3fs (%s)",
                    trace_prefix,
                    len(basic_features),
                    time.perf_counter() - basic_start,
                    ", ".join(sorted(basic_features)),
                )
        except Exception as exc:
            failed_sets["basic"] = f"{type(exc).__name__}: {exc}"
            features.update(_nan_placeholder_features(BASIC_METAFEATURE_COLUMNS))
            if trace:
                logger.warning(
                    "%s: feature set `basic` failed after %.3fs — %s",
                    trace_prefix,
                    time.perf_counter() - basic_start,
                    failed_sets["basic"],
                )

    if "redundancy" in feature_sets:
        redundancy_start = time.perf_counter() if trace else None
        try:
            categorical_columns = get_categorical_columns(X_train)
            X_num = X_train.drop(columns=categorical_columns, errors="ignore")
            redundancy_features = compute_redundancy_metafeatures(X_num)
            features.update(redundancy_features)
            if trace:
                logger.info(
                    "%s: feature set `redundancy`: calculated %d feature(s) in %.3fs (%s)",
                    trace_prefix,
                    len(redundancy_features),
                    time.perf_counter() - redundancy_start,
                    ", ".join(sorted(redundancy_features)),
                )
        except Exception as exc:
            failed_sets["redundancy"] = f"{type(exc).__name__}: {exc}"
            features.update(_nan_placeholder_features(REDUNDANCY_METAFEATURE_COLUMNS))
            if trace:
                logger.warning(
                    "%s: feature set `redundancy` failed after %.3fs — %s",
                    trace_prefix,
                    time.perf_counter() - redundancy_start,
                    failed_sets["redundancy"],
                )

    if "irregularity" in feature_sets:
        irregularity_start = time.perf_counter() if trace else None
        categorical_columns = get_categorical_columns(X_train)
        X_num = X_train.drop(columns=categorical_columns, errors="ignore")
        irregularity_features = compute_irregularity_components(X_num)
        features.update(irregularity_features)
        if trace:
            logger.info(
                "%s: feature set `irregularity`: calculated %d feature(s) in %.3fs (%s)",
                trace_prefix,
                len(irregularity_features),
                time.perf_counter() - irregularity_start,
                ", ".join(sorted(irregularity_features)),
            )

    if "pymfe" in feature_sets:
        pymfe_start = time.perf_counter() if trace else None
        try:
            pymfe_features = extract_pymfe_features(
                X_train,
                y_train,
                groups=pymfe_groups,
                summary=pymfe_summary,
                problem_type=problem_type,
                per_feature_timeout_s=pymfe_per_feature_timeout_s,
                raw_features=pymfe_raw_features,
                trace=trace,
                trace_label=trace_label,
            )
            features.update(pymfe_features)
            if trace:
                logger.info(
                    "%s: feature set `pymfe`: calculated %d output(s) in %.3fs",
                    trace_prefix,
                    len(pymfe_features),
                    time.perf_counter() - pymfe_start,
                )
        except Exception as exc:
            failed_sets["pymfe"] = f"{type(exc).__name__}: {exc}"
            if trace:
                logger.warning(
                    "%s: feature set `pymfe` failed after %.3fs — %s",
                    trace_prefix,
                    time.perf_counter() - pymfe_start,
                    failed_sets["pymfe"],
                )

    return features, failed_sets
