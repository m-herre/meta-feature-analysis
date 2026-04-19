from __future__ import annotations

from .basic import compute_basic_metafeatures, get_categorical_columns
from .irregularity import compute_irregularity_components
from .pymfe_features import extract_pymfe_features

VALID_FEATURE_SETS = {"basic", "irregularity", "pymfe"}


def validate_feature_sets(feature_sets: tuple[str, ...]) -> None:
    unknown = sorted(set(feature_sets) - VALID_FEATURE_SETS)
    if unknown:
        raise ValueError(f"Unknown feature sets: {unknown}")


def extract_requested_metafeatures(
    X_train,
    y_train,
    *,
    feature_sets: tuple[str, ...],
    pymfe_groups: tuple[str, ...],
    pymfe_summary: tuple[str, ...],
) -> tuple[dict[str, float], dict[str, str]]:
    """Extract all configured feature sets for one training split.

    Only `pymfe` is treated as best-effort: it calls an external library that
    can fail on unusual inputs (e.g. RecursionError on very wide data), and
    losing optional features is preferable to aborting the run. `basic` and
    `irregularity` are pure in-process numerics — if they raise, something is
    genuinely broken and the error must surface, not be downgraded to NaN.
    """
    validate_feature_sets(feature_sets)
    features: dict[str, float] = {}
    failed_sets: dict[str, str] = {}

    if "basic" in feature_sets:
        features.update(compute_basic_metafeatures(X_train))

    if "irregularity" in feature_sets:
        categorical_columns = get_categorical_columns(X_train)
        X_num = X_train.drop(columns=categorical_columns, errors="ignore")
        features.update(compute_irregularity_components(X_num))

    if "pymfe" in feature_sets:
        try:
            features.update(
                extract_pymfe_features(
                    X_train,
                    y_train,
                    groups=pymfe_groups,
                    summary=pymfe_summary,
                )
            )
        except Exception as exc:
            failed_sets["pymfe"] = f"{type(exc).__name__}: {exc}"

    return features, failed_sets
