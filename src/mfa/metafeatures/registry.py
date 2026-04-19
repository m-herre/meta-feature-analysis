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

    Each feature set is computed in isolation; if one raises, the error is
    recorded in the returned `failed_sets` mapping (set name -> error repr)
    and the remaining sets still run. The split still produces a row with
    whatever sets did succeed — the caller aggregates per-split rows into
    a table where failing splits simply carry NaN for the missing feature
    columns, preserving both the dataset and the other feature sets.
    """
    validate_feature_sets(feature_sets)
    features: dict[str, float] = {}
    failed_sets: dict[str, str] = {}

    if "basic" in feature_sets:
        try:
            features.update(compute_basic_metafeatures(X_train))
        except Exception as exc:
            failed_sets["basic"] = f"{type(exc).__name__}: {exc}"

    if "irregularity" in feature_sets:
        try:
            categorical_columns = get_categorical_columns(X_train)
            X_num = X_train.drop(columns=categorical_columns, errors="ignore")
            features.update(compute_irregularity_components(X_num))
        except Exception as exc:
            failed_sets["irregularity"] = f"{type(exc).__name__}: {exc}"

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
