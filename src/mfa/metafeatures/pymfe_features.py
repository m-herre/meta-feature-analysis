from __future__ import annotations

import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from .._logging import get_logger
from .basic import get_categorical_columns

logger = get_logger(__name__)


def _trace_prefix(trace_label: str | None) -> str:
    if trace_label:
        return f"Meta-features {trace_label}"
    return "Meta-features"


def _format_warning_messages(records: list[warnings.WarningMessage]) -> tuple[str, ...]:
    messages: list[str] = []
    for record in records:
        origin = f"{Path(record.filename).name}:{record.lineno}"
        messages.append(f"{record.category.__name__} from {origin}: {record.message}")
    return tuple(dict.fromkeys(messages))


def _log_warning_messages(
    warning_messages: tuple[str, ...],
    *,
    trace_label: str | None,
    group: str | None = None,
    phase: str,
) -> None:
    if not warning_messages:
        return
    prefix = _trace_prefix(trace_label)
    for message in warning_messages:
        if group is None:
            logger.warning("%s: pymfe warning during %s — %s", prefix, phase, message)
        else:
            logger.warning("%s: pymfe group `%s` warning during %s — %s", prefix, group, phase, message)


def _prepare_pymfe_input(
    X_train: pd.DataFrame,
) -> tuple[pd.DataFrame, list[int]]:
    X_encoded = X_train.copy()
    categorical_columns = get_categorical_columns(X_encoded)
    for column in categorical_columns:
        if X_encoded[column].isna().any():
            X_encoded[column] = X_encoded[column].fillna(X_encoded[column].mode().iloc[0])
        X_encoded[column] = X_encoded[column].astype("category").cat.codes
    X_encoded = X_encoded.apply(pd.to_numeric, errors="coerce")
    numeric_columns = [column for column in X_encoded.columns if column not in categorical_columns]
    if numeric_columns:
        numeric_frame = X_encoded.loc[:, numeric_columns]
        X_encoded.loc[:, numeric_columns] = numeric_frame.fillna(numeric_frame.median())
    categorical_indices = [X_encoded.columns.get_loc(column) for column in categorical_columns]
    return X_encoded, categorical_indices


def _normalize_pymfe_outputs(names, values) -> dict[str, float]:
    return {
        f"pymfe__{name}": float(value) if value is not None else np.nan
        for name, value in zip(names, values, strict=False)
    }


def extract_pymfe_features(
    X_train: pd.DataFrame,
    y_train: pd.Series | None,
    *,
    groups: tuple[str, ...],
    summary: tuple[str, ...],
    trace: bool = False,
    trace_label: str | None = None,
) -> dict[str, float]:
    """Extract optional pymfe meta-features."""
    try:
        from pymfe.mfe import MFE
    except ImportError as err:
        raise ImportError("`pymfe` is not installed. Install `meta-feature-analysis[pymfe]` to enable it.") from err

    if not groups:
        return {}

    trace_prefix = _trace_prefix(trace_label)
    X_encoded, categorical_indices = _prepare_pymfe_input(X_train)
    y_encoded = None if y_train is None else y_train.to_numpy()

    if not trace:
        mfe = MFE(groups=list(groups), summary=list(summary))
        mfe.fit(X_encoded.to_numpy(), y_encoded, cat_cols=categorical_indices)
        names, values = mfe.extract()
        return _normalize_pymfe_outputs(names, values)

    all_features: dict[str, float] = {}
    for group in groups:
        raw_features = tuple(MFE.valid_metafeatures(groups=(group,)))
        logger.info(
            "%s: pymfe group `%s`: calculating %d raw feature(s) with summary=%s (%s)",
            trace_prefix,
            group,
            len(raw_features),
            ",".join(summary),
            ", ".join(raw_features),
        )
        group_start = time.perf_counter()
        mfe = MFE(groups=[group], summary=list(summary), measure_time="total_summ")

        fit_records: list[warnings.WarningMessage] = []
        try:
            with warnings.catch_warnings(record=True) as fit_records:
                warnings.simplefilter("always")
                mfe.fit(X_encoded.to_numpy(), y_encoded, cat_cols=categorical_indices)
        except Exception:
            _log_warning_messages(
                _format_warning_messages(list(fit_records)),
                trace_label=trace_label,
                group=group,
                phase="fit",
            )
            raise
        _log_warning_messages(
            _format_warning_messages(list(fit_records)),
            trace_label=trace_label,
            group=group,
            phase="fit",
        )

        extract_records: list[warnings.WarningMessage] = []
        try:
            with warnings.catch_warnings(record=True) as extract_records:
                warnings.simplefilter("always")
                result = mfe.extract(out_type=dict)
        except Exception:
            _log_warning_messages(
                _format_warning_messages(list(extract_records)),
                trace_label=trace_label,
                group=group,
                phase="extract",
            )
            raise
        _log_warning_messages(
            _format_warning_messages(list(extract_records)),
            trace_label=trace_label,
            group=group,
            phase="extract",
        )

        names = result["mtf_names"]
        values = result["mtf_vals"]
        times = result["mtf_time"]
        logger.info(
            "%s: pymfe group `%s`: completed %d output(s) in %.3fs wall time",
            trace_prefix,
            group,
            len(names),
            time.perf_counter() - group_start,
        )
        for name, elapsed in zip(names, times, strict=False):
            logger.info(
                "%s: pymfe group `%s`: computed `%s` in %.6fs",
                trace_prefix,
                group,
                name,
                float(elapsed),
            )
        all_features.update(_normalize_pymfe_outputs(names, values))
    return all_features
