from __future__ import annotations

import multiprocessing as mp
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from .._logging import get_logger
from .basic import get_categorical_columns
from .pymfe_catalog import PYMFE_CLASSIFICATION_ONLY, should_filter_classification_only

logger = get_logger(__name__)

PROCESS_TERMINATE_GRACE_S = 5


def enumerate_pymfe_raw_features(
    groups: tuple[str, ...],
    *,
    problem_type: str | None = None,
    raw_features: tuple[str, ...] | None = None,
) -> tuple[str, ...]:
    """Enumerate raw pymfe feature names requested by the current config."""
    from pymfe.mfe import MFE

    selected = None if raw_features is None else set(raw_features)
    filter_classification_only = should_filter_classification_only(problem_type)
    if not hasattr(MFE, "valid_metafeatures"):
        if raw_features is None:
            return ()
        return tuple(
            dict.fromkeys(
                feature
                for feature in raw_features
                if not (filter_classification_only and feature in PYMFE_CLASSIFICATION_ONLY)
            )
        )
    enumerated: list[str] = []
    for group in groups:
        for feature in MFE.valid_metafeatures(groups=(group,)):
            if selected is not None and feature not in selected:
                continue
            if filter_classification_only and feature in PYMFE_CLASSIFICATION_ONLY:
                continue
            if feature not in enumerated:
                enumerated.append(feature)
    return tuple(enumerated)


def _regression_allowed_features(groups: tuple[str, ...]) -> list[str]:
    """Enumerate pymfe features for the given groups minus the classification-only deny list."""
    return list(enumerate_pymfe_raw_features(groups, problem_type="regression"))


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


def _compute_feature_worker(
    queue,
    X_np,
    y_np,
    cat_indices,
    group: str,
    feature: str,
    summary: tuple[str, ...],
) -> None:
    """Subprocess target: fit+extract a single pymfe feature, push result to `queue`.

    Payload shape: ``(status, data, warning_messages)`` where ``warning_messages``
    is a tuple of formatted strings captured during fit+extract so the parent
    process can log them with full provenance (mirrors batch-path behavior).
    """
    try:
        from pymfe.mfe import MFE
    except ImportError as err:
        queue.put(("err", f"ImportError: {err}", ()))
        return
    try:
        mfe = MFE(groups=[group], features=[feature], summary=list(summary))
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            mfe.fit(X_np, y_np, cat_cols=cat_indices)
            names, values = mfe.extract()
        warning_messages = _format_warning_messages(list(captured))
        queue.put(("ok", dict(zip(names, values, strict=False)), warning_messages))
    except Exception as exc:
        queue.put(("err", f"{type(exc).__name__}: {exc}", ()))


def _extract_per_feature_with_timeout(
    X_encoded: pd.DataFrame,
    y_encoded,
    categorical_indices: list[int],
    groups: tuple[str, ...],
    summary: tuple[str, ...],
    timeout_s: float,
    trace: bool,
    trace_label: str | None,
    problem_type: str | None,
    raw_features: tuple[str, ...] | None,
) -> dict[str, float]:
    """Extract pymfe features one at a time in isolated subprocesses.

    Each feature runs in a fresh subprocess; if it exceeds `timeout_s` the
    process is terminated and the feature skipped. Non-zero exit or in-worker
    exceptions are also caught and the feature is skipped (not dropped from
    the result dict as NaN — absent keys become NaN at DataFrame construction).
    """
    try:
        from pymfe.mfe import MFE
    except ImportError as err:
        raise ImportError("`pymfe` is not installed. Install `meta-feature-analysis[pymfe]` to enable it.") from err

    trace_prefix = _trace_prefix(trace_label)
    X_np = X_encoded.to_numpy()
    results: dict[str, float] = {}
    skipped: list[str] = []
    ctx = mp.get_context()

    filter_classification_only = should_filter_classification_only(problem_type)
    selected = None if raw_features is None else set(raw_features)
    for group in groups:
        raw_features = tuple(MFE.valid_metafeatures(groups=(group,)))
        if selected is not None:
            raw_features = tuple(f for f in raw_features if f in selected)
        if filter_classification_only:
            skipped_in_group = [f for f in raw_features if f in PYMFE_CLASSIFICATION_ONLY]
            raw_features = tuple(f for f in raw_features if f not in PYMFE_CLASSIFICATION_ONLY)
            if trace and skipped_in_group:
                logger.info(
                    "%s: pymfe group `%s`: skipping %d classification-only feature(s) for problem_type=%s (%s)",
                    trace_prefix,
                    group,
                    len(skipped_in_group),
                    problem_type,
                    ", ".join(skipped_in_group),
                )
        if trace:
            logger.info(
                "%s: pymfe group `%s`: computing %d feature(s) with per-feature timeout %.0fs",
                trace_prefix,
                group,
                len(raw_features),
                timeout_s,
            )
        for feature in raw_features:
            feature_start = time.perf_counter() if trace else None
            queue = ctx.Queue()
            proc = ctx.Process(
                target=_compute_feature_worker,
                args=(queue, X_np, y_encoded, categorical_indices, group, feature, summary),
                daemon=True,
            )
            proc.start()
            proc.join(timeout=timeout_s)

            if proc.is_alive():
                proc.terminate()
                proc.join(timeout=PROCESS_TERMINATE_GRACE_S)
                if proc.is_alive():
                    proc.kill()
                    proc.join()
                skipped.append(f"{group}.{feature}")
                logger.warning(
                    "%s: pymfe feature `%s.%s` exceeded %.0fs timeout, skipping",
                    trace_prefix,
                    group,
                    feature,
                    timeout_s,
                )
                _drain_queue(queue)
                continue

            if proc.exitcode != 0:
                skipped.append(f"{group}.{feature}")
                logger.warning(
                    "%s: pymfe feature `%s.%s` crashed (exit code %s), skipping",
                    trace_prefix,
                    group,
                    feature,
                    proc.exitcode,
                )
                _drain_queue(queue)
                continue

            try:
                message = queue.get(timeout=PROCESS_TERMINATE_GRACE_S)
            except Exception as exc:
                skipped.append(f"{group}.{feature}")
                logger.warning(
                    "%s: pymfe feature `%s.%s` produced no result (%s), skipping",
                    trace_prefix,
                    group,
                    feature,
                    exc,
                )
                continue

            status, payload, warning_messages = message
            _log_warning_messages(
                warning_messages,
                trace_label=trace_label,
                group=group,
                phase="fit+extract",
            )

            if status == "ok":
                results.update(_normalize_pymfe_outputs(list(payload.keys()), list(payload.values())))
                if trace:
                    logger.info(
                        "%s: pymfe group `%s`: computed `%s` in %.6fs",
                        trace_prefix,
                        group,
                        feature,
                        time.perf_counter() - feature_start,
                    )
            else:
                skipped.append(f"{group}.{feature}")
                logger.warning(
                    "%s: pymfe feature `%s.%s` failed: %s, skipping",
                    trace_prefix,
                    group,
                    feature,
                    payload,
                )

    if skipped:
        logger.warning(
            "%s: skipped %d pymfe feature(s) due to timeout or crash: %s",
            trace_prefix,
            len(skipped),
            ", ".join(skipped),
        )
    return results


def _drain_queue(queue) -> None:
    """Best-effort drain of a multiprocessing queue after the producer is gone."""
    try:
        while not queue.empty():
            queue.get_nowait()
    except Exception:
        pass


def extract_pymfe_features(
    X_train: pd.DataFrame,
    y_train: pd.Series | None,
    *,
    groups: tuple[str, ...],
    summary: tuple[str, ...],
    problem_type: str | None = None,
    per_feature_timeout_s: float | None = None,
    raw_features: tuple[str, ...] | None = None,
    trace: bool = False,
    trace_label: str | None = None,
) -> dict[str, float]:
    """Extract optional pymfe meta-features.

    When `per_feature_timeout_s` is set, each feature is computed in an
    isolated subprocess and skipped if it exceeds the timeout or crashes the
    worker. Otherwise the full set of groups is extracted in one fit/extract
    batch (the default, efficient path).
    """
    try:
        from pymfe.mfe import MFE
    except ImportError as err:
        raise ImportError("`pymfe` is not installed. Install `meta-feature-analysis[pymfe]` to enable it.") from err

    if not groups:
        return {}

    trace_prefix = _trace_prefix(trace_label)
    X_encoded, categorical_indices = _prepare_pymfe_input(X_train)
    y_encoded = None if y_train is None else y_train.to_numpy()
    selected_raw_features = None if raw_features is None else set(raw_features)

    filter_classification_only = should_filter_classification_only(problem_type)
    if filter_classification_only:
        skipped_count = sum(
            1
            for group in groups
            for feature in MFE.valid_metafeatures(groups=(group,))
            if selected_raw_features is None or feature in selected_raw_features
            if feature in PYMFE_CLASSIFICATION_ONLY
        )
        if skipped_count:
            logger.info(
                "%s: problem_type=%s; skipping %d classification-only pymfe feature(s)",
                trace_prefix,
                problem_type,
                skipped_count,
            )

    if per_feature_timeout_s is not None:
        return _extract_per_feature_with_timeout(
            X_encoded,
            y_encoded,
            categorical_indices,
            groups,
            summary,
            per_feature_timeout_s,
            trace,
            trace_label,
            problem_type,
            raw_features,
        )

    if not trace:
        if raw_features is not None:
            allowed = list(
                enumerate_pymfe_raw_features(
                    groups,
                    problem_type=problem_type,
                    raw_features=raw_features,
                )
            )
            if not allowed:
                return {}
            mfe = MFE(groups=list(groups), features=allowed, summary=list(summary))
        elif filter_classification_only:
            allowed = _regression_allowed_features(groups)
            if not allowed:
                return {}
            mfe = MFE(groups=list(groups), features=allowed, summary=list(summary))
        else:
            mfe = MFE(groups=list(groups), summary=list(summary))
        mfe.fit(X_encoded.to_numpy(), y_encoded, cat_cols=categorical_indices)
        names, values = mfe.extract()
        return _normalize_pymfe_outputs(names, values)

    all_features: dict[str, float] = {}
    for group in groups:
        raw_features = tuple(MFE.valid_metafeatures(groups=(group,)))
        if selected_raw_features is not None:
            raw_features = tuple(f for f in raw_features if f in selected_raw_features)
        if filter_classification_only:
            raw_features = tuple(f for f in raw_features if f not in PYMFE_CLASSIFICATION_ONLY)
            if not raw_features:
                logger.info(
                    "%s: pymfe group `%s`: all features classification-only, skipping group",
                    trace_prefix,
                    group,
                )
                continue
        logger.info(
            "%s: pymfe group `%s`: calculating %d raw feature(s) with summary=%s (%s)",
            trace_prefix,
            group,
            len(raw_features),
            ",".join(summary),
            ", ".join(raw_features),
        )
        group_start = time.perf_counter()
        mfe_kwargs = {
            "groups": [group],
            "summary": list(summary),
            "measure_time": "total_summ",
        }
        if filter_classification_only or selected_raw_features is not None:
            mfe_kwargs["features"] = list(raw_features)
        mfe = MFE(**mfe_kwargs)

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
