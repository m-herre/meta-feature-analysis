from __future__ import annotations

import sys
import types
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mfa.metafeatures import build_metafeature_table
from mfa.metafeatures.pymfe_features import _log_warning_messages, extract_pymfe_features
from mfa.metafeatures.registry import extract_requested_metafeatures


def test_extract_pymfe_features_imputes_numeric_missing_values_with_median(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class FakeMFE:
        def __init__(self, *, groups, summary) -> None:
            captured["groups"] = groups
            captured["summary"] = summary

        def fit(self, X, y, cat_cols) -> None:
            captured["X"] = X.copy()
            captured["y"] = y.copy()
            captured["cat_cols"] = list(cat_cols)

        def extract(self):
            return ["dummy"], [1.0]

    pymfe_module = types.ModuleType("pymfe")
    pymfe_module.__path__ = []
    mfe_module = types.ModuleType("pymfe.mfe")
    mfe_module.MFE = FakeMFE
    pymfe_module.mfe = mfe_module
    monkeypatch.setitem(sys.modules, "pymfe", pymfe_module)
    monkeypatch.setitem(sys.modules, "pymfe.mfe", mfe_module)

    X_train = pd.DataFrame(
        {
            "num": [1.0, np.nan, 5.0, 9.0],
            "cat": pd.Series(["a", "a", None, "b"], dtype="category"),
        }
    )
    y_train = pd.Series([0, 1, 0, 1])

    features = extract_pymfe_features(X_train, y_train, groups=("general",), summary=("mean",))

    np.testing.assert_allclose(captured["X"], np.array([[1.0, 0.0], [5.0, 0.0], [5.0, 0.0], [9.0, 1.0]]))
    np.testing.assert_array_equal(captured["y"], y_train.to_numpy())
    assert captured["cat_cols"] == [1]
    assert captured["groups"] == ["general"]
    assert captured["summary"] == ["mean"]
    assert features == {"pymfe__dummy": 1.0}


def test_extract_pymfe_features_trace_logs_group_contents_and_output_times(monkeypatch) -> None:
    info_messages: list[str] = []

    class FakeMFE:
        def __init__(self, *, groups, summary, measure_time=None) -> None:
            self.groups = groups
            self.summary = summary
            self.measure_time = measure_time

        @classmethod
        def valid_metafeatures(cls, groups=None):
            group = groups[0]
            if group == "general":
                return ("attr_to_inst", "nr_attr")
            return ("cor",)

        def fit(self, X, y, cat_cols) -> None:
            return None

        def extract(self, out_type=dict):
            assert out_type is dict
            if self.groups == ["general"]:
                return {
                    "mtf_names": ["attr_to_inst", "nr_attr"],
                    "mtf_vals": [0.5, 2.0],
                    "mtf_time": [0.125, 0.25],
                }
            return {
                "mtf_names": ["cor.mean"],
                "mtf_vals": [1.5],
                "mtf_time": [0.375],
            }

    pymfe_module = types.ModuleType("pymfe")
    pymfe_module.__path__ = []
    mfe_module = types.ModuleType("pymfe.mfe")
    mfe_module.MFE = FakeMFE
    pymfe_module.mfe = mfe_module
    monkeypatch.setitem(sys.modules, "pymfe", pymfe_module)
    monkeypatch.setitem(sys.modules, "pymfe.mfe", mfe_module)
    monkeypatch.setattr(
        "mfa.metafeatures.pymfe_features.logger.info",
        lambda message, *args: info_messages.append(message % args if args else message),
    )

    features = extract_pymfe_features(
        pd.DataFrame({"num": [1.0, 2.0, 3.0]}),
        pd.Series([0, 1, 0]),
        groups=("general", "statistical"),
        summary=("mean",),
        trace=True,
        trace_label="dataset_a r0 f0",
    )

    assert features == {
        "pymfe__attr_to_inst": 0.5,
        "pymfe__nr_attr": 2.0,
        "pymfe__cor.mean": 1.5,
    }
    assert any(
        "dataset_a r0 f0: pymfe group `general`: calculating 2 raw feature(s)" in message
        for message in info_messages
    )
    assert any("attr_to_inst, nr_attr" in message for message in info_messages)
    assert any(
        "dataset_a r0 f0: pymfe group `general`: computed `attr_to_inst` in 0.125000s" in message
        for message in info_messages
    )
    assert any(
        "dataset_a r0 f0: pymfe group `statistical`: computed `cor.mean` in 0.375000s" in message
        for message in info_messages
    )


def test_extract_pymfe_features_logs_captured_warning_messages(monkeypatch) -> None:
    warning_messages: list[str] = []
    monkeypatch.setattr(
        "mfa.metafeatures.pymfe_features.logger.warning",
        lambda message, *args: warning_messages.append(message % args if args else message),
    )

    _log_warning_messages(
        ("RuntimeWarning from fake_warning.py:7: precision loss",),
        trace_label="dataset_a r0 f0",
        group="general",
        phase="fit",
    )

    assert any("dataset_a r0 f0: pymfe group `general` warning during fit" in message for message in warning_messages)
    assert any("RuntimeWarning" in message and "precision loss" in message for message in warning_messages)


def test_extract_pymfe_features_non_trace_does_not_log_warning_causes(monkeypatch) -> None:
    logger_warning_messages: list[str] = []

    class WarningMFE:
        def __init__(self, *, groups, summary) -> None:
            pass

        def fit(self, X, y, cat_cols) -> None:
            warnings.warn("precision loss", RuntimeWarning, stacklevel=1)

        def extract(self):
            return ["dummy"], [1.0]

    pymfe_module = types.ModuleType("pymfe")
    pymfe_module.__path__ = []
    mfe_module = types.ModuleType("pymfe.mfe")
    mfe_module.MFE = WarningMFE
    pymfe_module.mfe = mfe_module
    monkeypatch.setitem(sys.modules, "pymfe", pymfe_module)
    monkeypatch.setitem(sys.modules, "pymfe.mfe", mfe_module)
    monkeypatch.setattr(
        "mfa.metafeatures.pymfe_features.logger.warning",
        lambda message, *args: logger_warning_messages.append(message % args if args else message),
    )

    with pytest.warns(RuntimeWarning, match="precision loss"):
        features = extract_pymfe_features(
            pd.DataFrame({"num": [1.0, 2.0, 3.0]}),
            pd.Series([0, 1, 0]),
            groups=("general",),
            summary=("mean",),
            trace=False,
            trace_label="dataset_a r0 f0",
        )

    assert features == {"pymfe__dummy": 1.0}
    assert logger_warning_messages == []


def test_extract_requested_metafeatures_isolates_pymfe_failure(monkeypatch) -> None:
    class ExplodingMFE:
        def __init__(self, *, groups, summary) -> None:
            pass

        def fit(self, X, y, cat_cols) -> None:
            raise RecursionError("maximum recursion depth exceeded")

        def extract(self):
            raise AssertionError("should not be reached")

    pymfe_module = types.ModuleType("pymfe")
    pymfe_module.__path__ = []
    mfe_module = types.ModuleType("pymfe.mfe")
    mfe_module.MFE = ExplodingMFE
    pymfe_module.mfe = mfe_module
    monkeypatch.setitem(sys.modules, "pymfe", pymfe_module)
    monkeypatch.setitem(sys.modules, "pymfe.mfe", mfe_module)

    X_train = pd.DataFrame({"num": [1.0, 2.0, 3.0, 4.0]})
    y_train = pd.Series([0, 1, 0, 1])

    features, failed_sets = extract_requested_metafeatures(
        X_train,
        y_train,
        feature_sets=("basic", "pymfe"),
        pymfe_groups=("general",),
        pymfe_summary=("mean",),
    )

    assert "pymfe" in failed_sets
    assert failed_sets["pymfe"].startswith("RecursionError")
    assert "basic" not in failed_sets
    assert features["n"] == 4
    assert not any(key.startswith("pymfe__") for key in features)


def test_build_metafeature_table_keeps_pymfe_columns_when_one_dataset_fails(
    monkeypatch, tmp_path: Path
) -> None:
    """One dataset's pymfe failure must not wipe pymfe columns from the others."""

    class SelectiveMFE:
        """Succeeds for all datasets except the one whose X has 7 rows."""

        def __init__(self, *, groups, summary) -> None:
            pass

        def fit(self, X, y, cat_cols) -> None:
            if X.shape[0] == 7:
                raise RecursionError("maximum recursion depth exceeded")

        def extract(self):
            return ["stat_a", "stat_b"], [0.5, 1.5]

    pymfe_module = types.ModuleType("pymfe")
    pymfe_module.__path__ = []
    mfe_module = types.ModuleType("pymfe.mfe")
    mfe_module.MFE = SelectiveMFE
    pymfe_module.mfe = mfe_module
    monkeypatch.setitem(sys.modules, "pymfe", pymfe_module)
    monkeypatch.setitem(sys.modules, "pymfe.mfe", mfe_module)

    class FakeTask:
        def __init__(self, n_rows: int) -> None:
            self._n_rows = n_rows

        def get_split_dimensions(self) -> tuple[int, int, int]:
            return 1, 1, 1

        def get_train_test_split(self, *, fold: int, repeat: int):
            X = pd.DataFrame({"num": np.arange(self._n_rows, dtype=float)})
            y = pd.Series(np.zeros(self._n_rows, dtype=int))
            return X, y, X, y

    def fake_from_task_id(task_id: int) -> FakeTask:
        # task_id 1 -> healthy (4 rows); task_id 2 -> poisoned (7 rows, raises).
        return FakeTask(n_rows=4 if task_id == 1 else 7)

    monkeypatch.setattr(
        "tabarena.benchmark.task.openml.OpenMLTaskWrapper.from_task_id", fake_from_task_id
    )

    metadata = pd.DataFrame(
        {
            "dataset": ["healthy", "poisoned"],
            "tid": [1, 2],
            "n_repeats": [1, 1],
            "n_folds": [1, 1],
        }
    )

    table = build_metafeature_table(
        metadata,
        cache_dir=tmp_path,
        use_cache=False,
        feature_sets=("basic", "pymfe"),
        pymfe_groups=("general",),
        pymfe_summary=("mean",),
        n_jobs=1,
    )

    assert set(table["dataset"]) == {"healthy", "poisoned"}
    # pymfe columns survive globally — the healthy dataset keeps its values.
    assert "pymfe__stat_a" in table.columns
    assert "pymfe__stat_b" in table.columns
    healthy = table.loc[table["dataset"] == "healthy"].iloc[0]
    poisoned = table.loc[table["dataset"] == "poisoned"].iloc[0]
    assert healthy["pymfe__stat_a"] == 0.5
    assert healthy["pymfe__stat_b"] == 1.5
    # Only the failing split carries NaN for pymfe features.
    assert np.isnan(poisoned["pymfe__stat_a"])
    assert np.isnan(poisoned["pymfe__stat_b"])
    # Basic features still computed for both datasets.
    assert healthy["n"] == 4
    assert poisoned["n"] == 7


def test_extract_requested_metafeatures_propagates_basic_failures(monkeypatch) -> None:
    """basic and irregularity are not best-effort — internal failures must surface."""
    import mfa.metafeatures.registry as registry

    def exploding_basic(_X_train):
        raise ValueError("deliberate basic failure")

    monkeypatch.setattr(registry, "compute_basic_metafeatures", exploding_basic)

    with pytest.raises(ValueError, match="deliberate basic failure"):
        extract_requested_metafeatures(
            pd.DataFrame({"num": [1.0, 2.0, 3.0]}),
            pd.Series([0, 1, 0]),
            feature_sets=("basic",),
            pymfe_groups=(),
            pymfe_summary=(),
        )


def test_extract_requested_metafeatures_propagates_irregularity_failures(monkeypatch) -> None:
    import mfa.metafeatures.registry as registry

    def exploding_irregularity(_X_num):
        raise RuntimeError("deliberate irregularity failure")

    monkeypatch.setattr(registry, "compute_irregularity_components", exploding_irregularity)

    with pytest.raises(RuntimeError, match="deliberate irregularity failure"):
        extract_requested_metafeatures(
            pd.DataFrame({"num": [1.0, 2.0, 3.0]}),
            pd.Series([0, 1, 0]),
            feature_sets=("irregularity",),
            pymfe_groups=(),
            pymfe_summary=(),
        )


def test_pymfe_failure_provenance_survives_rerun_from_cache(monkeypatch, tmp_path: Path) -> None:
    """A cached split that failed pymfe must still report the failure on rerun."""

    class FailingMFE:
        def __init__(self, *, groups, summary) -> None:
            pass

        def fit(self, X, y, cat_cols) -> None:
            raise RecursionError("maximum recursion depth exceeded")

        def extract(self):
            raise AssertionError("unreachable")

    pymfe_module = types.ModuleType("pymfe")
    pymfe_module.__path__ = []
    mfe_module = types.ModuleType("pymfe.mfe")
    mfe_module.MFE = FailingMFE
    pymfe_module.mfe = mfe_module
    monkeypatch.setitem(sys.modules, "pymfe", pymfe_module)
    monkeypatch.setitem(sys.modules, "pymfe.mfe", mfe_module)

    class FakeTask:
        def get_split_dimensions(self) -> tuple[int, int, int]:
            return 1, 1, 1

        def get_train_test_split(self, *, fold: int, repeat: int):
            X = pd.DataFrame({"num": [1.0, 2.0, 3.0, 4.0]})
            y = pd.Series([0, 1, 0, 1])
            return X, y, X, y

    monkeypatch.setattr(
        "tabarena.benchmark.task.openml.OpenMLTaskWrapper.from_task_id",
        lambda task_id: FakeTask(),
    )

    metadata = pd.DataFrame(
        {"dataset": ["poisoned"], "tid": [1], "n_repeats": [1], "n_folds": [1]}
    )

    # First run: pymfe fails, the split still caches basic features and a failed-sets marker.
    warnings_first: list[str] = []
    monkeypatch.setattr(
        "mfa.metafeatures.logger.warning",
        lambda message, *args: warnings_first.append(message % args if args else message),
    )
    build_metafeature_table(
        metadata,
        cache_dir=tmp_path,
        use_cache=True,
        feature_sets=("basic", "pymfe"),
        pymfe_groups=("general",),
        pymfe_summary=("mean",),
        n_jobs=1,
    )
    assert any("`pymfe` failed on 1/1 split(s)" in message for message in warnings_first)

    # Second run: cache hit. No extractor is invoked, but the failure summary must still fire.
    def must_not_fit(*args, **kwargs):
        raise AssertionError("cache hit must not re-invoke pymfe")

    mfe_module.MFE = type(
        "BoomMFE",
        (),
        {"__init__": lambda self, **_: None, "fit": must_not_fit, "extract": must_not_fit},
    )

    warnings_second: list[str] = []
    monkeypatch.setattr(
        "mfa.metafeatures.logger.warning",
        lambda message, *args: warnings_second.append(message % args if args else message),
    )
    build_metafeature_table(
        metadata,
        cache_dir=tmp_path,
        use_cache=True,
        feature_sets=("basic", "pymfe"),
        pymfe_groups=("general",),
        pymfe_summary=("mean",),
        n_jobs=1,
    )
    assert any("`pymfe` failed on 1/1 split(s)" in message for message in warnings_second)
