from __future__ import annotations

import sys
import types
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mfa.cache import metafeature_split_cache_dir
from mfa.metafeatures import build_metafeature_table
from mfa.metafeatures.pymfe_catalog import (
    PYMFE_CLASSIFICATION_ONLY,
    PYMFE_REGRESSION_SAFE,
    is_classification,
    should_filter_classification_only,
)
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
        "dataset_a r0 f0: pymfe group `general`: calculating 2 raw feature(s)" in message for message in info_messages
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


def test_extract_pymfe_features_per_feature_timeout_skips_hanging_feature(monkeypatch) -> None:
    import multiprocessing as mp

    if mp.get_start_method(allow_none=True) not in (None, "fork"):
        pytest.skip("per-feature subprocess timeout test relies on fork() inheritance of monkeypatches")

    warning_messages: list[str] = []
    monkeypatch.setattr(
        "mfa.metafeatures.pymfe_features.logger.warning",
        lambda message, *args: warning_messages.append(message % args if args else message),
    )

    class FakeMFE:
        def __init__(self, *, groups, features=None, summary) -> None:
            self.groups = groups
            self.features = list(features) if features is not None else None
            self.summary = summary

        @classmethod
        def valid_metafeatures(cls, groups=None):
            return ("fast_feature", "slow_feature")

        def fit(self, X, y, cat_cols) -> None:
            import time

            if self.features == ["slow_feature"]:
                time.sleep(5)

        def extract(self):
            assert self.features is not None
            name = self.features[0]
            return [name], [42.0]

    pymfe_module = types.ModuleType("pymfe")
    pymfe_module.__path__ = []
    mfe_module = types.ModuleType("pymfe.mfe")
    mfe_module.MFE = FakeMFE
    pymfe_module.mfe = mfe_module
    monkeypatch.setitem(sys.modules, "pymfe", pymfe_module)
    monkeypatch.setitem(sys.modules, "pymfe.mfe", mfe_module)

    features = extract_pymfe_features(
        pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0]}),
        pd.Series([0, 1, 0, 1]),
        groups=("general",),
        summary=("mean",),
        per_feature_timeout_s=0.5,
    )

    assert features == {"pymfe__fast_feature": 42.0}
    assert any("slow_feature" in message and "timeout" in message for message in warning_messages)


def test_extract_pymfe_features_per_feature_skips_crashing_feature(monkeypatch) -> None:
    import multiprocessing as mp

    if mp.get_start_method(allow_none=True) not in (None, "fork"):
        pytest.skip("per-feature subprocess crash test relies on fork() inheritance of monkeypatches")

    warning_messages: list[str] = []
    monkeypatch.setattr(
        "mfa.metafeatures.pymfe_features.logger.warning",
        lambda message, *args: warning_messages.append(message % args if args else message),
    )

    class FakeMFE:
        def __init__(self, *, groups, features=None, summary) -> None:
            self.groups = groups
            self.features = list(features) if features is not None else None
            self.summary = summary

        @classmethod
        def valid_metafeatures(cls, groups=None):
            return ("good_feature", "crashy_feature")

        def fit(self, X, y, cat_cols) -> None:
            import os

            if self.features == ["crashy_feature"]:
                os._exit(137)

        def extract(self):
            assert self.features is not None
            name = self.features[0]
            return [name], [7.0]

    pymfe_module = types.ModuleType("pymfe")
    pymfe_module.__path__ = []
    mfe_module = types.ModuleType("pymfe.mfe")
    mfe_module.MFE = FakeMFE
    pymfe_module.mfe = mfe_module
    monkeypatch.setitem(sys.modules, "pymfe", pymfe_module)
    monkeypatch.setitem(sys.modules, "pymfe.mfe", mfe_module)

    features = extract_pymfe_features(
        pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0]}),
        pd.Series([0, 1, 0, 1]),
        groups=("general",),
        summary=("mean",),
        per_feature_timeout_s=5.0,
    )

    assert features == {"pymfe__good_feature": 7.0}
    assert any("crashy_feature" in message and "crashed" in message for message in warning_messages)


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


def test_build_metafeature_table_keeps_pymfe_columns_when_one_dataset_fails(monkeypatch, tmp_path: Path) -> None:
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

    monkeypatch.setattr("tabarena.benchmark.task.openml.OpenMLTaskWrapper.from_task_id", fake_from_task_id)

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

    def exploding_basic(_X_train, _y_train=None, problem_type=None):
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

        @classmethod
        def valid_metafeatures(cls, groups=None):
            return ("deterministic_failure",)

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

    metadata = pd.DataFrame({"dataset": ["poisoned"], "tid": [1], "n_repeats": [1], "n_folds": [1]})

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

    class BoomMFE:
        def __init__(self, **kwargs) -> None:
            pass

        @classmethod
        def valid_metafeatures(cls, groups=None):
            return ("deterministic_failure",)

        fit = must_not_fit
        extract = must_not_fit

    mfe_module.MFE = BoomMFE

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


def test_complete_pymfe_split_cache_hit_does_not_load_task(monkeypatch, tmp_path: Path) -> None:
    class CompleteMFE:
        def __init__(self, *, groups, summary, features=None) -> None:
            self.features = list(features) if features is not None else None

        @classmethod
        def valid_metafeatures(cls, groups=None):
            return ("fast", "slow")

        def fit(self, X, y, cat_cols) -> None:
            return None

        def extract(self):
            names = self.features if self.features is not None else ["fast", "slow"]
            values = [1.0 if name == "fast" else 2.0 for name in names]
            return names, values

    pymfe_module = types.ModuleType("pymfe")
    pymfe_module.__path__ = []
    mfe_module = types.ModuleType("pymfe.mfe")
    mfe_module.MFE = CompleteMFE
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

    monkeypatch.setattr("tabarena.benchmark.task.openml.OpenMLTaskWrapper.from_task_id", lambda task_id: FakeTask())
    metadata = pd.DataFrame({"dataset": ["dataset_a"], "tid": [1], "n_repeats": [1], "n_folds": [1]})

    first = build_metafeature_table(
        metadata,
        cache_dir=tmp_path,
        use_cache=True,
        feature_sets=("basic", "pymfe"),
        pymfe_groups=("general",),
        pymfe_summary=("mean",),
    )

    def fail_from_task_id(task_id: int):
        raise AssertionError("complete pymfe split cache should not load the task")

    monkeypatch.setattr("tabarena.benchmark.task.openml.OpenMLTaskWrapper.from_task_id", fail_from_task_id)
    second = build_metafeature_table(
        metadata,
        cache_dir=tmp_path,
        use_cache=True,
        feature_sets=("basic", "pymfe"),
        pymfe_groups=("general",),
        pymfe_summary=("mean",),
    )

    pd.testing.assert_frame_equal(second, first)


def test_missing_pymfe_split_cache_cell_repairs_only_missing_raw_feature(monkeypatch, tmp_path: Path) -> None:
    captured: dict = {"instances": []}

    class SelectiveMFE:
        def __init__(self, *, groups, summary, features=None) -> None:
            self.features = list(features) if features is not None else None
            captured["instances"].append(self.features)

        @classmethod
        def valid_metafeatures(cls, groups=None):
            return ("fast", "slow")

        def fit(self, X, y, cat_cols) -> None:
            return None

        def extract(self):
            names = self.features if self.features is not None else ["fast", "slow"]
            values = [1.0 if name == "fast" else 22.0 for name in names]
            return names, values

    pymfe_module = types.ModuleType("pymfe")
    pymfe_module.__path__ = []
    mfe_module = types.ModuleType("pymfe.mfe")
    mfe_module.MFE = SelectiveMFE
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

    monkeypatch.setattr("tabarena.benchmark.task.openml.OpenMLTaskWrapper.from_task_id", lambda task_id: FakeTask())
    metadata = pd.DataFrame({"dataset": ["dataset_a"], "tid": [1], "n_repeats": [1], "n_folds": [1]})

    build_metafeature_table(
        metadata,
        cache_dir=tmp_path,
        use_cache=True,
        feature_sets=("basic", "pymfe"),
        pymfe_groups=("general",),
        pymfe_summary=("mean",),
    )

    split_path = metafeature_split_cache_dir(tmp_path) / "dataset_a__r0__f0.parquet"
    cached = pd.read_parquet(split_path)
    cached = cached.drop(columns=["pymfe__slow"])
    cached.to_parquet(split_path, index=False)
    captured["instances"].clear()

    repaired = build_metafeature_table(
        metadata,
        cache_dir=tmp_path,
        use_cache=True,
        feature_sets=("basic", "pymfe"),
        pymfe_groups=("general",),
        pymfe_summary=("mean",),
    )

    row = repaired.iloc[0]
    assert row["pymfe__fast"] == 1.0
    assert row["pymfe__slow"] == 22.0
    assert captured["instances"] == [["slow"]]
    rewritten = pd.read_parquet(split_path)
    assert rewritten["pymfe__slow"].iat[0] == 22.0


def test_existing_nan_pymfe_split_cache_cell_is_not_repaired(monkeypatch, tmp_path: Path) -> None:
    class NanMFE:
        def __init__(self, *, groups, summary, features=None) -> None:
            self.features = list(features) if features is not None else None

        @classmethod
        def valid_metafeatures(cls, groups=None):
            return ("defined", "undefined")

        def fit(self, X, y, cat_cols) -> None:
            return None

        def extract(self):
            names = self.features if self.features is not None else ["defined", "undefined"]
            values = [1.0 if name == "defined" else np.nan for name in names]
            return names, values

    pymfe_module = types.ModuleType("pymfe")
    pymfe_module.__path__ = []
    mfe_module = types.ModuleType("pymfe.mfe")
    mfe_module.MFE = NanMFE
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

    monkeypatch.setattr("tabarena.benchmark.task.openml.OpenMLTaskWrapper.from_task_id", lambda task_id: FakeTask())
    metadata = pd.DataFrame({"dataset": ["dataset_a"], "tid": [1], "n_repeats": [1], "n_folds": [1]})

    first = build_metafeature_table(
        metadata,
        cache_dir=tmp_path,
        use_cache=True,
        feature_sets=("basic", "pymfe"),
        pymfe_groups=("general",),
        pymfe_summary=("mean",),
    )
    assert first["pymfe__defined"].iat[0] == 1.0
    assert np.isnan(first["pymfe__undefined"].iat[0])

    def fail_from_task_id(task_id: int):
        raise AssertionError("existing pymfe NaN values should remain cache hits")

    monkeypatch.setattr("tabarena.benchmark.task.openml.OpenMLTaskWrapper.from_task_id", fail_from_task_id)
    second = build_metafeature_table(
        metadata,
        cache_dir=tmp_path,
        use_cache=True,
        feature_sets=("basic", "pymfe"),
        pymfe_groups=("general",),
        pymfe_summary=("mean",),
    )

    pd.testing.assert_frame_equal(second, first)


def test_timeout_increase_reuses_partial_split_cache_and_repairs_missing_pymfe(monkeypatch, tmp_path: Path) -> None:
    import multiprocessing as mp

    if mp.get_start_method(allow_none=True) not in (None, "fork"):
        pytest.skip("per-feature subprocess repair test relies on fork() inheritance of monkeypatches")

    marker_dir = tmp_path / "markers"
    marker_dir.mkdir()

    class TimeoutMFE:
        def __init__(self, *, groups, features=None, summary) -> None:
            self.features = list(features) if features is not None else None

        @classmethod
        def valid_metafeatures(cls, groups=None):
            return ("fast", "slow")

        def fit(self, X, y, cat_cols) -> None:
            import time

            feature = self.features[0]
            (marker_dir / feature).write_text("seen", encoding="utf-8")
            if feature == "slow":
                time.sleep(0.2)

        def extract(self):
            feature = self.features[0]
            return [feature], [1.0 if feature == "fast" else 2.0]

    pymfe_module = types.ModuleType("pymfe")
    pymfe_module.__path__ = []
    mfe_module = types.ModuleType("pymfe.mfe")
    mfe_module.MFE = TimeoutMFE
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

    monkeypatch.setattr("tabarena.benchmark.task.openml.OpenMLTaskWrapper.from_task_id", lambda task_id: FakeTask())
    metadata = pd.DataFrame({"dataset": ["dataset_a"], "tid": [1], "n_repeats": [1], "n_folds": [1]})

    partial = build_metafeature_table(
        metadata,
        cache_dir=tmp_path,
        use_cache=True,
        feature_sets=("basic", "pymfe"),
        pymfe_groups=("general",),
        pymfe_summary=("mean",),
        pymfe_per_feature_timeout_s=0.05,
    )
    assert partial["pymfe__fast"].iat[0] == 1.0
    assert "pymfe__slow" not in partial.columns

    for marker in marker_dir.iterdir():
        marker.unlink()

    repaired = build_metafeature_table(
        metadata,
        cache_dir=tmp_path,
        use_cache=True,
        feature_sets=("basic", "pymfe"),
        pymfe_groups=("general",),
        pymfe_summary=("mean",),
        pymfe_per_feature_timeout_s=1.0,
    )

    assert repaired["pymfe__fast"].iat[0] == 1.0
    assert repaired["pymfe__slow"].iat[0] == 2.0
    assert sorted(path.name for path in marker_dir.iterdir()) == ["slow"]


def test_non_pymfe_nan_in_cached_split_is_not_repaired(monkeypatch, tmp_path: Path) -> None:
    class SafeMFE:
        def __init__(self, *, groups, summary, features=None) -> None:
            self.features = list(features) if features is not None else None

        @classmethod
        def valid_metafeatures(cls, groups=None):
            return ("nr_attr",)

        def fit(self, X, y, cat_cols) -> None:
            return None

        def extract(self):
            names = self.features if self.features is not None else ["nr_attr"]
            return names, [3.0] * len(names)

    pymfe_module = types.ModuleType("pymfe")
    pymfe_module.__path__ = []
    mfe_module = types.ModuleType("pymfe.mfe")
    mfe_module.MFE = SafeMFE
    pymfe_module.mfe = mfe_module
    monkeypatch.setitem(sys.modules, "pymfe", pymfe_module)
    monkeypatch.setitem(sys.modules, "pymfe.mfe", mfe_module)

    class FakeTask:
        def get_split_dimensions(self) -> tuple[int, int, int]:
            return 1, 1, 1

        def get_train_test_split(self, *, fold: int, repeat: int):
            X = pd.DataFrame({"num": [1.0, 2.0, 3.0, 4.0]})
            y = pd.Series([0.1, 0.2, 0.3, 0.4])
            return X, y, X, y

    monkeypatch.setattr("tabarena.benchmark.task.openml.OpenMLTaskWrapper.from_task_id", lambda task_id: FakeTask())
    metadata = pd.DataFrame(
        {
            "dataset": ["regression_dataset"],
            "tid": [1],
            "problem_type": ["regression"],
            "n_repeats": [1],
            "n_folds": [1],
        }
    )
    first = build_metafeature_table(
        metadata,
        cache_dir=tmp_path,
        use_cache=True,
        feature_sets=("basic", "pymfe"),
        pymfe_groups=("general",),
        pymfe_summary=("mean",),
    )
    assert np.isnan(first["n_classes"].iat[0])

    def fail_from_task_id(task_id: int):
        raise AssertionError("non-pymfe NaNs should not trigger split repair")

    monkeypatch.setattr("tabarena.benchmark.task.openml.OpenMLTaskWrapper.from_task_id", fail_from_task_id)
    second = build_metafeature_table(
        metadata,
        cache_dir=tmp_path,
        use_cache=True,
        feature_sets=("basic", "pymfe"),
        pymfe_groups=("general",),
        pymfe_summary=("mean",),
    )

    pd.testing.assert_frame_equal(second, first)


def test_pymfe_catalog_lists_are_disjoint() -> None:
    assert PYMFE_CLASSIFICATION_ONLY.isdisjoint(PYMFE_REGRESSION_SAFE)


def test_is_classification_helpers() -> None:
    assert is_classification("binary")
    assert is_classification("MULTICLASS")
    assert not is_classification("regression")
    assert not is_classification(None)
    assert should_filter_classification_only("regression")
    assert should_filter_classification_only("Regression")
    assert not should_filter_classification_only("binary")
    assert not should_filter_classification_only(None)


def _install_fake_mfe(
    monkeypatch,
    *,
    valid_by_group: dict[str, tuple[str, ...]],
    outputs_by_features: dict[tuple[str, ...], tuple[list[str], list[float]]] | None = None,
    captured: dict | None = None,
) -> None:
    """Install a FakeMFE in sys.modules that records constructor kwargs.

    Mirrors the pattern used by earlier tests in this file. `captured` (if
    given) receives a list under key `instances`, each entry a dict of the
    kwargs passed to MFE().
    """

    class FakeMFE:
        def __init__(self, *, groups, summary, features=None, measure_time=None) -> None:
            self.groups = list(groups)
            self.summary = list(summary)
            self.features = list(features) if features is not None else None
            self.measure_time = measure_time
            if captured is not None:
                captured.setdefault("instances", []).append(
                    {
                        "groups": self.groups,
                        "summary": self.summary,
                        "features": self.features,
                        "measure_time": self.measure_time,
                    }
                )

        @classmethod
        def valid_metafeatures(cls, groups=None):
            if not groups:
                return ()
            collected: list[str] = []
            for group in groups:
                for feature in valid_by_group.get(group, ()):
                    if feature not in collected:
                        collected.append(feature)
            return tuple(collected)

        def fit(self, X, y, cat_cols) -> None:
            return None

        def extract(self, out_type=None):
            key = tuple(self.features) if self.features is not None else ("__all__",)
            if outputs_by_features and key in outputs_by_features:
                names, values = outputs_by_features[key]
            else:
                names = list(self.features) if self.features is not None else ["dummy"]
                values = [1.0] * len(names)
            if out_type is dict:
                return {
                    "mtf_names": names,
                    "mtf_vals": values,
                    "mtf_time": [0.0] * len(names),
                }
            return names, values

    pymfe_module = types.ModuleType("pymfe")
    pymfe_module.__path__ = []
    mfe_module = types.ModuleType("pymfe.mfe")
    mfe_module.MFE = FakeMFE
    pymfe_module.mfe = mfe_module
    monkeypatch.setitem(sys.modules, "pymfe", pymfe_module)
    monkeypatch.setitem(sys.modules, "pymfe.mfe", mfe_module)


def test_extract_pymfe_features_filters_classification_only_for_regression(monkeypatch) -> None:
    captured: dict = {}
    _install_fake_mfe(
        monkeypatch,
        valid_by_group={"general": ("nr_attr", "nr_class"), "statistical": ("cor", "can_cor")},
        captured=captured,
    )

    features = extract_pymfe_features(
        pd.DataFrame({"x": [1.0, 2.0, 3.0]}),
        pd.Series([1.1, 2.2, 3.3]),
        groups=("general", "statistical"),
        summary=("mean",),
        problem_type="regression",
    )

    assert len(captured["instances"]) == 1
    call = captured["instances"][0]
    assert set(call["features"]) == {"nr_attr", "cor"}
    assert "nr_class" not in call["features"]
    assert "can_cor" not in call["features"]
    assert features == {"pymfe__nr_attr": 1.0, "pymfe__cor": 1.0}


def test_extract_pymfe_features_no_filter_for_binary(monkeypatch) -> None:
    captured: dict = {}
    _install_fake_mfe(
        monkeypatch,
        valid_by_group={"general": ("nr_attr", "nr_class")},
        captured=captured,
    )

    extract_pymfe_features(
        pd.DataFrame({"x": [1.0, 2.0, 3.0]}),
        pd.Series([0, 1, 0]),
        groups=("general",),
        summary=("mean",),
        problem_type="binary",
    )

    assert len(captured["instances"]) == 1
    assert captured["instances"][0]["features"] is None


def test_extract_pymfe_features_no_filter_for_multiclass(monkeypatch) -> None:
    captured: dict = {}
    _install_fake_mfe(
        monkeypatch,
        valid_by_group={"general": ("nr_attr", "nr_class")},
        captured=captured,
    )

    extract_pymfe_features(
        pd.DataFrame({"x": [1.0, 2.0, 3.0]}),
        pd.Series([0, 1, 2]),
        groups=("general",),
        summary=("mean",),
        problem_type="multiclass",
    )

    assert captured["instances"][0]["features"] is None


def test_extract_pymfe_features_no_filter_when_problem_type_is_none(monkeypatch) -> None:
    captured: dict = {}
    _install_fake_mfe(
        monkeypatch,
        valid_by_group={"general": ("nr_attr", "nr_class")},
        captured=captured,
    )

    extract_pymfe_features(
        pd.DataFrame({"x": [1.0, 2.0, 3.0]}),
        pd.Series([0, 1, 0]),
        groups=("general",),
        summary=("mean",),
    )

    assert captured["instances"][0]["features"] is None


def test_extract_pymfe_features_trace_path_filters_for_regression(monkeypatch) -> None:
    captured: dict = {}
    _install_fake_mfe(
        monkeypatch,
        valid_by_group={"general": ("nr_attr", "nr_class"), "statistical": ("cor",)},
        captured=captured,
    )

    features = extract_pymfe_features(
        pd.DataFrame({"x": [1.0, 2.0, 3.0]}),
        pd.Series([1.1, 2.2, 3.3]),
        groups=("general", "statistical"),
        summary=("mean",),
        problem_type="regression",
        trace=True,
    )

    per_group_calls = [call for call in captured["instances"] if len(call["groups"]) == 1]
    general_call = next(call for call in per_group_calls if call["groups"] == ["general"])
    assert general_call["features"] == ["nr_attr"]
    statistical_call = next(call for call in per_group_calls if call["groups"] == ["statistical"])
    assert statistical_call["features"] == ["cor"]
    assert features == {"pymfe__nr_attr": 1.0, "pymfe__cor": 1.0}


def test_extract_pymfe_features_trace_path_skips_all_classification_only_group(monkeypatch) -> None:
    info_messages: list[str] = []
    monkeypatch.setattr(
        "mfa.metafeatures.pymfe_features.logger.info",
        lambda message, *args: info_messages.append(message % args if args else message),
    )
    _install_fake_mfe(
        monkeypatch,
        valid_by_group={"concept": ("conceptvar", "impconceptvar"), "general": ("nr_attr",)},
    )

    features = extract_pymfe_features(
        pd.DataFrame({"x": [1.0, 2.0, 3.0]}),
        pd.Series([1.1, 2.2, 3.3]),
        groups=("concept", "general"),
        summary=("mean",),
        problem_type="regression",
        trace=True,
    )

    assert features == {"pymfe__nr_attr": 1.0}
    assert any("pymfe group `concept`: all features classification-only" in message for message in info_messages)


def test_per_feature_timeout_path_skips_classification_only_for_regression(monkeypatch) -> None:
    import multiprocessing as mp

    if mp.get_start_method(allow_none=True) not in (None, "fork"):
        pytest.skip("per-feature subprocess test relies on fork() inheritance of monkeypatches")

    info_messages: list[str] = []
    monkeypatch.setattr(
        "mfa.metafeatures.pymfe_features.logger.info",
        lambda message, *args: info_messages.append(message % args if args else message),
    )

    fit_calls: list[list[str] | None] = []

    class FakeMFE:
        def __init__(self, *, groups, features=None, summary) -> None:
            self.groups = groups
            self.features = list(features) if features is not None else None
            self.summary = summary

        @classmethod
        def valid_metafeatures(cls, groups=None):
            return ("nr_attr", "nr_class")

        def fit(self, X, y, cat_cols) -> None:
            fit_calls.append(self.features)

        def extract(self):
            assert self.features is not None
            name = self.features[0]
            return [name], [9.0]

    pymfe_module = types.ModuleType("pymfe")
    pymfe_module.__path__ = []
    mfe_module = types.ModuleType("pymfe.mfe")
    mfe_module.MFE = FakeMFE
    pymfe_module.mfe = mfe_module
    monkeypatch.setitem(sys.modules, "pymfe", pymfe_module)
    monkeypatch.setitem(sys.modules, "pymfe.mfe", mfe_module)

    features = extract_pymfe_features(
        pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0]}),
        pd.Series([1.1, 2.2, 3.3, 4.4]),
        groups=("general",),
        summary=("mean",),
        problem_type="regression",
        per_feature_timeout_s=5.0,
        trace=True,
    )

    assert features == {"pymfe__nr_attr": 9.0}
    assert any(
        "skipping 1 classification-only feature(s) for problem_type=regression" in message for message in info_messages
    )


def test_pymfe_catalog_covers_default_groups() -> None:
    pymfe_mfe = pytest.importorskip("pymfe.mfe")
    MFE = pymfe_mfe.MFE
    classified = PYMFE_CLASSIFICATION_ONLY | PYMFE_REGRESSION_SAFE
    default_groups = (
        "general",
        "statistical",
        "info-theory",
        "model-based",
        "landmarking",
        "clustering",
        "concept",
        "itemset",
        "complexity",
    )
    for group in default_groups:
        enumerated = set(MFE.valid_metafeatures(groups=(group,)))
        uncategorized = enumerated - classified
        assert not uncategorized, (
            f"pymfe group `{group}` emits features not present in either catalog list: {sorted(uncategorized)}. "
            "Update PYMFE_CLASSIFICATION_ONLY or PYMFE_REGRESSION_SAFE in pymfe_catalog.py."
        )
