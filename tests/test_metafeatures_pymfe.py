from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd

from mfa.metafeatures import build_metafeature_table
from mfa.metafeatures.pymfe_features import extract_pymfe_features
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
