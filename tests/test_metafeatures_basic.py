from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

from mfa.metafeatures import build_metafeature_table
from mfa.metafeatures.basic import compute_basic_metafeatures


def test_compute_basic_metafeatures_known_values() -> None:
    X = pd.DataFrame(
        {
            "num": [1.0, 2.0, np.nan, 4.0],
            "cat": pd.Series(["a", "b", "a", None], dtype="category"),
            "flag": [True, False, True, True],
        }
    )
    features = compute_basic_metafeatures(X)
    assert features["n"] == 4
    assert features["d"] == 3
    assert math.isclose(features["log_n"], math.log10(4))
    assert math.isclose(features["n_over_d"], 4 / 3)
    assert math.isclose(features["cat_fraction"], 2 / 3)
    assert math.isclose(features["missing_fraction"], (1 / 4 + 1 / 4 + 0) / 3)


def test_compute_basic_metafeatures_zero_columns() -> None:
    X = pd.DataFrame(index=range(5))
    features = compute_basic_metafeatures(X)
    assert features["n"] == 5
    assert features["d"] == 0
    assert np.isnan(features["n_over_d"])
    assert np.isnan(features["cat_fraction"])
    assert np.isnan(features["missing_fraction"])


def test_build_metafeature_table_reuses_split_cache_and_invalidates_on_version(
    monkeypatch,
    tmp_path: Path,
) -> None:
    metadata = pd.DataFrame(
        {
            "dataset": ["dataset_a"],
            "tid": [1],
            "n_repeats": [1],
            "n_folds": [1],
        }
    )
    call_counter = {"count": 0}

    class FakeTask:
        def get_split_dimensions(self) -> tuple[int, int, int]:
            return 1, 1, 1

        def get_train_test_split(self, *, fold: int, repeat: int):
            X = pd.DataFrame({"num": [1.0, 2.0, 3.0]})
            y = pd.Series([0, 1, 0])
            return X, y, X, y

    def fake_from_task_id(task_id: int) -> FakeTask:
        call_counter["count"] += 1
        return FakeTask()

    monkeypatch.setattr("tabarena.benchmark.task.openml.OpenMLTaskWrapper.from_task_id", fake_from_task_id)

    first = build_metafeature_table(metadata, cache_dir=tmp_path, use_cache=True, cache_version=1)
    assert call_counter["count"] == 1

    def fail_from_task_id(task_id: int) -> FakeTask:
        raise AssertionError("Split cache hit should avoid loading the OpenML task.")

    monkeypatch.setattr("tabarena.benchmark.task.openml.OpenMLTaskWrapper.from_task_id", fail_from_task_id)
    second = build_metafeature_table(metadata, cache_dir=tmp_path, use_cache=True, cache_version=1)
    pd.testing.assert_frame_equal(second, first)

    monkeypatch.setattr("tabarena.benchmark.task.openml.OpenMLTaskWrapper.from_task_id", fake_from_task_id)
    third = build_metafeature_table(metadata, cache_dir=tmp_path, use_cache=True, cache_version=2)
    assert call_counter["count"] == 2
    pd.testing.assert_frame_equal(third, first)


def test_build_metafeature_table_logs_dataset_progress(
    monkeypatch,
    tmp_path: Path,
) -> None:
    metadata = pd.DataFrame(
        {
            "dataset": ["dataset_a", "dataset_b"],
            "tid": [1, 2],
            "n_repeats": [1, 1],
            "n_folds": [1, 1],
        }
    )
    messages: list[str] = []
    perf_counter_values = iter([0.0, 1.2, 2.0, 3.6, 4.1, 6.3, 7.0, 9.2, 10.0, 12.4])

    class FakeTask:
        def get_split_dimensions(self) -> tuple[int, int, int]:
            return 1, 1, 1

        def get_train_test_split(self, *, fold: int, repeat: int):
            X = pd.DataFrame({"num": [1.0, 2.0, 3.0]})
            y = pd.Series([0, 1, 0])
            return X, y, X, y

    def fake_from_task_id(task_id: int) -> FakeTask:
        return FakeTask()

    def capture_info(message: str, *args) -> None:
        messages.append(message % args if args else message)

    monkeypatch.setattr("tabarena.benchmark.task.openml.OpenMLTaskWrapper.from_task_id", fake_from_task_id)
    monkeypatch.setattr("mfa.metafeatures.logger.info", capture_info)
    monkeypatch.setattr("mfa.metafeatures.time.perf_counter", lambda: next(perf_counter_values))

    build_metafeature_table(metadata, cache_dir=tmp_path, use_cache=False, cache_version=1)

    assert any("Meta-features [1/2] dataset_a: starting" in message for message in messages)
    assert any("Meta-features [1/2] dataset_a: done in 00:00:02" in message for message in messages)
    assert any("Meta-features [2/2] dataset_b: starting" in message for message in messages)
    assert any("Meta-features [2/2] dataset_b: done in 00:00:02" in message for message in messages)
    assert any("total elapsed 00:00:04" in message for message in messages)
    assert any("Meta-features: complete in 00:00:12" in message for message in messages)
