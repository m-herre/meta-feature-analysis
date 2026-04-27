from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

from mfa.cache import metafeature_split_cache_dir
from mfa.metafeatures import build_metafeature_table
from mfa.metafeatures.basic import compute_basic_metafeatures
from mfa.metafeatures.redundancy import compute_redundancy_metafeatures
from mfa.metafeatures.registry import extract_requested_metafeatures


def test_compute_basic_metafeatures_known_values() -> None:
    X = pd.DataFrame(
        {
            "num": [0.0, 1.0, 2.0, 100.0],
            "constant": [1.0, 1.0, 1.0, 1.0],
            "cat": pd.Series(["a", "b", "a", None], dtype="category"),
            "flag": [True, False, True, True],
        }
    )
    y = pd.Series([0, 0, 1, 1])
    features = compute_basic_metafeatures(X, y, problem_type="binary")
    assert features["n"] == 4
    assert features["d"] == 4
    assert math.isclose(features["log_n"], math.log10(4))
    assert math.isclose(features["log_d"], math.log10(4))
    assert math.isclose(features["n_over_d"], 1.0)
    assert math.isclose(features["d_over_n"], 1.0)
    assert features["n_num_features"] == 2
    assert features["n_cat_features"] == 2
    assert math.isclose(features["num_fraction"], 0.5)
    assert math.isclose(features["cat_fraction"], 0.5)
    assert math.isclose(features["missing_fraction"], 1 / 16)
    assert features["n_classes"] == 2
    assert math.isclose(features["class_entropy"], 1.0)
    assert math.isclose(features["majority_class_fraction"], 0.5)
    assert math.isclose(features["minority_class_fraction"], 0.5)
    assert math.isclose(features["class_imbalance_ratio"], 1.0)
    assert math.isclose(features["mean_cat_cardinality"], 2.0)
    assert math.isclose(features["max_cat_cardinality"], 2.0)
    assert math.isclose(features["high_cardinality_fraction"], 0.0)
    assert math.isclose(features["cat_cardinality_to_n_ratio"], 0.5)
    assert math.isclose(features["row_missing_fraction"], 0.25)
    assert math.isclose(features["feature_missing_fraction"], 0.25)
    assert math.isclose(features["num_missing_fraction"], 0.0)
    assert math.isclose(features["cat_missing_fraction"], 0.125)
    assert math.isclose(features["max_feature_missing_fraction"], 0.25)
    assert math.isclose(features["outlier_fraction_iqr"], 1 / 8)
    assert math.isclose(features["zero_fraction"], 1 / 8)
    assert math.isclose(features["constant_feature_fraction"], 0.25)
    assert math.isclose(features["near_constant_feature_fraction"], 0.25)
    assert "mean_abs_corr" not in features
    assert "effective_rank" not in features
    expected_numeric = X[["num", "constant"]]
    assert math.isclose(features["mean_abs_skew"], expected_numeric.skew().abs().mean())
    assert math.isclose(features["mean_kurtosis"], expected_numeric.kurt().mean())


def test_compute_basic_metafeatures_zero_columns() -> None:
    X = pd.DataFrame(index=range(5))
    features = compute_basic_metafeatures(X)
    assert features["n"] == 5
    assert features["d"] == 0
    assert np.isnan(features["log_d"])
    assert np.isnan(features["n_over_d"])
    assert np.isnan(features["cat_fraction"])
    assert np.isnan(features["missing_fraction"])
    assert np.isnan(features["mean_cat_cardinality"])
    assert np.isnan(features["row_missing_fraction"])
    assert np.isnan(features["constant_feature_fraction"])


def test_compute_basic_metafeatures_requires_classification_problem_type_for_target_features() -> None:
    X = pd.DataFrame({"num": np.arange(100, dtype=float)})
    y = pd.Series([0, 1] * 50)

    features = compute_basic_metafeatures(X, y)

    assert np.isnan(features["n_classes"])
    assert np.isnan(features["class_entropy"])
    assert np.isnan(features["majority_class_fraction"])
    assert np.isnan(features["minority_class_fraction"])
    assert np.isnan(features["class_imbalance_ratio"])


def test_compute_basic_metafeatures_treats_regression_problem_type_targets_as_missing() -> None:
    X = pd.DataFrame({"num": np.arange(100, dtype=float)})
    y = pd.Series(np.arange(100))

    features = compute_basic_metafeatures(X, y, problem_type="regression")

    assert np.isnan(features["n_classes"])
    assert np.isnan(features["class_entropy"])
    assert np.isnan(features["majority_class_fraction"])
    assert np.isnan(features["minority_class_fraction"])
    assert np.isnan(features["class_imbalance_ratio"])


def test_compute_redundancy_metafeatures_known_values() -> None:
    X = pd.DataFrame(
        {
            "a": [1.0, 2.0, 3.0, 4.0, 5.0],
            "b": [2.0, 4.0, 6.0, 8.0, 10.0],
            "c": [5.0, 4.0, 3.0, 2.0, 1.0],
        }
    )

    features = compute_redundancy_metafeatures(X)

    assert math.isclose(features["mean_abs_corr"], 1.0)
    assert math.isclose(features["max_abs_corr"], 1.0)
    assert math.isclose(features["high_corr_pair_fraction"], 1.0)
    assert math.isclose(features["effective_rank"], 1.0)
    assert math.isclose(features["participation_ratio"], 1.0)


def test_extract_requested_metafeatures_supports_redundancy_feature_set() -> None:
    X = pd.DataFrame(
        {
            "a": [1.0, 2.0, 3.0, 4.0],
            "b": [2.0, 4.0, 6.0, 8.0],
        }
    )

    features, failed_sets = extract_requested_metafeatures(
        X,
        pd.Series([0, 1, 0, 1]),
        feature_sets=("redundancy",),
        pymfe_groups=(),
        pymfe_summary=(),
    )

    assert failed_sets == {}
    assert "n" not in features
    assert math.isclose(features["mean_abs_corr"], 1.0)
    assert math.isclose(features["effective_rank"], 1.0)


def test_compute_redundancy_metafeatures_respects_width_cap() -> None:
    from mfa.metafeatures.redundancy import MAX_REDUNDANCY_NUMERIC_FEATURES

    X = pd.DataFrame(
        np.tile(np.arange(3, dtype=float).reshape(-1, 1), (1, MAX_REDUNDANCY_NUMERIC_FEATURES + 1)),
        columns=[f"x_{idx}" for idx in range(MAX_REDUNDANCY_NUMERIC_FEATURES + 1)],
    )

    features = compute_redundancy_metafeatures(X)

    assert np.isnan(features["mean_abs_corr"])
    assert np.isnan(features["effective_rank"])


def test_build_metafeature_table_uses_metadata_problem_type_for_target_features(
    monkeypatch,
    tmp_path: Path,
) -> None:
    metadata = pd.DataFrame(
        {
            "dataset": ["classification_dataset", "regression_dataset"],
            "tid": [1, 2],
            "problem_type": ["binary", "regression"],
            "n_repeats": [1, 1],
            "n_folds": [1, 1],
        }
    )

    class FakeTask:
        def __init__(self, task_id: int) -> None:
            self.task_id = task_id

        def get_split_dimensions(self) -> tuple[int, int, int]:
            return 1, 1, 1

        def get_train_test_split(self, *, fold: int, repeat: int):
            X = pd.DataFrame({"num": [1.0, 2.0, 3.0, 4.0]})
            y = pd.Series([0, 1, 0, 1])
            return X, y, X, y

    monkeypatch.setattr(
        "tabarena.benchmark.task.openml.OpenMLTaskWrapper.from_task_id",
        lambda task_id: FakeTask(task_id),
    )

    table = build_metafeature_table(
        metadata,
        cache_dir=tmp_path,
        use_cache=False,
        feature_sets=("basic",),
    )

    classification = table.loc[table["dataset"] == "classification_dataset"].iloc[0]
    regression = table.loc[table["dataset"] == "regression_dataset"].iloc[0]
    assert classification["n_classes"] == 2
    assert math.isclose(classification["class_entropy"], 1.0)
    assert np.isnan(regression["n_classes"])
    assert np.isnan(regression["class_entropy"])


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


def test_build_metafeature_table_invalidates_split_cache_on_basic_schema_change(
    monkeypatch,
    tmp_path: Path,
) -> None:
    import mfa.metafeatures as mfa_metafeatures

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

    first = build_metafeature_table(
        metadata,
        cache_dir=tmp_path,
        use_cache=True,
        cache_version=1,
        feature_sets=("basic",),
    )
    assert call_counter["count"] == 1

    def fail_from_task_id(task_id: int) -> FakeTask:
        raise AssertionError("Same schema should hit the split cache.")

    monkeypatch.setattr("tabarena.benchmark.task.openml.OpenMLTaskWrapper.from_task_id", fail_from_task_id)
    second = build_metafeature_table(
        metadata,
        cache_dir=tmp_path,
        use_cache=True,
        cache_version=1,
        feature_sets=("basic",),
    )
    pd.testing.assert_frame_equal(second, first)

    monkeypatch.setattr(mfa_metafeatures, "BASIC_METAFEATURE_SCHEMA_VERSION", 999)
    monkeypatch.setattr("tabarena.benchmark.task.openml.OpenMLTaskWrapper.from_task_id", fake_from_task_id)
    third = build_metafeature_table(
        metadata,
        cache_dir=tmp_path,
        use_cache=True,
        cache_version=1,
        feature_sets=("basic",),
    )
    assert call_counter["count"] == 2
    pd.testing.assert_frame_equal(third, first)


def test_build_metafeature_table_trace_reuses_split_cache(
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

    first = build_metafeature_table(metadata, cache_dir=tmp_path, use_cache=True, cache_version=1, trace=True)
    assert call_counter["count"] == 1

    def fail_from_task_id(task_id: int) -> FakeTask:
        raise AssertionError("Trace mode should still reuse the split cache.")

    monkeypatch.setattr("tabarena.benchmark.task.openml.OpenMLTaskWrapper.from_task_id", fail_from_task_id)
    traced = build_metafeature_table(metadata, cache_dir=tmp_path, use_cache=True, cache_version=1, trace=True)

    pd.testing.assert_frame_equal(traced, first)


def test_build_metafeature_table_recomputes_corrupt_split_cache(
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
    task_sizes = iter([3, 4])

    class FakeTask:
        def __init__(self, n_rows: int) -> None:
            self.n_rows = n_rows

        def get_split_dimensions(self) -> tuple[int, int, int]:
            return 1, 1, 1

        def get_train_test_split(self, *, fold: int, repeat: int):
            X = pd.DataFrame({"num": np.arange(self.n_rows, dtype=float)})
            y = pd.Series(np.zeros(self.n_rows, dtype=int))
            return X, y, X, y

    monkeypatch.setattr(
        "tabarena.benchmark.task.openml.OpenMLTaskWrapper.from_task_id",
        lambda task_id: FakeTask(next(task_sizes)),
    )

    first = build_metafeature_table(
        metadata,
        cache_dir=tmp_path,
        use_cache=True,
        cache_version=1,
        feature_sets=("basic",),
    )
    assert first["n"].iat[0] == 3

    split_path = metafeature_split_cache_dir(tmp_path) / "dataset_a__r0__f0.parquet"
    split_path.write_text("not a parquet file", encoding="utf-8")
    warnings: list[str] = []
    monkeypatch.setattr(
        "mfa.metafeatures.logger.warning",
        lambda message, *args: warnings.append(message % args if args else message),
    )

    second = build_metafeature_table(
        metadata,
        cache_dir=tmp_path,
        use_cache=True,
        cache_version=1,
        feature_sets=("basic",),
    )

    assert second["n"].iat[0] == 4
    assert any("split cache" in message and "unreadable" in message for message in warnings)


def test_retry_failed_pymfe_repairs_failed_basic_split_cache(monkeypatch, tmp_path: Path) -> None:
    import mfa.metafeatures.registry as registry
    from mfa.metafeatures.basic import compute_basic_metafeatures as real_compute_basic_metafeatures

    metadata = pd.DataFrame({"dataset": ["dataset_a"], "tid": [1], "n_repeats": [1], "n_folds": [1]})

    class FakeTask:
        def get_split_dimensions(self) -> tuple[int, int, int]:
            return 1, 1, 1

        def get_train_test_split(self, *, fold: int, repeat: int):
            X = pd.DataFrame({"num": [1.0, 2.0, 3.0]})
            y = pd.Series([0, 1, 0])
            return X, y, X, y

    monkeypatch.setattr("tabarena.benchmark.task.openml.OpenMLTaskWrapper.from_task_id", lambda task_id: FakeTask())

    def exploding_basic(_X_train, _y_train=None, problem_type=None):
        raise ValueError("deliberate basic failure")

    monkeypatch.setattr(registry, "compute_basic_metafeatures", exploding_basic)
    first = build_metafeature_table(
        metadata,
        cache_dir=tmp_path,
        use_cache=True,
        feature_sets=("basic",),
        n_jobs=1,
    )
    assert np.isnan(first["n"].iat[0])

    split_path = metafeature_split_cache_dir(tmp_path) / "dataset_a__r0__f0.parquet"
    failed_cache = pd.read_parquet(split_path)
    assert '"basic"' in failed_cache["_cache_failed_sets"].iat[0]
    assert np.isnan(failed_cache["n"].iat[0])

    monkeypatch.setattr(registry, "compute_basic_metafeatures", real_compute_basic_metafeatures)
    repaired = build_metafeature_table(
        metadata,
        cache_dir=tmp_path,
        use_cache=True,
        feature_sets=("basic",),
        retry_failed_pymfe=True,
        n_jobs=1,
    )

    assert repaired["n"].iat[0] == 3
    rewritten = pd.read_parquet(split_path)
    assert rewritten["n"].iat[0] == 3
    assert rewritten["_cache_failed_sets"].iat[0] == "{}"


def test_retry_failed_pymfe_repairs_failed_redundancy_split_cache(monkeypatch, tmp_path: Path) -> None:
    import mfa.metafeatures.registry as registry
    from mfa.metafeatures.redundancy import compute_redundancy_metafeatures as real_compute_redundancy_metafeatures

    metadata = pd.DataFrame({"dataset": ["dataset_a"], "tid": [1], "n_repeats": [1], "n_folds": [1]})

    class FakeTask:
        def get_split_dimensions(self) -> tuple[int, int, int]:
            return 1, 1, 1

        def get_train_test_split(self, *, fold: int, repeat: int):
            X = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0], "b": [2.0, 4.0, 6.0, 8.0]})
            y = pd.Series([0, 1, 0, 1])
            return X, y, X, y

    monkeypatch.setattr("tabarena.benchmark.task.openml.OpenMLTaskWrapper.from_task_id", lambda task_id: FakeTask())

    def exploding_redundancy(_X_num):
        raise RuntimeError("deliberate redundancy failure")

    monkeypatch.setattr(registry, "compute_redundancy_metafeatures", exploding_redundancy)
    first = build_metafeature_table(
        metadata,
        cache_dir=tmp_path,
        use_cache=True,
        feature_sets=("basic", "redundancy"),
        n_jobs=1,
    )
    assert np.isnan(first["mean_abs_corr"].iat[0])
    assert first["n"].iat[0] == 4

    split_path = metafeature_split_cache_dir(tmp_path) / "dataset_a__r0__f0.parquet"
    failed_cache = pd.read_parquet(split_path)
    assert '"redundancy"' in failed_cache["_cache_failed_sets"].iat[0]
    assert np.isnan(failed_cache["mean_abs_corr"].iat[0])

    monkeypatch.setattr(registry, "compute_redundancy_metafeatures", real_compute_redundancy_metafeatures)
    repaired = build_metafeature_table(
        metadata,
        cache_dir=tmp_path,
        use_cache=True,
        feature_sets=("basic", "redundancy"),
        retry_failed_pymfe=True,
        n_jobs=1,
    )

    assert repaired["mean_abs_corr"].iat[0] == 1.0
    rewritten = pd.read_parquet(split_path)
    assert rewritten["mean_abs_corr"].iat[0] == 1.0
    assert rewritten["_cache_failed_sets"].iat[0] == "{}"


def test_missing_basic_split_cache_column_repairs_only_when_retry_enabled(monkeypatch, tmp_path: Path) -> None:
    metadata = pd.DataFrame({"dataset": ["dataset_a"], "tid": [1], "n_repeats": [1], "n_folds": [1]})

    class FakeTask:
        def get_split_dimensions(self) -> tuple[int, int, int]:
            return 1, 1, 1

        def get_train_test_split(self, *, fold: int, repeat: int):
            X = pd.DataFrame({"num": [1.0, 2.0, 3.0]})
            y = pd.Series([0, 1, 0])
            return X, y, X, y

    monkeypatch.setattr("tabarena.benchmark.task.openml.OpenMLTaskWrapper.from_task_id", lambda task_id: FakeTask())
    build_metafeature_table(
        metadata,
        cache_dir=tmp_path,
        use_cache=True,
        feature_sets=("basic",),
        n_jobs=1,
    )

    split_path = metafeature_split_cache_dir(tmp_path) / "dataset_a__r0__f0.parquet"
    cached = pd.read_parquet(split_path).drop(columns=["log_n"])
    cached.to_parquet(split_path, index=False)

    def fail_from_task_id(task_id: int):
        raise AssertionError("incomplete basic split cache should not load the task unless retry is enabled")

    monkeypatch.setattr("tabarena.benchmark.task.openml.OpenMLTaskWrapper.from_task_id", fail_from_task_id)
    reused = build_metafeature_table(
        metadata,
        cache_dir=tmp_path,
        use_cache=True,
        feature_sets=("basic",),
        n_jobs=1,
    )
    assert "log_n" not in reused.columns

    monkeypatch.setattr("tabarena.benchmark.task.openml.OpenMLTaskWrapper.from_task_id", lambda task_id: FakeTask())
    repaired = build_metafeature_table(
        metadata,
        cache_dir=tmp_path,
        use_cache=True,
        feature_sets=("basic",),
        retry_failed_pymfe=True,
        n_jobs=1,
    )
    assert math.isclose(repaired["log_n"].iat[0], math.log10(3))
    assert "log_n" in pd.read_parquet(split_path).columns


def test_existing_basic_nan_split_cache_cell_is_not_repaired(monkeypatch, tmp_path: Path) -> None:
    metadata = pd.DataFrame(
        {
            "dataset": ["regression_dataset"],
            "tid": [1],
            "problem_type": ["regression"],
            "n_repeats": [1],
            "n_folds": [1],
        }
    )

    class FakeTask:
        def get_split_dimensions(self) -> tuple[int, int, int]:
            return 1, 1, 1

        def get_train_test_split(self, *, fold: int, repeat: int):
            X = pd.DataFrame({"num": [1.0, 2.0, 3.0]})
            y = pd.Series([1.0, 2.0, 3.0])
            return X, y, X, y

    monkeypatch.setattr("tabarena.benchmark.task.openml.OpenMLTaskWrapper.from_task_id", lambda task_id: FakeTask())
    first = build_metafeature_table(
        metadata,
        cache_dir=tmp_path,
        use_cache=True,
        feature_sets=("basic",),
        n_jobs=1,
    )
    assert np.isnan(first["n_classes"].iat[0])

    def fail_from_task_id(task_id: int):
        raise AssertionError("existing basic NaN values should remain cache hits")

    monkeypatch.setattr("tabarena.benchmark.task.openml.OpenMLTaskWrapper.from_task_id", fail_from_task_id)
    second = build_metafeature_table(
        metadata,
        cache_dir=tmp_path,
        use_cache=True,
        feature_sets=("basic",),
        retry_failed_pymfe=True,
        n_jobs=1,
    )

    pd.testing.assert_frame_equal(second, first)


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


class _ImmediateExecutor:
    """Executes submissions inline, optionally raising BrokenProcessPool on marked units.

    `break_on` contains (dataset, repeat, fold) keys whose future will carry a
    BrokenProcessPool exception, mirroring what `ProcessPoolExecutor` does to
    every pending future after a worker dies.
    """

    def __init__(self, break_on: set[tuple[str, int, int]] | None = None) -> None:
        self.break_on = break_on or set()
        self.submitted_keys: list[tuple[str, int, int]] = []
        self.shutdown_calls: list[dict] = []

    def submit(self, fn, *args, **kwargs):
        from concurrent.futures import Future
        from concurrent.futures.process import BrokenProcessPool

        dataset, _task_id, repeat, fold = args[0], args[1], args[2], args[3]
        key = (dataset, repeat, fold)
        self.submitted_keys.append(key)
        future: Future = Future()
        if key in self.break_on:
            future.set_exception(BrokenProcessPool(f"simulated for {key}"))
        else:
            try:
                future.set_result(fn(*args, **kwargs))
            except Exception as exc:
                future.set_exception(exc)
        return future

    def shutdown(self, *, wait: bool = True, cancel_futures: bool = False) -> None:
        self.shutdown_calls.append({"wait": wait, "cancel_futures": cancel_futures})


def _install_fake_openml(monkeypatch) -> None:
    class FakeTask:
        def get_split_dimensions(self) -> tuple[int, int, int]:
            return 1, 3, 1

        def get_train_test_split(self, *, fold: int, repeat: int):
            X = pd.DataFrame({"num": [1.0, 2.0, 3.0, 4.0]})
            y = pd.Series([0, 1, 0, 1])
            return X, y, X, y

    monkeypatch.setattr(
        "tabarena.benchmark.task.openml.OpenMLTaskWrapper.from_task_id",
        lambda task_id: FakeTask(),
    )


def test_build_metafeature_table_retries_broken_process_pool(monkeypatch, tmp_path: Path) -> None:
    _install_fake_openml(monkeypatch)

    metadata = pd.DataFrame(
        {
            "dataset": ["dataset_a"],
            "tid": [1],
            "n_repeats": [1],
            "n_folds": [3],
        }
    )

    executors = [
        _ImmediateExecutor(break_on={("dataset_a", 0, 1)}),
        _ImmediateExecutor(),
    ]
    spawned: list[_ImmediateExecutor] = []

    def fake_get_executor(backend: str, max_workers: int) -> _ImmediateExecutor:
        exec_ = executors.pop(0)
        spawned.append(exec_)
        return exec_

    monkeypatch.setattr("mfa.metafeatures.get_executor", fake_get_executor)
    # Force deterministic completion order so the "retry only drops un-harvested
    # splits" assertion below is stable across test runs.
    monkeypatch.setattr("mfa.metafeatures.as_completed", lambda futures: list(futures))

    table = build_metafeature_table(
        metadata,
        cache_dir=tmp_path,
        use_cache=True,
        cache_version=1,
        n_jobs=2,
        feature_sets=("basic",),
    )

    assert len(table) == 3
    assert set(zip(table["repeat"], table["fold"], strict=True)) == {(0, 0), (0, 1), (0, 2)}
    assert len(spawned) == 2, "pool should be rebuilt exactly once after a break"
    # First pool submits all three splits; second pool retries only fold 1
    # (broken) and fold 2 (never harvested because the break short-circuited
    # iteration). fold 0 completed before the break, so must not be retried.
    assert spawned[0].submitted_keys == [("dataset_a", 0, 0), ("dataset_a", 0, 1), ("dataset_a", 0, 2)]
    assert spawned[1].submitted_keys == [("dataset_a", 0, 1), ("dataset_a", 0, 2)]
    assert spawned[0].shutdown_calls[0]["cancel_futures"] is True


def test_build_metafeature_table_falls_back_to_sequential_after_repeated_pool_breaks(
    monkeypatch, tmp_path: Path
) -> None:
    _install_fake_openml(monkeypatch)

    metadata = pd.DataFrame(
        {
            "dataset": ["dataset_a"],
            "tid": [1],
            "n_repeats": [1],
            "n_folds": [3],
        }
    )

    # Both pools break on the same split; the third attempt must be sequential.
    executors = [
        _ImmediateExecutor(break_on={("dataset_a", 0, 2)}),
        _ImmediateExecutor(break_on={("dataset_a", 0, 2)}),
    ]
    spawned: list[_ImmediateExecutor] = []

    def fake_get_executor(backend: str, max_workers: int) -> _ImmediateExecutor:
        if not executors:
            raise AssertionError("fallback must not spawn a third pool")
        exec_ = executors.pop(0)
        spawned.append(exec_)
        return exec_

    monkeypatch.setattr("mfa.metafeatures.get_executor", fake_get_executor)

    table = build_metafeature_table(
        metadata,
        cache_dir=tmp_path,
        use_cache=True,
        cache_version=1,
        n_jobs=2,
        feature_sets=("basic",),
    )

    assert len(table) == 3
    assert len(spawned) == 2


def test_single_slot_task_cache_evicts_on_new_id(monkeypatch) -> None:
    """_get_cached_task must release the previous wrapper before loading a new one.

    Otherwise an unbounded dict grows per-worker and OOMs the run
    (see job 222410, MaxRSS=896G on a 900G allocation).
    """
    from tabarena.benchmark.task.openml import OpenMLTaskWrapper

    import mfa.metafeatures as mfa_metafeatures

    loads: list[int] = []

    class FakeWrapper:
        def __init__(self, task_id: int) -> None:
            loads.append(task_id)
            self.task_id = task_id

        def get_split_dimensions(self) -> tuple[int, int, int]:
            return (1, 1, 1)

    monkeypatch.setattr(
        OpenMLTaskWrapper,
        "from_task_id",
        classmethod(lambda cls, task_id: FakeWrapper(task_id)),
    )

    mfa_metafeatures._CACHED_TASK_ID = None
    mfa_metafeatures._CACHED_TASK = None

    a1 = mfa_metafeatures._get_cached_task(1)
    a2 = mfa_metafeatures._get_cached_task(1)
    b1 = mfa_metafeatures._get_cached_task(2)
    a3 = mfa_metafeatures._get_cached_task(1)

    assert a1 is a2
    assert a1 is not b1
    assert a1 is not a3
    assert loads == [1, 2, 1]
    assert mfa_metafeatures._CACHED_TASK_ID == 1
