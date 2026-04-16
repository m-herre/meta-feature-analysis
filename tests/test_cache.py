from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from mfa.cache import (
    compute_config_hash,
    invalidate_downstream,
    metafeature_split_cache_dir,
    read_dataframe_cache,
    read_json_cache,
    write_dataframe_cache,
    write_json_cache,
)


def test_compute_config_hash_is_order_insensitive() -> None:
    left = compute_config_hash({"a": 1, "b": [1, 2]})
    right = compute_config_hash({"b": [1, 2], "a": 1})
    assert left == right


def test_dataframe_cache_round_trip(tmp_path: Path) -> None:
    frame = pd.DataFrame({"x": [1, 2], "y": [3, 4]})
    try:
        write_dataframe_cache(frame, tmp_path, 1, "raw_results", "hash123")
    except ImportError:
        pytest.skip("No parquet engine is installed.")
    loaded = read_dataframe_cache(tmp_path, 1, "raw_results", "hash123")
    pd.testing.assert_frame_equal(loaded, frame)


def test_json_cache_round_trip(tmp_path: Path) -> None:
    payload = {"alpha": 1, "beta": [1, 2, 3]}
    write_json_cache(payload, tmp_path, 5, "statistics", "hash456")
    assert read_json_cache(tmp_path, 5, "statistics", "hash456") == payload


def test_invalidate_downstream(tmp_path: Path) -> None:
    (tmp_path / "stage1_raw_results").mkdir()
    (tmp_path / "stage3_gaps").mkdir()
    (tmp_path / "stage5_statistics").mkdir()
    invalidate_downstream(tmp_path, from_stage=3)
    assert (tmp_path / "stage1_raw_results").exists()
    assert not (tmp_path / "stage3_gaps").exists()
    assert not (tmp_path / "stage5_statistics").exists()


def test_invalidate_downstream_clears_split_metafeature_cache(tmp_path: Path) -> None:
    split_dir = metafeature_split_cache_dir(tmp_path)
    split_dir.mkdir(parents=True)
    (split_dir / "dataset_a__r0__f0.parquet").write_text("placeholder", encoding="utf-8")
    invalidate_downstream(tmp_path, from_stage=2)
    assert not split_dir.exists()
