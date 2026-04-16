from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import pandas as pd


def _to_serializable(value: Any) -> Any:
    if is_dataclass(value):
        return _to_serializable(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "value") and isinstance(value.value, str):
        return value.value
    if isinstance(value, dict):
        return {str(key): _to_serializable(val) for key, val in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, (list, tuple)):
        return [_to_serializable(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return sorted(_to_serializable(item) for item in value)
    return value


def compute_config_hash(config: Any) -> str:
    """Compute a deterministic short hash for a config-like object."""
    serializable = _to_serializable(config)
    payload = json.dumps(serializable, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def compute_stage_hash(stage_name: str, upstream_hash: str, params: Any = None) -> str:
    """Compute a deterministic short hash for a pipeline stage."""
    return compute_config_hash(
        {
            "stage_name": stage_name,
            "upstream_hash": upstream_hash,
            "params": params,
        }
    )


def stage_cache_path(cache_dir: str | Path, stage: int, name: str, cache_hash: str, suffix: str) -> Path:
    cache_root = Path(cache_dir)
    return cache_root / f"stage{stage}_{name}" / f"{cache_hash}.{suffix}"


def metafeature_split_cache_dir(cache_dir: str | Path) -> Path:
    cache_root = Path(cache_dir)
    return cache_root / "metafeatures" / "splits"


def write_dataframe_cache(df: pd.DataFrame, cache_dir: str | Path, stage: int, name: str, cache_hash: str) -> Path:
    """Persist a DataFrame cache entry as parquet."""
    path = stage_cache_path(cache_dir, stage, name, cache_hash, "parquet")
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return path


def read_dataframe_cache(cache_dir: str | Path, stage: int, name: str, cache_hash: str) -> pd.DataFrame | None:
    """Load a cached DataFrame when it exists."""
    path = stage_cache_path(cache_dir, stage, name, cache_hash, "parquet")
    if not path.exists():
        return None
    return pd.read_parquet(path)


def write_json_cache(payload: Any, cache_dir: str | Path, stage: int, name: str, cache_hash: str) -> Path:
    """Persist a JSON-serializable cache entry."""
    path = stage_cache_path(cache_dir, stage, name, cache_hash, "json")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as outfile:
        json.dump(_to_serializable(payload), outfile, sort_keys=True)
    return path


def read_json_cache(cache_dir: str | Path, stage: int, name: str, cache_hash: str) -> Any:
    """Load a cached JSON payload when it exists."""
    path = stage_cache_path(cache_dir, stage, name, cache_hash, "json")
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as infile:
        return json.load(infile)


def invalidate_downstream(cache_dir: str | Path, from_stage: int) -> None:
    """Delete cache directories for stage N and above."""
    cache_root = Path(cache_dir)
    if not cache_root.exists():
        return
    for path in cache_root.iterdir():
        if not path.is_dir():
            continue
        prefix = path.name.split("_", maxsplit=1)[0]
        if not prefix.startswith("stage"):
            continue
        stage_number = int(prefix.removeprefix("stage"))
        if stage_number >= from_stage:
            shutil.rmtree(path)
    if from_stage <= 2:
        split_cache_dir = metafeature_split_cache_dir(cache_root)
        if split_cache_dir.exists():
            shutil.rmtree(split_cache_dir)
