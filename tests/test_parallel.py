from __future__ import annotations

import os

import pytest

from mfa.parallel import get_executor, resolve_n_jobs


def _square(value: int) -> int:
    return value * value


def test_resolve_n_jobs_positive() -> None:
    assert resolve_n_jobs(4) == 4
    assert resolve_n_jobs(1) == 1


def test_resolve_n_jobs_zero_returns_cpu_count() -> None:
    assert resolve_n_jobs(0) == (os.cpu_count() or 1)


def test_resolve_n_jobs_negative_returns_cpu_count() -> None:
    result = resolve_n_jobs(-1)
    assert result == (os.cpu_count() or 1)
    assert result >= 1


@pytest.mark.parametrize("backend", ["thread", "process"])
def test_get_executor_runs_top_level_worker(backend: str) -> None:
    with get_executor(backend, max_workers=2) as executor:
        results = list(executor.map(_square, [1, 2, 3]))

    assert results == [1, 4, 9]
