from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

VALID_BACKENDS = {"process", "thread"}


def resolve_n_jobs(n_jobs: int) -> int:
    """Resolve n_jobs: -1 -> os.cpu_count(), 0 or 1 -> 1, N -> N."""
    if n_jobs <= 0:
        return os.cpu_count() or 1
    return n_jobs


def get_executor(backend: str, max_workers: int) -> ProcessPoolExecutor | ThreadPoolExecutor:
    """Return an executor matching the requested backend."""
    if backend == "thread":
        return ThreadPoolExecutor(max_workers=max_workers)
    return ProcessPoolExecutor(max_workers=max_workers)
