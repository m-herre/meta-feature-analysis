"""Run `run_analysis` to populate the `.mfa_cache/` directory.

Designed to be launched from an sbatch job: it does the heavy compute once
so that subsequent notebook runs of `run_analysis(config)` hit the cache
and return in seconds.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path


def _setup_sys_path() -> Path:
    script_dir = Path(__file__).resolve().parent
    project_dir = script_dir.parent
    repo_root = project_dir.parent
    sys.path.insert(0, str(project_dir / "src"))
    sys.path.insert(0, str(repo_root / "tabarena" / "tabarena"))
    return project_dir


def main() -> int:
    project_dir = _setup_sys_path()

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=project_dir / "configs" / "config_0.yaml",
        help="Path to the analysis YAML config.",
    )
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=None,
        help="Optional subset of dataset names (for smoke tests).",
    )
    args = parser.parse_args()

    from dataclasses import replace

    from mfa import load_config, run_analysis
    from mfa.parallel import resolve_n_jobs

    config = load_config(args.config)
    # resolve_n_jobs(-1) uses os.cpu_count(), which on a shared cluster node
    # returns the node's physical CPU count (e.g. 40) rather than the cgroup
    # allocation. Clamp to SLURM_CPUS_PER_TASK so we don't oversubscribe.
    slurm_cpus_env = os.environ.get("SLURM_CPUS_PER_TASK")
    slurm_cpus = int(slurm_cpus_env) if slurm_cpus_env else None
    resolved = resolve_n_jobs(config.parallelism.n_jobs)
    if slurm_cpus is not None and resolved > slurm_cpus:
        resolved = slurm_cpus
        config = replace(
            config,
            parallelism=replace(config.parallelism, n_jobs=slurm_cpus),
        )
    print(
        f"[warm_cache] config={args.config.name} "
        f"n_jobs(resolved)={resolved} "
        f"SLURM_CPUS_PER_TASK={slurm_cpus_env} "
        f"backend={config.parallelism.backend}",
        flush=True,
    )
    if args.datasets:
        print(f"[warm_cache] dataset subset: {args.datasets}", flush=True)

    start = time.perf_counter()
    result = run_analysis(config, datasets=args.datasets)
    elapsed = time.perf_counter() - start

    print(
        f"[warm_cache] done in {elapsed:.1f}s "
        f"config_hash={result.config_hash} "
        f"analysis_table={result.analysis_table.shape} "
        f"metafeature_table={result.metafeature_table.shape} "
        f"gap_table={result.gap_table.shape} "
        f"correlation_tests={len(result.correlation_results)}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
