#!/usr/bin/env bash
#SBATCH --job-name=mfa-warm
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=48
#SBATCH --mem=750G
#SBATCH --output=logs/warm_cache_%j.out
#SBATCH --error=logs/warm_cache_%j.err

# Populate the .mfa_cache/ directory by running the full pipeline once on a
# beefy cpu node. Notebooks can then re-run `run_analysis(config)` and hit
# every stage cache instantly.
#
# Submit:
#   cd meta-feature-analysis
#   mkdir -p logs
#   sbatch scripts/warm_cache.sh                               # full run
#   sbatch scripts/warm_cache.sh --datasets qsar-biodeg Fitness_Club  # smoke test
#
# Monitor:
#   squeue -u $USER
#   tail -f logs/warm_cache_<jobid>.out

set -euo pipefail

# SLURM copies the batch script to /var/spool/slurmd/... so BASH_SOURCE is
# useless for locating the project. Use SLURM_SUBMIT_DIR (or PWD when run
# outside slurm) and assume sbatch was invoked from the project root.
PROJECT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
REPO_ROOT="$(cd "${PROJECT_DIR}/.." && pwd)"
VENV_PY="${REPO_ROOT}/tabarena/.venv/bin/python"

# Keep numeric libraries from oversubscribing when n_jobs worker processes
# each spawn their own BLAS thread pool. One BLAS thread per worker is the
# right default; the process pool provides the outer parallelism.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

cd "${PROJECT_DIR}"

echo "[warm_cache.sh] node=${SLURMD_NODENAME:-$(hostname)} job=${SLURM_JOB_ID:-no-slurm} cpus=${SLURM_CPUS_PER_TASK:-?}"
echo "[warm_cache.sh] python=${VENV_PY}"
echo "[warm_cache.sh] pwd=$(pwd)"

exec "${VENV_PY}" scripts/warm_cache.py "$@"
