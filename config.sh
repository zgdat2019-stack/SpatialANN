#!/usr/bin/env bash
# Paths are relative to this repository by default. Environment variables override them.
export S56_BUNDLE="${S56_BUNDLE:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
export S56_PROJECT_ROOT="$S56_BUNDLE"
export S56_PYTHON="${S56_PYTHON:-${CONDA_PREFIX:+$CONDA_PREFIX/bin/python}}"
export S56_MAIN_RESULT="${S56_MAIN_RESULT:-$S56_PROJECT_ROOT/data}"
export S56_INPUT_DIR="${S56_INPUT_DIR:-$S56_PROJECT_ROOT/inputs/grid}"
export S56_STATION_CACHE="${S56_STATION_CACHE:-$S56_PROJECT_ROOT/inputs/station_cache}"
export S56_OUTPUT="${S56_OUTPUT:-$S56_PROJECT_ROOT/result}"
export S56_CACHE="${S56_CACHE:-$S56_PROJECT_ROOT/cache}"
export S56_LOG_DIR="${S56_LOG_DIR:-$S56_PROJECT_ROOT/logs}"

# Set the partition to a valid name on your SLURM cluster before submission.
export S56_PARTITION="${S56_PARTITION:-}"
export S56_PREP_PARTITION="${S56_PREP_PARTITION:-$S56_PARTITION}"
export S56_CPUS="${S56_CPUS:-8}"
export S56_MEMORY="${S56_MEMORY:-48G}"
export S56_PREP_MEMORY="${S56_PREP_MEMORY:-48G}"
export S56_CONCURRENCY="${S56_CONCURRENCY:-10}"
# 0 requests unlimited wall time; change to the time limit allowed by your cluster.
export S56_TIME_LIMIT="${S56_TIME_LIMIT:-0}"
export S56_EXCLUDE="${S56_EXCLUDE:-}"
export S56_DEVICE="${S56_DEVICE:-cpu}"
# For GPU inference, also choose a GPU partition and set S56_GRES=gpu:1.
export S56_GRES="${S56_GRES:-}"
export S56_TILE_SIZE="${S56_TILE_SIZE:-2048}"
export S56_INFERENCE_CHUNK="${S56_INFERENCE_CHUNK:-512}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="$S56_CPUS"
export MKL_NUM_THREADS="$S56_CPUS"
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
