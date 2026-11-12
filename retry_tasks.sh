#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/config.sh"
cd "$S56_PROJECT_ROOT"
[[ $# -eq 1 && "$1" =~ ^[0-9,-]+$ ]] || { echo 'Usage: bash retry_tasks.sh 3,7' >&2; exit 2; }
[[ -n "$S56_PYTHON" && -x "$S56_PYTHON" ]] || { echo 'Activate the same Python environment first' >&2; exit 2; }
[[ -n "$S56_PARTITION" ]] || { echo 'Set S56_PARTITION for your SLURM cluster.' >&2; exit 2; }
mkdir -p "$S56_LOG_DIR"
args=(--export=ALL --job-name=s56_product --nodes=1 --ntasks=1 --time="$S56_TIME_LIMIT"
      --partition="$S56_PARTITION" --cpus-per-task="$S56_CPUS" --mem="$S56_MEMORY"
      --chdir="$S56_PROJECT_ROOT" --array="$1%$S56_CONCURRENCY"
      --output="$S56_LOG_DIR/product_%A_%a.out" --error="$S56_LOG_DIR/product_%A_%a.err")
[[ -z "$S56_EXCLUDE" ]] || args+=(--exclude="$S56_EXCLUDE")
[[ -z "$S56_GRES" ]] || args+=(--gres="$S56_GRES")
sbatch "${args[@]}" "$S56_BUNDLE/run_worker.sh" run
