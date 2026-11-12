#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/config.sh"
cd "$S56_PROJECT_ROOT"
trap 'rc=$?; echo "EXIT rc=$rc job=${SLURM_JOB_ID:-local} task=${SLURM_ARRAY_TASK_ID:-none} node=$(hostname) time=$(date -Is)" >&2' EXIT
echo "START command=$* job=${SLURM_JOB_ID:-local} task=${SLURM_ARRAY_TASK_ID:-none} node=$(hostname) time=$(date -Is)"
args=("$S56_BUNDLE/product.py" "${1:?command required}" --main-result "$S56_MAIN_RESULT"
      --station-cache-dir "$S56_STATION_CACHE"
      --output-dir "$S56_OUTPUT" --cache-dir "$S56_CACHE" --device "$S56_DEVICE"
      --tile-size "$S56_TILE_SIZE" --inference-chunk "$S56_INFERENCE_CHUNK"
      --threads "${SLURM_CPUS_PER_TASK:-$S56_CPUS}")
[[ -z "$S56_INPUT_DIR" ]] || args+=(--input-dir "$S56_INPUT_DIR")
if [[ "$1" == run ]]; then
  args+=(--task-id "${SLURM_ARRAY_TASK_ID:?Missing task ID}")
fi
"$S56_PYTHON" -u "${args[@]}"
