#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/config.sh"
cd "$S56_PROJECT_ROOT"
[[ $# -eq 0 ]] || { echo 'No positional arguments; configure config.sh or environment variables.' >&2; exit 2; }
[[ -n "$S56_PYTHON" && -x "$S56_PYTHON" ]] || { echo 'Activate your Python environment or set S56_PYTHON to its absolute python path.' >&2; exit 2; }
[[ -n "$S56_PARTITION" && -n "$S56_PREP_PARTITION" ]] || { echo 'Set S56_PARTITION and, if needed, S56_PREP_PARTITION for your SLURM cluster.' >&2; exit 2; }
[[ "$S56_CONCURRENCY" =~ ^([1-9]|10)$ ]] || { echo 'S56_CONCURRENCY must be 1..10' >&2; exit 2; }
command -v sbatch >/dev/null
mkdir -p "$S56_LOG_DIR"
bash "$S56_BUNDLE/run_worker.sh" preflight
common=(--parsable --export=ALL --nodes=1 --ntasks=1 --cpus-per-task="$S56_CPUS" --time="$S56_TIME_LIMIT"
        --chdir="$S56_PROJECT_ROOT")
[[ -z "$S56_EXCLUDE" ]] || common+=(--exclude="$S56_EXCLUDE")
prep=$(sbatch "${common[@]}" --job-name=s56_prepare --partition="$S56_PREP_PARTITION" --mem="$S56_PREP_MEMORY" \
       --output="$S56_LOG_DIR/prepare_%j.out" --error="$S56_LOG_DIR/prepare_%j.err" \
       "$S56_BUNDLE/run_worker.sh" prepare)
prep=${prep%%;*}
echo "Preparation job submitted: $prep"
printf 'PREP_JOB=%s\n' "$prep" > "$S56_LOG_DIR/last_submission.env"
# Always expand a NONEMPTY array: Bash 4.2/4.3 + nounset rejects empty arrays.
product_args=("${common[@]}" --job-name=s56_product --partition="$S56_PARTITION" --mem="$S56_MEMORY"
        --array="1-10%$S56_CONCURRENCY" --dependency="afterok:$prep"
        --output="$S56_LOG_DIR/product_%A_%a.out" --error="$S56_LOG_DIR/product_%A_%a.err")
[[ -z "$S56_GRES" ]] || product_args+=(--gres="$S56_GRES")
array=$(sbatch "${product_args[@]}" "$S56_BUNDLE/run_worker.sh" run)
array=${array%%;*}
printf 'Product array submitted: %s (10 tasks, maximum concurrent=%s)\n' "$array" "$S56_CONCURRENCY"
printf 'PRODUCT_JOB=%s\n' "$array" >> "$S56_LOG_DIR/last_submission.env"
echo "Output: $S56_OUTPUT"
echo "Logs: $S56_LOG_DIR"
echo 'If preparation fails, cancel its waiting product array before resubmitting; see README.md.'
