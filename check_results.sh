#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/config.sh"
cd "$S56_PROJECT_ROOT"
bash "$S56_BUNDLE/run_worker.sh" check
