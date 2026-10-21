#!/usr/bin/env bash
# One bounded health check; cron invokes this every twenty minutes.
set -euo pipefail
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-/home/user/anaconda3/envs/aris/bin/python}"
export PATH="/usr/local/bin:/usr/bin:/bin:/home/user/.local/bin:${PATH:-}"
export PYTHONPATH="$PROJECT_DIR/src:$PROJECT_DIR${PYTHONPATH:+:$PYTHONPATH}"
cd "$PROJECT_DIR"
# Keep the 30-second completion watcher alive across logout/reboot. A read-only
# invocation remains read-only; the watcher uses an independent singleton lock.
READ_ONLY=0
for ARG in "$@"; do
  if [[ "$ARG" == "--check-only" ]]; then READ_ONLY=1; fi
done
if [[ "$READ_ONLY" == 0 ]]; then
  mkdir -p "$PROJECT_DIR/artifacts/tosem_four_copies/experiments_20260930"
  "$PYTHON" -u -m dyrca_rca.experiments.auto_tosem_pipeline --ensure-running \
    >> "$PROJECT_DIR/artifacts/tosem_four_copies/experiments_20260930/health.log" 2>&1 || true
fi
exec "$PYTHON" -u -m dyrca_rca.collection_monitor "$@"
