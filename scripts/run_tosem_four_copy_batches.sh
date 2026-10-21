#!/usr/bin/env bash
# Launch protected, staged four-copy collection.
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
KUBECONFIG_PATH="${DYRCA_COPY_KUBECONFIG:-${PROJECT_DIR}/artifacts/tosem_four_copies/kubeconfig}"
PYTHON="${PYTHON:-/home/user/anaconda3/envs/aris/bin/python}"

export KUBECONFIG="$KUBECONFIG_PATH"
export PYTHONPATH="$PROJECT_DIR/src:$PROJECT_DIR${PYTHONPATH:+:$PYTHONPATH}"
export DYRCA_CLUSTER="kind-tosem-copies"

cd "$PROJECT_DIR"
exec "$PYTHON" -u -m dyrca_rca.four_copy_collection --kubeconfig "$KUBECONFIG_PATH" "$@"
