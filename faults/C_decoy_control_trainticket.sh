#!/usr/bin/env bash
# TrainTicket wrapper for the shared counterfactual control collector.
set -euo pipefail
export NAMESPACE="${NAMESPACE:-trainticket}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$SCRIPT_DIR/C_decoy_control.sh" "$@"
