#!/bin/bash
# ============================================================
# run_verify.sh — 验证单次实验的拓扑变化
# ============================================================
# Usage:
#   ./scripts/run_verify.sh data/experiments/T1_xxx_run001
#   ./scripts/run_verify.sh data/experiments/T1_xxx_run001 --pretty
# ============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
PYTHON="/home/user/anaconda3/envs/aris/bin/python"

EXP_DIR="${1:?Usage: $0 <experiment_dir> [--pretty]}"
shift || true

echo "╔══════════════════════════════════════════════╗"
echo "║  Topology Change Verification               ║"
echo "╚══════════════════════════════════════════════╝"
echo "  Experiment: $EXP_DIR"
echo ""

cd "$PROJECT_DIR"

$PYTHON -m verifier.topology_verifier \
    --experiment-dir "$EXP_DIR" \
    "$@"

echo ""
echo "Output files:"
echo "  $EXP_DIR/topology_diff.json"
echo "  $EXP_DIR/verification_report.json"
