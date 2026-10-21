#!/bin/bash
# ============================================================
# run_batch.sh — 一键启动批量实验
# ============================================================
# Usage:
#   ./scripts/run_batch.sh                                # 全部场景
#   ./scripts/run_batch.sh --scenarios T1,T3              # 仅 T1+T3
#   ./scripts/run_batch.sh --dry-run                      # 预览计划
#   ./scripts/run_batch.sh --repeats 1 --skip-export      # 快速测试
#   ./scripts/run_batch.sh --config config/my_config.yaml # 自定义配置
# ============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
PYTHON="/home/user/anaconda3/envs/aris/bin/python"
CONFIG="${PROJECT_DIR}/config/experiment_config.yaml"

cd "$PROJECT_DIR"

echo "╔══════════════════════════════════════════════╗"
echo "║  Dynamic Topology Fault Dataset — Batch Run  ║"
echo "╚══════════════════════════════════════════════╝"
echo "  Python: $PYTHON"
echo "  Config: $CONFIG"
echo "  Time:   $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo ""

$PYTHON -m runner.scheduler \
    --config "$CONFIG" \
    "$@"

echo ""
echo "Done. Check summary:"
echo "  cat $PROJECT_DIR/data/experiments/batch_summary.json | python3 -m json.tool | head -30"
