#!/bin/bash
# ============================================================
# run_export.sh — Export data for a completed experiment
# ============================================================
# Prerequisites:
#   - port-forward running: ./scripts/start-port-forward.sh
#   - experiment already completed (metadata.json exists)
#
# Usage:
#   ./scripts/run_export.sh <experiment_dir>
#   ./scripts/run_export.sh data/experiments/T3_prodcat_redirect_run042
#   ./scripts/run_export.sh data/experiments/T3_prodcat_redirect_run042 --loki-url http://custom:3100
# ============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

EXP_DIR="${1:?Usage: $0 <experiment_dir> [exporter args...]}"
shift || true

# Resolve to absolute path
EXP_DIR="$(cd "$EXP_DIR" && pwd)"

# Verify metadata.json exists
if [ ! -f "$EXP_DIR/metadata.json" ]; then
    echo "ERROR: metadata.json not found in $EXP_DIR"
    echo "  Is this a valid experiment directory?"
    exit 1
fi

# Read experiment_id for logging
PYTHON="${PYTHON:-/home/user/anaconda3/envs/aris/bin/python}"
EXP_ID=$($PYTHON -c "import json; print(json.load(open('$EXP_DIR/metadata.json')).get('experiment_id','unknown'))")

echo "╔══════════════════════════════════════════════╗"
echo "║  Data Export: $EXP_ID"
echo "╚══════════════════════════════════════════════╝"
echo "  Dir: $EXP_DIR"
echo ""

# Default API endpoints (override via env vars)
export PROM_URL="${PROM_URL:-http://localhost:9090}"
export LOKI_URL="${LOKI_URL:-http://localhost:3100}"
export TEMPO_URL="${TEMPO_URL:-http://localhost:3200}"

echo "  APIs: Prom=$PROM_URL  Loki=$LOKI_URL  Tempo=$TEMPO_URL"
echo ""

# Use conda aris environment (Python 3.10 + all deps)
PYTHON="${PYTHON:-/home/user/anaconda3/envs/aris/bin/python}"

# Run the Python exporter
cd "$PROJECT_DIR"
$PYTHON -m collector.exporter \
    --experiment-dir "$EXP_DIR" \
    --prom-url "$PROM_URL" \
    --loki-url "$LOKI_URL" \
    --tempo-url "$TEMPO_URL" \
    "$@"

echo ""
echo "Files exported:"
echo "─────────────────"
find "$EXP_DIR" -type f -newer "$EXP_DIR/metadata.json" | sort | \
    while read -r f; do
        size=$(stat --printf="%s" "$f" 2>/dev/null || stat -f%z "$f" 2>/dev/null || echo "?")
        rel="${f#$EXP_DIR/}"
        printf "  %-50s %8s bytes\n" "$rel" "$size"
    done

echo ""
echo "Done. Data ready for RCA training in:"
echo "  $EXP_DIR"
