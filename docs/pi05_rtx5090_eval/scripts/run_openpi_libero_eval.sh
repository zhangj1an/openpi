#!/bin/bash
# openpi pi0.5 LIBERO success-rate eval: JAX websocket policy server (openpi venv) + LIBERO sim client (lerobot venv).
# Usage: run_openpi_libero_eval.sh <suite> <task_ids> <episodes_per_task>
set -u
cd "$(dirname "$0")"
SUITE=${1:-libero_spatial}; TASKS=${2:-0,1,2,3,4,5,6,7,8,9}; EPS=${3:-10}
OUT=runs/openpi_eval_${SUITE}
mkdir -p "$OUT"
export OPENPI_DATA_HOME=/dev/shm/openpi XLA_PYTHON_CLIENT_MEM_FRACTION=0.6
(cd /workspace/openpi && exec .venv/bin/python scripts/serve_policy.py --port 18000 --env LIBERO) > "$OUT/server.log" 2>&1 &
SERVER=$!
trap 'kill $SERVER 2>/dev/null' EXIT
until grep -q "server listening" "$OUT/server.log" 2>/dev/null || ss -ltn | grep -q ":18000 "; do
  kill -0 $SERVER 2>/dev/null || { echo "server died"; tail -20 "$OUT/server.log"; exit 1; }
  sleep 2
done
echo "server up $(date)"
MUJOCO_GL=egl /workspace/lerobot/.venv/bin/python sim_client_openpi.py --port 18000 --suite "$SUITE" \
  --task-ids "$TASKS" --episodes-per-task "$EPS" --videos 3 --out "$OUT" < /dev/null > "$OUT/client.log" 2>&1
echo "client exit=$? $(grep SUMMARY $OUT/client.log)"
