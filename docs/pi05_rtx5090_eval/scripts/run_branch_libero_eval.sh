#!/bin/bash
# LIBERO-Spatial eval against the pi05-bs1-latency branch's serve_policy (openpi websocket protocol).
# Usage: run_branch_libero_eval.sh <label> <serve_policy args...>
set -u
cd "$(dirname "$0")"
LABEL=$1; shift
OUT=runs/eval_$LABEL
mkdir -p "$OUT"
export OPENPI_DATA_HOME=/dev/shm/openpi_data TORCHINDUCTOR_CACHE_DIR=/workspace/tmp_uv/inductor
(cd /workspace/openpi && exec /workspace/openpi_venv/bin/python scripts/serve_policy.py --port 18000 "$@") > "$OUT/server.log" 2>&1 &
SERVER=$!
trap 'kill $SERVER 2>/dev/null' EXIT
until ss -ltn | grep -q ":18000 "; do
  kill -0 $SERVER 2>/dev/null || { echo "server died"; tail -20 "$OUT/server.log"; exit 1; }
  sleep 2
done
echo "server up $(date)"
MUJOCO_GL=egl /workspace/lerobot/.venv/bin/python sim_client_openpi.py --port 18000 --suite libero_spatial \
  --task-ids 0,1,2,3,4,5,6,7,8,9 --episodes-per-task 10 --videos 3 --out "$OUT" < /dev/null > "$OUT/client.log" 2>&1
echo "client exit=$? $(grep SUMMARY $OUT/client.log)"
