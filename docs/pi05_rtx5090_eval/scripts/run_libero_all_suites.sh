#!/bin/bash
# Usage: run_libero_all_suites.sh <label> <port> <parallel>
# All 4 LIBERO suites x 10 tasks x 50 episodes (seed 7, official init states) against a running openpi policy server,
# sharded by (suite, task) over <parallel> clients. Resumable: finished tasks (results.json present) are skipped.
# Env: RUNS_DIR (default ./runs, relative to the repo root), LIBERO_PYTHON (Python 3.8 env with LIBERO + openpi-client).
set -u
LABEL=$1; PORT=$2; PAR=$3
cd "$(dirname "$0")/../../.."  # repo root
ROOT=${RUNS_DIR:-runs}/$LABEL; mkdir -p "$ROOT"
jobs=""
# longest suites first so they don't all end up at the tail
for s in libero_10 libero_goal libero_object libero_spatial; do for t in 0 1 2 3 4 5 6 7 8 9; do jobs="$jobs $s:$t"; done; done
echo $jobs | tr ' ' '\n' | xargs -P $PAR -I{} bash -c '
  s=${1%%:*}; t=${1##*:}; out='$ROOT'/$s/task$t
  [ -f $out/results.json ] && exit 0
  mkdir -p $out
  MUJOCO_GL=egl PYTHONPATH=third_party/libero ${LIBERO_PYTHON:-python} docs/pi05_rtx5090_eval/scripts/sim_client_openpi.py \
    --port '$PORT' --suite $s --task-ids $t --episodes-per-task 50 --seed 7 --videos 0 --out $out < /dev/null > $out/client.log 2>&1
  rc=$?
  echo "$(date +%T) $s task$t exit=$rc $(grep SUMMARY $out/client.log)"
' _ {}
echo ALL_DONE
