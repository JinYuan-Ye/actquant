#!/usr/bin/env bash
# Path/CLI adaptation only: official ActQuant server + untouched OpenPI client.
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ "${1:-}" != --run || -z "${2:-}" ]]; then
 cat <<'HELP'
Usage (starts an actual evaluation; never invoked by install.sh):
  run_eval_single_gpu.sh --run /absolute/EVAL_DIR [libero_spatial] [trials_per_task=1]
EVAL_DIR must already contain pi05.gguf, tokenizer.model, norm_stats.json.
One trial per task still evaluates all 10 tasks in the selected suite.
No downloads are performed. GPU=0; port can be set using ACTQUANT_PORT.
HELP
 exit 0
fi
MODEL_DIR="$(realpath "$2")"
SUITE="${3:-libero_spatial}"
TRIALS="${4:-1}"
PORT="${ACTQUANT_PORT:-8000}"
case "$SUITE" in libero_spatial|libero_object|libero_goal|libero_10) ;; *) echo 'Invalid suite'; exit 2;; esac
[[ "$TRIALS" =~ ^[1-9][0-9]*$ ]] && ((TRIALS <= 50)) || { echo 'Trials must be 1..50'; exit 2; }
for name in pi05.gguf tokenizer.model norm_stats.json; do
 [[ -s "$MODEL_DIR/$name" ]] || { echo "Missing local model file: $MODEL_DIR/$name"; exit 2; }
done
RUN_DIR="$ROOT/results/${SUITE}-$(date -u +%Y%m%dT%H%M%SZ)-$$"
mkdir -p "$RUN_DIR"
(
 source "$ROOT/deployment/activate_server.sh"
 export CUDA_VISIBLE_DEVICES=0
 exec python "$ROOT/src/ActQuant/tools/pi0.5/serve_policy.py" \
   --model-dir "$MODEL_DIR" --device CUDA0 --steps 10 --host 127.0.0.1 --port "$PORT"
) > "$RUN_DIR/server.log" 2>&1 &
SERVER_PID=$!
cleanup() { kill "$SERVER_PID" 2>/dev/null || true; wait "$SERVER_PID" 2>/dev/null || true; }
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
READY=0
for attempt in $(seq 1 60); do
 kill -0 "$SERVER_PID" 2>/dev/null || { cat "$RUN_DIR/server.log"; exit 1; }
 if curl --max-time 2 -fsS "http://127.0.0.1:$PORT/healthz" >/dev/null 2>&1; then READY=1; break; fi
 sleep 2
done
[[ "$READY" == 1 ]] || { echo "Server timeout; inspect $RUN_DIR/server.log"; exit 1; }
(
 source "$ROOT/deployment/activate_libero.sh"
 python "$ROOT/src/openpi/examples/libero/main.py" \
   --args.host 127.0.0.1 --args.port "$PORT" --args.task-suite-name "$SUITE" \
   --args.num-trials-per-task "$TRIALS" --args.replan-steps 5 --args.video-out-path "$RUN_DIR/videos"
) 2>&1 | tee "$RUN_DIR/client.log"
