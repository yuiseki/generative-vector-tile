#!/usr/bin/env bash
set -euo pipefail
# Launch N llama-server instances on consecutive ports to act as a parallel
# LLM pool for generative-vector-tile. The application client (llm.py) reads
# OPENAI_BASE_URLS and round-robins requests across them.
#
# Defaults target Qwen3.5-1.7B-Q4_K_M -- small enough that 4 instances comfortably
# fit on a 36GB M4 Max and still gives true parallel LLM calls so the browser's
# burst of tile requests isn't queued behind a single LLM.

POOL_SIZE="${POOL_SIZE:-4}"
PORT_START="${PORT_START:-18099}"
HOST="${HOST:-127.0.0.1}"
LLAMA_CPP_DIR="${LLAMA_CPP_DIR:-$HOME/llama.cpp}"
LLAMA_SERVER_BIN="${LLAMA_CPP_DIR}/build/bin/llama-server"
HF_REPO="${HF_REPO:-unsloth/Qwen3.5-2B-GGUF}"
HF_QUANT="${HF_QUANT:-Q4_K_M}"
ALIAS="${ALIAS:-gvt-llm}"

if ! command -v tmux >/dev/null 2>&1; then
  echo "tmux required (brew install tmux)" >&2
  exit 1
fi
if [ ! -x "${LLAMA_SERVER_BIN}" ]; then
  echo "llama-server not found: ${LLAMA_SERVER_BIN}" >&2
  exit 1
fi

# Per-instance tuning notes:
#   -c 8192    System prompt is ~1KB, q is small, 8K covers comfortably
#   -ngl 999   Metal offload everything
#   -t 3       Restrict each instance's CPU threads so 4 instances don't
#              fight over the 10 M4 Max P-cores; total = 12 with light
#              oversubscription that's actually fine for prompt prefill
#   -fa on     Flash Attention
#   --jinja    Use the model's chat template (required for proper Qwen
#              format with the json_schema response_format path)
#   --reasoning off  Suppress Qwen <think> emission for this short task
build_cmd() {
  local port="$1"
  local log="$2"
  cat <<EOF
${LLAMA_SERVER_BIN} \
  -hf ${HF_REPO}:${HF_QUANT} \
  -a ${ALIAS} \
  -ngl 999 \
  -c 8192 \
  -np 1 \
  -fa on \
  -t 3 \
  --jinja \
  --reasoning off \
  --host ${HOST} \
  --port ${port} \
  > ${log} 2>&1
EOF
}

# Launch instance 0 first and wait for it to be /v1/models-ready. This is the
# leader that downloads the GGUF into the HF cache. Subsequent instances mmap
# the same file -- without this serial step, all N would race to write the
# same .downloadInProgress blob in parallel, wasting bandwidth (4x) and
# risking corruption when one of them renames the temp file out from under
# the others.
LEADER_PORT=${PORT_START}
LEADER_SESSION="gvt-llm-0"
LEADER_LOG="/tmp/${LEADER_SESSION}.log"
if tmux has-session -t "${LEADER_SESSION}" 2>/dev/null; then
  echo "leader ${LEADER_SESSION} already running (port ${LEADER_PORT})"
else
  CMD=$(build_cmd "${LEADER_PORT}" "${LEADER_LOG}")
  tmux new-session -d -s "${LEADER_SESSION}" "bash -lc '${CMD}'"
  echo "leader started: ${LEADER_SESSION} (port ${LEADER_PORT})  log: ${LEADER_LOG}"
  echo "waiting for leader to download + load the model (this is the only fetch)..."
fi
until curl -sf -o /dev/null "http://${HOST}:${LEADER_PORT}/v1/models" 2>/dev/null; do
  sleep 3
done
echo "leader ready -- model is in HF cache, fanning out the rest"

urls="http://${HOST}:${LEADER_PORT}/v1"
for i in $(seq 1 $((POOL_SIZE - 1))); do
  PORT=$((PORT_START + i))
  SESSION_NAME="gvt-llm-${i}"
  LOG_PATH="/tmp/${SESSION_NAME}.log"

  if tmux has-session -t "${SESSION_NAME}" 2>/dev/null; then
    echo "session ${SESSION_NAME} already exists (port ${PORT})"
  else
    CMD=$(build_cmd "${PORT}" "${LOG_PATH}")
    tmux new-session -d -s "${SESSION_NAME}" "bash -lc '${CMD}'"
    echo "started: ${SESSION_NAME}  endpoint: http://${HOST}:${PORT}/v1  log: ${LOG_PATH}"
  fi
  urls="${urls},http://${HOST}:${PORT}/v1"
done

echo
echo "Configure generative-vector-tile to round-robin across this pool:"
echo "  export OPENAI_BASE_URLS=${urls}"
echo "  export OPENAI_API_KEY=dummy"
echo "  export OPENAI_MODEL=${ALIAS}"
echo "  export LLM_TIMEOUT_S=30"
echo
echo "Stop all:  tmux ls | awk -F: '/^gvt-llm-/{print \$1}' | xargs -I{} tmux kill-session -t {}"
