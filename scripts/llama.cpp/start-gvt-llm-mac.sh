#!/usr/bin/env bash
set -euo pipefail
# Launch llama-server as the LLM backend for generative-vector-tile on macOS.
#
# The server speaks the OpenAI /v1/chat/completions API, so gvt only needs:
#   OPENAI_BASE_URL=http://127.0.0.1:18099/v1
#   OPENAI_API_KEY=dummy
#   OPENAI_MODEL=gvt-llm
#
# Port 18099 is chosen so it doesn't collide with TRIDENT's 18091-18094 if
# you happen to have both running on the same machine.

SESSION_NAME="${SESSION_NAME:-gvt-llm}"
PORT="${PORT:-18099}"
HOST="${HOST:-127.0.0.1}"
LLAMA_CPP_DIR="${LLAMA_CPP_DIR:-$HOME/llama.cpp}"
LLAMA_SERVER_BIN="${LLAMA_CPP_DIR}/build/bin/llama-server"

MODEL_PATH="${MODEL_PATH:-$HOME/.cache/huggingface/hub/models--unsloth--Qwen3.5-35B-A3B-GGUF/snapshots/bc014a17be43adabd7066b7a86075ff935c6a4e2/Qwen3.5-35B-A3B-UD-IQ4_XS.gguf}"
ALIAS="${ALIAS:-gvt-llm}"
LOG_PATH="${LOG_PATH:-/tmp/${SESSION_NAME}.log}"

if ! command -v tmux >/dev/null 2>&1; then
  echo "tmux required (brew install tmux)" >&2
  exit 1
fi
if [ ! -x "${LLAMA_SERVER_BIN}" ]; then
  echo "llama-server not found: ${LLAMA_SERVER_BIN}" >&2
  exit 1
fi
if [ ! -f "${MODEL_PATH}" ]; then
  echo "model file not found: ${MODEL_PATH}" >&2
  exit 1
fi
if tmux has-session -t "${SESSION_NAME}" 2>/dev/null; then
  echo "session ${SESSION_NAME} already exists"
  echo "log: ${LOG_PATH}"
  exit 0
fi

# Tuning notes for M4 Max 36GB:
#   -c 16384         16K context is plenty for our system prompt + short q
#   -ngl 999         Metal offload all layers
#   -ncmoe 0         Apple Silicon unified memory -- start at 0, raise only if
#                    Metal recommendedMaxWorkingSetSize is exceeded
#   -fa on / -ctk/-ctv q8_0  Flash Attention + KV cache quantization
#   -t 10            M4 Max P-core count
#   --jinja          Enable jinja chat templates so the OAI chat format works
#                    correctly with structured outputs
#   --reasoning off  Suppress Qwen's <think> tags so chat.completions.parse
#                    sees clean JSON in the assistant message
CMD=$(cat <<EOF
${LLAMA_SERVER_BIN} \
  -m ${MODEL_PATH} \
  -a ${ALIAS} \
  -ngl 999 \
  -ncmoe 0 \
  -c 16384 \
  -np 1 \
  -fa on \
  -ctk q8_0 \
  -ctv q8_0 \
  -t 10 \
  --jinja \
  --reasoning off \
  --host ${HOST} \
  --port ${PORT} \
  > ${LOG_PATH} 2>&1
EOF
)

tmux new-session -d -s "${SESSION_NAME}" "bash -lc '${CMD}'"
echo "started: ${SESSION_NAME}  endpoint: http://${HOST}:${PORT}/v1"
echo "log:     ${LOG_PATH}"
echo
echo "Configure generative-vector-tile to use this LLM:"
echo "  export OPENAI_BASE_URL=http://${HOST}:${PORT}/v1"
echo "  export OPENAI_API_KEY=dummy"
echo "  export OPENAI_MODEL=${ALIAS}"
echo "  export LLM_TIMEOUT_S=30"
echo
echo "Stop:    tmux kill-session -t ${SESSION_NAME}"
