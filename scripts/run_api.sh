#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

export ORC_LLM_MODEL_PATH="${ORC_LLM_MODEL_PATH:-models/LFM2.5-1.2B-Instruct-Q4_K_M.gguf}"
export ORC_LLM_N_CTX="${ORC_LLM_N_CTX:-4096}"
export ORC_LLM_N_GPU_LAYERS="${ORC_LLM_N_GPU_LAYERS:-999}"
export ORC_LLM_MAX_TOKENS="${ORC_LLM_MAX_TOKENS:-256}"

if [ ! -f "$ORC_LLM_MODEL_PATH" ]; then
  python scripts/download_model.py
fi

python -m uvicorn orc_llm.api:app --host "${HOST:-127.0.0.1}" --port "${PORT:-8000}"

